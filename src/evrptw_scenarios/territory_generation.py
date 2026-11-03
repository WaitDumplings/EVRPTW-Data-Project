"""Connect the reviewed independent activation modules to the frozen CLE pipeline."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit
from scipy.spatial import cKDTree

from .activation import sample_conditional_bernoulli, unit_log_odds
from .calibration import service_curve
from .capacity import sample_capacity_order
from .demand import MARK_FIELDS, PARCEL_FIELDS, SITE_FIELDS
from .marks import DonorLibrary, instantiate_availability, reconcile_one_visit
from .physical import bind_physical
from .territory import TerritoryProblem, grow_clock, score_candidate, weighted_reference_length
from .util import identity, output_hashes, rng, sha256_file, write_json


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def _component_count(cells, adjacency):
    remaining, count = set(cells), 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            neighbors = remaining.intersection(adjacency.get(stack.pop(), ()))
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return count


def _geometry_diagnostics(frame, depot):
    """Published-coordinate geometry, separate from actual road costs."""
    if not len(frame):
        return {}
    from pyproj import CRS, Transformer
    from .territory_world import great_circle_km
    lon, lat = frame.location_lon.to_numpy(float), frame.location_lat.to_numpy(float)
    center_lon, center_lat = float(lon.mean()), float(lat.mean())
    projection = Transformer.from_crs(4326, CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"), always_xy=True)
    x, y = projection.transform(lon, lat)
    xy = np.column_stack([x, y])
    radial = np.sqrt((xy**2).sum(axis=1)) / 1000
    # This small N-by-N computation concerns only the selected task, never CLE.
    from scipy.spatial.distance import pdist
    unique = np.unique(xy, axis=0)
    nearest = cKDTree(unique).query(unique, k=2)[0][:, 1] / 1000 if len(unique) > 1 else np.array([])
    return {"geometry_method": "great_circle_center_radius_and_local_aeqd_pairs",
            "center_longitude": center_lon, "center_latitude": center_lat,
            "center_r90_km": float(np.quantile(radial, .9)),
            "diameter_km": float(pdist(xy).max() / 1000) if len(xy) > 1 else 0.,
            "unique_coordinate_nn_p95_km": float(np.quantile(nearest, .95)) if len(nearest) else None,
            "unique_coordinate_nn_p99_km": float(np.quantile(nearest, .99)) if len(nearest) else None,
            "depot_to_center_geodesic_km": float(great_circle_km(
                float(depot.location_lon.iloc[0]), float(depot.location_lat.iloc[0]), center_lon, center_lat))}


def activate_territory(context, config, *, day_index, size):
    """One independent size/day, sharing only the defined candidate clock field."""
    if config["activation"].get("method") == "sequential_tract":
        from .tract_activation import activate_tracts
        return activate_tracts(context, config, day_index=day_index, size=size)

    catalog, world = context.catalog, context.info
    settings = config["activation"]
    current_date = (date.fromisoformat(config["operations"]["start_date"]) + timedelta(days=day_index)).isoformat()
    geographic_parameters = {key: config["world"][key] for key in [
        "seed", "provider_id", "external_radius_km", "p_unit"]}
    numeric_parameters = {key: settings[key] for key in [
        "tau", "opening_penalty", "transfer_penalty", "max_centers"]}
    spatial_id = identity("territory-spatial-v1", world["source"]["cle_manifest_sha256"],
                          world["source"]["community_sha256"], world["depot_id"],
                          geographic_parameters, current_date, size, numeric_parameters)
    theta = unit_log_odds(catalog.modeled_residential_units.to_numpy(), config["world"]["p_unit"])
    pi, log_pi = expit(theta), log_expit(theta)
    result = {"spatial_id": spatial_id, "date": current_date, "day_index": day_index,
              "requested_n": size, "supported_size": False, "selected_n": 0,
              "selected_k": None, "expected_mass": None, "reference_length_km": None,
              "activation_eta": None, "activation_attempts": 0,
              "activation_proposal_variance": None, "candidate_count": 0,
              "coupling": "anchors_preselected_independent_of_shared_source_one_field",
              "parameter_status": settings["parameter_status"], "candidates": []}
    if len(catalog) < size:
        return {**result, "status": "unsupported_candidate_count"}, [], [], []
    anchor_rng = rng("territory-anchors", spatial_id)
    first = int(anchor_rng.choice(len(catalog), p=pi / pi.sum()))
    anchors = [first]
    road_costs = context.anchor_costs(anchors)
    reference = weighted_reference_length(road_costs[0], pi, size)
    result["reference_length_km"] = _json_number(reference)
    if not np.isfinite(reference) or reference <= 0:
        return {**result, "status": "unsupported_reference_scale"}, [], [], []
    if settings["max_centers"] == 2:
        weights = pi * (road_costs[0] / reference)**2
        weights[first] = 0.
        if weights.sum() > 0 and np.isfinite(weights).all():
            anchors.append(int(anchor_rng.choice(len(catalog), p=weights / weights.sum())))
            road_costs = context.anchor_costs(anchors)
    # Each source field is independent; source zero is reused across K candidates.
    exponentials = np.vstack([rng("territory-clock", spatial_id, str(catalog.location_id.iloc[a])).exponential(
        size=len(catalog)) for a in anchors])
    candidates = []
    for count in range(1, len(anchors) + 1):
        problem = TerritoryProblem(
            location_ids=catalog.location_id.astype(str).to_numpy(), cell_ids=catalog.cell_id.astype(str).to_numpy(),
            adjacency=context.adjacency, probabilities=pi, log_probabilities=log_pi,
            road_costs=road_costs[:count], anchor_cells=catalog.cell_id.iloc[anchors[:count]].astype(str).tolist(),
            target_count=size, reference_length=reference, tau=settings["tau"])
        grown = grow_clock(problem, exponentials=exponentials[:count])
        score = score_candidate(problem, grown, anchors[:count],
                                opening_penalty=settings["opening_penalty"],
                                transfer_penalty=settings["transfer_penalty"])
        candidates.append((score, count, grown))
        result["candidates"].append({"centers": count, "score": _json_number(score),
            "supported": bool(grown.supported), "expected_mass": grown.expected_mass,
            "candidate_count": len(grown.selected_indices), "stop_reason": grown.stop_reason,
            "winning_anchor_count": grown.winning_anchor_count,
            "contributing_anchor_count": grown.contributing_anchor_count,
            "events": [asdict(event) for event in grown.events]})
    result["anchor_location_ids"] = catalog.location_id.iloc[anchors].astype(str).tolist()
    result["anchor_depot_outward_km"] = (catalog.depot_outward_distance_m.iloc[anchors] / 1000).tolist()
    result["anchor_depot_inward_km"] = (catalog.depot_inward_distance_m.iloc[anchors] / 1000).tolist()
    possible = [(score, k, grown) for score, k, grown in candidates if grown.supported and np.isfinite(score)]
    if not possible:
        return {**result, "status": "unsupported_territory"}, [], [], []
    _, count, grown = min(possible, key=lambda item: (item[0], item[1]))
    territorial_indices = np.asarray(grown.selected_indices, dtype=int)
    sampled = sample_conditional_bernoulli(theta[territorial_indices], size,
                                         rng("territory-subset", spatial_id))
    selected_indices = territorial_indices[sampled.indices]
    selected_indices = np.asarray(sorted(selected_indices, key=lambda i: str(catalog.location_id.iloc[i])))
    diagnostics = sampled.diagnostics
    result.update(supported_size=True, selected_n=len(selected_indices), selected_k=count,
                  expected_mass=grown.expected_mass, candidate_count=len(territorial_indices),
                  activation_eta=_json_number(diagnostics.eta), activation_attempts=diagnostics.attempts,
                  activation_proposal_variance=diagnostics.proposal_variance,
                  activation_proposal_budget=diagnostics.proposal_budget, status="activated",
                  zero_tilt_budget_difference=grown.expected_mass-size)
    result["geometry"] = _geometry_diagnostics(catalog.iloc[selected_indices], context.depot)
    result["territory_cell_components"] = _component_count(
        catalog.cell_id.iloc[territorial_indices], context.adjacency)
    result["activated_cell_components"] = _component_count(
        catalog.cell_id.iloc[selected_indices], context.adjacency)
    territory_rows = [{"location_id": str(catalog.location_id.iloc[i]), "owner": int(owner),
                       "probability": float(pi[i]), "road_cost_km": float(road_costs[owner, i])}
                      for i, owner in zip(territorial_indices, grown.owners)]
    selection_rows = [{"location_id": str(catalog.location_id.iloc[i]), "rank": rank}
                      for rank, i in enumerate(selected_indices)]
    return result, selected_indices.tolist(), territory_rows, selection_rows


def _apply_active_unit_service(bundle, active, count, config):
    """Reallocate existing donor variable work; keep all raw source marks."""
    settings = config["workload"]
    mean = config["world"]["mean_parcels_per_active_unit"]
    share, exponent = settings["service_active_unit_share"], settings["service_active_unit_exponent"]
    curve = service_curve(count, settings)
    intercept = settings["service_a_s"]
    factor = (intercept + (curve - intercept) * (1 - share + share *
              (active / max(1., count / mean))**exponent)) / curve
    for mark in bundle["parcels"]:
        mark["assigned_planned_service_s"] *= factor
    return factor


def generate_selected_orders(catalog, selected_indices, config, donors, *, scenario_id, day_index):
    """Preserve S and sample whole per-location bundles with explicit failure."""
    op, world, settings = (config[k] for k in ["operations", "world", "workload"])
    current = (date.fromisoformat(op["start_date"]) + timedelta(days=day_index)).isoformat()
    parcels, sites, rejections = [], [], []
    for i in selected_indices:
        location = catalog.iloc[i]
        loc, cell = str(location.location_id), str(location.community_id)
        result = sample_capacity_order(
            units=int(location.modeled_residential_units), probability=world["p_unit"],
            mean_parcels=world["mean_parcels_per_active_unit"],
            counts_rng=rng("selected-order-count", scenario_id, loc),
            marks_rng=rng("selected-order-marks", scenario_id, loc),
            bundle_sampler=donors.sample_bundle, capacity_cm3=settings["cargo_capacity_cm3"],
            max_proposals=settings["maximum_full_proposals"])
        for failure in result.rejections:
            proposal = failure.proposal
            rejections.append({"scenario_id": scenario_id, "day_index": day_index, "location_id": loc,
                "proposal_index": failure.attempt, "proposed_active_units": proposal.active_units,
                "proposed_package_count": proposal.package_count, "proposed_volume_cm3": proposal.volume_cm3,
                "source_donor_id": proposal.bundle.get("source_donor_id"), "reason": failure.reason,
                "source_parcel_refs_json": json.dumps([m.get("source_parcel_ref") for m in proposal.bundle["parcels"]]),
                "exhausted": not result.accepted and failure.attempt == result.attempts})
        common = {"scenario_id": scenario_id, "operational_day_id": scenario_id,
                  "day_index": day_index, "date": current, "location_id": loc, "community_id": cell,
                  "capacity_conditioning_attempts": result.attempts}
        midnight = day_index * 86400
        if not result.accepted:
            sites.append({**common, "parcel_count": 0, "demand_cm3": None, "planned_service_time_s": None,
                "tw_start_s": midnight + op["shift_start_s"], "tw_end_s": midnight + op["shift_end_s"],
                "source_specified_window_count": 0, "source_missing_window_count": 0,
                "supported_one_visit_representation": False, "generation_supported": False,
                "location_tw_status": "unsupported_generation", "location_tw_source_donor_id": None,
                "reasons": json.dumps([result.failure_reason])})
            continue
        proposal = result.proposal
        active, count, bundle = proposal.active_units, proposal.package_count, proposal.bundle
        service_factor = _apply_active_unit_service(bundle, active, count, config)
        window = donors.sample_location_window(rng("selected-location-window", scenario_id, loc))
        rows = []
        for ordinal, mark in enumerate(bundle["parcels"]):
            pid = identity("selected-parcel-v1", scenario_id, loc, ordinal)
            clock = rng("selected-parcel-ready", scenario_id, loc, ordinal)
            row = {key: mark.get(key) for key in MARK_FIELDS}
            row.update(window)
            row.update({"scenario_id": scenario_id, "parcel_id": pid, "location_id": loc, "community_id": cell,
                "arrival_day_index": day_index, "arrival_date": current,
                "ready_s": midnight + float(clock.uniform(*op["ready_window_s"])),
                "promise_s": midnight + op["shift_end_s"], "promise_day_index": day_index,
                "eligible_for_deferral": False, "preferred_weekday": date.fromisoformat(current).weekday(),
                "scheduled_day_index": day_index, "dispatch_day_index": day_index,
                "commitment_failure": False, "status": "dispatched_to_workload",
                "source_donor_id": bundle["source_donor_id"], "exact_source_count": bundle["exact_source_count"],
                "source_package_count": bundle["source_package_count"],
                "source_count_extrapolated": bundle["count_outside_observed_range"],
                "active_units": active, "modeled_residential_units": int(location.modeled_residential_units),
                "capacity_conditioning_attempts": result.attempts,
                "service_active_unit_factor": service_factor})
            rows.append(row)
        availability = instantiate_availability({"parcels": rows}, service_day_start_s=midnight,
                                                mode=op["availability_pattern"])
        aggregate = reconcile_one_visit(availability, releases_s=[r["ready_s"] for r in rows],
            deadlines_s=[r["promise_s"] for r in rows],
            operating_window_s=[midnight + op["shift_start_s"], midnight + op["shift_end_s"]],
            source_window_end=op["source_window_end"], missing_window_policy=op["missing_window_policy"])
        sites.append({**common, **{key: aggregate[key] for key in SITE_FIELDS if key in aggregate},
                      "generation_supported": True, "active_units": active,
                      "modeled_residential_units": int(location.modeled_residential_units),
                      "service_active_unit_factor": service_factor, **window,
                      "reasons": json.dumps(aggregate["reasons"])})
        parcels.extend(rows)
    return parcels, sites, rejections


def _materialize(config, context, physical, scenario, selected, sites, root):
    from .build import materialize_instances
    # The established materializer sees one independent day/task at a time;
    # it cannot reuse a prefix or demand bundle from a different N.
    day = scenario["day_index"]
    selected_ids = context.catalog.location_id.iloc[selected].astype(str).tolist()
    job = root / "tasks" / scenario["scenario_id"]
    job.mkdir(parents=True)
    site_path, view_path, order_path = job / "sites.parquet", job / "views.parquet", job / "order.parquet"
    pd.DataFrame(sites, columns=[*SITE_FIELDS, "scenario_id"]).to_parquet(site_path, index=False)
    view = {"view_id": scenario["scenario_id"], "day_index": day, "kind": "territory_day",
            "requested_n": scenario["requested_n"], "selected_n": len(selected),
            "available_n": scenario["candidate_count"], "supported_size": scenario["supported_size"],
            "parent_operational_day_id": scenario["scenario_id"],
            "materialize_matrix": scenario["supported_size"] and day in config["instances"]["materialize_day_indices"],
            "sampler": "territory_first_conditional_bernoulli"}
    pd.DataFrame([view]).to_parquet(view_path, index=False)
    pd.DataFrame([{"day_index": day, "rank": j, "location_id": loc} for j, loc in enumerate(selected_ids)],
                 columns=["day_index", "rank", "location_id"]).to_parquet(order_path, index=False)
    legacy_shape = {**config, "views": {}}
    demand = {"sites_path": str(site_path), "process_id": scenario["process_id"]}
    views = {"views_path": str(view_path), "selection_order_path": str(order_path)}
    # Existing paired-road routing, charging selection and solver representation.
    output = materialize_instances(legacy_shape, context.info, demand, views, physical,
                                   root / "instances", progress=lambda *a, **k: None)
    path = Path(output[scenario["scenario_id"]])
    metadata = json.loads(path.read_text())
    metadata.update(scenario_id=scenario["scenario_id"], spatial_id=scenario["spatial_id"],
                    nested_sizes=False, territory_selected_centers=scenario["selected_k"])
    if any(metadata["assessment"][key] for key in ["unsupported_generation_locations",
                                                  "unsupported_one_visit_locations", "oversize_bundles"]):
        metadata["status"] = "unsupported_workload"
    write_json(path, metadata)
    return view, str(path)


def build_territory_run(config, *, progress=print):
    from .territory_world import build_territory_world
    destination = Path(config["output"]["path"]).resolve()
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Output or partial directory exists; choose a new path")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    code_files = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(temporary / "config.json", config)
    try:
        progress("constructing one depot world and road-supported community graph", flush=True)
        world_options = {}
        if config["world"].get("service_policy") == "fastest_time_energy":
            weekday = date.fromisoformat(config["operations"]["start_date"]).weekday()
            world_options = {"vehicle": config["physical"]["vehicle"],
                             "day_type": "weekend" if weekday >= 5 else "weekday"}
        context = build_territory_world(config["input"]["cle"], config["input"]["communities"],
            config["world"], temporary / "world", progress=progress, **world_options)
        donors = DonorLibrary(config["input"]["donor_artifact"])
        donors.configure_calibration(config["workload"])
        physical = bind_physical(config["input"]["cle"], context.info, config["physical"], temporary / "physical")
        process_id = identity("territory-order-process-v1", context.info["world_id"],
            donors.manifest["source_sha256"], donors.pool_identity, config["workload"], config["operations"])
        for folder in ["activation/scenarios", "demand", "views"]:
            (temporary / folder).mkdir(parents=True, exist_ok=True)
        all_parcels, all_sites, all_failures, days, views = [], [], [], [], []
        scenario_rows, territorial_rows, selection_rows, instances = [], [], [], {}
        for day in range(config["operations"]["days"]):
            for n in config["instances"]["sizes"]:
                progress(f"activating independent day {day}, Cus{n}", flush=True)
                scenario, selected, territory, selection = activate_territory(context, config, day_index=day, size=n)
                scenario_id = identity("territory-day-v1", scenario["spatial_id"], process_id)
                scenario.update(scenario_id=scenario_id, view_id=scenario_id, process_id=process_id)
                for collection in [territory, selection]:
                    for row in collection:
                        row["scenario_id"] = scenario_id
                parcels, sites, failures = generate_selected_orders(context.catalog, selected, config, donors,
                    scenario_id=scenario_id, day_index=day)
                all_parcels.extend(parcels)
                all_sites.extend(sites)
                all_failures.extend(failures)
                territorial_rows.extend(territory)
                selection_rows.extend(selection)
                success = sum(bool(s["generation_supported"]) for s in sites)
                scenario["capacity_supported_locations"] = success
                scenario["capacity_failed_locations"] = len(sites) - success
                write_json(temporary / "activation/scenarios" / (scenario_id + ".json"), scenario)
                scenario_rows.append({k: v for k, v in scenario.items() if not isinstance(v, (dict, list))})
                days.append({"scenario_id": scenario_id, "operational_day_id": scenario_id,
                    "day_index": day, "date": scenario["date"], "requested_n": n,
                    "arrivals": len(parcels), "dispatched": len(parcels), "previous_backlog": 0,
                    "backlog": 0, "active_locations": len(sites), "route_solution_claimed": False})
                view, instance_path = _materialize(config, context, physical, scenario, selected, sites, temporary)
                views.append(view)
                instances[scenario_id] = instance_path
        parcel_columns = list(dict.fromkeys([*PARCEL_FIELDS, "scenario_id", "service_active_unit_factor"]))
        site_columns = list(dict.fromkeys([*SITE_FIELDS, "scenario_id", "operational_day_id", "active_units",
                                          "modeled_residential_units", "service_active_unit_factor"]))
        tables = {"demand/parcels.parquet": pd.DataFrame(all_parcels, columns=parcel_columns),
                  "demand/location_dispatch.parquet": pd.DataFrame(all_sites, columns=site_columns),
                  "demand/operational_days.parquet": pd.DataFrame(days),
                  "views/views.parquet": pd.DataFrame(views),
                  "activation/scenarios.parquet": pd.DataFrame(scenario_rows),
                  "activation/territories.parquet": pd.DataFrame(territorial_rows,
                      columns=["scenario_id", "location_id", "owner", "probability", "road_cost_km"]),
                  "activation/selections.parquet": pd.DataFrame(selection_rows,
                      columns=["scenario_id", "location_id", "rank"]),
                  "demand/generation_failures.parquet": pd.DataFrame(all_failures,
                      columns=["scenario_id", "day_index", "location_id", "proposal_index", "proposed_active_units",
                               "proposed_package_count", "proposed_volume_cm3", "source_donor_id", "reason",
                               "source_parcel_refs_json", "exhausted"])}
        for name, frame in tables.items():
            frame.to_parquet(temporary / name, index=False, compression="zstd")
        write_json(temporary / "demand/demand.json", {
            "schema": "evrptw_territory_order_process_v1", "process_id": process_id,
            "world_id": context.info["world_id"], "parcel_count": len(all_parcels),
            "location_day_count": len(all_sites), "capacity_model": "fixed_subset_then_independent_truncated_whole_orders",
            "operations": config["operations"],
            "maximum_full_proposals": config["workload"]["maximum_full_proposals"],
            "donor_source_sha256": donors.manifest["source_sha256"], "donor_calibration": donors.calibration_summary})
        write_json(temporary / "views/views.json", {"nested_prefixes": False,
            "sampler": "territory_first_conditional_bernoulli", "full_site_bundles_preserved": True})
        def relative(value):
            if isinstance(value, dict):
                return {k: relative(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relative(v) for v in value]
            if isinstance(value, str) and value.startswith(str(temporary) + "/"):
                return str(Path(value).relative_to(temporary))
            return value
        for path in temporary.rglob("*.json"):
            write_json(path, relative(json.loads(path.read_text())))
        source = context.info["source"]
        if code_files != {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}:
            raise RuntimeError("Source changed during generation; partial artifacts retained")
        write_json(temporary / "manifest.json", {
            "schema": "evrptw_stage2_territory_run_v1", "run_id": identity("territory-run-v1", config, code_files),
            "world_id": context.info["world_id"], "process_id": process_id, "input": config["input"],
            "config": "config.json", "source_sha256": {"cle_manifest": source["cle_manifest_sha256"],
                "cle_verification": source["cle_verification_sha256"],
                "freeze": source["freeze_sha256"], "community": source["community_sha256"],
                "donor_manifest": sha256_file(Path(config["input"]["donor_artifact"]) / "manifest.json")},
            "tables": {"parcels": "demand/parcels.parquet", "operational_days": "demand/operational_days.parquet",
                       "views": "views/views.parquet"}, "instances": relative(instances),
            "files": output_hashes(temporary), "stage1_modified": False, "stage2_frozen": False,
            "public_release_performed": False, "route_solution_claimed": False,
            "independent_sizes": True, "build_seconds": time.monotonic() - started,
            "code_provenance": {"package": "evrptw_scenarios", "version": "territory_v1",
                                "source_files_sha256": code_files}})
        temporary.rename(destination)
    except Exception as error:
        if temporary.exists():
            write_json(temporary / "build_failure.json", {"status": "failed", "error_type": type(error).__name__,
                "message": str(error), "depot_redraw_count": 0, "source_data_modified": False})
        raise
    from .verify import verify_run
    progress("verifying saved territory-day artifacts", flush=True)
    return verify_run(destination)
