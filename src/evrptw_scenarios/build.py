"""Materialize a demand-first run without retrying an inconvenient workload."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .charging_selection import select_joint_chargers
from .config import JOINT_CHARGING_POLICY, joint_station_budget
from .demand import build_demand
from .marks import DonorLibrary
from .physical import bind_physical, select_chargers
from .routing import FrozenCostRouter
from .util import identity, output_hashes, sha256_file, write_json
from .views import build_views
from .world import build_world


def _terminals(depot, catalog, chargers, site_rows, ids, day_index, operations):
    customer = catalog.set_index("location_id").loc[ids].reset_index()
    day = site_rows.set_index("location_id").loc[ids]
    customer["kind"] = "customer"
    for column, source in [("demand_cm3", "demand_cm3"), ("service_time_s", "planned_service_time_s"),
                           ("tw_start_s", "tw_start_s"), ("tw_end_s", "tw_end_s"),
                           ("parcel_count", "parcel_count")]:
        customer[column] = day[source].to_numpy()
    for column in ["tw_start_s", "tw_end_s"]:
        customer[column] -= day_index * 86400
    depot = depot.copy()
    depot["kind"] = "depot"
    chargers = chargers.copy()
    chargers["kind"] = "charging"
    chargers["power_usable_kw"] = chargers["effective_power_kw"]
    chargers["power_capped_pre_derating_kw"] = chargers["stage2_capped_pre_derating_kw"]
    for frame in [depot, chargers]:
        frame["demand_cm3"] = 0.
        frame["service_time_s"] = 0.
        frame["tw_start_s"] = float(operations["shift_start_s"])
        frame["tw_end_s"] = float(operations["shift_end_s"])
        frame["parcel_count"] = 0
    # Preserve canonical native source fields while removing geometry objects.
    result = pd.concat([depot, customer, chargers], ignore_index=True, sort=False)
    if "geometry" in result:
        result = result.drop(columns="geometry")
    result = result.copy()
    result.insert(0, "node_index", np.arange(len(result)))
    if not result.location_id.is_unique:
        raise ValueError("Instance canonical location IDs are not unique")
    return result


def materialize_instances(config, world, demand, views, physical, out, *, progress=print):
    out = Path(out)
    catalog = pd.read_parquet(world["catalog_path"])
    depot = pd.read_parquet(world["depot_path"])
    charger_path = physical.get("chargers_path", physical.get("charger_path"))
    chargers = pd.read_parquet(charger_path)
    sites = pd.read_parquet(demand["sites_path"])
    table = pd.read_parquet(views["views_path"])
    order = pd.read_parquet(views["selection_order_path"])
    ledger = pd.read_parquet(Path(config["input"]["cle"]) / "location_access_costs.parquet")
    vehicle = config["physical"]["vehicle"]
    rate = vehicle["specific_energy_kwh_per_km"]
    routers = {}
    indexes = {}
    output = {}
    for day_index, group in table.groupby("day_index", sort=True):
        day_index = int(day_index)
        selected_order = order.loc[order.day_index.eq(day_index)].sort_values("rank").location_id.astype(str).tolist()
        today = sites.loc[sites.day_index.eq(day_index)]
        wanted = group.loc[group.materialize_matrix & group.supported_size]
        largest = int(wanted.selected_n.max()) if len(wanted) else 0
        cost = None
        current_date = date.fromisoformat(config["operations"]["start_date"]) + timedelta(days=day_index)
        # Compute a single maximum prefix, then use exact index views for N.
        if len(wanted):
            day_type = "weekend" if current_date.weekday() >= 5 else "weekday"
            if day_type not in routers:
                columns = ["edge_u", "edge_v", "edge_key", "length_m", "in_public_graph",
                           "operating_mode", f"reference_travel_time_{day_type}_s",
                           f"reference_speed_{day_type}_kph"]
                speeds = pd.read_parquet(Path(config["input"]["cle"]) / "speeds/directed_speeds.parquet", columns=columns)
                routers[day_type] = FrozenCostRouter(speeds, day_type=day_type)
            router = routers[day_type]
            maximum = _terminals(depot, catalog, chargers, today, selected_order[:largest], day_index, config["operations"])
            endpoints = router.bind_endpoints(maximum, ledger)
            pairs = sorted({(0, i) for i in range(len(maximum))} | {(i, 0) for i in range(len(maximum))})
            progress(f"routing day {day_index}: {largest} customer locations + {len(chargers)} charging locations", flush=True)
            cost = router.route(endpoints, specific_energy_kwh_per_km=rate, audit_pairs=pairs, require_all_reachable=False)
            indexes[day_index] = {loc: i for i, loc in enumerate(cost.location_ids)}
        for view in group.to_dict("records"):
            directory = out / str(view["view_id"])
            directory.mkdir(parents=True, exist_ok=False)
            supported = bool(view["supported_size"])
            n = int(view["selected_n"])
            ids = selected_order[:n] if supported else []
            selected_sites = today.loc[today.location_id.isin(ids)]
            charging = config["physical"]["charging"]
            policy = charging["supply_policy"]
            budget = (joint_station_budget(charging, int(view["requested_n"]))
                      if policy == JOINT_CHARGING_POLICY else charging["external_cs_budget"])
            if policy == "all_compatible_catchment":
                selected_chargers = chargers
                selection = {"policy": "all_compatible_catchment", "supported_budget": True,
                             "candidate_count": len(chargers),
                             "selected_location_ids": chargers.location_id.astype(str).tolist()}
            elif bool(view["materialize_matrix"]) and supported:
                depot_id = str(depot.location_id.iloc[0])
                if policy == JOINT_CHARGING_POLICY:
                    key = "bg_geoid" if "bg_geoid" in catalog else "community_id"
                    communities = catalog[["location_id", key]].set_index("location_id").loc[ids, key]
                    selected_chargers, selection = select_joint_chargers(
                        chargers, ids, communities, cost, budget=budget, depot_id=depot_id,
                        vehicle=vehicle, reference_departure_soc=charging["reference_departure_soc"])
                else:
                    selected_chargers, selection = select_chargers(
                        chargers, ids, cost, budget=budget, depot_id=depot_id)
            else:
                selected_chargers = chargers.iloc[:0].copy()
                selection = {"policy": policy, "status": "not_selected_without_materialized_roads",
                             "supported_budget": None, "requested_budget": budget,
                             "candidate_count": len(chargers), "selected_location_ids": []}
            write_json(directory / "charging_selection.json", selection)
            terminals = _terminals(depot, catalog, selected_chargers, today, ids, day_index, config["operations"])
            terminals.to_parquet(directory / "terminals.parquet", index=False, compression="zstd")
            bad_tw = int((~selected_sites.supported_one_visit_representation.astype(bool)).sum())
            oversize = int(selected_sites.demand_cm3.gt(vehicle["cargo_capacity_cm3"]).sum())
            generation_failures = int((~selected_sites.generation_supported.astype(bool)).sum()) if "generation_supported" in selected_sites else 0
            metadata = {"schema": "evrptw_scenario_instance_v1", **view,
                        "world_id": world["world_id"], "process_id": demand["process_id"],
                        "date": (pd.Timestamp(config["operations"]["start_date"]) + pd.Timedelta(days=day_index)).date().isoformat(),
                        "status": "unsupported_size" if not supported else "not_materialized",
                        "matrix_materialized": False, "node_order": terminals.location_id.astype(str).tolist(),
                        "vehicle": vehicle,
                        "operating_window_s": [config["operations"]["shift_start_s"], config["operations"]["shift_end_s"]],
                        "charging_assumptions": config["physical"]["charging"],
                        "charging_selection": selection,
                        "primary_objective": "total_distance",
                        "assessment": {"unsupported_one_visit_locations": bad_tw,
                                       "oversize_bundles": oversize,
                                       "unsupported_generation_locations": generation_failures,
                                       "direct_single_stop_certificate_count": None,
                                       "route_solution_claimed": False,
                                       "single_trip_no_split_delivery": True},
                        "cost_policy": {"path_policy": "shortest_distance_weekend" if pd.Timestamp(current_date).weekday() >= 5 else "shortest_distance_weekday",
                                        "day_type": "weekend" if pd.Timestamp(current_date).weekday() >= 5 else "weekday",
                                        "execution_policy": "shortest_distance", "objective": "total_distance",
                                        "connector_policy": "public_U_length_weighted_median_else_all_public",
                                        "lazy_reconstruction": True}}
            if view["materialize_matrix"] and supported:
                ix = np.array([indexes[day_index][loc] for loc in metadata["node_order"]], dtype=int)
                fields = cost.matrix_fields
                arrays = {name: getattr(cost, name)[np.ix_(ix, ix)] for name in fields}
                np.savez_compressed(directory / "costs.npz", **arrays)
                remap = {int(original): new for new, original in enumerate(ix)}
                for filename, records in [("path_audit.json", cost.audit_paths),
                                           ("fastest_path_audit.json", cost.fastest_audit_paths)]:
                    audit = {}
                    for key, value in records.items():
                        i, j = map(int, key.split(","))
                        if i in remap and j in remap:
                            value = {**value}
                            for field in ["distance_m", "time_s", "energy_kwh"]:
                                if not np.isfinite(value[field]):
                                    value[field] = None
                            audit[f"{remap[i]},{remap[j]}"] = value
                    write_json(directory / filename, audit)
                reachable = np.isfinite(arrays["time_s"]).all()
                metadata["status"] = "materialized" if reachable else "numeric_unreachable"
                metadata["matrix_materialized"] = True
                metadata["cost_policy"] = cost.metadata
                metadata["cost_policy"] = {**cost.metadata, "unreachable_directed_pairs": int((~np.isfinite(arrays["time_s"])).sum())}
                if generation_failures or oversize or bad_tw:
                    metadata["status"] = "unsupported_workload"
                elif selection.get("supported_budget") is False:
                    metadata["status"] = "unsupported_charging_budget"
                earliest = np.maximum(terminals.tw_start_s.to_numpy(dtype=float)[1:n + 1],
                                      config["operations"]["shift_start_s"] + arrays["time_s"][0, 1:n + 1])
                single = ((earliest <= terminals.tw_end_s.to_numpy(dtype=float)[1:n + 1]) &
                          (earliest + terminals.service_time_s.to_numpy(dtype=float)[1:n + 1] + arrays["time_s"][1:n + 1, 0] <= config["operations"]["shift_end_s"]) &
                          (terminals.demand_cm3.to_numpy(dtype=float)[1:n + 1] <= vehicle["cargo_capacity_cm3"]) &
                          (arrays["energy_kwh"][0, 1:n + 1] + arrays["energy_kwh"][1:n + 1, 0] <= vehicle["battery_capacity_kwh"] * (vehicle["initial_soc"] - vehicle["reserve_soc"])))
                metadata["assessment"]["direct_single_stop_certificate_count"] = int(single.sum())
                metadata["assessment"]["certificate_meaning"] = "sufficient individual no-charge depot-stop-depot witness; failure is not proof of EVRPTW infeasibility"
            write_json(directory / "instance.json", metadata)
            output[str(view["view_id"])] = str(directory / "instance.json")
        # Keep only one day closure, avoiding an accidental horizon-sized RAM cache.
        cost = None
    return output


def build_run(config, *, progress=print):
    """Build to a new directory, retaining explicit unsupported states, then verify."""
    if config.get("schema") == "evrptw_territory_first_config_v1":
        from .territory_generation import build_territory_run
        return build_territory_run(config, progress=progress)
    destination = Path(config["output"]["path"]).resolve()
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Output or partial directory exists; choose a new output path")
    temporary.mkdir(parents=True)
    code_files = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    started = time.monotonic()
    try:
        write_json(temporary / "config.json", config)
        progress("verifying frozen CLE and constructing fixed world", flush=True)
        world = build_world(config["input"]["cle"], config["input"]["communities"], config["world"], temporary / "world")
        progress(f"world: {world['customer_count']} locations; constructing arrival ledger", flush=True)
        donors = DonorLibrary(config["input"]["donor_artifact"])
        demand = build_demand(world, config, donors, temporary / "demand", progress=progress)
        del donors
        progress(f"demand: {demand['parcel_count']} parcels; constructing views and physical supply", flush=True)
        views = build_views(demand, config, temporary / "views")
        physical = bind_physical(config["input"]["cle"], world, config["physical"], temporary / "physical")
        instances = materialize_instances(config, world, demand, views, physical, temporary / "instances", progress=progress)
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
        source = world["source"]
        current_code = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        if current_code != code_files:
            raise RuntimeError("Stage 2 source changed during generation; partial output retained, run again with stable source")
        manifest = {"schema": "evrptw_stage2_run_v1", "run_id": identity("run-v1", world["world_id"], demand["process_id"], config["views"], config["physical"], code_files),
                    "world_id": world["world_id"], "process_id": demand["process_id"],
                    "input": config["input"], "config": "config.json",
                    "source_sha256": {"cle_manifest": source["cle_manifest_sha256"], "freeze": source["freeze_sha256"],
                                      "cle_verification": source["cle_verification_sha256"],
                                      "donor_manifest": sha256_file(Path(config["input"]["donor_artifact"]) / "manifest.json"),
                                      "community": source["community_sha256"]},
                    "tables": {"parcels": "demand/parcels.parquet", "operational_days": "demand/operational_days.parquet", "views": "views/views.parquet"},
                    "instances": relative(instances), "files": output_hashes(temporary),
                    "stage1_modified": False, "stage2_frozen": False,
                    "code_provenance": {"package": "evrptw_scenarios", "version": "0.1.0", "source_files_sha256": code_files},
                    "public_release_performed": False, "route_solution_claimed": False,
                    "build_seconds": time.monotonic() - started}
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception as exc:
        if temporary.exists():
            write_json(temporary / "build_failure.json", {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                                                          "demand_retried": False, "source_data_modified": False})
        raise
    from .verify import verify_run
    progress("independent saved-artifact verification", flush=True)
    return verify_run(destination)
