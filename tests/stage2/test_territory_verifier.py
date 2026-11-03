"""Tiny saved-artifact fixtures; no production generator is called."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from evrptw_scenarios.reader import ScenarioRun
from evrptw_scenarios.util import output_hashes, sha256_file, write_json
from evrptw_scenarios.verify import verify_run


def _parquet(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, index=False)
    return frame


def _refresh(root):
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"] = output_hashes(root, exclude=["manifest.json", "verification.json"])
    write_json(root/"manifest.json", manifest)


@pytest.fixture
def territory_run(tmp_path):
    root, cle, donors = (tmp_path/name for name in ["run", "cle", "donors"])
    for folder in [root, cle, donors]:
        folder.mkdir()
    canonical = _parquet(cle/"locations.parquet", [
        {"location_id": "depot", "location_type": "depot", "candidate_eligible": True,
         "residential_units": np.nan, "residential_type": "not_applicable",
         "location_lon": -117., "location_lat": 34.},
        {"location_id": "a", "location_type": "customer", "candidate_eligible": True,
         "residential_units": 1., "residential_type": "house",
         "location_lon": -116.999, "location_lat": 34.},
        {"location_id": "b", "location_type": "customer", "candidate_eligible": True,
         "residential_units": 2., "residential_type": "small_apt",
         "location_lon": -117.001, "location_lat": 34.},
    ])
    write_json(cle/"cle_manifest.json", {"schema": "evrptw_city_logistics_environment_v1",
        "city_slug": "fixture", "freeze_authority": "freeze.json",
        "files": {"locations.parquet": {"sha256": sha256_file(cle/"locations.parquet")}}})
    cle_hash = sha256_file(cle/"cle_manifest.json")
    write_json(cle/"cle_verification.json", {"passed": True, "cle_manifest_sha256": cle_hash})
    write_json(cle/"freeze.json", {"frozen": True, "cities": {"fixture": {
        "cle_manifest_sha256": cle_hash, "verification_sha256": sha256_file(cle/"cle_verification.json")}}})
    source_parcel = {"source_parcel_ref": "raw", "source_event_ref": "event",
        "source_station_day_id": "station-day", "source_route_ref": "route", "source_date": "2018-07-19",
        "source_kind": "delivered", "individual_marks_valid": True, "donor_id": "bundle",
        "source_tw_status": "missing", "depth_cm": 1., "height_cm": 2., "width_cm": 5.,
        "volume_cm3": 10., "source_planned_service_s": 5.,
        "source_tw_start_offset_s": np.nan, "source_tw_end_offset_s": np.nan}
    _parquet(donors/"parcels.parquet", [source_parcel])
    _parquet(donors/"bundles.parquet", [{"donor_id": "bundle", "package_count": 1,
        "source_kind": "delivered", "individual_marks_valid": True}])
    _parquet(donors/"quarantine.parquet", [], columns=["reason"])
    write_json(donors/"manifest.json", {"schema": "evrptw_unconditioned_joint_marks_v1",
        "source_sha256": {}, "source_paths": {}, "outputs_sha256": output_hashes(donors)})
    community = tmp_path/"communities.parquet"
    _parquet(community, [{"bg_geoid": "bg"}])
    op = {"start_date": "2026-11-02", "days": 1, "shift_start_s": 0., "shift_end_s": 100.,
        "ready_window_s": [0., 0.], "source_window_end": "service_start",
        "missing_window_policy": "operating_horizon"}
    world = {"seed": 1, "provider_id": "fixture", "depot_id": "depot", "external_radius_km": 1.,
             "p_unit": .1, "mean_parcels_per_active_unit": 1.4}
    workload = {"cargo_capacity_cm3": 100., "maximum_full_proposals": 2,
        "service_a_s": 2., "service_b_s": 3., "service_alpha": 1., "service_support_max_count": 16,
        "service_tail_gamma": .5, "service_scale": 1., "service_active_unit_share": .5,
        "service_active_unit_exponent": .5}
    vehicle = {"cargo_capacity_cm3": 100., "specific_energy_kwh_per_km": .4,
               "battery_capacity_kwh": 100., "initial_soc": 1., "reserve_soc": .1}
    config = {"schema": "evrptw_territory_first_config_v1",
        "input": {"cle": str(cle), "communities": str(community), "donor_artifact": str(donors)},
        "world": world, "activation": {"max_centers": 2}, "workload": workload, "operations": op,
        "instances": {"sizes": [2], "materialize_day_indices": []},
        "physical": {"vehicle": vehicle, "charging": {
            "external_cs_budget": 0, "supply_policy": "all_compatible_catchment"}}}
    write_json(root/"config.json", config)
    catalog = canonical.loc[canonical.location_type.eq("customer")].copy()
    catalog["world_id"] = "world"
    catalog["modeled_residential_units"] = [1, 2]
    catalog["units_imputed"] = False
    catalog["expected_daily_activation_probability"] = [.1, .19]
    catalog["depot_outward_distance_m"] = [20000., 21000.]
    catalog["depot_inward_distance_m"] = [22000., 23000.]
    dlon = np.deg2rad(.001)
    gc = 2*6371.0088*np.arcsin(np.sqrt(np.cos(np.deg2rad(34.))**2*np.sin(dlon/2)**2))
    catalog["depot_geodesic_distance_km"] = gc
    catalog.to_parquet(root/"world-catalog.tmp", index=False)
    (root/"world").mkdir()
    (root/"world-catalog.tmp").rename(root/"world/catalog.parquet")
    depot = canonical.loc[canonical.location_type.eq("depot")].copy()
    depot.to_parquet(root/"world/depot.parquet", index=False)
    write_json(root/"world/world.json", {"world_id": "world", "depot_id": "depot", "config": world,
        "source": {"cle_manifest_sha256": cle_hash}, "depot_redraw_count": 0,
        "terminal_through_routing_allowed": False, "customer_count": 2,
        "geography_policy": "great_circle_and_unbounded_bidirectional_reachability"})
    scenario = {"scenario_id": "scenario", "day_index": 0, "date": op["start_date"],
        "requested_n": 2, "selected_n": 2, "supported_size": True, "selected_k": 1,
        "expected_mass": .29, "activation_eta": None, "activation_attempts": 0,
        "activation_proposal_variance": 0., "reference_length_km": 1.}
    _parquet(root/"activation/scenarios.parquet", [scenario])
    _parquet(root/"activation/territories.parquet", [
        {"scenario_id": "scenario", "location_id": loc, "owner": 0, "probability": pi, "road_cost_km": cost}
        for loc, pi, cost in [("a", .1, .1), ("b", .19, .2)]])
    _parquet(root/"activation/selections.parquet", [
        {"scenario_id": "scenario", "location_id": loc, "rank": i} for i, loc in enumerate(["a", "b"])])
    sites, parcels = [], []
    for loc, units in [("a", 1), ("b", 2)]:
        parcel = {k: v for k, v in source_parcel.items() if k not in
                  {"donor_id", "source_kind", "individual_marks_valid"}}
        parcel.update(scenario_id="scenario", parcel_id="parcel-"+loc, location_id=loc,
            source_donor_id="bundle", source_package_count=1, exact_source_count=True,
            arrival_day_index=0, arrival_date=op["start_date"], dispatch_day_index=0, scheduled_day_index=0,
            promise_day_index=0, promise_s=100., ready_s=0., commitment_failure=False,
            active_units=1, modeled_residential_units=units, assigned_planned_service_s=5.,
            location_tw_status="missing", location_tw_start_offset_s=np.nan,
            location_tw_end_offset_s=np.nan, location_tw_source_donor_id=None,
            capacity_conditioning_attempts=1)
        parcels.append(parcel)
        sites.append({"scenario_id": "scenario", "day_index": 0, "location_id": loc,
            "generation_supported": True, "parcel_count": 1, "demand_cm3": 10.,
            "planned_service_time_s": 5., "tw_start_s": 0., "tw_end_s": 95.,
            "source_specified_window_count": 0, "source_missing_window_count": 1,
            "supported_one_visit_representation": True, "reasons": "[]", "capacity_conditioning_attempts": 1})
    _parquet(root/"demand/parcels.parquet", parcels)
    _parquet(root/"demand/location_dispatch.parquet", sites)
    _parquet(root/"demand/operational_days.parquet", [{"scenario_id": "scenario", "day_index": 0,
        "date": op["start_date"], "active_locations": 2, "arrivals": 2, "dispatched": 2,
        "previous_backlog": 0, "backlog": 0, "route_solution_claimed": False}])
    _parquet(root/"demand/generation_failures.parquet", [], columns=["scenario_id", "location_id",
        "proposal_index", "proposed_volume_cm3", "exhausted"])
    write_json(root/"demand/demand.json", {"world_id": "world", "process_id": "process",
        "operations": op, "parcel_count": 2, "donor_source_sha256": {}})
    view = {"view_id": "scenario", "day_index": 0, "kind": "territory_day", "requested_n": 2,
        "selected_n": 2, "supported_size": True, "parent_operational_day_id": "scenario", "materialize_matrix": False}
    _parquet(root/"views/views.parquet", [view])
    _parquet(root/"physical/chargers.parquet", [], columns=["location_id"])
    write_json(root/"physical/physical.json", {"config": config["physical"], "world_id": "world", "charger_count": 0})
    terminals = [{"node_index": 0, "location_id": "depot", "kind": "depot", "demand_cm3": 0.,
        "service_time_s": 0., "parcel_count": 0, "tw_start_s": 0., "tw_end_s": 100.}]
    terminals += [{"node_index": i+1, "location_id": loc, "kind": "customer", "demand_cm3": 10.,
        "service_time_s": 5., "parcel_count": 1, "tw_start_s": 0., "tw_end_s": 95.} for i, loc in enumerate(["a", "b"])]
    _parquet(root/"instances/scenario/terminals.parquet", terminals)
    charging = {"selected_location_ids": []}
    write_json(root/"instances/scenario/charging_selection.json", charging)
    write_json(root/"instances/scenario/instance.json", {"schema": "evrptw_scenario_instance_v1", **view,
        "world_id": "world", "process_id": "process", "status": "not_materialized",
        "matrix_materialized": False, "node_order": ["depot", "a", "b"], "vehicle": vehicle,
        "operating_window_s": [0., 100.], "charging_selection": charging,
        "assessment": {"route_solution_claimed": False, "unsupported_generation_locations": 0,
                       "unsupported_one_visit_locations": 0, "oversize_bundles": 0}})
    write_json(root/"manifest.json", {"schema": "evrptw_stage2_territory_run_v1", "run_id": "run",
        "world_id": "world", "process_id": "process", "input": config["input"],
        "source_sha256": {"cle_manifest": cle_hash, "freeze": sha256_file(cle/"freeze.json"),
            "donor_manifest": sha256_file(donors/"manifest.json"), "community": sha256_file(community)},
        "tables": {"parcels": "demand/parcels.parquet", "operational_days": "demand/operational_days.parquet",
                   "views": "views/views.parquet"}, "instances": {"scenario": "instances/scenario/instance.json"},
        "stage1_modified": False, "route_solution_claimed": False, "files": {}})
    _refresh(root)
    return root


def test_new_schema_dispatch_reader_and_unbounded_roads(territory_run):
    result = verify_run(territory_run, check_sources=True)
    assert result["passed"], result["errors"]
    assert result["metrics"]["selected_locations"] == 2
    assert not result["scope"]["growth_distribution_recreated"]
    run = ScenarioRun(territory_run)
    assert len(run.parcels()) == 2
    assert run.instance("scenario")["metadata"]["node_order"] == ["depot", "a", "b"]


@pytest.mark.parametrize("attack", ["selection", "units", "geography", "service", "inventory"])
def test_semantic_tampering_detected_after_hash_refresh(territory_run, attack):
    if attack == "inventory":
        (territory_run/"unlisted.txt").write_text("unmanifested")
    else:
        file, column, value = {
            "selection": ("activation/selections.parquet", "location_id", "outside"),
            "units": ("world/catalog.parquet", "modeled_residential_units", 17),
            "geography": ("world/catalog.parquet", "depot_geodesic_distance_km", 900.),
            "service": ("demand/parcels.parquet", "assigned_planned_service_s", 999.),
        }[attack]
        table = pd.read_parquet(territory_run/file)
        table.loc[0, column] = value
        table.to_parquet(territory_run/file, index=False)
        _refresh(territory_run)
    result = verify_run(territory_run)
    assert not result["passed"]


def test_failed_order_keeps_selected_id_and_explicit_status(territory_run):
    root = territory_run
    parcels = pd.read_parquet(root/"demand/parcels.parquet")
    parcels = parcels.loc[parcels.location_id.eq("a")]
    parcels.to_parquet(root/"demand/parcels.parquet", index=False)
    sites = pd.read_parquet(root/"demand/location_dispatch.parquet")
    sites.loc[1, ["generation_supported", "supported_one_visit_representation"]] = False
    sites.loc[1, "parcel_count"] = 0
    sites.loc[1, ["demand_cm3", "planned_service_time_s"]] = np.nan
    sites.loc[1, "capacity_conditioning_attempts"] = 2
    sites.loc[1, "reasons"] = '["capacity_proposal_limit_exhausted"]'
    sites.to_parquet(root/"demand/location_dispatch.parquet", index=False)
    _parquet(root/"demand/generation_failures.parquet", [{"scenario_id": "scenario", "location_id": "b",
        "proposal_index": i, "proposed_volume_cm3": 200., "exhausted": i == 2} for i in [1, 2]])
    days = pd.read_parquet(root/"demand/operational_days.parquet")
    days.loc[0, ["arrivals", "dispatched"]] = 1
    days.to_parquet(root/"demand/operational_days.parquet", index=False)
    info = json.loads((root/"demand/demand.json").read_text())
    info["parcel_count"] = 1
    write_json(root/"demand/demand.json", info)
    terminals = pd.read_parquet(root/"instances/scenario/terminals.parquet")
    terminals.loc[2, "parcel_count"] = 0
    terminals.loc[2, ["demand_cm3", "service_time_s"]] = np.nan
    terminals.to_parquet(root/"instances/scenario/terminals.parquet", index=False)
    instance_path = root/"instances/scenario/instance.json"
    instance = json.loads(instance_path.read_text())
    instance["status"] = "unsupported_workload"
    instance["assessment"].update(unsupported_generation_locations=1, unsupported_one_visit_locations=1)
    write_json(instance_path, instance)
    _refresh(root)
    result = verify_run(root)
    assert result["passed"], result["errors"]
    assert result["metrics"]["generation_unsupported_location_days"] == 1
    assert len(pd.read_parquet(root/"activation/selections.parquet")) == 2
    # Deleting the failed location is a semantic error, even with new hashes.
    sites.iloc[:1].to_parquet(root/"demand/location_dispatch.parquet", index=False)
    _refresh(root)
    assert not verify_run(root)["passed"]


def test_verifier_does_not_import_generation_modules():
    import ast
    from pathlib import Path
    import evrptw_scenarios.verify_territory as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not {"build", "territory_generation", "activation", "territory", "capacity", "routing"}.intersection(imports)


def test_same_day_different_sizes_have_independent_scenario_keys(territory_run):
    root = territory_run
    config = json.loads((root/"config.json").read_text())
    config["instances"]["sizes"] = [1, 2]
    write_json(root/"config.json", config)
    for filename in ["activation/scenarios.parquet", "activation/territories.parquet",
                     "activation/selections.parquet", "demand/parcels.parquet",
                     "demand/location_dispatch.parquet", "demand/operational_days.parquet",
                     "views/views.parquet"]:
        table = pd.read_parquet(root/filename)
        extra = table.iloc[:1].copy()
        for field in ["scenario_id", "view_id", "parent_operational_day_id"]:
            if field in extra:
                extra[field] = "other"
        for field in ["requested_n", "selected_n", "active_locations", "arrivals", "dispatched"]:
            if field in extra:
                extra[field] = 1
        if "parcel_id" in extra:
            extra["parcel_id"] = "other-parcel"
        if "expected_mass" in extra:
            extra["expected_mass"] = .1
        pd.concat([table, extra], ignore_index=True).to_parquet(root/filename, index=False)
    demand = json.loads((root/"demand/demand.json").read_text())
    demand["parcel_count"] = 3
    write_json(root/"demand/demand.json", demand)
    folder = root/"instances/other"
    folder.mkdir()
    terminals = pd.read_parquet(root/"instances/scenario/terminals.parquet").iloc[:2]
    terminals.to_parquet(folder/"terminals.parquet", index=False)
    info = deepcopy(json.loads((root/"instances/scenario/instance.json").read_text()))
    info.update(view_id="other", parent_operational_day_id="other", requested_n=1, selected_n=1,
                node_order=["depot", "a"])
    write_json(folder/"instance.json", info)
    write_json(folder/"charging_selection.json", info["charging_selection"])
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["instances"]["other"] = "instances/other/instance.json"
    write_json(root/"manifest.json", manifest)
    _refresh(root)
    result = verify_run(root)
    assert result["passed"], result["errors"]
    assert result["metrics"]["scenarios"] == 2


def test_nullable_unsupported_size_is_preserved(territory_run):
    root = territory_run
    config = json.loads((root/"config.json").read_text())
    config["instances"]["sizes"] = [3]
    write_json(root/"config.json", config)
    scenario = pd.read_parquet(root/"activation/scenarios.parquet")
    scenario.loc[0, ["requested_n", "selected_n", "supported_size"]] = [3, 0, False]
    scenario.loc[0, ["selected_k", "expected_mass", "activation_proposal_variance"]] = np.nan
    scenario.to_parquet(root/"activation/scenarios.parquet", index=False)
    for file in ["activation/territories.parquet", "activation/selections.parquet",
                 "demand/parcels.parquet", "demand/location_dispatch.parquet"]:
        table = pd.read_parquet(root/file)
        table.iloc[:0].to_parquet(root/file, index=False)
    days = pd.read_parquet(root/"demand/operational_days.parquet")
    days.loc[0, ["active_locations", "arrivals", "dispatched"]] = 0
    days.to_parquet(root/"demand/operational_days.parquet", index=False)
    demand = json.loads((root/"demand/demand.json").read_text())
    demand["parcel_count"] = 0
    write_json(root/"demand/demand.json", demand)
    view = pd.read_parquet(root/"views/views.parquet")
    view.loc[0, ["requested_n", "selected_n", "supported_size"]] = [3, 0, False]
    view.to_parquet(root/"views/views.parquet", index=False)
    info = json.loads((root/"instances/scenario/instance.json").read_text())
    info.update(requested_n=3, selected_n=0, supported_size=False,
                node_order=["depot"], status="unsupported_size")
    write_json(root/"instances/scenario/instance.json", info)
    terminals = pd.read_parquet(root/"instances/scenario/terminals.parquet")
    terminals.iloc[:1].to_parquet(root/"instances/scenario/terminals.parquet", index=False)
    _refresh(root)
    result = verify_run(root)
    assert result["passed"], result["errors"]


def test_paired_matrix_policy_mismatch_is_detected(territory_run):
    root = territory_run
    view = pd.read_parquet(root/"views/views.parquet")
    view.loc[0, "materialize_matrix"] = True
    view.to_parquet(root/"views/views.parquet", index=False)
    info = json.loads((root/"instances/scenario/instance.json").read_text())
    info.update(matrix_materialized=True, materialize_matrix=True, status="materialized")
    write_json(root/"instances/scenario/instance.json", info)
    distance = np.array([[0., 100., 200.], [100., 0., 100.], [200., 100., 0.]])
    time = distance/10
    integer = np.zeros((3, 3), dtype=np.int64)
    mode = integer.copy()
    np.fill_diagonal(mode, 2)
    arrays = {"distance_m": distance, "time_s": time, "energy_kwh": distance*.0004,
        "distance_shortest_path_distance": distance, "distance_shortest_path_time": time,
        "time_shortest_path_distance": distance*1.2,
        "time_shortest_path_time": time*2,  # Invalid: fastest takes longer.
        "time_shortest_path_energy_kwh": distance*1.2*.0004,
        "source_option": integer, "destination_option": integer, "path_mode": mode,
        "time_shortest_path_source_option": integer, "time_shortest_path_destination_option": integer,
        "time_shortest_path_mode": mode}
    np.savez_compressed(root/"instances/scenario/costs.npz", **arrays)
    _refresh(root)
    result = verify_run(root)
    assert not result["passed"]
    assert any("Fastest time exceeds shortest-distance-path time" in error for error in result["errors"])
