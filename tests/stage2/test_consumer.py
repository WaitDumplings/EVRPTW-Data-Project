import json

import numpy as np
import pandas as pd
import pytest

from evrptw_scenarios.consumer import UnsupportedSolverRepresentation, export_solver_input
from evrptw_scenarios.reader import ScenarioRun
from evrptw_scenarios.util import output_hashes, sha256_file, write_json


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "demand").mkdir()
    (root / "views").mkdir()
    folder = root / "instances/view"
    folder.mkdir(parents=True)
    pd.DataFrame({"parcel_id": ["parcel"], "volume_cm3": [100.]}).to_parquet(root / "demand/parcels.parquet", index=False)
    pd.DataFrame({"day_index": [0], "arrivals": [1], "dispatched": [1], "backlog": [0]}).to_parquet(root / "demand/operational_days.parquet", index=False)
    pd.DataFrame({"view_id": ["view"], "selected_n": [1]}).to_parquet(root / "views/views.parquet", index=False)
    terminals = pd.DataFrame({"node_index": [0, 1, 2], "location_id": ["depot", "customer", "charging"],
        "kind": ["depot", "customer", "charging"], "demand_cm3": [0., 100., 0.],
        "service_time_s": [0., 10., 0.], "tw_start_s": [0., 20., 0.], "tw_end_s": [100., 90., 100.],
        "power_usable_kw": [None, None, 50.]})
    terminals.to_parquet(folder / "terminals.parquet", index=False)
    distance = np.array([[0, 1000, 3000], [1300, 0, 500], [2900, 700, 0.]])
    time = np.array([[0, 100, 500], [120, 0, 80], [450, 90, 0.]])
    np.savez_compressed(folder / "costs.npz", distance_m=distance, time_s=time, energy_kwh=distance * .0004,
                        source_option=np.zeros((3, 3)), destination_option=np.zeros((3, 3)), path_mode=np.zeros((3, 3)))
    write_json(folder / "path_audit.json", {"0,1": {"distance_m": 1000, "time_s": 100}})
    metadata = {"schema": "evrptw_scenario_instance_v1", "view_id": "view", "world_id": "world", "process_id": "process",
        "date": "2026-11-02", "kind": "benchmark_view", "status": "materialized", "matrix_materialized": True,
        "node_order": ["depot", "customer", "charging"], "operating_window_s": [0, 100],
        "vehicle": {"cargo_capacity_cm3": 1000}, "cost_policy": {"distance_is_same_time_path": True},
        "assessment": {"unsupported_one_visit_locations": 0, "oversize_bundles": 0, "route_solution_claimed": False}}
    write_json(folder / "instance.json", metadata)
    manifest = {"schema": "evrptw_stage2_run_v1", "run_id": "run",
        "tables": {"parcels": "demand/parcels.parquet", "operational_days": "demand/operational_days.parquet", "views": "views/views.parquet"},
        "instances": {"view": "instances/view/instance.json"}, "files": output_hashes(root)}
    write_json(root / "manifest.json", manifest)
    return root


def refresh(root):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = output_hashes(root, exclude=["manifest.json"])
    write_json(root / "manifest.json", manifest)


def test_saved_reader_and_same_family_solver_export(saved_run, tmp_path):
    run = ScenarioRun(saved_run)
    assert len(run.parcels()) == len(run.operational_days()) == len(run.views()) == 1
    assert run.audited_path("view", 0, 1)["time_s"] == 100
    output = export_solver_input(saved_run, "view", tmp_path / "solver.json")
    result = json.loads(output.read_text())
    assert result["node_order"] == ["depot", "customer", "charging"]
    assert result["depot_index"] == 0 and result["customer_indices"] == [1] and result["charging_indices"] == [2]
    assert result["distance_matrix_km"][1][0] == 1.3
    assert result["travel_time_matrix_s"][1][0] == 120
    assert result["energy_matrix_kwh"][1][0] == .52
    assert not result["route_solution_claimed"]


@pytest.mark.parametrize("condition", ["oversize", "empty_tw", "assessment", "not_materialized", "unreachable"])
def test_unsupported_adapter_keeps_original_workload(saved_run, tmp_path, condition):
    folder = saved_run / "instances/view"
    parcel_bytes = (saved_run / "demand/parcels.parquet").read_bytes()
    if condition in {"oversize", "empty_tw"}:
        terminals = pd.read_parquet(folder / "terminals.parquet")
        terminals.loc[1, "demand_cm3" if condition == "oversize" else "tw_start_s"] = 2000
        terminals.to_parquet(folder / "terminals.parquet", index=False)
    elif condition in {"assessment", "not_materialized"}:
        metadata = json.loads((folder / "instance.json").read_text())
        if condition == "assessment":
            metadata["assessment"]["unsupported_one_visit_locations"] = 1
        else:
            metadata.update(status="not_materialized", matrix_materialized=False)
        write_json(folder / "instance.json", metadata)
    else:
        with np.load(folder / "costs.npz") as archive:
            costs = {key: archive[key] for key in archive.files}
        costs["time_s"][0, 1] = float("inf")
        np.savez_compressed(folder / "costs.npz", **costs)
    refresh(saved_run)
    with pytest.raises(UnsupportedSolverRepresentation):
        export_solver_input(saved_run, "view", tmp_path / "bad.json")
    assert (saved_run / "demand/parcels.parquet").read_bytes() == parcel_bytes
    assert not (tmp_path / "bad.json").exists()


def test_hash_tamper_rejected(saved_run):
    (saved_run / "demand/parcels.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        ScenarioRun(saved_run)


@pytest.mark.parametrize("attack", ["../outside", "/tmp/outside", "C:/outside", "a\\outside"])
def test_manifest_path_escape_rejected(saved_run, attack):
    manifest = json.loads((saved_run / "manifest.json").read_text())
    manifest["files"][attack] = "0" * 64
    write_json(saved_run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="path|Artifact"):
        ScenarioRun(saved_run)


def test_symlink_escape_rejected(saved_run, tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    (saved_run / "escaped").symlink_to(outside)
    manifest = json.loads((saved_run / "manifest.json").read_text())
    manifest["files"]["escaped"] = "0" * 64
    write_json(saved_run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="escapes"):
        ScenarioRun(saved_run)


@pytest.mark.parametrize("operating_mode", ["U", "R"])
def test_lazy_natural_path_replays_two_native_endpoints(saved_run, tmp_path, monkeypatch, operating_mode):
    cle = tmp_path / "frozen-cle"
    (cle / "speeds").mkdir(parents=True)
    speeds = pd.DataFrame({"edge_u": ["a", "b", "b", "c"], "edge_v": ["b", "a", "c", "b"],
        "edge_key": ["0"] * 4, "length_m": [1000.] * 4, "in_public_graph": [True] * 4,
        "reference_travel_time_weekday_s": [100.] * 4, "reference_speed_weekday_kph": [36.] * 4,
        "operating_mode": [operating_mode] * 4})
    speeds.to_parquet(cle / "speeds/directed_speeds.parquet", index=False)
    locations, ledger = [], []
    for kind, location, key, u, v, fraction in [
        ("depot", "depot", "candidate_id", "a", "b", 0.),
        ("customer", "customer", "site_id", "a", "b", 1.),
        ("charging", "charging", "charging_location_id", "b", "c", 1.),
    ]:
        native = {key: "native-" + kind, "requires_terminal_paths": False, "connector_length_m": 10.,
            "directed_projection_offsets": json.dumps([{"u": u, "v": v, "key": "0", "projection_fraction_from_u": fraction}]),
            "terminal_inbound_gateway": None, "terminal_outbound_gateway": None,
            "terminal_inbound_length_m": 0., "terminal_outbound_length_m": 0.,
            "terminal_inbound_path_json": "[]", "terminal_outbound_path_json": "[]"}
        table = kind + ".parquet"
        pd.DataFrame([native]).to_parquet(cle / table, index=False)
        locations.append({"location_id": location, "candidate_eligible": True, "location_type": kind,
                          "access_table": table, "access_row_id": "native-" + kind})
        ledger.append({"location_id": location, "connector_length_m": 10., "connector_time_s_per_inverse_kph": 36.,
            "inbound_terminal_length_m": 0., "outbound_terminal_length_m": 0.,
            "inbound_terminal_reference_weekday_time_s": 0., "outbound_terminal_reference_weekday_time_s": 0.})
    pd.DataFrame(locations).to_parquet(cle / "locations.parquet", index=False)
    pd.DataFrame(ledger).to_parquet(cle / "location_access_costs.parquet", index=False)
    cle_manifest = {"schema": "evrptw_city_logistics_environment_v1", "city_slug": "fixture", "freeze_authority": "freeze.json",
        "files": {name: {"sha256": digest} for name, digest in output_hashes(cle).items()}}
    write_json(cle / "cle_manifest.json", cle_manifest)
    cle_hash = sha256_file(cle / "cle_manifest.json")
    write_json(cle / "cle_verification.json", {"passed": True, "cle_manifest_sha256": cle_hash})
    write_json(cle / "freeze.json", {"frozen": True, "cities": {"fixture": {
        "cle_manifest_sha256": cle_hash, "verification_sha256": sha256_file(cle / "cle_verification.json")}}})
    folder = saved_run / "instances/view"
    metadata = json.loads((folder / "instance.json").read_text())
    metadata.update(status="not_materialized", matrix_materialized=False,
        cost_policy={"path_policy": "fastest_weekday", "connector_policy": "public_U_length_weighted_median_else_all_public", "lazy_reconstruction": True})
    metadata["vehicle"]["specific_energy_kwh_per_km"] = .4
    write_json(folder / "instance.json", metadata)
    manifest = json.loads((saved_run / "manifest.json").read_text())
    manifest.update(input={"cle": str(cle)}, source_sha256={"cle_manifest": cle_hash, "freeze": sha256_file(cle / "freeze.json")})
    write_json(saved_run / "manifest.json", manifest)
    refresh(saved_run)
    import evrptw_scenarios.routing as routing
    original = routing.canonical_endpoint_frame
    bound_ids = []

    def record_pair(path, ids):
        bound_ids.append(ids)
        return original(path, ids)

    monkeypatch.setattr(routing, "canonical_endpoint_frame", record_pair)
    run = ScenarioRun(saved_run)
    result = run.path("view", 0, 1)
    assert result["distance_m"] == 1020.
    assert result["time_s"] == 102.
    assert result["energy_kwh"] == pytest.approx(.408)
    assert bound_ids == [["depot", "customer"]]
    assert run.path("view", 1, 1)["distance_m"] == 0
    (cle / "speeds/directed_speeds.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="CLE data changed"):
        run.path("view", 0, 1)


def test_paired_export_modes_keep_workload_and_use_matching_times(saved_run, tmp_path):
    folder = saved_run / "instances/view"
    with np.load(folder / "costs.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update(distance_shortest_path_distance=arrays["distance_m"],
                  distance_shortest_path_time=arrays["time_s"],
                  time_shortest_path_distance=arrays["distance_m"] * 1.5,
                  time_shortest_path_time=arrays["time_s"] * .5,
                  time_shortest_path_energy_kwh=arrays["energy_kwh"] * 1.5)
    np.savez_compressed(folder / "costs.npz", **arrays)
    metadata = json.loads((folder / "instance.json").read_text())
    metadata["cost_policy"].update(execution_policy="shortest_distance", paired_path_policies=["shortest_distance", "fastest_time"])
    write_json(folder / "instance.json", metadata)
    refresh(saved_run)
    primary = json.loads(export_solver_input(saved_run, "view", tmp_path / "distance.json").read_text())
    fastest = json.loads(export_solver_input(saved_run, "view", tmp_path / "fastest.json", path_policy="fastest_time").read_text())
    energy = json.loads(export_solver_input(saved_run, "view", tmp_path / "energy.json", objective="total_energy", specific_energy_kwh_per_km=.2).read_text())
    for result in [fastest, energy]:
        assert result["source"] == primary["source"]
        assert result["node_order"] == primary["node_order"]
        assert result["nodes"] == primary["nodes"]
        assert result["physical_scenario_id"] != primary["physical_scenario_id"]
    assert primary["travel_time_matrix_s"][1][0] == 120
    assert fastest["travel_time_matrix_s"][1][0] == 60
    assert fastest["distance_matrix_km"][1][0] == 1.95
    assert energy["energy_matrix_kwh"][1][0] == pytest.approx(.26)
    assert primary["fastest_time_lower_bound_s"][1][0] == 60
    assert len(primary["paired_path_matrices"]) == 4
    assert energy["primary_objective"] == "total_energy"
