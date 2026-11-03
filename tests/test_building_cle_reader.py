import json

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from evrptw_cle.cle_reader import CityLogisticsEnvironment
from evrptw_cle.util import sha256_file


def building_reader_fixture(tmp_path, *, receipt=True):
    (tmp_path / "roads").mkdir()
    (tmp_path / "speeds").mkdir()
    gpd.GeoDataFrame({"location_id": ["customer:toy:a"], "location_type": ["customer"],
        "candidate_eligible": [True], "residential_units": [12],
        "residential_source": ["ai_inferred"], "units_method": ["web_extracted"]},
        geometry=[Point(-117, 33)], crs=4326).to_parquet(tmp_path / "locations.parquet")
    pd.DataFrame({"location_id": ["customer:toy:a"], "connector_length_m": [2.]}).to_parquet(
        tmp_path / "location_access_costs.parquet")
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, osmid=1, x=-117, y=33)
    graph.add_node(2, osmid=2, x=-116.9999, y=33)
    graph.add_edge(1, 2, key=0, osmid=55, length=10., oneway=True,
                   geometry=LineString([(-117, 33), (-116.9999, 33)]))
    ox.save_graphml(graph, tmp_path / "roads/graph_operational.graphml")
    pd.DataFrame({"edge_u": ["1"], "edge_v": ["2"], "edge_key": ["0"],
        "in_public_graph": [True], "in_delivery_terminal_graph": [False],
        "in_charging_terminal_graph": [False], "legal_speed_kph": [36.],
        "legal_travel_time_s": [1.], "reference_speed_weekday_kph": [30.],
        "reference_speed_weekend_kph": [25.], "reference_travel_time_weekday_s": [1.2],
        "reference_travel_time_weekend_s": [1.44]}).to_parquet(tmp_path / "speeds/directed_speeds.parquet")
    files = {name: {"sha256": sha256_file(tmp_path / name)} for name in [
        "locations.parquet", "location_access_costs.parquet", "roads/graph_operational.graphml",
        "speeds/directed_speeds.parquet"]}
    (tmp_path / "cle_manifest.json").write_text(json.dumps({
        "schema": "evrptw_building_first_cle_v1", "city_slug": "toy",
        "independent_verification": "stage1_validation.json", "files": files,
        "construction_options": {"allow_private_through": True}}))
    if receipt:
        (tmp_path / "stage1_validation.json").write_text(json.dumps({"passed": True,
            "errors": [], "summary": {"customer": {"available": 1}},
            "cle_manifest_sha256": sha256_file(tmp_path / "cle_manifest.json")}))
    return tmp_path


def test_new_building_schema_reads_locations_costs_and_oneway_graph(tmp_path):
    reader = CityLogisticsEnvironment(building_reader_fixture(tmp_path))
    assert reader.verified and not reader.frozen
    assert reader.locations().units_method.tolist() == ["web_extracted"]
    assert reader.access_costs().connector_length_m.tolist() == [2.]
    graph = reader.graph()
    assert graph.number_of_edges() == 1 and graph.has_edge("1", "2") and not graph.has_edge("2", "1")
    assert graph["1"]["2"][0]["reference_travel_time_weekday_s"] == 1.2
    assert graph.graph["ordinary_private_through_routing"]
    assert not (tmp_path / "cle_verification.json").exists()


def test_new_reader_rejects_receipt_for_another_manifest(tmp_path):
    building_reader_fixture(tmp_path)
    (tmp_path / "stage1_validation.json").write_text(json.dumps({"passed": True,
        "cle_manifest_sha256": "wrong"}))
    with pytest.raises(ValueError, match="validation bound to this manifest"):
        CityLogisticsEnvironment(tmp_path)


def test_new_reader_uses_building_validator_when_receipt_absent(tmp_path, monkeypatch):
    building_reader_fixture(tmp_path, receipt=False)
    calls = []

    def validate(path, *, verify_hashes):
        calls.append((path, verify_hashes))
        return {"passed": True, "errors": [], "summary": {"eligible_locations": 1}}

    monkeypatch.setattr("evrptw_cle.stage1_validation.validate_building_stage1", validate)
    reader = CityLogisticsEnvironment(tmp_path)
    assert reader.verified and len(calls) == 1
    assert calls[0][1] is True
    assert not (tmp_path / "stage1_validation.json").exists()


def test_explicit_new_reader_optout_does_not_claim_verification(tmp_path):
    reader = CityLogisticsEnvironment(building_reader_fixture(tmp_path, receipt=False), verify=False)
    assert not reader.verified and reader.verification is None
    assert len(reader.locations()) == 1
