import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from evrptw_scenarios import territory_world as tw
from evrptw_scenarios import cle_binding
from evrptw_scenarios.territory_config import DEFAULT
from evrptw_scenarios.world import stable_id


def roads_and_speeds(lines):
    roads, speeds = [], []
    for identity, u, v, coordinates, length, public in lines:
        refs = []
        for a, b, suffix in [(u, v, "f"), (v, u, "r")]:
            key = identity + suffix
            refs.append({"u": a, "v": b, "key": key,
                         "geometry_orientation": ("same_as_physical" if suffix == "f"
                                                  else "reverse_of_physical")})
            speeds.append({"edge_u": a, "edge_v": b, "edge_key": key,
                           "length_m": length, "in_public_graph": public})
        roads.append({"physical_edge_id": identity, "edge_u": u, "edge_v": v,
                      "directed_edge_refs": json.dumps(refs), "geometry": LineString(coordinates)})
    return gpd.GeoDataFrame(roads, crs=4326), pd.DataFrame(speeds)


def native(identity, x, y, edge, fraction, *, units=1, connector=0):
    name, u, v = edge
    offsets = [{"u": u, "v": v, "key": name + "f", "projection_fraction_from_u": fraction},
               {"u": v, "v": u, "key": name + "r", "projection_fraction_from_u": 1 - fraction}]
    return {"_id": identity, "location_id": identity, "site_id": identity,
            "facility_group_id": identity, "location_lon": x, "location_lat": y,
            "residential_units": units, "residential_type": "single_family",
            "requires_terminal_paths": False, "connector_length_m": connector,
            "directed_projection_offsets": json.dumps(offsets),
            "terminal_inbound_gateway": "", "terminal_outbound_gateway": "",
            "terminal_inbound_length_m": 0., "terminal_outbound_length_m": 0.,
            "physical_edge_id": name, "geometry": Point(x, y)}


def graph_catalog():
    return gpd.GeoDataFrame({"location_id": ["a", "b"],
                            "bg_geoid": ["000000000001", "000000000002"],
                            "community_id": ["city:bg:1", "city:bg:2"],
                            "cell_id": ["city:bg:1", "city:bg:2"],
                            "geometry": [Point(.01, 0), Point(.19, 0)]}, crs=4326)


def polygons():
    return gpd.GeoDataFrame({"GEOID": ["000000000001", "000000000002"],
                            "geometry": [box(-.01, -.01, .05, .01),
                                         box(.15, -.01, .21, .01)]}, crs=4326)


def test_actual_bridge_and_empty_road_corridor_have_witnesses():
    roads, speeds = roads_and_speeds([
        ("left", "1", "2", [(0, 0), (.06, 0)], 6000, True),
        ("empty", "2", "3", [(.06, 0), (.14, 0)], 8000, True),
        ("right", "3", "4", [(.14, 0), (.2, 0)], 6000, True),
    ])
    result = tw.road_supported_community_graph(roads, speeds, polygons(), graph_catalog())
    assert result["adjacency"] == {"city:bg:1": ["city:bg:2"], "city:bg:2": ["city:bg:1"]}
    witness = result["witnesses"][0]
    assert witness["unit_path"][0] == "city:bg:1"
    assert witness["unit_path"][-1] == "city:bg:2"
    assert any(node.startswith("road:") for node in witness["unit_path"])
    assert "empty" in witness["road_evidence_ids"]


def test_touching_polygons_without_public_bridge_do_not_create_adjacency():
    bg = polygons()
    bg.geometry = [box(-.01, -.01, .1, .01), box(.1, -.01, .21, .01)]
    roads, speeds = roads_and_speeds([
        ("left", "1", "2", [(0, 0), (.04, 0)], 4000, True),
        ("private_bridge", "2", "3", [(.04, 0), (.16, 0)], 12000, False),
        ("right", "3", "4", [(.16, 0), (.2, 0)], 4000, True),
    ])
    result = tw.road_supported_community_graph(roads, speeds, bg, graph_catalog())
    assert result["adjacency"] == {"city:bg:1": [], "city:bg:2": []}
    assert result["public_physical_roads"] == 2


def test_geometry_encounter_order_prevents_centroid_adjacency_shortcuts():
    bg = gpd.GeoDataFrame({"GEOID": ["000000000001", "000000000002", "000000000003"],
                           "geometry": [box(0, -1, 1, 1), box(1, -1, 2, 1), box(2, -1, 3, 1)]},
                          crs=4326)
    catalog = gpd.GeoDataFrame({"location_id": ["1", "2", "3"],
                               "bg_geoid": bg.GEOID, "community_id": ["a", "b", "c"],
                               "cell_id": ["a", "b", "c"], "geometry": [Point(.5, 0),
                               Point(1.5, 0), Point(2.5, 0)]}, crs=4326)
    roads, speeds = roads_and_speeds([("r", "u", "v", [(.2, 0), (2.8, 0)], 100, True)])
    result = tw.road_supported_community_graph(roads, speeds, bg, catalog)
    assert result["adjacency"]["a"] == ["b"]
    assert result["adjacency"]["b"] == ["a", "c"]
    assert result["adjacency"]["c"] == ["b"]


def test_reverse_reference_attaches_native_junction_to_correct_geometry_end():
    bg = gpd.GeoDataFrame({"GEOID": ["000000000001", "000000000002", "000000000003"],
                           "geometry": [box(0, -1, 1, 1), box(1, -1, 2, 1), box(2, -1, 3, 1)]},
                          crs=4326)
    catalog = gpd.GeoDataFrame({"location_id": ["1", "2", "3"],
                               "bg_geoid": bg.GEOID, "community_id": ["a", "b", "c"],
                               "cell_id": ["a", "b", "c"], "geometry": [Point(.5, 0),
                               Point(1.5, 0), Point(2.5, 0)]}, crs=4326)
    # Stored row endpoints follow the reverse directed ref. The physical line
    # runs L->R; the extra stub proves L belongs to the left BG a. Reversing
    # the BG sequence's junction attachment would falsely add a--c, skipping b.
    roads, speeds = roads_and_speeds([
        ("main", "R", "L", [(.2, 0), (2.8, 0)], 260, True),
        ("stub", "L", "leaf", [(.2, 0), (.2, -.1)], 10, True),
    ])
    refs = json.loads(roads.at[0, "directed_edge_refs"])
    refs[0]["geometry_orientation"] = "reverse_of_physical"
    refs[1]["geometry_orientation"] = "same_as_physical"
    roads.at[0, "directed_edge_refs"] = json.dumps(refs)
    result = tw.road_supported_community_graph(roads, speeds, bg, catalog)
    assert result["adjacency"] == {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    assert result["road_evidence"]["main"]["native_u"] == "L"
    assert result["road_evidence"]["main"]["native_v"] == "R"


def test_native_provider_is_unbounded_and_same_id_is_zero():
    _, speeds = roads_and_speeds([("r", "1", "2", [(0, 0), (.2, 0)], 40000, True)])
    frame = gpd.GeoDataFrame([native("a", 0, 0, ("r", "1", "2"), 0, connector=7),
                              native("b", .2, 0, ("r", "1", "2"), 1, connector=11)], crs=4326)
    provider = tw.NativeDistanceProvider(speeds)
    outward, inward = provider.distances(frame.iloc[[0]], frame)
    assert outward.tolist() == inward.tolist() == [0, 40018]
    assert np.isinf(provider.engine.limit_m)
    both_out, both_in = provider.distances(frame, frame)
    assert both_out.shape == both_in.shape == (2, 2)
    assert np.array_equal(both_out.diagonal(), [0, 0])


@pytest.fixture
def frozen_mock(tmp_path, monkeypatch):
    root = tmp_path / "cle"
    (root / "roads").mkdir(parents=True)
    (root / "speeds").mkdir()
    roads, speeds = roads_and_speeds([
        ("main", "1", "2", [(0, 0), (.2, 0)], 40000, True),
        ("island", "8", "9", [(.01, .05), (.03, .05)], 2000, True),
        ("outside", "2", "3", [(.2, 0), (.8, 0)], 60000, True),
    ])
    speeds["operating_mode"] = "U"
    speeds["reference_speed_weekday_kph"] = 100.
    speeds["reference_travel_time_weekday_s"] = speeds.length_m * .036
    roads.to_parquet(root / "roads/physical_roads.parquet", index=False)
    speeds.to_parquet(root / "speeds/directed_speeds.parquet", index=False)
    community = tmp_path / "bg.parquet"
    polygons().to_parquet(community, index=False)
    for name in ["cle_manifest.json", "cle_verification.json", "stage1_freeze.json"]:
        (root / name).write_text("{}")
    manifest = {"city_slug": "test-city", "freeze_authority": "stage1_freeze.json",
                "files": {"roads/physical_roads.parquet": {"sha256": "bound-roads"}}}
    cle = SimpleNamespace(path=root, manifest=manifest)
    customers = gpd.GeoDataFrame([
        native("c1", .01, 0, ("main", "1", "2"), .05, units=1),
        native("c2", .16, 0, ("main", "1", "2"), .8, units=8),
        native("missing", .1, 0, ("main", "1", "2"), .5, units=2),
        native("island", .02, .05, ("island", "8", "9"), .5),
        native("far", .8, 0, ("outside", "2", "3"), 1),
    ], crs=4326)
    depots = gpd.GeoDataFrame([
        native("d0", 0, 0, ("main", "1", "2"), 0),
        native("d1", .02, 0, ("main", "1", "2"), .1),
    ], crs=4326).sort_values("location_id").reset_index(drop=True)
    # Frozen depot tables publish a Point, unlike the customer's named fields.
    depots = depots.drop(columns=["location_lon", "location_lat"])
    for frame in (customers, depots):
        frame["connector_time_s_per_inverse_kph"] = 3.6 * frame.connector_length_m
        for side in ("inbound", "outbound"):
            frame[f"terminal_{side}_path_json"] = "[]"
            frame[f"{side}_terminal_length_m"] = 0.
            frame[f"{side}_terminal_reference_weekday_time_s"] = 0.
    monkeypatch.setattr(tw, "_verified_cle", lambda _: cle)
    monkeypatch.setattr(cle_binding, "cle_authority",
                        lambda *_: (root / "stage1_freeze.json", root / "cle_verification.json"))
    monkeypatch.setattr(tw, "_canonical_native",
                        lambda _cle, kind: (depots if kind == "depot" else customers).copy())
    return root, community, depots


def test_build_keeps_32km_road_candidate_and_missing_bg_but_not_disconnected(frozen_mock, tmp_path):
    root, community, _ = frozen_mock
    config = {**DEFAULT["world"], "depot_id": "d0"}
    world = tw.build_territory_world(root, community, config, tmp_path / "world", progress=None)
    assert world.catalog.location_id.tolist() == ["c1", "c2", "missing"]
    c2 = world.catalog.set_index("location_id").loc["c2"]
    assert c2.depot_outward_distance_m == c2.depot_inward_distance_m == 32000
    assert world.info["geodesic_candidate_count"] == 4
    assert world.info["nonfinite_road_candidates"] == 1
    assert world.info["community_missing_world_count"] == 1
    missing = world.catalog.set_index("location_id").loc["missing"]
    assert missing.cell_id == "missing-bg:missing"
    assert world.adjacency[missing.cell_id]
    costs = world.anchor_costs([0, 1])
    assert costs.shape == (2, 3)
    assert costs[0, 0] == costs[1, 1] == 0
    assert costs[0, 1] == pytest.approx(30)
    saved = json.loads((tmp_path / "world/world.json").read_text())
    assert saved["schema"] == "evrptw_territory_operating_world_v1"
    assert saved["depot_redraw_count"] == 0
    assert set(saved["artifacts"]) == {"catalog.parquet", "depot.parquet", "community_graph.json"}
    assert saved["config"] == config


def test_uniform_depot_stream_and_explicit_override_match_legacy(frozen_mock, tmp_path):
    root, community, depots = frozen_mock
    config = dict(DEFAULT["world"])
    seed = int(stable_id(["depot_uniform_v1", "test-city", config["provider_id"], config["seed"]]), 16)
    expected = depots.iloc[int(np.random.Generator(np.random.PCG64(seed)).integers(len(depots)))]
    world = tw.build_territory_world(root, community, config, tmp_path / "random", progress=None)
    assert world.info["depot_id"] == expected.location_id
    assert world.info["depot_selection"] == "one uniform sorted-ID draw"
    explicit = tw.build_territory_world(root, community, {**config, "depot_id": "d1"},
                                       tmp_path / "explicit", progress=None)
    assert explicit.info["depot_id"] == "d1"
    assert explicit.info["depot_selection"] == "explicit canonical ID"


@pytest.mark.parametrize(("limit_s", "capacity", "selected"), [
    (10000, 100., ["c1", "c2", "missing"]),
    (800, 1000., ["c1", "missing"]),
])
def test_fastest_service_gate_uses_path_energy_and_time_without_radius(
        frozen_mock, tmp_path, limit_s, capacity, selected):
    root, community, _ = frozen_mock
    config = {**DEFAULT["world"], "depot_id": "d0", "external_radius_km": 1.,
              "service_policy": "fastest_time_energy", "service_limit_s": limit_s,
              "depot_sampling": "bg_uniform"}
    vehicle = {"battery_capacity_kwh": capacity, "initial_soc": 1., "reserve_soc": .1,
               "specific_energy_kwh_per_km": 1.}
    world = tw.build_territory_world(root, community, config, tmp_path / "review",
                                    progress=None, vehicle=vehicle)
    assert world.catalog.location_id.tolist() == selected
    assert world.info["geodesic_candidate_count"] is None
    assert world.info["depot_bg_count"] == 1
    assert world.info["service_scope"]["finite_bidirectional_count"] == 4
    assert world.info["nonfinite_road_candidates"] == 1
    row = world.catalog.set_index("location_id").loc["missing"]
    assert row.depot_geodesic_distance_km > 1
    assert row.depot_fastest_outward_distance_m == pytest.approx(20000.)
    assert row.depot_fastest_outward_time_s == pytest.approx(720.)
    assert row.depot_fastest_outward_energy_kwh == pytest.approx(20.)


def test_bg_depot_sampler_excludes_unmatched_automatic_but_allows_explicit(frozen_mock):
    _, community, depots = frozen_mock
    depots = tw.published_coordinates(depots)
    unmatched = depots.iloc[[0]].copy()
    unmatched["location_id"], unmatched["location_lon"] = "outside", .11
    pool = pd.concat([unmatched, depots], ignore_index=True)
    config = {**DEFAULT["world"], "depot_sampling": "bg_uniform"}
    depot, label, n_bg, n_missing = tw._select_depot(pool, config, "test-city", community)
    assert depot.location_id.iat[0] in {"d0", "d1"}
    assert label == "uniform BG then uniform sorted canonical depot"
    assert (n_bg, n_missing) == (1, 1)
    explicit = tw._select_depot(pool, {**config, "depot_id": "outside"}, "test-city", community)
    assert explicit[0].location_id.iat[0] == "outside"


def test_haversine_matches_analysis_radius_and_broadcasts():
    result = tw.great_circle_km(0, 0, [0, 1], [0, 0])
    assert result == pytest.approx([0, np.pi / 180 * 6371.0088])
