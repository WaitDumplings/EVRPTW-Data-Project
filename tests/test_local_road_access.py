import json

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, box

from evrptw_cle.local_road_access import LocalAccessContext, _forest, classify_way
from evrptw_cle.osm_graph import _edge_is_drivable
from evrptw_cle.util import sha256_file


def test_saved_context_preserves_string_node_keys_and_forests(tmp_path):
    ctx = context()
    ctx.water = box(1000, 1000, 1010, 1010)
    ctx.save(tmp_path)
    ctx.physical.to_parquet(tmp_path / "physical_roads.parquet", index=False)
    names = [
        "graph_operational.graphml",
        "graph_terminal_access.graphml",
        "access_forests.json",
        "connector_water.parquet",
        "physical_roads.parquet",
    ]
    (tmp_path / "access_context_manifest.json").write_text(
        json.dumps(
            {
                "area_crs": ctx.area_crs,
                "outputs": {name: sha256_file(tmp_path / name) for name in names},
            }
        )
    )
    restored = LocalAccessContext.load(tmp_path)
    assert set(restored.core) == set(ctx.core)
    assert set(restored.access) == set(ctx.access)
    assert restored.reference == ctx.reference
    assert restored._path("c", restored.outbound) == ctx._path("c", ctx.outbound)


def context(*, permit=False, one_way=False):
    core = nx.MultiDiGraph(crs="EPSG:3857")
    for n, x in [("a", 0), ("b", 100)]:
        core.add_node(n, x=x, y=0)
    core.add_edge(
        "a",
        "b",
        key="pub",
        length=100.0,
        highway="residential",
        osmid="1",
        geometry=LineString([(0, 0), (100, 0)]),
    )
    core.add_edge(
        "b",
        "a",
        key="pubr",
        length=100.0,
        highway="residential",
        osmid="1",
        geometry=LineString([(100, 0), (0, 0)]),
    )
    access = nx.MultiDiGraph(crs="EPSG:3857")
    access.add_node("a", x=0, y=0)
    access.add_node("c", x=0, y=100)
    access.add_edge(
        "a",
        "c",
        key="term",
        length=100.0,
        highway="service",
        osmid="2",
        geometry=LineString([(0, 0), (0, 100)]),
    )
    if not one_way:
        access.add_edge(
            "c",
            "a",
            key="termr",
            length=100.0,
            highway="service",
            osmid="2",
            geometry=LineString([(0, 100), (0, 0)]),
        )
    refs = [
        dict(
            u="a",
            v="c",
            key="term",
            length_m=100,
            osmid=["2"],
            highway=["service"],
            name=[""],
            geometry_orientation="same_as_physical",
        )
    ]
    if not one_way:
        refs.append(
            dict(
                u="c",
                v="a",
                key="termr",
                length_m=100,
                osmid=["2"],
                highway=["service"],
                name=[""],
                geometry_orientation="reverse_of_physical",
            )
        )
    roads = gpd.GeoDataFrame(
        [
            dict(
                physical_edge_id="term",
                directed_edge_refs=json.dumps(refs),
                directed_edge_ref_count=len(refs),
                edge_u="a",
                edge_v="c",
                edge_key="term",
                highway="service",
                road_name="",
                oneway=one_way,
                access_layer="osm_terminal_only",
                connector_kind="driveway",
                legal_access_tier="permit_required" if permit else "unspecified",
                motor_access_allowed=not permit,
                source_way_motor_allowed=not permit,
                node_access_blocked=False,
                sensitive_interior_road=False,
                anchor_class_eligible=True,
            )
        ],
        geometry=[LineString([(0, 0), (0, 100)])],
        crs=3857,
    )
    return LocalAccessContext(
        core,
        access,
        roads,
        {"a", "b"},
        _forest(access, {"a"}),
        _forest(access, {"a"}, reverse=True),
        "EPSG:3857",
    )


def locations():
    return gpd.GeoDataFrame(
        {"latent_service_location_id": ["house"]},
        geometry=[Polygon([(10, 45), (20, 45), (20, 55)])],
        crs=3857,
    )


def test_terminal_access_keeps_footprint_and_real_asymmetric_partial_paths():
    c = context()
    f = c.anchor(locations())
    r = f.iloc[0]
    assert r.road_access_distance_m == 10
    assert r.local_access_confirmed
    assert r.requires_terminal_paths
    assert np.isclose(r.terminal_inbound_length_m, r.terminal_outbound_length_m)
    assert np.isclose(r.terminal_inbound_length_m, 45)
    assert not c.core.has_edge("a", "c")
    assert np.isclose(json.loads(r.terminal_outbound_path_json)[0]["length_m"], 45)


def test_private_local_road_retained_unavailable_without_distant_fallback():
    f = context(permit=True).anchor(locations())
    assert len(f) == 1
    assert not f.iloc[0].local_access_confirmed
    assert f.iloc[0].access_review_reason == "local_road_permit_required"
    assert f.iloc[0].road_access_distance_m == 10


def test_one_way_terminal_is_not_made_bidirectional():
    r = context(one_way=True).anchor(locations()).iloc[0]
    assert not r.local_access_confirmed
    assert r.access_review_reason == "local_road_no_verified_directed_roundtrip"


def test_specific_permission_overrides_generic_access_but_explicit_motor_no_blocks():
    assert _edge_is_drivable({"highway": "residential", "access": "no", "motorcar": "yes"})
    assert not _edge_is_drivable({"highway": "residential", "vehicle": "no"})
    assert classify_way({"highway": "service", "motor_vehicle": "no"})["allowed"] is False
    assert classify_way({"highway": "service", "access": "private"})["allowed"] is False


def test_mapped_water_blocks_connector_without_distance_threshold():
    r = context().anchor(locations(), water=box(4, 0, 6, 100)).iloc[0]
    assert not r.local_access_confirmed
    assert r.access_review_reason == "connector_crosses_mapped_water"


def test_pair_materializer_uses_real_terminal_paths_only_for_source_and_destination():
    from evrptw_cle.terminal_pair import materialize_local_access_pair

    c = context()
    first = locations()
    second = locations()
    second["latent_service_location_id"] = "other"
    second.geometry = second.geometry.translate(yoff=20)
    both = gpd.GeoDataFrame(pd.concat([first, second], ignore_index=True), crs=3857)
    rows = c.anchor(both)
    routed, report = materialize_local_access_pair(
        graph=c.core,
        source=rows.iloc[0],
        destination=rows.iloc[1],
        connector_speed_kph=10,
        terminal_speed_kph=10,
    )
    length = nx.shortest_path_length(
        routed, report["source_node"], report["destination_node"], weight="length"
    )
    assert np.isclose(length, 10 + 45 + 65 + 10)
    assert not nx.has_path(routed, report["destination_node"], report["source_node"])
    assert len(c.core) == 2
    assert not c.core.has_node("service_house")
    assert not report["terminal_through_routing_allowed"]
    refs = [
        d for u, v, k, d in routed.edges(keys=True, data=True) if d.get("real_osm_terminal_segment")
    ]
    assert refs and all(d["source_osmid"] == "2" for d in refs)
