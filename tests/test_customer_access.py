from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from evrptw_cle.customer_access import attach_footprint_access


def test_mixed_simplified_highway_edge_cannot_host_delivery_projection(tmp_path):
    import networkx as nx

    from evrptw_cle.customer_access import build_eligible_physical_edges

    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for i, x in enumerate([-117.1, -117.09, -117.08], start=1):
        graph.add_node(i, x=x, y=32.8)
    graph.add_edge(1, 2, osmid=1, highway=["primary", "motorway"], length=1000)
    graph.add_edge(2, 3, osmid=2, highway="residential", length=1000)
    edges = build_eligible_physical_edges(tmp_path / "unused", "EPSG:32611", graph=graph)
    assert len(edges) == 1
    assert edges.highway.iloc[0] == "residential"


def test_footprint_access_uses_polygon_boundary_and_retains_directed_refs() -> None:
    locations = gpd.GeoDataFrame(
        {
            "latent_service_location_id": ["a"],
            "residential_units": [1],
            "service_location_type": ["house"],
        },
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:3857",
    )
    edges = gpd.GeoDataFrame(
        {
            "physical_edge_id": ["edge"],
            "directed_edge_refs": [
                (
                    '[{"u":"1","v":"2","key":"0","length_m":30.0,'
                    '"geometry_orientation":"same_as_physical"}]'
                )
            ],
            "directed_edge_ref_count": [1],
            "edge_u": ["1"],
            "edge_v": ["2"],
            "edge_key": ["0"],
            "highway": ["residential"],
            "road_name": ["Sample"],
            "oneway": [False],
            "access_layer": ["operational_public"],
            "connector_kind": ["through_road"],
            "legal_access_tier": ["operational_eligible"],
        },
        geometry=[LineString([(20, -10), (20, 20)])],
        crs="EPSG:3857",
    )
    result = attach_footprint_access(locations, edges, "EPSG:3857")
    assert result.loc[0, "road_access_distance_m"] == 10
    assert result.loc[0, "road_anchor_lon"] > result.loc[0, "building_access_lon"]
    assert result.loc[0, "directed_edge_ref_count"] == 1
    assert result.loc[0, "access_layer"] == "operational_public"
    assert result.loc[0, "road_projection_offset_m_from_physical_start"] == 10
    directed = json.loads(result.loc[0, "directed_projection_offsets"])
    assert directed[0]["offset_from_u_m"] == 10
    assert directed[0]["offset_to_v_m"] == pytest.approx(20)
    assert bool(result.loc[0, "connector_bidirectional"])
    assert result.loc[0, "connector_speed_policy"] == "assigned_at_instance_generation"
    assert not bool(result.loc[0, "customer_release_eligible"])
