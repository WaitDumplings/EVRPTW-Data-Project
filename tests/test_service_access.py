from __future__ import annotations

import json

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from evrptw_cle.service_access import materialize_active_service_graph


def test_materializer_splits_both_directions_and_adds_symmetric_connector() -> None:
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=100.0, y=0.0)
    graph.add_edge(
        1,
        2,
        key=0,
        length=100.0,
        travel_time_s=10.0,
        geometry=LineString([(0, 0), (100, 0)]),
    )
    graph.add_edge(
        2,
        1,
        key=0,
        length=100.0,
        travel_time_s=12.0,
        geometry=LineString([(100, 0), (0, 0)]),
    )
    service_nodes = gpd.GeoDataFrame(
        {
            "latent_service_location_id": ["a"],
            "service_access_node_id": ["service_a"],
        },
        geometry=[Point(25, 10)],
        crs="EPSG:3857",
    )
    refs = [
        {"u": "1", "v": "2", "key": "0", "projection_fraction_from_u": 0.25},
        {"u": "2", "v": "1", "key": "0", "projection_fraction_from_u": 0.75},
    ]
    projection_nodes = gpd.GeoDataFrame(
        {
            "road_projection_node_id": ["projection_a"],
            "directed_projection_offsets": [json.dumps(refs)],
        },
        geometry=[Point(25, 0)],
        crs="EPSG:3857",
    )
    connectors = pd.DataFrame(
        {
            "latent_service_location_id": ["a"],
            "service_access_node_id": ["service_a"],
            "road_projection_node_id": ["projection_a"],
            "service_access_connector_id": ["connector_a"],
            "connector_length_m": [10.0],
        }
    )

    result, audit = materialize_active_service_graph(
        graph=graph,
        service_nodes=service_nodes,
        projection_nodes=projection_nodes,
        connectors=connectors,
        active_service_location_ids=["a"],
        connector_speed_kph=10.0,
    )
    assert not result.has_edge(1, 2, 0)
    assert nx.has_path(result, "service_a", 2)
    assert nx.has_path(result, 2, "service_a")
    out_edge = result.edges["service_a", "projection_a", "connector_a|out"]
    in_edge = result.edges["projection_a", "service_a", "connector_a|in"]
    assert out_edge["travel_time_s"] == pytest.approx(3.6)
    assert in_edge["travel_time_s"] == pytest.approx(out_edge["travel_time_s"])
    assert audit["source_directed_edge_count_split"] == 2
    assert audit["materialized_directed_connector_edge_count"] == 2


def test_materializer_uses_deterministic_reference_scc_when_sizes_tie() -> None:
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    for node, x in (("z", 200.0), ("y", 300.0), ("a", 0.0), ("b", 100.0)):
        graph.add_node(node, x=x, y=0.0)
    for u, v, start, end in (
        ("z", "y", 200.0, 300.0),
        ("y", "z", 300.0, 200.0),
        ("a", "b", 0.0, 100.0),
        ("b", "a", 100.0, 0.0),
    ):
        graph.add_edge(
            u,
            v,
            key=0,
            length=100.0,
            travel_time_s=10.0,
            geometry=LineString([(start, 0), (end, 0)]),
        )
    refs = [
        {"u": "a", "v": "b", "key": "0", "projection_fraction_from_u": 0.5},
        {"u": "b", "v": "a", "key": "0", "projection_fraction_from_u": 0.5},
    ]
    service_nodes = gpd.GeoDataFrame(
        {
            "latent_service_location_id": ["tie"],
            "service_access_node_id": ["service_tie"],
        },
        geometry=[Point(50, 10)],
        crs="EPSG:3857",
    )
    projection_nodes = gpd.GeoDataFrame(
        {
            "road_projection_node_id": ["projection_tie"],
            "directed_projection_offsets": [json.dumps(refs)],
        },
        geometry=[Point(50, 0)],
        crs="EPSG:3857",
    )
    connectors = pd.DataFrame(
        {
            "latent_service_location_id": ["tie"],
            "service_access_node_id": ["service_tie"],
            "road_projection_node_id": ["projection_tie"],
            "service_access_connector_id": ["connector_tie"],
            "connector_length_m": [10.0],
            "protected_roundtrip_eligible": [True],
        }
    )

    _, audit = materialize_active_service_graph(
        graph=graph,
        service_nodes=service_nodes,
        projection_nodes=projection_nodes,
        connectors=connectors,
        active_service_location_ids=["tie"],
        connector_speed_kph=10.0,
    )

    assert audit["reference_road_node_id"] == "a"
    assert audit["protected_roundtrip_checked"] is True
