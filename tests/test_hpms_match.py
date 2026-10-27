from __future__ import annotations

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Polygon

from evrptw_cle.hpms_match import build_hpms_edge_matches


def _write_inputs(tmp_path, *, bidirectional: bool):
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(1, x=-117.01, y=32.0)
    graph.add_node(2, x=-116.99, y=32.0)
    eastbound = LineString([(-117.01, 32.0), (-116.99, 32.0)])
    graph.add_edge(
        1,
        2,
        key=0,
        osmid=52,
        highway="primary",
        ref="CA 52",
        length=1890.0,
        oneway=not bidirectional,
        reversed=False,
        geometry=eastbound,
    )
    if bidirectional:
        graph.add_edge(
            2,
            1,
            key=0,
            osmid=52,
            highway="primary",
            ref="CA 52",
            length=1890.0,
            oneway=False,
            reversed=True,
            geometry=LineString(list(eastbound.coords)[::-1]),
        )
    graph_path = tmp_path / "graph.graphml"
    ox.save_graphml(graph, graph_path)

    hpms_path = tmp_path / "hpms.geojson"
    gpd.GeoDataFrame(
        {
            "OBJECTID": [1001],
            "ROUTE_NUMBER": [52],
            "F_SYSTEM": [2],
            "SPEED_LIMIT": [65],
        },
        geometry=[LineString([(-116.99, 32.00002), (-117.01, 32.00002)])],
        crs="EPSG:4326",
    ).to_file(hpms_path, driver="GeoJSON")

    boundary_path = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[
            Polygon(
                [
                    (-117.02, 31.99),
                    (-116.98, 31.99),
                    (-116.98, 32.01),
                    (-117.02, 32.01),
                ]
            )
        ],
        crs="EPSG:4326",
    ).to_file(boundary_path, driver="GeoJSON")
    return graph_path, hpms_path, boundary_path


def test_reversed_coordinate_storage_still_matches_one_way_corridor(tmp_path) -> None:
    graph_path, hpms_path, boundary_path = _write_inputs(tmp_path, bidirectional=False)
    output = tmp_path / "matches.parquet"
    summary = build_hpms_edge_matches(
        city_slug="test-city",
        hpms_path=hpms_path,
        graph_path=graph_path,
        boundary_path=boundary_path,
        output_path=output,
    )

    matches = pd.read_parquet(output)
    assert len(matches) == 1
    assert matches.loc[0, "edge_id"] == "1:2:0"
    assert matches.loc[0, "match_confidence"] == "high"
    assert bool(matches.loc[0, "corridor_match_usable"])
    assert bool(matches.loc[0, "direction_verified"])
    assert bool(matches.loc[0, "hpms_speed_usable"])
    assert summary["counts"]["hpms_speed_usable_edges"] == 1


def test_bidirectional_osm_corridor_can_supply_class_but_not_directional_speed(
    tmp_path,
) -> None:
    graph_path, hpms_path, boundary_path = _write_inputs(tmp_path, bidirectional=True)
    output = tmp_path / "matches.parquet"
    build_hpms_edge_matches(
        city_slug="test-city",
        hpms_path=hpms_path,
        graph_path=graph_path,
        boundary_path=boundary_path,
        output_path=output,
    )

    matches = pd.read_parquet(output)
    assert set(matches["edge_id"]) == {"1:2:0", "2:1:0"}
    assert matches["corridor_match_usable"].all()
    assert not matches["direction_verified"].any()
    assert not matches["hpms_speed_usable"].any()
