"""Materialize only active virtual service stops into a directed road graph."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .protected_connectivity import build_directed_component_index


def _edge_lookup(graph: nx.MultiDiGraph) -> dict[tuple[str, str, str], tuple[Any, Any, Any]]:
    return {(str(u), str(v), str(key)): (u, v, key) for u, v, key in graph.edges(keys=True)}


def _split_geometry(geometry: Any, start: float, end: float) -> Any:
    if geometry is None:
        return None
    return substring(geometry, start, end, normalized=True)


def materialize_active_service_graph(
    *,
    graph: nx.MultiDiGraph,
    service_nodes: gpd.GeoDataFrame,
    projection_nodes: gpd.GeoDataFrame,
    connectors: pd.DataFrame,
    active_service_location_ids: Iterable[str],
    connector_speed_kph: float,
    proportional_edge_attributes: tuple[str, ...] = (
        "length",
        "travel_time_s",
        "legal_travel_time_s",
        "reference_travel_time_s",
    ),
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    """Split referenced directed edges and attach a small active stop subset.

    The full CLE remains compact. This function copies the base graph,
    inserts only the requested road projections, splits every retained directed
    OSM edge at exact fractional offsets, and adds equal-speed connector edges in
    both directions.  Original one-way directionality is never relaxed.
    """

    if connector_speed_kph <= 0:
        raise ValueError("connector_speed_kph must be positive")
    active = {str(value) for value in active_service_location_ids}
    if not active:
        raise ValueError("At least one active service location is required")

    selected_connectors = connectors.loc[
        connectors["latent_service_location_id"].astype(str).isin(active)
    ].copy()
    if set(selected_connectors["latent_service_location_id"].astype(str)) != active:
        missing = sorted(
            active - set(selected_connectors["latent_service_location_id"].astype(str))
        )
        raise KeyError(f"Unknown active service locations: {missing[:5]}")
    if "protected_roundtrip_eligible" in selected_connectors:
        invalid = selected_connectors.loc[
            ~selected_connectors["protected_roundtrip_eligible"].astype(bool),
            "latent_service_location_id",
        ].astype(str)
        if not invalid.empty:
            raise ValueError(
                "Active locations include projections outside the reference directed SCC: "
                f"{invalid.head(5).tolist()}"
            )
    selected_services = service_nodes.loc[
        service_nodes["latent_service_location_id"].astype(str).isin(active)
    ].copy()
    if len(selected_services) != len(active):
        raise ValueError("Service-node ledger is not one-to-one for active locations")
    projection_ids = set(selected_connectors["road_projection_node_id"].astype(str))
    selected_projections = projection_nodes.loc[
        projection_nodes["road_projection_node_id"].astype(str).isin(projection_ids)
    ].copy()
    if set(selected_projections["road_projection_node_id"].astype(str)) != projection_ids:
        raise KeyError("A connector references a missing road projection node")

    component_index = build_directed_component_index(graph)
    reference_nodes = [
        node
        for node in graph.nodes
        if component_index.node_to_scc[str(node)] == component_index.reference_scc_id
    ]
    reference_node = min(reference_nodes, key=str)
    result = graph.copy()
    graph_crs = result.graph.get("crs", "EPSG:4326")
    service_wgs = selected_services.to_crs(graph_crs)
    projection_wgs = selected_projections.to_crs(graph_crs)
    for row in projection_wgs.itertuples(index=False):
        node_id = str(row.road_projection_node_id)
        result.add_node(
            node_id,
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            virtual_node=True,
            virtual_node_kind="road_projection",
        )
    for row in service_wgs.itertuples(index=False):
        node_id = str(row.service_access_node_id)
        result.add_node(
            node_id,
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            virtual_node=True,
            virtual_node_kind="service_access",
            latent_service_location_id=str(row.latent_service_location_id),
        )

    lookup = _edge_lookup(result)
    splits: dict[tuple[Any, Any, Any], list[tuple[float, str]]] = defaultdict(list)
    for row in selected_projections.itertuples(index=False):
        projection_id = str(row.road_projection_node_id)
        for ref in json.loads(row.directed_projection_offsets):
            identity = (str(ref["u"]), str(ref["v"]), str(ref["key"]))
            if identity not in lookup:
                raise KeyError(f"Projection references missing directed edge {identity}")
            fraction = min(max(float(ref["projection_fraction_from_u"]), 0.0), 1.0)
            splits[lookup[identity]].append((fraction, projection_id))

    endpoint_alias: dict[str, Any] = {}
    split_edge_count = 0
    added_road_segment_count = 0
    tolerance = 1e-10
    for (u, v, key), raw_stops in splits.items():
        attributes = dict(result.edges[u, v, key])
        unique: dict[float, str] = {}
        for fraction, projection_id in raw_stops:
            rounded = round(fraction, 12)
            if rounded in unique and unique[rounded] != projection_id:
                raise ValueError("Distinct projection ids occupy the same directed-edge offset")
            unique[rounded] = projection_id
        interior: list[tuple[float, str]] = []
        for fraction, projection_id in sorted(unique.items()):
            if fraction <= tolerance:
                endpoint_alias[projection_id] = u
            elif fraction >= 1.0 - tolerance:
                endpoint_alias[projection_id] = v
            else:
                interior.append((fraction, projection_id))
        if not interior:
            continue
        result.remove_edge(u, v, key)
        split_edge_count += 1
        sequence = [(0.0, u), *interior, (1.0, v)]
        geometry = attributes.get("geometry")
        if geometry is None:
            geometry = LineString(
                [
                    (float(result.nodes[u]["x"]), float(result.nodes[u]["y"])),
                    (float(result.nodes[v]["x"]), float(result.nodes[v]["y"])),
                ]
            )
        for index, ((start, a), (end, b)) in enumerate(pairwise(sequence)):
            ratio = end - start
            segment = dict(attributes)
            for field in proportional_edge_attributes:
                if field in segment and segment[field] not in (None, ""):
                    segment[field] = float(segment[field]) * ratio
            start_point = Point(float(result.nodes[a]["x"]), float(result.nodes[a]["y"]))
            end_point = Point(float(result.nodes[b]["x"]), float(result.nodes[b]["y"]))
            segment["geometry"] = substring(
                geometry, geometry.project(start_point), geometry.project(end_point)
            )
            segment["virtual_split"] = True
            segment["source_edge_u"] = str(u)
            segment["source_edge_v"] = str(v)
            segment["source_edge_key"] = str(key)
            result.add_edge(a, b, key=f"{key}|virtual_split|{index}", **segment)
            added_road_segment_count += 1

    connector_edge_count = 0
    connector_seconds_per_m = 3.6 / connector_speed_kph
    for row in selected_connectors.itertuples(index=False):
        service_id = str(row.service_access_node_id)
        projection_id = str(row.road_projection_node_id)
        road_id = endpoint_alias.get(projection_id, projection_id)
        length_m = float(row.connector_length_m)
        attributes = {
            "length": length_m,
            "travel_time_s": length_m * connector_seconds_per_m,
            "scenario_time_s": length_m * connector_seconds_per_m,
            "synthetic_access_connector": True,
            "connector_speed_kph": connector_speed_kph,
            "connector_speed_symmetry": "equal_both_directions",
            "latent_service_location_id": str(row.latent_service_location_id),
        }
        connector_id = str(row.service_access_connector_id)
        result.add_edge(service_id, road_id, key=f"{connector_id}|out", **attributes)
        result.add_edge(road_id, service_id, key=f"{connector_id}|in", **attributes)
        connector_edge_count += 2

    unreachable = []
    for service_id in selected_services["service_access_node_id"].astype(str):
        if result.in_degree(service_id) == 0 or result.out_degree(service_id) == 0:
            unreachable.append(service_id)
    if unreachable:
        raise RuntimeError(
            f"Materialized service nodes lack bidirectional access: {unreachable[:5]}"
        )
    forward_reachable = nx.descendants(result, reference_node) | {reference_node}
    reverse_reachable = nx.ancestors(result, reference_node) | {reference_node}
    roundtrip_unreachable = sorted(
        service_id
        for service_id in selected_services["service_access_node_id"].astype(str)
        if service_id not in forward_reachable or service_id not in reverse_reachable
    )
    if roundtrip_unreachable:
        raise RuntimeError(
            "Materialized service nodes fail reference-SCC round-trip access: "
            f"{roundtrip_unreachable[:5]}"
        )

    audit = {
        "active_service_location_count": len(active),
        "unique_road_projection_count": len(projection_ids),
        "source_directed_edge_count_split": split_edge_count,
        "replacement_directed_road_segment_count": added_road_segment_count,
        "materialized_directed_connector_edge_count": connector_edge_count,
        "endpoint_projection_alias_count": len(endpoint_alias),
        "connector_speed_kph": connector_speed_kph,
        "connector_speed_symmetry": "equal_both_directions",
        "oneway_policy": "original directed edges split only; no reverse road edges synthesized",
        "reference_road_node_id": str(reference_node),
        "protected_roundtrip_checked": True,
        "protected_roundtrip_failure_count": 0,
    }
    return result, audit
