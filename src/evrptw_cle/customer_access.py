"""Footprint-to-directed-road access audit for Gate 2 customer candidates."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.strtree import STRtree

from .protected_connectivity import (
    build_directed_component_index,
    connectivity_summary,
    label_projection_connectivity,
)
from .road_tags import (
    DELIVERY_ROAD_CLASSES,
    EXCLUDED_ANCHOR_ROAD_CLASSES,
    _as_bool,
    _parse_multivalue,
)
from .util import sha256_file, write_json


def _physical_edge_key(geometry: Any, osmid: Any) -> str:
    normalized = shapely.normalize(geometry)
    osmids = sorted(_parse_multivalue(osmid))
    digest = hashlib.sha1()
    digest.update(normalized.wkb)
    digest.update("|".join(osmids).encode("utf-8"))
    return digest.hexdigest()[:20]


def _orientation_against(reference: Any, candidate: Any) -> str:
    """Describe candidate line direction relative to the retained physical line."""

    reference_start = shapely.get_point(reference, 0)
    reference_end = shapely.get_point(reference, -1)
    candidate_start = shapely.get_point(candidate, 0)
    candidate_end = shapely.get_point(candidate, -1)
    same_cost = float(
        shapely.distance(reference_start, candidate_start)
        + shapely.distance(reference_end, candidate_end)
    )
    reverse_cost = float(
        shapely.distance(reference_start, candidate_end)
        + shapely.distance(reference_end, candidate_start)
    )
    return "same_as_physical" if same_cost <= reverse_cost else "reverse_of_physical"


def _projection_offsets_json(
    directed_refs_json: str,
    physical_offset_m: float,
    physical_length_m: float,
) -> str:
    refs = json.loads(directed_refs_json)
    if physical_length_m <= 0:
        raise ValueError("Physical edge geometry has nonpositive length")
    fraction = min(max(physical_offset_m / physical_length_m, 0.0), 1.0)
    enriched = []
    for ref in refs:
        same = ref.get("geometry_orientation") == "same_as_physical"
        from_u_fraction = fraction if same else 1.0 - fraction
        edge_length_m = float(ref["length_m"])
        item = dict(ref)
        item["projection_fraction_from_u"] = from_u_fraction
        item["offset_from_u_m"] = from_u_fraction * edge_length_m
        item["offset_to_v_m"] = (1.0 - from_u_fraction) * edge_length_m
        enriched.append(item)
    return json.dumps(enriched, sort_keys=True, separators=(",", ":"))


def build_eligible_physical_edges(
    graph_path: Path, area_crs: str, *, graph=None
) -> gpd.GeoDataFrame:
    """Collapse reciprocal graph edges while retaining every directed edge reference."""

    if graph is None:
        graph = ox.load_graphml(graph_path)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges = edges.reset_index()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    geometry_by_group: dict[str, Any] = {}
    for row in edges.itertuples(index=False):
        attributes = row._asdict()
        if _as_bool(attributes.get("transit_only", False)):
            continue
        classes = _parse_multivalue(attributes.get("highway"))
        if not classes or not any(value in DELIVERY_ROAD_CLASSES for value in classes):
            continue
        # Simplified OSM edges may carry mixed classes. If any segment is an
        # excluded highway/ramp, do not anchor a host to an unknown segment.
        if any(value in EXCLUDED_ANCHOR_ROAD_CLASSES for value in classes):
            continue
        geometry = attributes.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        physical_id = _physical_edge_key(geometry, attributes.get("osmid"))
        geometry_by_group.setdefault(physical_id, geometry)
        physical_geometry = geometry_by_group[physical_id]
        groups[physical_id].append(
            {
                "u": str(attributes.get("u")),
                "v": str(attributes.get("v")),
                "key": str(attributes.get("key")),
                "osmid": list(_parse_multivalue(attributes.get("osmid"))),
                "oneway": _as_bool(attributes.get("oneway", False)),
                "highway": list(classes),
                "name": list(_parse_multivalue(attributes.get("name"))),
                "length_m": float(attributes.get("length") or 0.0),
                "geometry_orientation": _orientation_against(physical_geometry, geometry),
            }
        )
    records = []
    geometries = []
    for physical_id in sorted(groups):
        refs = sorted(groups[physical_id], key=lambda value: (value["u"], value["v"], value["key"]))
        first = refs[0]
        records.append(
            {
                "physical_edge_id": physical_id,
                "directed_edge_refs": json.dumps(refs, sort_keys=True),
                "directed_edge_ref_count": len(refs),
                "edge_u": first["u"],
                "edge_v": first["v"],
                "edge_key": first["key"],
                "highway": "|".join(first["highway"]),
                "road_name": "|".join(first["name"]),
                "oneway": first["oneway"],
                "access_layer": "operational_public",
                "connector_kind": "through_road",
                "legal_access_tier": "operational_eligible",
            }
        )
        geometries.append(geometry_by_group[physical_id])
    if not records:
        raise ValueError("Operational graph contains no eligible inside-city delivery edges")
    return gpd.GeoDataFrame(records, geometry=geometries, crs=edges.crs).to_crs(area_crs)


def build_terminal_physical_edges(
    graph_path: Path,
    area_crs: str,
    *,
    include_permit_required: bool,
    retain_all_directed_refs: bool = True,
) -> gpd.GeoDataFrame:
    """Collapse a terminal-only graph into physical ways for access matching."""

    graph = ox.load_graphml(graph_path)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges = edges.reset_index()
    groups: dict[str, dict[str, Any]] = {}
    geometry_by_group: dict[str, Any] = {}
    for row in edges.itertuples(index=False):
        attributes = row._asdict()
        if not _as_bool(attributes.get("connected_to_operational", False)):
            continue
        legal_tier = str(attributes.get("legal_access_tier", "unresolved"))
        if legal_tier == "permit_required" and not include_permit_required:
            continue
        if legal_tier != "permit_required" and not _as_bool(
            attributes.get("terminal_core_eligible", False)
        ):
            continue
        geometry = attributes.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        physical_id = "terminal_" + _physical_edge_key(geometry, attributes.get("osmid"))
        geometry_by_group.setdefault(physical_id, geometry)
        ref = {
            "u": str(attributes.get("u")),
            "v": str(attributes.get("v")),
            "key": str(attributes.get("key")),
            "osmid": list(_parse_multivalue(attributes.get("osmid"))),
            "oneway": _as_bool(attributes.get("oneway", False)),
            "highway": list(_parse_multivalue(attributes.get("highway"))),
            "name": list(_parse_multivalue(attributes.get("name"))),
            "connector_kind": str(attributes.get("connector_kind", "unresolved")),
            "legal_access_tier": legal_tier,
        }
        if physical_id not in groups:
            groups[physical_id] = {
                "first": ref,
                "refs": [ref] if retain_all_directed_refs else None,
                "count": 1,
            }
        else:
            groups[physical_id]["count"] += 1
            if retain_all_directed_refs:
                groups[physical_id]["refs"].append(ref)
    records = []
    geometries = []
    for physical_id in sorted(groups):
        group = groups[physical_id]
        first = group["first"]
        refs = (
            sorted(group["refs"], key=lambda value: (value["u"], value["v"], value["key"]))
            if retain_all_directed_refs
            else [first]
        )
        records.append(
            {
                "physical_edge_id": physical_id,
                "directed_edge_refs": json.dumps(refs, sort_keys=True),
                "directed_edge_ref_count": group["count"],
                "directed_edge_refs_complete": retain_all_directed_refs,
                "edge_u": first["u"],
                "edge_v": first["v"],
                "edge_key": first["key"],
                "highway": "|".join(first["highway"]),
                "road_name": "|".join(first["name"]),
                "oneway": first["oneway"],
                "access_layer": "osm_terminal_only",
                "connector_kind": first["connector_kind"],
                "legal_access_tier": first["legal_access_tier"],
            }
        )
        geometries.append(geometry_by_group[physical_id])
    if not records:
        raise ValueError("Terminal graph contains no eligible connected physical edges")
    return gpd.GeoDataFrame(records, geometry=geometries, crs=edges.crs).to_crs(area_crs)


def attach_footprint_access(
    locations: gpd.GeoDataFrame,
    physical_edges: gpd.GeoDataFrame,
    area_crs: str,
    *,
    matched_edge_indices=None,
) -> gpd.GeoDataFrame:
    """Attach each footprint to its nearest eligible physical edge without accepting it."""

    projected = locations.to_crs(area_crs).reset_index(drop=True)
    edges = physical_edges.to_crs(area_crs).reset_index(drop=True)
    footprint_geometries = np.asarray(projected.geometry.to_numpy(), dtype=object)
    edge_geometries = np.asarray(edges.geometry.to_numpy(), dtype=object)
    tree = STRtree(edge_geometries)
    nearest_indices = np.asarray(
        tree.nearest(footprint_geometries)
        if matched_edge_indices is None
        else matched_edge_indices,
        dtype=int,
    )
    if (
        len(nearest_indices) != len(projected)
        or np.any(nearest_indices < 0)
        or np.any(nearest_indices >= len(edges))
    ):
        raise ValueError("Selected physical edges must match the source rows")
    matched_edges = edge_geometries[nearest_indices]
    distances = np.asarray(shapely.distance(footprint_geometries, matched_edges), dtype=float)
    connectors = shapely.shortest_line(footprint_geometries, matched_edges)
    building_access_local = shapely.get_point(connectors, 0)
    road_anchor_local = shapely.get_point(connectors, -1)
    building_access_wgs84 = gpd.GeoSeries(building_access_local, crs=area_crs).to_crs("EPSG:4326")
    road_anchor_wgs84 = gpd.GeoSeries(road_anchor_local, crs=area_crs).to_crs("EPSG:4326")

    matched = edges.iloc[nearest_indices].reset_index(drop=True)
    matched_physical_geometries = edge_geometries[nearest_indices]
    physical_offsets_m = np.asarray(
        shapely.line_locate_point(matched_physical_geometries, road_anchor_local),
        dtype=float,
    )
    physical_lengths_m = np.asarray(shapely.length(matched_physical_geometries), dtype=float)
    result = locations.reset_index(drop=True).copy()
    for field in (
        "physical_edge_id",
        "directed_edge_refs",
        "directed_edge_ref_count",
        "edge_u",
        "edge_v",
        "edge_key",
        "highway",
        "road_name",
        "oneway",
        "access_layer",
        "connector_kind",
        "legal_access_tier",
    ):
        result[field] = matched[field].to_numpy()
    result["road_anchor_method"] = np.where(
        result["access_layer"] == "osm_terminal_only",
        "footprint_boundary_to_osm_terminal_way_projection",
        "footprint_boundary_to_operational_edge_projection",
    )
    result["road_access_distance_m"] = distances
    result["building_access_lon"] = building_access_wgs84.x.to_numpy(dtype=float)
    result["building_access_lat"] = building_access_wgs84.y.to_numpy(dtype=float)
    result["road_anchor_lon"] = road_anchor_wgs84.x.to_numpy(dtype=float)
    result["road_anchor_lat"] = road_anchor_wgs84.y.to_numpy(dtype=float)
    result["physical_edge_geometry_length_m"] = physical_lengths_m
    result["road_projection_offset_m_from_physical_start"] = physical_offsets_m
    result["road_projection_fraction_from_physical_start"] = np.divide(
        physical_offsets_m,
        physical_lengths_m,
        out=np.zeros_like(physical_offsets_m),
        where=physical_lengths_m > 0,
    ).clip(0.0, 1.0)
    result["directed_projection_offsets"] = [
        _projection_offsets_json(refs, offset, length)
        for refs, offset, length in zip(
            result["directed_edge_refs"],
            physical_offsets_m,
            physical_lengths_m,
            strict=True,
        )
    ]
    result["service_access_node_id"] = "service_access_" + result[
        "latent_service_location_id"
    ].astype(str)
    result["road_projection_node_id"] = [
        "road_projection_" + hashlib.sha1(f"{physical_id}|{fraction:.9f}".encode()).hexdigest()[:20]
        for physical_id, fraction in zip(
            result["physical_edge_id"],
            result["road_projection_fraction_from_physical_start"],
            strict=True,
        )
    ]
    result["service_access_connector_id"] = "connector_" + result[
        "latent_service_location_id"
    ].astype(str)
    result["connector_length_m"] = distances
    result["connector_bidirectional"] = True
    result["connector_speed_symmetry"] = "equal_both_directions"
    result["connector_speed_policy"] = "assigned_at_instance_generation"
    result["access_threshold_m"] = np.nan
    result["customer_release_eligible"] = False
    result["quarantine_reason"] = "road_access_threshold_pending"
    return gpd.GeoDataFrame(result, geometry="geometry", crs=locations.crs)


def _write_service_access_contract(anchored: gpd.GeoDataFrame, output_dir: Path) -> dict[str, Any]:
    """Write compact virtual-node/connector ledgers without inflating GraphML."""

    identity = [
        "latent_service_location_id",
        "service_access_node_id",
        "road_projection_node_id",
        "service_access_connector_id",
    ]
    service_nodes = anchored[
        identity
        + [
            "building_access_lon",
            "building_access_lat",
            "anchor_scc_id",
            "reference_scc_id",
            "anchor_in_reference_scc",
            "protected_inbound_access_eligible",
            "protected_outbound_access_eligible",
            "protected_roundtrip_eligible",
            "protected_roundtrip_status",
        ]
    ].copy()
    service_nodes = gpd.GeoDataFrame(
        service_nodes,
        geometry=gpd.points_from_xy(
            service_nodes["building_access_lon"], service_nodes["building_access_lat"]
        ),
        crs="EPSG:4326",
    )
    projection_nodes = anchored[
        identity
        + [
            "physical_edge_id",
            "road_anchor_lon",
            "road_anchor_lat",
            "road_projection_fraction_from_physical_start",
            "road_projection_offset_m_from_physical_start",
            "physical_edge_geometry_length_m",
            "directed_projection_offsets",
            "anchor_scc_id",
            "reference_scc_id",
            "anchor_in_reference_scc",
            "protected_inbound_access_eligible",
            "protected_outbound_access_eligible",
            "protected_roundtrip_eligible",
            "protected_roundtrip_status",
        ]
    ].copy()
    projection_nodes = projection_nodes.drop_duplicates(
        "road_projection_node_id", keep="first"
    ).reset_index(drop=True)
    projection_nodes = gpd.GeoDataFrame(
        projection_nodes,
        geometry=gpd.points_from_xy(
            projection_nodes["road_anchor_lon"], projection_nodes["road_anchor_lat"]
        ),
        crs="EPSG:4326",
    )
    connectors = anchored[
        identity
        + [
            "connector_length_m",
            "connector_bidirectional",
            "connector_speed_symmetry",
            "connector_speed_policy",
            "anchor_scc_id",
            "reference_scc_id",
            "anchor_in_reference_scc",
            "protected_inbound_access_eligible",
            "protected_outbound_access_eligible",
            "protected_roundtrip_eligible",
            "protected_roundtrip_status",
        ]
    ].copy()
    connectors["from_node_id"] = connectors["service_access_node_id"]
    connectors["to_node_id"] = connectors["road_projection_node_id"]
    connectors["materialized_directed_edge_count"] = 2
    for column in [
        "road_access_policy",
        "access_status",
        "access_review_reason",
        "local_access_confirmed",
        "requires_terminal_paths",
        "terminal_inbound_path_json",
        "terminal_outbound_path_json",
        "terminal_inbound_gateway",
        "terminal_outbound_gateway",
        "terminal_inbound_length_m",
        "terminal_outbound_length_m",
    ]:
        if column in anchored:
            connectors[column] = anchored[column].to_numpy()
    for column in [
        "delivery_authorization_assumed",
        "owner_permission_observed",
        "access_permission_basis",
        "local_alternative_selected",
        "initial_access_review_reason",
        "frontage_policy",
        "unsupported_frontage_nearer",
        "unsupported_frontage_osmid",
        "unsupported_frontage_highway",
        "unsupported_frontage_distance_m",
    ]:
        if column in anchored:
            connectors[column] = anchored[column].to_numpy()

    paths = {
        "service_access_nodes": output_dir / "service_access_nodes.parquet",
        "road_projection_nodes": output_dir / "road_projection_nodes.parquet",
        "service_access_connectors": output_dir / "service_access_connectors.parquet",
    }
    service_nodes.to_parquet(paths["service_access_nodes"], index=False)
    projection_nodes.to_parquet(paths["road_projection_nodes"], index=False)
    connectors.to_parquet(paths["service_access_connectors"], index=False)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": {key: sha256_file(value) for key, value in paths.items()},
        "row_counts": {
            "service_access_nodes": len(service_nodes),
            "road_projection_nodes": len(projection_nodes),
            "service_access_connectors": len(connectors),
            "materialized_directed_connector_edges": 2 * len(connectors),
        },
    }


def _threshold_summary(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    selected = frame["road_access_distance_m"] <= threshold
    units = pd.to_numeric(frame["residential_units"], errors="coerce").fillna(0)
    total_units = float(units.sum())
    selected_units = float(units.loc[selected].sum())
    by_type = {}
    for location_type, group in frame.groupby("service_location_type", observed=True):
        group_selected = group["road_access_distance_m"] <= threshold
        group_units = pd.to_numeric(group["residential_units"], errors="coerce").fillna(0)
        group_total = float(group_units.sum())
        group_selected_units = float(group_units.loc[group_selected].sum())
        by_type[str(location_type)] = {
            "location_count": len(group),
            "covered_location_count": int(group_selected.sum()),
            "covered_location_share": float(group_selected.mean()),
            "modeled_unit_mass": group_total,
            "covered_modeled_unit_mass": group_selected_units,
            "covered_modeled_unit_share": (
                group_selected_units / group_total if group_total else 0.0
            ),
        }
    return {
        "covered_location_count": int(selected.sum()),
        "covered_location_share": float(selected.mean()),
        "modeled_unit_mass": total_units,
        "covered_modeled_unit_mass": selected_units,
        "covered_modeled_unit_share": selected_units / total_units if total_units else 0.0,
        "by_service_location_type": by_type,
    }


def build_footprint_access_audit(
    *,
    city_slug: str,
    location_path: Path,
    graph_path: Path,
    output_dir: Path,
    area_crs: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    locations = gpd.read_parquet(location_path)
    edges = build_eligible_physical_edges(graph_path, area_crs)
    anchored = attach_footprint_access(locations, edges, area_crs)
    graph = ox.load_graphml(graph_path)
    component_index = build_directed_component_index(graph)
    anchored = label_projection_connectivity(anchored, component_index)
    output_path = output_dir / "footprint_road_access_candidates.parquet"
    anchored.to_parquet(output_path, index=False)
    access_contract = _write_service_access_contract(anchored, output_dir)
    distances = anchored["road_access_distance_m"].to_numpy(dtype=float)
    quantiles = np.quantile(distances, [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1])
    report = {
        "schema": "evrptw_customer_footprint_access_audit_v2",
        "status": "access_threshold_selection_pilot_not_release_eligible",
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "location_path": str(location_path.resolve()),
            "location_sha256": sha256_file(location_path),
            "graph_path": str(graph_path.resolve()),
            "graph_sha256": sha256_file(graph_path),
            "area_crs": area_crs,
        },
        "edge_contract": {
            "eligible_physical_edge_count": len(edges),
            "inside_city_non_transit_only": True,
            "included_highway_classes": sorted(DELIVERY_ROAD_CLASSES),
            "excluded_highway_classes": sorted(EXCLUDED_ANCHOR_ROAD_CLASSES),
            "reciprocal_edge_policy": "one physical geometry with all directed u/v/key refs retained",
            "projection_contract": "per-directed-edge fractional offset retained for exact active-stop edge splitting",
            "connector_contract": "service access point to road projection is a virtual bidirectional connector with equal speed in both directions",
            "connector_speed_policy": "assigned at instance generation; Stage 1 stores geometry and symmetry only",
            "directed_roundtrip_policy": (
                "default benchmark candidates must inherit the largest directed road SCC; "
                "all other source locations are retained with quarantine labels"
            ),
        },
        "summary": {
            "candidate_location_count": len(anchored),
            "protected_connectivity": connectivity_summary(anchored, component_index),
            "distance_quantiles_m": {
                key: float(value)
                for key, value in zip(
                    ("min", "p50", "p90", "p95", "p99", "p999", "max"),
                    quantiles,
                    strict=True,
                )
            },
            "threshold_sensitivity": {
                str(threshold): _threshold_summary(anchored, threshold)
                for threshold in (25, 50, 100, 200)
            },
        },
        "outputs": {
            "footprint_road_access_candidates": str(output_path),
            **access_contract["paths"],
        },
        "output_sha256": {
            "footprint_road_access_candidates": sha256_file(output_path),
            **access_contract["sha256"],
        },
        "access_contract_row_counts": access_contract["row_counts"],
        "known_limitations": [
            "No access threshold is frozen from a one-city pilot.",
            "OSM entrance/driveway evidence has not yet been preferred over the geometric fallback.",
            "Virtual nodes are stored as compact ledgers; only an active instance subset is materialized into a routing graph.",
            "Connector speed magnitude remains an instance policy; Stage 1 freezes only equal bidirectional connector-speed semantics.",
            "G2 Microsoft/NSI geometry matches still require manual audit.",
            "SCC eligibility is a routing-topology gate and does not validate location provenance.",
        ],
    }
    write_json(output_dir / "footprint_road_access_audit.json", report)
    return report


def refresh_footprint_access_connectivity(
    *,
    graph_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Relabel a frozen access layer after adding the directed SCC contract.

    Geometry matching and nearest-edge projection are unchanged.  This is the
    reproducible migration path for an already generated CLE cohort.
    """

    candidate_path = output_dir / "footprint_road_access_candidates.parquet"
    report_path = output_dir / "footprint_road_access_audit.json"
    if not candidate_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Existing customer access candidate/report files are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    graph_sha = sha256_file(graph_path)
    if report.get("inputs", {}).get("graph_sha256") != graph_sha:
        raise ValueError("Customer access audit and operational graph hashes differ")

    anchored = gpd.read_parquet(candidate_path)
    graph = ox.load_graphml(graph_path)
    component_index = build_directed_component_index(graph)
    anchored = label_projection_connectivity(anchored, component_index)
    anchored.to_parquet(candidate_path, index=False)
    access_contract = _write_service_access_contract(anchored, output_dir)

    report["schema"] = "evrptw_customer_footprint_access_audit_v3"
    report["generated_utc"] = datetime.now(UTC).isoformat()
    report["status"] = "protected_connectivity_labeled_not_release_eligible"
    report.setdefault("edge_contract", {})["directed_roundtrip_policy"] = (
        "default benchmark candidates must inherit the largest directed road SCC; "
        "all other source locations are retained with quarantine labels"
    )
    report.setdefault("summary", {})["protected_connectivity"] = connectivity_summary(
        anchored, component_index
    )
    report["access_contract_row_counts"] = access_contract["row_counts"]
    report["outputs"] = {
        "footprint_road_access_candidates": str(candidate_path),
        **access_contract["paths"],
    }
    report["output_sha256"] = {
        "footprint_road_access_candidates": sha256_file(candidate_path),
        **access_contract["sha256"],
    }
    limitations = report.setdefault("known_limitations", [])
    note = "SCC eligibility is a routing-topology gate and does not validate location provenance."
    if note not in limitations:
        limitations.append(note)
    write_json(report_path, report)
    return report


def _scenario_summary(frame: pd.DataFrame) -> dict[str, Any]:
    distances = frame["road_access_distance_m"].to_numpy(dtype=float)
    quantiles = np.quantile(distances, [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1])
    return {
        "distance_quantiles_m": {
            key: float(value)
            for key, value in zip(
                ("min", "p50", "p90", "p95", "p99", "p999", "max"),
                quantiles,
                strict=True,
            )
        },
        "best_access_layer_counts": {
            str(key): int(value)
            for key, value in frame["access_layer"].value_counts().sort_index().items()
        },
        "threshold_sensitivity": {
            str(threshold): _threshold_summary(frame, threshold) for threshold in (25, 50, 100, 200)
        },
    }


def build_terminal_scenario_access_audit(
    *,
    city_slug: str,
    location_path: Path,
    operational_graph_path: Path,
    terminal_graph_path: Path,
    output_dir: Path,
    area_crs: str,
) -> dict[str, Any]:
    """Compare operational-only, nonprivate-terminal, and private sensitivity access."""

    output_dir.mkdir(parents=True, exist_ok=True)
    locations = gpd.read_parquet(location_path)
    operational = build_eligible_physical_edges(operational_graph_path, area_crs)
    terminal_graph_sha = sha256_file(terminal_graph_path)
    cache_path = output_dir / "terminal_physical_edges_audit_cache.parquet"
    cache_manifest_path = output_dir / "terminal_physical_edges_audit_cache.json"
    cache_manifest = (
        json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest_path.exists()
        else {}
    )
    cache_valid = (
        cache_path.exists()
        and cache_manifest.get("terminal_graph_sha256") == terminal_graph_sha
        and cache_manifest.get("area_crs") == area_crs
        and cache_manifest.get("directed_edge_refs_complete") is False
    )
    if cache_valid:
        terminal_with_private = gpd.read_parquet(cache_path)
    else:
        terminal_with_private = build_terminal_physical_edges(
            terminal_graph_path,
            area_crs,
            include_permit_required=True,
            retain_all_directed_refs=False,
        )
        terminal_with_private.to_parquet(cache_path, index=False)
        write_json(
            cache_manifest_path,
            {
                "schema": "evrptw_terminal_physical_edge_audit_cache_v1",
                "terminal_graph_sha256": terminal_graph_sha,
                "area_crs": area_crs,
                "physical_edge_count": len(terminal_with_private),
                "directed_edge_refs_complete": False,
                "parquet_sha256": sha256_file(cache_path),
            },
        )
    terminal_default = terminal_with_private.loc[
        terminal_with_private["legal_access_tier"] != "permit_required"
    ].copy()
    operational_only = attach_footprint_access(locations, operational, area_crs)
    default_edges = gpd.GeoDataFrame(
        pd.concat([operational, terminal_default], ignore_index=True),
        geometry="geometry",
        crs=area_crs,
    )
    all_edges = gpd.GeoDataFrame(
        pd.concat([operational, terminal_with_private], ignore_index=True),
        geometry="geometry",
        crs=area_crs,
    )
    default = attach_footprint_access(locations, default_edges, area_crs)
    permit = attach_footprint_access(locations, all_edges, area_crs)

    result = default.copy()
    result["operational_only_distance_m"] = operational_only["road_access_distance_m"].to_numpy(
        dtype=float
    )
    result["permit_sensitivity_distance_m"] = permit["road_access_distance_m"].to_numpy(dtype=float)
    result["permit_sensitivity_access_layer"] = permit["access_layer"].to_numpy()
    result["permit_sensitivity_legal_access_tier"] = permit["legal_access_tier"].to_numpy()
    output_path = output_dir / "footprint_terminal_access_scenarios.parquet"
    result.to_parquet(output_path, index=False)

    report = {
        "schema": "evrptw_customer_terminal_access_scenario_audit_v1",
        "status": "pilot_not_release_eligible",
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "location_path": str(location_path.resolve()),
            "location_sha256": sha256_file(location_path),
            "operational_graph_path": str(operational_graph_path.resolve()),
            "operational_graph_sha256": sha256_file(operational_graph_path),
            "terminal_graph_path": str(terminal_graph_path.resolve()),
            "terminal_graph_sha256": terminal_graph_sha,
            "area_crs": area_crs,
        },
        "physical_edge_counts": {
            "operational": len(operational),
            "terminal_nonprivate": len(terminal_default),
            "terminal_including_private": len(terminal_with_private),
        },
        "scenarios": {
            "operational_only": _scenario_summary(operational_only),
            "operational_plus_nonprivate_terminal": _scenario_summary(default),
            "operational_plus_private_permit_sensitivity": _scenario_summary(permit),
        },
        "outputs": {
            "footprint_terminal_access_scenarios": str(output_path.resolve()),
            "terminal_physical_edge_audit_cache": str(cache_path.resolve()),
        },
        "output_sha256": {
            "footprint_terminal_access_scenarios": sha256_file(output_path),
            "terminal_physical_edge_audit_cache": sha256_file(cache_path),
        },
        "decision": {
            "access_threshold_frozen": False,
            "private_access_in_default": False,
            "reason": "One-city scenario audit cannot freeze global threshold or private permission semantics.",
        },
    }
    write_json(output_dir / "footprint_terminal_access_scenario_audit.json", report)
    return report
