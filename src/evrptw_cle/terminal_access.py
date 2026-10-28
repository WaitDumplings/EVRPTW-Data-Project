"""OSM terminal-access connector extraction and audit.

The operational graph intentionally excludes service ways and explicitly
private roads.  This module retains those real OSM ways in a separate layer so
they can be used only to enter or leave a delivery location, never as a
through-routing shortcut.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import shapely
from shapely.strtree import STRtree

from .osm_graph import _run_osmium, _tag_values
from .util import sha256_file, write_json

_TERMINAL_HIGHWAYS = {"service", "residential", "living_street", "unclassified"}
_EXCLUDED_SERVICE_VALUES = {"emergency_access"}
_KNOWN_CONNECTOR_KINDS = {
    "alley",
    "drive-through",
    "driveway",
    "parking",
    "parking_aisle",
    "slipway",
}
_DEFAULT_LEGAL_ACCESS_TIERS = {
    "explicit_yes",
    "destination_or_customers",
    "delivery_only",
    "permissive",
    "unspecified",
}
_MOTOR_ACCESS_PRECEDENCE = ("motorcar", "motor_vehicle", "vehicle", "access:delivery", "access")
ACCESS_NODE_TAGS = (
    "access",
    "access:delivery",
    "vehicle",
    "motor_vehicle",
    "motorcar",
    "barrier",
    "locked",
)
ACCESS_NODE_TAG_CONTRACT = "complete_way_restriction_barrier_delivery_lock_nodes_v3"


def _normalized_values(value: Any) -> list[str]:
    return [item.strip().lower() for item in _tag_values(value) if item.strip()]


def _effective_motor_access(data: dict[str, Any]) -> tuple[str, str]:
    """Resolve motor access using the most specific populated OSM tag."""

    for field in _MOTOR_ACCESS_PRECEDENCE:
        values = _normalized_values(data.get(field))
        if not values:
            continue
        if any(value in {"no", "prohibited", "none"} for value in values):
            return field, "prohibited"
        if "private" in values:
            return field, "permit_required"
        if any(value in {"destination", "customers"} for value in values):
            return field, "destination_or_customers"
        if "delivery" in values:
            return field, "delivery_only"
        if field == "access:delivery" and any(value in {"yes", "permissive"} for value in values):
            # A delivery exception may override generic restrictions for the
            # invited delivery, but cannot grant permission for public transit.
            _, baseline = _effective_motor_access(
                {key: value for key, value in data.items() if key != "access:delivery"}
            )
            if baseline not in {"explicit_yes", "permissive", "unspecified"}:
                return field, "delivery_only"
        if "permissive" in values:
            return field, "permissive"
        if "yes" in values:
            return field, "explicit_yes"
        return field, "unresolved_tag_value"
    return "none", "unspecified"


def classify_terminal_access_way(data: dict[str, Any]) -> dict[str, Any]:
    """Classify one OSM edge for terminal-only delivery access.

    `core_eligible` means that the tag evidence does not explicitly require
    private permission.  It does not make the edge a public through road.
    """

    highways = set(_normalized_values(data.get("highway")))
    services = set(_normalized_values(data.get("service")))
    access_field, legal_tier = _effective_motor_access(data)
    reason = "eligible_terminal_way"

    if not highways or not highways.intersection(_TERMINAL_HIGHWAYS):
        reason = "unsupported_highway"
    elif "yes" in _normalized_values(data.get("area")):
        reason = "area_feature_not_road"
    elif services.intersection(_EXCLUDED_SERVICE_VALUES):
        reason = "emergency_access_excluded"
    elif legal_tier == "prohibited":
        reason = "motor_access_prohibited"
    elif "service" not in highways and legal_tier not in {"permit_required", "delivery_only"}:
        # Public residential/living-street/unclassified roads already belong in
        # graph_operational. Private and delivery-only examples stay separate
        # for the applicable permission/purpose-specific scenario.
        reason = "already_operational_or_not_terminal"

    candidate = reason == "eligible_terminal_way"
    core_eligible = candidate and legal_tier in _DEFAULT_LEGAL_ACCESS_TIERS
    if services:
        known = services.intersection(_KNOWN_CONNECTOR_KINDS)
        connector_kind = min(known) if known else "service_other"
    elif "service" in highways:
        connector_kind = "service_unspecified"
    else:
        connector_kind = "private_local_road"
    return {
        "terminal_candidate": candidate,
        "terminal_core_eligible": core_eligible,
        "terminal_reason": reason,
        "connector_kind": connector_kind,
        "legal_access_source_tag": access_field,
        "legal_access_tier": legal_tier,
    }


def _extract_terminal_candidate_graph(
    *, pbf_file: Path, boundary_file: Path, working_dir: Path
) -> nx.MultiDiGraph:
    city_pbf = working_dir / "city_complete_ways.osm.pbf"
    access_pbf = working_dir / "terminal_access_candidates.osm.pbf"
    access_xml = working_dir / "terminal_access_candidates.osm"
    _run_osmium(
        [
            "extract",
            "--strategy",
            "complete_ways",
            "--polygon",
            str(boundary_file),
            "--output",
            str(city_pbf),
            "--overwrite",
            str(pbf_file),
        ]
    )
    _run_osmium(
        [
            "tags-filter",
            "--output",
            str(access_pbf),
            "--overwrite",
            str(city_pbf),
            "w/highway=service",
            "w/service",
            "w/access=private",
            "w/vehicle=private",
            "w/motor_vehicle=private",
            "w/motorcar=private",
            "w/access=delivery",
            "w/vehicle=delivery",
            "w/motor_vehicle=delivery",
            "w/motorcar=delivery",
        ]
    )
    _run_osmium(
        [
            "cat",
            "--output",
            str(access_xml),
            "--output-format",
            "osm",
            "--overwrite",
            str(access_pbf),
        ]
    )
    return ox.graph.graph_from_xml(
        access_xml,
        bidirectional=False,
        simplify=False,
        retain_all=True,
    )


def _filter_and_classify_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    result = graph.copy()
    rejected: list[tuple[Any, Any, Any]] = []
    for u, v, key, data in result.edges(keys=True, data=True):
        classification = classify_terminal_access_way(data)
        if not classification["terminal_candidate"]:
            rejected.append((u, v, key))
            continue
        data.update(classification)
        data["access_layer"] = "osm_terminal_only"
        data["through_routing_allowed"] = False
    result.remove_edges_from(rejected)
    result.remove_nodes_from(list(nx.isolates(result)))
    if len(result) == 0:
        raise RuntimeError("No OSM terminal-access candidate ways remained after filtering")
    return result


def _assign_connection_components(
    graph: nx.MultiDiGraph,
    *,
    operational_graph_path: Path,
    area_crs: str,
    connection_tolerance_m: float,
) -> dict[str, Any]:
    components = sorted(
        nx.weakly_connected_components(graph),
        key=lambda nodes: (-len(nodes), min(str(node) for node in nodes)),
    )
    node_component: dict[Any, int] = {}
    for component_id, nodes in enumerate(components):
        for node in nodes:
            node_component[node] = component_id

    terminal_edges = ox.graph_to_gdfs(
        graph, nodes=False, edges=True, fill_edge_geometry=True
    ).reset_index()
    operational = ox.load_graphml(operational_graph_path)
    operational_edges = ox.graph_to_gdfs(
        operational, nodes=False, edges=True, fill_edge_geometry=True
    ).reset_index()
    if "transit_only" in operational_edges.columns:
        transit = operational_edges["transit_only"].astype(str).str.lower().isin({"true", "1"})
        operational_edges = operational_edges.loc[~transit].copy()
    terminal_projected = terminal_edges.to_crs(area_crs)
    operational_projected = operational_edges.to_crs(area_crs)
    operational_geometries = np.asarray(operational_projected.geometry.to_numpy(), dtype=object)
    tree = STRtree(operational_geometries)
    terminal_geometries = np.asarray(terminal_projected.geometry.to_numpy(), dtype=object)
    nearest = np.asarray(tree.nearest(terminal_geometries), dtype=int)
    edge_distances = np.asarray(
        shapely.distance(terminal_geometries, operational_geometries[nearest]), dtype=float
    )

    component_min_distance: dict[int, float] = {}
    for row, distance in zip(terminal_edges.itertuples(index=False), edge_distances, strict=True):
        component_id = node_component[row.u]
        component_min_distance[component_id] = min(
            component_min_distance.get(component_id, float("inf")), float(distance)
        )
    for node, component_id in node_component.items():
        graph.nodes[node]["terminal_component_id"] = component_id
        graph.nodes[node]["connected_to_operational"] = (
            component_min_distance[component_id] <= connection_tolerance_m
        )
    for u, v, key, data in graph.edges(keys=True, data=True):
        component_id = node_component[u]
        distance = component_min_distance[component_id]
        data["terminal_component_id"] = component_id
        data["operational_connection_distance_m"] = distance
        data["connected_to_operational"] = distance <= connection_tolerance_m
        data["terminal_core_eligible"] = bool(
            data["terminal_core_eligible"] and data["connected_to_operational"]
        )

    connected = sum(
        distance <= connection_tolerance_m for distance in component_min_distance.values()
    )
    return {
        "component_count": len(components),
        "connected_component_count": connected,
        "disconnected_component_count": len(components) - connected,
        "connection_tolerance_m": connection_tolerance_m,
        "component_connection_distance_quantiles_m": {
            key: float(value)
            for key, value in zip(
                ("min", "p50", "p90", "p95", "p99", "max"),
                np.quantile(
                    np.asarray(list(component_min_distance.values()), dtype=float),
                    [0, 0.5, 0.9, 0.95, 0.99, 1],
                ),
                strict=True,
            )
        },
    }


def build_terminal_access_layer(
    *,
    city_slug: str,
    pbf_file: Path,
    boundary_file: Path,
    operational_graph_path: Path,
    output_graph_path: Path,
    output_report_path: Path,
    area_crs: str,
    connection_tolerance_m: float = 2.0,
) -> dict[str, Any]:
    """Extract, classify, connect, and persist a terminal-only OSM graph."""

    for source in (pbf_file, boundary_file, operational_graph_path):
        if not source.exists():
            raise FileNotFoundError(source)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{city_slug}-terminal-access-", dir=output_graph_path.parent
    ) as temporary:
        graph = _extract_terminal_candidate_graph(
            pbf_file=pbf_file.resolve(),
            boundary_file=boundary_file.resolve(),
            working_dir=Path(temporary),
        )
    graph = _filter_and_classify_graph(graph)
    connection = _assign_connection_components(
        graph,
        operational_graph_path=operational_graph_path,
        area_crs=area_crs,
        connection_tolerance_m=connection_tolerance_m,
    )
    graph.graph["access_layer"] = "osm_terminal_only"
    graph.graph["through_routing_allowed"] = False
    graph.graph["source_pbf_sha256"] = sha256_file(pbf_file)
    ox.io.save_graphml(graph, output_graph_path)

    directed_edges = list(graph.edges(keys=True, data=True))
    physical_osm_ways = {
        str(osmid) for _, _, _, data in directed_edges for osmid in _tag_values(data.get("osmid"))
    }
    report = {
        "schema": "evrptw_osm_terminal_access_layer_v1",
        "status": "pilot_not_release_eligible",
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "semantics": {
            "through_routing_allowed": False,
            "default_profile": "nonprivate_terminal_only",
            "private_profile": "permit_required_sensitivity_only",
            "prohibited_access_policy": "excluded",
        },
        "inputs": {
            "pbf_file": str(pbf_file.resolve()),
            "pbf_sha256": sha256_file(pbf_file),
            "boundary_file": str(boundary_file.resolve()),
            "boundary_sha256": sha256_file(boundary_file),
            "operational_graph": str(operational_graph_path.resolve()),
            "operational_graph_sha256": sha256_file(operational_graph_path),
            "area_crs": area_crs,
        },
        "summary": {
            "node_count": len(graph.nodes),
            "directed_edge_count": len(directed_edges),
            "physical_osm_way_count": len(physical_osm_ways),
            "connected_directed_edge_count": sum(
                bool(data.get("connected_to_operational")) for _, _, _, data in directed_edges
            ),
            "default_core_directed_edge_count": sum(
                bool(data.get("terminal_core_eligible")) for _, _, _, data in directed_edges
            ),
            "permit_required_directed_edge_count": sum(
                data.get("legal_access_tier") == "permit_required"
                and bool(data.get("connected_to_operational"))
                for _, _, _, data in directed_edges
            ),
            "connector_kind_counts": dict(
                sorted(
                    Counter(str(data.get("connector_kind")) for *_, data in directed_edges).items()
                )
            ),
            "legal_access_tier_counts": dict(
                sorted(
                    Counter(
                        str(data.get("legal_access_tier")) for *_, data in directed_edges
                    ).items()
                )
            ),
            **connection,
        },
        "output": {
            "graph_terminal_access": str(output_graph_path.resolve()),
            "graph_terminal_access_sha256": sha256_file(output_graph_path),
        },
        "known_limitations": [
            "OSM service/access tags do not prove that an individual delivery vehicle has permission.",
            "Unspecified-access driveway ways are retained as terminal-only candidates, not public through roads.",
            "The 2 m topological connection tolerance is an entity-resolution tolerance, not a synthetic road.",
            "The customer access threshold remains unfrozen until eleven-city and manual audits pass.",
        ],
    }
    write_json(output_report_path, report)
    return report
