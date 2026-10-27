from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.ops import unary_union

_EXCLUDED_HIGHWAYS = re.compile(
    r"^(abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|"
    r"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|"
    r"rest_area|service|services|steps|track)$"
)
_EXCLUDED_SERVICE = re.compile(r"^(alley|driveway|emergency_access|parking|parking_aisle|private)$")
_REQUIRED_SPEED_AND_ACCESS_TAGS = (
    "maxspeed:forward",
    "maxspeed:backward",
    "maxspeed:hgv",
    "maxspeed:hgv:forward",
    "maxspeed:hgv:backward",
    "hgv",
    "vehicle",
    "motor_vehicle",
    "motorcar",
    "access:delivery",
)


def _ensure_useful_way_tags() -> None:
    current = list(ox.settings.useful_tags_way)
    for tag in _REQUIRED_SPEED_AND_ACCESS_TAGS:
        if tag not in current:
            current.append(tag)
    ox.settings.useful_tags_way = current


def configure_osmnx(
    cache_dir: Path,
    overpass_url: str,
    request_timeout_s: int,
    max_query_area_km2: float,
    overpass_rate_limit: bool,
) -> None:
    _ensure_useful_way_tags()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.requests_timeout = request_timeout_s
    ox.settings.overpass_url = overpass_url
    ox.settings.overpass_rate_limit = overpass_rate_limit
    ox.settings.max_query_area_size = max_query_area_km2 * 1_000_000
    ox.settings.log_console = True


def prepare_query_polygon(
    query_mask: gpd.GeoDataFrame,
    buffer_m: float,
    simplify_m: float,
    component_min_area_km2: float,
):
    local_crs = query_mask.estimate_utm_crs()
    parts = query_mask.to_crs(local_crs).explode(index_parts=False).reset_index(drop=True)
    areas = parts.geometry.area
    seeds = parts[areas >= component_min_area_km2 * 1_000_000].copy()
    if seeds.empty:
        seeds = parts.loc[[areas.idxmax()]].copy()
    geometry = unary_union(seeds.geometry.tolist())
    geometry = geometry.buffer(buffer_m).simplify(simplify_m, preserve_topology=True)
    return gpd.GeoSeries([geometry], crs=local_crs).to_crs("EPSG:4326").iloc[0]


def download_drive_graph(
    boundary: gpd.GeoDataFrame,
    query_mask: gpd.GeoDataFrame,
    *,
    network_type: str,
    retain_all: bool,
    query_buffer_m: float,
    query_simplify_m: float,
    query_component_min_area_km2: float,
) -> tuple[Any, Any, dict]:
    query_polygon = prepare_query_polygon(
        query_mask,
        query_buffer_m,
        query_simplify_m,
        query_component_min_area_km2,
    )
    envelope_graph = ox.graph.graph_from_polygon(
        query_polygon,
        network_type=network_type,
        simplify=True,
        retain_all=retain_all,
        truncate_by_edge=True,
    )
    graph = ox.truncate.truncate_graph_polygon(
        envelope_graph,
        boundary.geometry.iloc[0],
        truncate_by_edge=False,
    )
    if len(graph) == 0:
        raise RuntimeError("OSM query produced an empty graph after exact boundary clipping")
    if not graph.is_directed() or not graph.is_multigraph():
        raise RuntimeError("Expected a directed MultiDiGraph from OSMnx")
    return (
        graph,
        envelope_graph,
        {
            "network_type": network_type,
            "retain_all_during_download": retain_all,
            "query_buffer_m": query_buffer_m,
            "query_simplify_m": query_simplify_m,
            "query_component_min_area_km2": query_component_min_area_km2,
            "raw_city_clip": "node coordinates inside exact administrative boundary",
            "routing_envelope": "complete queried OSM drive graph retained for operational selection",
        },
    )


def _tag_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _edge_is_drivable(data: dict[str, Any]) -> bool:
    """Public through roads, respecting the most specific motor-access tag."""
    highways = _tag_values(data.get("highway"))
    if not highways or all(_EXCLUDED_HIGHWAYS.fullmatch(value) for value in highways):
        return False
    if "yes" in _tag_values(data.get("area")):
        return False
    effective = next(
        (
            _tag_values(data.get(key))
            for key in ("motorcar", "motor_vehicle", "vehicle", "access:delivery", "access")
            if _tag_values(data.get(key))
        ),
        [],
    )
    if any(value in {"no", "prohibited", "none", "private"} for value in effective):
        return False
    services = _tag_values(data.get("service"))
    return not any(_EXCLUDED_SERVICE.fullmatch(value) for value in services)


def _run_osmium(arguments: list[str]) -> str:
    binary = shutil.which("osmium")
    if binary is None:
        raise RuntimeError(
            "PBF mode requires `osmium-tool` on PATH. Install it from conda-forge "
            "or use the default Overpass backend."
        )
    process = subprocess.run(
        [binary, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"osmium {' '.join(arguments)} failed with code {process.returncode}: "
            f"{process.stderr.strip()}"
        )
    return process.stdout.strip()


def _filter_drive_edges(graph):
    excluded = [
        (u, v, key)
        for u, v, key, data in graph.edges(keys=True, data=True)
        if not _edge_is_drivable(data)
    ]
    graph.remove_edges_from(excluded)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if len(graph) == 0:
        raise RuntimeError("Local PBF contained no drivable OSM ways in the query mask")
    return graph


def load_drive_graph_from_pbf(
    boundary: gpd.GeoDataFrame,
    query_mask: gpd.GeoDataFrame,
    *,
    pbf_file: Path,
    work_dir: Path,
    query_buffer_m: float,
    query_simplify_m: float,
    query_component_min_area_km2: float,
    keep_intermediates: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build an OSMnx-compatible directed drive graph from a frozen local PBF."""
    _ensure_useful_way_tags()
    pbf_file = pbf_file.resolve()
    if not pbf_file.exists():
        raise FileNotFoundError(pbf_file)
    work_dir.mkdir(parents=True, exist_ok=True)

    query_polygon = prepare_query_polygon(
        query_mask,
        query_buffer_m,
        query_simplify_m,
        query_component_min_area_km2,
    )
    query_frame = gpd.GeoDataFrame(
        [{"geometry_role": "pbf_query_mask"}],
        geometry=[query_polygon],
        crs="EPSG:4326",
    )
    projected = query_frame.to_crs(query_frame.estimate_utm_crs())
    extraction_frame = projected.set_geometry(projected.geometry.buffer(500)).to_crs("EPSG:4326")

    polygon_file = work_dir / "query_polygon_buffered.geojson"
    clipped_pbf = work_dir / "city_complete_ways.osm.pbf"
    highway_pbf = work_dir / "city_highways.osm.pbf"
    highway_xml = work_dir / "city_highways.osm"
    extraction_frame.to_file(polygon_file, driver="GeoJSON")
    _run_osmium(
        [
            "extract",
            "--strategy",
            "complete_ways",
            "--polygon",
            str(polygon_file),
            "--output",
            str(clipped_pbf),
            "--overwrite",
            str(pbf_file),
        ]
    )
    _run_osmium(
        [
            "tags-filter",
            "--output",
            str(highway_pbf),
            "--overwrite",
            str(clipped_pbf),
            "w/highway",
        ]
    )
    _run_osmium(
        [
            "cat",
            "--output",
            str(highway_xml),
            "--output-format",
            "osm",
            "--overwrite",
            str(highway_pbf),
        ]
    )

    graph = ox.graph.graph_from_xml(
        highway_xml,
        bidirectional=False,
        simplify=False,
        retain_all=True,
    )
    graph = _filter_drive_edges(graph)
    graph = ox.truncate.truncate_graph_polygon(
        graph,
        extraction_frame.geometry.iloc[0],
        truncate_by_edge=True,
    )
    graph = ox.simplification.simplify_graph(graph)
    graph = ox.truncate.truncate_graph_polygon(
        graph,
        query_polygon,
        truncate_by_edge=True,
    )
    envelope_graph = graph
    graph_exact = ox.truncate.truncate_graph_polygon(
        envelope_graph,
        boundary.geometry.iloc[0],
        truncate_by_edge=False,
    )
    if len(graph_exact) == 0:
        raise RuntimeError("PBF graph was empty after exact administrative-boundary clipping")
    street_count = ox.stats.count_streets_per_node(graph, nodes=graph_exact.nodes)
    nx.set_node_attributes(graph_exact, street_count, "street_count")
    graph = graph_exact

    if not keep_intermediates:
        for path in (polygon_file, clipped_pbf, highway_pbf, highway_xml):
            path.unlink(missing_ok=True)
    replication_timestamp = _run_osmium(
        ["fileinfo", "--get", "header.option.osmosis_replication_timestamp", str(pbf_file)]
    )
    return (
        graph,
        envelope_graph,
        {
            "network_type": "drive",
            "retain_all_during_download": True,
            "query_buffer_m": query_buffer_m,
            "query_simplify_m": query_simplify_m,
            "query_component_min_area_km2": query_component_min_area_km2,
            "local_extraction_buffer_m": 500.0,
            "drive_filter_semantics": "OSMnx 2.1 drive exclusions reproduced after w/highway extraction",
            "raw_city_clip": "node coordinates inside exact administrative boundary",
            "routing_envelope": "complete queried OSM drive graph retained for operational selection",
            "keep_intermediates": keep_intermediates,
            "pbf_replication_timestamp_utc": replication_timestamp or None,
        },
    )
