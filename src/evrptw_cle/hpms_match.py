"""Conflate normalized HPMS line segments with directed OSM graph edges.

The matcher is deliberately split into two evidence levels:

* a high-confidence physical-corridor match can provide ``F_SYSTEM``; and
* ``SPEED_LIMIT`` can fill a directed OSM edge only when the matched physical
  OSM segment has a unique, one-way graph direction.

Line coordinate order is ignored during geometric matching.  The original OSM
``u -> v`` direction is never modified.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString, Point

from .road_tags import _as_bool
from .speed import (
    _canonical_values,
    _direction_label,
    _primary_highway,
    operating_mode_from_hpms,
    operating_mode_from_osm,
)
from .util import write_json

SUPPORTED_HPMS_SUFFIXES = (".parquet", ".geojson", ".json", ".gpkg", ".shp")
NUMBER_TOKEN = re.compile(r"\d+")
REQUIRED_MATCH_COLUMNS = {
    "edge_id",
    "F_SYSTEM",
    "SPEED_LIMIT",
    "match_confidence",
    "corridor_match_usable",
    "direction_verified",
    "hpms_speed_usable",
}

FIELD_ALIASES = {
    "segment_id": ("objectid", "object_id", "segment_id", "section_id", "id"),
    "route_id": ("routeid", "route_id"),
    "route_number": (
        "route_number",
        "route_numb",
        "route_num",
        "route_nu_1",
        "route_number_text",
    ),
    "f_system": ("f_system", "fsystem", "functional_system"),
    "speed_limit": ("speed_limit", "speed_limi", "speedlimit"),
    "data_year": ("datayear", "data_year", "year_record", "year_recor"),
    "facility_type": ("facility_type", "facility_t"),
}


@dataclass(frozen=True)
class HPMSMatchOptions:
    """Versioned geometric tolerances for candidate generation and confidence."""

    candidate_radius_m: float = 75.0
    overlap_buffer_m: float = 25.0
    minimum_overlap_ratio: float = 0.20
    maximum_orientation_delta_deg: float = 30.0
    high_confidence_distance_m: float = 25.0
    high_confidence_overlap_ratio: float = 0.50
    high_confidence_orientation_delta_deg: float = 15.0
    ambiguity_distance_margin_m: float = 10.0
    ambiguity_overlap_margin: float = 0.20

    def validate(self) -> None:
        positive = {
            "candidate_radius_m": self.candidate_radius_m,
            "overlap_buffer_m": self.overlap_buffer_m,
            "maximum_orientation_delta_deg": self.maximum_orientation_delta_deg,
            "high_confidence_distance_m": self.high_confidence_distance_m,
            "high_confidence_overlap_ratio": self.high_confidence_overlap_ratio,
            "high_confidence_orientation_delta_deg": (self.high_confidence_orientation_delta_deg),
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"HPMS match options must be positive: {invalid}")
        if not 0 < self.minimum_overlap_ratio <= 1:
            raise ValueError("minimum_overlap_ratio must be in (0, 1]")
        if self.high_confidence_overlap_ratio < self.minimum_overlap_ratio:
            raise ValueError("high_confidence_overlap_ratio cannot be below minimum_overlap_ratio")
        if self.high_confidence_distance_m > self.candidate_radius_m:
            raise ValueError("high_confidence_distance_m cannot exceed candidate_radius_m")
        if self.high_confidence_orientation_delta_deg > (self.maximum_orientation_delta_deg):
            raise ValueError(
                "high_confidence_orientation_delta_deg cannot exceed maximum_orientation_delta_deg"
            )
        if self.ambiguity_distance_margin_m < 0 or self.ambiguity_overlap_margin < 0:
            raise ValueError("HPMS ambiguity margins cannot be negative")


def discover_hpms_source(source_root: Path, source_stem: str) -> Path | None:
    """Resolve one supported HPMS source without hard-coding its file format."""

    for suffix in SUPPORTED_HPMS_SUFFIXES:
        candidate = source_root / f"{source_stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def validate_hpms_edge_matches(path: Path) -> dict[str, int]:
    """Validate the normalized v1 contract before a CLE consumes it."""

    frame = (
        pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".csv"
        else pd.read_parquet(path)
    )
    missing = REQUIRED_MATCH_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"HPMS edge evidence is missing v1 columns: {sorted(missing)}")
    if frame["edge_id"].astype(str).duplicated().any():
        raise ValueError("HPMS edge evidence contains duplicate edge_id values")
    return {
        "edge_count": len(frame),
        "corridor_usable_count": int(frame["corridor_match_usable"].map(_as_bool).sum()),
        "direction_verified_count": int(frame["direction_verified"].map(_as_bool).sum()),
        "speed_usable_count": int(frame["hpms_speed_usable"].map(_as_bool).sum()),
    }


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    by_lower = {str(name).lower(): str(name) for name in frame.columns}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _values(frame: pd.DataFrame, key: str, default: Any = None) -> pd.Series:
    column = _column(frame, FIELD_ALIASES[key])
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    return frame[column]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _route_tokens(value: Any) -> frozenset[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return frozenset()
    tokens = []
    for item in NUMBER_TOKEN.findall(str(value)):
        tokens.append(str(int(item)))
    return frozenset(tokens)


def _route_relation(hpms_value: Any, osm_value: Any) -> str:
    hpms_tokens = _route_tokens(hpms_value)
    osm_tokens = _route_tokens(osm_value)
    if not hpms_tokens or not osm_tokens:
        return "unknown"
    return "exact" if hpms_tokens & osm_tokens else "conflict"


def _as_line(geometry: Any) -> LineString | None:
    if isinstance(geometry, LineString):
        return geometry if not geometry.is_empty and geometry.length > 0 else None
    if isinstance(geometry, MultiLineString):
        merged = line_merge(geometry)
        if isinstance(merged, LineString):
            return merged if merged.length > 0 else None
        parts = [part for part in merged.geoms if part.length > 0]
        return max(parts, key=lambda item: item.length) if parts else None
    return None


def _orientation(line: LineString, anchor: Point) -> float:
    """Return a local, storage-direction-invariant orientation in [0, 180)."""

    position = float(line.project(anchor))
    window = min(25.0, max(2.0, line.length * 0.20))
    start = max(0.0, position - window)
    end = min(float(line.length), position + window)
    if math.isclose(start, end):
        start, end = 0.0, float(line.length)
    first = line.interpolate(start)
    second = line.interpolate(end)
    return math.degrees(math.atan2(second.y - first.y, second.x - first.x)) % 180.0


def _orientation_delta(first: float, second: float) -> float:
    delta = abs(first - second) % 180.0
    return min(delta, 180.0 - delta)


def _load_hpms(path: Path, boundary_path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        raw = gpd.read_parquet(path)
    else:
        raw = gpd.read_file(path)
    if raw.crs is None:
        raise ValueError(f"HPMS source has no CRS: {path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError(f"Boundary has no CRS: {boundary_path}")
    raw = raw.to_crs(boundary.crs)
    boundary_union = boundary.geometry.union_all()
    raw = raw.loc[raw.geometry.intersects(boundary_union)].copy()
    raw = raw.explode(index_parts=False, ignore_index=True)
    raw = raw.loc[raw.geometry.map(_as_line).notna()].copy()
    if raw.empty:
        raise ValueError(f"HPMS source has no line segments in city boundary: {path}")

    segment_ids = _values(raw, "segment_id")
    normalized = gpd.GeoDataFrame(
        {
            "hpms_segment_id": [
                str(value) if value not in (None, "") else f"row:{index}"
                for index, value in enumerate(segment_ids)
            ],
            "hpms_route_id": _values(raw, "route_id", "").fillna("").astype(str),
            "hpms_route_number": _values(raw, "route_number"),
            "F_SYSTEM": pd.to_numeric(_values(raw, "f_system"), errors="coerce"),
            "SPEED_LIMIT": pd.to_numeric(_values(raw, "speed_limit"), errors="coerce"),
            "hpms_data_year": pd.to_numeric(_values(raw, "data_year"), errors="coerce"),
            "hpms_facility_type": pd.to_numeric(_values(raw, "facility_type"), errors="coerce"),
        },
        geometry=[_as_line(value) for value in raw.geometry],
        crs=raw.crs,
    )
    return normalized


def _physical_id(row: pd.Series) -> str:
    osmid = _canonical_values(row.get("osmid"))
    u, v = str(row["edge_u"]), str(row["edge_v"])
    endpoints = f"{min(u, v)}:{max(u, v)}"
    length = float(row.get("length") or row.geometry.length)
    if osmid:
        return f"osmid:{osmid}|{endpoints}|{length:.3f}"
    return f"edge:{endpoints}|{length:.3f}"


def _load_osm_edges(graph_path: Path, local_crs: Any) -> gpd.GeoDataFrame:
    graph = ox.load_graphml(graph_path)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True).reset_index()
    edges = edges.rename(columns={"u": "edge_u", "v": "edge_v", "key": "edge_key"})
    edges["edge_id"] = (
        edges["edge_u"].astype(str)
        + ":"
        + edges["edge_v"].astype(str)
        + ":"
        + edges["edge_key"].astype(str)
    )
    edges["highway_primary"] = edges.get("highway", pd.Series(index=edges.index)).map(
        _primary_highway
    )
    edges["osm_ref"] = edges.get("ref", pd.Series(index=edges.index, dtype="object"))
    edges["oneway_normalized"] = edges.get("oneway", pd.Series(False, index=edges.index)).map(
        _as_bool
    )
    edges["direction_label"] = [
        _direction_label(value, edge_id)
        for value, edge_id in zip(
            edges.get("reversed", pd.Series(False, index=edges.index)),
            edges["edge_id"],
            strict=True,
        )
    ]
    projected = edges.to_crs(local_crs)
    projected["physical_segment_id"] = projected.apply(_physical_id, axis=1)
    return projected


def _consistent_values(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_f = _number(first.get("F_SYSTEM"))
    second_f = _number(second.get("F_SYSTEM"))
    first_s = _number(first.get("SPEED_LIMIT"))
    second_s = _number(second.get("SPEED_LIMIT"))
    return first_f == second_f and first_s == second_s


def _is_ambiguous(
    best: dict[str, Any], second: dict[str, Any] | None, options: HPMSMatchOptions
) -> bool:
    if second is None or _consistent_values(best, second):
        return False
    if best["route_relation"] == "exact" and second["route_relation"] != "exact":
        return False
    distance_better = (
        second["lateral_distance_m"] - best["lateral_distance_m"]
        >= options.ambiguity_distance_margin_m
    )
    overlap_better = (
        best["overlap_ratio"] - second["overlap_ratio"] >= options.ambiguity_overlap_margin
    )
    return not (distance_better or overlap_better)


def _candidate(
    edge_line: LineString,
    edge_highway: str,
    edge_ref: Any,
    hpms_row: pd.Series,
    options: HPMSMatchOptions,
) -> dict[str, Any] | None:
    hpms_line = _as_line(hpms_row.geometry)
    if hpms_line is None:
        return None
    distance = float(edge_line.distance(hpms_line))
    if distance > options.candidate_radius_m:
        return None
    anchor = hpms_line.interpolate(0.5, normalized=True)
    orientation_delta = _orientation_delta(
        _orientation(edge_line, anchor), _orientation(hpms_line, anchor)
    )
    if orientation_delta > options.maximum_orientation_delta_deg:
        return None
    overlap_length = float(
        edge_line.intersection(hpms_line.buffer(options.overlap_buffer_m, cap_style=2)).length
    )
    overlap_ratio = min(1.0, overlap_length / max(1e-9, min(edge_line.length, hpms_line.length)))
    if overlap_ratio < options.minimum_overlap_ratio:
        return None
    route_value = hpms_row["hpms_route_number"]
    if not _route_tokens(route_value):
        route_value = hpms_row["hpms_route_id"]
    route_relation = _route_relation(route_value, edge_ref)
    if route_relation == "conflict":
        return None
    hpms_mode = operating_mode_from_hpms(hpms_row["F_SYSTEM"])
    osm_mode = operating_mode_from_osm(edge_highway)
    class_relation = (
        "unknown" if hpms_mode is None else "compatible" if hpms_mode == osm_mode else "different"
    )
    return {
        **hpms_row.drop(labels="geometry").to_dict(),
        "route_relation": route_relation,
        "class_relation": class_relation,
        "lateral_distance_m": distance,
        "overlap_ratio": overlap_ratio,
        "orientation_difference_deg": orientation_delta,
    }


def _confidence(candidate: dict[str, Any], ambiguous: bool, options: HPMSMatchOptions) -> str:
    if ambiguous:
        return "ambiguous"
    exact_or_nearly_coincident = candidate["route_relation"] == "exact" or (
        candidate["lateral_distance_m"] <= 5.0 and candidate["overlap_ratio"] >= 0.80
    )
    if (
        exact_or_nearly_coincident
        and candidate["lateral_distance_m"] <= options.high_confidence_distance_m
        and candidate["overlap_ratio"] >= options.high_confidence_overlap_ratio
        and candidate["orientation_difference_deg"] <= options.high_confidence_orientation_delta_deg
    ):
        return "high"
    return "medium"


def build_hpms_edge_matches(
    *,
    city_slug: str,
    hpms_path: Path,
    graph_path: Path,
    boundary_path: Path,
    output_path: Path,
    options: HPMSMatchOptions | None = None,
) -> dict[str, Any]:
    """Build one normalized, auditable HPMS-to-directed-OSM evidence table."""

    options = options or HPMSMatchOptions()
    options.validate()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite HPMS match output: {output_path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError(f"Boundary has no CRS: {boundary_path}")
    local_crs = boundary.estimate_utm_crs()
    if local_crs is None:
        raise ValueError(f"Could not estimate local CRS for {city_slug}")
    hpms = _load_hpms(hpms_path, boundary_path).to_crs(local_crs)
    edges = _load_osm_edges(graph_path, local_crs)
    hpms_index = hpms.sindex
    records: list[dict[str, Any]] = []
    matched_physical = 0

    for physical_id, group in edges.groupby("physical_segment_id", sort=True):
        representative = group.iloc[0]
        edge_line = _as_line(representative.geometry)
        if edge_line is None:
            continue
        candidate_indices = hpms_index.query(
            edge_line.buffer(options.candidate_radius_m), predicate="intersects"
        )
        candidates = []
        for index in np.asarray(candidate_indices, dtype=int):
            item = _candidate(
                edge_line,
                str(representative["highway_primary"]),
                representative["osm_ref"],
                hpms.iloc[index],
                options,
            )
            if item is not None:
                candidates.append(item)
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                0 if item["route_relation"] == "exact" else 1,
                0 if item["class_relation"] == "compatible" else 1,
                -item["overlap_ratio"],
                item["lateral_distance_m"],
                item["orientation_difference_deg"],
                str(item["hpms_segment_id"]),
            )
        )
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        ambiguous = _is_ambiguous(best, second, options)
        confidence = _confidence(best, ambiguous, options)
        direction_labels = set(group["direction_label"].astype(str))
        direction_verified = (
            bool(group["oneway_normalized"].all())
            and len(direction_labels) == 1
            and direction_labels <= {"forward", "reverse"}
        )
        speed_value = _number(best.get("SPEED_LIMIT"))
        speed_present = speed_value is not None and speed_value > 0
        speed_usable = confidence == "high" and direction_verified and speed_present
        corridor_usable = confidence == "high"
        matched_physical += 1
        for _, edge in group.sort_values("edge_id").iterrows():
            records.append(
                {
                    "edge_id": str(edge["edge_id"]),
                    "edge_u": str(edge["edge_u"]),
                    "edge_v": str(edge["edge_v"]),
                    "edge_key": str(edge["edge_key"]),
                    "physical_segment_id": str(physical_id),
                    **best,
                    "match_method": "city_projected_corridor_v1",
                    "match_confidence": confidence,
                    "corridor_match_usable": corridor_usable,
                    "direction_verified": direction_verified,
                    "hpms_speed_usable": speed_usable,
                    "candidate_count": len(candidates),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "edge_id",
                "edge_u",
                "edge_v",
                "edge_key",
                "physical_segment_id",
                "hpms_segment_id",
                "F_SYSTEM",
                "SPEED_LIMIT",
                "match_confidence",
                "corridor_match_usable",
                "direction_verified",
                "hpms_speed_usable",
            ]
        )
    frame = frame.sort_values("edge_id").reset_index(drop=True)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        frame.to_parquet(output_path, index=False)

    summary = {
        "schema": "evrptw_hpms_osm_match_v1",
        "city_slug": city_slug,
        "inputs": {
            "hpms_path": str(hpms_path.resolve()),
            "graph_path": str(graph_path.resolve()),
            "boundary_path": str(boundary_path.resolve()),
        },
        "options": options.__dict__,
        "counts": {
            "hpms_segments_in_boundary": len(hpms),
            "directed_osm_edges": len(edges),
            "matched_physical_segments": int(matched_physical),
            "matched_directed_edges": len(frame),
            "high_confidence_edges": int(
                frame.get("match_confidence", pd.Series(dtype=str)).eq("high").sum()
            ),
            "direction_verified_edges": int(
                frame.get("direction_verified", pd.Series(dtype=bool)).fillna(False).sum()
            ),
            "hpms_speed_usable_edges": int(
                frame.get("hpms_speed_usable", pd.Series(dtype=bool)).fillna(False).sum()
            ),
        },
        "output": str(output_path.resolve()),
    }
    write_json(output_path.with_suffix(".json"), summary)
    return summary
