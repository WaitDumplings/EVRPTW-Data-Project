"""Build auditable directed legal and weekday/weekend reference speeds."""

from __future__ import annotations

import math
import re
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd

from .moves_speed import (
    DEFAULT_MOVES_SPEED_PROFILE_PATH,
    load_moves_speed_profile,
    moves_road_type_from_hpms,
    moves_road_type_from_osm,
    resolve_moves_speed_profile,
    speed_retention_factor,
)
from .road_tags import _as_bool, _parse_multivalue
from .util import sha256_file, write_json

MPH_TO_KPH = 1.609344
NUMERIC_SPEED = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mph|km/h|kph)?", re.IGNORECASE)

OSM_OPERATING_MODE = {
    "motorway": "H",
    "trunk": "H",
    "motorway_link": "M",
    "trunk_link": "M",
    "primary": "M",
    "primary_link": "M",
    "secondary": "M",
    "secondary_link": "M",
    "tertiary": "M",
    "tertiary_link": "M",
    "busway": "M",
    "unclassified": "U",
    "residential": "U",
    "living_street": "U",
    "service": "U",
    "road": "U",
    "unknown": "U",
}


def operating_mode_from_hpms(value: Any) -> str | None:
    """Map FHWA HPMS F_SYSTEM to the portable H/M/U operating modes."""

    try:
        f_system = int(float(value))
    except (TypeError, ValueError):
        return None
    if f_system in {1, 2}:
        return "H"
    if f_system in {3, 4, 5, 6}:
        return "M"
    if f_system == 7:
        return "U"
    return None


def operating_mode_from_osm(highway: str) -> str:
    """Return the U.S. reference fallback mode for an OSM highway class."""

    return OSM_OPERATING_MODE.get(highway, "U")


def parse_osm_maxspeed(value: Any) -> tuple[float | None, str]:
    """Parse numeric OSM maxspeed values; return a conservative minimum for lists."""

    tokens: list[str] = []
    for item in _parse_multivalue(value):
        tokens.extend(piece.strip() for piece in str(item).split(";") if piece.strip())
    if not tokens:
        return None, "missing"
    parsed: list[float] = []
    for token in tokens:
        match = NUMERIC_SPEED.fullmatch(token)
        if match is None:
            return None, "non_numeric_or_conditional"
        speed = float(match.group("value"))
        if (match.group("unit") or "").lower() == "mph":
            speed *= MPH_TO_KPH
        if not math.isfinite(speed) or speed <= 0:
            return None, "invalid_numeric"
        parsed.append(speed)
    status = "parsed_single" if len(parsed) == 1 else "parsed_multivalue_conservative_min"
    return min(parsed), status


def _primary_highway(value: Any) -> str:
    classes = _parse_multivalue(value)
    return str(classes[0]) if classes else "unknown"


def _canonical_values(value: Any) -> str:
    values = sorted({str(item) for item in _parse_multivalue(value) if str(item)})
    return "|".join(values)


def _direction_label(value: Any, edge_id: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "reverse" if bool(value) else "forward"
    values = {str(item).strip().lower() for item in _parse_multivalue(value)}
    truthy = {"true", "1", "yes"}
    falsey = {"false", "0", "no"}
    if values and values <= truthy:
        return "reverse"
    if values and values <= falsey:
        return "forward"
    return f"mixed:{edge_id}"


def _select_directional_speed(attributes: dict[str, Any], edge_id: str) -> dict[str, Any]:
    """Select the legal tag that applies to this directed OSMnx edge.

    OSMnx retains the original OSM way direction in ``reversed``.  Directional
    tags therefore apply as ``forward`` when reversed is false and ``backward``
    when reversed is true.  A malformed directional value does not erase a
    usable generic maxspeed; both raw values and parse statuses remain auditable.
    HGV tags are parsed as evidence only until a vehicle-class contract is
    frozen.
    """

    direction = _direction_label(attributes.get("reversed"), edge_id)
    directional_tag = None
    hgv_directional_tag = None
    if direction == "forward":
        directional_tag = "maxspeed:forward"
        hgv_directional_tag = "maxspeed:hgv:forward"
    elif direction == "reverse":
        directional_tag = "maxspeed:backward"
        hgv_directional_tag = "maxspeed:hgv:backward"

    directional_raw = attributes.get(directional_tag) if directional_tag else None
    directional_speed, directional_status = parse_osm_maxspeed(directional_raw)
    general_raw = attributes.get("maxspeed")
    general_speed, general_status = parse_osm_maxspeed(general_raw)
    if directional_speed is not None:
        selected_speed = directional_speed
        selected_status = directional_status
        selected_source = f"osm_{directional_tag.replace(':', '_')}"
    else:
        selected_speed = general_speed
        selected_status = general_status
        selected_source = "osm_maxspeed" if general_speed is not None else "missing"

    hgv_directional_raw = attributes.get(hgv_directional_tag) if hgv_directional_tag else None
    hgv_directional_speed, hgv_directional_status = parse_osm_maxspeed(hgv_directional_raw)
    hgv_general_raw = attributes.get("maxspeed:hgv")
    hgv_general_speed, hgv_general_status = parse_osm_maxspeed(hgv_general_raw)
    if hgv_directional_speed is not None:
        hgv_speed = hgv_directional_speed
        hgv_status = hgv_directional_status
        hgv_source = f"osm_{hgv_directional_tag.replace(':', '_')}"
    else:
        hgv_speed = hgv_general_speed
        hgv_status = hgv_general_status
        hgv_source = "osm_maxspeed_hgv" if hgv_general_speed is not None else "missing"

    return {
        "direction_label": direction,
        "raw_maxspeed": str(general_raw or ""),
        "raw_directional_maxspeed": str(directional_raw or ""),
        "directional_maxspeed_tag": directional_tag or "unresolved_mixed_direction",
        "directional_maxspeed_present": directional_raw not in (None, ""),
        "directional_maxspeed_parse_status": directional_status,
        "maxspeed_parse_status": selected_status,
        "speed_limit_kph": selected_speed,
        "speed_limit_observed_source": selected_source,
        "raw_hgv_maxspeed": str(hgv_general_raw or ""),
        "raw_directional_hgv_maxspeed": str(hgv_directional_raw or ""),
        "hgv_maxspeed_parse_status": hgv_status,
        "hgv_speed_limit_kph_evidence": hgv_speed,
        "hgv_speed_limit_source": hgv_source,
    }


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy(dtype=float), kind="mergesort")
    sorted_values = values.to_numpy(dtype=float)[order]
    sorted_weights = weights.to_numpy(dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2, side="left")])


def _load_hpms_edge_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load an optional, already-conflated HPMS-to-OSM edge table.

    Raw HPMS geometry is intentionally handled by a source adapter rather than
    hidden inside the core speed model.  The normalized table must contain an
    ``edge_id`` column (or ``edge_u/edge_v/edge_key``), ``F_SYSTEM``, optional
    ``SPEED_LIMIT`` in mph, and a ``match_confidence`` label.  Only high or
    verified matches can replace missing OSM evidence.
    """

    if path is None:
        return {}
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if "edge_id" not in frame.columns:
        required = {"edge_u", "edge_v", "edge_key"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                "HPMS edge evidence needs edge_id or edge_u/edge_v/edge_key; "
                f"missing {sorted(missing)}"
            )
        frame["edge_id"] = (
            frame["edge_u"].astype(str)
            + ":"
            + frame["edge_v"].astype(str)
            + ":"
            + frame["edge_key"].astype(str)
        )
    if frame["edge_id"].duplicated().any():
        raise ValueError("HPMS edge evidence contains duplicate edge_id values")
    return {str(row["edge_id"]): row.to_dict() for _, row in frame.iterrows()}


def _high_confidence_hpms(record: dict[str, Any]) -> bool:
    confidence = str(record.get("match_confidence", "")).strip().lower()
    if confidence in {"high", "verified", "accepted"}:
        return True
    try:
        return float(confidence) >= 0.8
    except (TypeError, ValueError):
        return False


def _hpms_corridor_is_usable(record: dict[str, Any]) -> bool:
    """Return whether HPMS functional class may replace the OSM fallback."""

    if not _high_confidence_hpms(record):
        return False
    if "corridor_match_usable" not in record:
        return True
    return _as_bool(record.get("corridor_match_usable"))


def _hpms_speed_is_usable(record: dict[str, Any]) -> bool:
    """Require an explicitly direction-verified match before using HPMS speed."""

    return (
        _hpms_corridor_is_usable(record)
        and _as_bool(record.get("direction_verified"))
        and _as_bool(record.get("hpms_speed_usable"))
    )


def build_legal_speed_layer(
    *,
    city_slug: str,
    graph_path: Path,
    output_dir: Path,
    hpms_edge_evidence_path: Path | None = None,
    moves_speed_profile_path: Path | None = None,
    state_fips: str | None = None,
    vehicle_speed_cap_kph: float | None = None,
) -> dict[str, Any]:
    """Build a directed legal/MOVES weekday/weekend reference-speed profile."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite speed output: {output_dir}")
    if vehicle_speed_cap_kph is not None and vehicle_speed_cap_kph <= 0:
        raise ValueError("vehicle_speed_cap_kph must be positive when supplied")
    moves_profile_path = resolve_moves_speed_profile(
        moves_speed_profile_path or DEFAULT_MOVES_SPEED_PROFILE_PATH,
        city_slug=city_slug,
        state_fips=state_fips,
    )
    moves_profile = load_moves_speed_profile(moves_profile_path)
    graph = ox.load_graphml(graph_path)
    hpms_evidence = _load_hpms_edge_evidence(hpms_edge_evidence_path)
    graph_edge_ids = {f"{u}:{v}:{key}" for u, v, key in graph.edges(keys=True)}
    stale_hpms_edges = sorted(set(hpms_evidence) - graph_edge_ids)
    if stale_hpms_edges:
        raise ValueError(
            "HPMS edge evidence belongs to a different graph; unknown edge IDs include "
            f"{stale_hpms_edges[:5]}"
        )
    records: list[dict[str, Any]] = []
    parse_counts: Counter[str] = Counter()
    for u, v, key, attributes in graph.edges(keys=True, data=True):
        length = float(attributes.get("length") or 0.0)
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"Invalid edge length for {(u, v, key)}")
        edge_id = f"{u}:{v}:{key}"
        selected_speed = _select_directional_speed(attributes, edge_id)
        hpms = hpms_evidence.get(edge_id, {})
        hpms_corridor_usable = _hpms_corridor_is_usable(hpms)
        hpms_speed_usable = _hpms_speed_is_usable(hpms)
        hpms_speed_kph = None
        try:
            raw_hpms_speed = float(hpms.get("SPEED_LIMIT"))
            if math.isfinite(raw_hpms_speed) and raw_hpms_speed > 0:
                hpms_speed_kph = raw_hpms_speed * MPH_TO_KPH
        except (TypeError, ValueError):
            pass
        osm_speed_kph = selected_speed["speed_limit_kph"]
        if osm_speed_kph is None and hpms_speed_usable and hpms_speed_kph is not None:
            selected_speed["speed_limit_kph"] = hpms_speed_kph
            selected_speed["speed_limit_observed_source"] = "hpms_speed_limit_direction_verified"
            selected_speed["maxspeed_parse_status"] = "hpms_numeric_mph"
        parse_status = selected_speed["maxspeed_parse_status"]
        parse_counts[f"{selected_speed['speed_limit_observed_source']}:{parse_status}"] += 1
        highway = _primary_highway(attributes.get("highway"))
        hpms_mode = operating_mode_from_hpms(hpms.get("F_SYSTEM"))
        operating_mode = (
            hpms_mode if hpms_corridor_usable and hpms_mode else operating_mode_from_osm(highway)
        )
        operating_mode_source = (
            "hpms_f_system_high_confidence"
            if hpms_corridor_usable and hpms_mode
            else "osm_highway_fallback"
        )
        hpms_moves_type = moves_road_type_from_hpms(hpms.get("F_SYSTEM"))
        moves_road_type = (
            hpms_moves_type
            if hpms_corridor_usable and hpms_moves_type
            else moves_road_type_from_osm(highway)
        )
        moves_road_type_source = (
            "hpms_f_system_high_confidence"
            if hpms_corridor_usable and hpms_moves_type
            else "osm_highway_fallback"
        )
        osmid = _canonical_values(attributes.get("osmid"))
        corridor_id = (
            f"osmid:{osmid}" if osmid else f"edge:{min(str(u), str(v))}:{max(str(u), str(v))}"
        )
        physical_segment_id = (
            f"{corridor_id}|{min(str(u), str(v))}|{max(str(u), str(v))}|{length:.3f}"
        )
        records.append(
            {
                "edge_u": str(u),
                "edge_v": str(v),
                "edge_key": str(key),
                "edge_id": edge_id,
                "physical_segment_id": physical_segment_id,
                "corridor_id": corridor_id,
                "length_m": length,
                "highway": highway,
                "operating_mode": operating_mode,
                "operating_mode_source": operating_mode_source,
                "moves_road_type": moves_road_type,
                "moves_road_type_source": moves_road_type_source,
                "moves_is_ramp": highway in {"motorway_link", "trunk_link"},
                "raw_hpms_f_system": hpms.get("F_SYSTEM"),
                "raw_hpms_speed_limit_mph": hpms.get("SPEED_LIMIT"),
                "hpms_match_confidence": hpms.get("match_confidence"),
                "hpms_corridor_match_usable": hpms_corridor_usable,
                "hpms_direction_verified": _as_bool(hpms.get("direction_verified")),
                "hpms_speed_usable": hpms_speed_usable,
                "hpms_segment_id": hpms.get("hpms_segment_id"),
                "hpms_match_method": hpms.get("match_method"),
                "hpms_lateral_distance_m": hpms.get("lateral_distance_m"),
                "hpms_overlap_ratio": hpms.get("overlap_ratio"),
                "hpms_orientation_difference_deg": hpms.get("orientation_difference_deg"),
                "hpms_speed_limit_kph_evidence": hpms_speed_kph,
                "speed_conflict_flag": bool(
                    osm_speed_kph is not None
                    and hpms_speed_kph is not None
                    and not math.isclose(float(osm_speed_kph), hpms_speed_kph, abs_tol=1.0)
                ),
                "oneway": _as_bool(attributes.get("oneway", False)),
                "transit_only": _as_bool(attributes.get("transit_only", False)),
                **selected_speed,
            }
        )
    frame = pd.DataFrame(records)
    observed = frame["speed_limit_kph"].notna()
    osm_observed = frame["speed_limit_observed_source"].astype(str).str.startswith("osm_")
    hpms_observed = frame["speed_limit_observed_source"].eq("hpms_speed_limit_direction_verified")
    observed_frame = frame.loc[observed]
    if observed_frame.empty:
        raise ValueError("Operational graph has no observed legal-speed evidence from OSM or HPMS")

    class_medians: dict[str, float] = {}
    for highway, group in observed_frame.groupby("highway", observed=True):
        class_medians[str(highway)] = _weighted_median(group["speed_limit_kph"], group["length_m"])
    mode_medians: dict[str, float] = {}
    for mode, group in observed_frame.groupby("operating_mode", observed=True):
        mode_medians[str(mode)] = _weighted_median(group["speed_limit_kph"], group["length_m"])
    city_median = _weighted_median(observed_frame["speed_limit_kph"], observed_frame["length_m"])
    parent_class = {
        "motorway_link": "motorway",
        "trunk_link": "trunk",
        "primary_link": "primary",
        "secondary_link": "secondary",
        "tertiary_link": "tertiary",
        "living_street": "residential",
        "service": "residential",
        "road": "residential",
        "unclassified": "residential",
        "busway": "tertiary",
    }
    sources: list[str] = []
    confidences: list[str] = []
    imputed: list[bool] = []
    values: list[float] = []
    for row in frame.itertuples(index=False):
        if row.speed_limit_kph is not None and not pd.isna(row.speed_limit_kph):
            values.append(float(row.speed_limit_kph))
            sources.append(str(row.speed_limit_observed_source))
            confidences.append("high" if row.maxspeed_parse_status == "parsed_single" else "medium")
            imputed.append(False)
            continue
        if row.highway in class_medians:
            values.append(class_medians[row.highway])
            sources.append("within_city_class_imputation")
            confidences.append("medium")
        elif row.operating_mode in mode_medians:
            values.append(mode_medians[row.operating_mode])
            sources.append("within_city_operating_mode_imputation")
            confidences.append("low")
        elif parent_class.get(row.highway) in class_medians:
            values.append(class_medians[parent_class[row.highway]])
            sources.append("within_city_parent_class_imputation")
            confidences.append("low")
        else:
            values.append(city_median)
            sources.append("within_city_global_imputation")
            confidences.append("low")
        imputed.append(True)
    frame["speed_limit_kph"] = values
    frame["speed_limit_source"] = sources
    frame["speed_limit_confidence"] = confidences
    frame["speed_limit_is_imputed"] = imputed
    # Canonical names are used by downstream country-independent code.  The
    # speed_limit_* aliases remain for the current graph and QA utilities.
    frame["legal_speed_kph"] = frame["speed_limit_kph"]
    frame["legal_speed_source"] = frame["speed_limit_source"]
    frame["legal_speed_confidence"] = frame["speed_limit_confidence"]
    frame["legal_speed_imputed"] = frame["speed_limit_is_imputed"]
    frame["legal_travel_time_s"] = frame["length_m"] / (frame["speed_limit_kph"] / 3.6)
    frame["free_flow_speed_proxy_kph"] = frame["legal_speed_kph"]
    for day_type in ("weekday", "weekend"):
        retention_column = f"moves_speed_retention_{day_type}"
        reference_column = f"reference_speed_{day_type}_kph"
        travel_time_column = f"reference_travel_time_{day_type}_s"
        retention_by_road_type = {
            road_type: speed_retention_factor(moves_profile, road_type=road_type, day_type=day_type)
            for road_type in moves_profile["road_types"]
        }
        frame[retention_column] = frame["moves_road_type"].map(retention_by_road_type)
        reference = frame["free_flow_speed_proxy_kph"].to_numpy(dtype=float) * frame[
            retention_column
        ].to_numpy(dtype=float)
        if vehicle_speed_cap_kph is not None:
            reference = np.minimum(reference, vehicle_speed_cap_kph)
        frame[reference_column] = np.minimum(
            frame["legal_speed_kph"].to_numpy(dtype=float), reference
        )
        frame[travel_time_column] = frame["length_m"] / (frame[reference_column] / 3.6)
    frame["vehicle_speed_cap_kph"] = vehicle_speed_cap_kph
    frame["reference_speed_source"] = moves_profile["profile_id"]
    numeric = frame[
        [
            "length_m",
            "speed_limit_kph",
            "legal_travel_time_s",
            "free_flow_speed_proxy_kph",
            "reference_speed_weekday_kph",
            "reference_speed_weekend_kph",
            "reference_travel_time_weekday_s",
            "reference_travel_time_weekend_s",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Speed layer contains nonpositive or nonfinite values")
    for day_type in ("weekday", "weekend"):
        if (frame[f"reference_speed_{day_type}_kph"] > frame["speed_limit_kph"] + 1e-9).any():
            raise ValueError(f"{day_type} reference speed exceeds legal limit")

    total_length = float(frame["length_m"].sum())
    observed_length = float(frame.loc[observed, "length_m"].sum())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{city_slug}-speed-", dir=output_dir.parent) as temp:
        staged = Path(temp) / output_dir.name
        staged.mkdir()
        table_path = staged / "directed_legal_speeds.parquet"
        frame.to_parquet(table_path, index=False)
        (staged / "moves_speed_profile.json").write_bytes(moves_profile_path.read_bytes())
        manifest = {
            "schema": "evrptw_directed_speed_profiles_v6",
            "status": "cle_reference_speed_profile_complete",
            "generated_utc": datetime.now(UTC).isoformat(),
            "city_slug": city_slug,
            "graph": {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)},
            "hpms_edge_evidence": (
                {
                    "path": str(hpms_edge_evidence_path.resolve()),
                    "sha256": sha256_file(hpms_edge_evidence_path),
                }
                if hpms_edge_evidence_path is not None
                else None
            ),
            "edge_count": len(frame),
            "observed_osm_maxspeed_edge_count": int(osm_observed.sum()),
            "observed_osm_maxspeed_edge_share": float(osm_observed.mean()),
            "observed_hpms_speed_limit_edge_count": int(hpms_observed.sum()),
            "observed_hpms_speed_limit_edge_share": float(hpms_observed.mean()),
            "hpms_corridor_match_usable_edge_count": int(frame["hpms_corridor_match_usable"].sum()),
            "hpms_direction_verified_edge_count": int(frame["hpms_direction_verified"].sum()),
            "hpms_speed_usable_edge_count": int(frame["hpms_speed_usable"].sum()),
            "observed_any_legal_speed_edge_count": int(observed.sum()),
            "observed_any_legal_speed_edge_share": float(observed.mean()),
            "observed_any_legal_speed_length_share": observed_length / total_length,
            "directional_maxspeed_present_edge_count": int(
                frame["directional_maxspeed_present"].sum()
            ),
            "directional_maxspeed_applied_edge_count": int(
                frame["speed_limit_observed_source"]
                .astype(str)
                .str.startswith("osm_maxspeed_")
                .sum()
            ),
            "hgv_maxspeed_evidence_edge_count": int(
                frame["hgv_speed_limit_kph_evidence"].notna().sum()
            ),
            "imputed_edge_count": int(frame["speed_limit_is_imputed"].sum()),
            "parse_status_counts": dict(sorted(parse_counts.items())),
            "class_weighted_median_kph": dict(sorted(class_medians.items())),
            "operating_mode_weighted_median_kph": dict(sorted(mode_medians.items())),
            "city_weighted_median_kph": city_median,
            "imputation_contract": {
                "priority": [
                    "parseable direction-applicable OSM maxspeed:forward/backward",
                    "parseable generic OSM maxspeed",
                    "direction-verified high-confidence HPMS SPEED_LIMIT when OSM is missing",
                    "same-city same-highway length-weighted median",
                    "same-city H/M/U operating-mode length-weighted median",
                    "same-city parent-highway length-weighted median",
                    "same-city all-observed length-weighted median",
                ],
                "multivalue_policy": "conservative minimum; preserve raw source text",
                "directional_tag_status": "retained and selected using OSMnx reversed way direction",
                "hpms_speed_policy": (
                    "requires corridor_match_usable, direction_verified, and "
                    "hpms_speed_usable; corridor-only matches may provide F_SYSTEM only"
                ),
                "hgv_tag_policy": "retained and parsed as evidence only; not applied until vehicle/GVWR access contract is frozen",
            },
            "reference_speed_contract": {
                "name": "delivery_reference_running_speed_not_door_to_door_average",
                "profile_id": moves_profile["profile_id"],
                "formula": (
                    "reference_speed(edge,day)=legal_speed(edge)*"
                    "MOVES_effective_speed(road_type,day)/MOVES_low_flow_Q85(road_type)"
                ),
                "free_flow_edge_proxy": "direction_applicable_legal_speed_kph",
                "moves_speed_profile_file": moves_profile_path.name,
                "moves_speed_profile_sha256": sha256_file(moves_profile_path),
                "moves_state_fips": moves_profile.get("state_fips"),
                "moves_observation_vintage": moves_profile.get("observation_vintage"),
                "moves_source_inputs": moves_profile.get("source_inputs"),
                "moves_temporal_conversion": moves_profile.get("temporal_conversion"),
                "moves_source_database": moves_profile["source_database"],
                "moves_source_type_id": moves_profile["source_type_id"],
                "moves_service_window": moves_profile["service_window"],
                "moves_low_flow_benchmark": moves_profile["low_flow_benchmark"],
                "moves_road_types": moves_profile["road_types"],
                "moves_scope_warning": moves_profile["default_data_scope"],
                "mode_source_priority": [
                    "high-confidence conflated HPMS F_SYSTEM",
                    "OSM highway fallback",
                ],
                "vehicle_speed_cap_kph": vehicle_speed_cap_kph,
                "vehicle_speed_cap_status": (
                    "applied_user_supplied_versioned_value"
                    if vehicle_speed_cap_kph is not None
                    else "not_applied_no_versioned_source"
                ),
                "turn_signal_start_stop_status": "not_in_edge speed; geometry-only turn penalties are Stage 2",
            },
            "stage_2_operational_speed_contract": {
                "semantics": "select the CLE weekday or weekend reference field",
                "stored_in_cle": False,
                "day_types": ["weekday", "weekend"],
                "canonical_residual_factor": 1.0,
                "independent_per_edge_noise": False,
                "departure_time_dependent_traffic": False,
            },
            "outputs": {
                "directed_legal_speeds": "directed_legal_speeds.parquet",
                "moves_speed_profile": "moves_speed_profile.json",
            },
        }
        manifest["output_sha256"] = {
            name: sha256_file(staged / relative) for name, relative in manifest["outputs"].items()
        }
        write_json(staged / "speed_manifest.json", manifest)
        staged.replace(output_dir)
    return manifest
