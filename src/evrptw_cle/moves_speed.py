"""EPA MOVES5 speed-profile extraction and portable lookup helpers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from .runtime_paths import CONFIG_ROOT

MOVES_PROFILE_SCHEMA = "evrptw_moves5_speed_retention_profile_v1"
MOVES_STATE_REGISTRY_SCHEMA = "evrptw_moves_state_profile_registry_v1"

DEFAULT_MOVES_SPEED_PROFILE_PATH = CONFIG_ROOT / "us_nei2023_streetlight_state_profiles_v1.json"
MOVES_ROAD_TYPE_FROM_HPMS = {
    1: "urban_restricted_access",
    2: "urban_restricted_access",
    3: "urban_unrestricted_access",
    4: "urban_unrestricted_access",
    5: "urban_unrestricted_access",
    6: "urban_unrestricted_access",
    7: "urban_unrestricted_access",
}
MOVES_RESTRICTED_OSM_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
}


def moves_road_type_from_hpms(value: Any) -> str | None:
    """Map FHWA functional class to a MOVES urban access class."""

    try:
        return MOVES_ROAD_TYPE_FROM_HPMS.get(int(float(value)))
    except (TypeError, ValueError):
        return None


def moves_road_type_from_osm(highway: str) -> str:
    """Return the MOVES urban access class used when HPMS is unavailable."""

    return (
        "urban_restricted_access"
        if str(highway) in MOVES_RESTRICTED_OSM_HIGHWAYS
        else "urban_unrestricted_access"
    )


def load_moves_speed_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a compact MOVES-compatible statistical profile."""

    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MOVES_PROFILE_SCHEMA:
        raise ValueError(f"Unsupported MOVES speed profile schema: {payload.get('schema')!r}")
    if int(payload.get("source_type_id", -1)) != 32:
        raise ValueError("The U.S. reference adapter requires MOVES sourceTypeID=32")
    road_types = payload.get("road_types")
    expected = {"urban_restricted_access", "urban_unrestricted_access"}
    if not isinstance(road_types, dict) or set(road_types) != expected:
        raise ValueError(f"MOVES speed profile road types must be {sorted(expected)}")
    for name, record in road_types.items():
        free_flow = float(record["low_flow_q85_mph"])
        if not math.isfinite(free_flow) or free_flow <= 0.0:
            raise ValueError(f"Invalid low-flow Q85 for {name}")
        for day_type in ("weekday", "weekend"):
            effective = float(record["effective_speed_mph"][day_type])
            retention = float(record["speed_retention_factor"][day_type])
            if not math.isfinite(effective) or effective <= 0.0:
                raise ValueError(f"Invalid effective speed for {name}/{day_type}")
            if not 0.0 < retention <= 1.0:
                raise ValueError(f"Invalid speed-retention factor for {name}/{day_type}")
            if not math.isclose(effective / free_flow, retention, rel_tol=1e-8):
                raise ValueError(f"Inconsistent MOVES profile ratio for {name}/{day_type}")
    return payload


def resolve_moves_speed_profile(
    path: str | Path | None = None,
    *,
    city_slug: str | None = None,
    state_fips: str | None = None,
) -> Path:
    """Select a state profile, or validate an explicitly supplied legacy profile.

    A registry never falls back to the historical national default. New cities
    supply a state FIPS code; the frozen cohort can use its recorded slug map.
    """

    source = Path(path or DEFAULT_MOVES_SPEED_PROFILE_PATH).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    requested_state = str(state_fips).zfill(2) if state_fips is not None else None
    if payload.get("schema") == MOVES_STATE_REGISTRY_SCHEMA:
        mapped_state = payload.get("city_state_fips", {}).get(city_slug)
        if mapped_state is not None:
            mapped_state = str(mapped_state).zfill(2)
        if requested_state and mapped_state and requested_state != mapped_state:
            raise ValueError(f"City/state mismatch for {city_slug!r}: {requested_state}")
        requested_state = requested_state or mapped_state
        if requested_state is None:
            raise ValueError(f"State FIPS is required for an unregistered city: {city_slug!r}")
        record = payload.get("states", {}).get(requested_state)
        if record is None:
            raise ValueError(f"No frozen StreetLight speed profile for state {requested_state}")
        selected = (source.parent / record["profile"]).resolve()
        profile = load_moves_speed_profile(selected)
        if profile.get("state_fips") != requested_state:
            raise ValueError(f"State registry/profile mismatch: {selected}")
        return selected
    profile = load_moves_speed_profile(source)
    if requested_state and profile.get("state_fips") not in {None, requested_state}:
        raise ValueError(f"Selected speed profile is for a different state: {source}")
    return source


def speed_retention_factor(profile: dict[str, Any], *, road_type: str, day_type: str) -> float:
    """Return the frozen MOVES class/day speed-retention factor."""

    try:
        return float(profile["road_types"][road_type]["speed_retention_factor"][day_type])
    except KeyError as exc:
        raise ValueError(f"MOVES profile does not contain {road_type!r}/{day_type!r}") from exc


def _sql_tuple(line: str) -> list[str] | None:
    text = line.strip().rstrip(",;")
    if not text.startswith("(") or not text.endswith(")"):
        return None
    return [value.strip().strip("'") for value in text[1:-1].split(",")]


def derive_moves_speed_profile_from_sql(
    sql_path: str | Path,
    *,
    source_type_id: int = 32,
    service_hour_ids: tuple[int, ...] = tuple(range(9, 25)),
    low_flow_day_id: int = 2,
    low_flow_hour_ids: tuple[int, ...] = (7, 8, 9, 10),
    low_flow_quantile: float = 0.85,
) -> dict[str, Any]:
    """Derive the compact U.S. reference profile from an official MOVES SQL dump.

    ``AvgSpeedDistribution`` is a within-hour VHT distribution. Hour-level mean
    speed is its speed-bin weighted mean. Service-window effective speed is total
    VMT divided by total VHT, using ``HourVMTFraction`` as the VMT allocation.
    The low-flow quantile pools weekend 06:00--10:00 bins using VHT weights.
    """

    if not 0.0 < low_flow_quantile < 1.0:
        raise ValueError("low_flow_quantile must be in (0, 1)")
    speed_bins: dict[int, float] = {}
    hour_day: dict[int, tuple[int, int]] = {}
    speed_fractions: dict[tuple[int, int, int], float] = {}
    hour_vmt: dict[tuple[int, int, int], float] = {}
    mode: str | None = None
    target_road_type_ids = {4, 5}

    with Path(sql_path).open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("INSERT INTO `avgspeedbin` VALUES"):
                mode = "speed_bins"
                continue
            if line.startswith("INSERT INTO `avgspeeddistribution` VALUES"):
                mode = "speed_distribution"
                continue
            if line.startswith("INSERT INTO `hourday` VALUES"):
                mode = "hour_day"
                continue
            if line.startswith("INSERT INTO `hourvmtfraction` VALUES"):
                mode = "hour_vmt"
                continue
            if line.startswith("UNLOCK TABLES"):
                mode = None
                continue
            if mode is None:
                continue
            values = _sql_tuple(line)
            if values is None:
                continue
            if mode == "speed_bins" and len(values) >= 2:
                speed_bins[int(values[0])] = float(values[1])
            elif mode == "hour_day" and len(values) >= 3:
                hour_day[int(values[0])] = (int(values[1]), int(values[2]))
            elif mode == "speed_distribution" and len(values) >= 5:
                source, road, hour_day_id, speed_bin = map(int, values[:4])
                if source == source_type_id and road in target_road_type_ids:
                    speed_fractions[(road, hour_day_id, speed_bin)] = float(values[4])
            elif mode == "hour_vmt" and len(values) >= 5:
                source, road, day, hour = map(int, values[:4])
                if source == source_type_id and road in target_road_type_ids:
                    hour_vmt[(road, day, hour)] = float(values[4])

    if len(speed_bins) != 16:
        raise ValueError(f"Expected 16 MOVES average-speed bins, found {len(speed_bins)}")
    if not hour_day:
        raise ValueError("MOVES SQL dump did not yield HourDay records")

    day_hour_to_id = {value: key for key, value in hour_day.items()}
    road_names = {
        4: "urban_restricted_access",
        5: "urban_unrestricted_access",
    }
    results: dict[str, Any] = {}
    for road_type_id, road_name in road_names.items():
        hourly_speeds: dict[tuple[int, int], float] = {}
        for day_id in (2, 5):
            for hour_id in range(1, 25):
                hour_day_id = day_hour_to_id[(day_id, hour_id)]
                fractions = {
                    speed_bin: speed_fractions[(road_type_id, hour_day_id, speed_bin)]
                    for speed_bin in speed_bins
                }
                if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-5):
                    raise ValueError(
                        f"MOVES speed fractions do not sum to one for road={road_type_id}, "
                        f"day={day_id}, hour={hour_id}"
                    )
                hourly_speeds[(day_id, hour_id)] = sum(
                    fractions[speed_bin] * speed_bins[speed_bin] for speed_bin in speed_bins
                )

        effective: dict[str, float] = {}
        for day_name, day_id in (("weekday", 5), ("weekend", 2)):
            window_vmt = sum(
                hour_vmt[(road_type_id, day_id, hour_id)] for hour_id in service_hour_ids
            )
            window_vht = sum(
                hour_vmt[(road_type_id, day_id, hour_id)] / hourly_speeds[(day_id, hour_id)]
                for hour_id in service_hour_ids
            )
            effective[day_name] = window_vmt / window_vht

        pooled_vht_by_bin: defaultdict[int, float] = defaultdict(float)
        total_vht = 0.0
        for hour_id in low_flow_hour_ids:
            hour_day_id = day_hour_to_id[(low_flow_day_id, hour_id)]
            hour_speed = hourly_speeds[(low_flow_day_id, hour_id)]
            hour_total_vht = hour_vmt[(road_type_id, low_flow_day_id, hour_id)] / hour_speed
            for speed_bin in speed_bins:
                weight = hour_total_vht * speed_fractions[(road_type_id, hour_day_id, speed_bin)]
                pooled_vht_by_bin[speed_bin] += weight
                total_vht += weight
        cumulative = 0.0
        q85_mph: float | None = None
        for speed_bin in sorted(speed_bins):
            cumulative += pooled_vht_by_bin[speed_bin] / total_vht
            if cumulative >= low_flow_quantile:
                q85_mph = speed_bins[speed_bin]
                break
        if q85_mph is None:
            raise ValueError(f"Unable to derive low-flow Q85 for MOVES road type {road_type_id}")
        results[road_name] = {
            "road_type_id": road_type_id,
            "low_flow_q85_mph": q85_mph,
            "effective_speed_mph": {key: round(value, 9) for key, value in effective.items()},
            "speed_retention_factor": {
                key: round(value / q85_mph, 9) for key, value in effective.items()
            },
        }

    return {
        "schema": MOVES_PROFILE_SCHEMA,
        "profile_id": "us_epa_moves5_20241112_source32_static_08_24_v1",
        "source_model": "U.S. EPA Motor Vehicle Emission Simulator (MOVES5)",
        "source_database": "movesdb20241112",
        "source_database_url": (
            "https://github.com/USEPA/EPA_MOVES_Model/blob/master/"
            "database/Setup/movesdb20241112.zip"
        ),
        "source_type_id": source_type_id,
        "source_type_label": "Light Commercial Truck",
        "service_window": {
            "local_time": "08:00-24:00",
            "hour_ids": list(service_hour_ids),
            "aggregation": "VMT_over_VHT_harmonic_effective_speed",
        },
        "low_flow_benchmark": {
            "day_id": low_flow_day_id,
            "day_type": "weekend",
            "local_time": "06:00-10:00",
            "hour_ids": list(low_flow_hour_ids),
            "quantile": low_flow_quantile,
            "aggregation": "VHT_weighted_speed_bin_quantile",
            "fhwa_basis": ("85th-percentile off-peak speed; weekend 06:00-10:00 low-flow window"),
        },
        "default_data_scope": {
            "avg_speed_distribution": "single national default by source/road/day/hour",
            "hour_vmt_fraction": (
                "national temporal allocation; not edge traffic volume and not EDV-specific"
            ),
        },
        "road_types": results,
    }
