"""Exclude identifiable security-sensitive or motor-prohibited location hosts.

Civil airports and seaports are allowed. Private commercial facilities are not
automatically prohibited. Excluded records never enter exported candidate rows.
Spatial masks are build inputs, not public location inventories.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

PUBLIC_LOCATION_POLICY = {
    "version": "public_location_exclusions_v1",
    "excluded_categories": [
        "military",
        "intelligence",
        "law_enforcement",
        "correctional",
        "nuclear",
        "motor_access_prohibited",
    ],
    "civil_airports_and_ports_allowed": True,
    "private_commercial_sites_allowed": True,
    "sensitive_point_buffer_m": 50.0,
    "spatial_match": "source geometry intersects excluded facility area or buffered sensitive POI",
    "export_removed_locations": False,
}
PUBLIC_MASK_OSMIUM_FILTERS = [
    "nwr/military",
    "nwr/landuse=military",
    "nwr/building=military",
    "nwr/amenity=police",
    "nwr/amenity=prison",
    "nwr/amenity=courthouse",
    "nwr/government",
    "nwr/office=government",
    "nwr/office=security",
    "nwr/plant:source=nuclear",
    "nwr/generator:source=nuclear",
    "nwr/nuclear",
    "nwr/motor_vehicle=no",
    "nwr/motorcar=no",
    "nwr/vehicle=no",
    "nwr/access=no",
    "nwr/name",
    "nwr/operator",
]
_TEXT_PATTERNS = [
    (
        "military",
        re.compile(
            r"\b(?:department of defense|defen[cs]e logistics agency|military|naval|(?:u\.?s\.?|united states)\s*(?:army|navy)|air force|marine corps|national guard|coast guard|space force|armed forces|dod)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "intelligence",
        re.compile(
            r"\b(?:central intelligence agency|national security agency|national geospatial.intelligence|defen[cs]e intelligence|intelligence agency|cia|nsa|fbi|federal bureau of investigation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "law_enforcement",
        re.compile(
            r"\b(?:police|sheriff|law enforcement|secret service|u\.?s\.? marshals|drug enforcement administration|homeland security|border patrol|customs and border protection|courthouse)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "correctional",
        re.compile(
            r"\b(?:prison|penitentiary|correctional|corrections|detention cent(?:er|re)|jail)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "nuclear",
        re.compile(
            r"\b(?:nuclear (?:power|reactor|fuel|weapon|facility|plant)|atomic (?:energy|weapon))\b",
            re.IGNORECASE,
        ),
    ),
]
_TEXT_FIELDS = (
    "name",
    "official_name",
    "operator",
    "owner",
    "brand",
    "company",
    "facility_name",
    "Station Name",
    "Owner Type Code",
    "Federal Agency Name",
    "Federal Agency Code",
    "Facility Type",
)
_TAG_FIELDS = (
    "landuse",
    "building",
    "military",
    "government",
    "office",
    "amenity",
    "nuclear",
    "plant:source",
    "generator:source",
    "motorcar",
    "motor_vehicle",
    "vehicle",
    "access",
    "access:delivery",
    "motor_vehicle:conditional",
)


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def location_tags(row: Any) -> dict[str, str]:
    values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    result = {}
    encoded = values.get("osm_tags_json")
    if not isinstance(encoded, (str, dict)):
        encoded = values.get("tags")
    if isinstance(encoded, dict):
        result.update(encoded)
    if isinstance(encoded, str):
        try:
            result.update(json.loads(encoded))
        except (ValueError, TypeError):
            pass
    for key in _TEXT_FIELDS + _TAG_FIELDS:
        value = _text(values.get(key))
        if value:
            result[key] = value
    return result


def excluded_location_reason(row: Any) -> str:
    tags = location_tags(row)

    def get(key):
        return _text(tags.get(key)).lower()

    agency = get("Federal Agency Code").upper()
    if (
        agency in {"DEF", "DAF", "DA", "DON", "USMC", "DOD"}
        or get("Facility Type").upper() == "MIL_BASE"
    ):
        return "military"
    if agency in {"DHS", "DOJ", "CBP", "CSOSA"}:
        return "law_enforcement"
    military = get("military")
    if military and military not in {"no", "false", "0", "historic", "abandoned", "disused"}:
        return "military"
    if get("landuse") == "military" or get("building") == "military":
        return "military"
    if get("government") in {"military", "defence", "defense"}:
        return "military"
    if get("government") in {"intelligence", "intelligence_agency"}:
        return "intelligence"
    if get("amenity") in {"police", "courthouse"} or get("government") in {
        "police",
        "law_enforcement",
    }:
        return "law_enforcement"
    if get("amenity") == "prison" or get("government") in {"prison", "correctional"}:
        return "correctional"
    if (
        get("plant:source") == "nuclear"
        or get("generator:source") == "nuclear"
        or get("nuclear") not in {"", "no", "false", "0"}
    ):
        return "nuclear"
    haystack = " ".join(_text(tags.get(key)) for key in _TEXT_FIELDS)
    for reason, pattern in _TEXT_PATTERNS:
        if pattern.search(haystack):
            return reason
    # Specific vehicle/delivery permission overrides a generic access tag.
    effective = next(
        (
            get(key)
            for key in ("motorcar", "motor_vehicle", "vehicle", "access:delivery", "access")
            if get(key)
        ),
        "",
    )
    if effective in {"no", "prohibited", "none"}:
        return "motor_access_prohibited"
    return ""


def prepare_exclusion_masks(raw: gpd.GeoDataFrame, area_crs: str) -> gpd.GeoDataFrame:
    """Return geometry/category only; strip names, IDs and facility details."""
    if raw.empty:
        return gpd.GeoDataFrame({"exclusion_category": []}, geometry=[], crs="EPSG:4326")
    categories = (
        raw["exclusion_category"].fillna("").astype(str)
        if "exclusion_category" in raw
        else raw.apply(excluded_location_reason, axis=1)
    )
    selected = raw.loc[categories.ne("")].copy()
    selected["exclusion_category"] = categories.loc[selected.index]
    selected = selected.loc[selected.geometry.notna() & ~selected.geometry.is_empty]
    metric = selected.to_crs(area_crs)
    pointlike = metric.geometry.geom_type.isin(["Point", "MultiPoint"])
    metric.loc[pointlike, "geometry"] = metric.loc[pointlike].geometry.buffer(
        PUBLIC_LOCATION_POLICY["sensitive_point_buffer_m"]
    )
    # Lines are not area masks (a prohibited road does not prohibit neighbours).
    metric = metric.loc[metric.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return metric[["exclusion_category", "geometry"]].to_crs(4326).reset_index(drop=True)


def filter_public_locations(
    frame: gpd.GeoDataFrame,
    masks: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    if frame.empty:
        result = frame.copy()
        result["public_location_eligible"] = pd.Series(dtype=bool)
        return result, {
            "policy": PUBLIC_LOCATION_POLICY,
            "input_count": 0,
            "removed_count": 0,
            "retained_count": 0,
            "removed_reason_counts": {},
        }
    reasons = frame.apply(excluded_location_reason, axis=1).to_numpy(dtype=object)
    if masks is not None and not masks.empty:
        aligned = masks.to_crs(frame.crs).reset_index(drop=True)
        valid = np.flatnonzero(reasons == "")
        tree = STRtree(aligned.geometry.to_numpy())
        pairs = tree.query(frame.iloc[valid].geometry.to_numpy(), predicate="intersects")
        for left, right in zip(pairs[0], pairs[1]):
            index = valid[int(left)]
            if not reasons[index]:
                reasons[index] = "within_" + str(aligned.iloc[int(right)].exclusion_category)
    keep = reasons == ""
    result = frame.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
    result["public_location_eligible"] = True
    return result, {
        "policy": PUBLIC_LOCATION_POLICY,
        "spatial_mask_applied": masks is not None,
        "input_count": len(frame),
        "removed_count": int((~keep).sum()),
        "retained_count": len(result),
        "removed_reason_counts": dict(sorted(Counter(str(x) for x in reasons if x).items())),
    }


def load_exclusion_masks(path: Path | None) -> gpd.GeoDataFrame | None:
    return gpd.read_parquet(path) if path is not None else None


def extract_exclusion_masks(
    *,
    osmium: str,
    pbf_path: Path,
    boundary_path: Path,
    area_crs: str,
) -> gpd.GeoDataFrame:
    """Extract contextual areas from the same PBF; never emit a facility list."""
    with tempfile.TemporaryDirectory(prefix="public-location-mask-") as temp:
        root = Path(temp)
        boundary = gpd.read_file(boundary_path).to_crs(area_crs)
        expanded = gpd.GeoDataFrame(geometry=boundary.geometry.buffer(100.0), crs=area_crs).to_crs(
            4326
        )
        expanded.to_file(root / "boundary.geojson", driver="GeoJSON")
        commands = [
            [
                osmium,
                "extract",
                "--polygon",
                str(root / "boundary.geojson"),
                "--strategy",
                "complete_ways",
                "--output",
                str(root / "city.pbf"),
                str(pbf_path),
            ],
            [
                osmium,
                "tags-filter",
                "--output",
                str(root / "selected.pbf"),
                str(root / "city.pbf"),
                *PUBLIC_MASK_OSMIUM_FILTERS,
            ],
            [
                osmium,
                "export",
                "--output-format",
                "geojson",
                "--output",
                str(root / "mask.geojson"),
                str(root / "selected.pbf"),
            ],
        ]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    "Public-location exclusion extraction failed: " + result.stderr[-500:]
                )
        raw = gpd.read_file(root / "mask.geojson").set_crs(4326, allow_override=True)
        return prepare_exclusion_masks(raw, area_crs)
