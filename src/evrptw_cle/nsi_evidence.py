"""Native NSI evidence; modeled bounds never classify housing or replace site units."""

from __future__ import annotations
from typing import Any
import gzip
import json

import geopandas as gpd
import pandas as pd

from .customer_spatial import resolve_containment_matches

NSI_OCCUPANCY_INTERVALS: dict[str, tuple[int, int | None]] = {
    "RES3A": (2, 2),
    "RES3B": (3, 4),
    "RES3C": (5, 9),
    "RES3D": (10, 19),
    "RES3E": (20, 49),
    "RES3F": (50, None),
}


def nsi_unit_interval(occtype: str, resunits: Any) -> tuple[int, int, int | None, str]:
    """Return modeled point/lower/upper units without hiding NSI zero-value fallback."""

    try:
        units = int(resunits)
    except (TypeError, ValueError):
        units = 0
    if units > 0:
        return units, units, units, "nsi_resunits_positive"
    code = str(occtype or "").upper()
    if code.startswith("RES1") or code == "RES2":
        return 1, 1, 1, "nsi_occtype_lower_bound"
    for prefix, (lower, upper) in NSI_OCCUPANCY_INTERVALS.items():
        if code.startswith(prefix):
            return lower, lower, upper, "nsi_occtype_lower_bound"
    raise ValueError(f"No ordinary-residential unit rule for NSI occupancy {occtype!r}")


def iter_geojsonseq(path):
    """Read archived RS-delimited gzip responses without making a live request."""
    buffer = b""
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            parts = (buffer + chunk).split(b"\x1e")
            buffer = parts.pop()
            for part in parts:
                if part.strip():
                    yield json.loads(part)
        if buffer.strip():
            yield json.loads(buffer)


def residential_points(features, *, boundary):
    """Retain native ordinary-residential points, identities and unit evidence.

    Modeled occupancy bounds are provenance only. The current site builder
    estimates nullable operational units separately from raw resunits, geometry
    ownership and supported component scope.
    """
    fields = [
        "fd_id",
        "bid",
        "occtype",
        "st_damcat",
        "resunits",
        "ftprntid",
        "ftprntsrc",
        "ftprntsqft",
        "usastrucid",
        "cbfips",
    ]
    rows, coordinates, seen = [], [], set()
    duplicates = total = 0
    for feature in features:
        total += 1
        props, geom = feature.get("properties") or {}, feature.get("geometry") or {}
        if not {"resunits", "ftprntsqft", "usastrucid"}.issubset(props):
            raise ValueError("Archived NSI response lacks the reviewed public schema")
        xy = geom.get("coordinates") or []
        if geom.get("type") != "Point" or len(xy) < 2:
            raise ValueError("NSI coordinate evidence must be a native Point")
        identity = (
            ("fd", str(props["fd_id"]))
            if props.get("fd_id") is not None
            else ("fallback", props.get("bid"), float(xy[0]), float(xy[1]), props.get("occtype"))
        )
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        code = str(props.get("occtype") or "UNKNOWN").upper()
        if not code.startswith(("RES1", "RES2", "RES3")):
            continue
        row = {k: props.get(k) for k in fields}
        row["occtype"] = code
        try:
            units, lower, upper, evidence = nsi_unit_interval(code, props.get("resunits"))
        except ValueError:
            units = lower = upper = None
            evidence = "unknown_occupancy_unit_rule"
        row.update(modeled_units=units, unit_lower=lower, unit_upper=upper, unit_evidence=evidence)
        rows.append(row)
        coordinates.append((float(xy[0]), float(xy[1])))
    points = gpd.GeoDataFrame(
        pd.DataFrame(
            rows, columns=fields + ["modeled_units", "unit_lower", "unit_upper", "unit_evidence"]
        ),
        geometry=gpd.points_from_xy([p[0] for p in coordinates], [p[1] for p in coordinates]),
        crs=4326,
    )
    inside = points.geometry.covered_by(boundary)
    return points.loc[inside].reset_index(drop=True), dict(
        raw_features=total,
        duplicate_records_removed=duplicates,
        ordinary_residential_points=int(inside.sum()),
        outside_service_boundary=int((~inside).sum()),
        scope="native NSI evidence, not final customer identities, household truth or daily demand",
    )


def match_crosswalk(points, composite):
    """Unique containment only; existing new-primary unique matches must survive."""
    matched = resolve_containment_matches(points, composite)
    primary = composite.loc[composite.source_kind.eq("global")]
    base = resolve_containment_matches(points, primary)
    unique = base.geometry_match_status.eq("matched_unique_containment")
    preserved = matched.geometry_match_status.eq(
        "matched_unique_containment"
    ) & matched.microsoft_building_id.eq(base.microsoft_building_id)
    if not preserved.loc[unique].all():
        raise ValueError("Legacy fill changed or made ambiguous a unique new-primary match")
    crosswalk = points.copy()
    crosswalk["composite_status"] = matched.geometry_match_status.to_numpy()
    crosswalk["composite_building_id"] = matched.microsoft_building_id.to_numpy()
    crosswalk["composite_source_kind"] = crosswalk.composite_building_id.map(
        composite.set_index("building_id").source_kind
    )
    return crosswalk
