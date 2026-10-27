"""Frozen national new-primary/legacy-gap geometry rules.

Footprints are geometric evidence, not final customer identities.
"""

from __future__ import annotations
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree
from .runtime_paths import CONFIG_ROOT


def exact_duplicates(geometries):
    return pd.Series(shapely.to_wkb(shapely.normalize(geometries))).duplicated().to_numpy()


def legacy_admission(new_guards, old_geometries, clearances, overlap_tolerance):
    """Return distances and one fixed legacy-conflict mask for all clearances."""
    limit = max(clearances)
    indices, distances = STRtree(new_guards).query_nearest(
        old_geometries, max_distance=limit, return_distance=True, all_matches=False
    )
    d = np.full(len(old_geometries), np.inf)
    nearest = np.full(len(old_geometries), -1, dtype=np.int64)
    d[indices[0]] = distances
    nearest[indices[0]] = indices[1]
    eligible_min = d > min(clearances)
    candidates = np.flatnonzero(eligible_min)
    duplicate = np.zeros(len(old_geometries), dtype=bool)
    duplicate[candidates] = exact_duplicates(old_geometries[candidates])
    candidates = candidates[~duplicate[candidates]]
    conflict = np.zeros(len(old_geometries), dtype=bool)
    pairs = STRtree(old_geometries[candidates]).query(
        old_geometries[candidates], predicate="intersects"
    )
    pairs = pairs[:, pairs[0] < pairs[1]]
    if pairs.shape[1]:
        area = shapely.area(
            shapely.intersection(
                old_geometries[candidates[pairs[0]]], old_geometries[candidates[pairs[1]]]
            )
        )
        pairs = pairs[:, area > overlap_tolerance]
        conflict[candidates[np.unique(pairs)]] = True
    masks = {str(clearance): (d > clearance) & ~duplicate & ~conflict for clearance in clearances}
    return {
        "distance": d,
        "nearest_guard_index": nearest,
        "duplicate": duplicate,
        "conflict": conflict,
        "masks": masks,
        "conflict_pair_count": int(pairs.shape[1]),
    }


def _source(frame, kind):
    frame = frame.to_crs(4326).copy()
    frame = frame.sort_values("building_id", kind="mergesort").reset_index(drop=True)
    prefix = "global20260813:" if kind == "global" else "legacyv2:"
    if not frame.building_id.astype(str).str.startswith(prefix).all():
        frame["original_building_id"] = frame.building_id
        frame["building_id"] = prefix + frame.building_id.astype(str)
    frame["source_kind"] = kind
    frame["source_snapshot"] = "2026-08-13" if kind == "global" else "legacy-v2-frozen-registry"
    return frame


def compose_footprints(new, legacy, *, area_crs, rules_path=None):
    """Apply the reviewed geometry kernel to normalized source GeoDataFrames.

    Return all rejected originals and legacy admission flags for audit. No
    footprint clipping/union, residential inference or customer activation occurs.
    """
    path = Path(rules_path) if rules_path else CONFIG_ROOT / "microsoft_geometry_rules_v1.json"
    rules = json.loads(path.read_text())
    new, old = _source(new, "global"), _source(legacy, "legacy")
    new_local = new.to_crs(area_crs).geometry.to_numpy()
    old_local = old.to_crs(area_crs).geometry.to_numpy()
    valid_new = (
        shapely.is_valid(new.geometry.to_numpy())
        & shapely.is_valid(new_local)
        & (shapely.area(new_local) > 0)
    )
    valid_old = (
        shapely.is_valid(old.geometry.to_numpy())
        & shapely.is_valid(old_local)
        & (shapely.area(old_local) > 0)
    )
    duplicate_new = np.zeros(len(new), dtype=bool)
    indices = np.flatnonzero(valid_new)
    duplicate_new[indices] = exact_duplicates(new.geometry.to_numpy()[indices])
    base_mask = valid_new & ~duplicate_new
    base = new.loc[base_mask].copy().reset_index(drop=True)
    base["admission_rule"] = "valid_new_primary"
    guards = np.concatenate([new_local[base_mask], shapely.envelope(new_local[~valid_new])])
    result = legacy_admission(
        guards,
        old_local[valid_old],
        rules["clearance_sensitivity_m"],
        rules["overlap_area_tolerance_m2"],
    )
    audit = old.copy()
    audit["valid_geometry"] = valid_old
    audit["nearest_new_or_guard_distance_m"] = np.nan
    audit["legacy_exact_duplicate"] = False
    audit["legacy_overlap_conflict"] = False
    audit.loc[valid_old, "nearest_new_or_guard_distance_m"] = np.where(
        np.isfinite(result["distance"]), result["distance"], np.nan
    )
    audit.loc[valid_old, "legacy_exact_duplicate"] = result["duplicate"]
    audit.loc[valid_old, "legacy_overlap_conflict"] = result["conflict"]
    for clearance in rules["clearance_sensitivity_m"]:
        name = f"admit_clearance_{clearance:g}m"
        audit[name] = False
        audit.loc[valid_old, name] = result["masks"][str(clearance)]
    added = audit.loc[audit[f"admit_clearance_{rules['primary_clearance_m']:g}m"]].copy()
    added["admission_rule"] = "legacy_gap_clearance_gt_10m_no_legacy_overlap"
    for frame in [base, added]:
        frame["current_existence_verified"] = False
        frame["requires_nsi_residential_evidence"] = True
    composite = gpd.GeoDataFrame(
        pd.concat([base, added.reindex(columns=base.columns)], ignore_index=True),
        geometry="geometry",
        crs=4326,
    )
    if not composite.building_id.is_unique:
        raise ValueError("Composite source identities must be unique")
    return dict(
        new_primary=base,
        legacy_admission_audit=audit,
        composite_footprints=composite,
        new_geometry_quarantine=new.loc[~base_mask].copy(),
        summary=dict(
            new_primary=len(base),
            legacy_additions=len(added),
            composite=len(composite),
            source_geometry_only=True,
            primary_clearance_m=rules["primary_clearance_m"],
            rule_file=str(path.name),
        ),
    )
