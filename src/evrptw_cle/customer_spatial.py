"""Current geometric containment matching and auxiliary unit bands.

The superseded unit-count-to-housing-type classifier is not included.
"""

from __future__ import annotations

import json


import geopandas as gpd

import numpy as np

import pandas as pd


def residential_unit_band(units: int) -> str:
    if units == 1:
        return "unit_1"
    if units <= 4:
        return "units_2_4"
    if units <= 19:
        return "units_5_19"
    if units <= 49:
        return "units_20_49"
    return "units_50_plus"


def resolve_containment_matches(
    nsi_points: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Return one audit row per NSI point; ambiguous containment is never auto-resolved."""

    points = nsi_points.reset_index(drop=True)
    polygons = footprints[["building_id", "geometry"]].reset_index(drop=True)
    if points.crs != polygons.crs:
        polygons = polygons.to_crs(points.crs)
    joined = gpd.sjoin(
        points[["geometry"]],
        polygons,
        how="left",
        predicate="covered_by",
    )
    candidates = (
        joined.dropna(subset=["building_id"]).groupby(level=0, sort=False)["building_id"].agg(list)
    )
    result = pd.DataFrame(index=points.index)
    result["footprint_match_count"] = candidates.map(len).reindex(points.index, fill_value=0)
    result["candidate_microsoft_building_ids"] = candidates.map(
        lambda values: json.dumps(sorted(str(value) for value in values))
    ).reindex(points.index, fill_value="[]")
    result["microsoft_building_id"] = candidates.map(
        lambda values: str(values[0]) if len(values) == 1 else None
    ).reindex(points.index)
    result["geometry_match_status"] = np.select(
        [result["footprint_match_count"] == 1, result["footprint_match_count"] > 1],
        ["matched_unique_containment", "quarantine_multiple_containment"],
        default="quarantine_no_containment",
    )
    result["geometry_match_method"] = np.where(
        result["footprint_match_count"] == 1,
        "nsi_point_within_microsoft_footprint",
        None,
    )
    result["geometry_match_distance_m"] = np.where(
        result["footprint_match_count"] == 1, 0.0, np.nan
    )
    return result.reset_index(drop=True)
