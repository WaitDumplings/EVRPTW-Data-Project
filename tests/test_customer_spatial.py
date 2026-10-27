from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from evrptw_cle.nsi_evidence import nsi_unit_interval
from evrptw_cle.residential_types import classify_nsi_occtypes
from evrptw_cle.customer_spatial import (
    residential_unit_band,
    resolve_containment_matches,
)


def test_nsi_unit_interval_preserves_positive_and_fallback_evidence() -> None:
    assert nsi_unit_interval("RES3F", 200) == (200, 200, 200, "nsi_resunits_positive")
    assert nsi_unit_interval("RES3B", 0) == (3, 3, 4, "nsi_occtype_lower_bound")
    assert nsi_unit_interval("RES3F", 0) == (50, 50, None, "nsi_occtype_lower_bound")
    assert nsi_unit_interval("RES2", 0) == (1, 1, 1, "nsi_occtype_lower_bound")


def test_units_keep_detailed_band_separate_from_routing_type() -> None:
    assert residential_unit_band(20) == "units_20_49"
    assert residential_unit_band(50) == "units_50_plus"
    assert classify_nsi_occtypes(["RES3E"]).service_location_type == "large_apt"
    assert classify_nsi_occtypes(["RES2"]).service_location_type == "manufactured_home"


def test_containment_match_quarantines_ambiguous_and_unmatched_points() -> None:
    footprints = gpd.GeoDataFrame(
        {"building_id": ["a", "b"]},
        geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3)],
        crs="EPSG:4326",
    )
    points = gpd.GeoDataFrame(
        geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(4, 4)], crs="EPSG:4326"
    )
    result = resolve_containment_matches(points, footprints)
    assert list(result["geometry_match_status"]) == [
        "matched_unique_containment",
        "quarantine_multiple_containment",
        "quarantine_no_containment",
    ]
    assert result.loc[0, "microsoft_building_id"] == "a"
    assert result.loc[1, "microsoft_building_id"] is None
