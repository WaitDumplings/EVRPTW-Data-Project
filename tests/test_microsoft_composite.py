"""Edge cases that could otherwise add two stops for one physical roof."""

import numpy as np
from shapely.geometry import box

from evrptw_cle.microsoft_composite import legacy_admission


def test_portable_composition_retains_lineage_and_never_claims_current_existence():
    import geopandas as gpd
    from evrptw_cle.microsoft_composite import compose_footprints

    new = gpd.GeoDataFrame(
        {"building_id": ["n2", "n1"]}, geometry=[box(0, 0, 10, 10)] * 2, crs=3857
    )
    old = gpd.GeoDataFrame(
        {"building_id": ["shadowed", "gap"]},
        geometry=[box(0, 0, 10, 10), box(50, 0, 55, 5)],
        crs=3857,
    )
    result = compose_footprints(new, old, area_crs=3857)
    frame = result["composite_footprints"]
    assert len(frame) == 2 and set(frame.source_kind) == {"global", "legacy"}
    assert frame.building_id.is_unique and not frame.current_existence_verified.any()
    assert frame.requires_nsi_residential_evidence.all()
    assert len(result["new_geometry_quarantine"]) == 1


def test_new_roof_changes_and_split_merge_are_never_backfilled():
    new = np.asarray([box(0, 0, 8, 8), box(10, 0, 18, 8)], dtype=object)
    old = np.asarray([box(0, 0, 8, 8), box(-1, -1, 9, 9), box(-1, -1, 19, 9)], dtype=object)
    result = legacy_admission(new, old, [2.0, 5.0, 10.0, 20.0], 0.01)
    assert not result["masks"]["2.0"].any()


def test_strict_clearance_boundary_and_separate_gap():
    new = np.asarray([box(0, 0, 10, 10)], dtype=object)
    old = np.asarray([box(20, 0, 25, 5), box(20.1, 15, 25.1, 20), box(50, 0, 55, 5)], dtype=object)
    result = legacy_admission(new, old, [2.0, 5.0, 10.0, 20.0], 0.01)
    assert result["masks"]["10.0"].tolist() == [False, True, True]


def test_duplicate_and_overlapping_legacy_do_not_create_extra_stops():
    new = np.asarray([box(0, 0, 1, 1)], dtype=object)
    old = np.asarray(
        [box(40, 0, 50, 10), box(40, 0, 50, 10), box(45, 0, 55, 10), box(80, 0, 90, 10)],
        dtype=object,
    )
    result = legacy_admission(new, old, [2.0, 5.0, 10.0, 20.0], 0.01)
    assert result["duplicate"].tolist() == [False, True, False, False]
    assert result["conflict"].tolist() == [True, False, True, False]
    assert result["masks"]["10.0"].tolist() == [False, False, False, True]
