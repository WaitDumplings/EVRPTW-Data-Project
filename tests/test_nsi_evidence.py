import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.nsi_evidence import residential_points, match_crosswalk
from evrptw_cle.residential_sites import infer_residential_sites


def test_native_point_matching_does_not_turn_unit_bounds_into_operational_households():
    feature = {
        "properties": dict(
            fd_id=1,
            bid=1,
            occtype="RES3B",
            resunits=0,
            ftprntsqft=100,
            usastrucid=0,
            ftprntid=1,
            ftprntsrc="test",
            cbfips="00100",
        ),
        "geometry": {"type": "Point", "coordinates": [2, 2]},
    }
    institutional = {
        "properties": dict(feature["properties"], fd_id=2, occtype="RES6"),
        "geometry": feature["geometry"],
    }
    points, audit = residential_points([feature, feature, institutional], boundary=box(0, 0, 5, 5))
    assert len(points) == 1 and audit["duplicate_records_removed"] == 1
    assert points.iloc[0].modeled_units == 3 and points.iloc[0].resunits == 0
    fp = gpd.GeoDataFrame(
        {"building_id": ["global20260813:roof"], "source_kind": ["global"]},
        geometry=[box(1, 1, 3, 3)],
        crs=4326,
    )
    crosswalk = match_crosswalk(points, fp)
    buildings = gpd.GeoDataFrame({"osm_id": [], "usable": []}, geometry=[], crs=4326)
    complexes = gpd.GeoDataFrame(
        {"osm_id": [], "site_evidence": [], "member_osm_ids_json": []}, geometry=[], crs=4326
    )
    sites, members, _ = infer_residential_sites(
        crosswalk, fp, buildings, complexes, city_slug="test", boundary=box(0, 0, 5, 5)
    )
    assert sites.iloc[0].service_location_type == "small_apt"
    assert pd.isna(sites.iloc[0].residential_units)
    assert sites.iloc[0].nsi_modeled_unit_source_sum == 3
    assert not members.operational_units_counted.any()


def test_legacy_fill_cannot_break_a_unique_new_primary_match():
    points = gpd.GeoDataFrame({"fd_id": [1]}, geometry=[Point(1, 1)], crs=4326)
    bad = gpd.GeoDataFrame(
        {"building_id": ["new", "overlap"], "source_kind": ["global", "legacy"]},
        geometry=[box(0, 0, 2, 2), box(0, 0, 2, 2)],
        crs=4326,
    )
    with pytest.raises(ValueError, match="changed or made ambiguous"):
        match_crosswalk(points, bad)


def test_empty_frozen_query_reports_zero_without_invented_records():
    points, audit = residential_points([], boundary=box(0, 0, 1, 1))
    assert points.empty and audit["ordinary_residential_points"] == 0
