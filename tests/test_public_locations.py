import json

import geopandas as gpd

import pandas as pd

import pytest

from shapely.geometry import Point, box

from evrptw_cle.depot_grouping import refine_depot_groups

from evrptw_cle.public_locations import (
    excluded_location_reason,
    filter_public_locations,
    prepare_exclusion_masks,
)


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"landuse": "military"}, "military"),
        ({"name": "Federal Bureau of Investigation"}, "intelligence"),
        ({"Federal Agency Code": "DAF"}, "military"),
        ({"Federal Agency Code": "DOJ"}, "law_enforcement"),
        ({"amenity": "prison"}, "correctional"),
        ({"plant:source": "nuclear"}, "nuclear"),
        ({"name": "County Sheriff Warehouse"}, "law_enforcement"),
        ({"motor_vehicle": "no"}, "motor_access_prohibited"),
        ({"access": "no", "motor_vehicle": "delivery"}, ""),
        ({"access": "private", "building": "warehouse"}, ""),
        ({"name": "Civil Airport Cargo Terminal"}, ""),
        ({"name": "Port Logistics Warehouse"}, ""),
        ({"name": "Salvation Army Warehouse"}, ""),
        ({"name": "Navy Pier"}, ""),
        ({"osm_tags_json": pd.NA, "name": "Old Navy warehouse"}, ""),
    ],
)
def test_public_host_categories(tags, expected):
    assert excluded_location_reason(tags) == expected


def test_unnamed_warehouse_in_sensitive_host_removed_without_exporting_inventory():
    hosts = gpd.GeoDataFrame(
        {"name": ["private context"], "landuse": ["military"]},
        geometry=[box(0, 0, 100, 100)],
        crs=3857,
    )
    masks = prepare_exclusion_masks(hosts, "EPSG:3857")
    assert set(masks.columns) == {"exclusion_category", "geometry"}
    candidates = gpd.GeoDataFrame(
        {"candidate_id": ["inside", "outside"], "building": ["warehouse", "warehouse"]},
        geometry=[Point(50, 50), Point(150, 50)],
        crs=3857,
    )
    kept, summary = filter_public_locations(candidates, masks)
    assert list(kept.candidate_id) == ["outside"]
    assert summary["removed_count"] == 1
    assert "inside" not in json.dumps(summary)
    assert "private context" not in json.dumps(summary)


def test_sensitive_poi_buffer_is_metric_and_no_unrelated_line_area():
    raw = gpd.GeoDataFrame({"amenity": ["police"]}, geometry=[Point(0, 0)], crs=3857)
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    points = gpd.GeoDataFrame(geometry=[Point(49, 0), Point(51, 0)], crs=3857)
    kept, summary = filter_public_locations(points, masks)
    assert len(kept) == 1
    assert summary["removed_count"] == 1


def test_same_address_merge_is_bounded_and_street_only_is_insufficient():
    frame = gpd.GeoDataFrame(
        {
            "source_osm_id": ["n1", "n2", "n3", "n4"],
            "facility_group_id": ["a", "b", "c", "d"],
            "operator": ["Company A", "Company B", "Company C", "Company D"],
            "osm_tags_json": [json.dumps({"addr:housenumber": "1", "addr:street": "Road"})] * 3
            + [json.dumps({"addr:street": "Road"})],
        },
        geometry=[Point(0, 0), Point(80, 0), Point(160, 0), Point(10, 0)],
        crs=3857,
    )
    result, _ = refine_depot_groups(frame, "EPSG:3857")
    assert result.facility_group_id.iloc[0] == result.facility_group_id.iloc[1]
    assert result.facility_group_id.iloc[1] != result.facility_group_id.iloc[2]
    assert result.facility_group_id.iloc[0] != result.facility_group_id.iloc[3]


def test_campus_context_merges_companies_but_rejects_long_industrial_corridor():
    frame = gpd.GeoDataFrame(
        {
            "source_osm_id": ["a", "b"],
            "facility_group_id": ["a", "b"],
            "osm_tags_json": ["{}", "{}"],
        },
        geometry=[Point(0, 0), Point(900, 0)],
        crs=3857,
    )
    context = gpd.GeoDataFrame(
        {"landuse": ["industrial"], "source_osm_id": ["area"]},
        geometry=[box(-10, -10, 2010, 10)],
        crs=3857,
    )
    merged, _ = refine_depot_groups(frame, "EPSG:3857", context=context)
    assert merged.facility_group_id.nunique() == 1
    frame.geometry = [Point(0, 0), Point(2000, 0)]
    distinct, _ = refine_depot_groups(frame, "EPSG:3857", context=context)
    assert distinct.facility_group_id.nunique() == 2
