from __future__ import annotations

import json

import pandas as pd


def _module():
    from evrptw_cle import depot_candidates

    return depot_candidates


def test_warehouse_area_is_a_flag_not_a_hard_filter() -> None:
    module = _module()
    row = pd.Series({"building": "warehouse", "name": "Unnamed warehouse"})
    assert module._evidence_tier(row, 1_500.0, 1_000.0)[0] == "B_warehouse_proxy"
    assert module._evidence_tier(row, 1_500.0, 2_000.0)[0] == "B_warehouse_proxy"


def test_unrelated_open_ended_tags_do_not_promote_candidates() -> None:
    module = _module()
    spaceflight = pd.Series(
        {"building": "industrial", "logistics": "spaceflight", "name": "TransAstra"}
    )
    taxi = pd.Series({"depot": "taxi", "name": "Taxi depot"})
    assert module._evidence_tier(spaceflight, 5_000.0, 1_000.0)[0] == "C_industrial_proxy"
    assert not module._is_matched_object(taxi)


def test_named_carrier_facility_is_tier_a() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "Amazon Delivery Station",
            "operator": "Amazon",
            "building": "warehouse",
        }
    )
    assert module._evidence_tier(row, 10_000.0, 1_000.0)[0] == "A_osm_explicit"


def test_carrier_retail_counter_is_rejected() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "The UPS Store",
            "operator": "UPS",
            "amenity": "post_depot",
        }
    )
    tier, reason, _ = module._evidence_tier(row, 0.0, 1_000.0)
    assert tier == "C_industrial_proxy"
    assert "retail counter" in reason


def test_carrier_point_requires_dispatch_facility_evidence() -> None:
    module = _module()
    row = pd.Series(
        {
            "operator": "FedEx",
            "amenity": "post_depot",
        }
    )
    tier, reason, _ = module._evidence_tier(row, 0.0, 1_000.0)
    assert tier == "C_industrial_proxy"
    assert "physical dispatch-facility evidence" in reason


def test_noncarrier_named_logistics_facility_remains_optional() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "SDSU Logistics",
            "amenity": "post_depot",
            "building": "yes",
        }
    )
    assert module._evidence_tier(row, 2_179.0, 1_000.0)[0] == "B_warehouse_proxy"


def test_office_only_remains_b_even_for_known_carrier():
    module = _module()
    row = pd.Series(
        {"name": "FedEx Regional Headquarters", "office": "logistics", "operator": "FedEx"}
    )
    assert module._evidence_tier(row, 0, 1000)[0] == "B_warehouse_proxy"
    row["logistics"] = "spaceflight"
    assert module._evidence_tier(row, 0, 1000)[0] == "C_industrial_proxy"


def _touching_warehouse_groups(source_name=None):
    import geopandas as gpd
    from shapely.geometry import box

    raw = gpd.GeoDataFrame(
        {"source_osm_id": ["w1", "w2"], "building": ["warehouse", "warehouse"]},
        geometry=[
            box(-117.100, 32.800, -117.099, 32.801),
            box(-117.099, 32.800, -117.098, 32.801),
        ],
        crs=4326,
    )
    if source_name is not None:
        raw["name"] = source_name
    boundary = gpd.GeoDataFrame(geometry=[box(-117.2, 32.7, -117.0, 32.9)], crs=4326)
    result, _ = _module().build_public_depot_candidates(raw, boundary, "example", None)
    return result


def test_touching_unnamed_warehouses_do_not_share_display_name_identity():
    result = _touching_warehouse_groups()
    assert result.facility_name.eq("Unnamed warehouse").all()
    assert result.facility_group_id.nunique() == 2


def test_touching_warehouses_with_shared_source_name_keep_identity_merge():
    result = _touching_warehouse_groups("Example Distribution Campus")
    assert result.facility_group_id.nunique() == 1


def test_depot_source_tags_preserve_native_dictionary_and_present_permission_fields():
    import geopandas as gpd
    from shapely.geometry import Point, box

    native = {
        "motorcar": "yes",
        "access:delivery": "yes",
        "access": "private",
        "operator:wikidata": "Q123",
        "source:permission": "original snapshot",
    }
    raw = gpd.GeoDataFrame(
        {"source_osm_id": ["n1"], "building": ["warehouse"],
         "motor_vehicle": ["permissive"], "osm_tags_json": [json.dumps(native)]},
        geometry=[Point(-117.1, 32.8)], crs=4326,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(-117.2, 32.7, -117., 32.9)], crs=4326)
    result, _ = _module().build_public_depot_candidates(raw, boundary, "example", None)
    tags = json.loads(result.osm_tags_json.iloc[0])
    assert all(tags[key] == value for key, value in native.items())
    assert tags["motor_vehicle"] == "permissive"
    assert "locked" not in tags  # An absent source tag must not be invented.
    assert "facility_name" not in tags  # Generated display labels are not raw tags.


def test_native_depot_delivery_prohibition_is_used_before_candidate_admission():
    row = pd.Series({"building": "warehouse", "osm_tags_json": json.dumps(
        {"motorcar": "yes", "access:delivery": "no"})})
    tier, reason, _ = _module()._evidence_tier(row, 1000., 1000.)
    assert tier == "C_industrial_proxy"
    assert reason == "excluded by public-location policy"


def test_flat_native_depot_properties_are_captured_before_derived_columns():
    import geopandas as gpd
    from shapely.geometry import Point, box

    raw = gpd.GeoDataFrame(
        {"source_osm_id": ["n1"], "source_osm_url": ["https://www.openstreetmap.org/node/1"],
         "building": ["warehouse"], "operator:variant": ["regional division"],
         "old_filter_hit": [True], "osm_object_version": [4],
         "osm_edit_timestamp": ["2026-10-01T00:00:00Z"], "geometry_priority": [2]},
        geometry=[Point(-117.1, 32.8)], crs=4326,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(-117.2, 32.7, -117., 32.9)], crs=4326)
    result, _ = _module().build_public_depot_candidates(raw, boundary, "example", None)
    tags = json.loads(result.osm_tags_json.iloc[0])
    assert tags == {"building": "warehouse", "operator:variant": "regional division"}
    assert result.facility_name.iloc[0] == "Unnamed warehouse"
    assert "osm_tags_json" not in raw  # The source table itself stays unchanged.
