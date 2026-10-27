import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from evrptw_cle.charger_locations import collapse_charging_locations
from evrptw_cle.charger_resolution import resolve_afdc_coordinates
from evrptw_cle.facilities import _normalize_chargers


def source():
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Station Name": ["Garage", "Garage DC", "Other garage"],
            "Street Address": ["100 Main St", "100 Main Street", "100 Main St"],
            "City": ["Example"] * 3,
            "State": ["CA"] * 3,
            "ZIP": ["90001"] * 3,
            "Longitude": [-118.0, -118.0, -118.001],
            "Latitude": [34.0] * 3,
            "Geocode Status": ["GPS", "200-9", "200-8"],
            "EV Level1 EVSE Num": [0] * 3,
            "EV Level2 EVSE Num": [2, 0, 4],
            "EV DC Fast Count": [0, 10, 0],
            "EV Connector Types": ["J1772", "J1772COMBO", "J1772"],
            "Restricted Access": [False, True, False],
            "Maximum Vehicle Class": ["MD"] * 3,
            "Access Code": ["public"] * 3,
            "Access Days Time": ["24 hours daily"] * 3,
        }
    )
    result = resolve_afdc_coordinates(frame)
    result = gpd.GeoDataFrame(
        result, geometry=[Point(x, y) for x, y in zip(result.Longitude, result.Latitude)], crs=4326
    )
    result = _normalize_chargers(result)
    result["charger_candidate_eligible"] = [True, False, False]
    return result


def test_one_location_preserves_access_and_avoids_restricted_equipment():
    locations, members = collapse_charging_locations(source())
    assert len(locations) == 2 and len(members) == 3
    location = locations.loc[locations.source_member_count.eq(2)].iloc[0]
    assert location.ID == 1 and location.reference_charge_mode == "ac_level2"
    assert json.loads(location.eligible_afdc_ids_json) == [1]
    configs = json.loads(location.equipment_access_members_json)
    assert configs[1]["Restricted Access"] is True
    assert configs[1]["EV DC Fast Count"] == 10
    assert location.location_total_ports is None
    assert members.ID.is_unique


def test_coordinate_collision_quarantines_every_member():
    frame = source()
    frame.loc[1, "Street Address"] = "200 Other St"
    locations, members = collapse_charging_locations(frame)
    assert not members.loc[:1, "charger_candidate_eligible"].any()
    assert (
        members.loc[:1, "location_quarantine_reason"].eq("same_coordinate_address_conflict").all()
    )
    assert not locations.charger_candidate_eligible.any()


def test_location_identity_is_order_independent_and_member_stable():
    locations, _ = collapse_charging_locations(source())
    reordered, _ = collapse_charging_locations(source().iloc[::-1])
    single, _ = collapse_charging_locations(source().iloc[[0]])
    assert set(locations.charger_id) == set(reordered.charger_id)
    assert single.iloc[0].charger_id in set(locations.charger_id)


def test_explicit_inactive_and_private_access_text_quarantines():
    frame = source()
    frame.loc[0, "Station Name"] = "Stations Removed"
    frame.loc[2, "Station Name"] = "Residents Only"
    frame["charger_candidate_eligible"] = True
    _, members = collapse_charging_locations(frame)
    assert members.loc[0, "location_quarantine_reason"] == "source_status_text_conflict"
    assert members.loc[2, "location_quarantine_reason"] == "source_public_access_text_conflict"
    assert not members.loc[[0, 2], "charger_candidate_eligible"].any()


def test_native_gps_and_point_need_no_address_and_osm_does_not_move_them():
    frame = source().drop(columns="geometry")
    osm = pd.DataFrame(
        {
            "addr_housenumber": ["100"],
            "addr_street": ["Main St"],
            "addr_city": ["Example"],
            "addr_state": ["CA"],
            "addr_postcode": ["90001"],
            "longitude": [-118.02],
            "latitude": [34.02],
            "source_osm_id": ["n1"],
        }
    )
    result = resolve_afdc_coordinates(frame, osm_pois=osm)
    assert result.loc[:1, "coordinate_candidate_eligible"].all()
    assert result.resolved_longitude.equals(result.raw_afdc_longitude)
    assert result.loc[0, "raw_to_osm_poi_m"] > 250
    assert not result.loc[2, "coordinate_candidate_eligible"]


def test_address_auxiliary_rejects_far_stale_and_unmatched_evidence():
    frame = source().drop(columns="geometry").iloc[[2]].copy()
    census = pd.DataFrame(
        {
            "afdc_id": [3],
            "census_match_status": ["match"],
            "census_longitude": [-118.0011],
            "census_latitude": [34.0],
            "census_input_address": ["100 Main St, Example, CA, 90001"],
        }
    )
    assert resolve_afdc_coordinates(
        frame, census_results=census
    ).coordinate_candidate_eligible.all()
    for column, value in [
        ("census_longitude", -119.0),
        ("census_input_address", "200 Changed St, Example, CA, 90001"),
        ("census_match_status", "tie"),
    ]:
        changed = census.copy()
        changed[column] = value
        assert not resolve_afdc_coordinates(
            frame, census_results=changed
        ).coordinate_candidate_eligible.any()


def test_extreme_unverified_road_connector_is_quarantined():
    frame = source()
    frame["road_access_distance_m"] = [1000.0, 1000.0, 1000.01]
    frame["charger_candidate_eligible"] = True
    _, members = collapse_charging_locations(frame)
    assert members.loc[0, "charger_candidate_eligible"]
    assert not members.loc[2, "charger_candidate_eligible"]
    assert (
        members.loc[2, "location_quarantine_reason"]
        == "road_coverage_or_access_unresolved_over_1000m"
    )
