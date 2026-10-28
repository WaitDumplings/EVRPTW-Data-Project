from __future__ import annotations

import pandas as pd
import pytest

from evrptw_cle.charger_resolution import address_key, resolve_afdc_coordinates


def _afdc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2],
            "Street Address": ["100 Main St", "200 Oak Ave"],
            "City": ["Example", "Example"],
            "State": ["CA", "CA"],
            "ZIP": ["90001", "90001-1234"],
            "Latitude": [34.0, 34.1],
            "Longitude": [-118.0, -118.1],
        }
    )


def test_address_key_is_case_and_zip_extension_insensitive() -> None:
    assert address_key("100 MAIN ST.", "Example", "ca", "90001-4321") == address_key(
        "100 Main St", "EXAMPLE", "CA", "90001"
    )


def test_resolution_precedence_and_raw_geometry_are_auditable() -> None:
    census = pd.DataFrame(
        {
            "afdc_id": [1],
            "census_match_status": ["match"],
            "census_longitude": [-118.001],
            "census_latitude": [34.001],
            "census_input_address": ["100 Main St, Example, CA, 90001"],
        }
    )
    osm = pd.DataFrame(
        {
            "source_osm_id": ["n9"],
            "addr_housenumber": ["100"],
            "addr_street": ["Main St"],
            "addr_city": ["Example"],
            "addr_state": ["CA"],
            "addr_postcode": ["90001"],
            "longitude": [-118.002],
            "latitude": [34.002],
        }
    )
    manual = pd.DataFrame(
        {
            "afdc_id": [1],
            "resolved_longitude": [-118.003],
            "resolved_latitude": [34.003],
            "review_note": ["operator map reviewed"],
        }
    )
    with pytest.raises(ValueError,match="manual coordinate overrides are disabled"):
        resolve_afdc_coordinates(_afdc(),manual_overrides=manual)
    afdc=_afdc()
    afdc["Geocode Status"]="200-8"
    result = resolve_afdc_coordinates(afdc, census_results=census, osm_pois=osm).set_index("afdc_id")

    assert result.loc[1, "raw_afdc_longitude"] == pytest.approx(-118.0)
    assert result.loc[1, "address_anchor_source"] == "us_census_geocoder"
    assert result.loc[1, "resolved_longitude"] == pytest.approx(-118.0)
    assert result.loc[1, "resolved_geometry_source"] == "afdc_raw"
    assert result.loc[1, "coordinate_validation_tier"] == "V_address_aux_250m"
    assert bool(result.loc[1, "coordinate_release_eligible"])
    assert result.loc[1, "matched_osm_charging_id"] == "n9"
    assert result.loc[1, "raw_to_address_anchor_m"] > 0
    assert result.loc[2, "resolved_geometry_source"] == "afdc_raw"
    assert result.loc[2, "location_resolution_status"] == "uncorroborated_source_coordinate"
    assert result.loc[2, "coordinate_validation_status"] == ("uncorroborated_source_coordinate")
    assert not bool(result.loc[2, "coordinate_candidate_eligible"])


@pytest.mark.parametrize("echoed_address", [None, "", "   "])
def test_address_coordinate_needs_current_census_input_binding(echoed_address) -> None:
    afdc = _afdc().iloc[:1].copy()
    afdc["Geocode Status"] = "200-8"
    evidence = {
        "afdc_id": [1],
        "census_match_status": ["match"],
        "census_longitude": [-118.0],
        "census_latitude": [34.0],
    }
    if echoed_address is not None:
        evidence["census_input_address"] = [echoed_address]
    result = resolve_afdc_coordinates(afdc, census_results=pd.DataFrame(evidence))
    assert result.iloc[0].census_match_status == "missing_input_address"
    assert result.iloc[0].address_anchor_source == "not_available"
    assert not result.iloc[0].coordinate_candidate_eligible
