from __future__ import annotations

from pathlib import Path

import pytest

from evrptw_cle.moves_speed import (
    DEFAULT_MOVES_SPEED_PROFILE_PATH,
    derive_moves_speed_profile_from_sql,
    load_moves_speed_profile,
    moves_road_type_from_hpms,
    moves_road_type_from_osm,
    resolve_moves_speed_profile,
    speed_retention_factor,
)


def test_frozen_moves5_profile_is_internally_consistent() -> None:
    path = Path(__file__).parent / "fixtures" / "legacy_moves5_profile.json"
    profile = load_moves_speed_profile(path)
    assert profile["source_database"] == "movesdb20241112"
    assert profile["source_type_id"] == 32
    assert speed_retention_factor(
        profile, road_type="urban_restricted_access", day_type="weekday"
    ) == pytest.approx(53.115367 / 70.0)
    assert speed_retention_factor(
        profile, road_type="urban_unrestricted_access", day_type="weekend"
    ) == pytest.approx(25.383511 / 45.0)


def test_moves_road_type_adapter_prefers_access_semantics() -> None:
    assert moves_road_type_from_hpms(1) == "urban_restricted_access"
    assert moves_road_type_from_hpms(4) == "urban_unrestricted_access"
    assert moves_road_type_from_osm("motorway") == "urban_restricted_access"
    assert moves_road_type_from_osm("motorway_link") == "urban_restricted_access"
    assert moves_road_type_from_osm("residential") == "urban_unrestricted_access"


@pytest.mark.parametrize(
    "city,state",
    [
        ("new-york", "36"),
        ("los-angeles", "06"),
        ("chicago", "17"),
        ("houston", "48"),
        ("phoenix", "04"),
        ("philadelphia", "42"),
        ("san-antonio", "48"),
        ("san-diego", "06"),
        ("dallas", "48"),
        ("fort-worth", "48"),
        ("jacksonville", "12"),
    ],
)
def test_default_city_selects_approved_state_profile(city, state) -> None:
    profile = load_moves_speed_profile(resolve_moves_speed_profile(city_slug=city))
    assert profile["state_fips"] == state
    assert profile["observation_vintage"]["annual_temporal_aggregation"].startswith("2023")


def test_new_cities_require_explicit_supported_state() -> None:
    with pytest.raises(ValueError, match="State FIPS is required"):
        resolve_moves_speed_profile(city_slug="austin")
    assert resolve_moves_speed_profile(city_slug="austin", state_fips="48") == (
        resolve_moves_speed_profile(city_slug="houston")
    )
    assert resolve_moves_speed_profile(city_slug="san-jose", state_fips="06") == (
        resolve_moves_speed_profile(city_slug="san-diego")
    )
    with pytest.raises(ValueError, match="No frozen StreetLight"):
        resolve_moves_speed_profile(city_slug="seattle", state_fips="53")
    with pytest.raises(ValueError, match="City/state mismatch"):
        resolve_moves_speed_profile(city_slug="houston", state_fips="42")
    tx = resolve_moves_speed_profile(city_slug="houston")
    with pytest.raises(ValueError, match="different state"):
        resolve_moves_speed_profile(tx, state_fips="42")


@pytest.mark.parametrize("state", ["04", "06", "12", "17", "36", "42", "48"])
def test_installed_state_statistics_reproduce_coefficients(state) -> None:
    path = resolve_moves_speed_profile(DEFAULT_MOVES_SPEED_PROFILE_PATH, state_fips=state)
    profile = load_moves_speed_profile(path)
    source = path.parent / profile["reproduction"]["canonical_sql"]
    reproduced = derive_moves_speed_profile_from_sql(source)
    assert reproduced["road_types"] == profile["road_types"]
