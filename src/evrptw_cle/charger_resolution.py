"""AFDC-native coordinates with independently recorded address corroboration."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

COORDINATE_POLICY = "afdc_native_gps_point_address_aux_v3"
ADDRESS_SUPPORT_MAX_M = 250.0
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(value: Any) -> str:
    return (
        ""
        if value is None or pd.isna(value)
        else NON_ALNUM.sub(" ", str(value).strip().lower()).strip()
    )


def address_key(street: Any, city: Any, state: Any, postal_code: Any) -> str:
    return "|".join(
        (
            normalize_text(street),
            normalize_text(city),
            normalize_text(state),
            normalize_text(postal_code).split(" ")[0][:5],
        )
    )


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    value = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(min(1.0, max(0.0, value))))


def valid_coordinate(lon: Any, lat: Any) -> bool:
    try:
        return (
            math.isfinite(float(lon))
            and math.isfinite(float(lat))
            and -180 <= float(lon) <= 180
            and -90 <= float(lat) <= 90
        )
    except (TypeError, ValueError):
        return False


def _osm_address_key(row: pd.Series) -> str:
    street = " ".join(normalize_text(row.get(key)) for key in ("addr_housenumber", "addr_street"))
    return address_key(
        street, row.get("addr_city"), row.get("addr_state"), row.get("addr_postcode")
    )


def resolve_afdc_coordinates(
    afdc: pd.DataFrame,
    *,
    census_results: pd.DataFrame | None = None,
    osm_pois: pd.DataFrame | None = None,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep native geometry. GPS/Point admits independently; Address needs support.

    Census interpolation and same-address OSM are QA evidence, never automatic
    replacement coordinates. A documented manual override is the only exception.
    Census input addresses are bound to each current AFDC row when supplied.
    """
    result = afdc.copy()
    result["afdc_id"] = pd.to_numeric(result["ID"], errors="raise").astype(int)
    if result.afdc_id.duplicated().any():
        raise ValueError("AFDC IDs must be unique")
    for axis in ("longitude", "latitude"):
        result[f"raw_afdc_{axis}"] = pd.to_numeric(result[axis.title()], errors="coerce")
        result[f"resolved_{axis}"] = result[f"raw_afdc_{axis}"]
        result[f"address_anchor_{axis}"] = np.nan
    result["normalized_address_key"] = result.apply(
        lambda r: address_key(r.get("Street Address"), r.get("City"), r.get("State"), r.get("ZIP")),
        axis=1,
    )
    result["address_anchor_source"] = "not_available"
    result["census_match_status"] = "not_requested"
    result["raw_to_address_anchor_m"] = np.nan
    result["raw_to_osm_poi_m"] = np.nan
    result["matched_osm_charging_id"] = ""
    result["coordinate_review_note"] = ""
    result["resolved_geometry_source"] = "afdc_raw"
    census = {}
    if census_results is not None and not census_results.empty:
        if census_results.afdc_id.duplicated().any():
            raise ValueError("Census evidence contains duplicate AFDC IDs")
        census = {int(r.afdc_id): r for _, r in census_results.iterrows()}
    osm = {}
    if osm_pois is not None:
        for _, r in osm_pois.iterrows():
            if valid_coordinate(r.get("longitude"), r.get("latitude")):
                osm.setdefault(_osm_address_key(r), []).append(r)
    manual = {}
    if manual_overrides is not None:
        if manual_overrides.afdc_id.duplicated().any():
            raise ValueError("Manual evidence contains duplicate AFDC IDs")
        manual = {int(r.afdc_id): r for _, r in manual_overrides.iterrows()}
    tiers, statuses, eligible = [], [], []
    for idx, row in result.iterrows():
        lon, lat = row.raw_afdc_longitude, row.raw_afdc_latitude
        valid = valid_coordinate(lon, lat)
        match = census.get(int(row.afdc_id))
        if match is not None:
            status = str(match.get("census_match_status", "unknown")).lower()
            # Batch API echoes the input address; reject evidence for a changed address.
            echoed = normalize_text(match.get("census_input_address"))
            expected = normalize_text(
                ", ".join(str(row.get(k, "")) for k in ("Street Address", "City", "State", "ZIP"))
            )
            if echoed and echoed != expected:
                status = "stale_input_address"
            result.at[idx, "census_match_status"] = status
            a_lon, a_lat = match.get("census_longitude"), match.get("census_latitude")
            if status == "match" and valid_coordinate(a_lon, a_lat):
                result.at[idx, "address_anchor_longitude"] = float(a_lon)
                result.at[idx, "address_anchor_latitude"] = float(a_lat)
                result.at[idx, "address_anchor_source"] = "us_census_geocoder"
                if valid:
                    result.at[idx, "raw_to_address_anchor_m"] = haversine_m(
                        lon, lat, float(a_lon), float(a_lat)
                    )
        candidates = osm.get(row.normalized_address_key, [])
        if valid and candidates:
            nearest = min(
                candidates,
                key=lambda r: haversine_m(lon, lat, float(r.longitude), float(r.latitude)),
            )
            result.at[idx, "raw_to_osm_poi_m"] = haversine_m(
                lon, lat, float(nearest.longitude), float(nearest.latitude)
            )
            result.at[idx, "matched_osm_charging_id"] = str(nearest.get("source_osm_id", ""))
        native = normalize_text(row.get("Geocode Status")) in {"gps", "200 9"}
        address_supported = (
            normalize_text(row.get("Geocode Status")) == "200 8"
            and result.at[idx, "raw_to_address_anchor_m"] <= ADDRESS_SUPPORT_MAX_M
        )
        tier = (
            "V_native_gps_point"
            if valid and native
            else "V_address_aux_250m"
            if valid and address_supported
            else "V0_quarantined"
        )
        state = (
            "native_source_precision_accepted"
            if valid and native
            else "address_corroborated_exact_geometry_unverified"
            if valid and address_supported
            else "uncorroborated_source_coordinate"
        )
        if int(row.afdc_id) in manual:
            override = manual[int(row.afdc_id)]
            if not valid_coordinate(
                override.get("resolved_longitude"), override.get("resolved_latitude")
            ) or not normalize_text(override.get("review_note")):
                raise ValueError("Manual overrides require valid coordinates and a review note")
            for axis in ("longitude", "latitude"):
                result.at[idx, f"resolved_{axis}"] = float(override[f"resolved_{axis}"])
            result.at[idx, "coordinate_review_note"] = str(override.review_note)
            result.at[idx, "resolved_geometry_source"] = "manual_reviewed_override"
            tier, state = "V1_manual_reviewed_exact", "exact_geometry_corroborated"
        tiers.append(tier)
        statuses.append(state)
        eligible.append(tier != "V0_quarantined")
    result["coordinate_policy_version"] = COORDINATE_POLICY
    result["coordinate_validation_tier"] = tiers
    result["coordinate_validation_status"] = statuses
    result["location_resolution_status"] = statuses
    result["coordinate_candidate_eligible"] = eligible
    # Location eligibility does not certify power, connector or vehicle parameters.
    result["coordinate_release_eligible"] = eligible
    return result
