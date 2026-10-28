"""Reviewed national depot source classification and spatial grouping.

Business function is a potential-location proxy, not verified ongoing operation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from evrptw_cle.depot_grouping import DEPOT_GROUPING_POLICY, refine_depot_groups
from evrptw_cle.public_locations import (
    excluded_location_reason,
    filter_public_locations,
)

ALLOWED_LOGISTICS_VALUES = {
    "yes",
    "cargo",
    "courier",
    "delivery",
    "distribution",
    "freight",
    "fulfillment",
    "logistics",
    "mail",
    "parcel",
    "post",
    "warehouse",
}

ALLOWED_DEPOT_VALUES = {
    "yes",
    "cargo",
    "courier",
    "delivery",
    "distribution",
    "freight",
    "fulfillment",
    "logistics",
    "mail",
    "parcel",
    "post",
    "warehouse",
}

CARRIER_PATTERNS = [
    (re.compile(r"\bamazon\b", re.IGNORECASE), "Amazon"),
    (re.compile(r"\b(?:ups|united parcel service)\b", re.IGNORECASE), "UPS"),
    (re.compile(r"\b(?:fedex|federal express)\b", re.IGNORECASE), "FedEx"),
    (re.compile(r"\bdhl\b", re.IGNORECASE), "DHL"),
    (re.compile(r"\b(?:usps|united states postal service)\b", re.IGNORECASE), "USPS"),
    (re.compile(r"\bontrac\b", re.IGNORECASE), "OnTrac"),
    (re.compile(r"\blasership\b", re.IGNORECASE), "LaserShip"),
    (re.compile(r"\bgls\b", re.IGNORECASE), "GLS"),
]

FACILITY_PATTERN = re.compile(
    r"\b(?:delivery|distribution|fulfillment|sortation|hub|processing|annex|depot|warehouse|logistics|parcel|courier|postal)\b",
    re.IGNORECASE,
)

RETAIL_COUNTER_PATTERN = re.compile(
    r"\b(?:the\s+ups\s+store|fedex\s+office|postalannex|mailboxes\s+etc|"
    r"print(?:ing)?\s*(?:and|&)\s*ship|retail\s+shipping\s+counter)\b",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _carrier(row: pd.Series) -> str:
    haystack = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "company", "owner")
    )
    for pattern, label in CARRIER_PATTERNS:
        if pattern.search(haystack):
            return label
    return ""


def _candidate_tags(row: pd.Series) -> dict[str, str]:
    columns = [
        "name",
        "official_name",
        "operator",
        "brand",
        "building",
        "building:use",
        "landuse",
        "industrial",
        "office",
        "shop",
        "amenity",
        "depot",
        "logistics",
        "access",
        "access:delivery",
        "addr:housenumber",
        "addr:street",
        "addr:city",
        "addr:postcode",
        "website",
        "source",
        "source:url",
    ]
    return {column: value for column in columns if (value := _text(row.get(column)))}


def _is_matched_object(row: pd.Series) -> bool:
    building = _text(row.get("building")).lower()
    landuse = _text(row.get("landuse")).lower()
    industrial = _text(row.get("industrial")).lower()
    office = _text(row.get("office")).lower()
    amenity = _text(row.get("amenity")).lower()
    depot_value = _text(row.get("depot")).lower()
    logistics_value = _text(row.get("logistics")).lower()
    return any(
        (
            building in {"warehouse", "industrial"},
            landuse in {"industrial", "depot", "logistics", "warehouse"},
            industrial in {"warehouse", "logistics", "distribution"},
            office == "logistics",
            amenity == "post_depot",
            depot_value in ALLOWED_DEPOT_VALUES,
            logistics_value in ALLOWED_LOGISTICS_VALUES,
        )
    )


def _evidence_tier(
    row: pd.Series,
    facility_area_m2: float,
    min_warehouse_area_m2: float,
) -> tuple[str, str, int]:
    if excluded_location_reason(row):
        return "C_industrial_proxy", "excluded by public-location policy", 0
    carrier = _carrier(row)
    name = _text(row.get("name")) or _text(row.get("official_name"))
    combined = " ".join(
        _text(row.get(column)) for column in ("name", "official_name", "operator", "brand")
    )
    building = _text(row.get("building")).lower()
    landuse = _text(row.get("landuse")).lower()
    industrial = _text(row.get("industrial")).lower()
    office = _text(row.get("office")).lower()
    amenity = _text(row.get("amenity")).lower()
    depot_value = _text(row.get("depot")).lower()
    logistics_value = _text(row.get("logistics")).lower()
    explicit_depot = (
        amenity == "post_depot"
        or office == "logistics"
        or depot_value in ALLOWED_DEPOT_VALUES
        or logistics_value in ALLOWED_LOGISTICS_VALUES
    )
    warehouse = (
        building == "warehouse"
        or industrial == "warehouse"
        or landuse in {"warehouse", "logistics", "depot"}
    )
    industrial_site = building == "industrial" or landuse == "industrial"
    facility_hint = bool(FACILITY_PATTERN.search(combined))
    has_address = bool(_text(row.get("addr:street")))

    physical_dispatch = (
        warehouse
        or amenity == "post_depot"
        or depot_value in ALLOWED_DEPOT_VALUES
        or logistics_value in ALLOWED_LOGISTICS_VALUES
    )
    office_only = office == "logistics" and not physical_dispatch
    if office_only and logistics_value in {"aerospace", "spaceflight"}:
        return "C_industrial_proxy", "explicit non-ground-logistics office", 0

    retail_haystack = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "website")
    )
    if RETAIL_COUNTER_PATTERN.search(retail_haystack):
        return "C_industrial_proxy", "parcel-carrier retail counter", 0

    if office_only:
        return (
            "B_warehouse_proxy",
            "ordinary logistics-office proxy; no warehouse verification required",
            2 + bool(name) + bool(has_address),
        )

    # A carrier name is not enough: it must be tied to a physical warehouse,
    # an explicit dispatch-function name, or an explicit depot/logistics tag.
    # Area is kept as a continuous attribute and sensitivity flag, not used as
    # ground-truth dispatch evidence.
    strong_dispatch_evidence = (
        warehouse or facility_hint or (explicit_depot and facility_area_m2 > 0)
    )
    carrier_facility = bool(carrier) and strong_dispatch_evidence
    if carrier == "USPS":
        usps_facility_hint = bool(
            re.search(r"\b(?:distribution|processing|annex|depot|warehouse)\b", name, re.IGNORECASE)
        )
        carrier_facility = (
            warehouse or usps_facility_hint or (amenity == "post_depot" and facility_area_m2 > 0)
        )

    if carrier_facility:
        reasons = []
        if carrier:
            reasons.append(f"named {carrier} facility")
        if explicit_depot:
            reasons.append("explicit depot/logistics tag")
        if warehouse:
            reasons.append("warehouse tag")
        if industrial_site and not warehouse:
            reasons.append("industrial-facility tag")
        if facility_hint:
            reasons.append("delivery/hub/logistics name")
        score = 6 + 3 * bool(explicit_depot) + 2 * bool(warehouse) + bool(has_address)
        return "A_osm_explicit", "; ".join(reasons), score

    if carrier and explicit_depot and not warehouse and not facility_hint:
        return (
            "C_industrial_proxy",
            "carrier-branded point lacks physical dispatch-facility evidence",
            0,
        )

    warehouse_proxy = warehouse or office == "logistics" or explicit_depot
    if warehouse_proxy:
        reasons = ["warehouse/logistics land-use evidence"]
        if facility_area_m2 >= min_warehouse_area_m2:
            reasons.append(f"area sensitivity flag >= {min_warehouse_area_m2:g} m2")
        elif facility_area_m2 > 0:
            reasons.append(f"area sensitivity flag < {min_warehouse_area_m2:g} m2")
        score = 2 + 2 * bool(explicit_depot) + bool(name) + bool(has_address)
        return "B_warehouse_proxy", "; ".join(reasons), score

    return "C_industrial_proxy", "generic industrial evidence only", 0


def _function_signal(row: pd.Series) -> str:
    amenity = _text(row.get("amenity")).lower()
    if amenity == "post_depot":
        return "osm_post_depot"
    combined = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "depot", "logistics")
    )
    match = FACILITY_PATTERN.search(combined)
    return match.group(0).lower() if match else ""


def _address(row: pd.Series) -> str:
    parts = [
        " ".join(filter(None, [_text(row.get("addr:housenumber")), _text(row.get("addr:street"))])),
        _text(row.get("addr:city")),
        _text(row.get("addr:postcode")),
    ]
    return ", ".join(part for part in parts if part)


def _classify_candidates(
    raw: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    min_warehouse_area_m2: float,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    local_crs = boundary.estimate_utm_crs()
    projected = raw.to_crs(local_crs)
    raw["facility_area_m2"] = projected.geometry.area.round(1)
    tier_values = raw.apply(
        lambda row: _evidence_tier(
            row,
            float(row["facility_area_m2"]),
            min_warehouse_area_m2,
        ),
        axis=1,
    )
    raw["evidence_tier"] = tier_values.map(lambda value: value[0])
    raw["evidence_reason"] = tier_values.map(lambda value: value[1])
    raw["evidence_score"] = tier_values.map(lambda value: value[2])
    raw["carrier"] = raw.apply(_carrier, axis=1)
    raw["facility_name"] = raw.apply(
        lambda row: (
            _text(row.get("name"))
            or _text(row.get("official_name"))
            or (_carrier(row) + " depot" if _carrier(row) else "Unnamed warehouse")
        ),
        axis=1,
    )
    raw["address"] = raw.apply(_address, axis=1)
    raw["osm_tags_json"] = raw.apply(
        lambda row: json.dumps(_candidate_tags(row), ensure_ascii=False, sort_keys=True), axis=1
    )
    raw["facility_geometry_type"] = raw.geometry.geom_type
    raw["facility_area_known"] = raw["facility_geometry_type"].isin({"Polygon", "MultiPolygon"})
    raw["function_signal"] = raw.apply(_function_signal, axis=1)
    raw["strict_candidate_class"] = "warehouse_proxy"
    raw.loc[
        (raw["evidence_tier"] == "A_osm_explicit") & raw["carrier"].ne(""),
        "strict_candidate_class",
    ] = "carrier_facility_signal"
    raw.loc[
        (raw["evidence_tier"] == "A_osm_explicit") & raw["function_signal"].ne(""),
        "strict_candidate_class",
    ] = "dispatch_function_signal"
    raw["verification_status"] = "unverified_osm_candidate"
    office_only = raw.apply(
        lambda row: (
            _text(row.get("office")).lower() == "logistics"
            and not (
                _text(row.get("building")).lower() == "warehouse"
                or _text(row.get("industrial")).lower() == "warehouse"
                or _text(row.get("landuse")).lower() in {"warehouse", "logistics", "depot"}
                or _text(row.get("amenity")).lower() == "post_depot"
                or _text(row.get("depot")).lower() in ALLOWED_DEPOT_VALUES
                or _text(row.get("logistics")).lower() in ALLOWED_LOGISTICS_VALUES
            )
        ),
        axis=1,
    )
    raw["depot_site_evidence"] = np.where(
        office_only, "ordinary_office_proxy", "warehouse_or_dispatch_proxy"
    )
    raw["public_location_eligible"] = True
    raw["depot_release_eligible"] = False
    boundary_geometry = boundary.geometry.union_all()
    display_points = raw.geometry.representative_point()
    raw = raw.loc[display_points.covered_by(boundary_geometry)].copy()
    tier_counts = {tier: int(count) for tier, count in raw["evidence_tier"].value_counts().items()}
    retained = raw.loc[raw["evidence_tier"].isin({"A_osm_explicit", "B_warehouse_proxy"})]
    return gpd.GeoDataFrame(retained, geometry="geometry", crs=4326), tier_counts


def _normalized_facility_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def _assign_legacy_facility_groups(
    candidates: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Group OSM objects only when shared-facility evidence is explicit.

    R2 uses containment by an industrial/logistics/warehouse/depot land-use
    polygon. R3 uses a normalized name/operator match plus either an identical
    address or touching polygons. All other objects remain R5 singletons.
    The former 1000 m2 area flag is not a grouping or eligibility condition.
    """

    frame = candidates.reset_index(drop=True).copy()
    parent = list(range(len(frame)))
    merge_rules: dict[int, set[str]] = {index: set() for index in range(len(frame))}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, rule: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            winner, loser = sorted((root_left, root_right))
            parent[loser] = winner
            merge_rules[winner] |= merge_rules[loser]
        merge_rules[find(left)].add(rule)

    metric = frame.to_crs(boundary.estimate_utm_crs())
    geometries = np.asarray(metric.geometry.to_numpy(), dtype=object)
    tree = STRtree(geometries)
    pairs = tree.query(geometries, predicate="intersects")
    tags = [json.loads(value) for value in frame["osm_tags_json"].astype(str)]
    # Display names include generated placeholders such as "Unnamed warehouse".
    # Only original source identity can support the R3 same-name merge rule.
    names = [
        _normalized_facility_text(
            _text(tag.get("name")) or _text(tag.get("official_name")) or _text(tag.get("operator"))
        )
        for tag in tags
    ]
    addresses = [
        _normalized_facility_text(value)
        for value in frame.get("address", pd.Series("", index=frame.index))
    ]
    accepted_landuse = {"industrial", "logistics", "warehouse", "depot"}
    for left, right in zip(pairs[0], pairs[1]):
        left = int(left)
        right = int(right)
        if left >= right:
            continue
        left_geometry = geometries[left]
        right_geometry = geometries[right]
        left_landuse = str(tags[left].get("landuse") or "").lower()
        right_landuse = str(tags[right].get("landuse") or "").lower()
        r2 = (
            left_landuse in accepted_landuse
            and left_geometry.covers(right_geometry.representative_point())
        ) or (
            right_landuse in accepted_landuse
            and right_geometry.covers(left_geometry.representative_point())
        )
        same_identity = bool(names[left]) and names[left] == names[right]
        same_address = bool(addresses[left]) and addresses[left] == addresses[right]
        touching = left_geometry.touches(right_geometry)
        if r2:
            union(left, right, "R2_shared_industrial_or_logistics_landuse")
        elif same_identity and (same_address or touching):
            union(left, right, "R3_name_operator_and_address_or_touching")

    groups: dict[int, list[int]] = {}
    for index in range(len(frame)):
        groups.setdefault(find(index), []).append(index)
    group_id_by_index: dict[int, str] = {}
    group_rules_by_index: dict[int, str] = {}
    group_size_by_index: dict[int, int] = {}
    for root, members in groups.items():
        member_ids = sorted(frame.iloc[members]["source_osm_id"].astype(str))
        token = "|".join(member_ids).encode()
        group_id = "depot_group_" + hashlib.blake2b(token, digest_size=10).hexdigest()
        rules = sorted(merge_rules.get(find(root), set())) or ["R5_singleton"]
        for index in members:
            group_id_by_index[index] = group_id
            group_rules_by_index[index] = json.dumps(rules, sort_keys=True)
            group_size_by_index[index] = len(members)
    frame["facility_group_id"] = frame.index.map(group_id_by_index)
    frame["facility_group_member_count"] = frame.index.map(group_size_by_index).astype(int)
    frame["facility_group_merge_rules_json"] = frame.index.map(group_rules_by_index)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=candidates.crs)


def _assign_facility_groups(
    candidates: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    legacy = _assign_legacy_facility_groups(candidates, boundary)
    result, events = refine_depot_groups(
        legacy,
        str(boundary.estimate_utm_crs()),
        context=context,
    )
    result.attrs["grouping_audit"] = {
        "policy": DEPOT_GROUPING_POLICY,
        "source_object_count": len(result),
        "group_count": int(result.facility_group_id.nunique()),
        "merge_operations": len(events),
    }
    return result


def build_public_depot_candidates(raw, boundary, city_slug, masks, reference_area_m2=1000.0):
    """One production path for source filtering, classification and grouping."""
    context, public_filter = filter_public_locations(raw, masks)
    candidates, tier_counts = _classify_candidates(context.copy(), boundary, reference_area_m2)
    if candidates.empty:
        raise ValueError(f"No public depot source candidates remain for {city_slug}")
    candidates = candidates.sort_values("source_osm_id").reset_index(drop=True)
    candidates = _assign_facility_groups(candidates, boundary, context=context)
    candidates["city_slug"] = city_slug
    candidates["candidate_id"] = city_slug + "-depot-" + candidates["source_osm_id"].astype(str)
    tier_a = candidates.evidence_tier.eq("A_osm_explicit")
    candidates["benchmark_depot_class"] = np.where(
        tier_a, "strict_osm_candidate", "optional_warehouse_proxy"
    )
    candidates["strict_candidate_eligible"] = tier_a
    candidates["optional_candidate_eligible"] = ~tier_a
    candidates["cle_candidate_eligible"] = True
    candidates["last_mile_function_verified"] = False
    candidates["depot_release_eligible"] = False
    return candidates, {
        "raw_tier_counts": tier_counts,
        "public_location_policy": public_filter,
        "facility_grouping": candidates.attrs["grouping_audit"],
    }
