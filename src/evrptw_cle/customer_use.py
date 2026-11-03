"""Whole-building OSM use semantics with traceable, same-object evidence.

classify_tags(tags, building=None) -> dict with state, basis, reason.
Only the complete building object's own tags may be supplied. Separate POIs and
landuse/context objects are not merged into this dictionary.
"""

from __future__ import annotations

import json
import re

POLICY_VERSION = "osm_customer_building_use_v1_20261102"

# Start from the reviewed compute.py RES/NONRES vocabulary, retaining exact
# tokens and making mixed use explicit. annexe is deliberately unknown.
RESIDENTIAL = frozenset(
    {
        "apartments",
        "apartment",
        "bungalow",
        "cabin",
        "coach_house",
        "detached",
        "farm",
        "farmhouse",
        "ger",
        "house",
        "houseboat",
        "residential",
        "housing",
        "dwelling",
        "semidetached_house",
        "semi_detached_house",
        "static_caravan",
        "stilt_house",
        "terrace",
        "single_family",
        "multi_family",
        "multifamily",
    }
)
NONRESIDENTIAL = frozenset(
    {
        "commercial",
        "industrial",
        "office",
        "offices",
        "business",
        "retail",
        "shop",
        "shops",
        "supermarket",
        "warehouse",
        "kiosk",
        "religious",
        "cathedral",
        "chapel",
        "church",
        "mosque",
        "synagogue",
        "temple",
        "school",
        "university",
        "college",
        "kindergarten",
        "hospital",
        "clinic",
        "civic",
        "public",
        "government",
        "garage",
        "garages",
        "carport",
        "shed",
        "roof",
        "barn",
        "stable",
        "greenhouse",
        "outbuilding",
        "service",
        "parking",
        "hangar",
        "storage_tank",
        "silo",
        "transformer_tower",
        "water_tower",
        "hotel",
        "motel",
        "hostel",
        "dormitory",
        "barracks",
        "prison",
        "fire_station",
        "police",
        "police_station",
        "train_station",
        "transportation",
        "bridge",
        "stadium",
        "sports_hall",
        "sports_centre",
        "grandstand",
        "pavilion",
        "toilets",
        "cowshed",
        "farm_auxiliary",
        "car_repair",
        "storage_rental",
        "boathouse",
        "power",
        "military",
        "restaurant",
        "cafe",
        "fast_food",
        "pub",
        "bar",
        "museum",
        "library",
        "townhall",
        "courthouse",
        "community_centre",
        "nursing_home",
        "shopping_centre",
        "shopping_center",
    }
)
MIXED = frozenset(
    {
        "mixed",
        "mixed_use",
        "mixeduse",
        "residential_commercial",
        "commercial_residential",
        "residential_and_commercial",
    }
)
GENERIC = frozenset(
    {
        "",
        "yes",
        "true",
        "1",
        "no",
        "false",
        "0",
        "unknown",
        "unspecified",
        "undefined",
        "unclassified",
        "building",
    }
)
INACTIVE = frozenset({"", "no", "false", "0", "vacant", "closed", "disused", "abandoned"})
TOURISM_NONRES = frozenset(
    {
        "hotel",
        "motel",
        "hostel",
        "guest_house",
        "apartment",
        "chalet",
        "museum",
        "gallery",
        "alpine_hut",
        "wilderness_hut",
    }
)
AMENITY_NONRES = frozenset(
    {
        "hospital",
        "clinic",
        "school",
        "college",
        "university",
        "kindergarten",
        "nursing_home",
        "prison",
        "police",
        "fire_station",
        "courthouse",
        "townhall",
        "library",
        "community_centre",
        "place_of_worship",
        "parking",
        "parking_space",
        "toilets",
        "restaurant",
        "cafe",
        "fast_food",
        "pub",
        "bar",
        "bank",
        "cinema",
        "theatre",
        "fuel",
        "car_wash",
        "marketplace",
        "bus_station",
    }
)


def _text(value):
    return "" if value is None else str(value).strip().lower()


def tokens(value):
    """Semicolon plus common explicit multi-value punctuation; no substrings."""
    raw = _text(value)
    return tuple(
        sorted({re.sub(r"[\s-]+", "_", t.strip()) for t in re.split(r"[;,|/]+", raw) if t.strip()})
    )


def classify_value(value):
    terms = set(tokens(value))
    res = terms & RESIDENTIAL
    nonres = terms & NONRESIDENTIAL
    unknown = terms - RESIDENTIAL - NONRESIDENTIAL - MIXED - GENERIC
    if terms & MIXED or (res and nonres):
        state = "mixed"
    elif res and not unknown:
        state = "residential"
    elif nonres and not unknown:
        state = "nonresidential"
    else:
        state = "unknown"
    return {
        "state": state,
        "tokens": sorted(terms),
        "residential_tokens": sorted(res),
        "nonresidential_tokens": sorted(nonres),
        "unrecognized_tokens": sorted(unknown),
    }


def _functional_nonres(tags):
    """Current use on this very full-building polygon, never contained POIs."""
    evidence = {}
    for key in ("shop", "office"):
        value = _text(tags.get(key))
        if value and value not in INACTIVE:
            evidence[key] = tags[key]
    for key, vocabulary in (("tourism", TOURISM_NONRES), ("amenity", AMENITY_NONRES)):
        if set(tokens(tags.get(key))) & vocabulary:
            evidence[key] = tags[key]
    return evidence


def classify_tags(tags, building=None):
    """Return residential/nonresidential/mixed/unknown plus traceable evidence.

    building:use overrides building typology. Explicit residential use plus
    business/facility tags on the SAME object is mixed. Building parts return
    unknown for whole-building eligibility; their own class is context only.
    """
    if isinstance(tags, str):
        tags = json.loads(tags)
    tags = dict(tags or {})
    building = tags.get("building") if building is None else building
    explicit = _text(tags.get("building:use"))
    actual = classify_value(explicit)
    typology = classify_value(building)
    function = _functional_nonres(tags)
    partial_use = actual["state"] == "unknown" and bool(
        actual["residential_tokens"] or actual["nonresidential_tokens"]
    )
    primary_present = bool(explicit and actual["state"] != "unknown")
    if partial_use:
        state = "mixed"
        basis = "building:use"
        reason = "unresolved_multi_value_use_with_unrecognized_tokens"
    elif primary_present:
        state = actual["state"]
        basis = "building:use"
        reason = "explicit_current_building_use"
        if actual["residential_tokens"] and function:
            state = "mixed"
            reason = "explicit_residential_use_with_same_object_nonres_function"
    else:
        state = typology["state"]
        basis = "building"
        reason = "building_typology"
        if function:
            if state == "mixed" or typology["residential_tokens"]:
                state = "mixed"
                reason = "residential_building_with_same_object_nonres_function"
            elif state == "unknown":
                state = "nonresidential"
                reason = "same_full_building_explicit_nonres_function"
            basis = "building+functional_tags"
    scope = "full_building"
    part = bool(_text(tags.get("building:part")) not in INACTIVE)
    part_state = None
    if part:
        part_state = state
        state = "unknown"
        scope = "building_part"
        basis = "building:part"
        reason = "part_semantics_do_not_establish_whole_building_use"
    return {
        "policy_version": POLICY_VERSION,
        "state": state,
        "basis": basis,
        "reason": reason,
        "scope": scope,
        "pure_nonresidential": scope == "full_building" and state == "nonresidential",
        "building": building,
        "building_use": tags.get("building:use"),
        "building_value_class": typology["state"],
        "building_use_value_class": actual["state"],
        "building_tokens": typology["tokens"],
        "building_use_tokens": actual["tokens"],
        "unrecognized_tokens": sorted(
            set(typology["unrecognized_tokens"] + actual["unrecognized_tokens"])
        ),
        "same_object_nonres_functional_tags": function,
        "part_state_context_only": part_state,
    }


def merge_states(states):
    """Merge WHOLE, same-entity evidence only; unknown is absence of evidence.

    The caller must establish geometry scope. Arbitrary overlapping neighbors,
    POIs, building parts and landuse areas must not be fed to this function.
    """
    states = set(states) - {"unknown"}
    if "mixed" in states or {"residential", "nonresidential"} <= states:
        return "mixed"
    if "nonresidential" in states:
        return "nonresidential"
    if "residential" in states:
        return "residential"
    return "unknown"
