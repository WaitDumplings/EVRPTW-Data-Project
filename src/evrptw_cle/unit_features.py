"""Rebuild the fixed building cohort and original unit-reference features.

Prepared canonical geometry, native Microsoft/OSM attributes and the original
positive residential NSI ledger remain separate inputs.  Matching all native
NSI records supplies zero-unit evidence only; it does not replace that ledger.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely

from .building_inputs import source_osm_blockers
from .customer_geometry import explicit_height_m
from .customer_matching import match_sources


AXIS_R = 0.2853432473555575
HOUSING_KEYS = ("building:flats", "building:units", "residential:units")
ROOM_KEYS = ("rooms", "tourism:rooms", "building:rooms")
HOUSE_TYPES = {"house", "detached", "semidetached_house", "terrace", "bungalow",
               "cabin", "static_caravan", "mobile_home"}
APARTMENT_TYPES = {"apartments", "apartment", "multi_family", "multifamily"}
OSM_FLAGS = [
    "apartment_tag", "building_apartment_tag", "building_use_apartment_tag",
    "residential_apartment_tag", "hotel_tag", "tourism_apartment_tag",
    "housing_tag_present", "housing_positive_integer", "housing_zero_reported",
    "housing_count_conflict", "room_tag_present", "room_positive_integer",
    "height_present", "height_available", "levels_present", "levels_available",
    "height_or_levels_available", "street_address", "address_unit_identifier",
    "name_present", "source_microsoft", "lacounty_ain",
    "apartment_with_housing_count", "hotel_with_room_count", "hotel_with_housing_count",
] + [key.replace(":", "_") + "_positive_integer" for key in HOUSING_KEYS]


def _number(value, integer=False):
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number) or number < 0 or (integer and number != int(number)):
        return None
    return int(number) if integer else number


def _tokens(value):
    return set(re.split(r"[;,|]", str(value or "").strip().lower()))


def _tag_flags(tags, building=None):
    building = _tokens(tags.get("building", building))
    use = _tokens(tags.get("building:use"))
    residential = _tokens(tags.get("residential"))
    tourism = _tokens(tags.get("tourism"))
    flags = {
        "building_apartment_tag": bool(building & APARTMENT_TYPES),
        "building_use_apartment_tag": bool(use & APARTMENT_TYPES),
        "residential_apartment_tag": bool(residential & {"apartments", "apartment"}),
        "hotel_tag": bool(building & {"hotel", "motel", "hostel", "guest_house"}
                          or tourism & {"hotel", "motel", "hostel", "guest_house"}),
        "tourism_apartment_tag": "apartment" in tourism,
    }
    flags["apartment_tag"] = (flags["building_apartment_tag"] or
                              flags["building_use_apartment_tag"] or
                              flags["residential_apartment_tag"])
    flags["height_present"] = bool(tags.get("height"))
    flags["height_available"] = bool(np.isfinite(explicit_height_m(tags.get("height"))))
    flags["levels_present"] = bool(tags.get("building:levels"))
    levels = _number(tags.get("building:levels"))
    flags["levels_available"] = levels is not None and levels > 0
    flags["street_address"] = bool(tags.get("addr:housenumber") and
                                   (tags.get("addr:street") or tags.get("addr:place")))
    flags["address_unit_identifier"] = bool(tags.get("addr:unit"))
    flags["name_present"] = bool(tags.get("name"))
    flags["source_microsoft"] = "microsoft" in str(tags.get("source", "")).lower()
    flags["lacounty_ain"] = bool(tags.get("lacounty:ain"))
    flags["housing_tag_present"] = any(tags.get(key) not in (None, "") for key in HOUSING_KEYS)
    flags["room_tag_present"] = any(tags.get(key) not in (None, "") for key in ROOM_KEYS)
    flags["room_positive_integer"] = any((_number(tags.get(key), True) or 0) > 0 for key in ROOM_KEYS)
    units = []
    for key in HOUSING_KEYS:
        value = _number(tags.get(key), True)
        flags[key.replace(":", "_") + "_positive_integer"] = value is not None and value > 0
        if value is not None:
            units.append(value)
    flags["housing_positive_integer"] = any(value > 0 for value in units)
    flags["housing_zero_reported"] = 0 in units
    return flags, units, levels


def load_paths(base, sources):
    """Restore prepared path polygons from their native MS/OSM identifiers."""
    base = Path(base)
    registry = pd.read_parquet(base / "geometry/path_registry.parquet")
    microsoft = pd.read_parquet(sources["microsoft"], columns=["building_id", "geometry"])
    osm = pd.read_parquet(sources["osm"], columns=["osm_id", "geometry"])
    ms_geometry = registry.path_id.map(microsoft.set_index("building_id").geometry)
    osm_geometry = registry.path_id.map(osm.set_index("osm_id").geometry)
    geometry = ms_geometry.where(registry.path_kind.eq("ms_primary"), osm_geometry)
    if geometry.isna().any():
        raise ValueError("Some prepared geometric paths have no original polygon")
    return gpd.GeoDataFrame(registry, geometry=shapely.from_wkb(geometry.to_numpy()), crs=4326)


def _full_osm(base, sources):
    registry = pd.read_parquet(Path(base) / "geometry/full_osm_registry.parquet")
    osm = pd.read_parquet(sources["osm"], columns=["osm_id", "geometry"])
    geometry = registry.canonical_id.map(osm.set_index("osm_id").geometry)
    return gpd.GeoDataFrame(registry, geometry=shapely.from_wkb(geometry.to_numpy()), crs=4326)


def geometry_context(base, sources):
    return load_paths(base, sources), _full_osm(base, sources)


def full_osm_containment(records, base, sources):
    """Attach complete original OSM-building containment without moving NSI."""
    full = _full_osm(base, sources)
    polygons = full.geometry.to_numpy()
    points = shapely.points(records.native_lon.to_numpy(), records.native_lat.to_numpy())
    source_index, polygon_index = shapely.STRtree(polygons).query(points, predicate="covered_by")
    count = np.bincount(source_index, minlength=len(records))
    sole = np.full(len(records), -1, dtype=np.int64)
    sole[source_index] = polygon_index
    result = records.copy()
    result["full_osm_count"] = count
    result["full_osm_owner_entity"] = -1
    result["full_osm_state"] = ""
    result["full_osm_owner_entities"] = ""
    is_nonres = full.state.isin(["nonresidential", "mixed"]).to_numpy()
    nonres_count = np.bincount(source_index, weights=is_nonres[polygon_index], minlength=len(records))
    result["full_osm_all_nonres"] = (count > 0) & (nonres_count == count)
    one = count == 1
    result.loc[one, "full_osm_owner_entity"] = full.osm_to_entity.to_numpy()[sole[one]]
    result.loc[one, "full_osm_state"] = full.state.to_numpy()[sole[one]]
    multiple = count[source_index] > 1
    if multiple.any():
        members = pd.DataFrame({"record_pos": source_index[multiple],
                                "owner": full.osm_to_entity.to_numpy()[polygon_index[multiple]]})
        members = members.loc[members.owner.ge(0)]
        choices = members.groupby("record_pos").owner.agg(
            lambda values: "|".join(str(int(value)) for value in sorted(values.unique())))
        result.loc[result.index[choices.index], "full_osm_owner_entities"] = choices.to_numpy()
    return result


def native_unit_dispositions(frame, records, base, sources, axis_r=AXIS_R):
    """Classify the geometry-clear service cohort using all-native zero evidence."""
    paths, full = geometry_context(base, sources)
    native = gpd.read_parquet(sources["nsi"]).sort_values("fd_id").reset_index(drop=True)
    matching, evidence, diagnostics = match_sources(paths, native, sources["area_crs"], axis_r)
    del evidence, paths
    _, blockers = source_osm_blockers(native.to_crs(sources["area_crs"]).geometry.array, matching,
        full.to_crs(sources["area_crs"]).geometry.array, full.state.to_numpy(),
        full.osm_to_entity.to_numpy(np.int64))
    values = pd.to_numeric(native.resunits, errors="coerce").to_numpy(float)
    owners = matching.owner.to_numpy(np.int64)
    assigned = owners >= 0
    zero = np.isfinite(values) & (values == 0)
    positive = np.isfinite(values) & (values > 0)
    counter = (matching.source_geometry_counterevidence.to_numpy(bool) & assigned &
               (owners == matching.base_direct_owner.to_numpy(np.int64)))
    clear = (assigned & ~matching.source_group_conflict.to_numpy(bool) &
             ~matching.source_internal_conflict.to_numpy(bool) & ~counter & (blockers == ""))
    group_positive = pd.Series(positive).groupby(matching.source_group_id).transform("any").to_numpy(bool)
    positive_owner = set(owners[assigned & positive])
    positive_owner.update(int(value) for value in records.owner if value >= 0)
    any_owner = set(owners[assigned])
    zero_owner = set(owners[assigned & zero])
    clear_zero_owner = set(owners[clear & zero & ~group_positive])
    zero_only = clear_zero_owner - positive_owner
    nsi1 = (frame.residential_use.eq("unknown") & frame.nsi_count.eq(1) &
            frame.nsi_attribution_clear.fillna(False).astype(bool) &
            frame.units_raw_nsi.eq(1) & frame.units_scope.eq("building"))
    residential = frame.residential_use.eq("residential") | nsi1
    unknown = frame.residential_use.eq("unknown") & ~nsi1
    explicit_zero = unknown & frame.entity_index.isin(zero_only)
    conflict = unknown & frame.entity_index.isin(zero_owner & positive_owner)
    pending = unknown & frame.entity_index.isin(zero_owner) & ~explicit_zero & ~conflict
    any_positive = unknown & frame.entity_index.isin(positive_owner)
    no_native = unknown & ~frame.entity_index.isin(any_owner)
    status = np.full(len(frame), "outside_scenario", dtype=object)
    for mask, value in [(residential, "residential_after_nsi1"), (unknown, "unknown_retained"),
                        (no_native, "unknown_no_nsi_match"), (any_positive, "unknown_positive_nsi"),
                        (pending, "unknown_zero_attribution_pending"), (conflict, "unknown_zero_positive_conflict"),
                        (explicit_zero, "excluded_explicit_zero_only")]:
        status[mask.to_numpy(bool)] = value
    result = pd.DataFrame({"entity_id": frame.entity_id.to_numpy(), "scenario_status": status})
    counts = {"native_nsi_records": len(native), "native_nsi_zero_records": int(zero.sum()),
              "clear_service_buildings": len(frame),
              "residential_after_nsi1": int(residential.sum()),
              "unknown_before_zero_exclusion": int(unknown.sum()),
              "unknown_explicit_zero_excluded": int(explicit_zero.sum()),
              "unknown_after_zero_exclusion": int((unknown & ~explicit_zero).sum()),
              "unknown_zero_positive_conflict_retained": int(conflict.sum()),
              "unknown_zero_attribution_pending_retained": int(pending.sum())}
    return result, {"counts": counts, "native_match_diagnostics": diagnostics}


def osm_reported_features(frame, base, sources):
    """Collect reported attributes from each canonical building's OSM aliases."""
    registry = pd.read_parquet(Path(base) / "geometry/full_osm_registry.parquet",
                               columns=["osm_group", "osm_ids", "building"]).set_index("osm_group")
    groups = np.where(frame.kind.eq("ms"), frame.trusted_osm_alias_group, frame.osm_group).astype(np.int64)
    member = {}
    group_members = registry.osm_ids.to_dict()
    for position, group in enumerate(groups):
        for identifier in str(group_members[group]).split("|"):
            member[identifier] = position
    native = pd.read_parquet(sources["osm"], columns=["osm_id", "building", "tags"])
    native["position"] = native.osm_id.map(member)
    native = native.loc[native.position.notna()]
    flags = {name: np.zeros(len(frame), dtype=bool) for name in OSM_FLAGS}
    unit_values, level_values, height_values = {}, {}, {}
    for _, building, raw_tags, position in native.itertuples(index=False, name=None):
        position = int(position)
        tags = (raw_tags if isinstance(raw_tags, dict) else
                json.loads(raw_tags) if isinstance(raw_tags, str) else {})
        attributes, units, levels = _tag_flags(tags, building)
        for name, value in attributes.items():
            if value:
                flags[name][position] = True
        if units:
            unit_values.setdefault(position, set()).update(units)
        if levels is not None and levels > 0:
            level_values.setdefault(position, set()).add(levels)
        height = explicit_height_m(tags.get("height"))
        if np.isfinite(height):
            height_values.setdefault(position, set()).add(height)
    for position, values in unit_values.items():
        flags["housing_count_conflict"][position] = len(values) > 1
    flags["height_or_levels_available"] = flags["height_available"] | flags["levels_available"]
    flags["apartment_with_housing_count"] = flags["apartment_tag"] & flags["housing_positive_integer"]
    flags["hotel_with_room_count"] = flags["hotel_tag"] & flags["room_positive_integer"]
    flags["hotel_with_housing_count"] = flags["hotel_tag"] & flags["housing_positive_integer"]
    result = pd.DataFrame({"entity_id": frame.entity_id.to_numpy(), **flags})
    for column, values in [("reported_housing_units", unit_values), ("reported_levels", level_values),
                           ("reported_height_m", height_values)]:
        numeric = np.full(len(frame), np.nan)
        for position, distinct in values.items():
            if len(distinct) == 1:
                numeric[position] = next(iter(distinct))
        result[column] = numeric
    typology = pd.Series(groups).map(registry.building).fillna("").str.lower()
    house = typology.map(lambda value: bool(_tokens(value) & HOUSE_TYPES)).to_numpy(bool)
    result["use_type"] = np.where(flags["apartment_tag"], "apartments", np.where(house, "house", "generic"))
    result["osm_building_type"] = typology.to_numpy()
    return result


def prepare_features(*, city, sources, base, axis_r=AXIS_R, config=None):
    """Build one fixed-cohort row per building from prepared and native inputs.

    ``base`` is the prepared ``customer_inputs`` directory.  The denominator
    includes geometry-clear, in-service MS/OSM matched buildings classified as
    residential or still unknown, after excluding clear unknown zero-only NSI
    evidence.  Conflicting unknown evidence remains in the denominator and is
    marked separately.  Original source matching and capacity screening select
    reference rows in the caller; this function imposes no household-count cap.
    """
    base = Path(base)
    original = pd.read_parquet(base / "population.parquet")
    scope = pd.read_parquet(base / "scope.parquet")
    frame = original.merge(scope, on="entity_id", how="left", validate="one_to_one")
    keep = (frame.matched_ms_osm.fillna(False).astype(bool) & frame.geometry_clear.fillna(False).astype(bool) &
            frame.service_domain_eligible.fillna(False).astype(bool))
    frame = frame.loc[keep].reset_index(drop=True)
    records = pd.read_parquet(base / "source_records.parquet")
    disposition, audit = native_unit_dispositions(frame, records, base, sources, axis_r)
    osm = osm_reported_features(frame, base, sources)
    frame = frame.merge(disposition, on="entity_id", how="left", validate="one_to_one")
    retained = (frame.scenario_status.eq("residential_after_nsi1") |
                frame.scenario_status.str.startswith("unknown_"))
    frame = frame.loc[retained].reset_index(drop=True)
    frame = frame.merge(osm, on="entity_id", how="left", validate="one_to_one")
    transform = Transformer.from_crs("EPSG:4326", sources["area_crs"], always_xy=True)
    frame["x"], frame["y"] = transform.transform(frame.canonical_longitude.to_numpy(float),
                                                   frame.canonical_latitude.to_numpy(float))
    frame["levels"] = pd.to_numeric(frame.reported_levels, errors="coerce")
    frame["height_osm_m"] = pd.to_numeric(frame.reported_height_m, errors="coerce")
    frame["height_m"] = pd.to_numeric(frame.height_m, errors="coerce")
    frame["height_original_m"] = frame.height_m
    frame["spatial_block"] = (np.floor(frame.x / 2000).astype("Int64").astype(str) + ":" +
                               np.floor(frame.y / 2000).astype("Int64").astype(str))
    owned = records.loc[records.owner.ge(0)].copy()
    aggregation = owned.groupby("owner").agg(
        source_record_count=("fd_id", "size"), raw_record_max=("resunits", "max"),
        raw_record_sum=("resunits", "sum"), source_internal_conflict=("source_internal_conflict", "max"),
        source_group_conflict=("source_group_conflict", "max"),
        source_scope_conflict=("final_source_geometry_units_scope_conflict", "max"))
    frame = frame.join(aggregation, on="entity_index")
    for name in ["source_internal_conflict", "source_group_conflict", "source_scope_conflict"]:
        frame[name] = frame[name].fillna(False).astype(bool)
    first = owned.drop_duplicates("owner", keep="first").set_index("owner")
    source_group = first.source_group_id.where(
        first.genuine_source_key.fillna(False).astype(bool) & first.source_group_id.fillna("").ne(""),
        "fd:" + first.fd_id.astype(str))
    frame["source_group"] = frame.entity_index.map(source_group).fillna("unmatched:" + frame.entity_id)
    frame["source_group"] = str(city) + ":" + frame.source_group.astype(str)
    frame["fd_id"] = frame.entity_index.map(first.fd_id).astype("Int64")
    raw = pd.to_numeric(frame.units_raw_nsi, errors="coerce").to_numpy(float)
    positive = np.isfinite(raw) & (raw > 0) & (raw == np.floor(raw))
    frame["clear_single"] = (frame.nsi_count.eq(1) & frame.nsi_attribution_clear.fillna(False).astype(bool) &
                             frame.units_scope.eq("building") & positive)
    frame["units"] = np.where(frame.clear_single, raw, np.nan)
    frame["baseline_residential"] = frame.scenario_status.eq("residential_after_nsi1")
    frame["unknown_positive_only"] = frame.scenario_status.eq("unknown_positive_nsi")
    frame["native_conflict_status"] = frame.scenario_status.isin(
        ["unknown_zero_positive_conflict", "unknown_zero_attribution_pending"])
    frame["baseline_units_missing"] = frame.baseline_residential & ~frame.clear_single
    frame["clean_direct"] = (frame.clear_single & ~frame.source_internal_conflict &
                              ~frame.source_group_conflict & ~frame.source_scope_conflict & ~frame.native_conflict_status)
    frame["newly_residential_source"] = ~frame.baseline_residential & frame.unknown_positive_only & frame.clean_direct
    frame["residential_supported"] = frame.baseline_residential | frame.newly_residential_source
    frame["source_retain_candidate"] = frame.residential_supported & frame.clean_direct
    frame["valid_features"] = (np.isfinite(frame.x) & np.isfinite(frame.y) &
                                np.isfinite(frame.area_m2) & frame.area_m2.gt(0))
    frame["reference_candidate"] = frame.source_retain_candidate & frame.valid_features
    frame["units_missing_target"] = (frame.residential_supported & ~frame.source_retain_candidate &
                                      ~frame.native_conflict_status & frame.valid_features)
    frame["city"] = str(city)
    frame.attrs["feature_preparation"] = audit
    return frame
