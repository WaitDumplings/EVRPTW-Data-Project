"""Independent validation for building-first Stage 1 CLEs.

This checks saved semantic/unit lineage and directed access-cost arithmetic. It
does not apply the historical apartment-complex or facility-mask contracts.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd

from .util import sha256_file

SCHEMA = "evrptw_building_first_cle_v1"


def _list(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, list) else []


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number >= 1 and number.is_integer() else None


def _web_building_evidence(value):
    for item in _list(value):
        url = urlparse(str(item.get("url") or ""))
        if (url.scheme in {"http", "https"} and url.netloc and item.get("quote")
                and item.get("scope") == "building"
                and (item.get("recorded_at_utc") or item.get("retrieved_at")
                     or item.get("accessed_at"))):
            return True
    return False


def validate_building_stage1(city_dir, *, verify_hashes=True):
    """Return ``passed/errors/summary`` from independently reading saved layers."""
    base = Path(city_dir)
    errors = []

    def check(name, condition, detail=None):
        if not condition:
            errors.append({"check": name, **({"detail": str(detail)} if detail else {})})

    required = {
        "manifest": "cle_manifest.json", "canonical": "locations.parquet",
        "costs": "location_access_costs.parquet", "customer": "customers/latent_locations.parquet",
        "customer_manifest": "customers/residential_site_manifest.json",
        "depot": "depots/depots.parquet", "charging": "charging/chargers.parquet",
        "speed": "speeds/directed_speeds.parquet", "road": "roads/access_context_manifest.json",
    }
    for kind, relative in required.items():
        check("required_component:" + kind, (base / relative).is_file(), relative)
    if errors:
        return {"passed": False, "errors": errors, "summary": {"city_dir": str(base)}}
    manifest = json.loads((base / required["manifest"]).read_text())
    check("building_first_schema", manifest.get("schema") == SCHEMA)
    customer_manifest = json.loads((base / required["customer_manifest"]).read_text())
    road = json.loads((base / required["road"]).read_text())
    canonical = gpd.read_parquet(base / required["canonical"])
    customer = gpd.read_parquet(base / required["customer"])
    depot = gpd.read_parquet(base / required["depot"])
    charging = gpd.read_parquet(base / required["charging"])
    speed = pd.read_parquet(base / required["speed"])
    costs = pd.read_parquet(base / required["costs"])

    if verify_hashes:
        for name, info in manifest.get("files", {}).items():
            path = base / name
            digest = info["sha256"] if isinstance(info, dict) else info
            check("saved_file:" + name, path.is_file() and sha256_file(path) == digest)
    options = manifest.get("construction_options", {})
    for option, field in [("selection_buffer_km", "selected_buffer_km"),
                          ("allow_private_through", "ordinary_private_through_routing"),
                          ("facility_masks", "facility_category_masks")]:
        if option in options:
            check("effective_road_option:" + option, field in road and road[field] == options[option])
    context = customer_manifest.get("inputs", {}).get("local_access_context")
    if context:
        check("customer_common_road_context", context["sha256"] == sha256_file(base / required["road"]))
    if (base / "speeds/common_speed_manifest.json").exists():
        sm = json.loads((base / "speeds/common_speed_manifest.json").read_text())
        binding = sm.get("inputs", {}).get("road_context")
        if binding:
            check("speed_common_road_context", binding["sha256"] == sha256_file(base / required["road"]))

    check("canonical_ids_unique", canonical.location_id.notna().all()
          and not canonical.location_id.duplicated().any())
    check("customer_ids_unique", customer.site_id.notna().all()
          and not customer.site_id.duplicated().any())
    check("charging_ids_unique", charging.charging_location_id.notna().all()
          and not charging.charging_location_id.duplicated().any())
    native = {"customer": (customer, "site_id", "cle_default_instance_eligible"),
              "depot": (depot, "facility_group_id", "depot_candidate_eligible"),
              "charging": (charging, "charging_location_id", "charging_parameter_eligible")}
    expected_ids = set()
    summary = {}
    for kind, (table, key, gate) in native.items():
        grouped = table.groupby(key, dropna=False)[gate].any()
        ids = {kind + ":" + str(value) for value in grouped.index}
        expected_ids.update(ids)
        observed = canonical.loc[canonical.location_type.eq(kind)]
        check("canonical_native_identity:" + kind, set(observed.location_id) == ids)
        expected_flags = {kind + ":" + str(key): bool(value) for key, value in grouped.items()}
        check("canonical_native_eligibility:" + kind, all(
            bool(row.candidate_eligible) == expected_flags.get(row.location_id)
            for row in observed.itertuples()))
        summary[kind] = {"retained": len(grouped), "available": int(grouped.sum())}
    check("canonical_total_count", len(canonical) == len(expected_ids)
          and set(canonical.location_id) == expected_ids)
    customer_canonical = canonical.loc[canonical.location_type.eq("customer")].reset_index(drop=True)
    customer_by_id = customer.drop_duplicates("site_id").set_index("site_id")
    new_unit_methods = customer.units_method.isin(["nsi_representative", "neighbor_estimated"])
    new_lineage_fields = ["units_source", "selected_nsi_fd_id", "selected_nsi_resunits", "capacity_units",
                          "height_neighbor_estimated", "neighbor_reference_id"]
    for field in ["residential_units", "residential_source", "units_method", "units_scope",
                  "units_raw_nsi", "units_review_status", "units_independently_verified",
                  "purpose_review_status", *new_lineage_fields, "scale_assessed", "scale_gate_status"]:
        if new_unit_methods.any() and field in new_lineage_fields and field in customer:
            check("canonical_customer_lineage_present:" + field, field in canonical)
        if field in canonical and field in customer:
            expected = customer_by_id[field].reindex(customer_canonical.source_location_id).reset_index(drop=True)
            actual = customer_canonical[field]
            equal = (actual.isna() & expected.isna()) | actual.eq(expected)
            check("canonical_customer_lineage:" + field, bool(equal.fillna(False).all()))

    profile = customer_manifest.get("unit_profile", "unrestricted")
    include_model = customer_manifest.get("include_model_imputed", True)
    nsi_owners = {}
    unit_fields = [column for column in ["site_id", "cle_default_instance_eligible", "units_final",
        "residential_units", "units_scope", "units_method", "residential_use", "matched_ms_osm",
        "geometry_clear", "units_review_status", "purpose_review_status", "units_evidence",
        "units_independently_verified", "units_raw_nsi", "nsi_fd_ids", "nsi_source_ids", "fd_id",
        "units_source", "selected_nsi_fd_id", "selected_nsi_resunits", "capacity_units",
        "protected_roundtrip_eligible"] if column in customer]
    for values in customer[unit_fields].itertuples(index=False, name=None):
        row = dict(zip(unit_fields, values, strict=True))
        if not row["cle_default_instance_eligible"]:
            continue
        identity = str(row["site_id"])
        units = _integer(row.get("units_final"))
        scope = row.get("units_scope")
        method = row.get("units_method")
        check("customer_residential_use", row.get("residential_use") == "residential", identity)
        check("customer_building_units_scope", scope == "building", identity)
        check("customer_positive_integer_units", units is not None
              and units == _integer(row.get("residential_units")), identity)
        check("customer_geometry_correspondence", bool(row.get("matched_ms_osm", True))
              and bool(row.get("geometry_clear", True)), identity)
        check("customer_review_resolved", not str(row.get("units_review_status", "")).startswith("pending")
              and not str(row.get("purpose_review_status", "")).startswith("pending"), identity)
        if units is not None:
            allowed = (units == 1 if profile in {"u1", "eq1", "1"} else units < 50
                       if profile in {"u_lt50", "lt50"} else units < 100
                       if profile in {"u_lt100", "lt100"} else profile in {"unrestricted", "all"})
            check("customer_unit_profile", allowed, identity)
        check("customer_units_method", method in {
            "nsi_retained", "nsi_representative", "neighbor_estimated", "web_extracted", "human_verified",
            "human_corrected", "model_imputed"}, identity)
        if method in {"nsi_representative", "neighbor_estimated"}:
            check("customer_units_source", row.get("units_source") == (
                "nsi" if method == "nsi_representative" else "neighbor"), identity)
            if units is not None and units > 1:
                try:
                    capacity = float(row.get("capacity_units"))
                except (TypeError, ValueError):
                    capacity = np.nan
                check("customer_multiunit_scale", math.isfinite(capacity) and units <= capacity, identity)
        if method == "neighbor_estimated":
            verified = row.get("units_independently_verified")
            check("neighbor_units_not_independently_verified", verified is False or
                  (isinstance(verified, np.bool_) and not verified), identity)
        if method == "nsi_representative":
            check("selected_nsi_units_preserved", units == _integer(row.get("selected_nsi_resunits")), identity)
        if method == "web_extracted":
            check("web_units_have_building_source", _web_building_evidence(
                row.get("units_evidence", [])), identity)
        if method == "model_imputed":
            check("imputed_units_enabled", include_model, identity)
            check("imputed_units_not_independently_verified", not row.get(
                "units_independently_verified", False), identity)
        if method == "nsi_retained":
            check("raw_nsi_units_preserved", units == _integer(row.get("units_raw_nsi")), identity)
            for field in ["nsi_fd_ids", "nsi_source_ids", "fd_id"]:
                if field not in row or row[field] is None:
                    continue
                value = row[field]
                if isinstance(value, str):
                    parsed = json.loads(value) if value.startswith("[") else value
                    ids = parsed if isinstance(parsed, list) else [parsed]
                else:
                    ids = value if isinstance(value, list) else [value]
                for fd_id in ids:
                    if str(fd_id) not in {"nan", "None", ""}:
                        previous = nsi_owners.setdefault(str(fd_id), identity)
                        check("nsi_source_counted_once", previous == identity, str(fd_id))
                break
        check("customer_verified_directed_roundtrip", bool(row.get("protected_roundtrip_eligible", False)), identity)
    check("stage1_customers_not_activated", "active_customer" not in customer or not customer.active_customer.any())
    source_records_path = base / "customer_inputs/source_records.parquet"
    representatives = customer.loc[customer.cle_default_instance_eligible & customer.units_method.eq("nsi_representative")]
    check("nsi_representative_original_ledger_present", representatives.empty or source_records_path.exists())
    if source_records_path.exists():
        source = pd.read_parquet(source_records_path)
        check("source_nsi_records_unique", source.fd_id.notna().all() and not source.fd_id.duplicated().any())
        joined = pd.DataFrame(columns=["fd_id"])
        if "entity_index" in customer:
            counts = source.owner.value_counts()
            single = source.loc[source.owner.isin(counts.index[counts.eq(1)]) & source.owner.ge(0)]
            retained = customer.loc[customer.cle_default_instance_eligible & customer.units_method.eq("nsi_retained"),
                                    ["entity_index", "units_raw_nsi"]]
            joined = retained.merge(single, left_on="entity_index", right_on="owner", how="left")
            check("nsi_retained_has_unique_original_record", joined.fd_id.notna().all()
                  and not joined.fd_id.duplicated().any())
            check("nsi_retained_matches_original_units", np.allclose(
                pd.to_numeric(joined.units_raw_nsi, errors="coerce"), joined.raw_resunits,
                rtol=0, atol=0, equal_nan=False))
            for fd_id, owner in zip(joined.fd_id.dropna(), joined.entity_index[joined.fd_id.notna()], strict=True):
                nsi_owners[str(fd_id)] = str(owner)
        if not representatives.empty:
            from .unit_matching import source_groups

            needed = {"selected_nsi_fd_id", "selected_nsi_resunits", "units_final"}
            have_fields = needed <= set(representatives)
            check("nsi_representative_source_fields", have_fields)
            if have_fields:
                original = source.drop_duplicates("fd_id").copy()
                original["_source_group"] = source_groups(original)
                chosen = representatives[["site_id", "selected_nsi_fd_id", "selected_nsi_resunits", "units_final"]].merge(
                    original[["fd_id", "raw_resunits", "_source_group"]],
                    left_on="selected_nsi_fd_id", right_on="fd_id", how="left")
                check("nsi_representative_has_original_record", chosen.fd_id.notna().all())
                check("nsi_representative_matches_original_units", np.allclose(
                    pd.to_numeric(chosen.selected_nsi_resunits, errors="coerce"),
                    pd.to_numeric(chosen.raw_resunits, errors="coerce"), rtol=0, atol=0, equal_nan=False))
                used = pd.concat([joined.fd_id, chosen.fd_id], ignore_index=True).dropna()
                check("nsi_source_counted_once", not used.duplicated().any())
                check("nsi_source_footprint_counted_once", not chosen._source_group.dropna().duplicated().any())
                for fd_id, identity in zip(chosen.fd_id.dropna(), chosen.site_id[chosen.fd_id.notna()], strict=True):
                    nsi_owners[str(int(fd_id))] = str(identity)

    edge_columns = ["edge_u", "edge_v", "edge_key"]
    check("directed_speed_edge_identity_unique", not speed.duplicated(edge_columns).any())
    time_fields = ["legal_travel_time_s", "reference_travel_time_weekday_s", "reference_travel_time_weekend_s"]
    numeric = speed[["length_m", *time_fields]].to_numpy(float)
    check("source_edge_costs_positive_finite", bool(np.isfinite(numeric).all() and (numeric > 0).all()))
    lookup = {tuple(str(key) for key in keys): tuple(values)
              for keys, values in zip(speed[edge_columns].to_numpy(), numeric, strict=True)}
    eligible_ids = set(canonical.loc[canonical.candidate_eligible, "location_id"])
    check("eligible_access_cost_coverage", not costs.location_id.duplicated().any()
          and set(costs.location_id) == eligible_ids)
    path_errors = []
    for kind, (table, key, gate) in native.items():
        selected = canonical.loc[canonical.candidate_eligible & canonical.location_type.eq(kind),
                                 ["location_id", "source_location_id"]].copy()
        access = table
        if kind == "depot":
            access = access.sort_values([gate, "candidate_id"], ascending=[False, True], kind="stable")
        access = access.drop_duplicates(key)[[key, "road_access_distance_m",
            "terminal_inbound_path_json", "terminal_outbound_path_json"]].copy()
        access[key] = access[key].astype(str)
        paired = selected.merge(access, left_on="source_location_id", right_on=key).merge(
            costs.drop_duplicates("location_id"), on="location_id")
        columns = paired.columns.tolist()
        for values in paired.itertuples(index=False, name=None):
            row = dict(zip(columns, values, strict=True))
            identity = row["location_id"]
            connector = float(row["road_access_distance_m"])
            check("connector_distance_binding", math.isfinite(connector) and connector >= 0
                  and np.isclose(float(row["connector_length_m"]), connector), identity)
            check("connector_time_coefficient", np.isclose(float(row["connector_time_s_per_inverse_kph"]),
                                                           3.6 * connector), identity)
            for direction in ["inbound", "outbound"]:
                expected = np.zeros(4)
                for ref in _list(row["terminal_" + direction + "_path_json"]):
                    keys = tuple(str(ref[name]) for name in ["u", "v", "key"])
                    if keys not in lookup:
                        path_errors.append(identity + ":" + ":".join(keys))
                        continue
                    edge = lookup[keys]
                    partial = float(ref["length_m"])
                    fraction = partial / edge[0]
                    check("directed_path_fraction", -1e-9 <= fraction <= 1 + 1e-8, identity)
                    expected[0] += partial
                    expected[1:] += fraction * np.asarray(edge[1:])
                fields = [direction + "_terminal_length_m"] + [
                    direction + "_terminal_" + name + "_time_s"
                    for name in ["legal", "reference_weekday", "reference_weekend"]]
                actual = np.asarray([row[field] for field in fields], dtype=float)
                check("directed_access_cost_arithmetic", bool(np.isfinite(actual).all()
                      and np.allclose(actual, expected, rtol=1e-8, atol=1e-6)), identity + ":" + direction)
    check("all_directed_path_edges_speed_bound", not path_errors, path_errors[:5])

    physical_path = base / "roads/physical_roads.parquet"
    if physical_path.exists():
        physical = pd.read_parquet(physical_path)
        if {"direction_status", "directed_edge_refs"} <= set(physical):
            expected = {"explicit_reverse": {"reverse_of_physical"},
                "explicit_forward": {"same_as_physical"},
                "implied_motorway_forward": {"same_as_physical"},
                "implied_junction_forward": {"same_as_physical"},
                "explicit_bidirectional": {"same_as_physical", "reverse_of_physical"},
                "default_bidirectional": {"same_as_physical", "reverse_of_physical"},
                "unresolved_oneway_value": set()}
            bad = 0
            for row in physical.itertuples():
                orientations = {ref["geometry_orientation"] for ref in _list(row.directed_edge_refs)}
                if row.direction_status in expected and orientations != expected[row.direction_status]:
                    bad += 1
            check("physical_directed_oneway_preserved", bad == 0, bad)
    summary.update(purpose_sources=dict(Counter(customer.residential_source)),
                   units_methods=dict(Counter(customer.units_method)),
                   eligible_locations=len(eligible_ids), directed_speed_edges=len(speed),
                   units_profile=profile, nsi_owned_records=len(nsi_owners),
                   file_hashes_verified=bool(verify_hashes))
    return {"passed": not errors, "errors": errors, "summary": summary}
