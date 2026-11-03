"""Bind canonical building-with-units records to the common Stage 1 road context."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import shapely

from .customer_enrichment import customer_eligible, positive_integer, within_profile
from .local_road_access import LocalAccessContext
from .util import sha256_file, write_json

SCHEMA = "evrptw_building_customer_layer_v1"
POLICY = "canonical_building_field_provenance_road_bound_v1"


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parquet_frame(frame):
    result = frame.copy()
    encoded = []
    for column in result.columns:
        if column == result.geometry.name:
            continue
        structured = (
            result[column]
            .map(lambda value: isinstance(value, (dict, list, tuple, np.ndarray)))
            .any()
        )
        cached_spatial_json = column in {"directed_edge_refs", "directed_projection_offsets"}
        if structured or cached_spatial_json:
            result[column] = result[column].map(
                lambda value: (
                    value
                    if isinstance(value, str)
                    else json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
                )
            )
            encoded.append(column)
    return result, encoded


def _information_reason(record, profile, include_model_imputed):
    reasons = []
    if not record.get("matched_ms_osm", True) or not record.get("geometry_clear", True):
        reasons.append("building_geometry_unresolved")
    if not record.get("service_domain_eligible", True):
        reasons.append("outside_service_boundary")
    use = record.get("residential_use", "unknown")
    if use != "residential":
        reasons.append("residential_use_" + str(use))
    if str(record.get("purpose_review_status", "")).startswith("pending"):
        reasons.append("purpose_review_pending")
    if record.get("units_scope") != "building":
        reasons.append("units_not_single_building_scope")
    if positive_integer(record.get("units_final")) is None:
        reasons.append("units_missing_or_invalid")
    elif not within_profile(record.get("units_final"), profile):
        reasons.append("units_outside_selected_profile")
    if str(record.get("units_review_status", "")).startswith("pending"):
        reasons.append("units_review_pending")
    if not include_model_imputed and record.get("units_method") in {"model_imputed", "neighbor_estimated"}:
        reasons.append("model_imputed_units_not_selected")
    return "|".join(reasons)


def _projection_column(name):
    """Access-generated fields only; stale source/annotation fields stay out."""
    prefixes = (
        "road_",
        "terminal_",
        "access_",
        "connector_",
        "protected_",
        "physical_edge_",
        "directed_",
        "anchor_",
        "reference_scc_",
        "unsupported_frontage_",
        "building_access_",
        "service_access_",
    )
    return name.startswith(prefixes) or name in {
        "edge_u",
        "edge_v",
        "edge_key",
        "highway",
        "oneway",
        "legal_access_tier",
        "motor_access_allowed",
        "source_way_motor_allowed",
        "node_access_blocked",
        "sensitive_interior_road",
        "native_area_access_prohibited",
        "source_oneway",
        "direction_status",
        "direction_resolved",
        "requires_terminal_paths",
        "local_access_confirmed",
        "local_alternative_selected",
        "initial_access_review_reason",
        "delivery_authorization_assumed",
        "owner_permission_observed",
        "frontage_policy",
    }


def _anchor_changed_sites(sites, context_dir, obstacles, projection_cache):
    """Combine validated previous spatial results with newly projected rows.

    The caller binds a cache to the same roads, obstacles and spatial generation
    implementation. This layer additionally requires the same ID and exact WKB.
    Every nonspatial value is taken from the new sites, never from the cache.
    """
    reused = np.zeros(len(sites), dtype=bool)
    cached_positions = np.full(len(sites), -1, dtype=np.int64)
    cache = None
    if projection_cache is not None and len(projection_cache):
        cache = projection_cache.to_crs(4326).reset_index(drop=True)
        if cache.site_id.isna().any() or cache.site_id.duplicated().any():
            raise ValueError("Projection cache requires unique site_id values")
        cached_positions = pd.Index(cache.site_id).get_indexer(sites.site_id)
        candidates = np.flatnonzero(cached_positions >= 0)
        if len(candidates):
            current_wkb = shapely.to_wkb(sites.geometry.array.take(candidates))
            prior_wkb = shapely.to_wkb(cache.geometry.array.take(cached_positions[candidates]))
            reused[candidates] = current_wkb == prior_wkb
    parts = []
    if reused.any():
        columns = [name for name in cache.columns if name not in sites and _projection_column(name)]
        if "protected_roundtrip_eligible" not in columns or "access_review_reason" not in columns:
            raise ValueError("Projection cache lacks generated access eligibility fields")
        parts.append(pd.DataFrame(cache.iloc[cached_positions[reused]][["site_id", *columns]]))
    needs_projection = np.flatnonzero(~reused)
    context_manifest = Path(context_dir) / "access_context_manifest.json"
    road_policy = json.loads(context_manifest.read_text()).get("policy", "unknown")
    if len(needs_projection):
        context = LocalAccessContext.load(context_dir)
        projected = context.anchor(
            sites.iloc[needs_projection].reset_index(drop=True), masks=None, footprints=obstacles
        )
        columns = [
            name for name in projected.columns if name not in sites and _projection_column(name)
        ]
        parts.append(pd.DataFrame(projected[["site_id", *columns]]))
        road_policy = getattr(context, "policy", road_policy)
    if parts:
        spatial = pd.concat(parts, ignore_index=True).set_index("site_id")
        if len(spatial) != len(sites) or not sites.site_id.isin(spatial.index).all():
            raise ValueError("Projection results must cover every current site exactly once")
        anchored = sites.join(spatial, on="site_id", how="left", validate="one_to_one")
    else:
        anchored = sites.copy()
        anchored["protected_roundtrip_eligible"] = pd.Series(dtype=bool)
        anchored["access_review_reason"] = pd.Series(dtype=str)
    anchored["customer_release_eligible"] = False
    return anchored, int(reused.sum()), len(needs_projection), road_policy


def _copy_lineage(city_dir, destination):
    inputs, outputs, requests = {}, [], {}
    source = Path(city_dir) / "customer_inputs"
    finalization = source / "finalization_manifest.json"
    deterministic = finalization.exists() and json.loads(finalization.read_text()).get("schema") == "building_customer_field_lineage_v2"
    files = [("finalization_manifest.json", "customer_lineage.json")]
    if deterministic:
        files.extend((name, name) for name in ["unit_preparation.json", "unit_coverage.json", "unit_matching.json"])
    else:
        files.append(("assessments.jsonl", "annotation_records.jsonl"))
    for original, name in files:
        path = source / original
        if not path.exists():
            continue
        shutil.copy2(path, destination / name)
        inputs[original] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        outputs.append(name)
    if deterministic:
        for name, item in json.loads(finalization.read_text()).get("unit_model", {}).items():
            path = Path(item["path"])
            relative = "unit_model/" + name
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            inputs[relative] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
            outputs.append(relative)
    annotations = source / "assessments.jsonl"
    if annotations.exists() and not deterministic:
        with annotations.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                for run in record.get("ai_runs", []):
                    if not run.get("record_path"):
                        continue
                    path = Path(run["record_path"])
                    if str(path) in requests:
                        continue
                    if not path.is_file():
                        continue
                    digest = sha256_file(path)
                    relative = "ai_requests/" + digest + ".json"
                    target = destination / relative
                    if not target.exists():
                        target.parent.mkdir(exist_ok=True)
                        shutil.copy2(path, target)
                        outputs.append(relative)
                    requests[str(path)] = {
                        "path": relative,
                        "sha256": digest,
                        "request_id": run.get("request_id"),
                        "model": run.get("model"),
                    }
    return inputs, outputs, requests


def build_building_customer_layer(
    *,
    buildings,
    context_dir,
    output_dir,
    city_slug,
    area_crs,
    profile="unrestricted",
    include_model_imputed=True,
    obstacle_buildings=None,
    projection_cache=None,
):
    """Write a building-level customer component under ``output_dir/customers``.

    ``buildings`` holds the finalized per-field enrichment records and their
    unchanged canonical polygons. The context's directed access rules are reused;
    this adapter adds no facility-category mask or apartment-complex inference.
    All records remain in the audit layer, including unknown use or units.
    """
    destination = Path(output_dir) / "customers"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "residential_site_manifest.json").exists():
        raise FileExistsError("Build this customer layer in a new output version")
    sites = buildings.copy().to_crs(4326).reset_index(drop=True)
    if sites.entity_id.isna().any() or sites.entity_id.duplicated().any():
        raise ValueError("Canonical buildings require unique nonempty entity_id values")
    sites["site_id"] = city_slug + ":" + sites.entity_id.astype(str)
    sites["latent_service_location_id"] = sites.site_id
    sites["site_source_key"] = sites.entity_id.astype(str)
    sites["site_identity_source"] = "microsoft_osm_canonical_building"
    sites["service_location_type"] = "residential_building"
    sites["active_customer"] = False
    sites["access_search_extent_wkt"] = sites.geometry.to_wkt()
    points = sites.geometry.representative_point()
    sites["representative_longitude"] = points.x
    sites["representative_latitude"] = points.y
    sites["location_lon"] = points.x
    sites["location_lat"] = points.y
    sites["residential_units"] = pd.array(
        [positive_integer(value) for value in sites.units_final], dtype="Float64"
    )
    records = sites.drop(columns="geometry").to_dict("records")
    sites["customer_information_eligible"] = [
        customer_eligible(record, profile, include_model_imputed=include_model_imputed)
        for record in records
    ]
    sites["customer_information_reason"] = [
        _information_reason(record, profile, include_model_imputed) for record in records
    ]
    sites["site_scope_unresolved"] = (
        sites.units_scope.ne("building") | sites.residential_units.isna()
    )
    pending_units = (
        sites.get("units_review_status", pd.Series("", index=sites.index))
        .fillna("")
        .astype(str)
        .str.startswith("pending")
    )
    sites["site_scope_review_required"] = sites.site_scope_unresolved | pending_units
    sites["site_scope_review_reason"] = np.where(
        sites.site_scope_review_required, sites.customer_information_reason, ""
    )
    context_path = Path(context_dir) / "access_context_manifest.json"
    # Complete polygons control new projections. Annotation-only changes keep
    # their exact prior spatial result, with all present eligibility recomputed.
    obstacles = sites if obstacle_buildings is None else obstacle_buildings
    anchored, reused_rows, projected_rows, road_policy = _anchor_changed_sites(
        sites, context_dir, obstacles, projection_cache
    )
    anchored["geometry_rule_candidate_eligible"] = [
        bool(record.get("matched_ms_osm", True) and record.get("geometry_clear", True))
        for record in records
    ]
    anchored["geometry_core_eligible"] = anchored.geometry_rule_candidate_eligible
    anchored["road_access_candidate_eligible"] = anchored.protected_roundtrip_eligible.astype(bool)
    anchored["cle_candidate_eligible"] = (
        anchored.customer_information_eligible & anchored.road_access_candidate_eligible
    )
    anchored["cle_default_instance_eligible"] = anchored.cle_candidate_eligible
    anchored["quarantine_reason"] = [
        "|".join(filter(None, [str(info), str(road)]))
        for info, road in zip(
            anchored.customer_information_reason,
            anchored.access_review_reason.fillna(""),
            strict=True,
        )
    ]
    anchored.loc[anchored.cle_candidate_eligible, "quarantine_reason"] = ""
    stored_sites, json_columns = _parquet_frame(sites)
    stored_anchors, anchor_json_columns = _parquet_frame(anchored)
    stored_sites.to_parquet(destination / "residential_sites.parquet", index=False)
    stored_anchors.to_parquet(destination / "latent_locations.parquet", index=False)
    lineage_inputs, lineage_outputs, requests = _copy_lineage(output_dir, destination)
    manifest = {
        "schema": SCHEMA,
        "policy": POLICY,
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "building_customer_candidates_bound_to_common_road_context",
        "area_crs": str(area_crs),
        "unit_profile": profile,
        "include_model_imputed": bool(include_model_imputed),
        "geometry": "unchanged canonical full-building polygons; polygon-boundary road projection",
        "apartment_complex_inference": False,
        "additional_facility_mask_applied": False,
        "obstacle_buildings": len(sites if obstacle_buildings is None else obstacle_buildings),
        "raw_nsi_rewritten": False,
        "default_day_or_vehicle_activation": False,
        "road_policy": road_policy,
        "inputs": {
            "local_access_context": {
                "path": str(context_path.resolve()),
                "sha256": sha256_file(context_path),
            },
            **lineage_inputs,
        },
        "summary": {
            "retained_buildings": len(sites),
            "reused_projection_rows": reused_rows,
            "projected_rows": projected_rows,
            "information_eligible_buildings": int(sites.customer_information_eligible.sum()),
            "road_available_buildings": int(anchored.road_access_candidate_eligible.sum()),
            "cle_available_buildings": int(anchored.cle_candidate_eligible.sum()),
            "residential_sources": dict(Counter(sites.residential_source)),
            "units_methods": dict(Counter(sites.units_method)),
            "units_sources": dict(Counter(sites.units_source)) if "units_source" in sites else {},
        },
        "json_encoded_columns": {
            "residential_sites.parquet": json_columns,
            "latent_locations.parquet": anchor_json_columns,
        },
        "outputs": {
            name: sha256_file(destination / name)
            for name in ["residential_sites.parquet", "latent_locations.parquet", *lineage_outputs]
        },
        "ai_request_records": requests,
        "generation_dependencies": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in [
                "customer_building_layer.py",
                "customer_enrichment.py",
                "local_road_access.py",
                "customer_access.py",
            ]
        },
    }
    write_json(destination / "residential_site_manifest.json", manifest)
    return manifest
