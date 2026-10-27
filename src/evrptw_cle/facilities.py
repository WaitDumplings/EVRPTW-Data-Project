"""Build road-anchored charging and depot candidate layers for one city."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.strtree import STRtree

from .charger_locations import LOCATION_POLICY, collapse_charging_locations
from .charger_resolution import COORDINATE_POLICY, resolve_afdc_coordinates
from .charging_parameters import (
    CONFIG_ROOT,
    load_frozen_charging_units,
    select_reference_configurations,
)
from .customer_access import _projection_offsets_json, build_eligible_physical_edges
from .protected_connectivity import (
    build_directed_component_index,
    connectivity_summary,
    label_projection_connectivity,
)
from .public_locations import (
    PUBLIC_LOCATION_POLICY,
    filter_public_locations,
    load_exclusion_masks,
)
from .util import sha256_file, write_json


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _connector_tokens(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {
        token.strip().upper() for token in str(value).replace(",", " ").split() if token.strip()
    }


def _anchor_points_to_edges(
    points: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    area_crs: str,
) -> gpd.GeoDataFrame:
    """Project facility points to eligible physical directed-road geometries."""

    projected = points.to_crs(area_crs).reset_index(drop=True)
    edge_frame = edges.to_crs(area_crs).reset_index(drop=True)
    point_geometries = np.asarray(projected.geometry.to_numpy(), dtype=object)
    edge_geometries = np.asarray(edge_frame.geometry.to_numpy(), dtype=object)
    tree = STRtree(edge_geometries)
    nearest_indices = np.asarray(tree.nearest(point_geometries), dtype=int)
    matched_geometries = edge_geometries[nearest_indices]
    connectors = shapely.shortest_line(point_geometries, matched_geometries)
    anchors_local = shapely.get_point(connectors, -1)
    anchors_wgs84 = gpd.GeoSeries(anchors_local, crs=area_crs).to_crs("EPSG:4326")
    distances = np.asarray(shapely.distance(point_geometries, matched_geometries), dtype=float)
    matched = edge_frame.iloc[nearest_indices].reset_index(drop=True)
    physical_offsets_m = np.asarray(
        shapely.line_locate_point(matched_geometries, anchors_local), dtype=float
    )
    physical_lengths_m = np.asarray(shapely.length(matched_geometries), dtype=float)

    result = points.reset_index(drop=True).copy()
    for field in (
        "physical_edge_id",
        "directed_edge_refs",
        "directed_edge_ref_count",
        "edge_u",
        "edge_v",
        "edge_key",
        "highway",
        "road_name",
        "oneway",
    ):
        result[field] = matched[field].to_numpy()
    result["road_anchor_method"] = "facility_point_to_operational_edge_projection"
    result["road_access_distance_m"] = distances
    result["road_anchor_lon"] = anchors_wgs84.x.to_numpy(dtype=float)
    result["road_anchor_lat"] = anchors_wgs84.y.to_numpy(dtype=float)
    result["physical_edge_geometry_length_m"] = physical_lengths_m
    result["road_projection_offset_m_from_physical_start"] = physical_offsets_m
    result["road_projection_fraction_from_physical_start"] = np.divide(
        physical_offsets_m,
        physical_lengths_m,
        out=np.zeros_like(physical_offsets_m),
        where=physical_lengths_m > 0,
    ).clip(0.0, 1.0)
    result["directed_projection_offsets"] = [
        _projection_offsets_json(refs, offset, length)
        for refs, offset, length in zip(
            result["directed_edge_refs"],
            physical_offsets_m,
            physical_lengths_m,
            strict=True,
        )
    ]
    result["road_projection_node_id"] = [
        "road_projection_" + hashlib.sha1(f"{physical_id}|{fraction:.9f}".encode()).hexdigest()[:20]
        for physical_id, fraction in zip(
            result["physical_edge_id"],
            result["road_projection_fraction_from_physical_start"],
            strict=True,
        )
    ]
    result["facility_access_node_id"] = "facility_access_" + result["facility_anchor_id"].astype(
        str
    )
    result["facility_access_connector_id"] = "facility_connector_" + result[
        "facility_anchor_id"
    ].astype(str)
    result["connector_length_m"] = distances
    result["connector_bidirectional"] = True
    result["connector_speed_symmetry"] = "equal_both_directions"
    result["connector_speed_policy"] = "assigned_at_instance_generation"
    return gpd.GeoDataFrame(result, geometry="geometry", crs=points.crs)


def _filter_afdc_city(
    afdc_path: Path,
    boundary: Any,
    city_slug: str,
) -> gpd.GeoDataFrame:
    raw = pd.read_csv(afdc_path, low_memory=False, dtype={"ZIP": str})
    required = {
        "Fuel Type Code": {"ELEC"},
        "Status Code": {"E"},
        "Access Code": {"public"},
        "Country": {"US"},
    }
    for column, expected in required.items():
        observed = set(raw[column].dropna().astype(str).unique())
        if observed != expected:
            raise ValueError(
                f"AFDC snapshot is not prefiltered as declared for {column}: {observed}"
            )
    if (
        "coordinate_policy_version" not in raw
        or not raw["coordinate_policy_version"].eq(COORDINATE_POLICY).all()
    ):
        # Old resolved files must not carry same-address OSM overrides forward.
        raw = resolve_afdc_coordinates(raw)
    longitude = pd.to_numeric(raw["resolved_longitude"], errors="coerce")
    latitude = pd.to_numeric(raw["resolved_latitude"], errors="coerce")
    points = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    )
    selected = points.loc[points.geometry.covered_by(boundary)].copy().reset_index(drop=True)
    selected["city_slug"] = city_slug
    return selected


def _normalize_chargers(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.copy()
    result["charger_id"] = "afdc_" + result["ID"].astype(int).astype(str)
    result["l1_ports"] = _number(result["EV Level1 EVSE Num"])
    result["l2_ports"] = _number(result["EV Level2 EVSE Num"])
    result["dc_fast_ports"] = _number(result["EV DC Fast Count"])
    result["restricted_public"] = result["Restricted Access"].map(_bool_or_none)
    connector_sets = result["EV Connector Types"].map(_connector_tokens)
    result["has_ccs1"] = connector_sets.map(lambda values: "J1772COMBO" in values)
    result["has_j1772"] = connector_sets.map(lambda values: "J1772" in values)
    result["has_nacs"] = connector_sets.map(lambda values: "TESLA" in values)
    result["has_chademo"] = connector_sets.map(lambda values: "CHADEMO" in values)
    result["reference_charge_mode"] = np.select(
        [
            result["dc_fast_ports"] > 0,
            result["l2_ports"] > 0,
        ],
        ["dc_fast", "ac_level2"],
        default="unsupported_or_unresolved",
    )
    result["reference_vehicle_connector_compatibility"] = np.select(
        [result["has_ccs1"], result["has_j1772"], result["has_nacs"]],
        ["ccs1_present", "j1772_present", "nacs_only_or_present"],
        default="unresolved",
    )
    maximum_class = result["Maximum Vehicle Class"].fillna("").astype(str).str.upper()
    result["medium_duty_vehicle_class_status"] = np.select(
        [maximum_class.isin(["MD", "HD"]), maximum_class.eq("LD")],
        ["supported", "not_supported"],
        default="unknown",
    )
    result["station_power_kw"] = np.nan
    result["station_power_status"] = "not_reported_in_afdc_snapshot"
    return result


def _read_depots(
    depot_candidates_path: Path,
    boundary: Any,
    city_slug: str,
) -> gpd.GeoDataFrame:
    frame = pd.read_csv(depot_candidates_path)
    frame = frame.loc[frame["city_slug"].astype(str) == city_slug].copy()
    if frame.empty:
        raise ValueError(f"No depot candidates for {city_slug}")
    required = {
        "benchmark_depot_class",
        "strict_candidate_eligible",
        "optional_candidate_eligible",
        "cle_candidate_eligible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Depot candidates predate the strict/optional contract; missing columns: "
            f"{sorted(missing)}"
        )
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    if not bool(points.geometry.covered_by(boundary).all()):
        raise ValueError("A retained depot candidate falls outside the frozen service boundary")
    return points


def _afdc_manifest_output_sha256(manifest: dict[str, Any]) -> str | None:
    """Read either a raw-snapshot or coordinate-resolution manifest hash."""

    output = manifest.get("output")
    if isinstance(output, dict):
        value = output.get("sha256")
        return str(value) if value else None
    value = manifest.get("sha256") or manifest.get("output_sha256")
    return str(value) if value else None


def _filter_public_anchors(frame, masks):
    anchors = frame.copy()
    anchors.geometry = gpd.points_from_xy(
        frame["road_anchor_lon"], frame["road_anchor_lat"], crs=4326
    )
    retained, summary = filter_public_locations(anchors, masks)
    keep = set(retained["facility_anchor_id"].astype(str))
    return frame.loc[frame["facility_anchor_id"].astype(str).isin(keep)].copy().reset_index(
        drop=True
    ), summary


def facility_inputs_current(manifest_path: Path, **paths: Path) -> bool:
    """Reuse a facility layer only when all inputs and generation code agree."""
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "evrptw_city_facility_layers_v6":
        return False
    if manifest.get("generation_module_sha256") != sha256_file(Path(__file__)):
        return False
    if manifest.get("generation_dependencies") != _generation_dependencies():
        return False
    if manifest.get("public_location_policy", {}).get("policy") != PUBLIC_LOCATION_POLICY:
        return False
    inputs = manifest.get("inputs", {})
    for name, path in paths.items():
        if not path.is_file() or inputs.get(name, {}).get("sha256") != sha256_file(path):
            return False
    for name in ("charging_units", "charging_parameter_contract"):
        record = inputs.get(name)
        if record and (
            not Path(record["path"]).is_file()
            or sha256_file(Path(record["path"])) != record["sha256"]
        ):
            return False
    mask = inputs.get("public_exclusion_masks", {})
    afdc_metadata = inputs.get("afdc_resolution_manifest")
    if afdc_metadata:
        metadata_path = Path(afdc_metadata["path"])
        if not metadata_path.is_file() or sha256_file(metadata_path) != afdc_metadata["sha256"]:
            return False
    mask_path = Path(mask.get("path", ""))
    return mask_path.is_file() and sha256_file(mask_path) == mask.get("sha256")


def _generation_dependencies() -> dict[str, str]:
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "customer_access.py",
            "protected_connectivity.py",
            "public_locations.py",
            "charger_resolution.py",
            "charger_locations.py",
            "charging_parameters.py",
        )
    }


def build_depot_layer(
    depots: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    component_index,
    area_crs: str,
    *,
    masks: gpd.GeoDataFrame | None = None,
    max_road_access_m: float = 250.0,
    access_context=None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Materialize the public depot pool against directed eligible road edges."""
    depots, source_filter = filter_public_locations(depots, masks)
    if depots.empty:
        raise ValueError("No public depot locations remain; never fall back to excluded hosts")
    depots["facility_anchor_id"] = depots["candidate_id"].astype(str)
    if access_context is None:
        depots = _anchor_points_to_edges(depots, edges, area_crs)
        depots = label_projection_connectivity(depots, component_index)
        depots, anchor_filter = _filter_public_anchors(depots, masks)
    else:
        local = depots.copy()
        local["latent_service_location_id"] = local["candidate_id"].astype(str)
        depots = access_context.anchor(local, masks=masks)
        depots["facility_access_node_id"] = depots.service_access_node_id
        depots["facility_access_connector_id"] = depots.service_access_connector_id
        anchor_filter = {
            "retention_policy": "retain source-safe locations; unresolved local access is unavailable",
            "review_required": int((~depots.local_access_confirmed).sum()),
        }
    depots["road_access_distance_qa_flag"] = depots["road_access_distance_m"] > max_road_access_m
    depots["road_anchor_eligible"] = depots["protected_roundtrip_eligible"].astype(bool)
    depots["strict_depot_candidate_eligible"] = (
        depots["strict_candidate_eligible"].astype(bool) & depots["road_anchor_eligible"]
    )
    depots["optional_depot_candidate_eligible"] = (
        depots["optional_candidate_eligible"].astype(bool) & depots["road_anchor_eligible"]
    )
    depots["depot_candidate_eligible"] = (
        depots["cle_candidate_eligible"].astype(bool) & depots["road_anchor_eligible"]
    )
    depots["depot_release_eligible"] = False
    return depots, {"source_filter": source_filter, "road_anchor_filter": anchor_filter}


def build_charger_layers(
    chargers, edges, component_index, area_crs, *, masks=None, max_road_access_m=250.0
):
    """Apply public-source, directed-road and unique-location gates uniformly."""
    chargers, charger_public_filter = filter_public_locations(chargers, masks)
    chargers["facility_anchor_id"] = chargers["charger_id"]
    chargers = _anchor_points_to_edges(chargers, edges, area_crs)
    chargers = label_projection_connectivity(chargers, component_index)
    chargers, charger_anchor_public_filter = _filter_public_anchors(chargers, masks)
    chargers["road_access_distance_qa_flag"] = (
        chargers["road_access_distance_m"] > max_road_access_m
    )
    chargers["road_anchor_eligible"] = chargers["protected_roundtrip_eligible"].astype(bool)
    chargers["charger_candidate_eligible"] = (
        chargers["restricted_public"].ne(True)
        & chargers["reference_charge_mode"].ne("unsupported_or_unresolved")
        & chargers["coordinate_candidate_eligible"].astype(bool)
        & chargers["road_anchor_eligible"]
    )
    chargers["charger_release_eligible"] = False
    chargers["release_blocker"] = np.select(
        [
            ~chargers["coordinate_candidate_eligible"].astype(bool),
            ~chargers["road_anchor_eligible"].astype(bool),
            chargers["restricted_public"].eq(True),
            chargers["reference_charge_mode"].eq("unsupported_or_unresolved"),
            chargers["medium_duty_vehicle_class_status"].eq("not_supported"),
        ],
        [
            "coordinate_not_corroborated",
            "outside_reference_directed_scc",
            "restricted_public_access",
            "connector_unresolved",
            "maximum_vehicle_class_light_duty",
        ],
        default="station_power_and_vehicle_compatibility_not_fully_verified",
    )

    locations, members = collapse_charging_locations(chargers)
    audit = {
        "coordinate_policy": COORDINATE_POLICY,
        "location_policy": LOCATION_POLICY,
        "source_filter": charger_public_filter,
        "road_anchor_filter": charger_anchor_public_filter,
        "safe_source_records": len(members),
        "unique_locations": len(locations),
        "candidate_eligible_source_records": int(members.charger_candidate_eligible.sum()),
        "candidate_eligible_locations": int(locations.charger_candidate_eligible.sum()),
        "quarantine_reason_counts": {
            str(k): int(v)
            for k, v in members.location_quarantine_reason.value_counts().items()
            if k
        },
        "port_count_policy": "retain per-source configurations; no automatic port total",
    }
    return locations, members, audit


def build_facility_layers(
    *,
    city_slug: str,
    afdc_path: Path,
    depot_candidates_path: Path,
    depot_summary_path: Path,
    boundary_path: Path,
    graph_path: Path,
    output_dir: Path,
    area_crs: str,
    max_road_access_m: float = 250.0,
    access_context=None,
) -> dict[str, Any]:
    """Build candidate facility layers; release eligibility remains evidence-gated."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite facility output: {output_dir}")
    boundary_frame = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    boundary = boundary_frame.geometry.union_all()
    afdc_resolution_manifest_path = afdc_path.with_suffix(".manifest.json")
    afdc_resolution_manifest = None
    if afdc_resolution_manifest_path.is_file():
        afdc_resolution_manifest = json.loads(
            afdc_resolution_manifest_path.read_text(encoding="utf-8")
        )
        recorded_hash = _afdc_manifest_output_sha256(afdc_resolution_manifest)
        if recorded_hash and recorded_hash != sha256_file(afdc_path):
            raise ValueError("AFDC table and coordinate-resolution manifest hashes differ")
    depot_summary = json.loads(depot_summary_path.read_text(encoding="utf-8"))
    if depot_summary.get("city_slug") != city_slug:
        raise ValueError("Depot summary city_slug differs from requested city")
    if depot_summary.get("candidate_csv_sha256") != sha256_file(depot_candidates_path):
        raise ValueError(
            "Depot candidate table is not bound to its current source summary; rebuild depot candidates"
        )
    if (
        depot_summary.get("facility_grouping", {}).get("policy", {}).get("version")
        != "spatial_depot_v2"
    ):
        raise ValueError("Depot inputs predate the current spatial location contract")
    if depot_summary.get("source", {}).get("boundary_sha256") != sha256_file(boundary_path):
        raise ValueError("Depot audit used a different service boundary")
    if depot_summary.get("graph_gate", {}).get("graph_path") != str(graph_path.resolve()):
        raise ValueError("Depot audit used a different operational graph")
    recorded_depot_graph_sha = depot_summary.get("graph_gate", {}).get("graph_sha256")
    if recorded_depot_graph_sha and recorded_depot_graph_sha != sha256_file(graph_path):
        raise ValueError("Depot audit used a different operational graph hash")

    public_record = depot_summary.get("public_location_policy", {})
    mask_path = Path(public_record["mask_path"]) if public_record.get("mask_path") else None
    if mask_path is None or not mask_path.is_file():
        raise ValueError(
            "Depot inputs lack the public-location exclusion mask; rebuild depot candidates"
        )
    if sha256_file(mask_path) != public_record.get("mask_sha256"):
        raise ValueError("Public-location exclusion mask checksum changed")
    masks = load_exclusion_masks(mask_path)

    context_manifest = Path(graph_path).parent / "access_context_manifest.json"
    if access_context is None and context_manifest.is_file():
        from .local_road_access import LocalAccessContext

        access_context = LocalAccessContext.load(context_manifest.parent)
    graph = access_context.core if access_context is not None else ox.load_graphml(graph_path)
    edges = build_eligible_physical_edges(graph_path, area_crs, graph=graph)
    component_index = build_directed_component_index(graph)
    chargers = _normalize_chargers(_filter_afdc_city(afdc_path, boundary, city_slug))
    chargers, charging_members, charging_location_audit = build_charger_layers(
        chargers,
        edges,
        component_index,
        area_crs,
        masks=masks,
        max_road_access_m=max_road_access_m,
    )
    units, charging_contract = load_frozen_charging_units()
    chargers, charging_members, charging_configurations, charging_parameters = (
        select_reference_configurations(
            chargers,
            charging_members,
            units,
            charging_contract["new_national_medians_kw"],
        )
    )
    charger_public_filter = charging_location_audit["source_filter"]
    charger_anchor_public_filter = charging_location_audit["road_anchor_filter"]

    depots = _read_depots(depot_candidates_path, boundary, city_slug)
    depots, depot_public_filter = build_depot_layer(
        depots,
        edges,
        component_index,
        area_crs,
        masks=masks,
        max_road_access_m=max_road_access_m,
        access_context=access_context,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{city_slug}-facilities-", dir=output_dir.parent
    ) as temp:
        staged = Path(temp) / output_dir.name
        staged.mkdir()
        charger_path = staged / "chargers.parquet"
        depot_path = staged / "depots.parquet"
        chargers.to_parquet(charger_path, index=False)
        charging_members.to_parquet(staged / "charging_members.parquet", index=False)
        charging_configurations.to_parquet(staged / "charging_configurations.parquet", index=False)
        depots.to_parquet(depot_path, index=False)
        manifest = {
            "schema": "evrptw_city_facility_layers_v6",
            "generation_module_sha256": sha256_file(Path(__file__)),
            "generation_dependencies": _generation_dependencies(),
            "status": "cle_build_candidates_not_release_eligible",
            "generated_utc": datetime.now(UTC).isoformat(),
            "city_slug": city_slug,
            "inputs": {
                "afdc": {"path": str(afdc_path.resolve()), "sha256": sha256_file(afdc_path)},
                "charging_units": {
                    "path": str(
                        (CONFIG_ROOT.parent / charging_contract["unit_source"]["path"]).resolve()
                    ),
                    "sha256": charging_contract["unit_source"]["sha256"],
                },
                "charging_parameter_contract": {
                    "path": str(CONFIG_ROOT / "charging_parameter_contract_v1.json"),
                    "sha256": sha256_file(CONFIG_ROOT / "charging_parameter_contract_v1.json"),
                },
                "afdc_resolution_manifest": (
                    {
                        "path": str(afdc_resolution_manifest_path.resolve()),
                        "sha256": sha256_file(afdc_resolution_manifest_path),
                        "schema": afdc_resolution_manifest.get("schema"),
                    }
                    if afdc_resolution_manifest is not None
                    else None
                ),
                "depot_candidates": {
                    "path": str(depot_candidates_path.resolve()),
                    "sha256": sha256_file(depot_candidates_path),
                },
                "depot_summary": {
                    "path": str(depot_summary_path.resolve()),
                    "sha256": sha256_file(depot_summary_path),
                },
                "public_exclusion_masks": {
                    "path": str(mask_path.resolve()),
                    "sha256": sha256_file(mask_path),
                },
                "boundary": {
                    "path": str(boundary_path.resolve()),
                    "sha256": sha256_file(boundary_path),
                },
                "graph": {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)},
            },
            "source_provenance": {
                "depot_osm": {
                    key: depot_summary.get("source", {}).get(key)
                    for key in (
                        "dataset",
                        "pbf_source_url",
                        "pbf_replication_timestamp_utc",
                        "pbf_sha256",
                        "boundary_sha256",
                    )
                },
                "road_osm": depot_summary.get("road_source", {}),
                "depot_generation": depot_summary.get("build_contract", {}),
                "grouping_policy": depot_summary.get("facility_grouping", {}).get("policy"),
                "afdc_snapshot": afdc_resolution_manifest,
            },
            "public_location_policy": {
                "policy": PUBLIC_LOCATION_POLICY,
                "mask_sha256": sha256_file(mask_path),
                "depot_filter": depot_public_filter,
                "charger_filter": charger_public_filter,
                "charger_road_anchor_filter": charger_anchor_public_filter,
            },
            "road_anchor": {
                "method": "facility point to eligible operational physical edge projection",
                "eligible_physical_edge_count": len(edges),
                "distance_qa_reference_m": max_road_access_m,
                "distance_policy": "retain all; flag the distance tail for coordinate/access review",
                "projection_contract": "per-directed-edge offsets retained for exact active-facility edge splitting",
                "connector_contract": "facility point to road projection is bidirectional with equal instance-assigned speed",
                "protected_roundtrip_policy": (
                    "default benchmark candidates must inherit the largest directed road SCC"
                ),
            },
            "charging": {
                "location_contract": charging_location_audit,
                "parameter_contract": charging_parameters,
                "inside_boundary_public_available_site_count": len(chargers),
                "road_anchor_eligible_count": int(chargers["road_anchor_eligible"].sum()),
                "road_access_distance_qa_flag_count": int(
                    chargers["road_access_distance_qa_flag"].sum()
                ),
                "candidate_eligible_count": int(chargers["charger_candidate_eligible"].sum()),
                "coordinate_candidate_eligible_count": int(
                    chargers["coordinate_candidate_eligible"].sum()
                ),
                "coordinate_exact_geometry_count": int(
                    chargers["coordinate_validation_status"].eq("exact_geometry_corroborated").sum()
                ),
                "coordinate_address_only_count": int(
                    chargers["coordinate_validation_status"]
                    .eq("address_corroborated_exact_geometry_unverified")
                    .sum()
                ),
                "coordinate_uncorroborated_count": int(
                    chargers["coordinate_validation_status"]
                    .eq("uncorroborated_source_coordinate")
                    .sum()
                ),
                "coordinate_validation_tier_counts": {
                    str(key): int(value)
                    for key, value in chargers["coordinate_validation_tier"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "protected_connectivity": connectivity_summary(chargers, component_index),
                "release_eligible_count": 0,
                "ccs1_site_count": int(chargers["has_ccs1"].sum()),
                "j1772_site_count": int(chargers["has_j1772"].sum()),
                "restricted_public_site_count": int(chargers["restricted_public"].eq(True).sum()),
                "power_policy": "compatible per-unit/connector reported power preferred; missing/QA uses frozen compatible-mode median; vehicle cap before single Stage2 derating",
            },
            "depots": {
                "retained_candidate_count": len(depots),
                "retained_location_count": int(depots["facility_group_id"].nunique()),
                "candidate_eligible_location_count": int(
                    depots.loc[depots["depot_candidate_eligible"], "facility_group_id"].nunique()
                ),
                "road_access_distance_qa_flag_count": int(
                    depots["road_access_distance_qa_flag"].sum()
                ),
                "candidate_eligible_count": int(depots["depot_candidate_eligible"].sum()),
                "protected_connectivity": connectivity_summary(depots, component_index),
                "strict_candidate_eligible_count": int(
                    depots["strict_depot_candidate_eligible"].sum()
                ),
                "optional_candidate_eligible_count": int(
                    depots["optional_depot_candidate_eligible"].sum()
                ),
                "release_eligible_count": 0,
                "release_policy": (
                    "modeled source-supported Tier A and optional Tier B departure locations; "
                    "current last-mile operation is not asserted; whole-CLE release gates remain separate"
                ),
            },
            "outputs": {
                "chargers": "chargers.parquet",
                "charging_members": "charging_members.parquet",
                "charging_configurations": "charging_configurations.parquet",
                "depots": "depots.parquet",
            },
        }
        write_json(staged / "facility_manifest.json", manifest)
        manifest["output_sha256"] = {
            name: sha256_file(staged / relative) for name, relative in manifest["outputs"].items()
        }
        write_json(staged / "facility_manifest.json", manifest)
        staged.replace(output_dir)
    return manifest
