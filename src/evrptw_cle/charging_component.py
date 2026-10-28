"""Source-bound charging locations on the current, purpose-specific road graph.

This is a CLE component builder. It neither activates stations nor asserts an
operating-day schedule, observed owner permission, queues or charging curves.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .charger_locations import LOCATION_POLICY, collapse_charging_locations
from .charger_resolution import COORDINATE_POLICY, resolve_afdc_coordinates
from .charging_parameters import PARAMETER_POLICY, select_reference_configurations
from .context_validation import validate_context_bindings
from .facilities import _normalize_chargers
from .local_road_access import LocalAccessContext
from .public_locations import filter_public_locations
from .runtime_paths import CONFIG_ROOT
from .util import sha256_file, write_json


COMPONENT_POLICY = "native_charging_current_common_graph_v1"
SOURCE_SCHEMA = "evrptw_charging_sources_v1"
ROAD_SNAPSHOT = "2026-10-25T21:21:21Z"


def _source_scope(frame, label):
    for column, expected in {
        "Country": "US",
        "Fuel Type Code": "ELEC",
        "Access Code": "public",
        "Status Code": "E",
    }.items():
        if column not in frame or not frame[column].fillna("").eq(expected).all():
            raise ValueError(f"{label} has missing or out-of-scope {column}")


def _dependencies():
    names = [
        "charging_component.py",
        "charging_road_access.py",
        "charger_resolution.py",
        "charger_locations.py",
        "charging_parameters.py",
        "facilities.py",
        "local_road_access.py",
        "delivery_access.py",
        "customer_access.py",
        "public_locations.py",
        "context_validation.py",
        "terminal_access.py",
        "road_tags.py",
        "frontage.py",
    ]
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _input(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_charging_layers(
    stations,
    units,
    context,
    *,
    masks,
    boundary,
    city_slug,
    medians,
    census_results=None,
    footprints=None,
):
    """Return locations, members, configurations, audit and charging access context.

    Potential location, physical access and reference-vehicle parameters remain
    separate flags. The historical 1000 m distance quarantine is deliberately
    absent: unresolved source-safe locations remain inspectable pending rows.
    """
    from .charging_road_access import ChargingAccessContext, build_charging_access_context

    _source_scope(stations, "AFDC stations")
    _source_scope(units, "AFDC charging units")
    charging_context = (
        context
        if isinstance(context, ChargingAccessContext)
        else build_charging_access_context(context)
    )
    resolved = resolve_afdc_coordinates(stations, census_results=census_results)
    points = gpd.GeoDataFrame(
        resolved,
        geometry=gpd.points_from_xy(resolved.resolved_longitude, resolved.resolved_latitude),
        crs=4326,
    )
    service = boundary.to_crs(4326).geometry.union_all()
    points = points.loc[points.geometry.covered_by(service)].copy().reset_index(drop=True)
    points["city_slug"] = city_slug
    points, source_filter = filter_public_locations(points, masks)
    points = _normalize_chargers(points)
    points["charger_candidate_eligible"] = points.coordinate_candidate_eligible.astype(
        bool
    ) & points.restricted_public.ne(True)
    # Collapse before road distances are attached, keeping the source-identity
    # quarantine separate from the old distance-based access quarantine.
    locations, members = collapse_charging_locations(points)
    members["source_location_eligible"] = members.charger_candidate_eligible
    locations["potential_location_eligible"] = locations.charger_candidate_eligible
    locations, members, configurations, parameters = select_reference_configurations(
        locations, members, units, medians
    )
    locations["reference_vehicle_compatible"] = locations.reference_vehicle_eligible
    if len(locations):
        locations["latent_service_location_id"] = locations.charging_location_id
        locations["facility_anchor_id"] = locations.charging_location_id
        locations = charging_context.anchor(locations, masks=masks, footprints=footprints)
        locations["road_anchor_method"] = "afdc_native_point_to_local_physical_way_projection"
        locations["facility_access_node_id"] = locations.service_access_node_id
        locations["facility_access_connector_id"] = locations.service_access_connector_id
    else:
        for field in ["local_access_confirmed", "protected_roundtrip_eligible"]:
            locations[field] = pd.Series(dtype=bool)
        locations["access_review_reason"] = pd.Series(dtype=str)
        locations["road_access_distance_m"] = pd.Series(dtype=float)
    locations["road_accessible"] = locations.local_access_confirmed.astype(bool)
    locations["road_anchor_eligible"] = locations.road_accessible
    locations["road_access_distance_qa_flag"] = locations.road_access_distance_m.gt(250.0)
    locations["charger_candidate_eligible"] = (
        locations.potential_location_eligible & locations.road_accessible
    )
    locations["charging_parameter_eligible"] = (
        locations.charger_candidate_eligible & locations.reference_vehicle_compatible
    )
    locations["charger_release_eligible"] = False
    locations["charging_component_policy"] = COMPONENT_POLICY
    locations["pending_reason"] = np.select(
        [
            locations.location_quarantine_reason.ne(""),
            ~locations.coordinate_candidate_eligible.astype(bool),
            locations.restricted_public.eq(True),
            ~locations.road_accessible,
            ~locations.reference_vehicle_compatible,
        ],
        [
            locations.location_quarantine_reason,
            "coordinate_not_corroborated",
            "restricted_public_access",
            locations.access_review_reason,
            "no_native_compatible_configuration",
        ],
        default="",
    )
    # Release remains a whole-CLE decision; an empty component reason only means
    # these component gates pass, never that an instance has been activated.
    locations["release_blocker"] = locations.pending_reason
    locations = locations.drop(
        columns=["customer_release_eligible", "quarantine_reason"], errors="ignore"
    )
    indexed = locations.set_index("charging_location_id")
    for field in [
        "road_accessible",
        "road_anchor_eligible",
        "reference_vehicle_compatible",
        "charging_parameter_eligible",
        "pending_reason",
    ]:
        members["location_" + field] = members.charging_location_id.map(indexed[field])
    members["charger_candidate_eligible"] = (
        members.source_location_eligible & members.location_road_accessible
    )
    members["charger_release_eligible"] = False
    audit = {
        "policy": COMPONENT_POLICY,
        "coordinate_policy": COORDINATE_POLICY,
        "location_policy": LOCATION_POLICY,
        "parameter_policy": PARAMETER_POLICY,
        "source_filter": source_filter,
        "source_safe_members": len(members),
        "source_safe_locations": len(locations),
        "potential_locations": int(locations.potential_location_eligible.sum()),
        "road_accessible_locations": int(locations.road_accessible.sum()),
        "reference_vehicle_compatible_locations": int(locations.reference_vehicle_compatible.sum()),
        "charging_parameter_eligible_locations": int(locations.charging_parameter_eligible.sum()),
        "pending_locations": int(locations.pending_reason.ne("").sum()),
        "pending_reason_counts": {
            str(k): int(v) for k, v in locations.pending_reason.value_counts().items() if k
        },
        "parameters": parameters,
        "distance_policy": "retain source-safe rows; no numeric access distance rejection",
        "distance_qa_reference_m": 250.0,
        "building_obstacle_coverage": "provided" if footprints is not None else "not_provided",
        "private_delivery_authorization_admitted": False,
        "instance_activation_included": False,
    }
    return locations, members, configurations, audit, charging_context


def build_charging_component(
    *,
    city_slug,
    station_path,
    source_manifest_path,
    units_path,
    boundary_path,
    mask_path,
    context_dir,
    area_crs,
    output_dir,
    census_path=None,
    footprints_path=None,
):
    """Construct a source-bound charging component in a fresh output directory."""
    output_dir, context_dir = Path(output_dir), Path(context_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh charging component output directory")
    sources = json.loads(Path(source_manifest_path).read_text())
    if sources.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Charging source manifest schema is not supported")
    for name, path in [("station", station_path), ("units", units_path)]:
        if sha256_file(Path(path)) != sources.get(name + "_sha256"):
            raise ValueError("Charging " + name + " snapshot hash differs")
    if census_path is None and sources.get("census_sha256"):
        raise ValueError("Declared frozen Census evidence is required")
    if census_path is not None:
        if sha256_file(Path(census_path)) != sources.get("census_sha256"):
            raise ValueError("Charging Census snapshot hash differs")
        if sources.get("census_input_station_sha256") != sources["station_sha256"]:
            raise ValueError("Charging Census evidence belongs to a different station snapshot")
    road_manifest_path = context_dir / "access_context_manifest.json"
    road_manifest = json.loads(road_manifest_path.read_text())
    if road_manifest.get("source", {}).get("pbf_replication_timestamp_utc") != ROAD_SNAPSHOT:
        raise ValueError("Charging requires the accepted common road snapshot")
    validate_context_bindings(
        road_manifest,
        city_slug=city_slug,
        boundary_path=boundary_path,
        mask_path=mask_path,
        area_crs=area_crs,
    )
    contract_path = CONFIG_ROOT / "charging_parameter_contract_v1.json"
    contract = json.loads(contract_path.read_text())
    if sources["units_sha256"] != contract["unit_source"]["sha256"]:
        raise ValueError("Charging units differ from the frozen national power contract")
    context = LocalAccessContext.load(context_dir)
    stations = pd.read_csv(station_path, low_memory=False, dtype={"ZIP": str})
    units = pd.read_csv(units_path, low_memory=False, dtype={"ZIP": str})
    census = pd.read_csv(census_path, low_memory=False) if census_path is not None else None
    masks = gpd.read_parquet(mask_path)
    boundary = gpd.read_file(boundary_path)
    footprints = gpd.read_parquet(footprints_path) if footprints_path is not None else None
    locations, members, configurations, audit, access = build_charging_layers(
        stations,
        units,
        context,
        masks=masks,
        boundary=boundary,
        city_slug=city_slug,
        medians=contract["new_national_medians_kw"],
        census_results=census,
        footprints=footprints,
    )
    inputs = {
        name: _input(path)
        for name, path in [
            ("station", station_path),
            ("units", units_path),
            ("source_manifest", source_manifest_path),
            ("boundary", boundary_path),
            ("masks", mask_path),
            ("road_context", road_manifest_path),
            ("parameter_contract", contract_path),
        ]
    }
    if census_path is not None:
        inputs["census"] = _input(census_path)
    if footprints_path is not None:
        inputs["building_obstacles"] = _input(footprints_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="charging-build-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "component"
        staged.mkdir()
        locations.to_parquet(staged / "chargers.parquet", index=False)
        members.to_parquet(staged / "charging_members.parquet", index=False)
        configurations.to_parquet(staged / "charging_configurations.parquet", index=False)
        private = staged / "private_access"
        private.mkdir()
        access.save(private)
        access.physical.to_parquet(private / "physical_roads.parquet", index=False)
        manifest = {
            "schema": "evrptw_charging_component_v1",
            "policy": COMPONENT_POLICY,
            "city_slug": city_slug,
            "area_crs": area_crs,
            "generated_utc": datetime.now(UTC).isoformat(),
            "status": "current_common_graph_component_not_whole_CLE_release",
            "inputs": inputs,
            "source_provenance": sources,
            "road_source": road_manifest["source"],
            "road_context_outputs": road_manifest["outputs"],
            "generation_dependencies": _dependencies(),
            "audit": audit,
            "outputs": {
                name: sha256_file(staged / name)
                for name in [
                    "chargers.parquet",
                    "charging_members.parquet",
                    "charging_configurations.parquet",
                ]
            },
            "private_outputs": {
                str(path.relative_to(staged)): sha256_file(path)
                for path in sorted(private.iterdir())
                if path.is_file()
            },
            "whole_cle_frozen": False,
            "instance_activation_included": False,
        }
        write_json(staged / "charging_component_manifest.json", manifest)
        staged.replace(output_dir)
    return manifest
