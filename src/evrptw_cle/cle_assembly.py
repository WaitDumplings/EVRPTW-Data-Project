"""Assemble persistent city context without activating an operating day."""

from datetime import UTC, datetime
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .runtime_paths import CONFIG_ROOT, policy_files
from .util import sha256_file, write_json

SCHEMA = "evrptw_city_logistics_environment_v1"


def canonical_locations(customer, depot, charging):
    """Create unique location identities while retaining all source access rows."""
    dep = depot.sort_values(
        ["depot_candidate_eligible", "candidate_id"], ascending=[False, True], kind="stable"
    ).drop_duplicates("facility_group_id")
    frames = []
    for kind, table, key, gate, native_file, row_id in [
        ("customer", customer, "site_id", "cle_default_instance_eligible", "customers/latent_locations.parquet", "site_id"),
        ("depot", dep, "facility_group_id", "depot_candidate_eligible", "depots/depots.parquet", "candidate_id"),
        ("charging", charging, "charging_location_id", "charging_parameter_eligible", "charging/chargers.parquet", "charging_location_id"),
    ]:
        out = gpd.GeoDataFrame({
            "source_location_id": table[key].astype(str).to_numpy(),
            "location_type": kind,
            "candidate_eligible": table[gate].astype(bool).to_numpy(),
            "access_table": native_file, "access_row_id": table[row_id].astype(str).to_numpy(),
        }, geometry=table.to_crs(4326).geometry.to_numpy(), crs=4326)
        out["location_id"] = kind + ":" + out.source_location_id
        out["pending_reason"] = table.get(
            "quarantine_reason" if kind == "customer" else "pending_reason" if kind == "charging" else "access_review_reason",
            pd.Series("", index=table.index),
        ).fillna("").astype(str).to_numpy()
        out.loc[out.candidate_eligible, "pending_reason"] = ""
        out["residential_units"] = pd.array(
            table.residential_units.to_numpy() if kind == "customer" else [pd.NA] * len(table), dtype="Float64"
        )
        out["residential_type"] = table.service_location_type.to_numpy() if kind == "customer" else "not_applicable"
        out["source_identity"] = table.site_identity_source.to_numpy() if kind == "customer" else "osm_spatial_facility_group" if kind == "depot" else "afdc_colocated_source_members"
        out["scope_unresolved"] = table.get("site_scope_unresolved",pd.Series(False,index=table.index)).astype(bool).to_numpy()
        out["scope_review_required"] = table.get("site_scope_review_required",pd.Series(False,index=table.index)).astype(bool).to_numpy()
        frames.append(out)
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    if result.location_id.duplicated().any() or result.source_location_id.isna().any():
        raise ValueError("Canonical locations require unique nonempty source identities")
    return result.sort_values("location_id").reset_index(drop=True)


def bind_access_costs(canonical, native_tables, speeds):
    """Bind real source-edge legal/reference times to eligible access chains."""
    time_fields = ["legal_travel_time_s", "reference_travel_time_weekday_s", "reference_travel_time_weekend_s"]
    lookup = {tuple(key): tuple(value) for key, value in zip(
        speeds[["edge_u", "edge_v", "edge_key"]].astype(str).to_numpy(),
        speeds[["length_m", *time_fields]].to_numpy(dtype=float),
    )}
    rows = []
    fields = ["road_access_distance_m", "terminal_inbound_path_json", "terminal_outbound_path_json"]
    for kind, selected in canonical.loc[canonical.candidate_eligible].groupby("location_type",sort=True):
        table, key = native_tables[kind]
        # Read only the needed columns and iterate tuples. Full per-location
        # pandas Series construction is prohibitive for million-site CLEs.
        native = table[[key,*fields]].set_index(key).loc[selected.access_row_id]
        for location_id, values in zip(selected.location_id,native.itertuples(index=False,name=None)):
            distance, inbound, outbound = values
            result = dict(location_id=location_id, connector_length_m=float(distance),
                          connector_time_s_per_inverse_kph=3.6 * float(distance))
            for direction, path in [("inbound",inbound),("outbound",outbound)]:
                total = np.zeros(3)
                length = 0.0
                for ref in json.loads(path):
                    key = tuple(str(ref[name]) for name in ["u", "v", "key"])
                    if key not in lookup:
                        raise ValueError("Access path lacks common source-edge speed: " + str(key))
                    source = lookup[key]
                    fraction = float(ref["length_m"]) / source[0]
                    if not -1e-9 <= fraction <= 1 + 1e-8:
                        raise ValueError("Access path exceeds native source edge length")
                    length += float(ref["length_m"])
                    total += fraction * np.array(source[1:])
                result[direction + "_terminal_length_m"] = length
                for name, value in zip(["legal", "reference_weekday", "reference_weekend"], total):
                    result[direction + "_terminal_" + name + "_time_s"] = float(value)
            rows.append(result)
    return pd.DataFrame(rows)


def assemble_city_cle(*, city_dir, boundary_paths, source_inventory_path, customer_registry_path, road_source_dir):
    """Write a manifest and compact canonical layer inside an existing city build.

    All graph/native/member/ledger layers remain explicit component files;
    terminal roads are never merged into the public routing graph.
    """
    base = Path(city_dir).resolve()
    target = base / "cle_manifest.json"
    if target.exists():
        raise FileExistsError("Use a new CLE assembly directory/version")
    components = {
        "roads": "roads/access_context_manifest.json",
        "customer": "customers/residential_site_manifest.json",
        "depot": "depots/depot_component_manifest.json",
        "charging": "charging/charging_component_manifest.json",
        "speed": "speeds/common_speed_manifest.json",
    }
    road = json.loads((base / components["roads"]).read_text())
    customer = gpd.read_parquet(base / "customers/latent_locations.parquet")
    depot = gpd.read_parquet(base / "depots/depots.parquet")
    charging = gpd.read_parquet(base / "charging/chargers.parquet")
    canonical = canonical_locations(customer, depot, charging)
    canonical.to_parquet(base / "locations.parquet", index=False)
    costs = bind_access_costs(canonical, {
        "customer": (customer, "site_id"), "depot": (depot, "candidate_id"),
        "charging": (charging, "charging_location_id"),
    }, pd.read_parquet(base / "speeds/directed_speeds.parquet"))
    costs.to_parquet(base / "location_access_costs.parquet", index=False)
    boundaries = {}
    import os
    for name, path in boundary_paths.items():
        dest = base / "boundaries" / (name + ".geojson")
        dest.parent.mkdir(exist_ok=True)
        os.link(path, dest)
        boundaries[name] = str(dest.relative_to(base))
    evidence = base / "private_sources"
    evidence.mkdir()
    for name in ["osm_ways.parquet", "osm_access_nodes.parquet", "extraction.json"]:
        os.link(Path(road_source_dir) / name, evidence / name)
    files = {}
    for folder in ["roads", "customers", "depots", "charging", "speeds", "boundaries", "private_sources"]:
        for path in sorted((base / folder).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(base))] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
    for name in ["locations.parquet", "location_access_costs.parquet", "depot_source_manifest.json", "charging_source_manifest.json"]:
        path = base / name
        if path.is_file():
            files[name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
    summary = {}
    for kind, group in canonical.groupby("location_type"):
        summary[kind] = dict(retained=len(group), available=int(group.candidate_eligible.sum()),
                             pending=int((~group.candidate_eligible).sum()))
    result = dict(
        schema=SCHEMA, city_slug=road["city_slug"], generated_utc=datetime.now(UTC).isoformat(),
        stage=1, status="assembled_stage1_data_qualification_in_separate_certificates", files=files,
        components=components, boundaries=boundaries,
        road_snapshot_utc=road["source"]["pbf_replication_timestamp_utc"],
        road_source_pbf_sha256=road["source"]["pbf_sha256"], area_crs=road["area_crs"],
        sources={name:dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))
                 for name,path in [("frozen_inventory", source_inventory_path), ("customer_registry", customer_registry_path)]},
        schema_contract=json.loads((CONFIG_ROOT / "cle_stage1_schema_v1.json").read_text()),
        policy_hashes={str(path.relative_to(CONFIG_ROOT)):sha256_file(path)
                       for path in policy_files()},
        assembler_sha256=sha256_file(Path(__file__)), summary=summary,
        native_location_tables_preserved=True,
        default_day_or_vehicle_activation=False,
        freeze_authority="../../stage1_freeze.json", independent_verification="cle_verification.json",
        instance_activation_included=False, public_release=False,
    )
    write_json(target, result)
    return result
