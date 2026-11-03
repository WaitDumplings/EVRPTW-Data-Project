"""Current Stage 1 non-customer components over frozen local source bundles.

The historical dispatcher remains unchanged. New orchestration calls this module
with a fresh city directory; components retain existing canonical file names.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from .current_access import CURRENT_DELIVERY_POLICY
from .runtime_paths import CONFIG_ROOT
from .source_preparation import source_path
from .util import sha256_file, write_json

CURRENT_ROAD_OPTIONS = dict(selection_buffer_km=20, allow_private_through=True,
                            facility_masks=False)


def build_stage1_noncustomer_component(component, *, city_slug, inventory, source_root,
                                      road_source_dir, city_dir, footprints_path=None,
                                      road_options=None, native_access_masks=None):
    """Build roads, depots, charging or speeds using the accepted current method.

    ``inventory`` is the existing parsed frozen inventory. ``road_source_dir``
    is one city's complete-way extraction folder, not the parent cohort folder.
    ``native_access_masks`` may contain actual access-prohibition polygons;
    facility-category masks from historical inventories are never substituted.
    """
    base, road_source_dir = Path(city_dir), Path(road_source_dir)
    source_root = Path(source_root)
    city = inventory["cities"][city_slug]

    def resolve(key):
        return source_path(inventory, source_root, key)

    boundary = resolve(city["boundaries"]["service_boundary"])
    admin = resolve(city["boundaries"]["admin_boundary"])
    masks_path = Path(native_access_masks) if native_access_masks is not None else None
    options = dict(CURRENT_ROAD_OPTIONS, **(road_options or {}))
    base.mkdir(parents=True, exist_ok=True)
    if component == "roads":
        from .delivery_access import build_delivery_access_context
        from .local_road_access import build_access_context

        target, seed = base / "roads", base / "_physical_seed_current"
        if target.exists() or seed.exists():
            raise FileExistsError("Use fresh current-method road output; historical layers are immutable")
        inputs = dict(source_dir=road_source_dir, admin_boundary_path=admin,
                      service_boundary_path=boundary, mask_path=masks_path,
                      area_crs=city["area_crs"], **options)
        context, _ = build_access_context(**inputs, output_dir=seed)
        del context
        gc.collect()
        context, report = build_delivery_access_context(**inputs, seed_dir=seed, output_dir=target)
        shutil.copyfile(seed / "access_context_manifest.json", target / "physical_seed_manifest.json")
        del context
        gc.collect()
        shutil.rmtree(seed)
        return report

    from .context_validation import validate_context_bindings
    from .local_road_access import LocalAccessContext

    road_manifest_path = base / "roads/access_context_manifest.json"
    road = json.loads(road_manifest_path.read_text())
    if road["policy"] != CURRENT_DELIVERY_POLICY:
        raise ValueError("Current Stage 1 requires current-method roads, not historical terminal-only roads")
    validate_context_bindings(road, city_slug=city_slug, boundary_path=boundary,
                              area_crs=city["area_crs"])
    if component == "speeds":
        from .common_speed_binding import build_common_speed_component
        return build_common_speed_component(city_slug=city_slug, context_dir=base / "roads",
            output_dir=base / "speeds", boundary_path=boundary, state_fips=city.get("state_fips"))

    context = LocalAccessContext.load(base / "roads")
    footprints = gpd.read_parquet(footprints_path) if footprints_path is not None else None
    masks = gpd.read_parquet(masks_path) if masks_path is not None else None
    if component == "depots":
        from .depot_candidates import build_public_depot_candidates
        from .facilities import build_depot_layer
        from .protected_connectivity import build_directed_component_index

        target = base / "depots"
        if target.exists():
            raise FileExistsError("Use a fresh depot component directory")
        raw_path = resolve(city["depot"]["raw"])
        if city["depot"]["pbf_sha256"] != road["source"]["pbf_sha256"]:
            raise ValueError("Depot and road snapshots differ")
        candidates, grouping = build_public_depot_candidates(gpd.read_parquet(raw_path),
            gpd.read_file(boundary), city_slug, masks,
            explicit_logistics_parks=True, facility_categories=False)
        points = candidates.copy()
        points["access_search_extent_wkt"] = points.geometry.to_wkt()
        points.geometry = points.to_crs(city["area_crs"]).geometry.representative_point().to_crs(4326)
        points["longitude"], points["latitude"] = points.geometry.x, points.geometry.y
        frame, audit = build_depot_layer(points, None, build_directed_component_index(context.core),
            city["area_crs"], masks=masks, access_context=context, footprints=footprints)
        target.mkdir()
        frame.to_parquet(target / "depots.parquet", index=False)
        extraction_path = resolve(city["depot"]["extraction"])
        provenance = dict(pbf_sha256=city["depot"]["pbf_sha256"], raw_depot_sha256=sha256_file(raw_path),
                          extraction=dict(path=str(extraction_path), sha256=sha256_file(extraction_path)))
        write_json(base / "depot_source_manifest.json", provenance)
        report = dict(city=city_slug, policy="depot_explicit_logistics_park_v3", source_provenance=provenance,
            grouping=grouping, access_audit=audit, context_sha256=sha256_file(road_manifest_path),
            boundary_sha256=sha256_file(boundary),
            building_obstacles=dict(path=str(Path(footprints_path).resolve()),
                                   sha256=sha256_file(Path(footprints_path))) if footprints_path is not None else None,
            output_sha256=sha256_file(target / "depots.parquet"),
            available_unique_groups=int(frame.loc[frame.depot_candidate_eligible].facility_group_id.nunique()),
            generation_dependencies={name: sha256_file(Path(__file__).with_name(name)) for name in
                ["stage1_components.py", "depot_candidates.py", "depot_grouping.py", "current_access.py", "facilities.py"]},
            whole_cle_frozen=False)
        write_json(target / "depot_component_manifest.json", report)
        return report
    if component == "charging":
        from .charging_component import SOURCE_SCHEMA, build_charging_component

        entry = inventory["charging"]
        source_manifest = dict(schema=SOURCE_SCHEMA,
            **{name + "_sha256": inventory["files"][entry[name]]["sha256"] for name in ["station", "units", "census"]},
            census_input_station_sha256=inventory["files"][entry["station"]]["sha256"],
            provenance={name: dict(path=str(resolve(entry[name + "_provenance"])),
                sha256=inventory["files"][entry[name + "_provenance"]]["sha256"]) for name in ["station", "units", "census"]},
            source_vintages=inventory["source_vintages"])
        source_manifest_path = base / "charging_source_manifest.json"
        write_json(source_manifest_path, source_manifest)
        return build_charging_component(city_slug=city_slug, station_path=resolve(entry["station"]),
            units_path=resolve(entry["units"]), census_path=resolve(entry["census"]),
            source_manifest_path=source_manifest_path, boundary_path=boundary, mask_path=masks_path,
            context_dir=base / "roads", area_crs=city["area_crs"], output_dir=base / "charging",
            footprints_path=footprints_path)
    raise ValueError("Unknown non-customer component: " + component)


def _component_footprints(project, city, base):
    """Current canonical obstacles, excluding legacy and retired MS parents."""
    target = base / "customer_inputs/component_obstacles.parquet"
    if target.exists():
        return target
    sources = project["cities"][city]
    geometry_dir = base / "customer_inputs/geometry"
    provenance = geometry_dir / "ms_provenance.parquet"
    registry = geometry_dir / "full_osm_registry.parquet"
    if not provenance.exists() or not registry.exists():
        raise FileNotFoundError("Prepare canonical customer geometry before facility road obstacles")
    ms_path = Path(sources["microsoft"])
    filters = [("source_kind", "=", "global")] if "source_kind" in pq.read_schema(ms_path).names else None
    ms = gpd.read_parquet(ms_path, filters=filters)
    index = pd.read_parquet(provenance, columns=["canonical_ms_id", "retired"])
    ids = set(index.loc[~index.retired, "canonical_ms_id"])
    ms = ms.loc[ms.building_id.isin(ids)]
    frames = [ms[["building_id", "geometry"]].copy()]
    groups = pd.read_parquet(registry)
    chosen = groups.loc[groups.is_split_child | groups.is_old_addition, "canonical_id"].astype(str)
    if len(chosen):
        osm = gpd.read_parquet(sources["osm"], columns=["osm_id", "geometry"],
                              filters=[("osm_id", "in", chosen.tolist())])
        frames.append(osm.rename(columns={"osm_id": "building_id"}).to_crs(ms.crs))
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=ms.crs)
    result = result.drop_duplicates("building_id")
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    return target


def assemble_building_city(*, project, city, inventory, source_root, road_source_dir, base):
    """Current assembly with field provenance, without historical-mask assumptions."""
    from datetime import UTC, datetime
    from .cle_assembly import bind_access_costs, canonical_locations

    target = base / "cle_manifest.json"
    if target.exists():
        recorded = json.loads(target.read_text())
        if recorded.get("schema") != "evrptw_building_first_cle_v1" or recorded.get("whole_cle_frozen"):
            raise FileExistsError("Historical/frozen CLE assembly cannot be overwritten")
        same = (recorded.get("customer_profile") == project.get("customer", {}).get("active_profile", "u_lt100")
                and recorded.get("independent_verification") == "stage1_validation.json"
                and recorded.get("assembler_sha256") == sha256_file(Path(__file__)))
        same &= _manifest_file_hashes_match(base, recorded.get("files", {}))
        inventory_path = Path(project["components"]["inventory"])
        same &= recorded.get("sources", {}).get("inventory", {}).get("sha256") == sha256_file(inventory_path)
        if same:
            return dict(recorded, reused=True)
        # Only the derived assembly is replaced. Preserve the complete previous
        # receipt and its canonical tables for review; components remain intact.
        archive = base / "draft_history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for name in ["cle_manifest.json", "locations.parquet", "location_access_costs.parquet",
                     "stage1_validation.json", "cle_verification.json"]:
            path = base / name
            if path.exists():
                path.replace(archive / name)
    customer = gpd.read_parquet(base / "customers/latent_locations.parquet")
    depot = gpd.read_parquet(base / "depots/depots.parquet")
    charging = gpd.read_parquet(base / "charging/chargers.parquet")
    canonical = canonical_locations(customer, depot, charging)
    # Classification, numerical attribution and human/AI provenance are distinct
    # fields; keep them on the canonical layer consumed by later stages.
    keys = customer.site_id.astype(str)
    for column in ["residential_use", "residential_source", "units_method", "units_source", "units_scope", "units_raw_nsi",
                   "units_low", "units_high", "annotation_evidence_id", "units_review_status",
                   "units_independently_verified", "purpose_review_status", "selected_nsi_fd_id",
                   "selected_nsi_resunits", "capacity_units", "height_neighbor_estimated",
                   "neighbor_reference_id", "scale_assessed", "scale_gate_status"]:
        if column in customer:
            lookup = pd.Series(customer[column].to_numpy(), index=keys)
            canonical[column] = canonical.source_location_id.map(lookup).where(canonical.location_type.eq("customer"))
    canonical.to_parquet(base / "locations.parquet", index=False)
    speeds = pd.read_parquet(base / "speeds/directed_speeds.parquet")
    costs = bind_access_costs(canonical, {"customer": (customer, "site_id"),
        "depot": (depot, "candidate_id"), "charging": (charging, "charging_location_id")}, speeds)
    if costs.empty:
        costs = pd.DataFrame(columns=["location_id", "connector_length_m", "connector_time_s_per_inverse_kph"])
    costs.to_parquet(base / "location_access_costs.parquet", index=False)
    components = {"roads": "roads/access_context_manifest.json", "customer": "customers/residential_site_manifest.json",
        "depot": "depots/depot_component_manifest.json", "charging": "charging/charging_component_manifest.json",
        "speed": "speeds/common_speed_manifest.json"}
    road = json.loads((base / components["roads"]).read_text())
    boundaries = {}
    boundary_dir = base / "boundaries"
    boundary_dir.mkdir(exist_ok=True)
    entry = inventory["cities"][city]
    for name, key in entry["boundaries"].items():
        source = source_path(inventory, source_root, key)
        dest = boundary_dir / (name + ".geojson")
        shutil.copy2(source, dest)
        boundaries[name] = str(dest.relative_to(base))
    private = base / "private_sources"
    private.mkdir(exist_ok=True)
    for name in ["osm_ways.parquet", "osm_access_nodes.parquet", "extraction.json"]:
        source, dest = Path(road_source_dir) / name, private / name
        if not dest.exists():
            try:
                dest.hardlink_to(source)
            except OSError:
                shutil.copy2(source, dest)
    files = {}
    for folder in ["roads", "customers", "depots", "charging", "speeds", "boundaries", "private_sources", "customer_inputs"]:
        for path in sorted((base / folder).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(base))] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
    for name in ["locations.parquet", "location_access_costs.parquet", "depot_source_manifest.json", "charging_source_manifest.json"]:
        path = base / name
        if path.exists():
            files[name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
    manifest = dict(schema="evrptw_building_first_cle_v1", city_slug=city, stage=1,
        generated_utc=datetime.now(UTC).isoformat(), components=components, boundaries=boundaries,
        files=files, construction_options=dict(CURRENT_ROAD_OPTIONS,
            **project.get("construction", {}).get("roads", {}), explicit_logistics_parks=True),
        road_snapshot_utc=road["source"]["pbf_replication_timestamp_utc"],
        road_source_pbf_sha256=road["source"]["pbf_sha256"], area_crs=road["area_crs"],
        sources={"inventory": {"path": project["components"]["inventory"],
                  "sha256": sha256_file(Path(project["components"]["inventory"]))}},
        customer_profile=project.get("customer", {}).get("active_profile", "u_lt100"),
        schema_contract=json.loads((CONFIG_ROOT / "building_stage1_schema_v1.json").read_text()),
        summary={kind: dict(retained=len(part), available=int(part.candidate_eligible.sum()),
                           pending=int((~part.candidate_eligible).sum()))
                 for kind, part in canonical.groupby("location_type")},
        independent_verification="stage1_validation.json", whole_cle_frozen=False,
        instance_activation_included=False, public_release=False,
        assembler_sha256=sha256_file(Path(__file__)))
    write_json(target, manifest)
    return manifest


def _manifest_file_hashes_match(root, files):
    if not files:
        return False
    for name, value in files.items():
        digest = value.get("sha256") if isinstance(value, dict) else value
        path = Path(root) / name
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def _verify_recorded_component(manifest_path):
    """Validate the existing component without importing its heavy graphs."""
    manifest_path = Path(manifest_path)
    report = json.loads(manifest_path.read_text())
    outputs = dict(report.get("outputs", {}), **report.get("private_outputs", {}))
    if isinstance(report.get("output_sha256"), str):
        outputs["depots.parquet"] = report["output_sha256"]
    if not _manifest_file_hashes_match(manifest_path.parent, outputs):
        raise ValueError("Recorded component output changed: " + str(manifest_path))
    dependencies = report.get("generation_dependencies", {})
    if not dependencies:
        raise ValueError("Component lacks generation dependencies: " + str(manifest_path))
    for name, digest in dependencies.items():
        source = Path(__file__).with_name(name)
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError("Component generation code changed: " + name)
    return report


def _reuse_component(project, city, component, inventory, base, road_source_dir):
    """Return an unchanged component, or reject stale output without replacing it."""
    stamps = {"roads": "roads/access_context_manifest.json", "depots": "depots/depot_component_manifest.json",
              "charging": "charging/charging_component_manifest.json", "speeds": "speeds/common_speed_manifest.json"}
    stamp = base / stamps[component]
    if not stamp.exists():
        return None
    try:
        from .context_validation import validate_context_bindings

        report = _verify_recorded_component(stamp)
        road_path = base / stamps["roads"]
        road = report if component == "roads" else _verify_recorded_component(road_path)
        if road.get("policy") != CURRENT_DELIVERY_POLICY:
            raise ValueError("Different road method")
        entry = inventory["cities"][city]
        source_root = Path(project["components"]["source_root"])
        boundary = source_path(inventory, source_root, entry["boundaries"]["service_boundary"])
        admin = source_path(inventory, source_root, entry["boundaries"]["admin_boundary"])
        validate_context_bindings(road, city_slug=city, boundary_path=boundary, area_crs=entry["area_crs"])
        if road.get("admin_boundary_sha256") != sha256_file(admin):
            raise ValueError("Administrative boundary changed")
        extraction = json.loads((Path(road_source_dir) / "extraction.json").read_text())
        if road.get("source") != extraction:
            raise ValueError("Source road extraction changed")
        if not _manifest_file_hashes_match(road_source_dir, extraction.get("outputs", {})):
            raise ValueError("Source road extraction output changed")
        options = dict(CURRENT_ROAD_OPTIONS, **project.get("construction", {}).get("roads", {}))
        for field, key in [("selected_buffer_km", "selection_buffer_km"),
                           ("ordinary_private_through_routing", "allow_private_through"),
                           ("facility_category_masks", "facility_masks")]:
            if road.get(field) != options[key]:
                raise ValueError("Road option changed: " + key)
        if component == "depots":
            if report.get("context_sha256") != sha256_file(road_path):
                raise ValueError("Depot road context changed")
            raw = source_path(inventory, source_root, entry["depot"]["raw"])
            if report.get("source_provenance", {}).get("raw_depot_sha256") != sha256_file(raw):
                raise ValueError("Depot source changed")
            obstacle = report.get("building_obstacles")
            if obstacle and sha256_file(Path(obstacle["path"])) != obstacle["sha256"]:
                raise ValueError("Depot building obstacles changed")
        elif component != "roads":
            if report.get("inputs", {}).get("road_context", {}).get("sha256") != sha256_file(road_path):
                raise ValueError("Component road context changed")
            for value in report.get("inputs", {}).values():
                if isinstance(value, dict) and value.get("path") and value.get("sha256"):
                    if sha256_file(Path(value["path"])) != value["sha256"]:
                        raise ValueError("Component input changed: " + str(value["path"]))
            if component == "charging":
                for name in ["station", "units", "census"]:
                    actual = source_path(inventory, source_root, inventory["charging"][name])
                    if report.get("inputs", {}).get(name, {}).get("sha256") != sha256_file(actual):
                        raise ValueError("Charging source changed: " + name)
        return dict(report, reused=True)
    except (ValueError, OSError, KeyError) as error:
        raise FileExistsError(f"Stale {component} component retained at {stamp.parent}: {error}. "
                              "Choose a new output directory for changed component inputs/code.") from error


def build_component(*, project, city, component):
    """Thin bridge for the building-first project.json orchestrator."""
    config = project.get("components", {})
    inventory = json.loads(Path(config["inventory"]).read_text())
    base = Path(project["output"]) / "cities" / city
    if any((folder / "stage1_freeze.json").exists() for folder in [base, base.parent, base.parent.parent]):
        raise FileExistsError("A frozen Stage 1 output is immutable; choose a new output directory")
    road_source_dir = Path(config["road_sources"]) / city
    if component == "assemble":
        return assemble_building_city(project=project, city=city, inventory=inventory,
            source_root=Path(config["source_root"]), road_source_dir=road_source_dir, base=base)
    reused = _reuse_component(project, city, component, inventory, base, road_source_dir)
    if reused is not None:
        return reused
    footprints = None if component in {"roads", "speeds"} else _component_footprints(project, city, base)
    return build_stage1_noncustomer_component(component, city_slug=city, inventory=inventory,
        source_root=Path(config["source_root"]), road_source_dir=road_source_dir, city_dir=base,
        footprints_path=footprints,
        road_options=project.get("construction", {}).get("roads"),
        native_access_masks=config.get("native_access_masks", {}).get(city))
