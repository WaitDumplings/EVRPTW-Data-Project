"""Regenerate Microsoft composition and native NSI matching from frozen inputs."""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .frozen_inputs import archive_path, validate_archive_inventory
from .microsoft_composite import compose_footprints
from .nsi_evidence import iter_geojsonseq, match_crosswalk, residential_points
from .runtime_paths import CONFIG_ROOT
from .util import sha256_file, write_json


def source_path(inventory, root, key):
    record = inventory["files"][key]
    path = archive_path(root, record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError("Missing or changed staged source: " + key)
    return path


def prepare_customer_city(*, inventory_path, source_root, city_slug, output_dir):
    """Reparse exact NSI payloads and rematch the reviewed footprint composite.

    Normalized per-city Microsoft extractions and OSM context tables are frozen
    inputs, not live API substitutes. The produced files form a new derived
    version; original archived file hashes are never relabeled as new outputs.
    """
    inventory_path, source_root, output_dir = map(Path, [inventory_path, source_root, output_dir])
    if output_dir.exists():
        raise FileExistsError("Use a fresh customer source preparation output")
    inventory = json.loads(inventory_path.read_text())
    validate_archive_inventory(inventory)
    city = inventory["cities"][city_slug]
    used = {}

    def resolve(key):
        path = source_path(inventory, source_root, key)
        used[key] = dict(path=str(path.resolve()), sha256=sha256_file(path))
        return path

    boundary_path = resolve(city["boundaries"]["service_boundary"])
    boundary = gpd.read_file(boundary_path).to_crs(4326).geometry.union_all()
    shapes = {}
    for kind in ["global", "legacy"]:
        parts = [gpd.read_parquet(resolve(key)) for key in city["geometry"][kind]]
        shapes[kind] = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs
        )
    composite = compose_footprints(shapes["global"], shapes["legacy"], area_crs=city["area_crs"])
    tile_paths = []
    tile_records = []
    for tile in city["nsi_tiles"]:
        path, metadata_path, request_path = (
            resolve(tile["response"]),
            resolve(tile["source"]),
            resolve(tile["request"]),
        )
        metadata = json.loads(metadata_path.read_text())
        if metadata["request_sha256"] != tile["request_sha256"] or metadata[
            "response_sha256"
        ] != sha256_file(path):
            raise ValueError("NSI tile evidence is not bound to the frozen request/response")
        tile_paths.append(path)
        tile_records.append(
            dict(
                tile_id=tile["tile_id"],
                request_sha256=tile["request_sha256"],
                response_sha256=sha256_file(path),
                request_file_sha256=sha256_file(request_path),
            )
        )
    points, parsing = residential_points(
        chain.from_iterable(iter_geojsonseq(p) for p in tile_paths), boundary=boundary
    )
    crosswalk = match_crosswalk(points, composite["composite_footprints"])
    output_dir.mkdir(parents=True)
    files = {}
    for key, frame in [
        ("microsoft_footprints", composite["composite_footprints"]),
        ("new_geometry_quarantine", composite["new_geometry_quarantine"]),
        ("legacy_admission_audit", composite["legacy_admission_audit"]),
        ("nsi_crosswalk", crosswalk),
    ]:
        path = output_dir / (key + ".parquet")
        frame.to_parquet(path, index=False)
        files[key] = dict(path=path.name, sha256=sha256_file(path), record_count=len(frame))
    manifest = dict(
        schema="evrptw_prepared_customer_sources_v1",
        city_slug=city_slug,
        policy="identified_residential_site_v3",
        inputs=used,
        outputs=files,
        inventory_sha256=sha256_file(inventory_path),
        boundary_sha256=sha256_file(boundary_path),
        microsoft=composite["summary"],
        nsi_parse=parsing,
        nsi_tiles=tile_records,
        matching_counts=crosswalk.composite_status.value_counts().to_dict(),
        geometry_scope="regenerated composite of archived normalized Microsoft city extractions",
        generation_dependencies={
            name: sha256_file(Path(__file__).with_name(name))
            for name in [
                "source_preparation.py",
                "microsoft_composite.py",
                "nsi_evidence.py",
                "customer_spatial.py",
            ]
        },
        geometry_rules_sha256=sha256_file(CONFIG_ROOT / "microsoft_geometry_rules_v1.json"),
        source_data_refreshed=False,
        source_vintages=inventory["source_vintages"],
    )
    write_json(output_dir / "source_preparation_manifest.json", manifest)
    return manifest


def assemble_customer_source_registry(*, inventory_path, source_root, prepared_root, output_path):
    """Bind regenerated derived files and unchanged frozen OSM context to one registry."""
    inventory_path, source_root, prepared_root, output_path = map(
        Path, [inventory_path, source_root, prepared_root, output_path]
    )
    if output_path.exists():
        raise FileExistsError("Use a fresh customer source registry")
    inventory = json.loads(inventory_path.read_text())
    registry = dict(
        schema="evrptw_frozen_customer_sources_v3",
        policy="identified_residential_site_v3",
        source_vintages=inventory["source_vintages"],
        cities={},
    )
    for city_slug, city in inventory["cities"].items():
        base = prepared_root / city_slug
        manifest_path = base / "source_preparation_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sources = {}
        for name in ["nsi_crosswalk", "microsoft_footprints"]:
            item = manifest["outputs"][name]
            path = base / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise ValueError("Changed prepared customer source: " + city_slug + ":" + name)
            sources[name] = dict(item, path=str(path.resolve()), private_build_input=False)
        for name in ["osm_buildings", "osm_complexes", "public_masks"]:
            key = city["sources"][name]
            item = inventory["files"][key]
            sources[name] = dict(
                path=str(source_path(inventory, source_root, key)),
                sha256=item["sha256"],
                private_build_input=item["private_build_input"],
            )
        provenance = {
            "regenerated_customer_inputs": dict(
                path=str(manifest_path.resolve()), sha256=sha256_file(manifest_path)
            )
        }
        for name, key in city["provenance"].items():
            provenance["archived_" + name] = dict(
                path=str(source_path(inventory, source_root, key)),
                sha256=inventory["files"][key]["sha256"],
            )
        registry["cities"][city_slug] = dict(
            area_crs=city["area_crs"],
            label=city["label"],
            boundary_sha256=city["boundary_sha256"],
            sources=sources,
            provenance=provenance,
        )
    write_json(output_path, registry)
    return registry
