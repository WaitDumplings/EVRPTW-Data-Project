"""Assign canonical buildings to the service domain without clipping polygons."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .building_inputs import BuildingEvidenceReader
from .util import sha256_file, write_json

SCHEMA = "evrptw_canonical_building_service_scope_v1"


def _path(value):
    return Path(value["path"] if isinstance(value, dict) else value).expanduser().resolve()


def prepare_city_scope(*, population_path, sources, geometry_dir, boundary_path, output_dir):
    """Save the service-boundary flag for every fixed matched building ID.

    Retained Microsoft entities use their own full polygon; OSM subdivision
    children use their own polygon. An OSM alias is never substituted for a
    retained Microsoft footprint when selecting the city's members.
    """
    population_path = _path(population_path)
    boundary_path = _path(boundary_path)
    output = _path(output_dir)
    inputs = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in [
        ("population", population_path), ("service_boundary", boundary_path),
        ("microsoft", _path(sources["microsoft"])), ("osm", _path(sources["osm"]))]}
    manifest_path, scope_path = output / "scope_manifest.json", output / "scope.parquet"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        old = {key: value["sha256"] for key, value in saved.get("inputs", {}).items()}
        current = {key: value["sha256"] for key, value in inputs.items()}
        if old != current or saved.get("schema") != SCHEMA:
            raise ValueError("Building service-scope inputs changed; use a fresh output directory")
        if not scope_path.is_file() or sha256_file(scope_path) != saved["scope_sha256"]:
            raise ValueError("Saved building service scope is missing or changed")
        return {**saved, "reused": True}
    if scope_path.exists():
        raise ValueError("Building service scope lacks its manifest; use a fresh output directory")
    columns = ["entity_index", "entity_id", "kind", "primary_path_id"]
    population = pd.read_parquet(population_path, columns=columns)
    if population.entity_id.isna().any() or population.entity_id.duplicated().any():
        raise ValueError("Service scope requires unique canonical building IDs")
    reader = BuildingEvidenceReader(sources, geometry_dir)
    buildings = reader.geometry(population)
    points = shapely.point_on_surface(buildings.geometry.array)
    boundary = gpd.read_file(boundary_path).to_crs(4326).geometry.union_all()
    # The prepared boundary must be the first covers() argument, not second in
    # covered_by(); large coastal city polygons otherwise dominate runtime.
    shapely.prepare(boundary)
    eligible = shapely.covers(boundary, points)
    scope = pd.DataFrame({"entity_id": population.entity_id.to_numpy(),
                          "service_domain_eligible": eligible,
                          "canonical_longitude": shapely.get_x(points),
                          "canonical_latitude": shapely.get_y(points)})
    output.mkdir(parents=True, exist_ok=True)
    scope.to_parquet(scope_path, index=False)
    by_kind = {str(kind): {"total": int(mask.sum()),
                           "inside": int((mask & eligible).sum()),
                           "outside": int((mask & ~eligible).sum())}
               for kind in population.kind.unique()
               for mask in [population.kind.eq(kind).to_numpy()]}
    manifest = {"schema": SCHEMA, "inputs": inputs,
        "geometry_dir": str(_path(geometry_dir)), "scope_path": str(scope_path),
        "scope_sha256": sha256_file(scope_path), "crs": "EPSG:4326",
        "membership_rule": "service boundary covers point_on_surface of unchanged canonical polygon",
        "retained_ms_point": "own Microsoft polygon, not OSM alias",
        "split_child_point": "own OSM child polygon, not former Microsoft parent",
        "polygons_clipped": False, "population_rows_dropped": False,
        "nsi_records_changed": False, "buildings": len(scope),
        "inside_service_boundary": int(np.count_nonzero(eligible)),
        "outside_service_boundary": int(np.count_nonzero(~eligible)), "by_kind": by_kind,
        "implementation_sha256": sha256_file(Path(__file__))}
    write_json(manifest_path, manifest)
    return manifest
