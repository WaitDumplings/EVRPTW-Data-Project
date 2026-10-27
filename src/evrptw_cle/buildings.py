from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape
from shapely.prepared import prep

from .util import sha256_file


@dataclass(frozen=True)
class BuildingCitySpec:
    slug: str
    label: str
    boundary_file: Path
    area_crs: str


@dataclass
class _CityAccumulator:
    spec: BuildingCitySpec
    boundary: Any
    boundary_prepared: Any
    bbox: tuple[float, float, float, float]
    output_dir: Path
    batch_size: int
    density_grid_m: float
    batch_index: int = 0
    accepted_count: int = 0
    bbox_candidate_count: int = 0
    invalid_geometry_count: int = 0
    area_values_m2: list[float] = field(default_factory=list)
    release_counts: Counter[str] = field(default_factory=Counter)
    capture_date_nonempty_count: int = 0
    density_cells: dict[tuple[int, int], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0.0])
    )
    pending_records: list[dict[str, Any]] = field(default_factory=list)
    pending_geometries: list[Any] = field(default_factory=list)

    @property
    def footprint_dir(self) -> Path:
        return self.output_dir / "footprints"

    @property
    def location_dir(self) -> Path:
        return self.output_dir / "locations"


def _iter_points(coordinates: Any) -> Iterable[tuple[float, float]]:
    if not coordinates:
        return
    first = coordinates[0]
    if isinstance(first, (int, float)):
        yield float(coordinates[0]), float(coordinates[1])
        return
    for child in coordinates:
        yield from _iter_points(child)


def _coordinate_bounds(coordinates: Any) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for x, y in _iter_points(coordinates):
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if not math.isfinite(min_x):
        raise ValueError("Geometry has no coordinates")
    return min_x, min_y, max_x, max_y


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0] or left[0] > right[2] or left[3] < right[1] or left[1] > right[3]
    )


def _feature_from_line(line: bytes) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(b'{"type":"Feature"'):
        return None
    if stripped.endswith(b","):
        stripped = stripped[:-1]
    return json.loads(stripped)


def _load_city_specs(preset_path: Path) -> tuple[dict[str, Any], list[BuildingCitySpec]]:
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    specs = []
    for item in payload["cities"]:
        specs.append(
            BuildingCitySpec(
                slug=item["slug"],
                label=item["label"],
                boundary_file=(preset_path.parent / item["boundary_file"]).resolve(),
                area_crs=item["area_crs"],
            )
        )
    return payload, specs


def _prepare_accumulators(
    specs: list[BuildingCitySpec],
    output_root: Path,
    batch_size: int,
    density_grid_m: float,
) -> list[_CityAccumulator]:
    accumulators = []
    for spec in specs:
        boundary_frame = gpd.read_file(spec.boundary_file).to_crs("EPSG:4326")
        boundary = boundary_frame.geometry.union_all()
        output_dir = output_root / spec.slug
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing building output: {output_dir}")
        (output_dir / "footprints").mkdir(parents=True)
        (output_dir / "locations").mkdir(parents=True)
        accumulators.append(
            _CityAccumulator(
                spec=spec,
                boundary=boundary,
                boundary_prepared=prep(boundary),
                bbox=tuple(float(value) for value in boundary.bounds),
                output_dir=output_dir,
                batch_size=batch_size,
                density_grid_m=density_grid_m,
            )
        )
    return accumulators


def _flush_city(accumulator: _CityAccumulator) -> None:
    if not accumulator.pending_records:
        return
    frame = gpd.GeoDataFrame(
        accumulator.pending_records,
        geometry=accumulator.pending_geometries,
        crs="EPSG:4326",
    )
    projected = frame.to_crs(accumulator.spec.area_crs)
    projected_points = projected.geometry.representative_point()
    areas = projected.geometry.area.to_numpy(dtype=float)
    point_wgs84 = gpd.GeoSeries(projected_points, crs=accumulator.spec.area_crs).to_crs("EPSG:4326")
    frame["footprint_area_m2"] = areas
    frame["location_lon"] = point_wgs84.x.to_numpy(dtype=float)
    frame["location_lat"] = point_wgs84.y.to_numpy(dtype=float)
    frame["membership_rule"] = "representative_point_within_land_boundary"

    part_name = f"part-{accumulator.batch_index:05d}.parquet"
    frame.to_parquet(accumulator.footprint_dir / part_name, index=False)

    location_frame = frame.drop(columns="geometry").copy()
    location_frame = gpd.GeoDataFrame(location_frame, geometry=point_wgs84, crs="EPSG:4326")
    location_frame.to_parquet(accumulator.location_dir / part_name, index=False)

    accumulator.area_values_m2.extend(float(value) for value in areas)
    for point, area in zip(projected_points, areas, strict=True):
        cell = (
            math.floor(point.x / accumulator.density_grid_m),
            math.floor(point.y / accumulator.density_grid_m),
        )
        accumulator.density_cells[cell][0] += 1.0
        accumulator.density_cells[cell][1] += float(area)

    accumulator.batch_index += 1
    accumulator.pending_records.clear()
    accumulator.pending_geometries.clear()


def _write_density_grid(accumulator: _CityAccumulator) -> Path:
    records = []
    geometries = []
    grid_m = accumulator.density_grid_m
    for (grid_x, grid_y), (count, total_area_m2) in sorted(accumulator.density_cells.items()):
        records.append(
            {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "building_count": int(count),
                "total_footprint_area_m2": total_area_m2,
                "mean_footprint_area_m2": total_area_m2 / count,
            }
        )
        geometries.append(
            box(
                grid_x * grid_m,
                grid_y * grid_m,
                (grid_x + 1) * grid_m,
                (grid_y + 1) * grid_m,
            )
        )
    grid = gpd.GeoDataFrame(records, geometry=geometries, crs=accumulator.spec.area_crs)
    grid = grid.to_crs("EPSG:4326")
    output = accumulator.output_dir / "building_density_grid.geojson"
    grid.to_file(output, driver="GeoJSON")
    return output


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "p25", "p50", "p75", "p95", "p99", "max")}
    array = np.asarray(values, dtype=float)
    quantile_values = np.quantile(array, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p05", "p25", "p50", "p75", "p95", "p99", "max"),
            quantile_values,
            strict=True,
        )
    }


def _write_city_summary(
    accumulator: _CityAccumulator,
    source_path: Path,
    source_sha256: str,
    source_feature_count: int,
    source_property_keys: list[str],
    preset_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema": "evrptw_building_footprints_v1",
        "city_slug": accumulator.spec.slug,
        "city_label": accumulator.spec.label,
        "source": {
            "dataset": preset_metadata.get("source_dataset", "Microsoft USBuildingFootprints"),
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "source_feature_count": source_feature_count,
            "source_property_keys": source_property_keys,
            "declared_crs": "EPSG:4326",
        },
        "boundary": {
            "path": str(accumulator.spec.boundary_file),
            "sha256": sha256_file(accumulator.spec.boundary_file),
            "membership_rule": "building representative point covered by land boundary",
            "boundary_bbox_wgs84": list(accumulator.bbox),
        },
        "area": {
            "computed_not_source_supplied": True,
            "projected_crs": accumulator.spec.area_crs,
            "unit": "m2",
            "quantiles_m2": _quantiles(accumulator.area_values_m2),
        },
        "building_count": accumulator.accepted_count,
        "bbox_candidate_count": accumulator.bbox_candidate_count,
        "invalid_geometry_count": accumulator.invalid_geometry_count,
        "release_counts": dict(sorted(accumulator.release_counts.items())),
        "capture_date_nonempty_count": accumulator.capture_date_nonempty_count,
        "capture_date_nonempty_share": (
            accumulator.capture_date_nonempty_count / accumulator.accepted_count
            if accumulator.accepted_count
            else 0.0
        ),
        "density_grid_m": accumulator.density_grid_m,
        "outputs": {
            "footprints": "footprints/*.parquet",
            "locations": "locations/*.parquet",
            "density_grid": "building_density_grid.geojson",
        },
        "semantic_status": "latent building opportunity; not yet a residential unit or customer",
    }
    output = accumulator.output_dir / "building_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def extract_building_footprints(
    source_path: Path,
    preset_path: Path,
    output_root: Path,
    batch_size: int = 50_000,
    density_grid_m: float = 500.0,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if density_grid_m <= 0:
        raise ValueError("density_grid_m must be positive")
    preset_metadata, specs = _load_city_specs(preset_path)
    output_root.mkdir(parents=True, exist_ok=True)
    accumulators = _prepare_accumulators(specs, output_root, batch_size, density_grid_m)

    digest = hashlib.sha256()
    source_feature_count = 0
    property_keys: set[str] = set()
    try:
        with source_path.open("rb") as source:
            for line in source:
                digest.update(line)
                feature = _feature_from_line(line)
                if feature is None:
                    continue
                source_feature_count += 1
                properties = feature.get("properties") or {}
                property_keys.update(str(key) for key in properties)
                geometry_payload = feature.get("geometry") or {}
                coordinates = geometry_payload.get("coordinates")
                if not coordinates:
                    continue
                footprint_bounds = _coordinate_bounds(coordinates)

                candidates = [
                    accumulator
                    for accumulator in accumulators
                    if _bounds_overlap(footprint_bounds, accumulator.bbox)
                ]
                if not candidates:
                    continue
                footprint = shape(geometry_payload)
                point = footprint.representative_point()
                for accumulator in candidates:
                    accumulator.bbox_candidate_count += 1
                    if not accumulator.boundary_prepared.covers(point):
                        continue
                    if not footprint.is_valid:
                        accumulator.invalid_geometry_count += 1
                    release = properties.get("release")
                    capture_dates_range = properties.get("capture_dates_range") or ""
                    accumulator.release_counts[str(release)] += 1
                    if capture_dates_range:
                        accumulator.capture_date_nonempty_count += 1
                    accumulator.pending_records.append(
                        {
                            "building_id": (
                                f"msft_usbf_{accumulator.spec.slug}_{source_feature_count:09d}"
                            ),
                            "source_feature_index": source_feature_count,
                            "source_release": release,
                            "capture_dates_range": capture_dates_range,
                        }
                    )
                    accumulator.pending_geometries.append(footprint)
                    accumulator.accepted_count += 1
                    if len(accumulator.pending_records) >= accumulator.batch_size:
                        _flush_city(accumulator)
    except Exception:
        for accumulator in accumulators:
            if accumulator.output_dir.exists():
                shutil.rmtree(accumulator.output_dir)
        raise

    for accumulator in accumulators:
        _flush_city(accumulator)
        _write_density_grid(accumulator)

    source_sha256 = digest.hexdigest()
    summaries = [
        _write_city_summary(
            accumulator,
            source_path,
            source_sha256,
            source_feature_count,
            sorted(property_keys),
            preset_metadata,
        )
        for accumulator in accumulators
    ]
    manifest = {
        "schema": "evrptw_building_extraction_run_v1",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "source_feature_count": source_feature_count,
        "source_property_keys": sorted(property_keys),
        "cities": summaries,
    }
    (output_root / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
