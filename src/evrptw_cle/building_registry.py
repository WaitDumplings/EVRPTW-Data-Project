"""Single-city Microsoft building extraction registry and staged runner."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyogrio.errors import DataSourceError
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.errors import GEOSException

from .buildings import _feature_from_line, extract_building_footprints
from .util import sha256_file, write_json

EXPECTED_SCHEMA = "evrptw_city_building_registry_v1"
EXPECTED_PROPERTIES = {"capture_dates_range", "release"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _first_source_feature(source_path: Path) -> dict[str, Any] | None:
    with source_path.open("rb") as handle:
        for line in handle:
            feature = _feature_from_line(line)
            if feature is not None:
                return feature
    return None


def _load_and_validate_registry(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_json(config_path)
    errors: list[str] = []
    if payload.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"unsupported registry schema {payload.get('schema')!r}")
    cities = payload.get("cities")
    if not isinstance(cities, dict) or not cities:
        errors.append("registry cities must be a nonempty object")
        return payload, errors

    official_value = payload.get("official_city_preset")
    if not official_value:
        errors.append("official_city_preset is missing")
        return payload, errors
    official_path = _resolve_from_config(config_path, str(official_value))
    if not official_path.exists():
        errors.append(f"official city preset is missing: {official_path}")
        return payload, errors
    official = _read_json(official_path)
    official_slugs = {str(item["slug"]) for item in official.get("cities", [])}
    registry_slugs = set(cities)
    if registry_slugs != official_slugs:
        errors.append(
            "registry city set differs from official preset: "
            f"missing={sorted(official_slugs - registry_slugs)}, "
            f"extra={sorted(registry_slugs - official_slugs)}"
        )

    integrity_mode = str(payload.get("source_integrity_mode", "frozen_registry"))
    if integrity_mode not in {"frozen_registry", "record_on_extraction"}:
        errors.append(f"unsupported source_integrity_mode {integrity_mode!r}")

    required = {
        "label",
        "state",
        "state_fips",
        "source_file",
        "source_sha256",
        "source_bytes",
        "source_feature_count",
        "boundary_file",
        "boundary_sha256",
        "area_crs",
    }
    for slug, entry in cities.items():
        if not isinstance(entry, dict):
            errors.append(f"{slug}: registry entry is not an object")
            continue
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{slug}: missing registry fields {missing}")
        source_sha256 = entry.get("source_sha256")
        source_bytes = entry.get("source_bytes")
        source_feature_count = entry.get("source_feature_count")
        if integrity_mode == "frozen_registry":
            if len(str(source_sha256 or "")) != 64:
                errors.append(f"{slug}: source_sha256 is not a full SHA-256")
            if not isinstance(source_bytes, int) or source_bytes <= 0:
                errors.append(f"{slug}: source_bytes must be a positive integer")
            if not isinstance(source_feature_count, int) or source_feature_count <= 0:
                errors.append(f"{slug}: source_feature_count must be a positive integer")
        else:
            if source_sha256 is not None and len(str(source_sha256)) != 64:
                errors.append(f"{slug}: optional source_sha256 is invalid")
            if source_bytes is not None and (
                not isinstance(source_bytes, int) or source_bytes <= 0
            ):
                errors.append(f"{slug}: optional source_bytes must be positive")
            if source_feature_count is not None and (
                not isinstance(source_feature_count, int) or source_feature_count <= 0
            ):
                errors.append(f"{slug}: optional source_feature_count must be positive")
        if len(str(entry.get("boundary_sha256", ""))) != 64:
            errors.append(f"{slug}: boundary_sha256 is not a full SHA-256")
        try:
            crs = CRS.from_user_input(entry.get("area_crs"))
            if not crs.is_projected:
                errors.append(f"{slug}: area_crs is not projected")
            units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
            if units and not all("metre" in unit or "meter" in unit for unit in units):
                errors.append(f"{slug}: area_crs axes are not measured in metres")
        except (CRSError, ValueError) as exc:
            errors.append(f"{slug}: invalid area_crs: {exc}")
    return payload, errors


def preflight_registered_city(
    *,
    config_path: Path,
    city_slug: str,
    source_root: Path,
    output_root: Path,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    """Validate one registry entry without writing building outputs."""

    config_path = config_path.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    payload, errors = _load_and_validate_registry(config_path)
    entry = payload.get("cities", {}).get(city_slug)
    if entry is None:
        errors.append(f"city {city_slug!r} is not present in the building registry")
        return {
            "schema": "evrptw_city_building_preflight_v1",
            "city_slug": city_slug,
            "passed": False,
            "errors": errors,
        }

    source_path = source_root / str(entry["source_file"])
    boundary_path = _resolve_from_config(config_path, str(entry["boundary_file"]))
    final_city_dir = output_root / city_slug
    checks: dict[str, Any] = {
        "source_exists": source_path.exists(),
        "boundary_exists": boundary_path.exists(),
        "final_output_exists": final_city_dir.exists(),
    }
    if not source_path.exists():
        errors.append(f"source file is missing: {source_path}")
    else:
        actual_bytes = source_path.stat().st_size
        checks["source_bytes"] = actual_bytes
        expected_bytes = entry.get("source_bytes")
        if expected_bytes is not None and actual_bytes != int(expected_bytes):
            errors.append(f"source byte size {actual_bytes} differs from registry {expected_bytes}")
        first = _first_source_feature(source_path)
        if first is None:
            errors.append("source file contains no line-oriented GeoJSON Feature")
        else:
            geometry_type = (first.get("geometry") or {}).get("type")
            property_keys = set((first.get("properties") or {}).keys())
            checks["first_feature_geometry_type"] = geometry_type
            checks["first_feature_property_keys"] = sorted(property_keys)
            if geometry_type != payload.get("expected_geometry_type"):
                errors.append(
                    f"first feature geometry is {geometry_type!r}; expected "
                    f"{payload.get('expected_geometry_type')!r}"
                )
            if property_keys != EXPECTED_PROPERTIES:
                errors.append(
                    f"first feature properties {sorted(property_keys)} differ from "
                    f"{sorted(EXPECTED_PROPERTIES)}"
                )
        expected_source_sha = entry.get("source_sha256")
        if verify_source_hash and expected_source_sha:
            actual_source_sha = sha256_file(source_path)
            checks["source_sha256"] = actual_source_sha
            if actual_source_sha != expected_source_sha:
                errors.append("source SHA-256 differs from the frozen registry")
        else:
            checks["source_sha256"] = "deferred_to_staged_extraction"

    if not boundary_path.exists():
        errors.append(f"boundary file is missing: {boundary_path}")
    else:
        actual_boundary_sha = sha256_file(boundary_path)
        checks["boundary_sha256"] = actual_boundary_sha
        if actual_boundary_sha != entry["boundary_sha256"]:
            errors.append("boundary SHA-256 differs from the frozen registry")
        try:
            boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
            geometry = boundary.geometry.union_all()
            checks["boundary_valid"] = bool(geometry.is_valid)
            checks["boundary_bounds_wgs84"] = [float(value) for value in geometry.bounds]
            if geometry.is_empty or not geometry.is_valid:
                errors.append("frozen land boundary is empty or invalid")
        except (DataSourceError, GEOSException, OSError, ValueError) as exc:
            errors.append(f"frozen land boundary cannot be read: {exc}")

    return {
        "schema": "evrptw_city_building_preflight_v1",
        "registry_id": payload.get("registry_id"),
        "city_slug": city_slug,
        "passed": not errors,
        "errors": errors,
        "resolved": {
            "config_path": str(config_path),
            "source_path": str(source_path.resolve()),
            "boundary_path": str(boundary_path.resolve()),
            "final_city_dir": str(final_city_dir),
            "area_crs": entry["area_crs"],
        },
        "checks": checks,
    }


def _output_hashes(city_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(city_dir)): sha256_file(path)
        for path in sorted(city_dir.rglob("*"))
        if path.is_file()
    }


def extract_registered_city(
    *,
    config_path: Path,
    city_slug: str,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Extract one city in staging, validate it, then publish atomically."""

    config_path = config_path.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    payload, registry_errors = _load_and_validate_registry(config_path)
    if registry_errors:
        raise ValueError("; ".join(registry_errors))
    if city_slug not in payload["cities"]:
        raise KeyError(f"Unknown registry city {city_slug!r}")
    entry = payload["cities"][city_slug]
    source_path = source_root / str(entry["source_file"])
    boundary_path = _resolve_from_config(config_path, str(entry["boundary_file"]))
    final_city_dir = output_root / city_slug
    final_manifest = output_root / "manifests" / f"{city_slug}.json"
    if final_city_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing city output: {final_city_dir}")
    if final_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing run manifest: {final_manifest}")

    preflight = preflight_registered_city(
        config_path=config_path,
        city_slug=city_slug,
        source_root=source_root,
        output_root=output_root,
        verify_source_hash=False,
    )
    if not preflight["passed"]:
        raise ValueError("Preflight failed: " + "; ".join(preflight["errors"]))

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent = output_root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{city_slug}-", dir=staging_parent) as temporary:
        temporary_path = Path(temporary)
        staging_output = temporary_path / "output"
        one_city_preset = temporary_path / "city_preset.json"
        write_json(
            one_city_preset,
            {
                "schema": "evrptw_building_city_preset_v1",
                "source_dataset": f"{payload['source_dataset']} {entry['state']}",
                "source_id": f"microsoft_usbf_{entry['state_fips']}",
                "cities": [
                    {
                        "slug": city_slug,
                        "label": entry["label"],
                        "boundary_file": str(boundary_path),
                        "area_crs": entry["area_crs"],
                    }
                ],
            },
        )
        extraction = extract_building_footprints(
            source_path=source_path,
            preset_path=one_city_preset,
            output_root=staging_output,
            batch_size=int(payload.get("batch_size", 50_000)),
            density_grid_m=float(payload.get("density_grid_m", 500.0)),
        )
        validation_errors: list[str] = []
        expected_source_sha = entry.get("source_sha256")
        expected_feature_count = entry.get("source_feature_count")
        if expected_source_sha and extraction.get("source_sha256") != expected_source_sha:
            validation_errors.append("extracted source SHA-256 differs from registry")
        if expected_feature_count is not None and extraction.get("source_feature_count") != int(
            expected_feature_count
        ):
            validation_errors.append("extracted source feature count differs from registry")
        if set(extraction.get("source_property_keys", [])) != EXPECTED_PROPERTIES:
            validation_errors.append("extracted source property keys differ from registry")
        summaries = extraction.get("cities", [])
        if len(summaries) != 1 or summaries[0].get("city_slug") != city_slug:
            validation_errors.append("staged extraction did not produce exactly the requested city")
        elif int(summaries[0].get("building_count", 0)) <= 0:
            validation_errors.append("staged extraction produced no city buildings")
        staged_city_dir = staging_output / city_slug
        if not staged_city_dir.exists():
            validation_errors.append("staged city output directory is missing")
        if validation_errors:
            raise RuntimeError(
                "Staged extraction validation failed: " + "; ".join(validation_errors)
            )

        output_hashes = _output_hashes(staged_city_dir)
        run_manifest = {
            "schema": "evrptw_city_building_extraction_run_v1",
            "status": "complete",
            "generated_utc": datetime.now(UTC).isoformat(),
            "registry": {
                "id": payload["registry_id"],
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "city_slug": city_slug,
            "state": entry["state"],
            "source": {
                "file": entry["source_file"],
                "path": str(source_path.resolve()),
                "sha256": extraction["source_sha256"],
                "bytes": source_path.stat().st_size,
                "feature_count": extraction["source_feature_count"],
                "property_keys": extraction["source_property_keys"],
                "integrity_mode": payload.get("source_integrity_mode", "frozen_registry"),
            },
            "boundary": {
                "path": str(boundary_path),
                "sha256": sha256_file(boundary_path),
            },
            "semantics": {
                "membership_rule": payload["membership_rule"],
                "polygon_policy": payload["polygon_policy"],
                "building_id_rule": payload["building_id_rule"],
            },
            "summary": summaries[0],
            "output_file_sha256": output_hashes,
        }
        staged_manifest = temporary_path / f"{city_slug}.json"
        write_json(staged_manifest, run_manifest)

        manifests_dir = output_root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        staged_city_dir.replace(final_city_dir)
        try:
            staged_manifest.replace(final_manifest)
        except OSError:
            final_city_dir.replace(staged_city_dir)
            raise

    return run_manifest
