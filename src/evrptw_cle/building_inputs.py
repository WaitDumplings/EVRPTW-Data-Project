"""Prepare the matched Microsoft--OSM population and its NSI evidence ledger.

All original ordinary-positive NSI records remain in a separate denominator,
including records without a matched population owner. No height or unit cap is
used to define this population. Cached imports preserve their recorded owners;
raw construction calls the current geometry and matching implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import CRS
import shapely

from .customer_geometry import build_geometry
from .customer_matching import match_sources
from .util import sha256_file


_REGISTRIES = ("ms_provenance", "parent_child", "osm_aliases", "osm_overlap_conflicts",
               "ms_osm_conflicts", "osm_duplicate_tags")
_OLD_ADMISSION_COLUMNS = ("volume_assessed", "volume_proxy_m3_per_unit", "volume_excluded",
                          "disposition", "admitted", "operational_units")


def _path(value) -> Path:
    if isinstance(value, dict):
        value = value["path"]
    return Path(value).expanduser().resolve()


def _wgs84(path: Path) -> None:
    metadata = pq.read_schema(path).metadata or {}
    geo = json.loads(metadata.get(b"geo", b"{}"))
    column = geo.get("primary_column", "geometry")
    description = geo.get("columns", {}).get(column, {})
    # GeoParquet's absent CRS defaults to WGS84 longitude/latitude (OGC:CRS84).
    crs = description.get("crs", "OGC:CRS84")
    if crs is None or not CRS.from_user_input(crs).equals(CRS.from_epsg(4326),
                                                       ignore_axis_order=True):
        raise ValueError(f"Building source must have declared WGS84 coordinates: {path}")


def _flag(frame, name):
    return (frame[name].fillna(False).to_numpy(bool) if name in frame else
            np.zeros(len(frame), dtype=bool))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def _input_refs(sources, geometry_cache, city):
    role = "actual_executed_input" if geometry_cache is None else "current_reference_inputs_not_reexecuted"
    source_refs = {key: {"path": str(_path(sources[key])),
                         "sha256": sha256_file(_path(sources[key])), "role": role}
                   for key in ("microsoft", "osm", "nsi")}
    cache_refs = None
    if geometry_cache is not None:
        cache = Path(geometry_cache).expanduser().resolve()
        cache_refs = {name: {"path": str(cache / folder / f"{city}.parquet"),
                            "sha256": sha256_file(cache / folder / f"{city}.parquet")}
                      for name, folder in [("locations", "locations"), ("records", "records")]}
    return source_refs, cache_refs


def _reuse_prepared(output, *, city, sources, axis_r, mode, source_refs, cache_refs):
    """Reuse an unchanged base; changed construction never inherits old reviews."""
    path = output / "preparation_manifest.json"
    if not path.exists():
        existing = ("population.parquet", "source_records.parquet", "assessments.jsonl",
                    "final_buildings.parquet")
        if any((output / name).exists() for name in existing):
            raise ValueError("Existing building outputs lack their preparation manifest; use a fresh output directory")
        return None
    saved = json.loads(path.read_text())
    old_sources = {key: value["sha256"] for key, value in saved.get("sources", {}).items()}
    new_sources = {key: value["sha256"] for key, value in source_refs.items()}
    old_cache = {key: value["sha256"] for key, value in saved.get("cache_inputs", {}).items()}
    new_cache = {key: value["sha256"] for key, value in (cache_refs or {}).items()}
    same = (saved.get("city") == city and saved.get("mode") == mode
            and float(saved.get("axis_r", -1)) == float(axis_r)
            and CRS.from_user_input(saved["area_crs"]).equals(CRS.from_user_input(sources["area_crs"]))
            and old_sources == new_sources and old_cache == new_cache)
    if not same:
        raise ValueError("Building inputs, matching mode or parameters changed; use a fresh output directory to preserve previous AI/human decisions")
    required = [output / "population.parquet", output / "source_records.parquet",
                output / "geometry/full_osm_registry.parquet", output / "geometry/parent_child.parquet"]
    required += [output / "geometry" / name for name in saved.get("geometry_files", [])]
    if any(not file.is_file() for file in required):
        raise ValueError("Prepared building outputs are incomplete; use a fresh output directory instead of rebuilding over previous decisions")
    return {**saved, "reused": True}


def source_osm_blockers(source_metric, source_result, full_osm_metric, states, osm_entity):
    """Preserve original full-building containment evidence without reassigning points."""
    n = len(source_result)
    count = np.zeros(n, dtype=np.int32)
    sole = np.full(n, -1, dtype=np.int64)
    pi, oi = shapely.STRtree(full_osm_metric).query(source_metric, predicate="covered_by")
    count[:] = np.bincount(pi, minlength=n)
    sole[pi] = oi
    reason = np.full(n, "", dtype=object)
    reason[count > 1] = "point_in_multiple_full_osm_objects"
    one = np.flatnonzero(count == 1)
    owner = source_result.owner.to_numpy(np.int64)
    partner = osm_entity[sole[one]]
    state = states[sole[one]]
    cross = (partner >= 0) & (owner[one] >= 0) & (partner != owner[one])
    reason[one[cross]] = "point_in_other_canonical_osm_entity"
    weak = (partner < 0) & np.isin(state, ["nonresidential", "mixed"]) & (owner[one] >= 0)
    reason[one[weak]] = "point_in_weak_nonresidential_or_mixed_osm"
    gap = source_result.polygon_gap_m.to_numpy(float)
    exterior = (partner < 0) & (owner[one] >= 0) & (gap[one] > 0)
    reason[one[exterior & ~weak]] = "inference_point_in_external_osm_object"
    return count, reason


def _save_geometry(bundle, directory):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bundle["paths"].drop(columns="geometry")).to_parquet(
        directory / "path_registry.parquet", index=False)
    pd.DataFrame(bundle["full_osm"].drop(columns=["geometry", "tags"], errors="ignore")).to_parquet(
        directory / "full_osm_registry.parquet", index=False)
    for name in _REGISTRIES:
        bundle[name].to_parquet(directory / f"{name}.parquet", index=False)
    _write_json(directory / "diagnostics.json", bundle.get("diagnostics", {}))


def _raw_inputs(sources, axis_r, geometry_dir):
    area_crs = sources["area_crs"]
    bundle = build_geometry(microsoft_path=_path(sources["microsoft"]),
                            osm_path=_path(sources["osm"]), area_crs=area_crs)
    _save_geometry(bundle, geometry_dir)
    native = gpd.read_parquet(_path(sources["nsi"]))
    values = pd.to_numeric(native.resunits, errors="coerce")
    ordinary = native.occtype.fillna("").str.upper().str.startswith(("RES1", "RES2", "RES3"))
    selected = ordinary & np.isfinite(values) & values.gt(0)
    ns = native.loc[selected].sort_values("fd_id").reset_index(drop=True).copy()
    ns["resunits"] = pd.to_numeric(ns.resunits).astype(float)
    if not ns.fd_id.is_unique:
        raise ValueError("Ordinary-positive NSI records must have unique fd_id")
    result, _, diagnostics = match_sources(bundle["paths"], ns, area_crs, axis_r)
    count, blockers = source_osm_blockers(
        ns.to_crs(area_crs).geometry.array, result,
        bundle["full_osm"].to_crs(area_crs).geometry.array,
        bundle["full_osm"].state.to_numpy(), bundle["full_osm"].osm_to_entity.to_numpy(np.int64))
    # Keep all source attributes, with the matcher's owner/candidate columns.
    original = pd.DataFrame(ns.drop(columns="geometry"))
    records = original.merge(result, on="fd_id", how="left", validate="one_to_one")
    records["raw_resunits"] = records.resunits
    records["native_lon"] = shapely.get_x(ns.geometry.array)
    records["native_lat"] = shapely.get_y(ns.geometry.array)
    records["full_osm_containment_count"] = count
    records["geometry_block_reason"] = blockers
    owner = records.owner.to_numpy(np.int64)
    ids = bundle["entities"].entity_id.to_numpy()
    records["entity_id"] = [ids[i] if i >= 0 else None for i in owner]
    return bundle["entities"], records, diagnostics, len(native)


def _verify_cached_nsi_roster(records, nsi_path):
    """Check the cached ordinary-positive ledger against its current NSI reference.

    A matching roster is not proof that every original cache-generation input
    or source attribute was identical; cached tables lack that full binding.
    """
    columns = ["fd_id", "resunits", "occtype"]
    native_names = set(pq.read_schema(nsi_path).names)
    copied_fields = [name for name in ("ftprntsrc", "ftprntid")
                     if name in records and name in native_names]
    coordinates = [name for name in ("native_lon", "native_lat") if name in records]
    columns += copied_fields + (["geometry"] if coordinates else [])
    native = pd.read_parquet(nsi_path, columns=columns)
    values = pd.to_numeric(native.resunits, errors="coerce")
    ordinary = native.occtype.fillna("").str.upper().str.startswith(("RES1", "RES2", "RES3"))
    original_rows = len(native)
    native = native.loc[ordinary & np.isfinite(values) & values.gt(0)].copy()
    if (not records.fd_id.is_unique or not native.fd_id.is_unique
            or records.fd_id.isna().any() or native.fd_id.isna().any()):
        raise ValueError("Cached NSI roster requires unique original fd_id values")
    cached = records.sort_values("fd_id").reset_index(drop=True)
    native = native.sort_values("fd_id").reset_index(drop=True)
    if not np.array_equal(cached.fd_id.to_numpy(), native.fd_id.to_numpy()):
        raise ValueError("Cached NSI roster differs from configured raw NSI: fd_id")
    original_units = pd.to_numeric(native.resunits, errors="coerce").to_numpy(float)
    cached_units = pd.to_numeric(cached.raw_resunits, errors="coerce").to_numpy(float)
    if not np.array_equal(cached_units, original_units):
        raise ValueError("Cached NSI roster differs from configured raw NSI: resunits")
    for name in ["occtype", *copied_fields]:
        left = cached[name].fillna("").astype(str).to_numpy()
        right = native[name].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Cached NSI roster differs from configured raw NSI: {name}")
    if coordinates:
        points = shapely.from_wkb(native.geometry.to_numpy())
        for name in coordinates:
            current = shapely.get_x(points) if name == "native_lon" else shapely.get_y(points)
            cached_values = pd.to_numeric(cached[name], errors="coerce").to_numpy(float)
            if not np.array_equal(cached_values, current, equal_nan=True):
                raise ValueError(f"Cached NSI roster differs from configured raw NSI: {name}")
    return {"verified": True, "records": len(native), "native_rows": original_rows,
            "fields": ["fd_id", "resunits", "occtype", *coordinates, *copied_fields],
            "full_original_raw_source_binding": "not_independently_verified"}


def _cache_inputs(city, cache, geometry_dir, axis_r, sources):
    cache = Path(cache).expanduser().resolve()
    entities = pd.read_parquet(cache / "locations" / f"{city}.parquet")
    records = pd.read_parquet(cache / "records" / f"{city}.parquet")
    roster = _verify_cached_nsi_roster(records, _path(sources["nsi"]))
    original_geometry = cache / "geometry" / city
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for source in original_geometry.glob("*.parquet"):
        shutil.copy2(source, geometry_dir / source.name)
    summary_path = cache / "summaries" / f"{city}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    saved_r = summary.get("shared_axial_R")
    if saved_r is not None and not np.isclose(float(saved_r), float(axis_r), rtol=0, atol=1e-12):
        raise ValueError("Cached matching uses a different axis_r; rebuild from raw sources")
    _write_json(geometry_dir / "diagnostics.json", {
        "mode": "cached_import", "original_geometry_directory": str(original_geometry),
        "original_geometry_diagnostics": json.loads((original_geometry / "diagnostics.json").read_text())
        if (original_geometry / "diagnostics.json").exists() else {},
    })
    return entities, records, summary.get("matching_diagnostics", {}), roster


def _population(entities, records, city):
    entities = entities.copy().drop(columns=list(_OLD_ADMISSION_COLUMNS), errors="ignore")
    owner = records.owner.to_numpy(np.int64)
    assigned = owner >= 0
    if not entities.entity_id.is_unique or not entities.entity_index.is_unique:
        raise ValueError("Canonical building identities must be unique")
    index = pd.Index(entities.entity_index)
    pos = index.get_indexer(owner[assigned])
    if (pos < 0).any():
        raise ValueError("NSI owner is absent from the complete canonical geometry population")
    n = len(entities)
    values = pd.to_numeric(records.raw_resunits, errors="coerce").to_numpy(float)
    if (not records.fd_id.is_unique or not np.all(np.isfinite(values) & (values > 0))
            or not records.occtype.fillna("").str.upper().str.startswith(("RES1", "RES2", "RES3")).all()):
        raise ValueError("Source denominator must retain unique original ordinary-positive NSI records")
    counts = np.bincount(pos, minlength=n)
    sums = np.bincount(pos, weights=values[assigned], minlength=n)
    blocks = records.geometry_block_reason.fillna("").ne("").to_numpy() if (
        "geometry_block_reason" in records) else np.zeros(len(records), bool)
    group_conflict = _flag(records, "source_group_conflict")
    if "base_direct_owner" in records:
        scope = (_flag(records, "source_geometry_counterevidence") & assigned
                 & (owner == records.base_direct_owner.to_numpy(np.int64)))
    else:
        scope = _flag(records, "final_source_geometry_units_scope_conflict") & assigned
    records = records.copy()
    records["final_source_geometry_units_scope_conflict"] = scope
    entities["nsi_count"] = counts
    entities["raw_units_sum"] = sums
    for name, flag in [("geometry_block_records", blocks),
                       ("source_group_conflict_records", group_conflict),
                       ("source_geometry_units_scope_conflict_records", scope)]:
        entities[name] = np.bincount(pos, weights=flag[assigned], minlength=n).astype(np.int64)
    matched = ((entities.kind.eq("ms") & entities.trusted_osm_alias_group.ge(0))
               | entities.kind.eq("split_osm"))
    pop = entities.loc[matched].copy()
    pop["city"] = city
    pop["matched_ms_osm"] = True
    area = pd.to_numeric(pop.area_m2, errors="coerce").to_numpy(float)
    pop["geometry_clear"] = (_flag(pop, "eligible") & ~_flag(pop, "geometry_conflict")
                             & ~_flag(pop, "weak_semantic_conflict")
                             & np.isfinite(area) & (area > 0))
    pop["osm_use"] = pop.semantic_state
    pop["residential_use"] = pop.osm_use
    pop["residential_source"] = np.where(pop.osm_use.eq("unknown"), "unresolved", "osm")
    one = pop.nsi_count.eq(1)
    pop["units_raw_nsi"] = pop.raw_units_sum.where(one)
    valid_units = (one & pop.raw_units_sum.gt(0) & np.isfinite(pop.raw_units_sum)
                   & pop.raw_units_sum.eq(np.floor(pop.raw_units_sum)))
    clear = (pop.geometry_clear & valid_units & pop.geometry_block_records.eq(0)
             & pop.source_group_conflict_records.eq(0)
             & pop.source_geometry_units_scope_conflict_records.eq(0))
    pop["nsi_attribution_clear"] = clear
    pop["nsi_scope"] = np.where(clear, "building", "unresolved")
    pop["units_final"] = pop.units_raw_nsi.where(clear)
    pop["units_method"] = np.where(clear, "nsi_retained", "unresolved")
    pop["units_scope"] = pop.nsi_scope
    pop["units_review_status"] = np.where(clear, "not_flagged", "unresolved")
    pop["units_independently_verified"] = False
    records["owner_in_population"] = records.owner.isin(pop.entity_index)
    return pop.reset_index(drop=True), records


def prepare_building_population(*, city, sources, output_dir, axis_r, geometry_cache=None):
    """Prepare a complete matched-building denominator plus original NSI records.

    ``geometry_cache`` is an explicit import of saved construction, not a claim
    to have rerun the current raw builder. Its source competition/owners remain
    unchanged, including any previously selected standalone OSM competitors.
    """
    output = Path(output_dir).expanduser().resolve()
    mode = "raw_geometry_and_matching" if geometry_cache is None else "cached_geometry_and_owners_import"
    source_refs, cache_refs = _input_refs(sources, geometry_cache, city)
    reused = _reuse_prepared(output, city=city, sources=sources, axis_r=axis_r,
                             mode=mode, source_refs=source_refs, cache_refs=cache_refs)
    if reused is not None:
        return reused
    output.mkdir(parents=True, exist_ok=True)
    geometry_dir = output / "geometry"
    for key in ("microsoft", "osm", "nsi"):
        _wgs84(_path(sources[key]))
    if geometry_cache is None:
        entities, records, diagnostics, native_rows = _raw_inputs(sources, axis_r, geometry_dir)
        roster = None
    else:
        entities, records, diagnostics, roster = _cache_inputs(
            city, geometry_cache, geometry_dir, axis_r, sources)
        native_rows = roster["native_rows"]
    population, records = _population(entities, records, city)
    population.to_parquet(output / "population.parquet", index=False)
    records.to_parquet(output / "source_records.parquet", index=False)
    manifest = {
        "schema": "matched_building_population_v1", "city": city, "mode": mode,
        "population_path": str(output / "population.parquet"),
        "source_records_path": str(output / "source_records.parquet"),
        "geometry_dir": str(geometry_dir), "area_crs": str(sources["area_crs"]),
        "geometry_files": sorted(path.name for path in geometry_dir.glob("*.parquet")),
        "axis_r": float(axis_r), "sources": source_refs,
        "geometry_cache": str(Path(geometry_cache).resolve()) if geometry_cache is not None else None,
        "population_buildings": len(population), "geometry_clear_buildings": int(population.geometry_clear.sum()),
        "ordinary_positive_nsi_records": len(records),
        "ordinary_positive_nsi_units": float(records.raw_resunits.sum()),
        "nsi_records_with_population_owner": int(records.owner_in_population.sum()),
        "native_nsi_rows": native_rows, "height_admission_gate": False,
        "unit_cap_applied_to_population": False,
        "matching_competitor_population": ("recorded cached canonical population" if geometry_cache is not None
                                            else "Global MS and OSM subdivision children; no standalone OSM additions"),
        "matching_diagnostics": diagnostics,
    }
    if cache_refs is not None:
        manifest["cache_inputs"] = cache_refs
        manifest["NSI_roster_verified"] = True
        manifest["cache_nsi_roster_check"] = roster
        manifest["cache_original_raw_source_binding"] = "not_independently_verified"
    _write_json(output / "preparation_manifest.json", manifest)
    return manifest


class BuildingEvidenceReader:
    """Read original tags and polygons only for requested fixed building IDs."""

    def __init__(self, sources, geometry_dir):
        self.sources = sources
        directory = Path(geometry_dir)
        self.geometry_dir = directory
        self.registry = pd.read_parquet(directory / "full_osm_registry.parquet").set_index("osm_group")
        self.parents = pd.read_parquet(directory / "parent_child.parquet")
        self._osm_objects = {}

    def _osm(self, ids):
        missing = sorted(set(ids) - self._osm_objects.keys())
        if missing:
            raw = pd.read_parquet(_path(self.sources["osm"]),
                                  columns=["osm_id", "building", "tags", "geometry"],
                                  filters=[("osm_id", "in", missing)])
            for row in raw.itertuples(index=False):
                self._osm_objects[str(row.osm_id)] = row._asdict()
        return [self._osm_objects[object_id] for object_id in ids]

    def payload(self, row):
        """Return only location/OSM evidence; the caller adds NSI for unit tasks."""
        row = dict(row)
        retained = row["kind"] == "ms"
        group_id = int(row["trusted_osm_alias_group"] if retained else row["osm_group"])
        group = self.registry.loc[group_id]
        ids = sorted(set(str(group.osm_ids).split("|")))
        raw = self._osm(ids)
        canonical = self._osm([str(group.canonical_id)])[0]
        geometry = shapely.from_wkb(canonical["geometry"])
        point = shapely.point_on_surface(geometry)
        ms_ids = ([row["entity_id"]] if retained else sorted(set(
            self.parents.loc[self.parents.entity_index.eq(row["entity_index"]), "parent_ms_id"])))
        if not ms_ids:
            raise ValueError("Matched building is missing Microsoft parent provenance")
        osm_objects = []
        for obj in raw:
            tags = json.loads(obj["tags"]) if isinstance(obj["tags"], str) else dict(obj["tags"] or {})
            osm_objects.append({"osm_id": obj["osm_id"], "building": obj["building"], "tags": tags})
        payload = {
            "entity_id": row["entity_id"], "microsoft_building_ids": ms_ids,
            "matching_kind": "retained_ms_trusted_osm_alias" if retained else "osm_split_child_with_ms_parent",
            "longitude": float(point.x), "latitude": float(point.y),
            "coordinate_source": "point_on_surface_of_matched_canonical_osm_building_wgs84",
            "coordinate_osm_id": str(group.canonical_id), "osm_objects": osm_objects,
            "building_features": {"canonical_geometry_area_m2": float(row["area_m2"]),
                                  "osm_bbox_wgs84": list(geometry.bounds)},
        }
        if self.sources.get("snapshots"):
            payload["source_snapshots"] = self.sources["snapshots"]
        return payload

    def geometry(self, entityrows):
        """Recover canonical final polygons without substituting alias geometries."""
        frame = pd.DataFrame(entityrows).copy().reset_index(drop=True)
        if frame.empty:
            return gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries([], crs=4326), crs=4326)
        ms_ids = frame.loc[frame.kind.eq("ms"), "primary_path_id"].astype(str).tolist()
        osm_ids = frame.loc[~frame.kind.eq("ms"), "primary_path_id"].astype(str).tolist()
        mapping = {}
        if ms_ids:
            native = pd.read_parquet(_path(self.sources["microsoft"]), columns=["building_id", "geometry"],
                                     filters=[("building_id", "in", ms_ids)])
            mapping.update(zip(native.building_id.astype(str), native.geometry))
        if osm_ids:
            mapping.update({obj["osm_id"]: obj["geometry"] for obj in self._osm(osm_ids)})
        polygons = shapely.from_wkb(np.asarray([mapping[str(x)] for x in frame.primary_path_id], dtype=object))
        return gpd.GeoDataFrame(frame, geometry=polygons, crs=4326)

    def obstacles(self):
        """Recover all active physical footprints, including nonresidential ones.

        Retired Microsoft parents are absent from primary paths, so they cannot
        obstruct frontage access to their own replacement OSM children. Alias
        polygons are not added a second time. Cached standalone OSM additions
        are outside this method's Microsoft-based geometry population.
        """
        paths = pd.read_parquet(self.geometry_dir / "path_registry.parquet",
                                columns=["path_id", "entity_index", "path_kind"])
        ms = paths.path_kind.eq("ms_primary")
        child_ids = set(self.registry.loc[self.registry.is_split_child, "canonical_id"].astype(str))
        children = paths.path_kind.eq("osm_primary") & paths.path_id.astype(str).isin(child_ids)
        frame = paths.loc[ms | children].copy().reset_index(drop=True)
        frame["entity_id"] = frame.path_id
        frame["building_id"] = frame.path_id
        frame["primary_path_id"] = frame.path_id
        frame["kind"] = np.where(frame.path_kind.eq("ms_primary"), "ms", "split_osm")
        return self.geometry(frame)
