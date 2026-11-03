"""Canonical Global Microsoft buildings, OSM subdivision and identity provenance."""

from __future__ import annotations

import json
import re
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

STATE_CODE = {"unknown": 0, "residential": 1, "nonresidential": 2, "mixed": 3}
STATE_NAME = np.array(["unknown", "residential", "nonresidential", "mixed"], dtype=object)


def explicit_height_m(value):
    """Same explicit single-length parser as previous volume analysis; no levels."""
    if value is None:
        return np.nan
    s = str(value).strip().lower().replace("′", "'").replace("″", '"')
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m|metres?|meters?)?", s)
    if m:
        v = float(m.group(1))
        return v if v > 0 else np.nan
    m = re.fullmatch(
        r'([0-9]+(?:\.[0-9]+)?)\s*(?:ft|feet|foot)(?:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:in|inches|inch|"))?',
        s,
    )
    if m:
        return float(m.group(1)) * 0.3048 + float(m.group(2) or 0) * 0.0254
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*'(?:\s*([0-9]+(?:\.[0-9]+)?)\s*\")?", s)
    return float(m.group(1)) * 0.3048 + float(m.group(2) or 0) * 0.0254 if m else np.nan


def _exact_groups(geometry, ids):
    codes, _ = pd.factorize(shapely.to_wkb(shapely.normalize(geometry)))
    _, first = np.unique(codes, return_index=True)
    canonical = np.asarray(ids, dtype=object)[first].copy()
    all_ids = canonical.copy()
    counts = np.bincount(codes, minlength=len(first))
    dup = counts[codes] > 1
    if dup.any():
        frame = pd.DataFrame({"group": codes[dup], "id": np.asarray(ids, dtype=object)[dup]})
        for group, sub in frame.groupby("group", sort=False):
            values = sorted(set(sub.id.astype(str)))
            canonical[int(group)] = values[0]
            all_ids[int(group)] = "|".join(values)
    return codes, first, canonical, all_ids, counts


def _positive_pairs(target_geometry, query_geometry, chunk=200000):
    """Return query index, target index and positive intersection area."""
    qi, ti = STRtree(target_geometry).query(query_geometry, predicate="intersects")
    area = np.empty(len(qi), dtype=float)
    for start in range(0, len(qi), chunk):
        sl = slice(start, start + chunk)
        area[sl] = shapely.area(
            shapely.intersection(query_geometry.take(qi[sl]), target_geometry.take(ti[sl]))
        )
    keep = area > 0
    return qi[keep], ti[keep], area[keep]


def _split_closure(nms, nosm, mi, oi, seed, alias_m, alias_o):
    """Seed on 2+ substantial children, propagate over identity-supported edges."""
    parents = np.arange(nms + nosm, dtype=np.int64)
    rank = np.zeros(nms + nosm, dtype=np.uint8)

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return int(x)

    def union(a, b):
        a, b = find(int(a)), find(int(b))
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parents[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    for m, o in zip(mi, oi):
        union(m, nms + o)
    for m, o in zip(alias_m, alias_o):
        union(m, nms + o)
    if not seed.any():
        return np.zeros(nms, bool), np.zeros(nosm, bool)
    seed_roots = set(find(x) for x in np.flatnonzero(seed))
    involved_m = np.unique(np.concatenate([mi, alias_m]))
    involved_o = np.unique(np.concatenate([oi, alias_o]))
    retired = np.zeros(nms, bool)
    active_osm = np.zeros(nosm, bool)
    retired[involved_m] = [find(int(x)) in seed_roots for x in involved_m]
    active_osm[involved_o] = [find(nms + int(x)) in seed_roots for x in involved_o]
    return retired, active_osm


def build_geometry(*, microsoft_path, osm_path, area_crs):
    """Build all competing MS entities and OSM children, without OSM-only additions.

    NSI matching and candidate eligibility are later stages. Only Global geometry
    is read, and the original MS/OSM polygons are never clipped to each other.
    """
    from . import customer_use as policy

    started = time.monotonic()
    metric_crs = area_crs

    original_ms = gpd.read_parquet(
        microsoft_path,
        columns=["building_id", "source_kind", "height", "geometry"],
        filters=[("source_kind", "=", "global")],
    ).reset_index(drop=True)
    ms_code, ms_first, ms_ids, all_ms_ids, ms_count = _exact_groups(
        original_ms.geometry.array, original_ms.building_id.to_numpy()
    )
    ms_wgs = original_ms.geometry.array.take(ms_first)
    ms_h = pd.to_numeric(original_ms.height, errors="coerce").to_numpy(float)
    ms_h[(~np.isfinite(ms_h)) | (ms_h <= 0)] = np.nan
    ms_height = np.full(len(ms_first), np.nan)
    np.fmax.at(ms_height, ms_code, ms_h)

    original_osm = gpd.read_parquet(
        osm_path, columns=["osm_id", "building", "building_part", "tags", "geometry"]
    )
    original_osm = original_osm.loc[~original_osm.building_part.fillna(False)].reset_index(
        drop=True
    )
    osm_code, osm_first, osm_ids, all_osm_ids, osm_count = _exact_groups(
        original_osm.geometry.array, original_osm.osm_id.to_numpy()
    )
    osm_wgs = original_osm.geometry.array.take(osm_first)
    state = np.zeros(len(osm_first), dtype=np.uint8)
    osm_height = np.full(len(osm_first), np.nan)
    raw_state, raw_h = [], []
    for building, raw in zip(original_osm.building, original_osm.tags):
        tags = json.loads(raw) if isinstance(raw, str) else (raw or {})
        raw_state.append(STATE_CODE[policy.classify_tags(tags, building)["state"]])
        raw_h.append(explicit_height_m(tags.get("height")))
    np.bitwise_or.at(state, osm_code, np.asarray(raw_state, dtype=np.uint8))
    np.fmax.at(osm_height, osm_code, np.asarray(raw_h, dtype=float))

    mg = gpd.GeoSeries(ms_wgs, crs=4326).to_crs(metric_crs).array
    og = gpd.GeoSeries(osm_wgs, crs=4326).to_crs(metric_crs).array
    ma, oa = shapely.area(mg), shapely.area(og)
    nm, no = len(mg), len(og)
    oi, mi, inter = _positive_pairs(mg, og)
    iou = inter / (oa[oi] + ma[mi] - inter)
    strong = iou >= 0.5
    # r9 identity degree is measured in the COMPLETE positive-overlap graph;
    # IoU is a qualification on the pair, not a way to remove competitors.
    positive_md = np.bincount(mi, minlength=nm)
    positive_od = np.bincount(oi, minlength=no)
    alias_mask = strong & (positive_md[mi] == 1) & (positive_od[oi] == 1)
    alias_m, alias_o = mi[alias_mask], oi[alias_mask]
    substantial = inter / oa[oi] >= 0.5
    at = np.flatnonzero(substantial)
    rp = shapely.point_on_surface(og)
    at = at[shapely.covers(mg.take(mi[at]), rp.take(oi[at]))]
    sub_m, sub_o = mi[at], oi[at]
    sub_md = np.bincount(sub_m, minlength=nm)
    sub_od = np.bincount(sub_o, minlength=no)
    seed = sub_md >= 2
    retired, split_osm = _split_closure(nm, no, sub_m, sub_o, seed, alias_m, alias_o)
    has_overlap = np.zeros(no, dtype=bool)
    has_overlap[oi] = True

    # The current customer population is the MS–OSM matched intersection.
    # Keep unmatched MS as NSI competitors, never add standalone OSM entities.
    previous_add = pd.DataFrame({"osm_id": pd.Series(dtype=str)})
    addition = np.zeros(no, dtype=bool)
    metric_overlapping_additions = 0
    selected_osm = split_osm | addition
    kept_ms = np.flatnonzero(~retired)
    chosen_osm = np.flatnonzero(selected_osm)
    ms_to_entity = np.full(nm, -1, dtype=np.int64)
    osm_to_entity = np.full(no, -1, dtype=np.int64)
    ms_to_entity[kept_ms] = np.arange(len(kept_ms))
    osm_to_entity[chosen_osm] = np.arange(len(kept_ms), len(kept_ms) + len(chosen_osm))

    retained_alias = ~retired[alias_m]
    kept_alias_m, kept_alias_o = alias_m[retained_alias], alias_o[retained_alias]
    assert not selected_osm[kept_alias_o].any(), (
        "A retained MS alias duplicates an active OSM child"
    )
    osm_to_entity[kept_alias_o] = ms_to_entity[kept_alias_m]
    aliases_by_ms = np.full(nm, -1, dtype=np.int64)
    aliases_by_ms[kept_alias_m] = kept_alias_o
    alias_state = np.zeros(nm, dtype=np.uint8)
    alias_state[kept_alias_m] = state[kept_alias_o]
    ms_entity_height = ms_height.copy()
    ms_entity_height[kept_alias_m] = np.fmax(
        ms_entity_height[kept_alias_m], osm_height[kept_alias_o]
    )

    # A single weak OSM N/M support is pending, never an inherited whole-MS use.
    weak = (~retired[sub_m]) & (aliases_by_ms[sub_m] != sub_o)
    weak_state = np.zeros(nm, dtype=np.uint8)
    np.bitwise_or.at(weak_state, sub_m[weak], state[sub_o[weak]])
    weak_conflict = (weak_state >= 2) & ~retired
    identity_ms = np.zeros(nm, dtype=bool)
    shared_without_split = (sub_od[sub_o] > 1) & ~retired[sub_m]
    identity_ms[sub_m[shared_without_split]] = True
    bad_strong = strong & ~alias_mask & ~retired[mi]
    identity_ms[mi[bad_strong]] = True

    entity_ids = np.concatenate([ms_ids[kept_ms], osm_ids[chosen_osm]])
    entity_n = len(entity_ids)
    entity_state = np.concatenate([alias_state[kept_ms], state[chosen_osm]])
    sub_keys = sub_m.astype(np.int64) * no + sub_o
    alias_keys = alias_m.astype(np.int64) * no + alias_o
    unique_relationships = np.unique(np.concatenate([sub_keys, alias_keys]))
    parent_counts = np.bincount(unique_relationships % no, minlength=no)
    entities = pd.DataFrame(
        {
            "entity_index": np.arange(entity_n),
            "entity_id": entity_ids,
            "kind": np.concatenate(
                [
                    np.full(len(kept_ms), "ms", dtype=object),
                    np.where(split_osm[chosen_osm], "split_osm", "osm_add"),
                ]
            ),
            "semantic_state": STATE_NAME[entity_state],
            "eligible": True,
            "geometry_conflict": np.concatenate(
                [identity_ms[kept_ms], np.zeros(len(chosen_osm), dtype=bool)]
            ),
            "weak_semantic_conflict": np.concatenate(
                [weak_conflict[kept_ms], np.zeros(len(chosen_osm), dtype=bool)]
            ),
            "weak_semantic_state": np.concatenate(
                [STATE_NAME[weak_state[kept_ms]], np.full(len(chosen_osm), "unknown", dtype=object)]
            ),
            "height_m": np.concatenate([ms_entity_height[kept_ms], osm_height[chosen_osm]]),
            "area_m2": np.concatenate([ma[kept_ms], oa[chosen_osm]]),
            "primary_path_id": entity_ids.copy(),
            "parent_count": np.concatenate([ms_count[kept_ms], parent_counts[chosen_osm]]),
            "is_original_osm_addition": np.concatenate(
                [np.zeros(len(kept_ms), bool), addition[chosen_osm]]
            ),
            "ms_group": np.concatenate([kept_ms, np.full(len(chosen_osm), -1, dtype=np.int64)]),
            "osm_group": np.concatenate([np.full(len(kept_ms), -1, dtype=np.int64), chosen_osm]),
            "trusted_osm_alias_group": np.concatenate(
                [aliases_by_ms[kept_ms], np.full(len(chosen_osm), -1, dtype=np.int64)]
            ),
        }
    )

    # OSM conflicts check all full objects, including unselected/nested parents.
    check_o = np.unique(np.concatenate([chosen_osm, kept_alias_o]))
    q, neighbor, ointer = _positive_pairs(og, og.take(check_o))
    target_o = check_o[q]
    not_self = target_o != neighbor
    target_o, neighbor, ointer = target_o[not_self], neighbor[not_self], ointer[not_self]
    hit_entities = osm_to_entity[target_o]
    hit_entities = hit_entities[hit_entities >= 0]
    entities.loc[np.unique(hit_entities), "geometry_conflict"] = True
    # Active children cannot coexist unexamined with other, unretired MS geometry.
    mixed_conflict = (~retired[mi]) & selected_osm[oi]
    mc_m, mc_o, mc_a = mi[mixed_conflict], oi[mixed_conflict], inter[mixed_conflict]
    if len(mc_m):
        entities.loc[
            np.unique(np.concatenate([ms_to_entity[mc_m], osm_to_entity[mc_o]])),
            "geometry_conflict",
        ] = True

    path_ids = np.concatenate([ms_ids[kept_ms], osm_ids[chosen_osm], osm_ids[kept_alias_o]])
    path_entities = np.concatenate(
        [ms_to_entity[kept_ms], osm_to_entity[chosen_osm], ms_to_entity[kept_alias_m]]
    )
    path_geom = np.concatenate(
        [ms_wgs.take(kept_ms), osm_wgs.take(chosen_osm), osm_wgs.take(kept_alias_o)]
    )
    paths = gpd.GeoDataFrame(
        {
            "path_id": path_ids,
            "entity_index": path_entities,
            "path_kind": np.concatenate(
                [
                    np.full(len(kept_ms), "ms_primary", object),
                    np.full(len(chosen_osm), "osm_primary", object),
                    np.full(len(kept_alias_o), "osm_alias", object),
                ]
            ),
            "height_m": np.concatenate(
                [ms_height[kept_ms], osm_height[chosen_osm], osm_height[kept_alias_o]]
            ),
            "area_m2": np.concatenate([ma[kept_ms], oa[chosen_osm], oa[kept_alias_o]]),
        },
        geometry=path_geom,
        crs=4326,
    )
    assert paths.path_id.is_unique and entities.entity_id.is_unique

    full_osm = gpd.GeoDataFrame(
        {
            "osm_group": np.arange(no),
            "osm_ids": all_osm_ids,
            "canonical_id": osm_ids,
            "state": STATE_NAME[state],
            "osm_to_entity": osm_to_entity,
            "height_m": osm_height,
            "area_m2": oa,
            "geometry_duplicate_count": osm_count,
            "building": original_osm.building.to_numpy()[osm_first],
            "tags": original_osm.tags.to_numpy()[osm_first],
            "is_split_child": split_osm,
            "is_old_addition": addition,
        },
        geometry=osm_wgs,
        crs=4326,
    )
    all_ms = gpd.GeoDataFrame(
        {
            "ms_group": np.arange(nm),
            "building_id": ms_ids,
            "ms_ids": all_ms_ids,
            "height_m": ms_height,
            "area_m2": ma,
            "retired": retired,
            "entity_index": ms_to_entity,
        },
        geometry=ms_wgs,
        crs=4326,
    )
    ms_provenance = pd.DataFrame(
        {
            "building_id": original_ms.building_id.to_numpy(),
            "ms_group": ms_code,
            "canonical_ms_id": ms_ids[ms_code],
            "retired": retired[ms_code],
            "entity_index": ms_to_entity[ms_code],
        }
    )
    relation_m = np.concatenate([sub_m, alias_m])
    relation_o = np.concatenate([sub_o, alias_o])
    relation_kind = np.concatenate(
        [np.full(len(sub_m), "substantial", object), np.full(len(alias_m), "trusted_alias", object)]
    )
    parent_child = pd.DataFrame(
        {
            "ms_group": relation_m,
            "parent_ms_id": ms_ids[relation_m],
            "osm_group": relation_o,
            "child_osm_id": osm_ids[relation_o],
            "entity_index": osm_to_entity[relation_o],
            "relationship": relation_kind,
            "parent_retired": retired[relation_m],
        }
    )
    osm_aliases = pd.DataFrame(
        {
            "ms_group": alias_m,
            "parent_ms_id": ms_ids[alias_m],
            "osm_group": alias_o,
            "osm_id": osm_ids[alias_o],
            "retired_ms": retired[alias_m],
            "entity_index": osm_to_entity[alias_o],
        }
    )
    osm_overlap = pd.DataFrame(
        {"osm_group": target_o, "other_osm_group": neighbor, "intersection_area_m2": ointer}
    )
    ms_osm_conflicts = pd.DataFrame(
        {"ms_group": mc_m, "osm_group": mc_o, "intersection_area_m2": mc_a}
    )
    duplicate_tags = original_osm.loc[
        osm_count[osm_code] > 1, ["osm_id", "building", "tags"]
    ].copy()
    duplicate_tags["osm_group"] = osm_code[osm_count[osm_code] > 1]
    diagnostics = {
        "metric_crs": metric_crs,
        "raw_global_records": len(original_ms),
        "canonical_global_geometries": nm,
        "raw_full_osm_records": len(original_osm),
        "canonical_full_osm_geometries": no,
        "split_seed_ms": int(seed.sum()),
        "retired_ms_geometries": int(retired.sum()),
        "split_osm_geometries": int(split_osm.sum()),
        "old_osm_addition_ids": len(previous_add),
        "canonical_old_additions": int(addition.sum()),
        "retained_ms": len(kept_ms),
        "old_additions_with_positive_metric_ms_intersection": metric_overlapping_additions,
        "trusted_aliases_retained_ms": len(kept_alias_m),
        "trusted_alias_edges_outside_substantial": int(
            (~np.isin(alias_keys, sub_keys, assume_unique=True)).sum()
        ),
        "entities": entity_n,
        "paths": len(paths),
        "geometry_conflict_entities": int(entities.geometry_conflict.sum()),
        "weak_semantic_conflict_entities": int(entities.weak_semantic_conflict.sum()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "height_rule": "split uses own explicit OSM height; retained MS may use trusted same-entity OSM height; no parent-height broadcasting",
        "addition_rule": "no standalone OSM additions in matched-building method",
    }
    return {
        "paths": paths,
        "entities": entities,
        "full_osm": full_osm,
        "all_ms": all_ms,
        "ms_provenance": ms_provenance,
        "parent_child": parent_child,
        "osm_aliases": osm_aliases,
        "osm_overlap_conflicts": osm_overlap,
        "ms_osm_conflicts": ms_osm_conflicts,
        "osm_duplicate_tags": duplicate_tags,
        "diagnostics": diagnostics,
    }
