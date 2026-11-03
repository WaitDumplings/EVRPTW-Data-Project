"""Geometry-first NSI ownership with source scope and axial offset evidence."""

from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .bid_geometry import decode_bid_details

BBOX_MIN_IOU = 0.25
AREA_RATIO_MAX = 4.0
STRONG_ALTERNATIVE_IOU = 0.5


def _groups(sources):
    src = sources.ftprntsrc.fillna("").astype(str).str.strip().str.upper()
    fid = sources.ftprntid.fillna("").astype(str).str.strip()
    bad = fid.str.upper().isin(
        ["", "0", "0.0", "NAN", "NONE", "NULL", "NA"]
    ) | fid.str.lower().str.startswith(("none_", "null_", "nan_"))
    genuine = src.isin(["BING", "NGA", "ORNL"]) & ~bad
    keys = pd.Series("fd:" + sources.fd_id.astype(str), index=sources.index)
    keys.loc[genuine] = src.loc[genuine] + "|" + fid.loc[genuine]
    codes, _ = pd.factorize(keys, sort=False)
    return keys.to_numpy(), codes.astype(np.int64), genuine.to_numpy()


def _source_boxes(bids):
    geometries = np.full(len(bids), None, dtype=object)
    informative = np.zeros(len(bids), dtype=bool)
    for i, value in enumerate(bids):
        detail = decode_bid_details(value)
        if detail is not None and any(int(x) > 0 for x in str(value).strip().split("-")[1:]):
            geometries[i] = detail["geometry"]
            informative[i] = True
    return geometries, informative


def _axes(polygons):
    """Whole-polygon MRR axes; full long/short edge lengths, not half widths."""
    mrr = shapely.minimum_rotated_rectangle(polygons)
    rings = shapely.get_exterior_ring(mrr)
    p0 = shapely.get_point(rings, 0)
    p1 = shapely.get_point(rings, 1)
    p2 = shapely.get_point(rings, 2)
    e0 = np.column_stack(
        (shapely.get_x(p1) - shapely.get_x(p0), shapely.get_y(p1) - shapely.get_y(p0))
    )
    e1 = np.column_stack(
        (shapely.get_x(p2) - shapely.get_x(p1), shapely.get_y(p2) - shapely.get_y(p1))
    )
    n0, n1 = np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1)
    long_length, short_length = np.maximum(n0, n1), np.minimum(n0, n1)
    edge = np.where((n0 >= n1)[:, None], e0, e1)
    u = np.divide(
        edge, long_length[:, None], out=np.full_like(edge, np.nan), where=long_length[:, None] > 0
    )
    v = np.column_stack((-u[:, 1], u[:, 0]))
    centroids = shapely.centroid(polygons)
    centers = np.column_stack((shapely.get_x(centroids), shapely.get_y(centroids)))
    return u, v, long_length, short_length, centers, mrr


def _axis_values(point_xy, path_indices, axes):
    u, v, long_length, short_length, centers, _ = axes
    delta = point_xy - centers[path_indices]
    along = np.abs(np.einsum("ij,ij->i", delta, u[path_indices]))
    across = np.abs(np.einsum("ij,ij->i", delta, v[path_indices]))
    z_long = np.divide(
        along,
        long_length[path_indices],
        out=np.full(len(along), np.nan),
        where=long_length[path_indices] > 0,
    )
    z_short = np.divide(
        across,
        short_length[path_indices],
        out=np.full(len(across), np.nan),
        where=short_length[path_indices] > 0,
    )
    return z_long, z_short, np.maximum(z_long, z_short)


def _frontier(indices, metrics):
    """Exact Pareto set. Every competing path uses the same finite dimensions."""
    if len(indices) <= 1:
        return indices
    values = metrics[indices]
    dominates = (values[:, None, :] <= values[None, :, :]).all(axis=2) & (
        values[:, None, :] < values[None, :, :]
    ).any(axis=2)
    return indices[~dominates.any(axis=0)]


def match_sources(paths_wgs_gdf, sources_wgs_gdf, area_crs, shared_R, out_evidence_path=None):
    """Return (one-row-per-fd result, complete candidate evidence, diagnostics).

    Path fields: path_id(str), entity_index(int), geometry in WGS84.
    Source fields: fd_id,bid,ftprntsrc,ftprntid,ftprntsqft,resunits,geometry.
    Geometry is never matched using previous Microsoft-parent ownership.
    """
    started = time.monotonic()
    paths = paths_wgs_gdf.reset_index(drop=True)
    sources = sources_wgs_gdf.reset_index(drop=True)
    assert paths.path_id.is_unique and sources.fd_id.is_unique
    assert np.isfinite(shared_R) and shared_R >= 0
    npth, nsrc = len(paths), len(sources)
    assert npth > 0
    entities = paths.entity_index.to_numpy(dtype=np.int64)
    assert (entities >= 0).all()
    n_entities = int(entities.max()) + 1
    path_ids = paths.path_id.astype(str).to_numpy()
    metric_paths = paths.geometry.to_crs(area_crs).to_numpy()
    metric_points = sources.geometry.to_crs(area_crs).to_numpy()
    point_xy = np.column_stack((shapely.get_x(metric_points), shapely.get_y(metric_points)))
    target_area = shapely.area(metric_paths)
    assert np.isfinite(target_area).all() and (target_area > 0).all()
    axes = _axes(metric_paths)
    hulls = shapely.convex_hull(metric_paths)
    # Target envelope is taken in WGS84 before projection, matching source BID.
    envelopes = (
        gpd.GeoSeries(shapely.envelope(paths.geometry.to_numpy()), crs=paths.crs)
        .to_crs(area_crs)
        .to_numpy()
    )
    envelope_area = shapely.area(envelopes)
    source_area = (
        pd.to_numeric(sources.ftprntsqft, errors="coerce").to_numpy(dtype=float) * 0.09290304
    )
    positive_source_area = np.isfinite(source_area) & (source_area > 0)
    boxes_wgs, informative = _source_boxes(sources.bid.to_numpy())
    boxes = gpd.GeoSeries(boxes_wgs, crs="EPSG:4326").to_crs(area_crs).to_numpy()
    own_box_covers = np.zeros(nsrc, dtype=bool)
    own_box_covers[informative] = shapely.covered_by(metric_points[informative], boxes[informative])
    internal_conflict = informative & ~own_box_covers

    # Direct candidates and meaningful source-bbox candidates are exhaustive for
    # an inference with q in its own BID and q in a candidate/alias convex hull.
    ds, dp = shapely.STRtree(metric_paths).query(metric_points, predicate="covered_by")
    pair_keys = [ds.astype(np.int64) * npth + dp]
    info_rows = np.flatnonzero(informative)
    if len(info_rows):
        bs, bp = shapely.STRtree(envelopes).query(boxes[info_rows], predicate="intersects")
        pair_keys.append(info_rows[bs].astype(np.int64) * npth + bp)
    keys = np.unique(np.concatenate(pair_keys))
    si, pi = keys // npth, keys % npth
    pair_entities = entities[pi]
    pair_points = metric_points[si]
    point_inside = shapely.covered_by(pair_points, metric_paths[pi])
    hull_inside = shapely.covered_by(pair_points, hulls[pi])
    polygon_gap = shapely.distance(pair_points, metric_paths[pi])
    centroid_gap = np.linalg.norm(point_xy[si] - axes[4][pi], axis=1)
    z_long, z_short, axis_R = _axis_values(point_xy[si], pi, axes)
    area_ratio = np.full(len(keys), np.nan)
    log_area = np.full(len(keys), np.nan)
    aok = positive_source_area[si]
    raw_ratio = np.divide(source_area[si[aok]], target_area[pi[aok]])
    area_ratio[aok] = np.maximum(raw_ratio, 1 / raw_ratio)
    log_area[aok] = np.abs(np.log(raw_ratio))
    bbox_iou = np.full(len(keys), np.nan)
    bok = informative[si]
    if bok.any():
        inter = shapely.area(shapely.intersection(boxes[si[bok]], envelopes[pi[bok]]))
        union = shapely.area(boxes[si[bok]]) + envelope_area[pi[bok]] - inter
        bbox_iou[bok] = np.divide(inter, union, out=np.zeros(len(inter)), where=union > 0)
    known_compatible = (~informative[si] | (bbox_iou >= BBOX_MIN_IOU)) & (
        ~positive_source_area[si] | (area_ratio <= AREA_RATIO_MAX)
    )
    full_source_gate = (
        informative[si]
        & positive_source_area[si]
        & own_box_covers[si]
        & (bbox_iou >= BBOX_MIN_IOU)
        & (area_ratio <= AREA_RATIO_MAX)
    )
    direct_admissible = point_inside & known_compatible
    inference_admissible = (
        ~point_inside & hull_inside & full_source_gate & np.isfinite(axis_R) & (axis_R <= shared_R)
    )
    admissible = (
        (direct_admissible | inference_admissible) & np.isfinite(axis_R) & ~internal_conflict[si]
    )

    # Unique entities, not unique alias paths, define the initial direct owner.
    direct_entity_keys = np.unique(ds.astype(np.int64) * n_entities + entities[dp])
    derow, deentity = direct_entity_keys // n_entities, direct_entity_keys % n_entities
    direct_count = np.bincount(derow, minlength=nsrc)
    direct_path_count = np.bincount(ds, minlength=nsrc)
    base_owner = np.full(nsrc, -1, dtype=np.int64)
    unique_direct = direct_count == 1
    base_owner[derow[unique_direct[derow]]] = deentity[unique_direct[derow]]
    chosen_pair = np.full(nsrc, -1, dtype=np.int64)
    if point_inside.any():
        dix = np.flatnonzero(point_inside)
        order = np.lexsort((path_ids[pi[dix]], ~known_compatible[dix], si[dix]))
        ordered = dix[order]
        row, first = np.unique(si[ordered], return_index=True)
        chosen_pair[row] = ordered[first]
    chosen_pair[~unique_direct] = -1
    base_chosen_pair = chosen_pair.copy()
    compatible_direct_counts = np.bincount(si[point_inside & known_compatible], minlength=nsrc)
    counterevidence = unique_direct & (compatible_direct_counts == 0)
    owner = base_owner.copy()
    proposed_owner = np.full(nsrc, -1, dtype=np.int64)
    reason = np.full(nsrc, "unmatched_no_supported_candidate", dtype=object)
    reason[direct_count > 1] = "pending_multiple_direct_entities"
    reason[unique_direct] = "direct_unique_entity"
    reason[counterevidence] = "direct_retained_source_counterevidence"
    reason[internal_conflict & unique_direct] = "direct_retained_source_internal_conflict"
    reason[internal_conflict & ~unique_direct] = "pending_source_internal_conflict"
    pareto = np.zeros(len(keys), dtype=bool)
    evaluated = np.zeros(nsrc, dtype=bool)

    need = (~unique_direct | counterevidence) & ~internal_conflict
    candidate_ix = np.flatnonzero(admissible & need[si])
    if len(candidate_ix):
        # Pair keys are source-major sorted; every group is contiguous here.
        rows, begin = np.unique(si[candidate_ix], return_index=True)
        ends = np.r_[begin[1:], len(candidate_ix)]
        for r, b, e in zip(rows, begin, ends):
            ix = candidate_ix[b:e]
            dimensions = []
            if informative[r]:
                dimensions.append(-bbox_iou[ix])
            if positive_source_area[r]:
                dimensions.append(log_area[ix])
            dimensions.append(axis_R[ix])
            # Only one source group's small candidate matrix is materialized.
            local_values = np.column_stack(dimensions)
            local_front = _frontier(np.arange(len(ix)), local_values)
            front = ix[local_front]
            pareto[front] = True
            evaluated[r] = True
            winners = np.unique(pair_entities[front])
            if len(winners) != 1:
                if base_owner[r] < 0:
                    reason[r] = "pending_pareto_multiple_entities"
                continue
            entity = int(winners[0])
            proposed_owner[r] = entity
            # Stable naming resolves evidence paths WITHIN the same entity only.
            pick = front[np.argmin(path_ids[pi[front]])]
            if base_owner[r] < 0:
                owner[r], chosen_pair[r] = entity, pick
                reason[r] = (
                    "direct_pareto_unique_entity" if point_inside[pick] else "inference_hull_axis"
                )
            elif entity != base_owner[r]:
                strong_front = front[bbox_iou[front] >= STRONG_ALTERNATIVE_IOU]
                if len(strong_front):
                    pick = strong_front[np.argmin(path_ids[pi[strong_front]])]
                    owner[r], chosen_pair[r] = entity, pick
                    reason[r] = "direct_replaced_strong_supported_alternative"

    # True source groups reconcile before any caller use/units filtering.
    group_keys, group_codes, genuine = _groups(sources)
    group_n = int(group_codes.max()) + 1 if nsrc else 0
    low = np.full(group_n, n_entities, dtype=np.int64)
    high = np.full(group_n, -1, dtype=np.int64)
    assigned = owner >= 0
    np.minimum.at(low, group_codes[assigned], owner[assigned])
    np.maximum.at(high, group_codes[assigned], owner[assigned])
    group_conflict = (high[group_codes] >= 0) & (low[group_codes] != high[group_codes])
    if group_conflict.any():
        owner[group_conflict] = base_owner[group_conflict]
        # Restore base representatives atomically, never tentative replacements.
        chosen_pair[group_conflict] = base_chosen_pair[group_conflict]
        reason[group_conflict] = "pending_source_group_cross_entity_scope"
    chosen_valid = chosen_pair >= 0
    chosen_path_index = np.full(nsrc, -1, dtype=np.int64)
    chosen_path_index[chosen_valid] = pi[chosen_pair[chosen_valid]]
    chosen_path_ids = np.full(nsrc, None, dtype=object)
    chosen_path_ids[chosen_valid] = path_ids[chosen_path_index[chosen_valid]]

    def selected(values):
        out = np.full(nsrc, np.nan)
        out[chosen_valid] = values[chosen_pair[chosen_valid]]
        return out

    result = pd.DataFrame(
        {
            "fd_id": sources.fd_id.to_numpy(),
            "owner": owner,
            "match_reason": reason,
            "source_group_conflict": group_conflict,
            "source_internal_conflict": internal_conflict,
            "source_geometry_counterevidence": counterevidence,
            "direct_count": direct_count,
            "direct_path_count": direct_path_count,
            "chosen_path_id": chosen_path_ids,
            "chosen_path_index": chosen_path_index,
            "bbox_iou": selected(bbox_iou),
            "R": selected(axis_R),
            "polygon_gap_m": selected(polygon_gap),
            "centroid_gap_m": selected(centroid_gap),
            "source_group_id": group_keys,
            "genuine_source_key": genuine,
            "proposed_owner": proposed_owner,
            "base_direct_owner": base_owner,
        }
    )
    accepted_pair = np.zeros(len(keys), dtype=bool)
    accepted_pair[chosen_pair[chosen_valid]] = True
    evidence = pd.DataFrame(
        {
            "fd_id": sources.fd_id.to_numpy()[si],
            "source_index": si,
            "path_id": path_ids[pi],
            "path_index": pi,
            "entity_index": pair_entities,
            "point_inside": point_inside,
            "hull_inside": hull_inside,
            "bbox_iou": bbox_iou,
            "area_ratio": area_ratio,
            "abs_log_area_ratio": log_area,
            "R": axis_R,
            "z_parallel": z_long,
            "z_perpendicular": z_short,
            "polygon_gap_m": polygon_gap,
            "centroid_gap_m": centroid_gap,
            "source_compatible_or_unavailable": known_compatible,
            "full_source_gate": full_source_gate,
            "admissible": admissible,
            "pareto_evaluated": evaluated[si],
            "on_evaluated_pareto_frontier": pareto,
            "selected_owner_path": accepted_pair,
        }
    )
    if out_evidence_path is not None:
        evidence.to_parquet(out_evidence_path, index=False)
    diagnostics = {
        "matcher_version": "geometry_first_path_pareto_v1_20261102",
        "sources": nsrc,
        "paths": npth,
        "canonical_entities": int(len(np.unique(entities))),
        "candidate_path_pairs": len(keys),
        "unique_direct_records": int(unique_direct.sum()),
        "multiple_direct_entity_records": int((direct_count > 1).sum()),
        "geometric_owner_records": int((owner >= 0).sum()),
        "operational_owner_eligible_before_semantics_units": int(
            ((owner >= 0) & ~group_conflict).sum()
        ),
        "source_group_conflict_records": int(group_conflict.sum()),
        "source_internal_conflict_records": int(internal_conflict.sum()),
        "source_counterevidence_records": int(counterevidence.sum()),
        "shared_R": float(shared_R),
        "bbox_min_iou": BBOX_MIN_IOU,
        "area_ratio_max": AREA_RATIO_MAX,
        "strong_alternative_iou": STRONG_ALTERNATIVE_IOU,
        "reason_counts": result.match_reason.value_counts().to_dict(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "shapely_version": shapely.__version__,
        "evidence_note": "Every candidate path is retained. Sources with no path remain in the one-row-per-fd result; no old parent owners or semantic filtering are used.",
    }
    return result, evidence, diagnostics
