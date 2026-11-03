"""Representative NSI matching, preserving every source observation.

Candidate geometry uses the existing nationwide matcher.  Numerical attribution
then selects one NSI record per building and per genuine source-footprint group;
units are never added.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely

from .customer_matching import match_sources


@dataclass(frozen=True)
class RepresentativeConfig:
    shared_R: float = 0.2853432473555575
    neighbor_column: str = "neighbor_units"
    capacity_column: str = "capacity_units"
    eligible_column: str = "location_eligible"
    height_column: str = "height_m"
    area_column: str = "area_m2"
    min_source_unit: int = 1
    max_dense_component_cells: int = 2_000_000


def source_groups(records: pd.DataFrame) -> np.ndarray:
    """Use true source keys; unkeyed records retain independent fd identities."""
    if "source_group_id" in records and "genuine_source_key" in records:
        genuine = records.genuine_source_key.fillna(False).to_numpy(bool)
        key = records.source_group_id.fillna("").astype(str).to_numpy()
        return np.where(genuine & (key != ""), key, "fd:" + records.fd_id.astype(str))
    src = records.get("ftprntsrc", pd.Series("", index=records.index)).fillna("").astype(str).str.upper()
    ident = records.get("ftprntid", pd.Series("", index=records.index)).fillna("").astype(str)
    genuine = src.isin(["BING", "NGA", "ORNL"]) & ~ident.str.upper().isin(["", "0", "NAN", "NONE", "NULL", "NA"])
    return np.where(genuine, src + "|" + ident, "fd:" + records.fd_id.astype(str))


def edges_from_current_ledger(records: pd.DataFrame) -> pd.DataFrame:
    """Recover already evaluated geometric owners without treating group flags as final.

    Unassigned or counterevidenced rows need ``build_candidate_edges``; this helper
    does not invent alternative owners from a nearest-point search.
    """
    r = records.loc[records.owner.ge(0)].copy()
    for name in ["source_internal_conflict", "source_geometry_counterevidence"]:
        if name in r:
            r = r.loc[~r[name].fillna(False).astype(bool)]
    result = pd.DataFrame({"fd_id": r.fd_id, "entity_index": r.owner.astype(np.int64)})
    for name in ["bbox_iou", "R", "polygon_gap_m", "centroid_gap_m"]:
        result[name] = pd.to_numeric(r.get(name, np.nan), errors="coerce")
    result["point_inside"] = result.polygon_gap_m.fillna(np.inf).le(1e-8)
    result["hull_inside"] = True
    result["admissible"] = True
    result["source_compatible_or_unavailable"] = True
    result["path_id"] = r.get("chosen_path_id", "").to_numpy()
    result["candidate_origin"] = "existing_geometric_owner"
    return result.reset_index(drop=True)


def build_candidate_edges(paths_wgs_gdf: gpd.GeoDataFrame, records: pd.DataFrame,
                          area_crs: str, config: RepresentativeConfig | None = None):
    """Re-evaluate all admissible paths for the caller's unresolved source subset."""
    config = config or RepresentativeConfig()
    if records.empty:
        return pd.DataFrame(), {"sources": 0}
    s = records.copy()
    lon = "native_lon" if "native_lon" in s else "geometry_lon"
    lat = "native_lat" if "native_lat" in s else "geometry_lat"
    gdf = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s[lon], s[lat]), crs="EPSG:4326")
    _, evidence, diagnostic = match_sources(paths_wgs_gdf, gdf, area_crs, config.shared_R)
    evidence = evidence.loc[evidence.admissible].copy()
    evidence["candidate_origin"] = "regenerated_geometric_candidate"
    return evidence, diagnostic


def _numeric(frame, name, default=np.nan):
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def score_edges(edges: pd.DataFrame, records: pd.DataFrame, entities: pd.DataFrame,
                config: RepresentativeConfig | None = None):
    """Filter genuine geometry/capacity candidates, then compute ranking evidence."""
    config = config or RepresentativeConfig()
    if edges.empty:
        return edges.copy()
    src_cols = [c for c in ["fd_id", "resunits", "ftprntsqft", "bldheight", "geometry_block_reason",
                           "source_internal_conflict", "owner", "source_group_id", "genuine_source_key",
                           "ftprntsrc", "ftprntid", "full_osm_owner_entity", "full_osm_state",
                           "full_osm_count", "full_osm_owner_entities", "full_osm_all_nonres"] if c in records]
    s = records[src_cols].copy()
    s["source_group"] = source_groups(s)
    # Candidate tables can include several alias paths to the same building.
    e = edges.copy().merge(s, on="fd_id", how="inner", validate="many_to_one")
    attrs = ["entity_index", config.eligible_column, config.area_column, config.height_column,
             config.neighbor_column, config.capacity_column]
    b = entities[[c for c in attrs if c in entities]].copy()
    rename = {c: "building_" + c for c in b if c != "entity_index"}
    b = b.rename(columns=rename)
    e = e.merge(b, on="entity_index", how="inner", validate="many_to_one")
    keep = np.ones(len(e), dtype=bool)
    if "admissible" in e:
        keep &= e.admissible.fillna(False).to_numpy(bool)
    if "source_internal_conflict" in e:
        keep &= ~e.source_internal_conflict.fillna(False).to_numpy(bool)
    eligible = "building_" + config.eligible_column
    if eligible in e:
        keep &= e[eligible].fillna(False).to_numpy(bool)
    units = _numeric(e, "resunits")
    keep &= np.isfinite(units) & (units >= config.min_source_unit) & (units == np.floor(units))
    inside = e.get("point_inside", pd.Series(False, index=e.index)).fillna(False).to_numpy(bool)
    blocked = e.get("geometry_block_reason", pd.Series("", index=e.index)).fillna("").astype(str)
    full_count = _numeric(e,"full_osm_count")
    fresh = np.isfinite(full_count)
    legacy = ~fresh
    # A fresh complete-building containment audit supersedes older containment
    # flags; the source's internal BID/point conflict remains a hard exclusion.
    keep &= ~(legacy & blocked.isin(["point_in_weak_nonresidential_or_mixed_osm", "point_in_multiple_full_osm_objects"]).to_numpy())
    keep &= ~(legacy & blocked.eq("inference_point_in_external_osm_object").to_numpy() & ~inside)
    if "owner" in e:
        keep &= ~(legacy & blocked.eq("point_in_other_canonical_osm_entity").to_numpy()
                  & e.entity_index.eq(e.owner).to_numpy())
    e["full_osm_fresh_containment"] = fresh
    e["full_osm_multiple_containers"] = fresh & (full_count > 1)
    e["full_osm_multi_unmapped_direct"] = False
    if fresh.any():
        full_owner = _numeric(e,"full_osm_owner_entity", -1)
        full_state = e.get("full_osm_state", pd.Series("",index=e.index)).fillna("").astype(str)
        all_nonres = (e.get("full_osm_all_nonres",pd.Series(False,index=e.index))
                      .fillna(False).to_numpy(bool))
        all_nonres |= (full_count == 1) & full_state.isin(["nonresidential", "mixed"]).to_numpy()
        keep &= ~(fresh & all_nonres)
        unique_owner = fresh & (full_count == 1) & (full_owner >= 0)
        keep &= ~(unique_owner & (e.entity_index.to_numpy() != full_owner))
        unmapped_object = fresh & (full_count == 1) & (full_owner < 0)
        keep &= ~(unmapped_object & ~inside)
        multiple = fresh & (full_count > 1)
        keep &= ~(multiple & ~inside)
        owner_sets = e.get("full_osm_owner_entities",pd.Series("",index=e.index)).fillna("").astype(str)
        e["full_osm_multi_unmapped_direct"] = multiple & owner_sets.eq("").to_numpy() & inside
        # A point shared by row-house boundaries can legitimately have several
        # eligible containers.  Resolve those containers rather than discarding
        # the source or sending it to an unrelated MS polygon containing it.
        mapped_multiple = multiple & owner_sets.ne("").to_numpy()
        for value, positions in owner_sets.loc[mapped_multiple].groupby(owner_sets.loc[mapped_multiple],sort=False).groups.items():
            allowed = np.asarray([int(v) for v in value.split("|") if v],dtype=np.int64)
            ix = np.asarray(list(positions),dtype=np.int64)
            keep[ix] &= np.isin(e.entity_index.to_numpy()[ix],allowed)
    capacity = _numeric(e, "building_" + config.capacity_column)
    e["capacity_test_available"] = np.isfinite(capacity) & (capacity >= 0)
    # The accepted NSI=1 admission policy remains a separate explicit exemption.
    e["capacity_exceeded"] = e.capacity_test_available.to_numpy() & (units > capacity) & (units > 1)
    keep &= ~e.capacity_exceeded.to_numpy()
    e = e.loc[keep].copy()
    if e.empty:
        return e
    e["geometry_tier"] = np.where(e.point_inside.fillna(False), 0, 1)
    neighbor = _numeric(e, "building_" + config.neighbor_column)
    units = _numeric(e, "resunits")
    valid_neighbor = np.isfinite(neighbor) & (neighbor >= 1)
    e["neighbor_available"] = valid_neighbor
    e["neighbor_log_discrepancy"] = np.where(valid_neighbor, np.abs(np.log(units/neighbor)), np.nan)
    # Missing evidence is neutral; it is not turned into a fabricated mismatch.
    area = _numeric(e, "building_" + config.area_column)
    src_area = _numeric(e, "ftprntsqft") * 0.09290304
    good_area = np.isfinite(area) & (area > 0) & np.isfinite(src_area) & (src_area > 0)
    e["source_area_log_discrepancy"] = np.where(good_area, np.abs(np.log(src_area/area)), np.nan)
    h = _numeric(e, "building_" + config.height_column)
    sh = _numeric(e, "bldheight")
    good_h = np.isfinite(h) & (h > 0) & np.isfinite(sh) & (sh > 0)
    e["source_height_log_discrepancy"] = np.where(good_h, np.abs(np.log(sh/h)), np.nan)
    sort = ["geometry_tier", "neighbor_log_discrepancy", "source_area_log_discrepancy",
            "source_height_log_discrepancy", "polygon_gap_m", "fd_id", "entity_index"]
    for c in sort:
        if c not in e:
            e[c] = np.nan
    # Remove aliases of one fd/building, but retain other real source records
    # for the geometric-coverage ledger before numerical representative choice.
    e = e.sort_values(sort, na_position="last", kind="stable").drop_duplicates(["fd_id", "entity_index"])
    return e.reset_index(drop=True)


def select_representatives(edges: pd.DataFrame, records: pd.DataFrame, entities: pd.DataFrame,
                           config: RepresentativeConfig | None = None):
    """Select an exact rank-maximal one-to-one partial group/building assignment.

    Geometry direct matches dominate inference matches in the objective.  Within
    identical rank counts, per-edge lexicographic ranks resolve neighborhood and
    source-size evidence.  Empty assignments are allowed via dummy columns.
    """
    config = config or RepresentativeConfig()
    start = time.monotonic()
    e = score_edges(edges, records, entities, config)
    eligible_source_ids = set(e.fd_id) if len(e) else set()
    eligible_record_building_edges = len(e)
    ledger = records[[c for c in ["fd_id", "resunits", "owner", "entity_id"] if c in records]].copy()
    ledger["selected_representative"] = False
    ledger["has_eligible_geometric_candidate"] = False
    ledger["selected_entity_index"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
    if e.empty:
        return e, ledger, {"eligible_edges": 0, "selected": 0, "elapsed_seconds": round(time.monotonic()-start, 3)}
    # Several source records can legitimately reference one true footprint.
    # They remain in the source ledger; one representative group/building edge
    # enters the assignment problem, selected by the existing evidence order.
    e = e.drop_duplicates(["source_group", "entity_index"]).reset_index(drop=True)
    group_degree = e.groupby("source_group", sort=False).source_group.transform("size")
    building_degree = e.groupby("entity_index", sort=False).entity_index.transform("size")
    isolated = group_degree.eq(1) & building_degree.eq(1)
    selected_indices = e.index[isolated].tolist()
    remaining = e.loc[~isolated].copy()
    e["component_id"] = -1
    gs, group_keys = pd.factorize(remaining.source_group, sort=True)
    bs, building_keys = pd.factorize(remaining.entity_index, sort=True)
    ng, nb = len(group_keys), len(building_keys)
    graph = coo_matrix((np.ones(2*len(remaining)), (np.r_[gs, bs+ng], np.r_[bs+ng, gs])), shape=(ng+nb, ng+nb)).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    remaining["component_id"] = labels[gs]
    e.loc[remaining.index,"component_id"] = remaining.component_id
    too_large_components = []
    component_sizes = [(1,1,1)] if isolated.any() else []
    for comp, ix in remaining.groupby("component_id", sort=False).groups.items():
        sub = e.loc[ix]
        group = sub.source_group.unique()
        buildings = sub.entity_index.unique()
        component_sizes.append((len(group), len(buildings), len(sub)))
        if len(group) == 1:
            selected_indices.append(sub.index[0])
            continue
        if len(buildings) == 1:
            selected_indices.append(sub.index[0])
            continue
        cells = len(group)*(len(buildings)+len(group))
        if cells > config.max_dense_component_cells:
            # Sparse exact assignment is implemented with private dummy nodes.
            from scipy.sparse.csgraph import min_weight_full_bipartite_matching
            solver = "sparse"
        else:
            solver = "dense"
        gcode = pd.Index(group).get_indexer(sub.source_group)
        bcode = pd.Index(buildings).get_indexer(sub.entity_index)
        m = min(len(group), len(buildings))
        # Base rank 0/1 geometry objective: one direct edge outweighs all
        # inference edges in this component; one inference outweighs all ties.
        tie_rank = np.arange(len(sub), dtype=float) / max(1, len(sub))
        reward_inference = m + 1.0
        reward_direct = (m+1.0)**2
        reward = np.where(sub.geometry_tier.to_numpy() == 0, reward_direct, reward_inference) - tie_rank
        if solver == "dense":
            cost = np.full((len(group), len(buildings)+len(group)), reward_direct*10, dtype=float)
            cost[np.arange(len(group)), len(buildings)+np.arange(len(group))] = 0
            cost[gcode, bcode] = -reward
            row, col = linear_sum_assignment(cost)
        else:
            # Add an offset to all selected cells so sparse zeros remain real edges.
            offset = reward_direct + 1.0
            matrix = coo_matrix((np.r_[offset-reward, np.full(len(group), offset)],
                                (np.r_[gcode, np.arange(len(group))], np.r_[bcode, len(buildings)+np.arange(len(group))])),
                               shape=(len(group), len(buildings)+len(group))).tocsr()
            row, col = min_weight_full_bipartite_matching(matrix)
        mapping = {(int(g), int(b)): index for g,b,index in zip(gcode,bcode,sub.index)}
        for a,b in zip(row,col):
            if b < len(buildings):
                selected_indices.append(mapping[int(a),int(b)])
    selected = e.loc[selected_indices].copy().sort_values("entity_index").reset_index(drop=True)
    selected["units_final"] = selected.resunits.astype(np.int64)
    selected["units_method"] = "nsi_best_representative_no_sum"
    mapping = selected.set_index("fd_id").entity_index
    ledger["selected_representative"] = ledger.fd_id.isin(mapping.index)
    ledger["selected_entity_index"] = ledger.fd_id.map(mapping).astype("Int64")
    ledger["has_eligible_geometric_candidate"] = ledger.fd_id.isin(eligible_source_ids)
    diag = {"source_records": len(records), "eligible_record_building_edges": eligible_record_building_edges,
            "eligible_source_records": len(eligible_source_ids), "eligible_group_building_edges": len(e),
            "selected": len(selected), "selected_units_sum": int(selected.units_final.sum()),
            "components": int(ncomp+isolated.sum()), "max_component_groups": max(x[0] for x in component_sizes),
            "max_component_buildings": max(x[1] for x in component_sizes),
            "direct_selected": int(selected.geometry_tier.eq(0).sum()),
            "inferred_selected": int(selected.geometry_tier.eq(1).sum()),
            "elapsed_seconds": round(time.monotonic()-start, 3),
            "objective": "lexicographically maximize direct-match count, inference-match count, then sum of per-edge evidence ranks",
            "units_policy": "one source record per building and genuine source-footprint group; never summed"}
    return selected, ledger, diag
