"""Company-independent spatial depot grouping, audited across eleven cities."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shapely
from scipy.spatial.distance import pdist
from shapely.strtree import STRtree

DEPOT_GROUPING_POLICY = {
    "version": "spatial_depot_v2",
    "multiple_operators_per_location": True,
    "address_max_span_m": 100.0,
    "supplemental_campus_max_span_m": 1000.0,
    "polygon_overlap_min_smaller_area_fraction": 0.8,
    "coordinate_alias_tolerance_m": 1.0,
    "office_only_policy": "retain_as_tier_b",
    "area_hard_gate": False,
}


def normalized(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def refine_depot_groups(
    frame, area_crs, address_distance_m=100.0, context=None, campus_span_m=1000.0
):
    """Preserve current campus groups; add geometry/coordinate/address aliases.

    No company-equality condition. No generic proximity-only clustering.
    Address unions are bounded against transitive chains of distant sites.
    """
    n = len(frame)
    parent = list(range(n))
    members = {i: {i} for i in range(n)}
    events = []
    metric = frame.to_crs(area_crs)
    geometries = np.asarray(metric.geometry.to_numpy(), dtype=object)
    points = shapely.point_on_surface(geometries)
    xy = np.column_stack([shapely.get_x(points), shapely.get_y(points)])
    areas = shapely.area(geometries)
    polygon = np.isin(shapely.get_type_id(geometries), [3, 6])
    point = shapely.get_type_id(geometries) == 0

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b, reason, bounded=False):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if bounded:
            merged = sorted(members[ra] | members[rb])
            max_before = max(
                (float(pdist(xy[sorted(members[r])]).max()) if len(members[r]) > 1 else 0.0)
                for r in [ra, rb]
            )
            max_after = float(pdist(xy[merged]).max())
            if max_after > max(address_distance_m, max_before) + 1e-6:
                return
        winner, loser = sorted([ra, rb])
        parent[loser] = winner
        members[winner] |= members.pop(loser)
        events.append(
            {
                "left_source_osm_id": str(frame.iloc[a].source_osm_id),
                "right_source_osm_id": str(frame.iloc[b].source_osm_id),
                "reason": reason,
                "representative_distance_m": float(np.linalg.norm(xy[a] - xy[b])),
            }
        )

    for indices in frame.groupby("facility_group_id", sort=True).indices.values():
        for i in indices[1:]:
            union(int(indices[0]), int(i), "preserve_existing_spatial_group")
    tree = STRtree(geometries)
    pairs = tree.query(geometries, predicate="intersects")
    for a, b in sorted(zip(map(int, pairs[0]), map(int, pairs[1]))):
        if a >= b or find(a) == find(b):
            continue
        if (point[a] and polygon[b]) or (polygon[a] and point[b]):
            union(a, b, "poi_covered_by_same_facility_polygon")
        elif polygon[a] and polygon[b] and min(areas[a], areas[b]) > 0:
            overlap = shapely.area(shapely.intersection(geometries[a], geometries[b])) / min(
                areas[a], areas[b]
            )
            if overlap >= 0.8:
                union(a, b, "overlapping_facility_geometry_min_area_fraction_ge_0.8")
        elif point[a] and point[b]:
            union(a, b, "identical_source_coordinate")
    # A complete house-number/street pair is required; street-only matches do
    # not identify an address. Existing address formatting is not an identity.
    addresses = defaultdict(list)
    for i, tags in enumerate(frame.osm_tags_json.map(json.loads)):
        number, street = (
            normalized(tags.get("addr:housenumber")),
            normalized(tags.get("addr:street")),
        )
        if number and street:
            addresses[(number, street)].append(i)
    for indices in addresses.values():
        for offset, a in enumerate(indices):
            for b in indices[offset + 1 :]:
                if np.linalg.norm(xy[a] - xy[b]) <= address_distance_m:
                    union(a, b, "same_full_street_address_with_bounded_distance", bounded=True)
    # One metre handles duplicated POI coordinates without treating arbitrary
    # neighbouring warehouses as a common site. Sensitivity is address-only.
    point_indices = np.flatnonzero(point)
    if len(point_indices):
        near = STRtree(geometries[point_indices]).query(
            geometries[point_indices], predicate="dwithin", distance=1.0
        )
        for aa, bb in sorted(zip(map(int, near[0]), map(int, near[1]))):
            if aa < bb:
                union(
                    int(point_indices[aa]),
                    int(point_indices[bb]),
                    "source_coordinate_alias_le_1m",
                    bounded=True,
                )
    # Industrial areas excluded as C-class depot *candidates* can still supply
    # spatial context for multiple genuine A/B objects. A global span bound
    # prevents a long industrial corridor becoming one departure location.
    if context is not None:
        context_metric = context.to_crs(area_crs)
        accepted = context_metric.get("landuse", pd.Series("", index=context_metric.index)).isin(
            {"industrial", "logistics", "warehouse", "depot"}
        )
        accepted &= context_metric.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        context_metric = context_metric.loc[accepted].copy()
        context_metric["_area"] = context_metric.geometry.area
        point_tree = STRtree(points)
        for row in context_metric.sort_values(["_area", "source_osm_id"]).itertuples():
            indices = sorted(map(int, point_tree.query(row.geometry, predicate="covers")))
            roots = sorted({find(i) for i in indices})
            if len(roots) < 2:
                continue
            merged = sorted(set().union(*(members[r] for r in roots)))
            if float(pdist(xy[merged]).max()) > campus_span_m:
                continue
            first = merged[0]
            for index in merged[1:]:
                before = len(events)
                union(first, index, "shared_osm_industrial_or_logistics_context_bounded_span")
                if len(events) > before:
                    events[-1]["context_source_osm_id"] = str(row.source_osm_id)
    group_ids = {}
    for indices in members.values():
        token = "|".join(sorted(frame.iloc[sorted(indices)].source_osm_id.astype(str))).encode()
        group_id = "depot_spatial_" + hashlib.blake2b(token, digest_size=10).hexdigest()
        for i in indices:
            group_ids[i] = group_id
    result = frame.copy()
    result["facility_group_id"] = [group_ids[i] for i in range(n)]
    result["facility_group_member_count"] = (
        result.groupby("facility_group_id")["source_osm_id"].transform("size").astype(int)
    )
    rules_by_group = {}
    for group_id, group in result.groupby("facility_group_id", sort=True):
        rules = set()
        for value in group.get(
            "facility_group_merge_rules_json", pd.Series("[]", index=group.index)
        ):
            rules.update(json.loads(value))
        source_ids = set(group.source_osm_id.astype(str))
        rules.update(e["reason"] for e in events if e["left_source_osm_id"] in source_ids)
        rules_by_group[group_id] = json.dumps(sorted(rules))
    result["facility_group_merge_rules_json"] = result.facility_group_id.map(rules_by_group)
    return result, events
