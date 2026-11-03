"""Current Stage 1 access assumptions, separate from historical terminal policy."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import shapely

from .public_locations import explicit_delivery_prohibited, location_tags
from .terminal_access import _effective_motor_access

CURRENT_DELIVERY_POLICY = "delivery_route_access_v4_private_through_no_facility_categories"
CURRENT_CHARGING_POLICY = "charging_motor_access_v3_private_through_no_facility_categories"
PHYSICAL_BLOCKS = {"block", "bollard", "cycle_barrier", "jersey_barrier", "kissing_gate",
                   "stile", "turnstile", "wall", "fence", "chain"}


def access_only_masks(masks):
    """Keep actual source access prohibitions, never facility category buffers."""
    if masks is None:
        return None
    if "exclusion_category" not in masks:
        return masks.iloc[:0].copy()
    return masks.loc[masks.exclusion_category.eq("motor_access_prohibited")].copy()


def motor_permission(tags, *, purpose="delivery", allow_private_through=True):
    """Resolve static native permissions; private is an explicit model assumption."""
    tags = dict(tags)
    generic = {k: v for k, v in tags.items() if k != "access:delivery"}
    field, tier = _effective_motor_access(generic)
    value = str(generic.get(field, "")).strip().lower()
    physical = (tags.get("locked") == "yes" or tags.get("barrier") in PHYSICAL_BLOCKS
                or tags.get("service") == "emergency_access")
    public = tier in {"explicit_yes", "permissive", "unspecified"} or value == "designated"
    private = tier == "permit_required" and value == "private" and allow_private_through
    destination = tier == "destination_or_customers"
    allowed = public or private or destination
    through = public or private
    basis = "private_through_routing_assumed" if private else tier
    if purpose == "delivery":
        dfield, dtier = _effective_motor_access(tags)
        if explicit_delivery_prohibited(tags):
            allowed = through = False
            basis = "explicit_delivery_prohibited"
        elif dtier == "delivery_only" or value == "residents":
            allowed, through = True, False
            # A redundant delivery exception need not restrict an otherwise
            # allowed ordinary-private road under this model.
            if private and dfield == "access:delivery":
                through = True
            basis = "delivery_purpose_access" if not through else basis
        elif dtier == "prohibited":
            allowed = through = False
            basis = "native_motor_prohibited"
    elif purpose != "charging":
        raise ValueError("Unknown access purpose: " + purpose)
    if physical:
        allowed = through = False
        basis = "native_physical_barrier"
    return dict(source_tag=field, legal_access_tier=tier, allowed=bool(allowed),
                through=bool(through), physical_block=bool(physical),
                private_through_routing_assumed=bool(private and allowed),
                basis=basis, source_tags=tags)


def native_location_reason(row, *, purpose="delivery"):
    permission = motor_permission(location_tags(row), purpose=purpose)
    return "" if permission["allowed"] else "motor_access_prohibited_or_unresolved"


def current_permission_inventory(physical, source_dir, *, allow_private_through=True, masks=None):
    """Re-evaluate source ways/nodes; do not inherit old facility-mask decisions."""
    from .local_road_access import OPERATING_ROAD_CLASSES

    roads = physical.copy()
    source = Path(source_dir)
    way_rows = pd.read_parquet(source / "osm_ways.parquet", columns=["osmid", "tags_json"])
    node_rows = pd.read_parquet(source / "osm_access_nodes.parquet", columns=["osmid", "tags_json"])
    ways = {str(r.osmid): json.loads(r.tags_json) for r in way_rows.itertuples()}
    nodes = {str(r.osmid): json.loads(r.tags_json) for r in node_rows.itertuples()}
    delivery_way = {k: motor_permission(v, allow_private_through=allow_private_through) for k, v in ways.items()}
    charging_way = {k: motor_permission(v, purpose="charging", allow_private_through=allow_private_through) for k, v in ways.items()}
    delivery_node = {k: motor_permission(v, allow_private_through=allow_private_through) for k, v in nodes.items()}
    charging_node = {k: motor_permission(v, purpose="charging", allow_private_through=allow_private_through) for k, v in nodes.items()}
    results = []
    for row in roads.itertuples():
        d, c = delivery_way[str(row.osmid)], charging_way[str(row.osmid)]
        endpoints = sorted({str(row.edge_u), str(row.edge_v)})
        dn = [delivery_node[n] for n in endpoints if n in delivery_node]
        cn = [dict(osmid=n, **charging_node[n]) for n in endpoints if n in charging_node]
        direction = bool(getattr(row, "direction_resolved", True))
        operating = row.highway in OPERATING_ROAD_CLASSES
        allowed = operating and direction and d["allowed"] and all(n["allowed"] for n in dn)
        charging = operating and direction and c["allowed"] and all(n["allowed"] for n in cn)
        private = d["private_through_routing_assumed"] or any(n["private_through_routing_assumed"] for n in dn)
        # Keep ordinary service connectors terminal unless the accepted private
        # through-routing assumption specifically applies to this segment.
        through = (allowed and d["through"] and all(n["through"] for n in dn)
                   and (row.access_layer == "operational_public" or private))
        results.append(dict(
            motor_access_allowed=allowed, strict_motor_access_allowed=allowed,
            source_way_motor_allowed=d["allowed"] and direction,
            node_access_blocked=not all(n["allowed"] for n in dn),
            sensitive_interior_road=False, delivery_only_interior_road=False,
            native_area_access_prohibited=False,
            charging_native_area_access_prohibited=False,
            delivery_authorization_assumed=bool(private or (allowed and not d["through"])),
            private_through_routing_assumed=bool(private and through),
            access_layer="operational_public" if through else "osm_terminal_only",
            connector_crossing_barrier=not row.anchor_class_eligible or not allowed,
            charging_motor_access_allowed=charging,
            charging_access_basis=c["basis"] if charging else "native_charging_permission_or_direction_blocked",
            charging_permission_requires_review=not charging,
            charging_way_access_tier=c["legal_access_tier"],
            charging_way_access_source_tag=c["source_tag"],
            charging_way_access_tags_json=json.dumps(c["source_tags"], sort_keys=True),
            charging_node_access_evidence_json=json.dumps(cn, sort_keys=True),
            charging_node_access_blocked=not all(n["allowed"] for n in cn),
        ))
    values = pd.DataFrame(results, index=roads.index)
    for column in values:
        roads[column] = values[column]
    native_masks = access_only_masks(masks)
    if native_masks is not None and len(native_masks):
        native_masks = native_masks.to_crs(roads.crs).reset_index(drop=True)
        pairs = shapely.STRtree(native_masks.geometry.to_numpy()).query(
            roads.geometry.to_numpy(), predicate="intersects")
        for left, right in zip(*pairs):
            geometry = roads.geometry.iloc[int(left)].intersection(native_masks.geometry.iloc[int(right)])
            if geometry.length <= 0:
                continue  # Boundary touch is not an interior access prohibition.
            index = roads.index[int(left)]
            roads.loc[index, ["motor_access_allowed", "strict_motor_access_allowed"]] = False
            roads.loc[index, "native_area_access_prohibited"] = True
            roads.loc[index, "connector_crossing_barrier"] = True
            if native_masks.iloc[int(right)].get("exclusion_scope", "all_motor") != "delivery_only":
                roads.loc[index, "charging_native_area_access_prohibited"] = True
                roads.loc[index, "charging_motor_access_allowed"] = False
                roads.loc[index, "charging_permission_requires_review"] = True
                roads.loc[index, "charging_access_basis"] = "native_area_access_prohibited"
    return roads, dict(
        facility_category_masks=False, ordinary_private_through_routing=allow_private_through,
        private_through_segments=int(roads.private_through_routing_assumed.sum()),
        charging_motor_access_allowed_segment_count=int(roads.charging_motor_access_allowed.sum()),
    )
