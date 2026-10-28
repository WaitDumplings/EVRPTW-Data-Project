"""Charging-purpose access on shared road topology without delivery exceptions."""

from __future__ import annotations

import json

import networkx as nx
import shapely

from .local_road_access import LocalAccessContext, _forest
from .frontage import apply_frontage_guard

POLICY = "charging_generic_motor_access_v1"


class ChargingAccessContext(LocalAccessContext):
    policy = POLICY

    def anchor(self, locations, **kwargs):
        result = super().anchor(locations, **kwargs)
        evidence = self.physical.set_index("physical_edge_id")
        for column in [
            "charging_motor_access_allowed",
            "charging_access_basis",
            "charging_permission_requires_review",
            "charging_way_access_tier",
            "charging_node_access_evidence_json",
        ]:
            result[column] = result.physical_edge_id.map(evidence[column])
        result["delivery_authorization_assumed"] = False
        result["owner_permission_observed"] = False
        return apply_frontage_guard(result, self.physical, self.area_crs)


def build_charging_access_context(delivery_context):
    """Keep the shared public core/reference and rebuild purpose-safe forests.

    The complete physical inventory is used because delivery-prohibited ways
    may still have ordinary motor permission. No invited-private permission or
    delivery-only exception is copied into the charging graph or path ledgers.
    """
    physical = delivery_context.physical.copy()
    required = {
        "charging_motor_access_allowed",
        "charging_access_basis",
        "charging_permission_requires_review",
        "charging_way_access_tier",
        "charging_node_access_evidence_json",
        "charging_node_access_blocked",
    }
    if not required <= set(physical.columns):
        raise ValueError("Road context lacks charging-purpose permission evidence; rebuild roads")
    physical["motor_access_allowed"] = physical.charging_motor_access_allowed.astype(bool)
    physical["delivery_legal_access_tier"] = physical.legal_access_tier
    physical["legal_access_tier"] = physical.charging_way_access_tier
    physical["source_way_motor_allowed"] = physical.motor_access_allowed
    physical["node_access_blocked"] = physical.charging_node_access_blocked.astype(bool)
    physical["delivery_authorization_assumed"] = False
    physical["connector_crossing_barrier"] = (
        ~physical.anchor_class_eligible | ~physical.motor_access_allowed
    )
    allowed_refs = {
        (ref["u"], ref["v"], ref["key"])
        for value in physical.loc[physical.motor_access_allowed, "directed_edge_refs"]
        for ref in json.loads(value)
    }
    core = delivery_context.core.copy()
    for u, v, key, data in core.edges(keys=True, data=True):
        if (str(u), str(v), str(key)) not in allowed_refs:
            raise ValueError("Shared public graph contains a charging-prohibited road edge")
        data["delivery_authorization_assumed"] = False
    access = nx.MultiDiGraph(crs=core.graph["crs"])
    wgs = physical.to_crs(core.graph["crs"])
    reference = set(delivery_context.reference)
    for row in wgs.loc[wgs.motor_access_allowed].itertuples():
        for ref in json.loads(row.directed_edge_refs):
            u, v, key = ref["u"], ref["v"], ref["key"]
            if u in reference and v in reference and core.has_edge(u, v, key):
                continue
            reversed_ = ref["geometry_orientation"] == "reverse_of_physical"
            geometry = shapely.reverse(row.geometry) if reversed_ else row.geometry
            for node, point in [(u, geometry.coords[0]), (v, geometry.coords[-1])]:
                if node not in access:
                    access.add_node(node, x=float(point[0]), y=float(point[1]))
            access.add_edge(
                u,
                v,
                key=key,
                length=float(ref["length_m"]),
                geometry=geometry,
                osmid=str(row.osmid),
                highway=row.highway,
                oneway=bool(row.oneway),
                reversed=reversed_,
                maxspeed=row.maxspeed,
                **{
                    "maxspeed:forward": row.maxspeed_forward,
                    "maxspeed:backward": row.maxspeed_backward,
                },
                access_layer="osm_terminal_only",
                through_routing_allowed=False,
                delivery_authorization_assumed=False,
                charging_access_basis=row.charging_access_basis,
            )
    gateways = set(access) & reference
    return ChargingAccessContext(
        core,
        access,
        physical,
        reference,
        _forest(access, gateways),
        _forest(access, gateways, reverse=True),
        delivery_context.area_crs,
        delivery_context.water,
    )
