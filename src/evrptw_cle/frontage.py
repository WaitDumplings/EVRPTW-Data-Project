"""Uniform evidence guard for geometric connectors past unsupported motor roads."""

import json

import numpy as np
import shapely

from .runtime_paths import CONFIG_ROOT

FRONTAGE_PATH = CONFIG_ROOT / "delivery_frontage_contract_v1.json"


def apply_frontage_guard(frame, physical, area_crs):
    """Preserve locations and raw geometry; quarantine unsupported frontage.

    No city, source ID or absolute distance is used. Pedestrian paths alone do
    not imply a motor entrance. The result remains a documented access proxy,
    not an assertion that a mapped track is a surveyed property entrance.
    """
    result = frame.copy()
    policy = json.loads(FRONTAGE_PATH.read_text())
    roads = physical.loc[physical.highway.isin(policy["unsupported_frontage_highways"])]
    result["unsupported_frontage_nearer"] = False
    result["unsupported_frontage_osmid"] = ""
    result["unsupported_frontage_highway"] = ""
    result["unsupported_frontage_distance_m"] = np.nan
    result["frontage_policy"] = policy["policy"]
    if len(roads) and len(result):
        roads = roads.sort_values(["osmid", "physical_edge_id"]).reset_index(drop=True)
        pairs, distances = shapely.STRtree(roads.geometry.to_numpy()).query_nearest(
            result.to_crs(area_crs).geometry.to_numpy(),
            all_matches=False,
            return_distance=True,
        )
        source, match = pairs
        for name in ["osmid", "highway"]:
            result.loc[result.index[source], "unsupported_frontage_" + name] = (
                roads.iloc[match][name].astype(str).to_numpy()
            )
        result.loc[result.index[source], "unsupported_frontage_distance_m"] = distances
        result["unsupported_frontage_nearer"] = (
            result.unsupported_frontage_distance_m + policy["comparison_tolerance_m"]
            < result.road_access_distance_m
        )
    blocked = result.unsupported_frontage_nearer
    # Preserve a more specific hard failure when the selected road already fails.
    newly_blocked = blocked & result.local_access_confirmed
    result.loc[newly_blocked, "access_review_reason"] = policy["pending_reason"]
    result.loc[newly_blocked, "access_status"] = "review_required"
    for name in [
        "local_access_confirmed",
        "protected_inbound_access_eligible",
        "protected_outbound_access_eligible",
        "protected_roundtrip_eligible",
        "anchor_in_reference_scc",
        "access_gateway_in_reference_scc",
    ]:
        result.loc[blocked, name] = False
    result.loc[blocked, "anchor_scc_id"] = "S_UNRESOLVED"
    result.loc[blocked, "protected_roundtrip_status"] = "review_required_local_access"
    return result
