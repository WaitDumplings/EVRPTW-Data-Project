import json

import geopandas as gpd
import pytest
from shapely.geometry import Point

from evrptw_cle.charging_road_access import build_charging_access_context
from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.local_road_access import build_access_context
from test_road_direction_barrier import _inputs


@pytest.mark.parametrize("node_only", [False, True])
@pytest.mark.parametrize(
    "restriction",
    [
        {"access": "private"},
        {"motor_vehicle": "delivery"},
        {"access": "private", "access:delivery": "yes"},
    ],
)
def test_charging_never_borrows_private_or_delivery_permission(tmp_path, node_only, restriction):
    options = _inputs(
        tmp_path,
        terminal_tags={} if node_only else restriction,
        node_tags=restriction if node_only else None,
    )
    build_access_context(**options, output_dir=tmp_path / "seed")
    delivery, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    locations = gpd.GeoDataFrame(
        {"latent_service_location_id": ["charging-test"]}, geometry=[Point(10, 150)], crs=3857
    )
    assert delivery.anchor(locations).local_access_confirmed.iloc[0]
    charging = build_charging_access_context(delivery)
    result = charging.anchor(locations).iloc[0]
    assert not result.local_access_confirmed
    assert not result.delivery_authorization_assumed
    assert not result.charging_motor_access_allowed
    assert not any(d["osmid"] == "2" for _, _, d in charging.access.edges(data=True))
    evidence = json.loads(result.charging_node_access_evidence_json)
    if node_only:
        assert evidence[0]["source_tags"] == restriction


@pytest.mark.parametrize("tier", ["yes", "permissive", "destination", "customers"])
def test_charging_uses_general_motor_permission_with_real_terminal_paths(tmp_path, tier):
    options = _inputs(
        tmp_path,
        terminal_tags={
            "motor_vehicle": tier,
            "maxspeed": "20 mph",
            "maxspeed:forward": "15 mph",
            "maxspeed:backward": "10 mph",
        },
    )
    build_access_context(**options, output_dir=tmp_path / "seed")
    delivery, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    charging = build_charging_access_context(delivery)
    locations = gpd.GeoDataFrame(
        {"latent_service_location_id": ["charging-test"]}, geometry=[Point(10, 150)], crs=3857
    )
    result = charging.anchor(locations).iloc[0]
    assert result.local_access_confirmed and result.requires_terminal_paths
    assert not result.delivery_authorization_assumed
    for _, _, key, data in charging.access.edges(keys=True, data=True):
        assert data["maxspeed"] == "20 mph"
        assert data["maxspeed:forward"] == "15 mph"
        assert data["maxspeed:backward"] == "10 mph"
        assert data["reversed"] == key.endswith("r")
    assert all(
        not item["delivery_authorization_assumed"]
        for item in json.loads(result.terminal_inbound_path_json)
        + json.loads(result.terminal_outbound_path_json)
    )
