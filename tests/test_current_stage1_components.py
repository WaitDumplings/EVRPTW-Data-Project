import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.charging_road_access import build_charging_access_context
from evrptw_cle.current_access import CURRENT_DELIVERY_POLICY, motor_permission
from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.depot_grouping import refine_depot_groups
from evrptw_cle.local_road_access import LocalAccessContext, build_access_context
from evrptw_cle.public_locations import filter_public_locations
from test_road_direction_barrier import _inputs


def test_current_road_build_private_through_oneway_and_shared_charging(tmp_path):
    inputs = _inputs(tmp_path, terminal_tags={"access": "private", "oneway": "yes"})
    inputs.update(mask_path=None, selection_buffer_km=20,
                  allow_private_through=True, facility_masks=False)
    _, seed = build_access_context(**inputs, output_dir=tmp_path / "seed")
    assert seed["selected_buffer_km"] == 20
    context, report = build_delivery_access_context(**inputs, seed_dir=tmp_path / "seed",
                                                    output_dir=tmp_path / "roads")
    assert report["policy"] == CURRENT_DELIVERY_POLICY
    assert report["selected_buffer_km"] == 20
    assert report["mask_sha256"] is None
    private = [(u, v, d) for u, v, d in context.core.edges(data=True) if d["osmid"] == "2"]
    assert private and all(d["private_through_routing_assumed"] for _, _, d in private)
    assert all((u, v) == ("a", "d") for u, v, _ in private)
    charging = build_charging_access_context(context)
    assert charging.core.has_edge("a", "d")
    assert not charging.core.has_edge("d", "a")
    loaded = LocalAccessContext.load(tmp_path / "roads")
    assert loaded.policy == CURRENT_DELIVERY_POLICY and not loaded.facility_masks


def test_current_private_keeps_native_physical_barriers(tmp_path):
    inputs = _inputs(tmp_path, terminal_tags={"access": "private"},
                     node_tags={"barrier": "bollard"})
    inputs.update(mask_path=None, selection_buffer_km=20,
                  allow_private_through=True, facility_masks=False)
    build_access_context(**inputs, output_dir=tmp_path / "seed")
    context, _ = build_delivery_access_context(**inputs, seed_dir=tmp_path / "seed",
                                              output_dir=tmp_path / "roads")
    assert not context.physical.loc[context.physical.osmid.eq("2"), "motor_access_allowed"].any()
    assert not context.physical.loc[context.physical.osmid.eq("2"), "charging_motor_access_allowed"].any()


@pytest.mark.parametrize("tags,purpose,expected", [
    ({"access": "private"}, "delivery", True),
    ({"access": "private"}, "charging", True),
    ({"access": "private", "access:delivery": "no"}, "delivery", False),
    ({"access": "private", "access:delivery": "no"}, "charging", True),
    ({"motor_vehicle": "no", "access": "private"}, "delivery", False),
    ({"access": "residents"}, "delivery", True),
    ({"access": "residents"}, "charging", False),
    ({"access": "private", "locked": "yes"}, "charging", False),
])
def test_private_assumption_respects_native_permissions(tags, purpose, expected):
    assert motor_permission(tags, purpose=purpose)["allowed"] == expected


def test_facility_category_removed_but_actual_access_prohibition_retained():
    frame = gpd.GeoDataFrame({"osm_tags_json": [json.dumps({"military": "logistics"}),
                            json.dumps({"motor_vehicle": "no"})]},
                            geometry=[Point(0, 0), Point(10, 0)], crs=3857)
    kept, _ = filter_public_locations(frame, facility_categories=False)
    assert len(kept) == 1
    assert json.loads(kept.iloc[0].osm_tags_json)["military"] == "logistics"


@pytest.mark.parametrize("tags,expected_groups", [
    ({"landuse": "industrial", "name": "Generic Industrial Park"}, 2),
    ({"landuse": "industrial", "name": "Example Logistics Park"}, 1),
    ({"landuse": "logistics"}, 1),
])
def test_only_explicit_logistics_campus_merges_without_legacy_span(tags, expected_groups):
    frame = gpd.GeoDataFrame({"source_osm_id": ["n1", "n2"],
        "facility_group_id": ["one", "two"], "osm_tags_json": ["{}", "{}"]},
        geometry=[Point(0, 0), Point(2000, 0)], crs=3857)
    context = gpd.GeoDataFrame({"source_osm_id": ["w10"], "landuse": [tags["landuse"]],
        "osm_tags_json": [json.dumps(tags)]}, geometry=[box(-10, -10, 2010, 10)], crs=3857)
    result, _ = refine_depot_groups(frame, 3857, context=context, explicit_logistics_parks=True)
    assert result.facility_group_id.nunique() == expected_groups


def test_current_charging_equipment_binding_uses_native_access_not_facility_category():
    from test_charging_parameters import fixture, MEDIANS
    from evrptw_cle.charging_parameters import select_reference_configurations

    locations, members, units = fixture()
    units["Facility Type"] = "MIL_BASE"
    result, _, _, _ = select_reference_configurations(locations, members, units, MEDIANS,
                                                       facility_categories=False)
    assert result.iloc[0].reference_vehicle_eligible
    units["motor_vehicle"] = "no"
    result, _, _, _ = select_reference_configurations(locations, members, units, MEDIANS,
                                                       facility_categories=False)
    assert not result.iloc[0].reference_vehicle_eligible


def test_national_power_medians_use_port_weights_and_pre_city_source_qualification():
    from evrptw_cle.charging_parameters import national_port_weighted_power_medians

    rows = []
    for ac, dc, ports, restricted in [(6, 50, 1, False), (10, 200, 3, False),
                                      (19, 500, 100, True), (50, 1000, 0, False)]:
        rows.append({"Country": "US", "Fuel Type Code": "ELEC", "Access Code": "public",
            "Status Code": "E", "Restricted Access": restricted,
            "EV J1772 Power Output (kW)": ac, "EV CCS Power Output (kW)": dc,
            "EV J1772 Connector Count": ports, "EV CCS Connector Count": ports})
    medians, counts = national_port_weighted_power_medians(pd.DataFrame(rows))
    assert medians == {"ac_level2": 10, "dc_fast": 200}
    assert counts["ac_level2"]["ports"] == 4


def test_current_component_wrappers_build_roads_then_real_depot_layer(tmp_path):
    from evrptw_cle.stage1_components import build_stage1_noncustomer_component
    from evrptw_cle.util import sha256_file

    inputs = _inputs(tmp_path, terminal_tags={"access": "private"})
    raw = tmp_path / "depots.parquet"
    gpd.GeoDataFrame({"source_osm_id": ["n100", "n101"], "building": ["warehouse", "warehouse"],
                     "name": ["First warehouse", "Second warehouse"]},
        geometry=[Point(20, 20), Point(60, 20)], crs=3857).to_crs(4326).to_parquet(raw)
    files = {}
    for key, path in {"admin": inputs["admin_boundary_path"], "service": inputs["service_boundary_path"],
                      "depot": raw, "depot_extract": inputs["source_dir"] / "extraction.json"}.items():
        files[key] = dict(path=str(path.relative_to(tmp_path)), sha256=sha256_file(path))
    inventory = {"files": files, "cities": {"synthetic": {"area_crs": "EPSG:3857",
        "boundaries": {"admin_boundary": "admin", "service_boundary": "service"},
        "depot": {"raw": "depot", "extraction": "depot_extract", "pbf_sha256": "frozen-synthetic-pbf"}}}}
    kwargs = dict(city_slug="synthetic", inventory=inventory, source_root=tmp_path,
                  road_source_dir=inputs["source_dir"], city_dir=tmp_path / "cle")
    build_stage1_noncustomer_component("roads", **kwargs)
    report = build_stage1_noncustomer_component("depots", **kwargs)
    assert report["available_unique_groups"] == 2
    layer = gpd.read_parquet(tmp_path / "cle/depots/depots.parquet")
    assert layer.depot_candidate_eligible.all()
    assert layer.road_access_distance_m.max() < 25
