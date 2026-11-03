import json
import shutil
import subprocess

import geopandas as gpd

import pandas as pd

import pytest

from shapely.geometry import Point, box

from evrptw_cle.depot_grouping import refine_depot_groups

from evrptw_cle.public_locations import (
    excluded_location_reason,
    filter_public_locations,
    prepare_exclusion_masks,
    is_territorial_boundary,
    location_tags,
    PUBLIC_LOCATION_POLICY,
)


BEAT_NATIVE_TAGS = {
    "admin_level": "11",
    "boundary": "administrative",
    "name": "Beat 2513",
    "operator": "Chicago Police Department",
    "place": "police_beat",
}


@pytest.mark.parametrize("encoded", [False, True])
def test_police_jurisdiction_does_not_exclude_a_residence_but_same_named_facility_does(encoded):
    territory = {"osm_tags_json": json.dumps(BEAT_NATIVE_TAGS)} if encoded else BEAT_NATIVE_TAGS
    assert is_territorial_boundary(location_tags(territory))
    assert not excluded_location_reason(territory)
    assert not excluded_location_reason(territory, purpose="charging")
    territory_raw = gpd.GeoDataFrame([territory], geometry=[box(0, 0, 500, 500)], crs=3857)
    assert prepare_exclusion_masks(territory_raw, "EPSG:3857").empty
    # A real facility keeps the same source name/operator and jurisdiction tags.
    facility = dict(BEAT_NATIVE_TAGS, amenity="police")
    if encoded:
        facility = {"osm_tags_json": json.dumps(facility)}
    hosts = gpd.GeoDataFrame(
        [territory, facility], geometry=[box(0, 0, 500, 500), box(200, 200, 300, 300)], crs=3857
    )
    masks = prepare_exclusion_masks(hosts, "EPSG:3857")
    residences = gpd.GeoDataFrame(
        {"location_id": ["ordinary_residence", "inside_actual_facility"]},
        geometry=[Point(100, 100), Point(250, 250)], crs=3857,
    )
    assert len(masks) == 1
    for purpose in ["delivery", "charging"]:
        kept, _ = filter_public_locations(residences, masks, purpose=purpose)
        assert kept.location_id.tolist() == ["ordinary_residence"]


@pytest.mark.parametrize("territory", [
    {"boundary": "administrative"}, {"boundary": "police"}, {"place": "police_beat"},
])
def test_explicit_jurisdiction_type_wins_over_sensitive_operator_text(territory):
    tags = {"name": "Regional jurisdiction", "operator": "County Sheriff", **territory}
    assert not excluded_location_reason(tags)
    assert excluded_location_reason({"name": tags["name"], "operator": tags["operator"]}) == "law_enforcement"


@pytest.mark.parametrize("facility, expected", [
    ({"amenity": "police"}, "law_enforcement"),
    ({"amenity": "prison"}, "correctional"),
    ({"government": "intelligence"}, "intelligence"),
    ({"landuse": "military"}, "military"),
    ({"building": "military"}, "military"),
    ({"plant:source": "nuclear"}, "nuclear"),
])
def test_real_sensitive_facility_tags_are_not_waived_by_jurisdiction_context(facility, expected):
    tags = {**BEAT_NATIVE_TAGS, **facility}
    assert excluded_location_reason(tags) == expected
    assert excluded_location_reason(tags, purpose="charging") == expected


@pytest.mark.parametrize("permission, scope", [
    ({"motorcar": "no"}, "all_motor"),
    ({"motorcar": "yes", "access:delivery": "no"}, "delivery_only"),
])
def test_jurisdiction_semantics_do_not_waive_native_access_prohibition(permission, scope):
    tags = {**BEAT_NATIVE_TAGS, **permission}
    assert excluded_location_reason(tags) == "motor_access_prohibited"
    raw = gpd.GeoDataFrame([tags], geometry=[box(0, 0, 100, 100)], crs=3857)
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    assert len(masks) == 1 and masks.exclusion_scope.iloc[0] == scope
    assert bool(excluded_location_reason(tags, purpose="charging")) == (scope == "all_motor")


def test_published_exclusion_and_access_contracts_match_runtime_policies():
    from evrptw_cle.charging_road_access import POLICY as charging_policy
    from evrptw_cle.delivery_access import POLICY as delivery_policy
    from evrptw_cle.runtime_paths import CONFIG_ROOT

    public = json.loads((CONFIG_ROOT / "public_location_exclusions_v1.json").read_text())
    roads = json.loads((CONFIG_ROOT / "local_road_access_contract_v2.json").read_text())
    assert public["policy"] == PUBLIC_LOCATION_POLICY
    assert roads["policy"] == delivery_policy
    assert roads["charging_access"]["policy"] == charging_policy
    assert not roads["independent_delivery_prohibition"]["vehicle_allowance_overrides"]
    for value in roads["independent_delivery_prohibition"]["prohibited_values"]:
        tags = {"motorcar": "yes", roads["independent_delivery_prohibition"]["source_tag"]: value}
        assert excluded_location_reason(tags) == "motor_access_prohibited"
        assert not excluded_location_reason(tags, purpose="charging")


@pytest.mark.skipif(shutil.which("osmium") is None, reason="OSM extraction needs osmium-tool")
def test_pbf_jurisdiction_does_not_become_sensitive_facility_mask(tmp_path):
    from evrptw_cle.public_locations import extract_exclusion_masks

    source, pbf = tmp_path / "context.osm", tmp_path / "context.pbf"
    territory = "".join(f'<tag k="{key}" v="{value}"/>' for key, value in BEAT_NATIVE_TAGS.items())
    source.write_text(f'''<osm version="0.6">
      <node id="1" version="1" lon="0" lat="0"/>
      <node id="2" version="1" lon="0.01" lat="0"/>
      <node id="3" version="1" lon="0.01" lat="0.01"/>
      <node id="4" version="1" lon="0" lat="0.01"/>
      <node id="5" version="1" lon="0.004" lat="0.004"/>
      <node id="6" version="1" lon="0.005" lat="0.004"/>
      <node id="7" version="1" lon="0.005" lat="0.005"/>
      <node id="8" version="1" lon="0.004" lat="0.005"/>
      <way id="10" version="1">
        <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>{territory}
      </way>
      <way id="20" version="1">
        <nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/>{territory}
        <tag k="amenity" v="police"/>
      </way>
    </osm>''')
    subprocess.run([shutil.which("osmium"), "cat", str(source), "-o", str(pbf)], check=True)
    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(geometry=[box(-.01, -.01, .02, .02)], crs=4326).to_file(
        boundary, driver="GeoJSON"
    )
    masks = extract_exclusion_masks(
        osmium=shutil.which("osmium"), pbf_path=pbf,
        boundary_path=boundary, area_crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"location_id": ["civilian_residence", "actual_police_facility"]},
        geometry=[Point(.002, .002), Point(.0045, .0045)], crs=4326,
    )
    kept, _ = filter_public_locations(points, masks)
    assert len(masks) == 1 and masks.exclusion_category.iloc[0] == "law_enforcement"
    assert kept.location_id.tolist() == ["civilian_residence"]


@pytest.mark.parametrize("vehicle_key", ["motorcar", "motor_vehicle", "vehicle"])
@pytest.mark.parametrize("prohibition", ["no", "prohibited", "none"])
def test_delivery_prohibition_is_independent_of_vehicle_permission(vehicle_key, prohibition):
    from evrptw_cle.osm_graph import _edge_is_drivable
    from evrptw_cle.terminal_access import _effective_motor_access

    tags = {"highway": "residential", vehicle_key: "yes", "access:delivery": prohibition}
    assert excluded_location_reason(tags) == "motor_access_prohibited"
    assert _effective_motor_access(tags) == ("access:delivery", "prohibited")
    assert not _edge_is_drivable(tags)


def test_delivery_permission_does_not_override_specific_vehicle_prohibition():
    from evrptw_cle.terminal_access import _effective_motor_access

    tags = {"motorcar": "no", "access:delivery": "yes"}
    assert excluded_location_reason(tags) == "motor_access_prohibited"
    assert _effective_motor_access(tags) == ("motorcar", "prohibited")


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"landuse": "military"}, "military"),
        ({"name": "Federal Bureau of Investigation"}, "intelligence"),
        ({"Federal Agency Code": "DAF"}, "military"),
        ({"Federal Agency Code": "DOJ"}, "law_enforcement"),
        ({"amenity": "prison"}, "correctional"),
        ({"plant:source": "nuclear"}, "nuclear"),
        ({"name": "County Sheriff Warehouse"}, "law_enforcement"),
        ({"motor_vehicle": "no"}, "motor_access_prohibited"),
        ({"access": "no", "motor_vehicle": "delivery"}, ""),
        ({"access": "private", "building": "warehouse"}, ""),
        ({"name": "Civil Airport Cargo Terminal"}, ""),
        ({"name": "Port Logistics Warehouse"}, ""),
        ({"name": "Salvation Army Warehouse"}, ""),
        ({"name": "Navy Pier"}, ""),
        ({"osm_tags_json": pd.NA, "name": "Old Navy warehouse"}, ""),
    ],
)
def test_public_host_categories(tags, expected):
    assert excluded_location_reason(tags) == expected


def test_unnamed_warehouse_in_sensitive_host_removed_without_exporting_inventory():
    hosts = gpd.GeoDataFrame(
        {"name": ["private context"], "landuse": ["military"]},
        geometry=[box(0, 0, 100, 100)],
        crs=3857,
    )
    masks = prepare_exclusion_masks(hosts, "EPSG:3857")
    assert set(masks.columns) == {
        "exclusion_category", "exclusion_scope", "exclusion_basis", "geometry"
    }
    candidates = gpd.GeoDataFrame(
        {"candidate_id": ["inside", "outside"], "building": ["warehouse", "warehouse"]},
        geometry=[Point(50, 50), Point(150, 50)],
        crs=3857,
    )
    kept, summary = filter_public_locations(candidates, masks)
    assert list(kept.candidate_id) == ["outside"]
    assert summary["removed_count"] == 1
    assert "inside" not in json.dumps(summary)
    assert "private context" not in json.dumps(summary)


def test_sensitive_poi_buffer_is_metric_and_no_unrelated_line_area():
    raw = gpd.GeoDataFrame({"amenity": ["police"]}, geometry=[Point(0, 0)], crs=3857)
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    points = gpd.GeoDataFrame(geometry=[Point(49, 0), Point(51, 0)], crs=3857)
    kept, summary = filter_public_locations(points, masks)
    assert len(kept) == 1
    assert summary["removed_count"] == 1


def test_mask_and_source_permissions_are_specific_to_delivery_or_charging():
    raw = gpd.GeoDataFrame(
        {
            "motorcar": ["yes", "no", "yes"],
            "access:delivery": ["no", "no", "no"],
            "military": [None, None, "base"],
        },
        geometry=[box(0, 0, 100, 100), box(200, 0, 300, 100), box(400, 0, 500, 100)],
        crs=3857,
    )
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    assert masks.exclusion_scope.tolist() == ["delivery_only", "all_motor", "all_motor"]
    assert masks.exclusion_basis.iloc[0] == "explicit_delivery_purpose_prohibition"
    points = gpd.GeoDataFrame(
        {"station_id": ["delivery_only", "motor_ban", "sensitive"]},
        geometry=[Point(50, 50), Point(250, 50), Point(450, 50)], crs=3857,
    )
    assert filter_public_locations(points, masks)[0].empty
    charging, _ = filter_public_locations(points, masks, purpose="charging")
    assert charging.station_id.tolist() == ["delivery_only"]
    assert filter_public_locations(points, masks.drop(columns="exclusion_scope"), purpose="charging")[0].empty
    assert not excluded_location_reason(raw.iloc[0], purpose="charging")
    assert excluded_location_reason(raw.iloc[1], purpose="charging") == "motor_access_prohibited"
    assert excluded_location_reason(raw.iloc[2], purpose="charging") == "military"


def test_access_only_barriers_have_no_area_but_sensitive_and_other_pois_keep_masks():
    raw = gpd.GeoDataFrame(
        {
            "name": ["Residential gate", "Motor prohibition", "Sensitive gate", "Other POI"],
            "barrier": ["gate", "gate", "gate", None],
            "access:delivery": ["no", "no", "no", None],
            "motorcar": ["yes", "no", "yes", "no"],
            "military": [None, None, "base", None],
        },
        geometry=[Point(0, 0), Point(200, 0), Point(400, 0), Point(600, 0)], crs=3857,
    )
    masks = prepare_exclusion_masks(raw, "EPSG:3857").to_crs(3857)
    assert len(masks) == 2
    assert not masks.geometry.intersects(Point(0, 0)).any()
    assert not masks.geometry.intersects(Point(200, 0)).any()
    assert masks.geometry.intersects(Point(400, 0)).any()
    assert masks.geometry.intersects(Point(600, 0)).any()


@pytest.mark.parametrize("permission", [{"motor_vehicle": "no"}, {"access": "no"}])
def test_generic_prohibited_gate_does_not_exclude_neighbours_but_prohibited_area_does(permission):
    tags = {"barrier": "gate", **permission}
    raw = gpd.GeoDataFrame(
        [tags, tags], geometry=[Point(0, 0), box(200, 0, 300, 100)], crs=3857,
    )
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    assert len(masks) == 1 and masks.exclusion_scope.iloc[0] == "all_motor"
    assert excluded_location_reason(tags) == "motor_access_prohibited"
    residences = gpd.GeoDataFrame(
        {"location_id": ["beside_gate", "inside_prohibited_area"]},
        geometry=[Point(10, 0), Point(250, 50)], crs=3857,
    )
    for purpose in ["delivery", "charging"]:
        kept, _ = filter_public_locations(residences, masks, purpose=purpose)
        assert kept.location_id.tolist() == ["beside_gate"]


@pytest.mark.parametrize("facility", [{"military": "base"}, {"amenity": "police"}])
def test_independent_sensitive_gate_keeps_its_facility_poi_buffer(facility):
    raw = gpd.GeoDataFrame(
        [{"barrier": "gate", "motor_vehicle": "no", **facility}], geometry=[Point(0, 0)], crs=3857,
    )
    masks = prepare_exclusion_masks(raw, "EPSG:3857")
    assert len(masks) == 1 and masks.exclusion_scope.iloc[0] == "all_motor"
    locations = gpd.GeoDataFrame(geometry=[Point(49, 0), Point(51, 0)], crs=3857)
    kept, _ = filter_public_locations(locations, masks)
    assert len(kept) == 1 and kept.geometry.iloc[0].x == 51


@pytest.mark.skipif(shutil.which("osmium") is None, reason="OSM extraction needs osmium-tool")
@pytest.mark.parametrize("permission", [
    {"motorcar": "yes", "access:delivery": "no"}, {"motor_vehicle": "no"}, {"access": "no"},
])
def test_unnamed_prohibited_area_keeps_scope_while_gate_blocks_only_incident_roads(tmp_path, permission):
    from evrptw_cle.delivery_access import permission_inventory
    from evrptw_cle.public_locations import extract_exclusion_masks
    from test_local_road_access import context

    pbf = tmp_path / "source.osm.pbf"
    source = tmp_path / "source.osm"
    gate_tags = {"barrier": "gate", **permission}
    gate_xml = "".join(f'<tag k="{key}" v="{value}"/>' for key, value in gate_tags.items())
    source.write_text(f'''<osm version="0.6">
      <node id="1" version="1" lon="0" lat="0"/>
      <node id="2" version="1" lon="0.001" lat="0"/>
      <node id="3" version="1" lon="0.001" lat="0.001"/>
      <node id="4" version="1" lon="0" lat="0.001"/>
      <node id="5" version="1" lon="0.004" lat="0.004">
        {gate_xml}
      </node>
      <node id="6" version="1" lon="0.006" lat="0.004">
        <tag k="name" v="Residential entry"/><tag k="barrier" v="gate"/>
        <tag k="motorcar" v="yes"/><tag k="access:delivery" v="no"/>
      </node>
      <way id="10" version="1">
        <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
        <tag k="building" v="warehouse"/><tag k="motorcar" v="yes"/>
        <tag k="access:delivery" v="no"/>
      </way>
    </osm>''')
    subprocess.run([shutil.which("osmium"), "cat", str(source), "-o", str(pbf)], check=True)
    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(geometry=[box(-.01, -.01, .01, .01)], crs=4326).to_file(
        boundary, driver="GeoJSON"
    )
    masks = extract_exclusion_masks(
        osmium=shutil.which("osmium"), pbf_path=pbf,
        boundary_path=boundary, area_crs="EPSG:3857",
    )
    assert len(masks) == 1
    assert masks.exclusion_category.iloc[0] == "motor_access_prohibited"
    assert masks.exclusion_scope.iloc[0] == "delivery_only"
    assert masks.geometry.iloc[0].covers(Point(.0005, .0005))
    assert not masks.geometry.iloc[0].covers(Point(.004, .004))
    assert not masks.geometry.iloc[0].covers(Point(.006, .004))

    # The anonymous gate creates no area mask, while its source node still
    # blocks incident roads even under the invited-private delivery model.
    roads = tmp_path / "roads"
    roads.mkdir()
    pd.DataFrame([{"osmid": "5", "tags_json": json.dumps(gate_tags)}]).to_parquet(
        roads / "osm_access_nodes.parquet", index=False
    )
    pd.DataFrame([{"osmid": "20", "tags_json": json.dumps(
        {"highway": "service", "access": "private"})}]).to_parquet(
        roads / "osm_ways.parquet", index=False
    )
    physical = context(permit=True).physical
    physical["osmid"], physical["edge_u"] = "20", "5"
    result, summary = permission_inventory(
        physical, roads, gpd.GeoDataFrame(geometry=[], crs=3857)
    )
    assert summary["hard_node_count"] == 1
    assert not result.motor_access_allowed.any()


def test_same_address_merge_is_bounded_and_street_only_is_insufficient():
    frame = gpd.GeoDataFrame(
        {
            "source_osm_id": ["n1", "n2", "n3", "n4"],
            "facility_group_id": ["a", "b", "c", "d"],
            "operator": ["Company A", "Company B", "Company C", "Company D"],
            "osm_tags_json": [json.dumps({"addr:housenumber": "1", "addr:street": "Road"})] * 3
            + [json.dumps({"addr:street": "Road"})],
        },
        geometry=[Point(0, 0), Point(80, 0), Point(160, 0), Point(10, 0)],
        crs=3857,
    )
    result, _ = refine_depot_groups(frame, "EPSG:3857")
    assert result.facility_group_id.iloc[0] == result.facility_group_id.iloc[1]
    assert result.facility_group_id.iloc[1] != result.facility_group_id.iloc[2]
    assert result.facility_group_id.iloc[0] != result.facility_group_id.iloc[3]


def test_campus_context_merges_companies_but_rejects_long_industrial_corridor():
    frame = gpd.GeoDataFrame(
        {
            "source_osm_id": ["a", "b"],
            "facility_group_id": ["a", "b"],
            "osm_tags_json": ["{}", "{}"],
        },
        geometry=[Point(0, 0), Point(900, 0)],
        crs=3857,
    )
    context = gpd.GeoDataFrame(
        {"landuse": ["industrial"], "source_osm_id": ["area"]},
        geometry=[box(-10, -10, 2010, 10)],
        crs=3857,
    )
    merged, _ = refine_depot_groups(frame, "EPSG:3857", context=context)
    assert merged.facility_group_id.nunique() == 1
    frame.geometry = [Point(0, 0), Point(2000, 0)]
    distinct, _ = refine_depot_groups(frame, "EPSG:3857", context=context)
    assert distinct.facility_group_id.nunique() == 2
