import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from evrptw_cle.cle_assembly import bind_access_costs, canonical_locations
from evrptw_cle.cle_validation import AccessRow, check_path, independent_reference, verify_city_cle


def tables():
    customer = gpd.GeoDataFrame(dict(
        site_id=["same"], cle_default_instance_eligible=[True], quarantine_reason=[""],
        residential_units=[pd.NA], service_location_type=["small_apartment"],
        site_identity_source=["microsoft_nsi_independent_or_provisional_site"],
        site_scope_unresolved=[True],site_scope_review_required=[False],
    ),geometry=[Point(0,0)],crs=4326)
    depot = gpd.GeoDataFrame(dict(
        candidate_id=["first_pending", "second_usable"], facility_group_id=["same", "same"],
        depot_candidate_eligible=[False,True], access_review_reason=["no_path", ""],
    ),geometry=[Point(1,1),Point(1,2)],crs=4326)
    charge = gpd.GeoDataFrame(dict(
        charging_location_id=["same"], charging_parameter_eligible=[False],
        pending_reason=["no_native_compatible_configuration"],
    ),geometry=[Point(2,2)],crs=4326)
    return customer,depot,charge


def test_canonical_locations_keep_distinct_roles_and_one_usable_depot_group():
    result = canonical_locations(*tables())
    assert set(result.location_id)=={"customer:same","depot:same","charging:same"}
    depot = result.loc[result.location_type.eq("depot")].iloc[0]
    assert depot.access_row_id == "second_usable" and depot.candidate_eligible
    assert depot.geometry.equals(Point(1,2))
    assert result.residential_units.isna().all()
    assert not {"demand", "active", "time_window"} & set(result)


def row():
    part=dict(u="a",v="b",key="road",osmid="1",length_m=50.,fraction=.5,
              geometry_wkt="LINESTRING (0 0, 0.5 0)",delivery_authorization_assumed=False)
    return AccessRow(
        requires_terminal_paths=True, road_anchor_lon=.5, road_anchor_lat=0.,
        terminal_inbound_path_json=json.dumps([part]), terminal_inbound_gateway="a",
        terminal_inbound_length_m=50., local_alternative_selected=False,
    )


def test_independent_access_chain_checks_partial_cost_and_missing_direction():
    key=("a","b","road")
    native={key:dict(length=100.,osmid="1",auth=False,strict=True)}
    speed={key:(100.,10.,12.,14.)}
    costs,count=check_path(row(),"inbound",native,{key},{"a"},speed)
    assert count==1 and np.allclose(costs,[5,6,7])
    with pytest.raises(ValueError,match="wrong-purpose"):
        check_path(row(),"inbound",native,{("b","a","road")},{"a"},speed)
    changed=row()
    changed["terminal_inbound_length_m"]=51.
    with pytest.raises(ValueError,match="does not conserve"):
        check_path(changed,"inbound",native,{key},{"a"},speed)


def test_partial_chain_must_reach_native_endpoint_and_its_fraction():
    from shapely.geometry import LineString
    key=("a","b","road")
    native={key:dict(length=100.,osmid="1",auth=False,strict=True,geometry=LineString([(0,0),(1,0)]))}
    speed={key:(100.,10.,12.,14.)}
    valid=row()
    check_path(valid,"inbound",native,{key},{"a"},speed)
    part=json.loads(valid.terminal_inbound_path_json)[0]
    part["geometry_wkt"]="LINESTRING (0.25 0, 0.5 0)"
    valid["terminal_inbound_path_json"]=json.dumps([part])
    with pytest.raises(ValueError,match="native endpoint"):
        check_path(valid,"inbound",native,{key},{"a"},speed)
    valid=row()
    part=json.loads(valid.terminal_inbound_path_json)[0]
    part["fraction"]=.7
    valid["terminal_inbound_path_json"]=json.dumps([part])
    with pytest.raises(ValueError,match="native length ratio"):
        check_path(valid,"inbound",native,{key},{"a"},speed)


def test_independent_scc_requires_actual_directed_roundtrip():
    edges={("a","b","1"),("b","a","2"),("b","c","3")}
    assert independent_reference(edges)=={"a","b"}


def test_saved_file_tampering_stops_verifier_before_builders_can_repair(tmp_path):
    path=tmp_path / "table.bin"
    path.write_bytes(b"changed bytes")
    (tmp_path / "cle_manifest.json").write_text(json.dumps(dict(
        stage=1,instance_activation_included=False,public_release=False,
        schema="evrptw_city_logistics_environment_v1",files={"table.bin":{"sha256":"wrong"}},
    )))
    with pytest.raises(ValueError,match="saved_hash"):
        verify_city_cle(tmp_path)
    assert path.read_bytes()==b"changed bytes"


def test_access_binding_preserves_unknown_units_and_does_not_assume_connector_speed():
    customer,depot,charging=tables()
    for table in [customer,depot,charging]:
        table["road_access_distance_m"]=20.
        for d in ["inbound","outbound"]:
            table["terminal_"+d+"_path_json"]="[]"
    canonical=canonical_locations(customer,depot,charging)
    speeds=pd.DataFrame(columns=["edge_u","edge_v","edge_key","length_m","legal_travel_time_s","reference_travel_time_weekday_s","reference_travel_time_weekend_s"])
    costs=bind_access_costs(canonical,{
        "customer":(customer,"site_id"),"depot":(depot,"candidate_id"),
        "charging":(charging,"charging_location_id"),
    },speeds)
    assert len(costs)==2 and costs.connector_time_s_per_inverse_kph.eq(72).all()
    assert costs.inbound_terminal_legal_time_s.eq(0).all()
    assert "connector_speed_kph" not in costs
def test_only_missing_derived_match_keys_normalize_null_and_empty():
    from evrptw_cle.cle_validation import same_evidence
    import pandas as pd
    assert same_evidence(pd.Series([None,"msft:a"]),pd.Series(["","msft:a"]),"composite_building_id")
    assert not same_evidence(pd.Series([None]),pd.Series(["msft:a"]),"composite_building_id")
    assert not same_evidence(pd.Series([None]),pd.Series([""]),"occtype")
    assert not same_evidence(pd.Series([None]),pd.Series([0]),"resunits")


def test_independent_owner_recount_rejects_an_unmarked_shared_component():
    from evrptw_cle.cle_validation import check_source_owner_flags
    sites=pd.DataFrame(dict(site_id=["a","b"],site_source_key=["complex:1","msft:1"],
                            site_scope_review_required=[False,False],site_scope_status=["no_competition"]*2))
    members=pd.DataFrame(dict(site_id=["a","b"],nsi_component_id=["same","same"],composite_building_id=["roof","roof"]))
    with pytest.raises(ValueError,match="component spans site owners"):
        check_source_owner_flags(sites,members)
    sites["site_scope_review_required"]=True
    assert check_source_owner_flags(sites,members)==dict(cross_owner_components=1,mixed_scope_roofs=1,affected_sites=2)


@pytest.mark.parametrize("field,value",[("length_m",51.),("source_length_m",51.),("source_length_m",float("nan")),("source_osmid","2"),("highway","motorway")])
def test_speed_metadata_must_match_the_actual_native_edge(field,value):
    from evrptw_cle.cle_validation import check_native_speed_metadata
    native={("a","b","k"):dict(length=50.,osmid="1",highway="service")}
    frame=pd.DataFrame([dict(edge_u="a",edge_v="b",edge_key="k",length_m=50.,source_length_m=50.,source_osmid="1",highway="service")])
    assert check_native_speed_metadata(frame,native)
    frame.loc[0,field]=value
    with pytest.raises(ValueError,match="actual native road"):
        check_native_speed_metadata(frame,native)


def test_geographic_degrees_and_survey_feet_cannot_be_labeled_as_metres():
    from evrptw_cle.cle_validation import projected_metres
    assert projected_metres("EPSG:32616") and projected_metres("EPSG:3857")
    assert not projected_metres("EPSG:4326")
    assert not projected_metres("EPSG:2229")

