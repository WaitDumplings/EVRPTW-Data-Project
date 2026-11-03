import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from evrptw_cle.cle_assembly import canonical_locations
from evrptw_cle.stage1_validation import validate_building_stage1


def city_fixture(tmp_path):
    for folder in ["customers", "depots", "charging", "roads", "speeds", "customer_inputs"]:
        (tmp_path / folder).mkdir()
    shared = {"road_access_distance_m": [2.0], "terminal_inbound_path_json": ["[]"],
              "terminal_outbound_path_json": ["[]"]}
    customer = gpd.GeoDataFrame({**shared, "site_id": ["toy:a"], "entity_id": ["a"], "entity_index": [0],
        "cle_default_instance_eligible": [True], "protected_roundtrip_eligible": [True],
        "residential_use": ["residential"], "residential_source": ["osm"],
        "units_final": [1], "residential_units": [1], "units_raw_nsi": [1],
        "units_method": ["nsi_retained"], "units_scope": ["building"],
        "units_review_status": ["not_flagged"], "active_customer": [False],
        "terminal_inbound_path_json": [json.dumps([{"u": "1", "v": "2", "key": "0", "length_m": 5}])]},
        geometry=[Point(-117, 33)], crs=4326)
    customer.to_parquet(tmp_path / "customers/latent_locations.parquet")
    pd.DataFrame({"fd_id": [100, 101], "owner": [0, -1], "raw_resunits": [1, 7]}).to_parquet(
        tmp_path / "customer_inputs/source_records.parquet")
    gpd.GeoDataFrame({**shared, "candidate_id": ["d"], "facility_group_id": ["g"],
        "depot_candidate_eligible": [True]}, geometry=[Point(-117, 33)], crs=4326).to_parquet(
        tmp_path / "depots/depots.parquet")
    gpd.GeoDataFrame({**shared, "charging_location_id": ["s"], "charging_parameter_eligible": [True]},
        geometry=[Point(-117, 33)], crs=4326).to_parquet(tmp_path / "charging/chargers.parquet")
    gpd.GeoDataFrame({"location_id": ["customer:toy:a", "depot:g", "charging:s"],
        "location_type": ["customer", "depot", "charging"],
        "source_location_id": ["toy:a", "g", "s"], "candidate_eligible": [True] * 3},
        geometry=[Point(-117, 33)] * 3, crs=4326).to_parquet(tmp_path / "locations.parquet")
    pd.DataFrame({"edge_u": ["1"], "edge_v": ["2"], "edge_key": ["0"], "length_m": [10.],
        "legal_travel_time_s": [10.], "reference_travel_time_weekday_s": [12.],
        "reference_travel_time_weekend_s": [14.]}).to_parquet(tmp_path / "speeds/directed_speeds.parquet")
    rows = []
    for identity in ["customer:toy:a", "depot:g", "charging:s"]:
        row = {"location_id": identity, "connector_length_m": 2., "connector_time_s_per_inverse_kph": 7.2}
        for direction in ["inbound", "outbound"]:
            values = [5., 5., 6., 7.] if identity.startswith("customer") and direction == "inbound" else [0.] * 4
            for key, value in zip([direction + "_terminal_length_m"] + [direction + "_terminal_" + key + "_time_s"
                    for key in ["legal", "reference_weekday", "reference_weekend"]], values, strict=True):
                row[key] = value
        rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp_path / "location_access_costs.parquet")
    gpd.GeoDataFrame({"direction_status": ["explicit_forward"],
        "directed_edge_refs": [json.dumps([{"geometry_orientation": "same_as_physical"}])]},
        geometry=[LineString([(-117, 33), (-116.99, 33)])], crs=4326).to_parquet(
        tmp_path / "roads/physical_roads.parquet")
    (tmp_path / "cle_manifest.json").write_text(json.dumps({"schema": "evrptw_building_first_cle_v1",
        "construction_options": {"selection_buffer_km": 20, "allow_private_through": True,
                                  "facility_masks": False}}))
    (tmp_path / "roads/access_context_manifest.json").write_text(json.dumps({"selected_buffer_km": 20,
        "ordinary_private_through_routing": True, "facility_category_masks": False}))
    (tmp_path / "customers/residential_site_manifest.json").write_text(json.dumps({"unit_profile": "unrestricted"}))
    return tmp_path


def test_valid_building_cle_preserves_native_directed_access_costs(tmp_path):
    result = validate_building_stage1(city_fixture(tmp_path))
    assert result["passed"], result["errors"]
    assert result["summary"]["eligible_locations"] == 3
    assert result["summary"]["nsi_owned_records"] == 1


def test_complex_units_cannot_be_admitted_as_one_building(tmp_path):
    city_fixture(tmp_path)
    path = tmp_path / "customers/latent_locations.parquet"
    frame = gpd.read_parquet(path)
    frame["units_scope"] = "complex"
    frame.to_parquet(path)
    result = validate_building_stage1(tmp_path)
    assert not result["passed"]
    assert "customer_building_units_scope" in {error["check"] for error in result["errors"]}


def test_missing_eligible_access_cost_row_is_detected(tmp_path):
    city_fixture(tmp_path)
    path = tmp_path / "location_access_costs.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[~frame.location_id.str.startswith("customer")]
    frame.to_parquet(path)
    result = validate_building_stage1(tmp_path)
    assert not result["passed"]
    assert "eligible_access_cost_coverage" in {error["check"] for error in result["errors"]}


def test_oneway_reverse_edge_is_detected(tmp_path):
    city_fixture(tmp_path)
    path = tmp_path / "roads/physical_roads.parquet"
    physical = gpd.read_parquet(path)
    physical["directed_edge_refs"] = json.dumps([{"geometry_orientation": "reverse_of_physical"}])
    physical.to_parquet(path)
    result = validate_building_stage1(tmp_path)
    assert not result["passed"]
    assert "physical_directed_oneway_preserved" in {error["check"] for error in result["errors"]}


def test_representative_and_neighbor_lineage_survives_assembly_and_validation(tmp_path):
    city_fixture(tmp_path)
    customer_path = tmp_path / "customers/latent_locations.parquet"
    customer = gpd.read_parquet(customer_path)
    customer["units_method"] = "nsi_representative"
    customer["units_source"] = "nsi"
    customer["units_final"] = customer["residential_units"] = 7
    customer["selected_nsi_fd_id"] = pd.array([101], dtype="Int64")
    customer["selected_nsi_resunits"] = 7
    customer["capacity_units"] = 10.
    customer["height_neighbor_estimated"] = False
    customer["neighbor_reference_id"] = "reference:a"
    customer["units_independently_verified"] = False
    customer["service_location_type"] = "building"
    customer["site_identity_source"] = "microsoft_osm_building"
    neighbor = customer.copy()
    neighbor["site_id"], neighbor["entity_id"], neighbor["entity_index"] = "toy:b", "b", 1
    neighbor["units_method"], neighbor["units_source"] = "neighbor_estimated", "neighbor"
    neighbor["units_final"] = neighbor["residential_units"] = 2
    neighbor["units_raw_nsi"] = float("nan")
    neighbor["selected_nsi_fd_id"] = pd.array([pd.NA], dtype="Int64")
    neighbor["selected_nsi_resunits"] = float("nan")
    neighbor["capacity_units"] = 3.
    neighbor["height_neighbor_estimated"] = True
    customer = gpd.GeoDataFrame(pd.concat([customer, neighbor], ignore_index=True), crs=4326)
    # A representative may come from a reassigned source, even when the
    # building's original owner ledger has several different records.
    source_path = tmp_path / "customer_inputs/source_records.parquet"
    pd.DataFrame({"fd_id": [100, 101, 102], "owner": [0, -1, 0],
        "raw_resunits": [1, 7, 9], "source_group_id": ["A", "B", "C"],
        "genuine_source_key": [True] * 3}).to_parquet(source_path)
    costs_path = tmp_path / "location_access_costs.parquet"
    costs = pd.read_parquet(costs_path)
    extra = costs.loc[costs.location_id.eq("customer:toy:a")].copy()
    extra["location_id"] = "customer:toy:b"
    pd.concat([costs, extra], ignore_index=True).to_parquet(costs_path)
    depot = gpd.read_parquet(tmp_path / "depots/depots.parquet")
    charging = gpd.read_parquet(tmp_path / "charging/chargers.parquet")

    def save_and_validate():
        customer.to_parquet(customer_path)
        canonical_locations(customer, depot, charging).to_parquet(tmp_path / "locations.parquet")
        return validate_building_stage1(tmp_path)

    result = save_and_validate()
    assert result["passed"], result["errors"]
    canonical = gpd.read_parquet(tmp_path / "locations.parquet")
    selected = canonical.loc[canonical.location_id.eq("customer:toy:a")].iloc[0]
    assert selected.units_source == "nsi" and selected.selected_nsi_fd_id == 101
    assert selected.selected_nsi_resunits == 7 and selected.capacity_units == 10
    estimated = canonical.loc[canonical.location_id.eq("customer:toy:b")].iloc[0]
    assert estimated.height_neighbor_estimated and estimated.neighbor_reference_id == "reference:a"

    source = pd.read_parquet(source_path)
    source.loc[source.fd_id.eq(101), "raw_resunits"] = 8
    source.to_parquet(source_path)
    result = validate_building_stage1(tmp_path)
    assert "nsi_representative_matches_original_units" in {error["check"] for error in result["errors"]}
    source.loc[source.fd_id.eq(101), "raw_resunits"] = 7
    source.to_parquet(source_path)
    customer.loc[customer.site_id.eq("toy:b"), "capacity_units"] = 1
    result = save_and_validate()
    assert "customer_multiunit_scale" in {error["check"] for error in result["errors"]}
