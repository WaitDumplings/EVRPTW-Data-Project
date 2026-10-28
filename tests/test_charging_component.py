import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.charging_component import build_charging_component, build_charging_layers
from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.local_road_access import build_access_context
from evrptw_cle.util import sha256_file
from test_road_direction_barrier import _inputs


def _station(x=10, y=150):
    point = gpd.GeoSeries([Point(x, y)], crs=3857).to_crs(4326).iloc[0]
    return pd.DataFrame(
        [
            {
                "ID": 1,
                "Station Name": "Public charging",
                "Street Address": "1 Main St",
                "City": "Synthetic",
                "State": "NY",
                "ZIP": "10001",
                "Country": "US",
                "Fuel Type Code": "ELEC",
                "Access Code": "public",
                "Status Code": "E",
                "Longitude": point.x,
                "Latitude": point.y,
                "Geocode Status": "GPS",
                "EV Level1 EVSE Num": 0,
                "EV Level2 EVSE Num": 1,
                "EV DC Fast Count": 1,
                "EV Connector Types": "J1772 J1772COMBO",
                "Restricted Access": False,
                "Maximum Vehicle Class": "LD",
                "Access Days Time": "Business hours",
                "Access Detail Code": "CUSTOMERS",
                "EV Pricing": "Paid",
                "EV Network": "Example",
            }
        ]
    )


def _units(stations):
    result = stations.copy()
    result["EV J1772 Connector Count"] = 1
    result["EV CCS Connector Count"] = 1
    result["EV J1772 Power Output (kW)"] = 7.2
    result["EV CCS Power Output (kW)"] = 80.0
    return result


def _road(tmp_path, **kwargs):
    options = _inputs(tmp_path, **kwargs)
    build_access_context(**options, output_dir=tmp_path / "seed")
    context, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "context"
    )
    return context, options


def _layers(context, stations, **kwargs):
    boundary = gpd.GeoDataFrame(geometry=[box(-5000, -5000, 5000, 5000)], crs=3857)
    masks = gpd.GeoDataFrame(geometry=[], crs=3857)
    return build_charging_layers(
        stations,
        _units(stations),
        context,
        masks=masks,
        boundary=boundary,
        city_slug="synthetic",
        medians={"ac_level2": 6.5, "dc_fast": 240},
        **kwargs,
    )


def test_charging_component_uses_shared_node_directed_terminal_and_native_identity(tmp_path):
    context, _ = _road(tmp_path, terminal_tags={"motor_vehicle": "customers"})
    source = _station()
    locations, members, configurations, audit, access = _layers(context, source)
    row = locations.iloc[0]
    assert row.potential_location_eligible and row.road_accessible
    assert row.reference_vehicle_compatible and row.charging_parameter_eligible
    assert row.geometry.x == source.iloc[0].Longitude
    assert row.geometry.y == source.iloc[0].Latitude
    assert row.requires_terminal_paths and not row.delivery_authorization_assumed
    assert np.isclose(row.terminal_inbound_length_m, 150)
    assert np.isclose(row.terminal_outbound_length_m, 150)
    assert row.capped_pre_derating_kw == 80 and row.station_power_kw == 80
    assert configurations.is_selected.sum() == 1 and len(members) == 1
    assert not row.charger_release_eligible and not audit["instance_activation_included"]
    assert not access.core.has_edge("a", "d")
    assert {p["osmid"] for p in json.loads(row.terminal_inbound_path_json)} == {"2"}


def test_public_station_cannot_borrow_invited_private_road_but_keeps_equipment(tmp_path):
    context, _ = _road(tmp_path, terminal_tags={"access": "private"})
    locations, members, configurations, audit, _ = _layers(context, _station())
    row = locations.iloc[0]
    assert row.potential_location_eligible and row.reference_vehicle_compatible
    assert not row.road_accessible and not row.charging_parameter_eligible
    assert row.pending_reason and len(configurations) == 2
    assert members.iloc[0].source_location_eligible
    assert not members.iloc[0].location_road_accessible
    assert audit["source_safe_locations"] == 1


def test_missing_reverse_terminal_path_is_pending(tmp_path):
    context, _ = _road(tmp_path, terminal_tags={"oneway": "yes"})
    locations, _, _, _, _ = _layers(context, _station())
    row = locations.iloc[0]
    assert row.potential_location_eligible and row.reference_vehicle_compatible
    assert not row.road_accessible
    assert row.pending_reason == "local_road_no_verified_directed_roundtrip"


def test_distance_tail_is_qa_and_not_the_legacy_1000m_quarantine(tmp_path):
    context, _ = _road(tmp_path)
    locations, _, _, _, _ = _layers(context, _station(x=1500))
    row = locations.iloc[0]
    assert row.road_access_distance_m > 1000 and row.road_access_distance_qa_flag
    assert row.location_quarantine_reason == "" and row.charging_parameter_eligible


def test_building_obstacle_keeps_native_point_pending(tmp_path):
    context, _ = _road(tmp_path)
    source = _station(x=30)
    buildings = gpd.GeoDataFrame(geometry=[box(10, 140, 20, 160)], crs=3857)
    locations, _, _, audit, _ = _layers(context, source, footprints=buildings)
    row = locations.iloc[0]
    assert not row.road_accessible and row.potential_location_eligible
    assert row.pending_reason == "connector_crosses_other_mapped_building"
    assert row.geometry.x == source.iloc[0].Longitude
    assert audit["building_obstacle_coverage"] == "provided"


@pytest.mark.parametrize("column", ["Country", "Fuel Type Code", "Access Code", "Status Code"])
def test_station_scope_is_checked_per_row_including_missing(tmp_path, column):
    context, _ = _road(tmp_path)
    source = _station()
    source[column] = None
    with pytest.raises(ValueError, match="out-of-scope"):
        _layers(context, source)


def test_component_rejects_changed_station_snapshot_before_using_context(tmp_path):
    station, units, registry = [
        tmp_path / name for name in ["stations.csv", "units.csv", "sources.json"]
    ]
    _station().to_csv(station, index=False)
    _units(_station()).to_csv(units, index=False)
    registry.write_text(
        json.dumps(
            {
                "schema": "evrptw_charging_sources_v1",
                "station_sha256": sha256_file(station),
                "units_sha256": sha256_file(units),
            }
        )
    )
    station.write_text(station.read_text() + "\n")
    with pytest.raises(ValueError, match="station snapshot"):
        build_charging_component(
            city_slug="synthetic",
            station_path=station,
            source_manifest_path=registry,
            units_path=units,
            boundary_path=tmp_path / "missing-boundary",
            mask_path=tmp_path / "missing-mask",
            context_dir=tmp_path / "missing-context",
            area_crs="EPSG:3857",
            output_dir=tmp_path / "output",
        )


def _component_inputs(tmp_path, monkeypatch):
    from evrptw_cle import charging_component as component

    _, options = _road(tmp_path)
    stations, units, registry = [
        tmp_path / name for name in ["stations.csv", "units.csv", "sources.json"]
    ]
    _station().to_csv(stations, index=False)
    _units(_station()).to_csv(units, index=False)
    registry.write_text(
        json.dumps(
            {
                "schema": "evrptw_charging_sources_v1",
                "station_sha256": sha256_file(stations),
                "units_sha256": sha256_file(units),
            }
        )
    )
    # This isolated source contract binds the small test snapshot, instead of
    # distributing or silently substituting for the real 100MB frozen input.
    contract = json.loads(
        (component.CONFIG_ROOT / "charging_parameter_contract_v1.json").read_text()
    )
    contract["unit_source"]["sha256"] = sha256_file(units)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "charging_parameter_contract_v1.json").write_text(json.dumps(contract))
    monkeypatch.setattr(component, "CONFIG_ROOT", configs)
    return dict(
        city_slug="synthetic",
        station_path=stations,
        source_manifest_path=registry,
        units_path=units,
        boundary_path=options["service_boundary_path"],
        mask_path=options["mask_path"],
        context_dir=tmp_path / "context",
        area_crs="EPSG:3857",
        output_dir=tmp_path / "charging",
    )


def test_file_component_outputs_are_hash_bound_and_never_activate(tmp_path, monkeypatch):
    options = _component_inputs(tmp_path, monkeypatch)
    manifest = build_charging_component(**options)
    result = gpd.read_parquet(options["output_dir"] / "chargers.parquet")
    assert result.charging_parameter_eligible.all()
    assert not result.charger_release_eligible.any()
    assert manifest["audit"]["charging_parameter_eligible_locations"] == 1
    assert not manifest["whole_cle_frozen"] and not manifest["instance_activation_included"]
    for group in ["outputs", "private_outputs"]:
        for name, digest in manifest[group].items():
            assert sha256_file(options["output_dir"] / name) == digest
    assert "private_access/physical_roads.parquet" in manifest["private_outputs"]
    assert not any("physical_roads" in name for name in manifest["outputs"])
    with pytest.raises(FileExistsError, match="fresh"):
        build_charging_component(**options)


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"city_slug": "other-city"}, "city differs"),
        ({"area_crs": "EPSG:32611"}, "metric CRS differs"),
    ],
)
def test_component_rejects_cross_city_or_crs_context(tmp_path, monkeypatch, change, expected):
    options = _component_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=expected):
        build_charging_component(**{**options, **change})


def test_component_rejects_wrong_road_snapshot(tmp_path, monkeypatch):
    options = _component_inputs(tmp_path, monkeypatch)
    path = options["context_dir"] / "access_context_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["pbf_replication_timestamp_utc"] = "2026-08-14T20:21:03Z"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="common road snapshot"):
        build_charging_component(**options)
