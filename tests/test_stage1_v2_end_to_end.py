"""One raw-source synthetic run through all five current Stage 1 components."""
import json
import shutil

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from evrptw_cle import charging_component
from evrptw_cle.cle_reader import CityLogisticsEnvironment
from evrptw_cle.runtime_paths import CONFIG_ROOT
from evrptw_cle.stage1_v2 import SCHEMA, build, load_project
from evrptw_cle.util import sha256_file, write_json
from test_charging_component import _station, _units
from test_road_direction_barrier import _inputs


def test_raw_building_first_stage1_build_verify_and_read(tmp_path, monkeypatch):
    """No cached geometry, fake AI replies, source network or Stage 2 activation."""
    road = _inputs(tmp_path, terminal_tags={"access": "private", "oneway": "yes"})
    road_root = tmp_path / "road_sources"
    road_root.mkdir()
    road_source = road_root / "synthetic"
    shutil.move(road["source_dir"], road_source)
    way_path = road_source / "osm_ways.parquet"
    ways = gpd.read_parquet(way_path)
    tags = json.loads(ways.loc[0, "tags_json"])
    ways.loc[0, "tags_json"] = json.dumps(dict(tags, maxspeed="25 mph"))
    ways.to_parquet(way_path, index=False)
    extraction = road_source / "extraction.json"
    meta = json.loads(extraction.read_text())
    meta.update(synthetic_test_fixture=True, access_nodes_complete_for_retained_ways=True)
    meta["outputs"][way_path.name] = sha256_file(way_path)
    write_json(extraction, meta)

    parent, child1, child2, single = [box(*bounds) for bounds in [
        (20, 10, 40, 20), (20, 10, 30, 20), (30, 10, 40, 20), (70, 10, 80, 20)]]
    microsoft = gpd.GeoDataFrame({"building_id": ["ms-parent", "ms-single"],
        "source_kind": ["global", "global"], "height": [None, None]},
        geometry=[parent, single], crs=3857).to_crs(4326)
    osm = gpd.GeoDataFrame({"osm_id": ["w-child1", "w-child2", "w-single"],
        "building": ["house"] * 3, "building_part": [False] * 3,
        "tags": [json.dumps({"building": "house"})] * 3},
        geometry=[child1, child2, single], crs=3857).to_crs(4326)
    nsi = gpd.GeoDataFrame({"fd_id": [1, 2, 3], "bid": [None] * 3,
        "ftprntsrc": ["BING"] * 3, "ftprntid": ["source-1", "source-2", "source-3"],
        "ftprntsqft": [100 / .09290304] * 3, "resunits": [1, 1, 2],
        "occtype": ["RES1", "RES1", "RES3A"]},
        geometry=[Point(25, 15), Point(35, 15), Point(75, 15)], crs=3857).to_crs(4326)
    customer_sources = {}
    for name, frame in [("microsoft", microsoft), ("osm", osm), ("nsi", nsi)]:
        path = tmp_path / (name + ".parquet")
        frame.to_parquet(path, index=False)
        customer_sources[name] = str(path)

    depot_path = tmp_path / "depot.parquet"
    gpd.GeoDataFrame({"source_osm_id": ["n100", "n101", "w-park"],
        "building": ["warehouse", "warehouse", None],
        "landuse": [None, None, "industrial"],
        "name": ["First warehouse", "Second warehouse", "Example Logistics Park"]},
        geometry=[Point(60, 20), Point(65, 20), box(55, 5, 69, 30)], crs=3857).to_crs(
            4326).to_parquet(depot_path, index=False)
    stations = _station(x=90, y=5)
    station_path, units_path, census_path = [tmp_path / name for name in [
        "stations.csv", "units.csv", "census.csv"]]
    stations.to_csv(station_path, index=False)
    _units(stations).to_csv(units_path, index=False)
    pd.DataFrame(columns=["afdc_id"]).to_csv(census_path, index=False)
    source_meta = tmp_path / "synthetic_sources.json"
    write_json(source_meta, {"synthetic_test_fixture": True,
                            "station_snapshot": "2026-10-26"})
    # This binds the tiny declared synthetic snapshot to a separate test power
    # contract; the archived national contract and real source bytes are untouched.
    policy = json.loads((CONFIG_ROOT / "charging_parameter_contract_v1.json").read_text())
    policy["unit_source"]["sha256"] = sha256_file(units_path)
    policy_dir = tmp_path / "test_power_contract"
    policy_dir.mkdir()
    write_json(policy_dir / "charging_parameter_contract_v1.json", policy)
    monkeypatch.setattr(charging_component, "CONFIG_ROOT", policy_dir)

    staged = {"admin": road["admin_boundary_path"], "service": road["service_boundary_path"],
        "depot": depot_path, "depot_extract": extraction,
        "station": station_path, "units": units_path, "census": census_path,
        "station_provenance": source_meta, "units_provenance": source_meta,
        "census_provenance": source_meta}
    inventory_path = tmp_path / "inventory.json"
    write_json(inventory_path, {"synthetic_test_fixture": True,
        "files": {key: {"path": str(path.relative_to(tmp_path)), "sha256": sha256_file(path)}
                  for key, path in staged.items()},
        "source_vintages": {"synthetic_test_fixture": True},
        "charging": {key: key for key in ["station", "units", "census", "station_provenance",
                                            "units_provenance", "census_provenance"]},
        "cities": {"synthetic": {"area_crs": "EPSG:3857", "state_fips": "36",
            "boundaries": {"admin_boundary": "admin", "service_boundary": "service"},
            "depot": {"raw": "depot", "extraction": "depot_extract",
                      "pbf_sha256": meta["pbf_sha256"]}}}})
    config_path = tmp_path / "stage1.json"
    write_json(config_path, {"schema": SCHEMA, "output": str(tmp_path / "result"),
        "cities": {"synthetic": {**customer_sources, "area_crs": "EPSG:3857",
            "admin_boundary": str(road["admin_boundary_path"]),
            "service_boundary": str(road["service_boundary_path"])}},
        "components": {"inventory": str(inventory_path), "source_root": str(tmp_path),
                       "road_sources": str(road_root)},
        "customer": {"active_profile": "u_lt100", "include_model_imputed": True},
        "source_snapshots": {"microsoft_index": "synthetic", "osm": "synthetic", "nsi_archive": "synthetic"},
        "ai": {"command": "must-not-be-called"}})
    project = load_project(config_path)
    result = build(project, ["synthetic"], skip_ai=True)
    assert result["status"] == "assembled_draft"
    assert result["validation"][0]["passed"], result["validation"][0]["errors"]
    assert not result["whole_cle_frozen"] and not result["stage2_included"]
    city = tmp_path / "result/cities/synthetic"
    prep = json.loads((city / "customer_inputs/preparation_manifest.json").read_text())
    assert prep["mode"] == "raw_geometry_and_matching"
    population = pd.read_parquet(city / "customer_inputs/population.parquet")
    assert set(population.entity_id) == {"w-child1", "w-child2", "ms-single"}
    assert population.units_final.sum() == 4
    reader = CityLogisticsEnvironment(city)
    locations = reader.locations()
    counts = locations.groupby("location_type").candidate_eligible.sum().to_dict()
    assert counts == {"charging": 1, "customer": 3, "depot": 1}
    public = reader.graph()
    assert public.has_edge("a", "d") and not public.has_edge("d", "a")
    assert all(edge["oneway"] for edge in public["a"]["d"].values())
    assert all(edge["reference_travel_time_weekday_s"] >= edge["legal_travel_time_s"]
               for _, _, edge in public.edges(data=True))
    assert len(reader.access_costs()) == 5 and reader.verified and not reader.frozen
    customers = gpd.read_parquet(city / "customers/latent_locations.parquet")
    assert customers.geometry.geom_type.eq("Polygon").all()
    assert not customers.active_customer.any()
    assert (city / "customers/customer_lineage.json").exists()
