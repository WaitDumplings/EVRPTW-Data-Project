"""Small end-to-end CLE build/read-back; synthetic inputs never enter production."""

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from evrptw_cle import charging_component
from evrptw_cle.cle_assembly import assemble_city_cle
from evrptw_cle.cle_validation import verify_city_cle
from evrptw_cle.cle_reader import CityLogisticsEnvironment
from evrptw_cle.cli import depots
from evrptw_cle.common_speed_binding import build_common_speed_component
from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.local_road_access import build_access_context
from evrptw_cle.residential_sites import build_registered_site_layer
from evrptw_cle.runtime_paths import CONFIG_ROOT
from evrptw_cle.util import sha256_file, write_json
from test_charging_component import _station, _units
from test_residential_sites import inputs
from test_road_direction_barrier import _inputs


def test_build_and_independently_read_complete_synthetic_CLE(tmp_path,monkeypatch):
    options=_inputs(tmp_path)
    extraction=options["source_dir"] / "extraction.json"
    meta=json.loads(extraction.read_text())
    meta["access_nodes_complete_for_retained_ways"]=True
    meta["synthetic_test_fixture"]=True
    way_path=options["source_dir"] / "osm_ways.parquet"
    ways=gpd.read_parquet(way_path)
    tags=json.loads(ways.loc[0,"tags_json"])
    tags["maxspeed"]="25 mph"
    ways.loc[0,"tags_json"]=json.dumps(tags)
    ways.to_parquet(way_path,index=False)
    meta["outputs"][way_path.name]=sha256_file(way_path)
    write_json(extraction,meta)
    admin=options["admin_boundary_path"]
    (tmp_path / "admin_boundary.geojson").write_bytes(admin.read_bytes())
    city=tmp_path / "cities/synthetic"
    build_access_context(**options,output_dir=tmp_path / "seed")
    build_delivery_access_context(**options,seed_dir=tmp_path / "seed",output_dir=city / "roads")
    records,fp,buildings,complexes=inputs()
    sources={}
    for name,frame in [("nsi_crosswalk",records),("microsoft_footprints",fp),("osm_buildings",buildings),("osm_complexes",complexes)]:
        frame=frame.set_crs(3857,allow_override=True).to_crs(4326)
        path=tmp_path / (name+".parquet")
        frame.to_parquet(path,index=False)
        sources[name]=dict(path=str(path),sha256=sha256_file(path))
    sources["public_masks"]=dict(path=str(options["mask_path"]),sha256=sha256_file(options["mask_path"]))
    registry=tmp_path / "customer_sources.json"
    write_json(registry,dict(policy="identified_residential_site_v3",source_vintages={"synthetic_test_only":True},cities=dict(synthetic=dict(
        sources=sources,provenance={},area_crs="EPSG:3857",boundary_sha256=sha256_file(options["service_boundary_path"]),
    ))))
    build_registered_site_layer(registry_path=registry,city_slug="synthetic",boundary_path=options["service_boundary_path"],graph_path=city / "roads/graph_operational.graphml",output_dir=city / "customers")
    raw=tmp_path / "depots.parquet"
    gpd.GeoDataFrame(dict(source_osm_id=["w10"],building=["warehouse"],name=["Example warehouse"]),geometry=[box(20,100,30,110)],crs=3857).to_crs(4326).to_parquet(raw,index=False)
    depmanifest=tmp_path / "depots.json"
    write_json(depmanifest,dict(pbf_sha256=meta["pbf_sha256"],raw_depot_sha256=sha256_file(raw)))
    depots(SimpleNamespace(city="synthetic",area_crs="EPSG:3857",context=city / "roads",boundary=options["service_boundary_path"],masks=options["mask_path"],raw_depots=raw,source_manifest=depmanifest,output=city / "depots",footprints=Path(sources["microsoft_footprints"]["path"])))
    station,unit=tmp_path / "stations.csv",tmp_path / "units.csv"
    frame=_station()
    frame.to_csv(station,index=False)
    _units(frame).to_csv(unit,index=False)
    testconfig=tmp_path / "test_policy"
    testconfig.mkdir()
    power=json.loads((CONFIG_ROOT / "charging_parameter_contract_v1.json").read_text())
    power["unit_source"]["sha256"]=sha256_file(unit)
    write_json(testconfig / "charging_parameter_contract_v1.json",power)
    monkeypatch.setattr(charging_component,"CONFIG_ROOT",testconfig)
    chmanifest=tmp_path / "charging_sources.json"
    write_json(chmanifest,dict(schema=charging_component.SOURCE_SCHEMA,station_sha256=sha256_file(station),units_sha256=sha256_file(unit)))
    charging_component.build_charging_component(
        city_slug="synthetic",station_path=station,units_path=unit,source_manifest_path=chmanifest,
        boundary_path=options["service_boundary_path"],mask_path=options["mask_path"],context_dir=city / "roads",
        area_crs="EPSG:3857",output_dir=city / "charging",footprints_path=Path(sources["microsoft_footprints"]["path"]),
    )
    build_common_speed_component(city_slug="synthetic",context_dir=city / "roads",output_dir=city / "speeds",state_fips="36")
    inventory=tmp_path / "inventory.json"
    write_json(inventory,dict(synthetic_test_only=True))
    assembly=assemble_city_cle(city_dir=city,boundary_paths=dict(admin_boundary=admin,service_boundary=options["service_boundary_path"]),source_inventory_path=inventory,customer_registry_path=registry,road_source_dir=options["source_dir"])
    assert assembly["summary"]["customer"]["retained"]==2
    assert assembly["summary"]["depot"]["retained"]==1
    result=verify_city_cle(city)
    assert result["passed"] and result["verified_terminal_paths"]>0
    assert result["no_data_or_eligibility_modified"]
    assert not result["instance_activation_included"]
    saved=pd.read_parquet(city / "locations.parquet")
    assert saved.residential_units.isna().all()
    reader=CityLogisticsEnvironment(city)
    public=reader.graph("public")
    assert len(reader.locations())==4
    assert public.graph["cle_layer"]=="public"
    assert all(d["reference_travel_time_weekday_s"]>=d["legal_travel_time_s"] for _,_,d in public.edges(data=True))
    verification_sha=sha256_file(city / "cle_verification.json")
    write_json(tmp_path / "stage1_freeze.json",dict(frozen=True,cities=dict(synthetic=dict(
        cle_manifest_sha256=sha256_file(city / "cle_manifest.json"),verification_sha256=verification_sha,
    ))))
    assert CityLogisticsEnvironment(city).frozen
    assert verify_city_cle(city)["passed"]
    assert sha256_file(city / "cle_verification.json")==verification_sha
