"""Small real-geometry preparation and explicit cache-import accounting."""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.building_inputs import (
    BuildingEvidenceReader, _verify_cached_nsi_roster, prepare_building_population,
)


def raw_fixture(tmp_path):
    ms = gpd.GeoDataFrame({"building_id": ["ms-parent", "ms-single", "ms-unmatched"],
                          "source_kind": ["global"] * 3, "height": [None, None, None]},
        geometry=[box(0, 0, .002, .001), box(.004, 0, .005, .001), box(.006, 0, .007, .001)], crs=4326)
    osm = gpd.GeoDataFrame({"osm_id": ["w-child1", "w-child2", "w-single", "w-only"],
        "building": ["yes", "apartments", "house", "house"], "building_part": [False] * 4,
        "tags": [json.dumps({"building": x}) for x in ["yes", "apartments", "house", "house"]]},
        geometry=[box(0, 0, .001, .001), box(.001, 0, .002, .001),
                  box(.004, 0, .005, .001), box(.009, 0, .010, .001)], crs=4326)
    footprint = gpd.GeoSeries([box(0, 0, .001, .001)], crs=4326).to_crs(3857).area.iloc[0] / .09290304
    nsi = gpd.GeoDataFrame({"fd_id": list(range(1, 9)), "bid": [None] * 8,
        "ftprntsrc": ["BING"] * 8, "ftprntid": [f"source-{i}" for i in range(8)],
        "ftprntsqft": [footprint] * 8, "resunits": [1, 4, 1, 1, 8, 1, 7, 0],
        "occtype": ["RES1", "RES3B", "RES1", "RES1", "RES3C", "RES1", "COM1", "RES1"]},
        geometry=[Point(.0005, .0005), Point(.0015, .0005), Point(.0045, .0005),
                  Point(.0065, .0005), Point(.02, .02), Point(.0095, .0005),
                  Point(.0045, .0005), Point(.0045, .0005)], crs=4326)
    sources = {"area_crs": "EPSG:3857", "snapshots": {"microsoft_index": "test", "osm": "test"}}
    for name, frame in [("microsoft", ms), ("osm", osm), ("nsi", nsi)]:
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        sources[name] = path
    return sources


def test_raw_subdivision_keeps_matched_population_and_all_original_nsi_denominator(tmp_path):
    sources = raw_fixture(tmp_path)
    output = tmp_path / "prepared"
    manifest = prepare_building_population(city="test", sources=sources, output_dir=output, axis_r=.3)
    population = pd.read_parquet(output / "population.parquet")
    records = pd.read_parquet(output / "source_records.parquet")
    assert set(population.entity_id) == {"w-child1", "w-child2", "ms-single"}
    assert set(records.fd_id) == set(range(1, 7)) and records.raw_resunits.sum() == 16
    assert records.owner_in_population.sum() == 3
    assert population.nsi_attribution_clear.all() and population.geometry_clear.all()
    assert population.set_index("entity_id").units_final.to_dict() == {
        "ms-single": 1, "w-child1": 1, "w-child2": 4}
    assert manifest["mode"] == "raw_geometry_and_matching"
    assert manifest["sources"]["nsi"]["role"] == "actual_executed_input"
    assert not manifest["height_admission_gate"] and not manifest["unit_cap_applied_to_population"]
    reader = BuildingEvidenceReader(sources, output / "geometry")
    row = population.loc[population.entity_id.eq("w-child1")].iloc[0]
    payload = reader.payload(row)
    assert payload["microsoft_building_ids"] == ["ms-parent"]
    assert payload["osm_objects"][0]["tags"] == {"building": "yes"}
    assert "units_raw_nsi" not in payload and "occtype" not in payload
    geometry = reader.geometry(population)
    assert geometry.geometry.covers(gpd.GeoSeries([
        Point(.0045, .0005), Point(.0005, .0005), Point(.0015, .0005)], crs=4326)).all()
    obstacles = reader.obstacles()
    assert set(obstacles.building_id) == {"ms-single", "ms-unmatched", "w-child1", "w-child2"}
    assert "ms-parent" not in set(obstacles.building_id) and "w-only" not in set(obstacles.building_id)


def test_cached_import_retains_owner_ledger_without_reusing_old_admission(tmp_path):
    sources = raw_fixture(tmp_path)
    native = gpd.read_parquet(sources["nsi"]).iloc[:3].copy()
    native["resunits"] = [1, 20, 30]
    native["occtype"] = ["RES1", "RES3E", "RES3E"]
    native.to_parquet(sources["nsi"], index=False)
    cache = tmp_path / "cache"
    for name in ["locations", "records", "geometry/test"]:
        (cache / name).mkdir(parents=True)
    entities = pd.DataFrame({"entity_index": [0, 1, 2], "entity_id": ["ms-single", "w-only", "unmatched"],
        "kind": ["ms", "osm_add", "ms"], "trusted_osm_alias_group": [2, -1, -1],
        "osm_group": [-1, 3, -1], "semantic_state": ["house", "residential", "unknown"],
        "eligible": [True] * 3, "geometry_conflict": [False] * 3,
        "weak_semantic_conflict": [False] * 3, "area_m2": [100.] * 3,
        "primary_path_id": ["ms-single", "w-only", "unmatched"],
        "admitted": [False] * 3, "volume_excluded": [True] * 3})
    entities["semantic_state"] = ["residential", "residential", "unknown"]
    entities.to_parquet(cache / "locations/test.parquet", index=False)
    pd.DataFrame({"fd_id": [1, 2, 3], "owner": [0, 1, -1], "raw_resunits": [1, 20, 30],
        "occtype": ["RES1", "RES3E", "RES3E"], "source_group_conflict": [False] * 3,
        "source_geometry_counterevidence": [False] * 3, "base_direct_owner": [0, 1, -1],
        "geometry_block_reason": [""] * 3}).to_parquet(cache / "records/test.parquet", index=False)
    pd.DataFrame({"osm_group": [2], "osm_ids": ["w-single"], "canonical_id": ["w-single"]}).to_parquet(
        cache / "geometry/test/full_osm_registry.parquet", index=False)
    pd.DataFrame({"entity_index": [0], "parent_ms_id": ["ms-single"]}).to_parquet(
        cache / "geometry/test/parent_child.parquet", index=False)
    output = tmp_path / "imported"
    manifest = prepare_building_population(city="test", sources=sources, output_dir=output,
                                           axis_r=.3, geometry_cache=cache)
    population = pd.read_parquet(output / "population.parquet")
    records = pd.read_parquet(output / "source_records.parquet")
    assert list(population.entity_id) == ["ms-single"]
    assert population.units_final.iloc[0] == 1 and population.nsi_attribution_clear.iloc[0]
    assert "admitted" not in population and "volume_excluded" not in population
    assert list(records.owner) == [0, 1, -1] and records.raw_resunits.sum() == 51
    assert list(records.owner_in_population) == [True, False, False]
    assert manifest["mode"] == "cached_geometry_and_owners_import"
    assert manifest["ordinary_positive_nsi_records"] == 3
    assert manifest["NSI_roster_verified"]
    assert manifest["cache_original_raw_source_binding"] == "not_independently_verified"
    assert manifest["sources"]["nsi"]["role"] == "current_reference_inputs_not_reexecuted"
    assert Path(manifest["geometry_dir"]).joinpath("full_osm_registry.parquet").is_file()
    (output / "assessments.jsonl").write_text('{"entity_id":"ms-single","human_review":true}\n')
    before = {name: (output / name).read_bytes() for name in [
        "population.parquet", "source_records.parquet", "preparation_manifest.json", "assessments.jsonl"]}
    reused = prepare_building_population(city="test", sources=sources, output_dir=output,
                                         axis_r=.3, geometry_cache=cache)
    assert reused["reused"]
    assert all((output / name).read_bytes() == value for name, value in before.items())
    altered = pd.read_parquet(cache / "records/test.parquet")
    altered.loc[0, "raw_resunits"] = 2
    altered.to_parquet(cache / "records/test.parquet", index=False)
    with pytest.raises(ValueError, match="fresh output directory"):
        prepare_building_population(city="test", sources=sources, output_dir=output,
                                     axis_r=.3, geometry_cache=cache)
    assert all((output / name).read_bytes() == value for name, value in before.items())


def test_preparation_reuses_unchanged_inputs_and_preserves_reviews_on_source_change(tmp_path):
    sources = raw_fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare_building_population(city="test", sources=sources, output_dir=output, axis_r=.3)
    (output / "assessments.jsonl").write_text('{"entity_id":"ms-single","human_review":true}\n')
    before = {name: (output / name).read_bytes() for name in [
        "population.parquet", "source_records.parquet", "preparation_manifest.json", "assessments.jsonl"]}
    assert prepare_building_population(city="test", sources=sources, output_dir=output, axis_r=.3)["reused"]
    with pytest.raises(ValueError, match="fresh output directory"):
        prepare_building_population(city="test", sources=sources, output_dir=output, axis_r=.4)
    native = gpd.read_parquet(sources["nsi"])
    native.loc[0, "resunits"] = 2
    native.to_parquet(sources["nsi"], index=False)
    with pytest.raises(ValueError, match="fresh output directory"):
        prepare_building_population(city="test", sources=sources, output_dir=output, axis_r=.3)
    assert all((output / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize("field", ["fd_id", "raw_resunits", "occtype", "native_lon", "native_lat",
                                   "ftprntsrc", "ftprntid"])
def test_first_cache_import_rejects_different_nsi_roster_fields(tmp_path, field):
    sources = raw_fixture(tmp_path)
    native = gpd.read_parquet(sources["nsi"]).iloc[:6].copy()
    cached = pd.DataFrame(native.drop(columns="geometry"))
    cached["raw_resunits"] = cached.pop("resunits")
    cached["native_lon"] = native.geometry.x
    cached["native_lat"] = native.geometry.y
    verified = _verify_cached_nsi_roster(cached, sources["nsi"])
    assert verified["verified"] and set(verified["fields"]) == {
        "fd_id", "resunits", "occtype", "native_lon", "native_lat", "ftprntsrc", "ftprntid"}
    if field == "fd_id":
        cached.loc[0, field] += 100
    elif field in {"raw_resunits", "native_lon", "native_lat"}:
        cached.loc[0, field] += 1
    else:
        cached.loc[0, field] = "RES3F" if field == "occtype" else "different"
    with pytest.raises(ValueError, match="Cached NSI roster differs"):
        _verify_cached_nsi_roster(cached, sources["nsi"])
