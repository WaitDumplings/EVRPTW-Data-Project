"""Final-child city membership uses its own polygon and retains source ledgers."""
import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from evrptw_cle.building_scope import prepare_city_scope


def scope_fixture(tmp_path):
    sources = {}
    ms = gpd.GeoDataFrame({"building_id": ["ms-retained", "ms-parent"]},
                         geometry=[box(0, 0, 2, 2), box(0, 3, 2, 5)], crs=4326)
    osm = gpd.GeoDataFrame({"osm_id": ["w-alias", "w-in", "w-out"],
                           "building": ["yes"] * 3, "tags": ["{}"] * 3},
                          geometry=[box(.2, 0, 2.2, 2), box(0, 3, .8, 5), box(.8, 3, 2, 5)], crs=4326)
    for name, frame in [("microsoft", ms), ("osm", osm)]:
        sources[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(sources[name], index=False)
    directory = tmp_path / "customer_inputs"
    geometry = directory / "geometry"
    geometry.mkdir(parents=True)
    pd.DataFrame({"osm_group": [0, 1, 2], "canonical_id": ["w-alias", "w-in", "w-out"],
                  "osm_ids": ["w-alias", "w-in", "w-out"]}).to_parquet(
        geometry / "full_osm_registry.parquet", index=False)
    pd.DataFrame({"entity_index": [1, 2], "parent_ms_id": ["ms-parent"] * 2}).to_parquet(
        geometry / "parent_child.parquet", index=False)
    population = pd.DataFrame({"entity_index": [0, 1, 2],
        "entity_id": ["ms-retained", "w-in", "w-out"], "kind": ["ms", "split_osm", "split_osm"],
        "primary_path_id": ["ms-retained", "w-in", "w-out"],
        "geometry_clear": [True, True, False]})
    population.to_parquet(directory / "population.parquet", index=False)
    pd.DataFrame({"fd_id": [1], "raw_resunits": [10]}).to_parquet(
        directory / "source_records.parquet", index=False)
    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(geometry=[box(-1, -1, 1.1, 7)], crs=4326).to_file(boundary)
    return sources, directory, geometry, boundary


def test_scope_uses_canonical_not_alias_point_and_keeps_all_buildings(tmp_path):
    sources, directory, geometry, boundary = scope_fixture(tmp_path)
    population_before = (directory / "population.parquet").read_bytes()
    sources_before = (directory / "source_records.parquet").read_bytes()
    manifest = prepare_city_scope(population_path=directory / "population.parquet",
        sources=sources, geometry_dir=geometry, boundary_path=boundary, output_dir=directory)
    scope = pd.read_parquet(directory / "scope.parquet").set_index("entity_id")
    assert scope.service_domain_eligible.to_dict() == {
        "ms-retained": True, "w-in": True, "w-out": False}
    assert scope.loc["ms-retained", "canonical_longitude"] == 1
    assert scope.loc["w-out", "canonical_longitude"] == 1.4
    assert manifest["outside_service_boundary"] == 1 and manifest["buildings"] == 3
    assert manifest["by_kind"]["split_osm"]["outside"] == 1
    assert (directory / "population.parquet").read_bytes() == population_before
    assert (directory / "source_records.parquet").read_bytes() == sources_before
    saved = json.loads((directory / "scope_manifest.json").read_text())
    assert set(saved["inputs"]) == {"population", "service_boundary", "microsoft", "osm"}
    assert not saved["polygons_clipped"] and not saved["population_rows_dropped"]


def test_identical_scope_reuses_sidecar_and_changed_boundary_requires_new_version(tmp_path):
    sources, directory, geometry, boundary = scope_fixture(tmp_path)
    kwargs = dict(population_path=directory / "population.parquet", sources=sources,
                  geometry_dir=geometry, boundary_path=boundary, output_dir=directory)
    prepare_city_scope(**kwargs)
    before = (directory / "scope.parquet").stat().st_mtime_ns
    assert prepare_city_scope(**kwargs)["reused"]
    assert (directory / "scope.parquet").stat().st_mtime_ns == before
    gpd.GeoDataFrame(geometry=[box(-1, -1, 3, 7)], crs=4326).to_file(boundary)
    with pytest.raises(ValueError, match="fresh output directory"):
        prepare_city_scope(**kwargs)
    assert (directory / "scope.parquet").stat().st_mtime_ns == before
