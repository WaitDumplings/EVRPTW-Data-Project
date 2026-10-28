import gzip
import json

import geopandas as gpd
import pytest
from shapely.geometry import Point, box, mapping

from evrptw_cle.source_preparation import prepare_customer_city
from evrptw_cle.util import sha256_file


def source_fixture(tmp_path, *, conflicting=False):
    root = tmp_path / "inputs"
    root.mkdir()
    files = {}

    def entry(key, path):
        files[key] = dict(path=path.name, archive_path=path.name, sha256=sha256_file(path))
        return key

    boundary = root / "land.geojson"
    gpd.GeoDataFrame(geometry=[box(0, 0, 0.01, 0.01)], crs=4326).to_file(boundary, driver="GeoJSON")
    new = root / "new.parquet"
    gpd.GeoDataFrame(
        dict(building_id=["roof"]), geometry=[box(0.001, 0.001, 0.002, 0.002)], crs=4326
    ).to_parquet(new)
    old = root / "old.parquet"
    gpd.GeoDataFrame(
        dict(building_id=["older"]), geometry=[box(0.001, 0.001, 0.002, 0.002)], crs=4326
    ).to_parquet(old)
    props = dict(
        fd_id=1,
        bid="a",
        occtype="RES1-1SNB",
        st_damcat="RES",
        resunits=1,
        ftprntid="roof",
        ftprntsrc="S",
        ftprntsqft=10,
        usastrucid=5,
        cbfips="00100",
    )
    feature = dict(type="Feature", properties=props, geometry=mapping(Point(0.0015, 0.0015)))
    payload = root / "tile.geojsonseq.gz"
    features = (
        [feature, dict(feature, properties=dict(props, resunits=2))]
        if conflicting
        else [feature, feature]
    )
    with gzip.open(payload, "wb") as handle:
        for row in features:
            handle.write(b"\x1e" + json.dumps(row).encode() + b"\n")
    metadata = root / "source.json"
    metadata.write_text(
        json.dumps(dict(request_sha256="f" * 64, response_sha256=sha256_file(payload)))
    )
    request = root / "request.json"
    request.write_text(json.dumps(dict(frozen=True)))
    baseline = root / "baseline.parquet"
    gpd.GeoDataFrame(
        dict(
            fd_id=[1],
            occtype=["RES1-1SNB"],
            resunits=[1],
            composite_status=["matched_unique_containment"],
            composite_building_id=["global20260813:roof"],
        ),
        geometry=[Point(0.0015, 0.0015)],
        crs=4326,
    ).to_parquet(baseline)
    city = dict(
        area_crs="EPSG:32631",
        boundaries=dict(service_boundary=entry("boundary", boundary)),
        geometry=dict(global_=[entry("new", new)], legacy=[entry("old", old)]),
        nsi_tiles=[
            dict(
                tile_id="tile",
                request_sha256="f" * 64,
                response=entry("payload", payload),
                source=entry("source", metadata),
                request=entry("request", request),
            )
        ],
        sources=dict(nsi_crosswalk=entry("baseline", baseline)),
    )
    city["geometry"]["global"] = city["geometry"].pop("global_")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            dict(
                schema="evrptw_frozen_source_archive_v1",
                files=files,
                source_vintages=dict(nsi="frozen"),
                cities=dict(example=city),
            )
        )
    )
    return inventory, root


def test_raw_nsi_reparse_and_composite_preserve_native_source_evidence(tmp_path):
    inventory, root = source_fixture(tmp_path)
    report = prepare_customer_city(
        inventory_path=inventory,
        source_root=root,
        city_slug="example",
        output_dir=tmp_path / "prepared",
    )
    assert report["nsi_parse"]["duplicate_records_removed"] == 1
    assert report["nsi_parse"]["ordinary_residential_points"] == 1
    assert "baseline_semantic_comparison" not in report
    assert report["microsoft"]["legacy_additions"] == 0
    table = gpd.read_parquet(tmp_path / "prepared/nsi_crosswalk.parquet")
    assert table.fd_id.tolist() == [1] and table.resunits.tolist() == [1]
    assert table.composite_status.tolist() == ["matched_unique_containment"]
    for item in report["outputs"].values():
        assert sha256_file(tmp_path / "prepared" / item["path"]) == item["sha256"]


def test_conflicting_raw_nsi_id_fails_before_derived_output(tmp_path):
    inventory, root = source_fixture(tmp_path, conflicting=True)
    with pytest.raises(ValueError, match="Conflicting duplicate NSI"):
        prepare_customer_city(
            inventory_path=inventory,
            source_root=root,
            city_slug="example",
            output_dir=tmp_path / "prepared",
        )
    assert not (tmp_path / "prepared").exists()


def test_new_city_does_not_require_a_historical_crosswalk(tmp_path):
    inventory, root = source_fixture(tmp_path)
    value = json.loads(inventory.read_text())
    del value["cities"]["example"]["sources"]["nsi_crosswalk"]
    inventory.write_text(json.dumps(value))
    report = prepare_customer_city(inventory_path=inventory, source_root=root,
                                   city_slug="example", output_dir=tmp_path / "new-city")
    assert "baseline_semantic_comparison" not in report
    assert report["nsi_parse"]["ordinary_residential_points"] == 1
