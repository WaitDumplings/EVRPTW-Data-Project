import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from evrptw_cle.terminal_access import ACCESS_NODE_TAG_CONTRACT
from evrptw_cle.util import sha256_file


def test_restriction_scan_keeps_outside_envelope_nodes_of_retained_complete_way(tmp_path):
    osmium = pytest.importorskip("osmium")
    path = Path(__file__).parents[1] / "scripts/extract_frozen_access_roads.py"
    spec = importlib.util.spec_from_file_location("road_extractor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pbf = tmp_path / "frozen.osm.pbf"
    header = osmium.io.Header()
    header.set("osmosis_replication_timestamp", "2026-10-25T21:21:21Z")
    with osmium.SimpleWriter(pbf, header=header) as writer:
        for node, location, tags in [
            (1, (0, 0), {}),
            (2, (0.25, 0), {"barrier": "bollard"}),
            (3, (0.30, 0), {"access:delivery": "no"}),
            (4, (0.40, 0), {"barrier": "gate", "access": "private"}),
            (5, (0.01, 0), {"locked": "yes"}),
        ]:
            writer.add_node(osmium.osm.mutable.Node(id=node, location=location, tags=tags))
        writer.add_way(osmium.osm.mutable.Way(id=10, nodes=[1, 2, 3], tags={"highway": "service"}))
    admin = tmp_path / "admin.geojson"
    gpd.GeoDataFrame(geometry=[box(-100, -100, 100, 100)], crs=3857).to_crs(4326).to_file(
        admin, driver="GeoJSON"
    )
    configs = tmp_path / "generator/configs"
    configs.mkdir(parents=True)
    (configs / "city_boundaries.json").write_text(
        json.dumps(
            {
                "cities": {
                    "sample": {"admin_boundary": {"path": str(admin), "sha256": sha256_file(admin)}}
                }
            }
        )
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"sha256": sha256_file(pbf)}))
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "regions": {
                    "sample": {
                        "path": str(pbf),
                        "metadata": str(metadata),
                        "frozen_url": "frozen-test-source",
                        "cities": ["sample"],
                    }
                }
            }
        )
    )
    output = tmp_path / "output"
    module.extract_region(
        cohort_path=cohort, generator_root=configs.parent, output_root=output, region="sample"
    )
    ways = gpd.read_parquet(output / "sample/osm_ways.parquet")
    nodes = gpd.read_parquet(output / "sample/osm_access_nodes.parquet")
    assert json.loads(ways.node_ids_json.iloc[0]) == [1, 2, 3]
    assert set(nodes.osmid) == {"2", "3", "5"}
    manifest = json.loads((output / "sample/extraction.json").read_text())
    assert manifest["outside_envelope_referenced_access_node_count"] == 2
    assert manifest["access_nodes_complete_for_retained_ways"]
    assert manifest["access_node_tag_contract"] == ACCESS_NODE_TAG_CONTRACT
    from evrptw_cle.raw_road_validation import verify_raw_road_nodes

    proof=verify_raw_road_nodes(output,tmp_path / "raw-node-proof.json")
    assert proof["passed"] and proof["cities"]["sample"]["checked_referenced_access_nodes"]==2
    # Even a newly labeled derived-file hash cannot hide a missing PBF barrier.
    nodes.loc[nodes.osmid.ne("2")].to_parquet(output / "sample/osm_access_nodes.parquet",index=False)
    manifest["outputs"]["osm_access_nodes.parquet"]=sha256_file(output / "sample/osm_access_nodes.parquet")
    (output / "sample/extraction.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="restriction node missing"):
        verify_raw_road_nodes(output,tmp_path / "must-not-pass.json")
