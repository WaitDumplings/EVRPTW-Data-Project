from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

from evrptw_cle.facilities import (
    _afdc_manifest_output_sha256,
    _anchor_points_to_edges,
    _connector_tokens,
)


def test_connector_tokens_normalize_afdc_values() -> None:
    assert _connector_tokens("CHADEMO J1772COMBO TESLA") == {
        "CHADEMO",
        "J1772COMBO",
        "TESLA",
    }
    assert _connector_tokens(None) == set()


def test_afdc_manifest_hash_supports_raw_and_resolved_schemas() -> None:
    assert _afdc_manifest_output_sha256({"sha256": "raw-hash", "output": "/tmp/a.csv"}) == (
        "raw-hash"
    )
    assert (
        _afdc_manifest_output_sha256({"output": {"path": "/tmp/a.csv", "sha256": "resolved-hash"}})
        == "resolved-hash"
    )


def test_facility_point_retains_directed_edge_refs() -> None:
    points = gpd.GeoDataFrame(
        {"charger_id": ["afdc_1"], "facility_anchor_id": ["afdc_1"]},
        geometry=[Point(-117.1, 32.8001)],
        crs="EPSG:4326",
    )
    refs = json.dumps(
        [
            {
                "u": "1",
                "v": "2",
                "key": "0",
                "length_m": 1870.0,
                "geometry_orientation": "same_as_physical",
            }
        ]
    )
    edges = gpd.GeoDataFrame(
        {
            "physical_edge_id": ["edge-a"],
            "directed_edge_refs": [refs],
            "directed_edge_ref_count": [1],
            "edge_u": ["1"],
            "edge_v": ["2"],
            "edge_key": ["0"],
            "highway": ["residential"],
            "road_name": ["Example Road"],
            "oneway": [True],
        },
        geometry=[LineString([(-117.11, 32.8), (-117.09, 32.8)])],
        crs="EPSG:4326",
    )
    anchored = _anchor_points_to_edges(points, edges, "EPSG:32611")
    assert anchored.loc[0, "physical_edge_id"] == "edge-a"
    assert anchored.loc[0, "directed_edge_refs"] == refs
    assert anchored.loc[0, "road_access_distance_m"] > 0
    assert bool(anchored.loc[0, "connector_bidirectional"])
    assert (
        json.loads(anchored.loc[0, "directed_projection_offsets"])[0]["projection_fraction_from_u"]
        > 0
    )
    assert anchored.loc[0, "road_anchor_method"] == (
        "facility_point_to_operational_edge_projection"
    )


def test_facility_cache_rejects_changed_source_or_mask(tmp_path):
    from evrptw_cle import facilities
    from evrptw_cle.public_locations import PUBLIC_LOCATION_POLICY
    from evrptw_cle.util import sha256_file

    mask = tmp_path / "masks.parquet"
    source = tmp_path / "depots.csv"
    mask.write_bytes(b"mask-v1")
    source.write_bytes(b"source-v1")
    manifest = tmp_path / "facility_manifest.json"
    payload = {
        "schema": "evrptw_city_facility_layers_v6",
        "generation_module_sha256": sha256_file(Path(facilities.__file__)),
        "generation_dependencies": facilities._generation_dependencies(),
        "public_location_policy": {"policy": PUBLIC_LOCATION_POLICY},
        "inputs": {
            "depot_candidates": {"sha256": sha256_file(source)},
            "public_exclusion_masks": {"path": str(mask), "sha256": sha256_file(mask)},
        },
    }
    manifest.write_text(json.dumps(payload))
    assert facilities.facility_inputs_current(manifest, depot_candidates=source)
    mask.write_bytes(b"mask-v2")
    assert not facilities.facility_inputs_current(manifest, depot_candidates=source)
    mask.write_bytes(b"mask-v1")
    source.write_bytes(b"source-v2")
    assert not facilities.facility_inputs_current(manifest, depot_candidates=source)


def test_retired_facility_paths_cannot_bypass_the_common_graph_contract():
    import pytest
    from evrptw_cle.facilities import build_charger_layers, build_facility_layers

    with pytest.raises(RuntimeError,match="retired"):
        build_charger_layers()
    with pytest.raises(RuntimeError,match="retired"):
        build_facility_layers()
