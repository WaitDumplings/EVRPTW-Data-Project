import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.local_road_access import _directions, build_access_context, classify_way
from evrptw_cle.protected_connectivity import build_directed_component_index
from evrptw_cle.terminal_access import (
    ACCESS_NODE_TAG_CONTRACT,
    ACCESS_NODE_TAGS,
    _effective_motor_access,
)
from evrptw_cle.util import sha256_file


def _inputs(tmp_path, *, terminal_tags=None, node_tags=None, tie=False):
    """Small real shared-node fixtures exercise the complete road builders."""
    source = tmp_path / "source"
    source.mkdir()
    admin = tmp_path / "admin.geojson"
    service = tmp_path / "service.geojson"
    masks = tmp_path / "masks.parquet"
    boundary = gpd.GeoDataFrame(geometry=[box(-100, -100, 400, 400)], crs=3857).to_crs(4326)
    boundary.to_file(admin, driver="GeoJSON")
    boundary.to_file(service, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[], crs=3857).to_parquet(masks)
    roads = [
        dict(
            osmid="1",
            node_ids_json=json.dumps(["a", "b"]),
            tags_json=json.dumps({"highway": "residential"}),
            geometry=LineString([(0, 0), (100, 0)]),
        ),
    ]
    if tie:
        roads.extend(
            [
                dict(
                    osmid="2",
                    node_ids_json=json.dumps(["y", "z"]),
                    tags_json=json.dumps({"highway": "residential"}),
                    geometry=LineString([(200, 0), (300, 0)]),
                ),
                dict(
                    osmid="3",
                    node_ids_json=json.dumps(["b", "y"]),
                    tags_json=json.dumps({"highway": "residential", "oneway": "yes"}),
                    geometry=LineString([(100, 0), (200, 0)]),
                ),
            ]
        )
    else:
        roads.append(
            dict(
                osmid="2",
                node_ids_json=json.dumps(["a", "c", "d"]),
                tags_json=json.dumps({"highway": "service", **(terminal_tags or {})}),
                geometry=LineString([(0, 0), (0, 100), (0, 200)]),
            )
        )
    gpd.GeoDataFrame(roads, geometry="geometry", crs=3857).to_crs(4326).to_parquet(
        source / "osm_ways.parquet", index=False
    )
    nodes = (
        []
        if node_tags is None
        else [dict(osmid="c", tags_json=json.dumps(node_tags), geometry=Point(0, 100))]
    )
    gpd.GeoDataFrame(
        nodes, columns=["osmid", "tags_json", "geometry"], geometry="geometry", crs=3857
    ).to_crs(4326).to_parquet(source / "osm_access_nodes.parquet", index=False)
    extraction = dict(
        city_slug="synthetic",
        pbf_sha256="frozen-synthetic-pbf",
        pbf_replication_timestamp_utc="2026-10-25T21:21:21Z",
        access_node_tag_contract=ACCESS_NODE_TAG_CONTRACT,
        admin_boundary_sha256=sha256_file(admin),
        outputs={
            name: sha256_file(source / name)
            for name in ["osm_ways.parquet", "osm_access_nodes.parquet"]
        },
    )
    (source / "extraction.json").write_text(json.dumps(extraction))
    return dict(
        source_dir=source,
        admin_boundary_path=admin,
        service_boundary_path=service,
        mask_path=masks,
        area_crs="EPSG:3857",
    )


def test_closed_terminal_at_shared_endpoint_keeps_geometry_and_length_consistent(tmp_path):
    import shapely
    from evrptw_cle.local_road_access import PATH_GEOMETRY_POLICY

    options=_inputs(tmp_path)
    way=options["source_dir"] / "osm_ways.parquet"
    roads=gpd.read_parquet(way).to_crs(3857)
    roads.loc[1,"node_ids_json"]=json.dumps(["a","c","d","e","a"])
    roads.loc[1,"geometry"]=LineString([(0,0),(-50,0),(-50,50),(0,50),(0,0)])
    roads.to_crs(4326).to_parquet(way,index=False)
    manifest=options["source_dir"] / "extraction.json"
    extraction=json.loads(manifest.read_text())
    extraction["outputs"][way.name]=sha256_file(way)
    manifest.write_text(json.dumps(extraction))
    build_access_context(**options,output_dir=tmp_path / "seed")
    current,_=build_delivery_access_context(**options,seed_dir=tmp_path / "seed",output_dir=tmp_path / "roads")
    ring=current.physical.loc[current.physical.osmid.eq("2")].iloc[0]
    location=gpd.GeoDataFrame(dict(latent_service_location_id=["shared-ring-endpoint"]),geometry=[Point(0,0)],crs=3857)
    result=current.anchor(location,selected_edge_ids=[ring.physical_edge_id]).iloc[0]
    assert result.local_access_confirmed and result.requires_terminal_paths
    assert result.terminal_path_geometry_policy==PATH_GEOMETRY_POLICY
    for direction in ["inbound","outbound"]:
        path=json.loads(result["terminal_"+direction+"_path_json"])
        assert path and result["terminal_"+direction+"_length_m"]==pytest.approx(0)
        for part in path:
            geometry=gpd.GeoSeries([shapely.from_wkt(part["geometry_wkt"])],crs=4326).to_crs(3857).iloc[0]
            assert geometry.length==pytest.approx(part["length_m"],abs=1e-7)


@pytest.mark.parametrize(
    "tags,expected",
    [
        ({"highway": "motorway"}, (True, False)),
        ({"highway": "motorway", "oneway": "no"}, (True, True)),
        ({"highway": "residential", "oneway": "-1"}, (False, True)),
        ({"highway": "residential", "oneway": "reversible"}, (False, False)),
        ({"highway": "residential", "oneway": "alternating"}, (False, False)),
        ({"highway": "residential", "oneway": "unknown"}, (False, False)),
    ],
)
def test_static_directions_never_invent_reverse_or_variable_access(tags, expected):
    assert _directions(tags) == expected


@pytest.mark.parametrize("field", ["access", "motor_vehicle", "vehicle", "motorcar"])
def test_delivery_only_tag_is_allowed_only_in_terminal_layer(field):
    result = classify_way({"highway": "residential", field: "delivery"})
    assert result["allowed"]
    assert result["legal_access_tier"] == "delivery_only"
    assert result["access_layer"] == "osm_terminal_only"


@pytest.mark.parametrize(
    "baseline",
    [
        "private",
        "no",
        "destination",
        "customers",
        "delivery",
        "agricultural",
    ],
)
@pytest.mark.parametrize("permission", ["yes", "permissive"])
def test_delivery_exception_never_grants_public_through_access(baseline, permission):
    tags = {"highway": "residential", "access": baseline, "access:delivery": permission}
    assert _effective_motor_access(tags) == ("access:delivery", "delivery_only")
    result = classify_way(tags)
    assert result["allowed"] and result["access_layer"] == "osm_terminal_only"
    tags["motor_vehicle"] = "no"
    assert _effective_motor_access(tags) == ("motor_vehicle", "prohibited")
    assert not classify_way(tags)["allowed"]


@pytest.mark.parametrize("baseline", [None, "yes", "permissive"])
def test_redundant_delivery_permission_preserves_public_baseline(baseline):
    tags = {"highway": "residential", "access:delivery": "yes"}
    if baseline is not None:
        tags["access"] = baseline
    assert classify_way(tags)["access_layer"] == "operational_public"


@pytest.mark.parametrize(
    "node_tags",
    [
        {"barrier": "bollard"},
        {"locked": "yes"},
        {"access:delivery": "no"},
    ],
)
def test_interior_hard_restriction_splits_and_blocks_both_incident_segments(tmp_path, node_tags):
    assert set(node_tags) <= set(ACCESS_NODE_TAGS)
    options = _inputs(tmp_path, node_tags=node_tags)
    seed, _ = build_access_context(**options, output_dir=tmp_path / "seed")
    service = seed.physical.loc[seed.physical.osmid.eq("2")]
    assert len(service) == 2
    assert all((service.edge_u.eq("c") | service.edge_v.eq("c")))
    current, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    assert not current.physical.loc[current.physical.osmid.eq("2"), "motor_access_allowed"].any()
    assert not any(d["osmid"] == "2" for _, _, d in current.access.edges(data=True))


@pytest.mark.parametrize("node_only", [False, True])
@pytest.mark.parametrize(
    "purpose_tags",
    [
        {"motor_vehicle": "delivery"},
        {"access": "private", "access:delivery": "yes"},
    ],
)
def test_delivery_purpose_way_or_node_never_enters_public_through_graph(
    tmp_path, node_only, purpose_tags
):
    options = _inputs(
        tmp_path,
        terminal_tags={"highway": "residential", **({} if node_only else purpose_tags)},
        node_tags=purpose_tags if node_only else None,
    )
    build_access_context(**options, output_dir=tmp_path / "seed")
    current, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    roads = current.physical.loc[current.physical.osmid.eq("2")]
    assert roads.motor_access_allowed.all()
    assert roads.access_layer.eq("osm_terminal_only").all()
    assert not any(d["osmid"] == "2" for _, _, d in current.core.edges(data=True))
    assert any(d["osmid"] == "2" for _, _, d in current.access.edges(data=True))
    location = gpd.GeoDataFrame(
        {"latent_service_location_id": ["delivery-site"]}, geometry=[Point(10, 150)], crs=3857
    )
    result = current.anchor(location).iloc[0]
    assert result.local_access_confirmed and result.requires_terminal_paths


def test_unresolved_private_oneway_stays_diagnostic_and_cannot_be_authorized(tmp_path):
    options = _inputs(tmp_path, terminal_tags={"access": "private", "oneway": "reversible"})
    build_access_context(**options, output_dir=tmp_path / "seed")
    current, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    road = current.physical.loc[current.physical.osmid.eq("2")].iloc[0]
    assert road.source_oneway == "reversible"
    assert road.direction_status == "unresolved_oneway_value"
    assert not road.direction_resolved and not road.motor_access_allowed
    assert json.loads(road.directed_edge_refs) == []


def test_both_road_builders_use_the_same_reference_scc_tie_rule(tmp_path):
    options = _inputs(tmp_path, tie=True)
    seed, _ = build_access_context(**options, output_dir=tmp_path / "seed")
    current, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "delivery"
    )
    for context in [seed, current]:
        index = build_directed_component_index(context.core)
        expected = {n for n in context.core if index.node_to_scc[n] == index.reference_scc_id}
        assert context.reference == expected == {"a", "b"}


@pytest.mark.parametrize("changed", ["admin", "service", "masks", "crs", "source", "topology"])
def test_reusing_a_seed_rejects_changed_inputs_before_writing_output(tmp_path, changed):
    options = _inputs(tmp_path)
    seed_dir = tmp_path / "seed"
    build_access_context(**options, output_dir=seed_dir)
    if changed == "crs":
        options["area_crs"] = "EPSG:32611"
    elif changed == "source":
        path = options["source_dir"] / "osm_access_nodes.parquet"
        pd.DataFrame({"changed": [True]}).to_parquet(path)
    elif changed == "topology":
        path = seed_dir / "access_context_manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("topology_contract")
        path.write_text(json.dumps(manifest))
    else:
        key = {
            "admin": "admin_boundary_path",
            "service": "service_boundary_path",
            "masks": "mask_path",
        }[changed]
        path = options[key]
        if changed == "masks":
            gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)], crs=3857).to_parquet(path)
        else:
            gpd.GeoDataFrame(geometry=[box(-50, -50, 50, 50)], crs=3857).to_crs(4326).to_file(
                path, driver="GeoJSON"
            )
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError):
        build_delivery_access_context(**options, seed_dir=seed_dir, output_dir=output)
    assert not output.exists()


def test_seed_builder_requires_current_extraction_node_contract_and_admin_hash(tmp_path):
    options = _inputs(tmp_path)
    metadata = options["source_dir"] / "extraction.json"
    extraction = json.loads(metadata.read_text())
    extraction.pop("access_node_tag_contract")
    metadata.write_text(json.dumps(extraction))
    with pytest.raises(ValueError, match="access-node"):
        build_access_context(**options, output_dir=tmp_path / "stale")
    assert not (tmp_path / "stale").exists()
    extraction["access_node_tag_contract"] = ACCESS_NODE_TAG_CONTRACT
    extraction["admin_boundary_sha256"] = "changed"
    metadata.write_text(json.dumps(extraction))
    with pytest.raises(ValueError, match="administrative boundary"):
        build_access_context(**options, output_dir=tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()
