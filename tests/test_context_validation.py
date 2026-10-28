import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evrptw_cle.cli import depots
from evrptw_cle import context_validation
from evrptw_cle.context_validation import validate_context_bindings
from evrptw_cle.residential_sites import build_registered_site_layer
from evrptw_cle.util import sha256_file


@pytest.fixture
def bound_inputs(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    boundary = tmp_path / "land.geojson"
    boundary.write_text("frozen-land-boundary")
    masks = tmp_path / "masks.parquet"
    masks.write_text("frozen-exclusion-masks")
    manifest = dict(
        policy="delivery_terminal_access_v2",
        topology_contract="source_direction_and_access_node_segmentation_v2",
        city_slug="example",
        service_boundary_sha256=sha256_file(boundary),
        mask_sha256=sha256_file(masks),
        area_crs="EPSG:32611",
        generation_dependencies={
            name: sha256_file(Path(context_validation.__file__).with_name(name))
            for name in [
                "local_road_access.py",
                "terminal_access.py",
                "road_tags.py",
                "delivery_access.py",
                "public_locations.py",
                "frontage.py",
            ]
        },
    )
    (context / "access_context_manifest.json").write_text(json.dumps(manifest))
    return context, boundary, masks, manifest


def test_bindings_accept_equivalent_metric_crs(bound_inputs):
    _, boundary, masks, manifest = bound_inputs
    validate_context_bindings(
        manifest,
        city_slug="example",
        boundary_path=boundary,
        mask_path=masks,
        area_crs=32611,
    )


def test_old_context_requires_topology_rebuild(bound_inputs):
    _, _, _, manifest = bound_inputs
    manifest.pop("topology_contract")
    with pytest.raises(ValueError, match="rebuild"):
        validate_context_bindings(manifest)


def test_changed_generation_code_requires_rebuild(bound_inputs):
    _, _, _, manifest = bound_inputs
    manifest["generation_dependencies"]["terminal_access.py"] = "stale_code_hash"
    with pytest.raises(ValueError, match="generation code changed"):
        validate_context_bindings(manifest)


def test_omitted_semantic_dependency_requires_rebuild(bound_inputs):
    _, _, _, manifest = bound_inputs
    manifest["generation_dependencies"].pop("terminal_access.py")
    with pytest.raises(ValueError, match="omits required"):
        validate_context_bindings(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("city_slug", "other", "city differs"),
        ("service_boundary_sha256", "other", "service boundary"),
        ("mask_sha256", "other", "exclusion masks"),
        ("area_crs", "EPSG:32612", "metric CRS"),
    ],
)
def test_depot_cli_rejects_mixed_context_before_source_reads(bound_inputs, field, value, message):
    context, boundary, masks, manifest = bound_inputs
    manifest[field] = value
    (context / "access_context_manifest.json").write_text(json.dumps(manifest))
    # Deliberately omit source_manifest: binding must fail before source parsing
    # or a geospatial build can begin.
    args = SimpleNamespace(
        context=context, city="example", boundary=boundary, masks=masks, area_crs=32611
    )
    with pytest.raises(ValueError, match=message):
        depots(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("city_slug", "other", "city differs"),
        ("service_boundary_sha256", "other", "service boundary"),
        ("mask_sha256", "other", "exclusion masks"),
        ("area_crs", "EPSG:32612", "metric CRS"),
    ],
)
def test_customer_builder_rejects_individually_valid_but_unbound_sources(
    tmp_path, bound_inputs, field, value, message
):
    context, boundary, masks, manifest = bound_inputs
    manifest[field] = value
    (context / "access_context_manifest.json").write_text(json.dumps(manifest))
    configs = tmp_path / "configs"
    configs.mkdir()
    registry = configs / "sources.json"
    registry.write_text(
        json.dumps(
            dict(
                policy="identified_residential_site_v3",
                cities=dict(
                    example=dict(
                        area_crs="EPSG:32611",
                        boundary_sha256=sha256_file(boundary),
                        sources=dict(public_masks=dict(path=masks.name, sha256=sha256_file(masks))),
                    )
                ),
            )
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        build_registered_site_layer(
            registry_path=registry,
            city_slug="example",
            boundary_path=boundary,
            graph_path=context / "graph_operational.graphml",
            output_dir=output,
        )
    assert not output.exists()
