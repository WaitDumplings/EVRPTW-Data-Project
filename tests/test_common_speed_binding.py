import json

import geopandas as gpd
import pandas as pd
import pytest

from evrptw_cle.common_speed_binding import (
    build_common_speed_component,
    common_source_edge_union,
    validate_common_speed_table,
)
from evrptw_cle.delivery_access import build_delivery_access_context
from evrptw_cle.local_road_access import build_access_context
from evrptw_cle.util import sha256_file
from test_road_direction_barrier import _inputs


def _speed_context(tmp_path, terminal_tags=None):
    options = _inputs(tmp_path, terminal_tags=terminal_tags)
    path = options["source_dir"] / "osm_ways.parquet"
    ways = gpd.read_parquet(path)
    tags = json.loads(ways.loc[0, "tags_json"])
    tags.update({"maxspeed": "25 mph", "maxspeed:forward": "30 mph", "maxspeed:backward": "20 mph"})
    ways.loc[0, "tags_json"] = json.dumps(tags)
    ways.to_parquet(path, index=False)
    extraction = options["source_dir"] / "extraction.json"
    metadata = json.loads(extraction.read_text())
    metadata["outputs"][path.name] = sha256_file(path)
    extraction.write_text(json.dumps(metadata))
    build_access_context(**options, output_dir=tmp_path / "seed")
    context, _ = build_delivery_access_context(
        **options, seed_dir=tmp_path / "seed", output_dir=tmp_path / "context"
    )
    return context, options


def test_common_speeds_keep_native_directions_and_impute_terminal_without_defaults(tmp_path):
    context, options = _speed_context(tmp_path)
    source_hashes = {
        name: sha256_file(tmp_path / "context" / name)
        for name in [
            "graph_operational.graphml",
            "graph_terminal_access.graphml",
            "physical_roads.parquet",
        ]
    }
    output = tmp_path / "speeds"
    report = build_common_speed_component(
        city_slug="synthetic",
        context_dir=tmp_path / "context",
        output_dir=output,
        state_fips="36",
        boundary_path=options["service_boundary_path"],
    )
    table = pd.read_parquet(output / "directed_speeds.parquet")
    assert table.loc[table.edge_u.eq("a") & table.edge_v.eq("b"), "legal_speed_kph"].iloc[
        0
    ] == pytest.approx(48.28032)
    assert table.loc[table.edge_u.eq("b") & table.edge_v.eq("a"), "legal_speed_kph"].iloc[
        0
    ] == pytest.approx(32.18688)
    terminal = table.loc[table.in_delivery_terminal_graph & ~table.in_public_graph]
    assert len(terminal) and terminal.legal_speed_imputed.all()
    assert terminal.legal_speed_kph.gt(10).all()
    assert terminal.reference_speed_weekday_kph.gt(0).all()
    assert report["observed_hpms_speed_limit_edge_count"] == 0
    assert report["hpms_evidence_status"] == "not_provided_zero_HPMS_evidence"
    assert report["connector_time_contract"]["stored_speed_kph"] is None
    assert not report["instance_activation_included"]
    _, inventory = common_source_edge_union(context)
    assert validate_common_speed_table(table, inventory)["edge_count"] == len(table)
    for group in ["outputs", "private_outputs"]:
        for name, digest in report[group].items():
            assert sha256_file(output / name) == digest
    assert source_hashes == {
        name: sha256_file(tmp_path / "context" / name) for name in source_hashes
    }


def test_charging_only_terminal_edges_have_speed_coverage(tmp_path):
    _, _ = _speed_context(tmp_path, terminal_tags={"access:delivery": "no", "maxspeed": "15 mph"})
    report = build_common_speed_component(
        city_slug="synthetic",
        context_dir=tmp_path / "context",
        output_dir=tmp_path / "speeds",
        state_fips="36",
    )
    table = pd.read_parquet(tmp_path / "speeds" / "directed_speeds.parquet")
    charging_only = table.loc[table.in_charging_terminal_graph & ~table.in_delivery_terminal_graph]
    assert len(charging_only) == 2
    assert charging_only.legal_speed_source.eq("osm_maxspeed").all()
    assert charging_only.legal_speed_kph.eq(24.14016).all()
    assert report["validation"]["in_charging_terminal_graph"] >= 2


def test_conflicting_duplicate_source_ref_is_rejected(tmp_path):
    context, _ = _speed_context(tmp_path)
    u, v, key, data = next(iter(context.core.edges(keys=True, data=True)))
    context.access.add_edge(u, v, key=key, **{**data, "length": data["length"] + 1})
    with pytest.raises(ValueError, match="Conflicting shared source edge"):
        common_source_edge_union(context)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("missing", "missing or foreign"),
        ("duplicate", "duplicate"),
        ("length", "source edge length"),
        ("time", "formula mismatch"),
    ],
)
def test_common_speed_verifier_rejects_missing_duplicate_or_conflicting_costs(
    tmp_path, damage, message
):
    context, _ = _speed_context(tmp_path)
    build_common_speed_component(
        city_slug="synthetic",
        context_dir=tmp_path / "context",
        output_dir=tmp_path / "speeds",
        state_fips="36",
    )
    table = pd.read_parquet(tmp_path / "speeds" / "directed_speeds.parquet")
    _, inventory = common_source_edge_union(context)
    if damage == "missing":
        table = table.iloc[1:]
    elif damage == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif damage == "length":
        table.loc[0, "length_m"] += 1
    else:
        table.loc[0, "reference_travel_time_weekday_s"] += 1
    with pytest.raises(ValueError, match=message):
        validate_common_speed_table(table, inventory)


def test_changed_saved_graph_cannot_be_used_for_speed_binding(tmp_path):
    _speed_context(tmp_path)
    graph = tmp_path / "context" / "graph_terminal_access.graphml"
    graph.write_text(graph.read_text() + "\n")
    with pytest.raises(ValueError, match="Changed access context"):
        build_common_speed_component(
            city_slug="synthetic",
            context_dir=tmp_path / "context",
            output_dir=tmp_path / "speeds",
            state_fips="36",
        )
