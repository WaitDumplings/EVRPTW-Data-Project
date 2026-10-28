"""Bind legal/reference speeds to every shared public and purpose-road edge."""

from __future__ import annotations

import json
import gc
import math
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .context_validation import validate_context_bindings
from .local_road_access import LocalAccessContext
from .road_tags import _as_bool
from .speed import _canonical_values, build_legal_speed_layer
from .util import sha256_file, write_json


POLICY = "common_source_edge_legal_reference_speed_v1"
LAYERS = ["in_public_graph", "in_delivery_terminal_graph", "in_charging_terminal_graph"]


def _identity(u, v, key):
    return str(u), str(v), str(key)


def _speed_attributes(data):
    return {
        "length": float(data["length"]),
        "osmid": _canonical_values(data.get("osmid")),
        "highway": _canonical_values(data.get("highway")),
        "maxspeed": _canonical_values(data.get("maxspeed")),
        "maxspeed:forward": _canonical_values(data.get("maxspeed:forward")),
        "maxspeed:backward": _canonical_values(data.get("maxspeed:backward")),
        "maxspeed:hgv": _canonical_values(data.get("maxspeed:hgv")),
        "maxspeed:hgv:forward": _canonical_values(data.get("maxspeed:hgv:forward")),
        "maxspeed:hgv:backward": _canonical_values(data.get("maxspeed:hgv:backward")),
        "reversed": _as_bool(data.get("reversed", False)),
        "oneway": _as_bool(data.get("oneway", False)),
        "transit_only": _as_bool(data.get("transit_only", False)),
    }


def common_source_edge_union(context):
    """Create a lean speed-input union without opening terminal through-routing.

    This graph is private calculation input only. It must never replace the
    public routing graph; terminal purpose memberships remain explicit columns.
    """
    graph = nx.MultiDiGraph(crs=context.core.graph["crs"])
    records, geometries = {}, {}

    def charging_edges():
        # Speed binding needs native edges, not a second million-edge graph
        # and two routing forests. Keep memory bounded on a 16GB build host.
        import shapely

        selected = context.physical.loc[context.physical.charging_motor_access_allowed]
        for row in selected.to_crs(context.core.graph["crs"]).itertuples():
            for ref in json.loads(row.directed_edge_refs):
                u, v, key = ref["u"], ref["v"], ref["key"]
                if u in context.reference and v in context.reference and context.core.has_edge(u, v, key):
                    continue
                reversed_ = ref["geometry_orientation"] == "reverse_of_physical"
                yield u, v, key, dict(
                    length=ref["length_m"], osmid=str(row.osmid), highway=row.highway,
                    maxspeed=row.maxspeed, oneway=bool(row.oneway), reversed=reversed_,
                    geometry=shapely.reverse(row.geometry) if reversed_ else row.geometry,
                    **{"maxspeed:forward": row.maxspeed_forward,
                       "maxspeed:backward": row.maxspeed_backward},
                )

    for layer, edges in zip(LAYERS, [context.core.edges(keys=True, data=True), context.access.edges(keys=True, data=True), charging_edges()]):
        for u, v, key, data in edges:
            identity = _identity(u, v, key)
            edge_id = ":".join(identity)
            attributes = _speed_attributes(data)
            if not math.isfinite(attributes["length"]) or attributes["length"] <= 0:
                raise ValueError("Common road edge has invalid source length: " + edge_id)
            if identity in records:
                previous = graph[identity[0]][identity[1]][identity[2]]
                for name, value in attributes.items():
                    if name == "transit_only":
                        continue  # A routing-envelope label, not source speed evidence.
                    same = (
                        math.isclose(previous[name], value, abs_tol=1e-6, rel_tol=1e-9)
                        if name == "length"
                        else previous[name] == value
                    )
                    if not same:
                        raise ValueError(f"Conflicting shared source edge {edge_id}: {name}")
                if not geometries[identity].equals_exact(data["geometry"], 1e-8):
                    raise ValueError("Conflicting shared source edge geometry: " + edge_id)
            else:
                for node in identity[:2]:
                    if node not in graph:
                        graph.add_node(node)
                graph.add_edge(*identity[:2], key=identity[2], **attributes)
                geometries[identity] = data["geometry"]
                records[identity] = {
                    "edge_u": identity[0],
                    "edge_v": identity[1],
                    "edge_key": identity[2],
                    "edge_id": edge_id,
                    "source_length_m": attributes["length"],
                    "source_osmid": attributes["osmid"],
                    **{name: False for name in LAYERS},
                }
            records[identity][layer] = True
    inventory = pd.DataFrame(records.values()).sort_values("edge_id").reset_index(drop=True)
    if inventory.empty or inventory.edge_id.duplicated().any():
        raise ValueError("Common speed edge identity is empty or ambiguous")
    return graph, inventory


def validate_common_speed_table(frame, inventory):
    """Independently require exact identity/length coverage and finite costs."""
    required = {
        "edge_id",
        "edge_u",
        "edge_v",
        "edge_key",
        "length_m",
        "legal_speed_kph",
        "legal_travel_time_s",
        "reference_speed_weekday_kph",
        "reference_speed_weekend_kph",
        "reference_travel_time_weekday_s",
        "reference_travel_time_weekend_s",
        *LAYERS,
    }
    if not required <= set(frame):
        raise ValueError("Common speed table lacks required fields")
    if frame.edge_id.duplicated().any() or frame.duplicated(["edge_u", "edge_v", "edge_key"]).any():
        raise ValueError("Common speed table contains duplicate edge identities")
    if set(frame.edge_id) != set(inventory.edge_id):
        raise ValueError("Common speed table has missing or foreign graph edges")
    expected = inventory.set_index("edge_id").loc[frame.edge_id]
    for name in ["edge_u", "edge_v", "edge_key"]:
        if not np.array_equal(frame[name].astype(str), expected[name].astype(str)):
            raise ValueError("Common speed table native edge identity mismatch")
    if not np.allclose(frame.length_m, expected.source_length_m, rtol=1e-9, atol=1e-6):
        raise ValueError("Common speed table source edge length mismatch")
    for name in LAYERS:
        if not np.array_equal(frame[name].astype(bool), expected[name].astype(bool)):
            raise ValueError("Common speed table graph layer membership mismatch")
    numeric = frame[list(required - {"edge_id", "edge_u", "edge_v", "edge_key", *LAYERS})].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Common speed table contains invalid speeds or costs")
    if not np.allclose(frame.legal_travel_time_s, 3.6 * frame.length_m / frame.legal_speed_kph):
        raise ValueError("Common legal travel time formula mismatch")
    for day in ["weekday", "weekend"]:
        speed, cost = frame[f"reference_speed_{day}_kph"], frame[f"reference_travel_time_{day}_s"]
        if (speed > frame.legal_speed_kph + 1e-9).any():
            raise ValueError("Common reference speed exceeds legal speed")
        if not np.allclose(cost, 3.6 * frame.length_m / speed):
            raise ValueError("Common reference travel time formula mismatch")
    return {"edge_count": len(frame), **{name: int(frame[name].sum()) for name in LAYERS}}


def build_common_speed_component(
    *,
    city_slug,
    context_dir,
    output_dir,
    state_fips=None,
    hpms_path=None,
    boundary_path=None,
):
    """Build source-edge speeds; connector operating speed remains a scenario input.

    hpms_path, when provided, is frozen normalized HPMS-to-native-edge evidence,
    not a rolling/raw geospatial download. Unknown/duplicate IDs fail in the
    existing speed kernel. No evidence is fabricated when it is omitted.
    """
    context_dir, output_dir = Path(context_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh common speed output directory")
    context_path = context_dir / "access_context_manifest.json"
    manifest = json.loads(context_path.read_text())
    validate_context_bindings(manifest, city_slug=city_slug, boundary_path=boundary_path)
    if manifest.get("source", {}).get("pbf_replication_timestamp_utc") != "2026-10-25T21:21:21Z":
        raise ValueError("Common speeds require the accepted road snapshot")
    context = LocalAccessContext.load(context_dir)
    graph, inventory = common_source_edge_union(context)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="common-speed-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "component"
        staged.mkdir()
        union_path = staged / "private_source_edge_union.graphml"
        ox.save_graphml(graph, union_path)
        del graph, context
        gc.collect()
        kernel = build_legal_speed_layer(
            city_slug=city_slug,
            graph_path=union_path,
            output_dir=staged / "kernel",
            hpms_edge_evidence_path=Path(hpms_path) if hpms_path is not None else None,
            state_fips=state_fips,
        )
        frame = pd.read_parquet(staged / "kernel" / "directed_legal_speeds.parquet")
        extra = inventory.drop(columns=["edge_u", "edge_v", "edge_key"])
        frame = frame.merge(extra, on="edge_id", how="left", validate="one_to_one")
        validation = validate_common_speed_table(frame, inventory)
        frame.to_parquet(staged / "directed_speeds.parquet", index=False)
        shutil.copyfile(
            staged / "kernel" / "moves_speed_profile.json", staged / "moves_speed_profile.json"
        )
        kernel["graph"]["path"] = str((output_dir / union_path.name).resolve())
        kernel["outputs"] = {
            "directed_legal_speeds": "directed_speeds.parquet",
            "moves_speed_profile": "moves_speed_profile.json",
        }
        kernel["output_sha256"] = {
            name: sha256_file(staged / path) for name, path in kernel["outputs"].items()
        }
        write_json(staged / "legal_reference_kernel_manifest.json", kernel)
        shutil.rmtree(staged / "kernel")
        dependencies = [
            "common_speed_binding.py",
            "speed.py",
            "moves_speed.py",
            "road_tags.py",
            "charging_road_access.py",
            "local_road_access.py",
            "context_validation.py",
        ]
        result = {
            "schema": "evrptw_common_speed_component_v1",
            "policy": POLICY,
            "city_slug": city_slug,
            "generated_utc": datetime.now(UTC).isoformat(),
            "inputs": {
                "road_context": {
                    "path": str(context_path.resolve()),
                    "sha256": sha256_file(context_path),
                },
                "road_graphs": {
                    name: manifest["outputs"][name]
                    for name in [
                        "graph_operational.graphml",
                        "graph_terminal_access.graphml",
                        "physical_roads.parquet",
                    ]
                },
                "hpms": {
                    "path": str(Path(hpms_path).resolve()),
                    "sha256": sha256_file(Path(hpms_path)),
                }
                if hpms_path is not None
                else None,
            },
            "road_source": manifest["source"],
            "generation_dependencies": {
                name: sha256_file(Path(__file__).with_name(name)) for name in dependencies
            },
            "validation": validation,
            "hpms_evidence_status": "provided_frozen_normalized_edge_evidence"
            if hpms_path is not None
            else "not_provided_zero_HPMS_evidence",
            "observed_hpms_speed_limit_edge_count": kernel["observed_hpms_speed_limit_edge_count"],
            "observed_osm_maxspeed_edge_count": kernel["observed_osm_maxspeed_edge_count"],
            "imputed_legal_speed_edge_count": int(frame.legal_speed_imputed.sum()),
            "reference_speed_contract": kernel["reference_speed_contract"],
            "connector_time_contract": {
                "policy": "retain_geometric_length_scenario_speed_required_v1",
                "stored_speed_kph": None,
                "stored_travel_time_s": None,
                "length_coefficient": "travel_time_s = connector_length_m * 3.6 / scenario_speed_kph",
                "scope": "connector speed is an explicit Stage2 operating assumption; no test-only 5kmh/10kmh production default",
            },
            "outputs": {
                name: sha256_file(staged / name)
                for name in [
                    "directed_speeds.parquet",
                    "moves_speed_profile.json",
                    "legal_reference_kernel_manifest.json",
                ]
            },
            "private_outputs": {union_path.name: sha256_file(union_path)},
            "union_graph_usage": "private speed calculation input only; never use as public through-routing graph",
            "whole_cle_frozen": False,
            "instance_activation_included": False,
        }
        write_json(staged / "common_speed_manifest.json", result)
        staged.replace(output_dir)
    return result
