#!/usr/bin/env python3
"""Record exact Stage 1 server inputs without copying or downloading data.

The public manifest contains portable target paths, sizes and hashes.  A
separate private map records the local source paths needed for data transfer.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path

from evrptw_cle.building_units import DEFAULTS
from evrptw_cle.stage1 import load_project
from evrptw_cle.util import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = "inputs/stage1_20261102"
DEFAULT_FREEZE_ID = "stage1-20261102-r1"


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _relative_from_config(config_path, repo_relative):
    return Path(os.path.relpath(ROOT / repo_relative, Path(config_path).resolve().parent)).as_posix()


def prepare_server_inputs(*, project_path, output, local_map, server_config,
                          data_root=DEFAULT_DATA_ROOT, freeze_id=DEFAULT_FREEZE_ID):
    """Write the public handoff files and private local transfer mapping.

    ``files[*].target_path`` is relative to ``data_root``; ``data_root`` is
    relative to the extracted code repository root. All source bytes are read
    only. The server project and manifest omit local source paths and AI keys.
    """
    project_path = Path(project_path).resolve()
    project = load_project(project_path)
    inventory_path = Path(project["components"]["inventory"]).resolve()
    inventory = json.loads(inventory_path.read_text())
    source_root = Path(project["components"]["source_root"])
    roads_root = Path(project["components"]["road_sources"])
    data_root = Path(data_root)
    if data_root.is_absolute() or ".." in data_root.parts:
        raise ValueError("data_root must be relative to the repository root")
    server_config = Path(server_config).resolve()
    inventory_repo_path = inventory_path.relative_to(ROOT).as_posix()
    entries, mappings, roads = {}, {}, {}

    def add(key, source, target, category, city=None, expected=None):
        source, target = Path(source).resolve(), Path(target).as_posix()
        if target in entries:
            entries[target]["roles"].append(key)
            return
        digest = sha256_file(source)
        if expected is not None and digest != expected:
            raise ValueError("Frozen input differs from its recorded source hash: " + key)
        entry = {"target_path": target, "sha256": digest, "bytes": source.stat().st_size,
                 "category": category, "roles": [key]}
        if city is not None:
            entry["city"] = city
        entries[target] = entry
        mappings[target] = {"source_path": str(source), **entry,
                            "repo_target_path": (data_root / target).as_posix()}

    def archive_input(key, category, city=None):
        item = inventory["files"][key]
        target = Path("archive") / item["path"]
        add(key, source_root / item["path"], target, category, city, item["sha256"])
        return (data_root / target).as_posix()

    server_cities = {}
    for city, sources in project["cities"].items():
        info = inventory["cities"][city]
        server_sources = {}
        for name, filename in [("microsoft", "microsoft_footprints.parquet"),
                               ("osm", "osm_buildings.parquet"), ("nsi", "nsi_native.parquet")]:
            target = Path("customer") / city / filename
            expected = None
            inventory_name = {"microsoft": "microsoft_footprints", "osm": "osm_buildings"}.get(name)
            if inventory_name and inventory_name in info.get("sources", {}):
                item = inventory["files"][info["sources"][inventory_name]]
                if (source_root / item["path"]).resolve() == Path(sources[name]).resolve():
                    expected = item["sha256"]
            add(city + "." + name, sources[name], target, "customer", city, expected)
            server_sources[name] = _relative_from_config(server_config, data_root / target)
        server_sources["area_crs"] = sources["area_crs"]
        for name in ["service_boundary", "admin_boundary"]:
            key = info["boundaries"][name]
            server_sources[name] = _relative_from_config(server_config, archive_input(key, "boundary", city))
        for name in ["raw", "extraction"]:
            archive_input(info["depot"][name], "depot", city)
        extraction_path = roads_root / city / "extraction.json"
        extraction = json.loads(extraction_path.read_text())
        add(city + ".road_extraction", extraction_path,
            Path("roads") / city / "extraction.json", "roads", city)
        for filename, expected in extraction["outputs"].items():
            add(city + ".roads." + filename, roads_root / city / filename,
                Path("roads") / city / filename, "roads", city, expected)
        roads[city] = {key: extraction[key] for key in
            ["pbf_sha256", "pbf_replication_timestamp_utc", "access_node_tag_contract"]}
        server_cities[city] = server_sources
    for key in inventory["charging"].values():
        archive_input(key, "charging")

    server_project = {
        "schema": project["schema"],
        "output": _relative_from_config(server_config, "outputs/" + freeze_id),
        "axis_r": project.get("axis_r", 0.2853432473555575),
        "cities": server_cities,
        "components": {
            "inventory": _relative_from_config(server_config, inventory_repo_path),
            "source_root": _relative_from_config(server_config, data_root / "archive"),
            "road_sources": _relative_from_config(server_config, data_root / "roads"),
        },
        "construction": {"roads": {"selection_buffer_km": 20,
                                      "allow_private_through": True, "facility_masks": False}},
        "customer": {"active_profile": "unrestricted", "include_model_imputed": True},
        "units": dict(DEFAULTS),
        "source_snapshots": dict(project.get("source_snapshots", {})),
    }
    _json(server_config, server_project)
    files = sorted(entries.values(), key=lambda row: row["target_path"])
    manifest = {
        "schema": "evrptw_stage1_server_input_manifest_v1",
        "freeze_id": freeze_id,
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_root": data_root.as_posix(),
        "target_path_base": "data_root",
        "server_config": server_config.relative_to(ROOT).as_posix(),
        "server_config_sha256": sha256_file(server_config),
        "component_inventory": {"repo_path": inventory_repo_path,
            "sha256": sha256_file(inventory_path), "bytes": inventory_path.stat().st_size,
            "transfer_group": "code_bundle"},
        "source_project_sha256": sha256_file(project_path),
        "source_snapshots": server_project["source_snapshots"],
        "road_snapshots": roads,
        "files": files,
        "summary": {"cities": len(server_cities), "files": len(files),
                    "bytes": sum(row["bytes"] for row in files)},
        "source_bytes_modified": False,
        "source_files_copied": False,
        "network_requests": 0,
    }
    _json(output, manifest)
    _json(local_map, {
        "schema": "evrptw_private_stage1_transfer_map_v1", "not_for_publication": True,
        "freeze_id": freeze_id, "data_root": data_root.as_posix(),
        "public_manifest_sha256": sha256_file(output),
        "files": [mappings[row["target_path"]] for row in files],
    })
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path, help="Existing local Stage1 project JSON")
    parser.add_argument("--output", type=Path, default=Path("freeze/20261102-r1/input_manifest.json"))
    parser.add_argument("--local-map", type=Path, default=Path("private/stage1-server-input-map.json"))
    parser.add_argument("--server-config", type=Path, default=Path("freeze/20261102-r1/stage1.server.json"))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--freeze-id", default=DEFAULT_FREEZE_ID)
    args = parser.parse_args(argv)
    manifest = prepare_server_inputs(project_path=args.project, output=args.output,
        local_map=args.local_map, server_config=args.server_config,
        data_root=args.data_root, freeze_id=args.freeze_id)
    print(json.dumps({"freeze_id": manifest["freeze_id"], "data_root": manifest["data_root"],
                      **manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
