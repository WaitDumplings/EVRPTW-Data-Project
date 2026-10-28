"""Restore a versioned source archive without refreshing or overwriting inputs."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from pathlib import Path

from .util import sha256_file, write_json

SCHEMA = "evrptw_frozen_source_archive_v1"


def archive_path(root, value):
    """Archive paths must stay inside their caller-selected root."""
    root = Path(root).resolve()
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Archive entries require relative paths without parent traversal")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Archive entry resolves outside its selected root")
    return path


def validate_archive_inventory(inventory):
    if inventory.get("schema") != SCHEMA or not inventory.get("files"):
        raise ValueError("A nonempty frozen source archive inventory is required")
    targets = set()
    for key, record in inventory["files"].items():
        if not key or len(str(record.get("sha256", ""))) != 64:
            raise ValueError("Every frozen source entry requires an identity and SHA-256")
        for field in ("archive_path", "path"):
            value = Path(record[field])
            if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
                raise ValueError("Frozen archive paths must be nonempty relative paths")
        if record["path"] in targets:
            raise ValueError("Two frozen source identities share an output path")
        targets.add(record["path"])


def hydrate_frozen_inputs(*, inventory_path, archive_root, output_root):
    """Hash-check and stage exact sources; existing differing bytes always fail.

    A same-filesystem hard link avoids duplicating large archives. Every reuse
    checks the frozen hash; source consumers must remain read-only. Cross-device
    archives use a byte copy. No permission bits on the original are changed.
    """
    inventory_path = Path(inventory_path).resolve()
    inventory = json.loads(inventory_path.read_text())
    validate_archive_inventory(inventory)
    archive_root, output_root = Path(archive_root).resolve(), Path(output_root).resolve()
    records = []
    # Preflight every file before creating any output. Missing snapshots never
    # trigger a network request or a replacement of the registry checksum.
    sources = {}
    for key, item in inventory["files"].items():
        source = archive_path(archive_root, item["archive_path"])
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise ValueError("Missing or changed frozen archive input: " + key)
        target = archive_path(output_root, item["path"])
        if target.exists() and (not target.is_file() or sha256_file(target) != item["sha256"]):
            raise FileExistsError("Refusing to replace a changed staged input: " + key)
        sources[key] = (source, target)
    for key, item in inventory["files"].items():
        source, target = sources[key]
        if target.exists():
            mode = "hash_verified_reuse"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".frozen-input-", dir=target.parent) as temp:
                staged = Path(temp) / "source"
                try:
                    os.link(source, staged)
                    mode = "hardlink"
                except OSError as error:
                    if error.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
                        raise
                    shutil.copyfile(source, staged)
                    mode = "byte_copy"
                if sha256_file(staged) != item["sha256"]:
                    raise ValueError("Frozen archive changed during staging: " + key)
                # link is a no-clobber atomic publication of the prepared bytes.
                os.link(staged, target)
        records.append(dict(key=key, path=item["path"], sha256=item["sha256"], mode=mode))
    result = dict(
        schema="evrptw_source_hydration_v1",
        inventory_sha256=sha256_file(inventory_path),
        file_count=len(records),
        sources=records,
        source_data_refreshed=False,
        rule="exact frozen bytes; sources and staged inputs are read-only consumers",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "hydration.json", result)
    return result
