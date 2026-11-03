"""Stable semantic random streams and portable artifact helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def identity(*parts):
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def rng(*parts):
    return np.random.default_rng(int(identity(*parts), 16))


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")


def output_hashes(path, *, exclude=()):
    root = Path(path)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and str(p.relative_to(root)) not in exclude}
