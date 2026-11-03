"""Deterministic spatial and source-group holdouts for original NSI references.

The split is assigned to connected components of spatial blocks joined by a
shared source-footprint key.  Thus a block, entity, or repeated source group
cannot contribute simultaneously to model fitting and evaluation.  Percentages
describe component assignment probabilities, not forced per-building quotas.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def make_reference_splits(
    frame: pd.DataFrame,
    city: str,
    *,
    seed: int = 20261102,
    block_m: float = 2000.0,
    train_pct: float = 80.0,
    cal_pct: float = 10.0,
    test_pct: float = 10.0,
) -> pd.Series:
    """Return ``train/calibration/test`` indexed exactly like ``frame``.

    Required fields are local-metric ``x`` and ``y``.  ``source_group`` should be
    the genuine upstream footprint key (or a unique fd-based fallback), as kept
    in the prepared reference table.  Missing/empty keys are not pooled into a
    synthetic shared source group.  Duplicate entity IDs, if present, also join
    their blocks; the intended reference table remains one row per building.

    No units, purpose labels, inferred heights, or prediction errors enter the
    split.  Sorting inputs differently does not change any building's split.
    """
    portions = np.asarray([train_pct, cal_pct, test_pct], dtype=float)
    if block_m <= 0 or np.any(portions < 0) or not np.isclose(portions.sum(), 100):
        raise ValueError("block_m must be positive and split percentages must sum to 100")
    if frame.empty:
        return pd.Series(index=frame.index, dtype="object", name="reference_split")
    xy = frame[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("Reference split requires finite local-metric coordinates")
    grid = np.floor(xy / block_m).astype(np.int64)
    labels = np.asarray([f"{x}:{y}" for x, y in grid], dtype=object)
    # Sorted labels make the minimum-root representative stable under row order.
    block_labels, block_codes = np.unique(labels, return_inverse=True)
    parent = np.arange(len(block_labels), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def join(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for column in ("source_group", "entity_id"):
        if column not in frame:
            continue
        keys = frame[column].fillna("").astype(str).reset_index(drop=True)
        repeated = keys.ne("") & keys.duplicated(keep=False)
        if not repeated.any():
            continue
        links = pd.DataFrame({"key": keys.loc[repeated], "block": block_codes[repeated.to_numpy()]})
        links = links.drop_duplicates(["key", "block"])
        for _, rows in links.groupby("key", sort=False):
            blocks = rows.block.to_numpy(dtype=np.int64)
            for other in blocks[1:]:
                join(int(blocks[0]), int(other))

    roots = np.asarray([find(i) for i in range(len(parent))], dtype=np.int64)
    root_draws = {}
    for root in np.unique(roots):
        value = f"{seed}:{city}:{block_labels[root]}".encode("utf-8")
        digest = hashlib.blake2b(value, digest_size=8).digest()
        root_draws[int(root)] = (int.from_bytes(digest, "little") % 1_000_000) / 10_000.0
    draws = np.asarray([root_draws[int(root)] for root in roots])[block_codes]
    values = np.where(draws < train_pct, "train",
                      np.where(draws < train_pct + cal_pct, "calibration", "test"))
    result = pd.Series(values, index=frame.index, name="reference_split")
    result.attrs.update({"city": str(city), "seed": int(seed), "block_m": float(block_m),
                         "spatial_blocks": len(block_labels),
                         "connected_components": len(root_draws),
                         "train_pct": float(train_pct), "calibration_pct": float(cal_pct),
                         "test_pct": float(test_pct)})
    return result
