"""Nationwide NSI source identity for residential ownership and unit QA."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.parent))])


def identity_fields(x):
    src = x.ftprntsrc.fillna("").astype(str).str.strip()
    fid = x.ftprntid.fillna("").astype(str).str.strip()
    county = x.cbfips.fillna("").astype(str).str[:5]
    credible = (
        ~src.str.upper().isin(["", "P"])
        & ~fid.str.lower().isin(["", "0", "none", "null", "nan"])
        & ~fid.str.lower().str.startswith("none_")
    )
    fp = (county + ":" + src + ":" + fid).where(credible, "")
    usa = pd.to_numeric(x.usastrucid, errors="coerce").fillna(0).astype("int64")
    return fp, usa, credible | usa.gt(0)


def source_groups(x, xy, tolerance=1.0):
    """Identity first. Near-coordinate merges cannot chain across row houses.

    Input sorted by fd_id for deterministic tie-breaking. Same-ID stacks and
    exact coincident records merge regardless of source classification.
    Different credible identities are not merged merely for being nearby.
    """
    assert x.fd_id.is_monotonic_increasing and x.fd_id.is_unique
    n = len(x)
    uf = UnionFind(n)
    fp, usa, credible = identity_fields(x)
    for series in (fp, usa.where(usa.gt(0), "")):
        first = {}
        for i, key in enumerate(series):
            if key == "":
                continue
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    first = {}
    for i, key in enumerate(zip(xy[:, 0], xy[:, 1])):
        if key in first:
            uf.union(first[key], i)
        else:
            first[key] = i
    initial = uf.labels()
    if tolerance <= 0:
        return initial
    # Near pairs are examined over original coordinates, preserving all stack
    # extents in the bounding-box cap. No centroid-only distance shortcut.
    pairs = cKDTree(xy).query_pairs(tolerance, output_type="ndarray")
    if not len(pairs):
        return initial
    parent = x.composite_building_id.fillna("").to_numpy()
    edges = []
    for i, j in pairs:
        if parent[i] and parent[i] == parent[j] and initial[i] != initial[j]:
            edges.append((float(np.linalg.norm(xy[i] - xy[j])), int(i), int(j)))
    bounds = {}
    cred = {}
    for i, r in enumerate(initial):
        if r not in bounds:
            bounds[r] = np.r_[xy[i], xy[i]]
            cred[r] = False
        bounds[r][:2] = np.minimum(bounds[r][:2], xy[i])
        bounds[r][2:] = np.maximum(bounds[r][2:], xy[i])
        cred[r] = cred[r] or bool(credible.iloc[i])
    for _, i, j in sorted(edges):
        a, b = uf.find(i), uf.find(j)
        if a == b or (cred[a] and cred[b]):
            continue
        mn = np.minimum(bounds[a][:2], bounds[b][:2])
        mx = np.maximum(bounds[a][2:], bounds[b][2:])
        if np.linalg.norm(mx - mn) > tolerance + 1e-9:
            continue
        bound = np.r_[mn, mx]
        flag = cred[a] or cred[b]
        uf.union(a, b)
        r = uf.find(a)
        bounds[r] = bound
        cred[r] = flag
    return uf.labels()
