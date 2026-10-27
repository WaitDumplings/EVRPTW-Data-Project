"""NSI occupancy labels independent of aggregated residential-unit estimates.

This module defines the reclassification audit policy. A unanimous source label
is modeled evidence, not verified architecture or a confidence probability.
The existing production customer builders are not switched by importing it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

CLASSIFICATION_SCHEMA = "nsi_occtype_type_separate_units_v1"
TYPE_ORDER = ("house", "manufactured_home", "small_apt", "medium_apt", "large_apt")
AMBIGUOUS_TYPE = "residential_type_ambiguous"
UNKNOWN_TYPE = "residential_type_unknown"


def nsi_occtype_type(code: str) -> str | None:
    code = str(code).strip().upper()
    if code.startswith("RES1"):
        return "house"
    if code == "RES2":
        return "manufactured_home"
    return {
        "RES3A": "small_apt",
        "RES3B": "small_apt",
        "RES3C": "medium_apt",
        "RES3D": "medium_apt",
        "RES3E": "large_apt",
        "RES3F": "large_apt",
    }.get(code)


@dataclass(frozen=True)
class ResidentialTypeEvidence:
    service_location_type: str
    type_evidence: str
    type_label_resolved: bool
    type_conflict: bool
    coarse_residential_type: str
    coarse_type_conflict: bool
    nsi_type_candidates: tuple[str, ...]


@lru_cache(maxsize=4096)
def _classify(codes: tuple[str, ...]) -> ResidentialTypeEvidence:
    labels = {nsi_occtype_type(code) for code in codes}
    incomplete = not codes or None in labels
    known = labels - {None}
    candidates = tuple(t for t in TYPE_ORDER if t in known)
    families = {
        "single_family"
        if t == "house"
        else "manufactured_home"
        if t == "manufactured_home"
        else "multi_family"
        for t in known
    }
    if incomplete:
        label, evidence = UNKNOWN_TYPE, "nsi_occtype_missing_or_unrecognized"
    elif len(known) == 1:
        label, evidence = candidates[0], "nsi_occtype_consensus"
    else:
        label, evidence = AMBIGUOUS_TYPE, "nsi_occtype_conflict"
    coarse = (
        "residential_family_unknown"
        if incomplete
        else next(iter(families))
        if len(families) == 1
        else "residential_family_ambiguous"
    )
    return ResidentialTypeEvidence(
        label,
        evidence,
        not incomplete and len(known) == 1,
        len(known) > 1,
        coarse,
        len(families) > 1,
        candidates,
    )


def classify_nsi_occtypes(raw: str | Sequence[str] | None) -> ResidentialTypeEvidence:
    """Classify all residential codes at one candidate without majority voting.

    raw may be the existing JSON-list field or an explicit sequence. Invalid
    values remain unknown; a missing label never silently becomes a house.
    Multiple RES1 records remain single-family evidence, with aggregation
    complexity recorded separately by the caller.
    """
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                values = json.loads(stripped)
            except json.JSONDecodeError:
                values = ["UNRECOGNIZED"]
        else:
            values = [stripped] if stripped else []
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        values = [] if raw is None else ["UNRECOGNIZED"]
    codes = tuple(
        sorted({str(v).strip().upper() if v is not None else "UNRECOGNIZED" for v in values})
    )
    return _classify(codes)
