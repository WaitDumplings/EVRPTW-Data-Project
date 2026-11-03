"""Reconcile blind Claude/GPT assessments without treating agreement as truth.

Provider calls, frozen input identity and raw request receipts are handled by the
runner. This module preserves each judgment and returns one domain assessment.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from urllib.parse import urlparse

TASKS = frozenset({"residential_use", "unit_review", "unit_impute"})
USES = frozenset({"residential", "nonresidential", "mixed", "unknown"})
SCOPES = frozenset({"building", "complex", "unresolved"})


def _count(value):
    if value is None or isinstance(value, (bool, list, tuple, dict)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number >= 1 and number.is_integer() else None


def _valid_output(task, output):
    if not isinstance(output, Mapping) or not output.get("entity_id"):
        return False
    if not isinstance(output.get("evidence", []), list):
        return False
    if task == "residential_use":
        return output.get("ai_inferred_use") in USES
    if output.get("units_scope") not in SCOPES:
        return False
    if task == "unit_review":
        value = output.get("recommended_units")
        return (
            isinstance(output.get("review_required"), bool)
            and isinstance(output.get("issues", []), list)
            and (value is None or _count(value) is not None)
        )
    value = output.get("units")
    if value is not None and _count(value) is None:
        return False
    if output.get("units_method") not in {"web_extracted", "model_imputed", "unresolved"}:
        return False
    low, high = output.get("units_low"), output.get("units_high")
    if low is None and high is None:
        return True
    if _count(low) is None or _count(high) is None or _count(value) is None:
        return False
    return float(low) <= float(value) <= float(high)


def _evidence_union(outputs):
    result, seen = [], set()
    for output in outputs:
        for item in output.get("evidence", []):
            if not isinstance(item, Mapping):
                continue
            key = json.dumps(dict(item), sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                result.append(deepcopy(dict(item)))
    return result


def _has_building_web_source(evidence):
    for item in evidence:
        url = urlparse(str(item.get("url") or ""))
        if (
            url.scheme in {"http", "https"}
            and url.netloc
            and item.get("kind") not in {"nsi", "osm_tag", "reference_case"}
            and item.get("scope") == "building"
            and item.get("quote")
            and (item.get("recorded_at_utc") or item.get("retrieved_at") or item.get("accessed_at"))
        ):
            return True
    return False


def reconcile_assessments(task, judgments, required_providers=("claude", "gpt")):
    """Return agreement, review, unresolved or incomplete without averaging units.

    An incomplete provider result has ``assessment=None`` and must not be passed
    to a domain update. A disagreement is a human-review proposal, not a vote in
    favor of either model. Raw judgments and their evidence stay separate.
    """
    if task not in TASKS:
        raise ValueError(f"Unsupported dual-review task: {task}")
    required = tuple(required_providers)
    if not required or len(set(required)) != len(required):
        raise ValueError("Required providers must be nonempty and distinct")
    preserved = deepcopy(dict(judgments))
    invalid = [
        provider for provider in required if not _valid_output(task, judgments.get(provider))
    ]
    outputs = [judgments[provider] for provider in required if provider not in invalid]
    if not invalid and len({output["entity_id"] for output in outputs}) != 1:
        invalid = list(required)
    result = {
        "task": task,
        "status": "incomplete",
        "assessment": None,
        "needs_human_review": False,
        "judgments": preserved,
        "provider_evidence": {
            provider: deepcopy(output.get("evidence", []))
            for provider, output in preserved.items()
            if isinstance(output, Mapping)
        },
    }
    if invalid:
        result["incomplete_providers"] = invalid
        result["reason"] = "Missing, failed, malformed, or mismatched-entity provider judgment"
        return result
    identity = outputs[0]["entity_id"]
    evidence = _evidence_union(outputs)
    if task == "residential_use":
        labels = {output["ai_inferred_use"] for output in outputs}
        if len(labels) == 1:
            label = next(iter(labels))
            result["status"] = "unresolved" if label == "unknown" else "agreed"
            reason = "Required judges returned the same use class; agreement is not independent ground truth"
        else:
            label = "unknown"
            result.update(status="disagreement", needs_human_review=True)
            reason = (
                "Required judges disagree on building use; preserve unknown pending human review"
            )
        result["assessment"] = {
            "entity_id": identity,
            "ai_inferred_use": label,
            "evidence": evidence,
            "reason": reason,
        }
        return result

    scopes = {output["units_scope"] for output in outputs}
    if task == "unit_review":
        counts = [_count(output.get("recommended_units")) for output in outputs]
        common_count = counts[0] if len(set(counts)) == 1 and counts[0] is not None else None
        # Keep observations in the ledger; the explicit review decision states
        # whether a note is an actionable attribution issue.
        flags = [output["review_required"] for output in outputs]
        issues = list(
            dict.fromkeys(str(issue) for output in outputs for issue in output.get("issues", []))
        )
        scope = outputs[0]["units_scope"] if len(scopes) == 1 else "unresolved"
        count_conflict = len({count for count in counts if count is not None}) > 1
        needs_review = any(flags) or len(scopes) > 1 or scope != "building" or count_conflict
        disagreement = len(set(flags)) > 1 or len(scopes) > 1
        if count_conflict:
            disagreement = True
            issues.append("independent_judges_disagree_on_recommended_units")
        if len(scopes) > 1:
            issues.append("independent_judges_disagree_on_units_scope")
        result.update(
            status="disagreement" if disagreement else "agreed", needs_human_review=needs_review
        )
        result["assessment"] = {
            "entity_id": identity,
            "review_required": needs_review,
            "issues": list(dict.fromkeys(issues)),
            "recommended_units": common_count,
            "units_scope": scope,
            "evidence": evidence,
            "reason": "Human review required by at least one judgment or scope conflict"
            if needs_review
            else "Both judges raise no unit issues; retain existing provenance without ground-truth promotion",
        }
        return result

    counts = [_count(output.get("units")) for output in outputs]
    unresolved = all(value is None for value in counts)
    common_scope = outputs[0]["units_scope"] if len(scopes) == 1 else "unresolved"
    assessment = {
        "entity_id": identity,
        "units": None,
        "units_low": None,
        "units_high": None,
        "units_method": "unresolved",
        "units_scope": common_scope,
        "evidence": evidence,
        "reason": "No common single-building unit estimate",
    }
    if unresolved and len(scopes) == 1:
        # Neither provider supplies a count; uncertainty is not an API failure.
        result.update(status="unresolved", assessment=assessment)
        return result
    if len(scopes) == 1 and common_scope == "complex" and len(set(counts)) == 1:
        assessment["reason"] = (
            "Both judgments concern complex-level units; no count assigned to one building"
        )
        result.update(status="unresolved", assessment=assessment)
        return result
    if len(set(counts)) != 1 or common_scope != "building" or len(scopes) != 1:
        result.update(status="disagreement", assessment=assessment, needs_human_review=True)
        assessment["reason"] = "Required judges disagree on units or scope; counts are not averaged"
        return result
    count = counts[0]
    method = "web_extracted" if _has_building_web_source(evidence) else "model_imputed"
    bounds = [
        (int(output["units_low"]), int(output["units_high"]))
        for output in outputs
        if output.get("units_low") is not None and output.get("units_high") is not None
    ]
    assessment.update(
        units=count,
        units_method=method,
        units_low=min(low for low, _ in bounds) if method == "model_imputed" and bounds else None,
        units_high=max(high for _, high in bounds)
        if method == "model_imputed" and bounds
        else None,
        reason="Required judges returned the same single-building count; source extraction and model estimation remain distinct",
    )
    result.update(status="agreed", assessment=assessment)
    return result
