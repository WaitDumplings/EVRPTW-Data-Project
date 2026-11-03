"""Field-level customer enrichment over fixed Microsoft--OSM building identities.

These functions apply recorded assessments; they do not call a model, move a
building, distribute a complex's units, or overwrite the original NSI value.
The AI provider and human-review files can therefore share one interpretation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from statistics import mean, median
from typing import Any
from urllib.parse import urlparse

USES = frozenset({"residential", "nonresidential", "mixed", "unknown"})
UNIT_SCOPES = frozenset({"building", "complex", "unresolved"})
UNIT_METHODS = frozenset({
    "nsi_retained", "web_extracted", "human_verified", "human_corrected",
    "model_imputed", "nsi_representative", "neighbor_estimated", "unresolved",
})
DEFAULT_PROFILES = ("u1", "u_lt50", "u_lt100", "unrestricted")


def positive_integer(value: Any) -> int | None:
    """Return a usable unit count without truncating an estimate or a list."""
    if isinstance(value, (bool, list, tuple, dict)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        return None
    return int(number)


def _use(value: Any) -> str:
    text = str(value or "unknown").lower()
    if text not in USES:
        raise ValueError(f"Unknown residential-use class: {value!r}")
    return text


def _scope(value: Any) -> str:
    text = str(value or "unresolved").lower()
    if text not in UNIT_SCOPES:
        raise ValueError(f"Unknown units scope: {value!r}")
    return text


def _evidence(value: Any) -> list[dict]:
    if not value:
        return []
    if isinstance(value, Mapping):
        value = [value]
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def evidence_is_traceable(evidence: Iterable[Mapping]) -> bool:
    """Accept a referenced OSM tag, web source, or documented human observation.

    This checks provenance form; it is not a claim that the evidence is true.
    """
    for item in evidence:
        if item.get("osm_id") and item.get("tag_key") and "tag_value" in item:
            return True
        if item.get("url") and (item.get("retrieved_at") or item.get("accessed_at")
                                or item.get("recorded_at_utc")):
            return True
        if item.get("kind") in {"osm_tag", "reference_case", "nsi"} and item.get(
                "source_id") and item.get("quote"):
            return True
        if item.get("type") == "manual" and item.get("note"):
            return True
        if item.get("evidence_id"):
            return True
    return False


def building_web_evidence(evidence: Iterable[Mapping]) -> bool:
    """A web-extracted building count needs an actual dated, quoted web source."""
    for item in evidence:
        url = urlparse(str(item.get("url") or ""))
        if (url.scheme in {"http", "https"} and url.netloc and item.get("quote")
                and item.get("scope") == "building"
                and (item.get("retrieved_at") or item.get("accessed_at")
                     or item.get("recorded_at_utc"))):
            return True
    return False


def initialize_record(building: Mapping[str, Any]) -> dict:
    """Create an enrichment record, retaining all input fields and raw values.

    ``nsi_attribution_clear`` must be supplied by geometric/source-scope matching;
    a positive NSI value by itself does not establish single-building ownership.
    """
    record = deepcopy(dict(building))
    if not record.get("entity_id"):
        raise ValueError("A fixed entity_id is required")
    osm_use = _use(record.get("osm_use", record.get("semantic_state", "unknown")))
    record["osm_use"] = osm_use
    record["residential_use"] = osm_use
    record["residential_source"] = "osm" if osm_use != "unknown" else "unresolved"
    record["matched_ms_osm"] = bool(record.get("matched_ms_osm", True))
    record["geometry_clear"] = bool(record.get("geometry_clear", True))
    record["units_raw_nsi"] = deepcopy(record.get("units_raw_nsi"))
    clear = bool(record.get("nsi_attribution_clear", record.get("nsi_scope_clear", False)))
    count = record.get("nsi_count", 1 if record["units_raw_nsi"] is not None else 0)
    scope = _scope(record.get("nsi_scope", "building" if clear else "unresolved"))
    raw = positive_integer(record["units_raw_nsi"])
    usable = clear and count == 1 and scope == "building" and raw is not None
    record["units_final"] = raw if usable else None
    record["units_scope"] = scope
    record["units_method"] = "nsi_retained" if usable else "unresolved"
    record["units_review_status"] = "not_flagged" if usable else "unresolved"
    record["units_independently_verified"] = False
    record["units_evidence"] = _evidence(record.get("units_evidence"))
    record["purpose_evidence"] = _evidence(record.get("purpose_evidence"))
    record["review_history"] = []
    record["unit_assessments"] = []
    record["purpose_assessments"] = []
    return record


def _assessment_record(record: Mapping, assessment: Mapping, field: str) -> dict:
    result = deepcopy(dict(record))
    if assessment.get("entity_id", result["entity_id"]) != result["entity_id"]:
        raise ValueError("Assessment entity_id differs from the fixed building")
    result.setdefault(field, []).append(deepcopy(dict(assessment)))
    return result


def apply_purpose_assessment(record: Mapping, assessment: Mapping) -> dict:
    """Apply evidence-backed AI use to unknown buildings; queue contradictions."""
    result = _assessment_record(record, assessment, "purpose_assessments")
    label = _use(assessment.get("ai_inferred_use", assessment.get(
        "residential_use", assessment.get("use", "unknown"))))
    evidence = _evidence(assessment.get("evidence"))
    existing = _use(result.get("residential_use", result.get("osm_use")))
    if label == "unknown":
        result["purpose_review_status"] = "unresolved" if existing == "unknown" else "not_flagged"
        return result
    if not evidence_is_traceable(evidence):
        result["purpose_review_status"] = "pending_evidence" if existing == "unknown" else "not_flagged"
        return result
    if existing != "unknown" and existing != label:
        result["purpose_review_status"] = "pending_conflict"
        result["proposed_residential_use"] = label
        return result
    if existing == "unknown":
        result["residential_use"] = label
        result["residential_source"] = "ai_inferred"
        result["purpose_evidence"] = evidence
    result["purpose_review_status"] = "not_flagged"
    return result


def apply_unit_assessment(record: Mapping, assessment: Mapping) -> dict:
    """Flag an existing value or apply a scoped extraction/imputation proposal.

    AI cannot silently replace an existing different count or qualify its own
    estimates as independently verified reference examples. Complex totals stay
    in the assessment ledger and are not assigned to one building.
    """
    result = _assessment_record(record, assessment, "unit_assessments")
    evidence = _evidence(assessment.get("evidence"))
    scope = _scope(assessment.get("units_scope", assessment.get("scope", "unresolved")))
    method = str(assessment.get("units_method", assessment.get("method", "unresolved")))
    derived_action = "impute" if method == "model_imputed" else (
        "replace" if method == "web_extracted" else "unknown")
    action = str(assessment.get("verdict", assessment.get("action", derived_action))).lower()
    value = positive_integer(assessment.get("units_final", assessment.get(
        "units", assessment.get("value"))))
    if "review_required" in assessment:
        if assessment["review_required"] or scope == "complex":
            result["units_review_status"] = "pending"
            result["unit_review_reason"] = assessment.get("reason", "AI raised a unit attribution question")
            result["unit_review_issues"] = deepcopy(assessment.get("issues", []))
            proposal = positive_integer(assessment.get("recommended_units"))
            if proposal is not None:
                result["proposed_units"] = proposal
            if scope == "complex":
                result["units_final"] = None
                result["units_scope"] = "complex"
                result["units_method"] = "unresolved"
        elif result.get("units_review_status") != "pending":
            result["units_review_status"] = "not_flagged" if result.get("units_final") else "unresolved"
        return result
    if action in {"flag", "review", "suspicious"} or assessment.get("suspicious"):
        result["units_review_status"] = "pending"
        result["unit_review_reason"] = assessment.get("reason", "AI raised a unit attribution question")
        return result
    if scope != "building" and value is not None:
        result["units_final"] = None
        result["units_scope"] = scope
        result["units_method"] = "unresolved"
        result["units_review_status"] = "pending"
        result["unit_review_reason"] = "Reported units do not have single-building scope"
        return result
    if action in {"unknown", "defer"}:
        return result
    if action in {"retain", "accept"} and method in {"unresolved", "nsi_retained"}:
        # Passing a plausibility screen is not independent unit verification.
        if result.get("units_review_status") != "pending":
            result["units_review_status"] = "not_flagged"
        return result
    if method not in {"web_extracted", "model_imputed"} or value is None:
        result["units_review_status"] = "pending"
        result["unit_review_reason"] = "No valid extraction or imputation proposal"
        return result
    if method == "web_extracted" and not building_web_evidence(evidence):
        result["units_review_status"] = "pending"
        result["unit_review_reason"] = "Web extraction lacks a dated, quoted single-building web source"
        return result
    low, high = assessment.get("units_low"), assessment.get("units_high")
    if low is not None or high is not None:
        try:
            valid_bounds = (low is not None and high is not None and math.isfinite(float(low))
                            and math.isfinite(float(high)) and 0 < float(low) <= value <= float(high))
        except (TypeError, ValueError):
            valid_bounds = False
        if not valid_bounds:
            result["units_review_status"] = "pending"
            result["unit_review_reason"] = "Unit interval must be positive, ordered, and contain the estimate"
            return result
    current = positive_integer(result.get("units_final"))
    if current is not None and current != value:
        result["proposed_units"] = value
        result["proposed_units_method"] = method
        result["units_review_status"] = "pending"
        result["unit_review_reason"] = "Proposed count differs from the retained building count"
        return result
    if result.get("units_review_status") == "pending":
        # Resolve a flagged existing value through the explicit human decision.
        result["proposed_units"] = value
        result["proposed_units_method"] = method
        return result
    result.update(units_final=value, units_scope="building", units_method=method,
                  units_review_status="estimated" if method == "model_imputed" else "source_supported",
                  units_independently_verified=False, units_evidence=evidence)
    for key in ("units_low", "units_high", "reference_entity_ids", "imputation_model"):
        if key in assessment:
            result[key] = deepcopy(assessment[key])
    return result


assess_units = apply_unit_assessment


def apply_review_decision(record: Mapping, decision: Mapping) -> dict:
    """Apply a recorded human decision, keeping original NSI and review history."""
    result = _assessment_record(record, decision, "review_history")
    field = str(decision.get("field", "units"))
    action = str(decision.get("action", "defer")).lower()
    if action not in {"accept", "confirm", "keep", "retain", "correct", "replace", "reject",
                      "discard", "discard_units", "defer", "pending"}:
        raise ValueError(f"Unknown human review action: {action}")
    evidence = _evidence(decision.get("evidence"))
    if action in {"defer", "pending"}:
        if field in {"units", "both"}:
            result["units_review_status"] = "pending"
        if field in {"purpose", "both"}:
            result["purpose_review_status"] = "pending"
        return result
    if field in {"purpose", "both"} and "residential_use" in decision:
        result["residential_use"] = _use(decision["residential_use"])
        result["residential_source"] = "human_reviewed"
        result["purpose_evidence"] = evidence
        result["purpose_review_status"] = "resolved"
    if field not in {"units", "both"}:
        return result
    if action in {"reject", "discard", "discard_units"}:
        result.update(units_final=None, units_scope="unresolved", units_method="unresolved",
                      units_review_status="rejected", units_independently_verified=False)
        return result
    scope = _scope(decision.get("units_scope", decision.get("scope", "building")))
    value = positive_integer(decision.get("units_final", decision.get("units",
                             result.get("units_final") or result.get("units_raw_nsi"))))
    if scope != "building" or value is None:
        result.update(units_final=None, units_scope=scope, units_method="unresolved",
                      units_review_status="unresolved", units_independently_verified=False)
        return result
    verified = bool(decision.get("independently_verified", False)) and evidence_is_traceable(evidence)
    raw = positive_integer(result.get("units_raw_nsi"))
    prior_method = result.get("units_method", "unresolved")
    previous_value = positive_integer(result.get("units_final"))
    # Reviewing a model estimate without independent unit evidence does not
    # turn it into a source-observed count or change its inclusion policy.
    method = (prior_method if not verified and value == previous_value and prior_method != "unresolved" else
              "model_imputed" if prior_method == "model_imputed" and not verified else
              "human_verified" if raw == value else "human_corrected")
    if not evidence and method == prior_method:
        evidence = _evidence(result.get("units_evidence"))
    result.update(units_final=value, units_scope="building",
                  units_method=method,
                  units_review_status="confirmed", units_independently_verified=verified,
                  units_evidence=evidence)
    if value != previous_value:
        # The old interval described the previous estimate, not the new value.
        result.pop("units_low", None)
        result.pop("units_high", None)
    result.pop("proposed_units", None)
    result.pop("proposed_units_method", None)
    return result


def within_profile(units: Any, profile: str = "unrestricted") -> bool:
    count = positive_integer(units)
    if count is None:
        return False
    if profile in {"u1", "1", "eq1"}:
        return count == 1
    if profile in {"u_lt50", "lt50"}:
        return count < 50
    if profile in {"u_lt100", "lt100"}:
        return count < 100
    if profile in {"unrestricted", "all"}:
        return True
    raise ValueError(f"Unknown units profile: {profile}")


def customer_eligible(record: Mapping, profile: str = "unrestricted", *,
                      include_model_imputed: bool = True) -> bool:
    return bool(
        record.get("matched_ms_osm", True) and record.get("geometry_clear", True)
        and record.get("service_domain_eligible", True)
        and record.get("residential_use") == "residential"
        and record.get("units_scope") == "building"
        and not str(record.get("units_review_status", "")).startswith("pending")
        and not str(record.get("purpose_review_status", "")).startswith("pending")
        and (include_model_imputed or record.get("units_method") not in {"model_imputed", "neighbor_estimated"})
        and within_profile(record.get("units_final"), profile)
    )


def select_reference_examples(records: Iterable[Mapping]) -> list[dict]:
    """Select independently checked single-building targets, never raw AI fills."""
    return [deepcopy(dict(record)) for record in records
            if customer_eligible(record, include_model_imputed=False)
            and record.get("units_independently_verified")
            and evidence_is_traceable(record.get("units_evidence", []))]


def mask_units_for_evaluation(record: Mapping) -> dict:
    """Whitelist building features, hiding raw counts and assessment evidence."""
    fields = {
        "entity_id", "city", "city_slug", "residential_use", "kind", "matching_kind",
        "longitude", "latitude", "location_lon", "location_lat", "area_m2",
        "height_m", "levels", "building_levels", "building_area_m2", "canonical_area_m2",
        "geometry_bbox_wgs84", "microsoft_building_ids", "source_snapshots",
    }
    result = {key: deepcopy(value) for key, value in record.items() if key in fields}
    fields.update({"canonical_geometry_area_m2", "osm_bbox_wgs84"})
    for container in ["features", "geometry_features", "building_features"]:
        if isinstance(record.get(container), Mapping):
            result[container] = {key: deepcopy(value) for key, value in record[container].items()
                                 if key in fields}
    if "osm_objects" in record:
        tags = {"building", "building:use", "building:levels", "height", "building:height",
                "name", "building:material"}
        result["osm_objects"] = [{
            "osm_id": item.get("osm_id"),
            "tags": {key: deepcopy(value) for key, value in item.get("tags", {}).items()
                     if key in tags or key.startswith(("addr:", "roof:"))},
        } for item in record["osm_objects"]]
    return result


def evaluate_unit_predictions(references: Iterable[Mapping], predictions: Iterable[Mapping]) -> dict:
    """Evaluate paired held-out estimates; absent predictions remain in counts."""
    truth = {item["entity_id"]: item for item in select_reference_examples(references)}
    predicted = {item["entity_id"]: item for item in predictions}
    pairs = []
    intervals = []
    methods: dict[str, list[tuple[int, int]]] = {}
    for entity_id, reference in truth.items():
        estimate = predicted.get(entity_id, {})
        value = positive_integer(estimate.get("units_final", estimate.get("units")))
        if value is None or estimate.get("units_scope", "building") != "building":
            continue
        actual = int(reference["units_final"])
        pairs.append((actual, value))
        method = str(estimate.get("units_method", "unspecified"))
        methods.setdefault(method, []).append((actual, value))
        low, high = estimate.get("units_low"), estimate.get("units_high")
        if low is not None and high is not None and float(low) <= float(high):
            intervals.append(float(low) <= actual <= float(high))
    errors = [estimate - actual for actual, estimate in pairs]
    absolute = [abs(value) for value in errors]
    return {
        "reference_buildings": len(truth), "predicted_buildings": len(pairs),
        "missing_or_invalid_predictions": len(truth) - len(pairs),
        "mae_units": mean(absolute) if pairs else None,
        "median_absolute_error_units": median(absolute) if pairs else None,
        "mean_signed_error_units": mean(errors) if pairs else None,
        "mean_absolute_relative_error": mean(abs(pred - actual) / actual
                                              for actual, pred in pairs) if pairs else None,
        "prediction_intervals": len(intervals),
        "interval_coverage": mean(intervals) if intervals else None,
        "by_method": {method: {
            "predicted_buildings": len(values),
            "mae_units": mean(abs(pred - actual) for actual, pred in values),
        } for method, values in sorted(methods.items())},
    }


def coverage_summary(records: Iterable[Mapping], profiles: Iterable[str] = DEFAULT_PROFILES) -> dict:
    """Use R/(R+U), with unknown/missing/unprocessed NSI rows retained in U.

    Mixed use is reported separately; it is not added to this metric's R or U.
    The same purpose denominator is used across every unit-count profile.
    """
    records = list(records)
    ids = [record["entity_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Coverage requires one row per fixed building entity_id")
    matched = [record for record in records
               if record.get("matched_ms_osm", True) and record.get("geometry_clear", True)
               and record.get("service_domain_eligible", True)]
    uses = Counter(_use(record.get("residential_use", record.get("osm_use")))
                   for record in matched)
    denominator = uses["residential"] + uses["unknown"]
    def ratio(count):
        return count / denominator if denominator else None
    results = {}
    for profile in profiles:
        eligible = [record for record in matched if customer_eligible(record, profile)]
        direct = [record for record in eligible if record.get("units_method") != "model_imputed"]
        results[profile] = {
            "eligible_customers": len(eligible),
            "eligible_source_supported_customers": len(direct),
            "eligible_model_imputed_customers": len(eligible) - len(direct),
            "customer_coverage": ratio(len(eligible)),
            "source_supported_customer_coverage": ratio(len(direct)),
            "residential_units": sum(int(record["units_final"]) for record in eligible),
        }
    return {"input_entities": len(records), "matched_geometry_clear_entities": len(matched),
            "residential": uses["residential"], "nonresidential": uses["nonresidential"],
            "unknown": uses["unknown"], "mixed": uses["mixed"],
            "purpose_denominator": denominator,
            "residential_identification_rate": ratio(uses["residential"]),
            "residential_sources": dict(Counter(record.get("residential_source", "unresolved")
                                               for record in matched
                                               if record.get("residential_use") == "residential")),
            "profiles": results}
