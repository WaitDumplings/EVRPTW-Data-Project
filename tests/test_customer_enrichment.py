from copy import deepcopy

import pytest

from evrptw_cle.customer_enrichment import (
    apply_purpose_assessment,
    apply_review_decision,
    apply_unit_assessment,
    coverage_summary,
    customer_eligible,
    evaluate_unit_predictions,
    initialize_record,
    mask_units_for_evaluation,
    select_reference_examples,
)


WEB = [{"url": "https://example.org/building", "retrieved_at": "2026-11-02",
        "scope": "building", "quote": "This building contains 12 residential units."}]
OSM = [{"osm_id": "w1", "tag_key": "building:use", "tag_value": "residential"}]


def building(entity_id="a", units=12, use="residential", **extra):
    return initialize_record({"entity_id": entity_id, "osm_use": use,
                              "units_raw_nsi": units, "nsi_attribution_clear": True,
                              **extra})


def verified(record, units=12):
    return apply_review_decision(record, {"field": "units", "action": "confirm",
        "units_final": units, "units_scope": "building", "independently_verified": True,
        "evidence": WEB})


def test_complex_total_never_becomes_single_building_units():
    record = building(units=10871, nsi_attribution_clear=False, nsi_scope="unresolved")
    result = apply_unit_assessment(record, {"action": "replace", "method": "web_extracted",
        "units": 10871, "scope": "complex", "evidence": WEB})
    assert result["units_raw_nsi"] == 10871
    assert result["units_final"] is None
    assert result["units_scope"] == "complex"
    assert not customer_eligible(result)


def test_ai_proposal_requires_review_to_replace_existing_value_and_preserves_raw():
    original = building(units=120)
    snapshot = deepcopy(original)
    proposed = apply_unit_assessment(original, {"action": "replace", "method": "web_extracted",
        "units": 12, "scope": "building", "evidence": WEB})
    assert original == snapshot
    assert proposed["units_final"] == 120
    assert proposed["proposed_units"] == 12
    assert not customer_eligible(proposed)
    fixed = verified(proposed)
    assert fixed["units_final"] == 12
    assert fixed["units_raw_nsi"] == 120
    assert fixed["units_method"] == "human_corrected"
    assert customer_eligible(fixed)


def test_purpose_and_units_have_independent_sources():
    initial = building(units=1, use="unknown")
    inferred = apply_purpose_assessment(initial, {"ai_inferred_use": "residential", "evidence": OSM})
    assert inferred["osm_use"] == "unknown"
    assert inferred["residential_source"] == "ai_inferred"
    assert inferred["units_method"] == "nsi_retained"
    assert inferred["units_raw_nsi"] == 1
    assert customer_eligible(inferred, "u1")
    unsupported = apply_purpose_assessment(initial, {"ai_inferred_use": "residential"})
    assert unsupported["residential_use"] == "unknown"


def test_nonresidential_ai_conflict_does_not_silently_override_osm():
    record = building(use="nonresidential")
    result = apply_purpose_assessment(record, {"ai_inferred_use": "residential", "evidence": OSM})
    assert result["residential_use"] == "nonresidential"
    assert result["purpose_review_status"] == "pending_conflict"
    assert not customer_eligible(result)


def test_model_fill_not_recycled_into_verified_examples():
    missing = building(units=None)
    filled = apply_unit_assessment(missing, {"action": "impute", "method": "model_imputed",
        "units": 12, "scope": "building", "units_low": 8, "units_high": 20,
        "independently_verified": True, "evidence": WEB})
    assert customer_eligible(filled)
    assert not customer_eligible(filled, include_model_imputed=False)
    assert not select_reference_examples([filled])
    assert len(select_reference_examples([verified(filled)])) == 1


def test_multiple_positive_nsi_records_not_summed_into_units():
    record = building(units=[1, 1], nsi_count=2)
    assert record["units_raw_nsi"] == [1, 1]
    assert record["units_final"] is None
    assert not customer_eligible(record)


def test_unknown_denominator_includes_missing_nsi_and_unprocessed_buildings():
    records = [building("one", 1), building("many", 120),
               building("unknown", None, use="unknown"),
               building("unknown_cap", 500, use="unknown"),
               building("shop", 1, use="nonresidential"),
               building("mixed", 2, use="mixed"),
               building("unmatched", 1, matched_ms_osm=False)]
    result = coverage_summary(records)
    assert result["purpose_denominator"] == 4
    assert result["residential_identification_rate"] == 0.5
    assert result["mixed"] == 1
    assert result["profiles"]["u1"]["eligible_customers"] == 1
    assert result["profiles"]["u_lt100"]["customer_coverage"] == 0.25
    assert result["profiles"]["unrestricted"]["customer_coverage"] == 0.5


def test_holdout_masks_target_and_reports_missing_predictions():
    references = [verified(building("a"), 12), verified(building("b"), 20)]
    masked = mask_units_for_evaluation(references[0])
    assert "units_raw_nsi" not in masked and "units_final" not in masked
    assert "units_evidence" not in masked and "review_history" not in masked
    result = evaluate_unit_predictions(references, [{"entity_id": "a", "units": 15,
        "units_low": 10, "units_high": 18}])
    assert result["reference_buildings"] == 2
    assert result["predicted_buildings"] == 1
    assert result["missing_or_invalid_predictions"] == 1
    assert result["mae_units"] == 3
    assert result["mean_absolute_relative_error"] == 0.25
    assert result["interval_coverage"] == 1


def test_provider_review_and_imputation_schemas_are_applied_without_adapter():
    review = apply_unit_assessment(building(units=120), {"entity_id": "a",
        "review_required": True, "issues": ["complex_scope"], "recommended_units": 12,
        "units_scope": "building", "reason": "The page's larger count describes the complex.",
        "evidence": WEB})
    assert review["units_review_status"] == "pending"
    assert review["units_final"] == 120
    assert review["proposed_units"] == 12
    missing = building(units=None)
    filled = apply_unit_assessment(missing, {"entity_id": "a", "units": 12,
        "units_low": 8, "units_high": 20, "units_method": "model_imputed",
        "units_scope": "building", "reason": "Comparable verified buildings.",
        "evidence": [{"kind": "reference_case", "source_id": "verified-1", "quote": "12 units"}]})
    assert filled["units_final"] == 12
    assert filled["units_method"] == "model_imputed"


def test_web_extraction_cannot_relabel_nsi_and_invalid_intervals_stay_pending():
    missing = building(units=None)
    not_web = apply_unit_assessment(missing, {"units": 12, "units_method": "web_extracted",
        "units_scope": "building", "evidence": OSM})
    assert not_web["units_final"] is None
    assert not_web["units_review_status"] == "pending"
    bad_interval = apply_unit_assessment(missing, {"units": 12,
        "units_method": "model_imputed", "units_scope": "building",
        "units_low": 15, "units_high": 20})
    assert bad_interval["units_final"] is None
    assert bad_interval["units_review_status"] == "pending"


def test_human_acceptance_without_independent_evidence_keeps_model_imputation_lineage():
    filled = apply_unit_assessment(building(units=None), {
        "units": 12, "units_method": "model_imputed", "units_scope": "building",
        "units_low": 8, "units_high": 20})
    accepted = apply_review_decision(filled, {"field": "units", "action": "accept",
        "units_final": 12, "units_scope": "building", "independently_verified": False})
    assert accepted["units_method"] == "model_imputed"
    assert accepted["units_raw_nsi"] is None
    assert not accepted["units_independently_verified"]
    assert customer_eligible(accepted, include_model_imputed=True)
    assert not customer_eligible(accepted, include_model_imputed=False)
    assert not select_reference_examples([accepted])


@pytest.mark.parametrize("independent", [False, True])
def test_human_changed_estimate_clears_previous_prediction_interval(independent):
    filled = apply_unit_assessment(building(units=None), {
        "units": 12, "units_method": "model_imputed", "units_scope": "building",
        "units_low": 8, "units_high": 20})
    corrected = apply_review_decision(filled, {"field": "units", "action": "correct",
        "units_final": 30, "units_scope": "building", "independently_verified": independent,
        "evidence": [dict(WEB[0], quote="This building contains 30 residential units.")]
                    if independent else []})
    assert corrected["units_final"] == 30
    assert corrected["units_raw_nsi"] is None
    assert "units_low" not in corrected and "units_high" not in corrected
    assert corrected["units_method"] == ("human_corrected" if independent else "model_imputed")


def test_purpose_only_human_decision_preserves_all_unit_fields():
    initial = building(use="unknown")
    initial["units_evidence"] = deepcopy(WEB)
    unit_state = {key: deepcopy(value) for key, value in initial.items()
                  if key.startswith(("units_", "unit_", "nsi_"))}
    result = apply_review_decision(initial, {"field": "purpose", "action": "confirm",
        "residential_use": "residential", "evidence": OSM, "units_final": 99})
    assert result["residential_use"] == "residential"
    assert result["residential_source"] == "human_reviewed"
    assert {key: result[key] for key in unit_state} == unit_state


def test_invalid_human_decision_action_raises_without_mutating_record():
    initial = building()
    before = deepcopy(initial)
    with pytest.raises(ValueError, match="Unknown human review action"):
        apply_review_decision(initial, {"field": "units", "action": "approve_everything",
                                        "units_final": 99})
    assert initial == before


def test_human_acceptance_of_unchanged_web_count_preserves_its_source_evidence():
    sourced = apply_unit_assessment(building(units=None), {
        "units": 12, "units_method": "web_extracted", "units_scope": "building",
        "evidence": WEB})
    accepted = apply_review_decision(sourced, {"field": "units", "action": "accept",
        "units_final": 12, "units_scope": "building", "independently_verified": False})
    assert accepted["units_method"] == "web_extracted"
    assert accepted["units_evidence"] == WEB
    assert not accepted["units_independently_verified"]
