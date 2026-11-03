from copy import deepcopy

from evrptw_cle.dual_review import reconcile_assessments


def purpose(label):
    return {"entity_id": "building-1", "ai_inferred_use": label, "reason": "Existing tags",
            "evidence": [{"kind": "osm_tag", "source_id": "w1", "quote": "building=yes"}]}


def review(flag=False, count=None, scope="building"):
    return {"entity_id": "building-1", "review_required": flag, "recommended_units": count,
            "units_scope": scope, "issues": ["unit scope uncertain"] if flag else [],
            "reason": "Review", "evidence": []}


def impute(count=12, scope="building", low=8, high=20, evidence=None):
    return {"entity_id": "building-1", "units": count, "units_scope": scope,
            "units_method": "model_imputed" if count is not None else "unresolved",
            "units_low": low if count is not None else None,
            "units_high": high if count is not None else None,
            "reason": "Comparable buildings", "evidence": evidence or []}


def test_purpose_consensus_preserves_raw_votes_and_unknown_is_unresolved():
    votes = {"claude": purpose("residential"), "gpt": purpose("residential")}
    before = deepcopy(votes)
    result = reconcile_assessments("residential_use", votes)
    assert result["status"] == "agreed" and not result["needs_human_review"]
    assert result["assessment"]["ai_inferred_use"] == "residential"
    assert votes == before and result["judgments"] == before
    both_unknown = reconcile_assessments("residential_use", {
        "claude": purpose("unknown"), "gpt": purpose("unknown")})
    assert both_unknown["status"] == "unresolved" and not both_unknown["needs_human_review"]


def test_purpose_disagreement_keeps_unknown_and_queues_human_review():
    result = reconcile_assessments("residential_use", {
        "claude": purpose("residential"), "gpt": purpose("nonresidential")})
    assert result["status"] == "disagreement" and result["needs_human_review"]
    assert result["assessment"]["ai_inferred_use"] == "unknown"


def test_unit_review_keeps_existing_source_and_either_judge_can_raise_issue():
    retained = reconcile_assessments("unit_review", {"claude": review(), "gpt": review()})
    assert retained["status"] == "agreed" and not retained["needs_human_review"]
    assert retained["assessment"]["review_required"] is False
    assert "independently_verified" not in retained["assessment"]
    flagged = reconcile_assessments("unit_review", {
        "claude": review(True, 12), "gpt": review(False, 15)})
    assert flagged["needs_human_review"] and flagged["assessment"]["review_required"]
    assert flagged["assessment"]["recommended_units"] is None
    conflict = reconcile_assessments("unit_review", {
        "claude": review(scope="building"), "gpt": review(scope="complex")})
    assert conflict["needs_human_review"] and conflict["assessment"]["units_scope"] == "unresolved"


def test_unit_review_different_counts_require_review_even_when_neither_judge_flags():
    result = reconcile_assessments("unit_review", {
        "claude": review(False, 12), "gpt": review(False, 15)})
    assert result["status"] == "disagreement"
    assert result["needs_human_review"] and result["assessment"]["review_required"]
    assert result["assessment"]["recommended_units"] is None
    assert "independent_judges_disagree_on_recommended_units" in result["assessment"]["issues"]


def test_unit_review_shared_unresolved_scope_still_requires_human_review():
    result = reconcile_assessments("unit_review", {
        "claude": review(False, scope="unresolved"),
        "gpt": review(False, scope="unresolved")})
    assert result["needs_human_review"] and result["assessment"]["review_required"]
    assert result["assessment"]["units_scope"] == "unresolved"
    assert result["assessment"]["recommended_units"] is None


def test_unit_review_informational_observation_does_not_override_explicit_no_review_verdict():
    claude = review(False, 2)
    claude["issues"] = ["Footprint area differs by 35%; this does not affect the two-unit count."]
    result = reconcile_assessments("unit_review", {"claude": claude, "gpt": review(False)})
    assert result["status"] == "agreed"
    assert not result["needs_human_review"] and not result["assessment"]["review_required"]
    assert result["assessment"]["recommended_units"] is None
    assert result["assessment"]["issues"] == claude["issues"]


def test_unit_imputation_exact_consensus_uses_interval_union_or_real_web_source():
    estimated = reconcile_assessments("unit_impute", {
        "claude": impute(low=8, high=15), "gpt": impute(low=10, high=20)})
    assert estimated["status"] == "agreed"
    assert estimated["assessment"]["units_method"] == "model_imputed"
    assert estimated["assessment"]["units_low"] == 8 and estimated["assessment"]["units_high"] == 20
    web = [{"kind": "web", "url": "https://example.org/building", "scope": "building",
            "quote": "This building has 12 homes.", "recorded_at_utc": "2026-11-02T00:00:00Z"}]
    extracted = reconcile_assessments("unit_impute", {
        "claude": impute(evidence=web), "gpt": impute()})
    assert extracted["assessment"]["units_method"] == "web_extracted"
    assert extracted["provider_evidence"]["claude"] == web
    assert extracted["provider_evidence"]["gpt"] == []
    recycled_nsi = reconcile_assessments("unit_impute", {
        "claude": impute(evidence=[dict(web[0], kind="nsi")]), "gpt": impute()})
    assert recycled_nsi["assessment"]["units_method"] == "model_imputed"


def test_unit_number_disagreement_and_complex_totals_are_never_assigned_or_averaged():
    conflict = reconcile_assessments("unit_impute", {"claude": impute(12), "gpt": impute(15)})
    assert conflict["status"] == "disagreement" and conflict["needs_human_review"]
    assert conflict["assessment"]["units"] is None
    complex_total = reconcile_assessments("unit_impute", {
        "claude": impute(120, "complex", 100, 140), "gpt": impute(120, "complex", 100, 140)})
    assert complex_total["status"] == "unresolved"
    assert complex_total["assessment"]["units"] is None
    complex_conflict = reconcile_assessments("unit_impute", {
        "claude": impute(120, "complex", 100, 160),
        "gpt": impute(150, "complex", 100, 160)})
    assert complex_conflict["status"] == "disagreement" and complex_conflict["needs_human_review"]
    assert complex_conflict["assessment"]["units"] is None
    missing = reconcile_assessments("unit_impute", {
        "claude": impute(None, "unresolved"), "gpt": impute(None, "unresolved")})
    assert missing["status"] == "unresolved" and not missing["needs_human_review"]
    scope_conflict = reconcile_assessments("unit_impute", {
        "claude": impute(None, "unresolved"), "gpt": impute(None, "complex")})
    assert scope_conflict["status"] == "disagreement" and scope_conflict["needs_human_review"]


def test_missing_failed_or_wrong_entity_provider_is_incomplete_not_unknown():
    for votes in [{"claude": purpose("residential")},
                  {"claude": purpose("residential"), "gpt": {"error": "runner unavailable"}},
                  {"claude": purpose("residential"),
                   "gpt": dict(purpose("residential"), entity_id="another-building")}]:
        result = reconcile_assessments("residential_use", votes)
        assert result["status"] == "incomplete" and result["assessment"] is None
        assert not result["needs_human_review"]
