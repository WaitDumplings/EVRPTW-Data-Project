"""Dual-run orchestration tests; every model response here is an explicit fixture."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from evrptw_cle.ai_provider import AIProviderError
from evrptw_cle.customer_enrichment import (
    customer_eligible, initialize_record, select_reference_examples,
)
from evrptw_cle.stage1_v2 import (
    _eligibility, apply_dual_record, apply_reviews, export_review, import_assessments, population, run_ai,
)


OSM_EVIDENCE = [{"kind": "osm_tag", "source_id": "w1", "quote": "name=Example Apartments"}]


def purpose(label="residential"):
    return {"entity_id": "b0", "ai_inferred_use": label, "evidence": deepcopy(OSM_EVIDENCE),
            "reason": "Synthetic fixture name semantics"}


def setup(tmp_path, monkeypatch, *, use="unknown", units=1):
    project = {"output": str(tmp_path / "out"), "cities": {"test": {}},
               "ai": {"require_dual_review": True, "batch_limit": 1}}
    base = Path(project["output"]) / "cities/test"
    inputs = base / "customer_inputs"
    inputs.mkdir(parents=True)
    record = initialize_record({"entity_id": "b0", "entity_index": 0,
        "osm_use": use, "units_raw_nsi": units, "nsi_count": int(units is not None),
        "nsi_attribution_clear": units is not None, "geometry_clear": True,
        "matched_ms_osm": True, "area_m2": 100.})
    table_record = {key: value for key, value in record.items()
                    if not isinstance(value, (dict, list))}
    pd.DataFrame([table_record]).to_parquet(inputs / "population.parquet", index=False)
    pd.DataFrame({"owner": [0] if units is not None else [],
        "fd_id": [10] if units is not None else [],
        "raw_resunits": [units] if units is not None else []}).to_parquet(
            inputs / "source_records.parquet", index=False)

    class EvidenceReader:
        def __init__(self, sources, geometry_dir):
            pass

        def payload(self, row):
            return {"entity_id": row.entity_id, "longitude": -117., "latitude": 33.,
                "microsoft_building_ids": ["ms1"],
                "osm_objects": [{"osm_id": "w1", "tags": {"building": "yes", "name": "Example Apartments"}}],
                "source_snapshots": {"osm": "synthetic-test"},
                "building_features": {"area_m2": 100.}}

    monkeypatch.setattr("evrptw_cle.building_inputs.BuildingEvidenceReader", EvidenceReader)
    return project, base, record


class RecordedFixtureProvider:
    def __init__(self, name, output, *, fail=False, mutate_input=False):
        self.name, self.output = name, output
        self.fail, self.mutate_input = fail, mutate_input
        self.calls = []

    def run(self, task, payload):
        self.calls.append((task, deepcopy(payload)))
        if self.mutate_input:
            payload["private_first_judge_scratch"] = "must not reach the other judge"
            payload["osm_objects"][0]["tags"]["private_first_judge_scratch"] = True
        if self.fail:
            raise AIProviderError("Synthetic second-provider failure", record_path=Path("mock-gpt-failure.json"))
        return {"output": deepcopy(self.output), "request_id": "mock-" + self.name,
                "record_path": "mock-" + self.name + ".json", "model": "synthetic-" + self.name}


def test_blind_judges_receive_identical_payload_even_if_first_mutates_its_copy(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch)
    claude = RecordedFixtureProvider("claude", purpose(), mutate_input=True)
    gpt = RecordedFixtureProvider("gpt", purpose())
    monkeypatch.setattr("evrptw_cle.stage1_v2.providers_for", lambda project: {"claude": claude, "gpt": gpt})
    result = run_ai(project, ["test"], "residential_use", limit=1)
    assert result["status"] == "batch_complete" and len(result["completed"]) == 1
    assert claude.calls[0] == gpt.calls[0]
    assert "private_first_judge_scratch" not in gpt.calls[0][1]
    assert "private_first_judge_scratch" not in gpt.calls[0][1]["osm_objects"][0]["tags"]
    _, states = population(base)
    record = states["b0"]
    assert record["residential_use"] == "residential" and record["residential_source"] == "ai_inferred"
    assert record["units_method"] == "nsi_retained"
    assert not record["units_independently_verified"] and not select_reference_examples([record])
    assert set(json.loads(record["ai_providers_json"])) == {"claude", "gpt"}


def test_second_provider_failure_never_applies_the_first_judgment(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch)
    claude = RecordedFixtureProvider("claude", purpose())
    gpt = RecordedFixtureProvider("gpt", purpose(), fail=True)
    monkeypatch.setattr("evrptw_cle.stage1_v2.providers_for", lambda project: {"claude": claude, "gpt": gpt})
    result = run_ai(project, ["test"], "residential_use", limit=1)
    assert result["status"] == "failed" and not result["completed"]
    assert result["failed"][0]["provider"] == "gpt"
    assert len(claude.calls) == len(gpt.calls) == 1
    frame, states = population(base)
    assert not states and frame.residential_use.tolist() == ["unknown"]
    assert not (base / "customer_inputs/assessments.jsonl").exists()


def test_purpose_disagreement_is_ineligible_and_exports_both_votes_for_human_review(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr("evrptw_cle.stage1_v2.providers_for", lambda project: {
        "claude": RecordedFixtureProvider("claude", purpose("residential")),
        "gpt": RecordedFixtureProvider("gpt", purpose("nonresidential"))})
    run_ai(project, ["test"], "residential_use", limit=1)
    _, states = population(base)
    record = states["b0"]
    assert record["residential_use"] == "unknown" and record["purpose_review_status"] == "pending_conflict"
    assert record["dual_review_status"] == "disagreement" and not customer_eligible(record)
    destination = tmp_path / "human.csv"
    assert export_review(project, ["test"], destination)["pending"] == 1
    queue = pd.read_csv(destination, keep_default_na=False)
    assert "claude=residential" in queue.reason.iloc[0]
    assert "gpt=nonresidential" in queue.reason.iloc[0]
    evidence = json.loads(queue.evidence_json.iloc[0])
    assert "claude" in json.dumps(evidence) and "gpt" in json.dumps(evidence)


@pytest.mark.parametrize("task", ["unit_review", "unit_impute"])
def test_matching_unit_votes_apply_without_becoming_human_verified(task):
    record = initialize_record({"entity_id": "b0", "osm_use": "residential",
        "units_raw_nsi": 1 if task == "unit_review" else None,
        "nsi_count": 1 if task == "unit_review" else 0,
        "nsi_attribution_clear": task == "unit_review"})
    output = ({"entity_id": "b0", "review_required": False, "issues": [],
        "recommended_units": None, "units_scope": "building", "reason": "Fixture no issues", "evidence": []}
        if task == "unit_review" else {"entity_id": "b0", "units": 12, "units_low": 8,
        "units_high": 20, "units_method": "model_imputed", "units_scope": "building",
        "reason": "Fixture numeric estimate", "evidence": []})
    replies = {provider: {"output": deepcopy(output), "model": "synthetic-" + provider,
                         "request_id": "mock-" + provider} for provider in ["claude", "gpt"]}
    result, reconciliation = apply_dual_record(record, task, replies)
    assert reconciliation["status"] == "agreed"
    assert result["units_method"] == ("nsi_retained" if task == "unit_review" else "model_imputed")
    assert not result["units_independently_verified"]
    assert not select_reference_examples([result])
    if task == "unit_impute":
        assert result["units_final"] == 12 and result["units_raw_nsi"] is None
        assert not customer_eligible(result, include_model_imputed=False)


def test_paired_import_applies_both_judgments_and_preserves_model_sources(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch)
    path = tmp_path / "paired.jsonl"
    path.write_text(json.dumps({"city": "test", "judgments": {
        "claude": {"output": purpose(), "model": "synthetic-claude"},
        "gpt": {"output": purpose(), "model": "synthetic-gpt"}}}) + "\n")
    result = import_assessments(project, ["test"], "residential_use", path)
    assert result["applied"] == 1 and result["pending_independent_review"] == 0
    _, states = population(base)
    record = states["b0"]
    assert record["residential_use"] == "residential" and record["dual_review_status"] == "agreed"
    assert record["residential_source"] == "ai_inferred"
    models = {run["provider"]: run["model"] for run in record["ai_runs"]
              if run["provider"] in {"claude", "gpt"}}
    assert models == {"claude": "synthetic-claude", "gpt": "synthetic-gpt"}
    assert not record["units_independently_verified"]
    assert not select_reference_examples([record])


def test_single_unit_review_import_stays_ineligible_until_second_judgment_arrives(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch, use="residential")
    output = {"entity_id": "b0", "review_required": False, "issues": [],
        "recommended_units": None, "units_scope": "building", "reason": "Synthetic no issues",
        "evidence": []}
    first, second = tmp_path / "claude.jsonl", tmp_path / "gpt.jsonl"
    first.write_text(json.dumps({"city": "test", "provider": "claude_cli", "output": output}) + "\n")
    second.write_text(json.dumps({"city": "test", "provider": "codex_cli", "output": output}) + "\n")
    pending = import_assessments(project, ["test"], "unit_review", first)
    assert pending["applied"] == 0 and pending["pending_independent_review"] == 1
    frame, states = population(base)
    assert states["b0"]["units_review_status"] == "pending_independent_review"
    assert not _eligibility(frame, "u1").any()
    assert not customer_eligible(states["b0"])
    finished = import_assessments(project, ["test"], "unit_review", second)
    assert finished["applied"] == 1 and finished["pending_independent_review"] == 0
    frame, states = population(base)
    assert states["b0"]["units_method"] == "nsi_retained"
    assert states["b0"]["units_raw_nsi"] == 1
    assert _eligibility(frame, "u1").all()
    assert not states["b0"]["units_independently_verified"]


@pytest.mark.parametrize("combined_file", [False, True])
def test_replaying_completed_import_does_not_reopen_independent_review(tmp_path, monkeypatch, combined_file):
    project, base, _ = setup(tmp_path, monkeypatch)
    first = {"city": "test", "provider": "claude_cli", "output": purpose()}
    second = {"city": "test", "provider": "codex_cli", "output": purpose()}
    replay = tmp_path / "claude.jsonl"
    if combined_file:
        replay.write_text("\n".join(json.dumps(row) for row in [first, second]) + "\n")
        initial = import_assessments(project, ["test"], "residential_use", replay)
        assert initial["applied"] == 1
    else:
        replay.write_text(json.dumps(first) + "\n")
        other = tmp_path / "gpt.jsonl"
        other.write_text(json.dumps(second) + "\n")
        import_assessments(project, ["test"], "residential_use", replay)
        import_assessments(project, ["test"], "residential_use", other)
    frame, states = population(base)
    assert states["b0"]["purpose_review_status"] == "not_flagged"
    assert _eligibility(frame, "u1").all()
    snapshot = (base / "customer_inputs/assessments.jsonl").read_bytes()
    repeated = import_assessments(project, ["test"], "residential_use", replay)
    assert repeated["applied"] == 0 and repeated["pending_independent_review"] == 0
    assert repeated["reused"] == (2 if combined_file else 1)
    assert (base / "customer_inputs/assessments.jsonl").read_bytes() == snapshot
    assert _eligibility(population(base)[0], "u1").all()


def test_deferring_purpose_keeps_it_in_human_queue_without_creating_unit_pending(tmp_path, monkeypatch):
    project, base, _ = setup(tmp_path, monkeypatch)
    paired = tmp_path / "conflict.jsonl"
    paired.write_text(json.dumps({"city": "test", "judgments": {
        "claude": purpose("residential"), "gpt": purpose("nonresidential")}}) + "\n")
    import_assessments(project, ["test"], "residential_use", paired)
    csv = tmp_path / "human.csv"
    export_review(project, ["test"], csv)
    rows = pd.read_csv(csv, keep_default_na=False)
    rows.loc[0, "decision"] = "defer"
    rows.to_csv(csv, index=False)
    apply_reviews(project, ["test"], csv)
    record = population(base)[1]["b0"]
    assert record["purpose_review_status"] == "pending"
    assert record["units_review_status"] == "not_flagged"
    assert record["units_raw_nsi"] == record["units_final"] == 1
    assert export_review(project, ["test"], csv)["pending"] == 1
    rows = pd.read_csv(csv, keep_default_na=False)
    rows.loc[0, "decision"] = "confirm"
    rows.loc[0, "residential_use"] = "residential"
    rows.to_csv(csv, index=False)
    apply_reviews(project, ["test"], csv)
    frame, states = population(base)
    assert states["b0"]["purpose_review_status"] == "resolved"
    assert states["b0"]["units_review_status"] == "not_flagged"
    assert _eligibility(frame, "u1").all()
    assert export_review(project, ["test"], csv)["pending"] == 0
