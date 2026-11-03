from copy import deepcopy
import csv
import json

import pytest

from evrptw_cle.sample_review_report import summarize


def case(index, *, task="residential_use", city="dallas", stratum="unknown_use", entity=None):
    return {
        "case_id": str(index),
        "city": city,
        "stratum": stratum,
        "task": task,
        "entity_id": entity or f"building-{index}",
        "selection_kind": "random",
    }


def purpose(entity, label):
    return {
        "entity_id": entity,
        "ai_inferred_use": label,
        "reason": "Recorded public evidence",
        "evidence": [
            {
                "kind": "web",
                "source_id": "property",
                "scope": "building",
                "url": "https://example.org/building",
                "quote": "Residential building.",
            }
        ],
    }


def review(entity, *, flag=True, scope="complex"):
    return {
        "entity_id": entity,
        "review_required": flag,
        "recommended_units": None,
        "units_scope": scope,
        "issues": ["complex total"],
        "reason": "Count is complex-wide",
        "evidence": [],
    }


def impute(entity, count=None, scope="unresolved"):
    return {
        "entity_id": entity,
        "units": count,
        "units_low": count,
        "units_high": count,
        "units_scope": scope,
        "units_method": "unresolved" if count is None else "model_imputed",
        "reason": "No direct count available",
        "evidence": [],
    }


def result(item, provider, output, **overrides):
    return {
        "case_id": item["case_id"],
        "provider": provider,
        "status": "success",
        "reply": {
            "output": output,
            "model": provider + "-test-model",
            "request_id": provider + item["case_id"],
            "record_path": "/recorded/" + provider + item["case_id"] + ".json",
        },
        **overrides,
    }


def load_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_disagreement_denominator_excludes_failed_invalid_and_keeps_agreed_human_cases(tmp_path):
    cases = [
        case(0),
        case(1, entity="shared"),
        case(2, task="unit_review", stratum="large_units", entity="shared"),
        case(3, task="unit_impute", city="los-angeles", stratum="missing_units"),
        case(4, city="los-angeles"),
        case(5, city="los-angeles"),
    ]
    outputs = [
        (purpose("building-0", "unknown"), purpose("building-0", "unknown")),
        (purpose("shared", "residential"), purpose("shared", "nonresidential")),
        (review("shared"), review("shared")),
        (impute("building-3"), impute("building-3")),
        (purpose("building-4", "residential"), None),
        (purpose("wrong-building", "residential"), purpose("wrong-building", "residential")),
    ]
    results = []
    for item, pair in zip(cases, outputs):
        for provider, output in zip(("claude", "gpt"), pair):
            results.append(
                result(item, provider, output)
                if output is not None
                else {
                    "case_id": item["case_id"],
                    "provider": provider,
                    "status": "failed",
                    "error": "API rate limit",
                }
            )
    original = deepcopy((cases, results))
    report = summarize(cases, results, tmp_path)
    overall = report["overall"]
    assert overall["sample_cases"] == 6 and overall["sample_locations"] == 5
    assert overall["paired_completed"] == 4 and overall["incomplete"] == 2
    assert overall["disagreement_count"] == 1 and overall["disagreement_rate"] == 0.25
    assert overall["human_review_count"] == 2 and overall["human_review_locations"] == 1
    assert overall["unresolved_count"] == 2
    assert overall["provider_failure_cases"] == 1 and overall["invalid_output_cases"] == 1
    assert (cases, results) == original
    human = load_csv(tmp_path / "human_review.csv")
    assert {row["case_id"] for row in human} == {"1", "2"}
    unresolved = load_csv(tmp_path / "unresolved_cases.csv")
    assert {row["case_id"] for row in unresolved} == {"0", "3", "4", "5"}
    rows = load_csv(tmp_path / "cases.csv")
    assert rows[4]["gpt_error"] == "API rate limit"
    assert rows[5]["gpt_status"] == "invalid_entity"
    assert rows[1]["claude_model"] == "claude-test-model"
    assert json.loads(rows[1]["claude_evidence_json"])[0]["url"] == "https://example.org/building"
    assert rows[1]["gpt_request_path"] == "/recorded/gpt1.json"
    markdown = (tmp_path / "REPORT.zh-CN.md").read_text()
    assert "25.0%" in markdown and "不能外推" in markdown
    assert "不表示人工核实" in markdown


def test_missing_results_have_no_disagreement_rate_and_no_false_manual_flag(tmp_path):
    item = case("none")
    report = summarize([item], [], tmp_path)
    assert report["overall"]["paired_completed"] == 0
    assert report["overall"]["disagreement_rate"] is None
    assert report["overall"]["human_review_count"] == 0
    assert report["overall"]["missing_provider_cases"] == 1
    assert load_csv(tmp_path / "human_review.csv") == []
    assert len(load_csv(tmp_path / "unresolved_cases.csv")) == 1


def test_last_result_is_current_and_purposive_extremes_are_separate(tmp_path):
    item = case("coop", task="unit_review", stratum="sentinel", city="new-york")
    item["selection_kind"] = "sentinel"
    item["units_raw_nsi"] = 10871
    good = review(item["entity_id"])
    records = [
        result(item, "claude", good),
        {"case_id": item["case_id"], "provider": "gpt", "status": "failed", "error": "timeout"},
        result(item, "gpt", good),
    ]
    report = summarize([item], records, tmp_path / "first")
    assert report["overall"]["paired_completed"] == 1
    assert report["overall"]["human_review_count"] == 1
    assert report["overall"]["disagreement_count"] == 0
    assert report["by_selection_kind"][0]["selection_kind"] == "sentinel"
    assert report["attempt_history"]["failed_attempts_total"] == 1
    assert report["attempt_history"]["failed_attempts_by_provider"] == {"claude": 0, "gpt": 1}
    assert report["attempt_history"]["recovered_failed_cases"] == 1
    records.append(
        {"case_id": item["case_id"], "provider": "gpt", "status": "failed", "error": "new failure"}
    )
    current = summarize([item], records, tmp_path / "later")
    assert current["overall"]["paired_completed"] == 0
    assert current["overall"]["human_review_count"] == 0
    assert current["overall"]["provider_failure_cases"] == 1
    assert current["attempt_history"]["failed_attempts_total"] == 2
    assert current["attempt_history"]["recovered_failed_cases"] == 0
    row = load_csv(tmp_path / "later/cases.csv")[0]
    assert row["units_raw_nsi"] == "10871"
    assert row["any_provider_review_flag"] == "True"


def test_retry_success_keeps_each_invalid_attempt_and_counts_recovered_case_once(tmp_path):
    item = case(1, task="unit_impute", stratum="missing_units")
    records = [
        {
            "case_id": "1",
            "provider": "claude",
            "status": "failed",
            "error": "units_low exceeds units_high",
            "record_path": "/raw/invalid-bounds.json",
        },
        {
            "case_id": "1",
            "provider": "gpt",
            "status": "failed",
            "error": "timeout",
            "record_path": "/raw/timeout.json",
        },
        {
            "case_id": "1",
            "provider": "gpt",
            "status": "failed",
            "error": "second timeout",
            "record_path": "/raw/timeout-2.json",
        },
        result(item, "claude", impute(item["entity_id"], 2, "building")),
        result(item, "gpt", impute(item["entity_id"], 2, "building")),
    ]
    report = summarize([item], records, tmp_path)
    assert report["overall"]["paired_completed"] == 1
    assert report["overall"]["provider_failure_cases"] == 0
    assert report["attempt_history"]["failed_attempts_total"] == 3
    assert report["attempt_history"]["failed_attempts_by_provider"] == {"claude": 1, "gpt": 2}
    assert report["attempt_history"]["recovered_failed_cases"] == 1
    failures = load_csv(tmp_path / "failed_attempts.csv")
    assert len(failures) == 3
    assert failures[0]["record_path"] == "/raw/invalid-bounds.json"
    assert failures[0]["error"] == "units_low exceeds units_high"
    assert failures[2]["provider_attempt_number"] == "2"
    assert all(row["final_provider_status"] == "success" for row in failures)
    assert all(row["final_paired_completed"] == "True" for row in failures)
    assert json.loads(failures[0]["result_json"]) == records[0]


def test_imputed_counts_are_not_averaged_and_complex_totals_stay_unresolved(tmp_path):
    one = case(1, task="unit_impute", stratum="missing_units")
    two = case(2, task="unit_impute", stratum="missing_units")
    records = [
        result(one, "claude", impute(one["entity_id"], 12, "building")),
        result(one, "gpt", impute(one["entity_id"], 15, "building")),
        result(two, "claude", impute(two["entity_id"], 10871, "complex")),
        result(two, "gpt", impute(two["entity_id"], 10871, "complex")),
    ]
    report = summarize([one, two], records, tmp_path)
    assert report["overall"]["disagreement_count"] == 1
    assert report["overall"]["human_review_count"] == 1
    assert report["overall"]["unresolved_count"] == 1
    rows = load_csv(tmp_path / "cases.csv")
    assert json.loads(rows[0]["assessment_json"])["units"] is None
    assert json.loads(rows[1]["assessment_json"])["units"] is None
    assert rows[0]["claude_units"] == "12" and rows[0]["gpt_units"] == "15"
    assert float(rows[0]["relative_count_difference"]) == pytest.approx(2 * 3 / 27)
    assert rows[0]["interval_overlap"] == "False"
    assert float(rows[1]["relative_count_difference"]) == 0
    assert rows[1]["interval_overlap"] == "True"


def test_sample_baseline_fields_and_overlapping_intervals_are_explicit(tmp_path):
    item = case(1, task="unit_impute", stratum="large_nsi_units")
    item.update(
        raw_units_max=10871,
        baseline_units=None,
        baseline_residential_use="unknown",
        nsi_count=1,
        nsi_fd_ids=[72668870],
        area_m2=125.5,
    )
    claude = impute(item["entity_id"], 12, "building")
    claude.update(units_low=10, units_high=16)
    gpt = impute(item["entity_id"], 15, "building")
    gpt.update(units_low=12, units_high=18)
    report = summarize([item], [result(item, "claude", claude), result(item, "gpt", gpt)], tmp_path)
    row = load_csv(tmp_path / "cases.csv")[0]
    assert row["units_raw_nsi"] == "10871" and row["raw_units_max"] == "10871"
    assert row["baseline_units"] == "" and row["baseline_residential_use"] == "unknown"
    assert row["nsi_count"] == "1" and json.loads(row["nsi_fd_ids_json"]) == [72668870]
    assert row["area_m2"] == "125.5"
    assert row["claude_units_low"] == "10" and row["gpt_units_high"] == "18"
    assert row["interval_overlap"] == "True"
    assert float(row["relative_count_difference"]) == pytest.approx(2 * 3 / 27)
    assert report["overall"]["disagreement_count"] == 1
    assert "NSI大户数压力测试" in (tmp_path / "REPORT.zh-CN.md").read_text()


def test_invalid_same_entity_schema_is_incomplete_not_disagreement(tmp_path):
    item = case(1)
    broken = {"entity_id": item["entity_id"], "reason": "Malformed use label"}
    report = summarize(
        [item],
        [
            result(item, "claude", broken),
            result(item, "gpt", purpose(item["entity_id"], "residential")),
        ],
        tmp_path,
    )
    assert report["overall"]["paired_completed"] == 0
    assert report["overall"]["disagreement_count"] == 0
    assert report["overall"]["invalid_output_cases"] == 1
    assert load_csv(tmp_path / "cases.csv")[0]["claude_status"] == "invalid_output"


def test_duplicate_sample_identity_is_not_double_counted(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        summarize([case(1), case(1)], [], tmp_path)


def test_rejected_structured_proposals_are_displayed_but_never_count_as_completed(tmp_path):
    item = case(1, task="unit_impute", stratum="missing_units")
    rejected = impute(item["entity_id"], 1, "building")
    rejected.update(
        units_method="web_extracted",
        evidence=[
            {
                "kind": "osm_tag",
                "source_id": "w1",
                "scope": "building",
                "url": "https://www.openstreetmap.org/way/1",
                "quote": "building:flats=1",
            }
        ],
    )
    claude_path, gpt_path = tmp_path / "claude-failure.json", tmp_path / "gpt-failure.json"
    claude_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": "ValueError",
                "cli_result": {"structured_output": rejected},
            }
        )
    )
    gpt_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": "ValueError",
                "events": [
                    {"item": {"type": "agent_message", "text": json.dumps(rejected)}},
                    {
                        "item": {
                            "type": "agent_message",
                            "text": "Non-JSON commentary after proposal",
                        }
                    },
                ],
            }
        )
    )
    records = [
        {
            "case_id": item["case_id"],
            "provider": provider,
            "status": "failed",
            "error": "web_extracted needs an actual web evidence source",
            "record_path": str(path),
        }
        for provider, path in (("claude", claude_path), ("gpt", gpt_path))
    ]
    report = summarize([item], records, tmp_path / "report")
    row = load_csv(tmp_path / "report/cases.csv")[0]
    for provider in ("claude", "gpt"):
        assert row[provider + "_status"] == "failed"
        assert row[provider + "_units"] == ""
        assert row[provider + "_failed_error_type"] == "ValueError"
        assert json.loads(row[provider + "_failed_raw_output_json"]) == rejected
    assert report["overall"]["paired_completed"] == 0
    assert report["overall"]["disagreement_count"] == 0
    assert report["overall"]["human_review_count"] == 0
    failures = load_csv(tmp_path / "report/failed_attempts.csv")
    assert all(json.loads(row["failed_raw_output_json"]) == rejected for row in failures)
    markdown = (tmp_path / "report/REPORT.zh-CN.md").read_text()
    assert "无效回答；原提议1户；web_extracted；证据osm_tag" in markdown


def test_shared_review_decision_with_scope_conflict_keeps_strict_disagreement_separate(tmp_path):
    one = case(1, task="unit_review", stratum="large_nsi_units")
    two = case(2)
    incomplete = case(3, task="unit_review", stratum="large_nsi_units")
    records = [
        result(one, "claude", review(one["entity_id"], flag=True, scope="complex")),
        result(one, "gpt", review(one["entity_id"], flag=True, scope="unresolved")),
        result(two, "claude", purpose(two["entity_id"], "residential")),
        result(two, "gpt", purpose(two["entity_id"], "nonresidential")),
        result(incomplete, "claude", review(incomplete["entity_id"])),
    ]
    report = summarize([one, two, incomplete], records, tmp_path)
    total = report["overall"]
    assert total["disagreement_count"] == 2
    assert total["human_review_count"] == 2
    assert total["unit_review_paired"] == 1
    assert total["review_decision_disagreement_count"] == 0
    assert total["review_decision_disagreement_rate"] == 0
    rows = load_csv(tmp_path / "cases.csv")
    assert rows[0]["disagreement"] == "True" and rows[0]["needs_human_review"] == "True"
    assert rows[0]["review_decision_disagreement"] == "False"
    assert rows[1]["review_decision_disagreement"] == ""
    assert rows[2]["review_decision_disagreement"] == ""
    group = next(row for row in report["by_stratum"] if row["stratum"] == "large_nsi_units")
    assert group["disagreement_count"] == 1 and group["review_decision_disagreement_count"] == 0


def test_provider_quota_failures_are_not_reported_as_judgment_errors(tmp_path):
    cases = [case(i, task="unit_impute", stratum="missing_units") for i in range(6)]
    errors = [
        "You're out of extra usage · resets 8:20pm (America/Los_Angeles)",
        "Selected model is at capacity. Please try a different model.",
        "Command timed out after 300 seconds",
        "Web-extracted units require cited building-level web evidence",
        "Connection unexpectedly closed",
        "Model-imputed units require an uncertainty range",
    ]
    records = []
    for item, error in zip(cases, errors):
        records.append(
            {"case_id": item["case_id"], "provider": "claude", "status": "failed", "error": error}
        )
        records.append(result(item, "gpt", impute(item["entity_id"], 1, "building")))
    report = summarize(cases, records, tmp_path / "blocked")
    assert report["status"] == "partial_provider_blocked"
    assert report["overall"]["paired_completed"] == 0
    assert report["overall"]["disagreement_count"] == 0
    assert report["attempt_history"]["failed_attempts_by_kind"] == {
        "quota_or_rate_limit": 1,
        "capacity": 1,
        "timeout": 1,
        "rejected_output": 2,
        "other": 1,
    }
    failures = load_csv(tmp_path / "blocked/failed_attempts.csv")
    assert [row["failure_kind"] for row in failures] == [
        "quota_or_rate_limit",
        "capacity",
        "timeout",
        "rejected_output",
        "other",
        "rejected_output",
    ]
    markdown = (tmp_path / "blocked/REPORT.zh-CN.md").read_text()
    assert "部分完成" in markdown and "claude有1个样本因额度或速率限制尚缺有效回答" in markdown
    assert "不作全11城结论" in markdown and "不是模型判断错误数" in markdown
    records.append(result(cases[0], "claude", impute(cases[0]["entity_id"], 1, "building")))
    recovered = summarize(cases, records, tmp_path / "unblocked")
    assert recovered["status"] == "partial_incomplete"
    assert recovered["attempt_history"]["failed_attempts_by_kind"]["quota_or_rate_limit"] == 1
    assert recovered["overall"]["paired_completed"] == 1


def test_stopped_report_preserves_each_provider_judgment_without_requiring_a_pair(tmp_path):
    purpose_case = case(1)
    missing = case(2, task="unit_impute", stratum="missing_units")
    large = case(3, task="unit_review", stratum="large_nsi_units", city="new-york")
    records = [
        {
            "case_id": "1",
            "provider": "claude",
            "status": "failed",
            "error": "You're out of extra usage",
        },
        result(purpose_case, "gpt", purpose(purpose_case["entity_id"], "residential")),
        result(missing, "claude", impute(missing["entity_id"])),
        result(missing, "gpt", impute(missing["entity_id"], 12, "building")),
        result(large, "claude", review(large["entity_id"], flag=False, scope="building")),
        result(large, "gpt", review(large["entity_id"], flag=True, scope="complex")),
    ]
    report = summarize(
        [purpose_case, missing, large], records, tmp_path, execution_status="stopped_by_user"
    )
    assert report["execution_status"] == "stopped_by_user"
    assert report["status"] == "partial_available"
    assert report["underlying_provider_status"] == "partial_provider_blocked"
    assert report["overall"]["paired_completed"] == 2
    providers = {row["provider"]: row for row in report["by_provider"]}
    assert providers["claude"]["selected_cases"] == 3
    assert providers["claude"]["valid_outputs"] == 2
    assert providers["claude"]["use_unknown"] == 0
    assert providers["claude"]["final_failed_quota_or_rate_limit"] == 1
    assert providers["claude"]["unit_impute_unresolved"] == 1
    assert providers["claude"]["unit_review_not_flagged"] == 1
    assert providers["gpt"]["valid_outputs"] == 3
    assert providers["gpt"]["use_residential"] == 1
    assert providers["gpt"]["unit_impute_count_supplied"] == 1
    assert providers["gpt"]["unit_impute_method_model_imputed"] == 1
    assert providers["gpt"]["unit_review_flagged"] == 1
    assert providers["gpt"]["unit_review_scope_complex"] == 1
    assert len(load_csv(tmp_path / "provider_summary.csv")) == 2
    assert len(load_csv(tmp_path / "city_provider_summary.csv")) == 4
    assert len(load_csv(tmp_path / "stratum_provider_summary.csv")) == 6
    markdown = (tmp_path / "REPORT.zh-CN.md").read_text()
    assert "用户已暂停" in markdown and "不再调用模型" in markdown
    assert "各模型已有判断" in markdown
