"""Describe a fixed Claude/GPT sample without inferring population accuracy.

This reporter only reads recorded judgments. It does not invoke a model, apply an
annotation or modify the source customer population.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import csv
from datetime import UTC, datetime
import json
from pathlib import Path

from .dual_review import reconcile_assessments

PROVIDERS = ("claude", "gpt")
FAILURE_KINDS = {
    "quota_or_rate_limit": "额度或速率限制",
    "capacity": "模型容量不足",
    "timeout": "超时",
    "rejected_output": "输出未通过约定",
    "other": "其他",
}
STRATA = {
    "unknown_use": "用途未知",
    "missing_units": "户数缺失",
    "large_units": "大户数",
    "large_nsi_units": "NSI大户数压力测试",
    "extreme_units": "异常大户数",
    "sentinel": "指定难例",
    "co_op_city": "Co-op City指定难例",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_results(results, identities):
    latest, ignored = {}, 0
    for result in results:
        key = (result.get("case_id"), result.get("provider"))
        if key[0] not in identities or key[1] not in PROVIDERS:
            ignored += 1
            continue
        latest[key] = result
    return latest, ignored


def _rejected_proposal(result):
    """Expose a failed provider's proposal for inspection, never for adoption."""
    error_type = result.get("error_type")
    path = result.get("record_path")
    if not path:
        return None, error_type
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None, error_type
    if not isinstance(receipt, Mapping):
        return None, error_type
    error_type = receipt.get("error_type", error_type)
    cli_result = receipt.get("cli_result", {})
    if isinstance(cli_result, Mapping) and isinstance(cli_result.get("structured_output"), Mapping):
        return cli_result["structured_output"], error_type
    for event in reversed(receipt.get("events", [])):
        item = event.get("item", {})
        if item.get("type") != "agent_message":
            continue
        try:
            output = json.loads(item.get("text", ""))
        except (TypeError, ValueError):
            continue
        if isinstance(output, Mapping):
            return output, error_type
    return None, error_type


def _failure_kind(error, error_type=None):
    message = str(error or "").lower().replace("_", " ").replace("-", " ")
    compact = message.replace(" ", "")
    if any(
        token in compact
        for token in ("outofextrausage", "hitlimit", "quota", "ratelimit", "usagelimit")
    ):
        return "quota_or_rate_limit"
    if "atcapacity" in compact:
        return "capacity"
    if (
        "timedout" in compact
        or "timeoutexpired" in str(error_type or "").lower()
        or "timeout" in compact
    ):
        return "timeout"
    validation = (
        "unit estimate",
        "unit interval",
        "model imputed",
        "uncertainty range",
        "units low",
        "units high",
        "web extracted",
        "web evidence",
        "building scope",
        "structured",
        "annotation",
        "entity id",
        "supporting evidence",
        "required fields",
        "json",
        "invalid",
        "public web",
    )
    if any(token in message for token in validation):
        return "rejected_output"
    return "other"


def _case_row(case, latest):
    judgments, provider_rows = {}, {}
    row = {
        "case_id": case["case_id"],
        "city": case["city"],
        "stratum": case["stratum"],
        "selection_kind": case.get("selection_kind", "not_recorded"),
        "task": case["task"],
        "entity_id": case["entity_id"],
        "longitude": case.get("longitude"),
        "latitude": case.get("latitude"),
        "units_raw_nsi": next(
            (
                case.get(key)
                for key in ("units_raw_nsi", "raw_resunits", "raw_units_max")
                if case.get(key) is not None
            ),
            None,
        ),
        "raw_units_max": case.get("raw_units_max"),
        "baseline_units": case.get("baseline_units"),
        "baseline_residential_use": case.get("baseline_residential_use"),
        "nsi_count": case.get("nsi_count"),
        "nsi_fd_ids_json": _json(case.get("nsi_fd_ids", [])),
        "area_m2": case.get("area_m2"),
        "case_json": _json(case),
    }
    for name in PROVIDERS:
        result = latest.get((case["case_id"], name))
        reply = result.get("reply", {}) if result else {}
        output = reply.get("output", {})
        status = result.get("status", "failed") if result else "missing"
        if status == "success" and isinstance(output, Mapping):
            if output.get("entity_id") == case["entity_id"]:
                judgments[name] = output
            else:
                status = "invalid_entity"
        elif status == "success":
            status = "invalid_output"
        rejected, failed_error_type = (
            _rejected_proposal(result) if status == "failed" else (None, None)
        )
        provider_rows[name] = (status, output if isinstance(output, Mapping) else {}, reply)
        details = {
            "status": status,
            "model": reply.get("model", result.get("model") if result else None),
            "use": output.get("ai_inferred_use") if isinstance(output, Mapping) else None,
            "units": output.get("units") if isinstance(output, Mapping) else None,
            "units_low": output.get("units_low") if isinstance(output, Mapping) else None,
            "units_high": output.get("units_high") if isinstance(output, Mapping) else None,
            "recommended_units": output.get("recommended_units")
            if isinstance(output, Mapping)
            else None,
            "units_scope": output.get("units_scope") if isinstance(output, Mapping) else None,
            "units_method": output.get("units_method") if isinstance(output, Mapping) else None,
            "review_required": output.get("review_required")
            if isinstance(output, Mapping)
            else None,
            "reason": output.get("reason") if isinstance(output, Mapping) else None,
            "issues_json": _json(output.get("issues", [])) if isinstance(output, Mapping) else "[]",
            "evidence_json": _json(output.get("evidence", []))
            if isinstance(output, Mapping)
            else "[]",
            "output_json": _json(output),
            "request_id": reply.get("request_id"),
            "request_path": reply.get("record_path", result.get("record_path") if result else None),
            "error": result.get("error") if result else None,
            "failed_raw_output_json": _json(rejected) if rejected is not None else None,
            "failed_error_type": failed_error_type,
            "failure_kind": _failure_kind(result.get("error"), failed_error_type)
            if status == "failed"
            else None,
        }
        row.update({name + "_" + key: value for key, value in details.items()})
    reconciliation = reconcile_assessments(case["task"], judgments)
    incomplete = set(reconciliation.get("incomplete_providers", []))
    for name in incomplete:
        if row[name + "_status"] == "success":
            row[name + "_status"] = "invalid_output"
    assessment = reconciliation.get("assessment") or {}
    row.update(
        paired_completed=reconciliation["status"] != "incomplete",
        reconciliation_status=reconciliation["status"],
        disagreement=reconciliation["status"] == "disagreement",
        needs_human_review=bool(reconciliation["needs_human_review"]),
        unresolved=reconciliation["status"] == "unresolved",
        assessment_json=_json(assessment),
        missing_provider_count=sum(row[name + "_status"] == "missing" for name in PROVIDERS),
        failed_provider_count=sum(row[name + "_status"] == "failed" for name in PROVIDERS),
        invalid_provider_count=sum(
            str(row[name + "_status"]).startswith("invalid") for name in PROVIDERS
        ),
        any_provider_review_flag=any(
            output.get("review_required") is True for _, output, _ in provider_rows.values()
        ),
        relative_count_difference=None,
        interval_overlap=None,
        review_decision_disagreement=None,
    )
    if case["task"] == "unit_review" and row["paired_completed"]:
        row["review_decision_disagreement"] = (
            row["claude_review_required"] != row["gpt_review_required"]
        )
    if case["task"] == "unit_impute" and row["paired_completed"]:
        counts = [row[name + "_units"] for name in PROVIDERS]
        if all(value is not None for value in counts):
            row["relative_count_difference"] = 2 * abs(counts[0] - counts[1]) / sum(counts)
        bounds = [
            row[name + bound] for name in PROVIDERS for bound in ("_units_low", "_units_high")
        ]
        if all(value is not None for value in bounds):
            row["interval_overlap"] = max(bounds[0], bounds[2]) <= min(bounds[1], bounds[3])
    return row, {"case_id": case["case_id"], **reconciliation}


def _aggregate(rows, dimensions=()):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in dimensions)
        groups.setdefault(key, []).append(row)
    if not dimensions and not rows:
        groups[()] = []
    aggregates = []
    for key, group in groups.items():
        paired = sum(row["paired_completed"] for row in group)
        disagreement = sum(row["disagreement"] for row in group)
        human = [row for row in group if row["needs_human_review"]]
        unit_review_paired = sum(
            row["task"] == "unit_review" and row["paired_completed"] for row in group
        )
        decision_disagreements = sum(row["review_decision_disagreement"] is True for row in group)
        item = dict(zip(dimensions, key))
        item.update(
            sample_cases=len(group),
            sample_locations=len({(row["city"], row["entity_id"]) for row in group}),
            paired_completed=paired,
            incomplete=len(group) - paired,
            agreed=sum(row["reconciliation_status"] == "agreed" for row in group),
            disagreement_count=disagreement,
            disagreement_rate=disagreement / paired if paired else None,
            human_review_count=len(human),
            human_review_locations=len({(row["city"], row["entity_id"]) for row in human}),
            human_review_rate=sum(row["needs_human_review"] for row in group) / paired
            if paired
            else None,
            unresolved_count=sum(row["unresolved"] for row in group),
            provider_failure_cases=sum(row["failed_provider_count"] > 0 for row in group),
            missing_provider_cases=sum(row["missing_provider_count"] > 0 for row in group),
            invalid_output_cases=sum(row["invalid_provider_count"] > 0 for row in group),
            unit_review_paired=unit_review_paired,
            review_decision_disagreement_count=decision_disagreements,
            review_decision_disagreement_rate=decision_disagreements / unit_review_paired
            if unit_review_paired
            else None,
        )
        for provider in PROVIDERS:
            for field in ("use", "units_scope", "units_method"):
                counts = Counter(
                    str(row[provider + "_" + field])
                    for row in group
                    if row[provider + "_status"] == "success"
                    and row[provider + "_" + field] is not None
                )
                item[provider + "_" + field + "_counts_json"] = _json(dict(counts))
            item[provider + "_review_flags"] = sum(
                row[provider + "_review_required"] is True for row in group
            )
        aggregates.append(item)
    return aggregates


def _attempt_history(cases, results, rows):
    lookup = {case["case_id"]: case for case in cases}
    final = {row["case_id"]: row for row in rows}
    failures, failed_by_provider, failed_by_kind, attempts = [], Counter(), Counter(), Counter()
    for index, result in enumerate(results, start=1):
        identity, provider = result.get("case_id"), result.get("provider")
        if identity not in lookup or provider not in PROVIDERS:
            continue
        attempts[identity, provider] += 1
        if result.get("status") != "failed":
            continue
        case, current = lookup[identity], final[identity]
        rejected, error_type = _rejected_proposal(result)
        failure_kind = _failure_kind(result.get("error"), error_type)
        failed_by_provider[provider] += 1
        failed_by_kind[failure_kind] += 1
        failures.append(
            {
                "history_index": index,
                "provider_attempt_number": attempts[identity, provider],
                "case_id": identity,
                "city": case["city"],
                "stratum": case["stratum"],
                "task": case["task"],
                "entity_id": case["entity_id"],
                "provider": provider,
                "error": result.get("error"),
                "failed_error_type": error_type,
                "failure_kind": failure_kind,
                "failed_raw_output_json": _json(rejected) if rejected is not None else None,
                "record_path": result.get("record_path"),
                "completed_at_utc": result.get("completed_at_utc"),
                "final_provider_status": current[provider + "_status"],
                "final_paired_completed": current["paired_completed"],
                "result_json": _json(result),
            }
        )
    recovered = {row["case_id"] for row in failures if row["final_paired_completed"]}
    return {
        "failed_attempts_total": len(failures),
        "failed_attempts_by_provider": {name: failed_by_provider[name] for name in PROVIDERS},
        "failed_attempts_by_kind": {name: failed_by_kind[name] for name in FAILURE_KINDS},
        "recovered_failed_cases": len(recovered),
        "recovered_case_definition": "sample cases with a historical failed attempt and final valid paired results",
    }, failures


def _provider_aggregate(rows, dimensions=()):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in dimensions), []).append(row)
    summaries = []
    for key, group in groups.items():
        for name in PROVIDERS:
            valid = [row for row in group if row[name + "_status"] == "success"]
            item = {
                **dict(zip(dimensions, key)),
                "provider": name,
                "selected_cases": len(group),
                "valid_outputs": len(valid),
                "final_failed_outputs": sum(row[name + "_status"] == "failed" for row in group),
                "missing_outputs": sum(row[name + "_status"] == "missing" for row in group),
                "invalid_outputs": sum(
                    str(row[name + "_status"]).startswith("invalid") for row in group
                ),
            }
            for task in ("residential_use", "unit_impute", "unit_review"):
                item[task + "_selected"] = sum(row["task"] == task for row in group)
                item[task + "_valid"] = sum(row["task"] == task for row in valid)
            for use in ("residential", "nonresidential", "mixed", "unknown"):
                item["use_" + use] = sum(
                    row[name + "_use"] == use for row in valid if row["task"] == "residential_use"
                )
            reviews = [row for row in valid if row["task"] == "unit_review"]
            imputed = [row for row in valid if row["task"] == "unit_impute"]
            item["unit_review_flagged"] = sum(
                row[name + "_review_required"] is True for row in reviews
            )
            item["unit_review_not_flagged"] = sum(
                row[name + "_review_required"] is False for row in reviews
            )
            item["unit_impute_count_supplied"] = sum(
                row[name + "_units"] is not None for row in imputed
            )
            item["unit_impute_unresolved"] = sum(row[name + "_units"] is None for row in imputed)
            for method in ("web_extracted", "model_imputed", "unresolved"):
                item["unit_impute_method_" + method] = sum(
                    row[name + "_units_method"] == method for row in imputed
                )
            for task, subset in (("unit_review", reviews), ("unit_impute", imputed)):
                for scope in ("building", "complex", "unresolved"):
                    item[task + "_scope_" + scope] = sum(
                        row[name + "_units_scope"] == scope for row in subset
                    )
            for kind in FAILURE_KINDS:
                item["final_failed_" + kind] = sum(
                    row[name + "_failure_kind"] == kind for row in group
                )
            summaries.append(item)
    return summaries


def _percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _verdict(row, provider):
    status = row[provider + "_status"]
    if status != "success":
        if row[provider + "_failed_raw_output_json"]:
            rejected = json.loads(row[provider + "_failed_raw_output_json"])
            proposal = rejected.get("units", rejected.get("recommended_units"))
            if proposal is not None:
                proposal = f"原提议{proposal}户"
            else:
                proposal = "原判断" + str(rejected.get("ai_inferred_use", "未确定"))
            kinds = list(
                dict.fromkeys(
                    str(item.get("kind", "未标记"))
                    for item in rejected.get("evidence", [])
                    if isinstance(item, Mapping)
                )
            )
            return "；".join(
                str(x)
                for x in [
                    "无效回答",
                    proposal,
                    rejected.get("units_method"),
                    "证据" + ",".join(kinds) if kinds else None,
                    str(row[provider + "_error"] or row[provider + "_failed_error_type"] or "")[
                        :100
                    ],
                ]
                if x
            )
        if status == "failed" and row[provider + "_error"]:
            return "失败；" + str(row[provider + "_error"])[:100]
        return status
    if row["task"] == "residential_use":
        return row[provider + "_use"] or "unknown"
    if row["task"] == "unit_review":
        count = row[provider + "_recommended_units"]
        label = "需复核" if row[provider + "_review_required"] else "无需复核"
        return f"{label}；建议{count if count is not None else '未给出'}；{row[provider + '_units_scope']}"
    return f"{row[provider + '_units'] if row[provider + '_units'] is not None else '未确定'}；{row[provider + '_units_scope']}；{row[provider + '_units_method']}"


def _markdown(summary, rows):
    total = summary["overall"]
    history = summary["attempt_history"]
    blocked = {
        name: sum(row[name + "_failure_kind"] == "quota_or_rate_limit" for row in rows)
        for name in PROVIDERS
    }
    partial = total["incomplete"] > 0
    operational_note = "；".join(
        f"{name}有{count}个样本因额度或速率限制尚缺有效回答"
        for name, count in blocked.items()
        if count
    )
    partial_note = f"当前固定样本尚未全部完成（{summary['underlying_provider_status']}）。"
    if operational_note:
        partial_note += operational_note + "。额度限制不表示这些建筑因数据或来源困难而无法判断。"
    partial_note += "当前不作全11城结论。"
    lines = [
        "# 11城Claude＋GPT抽样评审" + ("（部分完成）" if partial else ""),
        "",
        "用户已暂停本轮评审（stopped_by_user）。以下仅汇报已有判断，不再调用模型。"
        if summary.get("execution_status") == "stopped_by_user"
        else "",
        "",
        partial_note if partial else "当前固定样本的双模型结果已齐全。",
        "",
        f"本轮固定{total['sample_cases']}个抽样项，涉及{total['sample_locations']}个地点。"
        f"双模型有效结果齐全{total['paired_completed']}项，"
        f"分歧{total['disagreement_count']}项（{_percent(total['disagreement_rate'])}），"
        f"需要人工复核{total['human_review_count']}项、{total['human_review_locations']}个地点。",
        f"历史失败尝试{history['failed_attempts_total']}次"
        f"（Claude {history['failed_attempts_by_provider']['claude']}次、"
        f"GPT {history['failed_attempts_by_provider']['gpt']}次）；"
        f"其中{history['recovered_failed_cases']}个样本随后已补齐有效双模型结果。"
        "历史失败不会因重试成功而从记录中删除。",
        "失败尝试分类："
        + "、".join(
            f"{FAILURE_KINDS[kind]}{count}次"
            for kind, count in history["failed_attempts_by_kind"].items()
        )
        + "。失败尝试总数不是模型判断错误数。",
        "失败回答可能包含字段或证据类型不符合约定的原始提议；其JSON与错误类型单独保留，"
        "不计为有效判断、不进入一致性归并。",
        "",
        "分歧率＝分歧项数÷双模型有效结果齐全项数。运行失败、缺少回答或无效输出单列，"
        "不计作分歧；双方都无法判断的结果计作未解决。双方一致但均认为需复核的记录仍进入人工队列，"
        "因此人工复核数不等于分歧数。",
        "",
        "这是对固定分层样本的描述，极端值和指定难例为定向选择；合并比例不能外推为11城总体错误率、"
        "住宅比例或全量人工工作量。模型一致也不表示人工核实或户数真值。",
        "户数补全另记录连续差异：relative_count_difference＝|Claude−GPT|÷两者均值；"
        "interval_overlap表示双方闭区间是否相交。缺少数值或区间时留空，"
        "这两项只作描述，不改变一致性或作用范围的判定。",
        "",
        "## 各模型已有判断",
        "",
        "下表分别统计每个模型已通过输出约定的回答，不要求另一个模型已完成。"
        "R/N/M/U分别代表住宅、非住宅、混合和未知；户数给出值仍保留building/complex范围与来源，"
        "不代表独立核实。缺少回答不计为未知用途。",
        "",
        "| 模型 | 选定项 | 有效回答 | 用途R/N/M/U | 补全给数/未定 | 户数复核需/不需 | 当前失败 |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for item in summary["by_provider"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in [
                    item["provider"],
                    item["selected_cases"],
                    item["valid_outputs"],
                    "/".join(
                        str(item["use_" + label])
                        for label in ("residential", "nonresidential", "mixed", "unknown")
                    ),
                    f"{item['unit_impute_count_supplied']}/{item['unit_impute_unresolved']}",
                    f"{item['unit_review_flagged']}/{item['unit_review_not_flagged']}",
                    item["final_failed_outputs"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "各模型按城市和样本类别的完整统计分别见[city_provider_summary.csv](city_provider_summary.csv)"
            "和[stratum_provider_summary.csv](stratum_provider_summary.csv)；"
            "[provider_summary.csv](provider_summary.csv)同时列出补全方法、户数范围及最终失败原因。",
            "",
            "## 按城市",
            "",
            "| 城市 | 抽样项 | 双模型齐全 | 分歧 | 分歧率 | 人工复核 | 未解决 | 未完成 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["by_city"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in [
                    item["city"],
                    item["sample_cases"],
                    item["paired_completed"],
                    item["disagreement_count"],
                    _percent(item["disagreement_rate"]),
                    item["human_review_count"],
                    item["unresolved_count"],
                    item["incomplete"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 按样本类别",
            "",
            "| 类别 | 抽样项 | 双模型齐全 | 分歧 | 分歧率 | 人工复核 | 未解决 | 未完成 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["by_stratum"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in [
                    STRATA.get(item["stratum"], item["stratum"]),
                    item["sample_cases"],
                    item["paired_completed"],
                    item["disagreement_count"],
                    _percent(item["disagreement_rate"]),
                    item["human_review_count"],
                    item["unresolved_count"],
                    item["incomplete"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"户数复核有效配对{total['unit_review_paired']}项，"
            f"`review_required`决策分歧{total['review_decision_disagreement_count']}项"
            f"（{_percent(total['review_decision_disagreement_rate'])}，分母仅为有效unit_review配对）。"
            "大户数组的严格分歧还包括户数或作用范围冲突；双方都要求复核、"
            "但分别判断为complex和unresolved时，严格分歧计1，复核决策分歧计0。",
            "",
            "## 逐条结论",
            "",
            "| 样本 | 城市 | 类别 | Claude | GPT | 归并 | 人工复核 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in [
                    row["case_id"],
                    row["city"],
                    STRATA.get(row["stratum"], row["stratum"]),
                    _verdict(row, "claude"),
                    _verdict(row, "gpt"),
                    row["reconciliation_status"],
                    "是" if row["needs_human_review"] else "否",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "全部判断理由、原始户数、证据、模型和请求路径见[cases.csv](cases.csv)。"
            "[human_review.csv](human_review.csv)列出已确定需人工复核的记录；"
            "[unresolved_cases.csv](unresolved_cases.csv)列出双方未解决或尚缺有效结果的记录。"
            "[city_stratum_summary.csv](city_stratum_summary.csv)给出每城每类统计。"
            "[failed_attempts.csv](failed_attempts.csv)保留全部失败尝试及原始记录路径，"
            "与最终是否成功分别报告。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(cases, results, output_dir, *, execution_status=None):
    """Write one reproducible report from fixed cases and recorded result envelopes.

    A resumed runner can append results; the last record for a case/provider is
    its current result. No stale former success hides a later failed attempt.
    ``case_id`` identifies one task on one building, so location counts are also
    reported when multiple tasks refer to the same building.
    """
    cases = list(cases)
    results = list(results)
    identities = [case["case_id"] for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Sample case_id values must be unique")
    latest, ignored = _latest_results(results, set(identities))
    rows, reconciliations = [], []
    for case in cases:
        row, reconciliation = _case_row(case, latest)
        rows.append(row)
        reconciliations.append(reconciliation)
    history, failures = _attempt_history(cases, results, rows)
    summary = {
        "schema": "evrptw_dual_sample_report_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "required_providers": list(PROVIDERS),
        "disagreement_denominator": "paired valid completed sample items",
        "population_inference": False,
        "ai_agreement_is_human_verification": False,
        "overall": _aggregate(rows)[0],
        "by_city": _aggregate(rows, ("city",)),
        "by_stratum": _aggregate(rows, ("stratum",)),
        "by_city_stratum": _aggregate(rows, ("city", "stratum")),
        "by_selection_kind": _aggregate(rows, ("selection_kind",)),
        "ignored_result_records": ignored,
        "attempt_history": history,
        "execution_status": execution_status,
        "by_provider": _provider_aggregate(rows),
        "by_city_provider": _provider_aggregate(rows, ("city",)),
        "by_stratum_provider": _provider_aggregate(rows, ("stratum",)),
    }
    underlying_status = (
        "partial_provider_blocked"
        if any(
            row[name + "_failure_kind"] == "quota_or_rate_limit"
            for row in rows
            for name in PROVIDERS
        )
        else "partial_incomplete"
        if summary["overall"]["incomplete"]
        else "complete"
    )
    summary["underlying_provider_status"] = underlying_status
    summary["status"] = (
        "partial_available"
        if execution_status == "stopped_by_user" and underlying_status != "complete"
        else underlying_status
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["case_id", "city", "stratum", "task", "entity_id"]
    _write_csv(target / "cases.csv", rows, columns)
    _write_csv(
        target / "human_review.csv", [row for row in rows if row["needs_human_review"]], columns
    )
    _write_csv(
        target / "unresolved_cases.csv",
        [row for row in rows if row["unresolved"] or not row["paired_completed"]],
        columns,
    )
    _write_csv(
        target / "failed_attempts.csv",
        failures,
        [
            "history_index",
            "provider_attempt_number",
            "case_id",
            "city",
            "stratum",
            "task",
            "entity_id",
            "provider",
            "error",
            "failed_error_type",
            "failure_kind",
            "failed_raw_output_json",
            "record_path",
            "completed_at_utc",
            "final_provider_status",
            "final_paired_completed",
            "result_json",
        ],
    )
    for name, key in [
        ("city_summary.csv", "by_city"),
        ("stratum_summary.csv", "by_stratum"),
        ("city_stratum_summary.csv", "by_city_stratum"),
        ("selection_summary.csv", "by_selection_kind"),
        ("provider_summary.csv", "by_provider"),
        ("city_provider_summary.csv", "by_city_provider"),
        ("stratum_provider_summary.csv", "by_stratum_provider"),
    ]:
        _write_csv(target / name, summary[key])
    with (target / "reconciliations.jsonl").open("w") as stream:
        for item in reconciliations:
            stream.write(_json(item) + "\n")
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (target / "REPORT.zh-CN.md").write_text(_markdown(summary, rows))
    return summary


def _load(path, key):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return value[key] if isinstance(value, dict) else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(_load(args.manifest, "cases"), _load(args.results, "results"), args.output)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
