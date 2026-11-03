"""Building-first Stage 1, with recorded AI proposals and human unit decisions.

This entrypoint leaves the historical project workflow and frozen receipts intact.
Geographic construction and annotation can be resumed independently.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .ai_provider import AIProviderConfig, AIProviderError, ClaudeCLIProvider
from .customer_enrichment import (
    apply_purpose_assessment,
    apply_review_decision,
    apply_unit_assessment,
    evaluate_unit_predictions,
    initialize_record,
    mask_units_for_evaluation,
    select_reference_examples,
)
from .util import sha256_file, write_json

SCHEMA = "evrptw_building_first_stage1_v1"
AXIS_R = 0.2853432473555575
FINAL_FIELDS = (
    "residential_use",
    "residential_source",
    "units_final",
    "units_scope",
    "units_method",
    "units_review_status",
    "units_independently_verified",
    "purpose_review_status",
    "units_low",
    "units_high",
    "annotation_evidence_id",
    "dual_review_status",
    "ai_providers_json",
)
PROFILES = ("u1", "u_lt50", "u_lt100", "unrestricted")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_project(config):
    path = Path(config).resolve()
    settings = read_json(path)
    if settings.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}")
    settings["config_path"] = str(path)
    settings.setdefault("ai", {}).setdefault("require_dual_review", True)
    settings["output"] = str(resolve(path.parent, settings["output"]))
    for city, sources in settings["cities"].items():
        sources.setdefault("snapshots", settings.get("source_snapshots", {}))
        for key in ("microsoft", "osm", "nsi", "service_boundary", "admin_boundary"):
            if sources.get(key):
                sources[key] = str(resolve(path.parent, sources[key]))
        if sources.get("geometry_cache"):
            sources["geometry_cache"] = str(resolve(path.parent, sources["geometry_cache"]))
    for key in ("inventory", "source_root", "road_sources"):
        if settings.get("components", {}).get(key):
            settings["components"][key] = str(resolve(path.parent, settings["components"][key]))
    component_inputs = settings.get("components", {})
    inventory_path = component_inputs.get("inventory")
    if inventory_path and Path(inventory_path).is_file() and component_inputs.get("source_root"):
        from .source_preparation import source_path

        inventory = read_json(inventory_path)
        for city, sources in settings["cities"].items():
            for name in ("service_boundary", "admin_boundary"):
                key = inventory.get("cities", {}).get(city, {}).get("boundaries", {}).get(name)
                if not sources.get(name) and key:
                    sources[name] = str(
                        source_path(inventory, component_inputs["source_root"], key)
                    )
    return settings


def initialize(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "stage1.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(
        path,
        {
            "schema": SCHEMA,
            "output": "outputs/stage1-building-first",
            "axis_r": AXIS_R,
            "cities": {
                "dallas": {
                    "microsoft": "inputs/customer/dallas/microsoft_footprints.parquet",
                    "osm": "inputs/customer/dallas/osm_buildings.parquet",
                    "nsi": "inputs/customer/dallas/nsi.parquet",
                    "area_crs": "EPSG:32614",
                    "service_boundary": "inputs/boundaries/dallas/service_boundary.geojson",
                    "admin_boundary": "inputs/boundaries/dallas/admin_boundary.geojson",
                }
            },
            "components": {
                "inventory": "inputs/inventory.json",
                "source_root": "inputs",
                "road_sources": "inputs/roads",
            },
            "ai": {
                "command": "claude",
                "gpt_command": "codex",
                "gpt_model": None,
                "require_dual_review": True,
                "allow_web": True,
                "timeout_seconds": 300,
                "batch_limit": 10,
                "reference_count": 8,
            },
            "customer": {"active_profile": "u_lt100", "include_model_imputed": True},
            "source_snapshots": {
                "microsoft_index": "2026-08-13",
                "osm": "2026-10-25",
                "nsi_archive": "2026-10-26",
            },
        },
    )
    return path


def city_dir(project, city):
    return Path(project["output"]) / "cities" / city


def selected_cities(project, selected=None):
    cities = list(selected or project["cities"])
    for city in cities:
        if city not in project["cities"]:
            raise ValueError(f"City absent from project: {city}")
    return cities


def overrides(base):
    path = base / "customer_inputs/assessments.jsonl"
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                result[record["entity_id"]] = record
    return result


def save_override(base, record):
    path = base / "customer_inputs/assessments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + "\n")


def population(base):
    frame = pd.read_parquet(base / "customer_inputs/population.parquet")
    scope_path = base / "customer_inputs/scope.parquet"
    if scope_path.exists():
        frame = frame.merge(
            pd.read_parquet(scope_path), on="entity_id", how="left", validate="one_to_one"
        )
        if frame.service_domain_eligible.isna().any():
            raise ValueError("Service scope does not cover every prepared building")
    else:
        frame["service_domain_eligible"] = True
    for field in FINAL_FIELDS:
        if field not in frame:
            frame[field] = None
    # Sparse enrichment records do not require rewriting millions of unmodified rows.
    lookup = overrides(base)
    if lookup:
        positions = pd.Series(frame.index.to_numpy(), index=frame.entity_id)
        for entity, state in lookup.items():
            if entity not in positions:
                raise ValueError(f"Assessment building absent from population: {entity}")
            index = positions[entity]
            for field in FINAL_FIELDS:
                if field in state:
                    frame.at[index, field] = state[field]
            frame.at[index, "annotation_evidence_id"] = entity
    return frame, lookup


def record_for(row, prior):
    if row.entity_id in prior:
        return deepcopy(prior[row.entity_id])
    return initialize_record(clean(row.to_dict()))


def prepare(project, cities):
    from .building_inputs import prepare_building_population

    rows = []
    for city in cities:
        entry = project["cities"][city]
        result = prepare_building_population(
            city=city,
            sources=entry,
            output_dir=city_dir(project, city) / "customer_inputs",
            axis_r=float(project.get("axis_r", AXIS_R)),
            geometry_cache=entry.get("geometry_cache"),
        )
        result["service_scope"] = scope_city(project, city)
        rows.append({"city": city, "preparation": result})
        print(f"PREPARED {city}", flush=True)
    write_json(Path(project["output"]) / "preparation_summary.json", clean(rows))
    return rows


def scope_city(project, city):
    from .building_scope import prepare_city_scope

    sources = project["cities"][city]
    if not sources.get("service_boundary"):
        raise ValueError(f"{city}: service_boundary is required for canonical building membership")
    base = city_dir(project, city) / "customer_inputs"
    return prepare_city_scope(
        population_path=base / "population.parquet",
        sources=sources,
        geometry_dir=base / "geometry",
        boundary_path=sources["service_boundary"],
        output_dir=base,
    )


def providers_for(project):
    from .gpt_provider import GPTProviderConfig, CodexCLIProvider

    cfg = project.get("ai", {})
    common = dict(
        timeout_seconds=float(cfg.get("timeout_seconds", 300)),
        allow_web=bool(cfg.get("allow_web", True)),
        prompt_version=cfg.get("prompt_version", "customer_evidence_v1"),
    )
    return {
        "claude": ClaudeCLIProvider(
            AIProviderConfig(
                cache_dir=Path(project["output"]) / "ai_requests/claude",
                command=cfg.get("command", "claude"),
                model=cfg.get("model"),
                max_budget_usd=cfg.get("max_budget_usd"),
                **common,
            )
        ),
        "gpt": CodexCLIProvider(
            GPTProviderConfig(
                cache_dir=Path(project["output"]) / "ai_requests/gpt",
                command=cfg.get("gpt_command", "codex"),
                model=cfg.get("gpt_model"),
                **common,
            )
        ),
    }


def apply_dual_record(record, task, replies):
    from .dual_review import reconcile_assessments

    reconciliation = reconcile_assessments(
        task, {name: reply["output"] for name, reply in replies.items()}
    )
    if reconciliation["assessment"] is None:
        raise ValueError(reconciliation.get("reason", "Independent judgments incomplete"))
    apply = apply_purpose_assessment if task == "residential_use" else apply_unit_assessment
    record = apply(record, reconciliation["assessment"])
    record.setdefault("dual_reviews", []).append(reconciliation)
    record["dual_review_status"] = reconciliation["status"]
    record["ai_providers_json"] = json.dumps(list(replies))
    if reconciliation["needs_human_review"]:
        if task == "residential_use":
            record["purpose_review_status"] = "pending_conflict"
            record["purpose_review_reason"] = "Independent model disagreement: " + ", ".join(
                f"{name}={reply['output'].get('ai_inferred_use')}"
                for name, reply in replies.items()
            )
        else:
            record["units_review_status"] = "pending"
            record["unit_review_reason"] = reconciliation["assessment"].get(
                "reason", "Unit judgments disagree"
            )
    for name, reply in replies.items():
        record.setdefault("ai_runs", []).append(
            {
                "task": task,
                "provider": name,
                "request_id": reply.get("request_id"),
                "record_path": str(reply["record_path"]) if reply.get("record_path") else None,
                "model": reply.get("model", "not_reported"),
            }
        )
    return record, reconciliation


def _profile_mask(frame, profile):
    units = pd.to_numeric(frame.units_final, errors="coerce")
    valid = units.notna() & units.gt(0) & units.eq(np.floor(units))
    if profile == "u1":
        return valid & units.eq(1)
    if profile == "u_lt50":
        return valid & units.lt(50)
    if profile == "u_lt100":
        return valid & units.lt(100)
    if profile == "unrestricted":
        return valid
    raise ValueError(profile)


def _eligibility(frame, profile, include_model_imputed=True):
    good = (
        frame.matched_ms_osm.astype(bool)
        & frame.geometry_clear.astype(bool)
        & frame.service_domain_eligible.astype(bool)
        & frame.residential_use.eq("residential")
        & frame.units_scope.eq("building")
        & ~frame.units_review_status.fillna("").astype(str).str.startswith("pending")
        & ~frame.units_review_status.isin(["unresolved", "rejected"])
        & ~frame.purpose_review_status.fillna("").astype(str).str.startswith("pending")
        & _profile_mask(frame, profile)
    )
    if not include_model_imputed:
        good &= ~frame.units_method.eq("model_imputed")
    return good


def _references(base, states):
    # Independently verified overrides are sparse; raw NSI is not a ground-truth library.
    return select_reference_examples(list(states.values()))


def reference_library(project):
    cases = []
    for city in project["cities"]:
        for record in overrides(city_dir(project, city)).values():
            record = {"city": city, **record}
            cases.append(record)
    return select_reference_examples(cases)


def _nearest_references(record, references, count):
    choices = [x for x in references if x["entity_id"] != record["entity_id"]]
    target = float(record.get("area_m2") or 1)
    choices.sort(
        key=lambda x: (
            abs(np.log(max(float(x.get("area_m2") or 1), 1) / max(target, 1))),
            x["entity_id"],
        )
    )
    fields = {
        "entity_id",
        "city",
        "area_m2",
        "building_features",
        "longitude",
        "latitude",
        "residential_use",
        "units_final",
        "units_scope",
        "units_method",
        "units_evidence",
        "units_independently_verified",
        "osm_objects",
    }
    return [{k: clean(v) for k, v in row.items() if k in fields} for row in choices[:count]]


def run_ai(project, cities, task, limit=None, retry_unresolved=False):
    from .building_inputs import BuildingEvidenceReader

    budget = int(limit if limit is not None else project.get("ai", {}).get("batch_limit", 10))
    providers = providers_for(project)
    completed, failed = [], []
    for city in cities:
        base = city_dir(project, city)
        frame, states = population(base)
        mask = frame.geometry_clear.astype(bool) & frame.service_domain_eligible.astype(bool)
        if task == "residential_use":
            mask &= frame.residential_use.eq("unknown")
            already = {k for k, s in states.items() if s.get("purpose_assessments")}
        elif task == "unit_review":
            mask &= frame.residential_use.eq("residential") & frame.nsi_count.gt(0)
            already = {
                k
                for k, s in states.items()
                if any("review_required" in a for a in s.get("unit_assessments", []))
            }
        else:
            mask &= frame.residential_use.eq("residential") & frame.units_final.isna()
            mask &= ~frame.units_review_status.fillna("").astype(str).str.startswith("pending")
            already = {
                k
                for k, s in states.items()
                if any("units_method" in a for a in s.get("unit_assessments", []))
            }
        if retry_unresolved:
            already = set()
        targets = frame.loc[mask & ~frame.entity_id.isin(already)].sort_values("entity_id")
        reader = BuildingEvidenceReader(project["cities"][city], base / "customer_inputs/geometry")
        references = reference_library(project)
        for _, row in targets.iterrows():
            if len(completed) + len(failed) >= budget:
                break
            record = record_for(row, states)
            payload = reader.payload(row)
            record.update(
                {
                    key: payload[key]
                    for key in (
                        "longitude",
                        "latitude",
                        "osm_objects",
                        "microsoft_building_ids",
                        "source_snapshots",
                        "building_features",
                    )
                    if key in payload
                }
            )
            if retry_unresolved:
                payload["annotation_revision"] = datetime.now(UTC).isoformat()
            if task != "residential_use":
                payload["residential_use"] = record["residential_use"]
                payload["nsi_attribution_clear"] = record.get("nsi_attribution_clear", False)
                payload["nsi_records"] = clean(
                    pd.read_parquet(
                        base / "customer_inputs/source_records.parquet",
                        filters=[("owner", "=", int(row.entity_index))],
                    )
                    .drop(columns=["geometry"], errors="ignore")
                    .to_dict("records")
                )
                payload["reference_cases"] = _nearest_references(
                    record, references, int(project.get("ai", {}).get("reference_count", 8))
                )
            frozen_payload = clean(payload)
            replies = {}
            for name, provider in providers.items():
                try:
                    replies[name] = provider.run(task, deepcopy(frozen_payload))
                except AIProviderError as error:
                    failed.append(
                        {
                            "city": city,
                            "entity_id": row.entity_id,
                            "provider": name,
                            "error": str(error),
                            "record_path": str(error.record_path),
                        }
                    )
                    break
            if len(replies) != len(providers):
                # Keep successful cached calls, never apply the first judgment
                # when the independent second run is absent or has failed.
                break
            record, reconciliation = apply_dual_record(record, task, replies)
            save_override(base, record)
            states[row.entity_id] = record
            completed.append(
                {
                    "city": city,
                    "entity_id": row.entity_id,
                    "dual_review_status": reconciliation["status"],
                    "request_ids": {name: reply["request_id"] for name, reply in replies.items()},
                }
            )
            print(f"ANNOTATED {city} {task} {row.entity_id}", flush=True)
        if failed or len(completed) >= budget:
            break
    result = {
        "task": task,
        "completed": completed,
        "failed": failed,
        "status": "failed" if failed else "batch_complete",
        "limit": budget,
    }
    write_json(Path(project["output"]) / f"{task}_last_batch.json", clean(result))
    return result


def import_assessments(project, cities, task, path):
    """Import real recorded outputs from another model runner, never fabricate them."""
    from .building_inputs import BuildingEvidenceReader

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_city = {city: population(city_dir(project, city)) for city in cities}
    readers = {}
    applied = 0
    pending = 0
    reused = 0
    imported_sha256 = sha256_file(Path(path))
    for item in rows:
        judgments = item.get("judgments")
        output = item.get("output", item)
        if judgments:
            first = next(iter(judgments.values()))
            output = first.get("output", first)
        entity = output["entity_id"]
        matches = [
            (city, frame, prior)
            for city, (frame, prior) in by_city.items()
            if (frame.entity_id == entity).any() and (not item.get("city") or item["city"] == city)
        ]
        if len(matches) != 1:
            raise ValueError(f"Assessment needs a unique city/entity: {entity}")
        city, frame, prior = matches[0]
        row = frame.loc[frame.entity_id.eq(entity)].iloc[0]
        record = record_for(row, prior)
        import_provider = item.get("provider", "external_recorded_output")
        if any(
            run.get("task") == task
            and run.get("imported_sha256") == imported_sha256
            and run.get("provider") == import_provider
            for run in record.get("ai_runs", [])
        ):
            reused += 1
            continue
        source_info = project["cities"][city]
        if source_info.get("osm") and source_info.get("microsoft"):
            if city not in readers:
                readers[city] = BuildingEvidenceReader(
                    source_info, city_dir(project, city) / "customer_inputs/geometry"
                )
            payload = readers[city].payload(row)
            record.update(
                {
                    key: payload[key]
                    for key in (
                        "longitude",
                        "latitude",
                        "osm_objects",
                        "microsoft_building_ids",
                        "source_snapshots",
                        "building_features",
                    )
                    if key in payload
                }
            )
        if project.get("ai", {}).get("require_dual_review", False):
            stored = record.setdefault("pending_independent_judgments", {}).setdefault(task, {})
            inputs = judgments or {item.get("provider", "external_recorded_output"): item}
            for name, reply in inputs.items():
                provider = {"claude_cli": "claude", "codex_cli": "gpt"}.get(name, name)
                stored[provider] = reply if "output" in reply else {"output": reply}
            if all(name in stored for name in ("claude", "gpt")):
                record, _ = apply_dual_record(record, task, stored)
                record["pending_independent_judgments"].pop(task, None)
                applied += 1
            else:
                record["dual_review_status"] = "incomplete"
                if task == "residential_use":
                    record["purpose_review_status"] = "pending_independent_review"
                else:
                    record["units_review_status"] = "pending_independent_review"
                pending += 1
        else:
            apply = apply_purpose_assessment if task == "residential_use" else apply_unit_assessment
            record = apply(record, output)
            applied += 1
        record.setdefault("ai_runs", []).append(
            {
                "task": task,
                "imported_from": str(Path(path).resolve()),
                "imported_sha256": imported_sha256,
                "model": item.get("model", "not_reported"),
                "prompt_version": item.get("prompt_version", "not_reported"),
                "provider": import_provider,
            }
        )
        save_override(city_dir(project, city), record)
        prior[entity] = record
    return {
        "applied": applied,
        "pending_independent_review": pending,
        "reused": reused,
        "task": task,
    }


def _awaits_human(record):
    return any(
        str(record.get(key, "")).startswith("pending")
        and record.get(key) != "pending_independent_review"
        for key in ("units_review_status", "purpose_review_status")
    )


def export_review(project, cities, destination):
    rows = []
    for city in cities:
        base = city_dir(project, city)
        for record in overrides(base).values():
            if not _awaits_human(record):
                continue
            rows.append(
                {
                    "city": city,
                    "entity_id": record["entity_id"],
                    "longitude": record.get("longitude"),
                    "latitude": record.get("latitude"),
                    "osm_objects_json": json.dumps(
                        record.get("osm_objects", []), ensure_ascii=False
                    ),
                    "units_raw_nsi": json.dumps(record.get("units_raw_nsi")),
                    "units_current": record.get("units_final"),
                    "proposed_units": record.get("proposed_units"),
                    "reason": record.get("purpose_review_reason")
                    or record.get("unit_review_reason", record.get("proposed_residential_use", "")),
                    "evidence_json": json.dumps(
                        {
                            key: record.get(key, [])
                            for key in ("purpose_assessments", "unit_assessments", "dual_reviews")
                        },
                        ensure_ascii=False,
                    ),
                    "decision": "",
                    "units_final": "",
                    "units_scope": "",
                    "residential_use": "",
                    "independently_verified": "",
                    "reviewer": "",
                    "note": "",
                    "evidence": "",
                }
            )
    columns = [
        "city",
        "entity_id",
        "longitude",
        "latitude",
        "osm_objects_json",
        "units_raw_nsi",
        "units_current",
        "proposed_units",
        "reason",
        "evidence_json",
        "decision",
        "units_final",
        "units_scope",
        "residential_use",
        "independently_verified",
        "reviewer",
        "note",
        "evidence",
    ]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(destination, index=False)
    return {"pending": len(rows), "path": str(Path(destination).resolve())}


def apply_reviews(project, cities, path):
    path = Path(path)
    rows = (
        pd.read_csv(path, keep_default_na=False).to_dict("records")
        if path.suffix.lower() == ".csv"
        else [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    )
    count = 0
    for city in cities:
        base = city_dir(project, city)
        frame, prior = population(base)
        lookup = frame.set_index("entity_id", drop=False)
        for item in rows:
            if item.get("city") != city or not item.get("decision"):
                continue
            if item["entity_id"] not in lookup.index:
                raise ValueError(f"Unknown review entity: {item['entity_id']}")
            record = record_for(lookup.loc[item["entity_id"]], prior)
            decision = {k: v for k, v in item.items() if v != ""}
            decision["action"] = decision.pop("decision")
            if decision["action"] not in {
                "accept",
                "confirm",
                "keep",
                "retain",
                "correct",
                "replace",
                "reject",
                "discard",
                "discard_units",
                "defer",
                "pending",
            }:
                raise ValueError(f"Unknown human review decision: {decision['action']}")
            if "field" not in decision:
                has_purpose = bool(decision.get("residential_use"))
                has_units = any(k in decision for k in ("units_final", "units", "units_scope"))
                if not has_purpose and not has_units:
                    has_purpose = str(record.get("purpose_review_status", "")).startswith("pending")
                    has_units = str(record.get("units_review_status", "")).startswith("pending")
                decision["field"] = (
                    "both" if has_purpose and has_units else "purpose" if has_purpose else "units"
                )
            if isinstance(decision.get("evidence"), str):
                decision["evidence"] = json.loads(decision["evidence"])
            if isinstance(decision.get("independently_verified"), str):
                decision["independently_verified"] = decision["independently_verified"].lower() in {
                    "true",
                    "1",
                    "yes",
                }
            record = apply_review_decision(record, decision)
            save_override(base, record)
            prior[record["entity_id"]] = record
            count += 1
    return {"applied_human_decisions": count}


def finalize(project, cities):
    result = []
    allow_estimates = bool(project.get("customer", {}).get("include_model_imputed", True))
    for city in cities:
        base = city_dir(project, city)
        frame, states = population(base)
        clear = (
            frame.matched_ms_osm.astype(bool)
            & frame.geometry_clear.astype(bool)
            & frame.service_domain_eligible.astype(bool)
        )
        counts = frame.loc[clear].residential_use.value_counts()
        r, u = int(counts.get("residential", 0)), int(counts.get("unknown", 0))
        denominator = r + u
        row = {
            "city": city,
            "matched_geometry_clear": int(clear.sum()),
            "residential": r,
            "unknown": u,
            "nonresidential": int(counts.get("nonresidential", 0)),
            "mixed": int(counts.get("mixed", 0)),
            "residential_candidate_denominator": denominator,
            "residential_identification_rate": r / denominator if denominator else None,
            "pending_human_review": sum(_awaits_human(x) for x in states.values()),
        }
        row["matched_outside_service_boundary"] = int(
            (~frame.service_domain_eligible.astype(bool)).sum()
        )
        processed_purpose = {
            key for key, state in states.items() if state.get("purpose_assessments")
        }
        row["ai_purpose_processed_buildings"] = int(
            (clear & frame.entity_id.isin(processed_purpose)).sum()
        )
        row["unknown_not_yet_ai_processed"] = int(
            (
                clear
                & frame.residential_use.eq("unknown")
                & ~frame.entity_id.isin(processed_purpose)
            ).sum()
        )
        row["unknown_after_ai_processing"] = int(
            (
                clear
                & frame.residential_use.eq("unknown")
                & frame.entity_id.isin(processed_purpose)
            ).sum()
        )
        row["ai_unit_review_processed_buildings"] = sum(
            any("review_required" in a for a in state.get("unit_assessments", []))
            for state in states.values()
        )
        row["independently_verified_reference_cases"] = len(_references(base, states))
        for profile in PROFILES:
            accepted = _eligibility(frame, profile, allow_estimates)
            sourced = accepted & ~frame.units_method.eq("model_imputed")
            frame[f"eligible_{profile}"] = accepted
            row[f"customers_{profile}"] = int(accepted.sum())
            row[f"units_{profile}"] = int(pd.to_numeric(frame.loc[accepted, "units_final"]).sum())
            row[f"customer_coverage_{profile}"] = (
                int(accepted.sum()) / denominator if denominator else None
            )
            row[f"source_supported_customers_{profile}"] = int(sourced.sum())
            row[f"model_imputed_customers_{profile}"] = int((accepted & ~sourced).sum())
        dest = base / "customer_inputs/final_buildings.parquet"
        frame.to_parquet(dest, index=False)
        lineage_inputs = {}
        for name in (
            "population.parquet",
            "preparation_manifest.json",
            "assessments.jsonl",
            "scope.parquet",
            "scope_manifest.json",
        ):
            source = base / "customer_inputs" / name
            if source.exists():
                lineage_inputs[name] = {
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                }
        write_json(
            base / "customer_inputs/finalization_manifest.json",
            {
                "schema": "building_customer_field_lineage_v1",
                "city": city,
                "inputs": lineage_inputs,
                "final_buildings_sha256": sha256_file(dest),
                "annotation_evidence_key": "entity_id",
                "raw_nsi_immutable": True,
                "profiles": list(PROFILES),
                "include_model_imputed": allow_estimates,
            },
        )
        references = _references(base, states)
        write_json(base / "customer_inputs/reference_cases.json", clean(references))
        write_json(base / "customer_inputs/coverage.json", clean(row))
        result.append(row)
    path = Path(project["output"]) / "customer_coverage.csv"
    combined = list(result)
    for city in project["cities"]:
        saved = city_dir(project, city) / "customer_inputs/coverage.json"
        if city not in cities and saved.exists():
            combined.append(read_json(saved))
    order = {city: index for index, city in enumerate(project["cities"])}
    combined.sort(key=lambda row: order[row["city"]])
    pd.DataFrame(combined).to_csv(path, index=False)
    write_json(
        Path(project["output"]) / "stage1_method.json",
        {
            "schema": SCHEMA,
            "generated_utc": datetime.now(UTC).isoformat(),
            "project_config_sha256": sha256_file(Path(project["config_path"])),
            "method": {
                "customer_entity": "matched_microsoft_osm_building",
                "height_admission_gate": False,
                "ai_purpose_before_unit_review": True,
                "raw_nsi_immutable": True,
                "web_extracted_separate_from_model_imputed": True,
                "reference_cases": "independently verified cases only",
                "purpose_denominator": "matched clear residential + unknown, regardless of NSI availability",
                "mixed_use": "separate",
                "profiles": list(PROFILES),
            },
            "whole_cle_frozen": False,
            "stage2_included": False,
        },
    )
    return result


def evaluation_inputs(project, cities, destination):
    from .building_inputs import BuildingEvidenceReader

    rows = []
    for city in cities:
        base = city_dir(project, city)
        refs = select_reference_examples(list(overrides(base).values()))
        reader = BuildingEvidenceReader(project["cities"][city], base / "customer_inputs/geometry")
        for reference in refs:
            features = reader.payload(pd.Series(reference))
            features.update(mask_units_for_evaluation(reference))
            item = mask_units_for_evaluation(features)
            rows.append({"city": city, **item})
    with Path(destination).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
    return {"masked_cases": len(rows), "path": str(destination)}


def evaluate(project, cities, path):
    references = []
    for city in cities:
        references.extend(
            select_reference_examples(list(overrides(city_dir(project, city)).values()))
        )
    predictions = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    result = evaluate_unit_predictions(references, predictions)
    write_json(Path(project["output"]) / "unit_imputation_evaluation.json", clean(result))
    return result


def components(project, cities, component):
    from .stage1_components import build_component
    from .building_inputs import BuildingEvidenceReader
    from .customer_building_layer import build_building_customer_layer

    selected = (
        ["roads", "customers", "depots", "charging", "speeds", "assemble"]
        if component == "all"
        else [component]
    )
    rows = []
    for city in cities:
        base = city_dir(project, city)
        for name in selected:
            if name == "customers":
                frame = pd.read_parquet(base / "customer_inputs/final_buildings.parquet")
                # Unknown/nonresidential objects stay in population and coverage,
                # while the access layer only routes residential candidates.
                frame = frame.loc[
                    frame.geometry_clear
                    & frame.service_domain_eligible
                    & frame.residential_use.eq("residential")
                ]
                reader = BuildingEvidenceReader(
                    project["cities"][city], base / "customer_inputs/geometry"
                )
                geometries = reader.geometry(frame)
                previous = base / "customers/residential_site_manifest.json"
                cache = None
                cache_receipt = None
                if previous.exists():
                    import geopandas as gpd

                    old = read_json(previous)
                    current_lineage = read_json(base / "customer_inputs/finalization_manifest.json")
                    prior_lineage = read_json(base / "customers/customer_lineage.json")
                    context_same = old["inputs"]["local_access_context"]["sha256"] == sha256_file(
                        base / "roads/access_context_manifest.json"
                    )
                    inputs_same = all(
                        current_lineage["inputs"].get(key) == prior_lineage["inputs"].get(key)
                        for key in ("population.parquet", "preparation_manifest.json")
                    )
                    spatial_same = all(
                        old["generation_dependencies"].get(key)
                        == sha256_file(Path(__file__).with_name(key))
                        for key in ("local_road_access.py", "customer_access.py")
                    )
                    prior_path = base / "customers/latent_locations.parquet"
                    if (
                        context_same
                        and inputs_same
                        and spatial_same
                        and sha256_file(prior_path) == old["outputs"][prior_path.name]
                    ):
                        cache = gpd.read_parquet(prior_path)
                        cache_receipt = {
                            "manifest_sha256": sha256_file(previous),
                            "latent_locations_sha256": old["outputs"][prior_path.name],
                        }
                    profile = project.get("customer", {}).get("active_profile", "u_lt100")
                    estimates = project.get("customer", {}).get("include_model_imputed", True)
                    code_same = all(
                        old["generation_dependencies"].get(key)
                        == sha256_file(Path(__file__).with_name(key))
                        for key in old["generation_dependencies"]
                    )
                    final_same = old["inputs"]["finalization_manifest.json"][
                        "sha256"
                    ] == sha256_file(base / "customer_inputs/finalization_manifest.json")
                    if (
                        cache is not None
                        and final_same
                        and code_same
                        and old["unit_profile"] == profile
                        and old["include_model_imputed"] == estimates
                    ):
                        rows.append(
                            {"city": city, "component": name, "report": {**old, "status": "reused"}}
                        )
                        print(f"REUSED {city} {name}", flush=True)
                        continue
                    archive = base / "draft_history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                    archive.mkdir(parents=True)
                    shutil.move(str(base / "customers"), str(archive / "customers"))
                    if cache_receipt is not None:
                        cache_receipt["path"] = str(archive / "customers/latent_locations.parquet")
                report = build_building_customer_layer(
                    buildings=geometries,
                    context_dir=base / "roads",
                    output_dir=base,
                    city_slug=city,
                    area_crs=project["cities"][city]["area_crs"],
                    profile=project.get("customer", {}).get("active_profile", "u_lt100"),
                    include_model_imputed=project.get("customer", {}).get(
                        "include_model_imputed", True
                    ),
                    obstacle_buildings=reader.obstacles(),
                    projection_cache=cache,
                )
                if cache_receipt is not None:
                    report["projection_cache"] = cache_receipt
                    write_json(base / "customers/residential_site_manifest.json", report)
            else:
                report = build_component(project=project, city=city, component=name)
            rows.append({"city": city, "component": name, "report": clean(report)})
            print(f"COMPLETE {city} {name}", flush=True)
    return rows


def verify(project, cities):
    from .stage1_validation import validate_building_stage1

    reports = []
    for city in cities:
        base = city_dir(project, city)
        report = validate_building_stage1(base)
        report["cle_manifest_sha256"] = sha256_file(base / "cle_manifest.json")
        report["validated_utc"] = datetime.now(UTC).isoformat()
        write_json(base / "stage1_validation.json", clean(report))
        reports.append({"city": city, **report})
    write_json(Path(project["output"]) / "validation_summary.json", clean(reports))
    return reports


def build(project, cities, *, limit=None, skip_ai=False):
    prepare(project, cities)
    status = {"whole_cle_frozen": False, "stage2_included": False}
    if not skip_ai:
        for task in ("residential_use", "unit_review"):
            batch = run_ai(project, cities, task, limit)
            if batch["failed"]:
                status.update(status="awaiting_ai_runner", ai_failures=batch["failed"])
                finalize(project, cities)
                write_json(Path(project["output"]) / "build_progress.json", clean(status))
                return status
    review = export_review(project, cities, Path(project["output"]) / "human_review.csv")
    if review["pending"]:
        status.update(status="awaiting_human_review", human_review=review)
        finalize(project, cities)
        write_json(Path(project["output"]) / "build_progress.json", clean(status))
        return status
    if not skip_ai:
        batch = run_ai(project, cities, "unit_impute", limit)
        if batch["failed"]:
            status.update(status="awaiting_ai_runner", ai_failures=batch["failed"])
            write_json(Path(project["output"]) / "build_progress.json", clean(status))
            return status
        review = export_review(project, cities, Path(project["output"]) / "human_review.csv")
        if review["pending"]:
            status.update(status="awaiting_human_review", human_review=review)
            finalize(project, cities)
            write_json(Path(project["output"]) / "build_progress.json", clean(status))
            return status
    finalize(project, cities)
    status.update(status="assembled_draft", components=components(project, cities, "all"))
    status["validation"] = verify(project, cities)
    if any(not report["passed"] for report in status["validation"]):
        status["status"] = "validation_failed"
    write_json(Path(project["output"]) / "build_progress.json", clean(status))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create the building-first Stage1 project")
    init.add_argument("directory", type=Path)
    for command in (
        "plan",
        "prepare",
        "scope",
        "ai",
        "import-ai",
        "export-review",
        "apply-review",
        "finalize",
        "evaluation-inputs",
        "evaluate",
        "components",
        "verify",
        "build",
    ):
        p = sub.add_parser(command)
        p.add_argument("config", type=Path)
        p.add_argument("--cities", nargs="+")
        p.add_argument("--json", action="store_true", help="Print full saved result metadata")
        if command in {"ai", "import-ai"}:
            p.add_argument(
                "--task", required=True, choices=["residential_use", "unit_review", "unit_impute"]
            )
        if command in {"ai", "build"}:
            p.add_argument("--limit", type=int)
        if command == "ai":
            p.add_argument(
                "--retry-unresolved",
                action="store_true",
                help="Reassess unfinished cases after adding references or changing the model",
            )
        if command in {"import-ai", "apply-review", "evaluate"}:
            p.add_argument("--input", type=Path, required=True)
        if command in {"export-review", "evaluation-inputs"}:
            p.add_argument("--output", type=Path, required=True)
        if command == "components":
            p.add_argument(
                "--component",
                required=True,
                choices=["all", "roads", "customers", "depots", "charging", "speeds", "assemble"],
            )
        if command == "build":
            p.add_argument("--skip-ai", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "init":
        print(initialize(args.directory))
        return
    project = load_project(args.config)
    cities = selected_cities(project, args.cities)
    Path(project["output"]).mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        missing = [
            {"city": c, "input": k, "path": project["cities"][c].get(k)}
            for c in cities
            for k in ("microsoft", "osm", "nsi")
            if not project["cities"][c].get(k) or not Path(project["cities"][c][k]).is_file()
        ]
        component_inputs = project.get("components", {})
        geographic_missing = []
        inventory = component_inputs.get("inventory")
        if not inventory or not Path(inventory).is_file():
            geographic_missing.append({"input": "components.inventory", "path": inventory})
        roads_root = component_inputs.get("road_sources")
        for city in cities:
            for name in ("osm_ways.parquet", "osm_access_nodes.parquet", "extraction.json"):
                path = Path(roads_root) / city / name if roads_root else None
                if path is None or not path.is_file():
                    geographic_missing.append(
                        {"city": city, "input": name, "path": str(path) if path else None}
                    )
        result = {
            "ready": not missing and not geographic_missing,
            "customer_inputs_ready": not missing,
            "geographic_inputs_ready": not geographic_missing,
            "cities": cities,
            "missing": missing,
            "missing_geographic_inputs": geographic_missing,
            "output": project["output"],
            "ai": {name: provider.readiness() for name, provider in providers_for(project).items()},
        }
    elif args.command == "prepare":
        result = prepare(project, cities)
    elif args.command == "scope":
        result = [{"city": city, **scope_city(project, city)} for city in cities]
    elif args.command == "ai":
        result = run_ai(project, cities, args.task, args.limit, args.retry_unresolved)
    elif args.command == "import-ai":
        result = import_assessments(project, cities, args.task, args.input)
    elif args.command == "export-review":
        result = export_review(project, cities, args.output)
    elif args.command == "apply-review":
        result = apply_reviews(project, cities, args.input)
    elif args.command == "evaluation-inputs":
        result = evaluation_inputs(project, cities, args.output)
    elif args.command == "evaluate":
        result = evaluate(project, cities, args.input)
    elif args.command == "finalize":
        result = finalize(project, cities)
    elif args.command == "components":
        result = components(project, cities, args.component)
    elif args.command == "verify":
        result = verify(project, cities)
    else:
        result = build(project, cities, limit=args.limit, skip_ai=args.skip_ai)
    if args.json:
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
    elif args.command in {"prepare", "components"}:
        print(f"Saved {len(result)} completed records to {project['output']}")
    elif args.command == "finalize":
        print(f"Updated {len(result)} cities: {Path(project['output']) / 'customer_coverage.csv'}")
    elif args.command == "verify":
        for report in result:
            print(f"{report['city']}: {'PASS' if report['passed'] else 'FAIL'}")
            for error in report.get("errors", [])[:5]:
                print("  " + str(error))
        print(f"Reports: {Path(project['output']) / 'validation_summary.json'}")
    elif args.command == "build":
        print(f"Stage 1 status: {result['status']}\nOutput: {project['output']}")
        if result.get("human_review"):
            print(f"Human review: {result['human_review']['path']}")
        for failure in result.get("ai_failures", []):
            print(f"AI runner: {failure['error']}\nDetails: {failure['record_path']}")
    else:
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
    if isinstance(result, dict) and (
        result.get("failed")
        or result.get("ai_failures")
        or result.get("status") == "validation_failed"
    ):
        raise SystemExit(2)
    if args.command == "verify" and any(not r["passed"] for r in result):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
