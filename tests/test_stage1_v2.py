import json
from pathlib import Path

import numpy as np
import pandas as pd

from evrptw_cle.customer_enrichment import initialize_record, apply_unit_assessment
from evrptw_cle.stage1_v2 import (
    SCHEMA,
    apply_reviews,
    export_review,
    finalize,
    import_assessments,
    population,
    save_override,
)


def setup_project(tmp_path):
    project = {
        "schema": SCHEMA,
        "config_path": str(tmp_path / "project.json"),
        "output": str(tmp_path / "out"),
        "cities": {"test": {}},
        "customer": {},
    }
    Path(project["config_path"]).write_text(json.dumps({"schema": SCHEMA}))
    base = Path(project["output"]) / "cities/test"
    (base / "customer_inputs").mkdir(parents=True)
    rows = []
    for i, (label, units) in enumerate(
        [("residential", 1), ("unknown", None), ("unknown", 2), ("nonresidential", 1), ("mixed", 2)]
    ):
        r = initialize_record(
            {
                "entity_id": f"b{i}",
                "entity_index": i,
                "osm_use": label,
                "units_raw_nsi": units,
                "nsi_count": int(units is not None),
                "nsi_attribution_clear": units is not None,
            }
        )
        for field in (
            "purpose_assessments",
            "unit_assessments",
            "review_history",
            "units_evidence",
            "purpose_evidence",
        ):
            r.pop(field)
        rows.append(r)
    pd.DataFrame(rows).to_parquet(base / "customer_inputs/population.parquet", index=False)
    return project, base


def test_denominator_keeps_unprocessed_and_missing_nsi(tmp_path):
    project, _ = setup_project(tmp_path)
    row = finalize(project, ["test"])[0]
    assert row["residential_candidate_denominator"] == 3
    assert row["unknown"] == 2 and row["mixed"] == 1
    assert row["customer_coverage_u1"] == 1 / 3


def test_real_output_import_reclassifies_without_requiring_nsi(tmp_path):
    project, base = setup_project(tmp_path)
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps(
            {
                "city": "test",
                "provider": "test_fixture",
                "output": {
                    "entity_id": "b1",
                    "ai_inferred_use": "residential",
                    "reason": "Fixture use tag",
                    "evidence": [
                        {"osm_id": "w1", "tag_key": "building:use", "tag_value": "residential"}
                    ],
                },
            }
        )
        + "\n"
    )
    assert import_assessments(project, ["test"], "residential_use", path)["applied"] == 1
    row = finalize(project, ["test"])[0]
    assert row["residential"] == 2 and row["unknown"] == 1
    assert row["residential_candidate_denominator"] == 3
    assert row["customers_unrestricted"] == 1
    frame, _ = population(base)
    assert np.isnan(frame.loc[frame.entity_id.eq("b1"), "units_final"].iloc[0])


def test_human_csv_roundtrip_keeps_raw_units(tmp_path):
    project, base = setup_project(tmp_path)
    state = initialize_record(
        {
            "entity_id": "b0",
            "entity_index": 0,
            "osm_use": "residential",
            "units_raw_nsi": 1,
            "nsi_count": 1,
            "nsi_attribution_clear": True,
        }
    )
    state = apply_unit_assessment(
        state,
        {
            "entity_id": "b0",
            "review_required": True,
            "units_scope": "building",
            "reason": "Fixture discrepancy",
        },
    )
    save_override(base, state)
    path = tmp_path / "human.csv"
    assert export_review(project, ["test"], path)["pending"] == 1
    rows = pd.read_csv(path, keep_default_na=False)
    rows.loc[0, "decision"] = "correct"
    rows.loc[0, "units_final"] = "2"
    rows.loc[0, "units_scope"] = "building"
    rows.loc[0, "independently_verified"] = "true"
    rows.loc[0, "evidence"] = json.dumps(
        [{"type": "manual", "note": "Explicit fixture two-unit record"}]
    )
    rows.to_csv(path, index=False)
    assert apply_reviews(project, ["test"], path)["applied_human_decisions"] == 1
    frame, states = population(base)
    assert states["b0"]["units_raw_nsi"] == 1
    assert states["b0"]["units_final"] == 2
    assert states["b0"]["units_independently_verified"]
    assert frame.loc[frame.entity_id.eq("b0"), "units_method"].iloc[0] == "human_corrected"
