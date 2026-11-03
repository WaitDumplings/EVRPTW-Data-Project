import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from evrptw_scenarios.cle_binding import BUILDING_SCHEMA, R1_FREEZE_ID, R1_FREEZE_SHA256, cle_authority
from evrptw_scenarios.tract_activation import activate_tracts


def test_building_cle_authority_requires_bound_cohort_and_city_validation(tmp_path):
    city = tmp_path / "cities" / "dallas"
    city.mkdir(parents=True)
    manifest = {"schema": BUILDING_SCHEMA, "city_slug": "dallas"}
    source = json.dumps(manifest).encode()
    (city / "cle_manifest.json").write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    validation = city / "stage1_validation.json"
    validation.write_text(json.dumps({"passed": True, "cle_manifest_sha256": digest}))
    receipt = tmp_path / "stage1_server_build_receipt.json"
    with pytest.raises(FileNotFoundError):
        cle_authority(city, manifest)
    cohort = {"dallas": digest, **{f"other_{i}": "hash" for i in range(10)}}
    receipt.write_text(json.dumps({
        "schema": "evrptw_stage1_server_build_receipt_v1",
        "freeze_id": R1_FREEZE_ID, "freeze_manifest_sha256": R1_FREEZE_SHA256,
        "complete_cle_validation_passed": True, "validated_cities": list(cohort),
        "cle_manifests": cohort,
    }))
    assert cle_authority(city, manifest) == (receipt, validation)
    (city / "cle_manifest.json").write_bytes(source + b" ")
    with pytest.raises(ValueError, match="does not authorize"):
        cle_authority(city, manifest)


def test_sequential_tract_samples_exact_n_and_excludes_unmapped():
    catalog = pd.DataFrame({
        "location_id": ["a", "b", "c", "d"],
        "bg_geoid": pd.array(["481130101001", "481130101002", None, "481130102001"], dtype="string"),
        "cell_id": ["bg1", "bg2", "unmapped", "bg3"],
        "modeled_residential_units": [1, 4, 20, 2],
        "location_lon": [-96.80, -96.79, -96.77, -96.76],
        "location_lat": [32.80, 32.81, 32.82, 32.83],
        "depot_outward_distance_m": [1000., 2000., 3000., 4000.],
    })
    context = SimpleNamespace(catalog=catalog, info={"world_id": "world"},
        adjacency={"bg1": ["bg2"], "bg2": ["bg1", "unmapped"],
                   "unmapped": ["bg2", "bg3"], "bg3": ["unmapped"]},
        depot=catalog.iloc[:1])
    config = {"world": {"p_unit": .03}, "operations": {"start_date": "2026-11-02"},
              "activation": {"candidate_multiplier": 20., "opening_probability": .1,
                             "parameter_status": "test"}}
    info, indices, candidates, _ = activate_tracts(context, config, day_index=0, size=2)
    assert info["supported_size"] and len(indices) == 2 and 2 not in indices
    assert info["effective_candidate_target"] == info["candidate_count"] == 3
    assert info["unmapped_tract_count"] == 1 and not info["candidate_target_reached"]
    assert {row["location_id"] for row in candidates} == {"a", "b", "d"}
