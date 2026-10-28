import json

import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.site_scope_audit import (
    annotate_site_scope,
    apply_site_scope_eligibility,
    audit_saved_site_scope,
    audit_site_scope,
)
from evrptw_cle.residential_sites import validate_site_tables
from evrptw_cle.util import sha256_file
from test_residential_sites import infer, inputs


def cross_boundary_sites(*, same_component=True):
    records, footprints, buildings, complexes = inputs()
    records = records.iloc[:2].copy()
    if not same_component:
        records.loc[records.index[1], "ftprntid"] = "different"
    complexes.geometry = [box(0, 0, 1.1, 2)]
    return infer(records, footprints, buildings.iloc[:1], complexes)[:2]


def test_cross_boundary_candidate_keeps_native_ids_and_does_not_change_sites():
    sites, members = cross_boundary_sites()
    original_sites, original_members = sites.copy(deep=True), members.copy(deep=True)
    report, ledger = audit_site_scope(sites, members)
    assert report["counts"]["source_components_spanning_sites"] == 1
    assert report["counts"]["microsoft_roofs_spanning_complex_and_fallback"] == 1
    assert report["counts"]["affected_sites"] == 2
    assert report["counts"]["affected_member_rows"] == 2
    assert report["counts"]["candidate_ledger_rows"] == 4
    assert set(members.columns).issubset(ledger.columns)
    assert set(ledger.fd_id) == set(members.fd_id)
    assert set(ledger.audit_site_source_key) == {"complex:w4", "msft:a"}
    assert ledger.audit_site_scope_ambiguous.all()
    assert ledger.audit_site_scope_review_required.all()
    assert ledger.audit_complex_site_count.eq(1).all()
    assert ledger.audit_fallback_site_count.eq(1).all()
    pd.testing.assert_frame_equal(sites, original_sites)
    pd.testing.assert_frame_equal(members, original_members)


def test_roof_competition_does_not_require_the_same_source_component():
    sites, members = cross_boundary_sites(same_component=False)
    report, ledger = audit_site_scope(sites, members)
    assert report["counts"]["source_components_spanning_sites"] == 0
    assert report["counts"]["microsoft_roofs_spanning_complex_and_fallback"] == 1
    assert len(ledger) == 2


def test_component_can_span_fallback_sites_without_complex_roof_competition():
    sites = pd.DataFrame({"site_id": ["a", "b"], "site_source_key": ["msft:a", "msft:b"]})
    members = pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "fd_id": [1, 2],
            "nsi_component_id": [0, 0],
            "composite_building_id": ["a", "b"],
        }
    )
    report, ledger = audit_site_scope(sites, members)
    assert report["counts"]["source_components_spanning_sites"] == 1
    assert report["counts"]["microsoft_roofs_spanning_complex_and_fallback"] == 0
    assert len(ledger) == 2 and ledger.audit_fallback_site_count.eq(2).all()


def test_clean_snapshot_produces_empty_ledger_with_schema():
    sites, members, _ = infer(*inputs())
    report, ledger = audit_site_scope(sites, members)
    assert report["counts"]["affected_sites"] == 0
    assert report["counts"]["candidate_ledger_rows"] == 0
    assert ledger.empty and "audit_identity_key" in ledger
    assert set(members.columns).issubset(ledger.columns)
    assert sites.residential_units.isna().all()
    assert sites.site_scope_unresolved.any()
    sites["road_access_candidate_eligible"] = True
    assert apply_site_scope_eligibility(sites).cle_default_instance_eligible.all()


def test_saved_audit_preserves_inputs_and_records_hashes(tmp_path):
    sites, members = cross_boundary_sites()
    site_path, member_path = tmp_path / "sites.parquet", tmp_path / "members.parquet"
    sites.to_parquet(site_path, index=False)
    members.to_parquet(member_path, index=False)
    before = (sha256_file(site_path), sha256_file(member_path))
    output = tmp_path / "audit"
    report = audit_saved_site_scope(
        sites_path=site_path, members_path=member_path, output_dir=output
    )
    assert report["inputs"]["sites"]["sha256"] == before[0] == sha256_file(site_path)
    assert report["inputs"]["members"]["sha256"] == before[1] == sha256_file(member_path)
    assert json.loads((output / "site_scope_audit.json").read_text()) == report
    assert len(pd.read_csv(output / report["outputs"]["candidate_ledger"]["path"])) == 4
    with pytest.raises(FileExistsError, match="fresh audit"):
        audit_saved_site_scope(sites_path=site_path, members_path=member_path, output_dir=output)


def test_unknown_member_owner_is_rejected():
    sites, members = cross_boundary_sites()
    members.loc[members.index[0], "site_id"] = "missing"
    with pytest.raises(ValueError, match="known, nonempty site_id"):
        audit_site_scope(sites, members)


def test_missing_identity_columns_fail_before_misleading_zero_count():
    sites, members = cross_boundary_sites()
    with pytest.raises(ValueError, match="nsi_component_id"):
        audit_site_scope(sites, members.drop(columns="nsi_component_id"))


def supported_building_split_inputs():
    records, footprints, buildings, complexes = inputs()
    records = records.iloc[:2].copy()
    records.geometry = [Point(1, 1), Point(4, 1)]
    records["ftprntid"] = ["left", "right"]
    records["resunits"] = [10, 20]
    footprints = footprints.iloc[:1].copy()
    footprints.geometry = [box(0.5, 0.5, 4.5, 1.5)]
    buildings = buildings.iloc[:2].copy()
    complexes.geometry = [box(0, 0, 2, 2)]
    return records, footprints, buildings, complexes


def test_supported_source_building_split_keeps_distinct_sites_and_units():
    from evrptw_cle.residential_sites import infer_residential_sites

    sites, members, summary, ledger = infer_residential_sites(
        *supported_building_split_inputs(),
        city_slug="test",
        boundary=box(-1, -1, 10, 3),
        return_scope_audit=True,
    )
    assert len(sites) == 2 and sites.residential_units.sum() == 30
    assert sites.site_supported_building_split.all()
    assert not sites.site_scope_review_required.any()
    assert ledger.audit_scope_resolution.eq("supported_source_building_split").all()
    assert summary["site_scope_audit"]["scope_resolution"]["supported_building_split_groups"] == 1
    assert validate_site_tables(sites, members)["passed"]
    sites["road_access_candidate_eligible"] = True
    assert apply_site_scope_eligibility(sites).cle_default_instance_eligible.all()


def test_component_cross_owner_is_pending_and_cannot_retain_partial_site_units():
    records, footprints, buildings, complexes = inputs()
    records = records.loc[records.fd_id.isin([1, 3, 5])].copy()
    records.loc[records.fd_id.eq(3), "ftprntid"] = "a"
    records.loc[records.fd_id.eq(5), "resunits"] = 8
    complexes.geometry = [box(3, 0, 9, 2)]
    sites, members, _ = infer(records, footprints, buildings, complexes)
    assert sites.site_scope_review_required.all()
    assert sites.residential_units.isna().all()
    assert not members.operational_units_counted.any()
    assert members.counted_residential_units.isna().all()
    assert members.resunits.sum() == 38  # raw unit evidence remains inspectable
    assert {"ftprntsrc", "ftprntid", "usastrucid", "cbfips"}.issubset(members.columns)
    assert members.ftprntid.tolist() == ["a", "a", "d"]
    sites["road_access_candidate_eligible"] = True
    result = apply_site_scope_eligibility(sites)
    assert result.road_access_candidate_eligible.all()
    assert not result.cle_candidate_eligible.any()
    assert validate_site_tables(result, members)["passed"]
    result.loc[result.index[0], "cle_default_instance_eligible"] = True
    with pytest.raises(ValueError, match="unresolved owner scope is default eligible"):
        validate_site_tables(result, members)


def test_explicit_relation_member_competes_with_a_separate_fallback_extent():
    records, footprints, buildings, complexes = supported_building_split_inputs()
    complexes["member_osm_ids_json"] = [json.dumps(["w1", "w2"])]
    sites, members, _ = infer(records, footprints, buildings, complexes)
    assert sites.site_scope_review_required.all()
    assert sites.site_scope_review_reason.str.contains("explicit_complex_membership").all()
    assert sites.residential_units.isna().all()
    assert len(sites) == 2 and len(members) == 2  # no invented merge/extent


def test_scope_annotation_requires_source_evidence_to_clear_a_shared_roof():
    sites, members, _ = infer(*supported_building_split_inputs())
    result, members, ledger, _ = annotate_site_scope(sites, members)
    assert result.site_scope_review_required.all()
    assert ledger.audit_scope_resolution_reason.eq("source_geometry_evidence_unavailable").all()
    assert not members.operational_units_counted.any()
