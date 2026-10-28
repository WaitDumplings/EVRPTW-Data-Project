"""Read-only source-identity QA for unresolved residential site scope.

The reported groups are ambiguity candidates, never confirmed duplicate sites
or errors. No geometry clustering, merging, deletion or eligibility change is
performed. A source component/roof may legitimately have complicated scope.
"""

from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from .util import sha256_file, write_json

SCHEMA = "evrptw_site_scope_identity_audit_v1"
SCOPE_POLICY = "source_scope_identity_guard_v1"
SPLIT_OVERLAP_TOLERANCE_M2 = 0.01
SITE_METADATA = (
    "site_source_key",
    "site_identity_source",
    "service_location_type",
    "coarse_residential_type",
    "site_scope_unresolved",
    "site_scope_ambiguous",
    "residential_units_scope",
    "site_scope_review_required",
    "site_scope_review_reason",
    "site_scope_status",
    "site_scope_policy",
    "site_supported_building_split",
    "site_complex_boundary_ambiguous",
)
AUDIT_COLUMNS = (
    "audit_candidate_kind",
    "audit_identity_key",
    "audit_site_count",
    "audit_complex_site_count",
    "audit_fallback_site_count",
    "audit_member_row",
)


def _identities(values):
    result = values.astype("string").str.strip()
    return result.mask(result.isin(["", "nan", "None", "<NA>", "null"]))


def _owner_kind(sites):
    """Classify canonical source evidence; unknown kinds are not called fallback."""
    kind = pd.Series("unknown", index=sites.index, dtype="string")
    if "site_identity_source" in sites:
        source = sites.site_identity_source
        kind.loc[source.eq("osm_explicit_residential_complex")] = "complex"
        kind.loc[
            source.isin(
                [
                    "osm_supported_independent_or_provisional_site",
                    "microsoft_nsi_independent_or_provisional_site",
                ]
            )
        ] = "fallback"
    if "site_source_key" in sites:
        source = sites.site_source_key.astype("string").fillna("")
        from_key = pd.Series("unknown", index=sites.index, dtype="string")
        from_key.loc[source.str.startswith("complex:")] = "complex"
        from_key.loc[source.str.startswith(("msft:", "osm_building:"))] = "fallback"
        conflict = kind.ne("unknown") & from_key.ne("unknown") & kind.ne(from_key)
        if conflict.any():
            raise ValueError("Site source key and identity source disagree about complex scope")
        kind = kind.where(from_key.eq("unknown"), from_key)
    return kind


def audit_site_scope(sites, members):
    """Return a summary and detailed candidate ledger without modifying inputs.

    Required sites columns: site_id, and at least one of site_source_key or
    site_identity_source. Required members columns: site_id, fd_id,
    nsi_component_id, composite_building_id. All member columns are retained in
    the ledger; available site type/scope metadata is added with audit_ prefixes.
    Each member appears once per candidate group it contributes to.
    """
    if not {"site_id"}.issubset(sites.columns) or not {
        "site_source_key",
        "site_identity_source",
    }.intersection(sites.columns):
        raise ValueError("Sites require site_id and site_source_key or site_identity_source")
    required = {"site_id", "fd_id", "nsi_component_id", "composite_building_id"}
    missing = required - set(members.columns)
    if missing:
        raise ValueError("Members lack required columns: " + ", ".join(sorted(missing)))
    if any(str(name).startswith("audit_") for name in members.columns):
        raise ValueError("Member columns beginning audit_ are reserved for QA output")
    sites = sites.copy().reset_index(drop=True)
    members = members.copy().reset_index(drop=True)
    site_keys = _identities(sites.site_id)
    member_sites = _identities(members.site_id)
    if site_keys.isna().any() or site_keys.duplicated().any():
        raise ValueError("Sites must have unique, nonempty site_id values")
    if member_sites.isna().any() or not member_sites.isin(site_keys).all():
        raise ValueError("Every member must refer to a known, nonempty site_id")
    owner_kind = pd.Series(_owner_kind(sites).to_numpy(), index=site_keys)
    work = pd.DataFrame(
        {
            "site": member_sites,
            "component": _identities(members.nsi_component_id),
            "roof": _identities(members.composite_building_id),
        }
    )
    work["kind"] = work.site.map(owner_kind)

    def owner_counts(key):
        unique = work.loc[work[key].notna(), [key, "site", "kind"]].drop_duplicates()
        unique["complex"] = unique.kind.eq("complex").astype(int)
        unique["fallback"] = unique.kind.eq("fallback").astype(int)
        return unique.groupby(key, sort=True).agg(
            site_count=("site", "size"),
            complex_site_count=("complex", "sum"),
            fallback_site_count=("fallback", "sum"),
        )

    component = owner_counts("component")
    component = component.loc[component.site_count.gt(1)]
    roof = owner_counts("roof")
    roof = roof.loc[roof.complex_site_count.gt(0) & roof.fallback_site_count.gt(0)]
    frames = []
    affected_member_rows = set()
    for key, counts, label in [
        ("component", component, "source_component_spans_sites"),
        ("roof", roof, "microsoft_roof_spans_complex_and_fallback"),
    ]:
        selected = work[key].isin(counts.index)
        rows = members.loc[selected].copy()
        keys = work.loc[selected, key]
        rows["audit_candidate_kind"] = label
        rows["audit_identity_key"] = keys.to_numpy()
        rows["audit_member_row"] = rows.index
        affected_member_rows.update(rows.index)
        for column in ["site_count", "complex_site_count", "fallback_site_count"]:
            rows["audit_" + column] = keys.map(counts[column]).to_numpy()
        for column in SITE_METADATA:
            if column in sites:
                values = pd.Series(sites[column].to_numpy(), index=site_keys)
                rows["audit_" + column] = member_sites.loc[selected].map(values).to_numpy()
        frames.append(rows)
    ledger = pd.concat(frames, ignore_index=True)
    metadata = ["audit_" + name for name in SITE_METADATA if name in sites]
    ledger = ledger[list(AUDIT_COLUMNS) + list(members.columns) + metadata]
    ledger = ledger.sort_values(
        ["audit_candidate_kind", "audit_identity_key", "audit_member_row"], kind="stable"
    ).reset_index(drop=True)
    summary = {
        "schema": SCHEMA,
        "interpretation": "source-identity ambiguity candidates; not confirmed duplicates or errors",
        "operation": "read_only_no_merge_delete_or_eligibility_change",
        "coverage": "identity QA only; does not detect every geometric or real-world scope conflict",
        "counts": {
            "input_sites": len(sites),
            "input_member_rows": len(members),
            "sites_with_unknown_owner_kind": int(owner_kind.eq("unknown").sum()),
            "member_rows_missing_component_id": int(work.component.isna().sum()),
            "member_rows_missing_composite_building_id": int(work.roof.isna().sum()),
            "source_components_spanning_sites": len(component),
            "microsoft_roofs_spanning_complex_and_fallback": len(roof),
            "affected_sites": int(member_sites.loc[list(affected_member_rows)].nunique()),
            "affected_member_rows": len(affected_member_rows),
            "candidate_ledger_rows": len(ledger),
        },
        "ledger_row_policy": "one original member row per candidate kind and identity group",
        "retained_member_columns": list(members.columns),
        "available_site_metadata": [name for name in SITE_METADATA if name in sites],
    }
    if "site_scope_review_required" in sites:
        summary["counts"]["input_sites_pending_scope_review"] = int(
            sites.site_scope_review_required.sum()
        )
    if "site_supported_building_split" in sites:
        summary["counts"]["input_sites_participating_in_supported_building_splits"] = int(
            sites.site_supported_building_split.sum()
        )
    return summary, ledger


def _supported_roof_split(rows, sites, buildings, complexes, *, area_crs, cross_components):
    """Require source-backed, distinct building extents on both sides of a complex.

    A shared Microsoft parent alone cannot decide whether sites are duplicates.
    This narrowly recognizes the existing OSM building split when complex-owned
    buildings are wholly covered, fallback buildings are separate, and no source
    identity or explicit relation member contradicts that separation.
    """
    if buildings is None or complexes is None:
        return False, "source_geometry_evidence_unavailable"
    if rows.nsi_component_id.astype("string").isin(cross_components).any():
        return False, "source_component_cross_site_owner"
    owners = sites.loc[rows.site_id.unique()]
    keys = owners.site_source_key.astype("string")
    fallback = owners.loc[~keys.str.startswith("complex:")]
    comp = owners.loc[keys.str.startswith("complex:")]
    if not fallback.site_source_key.str.startswith("osm_building:").all():
        return False, "fallback_retains_shared_microsoft_extent"
    osm = _identities(rows.osm_building_id)
    if osm.isna().any() or not osm.isin(buildings.index).all():
        return False, "source_building_membership_incomplete"
    for row in fallback.itertuples():
        expected = row.site_source_key.removeprefix("osm_building:")
        member = rows.loc[rows.site_id.eq(row.Index)]
        if not _identities(member.osm_building_id).eq(expected).all():
            return False, "fallback_building_membership_disagrees"
        if not row.geometry.equals(buildings.loc[expected].geometry):
            return False, "fallback_extent_differs_from_source_building"
    metric = gpd.GeoSeries(owners.geometry, crs=sites.crs).to_crs(area_crs)
    for crow in comp.itertuples():
        complex_id = crow.site_source_key.removeprefix("complex:")
        if complex_id not in complexes.index:
            return False, "complex_source_evidence_unavailable"
        source = complexes.loc[complex_id]
        if not crow.geometry.equals(source.geometry):
            return False, "complex_extent_differs_from_source"
        ids = _identities(rows.loc[rows.site_id.eq(crow.Index), "osm_building_id"])
        if not all(crow.geometry.covers(buildings.loc[key].geometry) for key in ids):
            return False, "complex_boundary_cuts_owned_source_building"
        member_ids = set(json.loads(source.member_osm_ids_json))
        for frow in fallback.itertuples():
            fallback_id = frow.site_source_key.removeprefix("osm_building:")
            if fallback_id in member_ids:
                return False, "explicit_complex_membership_competes_with_fallback"
            area = metric.loc[crow.Index].intersection(metric.loc[frow.Index]).area
            if area > SPLIT_OVERLAP_TOLERANCE_M2:
                return False, "fallback_source_building_overlaps_complex"
    return True, "distinct_source_buildings_and_complex_extent"


def annotate_site_scope(sites, members, *, osm_buildings=None, complexes=None, area_crs=None):
    """Flag unresolved source scope while preserving every site and source record.

    Returns copied sites/members, a candidate ledger and a summary. Unresolved
    owner competition is pending for default use. Its operational unit estimate
    is cleared, while raw NSI evidence is retained. Missing units or provisional
    non-complex scope alone is not a reason to make a location pending.
    """
    sites, members = sites.copy().reset_index(drop=True), members.copy().reset_index(drop=True)
    if "residential_units" not in sites or not {
        "operational_units_counted",
        "counted_residential_units",
    }.issubset(members):
        raise ValueError(
            "Scope annotation requires canonical nullable units and counted-member fields"
        )
    report, ledger = audit_site_scope(sites, members)
    if "site_source_key" not in sites:
        raise ValueError("Scope annotation requires canonical site_source_key")
    if "osm_building_id" not in members:
        raise ValueError("Scope annotation requires member osm_building_id")
    by_site = sites.set_index("site_id", drop=False)
    buildings = None
    if osm_buildings is not None:
        buildings = osm_buildings.loc[osm_buildings.usable].set_index("osm_id")
    source_complexes = complexes.set_index("osm_id") if complexes is not None else None
    if buildings is not None and source_complexes is not None and area_crs is None:
        raise ValueError("Metric area_crs is required for source building-split evidence")
    component_rows = ledger.audit_candidate_kind.eq("source_component_spans_sites")
    cross_components = set(ledger.loc[component_rows, "audit_identity_key"])
    resolutions = {}
    for (kind, identity), rows in ledger.groupby(
        ["audit_candidate_kind", "audit_identity_key"], sort=True
    ):
        supported, reason = False, "source_component_cross_site_owner"
        if kind == "microsoft_roof_spans_complex_and_fallback":
            supported, reason = _supported_roof_split(
                rows,
                by_site,
                buildings,
                source_complexes,
                area_crs=area_crs,
                cross_components=cross_components,
            )
        resolutions[(kind, identity)] = (supported, reason)
    ledger["audit_supported_building_split"] = pd.Series(
        [
            resolutions[(kind, identity)][0]
            for kind, identity in zip(ledger.audit_candidate_kind, ledger.audit_identity_key)
        ],
        index=ledger.index,
        dtype=bool,
    )
    ledger["audit_scope_resolution_reason"] = [
        resolutions[(kind, identity)][1]
        for kind, identity in zip(ledger.audit_candidate_kind, ledger.audit_identity_key)
    ]
    ledger["audit_scope_resolution"] = ledger.audit_supported_building_split.map(
        {True: "supported_source_building_split", False: "review_required_source_scope"}
    )
    ambiguous = sites.get(
        "site_complex_boundary_ambiguous",
        sites.get("site_scope_ambiguous", pd.Series(False, index=sites.index)),
    ).fillna(False)
    sites["site_complex_boundary_ambiguous"] = ambiguous.astype(bool)
    reasons = {key: set() for key in sites.site_id}
    for key in sites.loc[ambiguous, "site_id"]:
        reasons[key].add("competing_complex_boundaries")
    pending = ledger.loc[~ledger.audit_supported_building_split]
    for row in pending.itertuples():
        reasons[row.site_id].add(row.audit_scope_resolution_reason)
    sites["site_scope_review_reason"] = sites.site_id.map(
        {key: "|".join(sorted(value)) for key, value in reasons.items()}
    )
    sites["site_scope_review_required"] = sites.site_scope_review_reason.ne("")
    sites["site_scope_ambiguous"] = sites.site_scope_review_required
    supported_sites = set(ledger.loc[ledger.audit_supported_building_split, "site_id"])
    sites["site_supported_building_split"] = sites.site_id.isin(supported_sites)
    sites["site_scope_policy"] = SCOPE_POLICY
    sites["site_scope_status"] = "no_detected_source_owner_competition"
    sites.loc[sites.site_supported_building_split, "site_scope_status"] = (
        "supported_source_building_split"
    )
    sites.loc[sites.site_scope_review_required, "site_scope_status"] = (
        "review_required_source_scope"
    )
    cleared = sites.site_scope_review_required & sites.residential_units.notna()
    cleared_units = float(sites.loc[cleared, "residential_units"].sum())
    sites.loc[sites.site_scope_review_required, "residential_units"] = pd.NA
    for field, value in {
        "residential_units_scope": "unresolved",
        "residential_units_source": "unresolved_source_owner_competition",
        "residential_units_confidence": "unknown",
        "residential_units_aggregation_status": "not_summed_unresolved",
        "residential_unit_band": "unknown",
        "unit_evidence_tier": "units_unknown",
    }.items():
        if field in sites:
            sites.loc[sites.site_scope_review_required, field] = value
    pending_members = members.site_id.isin(sites.loc[sites.site_scope_review_required, "site_id"])
    members.loc[pending_members, "operational_units_counted"] = False
    members.loc[pending_members, "counted_residential_units"] = pd.NA
    # Refresh ledger scope and unit annotations after clearing operational counts.
    for column in SITE_METADATA:
        if column in sites:
            ledger["audit_" + column] = ledger.site_id.map(sites.set_index("site_id")[column])
    for column in ["operational_units_counted", "counted_residential_units"]:
        ledger[column] = ledger.audit_member_row.map(members.reset_index(drop=True)[column])
    report["scope_policy"] = SCOPE_POLICY
    report["operation"] = "source_scope_annotation_no_merge_or_delete_default_pending_guard"
    report["scope_resolution"] = {
        "supported_building_split_groups": sum(value[0] for value in resolutions.values()),
        "unresolved_candidate_groups": sum(not value[0] for value in resolutions.values()),
        "review_required_sites": int(sites.site_scope_review_required.sum()),
        "supported_building_split_sites": int(sites.site_supported_building_split.sum()),
        "unit_estimate_sites_cleared": int(cleared.sum()),
        "modeled_unit_estimate_sum_cleared": cleared_units,
        "split_overlap_numerical_tolerance_m2": SPLIT_OVERLAP_TOLERANCE_M2,
        "resolution_rule": "source evidence only; no site merging, deletion or added source geometry",
    }
    return sites, members, ledger, report


def apply_site_scope_eligibility(sites):
    """Combine proved road access with unresolved source-scope exclusion."""
    result = sites.copy()
    result["cle_candidate_eligible"] = (
        result.road_access_candidate_eligible & ~result.site_scope_review_required
    )
    result["cle_default_instance_eligible"] = result.cle_candidate_eligible
    return result


def audit_saved_site_scope(*, sites_path, members_path, output_dir):
    """Audit saved Parquet tables and write a new JSON/CSV bundle only."""
    sites_path, members_path = Path(sites_path).resolve(), Path(members_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a fresh audit output directory: " + str(output_dir))
    available = set(pq.ParquetFile(sites_path).schema.names)
    columns = [name for name in ["site_id", *SITE_METADATA] if name in available]
    sites = pd.read_parquet(sites_path, columns=columns)
    members = pd.read_parquet(members_path)
    report, ledger = audit_site_scope(sites, members)
    report["inputs"] = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in [("sites", sites_path), ("members", members_path)]
    }
    report["generation_dependencies"] = {Path(__file__).name: sha256_file(Path(__file__))}
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = output_dir / "site_scope_ambiguity_candidates.csv"
    ledger.to_csv(ledger_path, index=False)
    report["outputs"] = {
        "candidate_ledger": {"path": ledger_path.name, "sha256": sha256_file(ledger_path)}
    }
    write_json(output_dir / "site_scope_audit.json", report)
    return report
