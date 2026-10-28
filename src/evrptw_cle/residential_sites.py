"""Canonical residential delivery sites, separate from building and household counts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

from .context_validation import validate_context_bindings
from .customer_access import (
    _threshold_summary,
    _write_service_access_contract,
    attach_footprint_access,
    build_eligible_physical_edges,
)
from .customer_spatial import residential_unit_band
from .protected_connectivity import (
    build_directed_component_index,
    connectivity_summary,
    label_projection_connectivity,
)
from .public_locations import filter_public_locations
from .residential_identity import source_groups
from .residential_types import TYPE_ORDER, classify_nsi_occtypes, nsi_occtype_type
from .site_scope_audit import SCOPE_POLICY, annotate_site_scope, apply_site_scope_eligibility
from .util import sha256_file, write_json

SITE_POLICY = "identified_residential_site_v3"


def validate_site_tables(sites, members):
    """Validate source ownership and nullable household estimates independently."""
    errors = []
    if (
        sites.site_id.duplicated().any()
        or not sites.site_id.eq(sites.latent_service_location_id).all()
    ):
        errors.append("Customer site identities must be unique and canonical")
    if members.fd_id.duplicated().any():
        errors.append("NSI source records have duplicate site owners")
    component_owners = members.groupby("nsi_component_id").site_id.nunique()
    cross_components = component_owners.index[component_owners.gt(1)]
    cross_members = members.nsi_component_id.isin(cross_components)
    if members.loc[cross_members, "operational_units_counted"].any():
        errors.append("A source component crossing site owners has operational units")
    pending = sites.get("site_scope_review_required", pd.Series(False, index=sites.index))
    pending_ids = set(sites.loc[pending, "site_id"])
    if not set(members.loc[cross_members, "site_id"]).issubset(pending_ids):
        errors.append("A source component crossing site owners lacks a scope review flag")
    if sites.loc[pending, "residential_units"].notna().any():
        errors.append("A site with unresolved owner scope has an operational unit estimate")
    for field in ["cle_candidate_eligible", "cle_default_instance_eligible"]:
        if field in sites and sites.loc[pending, field].any():
            errors.append("A site with unresolved owner scope is default eligible")
    if set(members.site_id) != set(sites.site_id):
        errors.append("Residential-site membership and parent tables differ")
    units = pd.to_numeric(sites.residential_units, errors="coerce")
    if (units.notna() & (~np.isfinite(units) | units.le(0) | units.mod(1).ne(0))).any():
        errors.append("Observed residential units must be positive finite integers")
    counted = members.loc[members.operational_units_counted]
    if counted.nsi_component_id.duplicated().any():
        errors.append("A source household component was counted more than once")
    sums = counted.groupby("site_id").counted_residential_units.sum()
    expected = sites.set_index("site_id").residential_units.dropna()
    if (
        set(sums.index) != set(expected.index)
        or not sums.reindex(expected.index).eq(expected).all()
    ):
        errors.append("Residential units differ from their unique source component owners")
    if members.loc[~members.operational_units_counted, "counted_residential_units"].notna().any():
        errors.append("Uncounted source rows contain operational household estimates")
    points = gpd.GeoSeries(
        gpd.points_from_xy(sites.location_lon, sites.location_lat), index=sites.index, crs=4326
    )
    if not points.covered_by(sites.geometry, align=False).all():
        errors.append("A customer representative point lies outside its site")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "passed": True,
        "unique_site_ids": len(sites),
        "unique_nsi_owners": len(members),
        "known_unit_sites": len(expected),
        "counted_units": float(expected.sum()),
    }


def build_registered_site_layer(
    *, registry_path, city_slug, boundary_path, graph_path, output_dir, access_context=None
):
    """Build from versioned inputs, or reuse only a complete hash-matched layer."""
    registry_path = Path(registry_path).resolve()
    context_manifest = Path(graph_path).parent / "access_context_manifest.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("policy") != SITE_POLICY:
        raise ValueError("Customer registry does not use the current site policy")
    entry = registry["cities"][city_slug]
    source_root = registry_path.parent.parent
    sources = {}
    for name, record in entry["sources"].items():
        path = source_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError("Missing or changed frozen customer source: " + name)
        sources[name] = path
    if sha256_file(Path(boundary_path)) != entry["boundary_sha256"]:
        raise ValueError("Customer crosswalk boundary differs from requested city boundary")
    for name, record in entry.get("provenance", {}).items():
        path = source_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError("Missing or changed frozen customer provenance: " + name)
    if context_manifest.is_file():
        manifest = json.loads(context_manifest.read_text())
        validate_context_bindings(
            manifest,
            city_slug=city_slug,
            boundary_path=boundary_path,
            mask_path=sources["public_masks"],
            area_crs=entry["area_crs"],
        )
        if access_context is None:
            from .local_road_access import LocalAccessContext

            access_context = LocalAccessContext.load(context_manifest.parent)
    expected = {
        **{name: record["sha256"] for name, record in entry["sources"].items()},
        "boundary": entry["boundary_sha256"],
        "graph": sha256_file(Path(graph_path)),
    }
    if access_context is not None:
        expected["local_access_context"] = sha256_file(context_manifest)
    manifest_path = Path(output_dir) / "residential_site_manifest.json"
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text())
        same_inputs = all(
            recorded.get("inputs", {}).get(name, {}).get("sha256") == digest
            for name, digest in expected.items()
        )
        same_code = all(
            sha256_file(Path(__file__).with_name(name)) == digest
            for name, digest in recorded.get("generation_dependencies", {}).items()
        ) and bool(recorded.get("generation_dependencies"))
        same_outputs = all(
            (manifest_path.parent / recorded["outputs"][name]).is_file()
            and sha256_file(manifest_path.parent / recorded["outputs"][name]) == digest
            for name, digest in recorded.get("output_sha256", {}).items()
        ) and bool(recorded.get("output_sha256"))
        if (
            same_inputs
            and same_code
            and same_outputs
            and recorded.get("policy") == SITE_POLICY
            and recorded.get("source_registry", {}).get("sha256") == sha256_file(registry_path)
        ):
            return recorded, False
        raise FileExistsError(
            "Customer layer is stale or incomplete; use a new customer-access root: "
            + str(output_dir)
        )
    report = build_residential_site_layer(
        city_slug=city_slug,
        records_path=sources["nsi_crosswalk"],
        footprints_path=sources["microsoft_footprints"],
        osm_buildings_path=sources["osm_buildings"],
        complexes_path=sources["osm_complexes"],
        boundary_path=Path(boundary_path),
        graph_path=Path(graph_path),
        mask_path=sources["public_masks"],
        output_dir=Path(output_dir),
        area_crs=entry["area_crs"],
        access_context=access_context,
    )
    if access_context is not None:
        report["inputs"]["local_access_context"] = {
            "path": str(context_manifest),
            "sha256": sha256_file(context_manifest),
        }
    report["source_registry"] = {"path": str(registry_path), "sha256": sha256_file(registry_path)}
    report["source_vintages"] = registry["source_vintages"]
    report["source_provenance"] = {
        name: {"path": str(source_root / record["path"]), "sha256": record["sha256"]}
        for name, record in entry.get("provenance", {}).items()
    }
    write_json(Path(output_dir) / "residential_site_manifest.json", report)
    write_json(Path(output_dir) / "footprint_road_access_audit.json", report)
    return report, True


def _unique_containment(points, areas):
    counts = np.zeros(len(points), dtype=np.int64)
    target = np.full(len(points), -1, dtype=np.int64)
    if not len(points) or not len(areas):
        return counts, target
    pairs = shapely.STRtree(areas.geometry.to_numpy()).query(
        points.geometry.to_numpy(), predicate="covered_by"
    )
    counts = np.bincount(pairs[0], minlength=len(points))
    good = counts[pairs[0]] == 1
    target[pairs[0, good]] = pairs[1, good]
    return counts, target


def canonical_complexes(frame):
    frame = frame.copy().reset_index(drop=True)
    if frame.empty:
        return frame
    frame["_key"] = shapely.to_wkb(shapely.normalize(frame.geometry.to_numpy()))
    frame["_relation"] = frame.site_evidence.str.startswith("explicit_site_relation")
    frame = (
        frame.sort_values(["_relation", "osm_id"], ascending=[False, True], kind="stable")
        .drop_duplicates("_key")
        .drop(columns=["_key", "_relation"])
    )
    # Explicit relation membership can identify a duplicate child area; spatial
    # containment or a shared name alone never merges different complexes.
    claimed = {}
    for row in frame.itertuples():
        if row.site_evidence.startswith("explicit_site_relation"):
            for member in json.loads(row.member_osm_ids_json):
                claimed.setdefault(member, []).append(row.osm_id)
    remove = []
    byid = frame.set_index("osm_id")
    for child, parents in claimed.items():
        if (
            child in byid.index
            and len(parents) == 1
            and byid.loc[parents[0]].geometry.covers(byid.loc[child].geometry)
        ):
            remove.append(child)
    return frame.loc[~frame.osm_id.isin(remove)].reset_index(drop=True)


def _component_unit_evidence(records, geometries):
    """Count a source component once; unresolved or overlapping scopes stay null."""
    x = records
    grp = x.groupby("_component", sort=False)
    component = grp.agg(
        site=("_site_key", "first"),
        site_options=("_site_key", "nunique"),
        raw_min=("_positive_units", "min"),
        raw_max=("_positive_units", "max"),
        geometry_key=("_unit_geometry_key", "first"),
        geometry_options=("_unit_geometry_key", "nunique"),
        all_house=("_is_house", "all"),
    )
    component["estimate"] = component.raw_min.where(component.raw_min.eq(component.raw_max))
    inferred = component.estimate.isna() & component.raw_max.isna() & component.all_house
    component.loc[inferred, "estimate"] = 1.0
    component["unit_source"] = np.where(
        inferred, "nsi_single_family_code", "nsi_resunits_consensus"
    )
    supported = (
        component.site_options.eq(1)
        & component.geometry_options.eq(1)
        & component.geometry_key.ne("")
    )
    # Distinct NSI identities in the same geometry are not automatically distinct
    # household components. Keep the raw record sum, but do not add their units.
    owners = component.loc[supported].groupby("geometry_key").size()
    supported &= component.geometry_key.map(owners).eq(1)
    used = component.loc[supported, "geometry_key"].drop_duplicates()
    shapes = gpd.GeoSeries(used.map(geometries).to_numpy(), index=used.to_numpy(), crs=4326)
    if len(shapes):
        pairs = shapely.STRtree(shapes.to_numpy()).query(shapes.to_numpy(), predicate="intersects")
        p = pairs[:, pairs[0] < pairs[1]]
        if p.shape[1]:
            overlaps = (
                shapely.area(shapely.intersection(shapes.to_numpy()[p[0]], shapes.to_numpy()[p[1]]))
                > 1e-14
            )
            bad = set(shapes.index[np.unique(p[:, overlaps])])
            supported &= ~component.geometry_key.isin(bad)
    component["scope_supported"] = supported & component.estimate.notna()
    component.loc[~component.scope_supported, "estimate"] = np.nan
    return component


def infer_residential_sites(
    records,
    footprints,
    osm_buildings,
    complexes,
    *,
    city_slug,
    boundary,
    masks=None,
    area_crs="EPSG:32618",
    return_scope_audit=False,
):
    """Infer one identity per explicit complex or provisional independent site.

    Ordinary-residential NSI source points may be inside a complex courtyard.
    Missing units do not change location eligibility. No building/area count is
    exported as a scale or substituted for residential units.
    """
    x = records.sort_values("fd_id", kind="stable").reset_index(drop=True).copy()
    if x.fd_id.duplicated().any():
        raise ValueError("NSI fd_id must have one source record")
    x = x.loc[x.geometry.covered_by(boundary)].reset_index(drop=True)
    if "composite_building_id" not in x:
        x["composite_building_id"] = x.get("microsoft_building_id", "")
    if "composite_status" not in x:
        x["composite_status"] = np.where(
            x.get("geometry_evidence_tier", "").eq("G1_containment"),
            "matched_unique_containment",
            "unmatched",
        )
    x["composite_building_id"] = x.composite_building_id.fillna("").astype(str)
    ordinary = x.occtype.map(nsi_occtype_type).notna()
    x = x.loc[ordinary].reset_index(drop=True)
    safe, source_filter = filter_public_locations(x[["fd_id", "geometry"]], masks)
    x = x.loc[x.fd_id.isin(safe.fd_id)].reset_index(drop=True)
    complexes, complex_filter = filter_public_locations(complexes, masks)
    complexes = canonical_complexes(complexes)
    buildings = (
        osm_buildings.loc[osm_buildings.usable].reset_index(drop=True)
        if len(osm_buildings)
        else osm_buildings
    )
    nc, nt = _unique_containment(x, complexes)
    bc, bt = _unique_containment(x, buildings)
    x["_complex_count"] = nc
    x["_complex_id"] = ""
    x["_osm_id"] = ""
    if len(complexes):
        x.loc[nt >= 0, "_complex_id"] = complexes.osm_id.to_numpy()[nt[nt >= 0]]
    if len(buildings):
        x.loc[bt >= 0, "_osm_id"] = buildings.osm_id.to_numpy()[bt[bt >= 0]]
    x["_component"] = source_groups(x, x.geometry.get_coordinates().to_numpy(), 0.0)
    g1 = x.composite_status.eq("matched_unique_containment")
    x["_has_osm"] = x._osm_id.ne("")
    roof = (
        x.loc[g1]
        .groupby("composite_building_id")
        .agg(all_osm=("_has_osm", "all"), osm_options=("_osm_id", "nunique"))
    )
    bridges = x.loc[x._osm_id.ne("")].groupby("_component")._osm_id.nunique()
    bridge_parents = set(
        x.loc[x._component.isin(bridges.loc[bridges.gt(1)].index), "composite_building_id"]
    )
    split = set(
        roof.index[roof.all_osm & roof.osm_options.gt(1) & ~roof.index.isin(bridge_parents)]
    )
    x["_site_key"] = ""
    x.loc[g1, "_site_key"] = "msft:" + x.loc[g1, "composite_building_id"]
    splitmask = g1 & x.composite_building_id.isin(split)
    x.loc[splitmask, "_site_key"] = "osm_building:" + x.loc[splitmask, "_osm_id"]
    baseline_site_count = int(x.loc[x._site_key.ne(""), "_site_key"].nunique())
    courtyard_record_count = int(((nc == 1) & ~g1).sum())
    # Explicit complex identity overrides roofs; generic landuse is not in this table.
    in_complex = nc == 1
    x.loc[in_complex, "_site_key"] = "complex:" + x.loc[in_complex, "_complex_id"]
    x["_unit_geometry_key"] = ""
    x.loc[g1, "_unit_geometry_key"] = "msft:" + x.loc[g1, "composite_building_id"]
    osm_known = x._osm_id.ne("")
    x.loc[osm_known, "_unit_geometry_key"] = "osm_building:" + x.loc[osm_known, "_osm_id"]
    assigned = x._site_key.ne("")
    unassigned = int((~assigned).sum())
    x = x.loc[assigned].reset_index(drop=True)
    fp = footprints.set_index("building_id")
    ob = buildings.set_index("osm_id")
    cc = complexes.set_index("osm_id")
    geometries = {"msft:" + str(k): v for k, v in zip(fp.index, fp.geometry)}
    geometries.update({"osm_building:" + str(k): v for k, v in zip(ob.index, ob.geometry)})
    geometries.update({"complex:" + str(k): v for k, v in zip(cc.index, cc.geometry)})
    numeric = pd.to_numeric(x.resunits, errors="coerce")
    x["_positive_units"] = numeric.where(np.isfinite(numeric) & numeric.gt(0))
    x["_is_house"] = x.occtype.map(nsi_occtype_type).eq("house")
    components = _component_unit_evidence(x, geometries)
    x["_complex_ambiguous"] = x._complex_count.gt(1)
    grouped = x.groupby("_site_key", sort=True)
    sites = grouped.agg(
        nsi_modeled_unit_source_sum=("modeled_units", "sum"),
        raw_nsi_resunits_sum=("resunits", "sum"),
        nsi_record_count=("fd_id", "size"),
        site_scope_ambiguous=("_complex_ambiguous", "any"),
    )
    cg = components.groupby("site")
    totals = cg.estimate.sum(min_count=1)
    all_supported = cg.scope_supported.all()
    sites["residential_units"] = totals.where(all_supported).reindex(sites.index).astype("Float64")
    sites["residential_units_scope"] = np.where(
        sites.residential_units.notna(), "non_overlapping_building_components", "unresolved"
    )
    sites["residential_units_source"] = np.where(
        sites.residential_units.notna(),
        "nsi_component_consensus",
        "unresolved_source_scope_or_missing_units",
    )
    sites["residential_units_confidence"] = np.where(
        sites.residential_units.notna(), "modeled_source_supported", "unknown"
    )
    sites["residential_units_aggregation_status"] = np.where(
        sites.residential_units.notna(),
        "deduplicated_supported_components",
        "not_summed_unresolved",
    )
    sites["residential_unit_band"] = sites.residential_units.map(
        lambda v: "unknown" if pd.isna(v) else residential_unit_band(int(v))
    )
    chosen = (
        x.sort_values(["_positive_units", "fd_id"], ascending=[False, True], kind="stable")
        .drop_duplicates("_site_key")
        .set_index("_site_key")
    )
    sites["representative_fd_id"] = chosen.fd_id
    sites["site_id"] = [
        "cus_site_" + city_slug + "_" + hashlib.sha256(v.encode()).hexdigest()[:20]
        for v in sites.index
    ]
    sites["latent_service_location_id"] = sites.site_id
    sites["site_identity_source"] = np.where(
        sites.index.str.startswith("complex:"),
        "osm_explicit_residential_complex",
        np.where(
            sites.index.str.startswith("osm_building:"),
            "osm_supported_independent_or_provisional_site",
            "microsoft_nsi_independent_or_provisional_site",
        ),
    )
    sites["customer_location_policy"] = SITE_POLICY
    patterns = x[["_site_key", "occtype"]].drop_duplicates()
    patterns["_type"] = patterns.occtype.map(nsi_occtype_type)
    bit = {v: 1 << i for i, v in enumerate(TYPE_ORDER)}
    patterns = patterns[["_site_key", "_type"]].drop_duplicates()
    patterns["_bit"] = patterns._type.map(bit)
    masks_by_site = patterns.groupby("_site_key")._bit.sum()
    code = {
        "house": "RES1-1SNB",
        "manufactured_home": "RES2",
        "small_apt": "RES3A",
        "medium_apt": "RES3C",
        "large_apt": "RES3E",
    }
    evidence = {
        v: classify_nsi_occtypes([code[t] for t in TYPE_ORDER if v & bit[t]])
        for v in masks_by_site.unique()
    }
    for field in [
        "service_location_type",
        "coarse_residential_type",
        "type_conflict",
        "coarse_type_conflict",
        "type_label_resolved",
    ]:
        sites[field] = masks_by_site.map({v: getattr(e, field) for v, e in evidence.items()})
    sites["site_scope_unresolved"] = (
        ~sites.index.str.startswith("complex:")
    ) & ~sites.service_location_type.eq("house")
    sites["geometry"] = sites.index.map(geometries)
    sites = gpd.GeoDataFrame(
        sites.reset_index(names="site_source_key"), geometry="geometry", crs=4326
    )
    if sites.geometry.isna().any():
        raise ValueError("Residential site lacks a source extent")
    rp = sites.geometry.representative_point()
    inside = rp.covered_by(boundary)
    if not inside.all():
        rp.loc[~inside] = sites.loc[~inside].geometry.intersection(boundary).representative_point()
    sites["representative_longitude"] = rp.x
    sites["representative_latitude"] = rp.y
    sites["location_lon"] = rp.x
    sites["location_lat"] = rp.y
    sites["geometry_evidence_tier"] = np.where(
        sites.site_identity_source.eq("osm_explicit_residential_complex"),
        "G_site_complex",
        "G1_containment",
    )
    sites["unit_evidence_tier"] = np.where(
        sites.residential_units.notna(), "source_component_estimate", "units_unknown"
    )
    sites["mixed_use_flag"] = False
    sites["mixed_use_evidence"] = "not_assessed_from_residential_records_only"
    sites["active_customer"] = False
    sites["customer_release_eligible"] = False
    # Reapply host geometry masks, then export only members of public-safe sites.
    sites, site_filter = filter_public_locations(sites, masks)
    owner = sites.set_index("site_source_key").site_id
    x["site_id"] = x._site_key.map(owner)
    x = x.loc[x.site_id.notna()].copy()
    ownership_columns = [
        "fd_id",
        "site_id",
        "_component",
        "composite_building_id",
        "_osm_id",
        "occtype",
        "resunits",
        "modeled_units",
    ]
    ownership_columns += [
        name for name in ["bid", "ftprntsrc", "ftprntid", "usastrucid", "cbfips"] if name in x
    ]
    ownership = x[ownership_columns].rename(
        columns={"_component": "nsi_component_id", "_osm_id": "osm_building_id"}
    )
    # A unit estimate has exactly one canonical owner row. Duplicate NSI source
    # records remain inspectable without becoming additional households.
    known_sites = set(sites.loc[sites.residential_units.notna(), "site_id"])
    ownership["operational_units_counted"] = (
        ownership.site_id.isin(known_sites)
        & ownership.nsi_component_id.map(components.scope_supported).fillna(False)
        & ~ownership.nsi_component_id.duplicated()
    )
    ownership["counted_residential_units"] = (
        ownership.nsi_component_id.map(components.estimate)
        .where(ownership.operational_units_counted)
        .astype("Float64")
    )
    sites, ownership, scope_ledger, scope_report = annotate_site_scope(
        sites, ownership, osm_buildings=buildings, complexes=complexes, area_crs=area_crs
    )
    result = (
        sites,
        ownership,
        {
            "policy": SITE_POLICY,
            "source_filter": source_filter,
            "complex_filter": complex_filter,
            "site_filter": site_filter,
            "safe_ordinary_nsi_records": source_filter["retained_count"],
            "unassigned_safe_records": unassigned,
            "before_complex_geometry_candidate_sites": baseline_site_count,
            "complex_courtyard_or_other_non_msft_records": courtyard_record_count,
            "potential_sites": len(sites),
            "identified_complexes": int(
                sites.site_identity_source.eq("osm_explicit_residential_complex").sum()
            ),
            "site_scope_unresolved_count": int(sites.site_scope_unresolved.sum()),
            "ambiguous_complex_scope_sites": int(sites.site_complex_boundary_ambiguous.sum()),
            "site_scope_review_required_count": int(sites.site_scope_review_required.sum()),
            "site_scope_policy": SCOPE_POLICY,
            "site_scope_audit": scope_report,
            "known_residential_units_sites": int(sites.residential_units.notna().sum()),
            "known_residential_units_sum": float(sites.residential_units.sum()),
            "modeled_source_unit_sum": float(sites.nsi_modeled_unit_source_sum.sum()),
            "record_ownership_count": len(ownership),
            "unit_policy": "source component estimates counted once; uncertainty stays null; raw NSI unit mass is separate",
        },
    )
    return (*result, scope_ledger) if return_scope_audit else result


def build_residential_site_layer(
    *,
    city_slug,
    records_path,
    footprints_path,
    osm_buildings_path,
    complexes_path,
    boundary_path,
    graph_path,
    mask_path,
    output_dir,
    area_crs,
    graph=None,
    edges=None,
    component_index=None,
    access_context=None,
):
    from .public_locations import load_exclusion_masks

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = gpd.read_file(boundary_path).to_crs(4326).geometry.union_all()
    masks = load_exclusion_masks(mask_path)
    sites, members, summary, scope_ledger = infer_residential_sites(
        gpd.read_parquet(records_path),
        gpd.read_parquet(footprints_path),
        gpd.read_parquet(osm_buildings_path),
        gpd.read_parquet(complexes_path),
        city_slug=city_slug,
        boundary=boundary,
        masks=masks,
        area_crs=area_crs,
        return_scope_audit=True,
    )
    sites.to_parquet(output_dir / "residential_sites.parquet", index=False)
    if graph is None and access_context is not None:
        graph = access_context.core
    if graph is None:
        graph = ox.load_graphml(graph_path)
    if edges is None and access_context is None:
        edges = build_eligible_physical_edges(graph_path, area_crs, graph=graph)
    if component_index is None:
        component_index = build_directed_component_index(graph)
    if access_context is None:
        anchored = attach_footprint_access(sites, edges, area_crs)
        anchored = label_projection_connectivity(anchored, component_index)
    else:
        admin_path = Path(boundary_path).with_name("admin_boundary.geojson")
        admin = gpd.read_file(admin_path).to_crs(area_crs).geometry.union_all()
        water = admin.difference(gpd.GeoSeries([boundary], crs=4326).to_crs(area_crs).iloc[0])
        anchored = access_context.anchor(
            sites, masks=masks, footprints=gpd.read_parquet(footprints_path), water=water
        )
    anchors = anchored[["site_id"]].copy()
    anchors = gpd.GeoDataFrame(
        anchors,
        geometry=gpd.points_from_xy(anchored.road_anchor_lon, anchored.road_anchor_lat),
        crs=4326,
    )
    keep, anchor_filter = filter_public_locations(anchors, masks)
    if access_context is None:
        anchored = anchored.loc[anchored.site_id.isin(keep.site_id)].reset_index(drop=True)
    else:
        # Failed/unknown access is retained, source identity and units unchanged.
        anchor_filter["retention_policy"] = (
            "retain source-safe location, redact excluded anchor, mark review_required"
        )
    anchored["geometry_rule_candidate_eligible"] = True
    anchored["geometry_core_eligible"] = True
    anchored["road_access_candidate_eligible"] = anchored.protected_roundtrip_eligible.astype(bool)
    anchored = apply_site_scope_eligibility(anchored)
    anchored["road_access_distance_qa_flag"] = anchored.road_access_distance_m.gt(200)
    anchored["quarantine_reason"] = (
        anchored.access_review_reason
        if access_context is not None
        else np.where(anchored.cle_candidate_eligible, "", "outside_reference_directed_scc")
    )
    pending_scope = anchored.site_scope_review_required
    anchored.loc[pending_scope, "quarantine_reason"] = [
        "|".join(filter(None, [str(access), "site_scope:" + str(scope)]))
        for access, scope in zip(
            anchored.loc[pending_scope, "quarantine_reason"],
            anchored.loc[pending_scope, "site_scope_review_reason"],
        )
    ]
    members = members.loc[members.site_id.isin(anchored.site_id)].copy()
    anchored.to_parquet(output_dir / "footprint_road_access_candidates.parquet", index=False)
    anchored.to_parquet(output_dir / "latent_locations.parquet", index=False)
    members.to_parquet(output_dir / "residential_site_members.parquet", index=False)
    scope_ledger.to_csv(output_dir / "site_scope_ambiguity_candidates.csv", index=False)
    extents = sites.loc[sites.site_id.isin(anchored.site_id)]
    verification = validate_site_tables(anchored, members)
    extents.to_parquet(output_dir / "residential_sites.parquet", index=False)
    access = _write_service_access_contract(anchored, output_dir)
    report = {
        "schema": "evrptw_residential_site_production_v3",
        "status": "production_site_layer",
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "policy": SITE_POLICY,
        "generation_dependencies": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in [
                "residential_sites.py",
                "road_tags.py",
                "runtime_paths.py",
                "context_validation.py",
                "residential_identity.py",
                "residential_types.py",
                "site_scope_audit.py",
                "customer_spatial.py",
                "customer_access.py",
                "public_locations.py",
                "local_road_access.py",
                "delivery_access.py",
                "frontage.py",
                "terminal_access.py",
                "protected_connectivity.py",
            ]
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
            for name, path in {
                "nsi_crosswalk": records_path,
                "microsoft_footprints": footprints_path,
                "osm_buildings": osm_buildings_path,
                "osm_complexes": complexes_path,
                "boundary": boundary_path,
                "public_masks": mask_path,
                "graph": graph_path,
            }.items()
        },
        "summary": {
            **summary,
            "road_anchor_public_filter": anchor_filter,
            "retained_sites_after_road_anchor": len(anchored),
            "road_available_sites": int(anchored.cle_candidate_eligible.sum()),
            "road_accessible_sites_before_scope_guard": int(
                anchored.road_access_candidate_eligible.sum()
            ),
            "scope_pending_with_road_access": int(
                (
                    anchored.site_scope_review_required & anchored.road_access_candidate_eligible
                ).sum()
            ),
            "cle_available_sites": int(anchored.cle_candidate_eligible.sum()),
            "known_units_available_sites": int(
                anchored.loc[anchored.cle_candidate_eligible, "residential_units"].notna().sum()
            ),
            "protected_connectivity": connectivity_summary(anchored, component_index),
            "threshold_sensitivity": {
                str(t): _threshold_summary(anchored, t) for t in [25, 50, 100, 200]
            },
        },
        "access_contract_row_counts": access["row_counts"],
        "outputs": {
            "residential_sites": "residential_sites.parquet",
            "latent_locations": "latent_locations.parquet",
            "residential_site_members": "residential_site_members.parquet",
            "site_scope_ambiguity_candidates": "site_scope_ambiguity_candidates.csv",
            **{k: Path(v).name for k, v in access["paths"].items()},
        },
    }
    report["summary"]["road_available_known_units_sum"] = float(
        anchored.loc[anchored.cle_candidate_eligible, "residential_units"].sum()
    )
    report["summary"]["retained_known_units_sum"] = float(anchored.residential_units.sum())
    report["verification"] = verification
    report["inputs"]["graph_sha256"] = sha256_file(Path(graph_path))
    report["output_sha256"] = {
        name: sha256_file(output_dir / relative) for name, relative in report["outputs"].items()
    }
    write_json(output_dir / "residential_site_manifest.json", report)
    write_json(output_dir / "footprint_road_access_audit.json", report)
    return report
