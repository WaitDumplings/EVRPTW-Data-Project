"""Deterministic building units: original NSI representatives and empirical neighbours."""
from __future__ import annotations

from datetime import UTC, datetime
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .unit_features import prepare_features, load_paths, full_osm_containment
from .unit_matching import RepresentativeConfig, edges_from_current_ledger, build_candidate_edges, select_representatives
from .unit_neighbors import EmpiricalMetricNeighborModel
from .unit_quality import fit_quality_thresholds, apply_quality
from .unit_reference import make_reference_splits
from .unit_scale import annotate_capacity
from .util import write_json, sha256_file

POLICY = 'building_nsi_representative_empirical_rms_units_v1'
DEFAULTS = dict(seed=20261102, block_m=2000, train_pct=80, cal_pct=10, test_pct=10,
    k=8, scale_probe_count=2000, calibration_per_city=5000,
    quality_target_retention=.95, quality_min_group_size=30, quality_min_neighbors=8,
    quality_dispersion_column='neighbor_log_spread_90', min_reference_count=1,
    min_unit_area_m2=10., floor_to_floor_m=3.048)


def settings(project):
    return {**DEFAULTS, **project.get('units', {})}


def input_dir(project, city):
    return Path(project['output']) / 'cities' / city / 'customer_inputs'


def capacity(frame, cfg):
    return annotate_capacity(frame, min_unit_area_m2=cfg['min_unit_area_m2'],
                             floor_to_floor_m=cfg['floor_to_floor_m'])


def model(cfg):
    return EmpiricalMetricNeighborModel({k: cfg[k] for k in
        ['seed', 'k', 'scale_probe_count', 'min_reference_count']})


def reference_split(frame, city, cfg):
    reference = frame.loc[frame.reference_anchor].copy()
    split = make_reference_splits(reference, city, seed=cfg['seed'], block_m=cfg['block_m'],
        train_pct=cfg['train_pct'], cal_pct=cfg['cal_pct'], test_pct=cfg['test_pct'])
    return reference, split


def prepare_units_city(project, city):
    """Prepare features and geometric candidates directly from the declared sources."""
    cfg = settings(project)
    base = input_dir(project, city)
    sources = project['cities'][city]
    f = prepare_features(city=city, sources=sources, base=base,
        axis_r=float(project.get('axis_r', RepresentativeConfig().shared_R)), config=cfg)
    f = capacity(f, cfg)
    records = full_osm_containment(pd.read_parquet(base / 'source_records.parquet'), base, sources)
    original = pd.read_parquet(base / 'population.parquet')
    edges = edges_from_current_ledger(records)
    count = original.set_index('entity_index').nsi_count
    clear = original.set_index('entity_index').nsi_attribution_clear
    redo = (records.owner.lt(0) | ~records.owner.map(clear).eq(True)
        | records.owner.map(count).fillna(0).gt(1)
        | records.source_group_conflict.fillna(False)
        | records.source_geometry_counterevidence.fillna(False)
        | records.geometry_block_reason.fillna('').ne(''))
    redo &= ~records.source_internal_conflict.fillna(False) & ~records.full_osm_all_nonres
    if redo.any():
        rebuilt, _ = build_candidate_edges(load_paths(base, sources), records.loc[redo], sources['area_crs'],
            RepresentativeConfig(shared_R=float(project.get('axis_r', RepresentativeConfig().shared_R))))
        rebuilt = rebuilt.drop(columns=[c for c in rebuilt if c in records and c not in ['fd_id', 'entity_index']], errors='ignore')
        edges = pd.concat([edges, rebuilt], ignore_index=True)
    f['location_eligible'] = f.valid_features & ~f.native_conflict_status
    f['neighbor_units_for_match'] = f.units.where(f.clean_direct)
    selected, _, _ = select_representatives(edges, records, f,
        RepresentativeConfig(neighbor_column='neighbor_units_for_match'))
    chosen = selected.set_index('entity_index')
    f['initial_selected_fd_id'] = f.entity_index.map(chosen.fd_id).astype('Int64')
    supported = f.units.eq(1) | (f.capacity_units.notna() & f.units.le(f.capacity_units))
    f['reference_anchor'] = (f.clean_direct & f.valid_features & ~f.native_conflict_status &
        f.units.ge(1) & supported & f.initial_selected_fd_id.eq(f.fd_id).fillna(False))
    f.to_parquet(base / 'unit_features.parquet', index=False)
    edges.to_parquet(base / 'unit_candidate_edges.parquet', index=False)
    reference, split = reference_split(f, city, cfg)
    roster = reference[['entity_id', 'source_group', 'fd_id', 'units']].copy()
    roster['reference_split'] = split
    roster.to_parquet(base / 'unit_reference_roster.parquet', index=False)
    write_json(base / 'unit_preparation.json', {
        'policy': POLICY, 'city': city, 'candidate_buildings': len(f),
        'reference_buildings': len(reference), 'train_reference_buildings': int(split.eq('train').sum()),
        'config': cfg, 'input_hashes': {name: sha256_file(base / name)
            for name in ['population.parquet', 'source_records.parquet', 'scope.parquet']},
    })
    return f


def load_features(project, city):
    path = input_dir(project, city) / 'unit_features.parquet'
    if not path.exists():
        return prepare_units_city(project, city)
    saved = json.loads((path.parent / 'unit_preparation.json').read_text())
    if any(saved['config'].get(key) != settings(project)[key] for key in ['min_unit_area_m2', 'floor_to_floor_m']):
        return prepare_units_city(project, city)
    f = pd.read_parquet(path)
    f['height_m'] = f.height_original_m
    return capacity(f, settings(project))


def calibrate_units(project, cities):
    cfg = settings(project)
    dest = Path(project['output']) / 'units'
    dest.mkdir(parents=True, exist_ok=True)
    predictions, scales = [], []
    for city in cities:
        f = load_features(project, city)
        ref, split = reference_split(f, city, cfg)
        roster = ref[['entity_id', 'source_group', 'fd_id', 'units']].copy()
        roster['reference_split'] = split
        roster.to_parquet(input_dir(project, city) / 'unit_reference_roster.parquet', index=False)
        fitted = model(cfg).fit(ref.loc[split.eq('train')])
        cal = ref.loc[split.eq('calibration')]
        if cal.empty:
            continue
        sample = cal.sample(min(cfg['calibration_per_city'], len(cal)), random_state=cfg['seed'])
        pred = fitted.predict(sample)
        pred['city'], pred['use_type'] = city, sample.use_type
        pred['entity_id'] = sample.entity_id
        pred['sampling_weight'] = len(cal) / len(sample)
        predictions.append(pred)
        scale = fitted.scales_table()
        scale['city'] = city
        scales.append(scale)
        print('CALIBRATED', city, len(sample), flush=True)
        del f, ref, fitted, sample, cal
        gc.collect()
    if not predictions:
        raise ValueError('No held-out calibration references; include a supported city or use a saved calibrated model')
    calibration = pd.concat(predictions, ignore_index=True)
    thresholds = fit_quality_thresholds(calibration, cfg)
    write_json(dest / 'quality_thresholds.json', thresholds)
    calibration.to_parquet(dest / 'calibration_predictions.parquet', index=False)
    pd.concat(scales, ignore_index=True).to_csv(dest / 'metric_scales.csv', index=False)
    write_json(dest / 'calibration.json', {'policy': POLICY, 'cities': list(cities), 'config': cfg,
        'calibration_buildings': len(calibration), 'target_labels_used': False,
        'quality_thresholds_sha256': sha256_file(dest / 'quality_thresholds.json'),
        'metric_scales_sha256': sha256_file(dest / 'metric_scales.csv'),
        'reference_inputs': {city: {'roster_sha256': sha256_file(input_dir(project, city) / 'unit_reference_roster.parquet'),
                                  'preparation_sha256': sha256_file(input_dir(project, city) / 'unit_preparation.json')}
                             for city in cities},
        'reference': 'fixed training split, original observations only'})
    return {'cities': list(cities), 'calibration_buildings': len(calibration), 'path': str(dest)}


def screen_and_label(frame):
    """Final one-sided scale gate and explicit per-building unit provenance.

    A source-supported one-unit observation remains admissible.  Multi-unit
    assignments require an assessed scale and must fit its loose upper proxy.
    Missing scale is recorded as unassessed rather than treated as a pass.
    """
    f = frame.copy()
    before = pd.to_numeric(f.units_final, errors='coerce')
    multi = before.gt(1)
    unassessed = multi & f.capacity_units.isna()
    oversized = multi & f.capacity_units.notna() & before.gt(f.capacity_units)
    f['scale_assessed'] = f.capacity_units.notna()
    f['scale_gate_status'] = np.select([unassessed, oversized, before.eq(1), multi],
        ['scale_unassessed', 'above_limit', 'single_unit', 'assessed_within_limit'], default='no_assignment')
    selected_units = before.where(f.units_method.eq('nsi_representative'))
    f['selected_nsi_resunits'] = (selected_units.combine_first(f.selected_nsi_resunits)
        if 'selected_nsi_resunits' in f else selected_units)
    f.loc[unassessed, 'result_reason'] = 'scale_unassessed_multiunit'
    f.loc[oversized, 'result_reason'] = 'assigned_units_exceed_scale'
    f.loc[unassessed | oversized, 'units_final'] = np.nan
    f.loc[unassessed | oversized, 'units_method'] = 'unresolved'
    f['admitted'] = f.units_final.ge(1)
    f['units_source'] = np.select([
        f.admitted & f.units_method.eq('nsi_representative'),
        f.admitted & f.units_method.eq('neighbor_estimated')], ['nsi', 'neighbor'], default='unresolved')
    f['units_scope'] = np.where(f.admitted, 'building', 'unresolved')
    f['units_review_status'] = np.select([f.units_source.eq('nsi'), f.units_source.eq('neighbor')],
        ['source_supported', 'estimated'], default='unresolved')
    f['units_independently_verified'] = False
    f['residential_use_observed'] = f.get('residential_use_observed', f.residential_use)
    f['residential_source'] = np.select([
        f.admitted & f.residential_use_observed.eq('residential'),
        f.units_source.eq('nsi'), f.units_source.eq('neighbor')],
        ['osm', 'nsi_representative', 'neighbor_reference_supported'], default='unresolved')
    f.loc[f.admitted, 'residential_use'] = 'residential'
    f['purpose_review_status'] = np.where(f.admitted, 'source_supported_or_inferred', 'unresolved')
    f['uncovered_reason'] = f.result_reason.where(~f.admitted, '')
    f['unit_model_policy'] = POLICY
    return f


def coverage(frame, city):
    source = frame.units_source.eq('nsi')
    neighbor = frame.units_source.eq('neighbor')
    return {'city': city, 'candidate_buildings': len(frame), 'nsi_buildings': int(source.sum()),
        'neighbor_estimated_buildings': int(neighbor.sum()), 'admitted_buildings': int(frame.admitted.sum()),
        'unfilled_buildings': int((~frame.admitted).sum()),
        'building_coverage_pct': 100 * frame.admitted.sum() / len(frame) if len(frame) else 0.,
        'max_resunit': int(frame.units_final.max()) if frame.admitted.any() else None}


def save_results(base, frame, city):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(base / 'unit_results.parquet', index=False)
    reasons = frame.loc[~frame.admitted, 'uncovered_reason'].value_counts().rename_axis('reason').reset_index(name='buildings')
    reasons.insert(0, 'city', city)
    reasons.to_csv(base / 'unit_uncovered_reasons.csv', index=False)
    summary = coverage(frame, city)
    write_json(base / 'unit_coverage.json', summary)
    return summary


def estimate_units_city(project, city, thresholds):
    cfg = settings(project)
    base = input_dir(project, city)
    f = load_features(project, city)
    ref, split = reference_split(f, city, cfg)
    train = ref.loc[split.eq('train')]
    fitted = model(cfg).fit(train)
    train_mask = f.index.isin(train.index)
    targets = f.loc[~train_mask]
    pred = fitted.predict(targets, estimate_missing_height=True)
    pred['city'], pred['use_type'] = city, targets.use_type
    pred = apply_quality(pred, thresholds)
    for col in pred:
        if col not in ['city', 'use_type']:
            f.loc[pred.index, col] = pred[col]
    for col in ['quality_pass', 'similarity_pass', 'dispersion_pass', 'quality_inputs_valid']:
        f[col] = f[col].eq(True)
    f['neighbor_units_for_match'] = f.neighbor_units.where(f.quality_pass)
    f.loc[f.reference_anchor, 'neighbor_units_for_match'] = f.loc[f.reference_anchor, 'units']
    missing = ~(np.isfinite(f.height_original_m) & f.height_original_m.gt(0)) & ~(np.isfinite(f.levels) & f.levels.gt(0))
    f['height_neighbor_estimated'] = missing & f.neighbor_height_p90.gt(0) & f.similarity_pass
    f.loc[f.height_neighbor_estimated, 'height_m'] = f.loc[f.height_neighbor_estimated, 'neighbor_height_p90']
    f = capacity(f, cfg)
    f['location_eligible'] = f.valid_features & ~f.native_conflict_status
    records = full_osm_containment(pd.read_parquet(base / 'source_records.parquet'), base, project['cities'][city])
    edges = pd.read_parquet(base / 'unit_candidate_edges.parquet')
    selected, ledger, diagnostics = select_representatives(edges, records, f,
        RepresentativeConfig(neighbor_column='neighbor_units_for_match'))
    selected.to_parquet(base / 'nsi_representatives.parquet', index=False)
    ledger.to_parquet(base / 'nsi_assignment_ledger.parquet', index=False)
    chosen = selected.set_index('entity_index')
    f['selected_nsi_fd_id'] = f.entity_index.map(chosen.fd_id).astype('Int64')
    f['units_final'] = f.entity_index.map(chosen.units_final)
    f['units_method'] = np.where(f.selected_nsi_fd_id.notna(), 'nsi_representative', 'unresolved')
    owner_capacity = records.owner.map(f.set_index('entity_index').capacity_units)
    failed = owner_capacity.notna() & records.resunits.gt(owner_capacity) & records.resunits.gt(1)
    flags = pd.DataFrame({'owner': records.owner, 'failed': failed}).loc[records.owner.ge(0)].groupby('owner').failed.agg(['size', 'sum'])
    bad = flags.index[flags['size'].eq(flags['sum'])]
    f['all_original_units_exceed_capacity'] = f.entity_index.isin(bad)
    unmatched = f.selected_nsi_fd_id.isna() & f.location_eligible & ~f.all_original_units_exceed_capacity
    physical = f.neighbor_units.eq(1) | (f.capacity_units.notna() & f.neighbor_units.le(f.capacity_units))
    accept = unmatched & f.quality_pass & f.neighbor_units.ge(1) & physical
    f.loc[accept, 'units_final'] = f.loc[accept, 'neighbor_units']
    f.loc[accept, 'units_method'] = 'neighbor_estimated'
    reason = np.full(len(f), 'unclassified', object)
    remaining = np.ones(len(f), bool)
    def mark(name, mask):
        ix = remaining & np.asarray(mask, bool)
        reason[ix] = name
        remaining[ix] = False
    mark('nsi_representative', f.units_method.eq('nsi_representative'))
    mark('neighbor_estimated', accept)
    mark('source_conflict', f.native_conflict_status)
    mark('invalid_building_features', ~f.valid_features)
    mark('original_nsi_scale_anomaly', f.all_original_units_exceed_capacity)
    mark('no_independent_neighbor_prediction', ~f.quality_inputs_valid)
    mark('neighbor_similarity_and_dispersion', ~f.similarity_pass & ~f.dispersion_pass)
    mark('neighbor_similarity', ~f.similarity_pass)
    mark('neighbor_dispersion', ~f.dispersion_pass)
    mark('neighbor_units_exceed_scale', ~physical)
    f['result_reason'] = reason
    f = screen_and_label(f)
    summary = save_results(base, f, city)
    write_json(base / 'unit_matching.json', diagnostics)
    print('ESTIMATED', city, summary['building_coverage_pct'], flush=True)
    return summary


def estimate_units(project, cities):
    path = Path(project['output']) / 'units/quality_thresholds.json'
    thresholds = json.loads(path.read_text())
    calibrated = json.loads(path.with_name('calibration.json').read_text())
    if calibrated['config'] != settings(project):
        raise ValueError('Unit parameters changed: run calibrate-units before estimate-units')
    return [estimate_units_city(project, city, thresholds) for city in cities]


def finalize(project, cities):
    """Write current building records for the unchanged polygon/road adapter."""
    cfg = settings(project)
    summaries, reasons = [], []
    for city in cities:
        base = input_dir(project, city)
        results = pd.read_parquet(base / 'unit_results.parquet')
        original = pd.read_parquet(base / 'population.parquet')
        scope = pd.read_parquet(base / 'scope.parquet')
        final = original.merge(scope, on='entity_id', validate='one_to_one')
        indexed = results.set_index('entity_id')
        override = {'units_final', 'units_scope', 'units_method', 'units_source', 'residential_use',
                    'residential_source', 'units_review_status', 'purpose_review_status', 'units_independently_verified'}
        updates = {}
        for col in results:
            if col == 'entity_id':
                continue
            values = final.entity_id.map(indexed[col])
            updates[col] = values if col in override or col not in final else values.where(values.notna(), final[col])
        final = pd.concat([final.drop(columns=list(updates), errors='ignore'), pd.DataFrame(updates, index=final.index)], axis=1)
        for col in ['admitted', 'reference_anchor', 'height_neighbor_estimated']:
            final[col] = final[col].eq(True)
        final['units_source'] = final.units_source.fillna('unresolved')
        final['units_method'] = final.units_method.fillna('unresolved')
        final['units_scope'] = final.units_scope.fillna('unresolved')
        final['residential_use'] = final.residential_use.fillna(final.osm_use)
        final['residential_source'] = final.residential_source.fillna('unresolved')
        final['uncovered_reason'] = final.uncovered_reason.fillna('outside_candidate_population')
        final.to_parquet(base / 'final_buildings.parquet', index=False)
        write_json(base / 'finalization_manifest.json', {
            'schema': 'building_customer_field_lineage_v2', 'policy': POLICY, 'city': city,
            'inputs': {name: {'sha256': sha256_file(base / name)} for name in
                ['population.parquet', 'preparation_manifest.json', 'scope.parquet', 'unit_results.parquet', 'unit_reference_roster.parquet']},
            'unit_model': {name: {'path': str(Path(project['output']) / 'units' / name),
                                  'sha256': sha256_file(Path(project['output']) / 'units' / name)}
                           for name in ['quality_thresholds.json', 'metric_scales.csv', 'calibration.json']
                           if (Path(project['output']) / 'units' / name).is_file()},
            'final_buildings_sha256': sha256_file(base / 'final_buildings.parquet'),
            'raw_nsi_immutable': True, 'units_sources': ['nsi', 'neighbor', 'unresolved'],
            'include_model_imputed': bool(project.get('customer', {}).get('include_model_imputed', True)),
            'config': cfg,
        })
        summaries.append(json.loads((base / 'unit_coverage.json').read_text()))
        reasons.append(pd.read_csv(base / 'unit_uncovered_reasons.csv'))
    root = Path(project['output'])
    for city in project['cities']:
        if city in cities:
            continue
        base = input_dir(project, city)
        if (base / 'unit_coverage.json').exists():
            summaries.append(json.loads((base / 'unit_coverage.json').read_text()))
            reasons.append(pd.read_csv(base / 'unit_uncovered_reasons.csv'))
    order = {city: i for i, city in enumerate(project['cities'])}
    summaries.sort(key=lambda row: order[row['city']])
    pd.DataFrame(summaries).to_csv(root / 'customer_coverage.csv', index=False)
    pd.concat(reasons, ignore_index=True).to_csv(root / 'customer_uncovered_reasons.csv', index=False)
    write_json(root / 'stage1_method.json', {'policy': POLICY, 'generated_utc': datetime.now(UTC).isoformat(),
        'units_sources': ['nsi', 'neighbor', 'unresolved'], 'config': cfg,
        'manual_record_repairs': False, 'additional_data_sources': False, 'stage2_included': False})
    return summaries
