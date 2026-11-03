import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from evrptw_cle.building_units import screen_and_label
from evrptw_cle.stage1 import main
from evrptw_cle.stage1_v2 import SCHEMA
import pytest


def test_scale_gate_rejects_small_impossible_and_unassessed_multiunit():
    frame = pd.DataFrame({
        'units_final': [20., 20., 20., 1.],
        'units_method': ['neighbor_estimated', 'nsi_representative', 'nsi_representative', 'nsi_representative'],
        'capacity_units': [1., 20., np.nan, np.nan],
        'area_m2': [5., 100., 100., 100.],
        'residential_use': ['unknown'] * 4,
        'result_reason': ['neighbor_estimated', 'nsi_representative', 'nsi_representative', 'nsi_representative'],
    })
    result = screen_and_label(frame)
    assert result.admitted.tolist() == [False, True, False, True]
    assert result.units_source.tolist() == ['unresolved', 'nsi', 'unresolved', 'nsi']
    assert result.uncovered_reason.iloc[0] == 'assigned_units_exceed_scale'
    assert result.uncovered_reason.iloc[2] == 'scale_unassessed_multiunit'
    assert result.selected_nsi_resunits.iloc[2] == 20
    assert np.isnan(result.units_final.iloc[2])
    assert frame.units_final.iloc[0] == 20


def test_current_stage1_raw_customer_pipeline_without_model_services(tmp_path):
    polygons = [box(200 * (i % 10), 200 * (i // 10),
                    200 * (i % 10) + 10, 200 * (i // 10) + 10) for i in range(103)]
    ms = gpd.GeoDataFrame({'building_id': [f'ms-{i}' for i in range(103)],
        'source_kind': ['global'] * 103, 'height': [6.] * 103}, geometry=polygons, crs=3857).to_crs(4326)
    osm = gpd.GeoDataFrame({'osm_id': [f'w{i}' for i in range(103)],
        'building': ['yes'] * 102 + ['retail'], 'building_part': [False] * 103,
        'tags': [json.dumps({'building': 'yes'})] * 102 + [json.dumps({'building': 'retail', 'shop': 'mall'})]},
        geometry=polygons, crs=3857).to_crs(4326)
    # 100 source-supported single-unit buildings, one unit-missing candidate,
    # one explicit zero-only unknown, and one nonresidential building.
    positions = list(range(100)) + [101, 102]
    nsi = gpd.GeoDataFrame({'fd_id': positions, 'bid': [None] * 102,
        'ftprntsrc': ['BING'] * 102, 'ftprntid': [f'source-{i}' for i in positions],
        'ftprntsqft': [100 / .09290304] * 102, 'resunits': [1] * 100 + [0, 0],
        'occtype': ['RES1'] * 101 + ['COM1']},
        geometry=[Point(polygons[i].centroid.x, polygons[i].centroid.y) for i in positions], crs=3857).to_crs(4326)
    sources = {}
    for key, data in [('microsoft', ms), ('osm', osm), ('nsi', nsi)]:
        path = tmp_path / f'{key}.parquet'
        data.to_parquet(path, index=False)
        sources[key] = str(path)
    boundary = tmp_path / 'boundary.geojson'
    gpd.GeoDataFrame(geometry=[box(-100, -100, 2200, 2400)], crs=3857).to_crs(4326).to_file(boundary, driver='GeoJSON')
    project = tmp_path / 'stage1.json'
    project.write_text(json.dumps({'schema': SCHEMA, 'output': str(tmp_path / 'result'),
        'cities': {'synthetic': {**sources, 'area_crs': 'EPSG:3857', 'service_boundary': str(boundary)}},
        'units': {'block_m': 100, 'train_pct': 60, 'cal_pct': 30, 'test_pct': 10},
        'ai': {'command': 'must-never-run'}}))
    main(['run', str(project), '--customer-only'])
    base = tmp_path / 'result/cities/synthetic/customer_inputs'
    result = pd.read_parquet(base / 'unit_results.parquet')
    final = pd.read_parquet(base / 'final_buildings.parquet')
    assert len(result) == 101
    assert set(result.units_source) <= {'nsi', 'neighbor', 'unresolved'}
    assert result.units_source.eq('nsi').sum() == 100
    assert not final.loc[final.entity_id.isin(['ms-101', 'ms-102']), 'admitted'].any()
    assert set(final.columns) >= {'units_source', 'selected_nsi_fd_id', 'scale_assessed', 'uncovered_reason'}
    assert result.loc[result.admitted & result.units_final.gt(1), 'scale_assessed'].all()
    assert (tmp_path / 'result/units/quality_thresholds.json').is_file()
    assert not (tmp_path / 'result/human_review.csv').exists()


def test_stage1_cli_returns_failure_when_saved_cle_validation_fails(tmp_path, monkeypatch):
    from evrptw_cle import stage1

    monkeypatch.setattr(stage1, 'load_project', lambda path: {
        'output': str(tmp_path), 'cities': {'synthetic': {}}})
    monkeypatch.setattr(stage1.geography, 'verify', lambda *args: [
        {'city': 'synthetic', 'passed': False, 'errors': [{'check': 'source_binding'}]}])
    with pytest.raises(SystemExit) as result:
        stage1.main(['verify', str(tmp_path / 'stage1.json')])
    assert result.value.code == 1
