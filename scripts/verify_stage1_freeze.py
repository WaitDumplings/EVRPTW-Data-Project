#!/usr/bin/env python3
"""Verify a frozen Stage1 code bundle, optionally its separately transferred inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, *, include_inputs=False, config=None):
    root = Path(root).resolve()
    directory = root / 'freeze/20261102-r1'
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    errors = []
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or digest(path) != expected['sha256']:
            errors.append(name)
    inputs_checked = 0
    if include_inputs:
        inputs = json.loads((directory / 'input_manifest.json').read_text())
        data_root = root / inputs['data_root']
        project = None
        if config is not None:
            from evrptw_cle.stage1 import load_project
            project = load_project(config)
        checks = {}
        for expected in inputs['files']:
            path = data_root / expected['target_path']
            if project is not None:
                parts = Path(expected['target_path']).parts
                if parts[0] == 'customer':
                    key = {'microsoft_footprints.parquet':'microsoft',
                           'osm_buildings.parquet':'osm', 'nsi_native.parquet':'nsi'}[parts[2]]
                    path = Path(project['cities'][parts[1]][key])
                elif parts[0] == 'roads':
                    path = Path(project['components']['road_sources']).joinpath(*parts[1:])
                else:
                    path = Path(project['components']['source_root']).joinpath(*parts[1:])
            checks[path] = expected['sha256']
            inputs_checked += 1
        if project is not None:
            reference = json.loads((directory / 'stage1.server.json').read_text())
            expected_paths = {str((data_root/r['target_path']).resolve()):r['sha256'] for r in inputs['files']}
            for city, sources in reference['cities'].items():
                for key in ['service_boundary','admin_boundary']:
                    target = str((directory/sources[key]).resolve())
                    checks[Path(project['cities'][city][key])] = expected_paths[target]
            checks[Path(project['components']['inventory'])] = inputs['component_inventory']['sha256']
        for path, expected in checks.items():
            if not path.is_file() or digest(path) != expected:
                errors.append(str(path))
    return {'freeze_id': manifest['freeze_id'], 'passed': not errors,
            'code_files_checked': len(manifest['files']), 'input_files_checked': inputs_checked,
            'missing_or_changed': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', action='store_true')
    parser.add_argument('--config', type=Path, help='Verify actual input paths from this server project')
    args = parser.parse_args()
    result = verify(args.root, include_inputs=args.inputs, config=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
