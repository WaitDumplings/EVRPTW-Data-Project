"""Build Stage 1 from Microsoft, OSM and NSI with calibrated unit completion."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import stage1_v2 as geography
from . import building_units
from .util import write_json


def load_project(path):
    project = geography.load_project(path)
    project.pop('ai', None)
    project.setdefault('customer', {}).setdefault('active_profile', 'unrestricted')
    project['customer'].setdefault('include_model_imputed', True)
    return project


def initialize(directory):
    path = geography.initialize(directory)
    project = geography.read_json(path)
    project.pop('ai', None)
    project['customer'] = {'active_profile': 'unrestricted', 'include_model_imputed': True}
    project['units'] = dict(building_units.DEFAULTS)
    write_json(path, project)
    return path


def run(project, cities, *, customer_only=False):
    geography.prepare(project, cities)
    calibrated = building_units.calibrate_units(project, cities)
    building_units.estimate_units(project, cities)
    coverage = building_units.finalize(project, cities)
    result = {'status': 'building_units_complete', 'customer_coverage': coverage,
              'calibration': calibrated, 'stage2_included': False}
    if not customer_only:
        result['components'] = geography.components(project, cities, 'all')
        result['validation'] = geography.verify(project, cities)
        result['status'] = 'stage1_complete' if all(r['passed'] for r in result['validation']) else 'validation_failed'
    write_json(Path(project['output']) / 'build_progress.json', geography.clean(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='Create a Stage1 project').add_argument('directory', type=Path)
    for name in ['plan', 'prepare', 'scope', 'calibrate-units', 'estimate-units',
                 'finalize', 'components', 'verify', 'run', 'build']:
        p = sub.add_parser(name)
        p.add_argument('config', type=Path)
        p.add_argument('--cities', nargs='+')
        p.add_argument('--json', action='store_true')
        if name in ['run', 'build']:
            p.add_argument('--customer-only', action='store_true', help='Build building units without road/facility components')
        if name == 'components':
            p.add_argument('--component', required=True,
                choices=['all', 'roads', 'customers', 'depots', 'charging', 'speeds', 'assemble'])
    args = parser.parse_args(argv)
    if args.command == 'init':
        print(initialize(args.directory))
        return
    project = load_project(args.config)
    cities = geography.selected_cities(project, args.cities)
    Path(project['output']).mkdir(parents=True, exist_ok=True)
    if args.command == 'plan':
        missing = [{'city': city, 'input': key, 'path': project['cities'][city].get(key)}
            for city in cities for key in ['microsoft', 'osm', 'nsi', 'service_boundary']
            if not project['cities'][city].get(key) or not Path(project['cities'][city][key]).is_file()]
        result = {'ready': not missing, 'cities': cities, 'missing': missing,
                  'output': project['output'], 'units': building_units.settings(project)}
    elif args.command == 'prepare':
        result = geography.prepare(project, cities)
    elif args.command == 'scope':
        result = [{'city': city, **geography.scope_city(project, city)} for city in cities]
    elif args.command == 'calibrate-units':
        result = building_units.calibrate_units(project, cities)
    elif args.command == 'estimate-units':
        result = building_units.estimate_units(project, cities)
    elif args.command == 'finalize':
        result = building_units.finalize(project, cities)
    elif args.command == 'components':
        result = geography.components(project, cities, args.component)
    elif args.command == 'verify':
        result = geography.verify(project, cities)
    else:
        result = run(project, cities, customer_only=args.customer_only)
    if args.json:
        import json
        print(json.dumps(geography.clean(result), ensure_ascii=False, indent=2))
    else:
        print(f'{args.command}: {len(cities)} city/cities; {project["output"]}')
    if ((args.command == 'verify' and any(not r['passed'] for r in result)) or
            (args.command in ['run', 'build'] and result.get('status') == 'validation_failed')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
