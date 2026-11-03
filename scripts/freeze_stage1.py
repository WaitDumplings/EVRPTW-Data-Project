#!/usr/bin/env python3
"""Create the immutable Stage1 source handoff from the actual working files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ID = 'stage1-20261102-r1'
FREEZE = ROOT / 'freeze/20261102-r1'
SCRIPTS = {'assemble_stage1.py','build_stage1_city.py','build_stage1_cohort.py',
    'extract_frozen_access_roads.py','prepare_all_customer_sources.py','prepare_customer_sources.py',
    'verify_raw_road_nodes.py','verify_stage1.py','verify_state_speed_profiles.py',
    'prepare_stage1_server_inputs.py','verify_stage1_freeze.py','freeze_stage1.py'}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def payload():
    paths = {ROOT/name for name in ['README.md','README.zh-CN.md','NOTICE.md',
                                   'pyproject.toml','requirements-tested.txt']}
    for folder, suffixes in [('src/evrptw_cle',{'.py'}),('configs',{'.json','.sql'}),
                              ('tests',{'.py','.json','.geojson'})]:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in suffixes
                     and '__pycache__' not in p.parts and 'stage2' not in p.relative_to(ROOT).parts)
    paths.update(p for p in (ROOT/'scripts').glob('*.py') if p.name in SCRIPTS)
    paths.update(ROOT/'docs'/name for name in ['STAGE1_SPEC.zh-CN.md','CLE_SCHEMA.md','INPUTS.md','METHODS.md','HISTORICAL_CLE.md'])
    paths.update(p for p in (ROOT/'docs/stage1').rglob('*') if p.is_file() and p.suffix in {'.md','.csv'}
                 and 'reference_code' not in p.parts)
    paths.update(p for p in FREEZE.iterdir() if p.is_file() and p.name not in {'MANIFEST.json','BUNDLE.json'})
    paths={p for p in paths if not (p.parent==ROOT/'configs' and p.name in {
        'customer_sources.template.json','road_extraction_cohort.template.json'})}
    result={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(paths)}
    # Packaging projection only: implementation files are byte-identical.
    text=result['pyproject.toml'].decode()
    text=text.replace('evrptw-stage2 = "evrptw_scenarios.cli:main"\n','')
    text=text.replace('packages = ["src/evrptw_cle", "src/evrptw_scenarios"]','packages = ["src/evrptw_cle"]')
    tomllib.loads(text)
    result['pyproject.toml']=text.encode()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    files=payload()
    if args.dry_run:
        print(json.dumps({'freeze_id':ID,'files':len(files),'bytes':sum(map(len,files.values()))}))
        return
    target=FREEZE/'MANIFEST.json'
    if target.exists():
        raise SystemExit('This freeze ID already exists; create a new revision instead of replacing it')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    manifest={'schema':'evrptw_stage1_implementation_freeze_v1','freeze_id':ID,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Stage1 implementation, methodology, parameters and source-input identities',
        'implementation_frozen':True,'cle_data_frozen':False,'server_generation_status':'pending',
        'stage2_included':False,'git_head_reference':head,'working_tree_modified':dirty,
        'identity_basis':'payload file hashes; includes current uncommitted Stage1 work',
        'packaging_projection':'pyproject excludes Stage2 package and CLI; implementation files unchanged',
        'files':{name:{'sha256':sha(content),'bytes':len(content),
                 **({'source_sha256':sha((ROOT/name).read_bytes())} if name=='pyproject.toml' else {})}
                 for name,content in files.items()},
        'local_validation':json.loads((FREEZE/'local_validation.json').read_text()),
        'source_input_manifest':'freeze/20261102-r1/input_manifest.json',
        'building_attribute_reference':{'candidate_buildings':4012462,'admitted_buildings':3922741,
            'coverage_pct':97.7639414404423,'road_access_completed':False}}
    content=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    target.write_bytes(content)
    files['freeze/20261102-r1/MANIFEST.json']=content
    directory=ROOT/'dist'; directory.mkdir(exist_ok=True)
    archive=directory/(ID+'.tar.gz')
    prefix='EVRPTW-Stage1-'+ID
    with archive.open('wb') as output, gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for name,data in sorted(files.items()):
                info=tarfile.TarInfo(prefix+'/'+name);info.size=len(data);info.mtime=0
                info.mode=0o755 if name.endswith('.sh') else 0o644
                tar.addfile(info,io.BytesIO(data))
    receipt={'freeze_id':ID,'archive':str(archive.relative_to(ROOT)),
        'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,
        'manifest_sha256':sha(content),'files':len(files),'top_level':prefix,
        'raw_inputs_included':False,'input_bytes':json.loads((FREEZE/'input_manifest.json').read_text())['summary']['bytes']}
    (FREEZE/'BUNDLE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (directory/(ID+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
