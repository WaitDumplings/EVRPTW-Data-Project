#!/usr/bin/env bash
# One cohort calibration followed by city-local CLE construction.
set -euo pipefail
FREEZE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FREEZE_DIR/../.." && pwd)"
CONFIG="${1:-$FREEZE_DIR/stage1.server.json}"
PYTHON="${PYTHON:-python}"
FROM="${STAGE1_FROM:-prepare}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

case "$FROM" in
  prepare) START=0;; calibrate) START=1;; estimate) START=2;;
  finalize) START=3;; components) START=4;; verify) START=5;;
  *) echo "STAGE1_FROM must be prepare, calibrate, estimate, finalize, components or verify" >&2; exit 2;;
esac

"$PYTHON" "$REPO_ROOT/scripts/verify_stage1_freeze.py" --root "$REPO_ROOT" --inputs --config "$CONFIG"
"$PYTHON" - "$CONFIG" "$FREEZE_DIR/stage1.server.json" <<'PY'
import json,sys
from evrptw_cle.stage1 import load_project
from evrptw_cle.building_units import settings
actual=load_project(sys.argv[1]); frozen=load_project(sys.argv[2])
for key in ['axis_r','customer','construction','source_snapshots']:
    if actual.get(key)!=frozen.get(key):
        raise SystemExit('Frozen method differs: '+key+'; use this revision with path edits only')
if settings(actual)!=settings(frozen) or list(actual['cities'])!=list(frozen['cities']):
    raise SystemExit('Frozen cohort or unit parameters differ')
for city in frozen['cities']:
    if actual['cities'][city]['area_crs']!=frozen['cities'][city]['area_crs']:
        raise SystemExit('Frozen city CRS differs: '+city)
PY

OUTPUT="$("$PYTHON" -c 'import sys; from evrptw_cle.stage1 import load_project; print(load_project(sys.argv[1])["output"])' "$CONFIG")"
mkdir -p "$OUTPUT/logs"
"$PYTHON" - "$OUTPUT/logs/environment.json" <<'PY'
import importlib.metadata as m,json,platform,sys
from pathlib import Path
names=['geopandas','shapely','osmnx','networkx','numpy','pandas','pyarrow','pyogrio','pyproj','scipy','requests','matplotlib','osmium']
versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
record={'python':sys.version,'platform':platform.platform(),'packages':versions}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
PY

run_step() {
  local LABEL="$1"; shift
  "$PYTHON" -u -m evrptw_cle.stage1 "$@" 2>&1 | tee "$OUTPUT/logs/$LABEL.log"
}
if (( START <= 0 )); then run_step prepare prepare "$CONFIG"; fi
if (( START <= 1 )); then run_step calibrate calibrate-units "$CONFIG"; fi
if (( START <= 2 )); then run_step estimate estimate-units "$CONFIG"; fi
if (( START <= 3 )); then run_step finalize finalize "$CONFIG"; fi
if (( START <= 4 )); then
  CITY_LIST="$("$PYTHON" -c 'import sys; from evrptw_cle.stage1 import load_project; print("\n".join(load_project(sys.argv[1])["cities"]))' "$CONFIG")"
  # Dallas is the first full-graph integration city; preserve the calibration cohort.
  if [[ "$CITY_LIST" == *dallas* ]]; then
    run_step components-dallas components "$CONFIG" --cities dallas --component all
  fi
  while IFS= read -r CITY; do
    [[ "$CITY" == dallas ]] && continue
    run_step "components-$CITY" components "$CONFIG" --cities "$CITY" --component all
  done <<< "$CITY_LIST"
fi
run_step verify verify "$CONFIG"
"$PYTHON" - "$CONFIG" "$FREEZE_DIR/MANIFEST.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from evrptw_cle.stage1 import load_project
project=load_project(sys.argv[1]); root=Path(project['output'])
reports=json.loads((root/'validation_summary.json').read_text())
if {r['city'] for r in reports}!=set(project['cities']) or not all(r['passed'] for r in reports):
    raise SystemExit('Complete-cohort CLE validation did not pass')
receipt={'schema':'evrptw_stage1_server_build_receipt_v1','freeze_id':'stage1-20261102-r1',
    'completed_utc':datetime.now(timezone.utc).isoformat(),'stage2_included':False,
    'freeze_manifest_sha256':hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    'config_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
    'validated_cities':[r['city'] for r in reports],
    'cle_manifests':{city:hashlib.sha256((root/'cities'/city/'cle_manifest.json').read_bytes()).hexdigest() for city in project['cities']},
    'complete_cle_validation_passed':True}
(root/'stage1_server_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Stage1 CLE cohort complete:',root)
PY
