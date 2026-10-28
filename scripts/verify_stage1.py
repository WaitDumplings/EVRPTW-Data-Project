"""Read-only independent Stage 1 verification in isolated city processes."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from evrptw_cle.cle_validation import verify_city_cle
from evrptw_cle.util import sha256_file, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--inventory", type=Path, required=True)
p.add_argument("--city")
p.add_argument("--cities", nargs="+", help="Selected cohort when the source inventory contains more cities")
p.add_argument("--report",type=Path)
a = p.parse_args()
if a.city:
    result = verify_city_cle(a.root / "cities" / a.city,report_path=a.report)
    print(json.dumps(dict(city=a.city, passed=result["passed"], checks=len(result["checks"]), summary=result["summary"])))
else:
    inventory = json.loads(a.inventory.read_text())
    cities = a.cities or sorted(inventory["cities"])
    if len(set(cities)) != len(cities) or not set(cities) <= set(inventory["cities"]):
        raise ValueError("Invalid city selection for this inventory")
    actual = {path.name for path in (a.root / "cities").iterdir() if path.is_dir()}
    if actual != set(cities):
        raise ValueError("Saved city cohort differs from the frozen inventory")
    results = {}
    frozen=(a.root / "stage1_freeze.json").is_file()
    report_root=a.root / "reaudits" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") if frozen else a.root
    report_root.mkdir(parents=True,exist_ok=True)
    for city in cities:
        print("VERIFY", city, flush=True)
        log_path=report_root / (city+".log") if frozen else a.root / "cities" / city / "verify.log"
        report_path=report_root / (city+".json") if frozen else a.root / "cities" / city / "cle_verification.json"
        with log_path.open("w") as log:
            subprocess.run([sys.executable, __file__, "--root", str(a.root.resolve()),
                            "--inventory", str(a.inventory.resolve()), "--city", city,"--report",str(report_path.resolve())],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        path = report_path
        result = json.loads(path.read_text())
        if not result["passed"]:
            raise ValueError("City independent verifier did not pass: " + city)
        results[city] = dict(verification_sha256=sha256_file(path),
                            cle_manifest_sha256=result["cle_manifest_sha256"], summary=result["summary"],
                            verified_terminal_paths=result["verified_terminal_paths"])
        print("VERIFIED", city, result["summary"], flush=True)
    write_json(report_root / "stage1_verification.json", dict(
        schema="evrptw_stage1_cohort_verification_v1", passed=True,
        generated_utc=datetime.now(UTC).isoformat(), city_count=len(cities), cities=results,
        inventory_sha256=sha256_file(a.inventory), instance_activation_included=False,
        public_release=False,
    ))
