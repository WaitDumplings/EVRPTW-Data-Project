"""Run frozen Stage 1 city components sequentially with persisted progress."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import UTC, datetime

from evrptw_cle.context_validation import validate_generation_dependencies
from evrptw_cle.util import sha256_file, write_json

p = argparse.ArgumentParser(description=__doc__)
for name in ["inventory", "source-root", "road-sources", "customer-registry", "output-root"]:
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--phase", choices=["roads", "locations"], required=True)
p.add_argument("--cities", nargs="+")
a = p.parse_args()
if (a.output_root / "stage1_freeze.json").exists():
    raise FileExistsError("Cannot build over a frozen CLE; choose a new output directory")
inventory = json.loads(a.inventory.read_text())
cities = a.cities or sorted(inventory["cities"], key=lambda c: (c not in ["jacksonville", "philadelphia"], c))
components = ["seed", "roads"] if a.phase == "roads" else ["customers", "depots", "charging", "speeds"]
stamps = {
    "seed": "_physical_seed/access_context_manifest.json", "roads": "roads/access_context_manifest.json",
    "customers": "customers/residential_site_manifest.json", "depots": "depots/depot_component_manifest.json",
    "charging": "charging/charging_component_manifest.json", "speeds": "speeds/common_speed_manifest.json",
}
a.output_root.mkdir(parents=True, exist_ok=True)
status_path = a.output_root / (a.phase + "_build_status.json")
status = dict(phase=a.phase, state="running", cities=cities, completed=[], failed=None,
              source_inventory_sha256=sha256_file(a.inventory), instance_activation_included=False)

def valid(path, component):
    if not path.exists():
        return False
    m = json.loads(path.read_text())
    city_name=path.parent.parent.name
    current_context=path.parent.parent / "roads/access_context_manifest.json"
    if component in ["seed", "roads"]:
        validate_generation_dependencies(m)
        for name, digest in m["outputs"].items():
            if sha256_file(path.parent / name) != digest:
                raise ValueError("Changed component output: " + name)
    elif component == "customers":
        if m.get("source_registry",{}).get("sha256")!=sha256_file(a.customer_registry):
            raise ValueError("Customer source registry changed")
        if m["inputs"]["local_access_context"]["sha256"]!=sha256_file(current_context):
            raise ValueError("Customer road context changed")
        for name, value in m["outputs"].items():
            if sha256_file(path.parent / value) != m["output_sha256"][name]:
                raise ValueError("Changed customer output: " + name)
    elif component == "depots":
        if sha256_file(path.parent / "depots.parquet")!=m["output_sha256"]:
            raise ValueError("Changed depot output")
        if m["context_sha256"]!=sha256_file(current_context):
            raise ValueError("Depot road context changed")
    else:
        if m["inputs"]["road_context"]["sha256"]!=sha256_file(current_context):
            raise ValueError("Component road context changed")
        for group in ["outputs", "private_outputs"]:
            for name, digest in m.get(group, {}).items():
                if sha256_file(path.parent / name) != digest:
                    raise ValueError("Changed component output: " + name)
    if component in ["charging","depots"]:
        registry=json.loads(a.customer_registry.read_text())
        expected=registry["cities"][city_name]["sources"]["microsoft_footprints"]["sha256"]
        actual=m["inputs"]["building_obstacles"]["sha256"] if component=="charging" else m["building_obstacles"]["sha256"]
        if actual!=expected:
            raise ValueError("Component building-obstacle source changed")
    for name, digest in m.get("generation_dependencies", {}).items():
        import evrptw_cle
        if sha256_file(Path(evrptw_cle.__file__).with_name(name)) != digest:
            raise ValueError("Changed component generation code: " + name)
    return True

for city in cities:
    base = a.output_root / "cities" / city
    for component in components:
        if component == "seed" and (base / stamps["roads"]).exists():
            continue
        stamp = base / stamps[component]
        if valid(stamp, component):
            status["completed"].append(dict(city=city, component=component, reused=True))
            print("VERIFIED REUSE", city, component, flush=True)
            continue
        if shutil.disk_usage(a.output_root).free < 6 * 1024**3:
            raise RuntimeError("Insufficient working space; no frozen data will be deleted")
        # A stopped worker may leave an incomplete folder. Keep it for diagnosis
        # and restart only that component; completed changed outputs still fail.
        folder = base / ("_physical_seed" if component == "seed" else component)
        if folder.exists() and not stamp.exists():
            archive = a.output_root / ".work/incomplete" / city / (component + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
            archive.parent.mkdir(parents=True, exist_ok=True)
            folder.replace(archive)
            old_log = base / (component + ".log")
            if old_log.exists():
                shutil.copyfile(old_log, archive / "previous_attempt.log")
        base.mkdir(parents=True, exist_ok=True)
        status.update(current_city=city, current_component=component, updated_utc=datetime.now(UTC).isoformat())
        write_json(status_path, status)
        print("BUILD", city, component, flush=True)
        with (base / (component + ".log")).open("w") as log:
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("build_stage1_city.py")), component,
                "--city", city,
                *[arg for key in ["inventory", "source_root", "road_sources", "customer_registry", "output_root"]
                  for arg in ["--" + key.replace("_", "-"), str(getattr(a, key).resolve())]],
            ], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state="failed", failed=dict(city=city, component=component, exit_code=result.returncode))
            write_json(status_path, status)
            raise SystemExit(result.returncode)
        if not valid(stamp, component):
            raise ValueError("Component finished without a verified manifest")
        status["completed"].append(dict(city=city, component=component, reused=False))
        write_json(status_path, status)
        print("COMPLETE", city, component, flush=True)
status.update(state="complete", updated_utc=datetime.now(UTC).isoformat())
write_json(status_path, status)
