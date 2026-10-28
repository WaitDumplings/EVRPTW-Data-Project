"""Assemble complete CLEs from saved components; no instance activation."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from evrptw_cle.cle_assembly import assemble_city_cle
from evrptw_cle.runtime_paths import CONFIG_ROOT
from evrptw_cle.source_preparation import source_path
from evrptw_cle.util import sha256_file

p = argparse.ArgumentParser(description=__doc__)
for name in ["inventory", "source-root", "road-sources", "customer-registry", "output-root"]:
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--city")
a = p.parse_args()
inventory = json.loads(a.inventory.read_text())
if a.city:
    city = inventory["cities"][a.city]
    base=a.output_root / "cities" / a.city
    if (a.output_root / "stage1_freeze.json").exists():
        raise FileExistsError("Cannot assemble over a frozen Stage 1 version")
    target=base / "cle_manifest.json"
    if target.exists():
        saved=json.loads(target.read_text())
        for name, item in saved["files"].items():
            if sha256_file(base / name)!=item["sha256"]:
                raise ValueError("Saved assembly bytes changed: "+name)
        current=(saved["assembler_sha256"]==sha256_file(Path(assemble_city_cle.__code__.co_filename))
                 and all(sha256_file(CONFIG_ROOT / name)==digest for name,digest in saved["policy_hashes"].items())
                 and all(sha256_file(Path(item["path"]))==item["sha256"] for item in saved["sources"].values()))
        if current:
            print(json.dumps(dict(city=a.city,assembled=True,reused=True,summary=saved["summary"])))
            sys.exit(0)
    assembly_names=["cle_manifest.json","cle_verification.json","locations.parquet","location_access_costs.parquet","boundaries","private_sources"]
    present=[base / name for name in assembly_names if (base / name).exists()]
    if present:
        archive=a.output_root / "superseded" / (a.city+"-assembly-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
        archive.mkdir(parents=True)
        for path in present:
            path.replace(archive / path.name)
    report = assemble_city_cle(
        city_dir=base,
        boundary_paths={name: source_path(inventory, a.source_root, key) for name,key in city["boundaries"].items()},
        source_inventory_path=a.inventory, customer_registry_path=a.customer_registry,
        road_source_dir=a.road_sources / a.city,
    )
    print(json.dumps(dict(city=a.city, assembled=True, summary=report["summary"])))
else:
    for city in inventory["cities"]:
        subprocess.run([
            sys.executable, __file__, "--city", city,
            *[arg for key in ["inventory", "source_root", "road_sources", "customer_registry", "output_root"]
              for arg in ["--"+key.replace("_","-"), str(getattr(a,key).resolve())]],
        ], check=True)
