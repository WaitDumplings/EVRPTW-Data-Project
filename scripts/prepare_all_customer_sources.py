"""Prepare the complete frozen cohort in isolated city processes."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from evrptw_cle.source_preparation import assemble_customer_source_registry
from evrptw_cle.util import sha256_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--inventory", type=Path, required=True)
p.add_argument("--source-root", type=Path, required=True)
p.add_argument("--prepared-root", type=Path, required=True)
p.add_argument("--registry", type=Path, required=True)
a = p.parse_args()
inventory = json.loads(a.inventory.read_text())
for city in inventory["cities"]:
    output = a.prepared_root / city
    manifest = output / "source_preparation_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if m["inventory_sha256"] != sha256_file(a.inventory):
            raise ValueError("Prepared source inventory differs: " + city)
        for file, record in m["outputs"].items():
            if sha256_file(output / record["path"]) != record["sha256"]:
                raise ValueError("Changed prepared input: " + city + ":" + file)
        import evrptw_cle.source_preparation as source_module

        for name, digest in m["generation_dependencies"].items():
            if sha256_file(Path(source_module.__file__).with_name(name)) != digest:
                raise ValueError("Prepared source code differs: " + city + ":" + name)
        print("VERIFIED REUSE", city, flush=True)
        continue
    if output.exists():
        archive = a.prepared_root / "incomplete" / (city + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
        archive.parent.mkdir(parents=True, exist_ok=True)
        output.replace(archive)
    a.prepared_root.mkdir(parents=True, exist_ok=True)
    with (a.prepared_root / (city + ".log")).open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("prepare_customer_sources.py")),
                "--inventory",
                str(a.inventory.resolve()),
                "--source-root",
                str(a.source_root.resolve()),
                "--city",
                city,
                "--output",
                str(output.resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("PREPARED", city, flush=True)
if a.registry.exists():
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / "registry.json"
        assemble_customer_source_registry(
            inventory_path=a.inventory, source_root=a.source_root,
            prepared_root=a.prepared_root, output_path=expected,
        )
        if json.loads(a.registry.read_text()) != json.loads(expected.read_text()):
            raise ValueError("Existing customer registry differs from the prepared sources")
else:
    assemble_customer_source_registry(
        inventory_path=a.inventory, source_root=a.source_root, prepared_root=a.prepared_root,
        output_path=a.registry,
    )
print("REGISTRY", a.registry, flush=True)
