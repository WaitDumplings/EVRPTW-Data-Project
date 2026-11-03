"""Inspect and re-audit an immutable CLE with its receipt-bound implementation.

Normal CLE consumption uses ``CityLogisticsEnvironment``. A historical re-audit
instead launches the saved verifier in a separate Python process, with the saved
package and policy resources on its import path. It never rewrites the old run.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .util import sha256_file, write_json

_LOCAL_INPUT_NAMES = {"project_status.json", "phase1_sources.template.json",
                      "customer_sources.template.json", "road_extraction_cohort.template.json"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_history(output):
    """Check the saved receipt, implementation and city certificate bindings.

    This is a small metadata check, not another scan of all city data or raw
    inputs. The full historical verifier performs those checks when requested.
    """
    root = Path(output).expanduser().resolve()
    receipt_path = root / "stage1_freeze.json"
    receipt = _json(receipt_path)
    if receipt.get("schema") != "evrptw_stage1_internal_freeze_v1" or not receipt.get("frozen"):
        raise ValueError("Historical verification requires a frozen Stage 1 receipt")
    snapshot = root / "code_snapshot"
    records = receipt.get("code_snapshot", {})
    required = {"scripts/verify_stage1.py", "src/evrptw_cle/cle_validation.py",
                "src/evrptw_cle/__init__.py", "pyproject.toml"}
    if not required <= records.keys():
        raise ValueError("Frozen receipt does not bind the complete verifier implementation")
    for name, digest in records.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Frozen implementation file changed or missing: " + name)
    cities = receipt.get("cities", {})
    if not cities or {p.name for p in (root / "cities").iterdir() if p.is_dir()} != set(cities):
        raise ValueError("Saved city cohort differs from its frozen receipt")
    for city, record in cities.items():
        city_dir = root / "cities" / city
        for filename, key in [("cle_manifest.json", "cle_manifest_sha256"),
                              ("cle_verification.json", "verification_sha256")]:
            if sha256_file(city_dir / filename) != record[key]:
                raise ValueError(f"Frozen city binding changed: {city}/{filename}")
        certificate = _json(city_dir / "cle_verification.json")
        if not certificate.get("passed") or certificate["cle_manifest_sha256"] != record["cle_manifest_sha256"]:
            raise ValueError("Frozen city lacks a matching passed certificate: " + city)
    manifest = _json(root / "cities" / sorted(cities)[0] / "cle_manifest.json")
    inventory = manifest["sources"]["frozen_inventory"]
    return dict(
        schema="evrptw_stage1_history_info_v1", output=str(root),
        frozen=True, frozen_generated_utc=receipt.get("generated_utc"),
        freeze_sha256=sha256_file(receipt_path), code_snapshot=str(snapshot),
        snapshot_files_verified=len(records), cities=sorted(cities), city_count=len(cities),
        inventory=inventory, source_inventory_present=Path(inventory["path"]).is_file(),
        verifier_sha256=records["src/evrptw_cle/cle_validation.py"],
        metadata_bindings_verified=True, full_city_data_reverified=False,
    )


def _policy_resource_aliases(root, selected):
    """Relocate declared policy inputs to identical receipt-bound saved bytes.

    Old provenance can mention an absolute construction-time configs path in
    addition to the verifier's CONFIG_ROOT. No raw source path is substituted.
    """
    receipt = _json(root / "stage1_freeze.json")
    snapshot = root / "code_snapshot"
    aliases = {}
    for city in selected:
        base = root / "cities" / city
        manifest = _json(base / "cle_manifest.json")
        declared = manifest.get("policy_hashes", {})

        def visit(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    original = Path(value["path"])
                    if original.is_absolute():
                        for name, digest in declared.items():
                            # Old manifests enumerated every config JSON,
                            # including source inventories. They are inputs,
                            # not implementation policies, and stay unmapped.
                            if name in _LOCAL_INPUT_NAMES:
                                continue
                            suffix = Path("configs") / name
                            if (original.parts[-len(suffix.parts):] == suffix.parts and
                                    value["sha256"] == digest):
                                bound = "configs/" + name
                                target = snapshot / bound
                                if (receipt["code_snapshot"].get(bound) != digest or
                                        sha256_file(target) != digest):
                                    raise ValueError("Historical policy is not bound by receipt: " + name)
                                record = dict(policy_name=name, sha256=digest,
                                              resolved_path=str(target))
                                old = aliases.setdefault(str(original), record)
                                if old != record:
                                    raise ValueError("Conflicting historical policy resource: " + str(original))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(manifest)
        for path in manifest.get("components", {}).values():
            visit(_json(base / path))
    return aliases


def verify_historical(output, *, report_dir=None, cities=None):
    """Run original city verifiers serially, writing only to a fresh external run.

    The original source files must still resolve at the paths recorded by the
    frozen manifests. Reading the CLE for Stage 2 does not have that requirement.
    """
    info = inspect_history(output)
    root, snapshot = Path(info["output"]), Path(info["code_snapshot"])
    selected = info["cities"] if cities is None else list(cities)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(info["cities"]):
        raise ValueError("Choose unique cities present in the frozen receipt")
    inventory = Path(info["inventory"]["path"])
    if not inventory.is_file() or sha256_file(inventory) != info["inventory"]["sha256"]:
        raise ValueError("Original frozen source inventory is missing or changed: " + str(inventory))
    now = datetime.now(UTC)
    reports = (Path(report_dir).expanduser().resolve() if report_dir is not None else
               root.parent / (root.name + "_reaudits") / now.strftime("%Y%m%dT%H%M%S%fZ"))
    if reports == root or reports.is_relative_to(root):
        raise ValueError("Historical reports must be outside the immutable frozen cohort")
    if reports.exists() and any(reports.iterdir()):
        raise FileExistsError("Use a fresh historical report directory: " + str(reports))
    reports.mkdir(parents=True, exist_ok=True)
    aliases = _policy_resource_aliases(root, selected)
    alias_path = reports / "policy_resource_aliases.json"
    write_json(alias_path, aliases)
    worker = Path(__file__).with_name("historical_worker.py")
    status = dict(schema="evrptw_stage1_historical_reaudit_v1", state="running", passed=False,
                  generated_utc=now.isoformat(), original_freeze_sha256=info["freeze_sha256"],
                  frozen_output=str(root), report_dir=str(reports),
                  snapshot_files_verified=info["snapshot_files_verified"],
                  snapshot_verifier_sha256=info["verifier_sha256"], cities={}, city_count=0,
                  resource_resolver_sha256=sha256_file(worker),
                  policy_resource_aliases_sha256=sha256_file(alias_path),
                  policy_resource_aliases=aliases,
                  reports_written_outside_frozen_cohort=True)
    write_json(reports / "stage1_verification.json", status)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    for city in selected:
        print("VERIFY HISTORICAL", city, flush=True)
        report = reports / (city + ".json")
        command = [sys.executable, "-B", str(worker), "--root", str(root),
                   "--city", city, "--report", str(report),
                   "--aliases", str(alias_path),
                   "--aliases-sha256", status["policy_resource_aliases_sha256"]]
        try:
            with (reports / (city + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=snapshot, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result = _json(report)
            expected = _json(root / "stage1_freeze.json")["cities"][city]
            if (not result.get("passed") or
                    result.get("cle_manifest_sha256") != expected["cle_manifest_sha256"] or
                    result.get("verifier_sha256") != info["verifier_sha256"]):
                raise ValueError("Historical verifier returned an unbound result: " + city)
            status["cities"][city] = dict(
                verification_sha256=sha256_file(report),
                cle_manifest_sha256=result["cle_manifest_sha256"],
                summary=result["summary"], verified_terminal_paths=result["verified_terminal_paths"])
            status["city_count"] = len(status["cities"])
            write_json(reports / "stage1_verification.json", status)
            print("VERIFIED HISTORICAL", city, flush=True)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            status.update(state="failed", failed_city=city)
            write_json(reports / "stage1_verification.json", status)
            raise RuntimeError(f"Historical verification failed for {city}; see {reports / (city + '.log')}") from error
    if inspect_history(root)["freeze_sha256"] != info["freeze_sha256"]:
        raise ValueError("Frozen receipt changed during historical verification")
    status.update(state="complete", passed=True,
                  full_selected_city_data_reverified=True,
                  source_inventory_sha256=sha256_file(inventory))
    write_json(reports / "stage1_verification.json", status)
    return status
