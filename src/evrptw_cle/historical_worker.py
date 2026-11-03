"""Run a saved verifier with explicitly recorded, byte-exact policy relocation.

This file is executed as a script. ``PYTHONPATH`` selects the frozen package,
not this installed implementation. Only declared policy-resource paths can be
relocated; raw geographic and statistical sources retain their original paths
and all of the saved verifier's existence/hash checks.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--aliases-sha256", required=True)
    args = parser.parse_args()
    raw = args.aliases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.aliases_sha256:
        raise ValueError("Historical policy resource map changed")
    aliases = json.loads(raw)
    snapshot = args.root.resolve() / "code_snapshot"
    receipt = json.loads((args.root / "stage1_freeze.json").read_text())
    resolved = {}
    for original, record in aliases.items():
        if record["policy_name"] in {
            "project_status.json", "phase1_sources.template.json",
            "customer_sources.template.json", "road_extraction_cohort.template.json",
        }:
            raise ValueError("A local input inventory cannot be relocated as a policy")
        name = "configs/" + record["policy_name"]
        target = (snapshot / name).resolve()
        if (not target.is_relative_to(snapshot / "configs") or
                receipt["code_snapshot"].get(name) != record["sha256"] or
                hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Unbound historical policy resource: " + original)
        resolved[original] = target

    from evrptw_cle import cle_validation

    if Path(cle_validation.__file__).resolve() != snapshot / "src/evrptw_cle/cle_validation.py":
        raise ValueError("Historical verifier was not loaded from its saved implementation")
    original_path = cle_validation.Path
    original_hash = cle_validation.sha256_file

    def resolve_path(*parts):
        path = original_path(*parts)
        return resolved.get(str(path), path)

    def resolved_hash(path):
        return original_hash(resolve_path(path))

    # Keep all original comparisons. Relocate only the policy paths whose
    # original provenance and saved replacement are declared in the audit map.
    cle_validation.Path = resolve_path
    cle_validation.sha256_file = resolved_hash
    result = cle_validation.verify_city_cle(
        args.root / "cities" / args.city, report_path=args.report)
    print(json.dumps(dict(city=args.city, passed=result["passed"],
                          checks=len(result.get("checks", [])), summary=result["summary"])))


if __name__ == "__main__":
    main()
