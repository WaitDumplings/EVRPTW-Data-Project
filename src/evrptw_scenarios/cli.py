"""A separate Stage 2 command; never changes the frozen Stage 1 builder."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .config import DEFAULT, load_config
from .util import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evrptw-stage2", description="Build daily delivery scenarios from a frozen CLE")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write a commented-in-documentation scenario template")
    init.add_argument("path")
    init.add_argument("--legacy", action="store_true", help="write the earlier nested natural-day profile")
    plan = commands.add_parser("plan", help="check inputs and show declared assumptions without building")
    plan.add_argument("config")
    build = commands.add_parser("build", help="build configured territory days or legacy workloads and verify saved results")
    build.add_argument("config")
    marks = commands.add_parser("prepare-marks", help="prepare the existing Amazon package attributes without feasibility filtering")
    marks.add_argument("--packages", required=True)
    marks.add_argument("--routes", required=True)
    marks.add_argument("--output", required=True)
    marks.add_argument("--partition", default="training")
    verify = commands.add_parser("verify", help="recheck saved artifacts independently")
    verify.add_argument("run")
    verify.add_argument("--check-sources", action="store_true")
    inspect = commands.add_parser("inspect", help="summarize natural workloads and fixed-N view support")
    inspect.add_argument("run")
    export = commands.add_parser("export", help="export one representable same-path numeric solver input")
    export.add_argument("run")
    export.add_argument("view_id")
    export.add_argument("--output", required=True)
    export.add_argument("--objective", choices=["total_distance", "total_energy"], default="total_distance")
    export.add_argument("--path-policy", choices=["shortest_distance", "fastest_time"], default="shortest_distance")
    validation = commands.add_parser("validate-modules", help="run seeded small distribution comparisons without city data")
    validation.add_argument("--output", required=True)
    validation.add_argument("--samples", type=int, default=12000)
    validation.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            path = Path(args.path)
            if path.exists():
                raise FileExistsError(path)
            if args.legacy:
                config = deepcopy(DEFAULT)
            else:
                from .territory_config import DEFAULT as TERRITORY_DEFAULT
                config = deepcopy(TERRITORY_DEFAULT)
            config["input"] = {"cle": "path/to/frozen/cities/your-city", "communities": "path/to/tiger2025_city_bg.parquet", "donor_artifact": "path/to/amazon_marks"}
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, config)
            print(f"Created {path}. Set the three input paths, review scenario assumptions, then run plan.")
        elif args.command == "prepare-marks":
            from .marks import build_donor_artifact
            result = build_donor_artifact(package_path=args.packages, route_path=args.routes,
                                          output_dir=args.output, partition=args.partition)
            print(json.dumps(result, indent=2))
        elif args.command == "plan":
            config = load_config(args.config)
            missing = [path for path in config["input"].values() if not Path(path).exists()]
            if missing:
                raise FileNotFoundError("Missing input paths: " + ", ".join(missing))
            print(json.dumps({"inputs_present": True, "source_hashes_verified": False,
                              "world": config["world"], "operations": config["operations"],
                              "workload": config["workload"],
                              "activation": config.get("activation"),
                              "instances": config.get("instances", config.get("views")),
                              "physical": config["physical"],
                              "output": config["output"],
                              "parameter_status": "declared synthetic scenario; not fitted national order rates",
                              "failure_policy": "retain unsupported sizes/TWs/loads; no redraw or repair"}, indent=2))
        elif args.command == "build":
            from .build import build_run
            result = build_run(load_config(args.config))
            print(json.dumps(result, indent=2))
            return 0 if result.get("passed") else 1
        elif args.command == "verify":
            from .verify import verify_run
            result = verify_run(args.run, check_sources=args.check_sources)
            print(json.dumps(result, indent=2))
            return 0 if result.get("passed") else 1
        elif args.command == "inspect":
            from .reader import ScenarioRun
            run = ScenarioRun(args.run)
            print(run.operational_days().to_string(index=False))
            print(run.views().to_string(index=False))
        elif args.command == "export":
            from .consumer import export_solver_input
            print(export_solver_input(args.run, args.view_id, args.output,
                                      objective=args.objective, path_policy=args.path_policy))
        elif args.command == "validate-modules":
            from .validation import validate_modules
            result = validate_modules(samples=args.samples, seed=args.seed, output=args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, f"Stage 2: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
