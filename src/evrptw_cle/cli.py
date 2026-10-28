"""Build and read Stage 1 city logistics environments from local source bundles."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .context_validation import validate_context_bindings
from .runtime_paths import CONFIG_ROOT
from .util import sha256_file, write_json


def validate_inputs(registry_path):
    """Check exact archived inputs, never substitute a new live download."""
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text())
    root = registry_path.parent.parent
    checks = []
    for city, entry in registry["cities"].items():
        for group in ["sources", "provenance"]:
            for name, item in entry.get(group, {}).items():
                path = root / item["path"]
                exists = path.is_file()
                passed = exists and sha256_file(path) == item["sha256"]
                checks.append(
                    dict(
                        city=city,
                        input=name,
                        group=group,
                        status="verified"
                        if passed
                        else "missing"
                        if not exists
                        else "hash_mismatch",
                    )
                )
    return dict(
        ready=bool(checks) and all(c["status"] == "verified" for c in checks),
        rule="exact frozen file hashes; no live/latest substitution",
        checks=checks,
    )


def require_current_context(path, **bindings):
    path = Path(path)
    manifest = json.loads((path / "access_context_manifest.json").read_text())
    validate_context_bindings(manifest, **bindings)
    return manifest


def roads(args):
    from .delivery_access import build_delivery_access_context
    from .local_road_access import build_access_context

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; never overwrite a verified bundle")
    options = dict(
        source_dir=args.source_dir,
        admin_boundary_path=args.admin_boundary,
        service_boundary_path=args.service_boundary,
        mask_path=args.masks,
        area_crs=args.area_crs,
    )
    seed = args.physical_seed
    if seed is None:
        seed = args.output / "_physical_seed"
        temporary, _ = build_access_context(**options, output_dir=seed)
        del temporary
        gc.collect()
    _, manifest = build_delivery_access_context(**options, seed_dir=seed, output_dir=args.output)
    print(
        json.dumps(
            dict(
                component="road_access",
                policy=manifest["policy"],
                osm_snapshot=manifest["source"]["pbf_replication_timestamp_utc"],
                whole_cle_frozen=False,
            ),
            indent=2,
        )
    )


def customers(args):
    from .residential_sites import build_registered_site_layer

    require_current_context(args.context)
    report, rebuilt = build_registered_site_layer(
        registry_path=args.registry,
        city_slug=args.city,
        boundary_path=args.boundary,
        graph_path=args.context / "graph_operational.graphml",
        output_dir=args.output,
    )
    print(json.dumps(dict(rebuilt=rebuilt, summary=report["summary"]), indent=2))


def depots(args):
    import geopandas as gpd
    from .depot_candidates import build_public_depot_candidates
    from .facilities import build_depot_layer
    from .local_road_access import LocalAccessContext
    from .protected_connectivity import build_directed_component_index

    manifest = require_current_context(
        args.context,
        city_slug=args.city,
        boundary_path=args.boundary,
        mask_path=args.masks,
        area_crs=args.area_crs,
    )
    provenance = json.loads(args.source_manifest.read_text())
    if provenance["pbf_sha256"] != manifest["source"]["pbf_sha256"]:
        raise ValueError("Depot and road OSM snapshots differ")
    if sha256_file(args.raw_depots) != provenance["raw_depot_sha256"]:
        raise ValueError("Depot raw evidence hash differs")
    if sha256_file(args.masks) != manifest["mask_sha256"]:
        raise ValueError("Depot and road exclusion masks differ")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh depot output directory")
    masks, boundary = gpd.read_parquet(args.masks), gpd.read_file(args.boundary)
    candidates, grouping = build_public_depot_candidates(
        gpd.read_parquet(args.raw_depots), boundary, args.city, masks
    )
    points = candidates.to_crs(4326).copy()
    points["access_search_extent_wkt"] = points.geometry.to_wkt()
    points.geometry = points.to_crs(args.area_crs).geometry.representative_point().to_crs(4326)
    points = points.loc[
        points.geometry.covered_by(boundary.to_crs(4326).geometry.union_all())
    ].reset_index(drop=True)
    points["longitude"], points["latitude"] = points.geometry.x, points.geometry.y
    context = LocalAccessContext.load(args.context)
    frame, access = build_depot_layer(
        points,
        None,
        build_directed_component_index(context.core),
        args.area_crs,
        masks=masks,
        access_context=context,
        footprints=gpd.read_parquet(args.footprints) if getattr(args, "footprints", None) else None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "depots.parquet"
    frame.to_parquet(path, index=False)
    result = dict(
        city=args.city,
        policy=manifest["policy"],
        source_sha256=sha256_file(args.raw_depots),
        building_obstacles=dict(path=str(args.footprints.resolve()),sha256=sha256_file(args.footprints)) if getattr(args,"footprints",None) else None,
        source_provenance=provenance,
        boundary_sha256=sha256_file(args.boundary),
        mask_sha256=sha256_file(args.masks),
        generation_dependencies={
            name: sha256_file(Path(__file__).with_name(name)) for name in [
                "cli.py", "facilities.py", "depot_candidates.py", "depot_grouping.py",
                "delivery_access.py", "frontage.py", "local_road_access.py", "customer_access.py",
                "terminal_access.py", "public_locations.py",
            ]
        },
        context_sha256=sha256_file(args.context / "access_context_manifest.json"),
        output_sha256=sha256_file(path),
        grouping=grouping,
        access_audit=access,
        available_unique_groups=int(
            frame.loc[frame.depot_candidate_eligible].facility_group_id.nunique()
        ),
        whole_cle_frozen=False,
    )
    write_json(args.output / "depot_component_manifest.json", result)
    print(
        json.dumps(
            dict(available_unique_groups=result["available_unique_groups"], whole_cle_frozen=False)
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")
    init = sub.add_parser("init", help="Create a project.json and local input folder")
    init.add_argument("directory", type=Path)
    for name, help_text in [("plan", "List required inputs and selected cities without building"),
                            ("build", "Build complete CLEs; resume unchanged components")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--cities", nargs="+", help="Override the city selection in project.json")
        if name == "plan":
            command.add_argument("--json", action="store_true")
        else:
            command.add_argument("--no-verify", action="store_true", help="Defer final data validation to the verify command")
    check = sub.add_parser("verify", help="Independently validate a built CLE cohort")
    check.add_argument("output", type=Path)
    inspect = sub.add_parser("inspect", help="Show location counts and saved certificate status")
    inspect.add_argument("output", type=Path)
    inspect.add_argument("--json", action="store_true")
    sub.add_parser("status", help="Show supported capabilities")
    v = sub.add_parser("validate-inputs", help="Verify archived customer source files")
    v.add_argument("registry", type=Path)
    r = sub.add_parser(
        "roads", help="Build directed public and purpose-specific terminal components"
    )
    for name in ["source-dir", "admin-boundary", "service-boundary", "masks", "output"]:
        r.add_argument("--" + name, type=Path, required=True)
    r.add_argument("--area-crs", required=True)
    r.add_argument("--physical-seed", type=Path)
    c = sub.add_parser("customers", help="Construct canonical residential potential locations")
    c.add_argument("--city", required=True)
    for name in ["registry", "boundary", "context", "output"]:
        c.add_argument("--" + name, type=Path, required=True)
    d = sub.add_parser("depots", help="Construct unique cross-operator departure locations")
    d.add_argument("--city", required=True)
    d.add_argument("--area-crs", required=True)
    d.add_argument("--footprints", type=Path, required=True)
    for name in ["raw-depots", "source-manifest", "boundary", "masks", "context", "output"]:
        d.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
    elif args.command in ["init", "plan", "build", "verify", "inspect"]:
        from .workflow import build_project, initialize, inspect_output, plan_project, verify_output

        try:
            if args.command == "init":
                path = initialize(args.directory)
                print(f"Created {path}\nAdd your local source bundle, then run:\n  evrptw-data plan {path}")
            elif args.command == "plan":
                plan = plan_project(args.config, args.cities)
                if args.json:
                    print(json.dumps({k: v for k, v in plan.items() if k != "inventory"}, default=str, indent=2))
                else:
                    print("Input status: " + ("ready" if plan["ready"] else "not ready"))
                    print("Cities: " + (", ".join(plan["cities"]) or "not available yet"))
                    print("Output: " + str(plan["output"]))
                    for issue in plan["issues"]:
                        print("  - " + issue)
                    if plan["ready"]:
                        print("Build: source preparation (if needed) -> roads -> locations and speeds -> CLE -> data validation")
                if not plan["ready"]:
                    raise SystemExit(2)
            elif args.command == "build":
                output = build_project(args.config, args.cities, verify=not args.no_verify)
                print(f"Build complete: {output}\nView counts: evrptw-data inspect {output}")
            elif args.command == "verify":
                result = verify_output(args.output)
                print(f"Data validation passed: {result['city_count']} cities")
            else:
                report = inspect_output(args.output)
                if args.json:
                    print(json.dumps(report, indent=2))
                else:
                    print(f"CLEs: {report['output']}")
                    print(f"{'City':<18} {'Customer':>12} {'Depot':>8} {'Charging':>10} {'Certificate':>12}")
                    for row in report["cities"]:
                        summary = row["summary"]
                        print(f"{row['city']:<18} {summary['customer']['available']:>12,} "
                              f"{summary['depot']['available']:>8,} {summary['charging']['available']:>10,} "
                              f"{'passed' if row['verified'] else 'missing':>12}")
                    print("Counts show eligible potential locations. No instance activation is performed.")
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            p.error(str(error))
    elif args.command == "status":
        print((CONFIG_ROOT / "project_status.json").read_text())
    elif args.command == "validate-inputs":
        result = validate_inputs(args.registry)
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(2)
    else:
        try:
            {"roads": roads, "customers": customers, "depots": depots}[args.command](args)
        except (ValueError, FileNotFoundError, FileExistsError) as e:
            p.error(str(e))
