"""Configuration-driven Stage 1 construction, independent of a research checkout."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

from .frozen_inputs import archive_path, validate_archive_inventory
from .runtime_paths import COMMAND_ROOT
from .util import sha256_file, write_json

SCHEMA = "evrptw_stage1_project_v1"
EXAMPLE = {
    "schema": SCHEMA,
    "sources": {
        "inventory": "inputs/inventory.json",
        "root": "inputs",
        "roads": "inputs/roads",
        "customers": None,
        "osm_cohort": None,
    },
    "output": "outputs/cle",
    "cities": ["dallas"],
}


def initialize(directory):
    root = Path(directory).expanduser().resolve()
    target = root / "project.json"
    if target.exists():
        raise FileExistsError(f"Project already exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    write_json(target, EXAMPLE)
    (root / "inputs/README.md").write_text(
        "# Local source bundle\n\n"
        "Place inventory.json and its source files here. Paths are relative to this folder.\n"
        "The code repository does not distribute the city data bundle.\n"
        "See docs/INPUTS.md in the repository for the manifest and input formats.\n"
        "Run `evrptw-data plan project.json` before starting a build.\n",
        encoding="utf-8",
    )
    return target


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: line {error.lineno}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Project paths must be nonempty strings")
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _references(value, files):
    if isinstance(value, str) and value in files:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _references(child, files)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child, files)


def plan_project(config, cities=None):
    """Inspect paths cheaply; component consumers check actual input hashes."""
    path = Path(config).expanduser().resolve()
    settings = _json(path)
    if settings.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA} in {path}")
    sources = settings.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("sources must be a JSON object")
    paths = {name: _resolve(path.parent, sources[name]) for name in ["inventory", "root", "roads"]}
    for name in ["customers", "osm_cohort"]:
        paths[name] = _resolve(path.parent, sources[name]) if sources.get(name) else None
    output = _resolve(path.parent, settings.get("output"))
    if any(
        output == item or output.is_relative_to(item) for item in [paths["root"], paths["roads"]]
    ):
        raise ValueError("output must be outside the read-only source directories")
    issues = []
    if not paths["inventory"].is_file():
        issues.append(f"Missing source inventory: {paths['inventory']} (see docs/INPUTS.md)")
        return dict(ready=False, issues=issues, config=path, paths=paths, output=output, cities=[])
    inventory = _json(paths["inventory"])
    for record in inventory.get("files", {}).values():
        record.setdefault("archive_path", record.get("path"))
        record.setdefault("private_build_input", False)
    validate_archive_inventory(inventory)
    available = inventory.get("cities", {})
    if not isinstance(available, dict) or not available:
        raise ValueError("Source inventory must contain at least one city")
    selected = cities if cities is not None else settings.get("cities")
    if selected is None:
        selected = list(available)
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(c, str) for c in selected)
    ):
        raise ValueError("cities must be a nonempty list of city slugs, or null for all cities")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate city in selection")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError("Unknown cities: " + ", ".join(sorted(unknown)))
    charging = inventory.get("charging", {})
    for name in [
        "station",
        "units",
        "census",
        "station_provenance",
        "units_provenance",
        "census_provenance",
    ]:
        if charging.get(name) not in inventory["files"]:
            issues.append(f"charging.{name} must reference a source file key")
    required = set(_references(inventory.get("charging", {}), inventory["files"]))
    for city in selected:
        entry = available[city]
        for name in ["boundaries", "sources", "depot", "area_crs"]:
            if name not in entry:
                issues.append(f"{city}: inventory is missing {name}")
        for group, names in [
            ("boundaries", ["admin_boundary", "service_boundary"]),
            ("sources", ["public_masks", "osm_buildings", "osm_complexes"]),
            ("depot", ["raw", "extraction"]),
        ]:
            for name in names:
                if entry.get(group, {}).get(name) not in inventory["files"]:
                    issues.append(f"{city}.{group}.{name} must reference a source file key")
        required.update(_references(entry, inventory["files"]))
    for key in sorted(required):
        file = archive_path(paths["root"], inventory["files"][key]["path"])
        if not file.is_file():
            issues.append(f"Missing input {key}: {file}")
    if paths["customers"] and not paths["customers"].is_file():
        issues.append(f"Missing prepared customer registry: {paths['customers']}")
    elif paths["customers"]:
        registry = _json(paths["customers"])
        for city in selected:
            if city not in registry.get("cities", {}):
                issues.append(f"Prepared customer registry does not contain {city}")
                continue
            for name in [
                "nsi_crosswalk",
                "microsoft_footprints",
                "osm_buildings",
                "osm_complexes",
                "public_masks",
            ]:
                item = registry["cities"][city].get("sources", {}).get(name, {})
                if not item.get("path") or not item.get("sha256"):
                    issues.append(f"Prepared registry: {city}.{name} requires path and sha256")
                elif not _resolve(paths["customers"].parent.parent, item["path"]).is_file():
                    issues.append(f"Prepared registry input is missing: {city}.{name}")
    if not paths["customers"]:
        for city in selected:
            if not all(k in available[city] for k in ["geometry", "nsi_tiles", "provenance"]):
                issues.append(
                    f"{city}: provide raw geometry/nsi_tiles/provenance or sources.customers"
                )
    needs_roads = any(
        not (paths["roads"] / city / file).is_file()
        for city in selected
        for file in ["extraction.json", "osm_ways.parquet", "osm_access_nodes.parquet"]
    )
    if needs_roads and not paths["osm_cohort"]:
        issues.append(
            "Road extractions are missing; provide inputs/roads/<city>/ or sources.osm_cohort"
        )
    if paths["osm_cohort"] and not paths["osm_cohort"].is_file():
        issues.append(f"Missing OSM cohort: {paths['osm_cohort']}")
    return dict(
        ready=not issues,
        issues=issues,
        config=path,
        paths=paths,
        output=output,
        cities=selected,
        inventory=inventory,
        prepare_roads=needs_roads,
        prepare_customers=paths["customers"] is None,
    )


def _stable_json(path, value):
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Run inputs changed: {path}. Choose a new output directory.")
    else:
        write_json(path, value)


def _stable_bytes(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Run inputs changed: {path}. Choose a new output directory.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _run(script, arguments, output, label):
    """Keep detailed worker output in a log; show useful city progress live."""
    command = COMMAND_ROOT / script
    if not command.is_file():
        raise FileNotFoundError(f"Missing packaged command: {command}; reinstall the project")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (label + ".log")
    print(f"  {label.replace('_', ' ')}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            [sys.executable, str(command), *map(str, arguments)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(
                    ("BUILD ", "COMPLETE ", "VERIFIED REUSE ", "PREPARED ", "VERIFY ")
                ):
                    print("    " + line.rstrip(), flush=True)
            code = process.wait()
    if code:
        tail = "\n".join(log_path.read_text().splitlines()[-8:])
        raise RuntimeError(f"{label} failed (exit {code}). Log: {log_path}\n{tail}")


def _build_project(config, cities=None, *, verify=True):
    plan = plan_project(config, cities)
    if not plan["ready"]:
        raise ValueError(
            "Inputs are not ready:\n" + "\n".join("- " + issue for issue in plan["issues"])
        )
    output, paths = plan["output"], plan["paths"]
    if (output / "stage1_freeze.json").exists():
        raise FileExistsError("This run is frozen. Choose a new output directory.")
    work = output / ".work"
    if output.exists() and any(output.iterdir()) and not (work / "project.json").exists():
        raise FileExistsError(f"Output is not a managed run: {output}. Choose an empty directory.")
    binding = dict(
        config=str(plan["config"]),
        cities=plan["cities"],
        inventory_sha256=sha256_file(paths["inventory"]),
        source_root=str(paths["root"]),
        road_inputs=str(paths["roads"]),
        customer_registry_sha256=sha256_file(paths["customers"]) if paths["customers"] else None,
    )
    _stable_json(work / "project.json", binding)
    inventory = dict(
        plan["inventory"], cities={c: plan["inventory"]["cities"][c] for c in plan["cities"]}
    )
    inventory_path = work / "inventory.json"
    if paths["customers"]:
        # Prepared provenance binds the original inventory bytes, not a
        # re-serialized city subset. Keep that identity intact.
        _stable_bytes(inventory_path, paths["inventory"].read_bytes())
    else:
        _stable_json(inventory_path, inventory)
    registry = work / "customer_sources.json"
    if paths["customers"]:
        original = _json(paths["customers"])
        prepared = dict(original, cities={c: original["cities"][c] for c in plan["cities"]})
        for city in prepared["cities"].values():
            for group in ["sources", "provenance"]:
                for item in city.get(group, {}).values():
                    item["path"] = str(_resolve(paths["customers"].parent.parent, item["path"]))
        _stable_json(registry, prepared)
    if plan["prepare_roads"]:
        cohort = _json(paths["osm_cohort"])
        regions = {}
        for name, record in cohort["regions"].items():
            selected = [c for c in record["cities"] if c in plan["cities"]]
            if selected:
                regions[name] = dict(
                    record,
                    cities=selected,
                    **{
                        key: str(_resolve(paths["osm_cohort"].parent, record[key]))
                        for key in ["path", "metadata"]
                    },
                )
        covered = {c for region in regions.values() for c in region["cities"]}
        if covered != set(plan["cities"]):
            raise ValueError("OSM cohort does not cover every selected city")
        cohort_path = work / "osm_cohort.json"
        _stable_json(cohort_path, dict(cohort, regions=regions))
        boundaries = {
            c: {
                k: dict(
                    path=str(archive_path(paths["root"], inventory["files"][v]["path"])),
                    sha256=inventory["files"][v]["sha256"],
                )
                for k, v in inventory["cities"][c]["boundaries"].items()
            }
            for c in plan["cities"]
        }
        _stable_json(work / "configs/city_boundaries.json", dict(cities=boundaries))
        _run(
            "extract_frozen_access_roads.py",
            [
                "--cohort",
                cohort_path,
                "--generator-root",
                work,
                "--output-root",
                paths["roads"],
                "--regions",
                *regions,
            ],
            output,
            "extract_roads",
        )
    if plan["prepare_customers"]:
        _run(
            "prepare_all_customer_sources.py",
            [
                "--inventory",
                inventory_path,
                "--source-root",
                paths["root"],
                "--prepared-root",
                work / "prepared",
                "--registry",
                registry,
            ],
            output,
            "prepare_customers",
        )
    options = [
        "--inventory",
        inventory_path,
        "--source-root",
        paths["root"],
        "--road-sources",
        paths["roads"],
        "--customer-registry",
        registry,
        "--output-root",
        output,
    ]
    status = dict(
        schema="evrptw_build_progress_v1",
        state="running",
        cities=plan["cities"],
        instance_activation_included=False,
    )
    try:
        for phase in ["roads", "locations"]:
            status.update(current_step=phase, updated_utc=datetime.now(UTC).isoformat())
            write_json(output / "build_status.json", status)
            _run(
                "build_stage1_cohort.py",
                ["--phase", phase, "--cities", *plan["cities"], *options],
                output,
                phase,
            )
        for city in plan["cities"]:
            _run("assemble_stage1.py", ["--city", city, *options], output, "assemble_" + city)
        if verify:
            verify_output(output)
        status.update(state="complete", independently_verified=verify, current_step=None)
        write_json(output / "build_status.json", status)
    except (ValueError, OSError, RuntimeError):
        status.update(state="failed")
        write_json(output / "build_status.json", status)
        raise
    return output


def build_project(config, cities=None, *, verify=True):
    """Serialize builds of one output; kernel locks release after interruption."""
    import fcntl

    plan = plan_project(config, cities)
    if not plan["ready"]:
        raise ValueError(
            "Inputs are not ready:\n" + "\n".join("- " + issue for issue in plan["issues"])
        )
    output = plan["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".build.lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another build is already using {output}") from error
        return _build_project(config, cities, verify=verify)


def verify_output(output):
    output = Path(output).expanduser().resolve()
    inventory = output / ".work/inventory.json"
    selection = []
    project = output / ".work/project.json"
    if project.is_file():
        selection = ["--cities", *_json(project)["cities"]]
    if not inventory.is_file():
        manifests = sorted((output / "cities").glob("*/cle_manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No assembled CLEs in {output}")
        inventory = Path(_json(manifests[0])["sources"]["frozen_inventory"]["path"])
    _run(
        "verify_stage1.py",
        ["--root", output, "--inventory", inventory, *selection],
        output,
        "verify",
    )
    return _json(output / "stage1_verification.json")


def inspect_output(output):
    root = Path(output).expanduser().resolve()
    rows = []
    for path in sorted((root / "cities").glob("*/cle_manifest.json")):
        manifest = _json(path)
        certificate = path.with_name("cle_verification.json")
        verified = certificate.is_file() and _json(certificate).get("passed") is True
        verified = verified and _json(certificate).get("cle_manifest_sha256") == sha256_file(path)
        rows.append(
            dict(city=manifest["city_slug"], summary=manifest["summary"], verified=verified)
        )
    if not rows:
        raise FileNotFoundError(f"No assembled CLEs in {root}")
    return dict(output=str(root), cities=rows, frozen=(root / "stage1_freeze.json").is_file())
