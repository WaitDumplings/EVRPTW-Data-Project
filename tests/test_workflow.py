import json

import pytest

from evrptw_cle import workflow
from evrptw_cle.cli import main
from evrptw_cle.util import sha256_file


def project(tmp_path):
    config = workflow.initialize(tmp_path / "project")
    root = config.parent / "inputs"
    source = root / "native.dat"
    source.write_bytes(b"native source")
    file = dict(path="native.dat", sha256=sha256_file(source))
    city = dict(
        area_crs="EPSG:32614",
        label="Dallas",
        state_fips="48",
        boundaries=dict(admin_boundary="x", service_boundary="x"),
        sources=dict(public_masks="x", osm_buildings="x", osm_complexes="x"),
        depot=dict(raw="x", extraction="x", pbf_sha256="a" * 64),
    )
    inventory = dict(
        schema="evrptw_frozen_source_archive_v1",
        files=dict(x=file),
        source_vintages={},
        cities=dict(dallas=city, other=city),
        charging={
            k: "x"
            for k in [
                "station",
                "units",
                "census",
                "station_provenance",
                "units_provenance",
                "census_provenance",
            ]
        },
    )
    # Noncanonical JSON is deliberate: prepared provenance binds these bytes.
    (root / "inventory.json").write_text(json.dumps(inventory, indent=1))
    registry = root / "customers.json"
    names = [
        "nsi_crosswalk",
        "microsoft_footprints",
        "osm_buildings",
        "osm_complexes",
        "public_masks",
    ]
    registry.write_text(
        json.dumps(
            dict(
                cities=dict(
                    dallas=dict(
                        sources={k: dict(path=str(source), sha256=file["sha256"]) for k in names}
                    )
                )
            )
        )
    )
    roads = root / "roads/dallas"
    roads.mkdir(parents=True)
    for name in ["extraction.json", "osm_ways.parquet", "osm_access_nodes.parquet"]:
        (roads / name).write_text("source")
    value = json.loads(config.read_text())
    value["sources"]["customers"] = "inputs/customers.json"
    config.write_text(json.dumps(value))
    return config


def test_init_and_missing_bundle_are_actionable(tmp_path, capsys):
    config = workflow.initialize(tmp_path / "city")
    before = config.read_bytes()
    with pytest.raises(FileExistsError):
        workflow.initialize(config.parent)
    assert config.read_bytes() == before
    with pytest.raises(SystemExit) as error:
        main(["plan", str(config)])
    assert error.value.code == 2
    assert "Missing source inventory" in capsys.readouterr().out
    assert not (config.parent / "outputs").exists()


def test_relative_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    config = project(tmp_path)
    monkeypatch.chdir(tmp_path)
    plan = workflow.plan_project(config)
    assert plan["ready"] and plan["cities"] == ["dallas"]
    assert plan["output"] == config.parent / "outputs/cle"
    assert not plan["prepare_customers"]


def test_unknown_city_and_output_inside_sources_fail_early(tmp_path):
    config = project(tmp_path)
    with pytest.raises(ValueError, match="Unknown cities"):
        workflow.plan_project(config, ["unknown"])
    value = json.loads(config.read_text())
    value["output"] = "inputs/cle"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="outside"):
        workflow.plan_project(config)


def test_prepared_inventory_identity_and_selected_cohort_are_preserved(tmp_path, monkeypatch):
    config = project(tmp_path)
    calls = []
    monkeypatch.setattr(
        workflow, "_run", lambda script, args, output, label: calls.append((script, args, label))
    )
    monkeypatch.setattr(
        workflow, "verify_output", lambda output: calls.append(("verify", [], str(output)))
    )
    output = workflow.build_project(config)
    assert (output / ".work/inventory.json").read_bytes() == (
        config.parent / "inputs/inventory.json"
    ).read_bytes()
    assembled = [x for x in calls if x[0] == "assemble_stage1.py"]
    assert len(assembled) == 1 and assembled[0][1][:2] == ["--city", "dallas"]
    assert calls[-1][0] == "verify"
    assert not any("pytest" in x[0] or "baseline" in x[0] for x in calls)
    value = json.loads(config.read_text())
    value["cities"] = ["other"]
    config.write_text(json.dumps(value))
    # Missing inputs or changed run selection must never rewrite the first run.
    with pytest.raises(ValueError):
        workflow.build_project(config)
    assert json.loads((output / ".work/project.json").read_text())["cities"] == ["dallas"]


def test_frozen_output_is_not_modified(tmp_path):
    config = project(tmp_path)
    output = config.parent / "outputs/cle"
    output.mkdir(parents=True)
    freeze = output / "stage1_freeze.json"
    freeze.write_text('{"frozen":true}')
    with pytest.raises(FileExistsError, match="frozen"):
        workflow.build_project(config)
    assert freeze.read_text() == '{"frozen":true}'
    assert not (output / ".work").exists()


def test_verify_uses_recorded_selection_not_every_inventory_city(tmp_path, monkeypatch):
    root = tmp_path / "out"
    work = root / ".work"
    work.mkdir(parents=True)
    (work / "inventory.json").write_text("{}")
    (work / "project.json").write_text('{"cities":["dallas"]}')
    (root / "stage1_verification.json").write_text('{"passed":true,"city_count":1}')
    calls = []
    monkeypatch.setattr(workflow, "_run", lambda *args: calls.append(args))
    assert workflow.verify_output(root)["passed"]
    assert calls[0][1][-2:] == ["--cities", "dallas"]


def test_all_internal_commands_are_available():
    for name in [
        "build_stage1_cohort.py",
        "build_stage1_city.py",
        "assemble_stage1.py",
        "prepare_all_customer_sources.py",
        "prepare_customer_sources.py",
        "extract_frozen_access_roads.py",
        "verify_stage1.py",
    ]:
        assert (workflow.COMMAND_ROOT / name).is_file()
