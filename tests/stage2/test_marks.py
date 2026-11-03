import json

import numpy as np
import pytest

from evrptw_scenarios.marks import (
    DonorLibrary, build_donor_artifact, instantiate_availability, reconcile_one_visit,
)


def raw_mark(status="DELIVERED", *, start=float("nan"), end=float("nan")):
    return {"scan_status": status, "planned_service_time_seconds": 10,
            "dimensions": {"depth_cm": 1000, "height_cm": 1000, "width_cm": 1000},
            "time_window": {"start_time_utc": start, "end_time_utc": end}}


@pytest.fixture
def donors(tmp_path):
    packages = {"r": {"a": {"p1": raw_mark()}, "b": {"p2": raw_mark(), "p3": raw_mark()},
                      "attempt": {"p1": raw_mark("DELIVERY_ATTEMPTED"), "p4": raw_mark("DELIVERY_ATTEMPTED")},
                      "ambiguous_a": {"p5": raw_mark()}, "ambiguous_b": {"p5": raw_mark()}}}
    routes = {"r": {"station_code": "DLA1", "date_YYYY_MM_DD": "2018-07-19",
                    "stops": {key: {"type": "Dropoff"} for key in packages["r"]}}}
    package_path, route_path = tmp_path / "packages.json", tmp_path / "routes.json"
    package_path.write_text(json.dumps(packages))
    route_path.write_text(json.dumps(routes))
    output = tmp_path / "artifact"
    build_donor_artifact(package_path=package_path, route_path=route_path, output_dir=output)
    return DonorLibrary(output)


def test_event_dedup_and_no_capacity_filter(donors):
    counts = donors.manifest["counts"]
    assert counts["canonical_delivered_parcels"] == 3
    assert counts["canonical_attempt_only_parcels"] == 1
    assert counts["quarantine_multiple_delivered_events"] == 1
    bundle = donors.sample_bundle(2, np.random.default_rng(0))
    assert bundle["joint_source_bundle_preserved"]
    assert sum(p["volume_cm3"] for p in bundle["parcels"]) == 2_000_000_000
    assert all(p["source_tw_status"] == "missing" for p in bundle["parcels"])
    assert donors.manifest["no_vehicle_or_route_feasibility_filter"]


def test_source_support_extrapolation_never_truncates(donors):
    bundle = donors.sample_bundle(200, np.random.default_rng(0))
    assert len(bundle["parcels"]) == 200
    assert bundle["count_outside_observed_range"]
    assert not bundle["joint_source_bundle_preserved"]
    assert bundle["joint_package_tuple_preserved"]
    assert bundle["source_package_count"] == 2
    with pytest.raises(ValueError):
        donors.sample_bundle(True, np.random.default_rng(0))


def test_empty_window_preserves_same_demand(donors):
    parcels = instantiate_availability(donors.sample_bundle(2, np.random.default_rng(0)), service_day_start_s=0)
    good = reconcile_one_visit(parcels, releases_s=[0, 10], deadlines_s=[100, 100], operating_window_s=[0, 100])
    bad = reconcile_one_visit(parcels, releases_s=[90, 90], deadlines_s=[100, 100], operating_window_s=[0, 100])
    assert (good["tw_start_s"], good["tw_end_s"]) == (10, 80)
    assert good["supported_one_visit_representation"]
    assert not bad["supported_one_visit_representation"]
    assert "empty_one_visit_start_window" in bad["reasons"]
    assert bad["demand_cm3"] == good["demand_cm3"]
    assert bad["parcel_count"] == 2
    assert bad["planned_service_time_s"] == 20


def test_completion_bounds_and_explicit_source_end(donors):
    parcels = instantiate_availability(donors.sample_bundle(2, np.random.default_rng(0)), service_day_start_s=0)
    for parcel in parcels:
        parcel.update(source_tw_status="specified", availability_start_s=30, availability_end_s=50)
    start = reconcile_one_visit(parcels, releases_s=[0, 0], deadlines_s=[100, 100], operating_window_s=[0, 100])
    complete = reconcile_one_visit(parcels, releases_s=[0, 0], deadlines_s=[100, 100], operating_window_s=[0, 100], source_window_end="service_completion")
    assert (start["tw_start_s"], start["tw_end_s"]) == (30, 50)
    assert (complete["tw_start_s"], complete["tw_end_s"]) == (30, 30)
    assert start["promises_and_shift_require_completion"]


def test_uninstantiated_or_unknown_source_window_not_silently_discarded(donors):
    parcels = donors.sample_bundle(1, np.random.default_rng(0))["parcels"]
    parcels[0].update(source_tw_status="specified", source_tw_start_offset_s=20, source_tw_end_offset_s=50)
    result = reconcile_one_visit(parcels, releases_s=[0], deadlines_s=[100], operating_window_s=[0, 100])
    assert "source_window_not_instantiated" in result["reasons"]
    parcels[0]["source_tw_status"] = "missing"
    result = reconcile_one_visit(parcels, releases_s=[0], deadlines_s=[100], operating_window_s=[0, 100], missing_window_policy="unknown_unsupported")
    assert "source_availability_unknown" in result["reasons"]


def test_hash_tamper_rejected(donors):
    (donors.path / "quarantine.parquet").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Changed donor artifact"):
        DonorLibrary(donors.path)



def test_calibrated_nonexact_service_residual_and_provenance(donors):
    from evrptw_scenarios.calibration import service_curve
    donors.configure_calibration({"donor_pool": "declared_pool", "minimum_donor_bundles": 1,
                                  "minimum_donor_station_days": 1, "specified_location_tw_probability": 0.})
    bundle = donors.sample_bundle(20, np.random.default_rng(0))
    assert bundle["source_package_count"] == 2
    assert len(bundle["parcels"]) == 20
    assigned = sum(p["assigned_planned_service_s"] for p in bundle["parcels"])
    assert assigned == pytest.approx(service_curve(20) * 20. / service_curve(2))
    assert sum(p["source_planned_service_s"] for p in bundle["parcels"]) == 200.
    assert all(p["service_count_model_applied"] for p in bundle["parcels"])
    assert donors.sample_location_window(np.random.default_rng(0))["location_tw_status"] == "missing"


def test_calibration_requires_declared_source_and_supported_neighborhood(donors):
    with pytest.raises(ValueError, match="hash-bound"):
        donors.configure_calibration()
    donors.configure_calibration({"donor_pool": "declared_pool"})
    with pytest.raises(ValueError, match="support thresholds"):
        donors.sample_bundle(1, np.random.default_rng(0))
