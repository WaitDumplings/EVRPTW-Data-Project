from copy import deepcopy
from datetime import date
import json

import pandas as pd

from evrptw_scenarios.config import DEFAULT, resolve_config
from evrptw_scenarios.demand import build_demand, schedule_day
from evrptw_scenarios.views import build_views


class SyntheticDonors:
    manifest = {"source_sha256": {"package_data": "synthetic"}, "partition": "test"}

    def sample_bundle(self, k, rng):
        return {"parcels": [{"source_parcel_ref": str(i), "source_event_ref": str(i),
                 "source_station_day_id": "station-day", "source_route_ref": "route", "source_date": "2018-07-19",
                 "depth_cm": 2., "height_cm": 3., "width_cm": 4., "volume_cm3": 24.,
                 "source_planned_service_s": 10., "source_tw_start_offset_s": None,
                 "source_tw_end_offset_s": None, "source_tw_status": "missing"} for i in range(k)],
                "source_donor_id": "bundle", "exact_source_count": False,
                "source_package_count": 1, "count_outside_observed_range": k > 1}


def config(tmp_path):
    value = deepcopy(DEFAULT)
    value["world"]["demand_model"] = "legacy_gamma_poisson"
    value["input"] = {"cle": str(tmp_path), "communities": str(tmp_path), "donor_artifact": str(tmp_path)}
    value["output"]["path"] = str(tmp_path / "run")
    return resolve_config(value)


def test_hard_promises_override_opt_in_and_closed_days():
    first = date(2026, 11, 2)  # Monday
    # Friday choice cannot hold a Monday parcel past its Wednesday commitment.
    assert schedule_day(0, first_date=first, promise_day=2, weekdays=range(7),
                        mode="address_day", eligible=True, preferred_weekday=4) == 0
    assert schedule_day(0, first_date=first, promise_day=6, weekdays=range(7),
                        mode="address_day", eligible=True, preferred_weekday=4) == 4
    assert schedule_day(5, first_date=first, promise_day=5, weekdays=range(5),
                        mode="daily", eligible=False, preferred_weekday=0) is None


def test_parcel_population_independent_of_view_sizes_and_vehicle(tmp_path):
    cfg = config(tmp_path)
    cfg["operations"]["days"] = 3
    cfg["world"]["rho"] = 3
    catalog = tmp_path / "catalog.parquet"
    pd.DataFrame({"location_id": ["a", "b", "c"], "community_id": ["x", "x", None], "G": [4., 1., 2.]}).to_parquet(catalog)
    world = {"catalog_path": str(catalog), "arrival_field_id": "stable-field", "world_id": "w"}
    first = build_demand(world, cfg, SyntheticDonors(), tmp_path / "first")
    cfg["views"]["sizes"] = [1]
    cfg["physical"]["vehicle"]["cargo_capacity_cm3"] = 1
    second = build_demand(world, cfg, SyntheticDonors(), tmp_path / "second")
    a, b = pd.read_parquet(first["parcels_path"]), pd.read_parquet(second["parcels_path"])
    pd.testing.assert_frame_equal(a, b)
    assert a.parcel_id.is_unique
    days = pd.read_parquet(first["days_path"])
    assert days.backlog.eq(0).all()
    assert a.volume_cm3.sum() == pd.read_parquet(first["sites_path"]).demand_cm3.sum()
    assert not a.commitment_failure.any()
    # Common arrival field keeps parcel identities/marks across alternative
    # depot worlds, while workload/day/view artifacts cannot collide.
    alternate = build_demand({**world, "world_id": "other-depot-world"}, cfg, SyntheticDonors(), tmp_path / "alternate")
    pd.testing.assert_frame_equal(a, pd.read_parquet(alternate["parcels_path"]))
    assert first["process_id"] != alternate["process_id"]
    assert set(pd.read_parquet(first["days_path"]).operational_day_id).isdisjoint(
        pd.read_parquet(alternate["days_path"]).operational_day_id)
    # Equal arrival realizations in two operating worlds must not alias saved
    # natural-day or fixed-N view identities, including their parent bindings.
    first_views = build_views(first, cfg, tmp_path / "first-views")
    alternate_views = build_views(alternate, cfg, tmp_path / "alternate-views")
    first_table = pd.read_parquet(first_views["views_path"])
    alternate_table = pd.read_parquet(alternate_views["views_path"])
    assert set(first_table.view_id).isdisjoint(alternate_table.view_id)
    assert set(first_table.parent_operational_day_id).isdisjoint(alternate_table.parent_operational_day_id)
    assert first_table.selected_n.to_list() == alternate_table.selected_n.to_list()


def test_empty_population_and_unsupported_size_are_retained(tmp_path):
    cfg = config(tmp_path)
    cfg["operations"]["days"] = 1
    cfg["world"]["rho"] = 0
    path = tmp_path / "catalog.parquet"
    pd.DataFrame({"location_id": ["a"], "community_id": ["c"], "G": [1.]}).to_parquet(path)
    info = build_demand({"catalog_path": str(path), "arrival_field_id": "f", "world_id": "w"}, cfg, SyntheticDonors(), tmp_path / "d")
    views = build_views(info, cfg, tmp_path / "v")
    table = pd.read_parquet(views["views_path"])
    assert info["parcel_count"] == 0
    assert table.loc[table.kind.eq("benchmark_view"), "supported_size"].eq(False).all()
    assert json.loads((tmp_path / "d/demand.json").read_text())["initial_backlog"] == 0


class CalibratedSyntheticDonors(SyntheticDonors):
    def configure_calibration(self, settings):
        self.calibration_summary = {"donor_pool": "synthetic_test"}

    def sample_bundle(self, k, generator):
        bundle = super().sample_bundle(k, generator)
        for mark in bundle["parcels"]:
            mark["assigned_planned_service_s"] = 100. / k
            mark["service_count_model_applied"] = True
        return bundle

    def sample_location_window(self, generator):
        return {"location_tw_status": "specified", "location_tw_start_offset_s": 32400.,
                "location_tw_end_offset_s": 61200., "location_tw_source_donor_id": "window-donor"}


def test_calibrated_single_location_window_service_and_capacity_failure(tmp_path):
    cfg = config(tmp_path)
    cfg["world"].update(demand_model="binomial_geometric", p_unit=1., mean_parcels_per_active_unit=1.)
    cfg["operations"]["days"] = 1
    path = tmp_path / "catalog.parquet"
    pd.DataFrame({"location_id": ["house", "apt"], "community_id": ["c", "c"],
                  "G": [1., 10.], "modeled_residential_units": [1, 10]}).to_parquet(path)
    world = {"catalog_path": str(path), "arrival_field_id": "calibrated", "world_id": "w"}
    good = build_demand(world, cfg, CalibratedSyntheticDonors(), tmp_path / "good")
    p = pd.read_parquet(good["parcels_path"])
    sites = pd.read_parquet(good["sites_path"]).set_index("location_id")
    assert len(p) == 11
    assert sites.loc["apt", "parcel_count"] == 10
    assert sites.loc["apt", "planned_service_time_s"] == 100
    assert sites.loc["house", "planned_service_time_s"] == 100
    assert p.source_tw_status.eq("missing").all()  # raw source not overwritten
    assert p.location_tw_status.eq("specified").all()
    assert sites.tw_start_s.eq(32400).all() and sites.tw_end_s.eq(61200).all()
    # The physical scenario is a separate view of the same workload.
    cfg["physical"]["vehicle"]["cargo_capacity_cm3"] = 1.
    variant = build_demand(world, cfg, CalibratedSyntheticDonors(), tmp_path / "variant")
    pd.testing.assert_frame_equal(p, pd.read_parquet(variant["parcels_path"]))
    cfg["workload"].update(cargo_capacity_cm3=1., maximum_full_proposals=3)
    bad = build_demand(world, cfg, CalibratedSyntheticDonors(), tmp_path / "bad")
    assert bad["parcel_count"] == 0
    failed = pd.read_parquet(bad["failed_proposal_path"])
    assert len(failed) == 6 and failed.exhausted.sum() == 2
    unsupported = pd.read_parquet(bad["sites_path"])
    assert not unsupported.generation_supported.any()
    assert unsupported.demand_cm3.isna().all()
    assert not unsupported.supported_one_visit_representation.any()


def test_city_unit_medians_use_whole_city_and_preserve_raw_units():
    import numpy as np
    from evrptw_scenarios.world import _propensities, WORLD_DEFAULTS
    frame = pd.DataFrame({"residential_type": ["house", "house", "large_apt", "large_apt", "large_apt",
         "residential_type_ambiguous", "residential_type_ambiguous"],
         "residential_units": [1., 1., 30., 31., np.nan, 1000., np.nan]})
    raw = frame.residential_units.copy()
    out = _propensities(frame, WORLD_DEFAULTS, "test")
    pd.testing.assert_series_equal(out.residential_units, raw)
    assert out.modeled_residential_units.tolist() == [1, 1, 30, 31, 31, 1000, 30]
    assert out.unit_imputation_policy.iloc[-1] == "city_all_residential_median_unknown_type"
    assert out.expected_daily_activation_probability.iloc[3] > out.expected_daily_activation_probability.iloc[0]


def test_active_unit_parcel_law_has_expected_mean_and_no_two_parcel_cap():
    import numpy as np
    from evrptw_scenarios.demand import active_location_count
    generator = np.random.default_rng(42)
    assert active_location_count(8, 1., 1., generator) == (8, 8)
    assert active_location_count(8, 1., 1., generator, conditioned_active=True) == (8, 8)
    values = np.array([active_location_count(1, 1., 1.4, generator)[1] for _ in range(20000)])
    assert abs(values.mean() - 1.4) < .03
    assert values.max() > 2
