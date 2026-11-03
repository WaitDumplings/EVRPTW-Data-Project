from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evrptw_scenarios.calibration import service_curve
from evrptw_scenarios.territory_config import DEFAULT, resolve_config
from evrptw_scenarios.territory_generation import activate_territory, generate_selected_orders


def context():
    n = 12
    frame = pd.DataFrame({"location_id": [f"loc-{i:02}" for i in range(n)],
        "community_id": [str(i // 3) for i in range(n)],
        "cell_id": [str(i // 3) for i in range(n)],
        "modeled_residential_units": [1, 20, 100, 200, 500, 40, 10, 50, 80, 300, 60, 25],
        "location_lon": -96 + np.arange(n) * .001, "location_lat": np.full(n, 32.),
        "depot_outward_distance_m": np.arange(n) * 100 + 1000.,
        "depot_inward_distance_m": np.arange(n) * 120 + 1000.})
    def costs(anchors):
        return np.array([np.abs(np.arange(n) - a) * .2 for a in anchors])
    return SimpleNamespace(catalog=frame, depot=pd.DataFrame({"location_lon": [-96.01], "location_lat": [32.]}),
        adjacency={"0": ["1"], "1": ["0", "2"], "2": ["1", "3"], "3": ["2"]},
        info={"source": {"cle_manifest_sha256": "cle", "community_sha256": "community"},
              "depot_id": "depot", "world_id": "world"}, anchor_costs=costs)


class Donors:
    def sample_bundle(self, count, generator):
        total = 60. * service_curve(count) / service_curve(1)
        return {"parcels": [{"source_parcel_ref": "source", "source_donor_id": "donor",
            "source_event_ref": "event", "source_station_day_id": "station-day", "source_route_ref": "route",
            "source_date": "2018-07-19", "depth_cm": 1., "height_cm": 2., "width_cm": 3.,
            "volume_cm3": 6., "source_planned_service_s": 60.,
            "assigned_planned_service_s": total / count, "source_tw_status": "missing",
            "source_tw_start_offset_s": None, "source_tw_end_offset_s": None,
            "service_count_model_applied": count != 1} for _ in range(count)],
            "source_donor_id": "donor", "source_package_count": 1, "exact_source_count": count == 1,
            "count_outside_observed_range": count > 77}

    def sample_location_window(self, generator):
        return {"location_tw_status": "missing", "location_tw_start_offset_s": None,
                "location_tw_end_offset_s": None, "location_tw_source_donor_id": None}


def test_new_config_is_clean_daily_and_parameters_remain_configurable(tmp_path):
    cfg = deepcopy(DEFAULT)
    cfg["input"] = {k: str(tmp_path) for k in cfg["input"]}
    cfg["activation"].update(tau=.7, opening_penalty=.3, max_centers=1)
    resolved = resolve_config(cfg)
    assert resolved["instances"]["sizes"] == [100]
    assert resolved["activation"]["tau"] == .7
    assert resolved["activation"]["max_centers"] == 1
    cfg["operations"]["promise_lead_days"] = 1
    with pytest.raises(ValueError, match="same-day"):
        resolve_config(cfg)


def test_size_is_independent_scenario_and_physical_changes_do_not_move_sites():
    ctx, cfg = context(), deepcopy(DEFAULT)
    first, ids, territory, selected = activate_territory(ctx, cfg, day_index=0, size=3)
    assert first["supported_size"] and first["selected_n"] == 3
    assert len(ids) == len(set(ids)) == len(selected) == 3
    assert set(ids) <= {int(row["location_id"][-2:]) for row in territory}
    cfg["instances"]["sizes"] = [3, 6]
    cfg["physical"]["vehicle"]["battery_capacity_kwh"] = 20
    cfg["workload"]["cargo_capacity_cm3"] = 1
    second, again, _, _ = activate_territory(ctx, cfg, day_index=0, size=3)
    assert first == second and ids == again
    alternate, _, _, _ = activate_territory(ctx, cfg, day_index=0, size=6)
    assert alternate["spatial_id"] != first["spatial_id"]


def test_selected_locations_keep_full_order_and_a_p_service_scale():
    ctx, cfg = context(), deepcopy(DEFAULT)
    selected = [2, 4, 9]
    parcels, sites, failures = generate_selected_orders(ctx.catalog, selected, cfg, Donors(),
                                                       scenario_id="test", day_index=0)
    assert not failures
    assert {x["location_id"] for x in sites} == set(ctx.catalog.location_id.iloc[selected])
    for site in sites:
        marks = [p for p in parcels if p["location_id"] == site["location_id"]]
        assert len(marks) == site["parcel_count"] >= site["active_units"] >= 1
        assert sum(p["volume_cm3"] for p in marks) == site["demand_cm3"]
        assert sum(p["assigned_planned_service_s"] for p in marks) == pytest.approx(site["planned_service_time_s"])
        count, active = len(marks), site["active_units"]
        mean = cfg["world"]["mean_parcels_per_active_unit"]
        central, a = service_curve(count), cfg["workload"]["service_a_s"]
        expected = 60. / service_curve(1) * (a + (central - a) * (.5 + .5 * (active / max(1., count / mean))**.5))
        assert site["planned_service_time_s"] == pytest.approx(expected)
        assert len({(p["location_tw_status"], p["location_tw_start_offset_s"], p["location_tw_end_offset_s"]) for p in marks}) == 1


def test_capacity_failure_does_not_replace_selected_locations():
    ctx, cfg = context(), deepcopy(DEFAULT)
    cfg["workload"].update(cargo_capacity_cm3=1., maximum_full_proposals=3)
    parcels, sites, failures = generate_selected_orders(ctx.catalog, [2, 4], cfg, Donors(),
                                                       scenario_id="failed", day_index=0)
    assert parcels == [] and len(sites) == 2 and len(failures) == 6
    assert {s["location_id"] for s in sites} == {"loc-02", "loc-04"}
    assert all(not s["generation_supported"] and s["capacity_conditioning_attempts"] == 3 for s in sites)
    assert sum(f["exhausted"] for f in failures) == 2


def test_too_small_world_is_reported_without_anchor_redraw():
    result, selected, territory, selections = activate_territory(context(), deepcopy(DEFAULT), day_index=0, size=20)
    assert result["status"] == "unsupported_candidate_count"
    assert selected == territory == selections == []
