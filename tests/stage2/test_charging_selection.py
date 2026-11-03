from types import SimpleNamespace

import numpy as np
import pandas as pd

from evrptw_scenarios.charging_selection import select_joint_chargers
from evrptw_scenarios.config import joint_station_budget, resolve_config


def test_joint_connector_prefers_compromise_and_protects_nearest_station():
    ids = ["depot", "c1", "c2", "near", "compromise", "fast", "unreachable"]
    distance = np.full((len(ids), len(ids)), 20_000.)
    np.fill_diagonal(distance, 0.)
    for a, b, km in [(1, 2, 10), (1, 3, 2), (2, 3, 8), (1, 4, 4),
                     (2, 4, 8), (1, 5, 8), (2, 5, 8)]:
        distance[a, b] = distance[b, a] = km * 1000
    distance[6, :6] = distance[:6, 6] = np.inf
    costs = SimpleNamespace(location_ids=ids, distance_shortest_path_distance=distance,
                            distance_shortest_path_time=distance * .06)
    chargers = pd.DataFrame({"location_id": ids[3:], "effective_power_kw": [6., 35., 90., 90.]})
    vehicle = {"battery_capacity_kwh": 100., "reserve_soc": .1, "specific_energy_kwh_per_km": .389}

    def choose(pool, customers, communities, budget=2):
        return select_joint_chargers(pool, customers, communities, costs, budget=budget,
                                     depot_id="depot", vehicle=vehicle, reference_departure_soc=.8)

    selected, audit = choose(chargers, ["c1", "c2"], ["481130001001", "481130002001"])
    assert selected.location_id.tolist() == ["compromise", "near"]
    assert audit["phase_counts"]["between_joint"] == 1
    assert audit["mean_nearest_depot_or_station_km"] == 5.
    _, shuffled = choose(chargers.iloc[::-1], ["c2", "c1"], [
        "dallas:tiger2025:bg:481130002001", "dallas:tiger2025:bg:481130001001",
    ])
    assert audit == shuffled
    _, short = choose(chargers, ["c1", "c2"], ["481130001001", "481130002001"], budget=4)
    assert short["usable_candidate_count"] == short["selected_count"] == 3
    assert not short["supported_budget"]


def test_territory_configuration_preserves_joint_size_budgets_and_scalar_fallback():
    config = resolve_config({
        "schema": "evrptw_territory_first_config_v1",
        "input": {"cle": "cle", "communities": "communities", "donor_artifact": "donor"},
        "physical": {"charging": {"supply_policy": "tract_joint_budget", "external_cs_budget": 2,
                                  "external_cs_budgets_by_size": {"500": 29}}},
    })
    charging = config["physical"]["charging"]
    assert charging["reference_departure_soc"] == .8
    assert [joint_station_budget(charging, n) for n in (100, 500, 1000, 2000, 50)] == [20, 29, 40, 60, 2]
