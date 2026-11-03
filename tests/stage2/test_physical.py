from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from evrptw_scenarios.physical import _configuration, select_chargers


def road_example():
    # Directed road lengths: a lies on the real depot/customer route; b is
    # geographically close to the depot but costs a detour in either direction.
    ids = ["depot", "customer", "station-a", "station-b"]
    graph = np.array([[0, 10, 5, 1], [20, 0, 3, 1], [17, 5, 0, 0], [20, 20, 0, 0.]])
    distance = dijkstra(csr_matrix(graph), directed=True)
    costs = SimpleNamespace(location_ids=ids, distance_shortest_path_distance=distance,
                            # Deliberately wrong alternative: selection must not
                            # take distances from fastest-time paths.
                            distance_m=np.ones((4, 4)))
    chargers = pd.DataFrame({"location_id": ["station-b", "station-a"],
                             "longitude": [0., 100.], "latitude": [0., 100.],
                             "effective_power_kw": [90., 9.9]})
    return chargers, costs


def test_road_detour_uses_directed_shortest_distance_not_geometry_or_power():
    chargers, costs = road_example()
    original = chargers.copy(deep=True)
    chosen, result = select_chargers(chargers, ["customer"], costs, budget=1, depot_id="depot")
    assert chosen.location_id.tolist() == ["station-a"]
    assert result["supported_budget"]
    assert result["road_distance_source"] == "distance_shortest_path_distance"
    assert result["selection_scores"][0]["individual_mean_detour_m"] == 0
    assert chosen.effective_power_kw.iat[0] == 9.9  # no second derating/capping
    assert not result["capacity_or_energy_route_feasibility_claimed"]
    pd.testing.assert_frame_equal(chargers, original)


def test_selection_is_stable_and_keeps_budget_shortage_explicit():
    chargers, costs = road_example()
    chosen, result = select_chargers(chargers, ["customer"], costs, budget=3, depot_id="depot")
    shuffled, repeated = select_chargers(chargers.iloc[::-1], ["customer"], costs,
                                          budget=3, depot_id="depot")
    assert chosen.location_id.tolist() == shuffled.location_id.tolist() == ["station-a", "station-b"]
    assert result == repeated
    assert result["status"] == "unsupported_charging_budget"
    assert not result["supported_budget"]
    assert result["candidate_count"] == result["selected_count"] == 2
    assert result["requested_budget"] == 3


def test_unreachable_station_is_diagnosed_without_replacement():
    chargers, costs = road_example()
    costs.distance_shortest_path_distance[3, :3] = np.inf
    costs.distance_shortest_path_distance[:3, 3] = np.inf
    chosen, result = select_chargers(chargers, ["customer"], costs, budget=2, depot_id="depot")
    assert chosen.location_id.tolist() == ["station-a"]
    assert result["candidate_count"] == 2
    assert result["usable_candidate_count"] == 1
    assert not result["supported_budget"]


def test_greedy_additions_cover_distinct_road_legs():
    ids = ["depot", "c1", "c2", "a1", "a2", "b"]
    # Two depot spokes. a1/a2 duplicate the first spoke, whereas b serves the
    # second. Greedy set detour selects b before a duplicate near a1.
    graph = np.zeros((6, 6))
    for i, j, length in [(0, 3, 5), (3, 1, 5), (0, 4, 5), (4, 1, 5),
                         (0, 5, 5), (5, 2, 5)]:
        graph[i, j] = graph[j, i] = length
    matrix = dijkstra(csr_matrix(graph), directed=True)
    costs = SimpleNamespace(location_ids=ids, distance_shortest_path_distance=matrix)
    chargers = pd.DataFrame({"location_id": ["a2", "b", "a1"]})
    chosen, result = select_chargers(chargers, ["c1", "c2"], costs, budget=2, depot_id="depot")
    assert set(chosen.location_id) == {"a1", "b"}
    assert result["selection_scores"][1]["selected_set_mean_best_detour_m"] == 0


def test_no_distance_policy_fallback_and_no_silent_missing_endpoints():
    chargers, costs = road_example()
    del costs.distance_shortest_path_distance
    with pytest.raises(ValueError, match="shortest-distance"):
        select_chargers(chargers, ["customer"], costs, budget=1, depot_id="depot")
    chargers, costs = road_example()
    with pytest.raises(ValueError, match="omit"):
        select_chargers(chargers, ["not-in-roads"], costs, budget=1, depot_id="depot")


def test_legitimate_terminal_gate_bridge_retains_negative_raw_detour():
    # The public depot/customer trip cannot use the station's private terminal
    # as a through-shortcut. A genuine station visit may enter/leave its gates.
    costs = SimpleNamespace(location_ids=["depot", "customer", "station"],
        distance_shortest_path_distance=np.array([[0, 1000, 100], [1000, 0, 100], [100, 100, 0.]]))
    chargers = pd.DataFrame({"location_id": ["station"]})
    chosen, result = select_chargers(chargers, ["customer"], costs, budget=1, depot_id="depot")
    assert chosen.location_id.tolist() == ["station"]
    assert result["negative_raw_detour_count"] == 2
    assert result["minimum_raw_detour_m"] == -800
    assert result["selection_scores"][0]["individual_mean_detour_m"] == 0


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_budget_rejected(budget):
    chargers, costs = road_example()
    with pytest.raises(ValueError, match="budget"):
        select_chargers(chargers, ["customer"], costs, budget=budget, depot_id="depot")


def test_fixed_reference_physical_assumptions_are_explicit():
    parameters = _configuration({})
    assert parameters["vehicle"]["initial_soc"] == 1
    assert parameters["vehicle"]["fleet_size"] is None
    assert parameters["charging"]["external_cs_budget"] == 20
    assert parameters["charging"]["supply_policy"] == "road_detour_budget"
    assert parameters["charging"]["charge_policy"] == "full"
    assert parameters["charging"]["charge_curve"] == "linear"
    with pytest.raises(ValueError, match="charge_policy"):
        _configuration({"charging": {"charge_policy": "partial"}})
    assert _configuration({"charging": {"supply_policy": "all_compatible_catchment"}})
