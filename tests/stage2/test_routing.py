"""Objective-specific path costs retain the same native edge chain."""
import numpy as np
import pandas as pd
import pytest

from evrptw_scenarios.routing import Endpoint, EndpointOption, FrozenCostRouter, Segment


def road_router():
    # The shortest route a-b-c is slow. a-d-c is longer but faster. Parallel
    # a-b arcs must select distinct native keys under the two objectives.
    rows = [
        ("a", "b", "short", 100., 100., True),
        ("a", "b", "fast", 120., 10., True),
        ("b", "c", "0", 100., 100., True),
        ("a", "d", "0", 200., 5., True),
        ("d", "c", "0", 200., 5., True),
        ("c", "a", "0", 400., 40., True),
        ("b", "a", "0", 100., 10., True),
        ("d", "a", "0", 200., 5., True),
        ("a", "c", "private_shortcut", 1., .1, False),
        ("p", "a", "terminal", 20., 2., False),
        ("a", "p", "terminal", 20., 2., False),
    ]
    frame = pd.DataFrame(rows, columns=["edge_u", "edge_v", "edge_key", "length_m",
                                       "reference_travel_time_weekday_s", "in_public_graph"])
    frame["reference_speed_weekday_kph"] = frame.length_m / frame.reference_travel_time_weekday_s * 3.6
    frame["operating_mode"] = "U"
    return FrozenCostRouter(frame, connector_speed_kph=36.)


def node(router, name):
    zero = Segment("geometry_connector", None, 0., 0.)
    option = EndpointOption(router.node_index[name], 0., 0., ())
    return Endpoint(name, (option,), (option,), (), zero)


def test_four_costs_and_arbitrary_paths_retain_native_lineage():
    router = road_router()
    endpoints = tuple(node(router, name) for name in ["a", "b", "c"])
    cost = router.route(endpoints, specific_energy_kwh_per_km=.4,
                        audit_pairs=[(0, 1), (0, 2)], batch_size=2)
    assert cost.distance_shortest_path_distance[0, 2] == 200
    assert cost.distance_shortest_path_time[0, 2] == 200
    assert cost.time_shortest_path_distance[0, 2] == 400
    assert cost.time_shortest_path_time[0, 2] == 10
    assert cost.distance_m is cost.distance_shortest_path_distance
    assert cost.time_s is cost.distance_shortest_path_time
    assert cost.energy_kwh[0, 2] == pytest.approx(.08)
    assert cost.time_shortest_path_energy_kwh[0, 2] == pytest.approx(.16)
    assert cost.audit_paths["0,1"]["segments"][0]["source_edge"] == ["a", "b", "short"]
    assert cost.fastest_audit_paths["0,1"]["segments"][0]["source_edge"] == ["a", "b", "fast"]
    for policy, distance_name, time_name in [
        ("shortest_distance", "distance_shortest_path_distance", "distance_shortest_path_time"),
        ("fastest_time", "time_shortest_path_distance", "time_shortest_path_time"),
    ]:
        for i in range(3):
            for j in range(3):
                record = router.path(endpoints, i, j, specific_energy_kwh_per_km=.4, policy=policy)
                assert record["distance_m"] == pytest.approx(getattr(cost, distance_name)[i, j])
                assert record["time_s"] == pytest.approx(getattr(cost, time_name)[i, j])
                assert sum(part["time_s"] for part in record["segments"]) == pytest.approx(record["time_s"])
    assert np.all(cost.distance_shortest_path_distance <= cost.time_shortest_path_distance + 1e-8)
    assert np.all(cost.time_shortest_path_time <= cost.distance_shortest_path_time + 1e-8)


def test_private_terminal_chain_is_access_only_and_not_public_shortcut():
    router = road_router()
    connector = Segment("geometry_connector", None, 7., .7)
    outbound = (connector, router._native(("p", "a", "terminal"), kind="terminal_native"))
    inbound = (router._native(("a", "p", "terminal"), kind="terminal_native"), connector)
    terminal = Endpoint("p", (router._option(router.node_index["a"], inbound),),
                        (router._option(router.node_index["a"], outbound),), (), connector)
    endpoints = (terminal, node(router, "c"))
    cost = router.route(endpoints, specific_energy_kwh_per_km=.4, audit_pairs=[(0, 1)])
    assert cost.distance_m[0, 1] == 227
    assert cost.time_s[0, 1] == pytest.approx(202.7)
    assert cost.time_shortest_path_distance[0, 1] == 427
    assert cost.time_shortest_path_time[0, 1] == pytest.approx(12.7)
    assert cost.audit_paths["0,1"]["segments"][1]["kind"] == "terminal_native"
    assert all(part["source_edge"] != ["a", "c", "private_shortcut"]
               for part in cost.audit_paths["0,1"]["segments"])


def test_access_options_to_same_gateway_are_chosen_for_each_objective():
    router = road_router()
    zero = Segment("geometry_connector", None, 0., 0.)
    gateway = router.node_index["a"]
    options = router._best_options([
        router._option(gateway, (Segment("geometry_connector", None, 10., 100.),)),
        router._option(gateway, (Segment("geometry_connector", None, 20., 1.),)),
    ])
    assert len(options) == 2
    source = Endpoint("source", (EndpointOption(gateway, 0., 0., ()),), options, (), zero)
    cost = router.route((source, node(router, "c")), specific_energy_kwh_per_km=.4)
    assert cost.distance_m[0, 1] == 210
    assert cost.time_s[0, 1] == 300
    assert cost.time_shortest_path_distance[0, 1] == 420
    assert cost.time_shortest_path_time[0, 1] == 11
    assert cost.source_option[0, 1] != cost.time_shortest_path_source_option[0, 1]


def test_same_native_edge_partial_paths_compare_both_policies():
    router = road_router()
    identity = ("a", "b", "short")
    def projection(name, fraction):
        connector = Segment("geometry_connector", None, 2., .2)
        inbound = router._option(router.node_index["a"], (router._native(identity, end=fraction), connector))
        outbound = router._option(router.node_index["b"], (connector, router._native(identity, start=fraction)))
        return Endpoint(name, (inbound,), (outbound,), ((identity, fraction),), connector)
    endpoints = (projection("left", .2), projection("right", .7))
    cost = router.route(endpoints, specific_energy_kwh_per_km=.4, audit_pairs=[(0, 1)])
    assert cost.distance_m[0, 1] == pytest.approx(54)
    assert cost.time_s[0, 1] == pytest.approx(50.4)
    assert cost.path_mode[0, 1] == cost.time_shortest_path_mode[0, 1] == 1
    direct = cost.audit_paths["0,1"]["segments"][1]
    assert direct["fraction_start"] == .2 and direct["fraction_end"] == .7
