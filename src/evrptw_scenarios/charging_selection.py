"""Task-conditioned charging supply on one canonical road cost bundle."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .config import JOINT_CHARGING_POLICY


TOL = 1e-9
PHASES = ("between_joint", "local_joint", "local_protection", "padding")


def _tract_legs(distance_m, customers, communities):
    by_tract = {}
    for customer, community in zip(customers, communities, strict=True):
        geoid = str(community).rsplit(":bg:", 1)[-1]
        if len(geoid) == 12 and geoid.isascii() and geoid.isdecimal():
            by_tract.setdefault(geoid[:11], []).append(customer)
    medoids = []
    for tract, members in sorted(by_tract.items()):
        within = distance_m[np.ix_(members, members)]
        medoids.append((tract, members[np.argmin((within + within.T).sum(axis=0))]))
    nodes = np.array([node for _, node in medoids], dtype=int)
    road = distance_m[np.ix_(nodes, nodes)]
    symmetric = (road + road.T) / 2
    weights = np.where(np.isfinite(symmetric), np.maximum(symmetric, TOL), 0.)
    tree = minimum_spanning_tree(csr_matrix(weights)).tocoo()
    source, target = nodes[tree.row], nodes[tree.col]
    return medoids, np.r_[source, target], np.r_[target, source]


def _joint_score(distance, minutes):
    """Score against the nearest and fastest feasible anchors for each reference."""
    scores = np.zeros_like(distance)
    for i, (d, t) in enumerate(zip(distance, minutes, strict=True)):
        feasible = np.flatnonzero(np.isfinite(d) & np.isfinite(t))
        if not len(feasible):
            continue
        d, t = d[feasible], t[feasible]
        nearest, fastest = d.min(), t.min()
        fastest_among_nearest = t[d <= nearest + TOL].min()
        nearest_among_fastest = d[t <= fastest + TOL].min()
        if nearest_among_fastest <= nearest + TOL or fastest_among_nearest <= fastest + TOL:
            scores[i, feasible] = (d <= nearest + TOL) & (t <= fastest + TOL)
        else:
            distance_regret = (d - nearest) / (nearest_among_fastest - nearest)
            time_regret = (t - fastest) / (fastest_among_nearest - fastest)
            scores[i, feasible] = 1 / (1 + np.maximum(distance_regret, time_regret))
    return scores


def _distance_gain(current, choices):
    return np.maximum(current[:, None] - choices, 0.).mean(axis=0)


def _winner(values, candidates, power, secondary=None):
    best = values[candidates].max()
    tied = candidates[np.abs(values[candidates] - best) <= TOL]
    if secondary is not None and len(tied) > 1:
        scores = secondary[tied]
        tied = tied[np.abs(scores - scores.max()) <= TOL]
    # Candidate indices are in canonical station ID order.
    chosen = int(tied[np.argmax(power[tied])])
    return chosen, float(values[chosen])


def _distance_witness(depot_distance, local_distance, power, target):
    nearest = depot_distance.copy()
    remaining = np.arange(len(power))
    witness = set()
    for _ in range(target):
        station, gain = _winner(_distance_gain(nearest, local_distance), remaining, power)
        if gain <= TOL:
            break
        witness.add(station)
        nearest = np.minimum(nearest, local_distance[:, station])
        remaining = remaining[remaining != station]
    return witness


def _choose(between_scores, local_scores, depot_distance, local_distance, power, target):
    pool = np.arange(len(power))
    witness = _distance_witness(depot_distance, local_distance, power, target)
    chosen, trace = set(), []
    between_best, local_best = np.zeros(len(between_scores)), np.zeros(len(local_scores))
    nearest = depot_distance.copy()
    public_first = np.full(len(nearest), np.inf)
    public_second = public_first.copy()

    def add(station, phase, gain):
        nonlocal nearest, public_first, public_second, local_best, between_best
        chosen.add(station)
        trace.append((station, phase, gain))
        nearest = np.minimum(nearest, local_distance[:, station])
        public_second = np.minimum(public_second, np.maximum(public_first, local_distance[:, station]))
        public_first = np.minimum(public_first, local_distance[:, station])
        local_best = np.maximum(local_best, local_scores[:, station])
        between_best = np.maximum(between_best, between_scores[:, station])

    for phase, scores in zip(PHASES[:2], [between_scores, local_scores], strict=True):
        while len(chosen) < target and len(scores):
            reserved = chosen | witness
            available = np.array([s for s in pool if s not in chosen
                                  and (s in witness or len(reserved) < target)], dtype=int)
            if not len(available):
                break
            best = between_best if phase == "between_joint" else local_best
            benefit = np.maximum(scores - best[:, None], 0.).mean(axis=0)
            station, gain = _winner(benefit, available, power, _distance_gain(nearest, local_distance))
            if gain <= TOL:
                break
            add(station, phase, gain)

    while witness - chosen:
        available = np.array(sorted(witness - chosen), dtype=int)
        station, gain = _winner(_distance_gain(nearest, local_distance), available, power)
        if gain <= TOL:
            break
        add(station, "local_protection", gain)

    proximity = -local_distance.min(axis=0)
    while len(chosen) < target:
        available = np.array([s for s in pool if s not in chosen], dtype=int)
        if chosen:
            second = np.minimum(public_second[:, None], np.maximum(public_first[:, None], local_distance))
            station, _ = _winner(-second.mean(axis=0), available, power, proximity)
        else:
            station, _ = _winner(proximity, available, power)
        add(station, "padding", None)
    return trace


def select_joint_chargers(chargers, customer_ids, communities, costs, *, budget,
                          depot_id, vehicle, reference_departure_soc):
    """Select a fixed external supply budget and retain the deterministic score trace.

    Route leg times follow the shortest-distance road paths. Reference SOC
    affects the station score, independent of the vehicle state in any route.
    """
    ids = list(map(str, costs.location_ids))
    index = {loc: i for i, loc in enumerate(ids)}
    task = sorted(zip(map(str, customer_ids), communities, strict=True))
    customers = np.array([index[loc] for loc, _ in task], dtype=int)
    depot = index[str(depot_id)]
    pool = chargers.sort_values("location_id", kind="stable").reset_index(drop=True)
    stations = np.array([index[loc] for loc in pool.location_id.astype(str)], dtype=int)
    # Keep the cost bundle in native units and materialize only needed slices.
    distance, time = costs.distance_shortest_path_distance, costs.distance_shortest_path_time
    medoids, origins, destinations = _tract_legs(distance, customers, [bg for _, bg in task])
    terminals = np.r_[depot, customers]
    outbound = np.ix_(terminals, stations)
    inbound = np.ix_(stations, terminals)
    usable = (np.isfinite(distance[outbound]).all(axis=0) & np.isfinite(distance[inbound]).all(axis=1) &
              np.isfinite(time[outbound]).all(axis=0) & np.isfinite(time[inbound]).all(axis=1))
    usable_pool = pool.loc[usable].reset_index(drop=True)
    stations = stations[usable]
    trace, nearest_mean, detour_mean = [], None, None
    target = min(budget, len(stations))
    if target and len(customers):
        power = usable_pool.effective_power_kw.to_numpy(float)
        rate = vehicle["specific_energy_kwh_per_km"]
        battery = vehicle["battery_capacity_kwh"]
        reserve = battery * vehicle["reserve_soc"]
        departure = battery * reference_departure_soc
        local_distance = distance[np.ix_(customers, stations)] / 1000
        local_energy = local_distance * rate
        local_charge = 60 * (battery - departure + local_energy) / power
        local_time = np.where(local_energy <= departure - reserve + TOL,
                              time[np.ix_(customers, stations)] / 60 + local_charge, np.inf)
        approach = distance[np.ix_(origins, stations)] / 1000
        onward = distance[np.ix_(stations, destinations)].T / 1000
        detour = np.maximum(0., approach + onward - distance[origins, destinations, None] / 1000)
        between_charge = 60 * (battery - departure + approach * rate) / power
        can_charge = ((approach * rate <= departure - reserve + TOL) &
                      (onward * rate <= battery - reserve + TOL))
        between_time = np.where(can_charge,
            (time[np.ix_(origins, stations)] + time[np.ix_(stations, destinations)].T) / 60 + between_charge,
            np.inf)
        depot_distance = distance[customers, depot] / 1000
        depot_distance = np.where(np.isfinite(depot_distance), depot_distance, local_distance.max(axis=1))
        trace = _choose(_joint_score(detour, between_time), _joint_score(local_distance, local_time),
                        depot_distance, local_distance, power, target)
        selected = [station for station, _, _ in trace]
        nearest_mean = float(np.minimum(distance[customers, depot] / 1000,
                                        local_distance[:, selected].min(axis=1)).mean())
        if len(origins):
            detour_mean = float(detour[:, selected].min(axis=1).mean())
    elif target:
        trace = [(station, "padding", None) for station in range(target)]
    chosen = usable_pool.iloc[[station for station, _, _ in trace]].copy().reset_index(drop=True)
    chosen["charging_selection_rank"] = np.arange(1, len(chosen) + 1)
    selected_ids = chosen.location_id.astype(str).tolist()
    report = {
        "schema": "evrptw_tract_joint_charger_selection_v1",
        "policy": JOINT_CHARGING_POLICY,
        "status": "selected" if len(chosen) == budget else "unsupported_charging_budget",
        "supported_budget": len(chosen) == budget,
        "requested_budget": budget, "selected_count": len(chosen), "candidate_count": len(pool),
        "usable_candidate_count": len(usable_pool), "selected_location_ids": selected_ids,
        "reference_departure_soc": reference_departure_soc,
        "reference_tract_medoids": {tract: ids[node] for tract, node in medoids},
        "directed_reference_legs": [[ids[a], ids[b]] for a, b in zip(origins, destinations, strict=True)],
        "road_distance_source": "distance_shortest_path_distance",
        "road_time_source": "distance_shortest_path_time",
        "mean_nearest_depot_or_station_km": nearest_mean,
        "mean_best_selected_directed_detour_km": detour_mean,
        "phase_counts": {phase: sum(p == phase for _, p, _ in trace) for phase in PHASES},
        "trace": [{"location_id": loc, "rank": i, "phase": phase, "marginal": gain}
                  for i, (loc, (_, phase, gain)) in enumerate(zip(selected_ids, trace, strict=True), 1)],
        "capacity_or_energy_route_feasibility_claimed": False,
        "source_supply_dropped": False,
    }
    return chosen, report
