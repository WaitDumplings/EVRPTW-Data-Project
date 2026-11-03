"""Static-cell territory growth, independent of road and customer activation adapters.

The adapter supplies immutable cell membership, a road-supported graph and K-by-n
two-way road costs. This module neither computes routes nor activates households.
All clock rates use complete cells, including the cell eventually cut by the
budget. Exponential draws for source 0 can be shared across different K values.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ContractedGraph:
    adjacency: dict[str, tuple[str, ...]]
    witness_paths: dict[tuple[str, str], tuple[str, ...]]


def contract_empty_units(
    adjacency: Mapping[str, Sequence[str]], occupied_cells: Sequence[str],
) -> ContractedGraph:
    """Contract paths whose internal nodes have no potential locations.

    Direction is preserved. A path never skips another occupied cell. Witnesses
    are deterministic unit-graph paths, not independently inferred OSM paths:
    the adapter remains responsible for actual road evidence and road costs.
    """
    graph = {str(g): tuple(sorted({str(v) for v in neighbors}))
             for g, neighbors in adjacency.items()}
    occupied = {str(g) for g in occupied_cells}
    nodes = set(graph) | occupied | {v for neighbors in graph.values() for v in neighbors}
    for node in nodes:
        graph.setdefault(node, ())
    result = {g: set() for g in occupied}
    witnesses = {}
    for source in sorted(occupied):
        paths = {source: (source,)}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in paths:
                    continue
                path = paths[node] + (neighbor,)
                paths[neighbor] = path
                if neighbor in occupied:
                    if neighbor != source:
                        result[source].add(neighbor)
                        witnesses[source, neighbor] = path
                else:
                    queue.append(neighbor)
    return ContractedGraph(
        {g: tuple(sorted(result[g])) for g in sorted(occupied)}, witnesses,
    )


def weighted_reference_length(
    road_costs_first: Sequence[float], probabilities: Sequence[float], target_count: int,
) -> float:
    """R_min(N,W)(a1), with ties included as complete equal-cost shells.

    Costs and the returned length have the same unit. Infinite costs are not
    silently discarded: an unattainable positive-weight quantile returns inf.
    A truly co-located support can return zero; a growth problem requires a
    separately declared positive scale rather than silently replacing zero.
    """
    costs = np.asarray(road_costs_first, dtype=float)
    pi = np.asarray(probabilities, dtype=float)
    if costs.ndim != 1 or costs.shape != pi.shape:
        raise ValueError("Reference costs and probabilities must be matching vectors")
    if target_count < 1 or not np.all(np.isfinite(pi) & (pi >= 0) & (pi <= 1)):
        raise ValueError("Positive target and probabilities in [0,1] are required")
    if np.any(np.isnan(costs) | (costs < 0)):
        raise ValueError("Road costs must be nonnegative, with inf allowed")
    positive = np.flatnonzero(pi > 0)
    if not len(positive):
        raise ValueError("Reference support has zero total probability mass")
    order = positive[np.argsort(costs[positive], kind="stable")]
    cumulative = np.cumsum(pi[order])
    total = float(cumulative[-1])
    threshold = min(float(target_count), total)
    position = min(int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1)
    return float(costs[order[position]])


@dataclass(frozen=True)
class TerritoryProblem:
    location_ids: Sequence[str]
    cell_ids: Sequence[str]
    adjacency: Mapping[str, Sequence[str]]
    probabilities: Sequence[float]
    road_costs: np.ndarray
    anchor_cells: Sequence[str]
    target_count: int
    reference_length: float
    tau: float
    log_probabilities: Sequence[float] | None = None
    cell_members: dict[str, np.ndarray] = field(init=False, repr=False)
    graph: ContractedGraph = field(init=False, repr=False)
    log_weights: np.ndarray = field(init=False, repr=False)
    log_rates: dict[tuple[int, str], float] = field(init=False, repr=False)

    def __post_init__(self):
        ids = np.asarray(self.location_ids, dtype=str).copy()
        cells = np.asarray(self.cell_ids, dtype=str).copy()
        pi = np.asarray(self.probabilities, dtype=float).copy()
        costs = np.asarray(self.road_costs, dtype=float).copy()
        anchors = tuple(str(g) for g in self.anchor_cells)
        n = len(ids)
        if ids.ndim != 1 or cells.shape != (n,) or pi.shape != (n,):
            raise ValueError("Location IDs, cell IDs and probabilities must be matching vectors")
        if len(set(ids)) != n or costs.shape != (len(anchors), n) or not anchors:
            raise ValueError("Unique locations and K-by-n road costs are required")
        if not isinstance(self.target_count, (int, np.integer)) or self.target_count < 1:
            raise ValueError("target_count must be a positive integer")
        if not np.all(np.isfinite(pi) & (pi >= 0) & (pi <= 1)):
            raise ValueError("Probabilities must lie in [0,1]")
        if np.any(np.isnan(costs) | (costs < 0)):
            raise ValueError("Road costs must be nonnegative, with inf allowed")
        if not (np.isfinite(self.reference_length) and self.reference_length > 0
                and np.isfinite(self.tau) and self.tau > 0):
            raise ValueError("reference_length and tau must be finite and positive")
        if self.log_probabilities is None:
            with np.errstate(divide="ignore"):
                log_pi = np.log(pi)
        else:
            log_pi = np.asarray(self.log_probabilities, dtype=float).copy()
            if log_pi.shape != (n,) or np.any(np.isnan(log_pi) | (log_pi > 0)):
                raise ValueError("log_probabilities must be matching nonpositive log values")
        order = np.argsort(cells, kind="stable")
        ordered_cells = cells[order]
        starts = np.concatenate(([0], np.flatnonzero(ordered_cells[1:] != ordered_cells[:-1]) + 1))
        ends = np.concatenate((starts[1:], [n]))
        members = {str(ordered_cells[start]): order[start:end]
                   for start, end in zip(starts, ends) if start < end}
        if any(g not in members for g in anchors):
            raise ValueError("Every anchor cell must contain potential locations")
        graph = contract_empty_units(self.adjacency, tuple(members))
        with np.errstate(over="ignore"):
            log_weights = log_pi[None, :] - costs / (self.reference_length * self.tau)
        rates = {(k, g): float(logsumexp(log_weights[k, indices]))
                 for k in range(len(anchors)) for g, indices in members.items()}
        for name, value in [("location_ids", ids), ("cell_ids", cells),
                            ("probabilities", pi), ("road_costs", costs),
                            ("anchor_cells", anchors), ("log_probabilities", log_pi),
                            ("cell_members", members), ("graph", graph),
                            ("log_weights", log_weights), ("log_rates", rates)]:
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GrowthEvent:
    log_time: float
    owner: int
    cell_id: str
    full_cell_count: int
    full_cell_mass: float
    selected_count: int
    selected_mass: float
    cumulative_mass: float
    budget_trigger: bool


@dataclass(frozen=True)
class GrowthResult:
    selected_indices: np.ndarray
    owners: np.ndarray
    expected_mass: float
    supported: bool
    stop_reason: str
    events: tuple[GrowthEvent, ...]
    claimed_cells: tuple[str, ...]
    winning_anchor_count: int
    contributing_anchor_count: int

    @property
    def B(self) -> float:
        return self.expected_mass


def _ordered_members(problem: TerritoryProblem, owner: int, cell: str) -> np.ndarray:
    indices = problem.cell_members[cell]
    order = np.lexsort((problem.location_ids[indices], problem.road_costs[owner, indices]))
    return indices[order]


def _take_cell(problem, owner, cell, selected, owners, mass, log_time):
    indices = _ordered_members(problem, owner, cell)
    weights = problem.probabilities[indices]
    cell_mass = float(np.sum(weights))
    trigger = mass + cell_mass >= problem.target_count
    if trigger:
        prefix = np.concatenate(([0.0], np.cumsum(weights)))
        first_q = max(0, problem.target_count - len(selected))
        possible_q = np.arange(first_q, len(indices) + 1)
        deviations = np.abs(mass + prefix[possible_q] - problem.target_count)
        # Resolve mathematical ties consistently despite cumulative-sum roundoff.
        tolerance = 8 * np.finfo(float).eps * max(1, problem.target_count, abs(mass))
        q = int(possible_q[np.flatnonzero(deviations <= deviations.min() + tolerance)[0]])
        indices = indices[:q]
        added = float(prefix[q])
    else:
        added = cell_mass
    selected.extend(indices.tolist())
    owners.extend([owner] * len(indices))
    mass += added
    event = GrowthEvent(log_time, owner, cell, len(weights), cell_mass,
                        len(indices), added, mass, trigger)
    return mass, event


def _finish(problem, selected, owners, mass, events, claimed, triggered):
    supported = len(selected) >= problem.target_count
    reason = ("budget_trigger" if triggered else
              "frontier_exhausted" if supported else "insufficient_candidate_count")
    return GrowthResult(
        np.asarray(selected, dtype=np.int64), np.asarray(owners, dtype=np.int64),
        float(mass), supported, reason, tuple(events), tuple(claimed),
        len({event.owner for event in events}), len(set(owners)),
    )


def grow_clock(
    problem: TerritoryProblem, *, rng: np.random.Generator | None = None,
    exponentials: np.ndarray | None = None,
) -> GrowthResult:
    """Persistent independent (source, cell) races, in logarithmic time.

    E[k,i] are independent Exp(1). A pair's delay is min_i E[k,i]/w[k,i],
    measured from its first eligibility. Losing deadlines never restart;
    duplicate neighbor exposures do not add clocks. Source cells enter the
    frontier at time zero but contribute no mass until their event wins.
    """
    if exponentials is None:
        rng = np.random.default_rng() if rng is None else rng
        exponentials = rng.exponential(size=problem.road_costs.shape)
    draws = np.asarray(exponentials, dtype=float)
    if draws.shape != problem.road_costs.shape or not np.all(np.isfinite(draws) & (draws > 0)):
        raise ValueError("exponentials must be a positive finite K-by-n array")
    log_e = np.log(draws)
    seen = set()
    claimed_set = set()
    claimed = []
    queue = []
    selected, owners, events = [], [], []
    mass = 0.0

    def expose(k, g, now):
        pair = (k, g)
        if g in claimed_set or pair in seen:
            return
        seen.add(pair)
        if not np.isfinite(problem.log_rates[pair]):
            return
        indices = problem.cell_members[g]
        delay = float(np.min(log_e[k, indices] - problem.log_weights[k, indices]))
        deadline = float(np.logaddexp(now, delay))
        heapq.heappush(queue, (deadline, k, g))

    for k, g in enumerate(problem.anchor_cells):
        expose(k, g, -np.inf)
    triggered = False
    while queue:
        now, k, g = heapq.heappop(queue)
        if g in claimed_set:
            continue
        claimed_set.add(g)
        claimed.append(g)
        mass, event = _take_cell(problem, k, g, selected, owners, mass, now)
        events.append(event)
        if event.budget_trigger:
            triggered = True
            break
        for neighbor in problem.graph.adjacency[g]:
            expose(k, neighbor, now)
    return _finish(problem, selected, owners, mass, events, claimed, triggered)


def grow_direct(
    problem: TerritoryProblem, *, rng: np.random.Generator | None = None,
) -> GrowthResult:
    """Independent categorical-H reference for small-graph distribution checks."""
    rng = np.random.default_rng() if rng is None else rng
    frontier = {(k, g) for k, g in enumerate(problem.anchor_cells)
                if np.isfinite(problem.log_rates[k, g])}
    seen = set(frontier)
    selected, owners, events, claimed = [], [], [], []
    claimed_set = set()
    mass, now = 0.0, -np.inf
    triggered = False
    while frontier:
        pairs = sorted(frontier)
        rates = np.asarray([problem.log_rates[pair] for pair in pairs])
        total = float(logsumexp(rates))
        probabilities = np.exp(rates - total)
        position = int(rng.choice(len(pairs), p=probabilities / probabilities.sum()))
        k, g = pairs[position]
        now = float(np.logaddexp(now, np.log(rng.exponential()) - total))
        claimed_set.add(g)
        claimed.append(g)
        frontier = {pair for pair in frontier if pair[1] != g}
        mass, event = _take_cell(problem, k, g, selected, owners, mass, now)
        events.append(event)
        if event.budget_trigger:
            triggered = True
            break
        for neighbor in problem.graph.adjacency[g]:
            pair = (k, neighbor)
            if neighbor not in claimed_set and pair not in seen:
                seen.add(pair)
                if np.isfinite(problem.log_rates[pair]):
                    frontier.add(pair)
    return _finish(problem, selected, owners, mass, events, claimed, triggered)


def score_candidate(
    problem: TerritoryProblem, result: GrowthResult,
    anchor_location_indices: Sequence[int], *, opening_penalty: float = 0.0,
    transfer_penalty: float = 0.0,
) -> float:
    """Actual-owner center cost + F(K-1) + prior-anchor transfer penalty."""
    if not result.supported or result.expected_mass <= 0:
        return float("inf")
    anchors = np.asarray(anchor_location_indices, dtype=np.int64)
    k_count = len(problem.anchor_cells)
    if anchors.shape != (k_count,):
        raise ValueError("One location index is required per preselected anchor")
    indices = result.selected_indices
    positive = problem.probabilities[indices] > 0
    indices, actual_owners = indices[positive], result.owners[positive]
    owned_costs = problem.road_costs[actual_owners, indices]
    center = float(np.sum(problem.probabilities[indices] * owned_costs)
                   / (result.expected_mass * problem.reference_length))
    transfer_cost = 0.0
    if transfer_penalty != 0:
        transfer_cost = transfer_penalty * sum(
            float(np.min(problem.road_costs[:k, anchors[k]])) for k in range(1, k_count)
        ) / problem.reference_length
    return center + opening_penalty * (k_count - 1) + transfer_cost
