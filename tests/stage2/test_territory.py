from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from evrptw_scenarios.territory import (
    TerritoryProblem, contract_empty_units, grow_clock, grow_direct,
    score_candidate, weighted_reference_length,
)


def problem(*, cells=("a", "b", "c"), pi=(1.0, 1.0, 1.0),
            graph=None, costs=None, anchors=("a",), target=2, length=1.0, tau=1.0):
    if graph is None:
        graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    if costs is None:
        costs = np.zeros((len(anchors), len(cells)))
    return TerritoryProblem(
        location_ids=[f"loc{i}" for i in range(len(cells))], cell_ids=cells,
        adjacency=graph, probabilities=pi, road_costs=costs, anchor_cells=anchors,
        target_count=target, reference_length=length, tau=tau,
    )


def test_contracts_empty_paths_without_skipping_occupied_units():
    graph = {"a": ["e1"], "e1": ["e2"], "e2": ["b"], "b": ["c"], "c": []}
    result = contract_empty_units(graph, ["a", "b", "c"])
    assert result.adjacency == {"a": ("b",), "b": ("c",), "c": ()}
    assert result.witness_paths["a", "b"] == ("a", "e1", "e2", "b")
    assert "a" not in result.adjacency["b"]


def test_reference_quantile_caps_target_by_total_mass_and_retains_inf():
    costs, pi = [4, 1, 2, 100], [.5, .25, .25, 0]
    assert weighted_reference_length(costs, pi, 1) == 4
    assert weighted_reference_length(costs, pi, 100) == 4
    assert weighted_reference_length([0, 0], [.5, .5], 1) == 0
    assert np.isinf(weighted_reference_length([1, np.inf], [.5, .5], 1))


def test_seed_cells_are_pending_and_share_one_budget():
    p = problem(cells=("a", "a", "b", "b"), pi=(.6, .6, .6, .6),
                graph={"a": [], "b": []}, anchors=("a", "b"), target=1)
    result = grow_clock(p, exponentials=np.array([[1, 2, 3, 4], [4, 3, 20, 30]]))
    assert len(result.events) == 1
    assert result.events[0].cell_id == "a"
    assert result.selected_indices.tolist() == [0, 1]
    assert result.B == pytest.approx(1.2)
    assert result.winning_anchor_count == result.contributing_anchor_count == 1


def test_boundary_prefix_respects_count_and_stable_location_ties():
    p = problem(cells=("a", "a", "a"), pi=(.6, .6, .6), graph={"a": []}, target=1)
    p = replace(p, location_ids=["z", "a", "m"])
    result = grow_clock(p, exponentials=np.ones((1, 3)))
    assert result.selected_indices.tolist() == [1, 2]
    assert result.supported
    # count constraint forbids a seemingly close empty prefix when N=2.
    p = replace(p, target_count=2)
    result = grow_clock(p, exponentials=np.ones((1, 3)))
    assert result.selected_indices.tolist() == [1, 2, 0]
    assert result.stop_reason == "frontier_exhausted"


def test_equal_budget_deviation_prefers_smaller_prefix_despite_roundoff():
    p = problem(cells=("a", "a"), pi=(.7, .6), graph={"a": []}, target=1)
    result = grow_clock(p, exponentials=np.ones((1, 2)))
    assert result.selected_indices.tolist() == [0]
    assert result.B == pytest.approx(.7)


def test_losing_deadline_is_not_restarted_after_other_events():
    p = problem(cells=("a", "b", "c", "d"), pi=(1, 1, 1, 1),
                graph={"a": ["c"], "b": [], "c": ["d"], "d": []},
                anchors=("a", "b"), target=4)
    draws = np.array([[.1, 20, .05, .99], [20, 1, 20, 20]])
    result = grow_clock(p, exponentials=draws)
    assert [e.cell_id for e in result.events] == ["a", "c", "b", "d"]
    assert np.exp([e.log_time for e in result.events]) == pytest.approx([.1, .15, 1, 1.14])


def test_duplicate_neighbor_exposure_does_not_add_or_restart_clock():
    p = problem(graph={"a": ["b", "c"], "b": ["c"], "c": []}, target=3)
    result = grow_clock(p, exponentials=np.array([[.1, .1, 10]]))
    assert [e.cell_id for e in result.events] == ["a", "b", "c"]
    assert np.exp([e.log_time for e in result.events]) == pytest.approx([.1, .2, 10.1])


def test_claim_cancels_other_source_clock_and_uses_fixed_complete_cell_rate():
    p = problem(cells=("a", "a", "b"), pi=(.2, .8, 1),
                graph={"a": ["b"], "b": []}, anchors=("a", "a"), target=2)
    assert np.exp(p.log_rates[0, "a"]) == pytest.approx(1)
    result = grow_clock(p, exponentials=np.array([[.02, 2, 1], [10, 10, 1]]))
    assert [e.cell_id for e in result.events] == ["a", "b"]
    assert result.owners.tolist() == [0, 0, 0]
    assert result.events[0].log_time == pytest.approx(np.log(.1))


def test_log_clock_does_not_drop_rates_underflowing_in_probability_space():
    p = problem(costs=np.array([[1000, 1001, 1002]]), target=2)
    assert np.exp(p.log_rates[0, "a"]) == 0
    result = grow_clock(p, exponentials=np.ones((1, 3)))
    assert result.supported
    assert result.selected_indices.tolist() == [0, 1]
    assert np.all(np.isfinite([e.log_time for e in result.events]))


def test_frontier_exhaustion_keeps_supported_low_expected_mass():
    p = problem(pi=(.01, .01, .01), target=2)
    result = grow_clock(p, exponentials=np.ones((1, 3)))
    assert result.supported
    assert result.stop_reason == "frontier_exhausted"
    assert result.B == pytest.approx(.03)
    isolated = replace(p, adjacency={"a": [], "b": [], "c": []})
    result = grow_clock(isolated, exponentials=np.ones((1, 3)))
    assert not result.supported
    assert result.stop_reason == "insufficient_candidate_count"


def test_shared_source_one_couples_k_when_source_two_never_wins():
    p1 = problem(cells=("a", "b", "c", "d"), pi=(.8, .8, .8, .8),
                 graph={"a": ["b"], "b": ["c"], "c": ["d"], "d": []}, target=2)
    p2 = replace(p1, road_costs=np.zeros((2, 4)), anchor_cells=("a", "d"))
    first = np.array([[.1, .2, .3, .4]])
    one = grow_clock(p1, exponentials=first)
    two = grow_clock(p2, exponentials=np.vstack((first, np.full(4, 1e6))))
    assert all(e.owner == 0 for e in two.events)
    assert one.selected_indices.tolist() == two.selected_indices.tolist()
    assert one.B == two.B


def test_zero_prefix_event_is_still_a_source_two_win_and_changes_outcome():
    p1 = problem(cells=("a", "a", "c", "b"), pi=(.9, .9, .2, .6),
                 graph={"a": ["c"], "c": [], "b": []},
                 costs=np.array([[1, 1, 2, 3]]), anchors=("a",), target=2)
    p2 = replace(p1, road_costs=np.array([[1, 1, 2, 3], [3, 3, 3, 1]]),
                 anchor_cells=("a", "b"))
    first = np.array([[.001, .001, 1, 10]])
    one = grow_clock(p1, exponentials=first)
    two = grow_clock(p2, exponentials=np.vstack((first, [10, 10, 10, .1])))
    assert one.selected_indices.tolist() == [0, 1, 2]
    assert two.selected_indices.tolist() == [0, 1]
    assert two.events[-1].owner == 1
    assert two.events[-1].selected_count == 0
    assert two.winning_anchor_count == 2
    assert two.contributing_anchor_count == 1


def test_actual_owner_score_and_prior_anchor_penalty():
    p = problem(pi=(.5, .5, .5), costs=np.array([[0, 2, 4], [4, 2, 0]]),
                anchors=("a", "c"), target=1, length=2)
    r = grow_clock(p, exponentials=np.ones((2, 3)))
    r = replace(r, selected_indices=np.array([1, 2]), owners=np.array([0, 1]),
                expected_mass=1.0, supported=True)
    assert score_candidate(p, r, [0, 2], opening_penalty=.3,
                           transfer_penalty=.4) == pytest.approx(1.6)


def _outcome(result):
    assignment = tuple(sorted(zip(result.selected_indices.tolist(), result.owners.tolist())))
    # Include the zero-prefix stopping cell and owner, not only selected locations.
    events = tuple((e.owner, e.cell_id, e.selected_count) for e in result.events)
    return assignment, events, result.supported, result.stop_reason


def test_complete_outcomes_match_independent_categorical_reference():
    p = problem(cells=("a", "a", "b", "c", "d"), pi=(.6, .6, .55, .8, .7),
                graph={"a": ["b", "c"], "b": ["a", "d"],
                       "c": ["a", "d"], "d": ["b", "c"]},
                costs=np.array([[0, .3, .8, 1.1, 1.5], [1.8, 1.5, 1, .6, 0]]),
                anchors=("a", "d"), target=2, length=1.4, tau=.9)
    repetitions = 12000
    clock_rng, direct_rng = np.random.default_rng(202610291), np.random.default_rng(202610292)
    clock = Counter(_outcome(grow_clock(p, rng=clock_rng)) for _ in range(repetitions))
    direct = Counter(_outcome(grow_direct(p, rng=direct_rng)) for _ in range(repetitions))
    outcomes = sorted(set(clock) | set(direct))
    counts = np.array([[clock[o], direct[o]] for o in outcomes])
    tv = float(np.abs(counts[:, 0] - counts[:, 1]).sum() / (2 * repetitions))
    common = counts.sum(axis=1) >= 20
    table = counts[common]
    if np.any(~common):
        table = np.vstack((table, counts[~common].sum(axis=0)))
    chi2, pvalue, _, _ = chi2_contingency(table.T, correction=False)
    print(f"clock/direct repetitions={repetitions} outcomes={len(outcomes)} "
          f"TV={tv:.6f} chi2={chi2:.4f} p={pvalue:.6g}")
    assert tv < .045
    assert pvalue > .00001
