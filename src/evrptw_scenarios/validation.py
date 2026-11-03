"""Small, seeded distribution comparisons; no city data or instance production."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import chi2_contingency, chisquare

from .activation import sample_conditional_bernoulli
from .capacity import OrderProposal, condition_order
from .territory import TerritoryProblem, grow_clock, grow_direct
from .util import write_json


def _exact_subsets(probabilities, size):
    """Independent elementary-symmetric DP normalizer, enumerated tiny support."""
    weights = np.asarray(probabilities) / (1 - np.asarray(probabilities))
    coefficients = np.zeros(size + 1)
    coefficients[0] = 1.
    for weight in weights:
        coefficients[1:] += weight * coefficients[:-1].copy()
    support = list(combinations(range(len(weights)), size))
    return support, np.asarray([np.prod(weights[list(s)]) / coefficients[size] for s in support])


def validate_modules(*, samples=12000, seed=20261029, output=None):
    if samples < 1000:
        raise ValueError("Use at least 1000 repetitions for these distribution comparisons")
    pi = np.array([.06, .14, .27, .43, .68, .88])
    support, expected = _exact_subsets(pi, 2)
    index = {s: j for j, s in enumerate(support)}
    observed = np.zeros(len(support), dtype=int)
    generator = np.random.default_rng(seed)
    attempts = []
    for _ in range(samples):
        result = sample_conditional_bernoulli(logit(pi), 2, generator)
        observed[index[tuple(result.indices)]] += 1
        attempts.append(result.diagnostics.attempts)
    chi, pvalue = chisquare(observed, samples * expected)
    activation = {"samples": samples, "support_size": len(support),
                  "total_variation": float(.5 * np.abs(observed / samples - expected).sum()),
                  "chi_square": float(chi), "p_value": float(pvalue),
                  "mean_proposals": float(np.mean(attempts)),
                  "passed": bool(pvalue > .001)}

    problem = TerritoryProblem(
        location_ids=[f"loc-{i}" for i in range(8)],
        cell_ids=["a", "a", "b", "c", "c", "d", "e", "e"],
        adjacency={"a": ["b", "c"], "b": ["a", "d"], "c": ["a", "d"],
                   "d": ["b", "c", "e"], "e": ["d"]},
        probabilities=[.35, .75, .6, .3, .8, .55, .4, .9],
        road_costs=np.array([[0, .2, .6, .7, .9, 1.3, 1.8, 2.],
                             [2., 1.8, 1.3, .9, .7, .6, .2, 0]]),
        anchor_cells=["a", "e"], target_count=3, reference_length=1.3, tau=.7,
    )
    clock_rng, direct_rng = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    clock, direct = Counter(), Counter()
    for _ in range(samples):
        for counts, result in [(clock, grow_clock(problem, rng=clock_rng)),
                               (direct, grow_direct(problem, rng=direct_rng))]:
            counts[(tuple(result.selected_indices), tuple(result.owners))] += 1
    outcomes = sorted(clock.keys() | direct.keys())
    a = np.array([clock[x] for x in outcomes])
    b = np.array([direct[x] for x in outcomes])
    chi, pvalue, dof, _ = chi2_contingency(np.vstack([a, b]), correction=False)
    frontier = {"samples_per_method": samples, "complete_outcomes": len(outcomes),
                "total_variation": float(.5 * np.abs(a / samples - b / samples).sum()),
                "chi_square": float(chi), "degrees_of_freedom": int(dof),
                "p_value": float(pvalue), "passed": bool(pvalue > .001)}

    # Three complete proposal types: .1 and .2 are feasible, .7 is oversized.
    generator = np.random.default_rng(seed + 3)
    labels, failures = Counter(), 0
    def propose(_attempt):
        label = str(generator.choice(["a", "b", "oversize"], p=[.1, .2, .7]))
        volume = 2. if label == "a" else 4. if label == "b" else 20.
        return OrderProposal(1, 1, {"label": label, "parcels": [{"volume_cm3": volume}]}, volume)
    for _ in range(samples):
        result = condition_order(propose, capacity_cm3=10., max_proposals=4)
        if result.accepted:
            labels[result.proposal.bundle["label"]] += 1
        else:
            failures += 1
    alpha, limit = .3, 4
    failure_probability = (1 - alpha) ** limit
    accepted = samples - failures
    failure_z = abs(failures / samples - failure_probability) / np.sqrt(
        failure_probability * (1 - failure_probability) / samples)
    conditional_z = abs(labels["a"] / accepted - 1 / 3) / np.sqrt((1 / 3) * (2 / 3) / accepted)
    capacity = {"samples": samples, "alpha": alpha, "proposal_limit": limit,
                "observed_failure_probability": failures / samples,
                "theoretical_failure_probability": failure_probability,
                "observed_accepted_a_fraction": labels["a"] / accepted,
                "theoretical_accepted_a_fraction": 1 / 3,
                "failure_z_score": float(failure_z), "conditional_z_score": float(conditional_z),
                "passed": bool(failure_z < 5 and conditional_z < 5)}
    report = {"schema": "evrptw_activation_module_validation_v1", "seed": seed,
              "scope": "seeded tiny distributions; no city production or route feasibility",
              "activation": activation, "frontier": frontier, "capacity": capacity,
              "passed": all(x["passed"] for x in [activation, frontier, capacity])}
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    return report
