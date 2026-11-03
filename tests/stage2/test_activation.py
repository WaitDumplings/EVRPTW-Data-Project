"""Small-population distribution checks against an independent probability DP."""
from copy import deepcopy
from itertools import combinations
import json

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import chi2

from evrptw_scenarios.activation import (
    sample_conditional_bernoulli,
    solve_bernoulli_tilt,
    unit_log_odds,
)


def exact_subset_probabilities(probabilities, size):
    """Independent reference: Poisson-binomial probability DP, then Bayes."""
    p = np.asarray(probabilities, dtype=float)
    table = np.zeros((len(p) + 1, size + 1))
    table[0, 0] = 1
    for i, probability in enumerate(p, start=1):
        table[i] = table[i - 1] * (1 - probability)
        table[i, 1:] += table[i - 1, :-1] * probability
    result = {}
    for selected in combinations(range(len(p)), size):
        mask = np.zeros(len(p), dtype=bool)
        mask[list(selected)] = True
        mass = np.prod(p[mask]) * np.prod(1 - p[~mask])
        result[selected] = float(mass / table[-1, size])
    return result


def distribution_comparison(probabilities, size, *, samples=20000, seed=20261029):
    """Return reproducible report metrics for tests and the small-sample audit."""
    exact = exact_subset_probabilities(probabilities, size)
    counts = dict.fromkeys(exact, 0)
    theta = logit(probabilities)
    generator = np.random.default_rng(seed)
    attempts = 0
    for _ in range(samples):
        result = sample_conditional_bernoulli(theta, size, generator)
        counts[tuple(result.indices)] += 1
        attempts += result.diagnostics.attempts
    expected = np.array(list(exact.values())) * samples
    observed = np.array(list(counts.values()))
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    tilt = solve_bernoulli_tilt(theta, size)
    return {
        "population": len(probabilities), "size": size, "samples": samples, "seed": seed,
        "subsets": len(exact), "exact_mass": float(sum(exact.values())),
        "minimum_expected_count": float(expected.min()),
        "total_variation": float(np.abs(observed / samples - expected / samples).sum() / 2),
        "chi_square": statistic, "degrees_of_freedom": len(exact) - 1,
        "chi_square_p_value": float(chi2.sf(statistic, len(exact) - 1)),
        "mean_attempts": attempts / samples, "eta": tilt.eta,
        "budget": tilt.budget, "proposal_variance": tilt.proposal_variance,
    }


@pytest.mark.parametrize("probabilities,size,seed", [
    ([.12, .23, .35, .47, .60, .72], 2, 20261029),
    ([.15, .25, .35, .45, .55, .65, .75], 3, 20261030),
])
def test_conditional_distribution_matches_independent_dp(probabilities, size, seed):
    report = distribution_comparison(probabilities, size, seed=seed)
    print(json.dumps(report, sort_keys=True))
    assert report["exact_mass"] == pytest.approx(1)
    assert report["minimum_expected_count"] > 5
    assert report["total_variation"] < .035
    assert report["chi_square_p_value"] > .001


def test_unit_log_odds_retains_finite_large_building_weights():
    theta = unit_log_odds([0, 1, 2, 100, 10000, 10**12], .01)
    assert theta[0] == -np.inf
    assert np.isfinite(theta[1:]).all()
    assert theta[1] == pytest.approx(np.log(.01 / .99))
    assert theta[2] == pytest.approx(np.log((1 - .99**2) / .99**2))
    assert np.all(np.diff(theta[1:]) > 0)
    # All raw activities round to one, but conditioning still selects two
    # exchangeable locations rather than incorrectly declaring all mandatory.
    large = unit_log_odds([10**9] * 4, .01)
    tilt = solve_bernoulli_tilt(large, 2)
    np.testing.assert_allclose(tilt.probabilities, .5)
    assert tilt.proposal_variance == pytest.approx(1)
    result = sample_conditional_bernoulli(large, 2, np.random.default_rng(42))
    assert len(result.indices) == 2


def test_fixed_count_reproducibility_and_common_tilt_invariance():
    theta = unit_log_odds([1, 2, 5, 10, 30, 100, 300, 900], .01)
    first = solve_bernoulli_tilt(theta, 3)
    shifted = solve_bernoulli_tilt(theta + 17, 3)
    assert first.probabilities.sum() == pytest.approx(3, abs=1e-11)
    np.testing.assert_allclose(first.probabilities, shifted.probabilities, atol=1e-13)
    assert shifted.eta == pytest.approx(first.eta - 17)
    generators = [np.random.default_rng(1729) for _ in range(3)]
    for _ in range(100):
        a = sample_conditional_bernoulli(theta, 3, generators[0])
        b = sample_conditional_bernoulli(theta, 3, generators[1])
        c = sample_conditional_bernoulli(theta + 17, 3, generators[2])
        assert len(a.indices) == 3 and len(np.unique(a.indices)) == 3
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.indices, c.indices)
        assert a.diagnostics == b.diagnostics
        assert a.diagnostics.attempts == c.diagnostics.attempts
        assert a.diagnostics.budget == pytest.approx(expit(theta).sum())


def test_deterministic_support_consumes_no_random_numbers():
    generator = np.random.default_rng(7)
    state = deepcopy(generator.bit_generator.state)
    all_sites = sample_conditional_bernoulli([-9., 0., 8.], 3, generator)
    np.testing.assert_array_equal(all_sites.indices, [0, 1, 2])
    assert all_sites.diagnostics.attempts == 0
    assert all_sites.diagnostics.proposal_variance == 0
    assert all_sites.diagnostics.eta == np.inf
    assert generator.bit_generator.state == state
    empty = sample_conditional_bernoulli([], 0, generator)
    assert not len(empty.indices)
    assert empty.diagnostics.attempts == 0
    certain = sample_conditional_bernoulli([-np.inf, np.inf, .5], 1, generator)
    np.testing.assert_array_equal(certain.indices, [1])
    assert generator.bit_generator.state == state


def test_mandatory_and_impossible_locations_are_respected():
    generator = np.random.default_rng(37)
    for _ in range(100):
        result = sample_conditional_bernoulli([-np.inf, .2, np.inf, -.8], 2, generator)
        assert 2 in result.indices and 0 not in result.indices
        assert len(result.indices) == 2
    np.testing.assert_array_equal(unit_log_odds([0, 1, 10], 0), [-np.inf] * 3)
    np.testing.assert_array_equal(unit_log_odds([0, 1, 10], 1), [-np.inf, np.inf, np.inf])


@pytest.mark.parametrize("theta,size", [
    ([np.nan], 0), ([[0., 1.]], 1), ([0.], 2), ([0.], 0.5),
    ([np.inf, np.inf], 1), ([-np.inf, 0.], 2),
])
def test_invalid_or_zero_probability_cardinality_is_not_repaired(theta, size):
    with pytest.raises(ValueError):
        sample_conditional_bernoulli(theta, size, np.random.default_rng(1))


@pytest.mark.parametrize("units,p", [([-1], .01), ([1.5], .01), ([1], 1.1)])
def test_invalid_household_inputs(units, p):
    with pytest.raises(ValueError):
        unit_log_odds(units, p)
