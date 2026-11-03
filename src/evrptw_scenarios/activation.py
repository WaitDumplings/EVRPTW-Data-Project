"""Fixed-count location activation by exponentially tilted rejection.

For log-odds ``theta``, an N-subset has probability proportional to
``exp(sum(theta[selected]))``. Adding a common tilt to every log-odds leaves
that conditional law unchanged. The tilt only centers the rejection proposal;
it does not change the household ordering probability used for later bundles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


@dataclass(frozen=True)
class TiltResult:
    eta: float
    probabilities: np.ndarray
    budget: float
    proposal_variance: float


@dataclass(frozen=True)
class ActivationDiagnostics:
    eta: float
    proposal_variance: float
    attempts: int
    budget: float
    proposal_budget: float


@dataclass(frozen=True)
class ActivationResult:
    indices: np.ndarray
    diagnostics: ActivationDiagnostics


def unit_log_odds(units, probability: float) -> np.ndarray:
    """Return logit(1-(1-p)**u) without rounding large-u probabilities to one.

    Zero units or p=0 have zero activity; p=1 makes every positive-unit
    location mandatory. Ordinary modeled residential units are positive ints.
    """
    units = np.asarray(units, dtype=float)
    if (units.ndim != 1 or not np.isfinite(units).all()
            or np.any(units < 0) or np.any(units != np.floor(units))):
        raise ValueError("units must be a one-dimensional array of nonnegative integers")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    if probability == 0:
        return np.full(units.shape, -np.inf)
    if probability == 1:
        return np.where(units > 0, np.inf, -np.inf)
    z = -units * np.log1p(-probability)
    with np.errstate(divide="ignore"):
        log_pi = np.log(-np.expm1(-z))
    return z + log_pi


def _validate(log_odds, size):
    theta = np.asarray(log_odds, dtype=float)
    if theta.ndim != 1 or np.isnan(theta).any():
        raise ValueError("log_odds must be one-dimensional and contain no NaN")
    if not isinstance(size, (int, np.integer)) or not 0 <= size <= len(theta):
        raise ValueError("size must be an integer between zero and the population size")
    mandatory = np.isposinf(theta)
    possible = ~np.isneginf(theta)
    if mandatory.sum() > size or possible.sum() < size:
        raise ValueError("size is outside the Bernoulli distribution's positive support")
    return theta, int(size), mandatory


def solve_bernoulli_tilt(log_odds, size: int) -> TiltResult:
    """Center independent Bernoulli proposals at ``size`` using a common tilt.

    Finite log-odds remain eligible even when their unconditioned probabilities
    round to one. Centering log-odds before root finding also handles large,
    nearly equal values without losing the small proposal offsets.
    """
    theta, size, mandatory = _validate(log_odds, size)
    budget = float(expit(theta).sum())
    finite = np.isfinite(theta)
    free = theta[finite]
    remaining = size - int(mandatory.sum())
    probabilities = mandatory.astype(float)
    if not len(free):
        eta = 0.0
    elif remaining == 0:
        eta = -np.inf
    elif remaining == len(free):
        probabilities[finite] = 1.0
        eta = np.inf
    else:
        # The smallest of the top `remaining` log-odds is a useful numerical
        # origin. A common shift cancels from all fixed-count subset weights.
        pivot = float(np.partition(free, len(free) - remaining)[len(free) - remaining])
        centered = free - pivot

        def difference(offset):
            return float(expit(centered + offset).sum()) - remaining

        lower, upper = -1.0, 1.0
        while difference(lower) > 0:
            lower *= 2
        while difference(upper) < 0:
            upper *= 2
        offset = brentq(difference, lower, upper, xtol=1e-12)
        probabilities[finite] = expit(centered + offset)
        eta = float(offset - pivot)
    variance = float(np.sum(probabilities * (1 - probabilities)))
    return TiltResult(eta, probabilities, budget, variance)


def sample_conditional_bernoulli(log_odds, size: int,
                                 rng: np.random.Generator) -> ActivationResult:
    """Sample exactly ``size`` indices from the conditional Bernoulli law.

    Rejection continues until the proposed cardinality is correct; there is no
    attempt cap, seed replacement, demand draw, or location repair. Deterministic
    support cases consume no random numbers and report zero proposal attempts.
    """
    tilt = solve_bernoulli_tilt(log_odds, size)
    q = tilt.probabilities
    attempts = 0
    if tilt.proposal_variance == 0:
        selected = np.flatnonzero(q)
    else:
        while True:
            attempts += 1
            selected = np.flatnonzero(rng.random(len(q)) < q)
            if len(selected) == size:
                break
    diagnostics = ActivationDiagnostics(
        eta=tilt.eta,
        proposal_variance=tilt.proposal_variance,
        attempts=attempts,
        budget=tilt.budget,
        proposal_budget=float(q.sum()),
    )
    return ActivationResult(selected, diagnostics)
