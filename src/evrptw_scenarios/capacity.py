"""Whole-order capacity conditioning after a location has been selected.

A proposal contains active households A, parcel count P, and all parcel marks.
An oversized proposal is rejected in full. Reaching the proposal limit leaves
the same location unsupported; this module never substitutes another location.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OrderProposal:
    active_units: int
    package_count: int
    bundle: dict[str, Any]
    volume_cm3: float


@dataclass(frozen=True)
class RejectedOrder:
    attempt: int
    proposal: OrderProposal
    reason: str


@dataclass(frozen=True)
class CapacityResult:
    accepted: bool
    proposal: OrderProposal | None
    attempts: int
    rejections: tuple[RejectedOrder, ...]
    failure_reason: str | None


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number != value or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def draw_active_units(units, probability, generator: np.random.Generator) -> int:
    """Draw Binomial(units, probability) conditional on at least one success.

First draw the first successful household from a geometric law truncated at
``units``, then draw successes among the remaining households. ``expm1`` and
``log1p`` preserve the positive-event mass even when ``units * probability``
is tiny, without a rejection loop or adding that mass to a CDF near one.
"""
    units = _positive_integer(units, "units")
    probability = float(probability)
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("An active location requires 0 < probability <= 1")
    if probability == 1:
        return units
    log_failure = math.log1p(-probability)
    positive_mass = -math.expm1(units * log_failure)
    quantile = math.log1p(-float(generator.random()) * positive_mass) / log_failure
    first_success = min(units, max(1, math.ceil(quantile)))
    return 1 + int(generator.binomial(units - first_success, probability))


def draw_order_counts(units, probability, mean_parcels, generator) -> tuple[int, int]:
    """Draw A | A > 0, then P = A + NB(A, 1 / mean_parcels).

The negative-binomial variate counts failures, so each active household has
at least one parcel and the mean parcel count per active household is the
declared ``mean_parcels``.
"""
    mean_parcels = float(mean_parcels)
    if not math.isfinite(mean_parcels) or mean_parcels < 1:
        raise ValueError("mean_parcels must be finite and at least one")
    active = draw_active_units(units, probability, generator)
    parcels = active + int(generator.negative_binomial(active, 1 / mean_parcels))
    return active, parcels


def condition_order(
    propose: Callable[[int], OrderProposal], *, capacity_cm3, max_proposals=64,
) -> CapacityResult:
    """Accept the first complete proposal with volume <= capacity.

``propose(attempt)`` receives a one-based attempt number. For independent
proposals from f, the accepted law is exactly f conditioned on fitting the
capacity. The finite limit controls only whether the location is supported;
every rejected proposal remains available in ``rejections``.
"""
    capacity_cm3 = float(capacity_cm3)
    if not math.isfinite(capacity_cm3) or capacity_cm3 <= 0:
        raise ValueError("capacity_cm3 must be finite and positive")
    max_proposals = _positive_integer(max_proposals, "max_proposals")
    rejected = []
    for attempt in range(1, max_proposals + 1):
        proposal = propose(attempt)
        active = _positive_integer(proposal.active_units, "active_units")
        parcels = _positive_integer(proposal.package_count, "package_count")
        if parcels < active:
            raise ValueError("package_count cannot be less than active_units")
        if not math.isfinite(proposal.volume_cm3) or proposal.volume_cm3 <= 0:
            raise ValueError("A complete order must have finite positive volume")
        if proposal.volume_cm3 <= capacity_cm3:
            return CapacityResult(True, proposal, attempt, tuple(rejected), None)
        rejected.append(RejectedOrder(attempt, proposal, "order_volume_exceeds_capacity"))
    return CapacityResult(
        False, None, max_proposals, tuple(rejected), "capacity_proposal_limit_exhausted",
    )


def sample_capacity_order(
    *, units, probability, mean_parcels, counts_rng, marks_rng, bundle_sampler,
    capacity_cm3, max_proposals=64,
) -> CapacityResult:
    """Condition a fixed location's complete (A, P, marks) order on capacity.

``bundle_sampler(P, marks_rng)`` must return exactly P parcels. Source donor
counts may be retained as metadata, but cannot replace P. Parcel dimensions
and other marks are left intact; volume is summed from the sampled parcels.
    """
    def propose(_attempt):
        active, count = draw_order_counts(units, probability, mean_parcels, counts_rng)
        bundle = bundle_sampler(count, marks_rng)
        if len(bundle["parcels"]) != count or bundle.get("package_count", count) != count:
            raise ValueError("bundle_sampler must preserve the proposed package_count")
        volumes = [float(parcel["volume_cm3"]) for parcel in bundle["parcels"]]
        if any(not math.isfinite(volume) or volume <= 0 for volume in volumes):
            raise ValueError("Parcel volumes must be finite and positive")
        return OrderProposal(active, count, bundle, math.fsum(volumes))

    return condition_order(propose, capacity_cm3=capacity_cm3, max_proposals=max_proposals)
