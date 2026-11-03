from collections import Counter
import math

import numpy as np
import pytest

from evrptw_scenarios.capacity import (
    OrderProposal, condition_order, draw_active_units, draw_order_counts, sample_capacity_order,
)


def proposal(active, volumes, label):
    bundle = {"package_count": len(volumes), "source_donor_id": label,
              "parcels": [{"volume_cm3": v, "source_parcel_ref": f"{label}:{i}"}
                          for i, v in enumerate(volumes)]}
    return OrderProposal(active, len(volumes), bundle, math.fsum(volumes))


def test_enumerated_conditioned_law_and_finite_limit_failure():
    """Independent enumerated f: alpha=.3; R=4; P(failure)=.7**4."""
    options = [proposal(1, [1.], "a"), proposal(2, [.5, .5, 1.], "b"),
               proposal(1, [2., 2.], "c"), proposal(3, [1.] * 4, "d")]
    generator = np.random.default_rng(20261029)
    outcomes = Counter()
    repetitions, limit, alpha = 20_000, 4, .3
    for _ in range(repetitions):
        result = condition_order(
            lambda _attempt: options[int(generator.choice(4, p=[.1, .2, .3, .4]))],
            capacity_cm3=2., max_proposals=limit,
        )
        if result.accepted:
            outcomes[result.proposal.bundle["source_donor_id"]] += 1
            assert result.attempts == len(result.rejections) + 1
            assert result.failure_reason is None
        else:
            outcomes["unsupported"] += 1
            assert result.attempts == limit
            assert result.proposal is None
        assert all(r.proposal.volume_cm3 > 2 for r in result.rejections)
    expected_failure = (1 - alpha) ** limit
    failure_rate = outcomes["unsupported"] / repetitions
    failure_sigma = math.sqrt(expected_failure * (1 - expected_failure) / repetitions)
    assert abs(failure_rate - expected_failure) < 5 * failure_sigma
    accepted = repetitions - outcomes["unsupported"]
    expected_a = .1 / alpha
    sigma_a = math.sqrt(expected_a * (1 - expected_a) / accepted)
    assert abs(outcomes["a"] / accepted - expected_a) < 5 * sigma_a
    assert outcomes["a"] + outcomes["b"] == accepted


def test_all_rejected_preserves_every_whole_order_and_fixed_attempts():
    orders = []

    def propose(attempt):
        order = proposal(2, [2., 3., 4.], f"donor-{attempt}")
        orders.append(order)
        return order

    result = condition_order(propose, capacity_cm3=8.)
    assert not result.accepted
    assert result.proposal is None
    assert result.failure_reason == "capacity_proposal_limit_exhausted"
    assert result.attempts == 64
    assert [r.attempt for r in result.rejections] == list(range(1, 65))
    assert [r.proposal for r in result.rejections] == orders
    assert all(r.proposal.active_units == 2 and r.proposal.package_count == 3
               and r.proposal.volume_cm3 == 9. for r in result.rejections)


def test_full_order_redraw_keeps_proposed_count_and_joint_marks():
    received = []

    def sampler(count, generator):
        received.append((count, generator))
        # Deliberately different donor count: it remains provenance only.
        volume = 3. if len(received) == 1 else 1.
        return {"package_count": count, "source_package_count": 99,
                "parcels": [{"volume_cm3": volume, "depth_cm": volume,
                             "height_cm": 1., "width_cm": 1., "joint_mark": "retained"}
                            for _ in range(count)]}

    counts_rng, marks_rng = np.random.default_rng(1), np.random.default_rng(2)
    result = sample_capacity_order(
        units=2, probability=1., mean_parcels=1., counts_rng=counts_rng,
        marks_rng=marks_rng, bundle_sampler=sampler, capacity_cm3=2.,
    )
    assert result.accepted and result.attempts == 2
    assert [count for count, _ in received] == [2, 2]
    assert all(generator is marks_rng for _, generator in received)
    assert result.rejections[0].proposal.volume_cm3 == 6.
    assert result.rejections[0].proposal.bundle["parcels"][0]["depth_cm"] == 3.
    assert result.proposal.active_units == result.proposal.package_count == 2
    assert result.proposal.bundle["source_package_count"] == 99
    assert all(p["joint_mark"] == "retained" for p in result.proposal.bundle["parcels"])


@pytest.mark.parametrize("changed", ["length", "metadata"])
def test_sampler_cannot_replace_proposed_parcel_count(changed):
    def sampler(count, _generator):
        return {"package_count": count + (changed == "metadata"),
                "parcels": [{"volume_cm3": 1.}] * (count + (changed == "length"))}

    with pytest.raises(ValueError, match="preserve the proposed package_count"):
        sample_capacity_order(units=2, probability=1., mean_parcels=1.,
                              counts_rng=np.random.default_rng(0), marks_rng=None,
                              bundle_sampler=sampler, capacity_cm3=100.)


def test_zero_truncated_binomial_matches_exact_mass_and_is_stable_for_rare_activation():
    generator = np.random.default_rng(351)
    draws = np.array([draw_active_units(4, .2, generator) for _ in range(20_000)])
    positive_mass = 1 - .8 ** 4
    for active in range(1, 5):
        expected = math.comb(4, active) * .2 ** active * .8 ** (4 - active) / positive_mass
        sigma = math.sqrt(expected * (1 - expected) / len(draws))
        assert abs(np.mean(draws == active) - expected) < 5 * sigma
    assert all(draw_active_units(10, 1e-20, generator) == 1 for _ in range(100))
    assert draw_active_units(10, 1., generator) == 10
    assert draw_active_units(1_000_000, 1e-15, generator) == 1


def test_geometric_parcels_are_failures_plus_active_households():
    generator = np.random.default_rng(950)
    counts = np.array([draw_order_counts(3, 1., 1.4, generator)[1] for _ in range(20_000)])
    # Sum of three positive geometric variates: E[P]=4.2, Var[P]=1.68.
    assert counts.min() >= 3
    assert abs(counts.mean() - 4.2) < 5 * math.sqrt(1.68 / len(counts))
    assert draw_order_counts(3, 1., 1., generator) == (3, 3)


@pytest.mark.parametrize("units,p", [(0, .2), (1.5, .2), (True, .2), (5, 0),
                                       (5, -1), (5, 1.1), (5, float("nan"))])
def test_invalid_active_location_parameters(units, p):
    with pytest.raises(ValueError):
        draw_active_units(units, p, np.random.default_rng(0))


@pytest.mark.parametrize("capacity,limit", [(0., 64), (-1., 64), (float("inf"), 64),
                                           (1., 0), (1., 1.5), (1., True)])
def test_invalid_capacity_parameters(capacity, limit):
    with pytest.raises(ValueError):
        condition_order(lambda _attempt: proposal(1, [1.], "a"),
                        capacity_cm3=capacity, max_proposals=limit)
