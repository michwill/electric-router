"""Split-ratio optimisation against the quoter (§7) -- no chain.

The quoter is stubbed with a concave payoff whose optimum is known in closed
form, so these check the optimiser actually climbs rather than merely runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core.split import (
    BPS,
    apply_weights,
    optimise,
    should_optimise,
    split_groups,
    weights_of,
)
from erouter.core.types import ArcKind, Leg

POOL_A = "0x" + "a1" * 20
POOL_B = "0x" + "b2" * 20
POOL_C = "0x" + "c3" * 20


def leg(src: int, dst: int, bps: int, target: str = POOL_A) -> Leg:
    return Leg(target=target, kind=ArcKind.SWAP_STABLE, i=0, j=1, n=2,
               src_slot=src, dst_slot=dst, bps=bps)


# Two branches out of slot 0, then a single leg out of slot 1.
SPLIT = [leg(0, 1, 6000, POOL_A), leg(0, 2, 0, POOL_B), leg(1, 2, 0, POOL_C)]


class ConcaveQuoter:
    """`out = sum_k (x_k - x_k^2 / (2 K_k))`, maximised by equal marginals.

    Depth differs per branch, so the optimum is *not* the even split -- an
    optimiser that merely jitters symmetrically would not find it.
    """

    def __init__(self, depth: tuple[float, ...] = (4e6, 1e6)) -> None:
        self.depth = depth
        self.calls = 0

    def payoff(self, shares: list[float], amount: int) -> int:
        total = 0.0
        for share, k in zip(shares, self.depth, strict=True):
            x = share * amount
            total += x - x * x / (2 * k)
        return int(max(total, 0.0))

    def quote_routes(self, routes, amounts_in, dst_slots):
        self.calls += 1
        out = []
        for legs, amount in zip(routes, amounts_in, strict=True):
            head = legs[0].bps / BPS
            out.append(self.payoff([head, 1.0 - head], amount))
        return out


def best_share(depth: tuple[float, float], amount: int) -> float:
    """Equal-marginal split: 1 - x/Ka = 1 - y/Kb with x + y = amount."""
    ka, kb = depth
    return (ka / (ka + kb))


def test_split_groups_finds_contiguous_runs_that_actually_split():
    assert split_groups(SPLIT) == [[0, 1]]
    assert split_groups([leg(0, 1, 0)]) == []
    assert split_groups([]) == []


def test_a_slot_drained_by_two_separate_runs_is_two_groups():
    """Grouping must follow the quoter's rule -- contiguity, not slot identity."""
    legs = [leg(0, 1, 5000), leg(0, 2, 0), leg(1, 3, 0), leg(0, 4, 3000), leg(0, 5, 0)]
    assert split_groups(legs) == [[0, 1], [3, 4]]


def test_weights_round_trip_through_bps():
    groups = split_groups(SPLIT)
    w = weights_of(SPLIT, groups)
    assert len(w) == 1
    assert w[0] == pytest.approx([0.6, 0.4], abs=1e-9)
    back = apply_weights(SPLIT, groups, w)
    assert back[0].bps == 6000
    assert back[1].bps == 0, "the last leg of a group must sweep"


def test_apply_weights_never_starves_the_sweep():
    """A group whose head takes everything would leave the sweep a no-op leg."""
    groups = split_groups(SPLIT)
    out = apply_weights(SPLIT, groups, [np.array([0.999999, 1e-9])])
    assert out[0].bps <= BPS - 1
    assert out[1].bps == 0
    assert out[0].bps >= 1


def test_the_gate_skips_a_route_with_nothing_to_split():
    assert should_optimise([leg(0, 1, 0)], [0.9], modelled_out=100, verified_out=50) == ""


def test_the_gate_fires_on_a_deep_leg():
    assert "theta" in should_optimise(SPLIT, [0.02, 0.44])


def test_the_gate_fires_when_the_model_already_disagrees():
    """Shallow legs, but the route as a whole missed -- theta alone misses this."""
    reason = should_optimise(SPLIT, [0.01], modelled_out=1_000_000, verified_out=999_000)
    assert "model off by" in reason


def test_the_gate_stays_quiet_when_the_model_was_right_and_legs_are_small():
    assert should_optimise(SPLIT, [0.01], modelled_out=1_000_000, verified_out=999_999) == ""


def test_it_climbs_towards_the_equal_marginal_split():
    amount = 1_000_000
    quoter = ConcaveQuoter()
    tuned, report = optimise(SPLIT, quoter, amount_in=amount, dst_slot=2, baseline=0)
    assert report.improved
    assert report.after > report.before
    # The analytic optimum puts 80% through the deeper branch; the model's
    # starting guess was 60%.
    got = tuned[0].bps / BPS
    assert abs(got - best_share(quoter.depth, amount)) < 0.05, got


def test_it_never_returns_something_worse():
    """The whole safety argument: only a strict improvement is accepted."""
    quoter = ConcaveQuoter()
    baseline = quoter.quote_routes([SPLIT], [1_000_000], [2])[0]
    _tuned, report = optimise(SPLIT, quoter, amount_in=1_000_000, dst_slot=2,
                              baseline=baseline)
    assert report.after >= report.before


def test_it_leaves_an_unsplittable_route_alone_and_spends_nothing():
    quoter = ConcaveQuoter()
    legs = [leg(0, 1, 0)]
    tuned, report = optimise(legs, quoter, amount_in=1000, dst_slot=1, baseline=99)
    assert tuned == legs
    assert report.calls == 0
    assert quoter.calls == 0
    assert report.skipped


def test_it_stays_within_its_round_trip_budget():
    quoter = ConcaveQuoter()
    _tuned, report = optimise(SPLIT, quoter, amount_in=1_000_000, dst_slot=2,
                              baseline=0, max_rounds=3)
    # One baseline quote, then at most two batches per round.
    assert report.calls <= 1 + 2 * 3
