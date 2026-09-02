"""The small end of the ladder, where the output token is coarse.

A probe whose answer is a few units of the output token measures rounding, not
a curve.  `plan_deltas` therefore floors the ladder by what the *far* side can
express, not only by what the near side can -- and every pass that re-asks
sizes on an arc owes the same floor, `plan_refine` off the reserve and
`plan_sized` off the ladder it already has.
"""

from erouter.core.probe import (
    GRID,
    MIN_OUT_QUANTA,
    ArcRef,
    Ladder,
    plan_deltas,
    plan_refine,
    plan_sized,
)
from erouter.core.types import ArcKind


def test_a_coarse_output_moves_the_small_end_out():
    # Mainnet GUSD/3Crv: 648,237 3Crv (18 decimals) against 393,473 GUSD (2).
    reserve_in = 648_237 * 10**18
    reserve_out = 393_473 * 10**2

    deltas = plan_deltas(reserve_in, 18, GRID, reserve_out)

    smallest = deltas[0] / 10**18
    answer_units = deltas[0] * reserve_out // reserve_in
    assert answer_units >= MIN_OUT_QUANTA, answer_units
    # And the ladder is still a ladder, spanning decades to the top of the grid.
    assert len(deltas) >= 2
    assert deltas[-1] == int(reserve_in * GRID[-1])
    assert smallest > 1.0, "the 1e-6 node cannot be measured in GUSD"


def test_an_18_decimal_pair_keeps_the_whole_grid():
    reserve_in = reserve_out = 1_000_000 * 10**18

    with_out = plan_deltas(reserve_in, 18, GRID, reserve_out)
    without = plan_deltas(reserve_in, 18, GRID)

    assert with_out == without


def test_no_far_reserve_is_not_a_floor_of_zero():
    """An unknown far side must not silently drop the floor."""
    reserve_in = 1_000_000 * 10**18
    assert plan_deltas(reserve_in, 18, GRID, 0) == plan_deltas(reserve_in, 18, GRID)


def _arc(**kw):
    return ArcRef(pool="0x" + "11" * 20, kind=ArcKind.SWAP_STABLE, i=0, j=1,
                  n_coins=2, **kw)


def test_a_trade_smaller_than_the_ladder_asks_for_nothing():
    """`a` is read off the smallest node; a smaller one is a coarser quote."""
    arc = _arc(reserve_in=10**24, decimals_in=18, decimals_out=6,
               reserve_out=10**12)
    ladder = Ladder(arc=arc, deltas=[10**18, 10**20], quotes=[10**6, 10**8])

    # 5%, 10% and 20% of a trade four orders below the ladder's own floor.
    plan = plan_sized([ladder], {arc.id: [5 * 10**13, 10**14, 2 * 10**14]})

    assert len(plan.probes) == 0


def test_a_trade_above_the_ladder_still_gets_its_nodes():
    arc = _arc(reserve_in=10**24, decimals_in=18, decimals_out=6,
               reserve_out=10**12)
    ladder = Ladder(arc=arc, deltas=[10**18, 10**20], quotes=[10**6, 10**8])

    plan = plan_sized([ladder], {arc.id: [5 * 10**13, 5 * 10**18, 10**19]})

    # The two the ladder can resolve, and only those.
    assert [p.dx for p in plan.probes] == [5 * 10**18, 10**19]


def test_a_retry_owes_the_same_output_floor_as_the_pass_it_retries():
    """`plan_refine` re-asks sizes on an arc `plan_grid` already floored."""
    reserve_in, reserve_out = 648_237 * 10**18, 393_473 * 10**2
    arc = _arc(reserve_in=reserve_in, decimals_in=18, decimals_out=2,
               reserve_out=reserve_out)

    plan = plan_refine([Ladder(arc=arc)], {arc.id}, grid=GRID)

    smallest = min(p.dx for p in plan.probes)
    assert smallest * reserve_out // reserve_in >= MIN_OUT_QUANTA
