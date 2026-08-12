"""The coarse grid's sizing rule, and the cap that makes it safe.

Sizing probes as a fraction of each pool's own reserve means a probe measures
"a bit of this pool", so the same nominal grid samples $16k in 3pool and $2 in
a $20k venue -- and the shallow one gets fitted three orders below where any
real trade lands.  A dollar grid fixes that.

On its own it also breaks things, which is what most of this file is about:
$50,000 into a $20k venue is not a probe, it is the whole pool.  The chord it
measures sits nowhere near the mid, `sqrt(a_f * a_r)` collapses, §4's
round-trip floor mutes the arc, and the price frame that falls out broke flow
conservation by 1.3e-4 on a live crvUSD->sDOLA route.  Hence the depth cap.
"""

from __future__ import annotations

from erouter.core.probe import (
    RESERVE_CEILING,
    VALUE_GRID_USD,
    ArcRef,
    plan_deltas,
    plan_grid,
    plan_value_deltas,
)
from erouter.core.types import ArcKind

DEEP = 400_000_000 * 10**6  # $400M of a 6-decimal token
SHALLOW = 20_000 * 10**6  # $20k of the same


def _arc(reserve: int, decimals: int = 6, token: str = "0xUSDC") -> ArcRef:
    return ArcRef(
        pool="0xpool", kind=ArcKind.SWAP_STABLE, i=0, j=1, n_coins=2,
        reserve_in=reserve, decimals_in=decimals, decimals_out=18, token_in=token,
    )


def test_dollar_sizing_is_the_same_size_in_every_pool():
    """The whole point: a deep and a shallow pool get the same notional."""
    deep = plan_value_deltas(1.0, 6, reserve_in=DEEP)
    assert deep == [int(usd * 10**6) for usd in VALUE_GRID_USD]


def test_the_cap_binds_on_a_pool_too_shallow_to_probe_that_hard():
    """$50k into a $20k pool becomes a fraction of the pool instead."""
    sizes = plan_value_deltas(1.0, 6, reserve_in=SHALLOW)
    assert max(sizes) <= RESERVE_CEILING * SHALLOW
    # ...and it is the cap that bound, not the floor or a rounding accident.
    assert max(sizes) == int(RESERVE_CEILING * SHALLOW)


def test_the_cap_scales_the_ladder_rather_than_clipping_it():
    """Both points must survive, or there is no curvature estimate.

    Clipping each point at the cap collapses them onto the same value on
    exactly the shallow pools that need the span most, and two equal deltas are
    a zero denominator in the divided differences.
    """
    sizes = plan_value_deltas(1.0, 6, reserve_in=SHALLOW)
    assert len(sizes) == len(VALUE_GRID_USD)
    assert sizes[0] < sizes[1]
    # The geometric span is preserved, because the cap is one scale factor.
    uncapped = VALUE_GRID_USD[1] / VALUE_GRID_USD[0]
    assert sizes[1] / sizes[0] == uncapped


def test_an_unpriced_token_falls_back_to_the_reserve_grid():
    """Coverage may not depend on a third party knowing about a token.

    §5.5's certificate prices out *every* arc, so an arc that silently lost its
    `a` would quietly weaken the optimality claim rather than fail loudly.
    """
    arcs = [_arc(DEEP, token="0xexotic")]
    plan = plan_grid(arcs, prices={"0xusdc": 1.0})
    assert plan.deltas == [plan_deltas(DEEP, 6)]


def test_a_priced_token_uses_the_value_grid():
    arcs = [_arc(DEEP, token="0xUSDC")]
    plan = plan_grid(arcs, prices={"0xusdc": 1.0})
    assert plan.deltas == [plan_value_deltas(1.0, 6, reserve_in=DEEP)]


def test_no_prices_at_all_is_the_old_behaviour_exactly():
    arcs = [_arc(DEEP), _arc(SHALLOW)]
    assert plan_grid(arcs, prices=None).deltas == plan_grid(arcs).deltas


def test_a_worthless_token_does_not_ask_for_an_astronomical_probe():
    """`1e-18 USD` would size a $1,000 probe at 1e21 tokens; take the reserve."""
    arcs = [_arc(DEEP, decimals=18, token="0xdust")]
    plan = plan_grid(arcs, prices={"0xdust": 1e-30})
    # Sized off the reserve, and in any case never larger than the cap allows.
    assert max(plan.deltas[0]) <= max(plan_deltas(DEEP, 18))


def test_deltas_stay_strictly_increasing_on_a_two_decimal_token():
    """Two-decimal tokens exist in the live universe and round deltas together."""
    sizes = plan_value_deltas(1.0, 2, reserve_in=10_000 * 10**2)
    assert sizes == sorted(set(sizes))
