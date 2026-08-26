"""A slippage budget the caller names, divided between the legs.

The automatic rule sets each leg from its own pool and the route's total is
whatever that adds up to.  This is the other direction, and the property worth
testing is not the arithmetic of one leg but the one that holds across the
route: every path from the source to the destination spends the budget once.
"""

from __future__ import annotations

import pytest

from erouter.core import routecall as rc
from erouter.core import slippage
from erouter.core.realize import RealizedLeg, RealizedRoute
from erouter.core.types import ArcKind, Leg

UNIT = 10**18
TOKEN = {k: "0x" + f"{k:02x}" * 20 for k in range(8)}
POOL = {k: "0x" + f"{k + 0xa0:02x}" * 20 for k in range(8)}


def leg(src, dst, amount_in, amount_out, *, fee=0.0, pool=0):
    return RealizedLeg(
        leg=Leg(target=POOL[pool], kind=ArcKind.SWAP_STABLE, i=0, j=1, n=2,
                src_slot=src, dst_slot=dst),
        kind=ArcKind.SWAP_STABLE, target=POOL[pool],
        token_in=TOKEN[src], token_out=TOKEN[dst],
        amount_in=amount_in, amount_out=amount_out, gamma_live=1.0 - fee)


def route(legs, *, amount_in, dst_slot):
    return RealizedRoute(
        legs=legs, amount_in=amount_in, dst_slot=dst_slot,
        src_token=TOKEN[0], dst_token=TOKEN[dst_slot],
        slots={TOKEN[k]: k for k in TOKEN},
        modelled_out=sum(x.amount_out for x in legs if x.leg.dst_slot == dst_slot))


def granted(built, rates) -> list[float]:
    """What each leg's bound really allows, in bp, as the contract applies it."""
    return [(1 - rc.leg_in(x) * rate // rc.ONE / rc.leg_out(x)) * 1e4
            for x, rate in zip(built.legs, rates, strict=True)]


def series(*fees):
    """One leg per fee, chained, each halving the amount so no rate quantises."""
    legs = [leg(k, k + 1, 1000 * UNIT >> k, 1000 * UNIT >> (k + 1), fee=fee, pool=k)
            for k, fee in enumerate(fees)]
    return route(legs, amount_in=1000 * UNIT, dst_slot=len(fees))


# --------------------------------------------------------------- in series


def test_two_pools_in_series_share_in_proportion_to_their_fees():
    # crvUSD/USDT at 1 bp then crvUSD/YB at 20 bp, given 50 bp between them:
    # 50/21 and 20/21 * 50, exactly as two resistors divide a voltage.
    share = granted(series(1e-4, 20e-4),
                    rc.min_rates(series(1e-4, 20e-4), slippage_bp=50.0)[0])
    assert share[0] == pytest.approx(50 / 21, abs=1e-3)
    assert share[1] == pytest.approx(20 / 21 * 50, abs=1e-3)
    assert sum(share) == pytest.approx(50.0, abs=1e-3)


def test_a_smaller_budget_scales_every_leg_by_the_same_factor():
    built = series(1e-4, 20e-4)
    big = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    small = granted(built, rc.min_rates(built, slippage_bp=5.0)[0])
    assert [x / 10 for x in big] == pytest.approx(small, abs=1e-3)


def test_the_only_leg_of_a_route_takes_the_whole_budget():
    built = series(3e-4)
    assert granted(built, rc.min_rates(built, slippage_bp=30.0)[0]) == \
        pytest.approx([30.0], abs=1e-3)


def test_a_fee_free_leg_costs_the_budget_almost_nothing():
    # A wrap takes no fee, so its weight is the 0.1 bp of rounding room against
    # the pool's 4 bp: it is handed a fortieth of the budget, not half of it.
    built = series(0.0, 20e-4)
    share = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    assert share[0] == pytest.approx(50 / 41, abs=1e-3)
    assert share[1] == pytest.approx(40 / 41 * 50, abs=1e-3)


# --------------------------------------------------------------- and in parallel


def test_parallel_branches_each_drop_the_whole_budget():
    # USDC -> USDT over 3pool and over USDC/USDT at once.  Either branch alone
    # is the trade, so either one alone may spend it all -- and the fees do not
    # come into it, because there is nothing in series to divide against.
    built = route([leg(0, 1, 600 * UNIT, 600 * UNIT, fee=1e-4, pool=0),
                   leg(0, 1, 400 * UNIT, 400 * UNIT, fee=30e-4, pool=1)],
                  amount_in=1000 * UNIT, dst_slot=1)
    share = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    assert share == pytest.approx([50.0, 50.0], abs=1e-3)


def test_branches_of_unequal_length_still_net_to_the_same_tolerance():
    # Direct against a two-hop, in parallel.  The budget is a property of the
    # branch and not of its leg count: both net to it, and the two-hop's legs
    # divide their half of the circuit by their own fees.
    built = route([leg(0, 3, 500 * UNIT, 500 * UNIT, fee=4e-4, pool=0),
                   leg(0, 1, 500 * UNIT, 500 * UNIT, fee=1e-4, pool=1),
                   leg(1, 3, 500 * UNIT, 500 * UNIT, fee=3e-4, pool=2)],
                  amount_in=1000 * UNIT, dst_slot=3)
    share = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    assert share[0] == pytest.approx(50.0, abs=1e-3)
    assert share[1] + share[2] == pytest.approx(50.0, abs=1e-3)
    assert share[1] == pytest.approx(50 / 4, abs=1e-3)
    assert share[2] == pytest.approx(3 / 4 * 50, abs=1e-3)


def diamond():
    """Split two ways, merge, then one leg both branches pay for."""
    return route([leg(0, 1, 600 * UNIT, 600 * UNIT, fee=1e-4, pool=0),
                  leg(0, 1, 400 * UNIT, 400 * UNIT, fee=30e-4, pool=1),
                  leg(1, 2, 1000 * UNIT, 500 * UNIT, fee=20e-4, pool=2)],
                 amount_in=1000 * UNIT, dst_slot=2)


def test_every_path_through_a_split_spends_the_budget_exactly_once():
    built = diamond()
    share = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    assert share[0] + share[2] == pytest.approx(50.0, abs=1e-3)
    assert share[1] + share[2] == pytest.approx(50.0, abs=1e-3)


def test_a_shared_leg_carries_one_bound_and_the_branches_absorb_the_difference():
    # Per-path proportions would want two different bounds on the leg below the
    # merge -- 40 bp for one branch, 20 for the other.  The network settles it,
    # and the branches take the same drop however far apart their fees are.
    built = diamond()
    share = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    assert share[0] == pytest.approx(share[1], abs=1e-3)
    assert share[2] > share[0]


# --------------------------------------------------------------- the promise


def test_the_route_ships_the_tolerance_it_was_asked_for_and_not_a_wei_more():
    for shape in (series(1e-4, 20e-4), diamond()):
        call = rc.encode_route(shape, receiver=TOKEN[0], slippage_bp=50.0)
        assert 49.9 < call.tolerance_bp <= 50.0


def test_the_wei_of_rounding_survives_a_budget_too_small_to_pay_for_it():
    # 0.1 bp is not slippage, it is room for what a wrap rounds away, so the
    # budget does not get to take it back.
    built = series(1e-4, 20e-4)
    share = granted(built, rc.min_rates(built, slippage_bp=0.05)[0])
    assert min(share) >= rc.FLOOR_BP - 1e-9


def test_naming_no_budget_leaves_the_automatic_rule_alone():
    built = series(1e-4, 20e-4)
    assert granted(built, rc.min_rates(built)[0]) == \
        pytest.approx([0.2 * 1.0, 0.2 * 20.0], abs=1e-3)


def test_a_volatile_pair_is_weighed_by_the_floor_it_would_have_been_given():
    # The 5 bp floor shapes the weights even when it does not set the bound, so
    # the leg that needs room to survive honest movement gets the larger share.
    built = series(1e-4, 1e-4)
    pegged = granted(built, rc.min_rates(built, slippage_bp=50.0)[0])
    loose = granted(built, rc.min_rates(built, slippage_bp=50.0,
                                        volatile=[POOL[1]])[0])
    assert pegged == pytest.approx([25.0, 25.0], abs=1e-3)
    assert loose[1] > 45.0


def test_a_volatile_leg_keeps_its_floor_under_a_budget_that_would_starve_it():
    # The volatile branch's share of a thin parallel section is half a bp, and
    # 5 of them is what the pair needs to survive honest movement.  Movement is
    # not slippage, so the budget does not get to take it back.
    built = route([leg(0, 1, 500 * UNIT, 500 * UNIT, fee=1e-4, pool=0),
                   leg(0, 1, 500 * UNIT, 500 * UNIT, fee=1e-4, pool=1),
                   leg(1, 2, 1000 * UNIT, 500 * UNIT, fee=100e-4, pool=2)],
                  amount_in=1000 * UNIT, dst_slot=2)
    share = granted(built, rc.min_rates(built, slippage_bp=50.0,
                                        volatile=[POOL[1]])[0])
    assert share[0] == pytest.approx(0.476, abs=1e-2)
    assert share[1] == pytest.approx(rc.VOLATILE_FLOOR_BP, abs=1e-3)


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError):
        rc.min_rates(series(1e-4), slippage_bp=-1.0)


# --------------------------------------------------------------- the division itself


def test_the_drops_are_normalised_so_no_path_can_overspend():
    built = diamond()
    share = slippage.divide(built, [1.0, 1.0, 1.0], 0.005)
    assert slippage.longest(built, share) == pytest.approx(0.005)


def bridge():
    """S -> A -> T beside S -> B -> T, with a leg across the middle, A -> B.

    A Wheatstone bridge, and the only shape whose potentials can sit the head
    of a leg above its tail: an undirected network has no idea the route is a
    DAG.  Measured on mainnet, it is the BTC-to-BTC pool between two branches.
    """
    return route([leg(0, 1, 500 * UNIT, 500 * UNIT, pool=0),
                  leg(0, 2, 500 * UNIT, 500 * UNIT, pool=1),
                  leg(1, 2, 100 * UNIT, 100 * UNIT, pool=2),
                  leg(1, 3, 400 * UNIT, 400 * UNIT, pool=3),
                  leg(2, 3, 600 * UNIT, 600 * UNIT, pool=4)],
                 amount_in=1000 * UNIT, dst_slot=3)


#: S>A dear and A>T cheap, B the reverse -- so the network wants current in the
#: bridge running B to A, against the way the route sends value through it.
BENT = [w / 1e4 for w in (20.0, 1.0, 1.0, 1.0, 20.0)]


def test_a_bridge_leg_can_come_back_below_zero():
    raw = slippage.drops(bridge(), BENT, 0.005)
    assert raw[2] < 0.0
    # Negative or not, the potentials telescope: every path is still the budget
    # before anything is clamped.  That is what clamping breaks.
    for path in ([0, 3], [1, 4], [0, 2, 4]):
        assert sum(raw[k] for k in path) == pytest.approx(0.005)


def test_clamping_a_backwards_leg_alone_would_overspend_the_budget():
    built = bridge()
    clamped = [max(0.0, drop) for drop in slippage.drops(built, BENT, 0.005)]
    assert slippage.longest(built, clamped) > 0.0065  # 65 bp against a 50 bp ask


def test_a_backwards_leg_is_held_at_the_drop_it_came_back_by():
    # Not clamped to nothing: shipping a leg carrying a fifth of the route at
    # the rounding floor reverts on any movement.  The magnitude is what the
    # network says the imbalance across it is, so it is what the leg is owed.
    built = bridge()
    raw = slippage.drops(built, BENT, 0.005)
    share = slippage.divide(built, BENT, 0.005)
    assert share[2] == pytest.approx(-raw[2])
    assert min(share) > 0.0
    assert slippage.longest(built, share) == pytest.approx(0.005)


def test_the_rest_of_the_route_gives_way_to_make_room_for_it():
    built = bridge()
    alone = slippage.divide(built, BENT, 0.005)
    lifted = slippage.divide(built, BENT, 0.005,
                             backstop=[0.0, 0.0, 0.0030, 0.0, 0.0])
    assert lifted[2] == pytest.approx(0.0030)      # 30 bp, past the 15.6 it bent by
    assert all(a > b for a, b in zip(alone[:2], lifted[:2], strict=True))
    assert slippage.longest(built, lifted) == pytest.approx(0.005)


def test_a_backstop_past_the_budget_stands_and_the_total_says_so():
    # A floor is what a leg needs to survive movement.  It does not give way to
    # a budget too small to hold it; the total is what it is, and is reported.
    built = bridge()
    share = slippage.divide(built, BENT, 0.005,
                            backstop=[0.0, 0.0, 0.0060, 0.0, 0.0])
    assert share[2] == pytest.approx(0.0060)
    assert slippage.longest(built, share) == pytest.approx(0.0060)


def test_a_backwards_leg_never_ships_under_the_automatic_rule():
    # `min_rates` passes the automatic tolerances as the backstop, so a leg the
    # network bends only slightly still gets what it would have with no budget
    # named at all.
    built = bridge()
    weights = [w / 1e4 for w in (20.0, 1.0, 1.0, 1.0, 20.0)]
    raw = slippage.drops(built, weights, 0.005)
    auto = [0.0, 0.0, 0.0040, 0.0, 0.0]            # 40 bp, past the 15.6 it bent by
    assert slippage.divide(built, weights, 0.005, backstop=auto)[2] == \
        pytest.approx(max(-raw[2], 0.0040))


def test_a_route_of_pure_shorts_is_divided_by_depth():
    # Every resistance zero: nothing to be proportional to, so the budget is
    # spread evenly along the longest path rather than dropped on one leg.
    built = series(0.0, 0.0, 0.0)
    share = slippage.divide(built, [0.0, 0.0, 0.0], 0.005)
    assert share == pytest.approx([0.005 / 3] * 3)


def test_a_network_that_will_not_solve_still_bounds_every_path():
    # A slot nothing reaches from the source: the Laplacian is singular and the
    # fallback shares in proportion to the weights alone, normalised the same.
    built = route([leg(0, 1, 1000 * UNIT, 1000 * UNIT, fee=1e-4, pool=0),
                   leg(3, 4, 1000 * UNIT, 1000 * UNIT, fee=1e-4, pool=1),
                   leg(1, 2, 1000 * UNIT, 500 * UNIT, fee=20e-4, pool=2)],
                  amount_in=1000 * UNIT, dst_slot=2)
    assert slippage.drops(built, [1.0, 1.0, 1.0], 0.005) is None
    share = slippage.divide(built, [1.0, 2.0, 1.0], 0.005)
    assert slippage.longest(built, share) == pytest.approx(0.005)
    assert all(value >= 0.0 for value in share)


def test_an_empty_route_divides_into_nothing():
    assert slippage.divide(route([], amount_in=0, dst_slot=1), [], 0.005) == []
