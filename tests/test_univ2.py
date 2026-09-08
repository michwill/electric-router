"""A v2 pair, priced two ways: exactly, and by the arc the solver gets.

`output` reproduces the contract, so it is checked against the contract's own
arithmetic rather than against a rearrangement of itself.  The arc law is a
second-order truncation of that, so it is checked for *agreement where it
claims to hold* and for degrading the way a truncation should past there --
which is what `cap` exists to bound.
"""

from __future__ import annotations

import math

import pytest

from erouter.core.types import ArcKind
from erouter.venues import univ2

WETH = "0x" + "ee" * 20
USDC = "0x" + "cc" * 20


class Nodes:
    """Canonical rates, so `rescale` is the identity and the fit shows through."""

    def __init__(self, merged=False):
        self._merged = merged

    def has(self, token):
        return token in (WETH, USDC)

    def node(self, token):
        if self._merged:
            return 0
        return 0 if token == WETH else 1

    def rate(self, _token):
        return 1.0


def pair(x=1_000 * 10**18, y=2_500_000 * 10**6, fee=30):
    """1,000 WETH against 2.5M USDC: $2,500 a WETH, both sides real."""
    return univ2.PairState(reserve0=x, reserve1=y, decimals0=18, decimals1=6,
                           fee_bps=fee)


def contract_get_amount_out(dx, x, y, fee_bps):
    """`UniswapV2Library.getAmountOut`, transcribed.

    Written out rather than imported so the test states the reference instead of
    agreeing with the implementation by construction.
    """
    dx_with_fee = dx * (10_000 - fee_bps)
    return (dx_with_fee * y) // (x * 10_000 + dx_with_fee)


@pytest.mark.parametrize("dx", [1, 10**15, 10**18, 10 * 10**18, 100 * 10**18])
@pytest.mark.parametrize("fee", [30, 25, 20, 0])
def test_output_is_the_contract_to_the_wei(dx, fee):
    state = pair(fee=fee)
    assert univ2.output(state, True, dx) == contract_get_amount_out(
        dx, state.reserve0, state.reserve1, fee)
    assert univ2.output(state, False, dx) == contract_get_amount_out(
        dx, state.reserve1, state.reserve0, fee)


def test_an_empty_side_pays_nothing_rather_than_quoting_a_price():
    """The failure `find_broken_pools` catches on Curve, in its v2 form."""
    assert not univ2.PairState(0, 10**18, 18, 18).live
    assert univ2.output(univ2.PairState(0, 10**18, 18, 18), True, 10**18) == 0
    assert univ2.arc_params(univ2.PairState(10**18, 0, 18, 18), True) == (0.0, 0.0, 0.0)


def test_the_arc_opens_at_the_pools_own_marginal_rate():
    """`a = k y / x` in human units: 2,500 USDC a WETH, less the fee."""
    a, _b, _cap = univ2.arc_params(pair(), True)
    assert a == pytest.approx(2_500 * 0.997, rel=1e-12)


def test_the_arc_law_tracks_the_curve_it_was_expanded_from():
    """Second order, so it holds where the trade is small against the pool.

    At the cap the truncation is still within a basis point of the exact curve;
    that is the claim `MAX_SHARE` is chosen to make.
    """
    state = pair()
    a, b, cap = univ2.arc_params(state, True)
    for share in (0.001, 0.01, 0.05, 0.1):
        dx_human = 1_000 * share
        modelled = a * dx_human - b * dx_human * dx_human / 2
        truth = univ2.output(state, True, int(dx_human * 10**18)) / 10**6
        assert abs(modelled / truth - 1) < 1e-2, f"at {share:.1%} of the pool"
    assert cap == pytest.approx(1_000 * univ2.MAX_SHARE)


def test_past_the_cap_the_quadratic_turns_over_and_pays_less():
    """Why there is a cap at all, stated as the failure it prevents.

    The truncation peaks at `dx = a / B = x / 2k` and falls after it, so a big
    enough trade is modelled as paying *less* than a smaller one -- a solver
    reading that routes around a pool it has just emptied.  At a whole reserve
    it claims `k y (1 - k)` against the pool's `k y / (1 + k)`: 0.6% of the
    truth, not an overshoot.
    """
    state = pair()
    a, b, cap = univ2.arc_params(state, True)
    vertex = a / b
    assert cap < vertex / 4, "the cap must sit well below the turnover"

    def modelled(dx):
        return a * dx - b * dx * dx / 2

    assert modelled(2 * vertex) < modelled(vertex)
    truth = univ2.output(state, True, 1_000 * 10**18) / 10**6
    assert modelled(1_000.0) / truth < 0.01


def test_both_directions_become_arcs_with_the_venue_on_them():
    arcs = univ2.arcs_for("0x" + "ab" * 20, pair(), WETH, USDC, Nodes())
    assert len(arcs) == 2
    assert {a.kind for a in arcs} == {ArcKind.SWAP_UNIV2}
    assert {a.venue for a in arcs} == {"uniswap v2"}
    assert all(not a.parallel for a in arcs), "one arc is the whole direction"
    assert all(math.isfinite(a.cap) and a.cap > 0 for a in arcs)
    assert {(a.i, a.j) for a in arcs} == {(0, 1), (1, 0)}


def test_a_pair_whose_coins_share_a_node_gives_nothing():
    """Both sides of one merged node is not a trade -- as in `univ3.arcs_for`."""
    assert univ2.arcs_for("0x" + "ab" * 20, pair(), WETH, USDC,
                          Nodes(merged=True)) == []


def test_a_token_the_frame_cannot_price_gives_nothing():
    class Half(Nodes):
        def has(self, token):
            return token == WETH

    assert univ2.arcs_for("0x" + "ab" * 20, pair(), WETH, USDC, Half()) == []


def test_the_fee_travels_with_the_pool_rather_than_being_assumed():
    """Sushi and Pancake run the same bytecode at 25 bp.

    A pool that stated its fee must be priced at it, or every fork is quoted as
    Uniswap and the error is one-sided.
    """
    thirty = univ2.arc_params(pair(fee=30), True)[0]
    twenty_five = univ2.arc_params(pair(fee=25), True)[0]
    assert twenty_five > thirty
    assert univ2.output(pair(fee=25), True, 10**18) > \
        univ2.output(pair(fee=30), True, 10**18)


def test_decimals_are_carried_into_the_rate():
    """18 in, 6 out: the raw ratio is 1e12 times the human one."""
    a, _b, _cap = univ2.arc_params(pair(), True)
    raw = univ2.output(pair(), True, 10**18)
    assert raw / 10**6 == pytest.approx(a, rel=1e-3)


def test_a_short_answer_is_an_error_and_not_an_empty_pool():
    """`eth_call` to an address with no code returns `0x`.

    Reading that as a pool holding nothing says the pair is dead when the truth
    is that nothing was asked -- the same conflation as reading the storage
    layout, which this function already refuses.
    """
    from erouter.venues.univ2_chain import decode_reserves

    with pytest.raises(ValueError):
        decode_reserves(b"")
    with pytest.raises(ValueError):
        decode_reserves(bytes(63))
    assert decode_reserves(bytes(96)) == (0, 0), "a real answer of zeros is zeros"
