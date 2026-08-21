"""The fee a trade pays, read back out of the pool's own model.

A dynamic fee is a function of the trade.  `gamma_live` measures the marginal
one -- the fee on a trade so small it moves nothing -- which on a cryptoswap
pool is `mid_fee` and on a big trade is nowhere near what gets charged.  These
check that what comes back is the pool's own charge and not something adjacent
to it.
"""

from __future__ import annotations

import math

import pytest

from erouter.core.poolfee import charged_fee, fee_free
from erouter.core.stableswap import StableSwap
from erouter.core.tricrypto import Tricrypto
from erouter.core.twocrypto import Twocrypto
from test_tricrypto import STATE as TRICRYPTO_STATE
from test_twocrypto import YB_WETH

FIXED = StableSwap(balances=(10**24,) * 3, rates=(10**18,) * 3, amp=2000 * 100,
                   fee=3 * 10**6, a_precision=100)
NG = StableSwap(balances=(10**24, 10**24), rates=(10**18,) * 2, amp=200 * 100,
                fee=10**7, offpeg_fee_multiplier=2 * 10**10, a_precision=100)


def test_a_fixed_fee_pool_reads_its_fee_at_every_size():
    for dx in (10**18, 10**21, 10**23, 5 * 10**23):
        assert charged_fee(FIXED, 0, 1, dx) == pytest.approx(3e-4, rel=1e-3)


def test_a_dynamic_fee_climbs_with_the_trade():
    """The whole point: the marginal fee is not the one being paid."""
    small = charged_fee(NG, 0, 1, 10**18)
    large = charged_fee(NG, 0, 1, 9 * 10**23)
    assert small == pytest.approx(1e-3, rel=1e-3)
    assert large > small * 1.1


@pytest.mark.parametrize("family,model,pair", [
    ("tricrypto", Tricrypto(**TRICRYPTO_STATE), (0, 2)),
    ("twocrypto", Twocrypto(**YB_WETH), (0, 1)),
])
def test_the_fee_stays_inside_the_pool_s_own_bracket(family, model, pair):
    """`mid_fee` and `out_fee` are what the pool says its fee can be.

    A value outside them would mean the twin is measuring impact, or rounding,
    or anything but the fee.
    """
    i, j = pair
    low = min(model.mid_fee, model.out_fee) / 1e10
    high = max(model.mid_fee, model.out_fee) / 1e10
    reserve = model.balances[i]
    seen = []
    for theta in (1e-6, 1e-3, 1e-2, 5e-2):
        fee = charged_fee(model, i, j, max(1, int(reserve * theta)))
        if fee is None:
            continue
        seen.append(fee)
        assert low - 1e-5 <= fee <= high + 1e-5, (
            f"{family}: {fee * 1e4:.2f} bp is outside [{low * 1e4:.2f}, "
            f"{high * 1e4:.2f}] bp")
    assert len(seen) >= 2
    # It has to *move* with the size -- that is what makes it dynamic -- but
    # not in a fixed direction: the fee follows how balanced the trade leaves
    # the pool, so a trade into the short side walks it back toward `mid_fee`.
    assert max(seen) > min(seen), f"{family}: the fee ignored the trade size"


def test_a_model_with_no_fee_fields_is_refused_rather_than_guessed():
    class Opaque:
        def get_dy(self, i, j, dx):
            return dx

    assert fee_free(Opaque()) is None
    assert charged_fee(Opaque(), 0, 1, 10**18) is None


def test_a_model_that_will_not_quote_is_refused():
    empty = StableSwap(balances=(0, 0), rates=(10**18,) * 2, amp=100, fee=10**6)
    assert charged_fee(empty, 0, 1, 10**18) is None


def test_zero_size_has_no_fee_to_read():
    assert charged_fee(FIXED, 0, 1, 0) is None


def test_the_twin_keeps_everything_that_is_not_a_fee():
    """`fee_gamma` shapes the curve and `admin_fee` is the DAO's cut; zeroing
    either would change the pool rather than disarm it."""
    twin = fee_free(Twocrypto(**YB_WETH))
    original = Twocrypto(**YB_WETH)
    assert twin.fee_gamma == original.fee_gamma
    assert twin.balances == original.balances
    assert twin.mid_fee == 0 and twin.out_fee == 0


# --------------------------------------------------------------- on a leg


def test_a_leg_prefers_the_fee_at_its_own_size():
    from erouter.core.realize import RealizedLeg
    from erouter.core.routecall import leg_fee
    from erouter.core.types import ArcKind, Leg

    def leg(**kw):
        return RealizedLeg(
            leg=Leg(target="0x" + "aa" * 20, kind=ArcKind.SWAP_CRYPTO),
            kind=ArcKind.SWAP_CRYPTO, target="0x" + "aa" * 20,
            token_in="0x" + "01" * 20, token_out="0x" + "02" * 20,
            amount_in=10**18, amount_out=10**18, **kw)

    assert leg_fee(leg(fee_frac=0.003, gamma_live=0.9999)) == pytest.approx(0.003)
    assert leg_fee(leg(gamma_live=0.9999)) == pytest.approx(1e-4)
    assert leg_fee(leg()) == 0.0
    assert leg_fee(leg(fee_frac=math.nan, gamma_live=math.nan)) == 0.0


class Client:
    """A quoter that answers every leg and prices only real pools."""

    def __init__(self, value=2 * 10**18):
        self.value = value
        self.probed: list = []
        self.asked: list = []

    def probe(self, probes):
        from erouter.core.quoter import Quote
        from erouter.core.transport import Status

        self.probed.extend(probes)
        return [Quote(Status.VALUE, self.value) for _ in probes]

    def fee_at(self, pool, kind, i, j, dx):
        self.asked.append((pool, kind, dx))
        return 0.0004


def _route():
    from erouter.core.realize import RealizedLeg, RealizedRoute
    from erouter.core.types import ArcKind, Leg

    def leg(kind, slot):
        return RealizedLeg(
            leg=Leg(target="0x" + "aa" * 20, kind=kind, src_slot=slot,
                    dst_slot=slot + 1),
            kind=kind, target="0x" + "aa" * 20, token_in="0x" + "01" * 20,
            token_out="0x" + "02" * 20, amount_in=10**18, amount_out=10**18)

    return RealizedRoute(legs=[leg(ArcKind.SWAP_STABLE, 0),
                               leg(ArcKind.WRAP_NATIVE, 1)])


def test_the_pipeline_pass_prices_every_leg_and_charges_only_pools():
    from erouter.core.pipeline import price_legs

    route, client = _route(), Client()
    assert price_legs(route, client) == 2
    assert [leg.verified_out for leg in route.legs] == [2 * 10**18] * 2
    assert route.legs[0].fee_frac == pytest.approx(0.0004)
    # A wrap charges nothing and has no pool to ask.
    assert math.isnan(route.legs[1].fee_frac)
    assert len(client.asked) == 1
    assert len(client.probed) == 2


def test_the_leg_s_own_quote_is_what_a_bound_is_set_against():
    """The route's modelled figure is a choice; this is a measurement."""
    from erouter.core.pipeline import price_legs
    from erouter.core.routecall import leg_out, min_rates

    route, client = _route(), Client()
    modelled, _ = min_rates(route)
    price_legs(route, client)
    measured, _ = min_rates(route)
    assert leg_out(route.legs[0]) == 2 * 10**18
    assert measured[0] > modelled[0] * 1.9


def test_a_route_with_nothing_to_price_asks_nothing():
    from erouter.core.pipeline import price_legs
    from erouter.core.realize import RealizedRoute

    client = Client()
    assert price_legs(RealizedRoute(legs=[]), client) == 0
    assert client.probed == []
