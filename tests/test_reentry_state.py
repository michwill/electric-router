"""What a reused pool looks like to the leg after it.

Two quantities move on different legs, and both have to be carried:

* a **swap** moves the balances and leaves `total_supply` alone;
* a **deposit** mints, so it moves both.

Reading each from wherever it happened to be lying around is the bug this
pins.  The route that found it was 3pool three ways -- swap USDT->DAI,
deposit USDT->3Crv, withdraw 3Crv->DAI -- where the withdrawal burned
1.92e24 fresh LP against the *pre-deposit* supply of 153.4M, a 1.25% larger
share of `D` than it owns.  It reads as the deposit-and-withdraw round trip
being free, which is exactly the shape of a route worth distrusting.
"""

from __future__ import annotations

import pytest

from erouter.chain.exact_probe import ExactQuoterClient
from erouter.core.stableswap import StableSwap, StableSwapLP
from erouter.core.types import ArcKind, Leg

UNIT = 10**18
POOL = StableSwap(
    balances=(1_000_000 * UNIT, 1_000_000 * UNIT, 1_000_000 * UNIT),
    rates=(UNIT, UNIT, UNIT),
    amp=2000, fee=4_000_000, offpeg_fee_multiplier=0,
    a_precision=1, fee_on_xp=False, admin_fee=5_000_000_000,
)
SUPPLY = 3_000_000 * UNIT
POOL_ADDRESS = "0x" + "3c" * 20


class FakeSet:
    """The shape `ExactQuoterClient` reads: the two directions, each with a getter.

    Deposits and withdrawals are admitted separately by `build_exact_lp` -- a
    pool can reproduce one and not the other -- so a double that serves only
    `get` would let a caller drop the deposit path without any test noticing.
    """

    def __init__(self, model, deposits=True):
        self.by_pool = {POOL_ADDRESS: model}
        self.deposits = {POOL_ADDRESS: model} if deposits else {}

    def get(self, pool):
        return self.by_pool.get(pool.lower())

    def get_deposit(self, pool):
        return self.deposits.get(pool.lower())


def client() -> ExactQuoterClient:
    """An `ExactQuoterClient` that knows one pool and talks to nothing."""
    lp = StableSwapLP(pool=POOL, total_supply=SUPPLY)
    return ExactQuoterClient(None, FakeSet(POOL), lp=FakeSet(lp))


def leg(kind, i, j):
    return Leg(target=POOL_ADDRESS, kind=kind, i=i, j=j, n=3,
               src_slot=0, dst_slot=1, bps=0)


DEPOSIT = leg(ArcKind.DEPOSIT_FIXED, 2, 0)
WITHDRAW = leg(ArcKind.WITHDRAW_STABLE, 0, 0)
SWAP = leg(ArcKind.SWAP_STABLE, 2, 0)


def test_a_withdrawal_burns_against_the_supply_the_deposit_left():
    """The minted LP has to be in the supply it is then burned against."""
    quote = client()._stateful_leg([DEPOSIT, WITHDRAW])
    minted = quote(DEPOSIT, 100_000 * UNIT)
    got = quote(WITHDRAW, minted)

    grown = StableSwapLP(pool=POOL, total_supply=SUPPLY)
    _, after = grown.add_liquidity([0, 0, 100_000 * UNIT])
    assert after.total_supply > SUPPLY, "the deposit minted nothing"
    want = after.calc_withdraw_one_coin(minted, 0)
    assert got == pytest.approx(want, rel=1e-9), (
        "the withdrawal did not see the deposit's mint")

    # And the pre-deposit supply -- the figure this used to use -- is a
    # materially different, larger answer.
    stale = StableSwapLP(pool=after.pool, total_supply=SUPPLY)
    assert stale.calc_withdraw_one_coin(minted, 0) > want * 1.001, (
        "the bug and the fix are indistinguishable on these numbers")


def test_a_swap_prices_against_the_pool_the_withdrawal_left():
    """Burn first and swap after: the order FRAX->USDC $10M's circuit route
    took through 3pool, which the walk could not price until it could burn."""
    quote = client()._stateful_leg([WITHDRAW, SWAP])
    paid = quote(WITHDRAW, 100_000 * UNIT)
    got = quote(SWAP, 200_000 * UNIT)

    want_paid, after = StableSwapLP(pool=POOL, total_supply=SUPPLY).remove_liquidity_one_coin(
        100_000 * UNIT, 0)
    assert paid == want_paid
    assert got == pytest.approx(after.pool.get_dy(2, 0, 200_000 * UNIT), rel=1e-9)
    assert got != pytest.approx(POOL.get_dy(2, 0, 200_000 * UNIT), rel=1e-6), (
        "the burn moved nothing")


def test_a_pool_the_walk_can_burn_through_is_listed_so():
    from erouter.core.candidates import BURNS

    assert client().reentrant_pools == {POOL_ADDRESS, POOL_ADDRESS + BURNS}


def test_a_deposit_prices_into_the_balances_the_swap_left():
    """A swap moves the balances the deposit is then imbalancing against."""
    quote = client()._stateful_leg([SWAP, DEPOSIT, WITHDRAW])
    dy = quote(SWAP, 200_000 * UNIT)
    assert dy > 0
    minted = quote(DEPOSIT, 100_000 * UNIT)

    _, swapped = POOL.exchange(2, 0, 200_000 * UNIT)
    want, _ = StableSwapLP(pool=swapped,
                           total_supply=SUPPLY).add_liquidity([0, 0, 100_000 * UNIT])
    assert minted == want, "the deposit priced into pre-swap balances"


def test_a_swap_does_not_move_the_supply():
    """Only the balances -- a swap mints and burns nothing.

    Read through the deposit that follows it: the mint is `supply * dD / D0`,
    so a supply the swap had touched would show up here.
    """
    quote = client()._stateful_leg([SWAP, DEPOSIT, WITHDRAW])
    quote(SWAP, 200_000 * UNIT)
    minted = quote(DEPOSIT, 100_000 * UNIT)

    _, swapped = POOL.exchange(2, 0, 200_000 * UNIT)
    want, _ = StableSwapLP(pool=swapped,
                           total_supply=SUPPLY).add_liquidity([0, 0, 100_000 * UNIT])
    assert minted == want


class Chain:
    """The chain as `ExactQuoterClient` sees it: one probe per leg, and whole
    routes it must not be handed."""

    def __init__(self):
        self.asked = []

    def probe(self, probes):
        from erouter.core.quoter import Quote
        from erouter.core.transport import Status

        self.asked.extend(probes)
        return [Quote(Status.VALUE, 2 * p.dx) for p in probes]

    def quote_routes(self, *_):
        raise AssertionError("a route the chain cannot price went to it whole")


OTHER = "0x" + "0f" * 20


def test_a_route_with_a_uniswap_leg_asks_the_chain_one_leg_at_a_time():
    """The chain knows no Uniswap leg, so a route with one cannot go there
    whole; a circuit route through TricryptoUSDC -- no model -- and v3 quoted 0
    and lost WETH->USDC $10M 2,693 bp.  Walk it, and ask the chain
    only about the leg nothing here models."""
    chain = Chain()
    lp = StableSwapLP(pool=POOL, total_supply=SUPPLY)
    exact = ExactQuoterClient(chain, FakeSet(POOL), lp=FakeSet(lp))
    real = exact._quote_leg
    exact._quote_leg = lambda leg, dx: 3 * dx if leg.kind is ArcKind.SWAP_UNIV3 else real(leg, dx)
    legs = [Leg(target=OTHER, kind=ArcKind.SWAP_STABLE, i=0, j=1, n=2,
                src_slot=0, dst_slot=1, bps=0),
            Leg(target="0x" + "33" * 20, kind=ArcKind.SWAP_UNIV3, i=0, j=1, n=2,
                src_slot=1, dst_slot=2, bps=0)]
    assert exact.quote_routes([legs], [1000], [2]) == [6000]
    assert [(p.pool, p.dx) for p in chain.asked] == [(OTHER, 1000)]


def test_a_pool_entered_twice_still_needs_a_model():
    chain = Chain()
    exact = ExactQuoterClient(chain, FakeSet(POOL), lp=None)
    legs = [Leg(target=OTHER, kind=ArcKind.SWAP_STABLE, i=0, j=1, n=3,
                src_slot=0, dst_slot=1, bps=0),
            Leg(target=OTHER, kind=ArcKind.SWAP_STABLE, i=1, j=2, n=3,
                src_slot=1, dst_slot=2, bps=0)]
    assert exact.quote_routes([legs], [1000], [2]) == [0]
    assert not chain.asked
