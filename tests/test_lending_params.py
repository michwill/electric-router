"""The lending pools: stableswap with wrapped tokens.

`cDAI/cUSDC` is the first Curve pool ever deployed and still quotes.  Nothing
about its invariant is unusual -- what is unusual is where its rates come from,
and that its coin list runs past its balances.
"""

from __future__ import annotations

import pytest

from erouter.core.stableswap import PRECISION, StableSwap
from erouter.dev.lending_params import LENDING_PRECISION, VARIANTS, candidates


class Coin:
    def __init__(self, decimals, address="0x" + "11" * 20, symbol="X"):
        self.decimals = decimals
        self.address = address
        self.symbol = symbol


class Pool:
    def __init__(self, coins, balances, name="pool"):
        self.coins = coins
        self.balances = balances
        self.name = name
        self.address = "0x" + "22" * 20


def test_a_wrapped_pool_is_recognised_by_its_trailing_zeros():
    """cDAI/cUSDC lists four coins and holds two.

    The underlying pair is listed because `exchange_underlying` trades it, and
    reported with a zero balance.  `all(pool.balances)` reads that as an empty
    pool, which is why the first Curve pool was never offered to any builder.
    """
    pool = Pool([Coin(8), Coin(8), Coin(18), Coin(6)],
                [833963285690023, 1412413376521133, 0, 0])
    got = candidates([pool])
    assert got and got[0][1] == 2, "should see two held coins of four listed"


def test_a_genuinely_empty_pool_is_not_a_lending_pool():
    pool = Pool([Coin(18), Coin(18)], [0, 0])
    assert candidates([pool]) == []


def test_a_full_pool_is_not_a_lending_pool():
    pool = Pool([Coin(18), Coin(18)], [10**21, 10**21])
    assert candidates([pool]) == []


def test_the_variants_cover_both_deployed_shapes():
    """The generation is not one implementation.

    The Compound pools take a static fee in token space and round the output
    down by a wei; the Aave pool takes a dynamic fee in `xp` space and keeps
    it.  Both shapes have to be reachable, or one family is unmodellable.
    """
    assert (False, True) in VARIANTS, "Compound: token-space fee, minus one wei"
    assert (True, False) in VARIANTS, "Aave: dynamic xp fee, keeps the wei"


def test_keeping_the_wei_is_worth_exactly_one_wei():
    """`subtract_one` is the whole difference between the two roundings."""
    shaped = dict(balances=(10**24, 10**24), rates=(PRECISION, PRECISION),
                  amp=10_000, fee=4_000_000, a_precision=100)
    keeps = StableSwap(subtract_one=False, **shaped)
    rounds = StableSwap(subtract_one=True, **shaped)
    dx = 10**21
    assert keeps.get_dy(0, 1, dx) - rounds.get_dy(0, 1, dx) == 1


def test_a_coin_that_is_not_lent_is_worth_one_underlying():
    """`LENDING_PRECISION` is the rate of a coin with no wrapper."""
    assert LENDING_PRECISION == 10**18
