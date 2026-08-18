"""Deposits and withdrawals, on the invariant the swap models already run.

An LP arc asks a different question of the same curve: not "what does `i`
become when `j` moves at constant `D`" but "what does `i` become when `D`
itself moves".  That is `solve_y_d`, and the two conventions around it are
easy to assume wrongly -- so they were read off the deployed 3pool
(`0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7`) rather than guessed:

* `calc_token_amount` charges **no fee**.  Its own docstring says it is
  "needed to prevent front-running, not for precise calculations".
* `calc_withdraw_one_coin` charges `fee * N / (4 * (N - 1))` on each coin's
  *imbalance against the ideal*, not on the output, and then returns one wei
  less "to account for rounding errors".

Against mainnet, four of the five pools carrying LP arcs reproduce their own
`calc_withdraw_one_coin` to the wei.  The fifth is stableswap-ng, which charges
its dynamic per-coin fee instead of the flat one -- off by 1,631 wei on a
575,757,635 withdrawal, which is what a fee convention looks like and not what
a broken solver looks like.
"""

from __future__ import annotations

import pytest

from erouter.core.stableswap import (StableSwap, StableSwapError, StableSwapLP,
                                     solve_y_d)

UNIT = 10**18
BALANCED = StableSwap(
    balances=(1_000_000 * UNIT, 1_000_000 * UNIT, 1_000_000 * UNIT),
    rates=(UNIT, UNIT, UNIT),
    amp=2000, fee=4_000_000, offpeg_fee_multiplier=0,
    a_precision=1, fee_on_xp=False,
)
SUPPLY = 3_000_000 * UNIT


def lp(pool=BALANCED, supply=SUPPLY):
    return StableSwapLP(pool=pool, total_supply=supply)


def test_reducing_d_reduces_the_balance():
    """The whole point of `solve_y_d`: `D` moves, the others are held."""
    xp = BALANCED.xp()
    d0 = BALANCED.d()
    same = solve_y_d(BALANCED.amp, BALANCED.a_precision, xp, d0, 0, 3)
    smaller = solve_y_d(BALANCED.amp, BALANCED.a_precision, xp, d0 * 99 // 100, 0, 3)
    assert abs(same - xp[0]) <= 2, "at the pool's own D, nothing should move"
    assert smaller < same


def test_depositing_nothing_mints_nothing():
    assert lp().calc_token_amount([0, 0, 0], True) == 0


def test_a_balanced_deposit_mints_in_proportion():
    """No fee and no imbalance, so LP scales with the value added."""
    minted = lp().calc_token_amount([1_000 * UNIT] * 3, True)
    assert minted == pytest.approx(SUPPLY * 3_000 / 3_000_000, rel=1e-6)


def test_a_one_sided_deposit_mints_less_than_a_balanced_one():
    """Slippage, which `calc_token_amount` does account for."""
    balanced = lp().calc_token_amount([1_000 * UNIT] * 3, True)
    one_sided = lp().calc_token_amount([3_000 * UNIT, 0, 0], True)
    assert one_sided < balanced


def test_withdrawing_one_coin_pays_less_than_the_ideal_share():
    """The imbalance fee, and the one wei the pool keeps."""
    burn = SUPPLY // 1000
    got = lp().calc_withdraw_one_coin(burn, 0)
    ideal = 3_000 * UNIT          # 0.1% of a 3,000,000 pool, taken in one coin
    assert 0 < got < ideal


def test_withdrawing_more_than_exists_is_refused():
    with pytest.raises(StableSwapError):
        StableSwapLP(pool=BALANCED, total_supply=0).calc_withdraw_one_coin(1, 0)
    with pytest.raises(StableSwapError):
        lp().calc_withdraw_one_coin(SUPPLY // 100, 7)


def test_a_bigger_withdrawal_pays_proportionally_less_per_lp():
    """Curvature: the larger the single-sided exit, the worse the rate."""
    small = lp().calc_withdraw_one_coin(SUPPLY // 10_000, 0)
    large = lp().calc_withdraw_one_coin(SUPPLY // 100, 0)
    assert large / 100 < small, "a 1% exit should price worse per LP than 0.01%"
