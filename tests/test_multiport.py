"""One pool, several ports -- the representation and its arithmetic.

The point of an element is that the pool is touched *once*, so the coupling
between its ports is the advancing state rather than a cross-term nobody
modelled.  These tests are about that: an element must equal the legs it
stands for, executed in order, to the wei.
"""

from __future__ import annotations

import pytest

from erouter.core.multiport import LP, MultiPort, MultiPortError, Port, evaluate
from erouter.core.stableswap import StableSwap, StableSwapLP

UNIT = 10**18
POOL = StableSwap(
    balances=(1_000_000 * UNIT, 1_000_000 * UNIT, 1_000_000 * UNIT),
    rates=(UNIT, UNIT, UNIT),
    amp=2000, fee=4_000_000, offpeg_fee_multiplier=0,
    a_precision=1, fee_on_xp=False, admin_fee=5_000_000_000,
)
SUPPLY = 3_000_000 * UNIT
ADDRESS = "0x" + "3c" * 20


def lp():
    return StableSwapLP(pool=POOL, total_supply=SUPPLY)


def element(inputs, outputs, n=3):
    return MultiPort(pool=ADDRESS, n_coins=n, inputs=tuple(inputs),
                     outputs=tuple(outputs))


# --- the port bound ---------------------------------------------------

def test_the_injection_is_what_bounds_the_ports():
    """`#in + #out <= N` is not a separate rule -- it falls out.

    Each coin holds at most one port and every port names a coin in range,
    so there is nowhere for an `N+1`th coin-port to go.  Asking for one on a
    two-coin pool fails as an out-of-range coin, which is the same refusal
    arriving earlier.
    """
    with pytest.raises(MultiPortError, match="out of range"):
        element([Port(0, 10_000)],
                [Port(1, 5_000), Port(2, 5_000)], n=2)


def test_a_coin_may_not_sit_on_both_sides():
    """That is a wash, not an element."""
    with pytest.raises(MultiPortError, match="at most one port"):
        element([Port(0, 10_000)], [Port(0, 5_000), Port(1, 5_000)])


def test_the_lp_port_consumes_no_coin_slot():
    """`add_liquidity` of both coins of a 2-coin pool is a real operation."""
    made = element([Port(0, 5_000), Port(1, 5_000)], [Port(LP, 10_000)], n=2)
    assert made.ports == 3 > made.n_coins


def test_weights_must_be_a_split():
    with pytest.raises(MultiPortError, match="sum to"):
        element([Port(0, 10_000)], [Port(1, 5_000), Port(2, 2_000)])


# --- the arithmetic ---------------------------------------------------

def test_one_in_two_out_equals_the_two_swaps_in_order():
    """The second leg must see what the first one left."""
    made = element([Port(2, 10_000)], [Port(0, 2_500), Port(1, 7_500)])
    dx = 400_000 * UNIT
    outs, _, _ = evaluate(made, POOL, lp(), dx)

    first, after = POOL.exchange(2, 0, dx // 4)
    second, _ = after.exchange(2, 1, dx - dx // 4)
    assert outs == [first, second]


def test_the_ports_are_not_independent():
    """Priced against the block state instead, the answer is different.

    This is the whole reason the element exists: two arcs would each quote
    against the untouched pool and neither would be what executes.
    """
    made = element([Port(2, 10_000)], [Port(0, 5_000), Port(1, 5_000)])
    dx = 600_000 * UNIT
    outs, _, _ = evaluate(made, POOL, lp(), dx)

    naive = [POOL.exchange(2, 0, dx // 2)[0], POOL.exchange(2, 1, dx // 2)[0]]
    assert outs[0] == naive[0], "the first leg is unaffected"
    assert outs[1] != naive[1], (
        "the second leg saw the same pool as the first, so nothing advanced")
    assert outs[1] < naive[1], "the first leg should have made the second worse"


def test_a_coin_and_an_lp_port_together():
    """`XDAI -> 3Crv + USDC.e`, which is the case this was built for."""
    made = element([Port(2, 10_000)], [Port(LP, 5_000), Port(0, 5_000)])
    dx = 200_000 * UNIT
    outs, _, after_lp = evaluate(made, POOL, lp(), dx)

    minted, grown = lp().add_liquidity([0, 0, dx // 2])
    swapped, _ = grown.pool.exchange(2, 0, dx // 2)
    assert outs == [minted, swapped]
    assert after_lp.total_supply > SUPPLY


def test_many_in_one_out_is_a_deposit():
    made = element([Port(0, 2_500), Port(1, 7_500)], [Port(LP, 10_000)])
    dx = 400_000 * UNIT
    outs, _, _ = evaluate(made, POOL, lp(), dx)

    minted, _ = lp().add_liquidity([dx // 4, dx - dx // 4, 0])
    assert outs == [minted]


def test_many_in_many_out_is_refused_rather_than_guessed():
    """It needs a pairing rule between the sides, and there isn't one yet."""
    made = element([Port(0, 5_000), Port(1, 5_000)], [Port(2, 5_000), Port(LP, 5_000)])
    with pytest.raises(MultiPortError, match="pairing rule"):
        evaluate(made, POOL, lp(), 100 * UNIT)
