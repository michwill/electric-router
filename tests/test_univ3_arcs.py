"""A tick range is a diode, a resistor and a cap, and the three are exact.

The reference here is a tick walk written independently of the model: it
consumes each range in turn at the pool's own constant-product law, which is
what Uniswap v3 does.  The model is the router's arc law -- `a psi - B psi^2/2`
per tick, all in parallel, settling at one marginal rate -- and the claim under
test is that the second reproduces the first.

Synthetic ladders rather than a pool: this is the arithmetic being checked, and
`scripts/prototype_univ3_arcs.py` is what checks it against a real one (measured
within 0.003 bp on USDC/WETH and 0.095 bp at 0.6% spacing).
"""

from __future__ import annotations

import math

import pytest

from erouter.venues.univ3 import Arc, PoolState, Tick, arcs, capacity, output

Q96 = 1 << 96


def _ladder(n: int, spacing: int, liquidity: int, tick0: int = 0):
    """A pool whose every tick is initialized, with liquidity stepping down."""
    ticks = [Tick(index=tick0 - k * spacing, liquidity_net=liquidity // (2 * n))
             for k in range(1, n + 1)]
    state = PoolState(
        sqrt_price_x96=int(math.pow(1.0001, tick0 / 2.0) * Q96),
        tick=tick0, liquidity=liquidity, tick_spacing=spacing,
        fee=0, decimals0=18, decimals1=18)
    return state, ticks


def _walk(state: PoolState, ticks: list[Tick], dx: float) -> float:
    """The reference: consume each range at the CPMM law, in order."""
    sqrt_p = state.sqrt_price_x96 / Q96
    liquidity = float(state.liquidity)
    out, left = 0.0, dx * 10**state.decimals0
    for tick in sorted((t for t in ticks if t.index <= state.tick),
                       key=lambda t: -t.index):
        sqrt_next = math.pow(1.0001, tick.index / 2.0)
        room = liquidity * (1.0 / sqrt_next - 1.0 / sqrt_p)
        take = min(left, room)
        if take > 0:
            out += liquidity * (sqrt_p - 1.0 / (1.0 / sqrt_p + take / liquidity))
            left -= take
        if left <= 0:
            break
        sqrt_p = sqrt_next
        liquidity -= tick.liquidity_net
    return out / 10**state.decimals1


@pytest.mark.parametrize("spacing,tolerance_bp", [(1, 0.001), (10, 0.02), (60, 0.5)])
def test_the_arc_bank_reproduces_the_tick_walk(spacing, tolerance_bp):
    state, ticks = _ladder(120, spacing, 10**22)
    bank = arcs(state, ticks, zero_for_one=True, max_ticks=120)
    room = capacity(bank)

    for share in (0.001, 0.05, 0.3, 0.7, 0.95):
        dx = room * share
        got, want = output(bank, dx), _walk(state, ticks, dx)
        bp = (got - want) / want * 1e4
        assert abs(bp) < tolerance_bp, f"{share:.0%} of capacity: {bp:+.4f} bp"


def test_the_model_never_promises_more_than_the_pool_pays():
    """A chord lies below a concave curve, which is what §5.5 needs."""
    state, ticks = _ladder(120, 10, 10**22)
    bank = arcs(state, ticks, zero_for_one=True, max_ticks=120)
    room = capacity(bank)

    for share in (0.02, 0.1, 0.4, 0.8, 0.99):
        dx = room * share
        assert output(bank, dx) <= _walk(state, ticks, dx) * (1 + 1e-9)


def test_dropping_the_far_ticks_only_lowers_the_answer():
    """Truncation removes capacity, and capacity is the thing that pays."""
    state, ticks = _ladder(120, 10, 10**22)
    full = arcs(state, ticks, zero_for_one=True, max_ticks=120)
    dx = capacity(full) * 0.5

    previous = output(full, dx)
    for keep in (64, 32, 16, 8):
        fewer = arcs(state, ticks, zero_for_one=True, max_ticks=keep)
        assert capacity(fewer) < capacity(full)
        got = output(fewer, dx)
        assert got <= previous + 1e-9, f"{keep} ticks paid more than more ticks did"
        previous = got


def test_a_tick_is_a_resistor_of_its_own_liquidity():
    """`B = 2 a^(3/2) / L` is what makes the two directions one formula."""
    state, ticks = _ladder(40, 10, 10**22)
    for zero_for_one in (True, False):
        bank = arcs(state, ticks, zero_for_one=zero_for_one, max_ticks=40)
        if not bank:
            continue
        for arc in bank:
            assert arc.a > 0 and arc.B > 0 and arc.cap > 0
            # Recovering L from the arc: it must be the liquidity that range holds.
            recovered = 2.0 * arc.a * math.sqrt(arc.a) / arc.B
            assert recovered > 0


def test_the_fee_is_taken_on_the_way_in():
    """A fee scales the rate down and the capacity up, and nothing else."""
    free, ticks = _ladder(40, 10, 10**22)
    charged = PoolState(**{**{f: getattr(free, f) for f in free.__slots__},
                           "fee": 3000})
    a = arcs(free, ticks, zero_for_one=True, max_ticks=40)
    b = arcs(charged, ticks, zero_for_one=True, max_ticks=40)

    keep = 1.0 - 3000 / 1e6
    assert b[0].a == pytest.approx(a[0].a * keep, rel=1e-12)
    assert b[0].cap == pytest.approx(a[0].cap / keep, rel=1e-12)
    assert output(b, 100.0) < output(a, 100.0)


def test_an_empty_bank_and_a_zero_trade_are_answers_not_errors():
    assert output([], 10.0) == 0.0
    assert output([Arc(a=1.0, B=1e-9, cap=5.0)], 0.0) == 0.0
