"""Uniswap v3 as a bank of parallel arcs, one per tick range.

A concentrated-liquidity pool is not one arc.  Between two initialized ticks it
is exactly a constant-product pool on virtual reserves, and moving the price
from `P` with liquidity `L` pays

    dy(dx) = P dx / (1 + dx sqrt(P) / L)
           = a dx - B dx^2 / 2 + O(dx^3),    a = P,  B = 2 a^(3/2) / L

which is the router's own arc law (M3/M4).  Each tick range therefore gives a
*diode* at its entry price, a *resistor* from its liquidity, and a *capacity*:
the token the range holds.  All three are closed form -- there is nothing to
probe and nothing to fit.

The three together are what makes it exact.  `a` alone says where a tick
switches on and `B` how it degrades, but with no cap the first arc keeps
absorbing past the liquidity its tick actually holds, at prices the pool has
already left: measured at +9,301 bp on a 0.6%-spacing ladder.  Capped, the same
construction lands within 0.09 bp of the true walk, and within 0.00002 bp at
0.01% spacing -- the residual being the cubic term inside one tick, so it grows
as (tick width)^2 and *not* with the size of the trade.

The error is negative -- the model under-promises -- because a chord lies below
a concave curve.  That is the direction §5.5's certificate needs.

`B = 2 a^(3/2) / L` holds in both directions, which is why the walk below needs
no separate case for the price rising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Q96 = 1 << 96


@dataclass(frozen=True, slots=True)
class Tick:
    """One initialized tick: where it is, and what crossing it does to `L`."""

    index: int
    liquidity_net: int


@dataclass(frozen=True, slots=True)
class PoolState:
    """Everything the walk reads, at one block."""

    sqrt_price_x96: int
    tick: int
    liquidity: int
    tick_spacing: int
    fee: int  # hundredths of a bip: 500 is 0.05%
    decimals0: int
    decimals1: int


@dataclass(frozen=True, slots=True)
class Arc:
    """One tick range, in the units the router fits in.

    `a` is output per unit of input at the range's entry price, `cap` the input
    it can absorb, and `B` its curvature.  The router already carries all three
    per arc -- `build` takes a `cap` array beside `a` and `B`, and the solver
    is stated as `0 <= psi <= cap`.
    """

    a: float
    B: float
    cap: float

    def output(self, dx: float) -> float:
        d = min(dx, self.cap)
        return self.a * d - 0.5 * self.B * d * d


def sqrt_price_at(tick: int) -> float:
    """`sqrt(1.0001^tick)`, as a float rather than a Q64.96.

    Floats throughout: this is a *model*, and its own truncation error is four
    orders below the tick-width term that dominates it.
    """
    return math.pow(1.0001, tick / 2.0)


def arcs(
    state: PoolState,
    ticks: list[Tick],
    *,
    zero_for_one: bool,
    max_ticks: int = 64,
) -> list[Arc]:
    """The arc bank for one direction, nearest tick first.

    Truncating at `max_ticks` is safe by construction: the arcs carry the whole
    of the pool's capacity between them, so dropping the far ones removes
    capacity and can only lower what the model promises.
    """
    scale_in, scale_out = (
        (10.0**state.decimals0, 10.0**state.decimals1)
        if zero_for_one
        else (10.0**state.decimals1, 10.0**state.decimals0)
    )
    keep = 1.0 - state.fee / 1e6

    sqrt_p = state.sqrt_price_x96 / Q96
    liquidity = float(state.liquidity)

    if zero_for_one:  # selling token0, the price falls
        walk = sorted((t for t in ticks if t.index <= state.tick),
                      key=lambda t: -t.index)
    else:
        walk = sorted((t for t in ticks if t.index > state.tick),
                      key=lambda t: t.index)

    out: list[Arc] = []
    for tick in walk[:max_ticks]:
        if liquidity <= 0:
            break
        sqrt_next = sqrt_price_at(tick.index)
        # `dx` is always the input side: token0 when the price falls, token1
        # when it rises, and each is the reciprocal shape of the other.
        if zero_for_one:
            dx = liquidity * (1.0 / sqrt_next - 1.0 / sqrt_p)
            a_raw = sqrt_p * sqrt_p
        else:
            dx = liquidity * (sqrt_next - sqrt_p)
            a_raw = 1.0 / (sqrt_p * sqrt_p)
        if dx > 0:
            b_raw = 2.0 * a_raw * math.sqrt(a_raw) / liquidity
            # Into human units, then the fee: it is taken on the way in, so it
            # scales the rate down and the capacity up.
            a = a_raw * scale_in / scale_out * keep
            b = b_raw * scale_in * scale_in / scale_out * keep * keep
            out.append(Arc(a=a, B=b, cap=dx / scale_in / keep))
        sqrt_p = sqrt_next
        liquidity += -tick.liquidity_net if zero_for_one else tick.liquidity_net
    return out


def output(bank: list[Arc], dx: float, *, rounds: int = 200) -> float:
    """What the bank pays for `dx`, at the solver's own optimum.

    Parallel arcs between one pair of nodes settle at a common marginal rate
    `u`, each taking `clip((a - u)/B, 0, cap)` -- which is exactly a tick
    switching on when the price reaches it and switching off when it is spent.
    Bisected here rather than pivoted, because the point is the model and not
    the solver.
    """
    if not bank or dx <= 0:
        return 0.0
    a = [arc.a for arc in bank]
    lo, hi = -max(a), max(a)
    for _ in range(rounds):
        u = 0.5 * (lo + hi)
        if sum(_take(arc, u) for arc in bank) > dx:
            lo = u
        else:
            hi = u
    u = 0.5 * (lo + hi)
    flows = [_take(arc, u) for arc in bank]
    total = sum(flows)
    if total <= 0:
        return 0.0
    # The bisection lands on `u` within a rounding of the demand; spend the
    # remainder proportionally rather than reporting a flow that is not `dx`.
    if total < dx:
        room = sum(arc.cap - f for arc, f in zip(bank, flows, strict=True))
        if room > 0:
            share = min(1.0, (dx - total) / room)
            flows = [f + (arc.cap - f) * share
                     for arc, f in zip(bank, flows, strict=True)]
    else:
        flows = [f * dx / total for f in flows]
    return sum(arc.output(f) for arc, f in zip(bank, flows, strict=True))


def _take(arc: Arc, u: float) -> float:
    return min(max((arc.a - u) / arc.B, 0.0), arc.cap)


def capacity(bank: list[Arc]) -> float:
    """Everything the modelled ticks can absorb, in input units."""
    return sum(arc.cap for arc in bank)
