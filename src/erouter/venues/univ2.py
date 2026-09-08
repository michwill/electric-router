"""Uniswap v2 as arcs, and nothing that needs a socket.

A v2 pair is the simplest thing this router prices.  One curve, no ticks, no
positions: reserves `(x, y)` with `x y = k` and a flat fee, so

    dy = k dx y / (x + k dx),        k = 1 - fee

is exact at every size rather than a fit, and the arc law falls out of it by
expansion:

    dy = (k y / x) dx * 1 / (1 + k dx / x)
       = a dx - B dx^2 / 2 + O(dx^3),      a = k y / x,   B = 2 a k / x

which is the same statement `univ3` makes about a tick, with `a = P` and
`B = 2 a^(3/2) / L`.  Both are second-order truncations of a known curve, so
neither needs the probe ladder `calibrate.py` runs against a Curve pool.

**One arc per direction, not a bank.**  A v3 pool is piecewise by construction --
liquidity changes at every tick boundary -- and its arcs have to be collapsed
before a route is realised.  A v2 pair has one range, so it produces one arc
each way and needs no collapse, no `parallel` flag and no Decision 3 exemption.
That is most of why this file is a fifth of `univ3.py`.

**The fee is per pool, not per protocol.**  Uniswap's own pairs charge 30 bp,
but the same bytecode is deployed at 25 bp (Sushi, Pancake) and 20 bp, and a
fork that changed it and kept the interface reads as a Uniswap pair to every
call this makes.  So `fee_bps` travels with the state rather than being a
constant here, and a census that does not know it is not admitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Uniswap's own fee, in basis points.  A default for a census row that omits
#: it, never an assumption made about a pool that stated something else.
DEFAULT_FEE_BPS = 30
#: The most of a pair's input reserve one arc may take.
#:
#: Past this the quadratic is describing a curve it has left, and it fails in
#: the direction that is easy to miss: `a dx - B dx^2 / 2` **turns over** at
#: `dx = a / B = x / 2k` and pays less the more it is given.  At a whole reserve
#: it claims `k y (1 - k)` where the pool pays `k y / (1 + k)` -- 7,477 USDC
#: against 1,248,122 on a 1,000 WETH pair, 0.6% of the truth rather than an
#: overshoot.  A solver reading that would route around a pool it had just
#: emptied.
#:
#: A tenth keeps the arc an order below the vertex and below §12.1's own
#: escalation threshold, so the size check speaks before the fit stops meaning
#: anything.
MAX_SHARE = 0.10


@dataclass(frozen=True, slots=True)
class PairState:
    """One pair, at one block, in raw token units."""

    reserve0: int
    reserve1: int
    decimals0: int
    decimals1: int
    fee_bps: int = DEFAULT_FEE_BPS

    @property
    def live(self) -> bool:
        """Both sides hold something and the fee is a fee.

        A pair with an empty side quotes a price and pays nothing, which is the
        same failure `find_broken_pools` catches on Curve: the arc would be
        built from a rate no trade can realise.
        """
        return (self.reserve0 > 0 and self.reserve1 > 0
                and 0 <= self.fee_bps < 10_000)


def output(state: PairState, zero_for_one: bool, dx: int) -> int:
    """Exactly what the pair pays for `dx`, in raw units.

    The contract's own arithmetic: `getAmountOut` multiplies before it divides
    and floors once, so doing it in that order is what makes this wei-exact
    rather than nearly so.
    """
    if dx <= 0 or not state.live:
        return 0
    x, y = ((state.reserve0, state.reserve1) if zero_for_one
            else (state.reserve1, state.reserve0))
    kept = dx * (10_000 - state.fee_bps)
    return (kept * y) // (x * 10_000 + kept)


def arc_params(state: PairState, zero_for_one: bool) -> tuple[float, float, float]:
    """`(a, B, cap)` in raw token units, from the reserves alone.

    `a` is the marginal rate at the origin and `B` its curvature, both exact for
    a constant product -- there is nothing to probe and nothing to fit.  `cap`
    is `MAX_SHARE` of the input reserve, which keeps the quadratic inside the
    range where it still describes the curve.
    """
    if not state.live:
        return 0.0, 0.0, 0.0
    x, y = ((state.reserve0, state.reserve1) if zero_for_one
            else (state.reserve1, state.reserve0))
    keep = (10_000 - state.fee_bps) / 10_000
    scale_in = 10.0 ** (state.decimals0 if zero_for_one else state.decimals1)
    scale_out = 10.0 ** (state.decimals1 if zero_for_one else state.decimals0)
    # Human units, because that is what `calibrate` fits and `rescale` expects.
    x_h, y_h = x / scale_in, y / scale_out
    if x_h <= 0 or y_h <= 0:
        return 0.0, 0.0, 0.0
    a = keep * y_h / x_h
    b = 2.0 * a * keep / x_h
    return a, b, x_h * MAX_SHARE


def tvl_usd(state: PairState, price0: float, price1: float) -> float:
    """Both sides valued, which is what a floor is applied to."""
    if not state.live:
        return 0.0
    return (state.reserve0 / 10.0 ** state.decimals0 * price0
            + state.reserve1 / 10.0 ** state.decimals1 * price1)


def arcs_for(pool: str, state: PairState, token0: str, token1: str, nodes,
             *, tvl: float = 0.0, note: str = ""):
    """Both directions of one pair as `PoolArc`s, or nothing.

    Mirrors `univ3.arcs_for` and is shorter for the reason the module docstring
    gives: no bank, so no `parallel`, no cap-share filter and no collapse.
    """
    from ..core.nodes import rescale
    from ..core.types import ArcKind, PoolArc

    out = []
    if not state.live:
        return out
    for zero_for_one in (True, False):
        token_in, token_out = ((token0, token1) if zero_for_one
                               else (token1, token0))
        if not (nodes.has(token_in) and nodes.has(token_out)):
            continue
        tau, sigma = nodes.node(token_in), nodes.node(token_out)
        if tau == sigma:                     # a node merge swallowed the pair
            continue
        a_raw, b_raw, cap_raw = arc_params(state, zero_for_one)
        if a_raw <= 0 or not math.isfinite(a_raw):
            continue
        rate_in, rate_out = nodes.rate(token_in), nodes.rate(token_out)
        if rate_in <= 0 or rate_out <= 0:
            continue
        a, b = rescale(a_raw, b_raw, rate_in, rate_out)
        i, j = (0, 1) if zero_for_one else (1, 0)
        decimals_in = state.decimals0 if zero_for_one else state.decimals1
        decimals_out = state.decimals1 if zero_for_one else state.decimals0
        reserve_in = state.reserve0 if zero_for_one else state.reserve1
        out.append(PoolArc(
            id=f"{pool.lower()}:{int(ArcKind.SWAP_UNIV2)}:{i}>{j}",
            pool=pool.lower(), kind=ArcKind.SWAP_UNIV2, i=i, j=j, n_coins=2,
            token_in=token_in, token_out=token_out, tau=tau, sigma=sigma,
            a=a, B=b, cap=cap_raw * rate_in,
            rate_in=rate_in, rate_out=rate_out,
            decimals_in=decimals_in, decimals_out=decimals_out,
            reserve_in=int(reserve_in),
            # Not `parallel`: one arc is the whole pair in this direction, so
            # Decision 3 is satisfied without an exemption to claim.
            venue="uniswap v2",
            tvl_usd=tvl,
            note=note or f"Uniswap v2 {state.fee_bps / 100:g}%",
        ))
    return out
