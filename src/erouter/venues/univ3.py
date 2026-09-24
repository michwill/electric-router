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

from ..core.bank import Arc, _take, capacity, collapse, output  # noqa: F401

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


    @property
    def clamped(self) -> bool:
        return self.B <= 0


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


def exact_output(bank: list[Arc], dx: float) -> float:
    """What the pool pays for `dx`, tick by tick, in human units.

    Within a range v3 is constant product on virtual reserves, and an arc's own
    fields already say which: `dy = a dx / (1 + k dx)` with `k = B / 2a`, since
    `B` is that curve's curvature at the range's start.  The arc law is its
    tangent quadratic, which undershoots a range the trade moves a long way
    through: -2,490 bp on a v3 CRV/WETH leg at $5M, where one initialized range
    carried the price down 70%.  Ranges are consumed nearest first, and a trade
    past the last one is paid for what the bank holds.
    """
    got, left = 0.0, dx
    for arc in bank:
        if left <= 0:
            break
        take = min(left, arc.cap)
        k = arc.B / (2.0 * arc.a) if arc.a > 0 else 0.0
        got += arc.a * take / (1.0 + k * take)
        left -= take
    return got


# --------------------------------------------------------------- the router

#: The smallest share of a bank's own capacity worth an arc.
#
# A tick range the price is sitting almost exactly on holds nearly nothing, and
# its `B = 2 a^(3/2) / L` is then enormous while its cap is ~0.  Measured over
# 144 mainnet pools, caps ran from 9.0e+08 down to 1.7e-14 and the conductance
# spread reached 1.4e25 -- past §9.7's 1e15, which exists to catch exactly this
# shape and is right to.  Such an arc cannot carry anything, so it is not built.
MIN_CAP_SHARE = 1e-6


def pool_arcs(pool: str, state: PoolState, ticks: list[Tick], nodes, *,
              token0: str, token1: str, max_ticks: int = 16,
              tvl_usd: float = 0.0, min_cap_share: float = MIN_CAP_SHARE,
              kind=None, venue: str = "uniswap v3", label: str = "Uniswap v3"):
    """Both directions of one pool, as arcs the solver can take straight.

    Nothing here is probed and nothing is fitted, so these arcs skip the refine
    and size-check stages entirely: `_recalibrate` keys on the ladder store and
    a v3 arc has no ladder, so it is passed over rather than re-measured.  That
    is the whole point -- the model is already exact to a hundredth of a basis
    point, and a probe could only make it worse.

    Ids carry a `#k` segment because a pool contributes many arcs between the
    same pair of nodes.  They are a decomposition of one swap, not many swaps,
    and `collapse` puts them back together before the route is realised.

    `kind`, `venue` and `label` exist for v4, whose swap arithmetic is this one
    exactly -- same ticks, same liquidity, same closed form.  What differs there
    is which pools are allowed to have arcs at all, and that is
    `venues.univ4.tier`'s business rather than this function's.  `pool` is a
    `PoolId` there instead of an address, which nothing here minds: it is used
    as an identity, and v4 has no per-pool address to offer.
    """
    from ..core.nodes import rescale
    from ..core.types import ArcKind, PoolArc

    kind = ArcKind.SWAP_UNIV3 if kind is None else kind

    out = []
    for zero_for_one in (True, False):
        token_in, token_out = ((token0, token1) if zero_for_one
                               else (token1, token0))
        if not (nodes.has(token_in) and nodes.has(token_out)):
            continue
        bank = arcs(state, ticks, zero_for_one=zero_for_one, max_ticks=max_ticks)
        if not bank:
            continue
        rate_in, rate_out = nodes.rate(token_in), nodes.rate(token_out)
        tau, sigma = nodes.node(token_in), nodes.node(token_out)
        if tau == sigma:                     # a node merge swallowed the pair
            continue
        # Never the first: the price sits inside that tick, so its arc holds
        # the remainder of the range the pool is trading in *now* and is the
        # best-priced liquidity there is.  It is also the one most likely to be
        # a sliver, since the sliver is exactly "the price is near this
        # boundary".  Dropping it cost 0.94 bp on a small leg while the rest of
        # the same route was within 0.002.
        floor = capacity(bank) * min_cap_share
        bank = [arc for k, arc in enumerate(bank) if k == 0 or arc.cap > floor]
        if not bank:
            continue
        i, j = (0, 1) if zero_for_one else (1, 0)
        decimals_in = state.decimals0 if zero_for_one else state.decimals1
        decimals_out = state.decimals1 if zero_for_one else state.decimals0
        for k, arc in enumerate(bank):
            a, b = rescale(arc.a, arc.B, rate_in, rate_out)
            out.append(PoolArc(
                id=f"{pool.lower()}:{int(kind)}:{i}>{j}#{k}",
                pool=pool.lower(), kind=kind, i=i, j=j, n_coins=2,
                token_in=token_in, token_out=token_out, tau=tau, sigma=sigma,
                a=a, B=b, cap=arc.cap * rate_in,
                rate_in=rate_in, rate_out=rate_out,
                decimals_in=decimals_in, decimals_out=decimals_out,
                reserve_in=int(capacity(bank) * 10**decimals_in),
                # `collapse` sums these back into one arc before realisation,
                # which is what lets a bank of them satisfy Decision 3.
                parallel=True,
                # And this is what buys the ballot a candidate without them, so
                # that adding the venue cannot cost the answer.
                venue=venue,
                # Named for the venue and the fee tier, because the note is
                # what the route diagram prints: "v3, 16 tick(s)" is not a
                # thing anyone scanning a route for Uniswap will recognise.
                tvl_usd=tvl_usd,
                note=f"{label} {state.fee / 10_000:g}% tick {k}"))
    return out


