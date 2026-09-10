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
    """How much this arc wants at marginal rate `u`.

    A clamped arc has no curvature to trade off against, so it is all or
    nothing: §2.3's linear leg, taken to its cap while it is the better price.
    """
    if arc.B <= 0:
        return arc.cap if arc.a > u else 0.0
    return min(max((arc.a - u) / arc.B, 0.0), arc.cap)


def capacity(bank: list[Arc]) -> float:
    """Everything the modelled ticks can absorb, in input units."""
    return sum(arc.cap for arc in bank)


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


def collapse(live, psi, nu, nodes, banks):
    """Put each pool's tick-arcs back into one arc before the route is built.

    A pool appears once in an executable route or its legs form one element
    (§7 rule 1), and K tick-arcs carrying flow at once would read as K visits.
    They are not: they are one swap the solver was allowed to describe
    piecewise, so they are summed here and priced by the bank at the size that
    actually landed -- `a = dy/dx` with `B = 0`, a chord exact at that point.

    Returns `(arcs, psi)` with every other arc untouched and in order.
    """
    import copy

    import numpy as np

    from ..core.types import ArcKind

    keep, flows, seen = [], [], {}
    for arc, flow in zip(live, psi, strict=True):
        if arc.kind is not ArcKind.SWAP_UNIV3:
            keep.append(arc)
            flows.append(float(flow))
            continue
        key = (arc.pool, arc.i, arc.j)
        if key in seen:
            flows[seen[key]] += float(flow)
            continue
        seen[key] = len(keep)
        keep.append(copy.copy(arc))
        flows.append(float(flow))

    for key, at in seen.items():
        arc = keep[at]
        total = flows[at]
        if total <= 0:
            continue
        # `psi` is value flow: canonical = psi / nu[tau], and human is that
        # over the node-merge rate.  Getting this wrong is silent -- the arc
        # still solves, just at the wrong price (see `nodes.rescale`).
        price = float(nu[arc.tau])
        if price <= 0 or arc.rate_in <= 0:
            continue
        dx_canonical = total / price
        dx = dx_canonical / arc.rate_in
        dy = output(banks[key], dx)
        # The chord, not the first tick's tangent: exact at the size realised,
        # and it is the only point this arc will be asked about.
        arc.a = (dy * arc.rate_out) / dx_canonical if dx_canonical > 0 else 0.0
        arc.B = 0.0
        # The bank's whole capacity, *not* the amount that happened to land.
        # `verify` refuses a candidate whose `over_capacity` is set, and
        # `_forward_simulate` rescales leg amounts after realisation -- so a cap
        # pinned to the realised size is a cap the very next step steps over.
        # Measured: every v3 leg vanished from the winning candidate while the
        # solve was still routing nineteen v3 arcs through it.
        arc.cap = capacity(banks[key]) * arc.rate_in
        # The tick arcs carry the venue and fee tier; the collapsed leg keeps
        # that and drops the tick index, which no longer means anything once
        # they are one leg.
        arc.note = f"{arc.note.split(' tick ')[0]} x{len(banks[key])}"
    return keep, np.array(flows, dtype=float)
