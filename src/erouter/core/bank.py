"""A bank: parallel arcs between one node pair that describe one swap piecewise.

Uniswap v3 and v4 contribute one arc per tick range, and a cryptoswap pool at
size is banked the same way (`core/cryptobank.py`).  The maths is the same for
both -- a common marginal rate fills the arcs in price order -- so it lives
here, where `core` can reach it, and the venues import it.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def collapse(live, psi, nu, nodes, banks, kind=None, bounded=True):
    """Put each pool's tick-arcs back into one arc before the route is built.

    `kind` is a parameter for the reason `pool_arcs` and `univ3_client.teach`
    take one: v4's banks are these banks, and a fold that assumes `SWAP_UNIV3`
    does the wrong thing twice over when both venues are live.  v4's own arcs
    are left uncollapsed because their kind does not match, and v3's arcs are
    folded against **v4's** banks, which do not hold them -- `KeyError` on
    `('0x60594a405d53811d3bc4766596efd80fd545a270', 0, 1)`, the v3 DAI/WETH
    pool, and every major pair failing to quote with `--univ3 --univ4` both on.
    Defaults to `SWAP_UNIV3` so v3's own call is unchanged.

    A pool appears once in an executable route or its legs form one element
    (§7 rule 1), and K tick-arcs carrying flow at once would read as K visits.
    They are not: they are one swap the solver was allowed to describe
    piecewise, so they are summed here and priced by the bank at the size that
    actually landed -- `a = dy/dx` with `B = 0`, a chord exact at that point.

    `bounded=False` leaves the folded arc uncapped, for a stableswap bank: it
    takes any size and only saturates, so its reach is where the model ends.
    Held to it, the cap guard refused every GHO->USDT $10M candidate.  Not a
    cryptoswap: Rocketpool rETH/ETH refuses 8x its reach ("unsafe value for y").

    Returns `(arcs, psi)` with every other arc untouched and in order.
    """
    import copy
    import math

    import numpy as np

    from .types import ArcKind

    want = ArcKind.SWAP_UNIV3 if kind is None else kind
    keep, flows, seen = [], [], {}
    for arc, flow in zip(live, psi, strict=True):
        key = (arc.pool, arc.i, arc.j)
        # A kind this fold owns but a pool with no bank -- a cryptoswap arc the
        # client could not compute -- is one arc already and stays as it is.
        if arc.kind is not want or key not in banks:
            keep.append(arc)
            flows.append(float(flow))
            continue
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
        # The node's rates, which is what `realize` converts with; an arc's own
        # `rate_in` is 1.0 on a Curve arc whatever its token.
        price = float(nu[arc.tau])
        rate_in, rate_out = nodes.rate(arc.token_in), nodes.rate(arc.token_out)
        if price <= 0 or rate_in <= 0:
            continue
        dx_canonical = total / price
        dx = dx_canonical / rate_in
        dy = output(banks[key], dx)
        # The chord, not the first tick's tangent: exact at the size realised,
        # and it is the only point this arc will be asked about.
        arc.a = (dy * rate_out) / dx_canonical if dx_canonical > 0 else 0.0
        arc.B = 0.0
        # The bank's whole capacity, *not* the amount that happened to land.
        # `verify` refuses a candidate whose `over_capacity` is set, and
        # `_forward_simulate` rescales leg amounts after realisation -- so a cap
        # pinned to the realised size is a cap the very next step steps over.
        # Measured: every v3 leg vanished from the winning candidate while the
        # solve was still routing nineteen v3 arcs through it.
        arc.cap = capacity(banks[key]) * rate_in if bounded else math.inf
        # The tick arcs carry the venue and fee tier; the collapsed leg keeps
        # that and drops the tick index, which no longer means anything once
        # they are one leg.
        arc.note = f"{arc.note.split(' tick ')[0]} x{len(banks[key])}"
    return keep, np.array(flows, dtype=float)
