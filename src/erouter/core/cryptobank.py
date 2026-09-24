"""Cryptoswap arcs as a bank, not one parabola.

A Curve arc is one quadratic `a*d - B*d^2/2`, fitted as a secant through the
origin and one probe size.  That is a good local model and a bad global one:
its marginal `a - B*d` is a straight line, so it has to cross zero, and past
its peak `realize` clips it flat.  For a cryptoswap pool traded past a fraction
of its depth that is not a detail.  Measured on TriCRV at block 26,024,400 the
arc peaked at 1.55M CRV and capped out at 100 WETH, while the pool -- which is
constant product at size, bar a small amplified region near spot -- paid 260
WETH for 7.7M CRV.  Refitting at the trade size cannot fix it: a parabola can
match the *value* there but not the *slope*, and a split is decided by slopes.

So each such arc becomes the bank a Uniswap v3 pool already is: parallel arcs,
one per segment of a geometric size grid reaching past the trade, each a
quadratic fitted from its segment's two ends and midpoint so value and slope
are both right locally.  The pool's own invariant supplies every point --
these are the arcs the client can *compute* -- so no round trip is spent.
`univ3.collapse` folds the bank back into one leg before it is realised, as it
does for ticks.
"""

from __future__ import annotations

import copy
import math

from .bank import Arc
from .transport import Status
from .types import ArcKind, Probe

#: Segments a bank is cut into, and how far past the trade it reaches.  The
#: grid halves from the top.  Not finer: ten segments fit TriCRV to 0.17%
#: rather than 0.24%, but their near-linear slivers made the base solve churn
#: 7,300 pivots into "src not connected" and cost CRV->WETH $1M 332 bp.
SEGMENTS = 4
REACH = 1.25
#: A pool whose price moves less than this across the reach keeps its one
#: quadratic: a bank of it is parallel near-identical arcs, which is the
#: degeneracy above, and it buys no fidelity.
MIN_IMPACT = 0.02
#: Nor past this multiple of the pool's input reserve.  A grid sized to the
#: trade put a small twocrypto pool's first probe at 26x its reserve; the
#: invariant refused it, the pool kept an uncapped near-linear quadratic, and
#: CRV->WETH $5M handed it 8M CRV in 52 of 65 candidates, all reverting.  At 4x
#: a constant-product pool has paid out 80% of its other side.
DEPTH = 4.0


def bank_arcs(arcs, nu, nodes, client, Psi: float):
    """`(arcs, banks)` with every computable cryptoswap arc replaced by a bank.

    `banks` maps `(pool, i, j)` to `univ3.Arc`s in human token units, which is
    what `univ3.collapse` and `univ3.output` read.  An arc whose pool cannot be
    computed, whose price is unknown, or whose invariant refuses the smallest
    segment is left exactly as it was.
    """
    from .nodes import rescale

    computes = getattr(client, "computes", None)
    if computes is None or not (Psi > 0.0):
        return arcs, {}
    plans = []
    for k, arc in enumerate(arcs):
        if arc.kind is not ArcKind.SWAP_CRYPTO or not computes(arc.pool):
            continue
        price, rate = float(nu[arc.tau]), nodes.rate(arc.token_in)
        if price <= 0 or rate <= 0:
            continue
        whole = Psi / price / rate * REACH
        if arc.reserve_in > 0:
            whole = min(whole, DEPTH * arc.reserve_in / 10 ** arc.decimals_in)
        if not math.isfinite(whole) or whole <= 0:
            continue
        ends = [whole * 2.0 ** (s - SEGMENTS + 1) for s in range(SEGMENTS)]
        starts = [0.0, *ends[:-1]]
        mids = [(lo + hi) / 2 for lo, hi in zip(starts, ends, strict=True)]
        plans.append((k, arc, starts, ends, mids))
    if not plans:
        return arcs, {}

    probes = []
    for _, arc, _, ends, mids in plans:
        scale = 10 ** arc.decimals_in
        for x in (*ends, *mids):
            probes.append(Probe(pool=arc.pool, kind=arc.kind, i=arc.i, j=arc.j,
                                n=arc.n_coins, dx=max(1, int(x * scale))))
    quotes = iter(client.probe(probes))

    banks: dict[tuple, list] = {}
    replace: dict[int, list] = {}
    for k, arc, starts, ends, mids in plans:
        out_scale = 10 ** arc.decimals_out
        at_end = [next(quotes) for _ in ends]
        at_mid = [next(quotes) for _ in mids]

        bank, pieces = [], []
        previous_end = 0.0
        exit_price = math.inf
        for seg, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
            f_hi = _human(at_end[seg], out_scale)
            f_mid = _human(at_mid[seg], out_scale)
            if f_hi is None or f_mid is None:
                break                       # the invariant stops answering here
            width = hi - lo
            F1 = f_hi - previous_end        # what the segment pays in total
            F2 = f_mid - previous_end       # what its first half pays
            a = (4.0 * F2 - F1) / width
            B = 4.0 * (2.0 * F2 - F1) / (width * width)
            # Concave by construction for a real pool; noise at the far end is
            # not, and a segment priced above its predecessor's exit would fill
            # before it and undo the ordering the bank relies on.
            a = min(a, exit_price)
            if not (a > 0.0):
                break
            B = max(B, a * 1e-9 / width)    # finite conductance, never zero
            bank.append(Arc(a=a, B=B, cap=width))
            ra, rB = rescale(a, B, arc.rate_in or nodes.rate(arc.token_in),
                             arc.rate_out or nodes.rate(arc.token_out))
            piece = copy.copy(arc)
            piece.id = f"{arc.id}#{seg}"
            piece.a, piece.B = ra, rB
            piece.cap = width * nodes.rate(arc.token_in)
            piece.parallel = True
            piece.note = f"{arc.note} tick {seg}"
            pieces.append(piece)
            previous_end = f_hi
            exit_price = a - B * width
        if len(bank) < 2 or exit_price > (1.0 - MIN_IMPACT) * bank[0].a:
            continue
        banks[(arc.pool, arc.i, arc.j)] = bank
        replace[k] = pieces
    if not replace:
        return arcs, {}
    out = []
    for k, arc in enumerate(arcs):
        out.extend(replace.get(k, (arc,)))
    return out, banks


def _human(quote, scale: int) -> float | None:
    """A quote in output-token units, or `None` where the invariant refused."""
    if quote.status is Status.VALUE and quote.value > 0:
        return quote.value / scale
    return None
