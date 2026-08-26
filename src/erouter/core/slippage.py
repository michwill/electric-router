"""Divide a slippage budget between the legs of a route.

The automatic bound sets every leg from its own pool -- a fifth of the least
that pool can charge -- and the route's total is whatever those add up to.  A
caller who names the total instead is asking the opposite question, and it has
one more constraint than it looks: a route that branches and merges must spend
the budget *once*, not once per branch, and the legs a branch shares with its
sibling cannot be given two different bounds.

Which is voltage division.  Give each leg a resistance equal to the tolerance
the automatic rule would have granted it -- proportional to the pool's fee
wherever the fee rule binds -- put the whole budget across the route's source
and destination, and read the drops.  Series shares in proportion: 1 bp and
20 bp of fee split 50 bp into 2.38 and 47.62.  Parallel branches each drop the
whole budget, because either one of them alone is the trade.  Where the two
rules disagree -- a leg two branches share, which per-path proportions would
want to bound twice -- the network is what settles it.

The result is normalised so that the longest path spends exactly the budget,
which is what makes the guarantee structural: no path can spend more, whatever
the solve returned.  `1 - sum` under-states `prod(1 - t)`, so the bound the
caller gets is never looser than the one they asked for.

**A leg the network runs backwards keeps its own drop.**  An undirected network
has no idea the route is a DAG, so on a shape that is not series-parallel the
solve can sit a leg's head above its tail -- a Wheatstone bridge, and on
mainnet a BTC-to-BTC pool between two branches.  Its drop comes back negative,
which is no bound at all, and shipping it at the rounding floor would revert on
any movement: measured, it carries 17-25% of the route.  So it is granted the
magnitude it came back by -- what the network says the imbalance across it is,
0.3 to 2.7 bp against a 50 bp budget -- as a floor the rescale may not erode,
and the rest of the route is pulled back around it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .linalg import DEFAULT_SOLVER, SingularSystem
from .realize import RealizedRoute

#: Slot 0 holds the input.  `routecall.fractions` seeds its walk the same way.
SOURCE = 0

#: A resistance of zero is a short circuit and has no conductance to write.
TINY = 1e-12


def _slots(route: RealizedRoute) -> list[int]:
    seen: list[int] = []
    for realized in route.legs:
        for slot in (realized.leg.src_slot, realized.leg.dst_slot):
            if slot not in seen:
                seen.append(slot)
    return seen


def drops(route: RealizedRoute, resistance: Sequence[float],
          total: float) -> list[float] | None:
    """The potential each leg drops, or None if the network will not solve.

    Dirichlet at the two terminals and Kirchhoff everywhere else, which is the
    same Laplacian as the solver on a graph of three or four nodes.
    """
    slots = _slots(route)
    index = {slot: k for k, slot in enumerate(slots)}
    source, sink = index.get(SOURCE), index.get(route.dst_slot)
    if source is None or sink is None or source == sink:
        return None
    matrix = np.zeros((len(slots), len(slots)))
    for realized, r in zip(route.legs, resistance, strict=True):
        a, b = index[realized.leg.src_slot], index[realized.leg.dst_slot]
        conductance = 1.0 / max(float(r), TINY)
        matrix[a, a] += conductance
        matrix[b, b] += conductance
        matrix[a, b] -= conductance
        matrix[b, a] -= conductance
    free = [k for k in range(len(slots)) if k not in (source, sink)]
    potential = np.zeros(len(slots))
    potential[source] = total
    if free:
        rhs = -matrix[np.ix_(free, [source])].ravel() * total
        try:
            potential[free] = DEFAULT_SOLVER.solve(matrix[np.ix_(free, free)], rhs)
        except SingularSystem:
            return None
    return [float(potential[index[realized.leg.src_slot]]
                  - potential[index[realized.leg.dst_slot]])
            for realized in route.legs]


def longest(route: RealizedRoute, spend: Sequence[float]) -> float:
    """The most any one path spends.  Legs arrive topologically ordered."""
    best = {SOURCE: 0.0}
    for realized, drop in zip(route.legs, spend, strict=True):
        reached = best.get(realized.leg.src_slot)
        if reached is None:
            continue
        head = realized.leg.dst_slot
        best[head] = max(best.get(head, 0.0), reached + drop)
    return best.get(route.dst_slot, 0.0)


#: Enough halvings to land the scale factor inside a float's last digits.
BISECTIONS = 60


def backstops(raw: Sequence[float], floor: Sequence[float] | None) -> list[float]:
    """What each leg is owed however the rest of the route is scaled.

    Zero for a leg the network drops forwards, which is nearly all of them.  A
    leg it runs backwards is owed the magnitude it came back by, and never less
    than `floor` -- the automatic rule's own answer, which is what that leg
    would have shipped with no budget named at all.
    """
    return [max(-value, float(floor[k]) if floor is not None else 0.0)
            if value < 0.0 else 0.0 for k, value in enumerate(raw)]


def divide(route: RealizedRoute, resistance: Sequence[float], total: float,
           *, backstop: Sequence[float] | None = None) -> list[float]:
    """Split `total` between the legs, as fractions rather than basis points.

    `backstop` is read only where the network runs a leg backwards; see
    `backstops`.  Those legs are held at their floor and everything else is
    scaled around them, by bisection because the longest path is a maximum over
    paths and not a sum -- monotone in the scale, so it converges, and the side
    it converges from is the one that cannot overspend.

    Falls back to sharing in proportion to `resistance` alone where the network
    does not solve -- a slot nothing reaches -- and to depth where every
    resistance is zero.  All three are normalised the same way, so the promise
    a caller reads off `RouteCall.tolerance_bp` holds however it got there.
    """
    if total < 0.0:
        raise ValueError(f"a slippage budget cannot be negative, got {total}")
    if not route.legs:
        return []
    raw = drops(route, resistance, total)
    if raw is None:
        raw = [float(r) for r in resistance]
    held = backstops(raw, backstop)
    share = [max(0.0, value) for value in raw]
    spent = longest(route, share)
    if spent <= 0.0:
        share = [1.0] * len(route.legs)
        spent = longest(route, share)
    if spent <= 0.0:
        return held
    share = [value * total / spent for value in share]
    if not any(held):
        return share
    if longest(route, held) >= total:
        # The floors are the budget on their own.  They are what a leg needs to
        # survive movement, so they stand and the total is what it is.
        return held
    low, high = 0.0, 1.0
    for _ in range(BISECTIONS):
        mid = 0.5 * (low + high)
        if longest(route, _blend(held, share, mid)) <= total:
            low = mid
        else:
            high = mid
    return _blend(held, share, low)


def _blend(held: Sequence[float], share: Sequence[float],
           scale: float) -> list[float]:
    return [max(floor, scale * value)
            for floor, value in zip(held, share, strict=True)]
