"""What a route's own minimum-out costs it in expectation.

Every leg executes with a minimum-out set to a fraction of that pool's fee --
the level at which a sandwich stops paying for itself.  That bound is also a
trigger: if the pool's rate moves further than it between the quote and the
transaction landing, the leg reverts and the whole route with it.  A route
lands only if *every* pool it touches stays inside its own bound, so

    P(lands) = product over legs of (1 - p_i)

and the quantity worth maximising is the expected outcome,

    output * P(lands) - gas

because gas is paid whether or not the route lands.  This is what makes an
extra leg cost something beyond its gas, and it is a better statement of the
cost than a leg budget or a flat per-leg threshold: it names *which* pools are
expensive rather than penalising length as such.

The measurement (`dev/revert_risk.py`) says that matters.  The median pool
never breached its bound in half an hour of samples; the risk is concentrated
in a handful whose fee is small against their own volatility.  TriCRV charges
3.36 bp, so its bound is 0.67 bp against a rate that moves 2.4 bp a minute --
it breaches a quarter of the time.  Yield Basis WETH holds a far more volatile
asset and never breached, because its 218 bp fee puts the bound at 43.7.  Asset
class does not predict this at all; fee against volatility does.  A leg budget
would have charged those two the same.

Only pool arcs carry the risk.  A wrap, a stake, a lending mint or a vault
redemption is priced by a rate that moves with accrual -- upward, slowly, and
not against us -- so no minimum-out of this kind can trigger on it.

An unmeasured pool is not a free one.  `DEFAULT_RISK` stands in until it is
probed, which keeps a thin pool nobody has sampled from looking safer than the
deep ones we did measure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .types import ArcKind, Leg

#: Kinds whose output is not a market rate, so no bound of this sort binds.
RISKLESS: frozenset[ArcKind] = frozenset({
    ArcKind.WRAP_NATIVE,
    ArcKind.UNWRAP_NATIVE,
    ArcKind.WSTETH_WRAP,
    ArcKind.WSTETH_UNWRAP,
    ArcKind.STAKE_NATIVE,
    ArcKind.LEND_MINT,
    ArcKind.LEND_REDEEM,
    ArcKind.ERC4626_DEPOSIT,
    ArcKind.ERC4626_REDEEM,
})

#: What an arc nobody has measured is assumed to cost, as a probability.
#
# Deliberately small but not zero.  Zero would say "this pool provably never
# moves", which is the one thing a missing measurement cannot say, and would
# make an unprobed pool the cheapest thing in the graph.
#
# The measured distribution is sharply bimodal -- nine arcs in ten sit at the
# 1e-5 floor and the rest run from 1% to 50% -- so neither its median nor its
# upper decile is a sensible stand-in: the first is indistinguishable from free
# and the second charges 120 bp for a gap in our own sampling.  0.2% is chosen
# instead as what an unknown arc must beat: about 20 bp, which is more than any
# tail-end routing gain and less than the cost of dropping a good pool.
# `FactsCache.risk_table` raises it to the measured 75th percentile when that
# is higher.
DEFAULT_RISK = 0.002


class RiskTable:
    """Per-arc probability that a leg's minimum-out trips before inclusion.

    Keyed by direction, like `GasTable` and for the same reason: a pool's own
    pairs do not behave alike.  TriCRV's CRV/ETH rate moves several times as
    far in a minute as its crvUSD/USDC one, and the minimum-out is written per
    leg, so pricing the pool by its worst pair would charge every route for the
    riskiest thing in it.

    Lookup walks from the specific to the general:

    1. this direction of this pool, measured;
    2. any direction of this pool, under `(-1, -1)` -- a pair the sweep missed
       on a pool it knows;
    3. `default`, for a pool that has never been sampled.
    """

    __slots__ = ("arcs", "default")

    def __init__(self, arcs: Mapping[tuple[str, int, int], float] | None = None,
                 default: float = DEFAULT_RISK):
        self.arcs = {(str(a).lower(), int(i), int(j)): float(p)
                     for (a, i, j), p in (arcs or {}).items()}
        self.default = float(default)

    def of(self, kind: ArcKind, target: str = "", i: int = 0, j: int = 0) -> float:
        kind = ArcKind(kind)
        if kind in RISKLESS:
            return 0.0
        address = target.lower()
        got = None
        if kind.is_swap:
            # `(i, j)` means coin indices only on a swap.  A deposit or a
            # single-coin withdrawal numbers its legs differently, so reading
            # the specific tier for one would hand it whichever swap happened
            # to share the pair -- it goes to the pool-level entry instead,
            # which is the right granularity for it anyway.
            got = self.arcs.get((address, int(i), int(j)))
        if got is None:
            got = self.arcs.get((address, -1, -1))
        return self.default if got is None else got

    def survival(self, legs: Iterable[Leg]) -> float:
        """P(every leg stays inside its bound).

        A pool touched twice counts twice, which is right: two legs are two
        separate minimum-outs, both of which have to hold.  (Routes are one
        arc per pool anyway, so this is a statement about wraps sharing a
        target, not about split pools.)
        """
        product = 1.0
        for leg in legs:
            product *= 1.0 - self.of(leg.kind, leg.target, leg.i, leg.j)
        return product

    def __len__(self) -> int:
        return len(self.arcs)

    def __bool__(self) -> bool:
        return bool(self.arcs)


#: No measurement anywhere: every pool arc is priced at the default.
STATIC = RiskTable()
