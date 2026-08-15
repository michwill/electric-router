"""What a route's own minimum-out costs it.

Every leg executes with a minimum-out set to a fraction of that pool's fee --
the level at which a sandwich stops paying for itself.  That bound is also a
trigger: if the pool's rate moves further than it between the quote and the
transaction landing, the leg reverts and the whole route with it.  A route
lands only if *every* pool it touches stays inside its own bound, so

    P(lands) = product over legs of (1 - p_i)

**A revert does not cost the trade, though, and that distinction sets the whole
scale of this term.**  Nothing is lost when a route fails: the gas is spent and
the user resubmits, so what a failure costs is one more transaction plus
whatever the price did in the meantime -- around a basis point, not a hundred.
Ranking on `output * P(lands)` instead prices a failure as losing the entire
notional, which is wrong by three orders of magnitude, and it shows: at the
measured 20-40% breach probabilities that objective pays 17 to 126 bp for
safety, and flips between routes 1-20% apart in price on nothing more than
which candidates a given run happened to generate.  So:

    output * (1 - P(fails) * REVERT_COST_BP / 1e4) - gas * (1 + P(fails))

The gas term carries the `(1 + P(fails))` because a failed attempt pays for
itself and the retry pays again; the output term is scaled by what a retry is
actually worth.  A leg is then priced at what it really adds -- its gas, and a
fraction of a basis point of failure risk -- which is enough to separate two
otherwise equal routes and never enough to buy a materially worse price.

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

#: What one failed attempt costs, as basis points of the trade.
#
# Gas, plus whatever the price did while the user resubmitted.  Set so that the
# most dangerous arc measured -- 39%, TricryptoUSDC's ETH side -- costs a route
# 0.4 bp, and a typical crypto pair a fraction of that.  Anything larger stops
# being a tie-break and starts buying worse prices: at the scale implied by
# "a failure loses the trade" the same 39% would be worth 390 bp.
#
# It is deliberately one number rather than a per-pair drift estimate.  What
# varies between pairs is `p`, which is measured; the cost of resubmitting is
# roughly the same trade either way.
REVERT_COST_BP = 1.0


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


def expected_value(output: float, survival: float, *, gas_cost: float = 0.0,
                   revert_cost_bp: float = REVERT_COST_BP) -> float:
    """What this route is worth, netting the cost of it not landing.

    `output` and `gas_cost` are both in output-token units.  A failure costs
    `revert_cost_bp` of the trade plus one extra transaction -- not the trade.
    """
    failure = max(0.0, min(1.0, 1.0 - survival))
    return (output * (1.0 - failure * revert_cost_bp / 1e4)
            - gas_cost * (1.0 + failure))


#: No measurement anywhere: every pool arc is priced at the default.
STATIC = RiskTable()
