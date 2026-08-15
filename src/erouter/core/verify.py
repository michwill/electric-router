"""On-chain verification of candidates (spec §7).

The model is used for combinatorics and the chain for arithmetic.  Every
candidate goes out in **one** `quote_routes` call at the pinned block, because
chained `get_dy` cannot be batched any other way -- without the quoter, twenty
multi-hop candidates would be twenty sequential round trips.

Four rules from §7, all load-bearing here:

1. *Net shared pools before quoting.*  Enforced upstream by the edge-flow
   formulation, and asserted here: a candidate touching a pool twice is
   rejected rather than quoted, because a view-only call cannot see its own
   earlier leg.
2. *Pin the block.*  All candidates share one `Transport`, hence one block.
3. *Quote at the real size.*  The legs carry the amounts the model chose.
4. *Treat a failed quote as arc removal.*  A zero comes back as `reverted` and
   the candidate is dropped, never as an error.

What this catches that no amount of modelling will: paused pools, reentrancy
locks, fee-on-transfer tokens, stale indexer state -- and, most importantly
here, the difference between a fitted reference price and the real curve.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, CandidateSet
from .gas import GasTable, leg_gas, plan_gas
from .nodes import NodeMap
from .quoter import MAX_LEGS, MAX_SLOTS, QuoterClient
from .realize import RealizationError, check_one_arc_per_pool, realize
from .types import ArcKind, PoolArc

# Per-kind execution gas lives in `gas.py`; a flat per-leg figure over-charged
# wraps by 3x and under-charged single-coin withdrawals.
# Outputs within this relative distance are the same answer; take the cheaper
# route to execute.  0.05 bp is far below any gas cost worth the extra hop.
# What counts as "the same answer".  A flat fraction was wrong in both
# directions, because it stands in for a cost `score` has already subtracted.
#
# `score` nets gas off every candidate before ranking, so the only thing left
# for a tolerance to absorb is noise in the quotes themselves -- integer
# rounding, around 1e-12 relative.  A flat 5e-6 was instead absorbing a whole
# basis point of real difference at low gas: measured on WETH->stETH 100 at
# 0.049 gwei, a 4-leg route quoting 100.000239 stETH lost to a 2-leg one at
# 99.999824, because 0.0415 bp fell under the 0.05 bp line -- while the two
# extra legs cost 1.5e-5 stETH of gas, thirty times less than the gain thrown
# away.  Minting stETH is exactly 1:1 and can never lose; the router had the
# answer and discarded it.
#
# So the tolerance is what one more leg actually costs, in output units, and
# nothing more.  At 30 gwei that is large and the preference for short routes
# comes back on its own; at 0.05 gwei it is nearly zero and a real gain is
# taken.  The floor keeps a tie from turning on the last bit of a quote.
TIE_FLOOR = 1e-12


def realize_candidates(
    candidates: CandidateSet,
    arcs: list[PoolArc],
    nu: np.ndarray,
    nodes: NodeMap,
    *,
    src_token: str,
    dst_token: str,
    amount_in: int,
    potentials: np.ndarray | None = None,
    max_legs: int = MAX_LEGS,
) -> None:
    """Turn each candidate's flow into legs, marking the ones that cannot be.

    `max_legs` defaults to the quoter's ABI capacity, which is what *we* can
    price -- not what anything can execute.  A deployed router has its own,
    much tighter, limit: curve_solver's caps at 4 legs and 16 ops.  Until this
    router emits calldata there is nothing to violate, so the default stays
    permissive and the knob exists for whoever has an executor to satisfy.
    """
    for candidate in candidates.candidates:
        active = np.flatnonzero(candidate.psi > 0)
        if active.size == 0:
            candidate.status = "empty"
            continue
        try:
            route = realize(
                [arcs[int(k)] for k in active],
                candidate.psi[active],
                nu,
                nodes,
                src_token=src_token,
                dst_token=dst_token,
                amount_in=amount_in,
                potentials=potentials,
            )
        except RealizationError as exc:
            candidate.status = "infeasible"
            candidate.note = str(exc)
            continue

        conflicts = check_one_arc_per_pool(route)
        if conflicts:
            candidate.status = "conflict"
            candidate.note = f"{len(conflicts)} pool(s) used twice"
            continue
        if len(route.slots) > MAX_SLOTS:
            candidate.status = "too_wide"
            candidate.note = f"{len(route.slots)} tokens > quoter limit {MAX_SLOTS}"
            continue
        if len(route.legs) > max_legs:
            candidate.status = "too_long"
            candidate.note = f"{len(route.legs)} legs > limit {max_legs}"
            continue
        candidate.route = route
        candidate.status = "ready"


def verify(
    candidates: CandidateSet,
    client: QuoterClient,
    *,
    amount_in: int,
    gas_price_wei: int = 0,
    dst_wei_per_eth: float = 0.0,
    gas_table: GasTable | None = None,
    min_gain_bp: float = 0.0,
) -> CandidateSet:
    """Quote every ready candidate in one call and rank them.

    Ranking is by *verified* output.  Gas only breaks ties, and only when the
    caller supplied a price and a way to value it in the output token -- a
    fixed cost per leg is not an element law and must not enter the convex
    core (§11.1).
    """
    # Quote whatever is new.  Ranking happens unconditionally below: this is
    # called more than once per route (candidates, then the direct floor, then
    # the refit), and returning early when nothing needs quoting used to leave
    # ranks stale from an earlier call -- including a rank of 1 that a solo
    # verification had handed to the refit candidate.  The winner is chosen by
    # rank, so a stale rank silently picks the wrong route.
    ready = [c for c in candidates.candidates if c.status == "ready" and c.route]
    if ready:
        outs = client.quote_routes(
            [c.route.wire_legs for c in ready],
            [amount_in] * len(ready),
            [c.route.dst_slot for c in ready],
        )
        for candidate, value in zip(ready, outs, strict=True):
            if value <= 0:
                candidate.status = "reverted"
                candidate.note = candidate.note or "quoter returned 0"
                continue
            candidate.verified_out = int(value)
            candidate.status = "ok"

    def score(candidate: Candidate) -> float:
        value = float(candidate.verified_out or 0)
        if gas_price_wei > 0 and dst_wei_per_eth > 0 and candidate.route:
            gas = plan_gas((leg.leg for leg in candidate.route.legs), gas_table)
            candidate.gas = gas
            value -= gas * gas_price_wei / 1e18 * dst_wei_per_eth
        return value

    def legs(candidate: Candidate) -> int:
        return len(candidate.route.legs) if candidate.route else 1_000

    # Prefer the simpler route when the outputs are indistinguishable.  Measured
    # on stablecoin pairs, the relaxation happily takes a 25-leg route to gain
    # 0.02 bp over a 1-leg one; any real gas price makes that strictly worse,
    # and §11.1 is explicit that a fixed per-arc cost belongs in candidate
    # selection rather than in the convex core.  Quantising the score is how a
    # tie becomes visible to the sort at all.
    for candidate in candidates.candidates:
        if not candidate.ok:
            candidate.rank = None  # a stale rank must never survive a re-verify

    usable = [c for c in candidates.candidates if c.ok]
    if not usable:
        return candidates
    best_score = max(score(c) for c in usable)
    # What one more leg has to earn.  Gas is the floor -- `score` has already
    # subtracted it, so this is what the *next* leg costs -- but gas is not the
    # whole price of a leg.  Each one is another chance for the state to move
    # between quoting and inclusion, and that risk is not modelled anywhere, so
    # `min_gain_bp` stands in for it.
    #
    # Measured across leg counts at three sizes: what splitting buys is set by
    # trade size against pool depth, not by which assets are involved.  Every
    # pair plateaus by 3 legs at $10k; the same pairs still gain 31-38 bp at 12
    # legs at $1M.  The waste is the tail -- steps worth 0.017 to 0.13 bp --
    # and a floor per leg removes exactly that while leaving the rest.
    per_leg = leg_gas(ArcKind.SWAP_STABLE) * gas_price_wei / 1e18 * dst_wei_per_eth
    # `min_gain_bp` is charged once, not per leg.  A route executes as a single
    # transaction, so the price moving between the quote and inclusion hits all
    # of it together -- the risk does not scale with how many pools are
    # touched.  Charging it per leg billed a 12-leg route on a pair drifting
    # 21 bp some 131 bp, drove it to 2 legs, and threw away a 71 bp advantage
    # that was really there.  What extra legs do add is gas, which `score` has
    # already subtracted, and revert surface, which is not a price.
    #
    # So this is a threshold on the *gain*: an advantage smaller than what the
    # pair moves on its own is not an advantage, however many legs bought it.
    # Below it, the shorter route wins on the tie-break.
    tolerance = max(per_leg, abs(best_score) * min_gain_bp / 1e4,
                    abs(best_score) * TIE_FLOOR)

    def rank_key(candidate: Candidate) -> tuple:
        value = score(candidate)
        bucket = 0 if best_score - value <= tolerance else 1
        return (bucket, legs(candidate) if bucket == 0 else 0, -value)

    ranked = sorted(usable, key=rank_key)

    # Hard floor: never rank below a plain one-hop swap.  The tie-break should
    # already give this -- a direct candidate has one leg, so it wins any tie --
    # but "the router is never worse than a pool you could find by inspection"
    # is a promise, not an emergent property, so it is enforced rather than
    # assumed.
    floor = max(
        (c for c in usable if c.kind == "direct"),
        key=lambda c: c.verified_out or 0,
        default=None,
    )
    if floor is not None and ranked and (ranked[0].verified_out or 0) < (
        floor.verified_out or 0
    ):
        ranked.remove(floor)
        ranked.insert(0, floor)

    for position, candidate in enumerate(ranked, start=1):
        candidate.rank = position
    return candidates


def summary(candidates: CandidateSet, decimals: int, limit: int = 12) -> list[dict]:
    """Rows for the terminal and the JSON, best first, failures last."""
    def sort_key(candidate: Candidate):
        return (candidate.rank if candidate.rank else 10_000, candidate.label)

    rows = []
    best = candidates.best
    reference = float(best.verified_out) if best and best.verified_out else 0.0
    for candidate in sorted(candidates.candidates, key=sort_key)[:limit]:
        delta = None
        if candidate.verified_out and reference:
            delta = (candidate.verified_out / reference - 1.0) * 10_000
        rows.append(
            {
                "label": candidate.label,
                "kind": candidate.kind,
                "rank": candidate.rank,
                "status": candidate.status,
                "arcs": candidate.n_arcs,
                "legs": len(candidate.route.legs) if candidate.route else 0,
                "verified_out": (
                    None if candidate.verified_out is None else str(candidate.verified_out)
                ),
                "out": (
                    ""
                    if candidate.verified_out is None
                    else f"{candidate.verified_out / 10**decimals:,.6f}"
                ),
                "delta_bp": None if delta is None else round(delta, 2),
                "note": candidate.note,
                "certificate": candidate.certificate,
            }
        )
    return rows
