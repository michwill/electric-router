"""Candidate generation (spec §6).

The solved flow is the optimum of a *relaxation*, and three things make it
unfit to quote directly:

* clamped arcs look bottomless, so §2.3 predicts they are preferentially
  filled rather than probed away;
* reference prices are fitted, so real dislocations read as free money and the
  solver spreads across dozens of arcs chasing them;
* a pool may end up carrying flow on two arcs, which a view-only quoter cannot
  evaluate because it cannot see its own earlier leg.

So the model chooses *which pools*, and the on-chain quote chooses *between
candidates*.  Every generator below is a cheap re-solve -- a rank-1 change to
the active set -- and the multicall adjudicates.

Priority order matters when the budget truncates.  The pin sweep outranks the
drop candidates because §13.1's chord regression is explicit that the active
set is *identical* across the endpoint allocations, so no drop-an-arc candidate
can find the interior optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import ArcArrays
from .quoter import MAX_LEGS, MAX_SLOTS
from .realize import RealizedRoute, cancel_cycles, prune_dust
from .seed import k_shortest_paths
from .solve import Solution, active_set_solve
from .types import PoolArc

# Sparsification levels.  The relaxation routinely activates dozens of arcs
# chasing fitted dislocations; these ask "what if you only had k pools?" and
# double as §11.1's gas-sparsification candidates.
TOP_K = (1, 2, 3, 4, 6, 8, 12)
PIN_LADDER = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
# An arc carrying less than this fraction of the trade cannot change the
# outcome, but chasing it costs pivots.  Candidates are heuristics adjudicated
# by the quoter, so they are solved to this screen rather than to machine
# tolerance; the certified base solve is not.
MIN_FLOW_FRACTION = 1e-4
# A candidate is realised into the quoter's slot accumulator, one slot per
# distinct token it touches.  A support wider than that cannot be priced no
# matter how good it is, so counting nodes is a cheap, sound rejection: the
# realised slot count is never *fewer* than the distinct nodes carrying flow
# (conversions only add spokes).
#
# Predicting it from the relaxation's own width does not work.  The obvious
# argument -- pin/drop/repair perturb C0 by one arc, so they inherit its width
# -- is false: forbidding an arc makes the re-solve find a different and
# sometimes far narrower support.  Skipping those families on that reasoning
# cost 5.65 bp on a $100 USDC->USDT swap and 0.49 bp at $1,000.  So the family
# is stopped only once it has actually produced this many unrealisable
# candidates in a row, which costs a few pivots to learn and never guesses.
WIDE_STREAK = 2
# Candidates are heuristics; stopping early yields a feasible flow, not a
# broken one, and the quoter is what decides between them anyway.
CANDIDATE_PIVOTS = 60


@dataclass(slots=True)
class Candidate:
    label: str
    psi: np.ndarray
    certificate: bool
    reason: str = ""
    kind: str = "solve"
    n_arcs: int = 0
    modelled_loss: float = 0.0
    route: RealizedRoute | None = None
    verified_out: int | None = None
    status: str = "pending"
    note: str = ""
    rank: int | None = None
    gas: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.verified_out is not None


@dataclass(slots=True)
class CandidateSet:
    candidates: list[Candidate] = field(default_factory=list)
    skipped: int = 0
    # Node count of the relaxation when it was too wide for any perturbation of
    # it to be realisable; 0 when the pin/drop/repair families ran normally.
    skipped_wide: int = 0

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def best(self) -> Candidate | None:
        """The winner, by the rank `verify` assigned -- which is *not* simply
        the largest output: outputs within a hair of each other are the same
        answer, and the cheaper route to execute wins (§11.1)."""
        ranked = [c for c in self.candidates if c.ok and c.rank is not None]
        if ranked:
            return min(ranked, key=lambda c: c.rank)
        usable = [c for c in self.candidates if c.ok]
        return max(usable, key=lambda c: c.verified_out or 0) if usable else None


def _signature(psi: np.ndarray, tol: float = 1e-12) -> tuple:
    """Active set plus split rounded to 10 bp -- §6.2's dedup key."""
    total = psi.sum()
    if total <= 0:
        return ()
    return tuple(
        sorted(
            (int(k), round(float(psi[k] / total), 3))
            for k in np.flatnonzero(psi > tol)
        )
    )


def _pool_of(arcs: list[PoolArc]) -> np.ndarray:
    return np.array([a.pool.lower() for a in arcs], dtype=object)


def conflicting_pools(arcs: list[PoolArc], psi: np.ndarray) -> dict[str, list[int]]:
    """Pools carrying flow on more than one arc (decision 3)."""
    groups: dict[str, list[int]] = {}
    for k in np.flatnonzero(psi > 0):
        groups.setdefault(arcs[int(k)].pool.lower(), []).append(int(k))
    return {pool: idx for pool, idx in groups.items() if len(idx) > 1}


def generate(
    g: ArcArrays,
    arcs: list[PoolArc],
    src: int,
    dst: int,
    Psi: float,
    base: Solution,
    *,
    base_certificate: bool = False,
    seed: np.ndarray | None = None,
    max_candidates: int = 20,
    top_k: tuple[int, ...] = TOP_K,
    gas_floor: float = 0.0,
    max_legs: int = MAX_LEGS,
) -> CandidateSet:
    out = CandidateSet()
    seen: set[tuple] = set()

    streak: dict[str, int] = {}

    def exhausted(kind: str) -> bool:
        """Has this family produced only unrealisable candidates lately?"""
        return streak.get(kind, 0) >= WIDE_STREAK

    def width(psi: np.ndarray) -> int:
        """Distinct nodes carrying flow -- a lower bound on realised slots."""
        live = psi > 0
        if not live.any():
            return 0
        return int(np.unique(np.concatenate([g.tau[live], g.sig[live]])).size)

    def add(psi: np.ndarray, label: str, kind: str, certificate: bool, reason: str = "") -> bool:
        psi, _ = cancel_cycles(g.tau, g.sig, psi)
        # Before the dedup signature, so two candidates differing only by a
        # dust branch collapse into one instead of spending two verify slots.
        psi, _ = prune_dust(g.tau, g.sig, psi, src, dst)
        active = int(np.count_nonzero(psi > 0))
        if active == 0:
            return False
        key = _signature(psi)
        if not key or key in seen:
            return False
        seen.add(key)
        loss = float(np.sum(g.eps * psi) + np.sum(np.where(g.G > 0, psi**2 / (2 * g.G), 0.0)))
        out.candidates.append(
            Candidate(
                label=label, psi=psi, certificate=certificate, reason=reason,
                kind=kind, n_arcs=active, modelled_loss=loss,
            )
        )
        return True

    def resolve(forbidden: np.ndarray, label: str, kind: str, pinned=None) -> bool:
        """Re-solve, then repair pool conflicts rather than discarding them.

        Decision 3 allows a pool at most one arc per route, and the Laplacian
        knows nothing about that.  Repairing in place -- keep the arc carrying
        most, forbid its siblings, solve again -- turns what would be a wasted
        candidate into a usable one, and every generator gets it for free.
        """
        banned = forbidden.copy()
        for _ in range(3):
            # One warm-started active-set solve, not column generation.  The
            # base solve already priced out all m arcs, so a candidate is a
            # small perturbation of a known optimum: re-deriving the support
            # from scratch costs ~150 pivots and buys nothing, since a
            # restricted candidate cannot be certified anyway.
            solution = active_set_solve(
                g, src, dst, Psi, A0=warm, forbidden=banned, forced_upper=pinned,
                # §11.1: gas cannot enter the objective without making the
                # program mixed-integer, but it bounds it from outside.  An arc
                # carrying less value than its leg costs to execute cannot pay
                # for itself even if it were pure profit, so screening it out
                # is sound rather than heuristic -- and it is what stops a
                # small trade sprouting branches that gas would eat.
                min_flow=MIN_FLOW_FRACTION * Psi,
                gas_cost=gas_floor,
                maxit=CANDIDATE_PIVOTS, partial_ok=True,
            )
            if not solution.feasible:
                return False
            conflicts = conflicting_pools(arcs, solution.psi)
            if not conflicts:
                break
            for indices in conflicts.values():
                keep = max(indices, key=lambda k: solution.psi[k])
                for k in indices:
                    if k != keep and not (pinned and k in pinned):
                        banned[k] = True
        else:
            return False
        # Two ways to be unrealisable, and both are known before realising:
        # more distinct tokens than the quoter has slots, or more arcs than the
        # caller will accept legs (each arc is at least one leg).
        support = int(np.count_nonzero(solution.psi > 0))
        if width(solution.psi) > MAX_SLOTS or support > max_legs:
            # Solved, and unrealisable.  Adding it would spend a realise and a
            # slot in the verification batch to learn what the node count
            # already said.
            out.skipped += 1
            streak[kind] = streak.get(kind, 0) + 1
            return False
        streak[kind] = 0
        return add(solution.psi, label, kind, False, "RESTRICTED")

    # 1. the relaxation itself
    add(base.psi, "C0 full", "base", base_certificate)

    pools = _pool_of(arcs)
    base_active = np.flatnonzero(base.psi > 0)
    # Warm-start from the *circulation-free* support.  The raw optimum carries
    # flow on arcs that only exist to go round a negative-eps loop; they are
    # cancelled before execution anyway, and leaving them in the start set is
    # what makes every candidate re-solve churn through them again.
    acyclic, _ = cancel_cycles(g.tau, g.sig, base.psi)
    warm = np.flatnonzero(acyclic > 0)
    if warm.size == 0:
        warm = base_active

    # Per-family budgets.  Ordering alone is not enough: with many flagged
    # arcs the pin sweep alone is 3 x 7 = 21 candidates, which used to consume
    # the whole budget before sparsification ran.  That mattered because every
    # un-sparsified candidate inherits the relaxation's sprawl and blows the
    # quoter's leg limit -- measured on a 5M USDC->USDT swap, *all twenty*
    # candidates came back too_long and the router fell back to a single pool,
    # missing a 0.08 bp two-pool split.
    sparse_budget = max(6, int(max_candidates * 0.45))
    pin_budget = max(4, int(max_candidates * 0.30))

    # 2. sparsification, over the k cheapest *paths* rather than the k largest
    #    arcs.  Restricting to arbitrary arcs usually leaves src and dst
    #    disconnected and the re-solve is infeasible; a union of shortest paths
    #    is connected by construction.  `k = 1` is §6.2's `C_*`, the
    #    no-splitting fallback, and the ladder doubles as §11.1's gas move --
    #    it is also the only family that reliably fits the quoter.
    made = 0
    paths = k_shortest_paths(g, src, dst, k=max(top_k) if top_k else 6)
    union: set[int] = set()
    for k, path in enumerate(paths, start=1):
        union.update(int(a) for a in path)
        if k not in top_k and k != 1:
            continue
        forbidden = np.ones(g.m, bool)
        for index in union:
            forbidden[index] = False
        label = "C* best single path" if k == 1 else f"top {k} paths"
        made += bool(resolve(forbidden, label, "sparse"))
        if made >= sparse_budget:
            break

    # 2b. keep only the pools the relaxation liked best, but let the solver use
    #     any arc of those pools so it can still find a connected route.
    order = sorted(base_active, key=lambda k: -base.psi[k])
    ranked_pools: list[str] = []
    for k in order:
        if pools[k] not in ranked_pools:
            ranked_pools.append(pools[k])
    for k in top_k:
        if k >= len(ranked_pools) or made >= sparse_budget:
            continue
        keep = set(ranked_pools[:k])
        forbidden = np.array([pool not in keep for pool in pools], dtype=bool)
        made += bool(resolve(forbidden, f"top {k} pool{'s' if k > 1 else ''}", "sparse"))

    out.skipped_wide = width(base.psi)

    # 3. pin sweep on every active flagged arc (§6.3).  Still ahead of the drop
    #    candidates, because no drop candidate can find a chord interior.
    made = 0
    flagged_active = [int(k) for k in base_active if g.flagged[k]]
    flagged_active.sort(key=lambda k: -base.psi[k])
    for arc_index in flagged_active[:3]:
        star = float(base.psi[arc_index])
        for step in PIN_LADDER:
            pin = min(star * step, float(g.cap[arc_index]))
            if step > 0 and pin <= 0:
                continue
            made += bool(
                resolve(
                    np.zeros(g.m, bool),
                    f"pin {arcs[arc_index].note[:18]} x{step:g}",
                    "pin",
                    pinned={arc_index: pin},
                )
            )
            if made >= pin_budget or len(out) >= max_candidates or exhausted("pin"):
                break
        if made >= pin_budget or len(out) >= max_candidates or exhausted("pin"):
            break

    # 4. one arc per pool (decision 3) -- keep the largest, forbid the rest
    conflicts = conflicting_pools(arcs, base.psi)
    if conflicts:
        forbidden = np.zeros(g.m, bool)
        for indices in conflicts.values():
            keep_index = max(indices, key=lambda k: base.psi[k])
            for k in indices:
                if k != keep_index:
                    forbidden[k] = True
        resolve(forbidden, f"repair {len(conflicts)} pool conflict(s)", "repair")
        worst = max(conflicts.items(), key=lambda kv: len(kv[1]))
        for keep_index in sorted(worst[1], key=lambda k: -base.psi[k])[1:2]:
            alt = np.zeros(g.m, bool)
            for k in worst[1]:
                if k != keep_index:
                    alt[k] = True
            resolve(alt, f"repair alt {arcs[keep_index].note[:18]}", "repair")

    # 5. drop each active arc in turn (§6.2)
    for k in order[: max(0, max_candidates - len(out))]:
        forbidden = np.zeros(g.m, bool)
        forbidden[k] = True
        resolve(forbidden, f"drop {arcs[int(k)].note[:20]}", "drop")
        if len(out) >= max_candidates or exhausted("drop"):
            break

    out.candidates = out.candidates[:max_candidates]
    return out
