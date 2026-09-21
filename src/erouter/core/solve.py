"""Active-set solve and the optimality certificate (spec §5.4, §5.5).

The routing program is

    (P)  min_psi  sum_p [ eps_p psi_p + psi_p^2 / (2 G_p) ]
         s.t.     B^T psi = s_hat,   0 <= psi <= cap

-- a strictly convex QP over a network polyhedron.  Its dual is
`max_u  Psi (u_src - u_dst) - sum_p (G_p/2) (u_tau - u_sig - eps_p)_+^2`, whose
gradient is Kirchhoff's current law and whose Hessian is an ordinary graph
Laplacian.  So each active set is one linear solve, the combinatorics live
entirely in which arcs are on, and no line search or trust region is needed.

The element law `psi_p = G_p (u_tau - u_sig - eps_p)_+` is a diode in series with
a resistor: zero flow until the potential difference exceeds the fee.  That
threshold is the origin of sparsity -- real optima light up 3-10 pools out of a
thousand.

Follows §14's reference listing with the corrections §14 defers: connectivity
recomputed every pivot (§9.4), upper-bounded arcs carrying an arbitrary pinned
value rather than only their cap (so §6.3's sweep is the same code path), and
arcs forbidden outright (column generation, one-arc-per-pool repair).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import accel as _accel
from .graph import ArcArrays, component_of, laplacian
from .linalg import DEFAULT_SOLVER, SingularSystem
from .seed import build_adjacency, spfa

# §9.2 absolute, never relative: rho legitimately passes through zero, and this
# is far below any real fee (1e-9 is 1e-5 bp).
TOL = 1e-9
# Flow below this fraction of the trade cannot matter, but chasing it can keep
# the active set oscillating forever.  Only used as a fallback (see `solve`).
DEGENERACY_SCREEN = 1e-4
# How many repeated bases to tolerate *after* Bland's rule is on before calling it
# a cycle.  Bland changes the pivot sequence, so it deserves a few iterations to
# break out on its own; measured cycles repeat every 2 pivots and never recover.
CYCLE_PATIENCE = 3
#: OSQP's tolerances.  Tight, because this is an oracle rather than a guess and
#: the differential test has always held the active set to it at 1e-10.
QP_EPS = 1e-10
QP_MAX_ITER = 200_000
#: Support size past which the active set is outside its design range and the
#: QP path is worth asking.  §5.4's "30-60 out of 1000", taken at the top.
QP_ACTIVE_FLOOR = 60
#: Flow below this fraction of the trade is interior-point noise, not a leg.
QP_DUST = 1e-9
#: Whether `solve` asks at all.  Off until the swap is measured end to end:
#: a lower modelled loss is not a better quote by itself -- see `DAI -> WETH`
#: at $1M, where the objective runs *against* realised output -- so this waits
#: on a size ladder and an all-venue sweep rather than on the objective alone.
QP_RESCUE = False
#: `active_set_solve`'s own default, and the floor here.
DEFAULT_PIVOTS = 600
#: The base solve gets pivots in proportion to the graph, not a flat 600.
#
# 600 is plenty for a small trade and nowhere near enough for a large one,
# because the active set grows with the trade rather than with the graph.  On
# `WETH -> WBTC` over one 2,975-arc graph:
#
#     trade      Psi      inner      gap     pivots   active
#     $100k   0.01423  converged  2.1e-17       41       42
#     $1M     0.1586   converged  2.4e-17      124       98
#     $10M    1.634      PARTIAL  7.9e-04      600      357
#     $40M    4.794      PARTIAL  1.0e-02      600      437
#
# The large ones stop at *exactly* the cap.  Given room they run to the cycle
# detector instead -- 2,864 pivots at $10M -- and land on a much better flow:
# the answer goes 11,692,969,838 to 12,406,125,434 and the spread over a 1e-8
# size sweep falls from 399 bp to 74.  Twice the arc count is enough for that
# everywhere measured, and costs nothing on the small trades that finish in
# 41 pivots and never reach it.
#
# It does not make the solve converge.  These exits are still PARTIAL, now on
# the cycle detector rather than the budget, and the reported `gap` is worse
# for it.  What it buys is a better incumbent before the cycling stops it.
BASE_PIVOT_FACTOR = 2
# How many times a cycling basis is perturbed before the patience above applies.
#
# Degeneracy is what makes the basis repeat: several arcs tie on the quantity the
# pivot rule ranks, so which one moves is arbitrary and the sequence can return
# to where it was.  The textbook remedy is to break every tie in advance by
# shifting each arc's cost by a distinct vanishing amount, and it is what the
# Rust solver gets by accident -- its rank-1 Cholesky update leaves a different
# rounding in `u` than a fresh factorisation, and that noise is enough to fall
# out of the cycle.  Measured: with rank-1 disabled the two agree to 1.3e-9 and
# cycle identically; with it on, Rust converges on solves where Python gives up.
#
# So do deliberately what it does by luck.  Deterministic, so two runs at one
# block still agree to the wei.
# Eight rather than four, because four was tuned on the synthetic universes and
# a real one needs more.  Replaying the deployed Router's own pairs at their
# median size and two decades above it, plus the sweep's, 39 cases at one
# block: 2 better (+9.67 bp on USDC->WBTC at $2M, +0.04 on WETH->USDC 800), 36
# unchanged, 1 worse by 0.02 bp.  At four the reference was giving up -- exiting
# PARTIAL where the port converged, on 98 of 208 candidate re-solves, and losing
# 92 of those on objective.
#
# Not more than eight: the ladder multiplies by ten a round, so eight reaches
# 1e-4 of the `eps` spread -- around a thousandth of a basis point, still far
# under anything a quote can see -- and twelve would reach 1.0, which is not a
# perturbation of the problem but a different one.
#
# And eight *here only*.  `rust/src/solve.rs` stays at four, because it does not
# need the ladder in the first place and the later rounds cost it: the same
# replay moved it 35.5 bp the wrong way on one case and nothing anywhere else.
# The two solvers need different amounts of shaking to reach the same answer,
# which is the whole premise of this constant existing.
PERTURB_ROUNDS = 8
# The first shift, relative to the largest `eps` in the graph.  Far below `TOL`
# and further below a basis point, so the perturbed problem is the same problem;
# each round multiplies by ten in case the first was inside the noise it meant
# to clear.
PERTURB_SCALE = 1e-11
#: Steps of iterative refinement on the flow residual, at a feasible exit.
#
# One is nearly always enough -- measured, it takes the worst live residual from
# 7.8e-06 to 1.8e-08 -- and the second costs one solve against a factorisation
# already formed and is refused the moment it stops helping.  Zero restores the
# behaviour before this existed, which is what the replay compares against.
REFINE_ROUNDS = 2
# When two candidates score within this *relative* distance of each other, treat
# them as tied and take the lower index.
#
# Degenerate ties are real here: two arcs carrying exactly the same flow score
# identically in exact arithmetic, and the last bits that separate them are
# whatever the linear solve happened to leave.  Comparing with `>` then lets
# rounding pick the pivot, and the two sides of the port round differently --
# measured on a pinned re-solve where `psi` was -0.499999999999865 against
# -0.499999999998981 for the same pair, the arcs swapping rank between
# implementations and the solves ending in different places.
#
# Relative with no absolute floor, deliberately: the four pivot categories score
# in two different units (flow and potential), and a floor large enough to cover
# `psi ~ 1` would swallow genuinely distinct candidates in `rho`.  At 1e-9 the
# window is ~500x the observed noise and still far below anything a quote can
# see.
PIVOT_TIE = 1e-9

# The compiled solve is **opt-in**, and the reason it stays that way has
# changed -- read this before assuming the old one still holds.
#
# It was the degenerate tail.  Replaying 94 problems off live quotes, the port
# took the identical pivot count on 93 and agreed on feasibility on all 94, at
# 23.8us per pivot against numpy's 130us -- but where the reference itself came
# back PARTIAL or cycled, the two wandered apart and the quote landed hundreds
# of bp away.  "The answer depends on which solver ran" is not shippable.
#
# Three defects were behind that, all of them in code a converging problem
# never reaches: a Cholesky factor reused across a basis it did not describe, a
# cycle-breaking perturbation that reached the drop rule but not the
# right-hand side, and ties settled by whichever arc the solve rounded higher
# (`PIVOT_TIE`, above).  `test_the_degenerate_tail_agrees` now holds the two to
# exact agreement on reason *and* flow over 216 solves, 144 of them PARTIAL,
# and `test_over_constrained_pins_agree` over 72 pinned re-solves -- the sweep
# that found the defects ran 676 of those and is in `docs/performance.md`.
#
# What that does not settle: both are synthetic, and the claim above was
# measured on live quotes at theta in the hundreds of percent. Flipping the
# default wants that replay repeated, not a synthetic stand-in for it.  Until
# then `EROUTER_ACCEL=1` opts in.
_ACCEL_ON = os.environ.get("EROUTER_ACCEL", "") == "1"


def accel_in_use() -> bool:
    """Whether `solve` will really take the compiled path.

    Both halves, because a caller asking which solver answered gets the wrong
    answer from either alone: the module can import while the opt-in is off.
    """
    return _ACCEL_ON and _accel.available()


def _steepest_pick(mask: np.ndarray, score: np.ndarray) -> int:
    """Most-violating candidate; ties -- exact or within `PIVOT_TIE` -- go to
    the lowest index."""
    where = np.flatnonzero(mask)
    best = float(np.max(score[where]))
    tied = where[score[where] >= best - PIVOT_TIE * abs(best)]
    return int(tied[0])


def _bland_pick(mask: np.ndarray, score: np.ndarray) -> int:
    """Bland's rule -- guarantees termination if a basis ever repeats."""
    return int(np.flatnonzero(mask)[0])


@dataclass(slots=True)
class Solution:
    psi: np.ndarray
    u: np.ndarray
    A: np.ndarray
    U: np.ndarray
    psi_upper: np.ndarray
    rho: np.ndarray
    pivots: int = 0
    feasible: bool = True
    reason: str = ""

    @property
    def active(self) -> np.ndarray:
        return np.flatnonzero(self.psi > 0)

    def objective(self, g: ArcArrays) -> float:
        """Modelled value loss: the diode term plus the resistor term."""
        psi = self.psi
        with np.errstate(divide="ignore", invalid="ignore"):
            impact = np.where(g.G > 0, psi**2 / (2 * g.G), 0.0)
        return float(np.sum(g.eps * psi) + np.sum(impact))

    def reduced(self, g: ArcArrays) -> np.ndarray:
        """Gradient of the Lagrangian: `eps_p + psi_p/G_p - (u_tau - u_sig)`.

        Not the same as `rho`.  `rho = u_tau - u_sig - eps` is the voltage
        driving the arc, and an *active* arc has `rho = psi/G > 0` by the element
        law (M6), so `psi * rho` is not the complementarity product.  This is:
        zero on every free arc, >= 0 at zero, <= 0 at the cap.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(g.G > 0, self.psi / g.G, 0.0)
        return slope - self.rho

    def loss_split(self, g: ArcArrays) -> tuple[float, float]:
        with np.errstate(divide="ignore", invalid="ignore"):
            impact = np.where(g.G > 0, self.psi**2 / (2 * g.G), 0.0)
        return float(np.sum(g.eps * self.psi)), float(np.sum(impact))


def _reconnect(g, src: int, dst: int, allowed: np.ndarray, adj) -> np.ndarray | None:
    """Cheapest directed `src -> dst` path over `allowed`, or None.

    Directed where `component_of` is not: `psi >= 0` runs `tau -> sig`.
    """
    banned = set(np.flatnonzero(~allowed).tolist())
    found = spfa(g, src, dst, adj, banned_arcs=banned)
    if found.negative_cycle:
        shifted = g.eps - min(0.0, float(g.eps.min()))
        found = spfa(g, src, dst, adj, banned_arcs=banned, weights=shifted)
    return np.asarray(found.arcs, dtype=np.int64) if found.found else None


def _why_unreachable(g, src: int, dst: int, Psi: float) -> str:
    """Why no flow reaches `dst`, in terms of what the pools can actually do.

    "src not connected to dst through the active set" is true and useless.  It is
    what a user sees when the pools *are* there and simply cannot carry the trade
    -- an arc capped at a tenth of the size, a pool holding 0.0026 of a coin
    against an API reporting $333,401 -- which sends someone hunting for a missing
    pool rather than looking at the empty one in front of them.

    Two cuts are worth naming because they are the ones a user can act on: what
    leaves the source, and what enters the destination.  Checking the source alone
    reports nothing when the drained pool is the only way *in*.  Anything subtler
    is a real cut somewhere in the middle, and "not connected" is then honest.

    Capacity belongs to the whole graph rather than the active set, so caps are
    summed over every arc across the cut, including ones already set aside.
    """
    import numpy as _np

    for side, arcs, phrase in (
        ("source", _np.flatnonzero(g.tau == src), "out of the source"),
        ("destination", _np.flatnonzero(g.sig == dst), "into the destination"),
    ):
        if arcs.size == 0:
            return f"no pool trades the {side} token"
        caps = g.cap[arcs]
        if _np.isinf(caps).any():
            continue
        room = float(caps.sum())
        if room < Psi:
            share = room / Psi if Psi else 0.0
            return (f"the pools {phrase} can carry {share:.3%} of this size "
                    f"-- their quotes stop rising beyond that")
    return "src not connected to dst through the active set"


def active_set_solve(
    g: ArcArrays,
    src: int,
    dst: int,
    Psi: float,
    *,
    A0: np.ndarray | None = None,
    forced_upper: dict[int, float] | None = None,
    forbidden: np.ndarray | None = None,
    tol: float = TOL,
    maxit: int = 600,
    solver=None,
    min_flow: float = 0.0,
    gas_cost: float = 0.0,
    partial_ok: bool = False,
) -> Solution:
    """Solve (P) restricted to the non-forbidden arcs.

    `min_flow` refuses entry to an arc that would carry less than that much value.
    With fitted reference prices, dozens of arcs sit within a hair of the diode
    threshold and oscillate in and out carrying dust -- 150 pivots to move 0.01%
    of the trade.  Screening them approximates the *tie*, not the answer: an arc
    below the screen cannot change the output measurably.  Keep it at 0 for the
    certified solve, where exactness is the point.

    `forced_upper` pins an arc at a given flow by moving it into the `U` set,
    where it contributes `-B_U^T psi_U` to the right-hand side -- exactly the
    mechanism a saturated capacity uses, so §6.3's pin-and-resolve sweep is a
    keyword argument rather than a new branch.
    """
    # The Rust solve, when it is installed and nothing has asked for a specific
    # linear solver.  One crossing per solve, not per pivot.  It is a port of
    # exactly this function -- `tests/test_accel_differential.py` differs the two,
    # and both against OSQP -- so the only thing that changes is how long it takes.
    if solver is None and _ACCEL_ON and _accel.available():
        got = _accel.solve_arrays(
            g, src, dst, Psi, a0=A0, forbidden=forbidden, pinned=forced_upper,
            tol=tol, maxit=maxit, min_flow=min_flow, gas_cost=gas_cost,
            partial_ok=partial_ok,
        )
        if got is not None:
            return Solution(
                got["psi"], got["u"], got["active"], got["upper"],
                got["psi_upper"], got["rho"],
                int(got["pivots"]), feasible=bool(got["feasible"]),
                reason=str(got["reason"]),
            )

    solver = solver or DEFAULT_SOLVER
    m, n = g.m, g.n_nodes
    forbidden = np.zeros(m, bool) if forbidden is None else np.asarray(forbidden, bool)
    pinned = dict(forced_upper or {})

    s_hat = np.zeros(n)
    s_hat[src] += Psi
    s_hat[dst] -= Psi

    A = np.zeros(m, bool)
    U = np.zeros(m, bool)
    psi_upper = np.zeros(m)

    for arc, value in pinned.items():
        U[arc] = True
        psi_upper[arc] = value

    if A0 is not None:
        A[np.asarray(A0)] = True
    A &= ~forbidden & ~U
    if not A.any():
        # §5.4 warm start: all arcs active is the pure (f', f'') answer and is
        # exact in the small-trade limit; everything after corrects for the
        # diode combinatorics.
        A = ~forbidden & ~U

    psi = np.zeros(m)
    u = np.zeros(n)
    rho = np.zeros(m)
    pivots = 0
    seen_bases: set[tuple] = set()
    bland = False
    cycles = 0

    reseeded = False

    def polished(u, psi, rounds=None):
        """One or two steps of iterative refinement, on the *flow* residual.

        The claim below -- that every iterate satisfies conservation exactly
        because `u` solves the Laplacian system -- is true in exact arithmetic
        and false in double precision the moment `G` is wide.  `psi = G(du -
        eps)` multiplies any error in a potential by `G`, and once a venue of
        near-linear arcs is in the graph `G` spans eleven orders: a potential
        good to 1e-15 gives a flow that misses conservation by 1e-5, and §12.4
        refuses a quote whose arithmetic is otherwise fine.

        The correction is the textbook one and costs one solve against a
        factorisation already formed: `B^T dpsi = L du`, so solving `L du = s -
        B^T psi` on the free nodes and re-deriving `psi` removes exactly the
        amplified error.  Applied only while it *reduces* the imbalance, so a
        step that would trade conservation for something else is dropped.
        """
        if rounds is None:
            rounds = REFINE_ROUNDS

        def snap(flow):
            # §5.4's cleanup, and it belongs *inside* the refinement: snapping
            # afterwards puts back an imbalance of up to `tol` per arc, which on
            # the graph this exists for is the whole error again.
            return np.where(np.abs(flow) < tol, 0.0, flow)

        psi = snap(psi)
        if not rounds or not keep.size or not idx.size:
            return u, psi

        # Rebuilt from the basis being returned rather than reused from the
        # loop: `keep` is recomputed at the top of an iteration and several
        # pivot branches `continue` before the Laplacian is rebuilt, so the two
        # can disagree by an arc at an exit -- which numpy reports as a
        # dimension mismatch inside `solve`, and which read as a routing refusal
        # on three of seventeen live cases.
        matrix = laplacian(g.tau[idx], g.sig[idx], g.G[idx], n, keep)

        def imbalance(flow):
            net = np.zeros(n)
            np.add.at(net, g.tau, flow)
            np.subtract.at(net, g.sig, flow)
            return s_hat - net

        for _ in range(rounds):
            residual = imbalance(psi)
            before = float(np.max(np.abs(residual[keep]))) if keep.size else 0.0
            if before == 0.0:
                break
            try:
                delta = solver.solve(matrix, residual[keep])
            except SingularSystem:
                break
            # The correction is added to the *flow*, never to `u` and back.
            #
            # `psi = G(u_tau - u_sig - eps)` and the potentials carry a large
            # common offset -- 5.86e3 on the graph this was found on, where one
            # ULP is 9.1e-13.  The corrections themselves are ~2.0e-08, but what
            # the flow depends on is the difference between two neighbouring
            # ones: 6.5e-13 there, *below* that ULP.  So `(u + du)_tau -
            # (u + du)_sig` rounds to `u_tau - u_sig` and the whole correction
            # is annihilated by the subtraction -- `dpsi` came out exactly 0.0
            # on an arc the system said should move by 2.1e-08.
            #
            # Differencing `du` first keeps every digit of it, because `du` has
            # no offset to lose them to.
            step = np.zeros(n)
            step[keep] = delta
            trial = psi.copy()
            trial[idx] += g.G[idx] * (step[g.tau[idx]] - step[g.sig[idx]])
            trial[U] = psi_upper[U]
            trial[~(comp[g.tau] & comp[g.sig])] = 0.0
            trial = snap(trial)
            trial_u = u.copy()
            trial_u[keep] += delta
            after = float(np.max(np.abs(imbalance(trial)[keep])))
            if not (after < before):
                break
            u, psi = trial_u, trial
        return u, psi
    adjacency = None
    # One turn back per arc, which is what bounds the repair below.
    sent_to_zero = np.zeros(m, bool)
    readmitted = np.zeros(m, bool)
    # `eps` the pivot rule actually sees.  Identical to the graph's until a cycle
    # forces a perturbation, and rebound to a shifted copy after.
    eps = g.eps
    perturbed = 0
    for _ in range(maxit):
        idx = np.flatnonzero(A)

        comp = component_of(dst, g.tau[idx], g.sig[idx], n)
        if not comp[src] and Psi != 0:
            # The *active set* being disconnected is not the graph being
            # disconnected.  An arc that saturates moves to `U`, and if it was
            # the one joining src to dst the set left behind joins nothing --
            # which is a starting point, not a verdict.  §5.4 admits every arc at
            # initialisation for exactly this reason; doing it again here is the
            # same step, taken when a pivot rather than the caller emptied the
            # set.  Three ways in: a stale warm start whose single arc caps out
            # at a larger size, a cheap capped arc in parallel with a dearer open
            # one, and the drop rule stripping negative arcs until the last one
            # across the cut points backwards.  Restarting replays a
            # deterministic descent, so admit a path instead; a dropped arc gets
            # one turn back, which terminates.
            if adjacency is None:
                adjacency = build_adjacency(g.tau, g.sig, n)
            open_arcs = ~forbidden & ~U & ~readmitted
            path = _reconnect(g, src, dst, open_arcs & ~sent_to_zero, adjacency)
            if path is None:
                path = _reconnect(g, src, dst, open_arcs, adjacency)
            if path is not None and not A[path].all():
                A[path] = True
                readmitted[path[sent_to_zero[path]]] = True
                pivots += 1
                continue
            candidates = ~forbidden & ~U
            if not reseeded and candidates.any() and not np.array_equal(candidates, A):
                A, reseeded = candidates.copy(), True
                continue
            return Solution(
                np.zeros(m), np.zeros(n), A, U, psi_upper, np.zeros(m), pivots,
                feasible=False, reason=_why_unreachable(g, src, dst, Psi),
            )
        u_idx = np.flatnonzero(comp)
        keep = u_idx[u_idx != dst]

        rhs = s_hat.copy()
        if idx.size:
            fee_flow = g.G[idx] * eps[idx]
            np.add.at(rhs, g.tau[idx], fee_flow)
            np.subtract.at(rhs, g.sig[idx], fee_flow)
        uidx = np.flatnonzero(U)
        if uidx.size:
            np.subtract.at(rhs, g.tau[uidx], psi_upper[uidx])
            np.add.at(rhs, g.sig[uidx], psi_upper[uidx])
            outside = ~comp[g.tau[uidx]] | ~comp[g.sig[uidx]]
            if outside.any():
                # An arc at its upper bound whose endpoints have left `dst`'s
                # component cannot deliver what it is pinned to carry.  Whether
                # that is fatal depends on *why* the arc is there.
                #
                # A caller's pin (§6.3's sweep) is the candidate's whole point,
                # so a detached one makes that candidate infeasible and
                # generation drops it.  An arc that merely *saturated* during
                # pivoting is different -- nothing asked for it to be at its cap
                # -- so releasing it is the pivot the loop would have made had it
                # looked.  Refusing instead threw away reachable routes, and
                # failed the whole quote when every candidate hit it.
                stray = uidx[outside]
                loose = np.array([j for j in stray if int(j) not in pinned],
                                 dtype=np.int64)
                if loose.size:
                    U[loose] = False
                    psi_upper[loose] = 0.0
                    pivots += 1
                    continue
                return Solution(
                    np.zeros(m), np.zeros(n), A, U, psi_upper, np.zeros(m), pivots,
                    feasible=False, reason="a pinned arc is detached from the active network",
                )

        u = np.zeros(n)
        if keep.size:
            L = laplacian(g.tau[idx], g.sig[idx], g.G[idx], n, keep)
            try:
                u[keep] = solver.solve(L, rhs[keep])
            except SingularSystem as exc:
                return Solution(
                    np.zeros(m), np.zeros(n), A, U, psi_upper, np.zeros(m), pivots,
                    feasible=False, reason=f"singular Laplacian: {exc}",
                )

        psi = np.zeros(m)
        psi[U] = psi_upper[U]
        if idx.size:
            psi[idx] = g.G[idx] * (u[g.tau[idx]] - u[g.sig[idx]] - eps[idx])
        # §9.4: nodes outside `dst`'s component carry zero flow by construction.
        # Without this an arc with both ends outside gets u = 0 at both, so a
        # favourable eps yields psi = -G*eps > 0 -- flow conjured from nothing,
        # satisfying no conservation law and impossible to order for execution.
        psi[~(comp[g.tau] & comp[g.sig])] = 0.0
        rho = u[g.tau] - u[g.sig] - eps


        # A repeated basis means the pivot sequence is going in circles.  The
        # first remedy is Bland's rule (lowest index), which guarantees
        # termination for a simplex method on a standard LP -- but this is a
        # bound-constrained QP with four pivot categories tried in a fixed order,
        # and Bland's guarantee does not transfer to that structure.  Measured on
        # USDC->CRV $100k, the basis repeated a further 410 times after Bland
        # switched on, every one a period-2 flip of the same pair.
        #
        # So once Bland is on and the basis is *still* repeating, stop and say so.
        # `solve` answers by screening out the oscillating dust arcs and accepting
        # the incumbent, which is a valid flow: every iterate satisfies
        # conservation exactly, only optimality is incomplete.
        signature = (A.tobytes(), U.tobytes())
        if signature in seen_bases:
            if bland and perturbed < PERTURB_ROUNDS:
                # Break the ties that let the basis return here.  A distinct
                # shift per arc, monotone in index so it is reproducible, and
                # small enough that the perturbed optimum is the real one to
                # far more digits than any quote can carry.
                perturbed += 1
                scale = PERTURB_SCALE * (10.0 ** (perturbed - 1))
                spread = float(np.max(np.abs(g.eps))) if g.m else 0.0
                eps = g.eps + scale * max(spread, 1.0) * (
                    np.arange(1, g.m + 1, dtype=float) / g.m)
                seen_bases.clear()
                continue
            if bland:
                cycles += 1
                if cycles >= CYCLE_PATIENCE:
                    # Same contract as running out of `maxit`: the caller
                    # decides whether an unconverged flow is usable.  The
                    # screened retry passes `partial_ok`, and refusing it there
                    # turns a route that used to be quoted into no route at all.
                    if partial_ok:
                        u, psi = polished(u, psi)
                        return Solution(psi, u, A, U, psi_upper, rho, pivots,
                                        feasible=True, reason="PARTIAL")
                    return Solution(
                        psi, u, A, U, psi_upper, rho, pivots, feasible=False,
                        reason=f"no convergence: cycling under Bland's rule "
                               f"after {pivots} pivots",
                    )
            bland = True
        seen_bases.add(signature)

        pick = _bland_pick if bland else _steepest_pick

        negative = A & (psi < -tol)
        if negative.any():
            leaving = pick(negative, -psi)
            A[leaving] = False
            sent_to_zero[leaving] = True
            pivots += 1
            continue

        over = A & (psi > g.cap + tol)
        if over.any():
            j = pick(over, psi - g.cap)
            A[j] = False
            U[j] = True
            psi_upper[j] = g.cap[j]
            pivots += 1
            continue

        Z = ~A & ~U & ~forbidden
        entering = Z & (rho > tol)
        if min_flow > 0 and entering.any():
            entering &= (g.G * rho) > min_flow
        if gas_cost > 0 and entering.any():
            # What admitting this arc is actually worth.  At reduced cost `rho`
            # it settles at `psi = G rho`, and the objective falls by
            # `G rho^2 / 2` -- so that, not the flow through it, is what has to
            # beat the gas of one more leg.  Screening on flow alone is far too
            # loose: measured on a $1,000 trade, 31 legs each cleared a flow floor
            # while together burning 3.25M gas to gain a fraction of a basis point.
            entering &= (0.5 * g.G * rho * rho) > gas_cost
        if entering.any():
            A[pick(entering, rho)] = True
            pivots += 1
            continue

        releasable = U & (rho < -tol)
        for arc in pinned:
            releasable[arc] = False  # a pinned arc stays pinned
        if releasable.any():
            j = pick(releasable, -rho)
            U[j] = False
            A[j] = True
            pivots += 1
            continue

        break
    else:
        # Every iterate satisfies conservation exactly -- `u` solves the Laplacian
        # system with the conservation right-hand side -- so only *optimality* is
        # incomplete, never feasibility.  A candidate is a heuristic the quoter
        # adjudicates, so an unconverged one is still a valid route to offer it.
        if not partial_ok:
            return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=False,
                            reason=f"no convergence in {maxit} pivots")
        u, psi = polished(u, psi)
        return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=True,
                        reason="PARTIAL")

    u, psi = polished(u, psi)
    return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=True)


def optimality_gap(
    solution: Solution, g: ArcArrays, available: np.ndarray, dst_node: int,
    tol: float = TOL,
) -> float:
    """How much objective is still on the table at this point (§5.5).

    An arc held at zero whose reduced cost `rho` is positive wants flow.  Admit it
    and it settles where the element law puts it, `psi = min(G rho, cap)`, taking
    the objective down by `rho psi - psi^2 / (2 G)`.  Summing that over every arc
    that wants in bounds the total remaining improvement from above: they are
    priced against the *current* potentials, and admitting one moves the
    potentials against the others, so the true gain is no larger.

    The cap is not a detail.  Without it the term is `G rho^2 / 2`, which is what
    this returns for an uncapped arc and what it returned for every arc before a
    venue of narrow arcs arrived: a Uniswap v3 tick has a `G` of 1e9 and a
    capacity of a few tens of thousands of dollars, so the flow it would settle
    at is four orders past what it can carry and the bound built out of it says
    8.8e9 where the objective is 1e-4.  Nothing was wrong with the solve -- 2,369
    capped arcs were being credited with an improvement none of them could
    deliver, and a certificate was refused on the strength of it.

    Active arcs need no term: `psi` is computed as `G rho` for them, so the
    element law holds identically and their contribution is zero by construction.

    This is what makes the certificate a statement about the answer instead of
    about the loop that produced it -- a solve that stopped early because it was
    cycling is still optimal if nothing wants in.

    Only arcs the trade can actually reach are counted.  §9.4 leaves `u = 0` at
    both ends of anything outside `dst`'s component, so a favourably dislocated
    arc out there shows `rho = -eps > 0` and appears to want flow that no route
    could carry; counting those put the bound five orders above the objective.

    `tol` is the solver's own entering threshold, deliberately the same one: an
    arc the pivoting would not admit is not an arc that wants in, and counting it
    would build a gap out of arithmetic noise.
    """
    live = np.flatnonzero(solution.psi > 0)
    if live.size == 0:
        return 0.0
    reach = component_of(dst_node, g.tau[live], g.sig[live], g.n_nodes)
    connected = reach[g.tau] & reach[g.sig]
    wants_in = available & connected & (solution.psi <= 0) & (solution.rho > tol)
    if not wants_in.any():
        return 0.0
    G = g.G[wants_in]
    rho = solution.rho[wants_in]
    settles = np.minimum(G * rho, g.cap[wants_in])
    return float(np.sum(rho * settles - settles**2 / (2.0 * G)))


def qp_solution(
    g: ArcArrays,
    src: int,
    dst: int,
    Psi: float,
    *,
    forbidden: np.ndarray | None = None,
    forced_upper: dict[int, float] | None = None,
    tol: float = TOL,
) -> Solution | None:
    """`(P)` handed to OSQP whole, or `None` where OSQP is not installed.

    An optional fast path, on the same terms `scipy` has in `core/linalg.py`:
    imported inside the function so `erouter.core` still runs on numpy alone
    under Pyodide, where neither wheel exists.  `tests/test_purity.py` holds
    that rule.

    Why bother, given §5.4 proves the active set finite: it was sized for
    "final |S| ~ 30-60 out of 1000" (`docs/quadratic-flow-router.md`), and a
    $10M trade over curve+v2+v3+v4 puts **357 to 437 arcs of 2,975** in the
    active set.  Seven times the design point, and the method does not degrade
    gracefully there -- measured on one `WETH -> WBTC` graph:

        trade   active set                     OSQP            OSQP time
        $1M     converged gap=2.4e-17 0.000358  0.000228 -36%    0.6 s
        $10M    PARTIAL   gap=7.9e-04 0.508     0.0225   -96%    1.0 s
        $40M    PARTIAL   gap=1.0e-02 1.026     0.191    -81%    2.4 s

    The $1M row is why this is not merely a speed question: ours reports §5.5
    optimality to machine precision and OSQP finds 36% less loss on the same
    graph, so at this scale the certificate is *wrong* rather than weak.

    The flows are checked, not assumed: KCL residual lands at 1e-13, no
    component below -1e-9, and `cancel_cycles` removes only dust -- 13, 11 and
    3 cycles keeping 99.4% to 100% of the flow, moving the objective in the
    seventh digit.  The gain is not circulation.
    """
    try:                                    # optional, and absent in a browser
        import osqp
        from scipy import sparse
    except ImportError:
        return None

    m, n = g.m, g.n_nodes
    banned = (np.zeros(m, bool) if forbidden is None
              else np.asarray(forbidden, bool))
    pinned = dict(forced_upper or {})

    rows = np.empty(2 * m, dtype=np.int64)
    cols = np.empty(2 * m, dtype=np.int64)
    vals = np.empty(2 * m, dtype=float)
    idx = np.arange(m)
    rows[0::2], cols[0::2], vals[0::2] = g.tau, idx, 1.0
    rows[1::2], cols[1::2], vals[1::2] = g.sig, idx, -1.0
    inc = sparse.csc_matrix((vals, (rows, cols)), shape=(n, m))

    s_hat = np.zeros(n)
    s_hat[src] += Psi
    s_hat[dst] -= Psi

    big = 1e12 * max(abs(Psi), 1.0)
    hi_arc = np.where(np.isfinite(g.cap), g.cap, big).astype(float)
    lo_arc = np.zeros(m)
    hi_arc[banned] = 0.0                    # a banned arc may carry nothing
    for k, at in pinned.items():            # §6.3's pin, as an equality
        lo_arc[int(k)] = hi_arc[int(k)] = float(at)

    A = sparse.vstack([inc, sparse.eye(m, format="csc")], format="csc")
    lo = np.concatenate([s_hat, lo_arc])
    hi = np.concatenate([s_hat, hi_arc])
    # `psi^2 / (2 G)` against OSQP's own half, so `P = diag(1/G)`.  A clamped
    # arc has `G = inf` and contributes nothing quadratic, which is the
    # zero-curvature limit §2.3 admits.
    inv = np.where(g.G > 0, 1.0 / np.where(np.isfinite(g.G), g.G, np.inf), 0.0)
    P = sparse.diags(np.nan_to_num(inv, posinf=0.0), format="csc")

    problem = osqp.OSQP()
    problem.setup(P=P, q=np.asarray(g.eps, float), A=A, l=lo, u=hi,
                  verbose=False, eps_abs=QP_EPS, eps_rel=QP_EPS,
                  max_iter=QP_MAX_ITER, polishing=True)
    res = problem.solve()
    if "solved" not in str(getattr(res.info, "status", "")):
        return None

    psi = np.clip(np.asarray(res.x, float), 0.0, hi_arc)
    # (P) does not forbid circulation and the router cannot execute one, so the
    # flow is projected onto the acyclic part before anyone sees it.  Skipping
    # this made every quote die in `realize` with "the active arcs contain a
    # cycle": OSQP leaves a handful of dust loops -- 13, 11 and 3 on the three
    # sizes measured -- and they cost nothing to remove, keeping 99.4% to 100%
    # of the flow and moving the objective in the seventh digit.  The active
    # set never produces them, which is why nothing downstream tolerates them.
    # Dust first, then cycles.  An interior-point answer carries ~1e-15 on
    # arcs it does not really use, and `cancel_cycles` nets the *flow* around a
    # two-cycle without clearing those residues -- so both directions still read
    # as carrying, `realize.topological_nodes` sees a cycle in the node graph
    # and refuses the route.  The active set leaves exact zeros there, which is
    # why nothing downstream had to care before.
    psi[psi < QP_DUST * max(abs(Psi), 1.0)] = 0.0
    from .realize import cancel_cycles
    psi, _spun = cancel_cycles(g.tau, g.sig, psi)
    psi[psi < QP_DUST * max(abs(Psi), 1.0)] = 0.0
    # Potentials from the duals of the conservation rows.  Sign is fixed by
    # the KKT condition the active set writes: an arc carrying flow between
    # its bounds has `u_tau - u_sig - eps = psi / G`, so the orientation is
    # checked against that rather than assumed.
    y = np.asarray(res.y, float)[:n]
    u = -y
    carrying = psi > tol
    if carrying.any():
        want = psi[carrying] / np.where(g.G[carrying] > 0, g.G[carrying], np.inf)
        got = u[g.tau[carrying]] - u[g.sig[carrying]] - g.eps[carrying]
        if np.linalg.norm(got - want) > np.linalg.norm(-got - want):
            u = y
    rho = u[g.tau] - u[g.sig] - np.asarray(g.eps, float)
    upper = psi >= hi_arc - tol
    active = carrying & ~upper
    return Solution(psi, u, active, upper, hi_arc, rho,
                    int(getattr(res.info, "iter", 0)), feasible=True,
                    reason="OSQP")


def price_out(
    u: np.ndarray, g: ArcArrays, in_S: np.ndarray, tol: float = TOL
) -> np.ndarray:
    """Arcs outside `S` that would improve the objective.  Empty ⟹ optimal.

    > Theorem (§5.5).  If rho_p = u_tau - u_sig - eps_p <= 0 for all p not in S,
    > then the subproblem solution extended by zero is the global optimum of (P)
    > over *all* m arcs.

    One vectorised pass over every arc proves optimality without ever forming
    those arcs' contributions to the Laplacian.  That is the scaling result: it
    replaces exponential path enumeration with an O(m) scalar test.
    """
    rho = u[g.tau] - u[g.sig] - g.eps
    return np.flatnonzero((~in_S) & (rho > tol))


@dataclass(slots=True)
class SolveReport:
    solution: Solution
    certificate: bool
    cg_rounds: int
    in_S: np.ndarray
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    #: Upper bound on the objective still available, in the solve's own units.
    #: Zero means nothing wants in -- the point is optimal however it was
    #: reached.  See `optimality_gap`.
    gap: float = 0.0


def solve(
    g: ArcArrays,
    src: int,
    dst: int,
    Psi: float,
    *,
    seed: np.ndarray | None = None,
    max_rounds: int = 8,
    tol: float = TOL,
    solver=None,
    forced_upper: dict[int, float] | None = None,
    forbidden: np.ndarray | None = None,
    A0: np.ndarray | None = None,
    min_flow: float = 0.0,
    gas_cost: float = 0.0,
) -> SolveReport:
    """Column generation around `active_set_solve` (spec §5.1 lines 5-10).

    `A0` warm-starts the active set.  Re-solving a near-identical problem --
    which is what every candidate generator does -- then costs a handful of
    pivots instead of rediscovering the support from scratch.
    """
    m = g.m
    banned = np.zeros(m, bool) if forbidden is None else np.asarray(forbidden, bool)
    degenerate = False
    in_S = np.ones(m, bool) if seed is None else np.asarray(seed, bool).copy()
    in_S &= ~banned

    report_solution: Solution | None = None
    rounds = 0
    widened = False
    restarted = False
    warm = A0
    screen = min_flow
    for rounds in range(1, max_rounds + 4):
        report_solution = active_set_solve(
            g, src, dst, Psi,
            A0=warm,
            forbidden=~in_S | banned,
            forced_upper=forced_upper,
            tol=tol,
            solver=solver,
            min_flow=screen,
            gas_cost=gas_cost,
            partial_ok=degenerate,
            maxit=DEFAULT_PIVOTS,
        )
        # Only the solves that actually ran out get more.  Handing every solve
        # the larger budget up front costs 11-14% of a quote and changes
        # nothing at $100k, where the base solve converges in 41 pivots and
        # never reaches 600.  Exhaustion is only knowable afterwards -- it is
        # `pivots == maxit`, where cycling stops short of it -- so this is a
        # retry rather than a bigger first attempt.
        wider = max(DEFAULT_PIVOTS, BASE_PIVOT_FACTOR * m)
        if (wider > DEFAULT_PIVOTS
                and report_solution.reason == "PARTIAL"
                and report_solution.pivots >= DEFAULT_PIVOTS):
            report_solution = active_set_solve(
                g, src, dst, Psi,
                A0=warm,
                forbidden=~in_S | banned,
                forced_upper=forced_upper,
                tol=tol,
                solver=solver,
                min_flow=screen,
                gas_cost=gas_cost,
                partial_ok=degenerate,
                maxit=wider,
            )
        if (
            not report_solution.feasible
            and report_solution.reason.startswith("no convergence")
            and screen <= 0
        ):
            # Widen before degrading.  §5.3's "seed quality only affects the
            # number of column-generation rounds, never correctness" applies to
            # a seed that *cycles* as much as to one that disconnects, and the
            # screen below is not free: it drops the oscillating arcs, so the
            # answer comes back on a fraction of the support it should have.
            # Measured on `WETH -> WBTC` at $1M, the amount that lands here:
            # screened gives gap 2.0e-03 on 35 arcs, widening gives 3.4e-07 on
            # 96 and 40% less modelled loss -- and 26 bp more on the chain.
            # This branch has to come first because the screen sets
            # `degenerate`, after which `partial_ok` makes every later round
            # feasible and the widening below is never reached.
            if not widened and in_S.sum() < (~banned).sum():
                in_S = ~banned
                widened = True
                warm = None
                continue
            # Dozens of arcs within a hair of the diode threshold can oscillate
            # forever, each carrying dust.  Retry with a flow screen and accept
            # the incumbent: every iterate satisfies conservation exactly, so a
            # partial solve is a valid flow for candidates and the quoter to work
            # from, and failing the whole route would be strictly worse.  The
            # certificate goes with it.
            screen = DEGENERACY_SCREEN * Psi
            degenerate = True
            continue
        # Carry the support into the next round.  Column generation only *adds*
        # arcs, so re-deriving the active set each round repeats work -- and each
        # repeat starts from a large active set, which is what makes solves big.
        if report_solution.feasible:
            warm = np.flatnonzero(report_solution.A)
        if not report_solution.feasible:
            # A warm start is an optimisation and must never decide the answer.
            #
            # `A0` is the previous size's support, which is how an interactive
            # session could quote $100 and then fail outright on $2,000,000: the
            # small quote's single arc hits its cap at the larger size, moves to
            # the upper-bounded set, and leaves nothing joining src to dst.  The
            # widening below cannot help -- it widens the *column* set, which was
            # never the restriction.  Starting cold is what `A0=None` already
            # means to `active_set_solve`, so this costs pivots only on a path
            # that was about to fail anyway.
            if warm is not None and not restarted:
                warm, restarted = None, True
                continue
            # "Seed quality only affects the number of column-generation rounds,
            # never correctness" (§5.3).  A seed that fails to connect src to
            # dst must therefore widen, not fail: pricing-out cannot rescue it,
            # because an infeasible restriction produces no potentials to price
            # against.
            if not widened and in_S.sum() < (~banned).sum():
                in_S = ~banned
                widened = True
                continue
            return SolveReport(report_solution, False, rounds, in_S,
                               reason=report_solution.reason)
        violators = price_out(report_solution.u, g, in_S | banned, tol)
        if violators.size == 0:
            break
        in_S[violators] = True
    else:
        return SolveReport(report_solution, False, rounds, in_S, reason="CG_TRUNCATED")

    if report_solution is not None and not report_solution.feasible:
        return SolveReport(report_solution, False, rounds, in_S,
                           reason=report_solution.reason)

    assert report_solution is not None

    # --- the QP path, where the active set is past what §5.4 sized it for ---
    #
    # `docs/quadratic-flow-router.md` §5.4 expects "final |S| ~ 30-60 out of
    # 1000".  A $10M trade over curve+v2+v3+v4 puts 357 of 2,975 arcs in the
    # active set, and there the method does not merely slow down -- at $1M it
    # reports §5.5 optimality to machine precision while OSQP finds 36% less
    # loss on the same graph.  So this is asked whenever the support outgrows
    # the design point or the solve gave up, and the *objective* decides, which
    # is exact because both flows are scored on the same graph.
    #
    # `qp_solution` sees everything `banned` does not forbid, rather than the
    # column-generation subset, which is the point of asking it: it solves (P)
    # globally in one shot.  `in_S` widens to match when it wins, so the gap
    # below is measured over what actually produced the answer.
    if QP_RESCUE:
        support = int(np.count_nonzero(report_solution.psi > tol))
        if support > QP_ACTIVE_FLOOR or report_solution.reason == "PARTIAL":
            other = qp_solution(g, src, dst, Psi, forbidden=banned,
                                forced_upper=forced_upper, tol=tol)
            if (other is not None
                    and other.objective(g) < report_solution.objective(g)):
                report_solution = other
                in_S = ~banned

    # The certificate needs both: no arc outside S wants flow, and no
    # non-concave arc carries any -- §5.5 proves nothing about a flagged arc.
    flagged_active = bool(np.any(g.flagged & (report_solution.psi > 0)))
    certificate = not flagged_active and not banned.any()
    reason = ""
    # What a solve that stopped early actually left behind, rather than whether it
    # stopped early.  A screened or cycling solve that nothing wants to improve on
    # *is* optimal, and saying otherwise throws away a certificate the answer has
    # earned -- measured at a 0.024 bp bound against a 660 bp modelled loss.
    gap = optimality_gap(report_solution, g, in_S & ~banned, dst, tol)
    if degenerate and gap > 0.0:
        certificate = False
        reason = "DEGENERATE"
    if flagged_active:
        reason = "CHORD_ACTIVE"
    elif banned.any():
        reason = "RESTRICTED"
    return SolveReport(report_solution, certificate, rounds, in_S, reason=reason,
                       gap=gap)
