"""Active-set solve and the optimality certificate (spec §5.4, §5.5).

The routing program is

    (P)  min_psi  sum_p [ eps_p psi_p + psi_p^2 / (2 G_p) ]
         s.t.     B^T psi = s_hat,   0 <= psi <= cap

-- a strictly convex QP over a network polyhedron.  Its dual is
`max_u  Psi (u_src - u_dst) - sum_p (G_p/2) (u_tau - u_sig - eps_p)_+^2`, whose
gradient is Kirchhoff's current law and whose Hessian is an ordinary graph
Laplacian.  So each active set is one linear solve, and the combinatorics live
entirely in which arcs are on -- which is why finite pivoting works and no line
search or trust region is needed.

The element law `psi_p = G_p (u_tau - u_sig - eps_p)_+` is a diode in series
with a resistor: zero flow until the potential difference exceeds the fee.
That threshold is the origin of sparsity -- real optima light up 3-10 pools out
of a thousand instead of smearing across all of them.

This follows §14's reference listing, with the corrections §14 defers: the
connectivity restriction is recomputed every pivot (§9.4), upper-bounded arcs
carry an arbitrary pinned value rather than only their cap (so §6.3's sweep is
the same code path), and arcs can be forbidden outright (column generation and
the one-arc-per-pool repair).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import ArcArrays, component_of, laplacian
from .linalg import DEFAULT_SOLVER, SingularSystem

# §9.2 absolute, never relative: rho legitimately passes through zero, and this
# is far below any real fee (1e-9 is 1e-5 bp).
TOL = 1e-9


def _steepest_pick(mask: np.ndarray, score: np.ndarray) -> int:
    """Most-violating candidate; ties go to the lowest index."""
    where = np.flatnonzero(mask)
    return int(where[np.argmax(score[where])])


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
        driving the arc, and an *active* arc has `rho = psi/G > 0` by the
        element law (M6) -- so `psi * rho` is not the complementarity product.
        This is: zero on every free arc, >= 0 on arcs held at zero, <= 0 on
        arcs held at their cap.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(g.G > 0, self.psi / g.G, 0.0)
        return slope - self.rho

    def loss_split(self, g: ArcArrays) -> tuple[float, float]:
        with np.errstate(divide="ignore", invalid="ignore"):
            impact = np.where(g.G > 0, self.psi**2 / (2 * g.G), 0.0)
        return float(np.sum(g.eps * self.psi)), float(np.sum(impact))


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
    partial_ok: bool = False,
) -> Solution:
    """Solve (P) restricted to the non-forbidden arcs.

    `min_flow` refuses entry to an arc that would carry less than that much
    value.  With fitted reference prices, dozens of arcs sit within a hair of
    the diode threshold and oscillate in and out, each carrying dust -- 150
    pivots to move 0.01% of the trade.  Screening them out is not an
    approximation of the answer so much as of the *tie*: an arc below the
    screen cannot change the output measurably, and it is left at zero.
    Keep it at 0 for the certified solve, where exactness is the point.

    `forced_upper` pins an arc at a given flow by moving it into the `U` set,
    where it contributes `-B_U^T psi_U` to the right-hand side.  That is exactly
    the mechanism a saturated capacity uses, so §6.3's pin-and-resolve sweep is
    a keyword argument rather than a new branch.
    """
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

    for _ in range(maxit):
        idx = np.flatnonzero(A)

        comp = component_of(dst, g.tau[idx], g.sig[idx], n)
        if not comp[src] and Psi != 0:
            return Solution(
                np.zeros(m), np.zeros(n), A, U, psi_upper, np.zeros(m), pivots,
                feasible=False, reason="src not connected to dst through the active set",
            )
        u_idx = np.flatnonzero(comp)
        keep = u_idx[u_idx != dst]

        rhs = s_hat.copy()
        if idx.size:
            fee_flow = g.G[idx] * g.eps[idx]
            np.add.at(rhs, g.tau[idx], fee_flow)
            np.subtract.at(rhs, g.sig[idx], fee_flow)
        uidx = np.flatnonzero(U)
        if uidx.size:
            np.subtract.at(rhs, g.tau[uidx], psi_upper[uidx])
            np.add.at(rhs, g.sig[uidx], psi_upper[uidx])
            outside = ~comp[g.tau[uidx]] | ~comp[g.sig[uidx]]
            if outside.any():
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
            psi[idx] = g.G[idx] * (u[g.tau[idx]] - u[g.sig[idx]] - g.eps[idx])
        # §9.4: nodes outside `dst`'s component carry zero flow by construction.
        # Without this an arc with both ends outside gets u = 0 on both, so a
        # favourable eps yields psi = -G*eps > 0 -- flow conjured from nothing,
        # in a component the trade never reaches.  It satisfies no conservation
        # law and cannot be ordered for execution.
        psi[~(comp[g.tau] & comp[g.sig])] = 0.0
        rho = u[g.tau] - u[g.sig] - g.eps

        # Degeneracy is only seen with exact-duplicate pools, which §9.5 merges;
        # if a basis ever repeats, fall back to Bland's rule (lowest index).
        signature = (A.tobytes(), U.tobytes())
        if signature in seen_bases:
            bland = True
        seen_bases.add(signature)

        pick = _bland_pick if bland else _steepest_pick

        negative = A & (psi < -tol)
        if negative.any():
            A[pick(negative, -psi)] = False
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
        # Every iterate satisfies conservation exactly -- `u` solves the
        # Laplacian system with the conservation right-hand side, so only
        # *optimality* is incomplete, never feasibility.  A candidate is a
        # heuristic the quoter adjudicates, so an unconverged one is still a
        # perfectly valid route to put in front of it.
        if not partial_ok:
            return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=False,
                            reason=f"no convergence in {maxit} pivots")
        psi = np.where(np.abs(psi) < tol, 0.0, psi)
        return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=True,
                        reason="PARTIAL")

    psi = np.where(np.abs(psi) < tol, 0.0, psi)
    return Solution(psi, u, A, U, psi_upper, rho, pivots, feasible=True)


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
) -> SolveReport:
    """Column generation around `active_set_solve` (spec §5.1 lines 5-10).

    `A0` warm-starts the active set.  Re-solving a near-identical problem --
    which is what every candidate generator does -- then costs a handful of
    pivots instead of rediscovering the support from scratch.
    """
    m = g.m
    banned = np.zeros(m, bool) if forbidden is None else np.asarray(forbidden, bool)
    in_S = np.ones(m, bool) if seed is None else np.asarray(seed, bool).copy()
    in_S &= ~banned

    report_solution: Solution | None = None
    rounds = 0
    widened = False
    warm = A0
    for rounds in range(1, max_rounds + 2):
        report_solution = active_set_solve(
            g, src, dst, Psi,
            A0=warm,
            forbidden=~in_S | banned,
            forced_upper=forced_upper,
            tol=tol,
            solver=solver,
            min_flow=min_flow,
        )
        # Carry the support into the next round.  Column generation only *adds*
        # arcs, so re-deriving the active set from scratch each round repeats
        # work that is already done -- and each repeat starts from a large
        # active set, which is exactly what makes the linear solves big.
        if report_solution.feasible:
            warm = np.flatnonzero(report_solution.A)
        if not report_solution.feasible:
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

    assert report_solution is not None
    # The certificate needs both: no arc outside S wants flow, and no
    # non-concave arc carries any -- §5.5 proves nothing about a flagged arc.
    flagged_active = bool(np.any(g.flagged & (report_solution.psi > 0)))
    certificate = not flagged_active and not banned.any()
    reason = ""
    if flagged_active:
        reason = "CHORD_ACTIVE"
    elif banned.any():
        reason = "RESTRICTED"
    return SolveReport(report_solution, certificate, rounds, in_S, reason=reason)
