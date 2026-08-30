"""Spec §13.3: the active-set solve against an independent QP solver.

Every other test here checks the solver against its own reasoning -- the same
active-set machinery that produced the answer, or an invariant the code was
written to satisfy.  This one poses `(P)` to OSQP, which shares no code with
`core/solve.py`, and compares the objective.

    minimise   sum_p [ eps_p psi_p + psi_p^2 / (2 G_p) ]
    subject to B^T psi = s_hat,  0 <= psi <= cap

That makes it the oracle a *port* has to be checked against: validating a rewrite
by "does it match the Python" reproduces the Python's bugs faithfully.

The comparison is on the **objective**, not on `psi`.  Two flows can tie exactly
-- parallel arcs with equal `eps` and `G` split arbitrarily -- so comparing
arc-by-arc would fail where both answers are right.  Where the optimum is unique
the flows are compared too.
"""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core import graph
from erouter.core.solve import active_set_solve

osqp = pytest.importorskip("osqp")
sparse = pytest.importorskip("scipy.sparse")


def make(tau, sig, a, B, *, cap=None, Psi=1.0, n=None):
    tau = np.asarray(tau, np.int64)
    sig = np.asarray(sig, np.int64)
    n = n or int(max(tau.max(), sig.max()) + 1)
    return graph.build(tau, sig, np.asarray(a, float), np.asarray(B, float),
                       np.ones(n), Psi, cap=cap, n_nodes=n,
                       merge_duplicates=False)


def objective(g, psi):
    """`sum eps psi + psi^2 / 2G`, the program's own value."""
    quad = np.where(g.G > 0, psi**2 / (2 * g.G), 0.0)
    return float(np.sum(g.eps * psi) + np.sum(quad))


def reference(g, src, dst, Psi):
    """Solve `(P)` with OSQP.  Returns (objective, psi)."""
    m, n = len(g.tau), g.n_nodes
    # B^T: one row per node, +1 where the arc arrives, -1 where it leaves.
    rows, cols, vals = [], [], []
    for p in range(m):
        rows.append(int(g.tau[p]))
        cols.append(p)
        vals.append(1.0)
        rows.append(int(g.sig[p]))
        cols.append(p)
        vals.append(-1.0)
    incidence = sparse.csc_matrix((vals, (rows, cols)), shape=(n, m))
    s_hat = np.zeros(n)
    s_hat[src] += Psi
    s_hat[dst] -= Psi

    upper = np.where(np.isfinite(g.cap), g.cap, 1e12 * max(Psi, 1.0))
    A = sparse.vstack([incidence, sparse.eye(m, format="csc")], format="csc")
    lo = np.concatenate([s_hat, np.zeros(m)])
    hi = np.concatenate([s_hat, upper])
    # The quadratic term is `psi^2 / (2 G)`, so `P = diag(1/G)` given OSQP's
    # own factor of a half.  An arc with `G = inf` (a clamped, linear element)
    # contributes nothing.
    inv = np.where(g.G > 0, 1.0 / np.where(np.isfinite(g.G), g.G, np.inf), 0.0)
    P = sparse.diags(np.nan_to_num(inv, posinf=0.0), format="csc")

    prob = osqp.OSQP()
    prob.setup(P=P, q=g.eps.astype(float), A=A, l=lo, u=hi, verbose=False,
               eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000, polishing=True)
    res = prob.solve()
    status = getattr(res.info, "status", "")
    if "solved" not in str(status):
        pytest.skip(f"OSQP did not converge: {status}")
    psi = np.clip(np.asarray(res.x, float), 0.0, None)
    return objective(g, psi), psi


# `tau` is an arc's origin and `sig` its head, which is the orientation the
# rest of the tests use and the one that makes `s_hat[src] = +Psi` come out.
CASES = {
    # A single pool: the answer is forced, so any disagreement is a units bug.
    "one arc": {"tau": [0], "sig": [1], "a": [1.0], "B": [1.0], "Psi": 0.25},
    # Two lanes of different depth: the split is the whole question.
    "parallel": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 1.0], "B": [1.0, 2.0], "Psi": 1.0},
    # Unequal `a`: the diode decides whether the dearer lane opens at all.
    "diode": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 0.9999], "B": [1.0, 1.0], "Psi": 0.5},
    # Two hops against one: 0->1->2 versus 0->2 direct.
    "series": {"tau": [0, 0, 1], "sig": [2, 1, 2], "a": [0.997, 0.9995, 0.9995],
                   "B": [1.0, 1.0, 1.0], "Psi": 0.4},
    # A capped lane: the bound has to bind in both solvers.
    "capped": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 0.999], "B": [1.0, 1.0],
                   "cap": [0.2, np.inf], "Psi": 1.0},
    # Five arcs over four nodes, the shape a real split takes.
    "network": {"tau": [0, 0, 1, 2, 0], "sig": [1, 2, 3, 3, 3],
                    "a": [0.9995, 0.9990, 0.9995, 0.9998, 0.996],
                    "B": [1.0, 2.0, 1.0, 3.0, 0.5], "Psi": 2.0},
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_objective_matches_an_independent_solver(name):
    spec = dict(CASES[name])
    Psi = spec.pop("Psi")
    g = make(Psi=Psi, **spec)
    dst = int(max(g.tau.max(), g.sig.max()))

    ours = active_set_solve(g, 0, dst, Psi)
    assert ours.feasible, f"{name}: our solve refused a problem OSQP accepts"
    mine = ours.objective(g)
    theirs, _ = reference(g, 0, dst, Psi)

    # Relative to the trade, because the objective is a loss in value terms.
    assert mine <= theirs + 1e-9 * Psi, (
        f"{name}: OSQP found a better flow -- ours {mine:.12g}, "
        f"OSQP {theirs:.12g}, worse by {(mine - theirs) / Psi * 1e4:.6f} bp"
    )
    assert abs(mine - theirs) <= 1e-9 * Psi, (
        f"{name}: objectives disagree by {(mine - theirs) / Psi * 1e4:.6f} bp"
    )


@pytest.mark.parametrize("name", ["one arc", "series", "capped"])
def test_the_flow_matches_where_the_optimum_is_unique(name):
    """Where no two arcs are interchangeable, the arcs themselves must agree."""
    spec = dict(CASES[name])
    Psi = spec.pop("Psi")
    g = make(Psi=Psi, **spec)
    dst = int(max(g.tau.max(), g.sig.max()))

    ours = active_set_solve(g, 0, dst, Psi)
    _, theirs = reference(g, 0, dst, Psi)
    assert np.allclose(ours.psi, theirs, atol=1e-6 * Psi), (
        f"{name}: flows differ by {np.max(np.abs(ours.psi - theirs)) / Psi:.3e} of Psi"
    )


def test_a_restricted_resolve_from_a_narrow_warm_start_agrees():
    """A regression test for a divergence that used to be xfailed here.

    A candidate is a *restricted* re-solve warm-started from the base optimum's
    acyclic support.  When that support is almost entirely forbidden, the two
    solvers used to take different pivot sequences: the reference converged in
    four pivots and the port cycled out to PARTIAL on a single arc.

    The cause was not the pivot rule.  `psi = G (u_tau - u_sig - eps)` is a
    small difference of larger numbers times a conductance running to 1e8, so a
    residual of `||L|| ||u|| eps` -- all backward stability promises -- lands in
    `psi` at about `TOL`.  Measured on this very system: `cond(L)` is only
    1.6e4, numpy's LU left a residual of exactly zero and the port's factored
    solve left 9.3e-10, which put a degenerate arc's flow at -1.5e-9 here
    against -4.1e-11 there.  One side of `TOL` each, and from there the pivot
    sequences part.  Solved by rational arithmetic on the same floats, the true
    flow is +1.0e-10: the arc carries nothing, positively.

    The port now runs iterative refinement on every solve, which costs a matvec
    and a triangular solve against a factor already in hand -- and *saves* time
    overall, because an accurate `u` wastes fewer pivots.  Over the ten
    generated universes, restricted re-solves went from two disagreements in
    sixty-three to none in a hundred and twenty.

    `rust/src/solve.rs` pins the same system as a unit test, at the level of
    the decision it used to get wrong.
    """
    import numpy as np

    from erouter.core.realize import cancel_cycles
    from erouter.core.solve import active_set_solve

    rng = np.random.default_rng(4)
    n = 5
    tau, sig, a, B = [], [], [], []
    for tail in range(n):
        for head in range(n):
            if tail == head:
                continue
            for _ in range(1 + int(rng.integers(0, 2))):
                tau.append(tail)
                sig.append(head)
                a.append(float(np.exp(rng.normal(0.0, 0.05))))
                B.append(float(np.exp(rng.uniform(np.log(1e-8), np.log(1e-4)))))
                rng.uniform(1e6, 1e8)  # the TVL draw, so the stream lines up
    tau.append(1)
    sig.append(0)
    a.append(0.999)
    B.append(2e-6)
    g = graph.build(np.array(tau, np.int64), np.array(sig, np.int64),
                    np.array(a, float), np.array(B, float), np.ones(n), 1.0,
                    n_nodes=n, merge_duplicates=False)

    base = active_set_solve(g, 0, 4, 1.0)
    acyclic, _ = cancel_cycles(g.tau, g.sig, base.psi)
    forbidden = np.ones(g.m, bool)
    for arc in (2, 3, 5, 10, 15):
        forbidden[arc] = False

    ours = active_set_solve(g, 0, 4, 1.0, A0=np.flatnonzero(acyclic > 0),
                            forbidden=forbidden, min_flow=1e-4,
                            maxit=60, partial_ok=True)
    import erouter_solve

    problem = erouter_solve.Problem(
        [int(v) for v in g.tau], [int(v) for v in g.sig],
        [float(v) for v in g.G], [float(v) for v in g.eps],
        [float(v) for v in g.cap], g.n_nodes)
    got = problem.solve(0, 4, 1.0, a0=[bool(v) for v in (acyclic > 0)],
                        forbidden=[bool(v) for v in forbidden],
                        min_flow=1e-4, maxit=60, partial_ok=True)
    theirs = np.frombuffer(got["psi"], dtype=np.float64)

    assert ours.reason == "" and got["reason"] == "", (ours.reason, got["reason"])
    assert ours.pivots == got["pivots"] == 4
    assert np.allclose(ours.psi, theirs, atol=1e-6), (ours.psi, theirs)


def test_over_constrained_pins_agree():
    """§6.3's pin sweep, which is where the last solver divergence lived.

    The sweep deliberately over-constrains: it forces an arc to carry
    `{0, 1/8, 1/4, 1/2, 1, 2, 4}` times what the relaxation gave it, because
    the model's *allocation* is the thing not to be trusted.  The high pins put
    the program in a degenerate region where the pivot sequence runs forty to
    sixty steps, and it used to be that the two solvers parted there -- six of
    112 re-solves, in both directions.

    Three defects, none of them a tolerance:

    * the port reused its Cholesky factor across the two paths that move the
      active set by more than one arc (a reconnect admits a whole path, a
      reseed replaces the set outright).  Both cleared the pending rank-1 term,
      neither cleared the factor, and with `keep` unchanged and the size still
      matching, nothing downstream could tell it factorised a different matrix.
      `basis_dirty` now says so.  Because the guard only prices a factor that
      came from an *update*, a stale one was never even measured;
    * the port perturbed a cycling basis into a local `eps` but still built the
      right-hand side from `arcs.eps`, so the perturbation reached the drop
      rule and not the solve.  The reference perturbs both;
    * both then picked pivots by `>` on scores that tie in exact arithmetic --
      two arcs at `psi` -0.499999999999865 against -0.499999999998981 -- so the
      last bits of the solve chose, and the two sides round differently.
      `PIVOT_TIE` makes a near-tie go to the lower index in both.

    Only the third needed a decision rather than a fix; the first two were the
    port failing to be the mirror it claims to be.

    Swept wide because the failures were sparse: on this generator they were
    three universes in twelve.  676 re-solves over 120 universes agree at both
    budgets; this keeps a cheaper slice of that in the suite.
    """
    import erouter_solve
    import numpy as np

    from erouter.core.realize import cancel_cycles
    from erouter.core.solve import active_set_solve

    def universe(seed):
        rng = np.random.default_rng(seed)
        n = 5
        tau, sig, a, B = [], [], [], []
        for tail in range(n):
            for head in range(n):
                if tail == head:
                    continue
                for _ in range(1 + int(rng.integers(0, 2))):
                    tau.append(tail)
                    sig.append(head)
                    a.append(float(rng.uniform(0.980, 0.999)))
                    B.append(float(np.exp(rng.uniform(np.log(1e-7), np.log(1e-5)))))
                    rng.uniform(1e20, 1e24)  # the reserve draw, so the stream
                    rng.uniform(1e6, 1e8)    # and the TVL draw, line up
        return graph.build(np.array(tau, np.int64), np.array(sig, np.int64),
                           np.array(a, float), np.array(B, float), np.ones(n),
                           1.0, n_nodes=n, merge_duplicates=False)

    disagreed = []
    for seed in range(12):
        g = universe(seed)
        base = active_set_solve(g, 0, 4, 1.0)
        acyclic, _ = cancel_cycles(g.tau, g.sig, base.psi)
        problem = erouter_solve.Problem(
            [int(v) for v in g.tau], [int(v) for v in g.sig],
            [float(v) for v in g.G], [float(v) for v in g.eps],
            [float(v) for v in g.cap], g.n_nodes)
        for idx in [k for k in range(g.m) if base.psi[k] > 0][:6]:
            for step in (1.5, 2.0, 4.0, 8.0):
                pin = min(base.psi[idx] * step, g.cap[idx])
                if pin <= 0:
                    continue
                ours = active_set_solve(
                    g, 0, 4, 1.0, A0=np.flatnonzero(acyclic > 0),
                    forbidden=np.zeros(g.m, bool), forced_upper={idx: pin},
                    min_flow=1e-4, maxit=600, partial_ok=True)
                got = problem.solve(
                    0, 4, 1.0, a0=[bool(v) for v in (acyclic > 0)],
                    forbidden=[False] * g.m, pinned=[(idx, pin)],
                    min_flow=1e-4, maxit=600, partial_ok=True)
                theirs = np.frombuffer(got["psi"], dtype=np.float64)
                if not np.allclose(ours.psi, theirs, atol=1e-6):
                    disagreed.append((seed, idx, step, ours.reason or "OK",
                                      str(got["reason"]) or "OK"))

    assert not disagreed, (
        f"{len(disagreed)} over-constrained pin(s) diverge across "
        f"{len({d[0] for d in disagreed})} universes: {disagreed[:4]}")
