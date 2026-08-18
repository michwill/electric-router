"""A flow handed out of the solver must satisfy `psi >= 0`.

(P)'s element law is `psi_p = G_p (u_tau - u_sig - eps_p)+`, so a negative
`psi` is not a point of the problem at all -- it is an arc the active-set
method had not yet pivoted out when it ran out of pivots or started cycling.
A converged solve never returns one, because dropping negatives is the first
pivot tried; the early exits used to.

That was not a cosmetic violation, because the two consumers disagree about
what a negative arc means:

* §12.4's flow-conservation gate counts it, so the flow it checks includes it;
* `realize` takes `psi > 0`, so the route that executes does not.

Both bugs found on USDC->WBTC $1M came from that gap.  Zeroing the arc stranded
its magnitude and the gate refused the route -- conservation went from 3.2e-10
out of the solve to 8.4e-03 after cleanup.  Keeping the arc satisfies the gate
but lets the executed route carry less than the model priced.

Neither is fixable downstream: whichever consumer you satisfy, the other is
reading a different flow.  So the solve restores feasibility before returning,
by taking the pivot it would have taken anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core.graph import ArcArrays
from erouter.core.solve import TOL, active_set_solve

N_NODES = 4
SRC, DST = 0, 3


def graph(eps):
    """Four nodes, two parallel two-hop lanes, and a tempting back-arc.

    The back-arc (3->1) has a favourable `eps`, which is what invites the
    solver to put flow on it in the wrong direction and leaves a negative
    `psi` behind when the pivoting is cut short.
    """
    tau = np.array([0, 1, 0, 2, 3])
    sig = np.array([1, 3, 2, 3, 1])
    n = len(tau)
    return ArcArrays(
        tau=tau, sig=sig,
        a=np.ones(n), B=np.ones(n),
        G=np.array([1.0, 1.0, 1.0, 1.0, 5.0]),
        eps=np.array(eps),
        cap=np.full(n, np.inf),
        flagged=np.zeros(n, dtype=bool),
        clamped=np.zeros(n, dtype=bool),
        n_nodes=N_NODES,
        sources=[[k] for k in range(n)],
    )


EPS = [1e-3, 1e-3, 2e-3, 1e-3, -5e-3]


def conservation(g, psi, Psi):
    net = np.zeros(g.n_nodes)
    np.add.at(net, g.tau, psi)
    np.subtract.at(net, g.sig, psi)
    want = np.zeros(g.n_nodes)
    want[SRC] += Psi
    want[DST] -= Psi
    return float(np.max(np.abs(net - want)))


@pytest.mark.parametrize("maxit", [1, 2, 3, 5, 40])
def test_no_negative_flow_however_early_the_solve_stops(maxit):
    """Cut the pivoting off at every depth; every usable answer is feasible."""
    g = graph(EPS)
    Psi = 100.0
    sol = active_set_solve(g, SRC, DST, Psi, maxit=maxit, partial_ok=True)
    if not sol.feasible:
        return  # refusing is allowed; returning something invalid is not
    assert sol.psi.min() >= -TOL, (
        f"solve stopped after {maxit} pivot(s) and returned flow running "
        f"backwards on {int((sol.psi < -TOL).sum())} arc(s), min "
        f"{sol.psi.min():.3e}"
    )


@pytest.mark.parametrize("maxit", [1, 2, 3, 5, 40])
def test_conservation_survives_the_cleanup(maxit):
    """Dropping the negative arcs must not strand what they were carrying."""
    g = graph(EPS)
    Psi = 100.0
    sol = active_set_solve(g, SRC, DST, Psi, maxit=maxit, partial_ok=True)
    if not sol.feasible:
        return
    assert conservation(g, sol.psi, Psi) < 1e-6 * Psi


def test_a_converged_solve_is_still_optimal_and_non_negative():
    """The cleanup must not disturb the ordinary path."""
    g = graph(EPS)
    sol = active_set_solve(g, SRC, DST, 100.0)
    assert sol.feasible and sol.reason != "PARTIAL"
    assert sol.psi.min() >= -TOL
    assert conservation(g, sol.psi, 100.0) < 1e-6 * 100.0
