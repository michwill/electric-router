"""Reference prices by weighted least squares on log-prices (spec §4).

`nu` is needed to define both `G_p` and `eps_p`, and the best estimator is a
weighted LS fit -- which is itself one Laplacian solve, reusing the same
machinery as the router:

    min_z  sum_p w_p ( z_sig - z_tau + log a_p )^2      =>   L_w z = -M^T W log a

It is robust in the way that matters: it fits the best *consistent* price
system rather than trusting any single quote path, and with `w_p = TVL_p` a
single manipulated shallow pool cannot move the frame.

**Feed both directions at half weight.**  For one pool the fit then lands at
`z_tau - z_sig = (log a_f - log a_r) / 2`, which is the fee-free mid price
exactly: the two one-sided quotes bracket it by +/- log(Gamma), and the
least-squares optimum sits in the middle.  Using one direction biases every
reference price to that side of the spread, by the fee, systematically -- and a
mis-estimated `nu` can manufacture a negative 2-cycle that the solver will
happily route around (§2.6).
"""

from __future__ import annotations

import numpy as np

from .graph import component_of, laplacian
from .linalg import DEFAULT_SOLVER, SingularSystem


def reference_prices(
    tau: np.ndarray,
    sig: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    n_nodes: int,
    numeraire: int,
    *,
    solver=None,
) -> np.ndarray:
    """Fit `nu` with `nu[numeraire] == 1`.

    `a` must be strictly positive.  A zero marginal rate is not a cheap pool,
    it is a broken probe: `log 0` would NaN the entire reference frame, and
    6% of arcs fail their smallest probe on mainnet, so this is a live concern
    rather than a hypothetical.
    """
    tau = np.asarray(tau, np.int64)
    sig = np.asarray(sig, np.int64)
    a = np.asarray(a, float)
    w = np.asarray(w, float)
    solver = solver or DEFAULT_SOLVER

    if a.size and not np.all(a > 0):
        bad = int(np.argmin(a))
        raise ValueError(
            f"arc {bad} has a={a[bad]!r}; reference prices need a > 0 "
            "(a failed probe must drop the arc, not enter the fit as zero)"
        )
    if w.size and not np.all(w > 0):
        raise ValueError("reference-price weights must be positive")

    z = np.zeros(n_nodes)
    if tau.size == 0:
        return np.ones(n_nodes)

    # Only the numeraire's component is determined; anything disconnected keeps
    # nu = 1 and will be dropped later for want of a route.
    comp = component_of(numeraire, tau, sig, n_nodes)
    keep = np.flatnonzero(comp & (np.arange(n_nodes) != numeraire))
    if keep.size:
        log_a = np.log(a)
        rhs = np.zeros(n_nodes)
        np.subtract.at(rhs, sig, w * log_a)
        np.add.at(rhs, tau, w * log_a)
        L = laplacian(tau, sig, w, n_nodes, keep)
        try:
            z[keep] = solver.solve(L, rhs[keep])
        except SingularSystem as exc:  # pragma: no cover - guarded by `comp`
            raise SingularSystem(f"reference-price fit is singular: {exc}") from exc
    return np.exp(z)


def dislocations(
    tau: np.ndarray, sig: np.ndarray, a: np.ndarray, nu: np.ndarray
) -> np.ndarray:
    """Residuals `r_p = z_sig - z_tau + log a_p`.

    Large |r_p| flags a stale pool or a genuine arbitrage, so these are worth
    surfacing rather than discarding.
    """
    return np.log(nu[sig]) - np.log(nu[tau]) + np.log(a)


def pool_mid(a_forward: float, a_reverse: float) -> float:
    """Fee-free mid price implied by the two one-sided quotes."""
    return float(np.sqrt(a_forward / a_reverse))


def gamma_live(a_forward: float | np.ndarray, a_reverse: float | np.ndarray):
    """Measured effective retention, `sqrt(a_f * a_r)` (§2.6).

    Reads the pool's *current* fee straight off two tiny probes -- no fee
    parameters, no `k` computation, no ABI knowledge of the fee law.  For a
    fixed-fee pool it must equal `1 - fee` to full precision, so a deviation
    means the probe pipeline is broken; for a dynamic-fee pool it is the live
    value and its drift across blocks is directly observable.
    """
    return np.sqrt(np.asarray(a_forward) * np.asarray(a_reverse))


def check_pair_drops(
    eps_forward: np.ndarray, eps_reverse: np.ndarray, tol: float = 0.0
) -> np.ndarray:
    """Indices where `eps_f + eps_r <= tol` -- a spurious negative 2-cycle.

    Round-tripping a pool always loses (`a_f a_r = Gamma^2 < 1`), but the
    *linearised* drops are frame-dependent and their sum falls below zero when
    `nu` is far enough off for that pair.  The model then sees a two-arc
    negative cycle that does not exist and allocates flow around it.  Violation
    means `nu` is inconsistent with the pool, not that arbitrage exists: snap
    `nu_sig/nu_tau` toward the pool's own mid, or drop the pool.
    """
    return np.flatnonzero(np.asarray(eps_forward) + np.asarray(eps_reverse) <= tol)
