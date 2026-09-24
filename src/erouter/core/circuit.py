"""Routing as a nonlinear circuit: Newton on node prices.

The ballot solves one quadratic model hundreds of times and lets the quoter
pick, because the model it solves was never trusted.  This solves the routing
problem itself -- convex, once the pools stop being approximated at a guessed
size -- by the method SPICE uses for a DC operating point.

Unknowns are node prices `nu` (value per canonical token), the destination held
at 1.  Each arc is a two-terminal device: at prices `nu` it trades the `d` that
maximises its value plus a barrier,

    nu_out f(d) - nu_in d + t (ln d + ln(cap - d)),

so the routing dual

    g(nu) = Q nu_src + sum_arcs max_d [...] - kappa sum_v ln nu_v

is convex.  Its gradient is the KCL residual in tokens -- `(-d, f(d))` per arc
by the envelope theorem -- and its Hessian a sum of rank-one stamps with
conductance `-1 / (nu_out f'' - t/d^2 - t/(cap - d)^2)`, positive on every arc.
The barriers are SPICE's gmin: no arc is ever quite off or quite saturated, so
the kinks where one switches on or hits its cap do not stall Newton, and a
token nothing downstream will buy sits at a small positive price instead of
being driven through zero.  `mu` steps both down, each stage damped Newton from
the last -- gmin stepping.

Measured on 13 cases at two blocks against the ballot, net of gas: 21 of 26 at
or above it, none more than 1.31 bp below, +45.4 bp on CRV->WETH $5M.  It only
got there once the graph stopped inventing value: two bank unit bugs and depth
counted twice in multi-port pools, which the ballot had been hiding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import accel as _accel
from .candidates import (
    MIN_FLOW_FRACTION,
    Candidate,
    CandidateSet,
    _from_ballot,
    conflicting_pools,
    keep_only,
    port_ids,
    repair_order,
)
from .gas import STATIC
from .realize import cancel_cycles, prune_dust
from .types import ArcKind

#: Opt-in on the same switch as the rest of the port.
_ACCEL_ON = os.environ.get("EROUTER_ACCEL", "") == "1"

ARMIJO = 1e-4
MAX_ITER = 60             # Newton iterations per barrier stage
MU_START, MU_END = 1e-2, 1e-10
#: Fewer, longer barrier steps cost as many iterations as they save: 3 stages
#: took 47 on CRV->WETH $1M, 5 took 61, 9 took 66, to the same answer.
STAGES = 3
#: A repair round changes a few pools; it resumes near the end of the schedule.
RESUME_MU, RESUME_STAGES = 1e-7, 2
TOL = 1e-9                # worst node's value mismatch, relative to the trade
INNER = 80                # bracketed Newton steps per arc, in log d
GMIN = 1e-14              # isolated nodes only: every arc leaks
#: Carrying the whole trade, a linear arc gives up at most this of its output:
#: `ceiling_conductance` in the trade's own size.  A near-linear arc with no
#: real cap otherwise put conductances near 1e12 into the Hessian.
LINEAR = 1e-6
MAX_ROUNDS = 12

CONVERSIONS = frozenset({
    ArcKind.WRAP_NATIVE, ArcKind.UNWRAP_NATIVE, ArcKind.WSTETH_WRAP,
    ArcKind.WSTETH_UNWRAP, ArcKind.ERC4626_DEPOSIT, ArcKind.ERC4626_REDEEM,
})
EXACT = frozenset({ArcKind.SWAP_UNIV3, ArcKind.SWAP_UNIV4})


@dataclass(slots=True)
class Devices:
    """Every arc as a device: `a d - B d^2/2`, or `a d / (1 + k d)` with
    `k = B / 2a` for a v3/v4 tick range, on `[0, cap]`."""

    tau: np.ndarray
    sig: np.ndarray
    a: np.ndarray
    B: np.ndarray
    cap: np.ndarray       # finite; 0 switches the arc off
    exact: np.ndarray
    n: int
    d: np.ndarray         # warm start for the per-arc solve

    def live(self) -> np.ndarray:
        return self.cap > 0


def devices(arcs, n_nodes: int, nu0: np.ndarray, V0: float) -> Devices:
    """The arcs as devices, in canonical units, for a trade worth `V0`."""
    a = np.array([arc.a for arc in arcs], float)
    B = np.array([arc.B for arc in arcs], float)
    tau = np.array([arc.tau for arc in arcs], np.int64)
    sig = np.array([arc.sigma for arc in arcs], np.int64)
    whole = V0 / np.maximum(nu0[tau], 1e-300)
    B = np.maximum(B, 1e-12 * np.maximum(a, 1e-300))
    B = np.maximum(B, 2.0 * LINEAR * a / whole)
    cap = np.array([arc.cap for arc in arcs], float)
    exact = np.array([arc.kind in EXACT and arc.parallel for arc in arcs], bool)
    # A quadratic pays nothing past its peak.
    cap = np.where(exact, cap, np.minimum(cap, a / B))
    cap = np.where(np.isfinite(cap) & (a > 0), cap, 0.0)
    return Devices(tau, sig, a, B, cap, exact, n_nodes, cap * 1e-6)


def _slopes(dev: Devices, k: np.ndarray, d: np.ndarray):
    """f' and f'' at d for the arcs k."""
    a, B, ex = dev.a[k], dev.B[k], dev.exact[k]
    kk = np.where(ex, B / (2.0 * a), 0.0)
    x = 1.0 + kk * d
    return (np.where(ex, a / x ** 2, a - B * d),
            np.where(ex, -2.0 * a * kk / x ** 3, -B))


def _values(dev: Devices, d: np.ndarray) -> np.ndarray:
    kk = np.where(dev.exact, dev.B / (2.0 * np.maximum(dev.a, 1e-300)), 0.0)
    return np.where(dev.exact, dev.a * d / (1.0 + kk * d), dev.a * d - 0.5 * dev.B * d * d)


def respond(dev: Devices, nu: np.ndarray, t: float, d0=None):
    """Every live arc's trade at prices `nu`: `(d, f(d), f'(d), dF/dd)`.

    Solves `nu_out f'(d) - nu_in + t/d - t/(cap - d) = 0`, strictly decreasing
    in `d`.  It is positive at `d = t / 2 nu_in`, where the leak alone pays
    twice the input price, and falls to -inf at `cap`, so the root is
    bracketed.  Newton runs in `log d`, where the leak regime `d ~ t/gap` is
    not a cliff; only the arcs not yet at machine precision keep iterating.
    """
    live = dev.live()
    cap = np.where(live, dev.cap, 1.0)
    nu_in, nu_out = nu[dev.tau], nu[dev.sig]
    lo = np.minimum(t / (2.0 * np.maximum(nu_in, 1e-300)), cap * 0.5)
    hi = cap.copy()
    d = np.clip(dev.d if d0 is None else d0, lo, cap * (1 - 1e-12))
    d = np.where(live & (d > lo) & (d < hi), d, np.sqrt(lo * hi))
    act = np.flatnonzero(live)
    for _ in range(INNER):
        if act.size == 0:
            break
        da, ca = d[act], cap[act]
        f1, f2 = _slopes(dev, act, da)
        pay, leak, wall = nu_out[act] * f1, t / da, t / (ca - da)
        F = pay - nu_in[act] + leak - wall
        # Zero to rounding: F is a difference of terms of order one, and in the
        # leak regime its derivative is t/d small, so a step test at 1e-14
        # alone could never pass and every such arc ran to INNER.
        exact_ = np.abs(F) <= 8.0 * np.finfo(float).eps * (np.abs(pay) + nu_in[act] + leak + wall)
        act, da, ca, f1, f2, F = (x[~exact_] for x in (act, da, ca, f1, f2, F))
        if act.size == 0:
            break
        dF = nu_out[act] * f2 - t / da ** 2 - t / (ca - da) ** 2
        up = F > 0
        lo[act] = np.where(up, da, lo[act])
        hi[act] = np.where(up, hi[act], da)
        new = da * np.exp(np.clip(-F / (dF * da), -50.0, 50.0))
        bad = ~((new > lo[act]) & (new < hi[act]))
        new = np.where(bad, np.sqrt(lo[act] * hi[act]), new)
        # At machine precision, not at a tolerance on F: a looser test left
        # the objective noisier than Armijo's decrease, and the line search
        # stalled -- 222 iterations against 65.
        done = ((~bad) & (np.abs(np.log(new / da)) <= 1e-14)) | (hi[act] <= lo[act] * (1 + 1e-15))
        d[act] = new
        act = act[~done]
    d = np.where(live, d, 0.0)
    f1, f2 = _slopes(dev, np.arange(d.size), d)
    safe_d, safe_room = np.where(live, d, 1.0), np.where(live, cap - d, 1.0)
    D = nu_out * f2 - np.where(live, t / safe_d ** 2 + t / safe_room ** 2, 1.0)
    return d, _values(dev, d), f1, D


def _objective(dev, nu, Q, src, t, kappa, free, d0):
    d, f, _, _ = respond(dev, nu, t, d0)
    live = dev.live()
    barrier = np.where(live, np.log(np.maximum(d, 1e-300))
                       + np.log(np.maximum(dev.cap - d, 1e-300)), 0.0)
    arcs = np.sum(np.where(live, nu[dev.sig] * f - nu[dev.tau] * d + t * barrier, 0.0))
    return Q * nu[src] + float(arcs) - kappa * float(np.sum(np.log(nu[free]))), d


def _gradient_hessian(dev, nu, Q, src, t, kappa, free):
    d, f, f1, D = respond(dev, nu, t)
    live = dev.live()
    grad = np.zeros(dev.n)
    np.add.at(grad, dev.tau, np.where(live, -d, 0.0))
    np.add.at(grad, dev.sig, np.where(live, f, 0.0))
    grad[src] += Q
    w = np.where(live, -1.0 / D, 0.0)
    H = np.zeros((dev.n, dev.n))
    np.add.at(H, (dev.tau, dev.tau), w)
    np.add.at(H, (dev.tau, dev.sig), -w * f1)
    np.add.at(H, (dev.sig, dev.tau), -w * f1)
    np.add.at(H, (dev.sig, dev.sig), w * f1 * f1)
    # A token nothing downstream will buy is worth 0 and may be left over;
    # the price barrier keeps it positive with a surplus of kappa / nu.
    grad[free] -= kappa / nu[free]
    H[free, free] += kappa / nu[free] ** 2
    return grad, H, d, f


@dataclass(slots=True)
class Result:
    nu: np.ndarray
    delta: np.ndarray
    out: np.ndarray
    iterations: int
    residual: float
    stages: list = field(default_factory=list)


def solve(dev: Devices, Q: float, src: int, dst: int, nu0: np.ndarray, *,
          nu_start=None, mu_start: float = MU_START, stages: int = STAGES,
          tol: float = TOL) -> Result:
    """Sell `Q` canonical units of `src`: damped Newton on the dual, one
    barrier stage at a time, in prices scaled by the starting guess `nu0`."""
    scale = np.where(nu0 > 0, nu0, 1.0)
    nu = (nu0 if nu_start is None else nu_start).copy()
    free = np.ones(dev.n, bool)
    free[dst] = False
    V0 = Q * nu0[src]
    m = max(int(dev.live().sum()), 1)
    total, record, residual = 0, [], float("inf")
    schedule = np.logspace(np.log10(mu_start), np.log10(MU_END), stages)
    for stage, mu in enumerate(schedule):
        t, kappa = mu * V0 / m, mu * V0 / dev.n
        stage_tol = tol if stage == len(schedule) - 1 else max(tol, 10 * mu)
        its, g_cur = 0, None
        for _ in range(MAX_ITER):
            its += 1
            grad, H, d, _ = _gradient_hessian(dev, nu, Q, src, t, kappa, free)
            dev.d = d
            residual = float(np.abs(grad * nu)[free].max() / V0)
            if residual < stage_tol:
                break
            Hs = H[np.ix_(free, free)] * np.outer(scale[free], scale[free])
            Hs[np.diag_indices_from(Hs)] += GMIN * max(float(np.diag(Hs).max()), 1e-300)
            try:
                step = -np.linalg.solve(Hs, grad[free] * scale[free])
            except np.linalg.LinAlgError:
                step = -grad[free] * scale[free]
            direction = np.zeros(dev.n)
            direction[free] = step * scale[free]
            if g_cur is None:
                g_cur = _objective(dev, nu, Q, src, t, kappa, free, d)[0]
            slope = float(grad @ direction)
            s = 1.0
            neg = direction < 0
            if neg.any():
                s = min(1.0, 0.99 * float(np.min(-nu[neg] / direction[neg])))
            while s > 1e-12:
                g1, d1 = _objective(dev, nu + s * direction, Q, src, t, kappa, free, d)
                if g1 <= g_cur + ARMIJO * s * slope:
                    dev.d, g_cur = d1, g1
                    break
                s *= 0.5
            else:
                g_cur = None
            nu = nu + s * direction
        total += its
        record.append((float(mu), its, residual))
    d, f, _, _ = respond(dev, nu, t)
    return Result(nu, d, f, total, residual, record)


def candidates(g, arcs, nu, src: int, dst: int, Psi: float, *,
               advanceable=None, leg_cost_bp: float = 0.0, per_gas: float = 0.0,
               gas_table=None) -> CandidateSet:
    """The circuit's answer, as a ballot of two for `verify` to adjudicate.

    `Psi` is the trade in the graph's scaled value units, as `generate` takes
    it, and the candidates come back in them.  `per_gas` is one unit of gas in
    canonical destination units.

    The first candidate is the solve itself, repaired until no pool is used on
    two ports it cannot be priced across (Decision 3).  The second prunes once:
    a port earning less surplus at the solved prices than `verify` charges its
    leg is dropped and the solve resumed.  Surplus is right for a thin branch
    and wrong for a deep near-linear pool, which carries a great deal at almost
    no rent -- the prune alone cost 39 bp on USDC->WBTC $5M -- so the quoter
    chooses, as it does for the ballot.
    """
    table = gas_table or STATIC
    if _ACCEL_ON and _accel.available():
        got = _accel.circuit(
            arcs, g.n_nodes, g.g_scale, nu, src, dst, Psi, advanceable=advanceable,
            leg_cost_bp=leg_cost_bp, per_gas=per_gas,
            gas=[table.gas(arc.kind, arc.pool, arc.i, arc.j) for arc in arcs])
        if got is not None:
            return _from_ballot(got)
    V = Psi * g.g_scale
    Q = V / nu[src]
    nu0 = nu / nu[dst]
    dev = devices(arcs, g.n_nodes, nu0, Q * nu0[src])
    port = port_ids(arcs)
    out = CandidateSet()

    def flow(res):
        psi = res.delta * nu[dev.tau] / g.g_scale
        return np.where(psi >= MIN_FLOW_FRACTION * Psi, psi, 0.0)

    def settle(res, prune: bool):
        """Repair (and, once, prune) from `res`, resuming the solve each time."""
        psi = flow(res)
        for rounds in range(MAX_ROUNDS):
            banned = np.zeros(len(arcs), bool)
            clash = conflicting_pools(arcs, psi, Psi, advanceable=advanceable)
            if clash:
                keep_only(banned, repair_order(clash, psi, port), 0, None, port)
            if prune and rounds == 0:
                banned |= _unearned(dev, res, arcs, psi, port, dst,
                                    leg_cost_bp, per_gas, table)
            if not banned.any():
                break
            dev.cap[banned] = 0.0
            res = solve(dev, Q, src, dst, nu0, nu_start=res.nu,
                        mu_start=RESUME_MU, stages=RESUME_STAGES)
            out.solves += 1
            out.pivots += res.iterations
            psi = flow(res)
        return res, psi

    first = solve(dev, Q, src, dst, nu0)
    out.solves += 1
    out.pivots += first.iterations
    settled, psi = settle(first, prune=False)
    _offer(out, g, psi, src, dst, "circuit")
    saved, warm = dev.cap.copy(), dev.d.copy()
    _, psi = settle(settled, prune=True)
    _offer(out, g, psi, src, dst, "circuit, pruned")
    dev.cap, dev.d = saved, warm
    return out


def _unearned(dev, res, arcs, psi, port, dst, leg_cost_bp, per_gas, table) -> np.ndarray:
    """Arcs of every port whose surplus is below its leg charge."""
    surplus = np.where(dev.live(), res.nu[dev.sig] * res.out - res.nu[dev.tau] * res.delta, 0.0)
    premium = float(res.out[dev.sig == dst].sum()) * leg_cost_bp / 1e4
    earned: dict[int, float] = {}
    for k in np.flatnonzero(psi > 0):
        earned[int(port[k])] = earned.get(int(port[k]), 0.0) + float(surplus[k])
    banned = np.zeros(len(arcs), bool)
    for head, got in earned.items():
        arc = arcs[head]
        gas = table.gas(arc.kind, arc.pool, arc.i, arc.j) * per_gas
        if got < (gas if arc.kind in CONVERSIONS else max(premium, gas)):
            banned |= port == head
    return banned


def _offer(out: CandidateSet, g, psi, src, dst, label: str) -> None:
    psi, _ = cancel_cycles(g.tau, g.sig, psi)
    psi, _ = prune_dust(g.tau, g.sig, psi, src, dst)
    if not (psi > 0).any():
        return
    if any(np.allclose(psi, c.psi, rtol=1e-12, atol=0.0) for c in out.candidates):
        return
    out.candidates.append(Candidate(
        label=label, psi=psi, certificate=False, kind="circuit",
        n_arcs=int(np.count_nonzero(psi > 0)),
    ))
