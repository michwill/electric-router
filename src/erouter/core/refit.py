"""Refit at the realised size and re-solve (spec §8, §7 rule 3).

The first solve calibrates every arc at a *guessed* size -- `d_bar` is
bootstrapped from TVL before anything is known about the split.  Once the
optimum says how much each arc actually carries, two more quotes per active arc
re-anchor the model there:

    quote f(delta_p) and f(1.01 delta_p)
    B_p <- 2 (a_p delta_p - f(delta_p)) / delta_p^2       (M2, at the real size)
    recompute G, eps ; re-solve warm-started
    stop when max|d psi| / Psi < 1e-4

`a` deliberately stays the tangent from the probe grid: it is what `eps` is
built from, it is stable, and re-deriving it from a large trade would fold
price impact into what is supposed to be the zero-size marginal rate.  The
extra `1.01 delta` quote is what §7 rule 3 buys -- the *realised* slope, which
gives `f''` at the true operating point and, more importantly, catches an arc
whose returns are increasing right where the route wants to use it.

Converges in two rounds essentially always, because `B_p` varies slowly with
`delta_p` -- it is a fixed point in a quantity that is nearly constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import ArcArrays, arc_params, ceiling_conductance
from .quoter import QuoterClient
from .types import PoolArc, Probe

SLOPE_STEP = 0.01  # §7 rule 3's 1.01 * delta
CONVERGED = 1e-4  # §8's stopping rule on max|d psi| / Psi


@dataclass(slots=True)
class RefitRound:
    quoted: int = 0
    reflagged: int = 0
    max_delta_psi: float = float("inf")
    max_b_change: float = 0.0
    converged: bool = False


@dataclass(slots=True)
class RefitReport:
    rounds: list[RefitRound] = field(default_factory=list)
    psi: np.ndarray | None = None
    changed: bool = False
    note: str = ""

    @property
    def converged(self) -> bool:
        return bool(self.rounds and self.rounds[-1].converged)


def _probe_pair(arc: PoolArc, delta_raw: int) -> list[Probe]:
    bumped = int(delta_raw * (1 + SLOPE_STEP))
    return [
        Probe(arc.pool, arc.kind, arc.i, arc.j, arc.n_coins, delta_raw),
        Probe(arc.pool, arc.kind, arc.i, arc.j, arc.n_coins, max(bumped, delta_raw + 1)),
    ]


def refit_arcs(
    g: ArcArrays,
    arcs: list[PoolArc],
    psi: np.ndarray,
    nu: np.ndarray,
    client: QuoterClient,
    *,
    rate_in,
    rate_out,
) -> tuple[int, int]:
    """Re-anchor `B` for every arc carrying flow.  Returns (quoted, reflagged).

    `rate_in`/`rate_out` map an arc to its node-merge scaling, so the fit is
    done in the same canonical units the solver works in.
    """
    active = np.flatnonzero(psi > 0)
    if active.size == 0:
        return 0, 0

    probes: list[Probe] = []
    plan: list[tuple[int, int, float]] = []  # (arc index, probe offset, delta canonical)
    for k in active:
        arc = arcs[int(k)]
        delta_canonical = float(psi[k]) / float(nu[arc.tau])
        delta_token = delta_canonical / rate_in(arc)
        delta_raw = int(delta_token * 10**arc.decimals_in)
        if delta_raw <= 0:
            continue
        plan.append((int(k), len(probes), delta_canonical))
        probes.extend(_probe_pair(arc, delta_raw))

    if not probes:
        return 0, 0
    answers = client.probe(probes)

    quoted = 0
    reflagged = 0
    for index, offset, delta_canonical in plan:
        arc = arcs[index]
        at_delta, at_bumped = answers[offset], answers[offset + 1]
        if not at_delta.ok or at_delta.value <= 0:
            continue

        scale_out = 10**arc.decimals_out
        f_delta = at_delta.value / scale_out * rate_out(arc)
        # (M2) at the realised size: match the true curve at 0 and at delta.
        B = 2.0 * (arc.a * delta_canonical - f_delta) / delta_canonical**2
        quoted += 1

        if at_bumped.ok and at_bumped.value > at_delta.value:
            f_bumped = at_bumped.value / scale_out * rate_out(arc)
            slope = (f_bumped - f_delta) / (delta_canonical * SLOPE_STEP)
            # The marginal rate must fall with size.  If it rises here, the arc
            # has increasing returns exactly where the route wants to use it --
            # inadmissible in a convex program, so clamp and flag (§11.2).
            if slope > arc.a * (1 + 1e-9):
                B = 0.0
                if not arc.convex_flag:
                    reflagged += 1
                arc.convex_flag = True

        if B <= 0.0:
            # The zero-curvature limit, with the mandatory cap (§2.3 rule 2).
            arc.B = 0.0
            arc.clamped = True
            arc.convex_flag = True
            arc.cap = min(arc.cap, delta_canonical)
        else:
            arc.B = B
            arc.clamped = False
        arc.calib_delta = delta_canonical
    return quoted, reflagged


def rebuild(g: ArcArrays, arcs: list[PoolArc], nu: np.ndarray) -> ArcArrays:
    """Recompute G and eps in place after a refit, keeping the same indexing."""
    a = np.array([arc.a for arc in arcs])
    B = np.array([arc.B for arc in arcs])
    G, eps = arc_params(g.tau, g.sig, a, B, nu)
    flagged = np.array([arc.convex_flag for arc in arcs])
    g.a, g.B = a, B
    g.G = ceiling_conductance(G, flagged) / g.g_scale
    g.eps = eps
    g.flagged = flagged
    g.clamped = np.array([arc.clamped for arc in arcs])
    caps = []
    for arc in arcs:
        if np.isfinite(arc.cap):
            caps.append(float(nu[arc.tau] * arc.cap) / g.g_scale)
        else:
            caps.append(np.inf)
    g.cap = np.array(caps)
    return g


def refit(
    g: ArcArrays,
    arcs: list[PoolArc],
    psi: np.ndarray,
    nu: np.ndarray,
    client: QuoterClient,
    solve_fn,
    Psi: float,
    *,
    rate_in,
    rate_out,
    rounds: int = 2,
) -> RefitReport:
    """§8's loop.  `solve_fn(g, A0) -> Solution` re-solves the updated graph."""
    report = RefitReport()
    current = np.array(psi, dtype=float)

    for _ in range(rounds):
        entry = RefitRound()
        before = np.array([arc.B for arc in arcs])
        entry.quoted, entry.reflagged = refit_arcs(
            g, arcs, current, nu, client, rate_in=rate_in, rate_out=rate_out
        )
        if entry.quoted == 0:
            report.note = "no arc could be re-quoted at its realised size"
            report.rounds.append(entry)
            break

        after = np.array([arc.B for arc in arcs])
        moved = np.abs(after - before) / np.maximum(np.abs(before), 1e-30)
        entry.max_b_change = float(moved.max()) if moved.size else 0.0

        rebuild(g, arcs, nu)
        solution = solve_fn(g, np.flatnonzero(current > 0))
        if not solution.feasible:
            report.note = f"refit made the problem infeasible: {solution.reason}"
            report.rounds.append(entry)
            break

        entry.max_delta_psi = (
            float(np.max(np.abs(solution.psi - current)) / Psi) if Psi > 0 else 0.0
        )
        entry.converged = entry.max_delta_psi < CONVERGED
        report.rounds.append(entry)

        current = solution.psi
        report.psi = current
        report.changed = True
        if entry.converged:
            break

    return report
