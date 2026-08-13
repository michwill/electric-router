"""Optimise a finished route's split ratios against the quoter itself (§7).

The model decides *which* pools; this decides *how much* through each, and it
does so without the model.  Once the topology is fixed the output is a smooth
function of the split weights alone, every evaluation is an exact chained
on-chain quote, and `quote_routes` prices 32 of them per round trip -- so a
finite-difference gradient plus a batched line search converges in a handful of
calls.

Why this rather than re-fitting the model (§8):

* **The objective has no model error.**  That matters because the error is not
  uniform -- measured over 71 legs, median |model - chain| is 0.00 bp below 10%
  of pool depth and 36 bp above 50%, with a worst case of 2,450 bp.  Re-fitting
  repairs the model and then trusts it again; this never consults it.
* **It cannot go infeasible.**  The topology is fixed, so src stays connected
  to dst by construction.  That is exactly how §8's refit fails in practice --
  "src not connected to dst through the active set" -- after which it discards
  its own correction.
* **It cannot lose.**  Only a strict improvement is accepted, so the worst case
  is the route we already had, minus the round trips.

Measured at block 25742250, six calls each: USDC->sUSDS +19.99 bp,
crvUSD->sDOLA 5M +332.83 bp, rETH->WETH +23.13 bp -- and +0.03 bp on a route
whose legs are all shallow, which is what the gate below is for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .quoter import QuoterClient
from .types import Leg

BPS = 10_000

# A leg trading this much of its pool's reserve is where the quadratic stops
# being trustworthy -- see the band table above.
GATE_THETA = 0.25
# ...or the route as a whole already disagrees with the chain by this much,
# which catches a bad split in shallow pools that theta alone misses.
GATE_GAP_BP = 2.0

MAX_ROUNDS = 3
# Finite-difference step, in weight units.  Large enough to clear integer `bps`
# granularity (1e-4) by two orders, small enough to stay local.
FD_STEP = 0.02
# Multipliers along the normalised gradient, spanning two decades so one batch
# covers both a cautious and an aggressive step.
LINE_STEPS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
# No branch may be driven to nothing: that is a topology change, and the point
# of this pass is that the topology is fixed.
MIN_WEIGHT = 1e-4
# Stop when a round buys less than this.
TOL_BP = 0.01


@dataclass(slots=True)
class SplitReport:
    groups: int = 0
    free: int = 0
    rounds: int = 0
    calls: int = 0
    evaluations: int = 0
    before: int = 0
    after: int = 0
    improved: bool = False
    skipped: str = ""

    @property
    def gain_bp(self) -> float:
        return (self.after / self.before - 1) * 10_000 if self.before else 0.0


def split_groups(legs: list[Leg]) -> list[list[int]]:
    """Indices of each contiguous run of legs leaving one slot, where it splits.

    Contiguity is the quoter's own grouping rule: it snapshots a slot's balance
    when `src_slot` changes, so a run is exactly one `bps` budget.
    """
    if not legs:
        return []
    runs: list[list[int]] = [[0]]
    for k in range(1, len(legs)):
        if legs[k].src_slot == legs[runs[-1][-1]].src_slot:
            runs[-1].append(k)
        else:
            runs.append([k])
    return [run for run in runs if len(run) > 1]


def weights_of(legs: list[Leg], groups: list[list[int]]) -> list[np.ndarray]:
    """Each group's split as weights summing to 1, the sweep leg taking the rest."""
    out: list[np.ndarray] = []
    for run in groups:
        w = np.array([legs[k].bps / BPS for k in run], dtype=float)
        w[-1] = max(0.0, 1.0 - float(w[:-1].sum()))
        out.append(_project(w))
    return out


def _project(w: np.ndarray) -> np.ndarray:
    """Onto the simplex interior: non-degenerate weights that sum to 1."""
    w = np.clip(np.asarray(w, dtype=float), MIN_WEIGHT, None)
    total = float(w.sum())
    return w / total if total > 0 else np.full(w.size, 1.0 / w.size)


def apply_weights(
    legs: list[Leg], groups: list[list[int]], weights: list[np.ndarray]
) -> list[Leg]:
    """Write weights back as integer `bps`, the last leg of each group sweeping.

    The sweep is what absorbs the rounding, so the group always spends exactly
    the balance it was handed however the integers land.
    """
    out = list(legs)
    for run, w in zip(groups, weights, strict=True):
        w = _project(w)
        head = [max(1, round(float(v) * BPS)) for v in w[:-1]]
        # Leave the sweep at least one bp of room, so it is never a no-op leg.
        room = BPS - 1
        while sum(head) > room and len(head) > 0:
            biggest = max(range(len(head)), key=lambda i: head[i])
            head[biggest] = max(1, head[biggest] - (sum(head) - room))
            if sum(head) <= room:
                break
        for index, value in zip(run[:-1], head, strict=True):
            out[index] = replace(out[index], bps=int(value))
        out[run[-1]] = replace(out[run[-1]], bps=0)
    return out


def should_optimise(
    legs: list[Leg],
    thetas: list[float],
    *,
    modelled_out: int = 0,
    verified_out: int = 0,
    gate_theta: float = GATE_THETA,
    gate_gap_bp: float = GATE_GAP_BP,
) -> str:
    """Why this route is worth optimising, or "" to skip it.

    Deliberately permissive: firing needlessly costs round trips, not basis
    points, and the two triggers catch different failures.  `theta` catches a
    leg large enough for the quadratic to have drifted; the model-vs-chain gap
    catches a route where it already demonstrably has, including the shallow
    pools theta misses.
    """
    if not split_groups(legs):
        return ""
    if thetas and max(thetas) >= gate_theta:
        return f"theta {max(thetas) * 100:.0f}%"
    if modelled_out > 0 and verified_out > 0:
        gap = abs(verified_out / modelled_out - 1) * 10_000
        if gap >= gate_gap_bp:
            return f"model off by {gap:.1f} bp"
    return ""


def optimise(
    legs: list[Leg],
    client: QuoterClient,
    *,
    amount_in: int,
    dst_slot: int,
    baseline: int = 0,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[list[Leg], SplitReport]:
    """Hill-climb the split weights.  Returns the best legs found and a report."""
    groups = split_groups(legs)
    report = SplitReport(groups=len(groups), before=baseline)
    if not groups:
        report.skipped = "no split to optimise"
        report.after = baseline
        return legs, report

    weights = weights_of(legs, groups)
    free = [(g, j) for g, run in enumerate(groups) for j in range(len(run) - 1)]
    report.free = len(free)
    if not free:
        report.skipped = "no free split weight"
        report.after = baseline
        return legs, report

    def quote(candidates: list[list[np.ndarray]]) -> list[int]:
        routes = [apply_weights(legs, groups, w) for w in candidates]
        report.calls += 1
        report.evaluations += len(routes)
        return client.quote_routes(routes, [amount_in] * len(routes), [dst_slot] * len(routes))

    best_w = [w.copy() for w in weights]
    best = baseline
    if best <= 0:
        got = quote([best_w])
        best = got[0] if got else 0
    if best <= 0:
        report.skipped = "baseline does not quote"
        report.after = baseline
        return legs, report
    report.before = best

    for _ in range(max_rounds):
        report.rounds += 1
        # --- finite-difference gradient, one batch ------------------------
        bumped: list[list[np.ndarray]] = []
        for g, j in free:
            trial = [w.copy() for w in best_w]
            trial[g][j] += FD_STEP
            trial[g][-1] -= FD_STEP
            bumped.append([_project(w) for w in trial])
        values = quote(bumped)
        grad = np.array(
            [(v - best) / FD_STEP if v > 0 else 0.0 for v in values], dtype=float
        )
        scale = float(np.max(np.abs(grad)))
        if not np.isfinite(scale) or scale <= 0:
            break
        grad = grad / scale

        # --- line search along it, one batch ------------------------------
        trials: list[list[np.ndarray]] = []
        for step in LINE_STEPS:
            trial = [w.copy() for w in best_w]
            for (g, j), d in zip(free, grad, strict=True):
                move = step * float(d)
                trial[g][j] += move
                trial[g][-1] -= move
            trials.append([_project(w) for w in trial])
        values = quote(trials)
        pick = int(np.argmax(values)) if values else -1
        if pick < 0 or values[pick] <= best:
            break
        gain_bp = (values[pick] / best - 1) * 10_000
        best, best_w = values[pick], trials[pick]
        report.improved = True
        if gain_bp < TOL_BP:
            break

    report.after = best
    return (apply_weights(legs, groups, best_w) if report.improved else legs), report
