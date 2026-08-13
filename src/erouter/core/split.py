"""Optimise a finished route's split ratios against the quoter itself (§7).

The model decides *which* pools; this decides *how much* through each, and it
does so without the model.  Once the topology is fixed the output is a smooth
function of the split weights alone, every evaluation is an exact chained
on-chain quote, and one `quote_routes` prices a whole batch per round trip --
so a central-difference gradient plus a batched line search converges in two
round trips per round.

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

**A round trip costs about eight quotes.**  Measured on an 11-leg route: one
route in a batch is 80 ms, 32 are 358 ms, so the link is ~70 ms and each extra
route ~9 ms.  That ratio is what sizes the design -- probing a coordinate in
both directions and at two scales is cheap next to the trip that carries it, so
the batch is spent on better information per round rather than on more rounds.
Every probe is also kept as a candidate answer in its own right, not spent only
on a difference quotient.

Measured at block 25742250 against a forward-difference search that always ran
three rounds, over the 6 of 16 reference routes that trip the gate (the other
10 cost nothing at all, then and now):

    crvUSD->sDOLA 5M   6 trips ->  8   +63.93 bp
    rETH->WETH 50      6 trips ->  8    +2.93 bp
    USDC->WETH 100k    2 trips ->  6    +2.73 bp
    crvUSD->sDOLA 2M   6 trips ->  4    +0.04 bp
    USDC->sUSDS 1M     6 trips ->  4    +0.08 bp
    USDC->USDT 5M      4 trips ->  4    +0.00 bp

Two thirds of them got *cheaper*, and the trips went where the basis points
were: the two routes that escalated returned 64 and 3 bp for two extra trips
each.  A flat budget cannot do that in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .quoter import MAX_ALL_LEGS, MAX_ROUTES, QuoterClient
from .types import Leg

BPS = 10_000

# A leg trading this much of its pool's reserve is where the quadratic stops
# being trustworthy -- see the band table above.
GATE_THETA = 0.25
# ...or the route as a whole already disagrees with the chain by this much,
# which catches a bad split in shallow pools that theta alone misses.
GATE_GAP_BP = 2.0

# Two rounds is the budget: on 4 of the 6 reference routes that trip the gate,
# round 2 gains under 0.4 bp and a third would be pure latency.
MAX_ROUNDS = 2
# ...but the pathological route is exactly the one worth chasing.  On
# crvUSD->sDOLA 5M -- where the model's own split was 300 bp wrong -- rounds 2
# and 3 were still buying 18 and 25 bp.  So a round that is *still* finding
# this much extends the budget by one, to a hard cap.  Cost then scales with
# the money on the table instead of with a constant.
CONTINUE_BP = 1.0
HOT_ROUNDS = 4
# Central-difference probe scales, in weight units.  Two scales because the
# useful step is not knowable in advance: 0.02 clears integer `bps`
# granularity by two orders and stays local, 0.08 still resolves a coordinate
# whose curvature is flat at 0.02.  They cost quotes, not round trips.
PROBE_SCALES = (0.02, 0.08)
# Multipliers along the normalised gradient, spanning two decades so one batch
# covers both a cautious and an aggressive step.
LINE_STEPS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
# Fractions of the coordinate-wise best move, applied to every coordinate at
# once.  A gradient step is one direction; this is a second one, free to the
# round because it rides the batch the line search was already paying for.
COMBINED_STEPS = (1.0, 0.5, 0.25)
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


def _move(weights: list[np.ndarray], moves) -> list[np.ndarray]:
    """`weights` shifted by `[(group, index, delta)]`, the sweep leg absorbing each."""
    out = [w.copy() for w in weights]
    for g, j, delta in moves:
        out[g][j] += delta
        out[g][-1] -= delta
    return [_project(w) for w in out]


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


def batch_budget(client: QuoterClient, legs: int) -> int:
    """How many routes fit in *one* round trip.

    `quote_routes` silently re-chunks a batch that busts either the route count
    or the total leg count, and a re-chunk is another sequential round trip --
    so the search sizes itself to the smaller of the two limits rather than
    discovering the boundary as unexplained latency.
    """
    routes = int(getattr(client, "max_routes", MAX_ROUTES))
    all_legs = int(getattr(client, "max_all_legs", MAX_ALL_LEGS))
    return max(1, min(routes, all_legs // max(legs, 1)))


def optimise(
    legs: list[Leg],
    client: QuoterClient,
    *,
    amount_in: int,
    dst_slot: int,
    baseline: int = 0,
    max_rounds: int = MAX_ROUNDS,
    hot_rounds: int = HOT_ROUNDS,
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

    budget = batch_budget(client, len(legs))

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

    # As many probe scales as the trip can carry, smallest first -- the
    # gradient wants the smallest scale that quotes on both sides.
    depth = max(1, budget // (2 * len(free)))
    scales = PROBE_SCALES[:depth] or PROBE_SCALES[:1]

    allowed = max_rounds
    while report.rounds < allowed:
        report.rounds += 1
        opened, centre = best, [w.copy() for w in best_w]

        # --- one batch: both directions, every scale, every coordinate ----
        design = [(g, j, s, sign)
                  for (g, j) in free for s in scales for sign in (1.0, -1.0)][:budget]
        probes = [_move(centre, [(g, j, sign * s)]) for g, j, s, sign in design]
        values = quote(probes)
        seen: dict[tuple[int, int, float], dict[float, int]] = {}
        for (g, j, s, sign), value in zip(design, values, strict=True):
            seen.setdefault((g, j, s), {})[sign] = value
        # Every probe is a real quote of a real route, so the best of them is a
        # candidate answer and not merely a difference quotient.
        for candidate, value in zip(probes, values, strict=True):
            if value > best:
                best, best_w = value, candidate

        # --- the gradient, from the smallest scale that quoted -------------
        grad = np.zeros(len(free))
        for k, (g, j) in enumerate(free):
            for s in scales:
                up = seen.get((g, j, s), {}).get(1.0, 0)
                down = seen.get((g, j, s), {}).get(-1.0, 0)
                if up > 0 and down > 0:
                    grad[k] = (up - down) / (2.0 * s)
                    break
                if up > 0:
                    grad[k] = (up - opened) / s
                    break
                if down > 0:
                    grad[k] = (opened - down) / s
                    break

        # --- one batch: the gradient line, plus the coordinate-wise move ---
        trials: list[list[np.ndarray]] = []
        scale = float(np.max(np.abs(grad)))
        if np.isfinite(scale) and scale > 0:
            unit = grad / scale
            for step in LINE_STEPS:
                trials.append(_move(centre, [
                    (g, j, step * float(unit[k])) for k, (g, j) in enumerate(free)
                ]))
        picks = []
        for g, j in free:
            options = [(sign * s, value) for s in scales
                       for sign, value in seen.get((g, j, s), {}).items()]
            delta, value = max(options, key=lambda pair: pair[1], default=(0.0, 0))
            if value > opened:
                picks.append((g, j, delta))
        if len(picks) > 1:
            trials.extend(
                _move(centre, [(g, j, d * factor) for g, j, d in picks])
                for factor in COMBINED_STEPS
            )
        if trials:
            trials = trials[:budget]
            for candidate, value in zip(trials, quote(trials), strict=True):
                if value > best:
                    best, best_w = value, candidate

        if best <= opened:
            break
        report.improved = True
        gain_bp = (best / opened - 1) * 10_000
        if gain_bp < TOL_BP:
            break
        if report.rounds >= allowed and gain_bp >= CONTINUE_BP:
            allowed = min(hot_rounds, allowed + 1)

    report.after = best
    return (apply_weights(legs, groups, best_w) if report.improved else legs), report
