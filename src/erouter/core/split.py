"""Optimise a finished route's split ratios, without consulting the model (§7).

The model decides *which* pools; this decides *how much* through each.  Once the
topology is fixed the output is a smooth function of the split weights alone,
and the model's `O(theta^2)` Taylor error -- median 0.00 bp below 10% of pool
depth but 36 bp above 50%, worst case 2,450 bp -- has no business anywhere near
that decision.

Why not re-fit the model instead (§8):

* **It cannot go infeasible.**  The topology is fixed, so src stays connected to
  dst by construction.  That is exactly how §8's refit fails in practice --
  "src not connected to dst through the active set" -- after which it discards
  its own correction.
* **It cannot lose.**  Only a strict improvement, measured by a real chained
  quote, is ever accepted.  The worst case is the route we already had.

**Sample the legs; do not chain them.**  A chained route quote forces a round
trip per iteration, because leg `k`'s input is leg `k-1`'s output -- which is
what shapes a search into batched line searches and rationed rounds.  Probing
one pool at many sizes does not chain, so a whole route's curves arrive in one
`probe_batch` and composition happens in Python, where an evaluation is
microseconds and each coordinate can simply be solved by golden section rather
than stepped towards.  See `curves.py` for why `u = x/f(x)` is the thing
interpolated.

**The curves check themselves against the chain before being trusted.**
`baseline` is a real chained quote at known weights, so composing at those same
weights compares like with like, for free.  It has to be done: a stableswap
crossing its peg edge inside one ladder interval measured 497 bp out on the
coarse ladder, and optimising on that would be optimising on fiction.  A curve
set that misses gets one dense pass -- at the sizes the legs actually operate
at, and only for the legs whose own secants say they are badly interpolated.
If it still misses, the chained hill-climb below takes over, which is also what
runs when a leg refuses to probe at all.

Measured at block 25742250 over the 6 of 16 reference routes that trip the gate
(the other 10 cost nothing at all), against the chained search:

    route                trips   probes   gain      vs chained   curves miss
    crvUSD->sDOLA 5M       3       672   +376.90 bp    +9.36 bp    -0.95 bp
    rETH->WETH 50          3       432    +35.61 bp    +0.46 bp    -0.67 bp
    USDC->sUSDS 1M         3       456    +20.46 bp    +0.11 bp    -0.14 bp
    crvUSD->sDOLA 2M       2       360     +0.37 bp    +0.01 bp    +0.01 bp
    USDC->WETH 100k        2       264     +2.65 bp    -0.08 bp    -0.26 bp
    USDC->USDT 5M          2        96     +0.03 bp     0.00 bp    +0.01 bp

Fifteen round trips against the chained search's twenty-four, converged rather
than hill-climbed, and better on four routes of six.  The three that spent a
third trip are the three whose coarse curves failed their own check.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import curves as curve_mod
from .quoter import MAX_ALL_LEGS, MAX_ROUTES, QuoterClient
from .types import ArcKind, Leg, Probe

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

# --- the sampled-curve search ---------------------------------------------
#
# Golden section.  Each iteration shrinks the bracket by 0.618, so 20 take a
# unit interval to 6.6e-5 -- just under the 1e-4 that integer `bps` can express,
# and therefore exact as far as the executable format is concerned.  It was 32,
# which reaches 5.3e-7: three orders of refinement that `apply_weights` rounds
# away, paid for on every one of ~100,000 evaluations.
GOLDEN = (5.0 ** 0.5 - 1.0) / 2.0
GOLDEN_ITERS = 20
# Screening a start does not need that precision -- it only has to say which
# basin is worth refining.  Measured on USDC->WETH $100k: 10 starts, 11,644
# evaluations, and 11,532 of them spent on the nine that lost.  Eight converged
# to the same value to six figures; the winner took 112.
SCREEN_ITERS = 6
SCREEN_SWEEPS = 3
# How many of the screened starts earn a full-precision ascent.  More than one
# because screening is coarse and the margin between basins here was 0.155 bp,
# which is not a gap a 6-bisection search can be trusted to rank.
REFINE_STARTS = 3
# Coordinate sweeps stop when one buys less than this, relatively.
SWEEP_TOL = 1e-9
MAX_SWEEPS = 12
# The optimum is proposed to the chain together with its neighbours, so one
# verification batch both picks the winner and measures how far the composed
# curves drifted from the real chained quote.
PROBE_OFFSETS = (0.01, 0.04)
# A leg's ladder always reaches at least this far past its realised input, even
# when the slot bound collapses -- which it does whenever the model gave some
# upstream leg zero flow, leaving a rate of 0 to propagate.
NOMINAL_HEADROOM = 4.0
# The composed curves are checked against the one chained quote already in
# hand, at the weights it was taken at.  Beyond this the objective is not
# trustworthy and the search refines rather than optimising on it.
CHECK_TOL_BP = 1.0
# Refinement: a narrow window around each leg's operating size, sampled hard.
# Linear-`u` error falls as `(r - 1)^2` in the node ratio, and the coarse
# ladder's 1.42 is what put a stableswap's peg edge 497 bp out; 1.015 is two
# orders better, which is what it takes.
REFINE_SPAN = 2.0
REFINE_NODES = 96
# Only legs whose own ladder says they are this badly interpolated get the
# dense pass.  On the measured routes that is three legs of twelve, not twelve.
LEG_TOL_BP = 0.05

# --- the exact pass, when a quote is cheap ---------------------------------
#
# An in-process EVM prices a route in milliseconds, so the last word can be
# taken from the true chained function instead of an interpolant.  The curves
# still do the searching -- a composed evaluation is microseconds against
# milliseconds -- but they no longer decide the answer, which is the one place
# interpolation error would have shown up in basis points.
POLISH_ITERS = 12
POLISH_SWEEPS = 2
# The curve search has already found the basin; this only refines inside it.
POLISH_WINDOW = 0.05
#: Skip the polish when the curves already agree with the chain this closely.
#
# Polish exists to correct interpolation drift, and the run measures that drift
# for free -- `check_bp` composes the curves at the weights a real chained quote
# was taken at.  When it is small there is nothing left to find, and measured,
# it is usually small: WETH->USDC 300 checked at -0.078 bp and polishing bought
# 0.031 for 241 sequential quotes, about 770 ms; USDC->sUSDS $1M checked at
# -0.045 and bought exactly nothing for 121 calls.  Each of those calls is a
# chained `quote_routes`, ~3.2 ms in-process and a round trip on the wire, so
# this is the single most expensive thing the router does per basis point.
#
# The threshold is above the drift both those cases showed and below the
# 1.0 bp at which `_trusted_curves` stops believing the curves at all.
POLISH_CHECK_BP = 0.15
# With an exact finish available, the curves only have to be right enough to
# land in the correct basin, so the dense second sampling pass is not needed.
LOCAL_CHECK_TOL_BP = 10.0


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
    mode: str = "chained"
    probes: int = 0
    polish_calls: int = 0
    polish_skipped: bool = False
    polish_bp: float = 0.0
    local: int = 0
    predicted: int = 0
    refined: bool = False
    check_bp: float = 0.0

    @property
    def gain_bp(self) -> float:
        return (self.after / self.before - 1) * 10_000 if self.before else 0.0

    @property
    def curve_error_bp(self) -> float:
        """How far the composed curves missed the real chained quote.

        The whole case for sampling rests on this being small, so it is
        reported rather than assumed.  Signed: positive means the curves
        over-promised.
        """
        if not (self.predicted > 0 and self.after > 0):
            return 0.0
        return (self.predicted / self.after - 1) * 10_000


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


# ------------------------------------------------------- sampled-curve search


def reachable_tops(
    legs: list[Leg], nominal_in: list[int], nominal_out: list[int], amount_in: int
) -> list[float]:
    """The largest input each leg can be handed, over the whole weight simplex.

    A leg can take at most its source slot's entire balance, so one forward
    pass bounds every leg: what arrives at a slot is bounded by what its
    feeders can send, and each feeder's rate is taken from the realised
    amounts.  Linear rates over-state a concave leg's output, which is the
    direction that keeps this an upper bound -- and an upper bound is all it
    has to be, since it only decides where the probe ladder stops.
    """
    cap: dict[int, float] = {0: float(amount_in)}
    tops: list[float] = []
    for k, leg in enumerate(legs):
        # The slot bound alone is not enough.  A leg whose modelled output was
        # zero propagates a rate of zero, and every leg downstream of it then
        # bounds at nothing -- so each leg also floors on its own realised
        # input, which is a size the chain demonstrably served.
        top = max(cap.get(leg.src_slot, 0.0), nominal_in[k] * NOMINAL_HEADROOM)
        tops.append(top)
        rate = (nominal_out[k] / nominal_in[k]) if nominal_in[k] > 0 else 0.0
        cap[leg.dst_slot] = cap.get(leg.dst_slot, 0.0) + top * rate
    return tops


# Exactly 1:1 in both directions and implemented without an external call, so
# probing them spends a slot in the batch to be told what the enum already says.
UNIT_KINDS = frozenset({ArcKind.WRAP_NATIVE, ArcKind.UNWRAP_NATIVE, ArcKind.STAKE_NATIVE})


def _probe_ladders(legs, client, ladders, report, *, optional: bool = False):
    """Quote every leg's ladder in one batch.  None if a leg comes back unusable.

    These probes are independent, so a whole route's sampling is one round trip
    -- the property a chained quote does not have, and the reason any of this
    is worth doing.
    """
    plans: list[tuple[list[int], int, int] | None] = []
    probes: list[Probe] = []
    for leg, deltas in zip(legs, ladders, strict=True):
        if leg.kind in UNIT_KINDS:
            plans.append(None)
            continue
        if len(deltas) < 3:
            if not optional:
                return None
            plans.append(None)
            continue
        start = len(probes)
        probes.extend(Probe(leg.target, leg.kind, leg.i, leg.j, leg.n, d) for d in deltas)
        plans.append((deltas, start, len(probes)))
    if report is not None:
        report.calls += 1
        report.probes += len(probes)
    results = client.probe(probes) if probes else []
    if len(results) != len(probes):
        return None
    out: list[tuple[list[float], list[float]] | None] = []
    for plan in plans:
        if plan is None:
            out.append(None)
            continue
        deltas, lo, hi = plan
        xs: list[float] = []
        ys: list[float] = []
        for delta, answer in zip(deltas, results[lo:hi], strict=True):
            status = getattr(answer, "status", None)
            if status is not None and status.name != "VALUE":
                continue
            value = int(getattr(answer, "value", 0) or 0)
            if value <= 0:
                continue
            xs.append(float(delta))
            ys.append(float(value))
        if len(xs) < 2:
            return None
        out.append((xs, ys))
    return out


def _build(points):
    """Turn probe points into curves, `None` entries becoming the identity."""
    out = []
    for pair in points:
        if pair is None:
            out.append(curve_mod.linear(1.0))
            continue
        try:
            out.append(curve_mod.fit(*pair))
        except curve_mod.CurveError:
            return None
    return out


def sample_curves(legs, client, tops, report=None):
    """The wide, coarse first pass: enough shape to let the weights move far."""
    points = _probe_ladders(legs, client, [curve_mod.sizes(t) for t in tops], report)
    return None if points is None else _build(points)


def refine_curves(legs, client, curves, takes, points, report=None):
    """A second pass, dense and narrow, around where the legs actually operate.

    Only the legs that need it: each curve estimates its own interpolation
    error from the secants it already has, so a route where one stableswap sits
    on a peg edge refines that one leg rather than all fifteen.

    Merged into the coarse points rather than replacing them, so a weight that
    later wanders outside the refined window degrades to the wide ladder
    instead of extrapolating off the end of a narrow one.
    """
    ladders = []
    for curve, take in zip(curves, takes, strict=True):
        if take <= 1.0 or curve.error_bp_at(take) <= LEG_TOL_BP:
            ladders.append([])
            continue
        lo, hi = take / REFINE_SPAN, take * REFINE_SPAN
        ladders.append(curve_mod.sizes(hi, nodes=REFINE_NODES, span=hi / max(lo, 1.0)))
    if not any(ladders):
        return points  # nothing to refine; the caller's check will decide
    extra = _probe_ladders(legs, client, ladders, report, optional=True)
    if extra is None:
        return None
    merged = []
    for before, more in zip(points, extra, strict=True):
        if before is None or more is None:
            merged.append(before)
            continue
        pairs = sorted({**dict(zip(*before, strict=True)), **dict(zip(*more, strict=True))}.items())
        merged.append(([x for x, _ in pairs], [y for _, y in pairs]))
    return merged


def _fractions(legs: list[Leg], groups: list[list[int]], weights) -> list[float | None]:
    """Per-leg share of its group's base balance; None means "sweep the rest"."""
    grouped = {k for run in groups for k in run}
    out: list[float | None] = [
        None if (k in grouped or leg.bps == 0) else leg.bps / BPS
        for k, leg in enumerate(legs)
    ]
    for run, w in zip(groups, weights, strict=True):
        w = _project(w)
        for index, value in zip(run[:-1], w[:-1], strict=True):
            out[index] = float(value)
        out[run[-1]] = None
    return out


def walk(legs: list[Leg], curves, fractions, amount_in: float, dst_slot: int) -> float:
    """Compose the leg curves the way the quoter composes the real calls.

    Same group-snapshot semantics as `RouteQuoter._walk` and
    `realize._forward_simulate`: the base balance is captured when `src_slot`
    changes, so a share is stable regardless of draining order.  Floats, not
    integers -- the executable rounding happens once, in `apply_weights`.
    """
    balances: dict[int, float] = {0: float(amount_in)}
    current: int | None = None
    base = 0.0
    for k, leg in enumerate(legs):
        src = leg.src_slot
        if src != current:
            current = src
            base = balances.get(src, 0.0)
        available = balances.get(src, 0.0)
        share = fractions[k]
        take = available if share is None else min(base * share, available)
        if take <= 0.0:
            continue
        balances[src] = available - take
        balances[leg.dst_slot] = balances.get(leg.dst_slot, 0.0) + curves[k].at(take)
    return balances.get(dst_slot, 0.0)


def takes_of(legs: list[Leg], curves, fractions, amount_in: float) -> list[float]:
    """Each leg's input under one weight vector -- where to refine its ladder."""
    balances: dict[int, float] = {0: float(amount_in)}
    current: int | None = None
    base = 0.0
    out: list[float] = []
    for k, leg in enumerate(legs):
        src = leg.src_slot
        if src != current:
            current = src
            base = balances.get(src, 0.0)
        available = balances.get(src, 0.0)
        share = fractions[k]
        take = available if share is None else min(base * share, available)
        out.append(max(0.0, take))
        if take <= 0.0:
            continue
        balances[src] = available - take
        balances[leg.dst_slot] = balances.get(leg.dst_slot, 0.0) + curves[k].at(take)
    return out


def make_evaluator(legs: list[Leg], groups: list[list[int]], curves,
                   amount_in: float, dst_slot: int):
    """`walk(_fractions(...))`, specialised for the inner loop.

    The search calls this ~100,000 times on a wide route, and the reference
    pair spends most of that rebuilding things that never change: `_fractions`
    reconstructs the grouped-index set and the whole fractions list per call,
    `_project` runs three numpy calls on arrays of two to five elements where
    the call overhead dwarfs the arithmetic, and `walk` carries balances in a
    dict and re-reads `leg.src_slot` per leg.  Measured on USDC->WETH $100k,
    29 legs: 108,520 evaluations cost 10.0 s, of which `_project` alone was
    3.6 s across 545,065 calls.

    None of it can be vectorised across legs -- the walk is sequential by
    construction, each leg spending what its predecessors left -- so this
    hoists the invariants out of the loop and keeps the arithmetic identical,
    rather than replacing it with something array-shaped.

    `walk` and `_fractions` stay as the readable definition of what this
    computes, and `test_split.py` holds the two to each other.
    """
    count = len(legs)
    grouped = {k for run in groups for k in run}
    # Ungrouped legs keep whatever share they were realised with, forever.
    static: list[float | None] = [
        None if (k in grouped or leg.bps == 0) else leg.bps / BPS
        for k, leg in enumerate(legs)
    ]
    src_of = [leg.src_slot for leg in legs]
    dst_of = [leg.dst_slot for leg in legs]
    # Bound methods, so the loop skips attribute lookup on every leg.
    at_of = [curve.at for curve in curves]
    slots = max([dst_slot, *src_of, *dst_of]) + 1
    fractions: list[float | None] = list(static)
    heads = [run[:-1] for run in groups]
    tails = [run[-1] for run in groups]
    start = float(amount_in)

    def evaluate(weights) -> float:
        for head, tail, w in zip(heads, tails, weights, strict=True):
            # `_project`, unrolled: clip up to MIN_WEIGHT, then normalise.
            total = 0.0
            clipped = []
            for value in w:
                one = value if value > MIN_WEIGHT else MIN_WEIGHT
                clipped.append(one)
                total += one
            if total > 0.0:
                scale = 1.0 / total
                for index, one in zip(head, clipped, strict=False):
                    fractions[index] = one * scale
            else:
                share = 1.0 / len(clipped)
                for index in head:
                    fractions[index] = share
            fractions[tail] = None

        balances = [0.0] * slots
        balances[0] = start
        current = -1
        base = 0.0
        for k in range(count):
            source = src_of[k]
            if source != current:
                current = source
                base = balances[source]
            available = balances[source]
            share = fractions[k]
            take = available if share is None else min(base * share, available)
            if take <= 0.0:
                continue
            balances[source] = available - take
            balances[dst_of[k]] += at_of[k](take)
        return balances[dst_slot]

    return evaluate


def _golden(objective, lo: float, hi: float, *, iters: int = GOLDEN_ITERS) -> float:
    """Maximise a unimodal `objective` on `[lo, hi]` without derivatives."""
    a, b = lo, hi
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = objective(d)
        else:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = objective(c)
    return 0.5 * (a + b)


def _ascend(start, evaluate, free, counter, *, iters: int = GOLDEN_ITERS,
            sweeps: int = MAX_SWEEPS, window: float = 0.0) -> tuple[list, float]:
    """Coordinate ascent, each coordinate maximised exactly by golden section.

    No step size, no gradient, no normalisation -- all of which existed only to
    ration round trips.  With the curves in hand an evaluation is microseconds,
    so each coordinate can simply be solved.
    """
    weights = [w.copy() for w in start]
    best = evaluate(weights)
    for _ in range(sweeps):
        opened = best
        for g, j in free:
            others = float(weights[g][:-1].sum()) - float(weights[g][j])
            room = 1.0 - MIN_WEIGHT - others
            if room <= MIN_WEIGHT:
                continue
            low = MIN_WEIGHT
            if window > 0.0:
                here = float(weights[g][j])
                low, room = max(low, here - window), min(room, here + window)
                if room <= low:
                    continue

            def objective(value, g=g, j=j, here=weights):
                # Only group `g` changes, and `evaluate` never mutates what it
                # is given, so the other groups can be shared rather than
                # copied -- this runs once per golden-section probe.
                counter[0] += 1
                trial = list(here)
                row = here[g].copy()
                row[j] = value
                row[-1] = max(MIN_WEIGHT, 1.0 - float(row[:-1].sum()))
                trial[g] = row
                return evaluate(trial)

            where = _golden(objective, low, room, iters=iters)
            candidate = [w.copy() for w in weights]
            candidate[g][j] = where
            candidate[g][-1] = max(MIN_WEIGHT, 1.0 - float(candidate[g][:-1].sum()))
            value = evaluate(candidate)
            if value > best:
                best, weights = value, candidate
        if best <= opened * (1.0 + SWEEP_TOL):
            break
    return weights, best


def optimise(
    legs: list[Leg],
    client: QuoterClient,
    *,
    amount_in: int,
    dst_slot: int,
    baseline: int = 0,
    nominal_in: list[int] | None = None,
    nominal_out: list[int] | None = None,
    max_rounds: int = MAX_ROUNDS,
    hot_rounds: int = HOT_ROUNDS,
) -> tuple[list[Leg], SplitReport]:
    """Re-split a finished route.  Returns the best legs found and a report.

    Sampled curves when the caller can supply the realised per-leg amounts --
    two round trips, converged.  The chained hill-climb otherwise, and as a
    fallback whenever a leg refuses to probe.
    """
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

    if nominal_in and nominal_out and hasattr(client, "probe") and baseline > 0:
        curves = _trusted_curves(
            legs, client, groups, weights, report, amount_in=amount_in,
            dst_slot=dst_slot, baseline=baseline,
            nominal_in=nominal_in, nominal_out=nominal_out,
        )
        if curves is not None:
            return _search_curves(
                legs, client, groups, weights, free, curves, report,
                amount_in=amount_in, dst_slot=dst_slot, baseline=baseline,
                budget=budget,
            )

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


def polish(
    legs, client, groups, start, free, report, *, amount_in: int, dst_slot: int,
    baseline: int,
) -> tuple[list, int]:
    """Finish on the true chained function, not on an interpolant.

    Only worth doing where a quote is cheap: each evaluation here is a real
    `quote_routes`, which on the wire is a round trip and in-process is
    milliseconds.  The window is deliberately small -- the curve search has
    already chosen the basin, and this decides only where inside it the answer
    sits, which is exactly the decision interpolation error was distorting.
    """
    calls = [0]

    def exact(candidate) -> float:
        calls[0] += 1
        routes = [apply_weights(legs, groups, candidate)]
        got = client.quote_routes(routes, [amount_in], [dst_slot])
        return float(got[0]) if got else 0.0

    tuned, value = _ascend(
        [_project(w) for w in start], exact, free, [0],
        iters=POLISH_ITERS, sweeps=POLISH_SWEEPS, window=POLISH_WINDOW,
    )
    report.polish_calls = calls[0]
    report.evaluations += calls[0]
    if value <= baseline:
        return start, baseline
    report.polish_bp = (value / baseline - 1) * 10_000 if baseline else 0.0
    return tuned, int(value)


def _trusted_curves(
    legs, client, groups, weights, report, *,
    amount_in: int, dst_slot: int, baseline: int, nominal_in, nominal_out,
):
    """Sample the legs, and refuse to optimise on curves that fail their check.

    The check is free: `baseline` is a real chained quote at known weights, so
    composing the curves at those same weights compares like with like.  A
    stableswap crossing its peg edge inside one ladder interval was measured
    497 bp out on a coarse ladder, which would have made the objective fiction
    -- so a curve set that misses gets one dense pass at the sizes the legs
    actually operate at, and if it still misses, the chained search takes over.
    """
    tops = reachable_tops(legs, nominal_in, nominal_out, amount_in)
    points = _probe_ladders(legs, client, [curve_mod.sizes(t) for t in tops], report)
    if points is None:
        report.skipped = "a leg would not probe"
        return None
    curves = _build(points)
    start = _fractions(legs, groups, weights)

    def missed() -> float:
        composed = walk(legs, curves, start, amount_in, dst_slot)
        report.check_bp = (composed / baseline - 1) * 10_000
        return abs(report.check_bp)

    tolerance = LOCAL_CHECK_TOL_BP if getattr(client, "local", False) else CHECK_TOL_BP
    if curves is None or missed() > tolerance:
        if curves is None:
            report.skipped = "a leg would not fit"
            return None
        spent = report.probes
        points = refine_curves(
            legs, client, curves, takes_of(legs, curves, start, amount_in),
            points, report,
        )
        if points is None:
            report.skipped = "refinement would not probe"
            return None
        curves = _build(points)
        report.refined = report.probes > spent
        tolerance = LOCAL_CHECK_TOL_BP if getattr(client, "local", False) else CHECK_TOL_BP
    if curves is None or missed() > tolerance:
            report.skipped = f"curves miss the chain by {report.check_bp:.1f} bp"
            return None
    return curves


def _search_curves(
    legs, client, groups, weights, free, curves, report, *,
    amount_in: int, dst_slot: int, baseline: int, budget: int,
) -> tuple[list[Leg], SplitReport]:
    """Optimise on the sampled curves, then let the chain adjudicate once.

    The curves are exact at their nodes but interpolated between them, so the
    optimum they find is proposed to the quoter *with its neighbours*: one
    batch picks the real winner and simultaneously measures how far the
    composition drifted, which is the number the whole approach stands on.
    """
    report.mode = "curves"
    counter = [0]

    fast = make_evaluator(legs, groups, curves, amount_in, dst_slot)

    def evaluate(candidate) -> float:
        counter[0] += 1
        return fast(candidate)

    # Several starts, because coordinate ascent finds a local optimum and the
    # model's own split is not a neutral place to begin from.
    starts = [[w.copy() for w in weights]]
    starts.append([np.full(w.size, 1.0 / w.size) for w in weights])
    for g, run in enumerate(groups):
        for j in range(len(run)):
            corner = [w.copy() for w in weights]
            corner[g] = np.full(len(run), MIN_WEIGHT)
            corner[g][j] = 1.0 - MIN_WEIGHT * (len(run) - 1)
            starts.append(corner)

    # Screen, then refine.  Every start used to run to full precision, and on a
    # wide route that is where the whole cost sat: 11,644 evaluations over ten
    # starts, of which 11,532 went to the nine that lost -- eight of them
    # converging to the same value to six figures.  A start only has to say
    # which basin it is in to be ranked, and that is a much cheaper question
    # than where the optimum inside it lies.
    #
    # `REFINE_STARTS` of them go on to a full ascent rather than one, because
    # the winner here beat the rest by 0.155 bp and a 6-bisection screen cannot
    # be trusted to resolve that.
    screened = []
    for start in starts:
        projected = [_project(w) for w in start]
        _, value = _ascend(projected, evaluate, free, counter,
                           iters=SCREEN_ITERS, sweeps=SCREEN_SWEEPS)
        screened.append((value, projected))
    screened.sort(key=lambda pair: -pair[0])

    best_w, best_value = None, -1.0
    for _, projected in screened[:REFINE_STARTS]:
        found, value = _ascend(projected, evaluate, free, counter)
        if value > best_value:
            best_w, best_value = found, value
    report.local = counter[0]
    if best_w is None:
        report.skipped = "no local optimum"
        report.after = baseline
        return legs, report

    # The optimum, plus a neighbourhood wide enough to cover the interpolation
    # error and the `bps` rounding that `apply_weights` is about to impose.
    candidates = [best_w]
    for g, j in free:
        for offset in PROBE_OFFSETS:
            for sign in (1.0, -1.0):
                trial = [w.copy() for w in best_w]
                trial[g][j] += sign * offset
                trial[g][-1] -= sign * offset
                candidates.append([_project(w) for w in trial])
    candidates = candidates[:budget]

    routes = [apply_weights(legs, groups, w) for w in candidates]
    report.calls += 1
    report.evaluations += len(routes)
    values = client.quote_routes(routes, [amount_in] * len(routes), [dst_slot] * len(routes))
    report.rounds = 1
    if not values:
        report.after = baseline
        return legs, report

    report.predicted = int(best_value)
    pick = int(np.argmax(values))
    winner, best = candidates[pick], int(values[pick])

    # The batch above is the last word only when a quote is expensive.  With an
    # in-process EVM it is merely a good starting point, and the true function
    # is affordable enough to be optimised directly -- but only where the
    # curves and the chain actually disagree.  See `POLISH_CHECK_BP`.
    drifting = abs(report.check_bp) >= POLISH_CHECK_BP
    report.polish_skipped = not drifting
    if getattr(client, "local", False) and best > 0 and drifting:
        winner, best = polish(
            legs, client, groups, winner, free, report,
            amount_in=amount_in, dst_slot=dst_slot, baseline=best,
        )

    if best <= baseline:
        report.after = baseline
        return legs, report
    report.improved = True
    report.after = best
    return apply_weights(legs, groups, winner), report


@dataclass(slots=True)
class ScoutResult:
    """One candidate topology, re-split against the shared curves."""

    index: int
    predicted: float
    legs: list[Leg]


def scout(plans, client: QuoterClient, *, amount_in: int,
          report: SplitReport | None = None) -> list[ScoutResult]:
    """Re-split several candidate topologies against **one** probe batch.

    The problem this exists for: candidates are ranked on the split the model
    gave them, and the model's split is excellent on narrow topologies and
    terrible on wide ones.  Measured on WETH->USDC 300, the winning 5-leg route
    gained 0.14 bp when its split was tuned while a 14-leg candidate gained
    102 -- so the wide one looked 100 bp worse than it was, lost the ranking,
    and never reached the pass that would have fixed it.  The router shipped a
    route 22 bp below its own best idea.

    Tuning each candidate properly is not affordable: it triples a warm route,
    median +1.4 s and up to +12 s, for a median gain of 0.01 bp.  Nor does
    scouting with a cheap round then converging the winner -- measured at 136%
    of the cost, because the optimiser's expense is its *first* round, the one
    that samples curves, not the convergence afterwards.

    What makes it cheap is that the candidates are nested: `top 6 pools`,
    `top 8 pools` and `top 12 pools` are largely the same arcs.  So the arcs
    are pooled, sampled once at the widest ladder anyone needs, and every
    candidate is then optimised against that shared set in pure Python, where
    an evaluation is microseconds.  One batch, no matter how many candidates.

    `plans` is `(legs, dst_slot, nominal_in, nominal_out)` per candidate.
    Returns a `ScoutResult` per usable candidate with the *predicted* output --
    predicted, not verified: composing interpolated curves is what the model
    was doing wrong in the first place, so nothing here may be believed without
    a real quote.  The caller quotes the survivors.
    """
    tops: dict[tuple, float] = {}
    for legs, _dst, nominal_in, nominal_out in plans:
        try:
            reach = reachable_tops(legs, nominal_in, nominal_out, amount_in)
        except (ValueError, ZeroDivisionError):
            continue
        for leg, top in zip(legs, reach, strict=True):
            key = (leg.target.lower(), int(leg.kind), leg.i, leg.j)
            tops[key] = max(tops.get(key, 0.0), float(top))
    if not tops:
        return []

    # One synthetic leg per distinct arc, laddered to the widest size any
    # candidate could hand it.  A curve sampled wide still serves a narrower
    # use: it is a function of the input, and the extra nodes only sit above.
    keys = list(tops)
    probe_legs = [
        Leg(target=key[0], kind=ArcKind(key[1]), i=key[2], j=key[3],
            n=max(key[2], key[3]) + 1, src_slot=0, dst_slot=1)
        for key in keys
    ]
    points = _probe_ladders(
        probe_legs, client, [curve_mod.sizes(tops[key]) for key in keys],
        report, optional=True,
    )
    if points is None:
        return []
    built = _build(points)
    if built is None:
        return []
    shared = dict(zip(keys, built, strict=True))

    out: list[ScoutResult] = []
    for index, (legs, dst_slot, _nominal_in, _nominal_out) in enumerate(plans):
        try:
            curves = [
                shared[(leg.target.lower(), int(leg.kind), leg.i, leg.j)]
                for leg in legs
            ]
        except KeyError:  # an arc the batch could not price
            continue
        groups = split_groups(legs)
        if not groups:
            continue
        weights = weights_of(legs, groups)
        free = [(g, j) for g, run in enumerate(groups) for j in range(len(run) - 1)]
        if not free:
            continue
        evaluate = make_evaluator(legs, groups, curves, amount_in, dst_slot)
        counter = [0]
        # Screening depth only.  This has to rank topologies, not locate an
        # optimum inside one -- the winner gets the real optimiser afterwards.
        best_w, best_value = None, -1.0
        for start in ([w.copy() for w in weights],
                      [np.full(w.size, 1.0 / w.size) for w in weights]):
            projected = [_project(w) for w in start]
            found, value = _ascend(projected, evaluate, free, counter,
                                   iters=SCREEN_ITERS, sweeps=SCREEN_SWEEPS)
            if value > best_value:
                best_w, best_value = found, value
        if best_w is None:
            continue
        out.append(ScoutResult(index=index, predicted=best_value,
                               legs=apply_weights(legs, groups, best_w)))
    out.sort(key=lambda found: -found.predicted)
    return out
