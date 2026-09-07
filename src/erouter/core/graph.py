"""Value-coordinate graph assembly (spec §3.1, §9.5-9.7).

Working in *value* rather than token units is what makes the dual Hessian a
plain graph Laplacian instead of a gain-graph one, and it is also why arc
conductance is direction-symmetric while `a` and `B` are wildly asymmetric.

Everything is struct-of-arrays: the solver never sees a Python object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# §9.6 an arc that cannot carry meaningful flow only adds pivots
DUST_FLOOR = 1e-6
#: How much free value an arc may claim before it is a bug rather than an
#: opportunity.  `eps = -1` is already "pays twice its input"; see `build`.
EPS_FLOOR = -1.0
# §9.7 clamped (B=0) arcs would otherwise carry G = inf
CEILING_FACTOR = 1e3
MAX_CONDITION = 1e12
# What the adaptive dust floor aims at, with headroom below MAX_CONDITION.
TARGET_CONDITION = 1e11
# Beyond this, the spread is not a wide universe -- it is a bug.
PATHOLOGICAL_CONDITION = 1e15


def arc_params(
    tau: np.ndarray,
    sig: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    nu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(M3) conductance and (M4) forward drop, in value coordinates.

        G_p = nu_tau * a_p / B_p        value scale x token-space conductance
        eps_p = 1 - a_p * nu_sig / nu_tau

    For a constant-product pool G collapses to TVL/4 -- the elementary
    "resistance of a pool is 4/TVL" result.

    `eps` may be negative: that is a favourably dislocated pool, an EMF, and it is
    exactly how arbitrage enters the routing problem.

    B == 0 is the admissible zero-curvature limit (§2.3), giving G = inf here;
    `ceiling_conductance` bounds it afterwards.  B < 0 is *not* admissible and
    must have been clamped at calibration -- it is rejected loudly rather than
    turned into a negative resistor.
    """
    if np.any(B < 0):
        bad = int(np.argmin(B))
        raise ValueError(
            f"negative curvature reached the graph (arc {bad}, B={B[bad]:.3e}). "
            "calibrate() must clamp B to 0; a negative G makes the Laplacian "
            "indefinite and voids the certificate (§11.2)."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(B > 0, nu[tau] * a / np.where(B > 0, B, 1.0), np.inf)
    eps = 1.0 - a * nu[sig] / nu[tau]
    return G, eps


def reference_conductance(
    G: np.ndarray,
    flagged: np.ndarray,
    cap: np.ndarray | None = None,
) -> float:
    """The largest conductance the graph's scale should be read from.

    An arc's `G` says how much flow it takes per unit of potential, which is
    what it is worth *only while the arc is free*.  An arc with a finite cap
    stops taking flow at its cap and leaves the active set, so past that point
    its `G` describes nothing -- and a `G` that describes nothing must not set
    the scale everything else is measured against.

    On a Curve universe this is exactly `max(finite)`: a clamped arc there has
    `B = 0` and so `G = inf`, and every arc with a finite `G` is uncapped.  It
    starts mattering when a venue supplies both at once -- a Uniswap v3 tick is
    nearly linear over its own range, so its `B` is tiny, its `G` is enormous,
    and its cap is a few tens of thousands of dollars.  Letting one of those set
    the reference raised `build`'s dust floor by eleven orders and evicted 1,342
    Curve arcs that had done nothing wrong.
    """
    free = np.isfinite(G) & ~flagged
    if cap is not None:
        unbounded = free & ~np.isfinite(cap)
        if unbounded.any():
            return float(G[unbounded].max())
    return float(G[free].max()) if free.any() else 1.0


def ceiling_conductance(
    G: np.ndarray,
    flagged: np.ndarray,
    factor: float = CEILING_FACTOR,
    cap: np.ndarray | None = None,
) -> np.ndarray:
    """§2.3 rule (3) / §9.7 -- clamp in G-space, never by flooring B.

    A 1e-30 floor on B becomes a 1e30 conductance and destroys the condition
    number of the whole Laplacian, which is worse than the defect it patches.

    Only clamped arcs (`G = inf`) are lifted.  Do **not** lower this ceiling to
    bound the condition number -- flattening real conductances makes every deep
    pool look identical, and the solver then splits arbitrarily among them
    instead of by depth.  Bound the spread from below instead (see `dust_floor`
    in `build`).

    A capped arc is neither, and gets `compress_conductance` instead: it must
    not *set* the scale, because its `G` stops describing it at its cap, and it
    must not be flattened to that scale either, because between deep and
    shallow the difference is exactly the depth the solver splits by.
    """
    reference = reference_conductance(G, flagged, cap)
    lifted = np.minimum(np.where(np.isfinite(G), G, np.inf), factor * reference)
    if cap is None:
        return lifted
    bounded = np.isfinite(cap) & np.isfinite(G)
    return np.where(bounded, compress_conductance(G, bounded, reference, factor),
                    lifted)


def compress_conductance(
    G: np.ndarray,
    bounded: np.ndarray,
    reference: float,
    factor: float = CEILING_FACTOR,
) -> np.ndarray:
    """Squeeze capped conductances into `[reference, factor * reference]`,
    monotonically.

    Both of the obvious answers were measured on a Curve+v3 graph and both are
    wrong.  Flattening capped arcs to the ceiling cost 7.9 bp: every deep pool
    looks identical and the solver splits arbitrarily among them.  Leaving them
    alone leaves a spread §12.4 cannot accept: v3 ticks reach `G = 1.4e11`
    beside Curve arcs at 7e-1, and once `condition` stops looking away from them
    `build` refuses the graph at 2.758e12 -- 6 of 21 sweep cases, every one of
    them carrying WBTC ticks.

    So neither: keep the order and lose the range.  Below the reference nothing
    moves.  Above it, `G -> reference * (G/reference)^alpha` with `alpha` set so
    the largest lands exactly on the ceiling -- continuous at the knee, strictly
    increasing either side of it, and `alpha < 1` precisely when there is
    something above the ceiling to bring down.  A deeper tick stays deeper; it
    just stops being deeper by eleven orders of magnitude.

    Choosing the band to be the one the ceiling already defines is what keeps
    `build`'s dust floor honest: it budgets against `factor * reference`, so
    with nothing above that the post-ceiling spread really is TARGET_CONDITION.
    """
    if not bounded.any() or not np.isfinite(reference) or reference <= 0:
        return G
    top = float(G[bounded].max())
    if top <= factor * reference or factor <= 1.0:
        return G
    alpha = np.log(factor) / np.log(top / reference)
    over = bounded & (G > reference)
    ratio = np.where(over, G / reference, 1.0)
    return np.where(over, reference * ratio**alpha, G)


@dataclass(slots=True)
class ArcArrays:
    """Solver input.  Index space is post-dust, post-duplicate-merge."""

    tau: np.ndarray
    sig: np.ndarray
    a: np.ndarray
    B: np.ndarray
    G: np.ndarray
    eps: np.ndarray
    cap: np.ndarray
    flagged: np.ndarray
    clamped: np.ndarray
    n_nodes: int
    g_scale: float = 1.0
    # Set when the dust floor had to be backed off for connectivity and the
    # resulting spread exceeds MAX_CONDITION.  Non-fatal, but worth surfacing.
    ill_conditioned: float = 0.0
    # index -> original arc indices (a merged duplicate group has several)
    sources: list[list[int]] = field(default_factory=list)
    dropped: dict[int, str] = field(default_factory=dict)
    # Somewhere for an accelerator to keep a resident copy of this graph across
    # the ~90 solves one quote runs over it.  `slots=True` means there is
    # otherwise nowhere to put it, and the attempt fails silently.  Not part of
    # the value -- excluded from `__eq__` and `repr`.
    accel: object | None = field(default=None, repr=False, compare=False)

    @property
    def m(self) -> int:
        return len(self.tau)

    def condition(self) -> float:
        """The conductance spread the solve has to live with -- every arc of it.

        Capped arcs were excluded here for a while, on the argument that an arc
        which reaches its cap leaves the active set and so does not condition
        the system for long.  The factorisation disagreed: it sees the whole
        matrix, and on a Curve+v3 graph the excluded part reached 2.8e12 while
        this returned 7e6, so §12.4's bound was being checked against a number
        that left out the entire problem.  `compress_conductance` is what makes
        counting them affordable again.
        """
        positive = self.G[self.G > 0]
        return float(positive.max() / positive.min()) if positive.size else 1.0


def build(
    tau: np.ndarray,
    sig: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    nu: np.ndarray,
    Psi: float,
    *,
    cap: np.ndarray | None = None,
    flagged: np.ndarray | None = None,
    clamped: np.ndarray | None = None,
    n_nodes: int | None = None,
    dust_floor: float = DUST_FLOOR,
    ceiling_factor: float = CEILING_FACTOR,
    max_spread: float = PATHOLOGICAL_CONDITION,
    merge_duplicates: bool = True,
    require: tuple[int, int] | None = None,
) -> ArcArrays:
    """Assemble solver arrays, in the order §9.5-9.7 requires.

    Dust first (so the ceiling reference is meaningful), then duplicate merge,
    then the conductance ceiling, then the invariants.
    """
    tau = np.asarray(tau, dtype=np.int64)
    sig = np.asarray(sig, dtype=np.int64)
    a = np.asarray(a, dtype=float)
    B = np.asarray(B, dtype=float)
    m = len(tau)
    n = int(n_nodes if n_nodes is not None else max(tau.max(), sig.max()) + 1)

    cap = np.full(m, np.inf) if cap is None else np.asarray(cap, dtype=float)
    flagged = np.zeros(m, bool) if flagged is None else np.asarray(flagged, bool)
    clamped = (B == 0.0) if clamped is None else np.asarray(clamped, bool)

    # §12.4: a zero-curvature arc has no self-limiting term, so without a finite
    # cap a negative-eps cycle gives unbounded flow.  Fail here, not in the solve.
    unbounded = clamped & ~np.isfinite(cap)
    if unbounded.any():
        raise ValueError(
            f"clamped arcs {[int(k) for k in np.flatnonzero(unbounded)]} have no finite cap; "
            "flow would be unbounded (§2.3 rule 2)"
        )

    G, eps = arc_params(tau, sig, a, B, nu)

    sources = [[k] for k in range(m)]
    dropped: dict[int, str] = {}

    # --- §9.6 dust ------------------------------------------------------
    #
    # §9.6's floor is `1e-6 * Psi`.  On a real universe that is not enough on its
    # own: genuine conductances span 4e10 on Ethereum -- a $45k pool against a
    # deep stableswap near its peg -- so `max/min` can breach §12.4's 1e12 bound
    # without anything being wrong.
    #
    # Raise the floor rather than lower the ceiling.  An arc with
    # `G = 1e-4 * Psi` carrying even a 0.1% share would lose ~0.5% to impact, so
    # it can never be part of a sensible route.
    base_floor = dust_floor * Psi
    floor = base_floor
    finite_G = G[np.isfinite(G)]
    positive_G = finite_G[finite_G > 0]
    # Measured over the arcs that will survive the floor, not over every arc.
    #
    # What this catches is a `B` floored instead of a `G` ceilinged, which puts a
    # spike at the *top*.  The bottom is a different animal: a dust pool almost
    # entirely on one side genuinely quotes a huge rate, and `a` is then correct
    # rather than broken -- measured on `oBTC/sbtcCRV`, where a tiny probe really
    # does return 9.55x and the chain agrees to the wei.  Such an arc is about to
    # be dropped by the floor anyway, but the spread was computed first and the
    # whole quote died on an assertion about a pool no route could have used.
    # Over the arcs whose `G` has to be comparable: a capped arc's does not,
    # since its cap is what limits it.  See `reference_conductance`.
    unbounded = ~np.isfinite(cap)
    comparable = G[np.isfinite(G) & unbounded]
    comparable = comparable[comparable >= base_floor]
    usable_G = comparable if comparable.size else positive_G[positive_G >= base_floor]
    if usable_G.size > 1:
        raw_spread = float(usable_G.max() / usable_G.min())
        if raw_spread > max_spread:
            # No *Curve* universe looks like this: the widest genuine spread
            # measured on Ethereum is ~4e10.  A spread of 1e15+ means B was
            # floored instead of G being ceilinged.  Say so, rather than letting
            # the adaptive dust floor "fix" it by dropping every other arc.
            #
            # A venue of narrow arcs breaks the inference rather than the rule:
            # a Uniswap v3 tick is nearly linear over its own range, so its `B`
            # is genuinely tiny and 144 mainnet pools reach 1.9e15 between them
            # with nothing floored anywhere.  Hence the parameter -- the bound
            # is a property of the universe, not of the arithmetic.
            raise ValueError(
                f"max(G)/min(G) = {raw_spread:.3e} before flooring; "
                "something is being clamped in the wrong space (§9.7)"
            )
    if finite_G.size:
        # Aim at the spread that will exist *after* the ceiling runs: a clamped
        # arc is lifted to `ceiling_factor * reference`, so budgeting against the
        # pre-ceiling maximum alone leaves the assertion tripping on real data.
        # The reference is the uncapped part of the graph, for the reason
        # `reference_conductance` gives.
        top = reference_conductance(G, flagged, cap)
        if (~np.isfinite(G)).any() or np.isfinite(cap).any():
            top *= ceiling_factor
        floor = max(floor, top / TARGET_CONDITION)

    # Conditioning must never cost connectivity: back the floor off until the
    # nodes we have to route between are still joined.  A badly conditioned
    # solve is recoverable; a graph with no path is not.
    dust = G < floor
    backed_off = False
    if require is not None:
        while floor > base_floor:
            alive = ~dust
            reachable = component_of(require[1], tau[alive], sig[alive], n)
            if reachable[require[0]]:
                break
            floor /= 10.0
            backed_off = True
            dust = G < max(floor, base_floor)
    keep = ~dust
    for k in np.flatnonzero(dust):
        dropped[int(k)] = "DUST"

    # --- an arc that claims absurd free value is not priced ---------------
    #
    # `eps` is allowed to be negative -- that is a dislocated pool, and §2.3
    # says it is how arbitrage enters the problem.  What it is not allowed to
    # be is impossible.  `eps = -1` already says the arc pays twice what it is
    # given; the worst real Curve arc measured is -4.5e-02, and cross-venue
    # dislocations are basis points.  So anything past this floor is a broken
    # calibration or an unpriced endpoint, not an opportunity, and the solver
    # will empty the trade into it: one WETH -> RSR arc at `eps = -5.4e7` took
    # the objective to -1.2e14 and the base solve to PARTIAL.
    #
    # Dropped rather than clamped, for the reason §2.3 clamps `B` instead: a
    # clamped `eps` of -1 is still the most attractive arc in the graph.  An
    # arc nothing can price is not an arc.
    absurd = keep & (eps < EPS_FLOOR)
    if absurd.any():
        keep = keep & ~absurd
        for k in np.flatnonzero(absurd):
            dropped[int(k)] = "UNPRICED"

    # --- §9.5 duplicates, as parallel resistors -------------------------
    if merge_duplicates:
        groups: dict[tuple, int] = {}
        order: list[int] = []
        for k in np.flatnonzero(keep):
            key = (
                int(tau[k]),
                int(sig[k]),
                round(float(a[k]), 12),
                round(float(B[k]), 12),
            )
            if key in groups:
                head = groups[key]
                G[head] += G[k]
                cap[head] = cap[head] + cap[k]
                sources[head].append(int(k))
                keep[k] = False
                dropped[int(k)] = "MERGED"
            else:
                groups[key] = int(k)
                order.append(int(k))

    idx = np.flatnonzero(keep)
    arrays = ArcArrays(
        tau=tau[idx],
        sig=sig[idx],
        a=a[idx],
        B=B[idx],
        G=G[idx],
        eps=eps[idx],
        cap=cap[idx],
        flagged=flagged[idx],
        clamped=clamped[idx],
        n_nodes=n,
        sources=[sources[k] for k in idx],
        dropped=dropped,
    )

    # --- §9.7 ceiling, after the dust floor -----------------------------
    arrays.G = ceiling_conductance(arrays.G, arrays.flagged, ceiling_factor,
                                   cap=arrays.cap)

    # --- §12.4 invariants -----------------------------------------------
    if arrays.m and not np.all(arrays.G > 0):
        bad = int(np.argmin(arrays.G))
        raise ValueError(f"arc {bad} has G={arrays.G[bad]:.3e}; Laplacian would not be PSD")
    condition = arrays.condition()
    if condition >= MAX_CONDITION:
        if not backed_off:
            raise ValueError(
                f"max(G)/min(G) = {condition:.3e} >= {MAX_CONDITION:.0e}; "
                "something is being clamped in the wrong space (§9.7)"
            )
        # The dust floor was lowered above precisely to keep `src` joined to
        # `dst`, and that is the whole reason the spread is this wide.  Failing
        # here would contradict the rule that produced the state: a badly
        # conditioned solve is recoverable and the §12.4 KCL residual check
        # adjudicates it, whereas a graph with no path is simply no route.
        arrays.ill_conditioned = condition
    return arrays


#: The smallest the scaled demand may become.  `solve` snaps any `|psi|` under
#: its `TOL` of 1e-9 to zero, and it does that in **scaled** units -- so a
#: normalisation that divides the demand below that annihilates the whole flow
#: and returns a "feasible" solution carrying nothing.  Measured on gnosis at
#: block 47,871,103: a $0.008 XDAI -> EURe quote against a median `G` of 7.6e7
#: scaled to 9.0e-11, and nineteen sizes between 0.0080 and 0.0120 failed with
#: "the optimal flow is empty" while their neighbours quoted normally.
#:
#: 1e-6 is a thousand times `TOL`, and the smallest scaled demand measured to
#: solve correctly on that pair -- so the floor binds only where the median
#: would otherwise have gone too far, and leaves every quote that works today
#: on exactly the scale it has now.
MIN_SCALED_PSI = 1e-6


def scale(arrays: ArcArrays, Psi: float) -> tuple[ArcArrays, float]:
    """§9.1 -- (P) is homogeneous in (G, Psi), so normalise G by its median.

    `u`, `eps` and `rho` are dimensionless and unchanged; only psi rescales.
    Without this, G spans 10+ orders (dust pool vs 100M pool) and the numbers
    the solver compares against its fixed tolerances sit nowhere near 1.

    Which is the whole point, and the reason the demand gets a say: uniform
    scaling cannot change the Laplacian's condition number -- dividing every
    `G` by one number divides every eigenvalue by it -- so what this buys is
    magnitude, not conditioning, and a normalisation that puts `G` near 1 by
    putting `Psi` under `TOL` has bought nothing and lost the route.  Any
    positive `s` leaves the problem the same problem, so where the two pull
    apart the demand wins.
    """
    positive = arrays.G[arrays.G > 0]
    s = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    if Psi > 0.0 and np.isfinite(Psi):
        s = min(s, Psi / MIN_SCALED_PSI)
    arrays.G = arrays.G / s
    arrays.cap = arrays.cap / s
    arrays.g_scale = s
    return arrays, Psi / s


# ------------------------------------------------------------ topology


def laplacian(
    tau: np.ndarray, sig: np.ndarray, G: np.ndarray, n: int, keep: np.ndarray
) -> np.ndarray:
    """L = B^T diag(G) B restricted to `keep`, assembled in O(nnz).

    Built directly on the *kept* index space rather than as an n x n matrix that
    is then sliced.  The active set is usually a handful of nodes out of ~300, so
    allocating the full matrix every pivot dominated the solve -- measured at ~10x
    the cost of the factorisation itself.

    An arc with exactly one endpoint kept still contributes its diagonal term:
    that is what grounds the system at `dst`.
    """
    size = len(keep)
    position = np.full(n, -1, dtype=np.int64)
    position[keep] = np.arange(size)

    head = position[tau]
    tail = position[sig]
    matrix = np.zeros((size, size))

    live_head = head >= 0
    np.add.at(matrix, (head[live_head], head[live_head]), G[live_head])
    live_tail = tail >= 0
    np.add.at(matrix, (tail[live_tail], tail[live_tail]), G[live_tail])
    both = live_head & live_tail
    np.add.at(matrix, (head[both], tail[both]), -G[both])
    np.add.at(matrix, (tail[both], head[both]), -G[both])
    return matrix


def component_of(root: int, tau: np.ndarray, sig: np.ndarray, n: int) -> np.ndarray:
    """Nodes reachable from `root` over the given (undirected) arcs.

    §9.4: `L_A > 0` iff every free node connects to `dst` through the active
    set.  §14's reference listing deletes only `dst`, which produces a singular
    factorisation the first time a pivot orphans a leaf -- so this is recomputed
    every pivot rather than once.
    """
    seen = np.zeros(n, bool)
    if n == 0:
        return seen
    seen[root] = True
    if len(tau) == 0:
        return seen
    # Iterate to a fixed point; the arc count is small and this is branch-free.
    # Both directions in one sweep, over an arc list doubled once up front, rather
    # than two half-sweeps per iteration: at this size the cost is numpy's
    # per-call dispatch, not the work.  Called 679 times in a route, once per
    # pivot per §9.4.
    src = np.concatenate((tau, sig))
    dst = np.concatenate((sig, tau))
    for _ in range(n):
        reach = seen[src] & ~seen[dst]
        if not reach.any():
            break
        seen[dst[reach]] = True
    return seen
