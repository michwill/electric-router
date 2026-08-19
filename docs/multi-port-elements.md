# Multi-port elements

Status: **design, not implemented.**  Written at the end of the session that
did the reentry work, so the next one starts from a plan rather than a blank
page.  Every file and function named here was read while writing it.

## The problem it solves

A pool with `N > 2` coins can serve several ports in one interaction:

    XDAI -> (3pool) -> 3Crv + USDC.e
    DAI  -> (3pool) -> USDC + USDT

Today that is impossible to represent well.  `build_arcs`
(`core/pipeline.py`) emits one arc per directed `(pool, i, j)` pair, the
solver treats each as an independent resistor, and `check_one_arc_per_pool`
(`core/realize.py`, decision 3) forbids using two of them at once.  The
reentry work lifted that ban for stableswap, and doing so exposed why the
ban existed: two arcs of one pool are two separate `psi^2/2G` terms with no
cross-term, both calibrated at a state neither will see, because the first
leg moves the pool before the second arrives.

`candidates.generate` now sweeps the allocation between co-active arcs of one
pool and lets the walk adjudicate (see the commit "Sweep the allocation when
one pool carries two arcs").  That brackets the error -- worth +1.79 bp on
crvUSD -> sDOLA at 2M -- but it does not remove it.

A multi-port element removes it.  One element, one input, a vector of
outputs, priced by one call into the pool model.  The pool appears **once**
in the circuit, so there is no stale second calibration, no omitted
cross-term, and decision 3 is satisfied by construction rather than enforced.

## What it is, precisely

An element `E = (pool, i, J, w)` where `i` is the coin paid in, `J` a subset
of the other coins, and `w` a split over `J` summing to 1.  Its response is

    dy_j = f_j(dx; w)   for j in J

evaluated on **advancing state**: coin `j1` is priced against the pool as it
stands, `j2` against the pool after `j1`'s leg, and so on.  That machinery
already exists -- `ExactQuoterClient._stateful_leg` and
`StableSwap.exchange` / `StableSwapLP.add_liquidity` in `core/stableswap.py`
do exactly this, verified against forked execution to within 1 bp
(`tests/forked/test_reentry_execution.py`).

So the *pricing* is done.  What is missing is the representation in the
solve.

## Why it is not a small change

`core/graph.py` builds `ArcArrays` with one scalar `G` per arc and assembles
a graph Laplacian whose Hessian is diagonal in arc space.  `core/solve.py`
runs an active-set loop over that, and §5.5's certificate comes from pricing
out **all** `m` arcs against it.  A multi-port element is a `|J| x |J|`
dense block on the diagonal, not a scalar, so:

* **assembly** -- `arc_params` / incidence must accept a column that touches
  `1 + |J|` nodes instead of 2;
* **pricing-out** -- the reduced cost of an element is a vector quantity;
  entering the basis means choosing `J` *and* `w`, which is a small inner
  optimisation rather than a comparison;
* **§9.4 component grounding** is recomputed per pivot and assumes 2-node
  columns; a multi-port column can bridge three components at once;
* **§12.4's KCL gate** and `core/diagnostics.py` check flow conservation
  arc-by-arc and need the element's vector form.

None of these is hard on its own.  Together they are the load-bearing core,
and the failure mode is not a crash -- it is a plausible route with a
certificate that no longer means what it says.

## Suggested order

1. **Represent and price, do not solve.**  A `MultiPort` type in
   `core/types.py` and an evaluator over the existing stateful walk.  Test it
   against `tests/forked/test_reentry_execution.py`'s harness: a two-output
   element must equal the two legs executed in sequence, to the wei.  This
   step is independent of the solver and cannot break a route.
2. **Candidate-only.**  Emit multi-port candidates from
   `candidates.generate` for pools the solver already wants twice, priced by
   the walk, ranked alongside everything else.  The solve stays diagonal; the
   element competes on measured output.  This is where the value shows up
   first, and it is reversible.
3. **Into the solve.**  Dense block, vector reduced cost, the four bullets
   above.  Do this only once (2) shows the elements are winning often enough
   to pay for it -- and only with the boa executor from step (3) of the wider
   plan in place, so "the certificate still means something" is a thing that
   can be *checked* rather than argued.

Step 2 is the one to start with.  It is most of the benefit, none of the
risk, and it measures whether step 3 is worth doing.

## Ports, and how many an element may have

`w` is the split across the element's **ports**, on either side -- an element
may take several coins in as well as pay several out.

    #coin-ports in + #coin-ports out <= N

Each port occupies a distinct coin, which is why this is the right bound and
not an arbitrary cap: one coin cannot be both an input and an output (a
wash), and two input ports on one coin are just one larger port.  So ports
map injectively onto coins.  It also bounds enumeration -- every coin is in,
out, or unused, so `3^N` before pruning: nothing for `N = 3`, and cappable
for the wide stableswap-ngs.

**The LP token does not consume a coin slot.**  It is not one of the `N`, so
it cannot collide with a coin-port.  Counting it would reject
`add_liquidity` of both coins of a 2-coin pool -- 2 in, 1 out, against
`N = 2` -- which is a perfectly good operation.

This settles what looked like an open question.  With several inputs
allowed, the element kinds are one kind:

| operation | ports |
|---|---|
| `exchange` | 1 in, 1 out |
| `add_liquidity` | k in, 1 out (LP) |
| `remove_liquidity_one_coin` | 1 in (LP), 1 out |
| `remove_liquidity` | 1 in (LP), k out |
| the case this exists for | j in, k out |

`w` is free on the coin side, and for a deposit the amounts vector *is* `w`
-- the same variable, not a second element kind.  So pricing-out optimises
over one thing, which is what makes the inner problem well-posed.


## Wiring step 2 -- what it actually takes

Checked rather than assumed, and it is smaller than it looked.

**`realize.py` needs no change at all.**  The reentry work already lays out
two arcs of one pool in an admissible order (`ADVANCEABLE`, the group sort in
`_emit`, `check_one_arc_per_pool(..., reentrant=)`).  An element's legs *are*
those legs; what the element changes is which split they carry.

**`candidates.py` needs one generator, and it reuses `resolve`.**  A tuned
element is expressible as a pin: `resolve` already takes
`pinned -> forced_upper`, a dict of per-arc bounds, so emitting the tuned
allocation is

    resolve(np.zeros(g.m, bool), label, "element",
            pinned={k1: psi1, k2: psi2})

with `psi1, psi2` from `best_split`.  `forced_upper` is an upper bound rather
than an equality, so the solver may take less -- which can only improve the
candidate, never invalidate it.

**The one real obstacle is that `core/candidates.py` cannot see a pool
model.**  It is pure, and holds arcs with `a` and `B`, not `StableSwap`
objects; the models live in `dev/exact_probe.py`.  So `generate` needs an
optional pricer threaded in from the pipeline, the way `reentrant` already
is:

    generate(..., element_split=None)
    element_split(pool, i, (j1, j2), amount) -> (psi1, psi2) | None

supplied by `ExactQuoterClient`, defaulting to `None` so the generator is
inert until it is.  That keeps `test_purity.py` green and makes the change
switch-off-able, which matters because this is the first step here that can
alter a live quote.

Estimated shape: ~40 lines in `candidates.py`, ~20 in `pipeline.py`, ~30 in
`exact_probe.py`, plus a route-level regression that the candidate never
*loses* to the pin ladder it replaces.
