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

## Open question worth settling first

Is `w` a free variable, or is it pinned by the pool?  For two `exchange`
legs it is free -- any split of `dx` between two output coins is executable.
For `add_liquidity` it is not: the amounts vector *is* the split.  These may
want to be two element kinds rather than one, and getting that wrong makes
the inner optimisation in pricing-out either impossible or trivial.
