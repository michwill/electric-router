# What a quote costs

Measured on ethereum at block 25,841,728, USDC -> WETH 100,000, with the Rust
solver on. Reproduce with `scripts/bench_quote.py`.

## How to measure it at all

Three rules, each learned by getting it wrong first. They are in the harness
because none of them is obvious and all three change the answer.

**Min, not median.** Sequential arms on this machine drift 30 ms between them,
and whichever ran second lost -- whichever it was:

```
run 1   AFTER 101.84   BEFORE 102.32
run 2   BEFORE 134.30  AFTER  134.75      arms swapped, result flips
```

The min is the least contaminated sample. The median measures the machine.

**CPU, not wall.** `process_time` drops the scheduler. The ratio is printed so
a run with network still in it is obvious rather than quietly averaged in.

**Ablation, not attribution.** cProfile charges per call, so a function called
many times reads several times its cost. `copy.copy` looked like 8% of a quote
and is 2%: `copy._reconstruct` makes 26,240 tiny `setattr` calls and the
profiler charged each of them. Every number below comes from a span around real
work, or from a counter, or from turning something off.

## Where it goes

A warm quote is **95 ms and `cpu/wall` is 0.999** -- there is no network left in
it. The stages do not overlap; they account for 91 of the 95.

| stage | ms | share |
|---|---|---|
| candidates | 40.66 | 42.6% |
| refine | 18.63 | 19.5% |
| split | 17.29 | 18.1% |
| direct | 3.49 | 3.7% |
| solve | 3.29 | 3.4% |
| verify · legs · realize · seed · graph · impact | ~7.8 | 8.2% |

Cutting across those:

| | ms | note |
|---|---|---|
| Rust solver | 20.4 | 41 solves at 497 us; **already native** |
| pool models | ~15 | 1,641 evaluations of Python integer arithmetic |
| FFI crossing | 0.6 | masks in, buffers out; not the cost |
| Python orchestration | ~59 | everything else |

**37 of the 41 solves come from `candidates.resolve`** -- the drop-arc, pin
sweep and conflict-repair families -- at 484 us each. Four are the route.

## What a port buys, measured rather than estimated

`rust/proto` replays a frozen quote with no Python in it: `scripts/dump_quote.py`
writes the solver's own index space, the base solution, and every candidate
solve exactly as it was asked, warm start and bans and pins.

| | Python | Rust | how |
|---|---|---|---|
| the 37 solves | 18.1 ms | 17.0 ms | same code both sides |
| realising 18 candidates | ~4 ms | 0.05 ms | prototype; a skeleton, so a floor |
| `StableSwap.get_dy` | 9.30 us | 1.13 us | 382 vectors, 60 pools, 0 wrong |

`U256` against `f64` is 1.13 us against 0.07 -- sixteen times -- and it does not
matter: 1,641 evaluations is 1.9 ms exact against 0.1 approximate, on a 95 ms
quote. The exact width costs under two milliseconds once it is out of Python
and buys a bound that agrees with the chain to the wei.

## The models, ported

All three families, against real pool state at the pinned block. Every vector
is a pool a quote actually admitted, asked at three sizes in both directions.

| family | python | rust | vectors | pools |
|---|---|---|---|---|
| stableswap | 9.30 us | 1.13 us | 382 | 60 |
| twocrypto | 22.01 | 2.87 | 504 | 84 |
| tricrypto | 24.38 | 4.01 | 270 | 15 |

0 wrong, 0 refused, on all 1,156 -- plus 804 for `get_y` and 124 for the
primitives underneath. A little over sixfold either way, so ~15 ms of a quote
becomes ~2.

Four things a reading would have passed and a vector did not:

* `cbrt` scales its input up by 1e18 or 1e36 and scales the answer back by 1e6
  or 1e12 *after* the Newton loop. Missing those five lines put the port a
  factor of a million out on exactly the inputs a deep pool reaches.
* tricrypto reaches its Newton fallback on real state -- 4 of 180 -- where
  twocrypto never does, 0 of 324. The cubic alone passes its own reading.
* `a_multiplier` is 100 on the 2021 tricrypto pools and 10,000 on everything
  after. The wrong one quotes about twice wrong.
* twocrypto's three legacy flags each name a place two deployed generations
  differ in the last wei. A pool does not announce which it is; it behaves
  like one.

## Where the boundary is

726 crossings a quote, 25.4 ms inside Rust -- so **74% of a quote is Python**.

| crossing | calls | ms | us each |
|---|---|---|---|
| `solve_arrays` | 41 | 19.75 | 481.6 |
| `calibrate_ladder` | 452 | 2.46 | 5.4 |
| `shortest_path` | 83 | 1.46 | 17.6 |
| `split_ascend` | 13 | 0.86 | 66.3 |
| `cancel_cycles` | 13 | 0.63 | 48.7 |
| `problem_for` | 124 | 0.22 | 1.8 |

`calibrate` crossing 452 times is the shape the stage move exists to fix --
not because a crossing is dear (1 to 2 us) but because of what sits between
them: 452 argument lists built and 452 dataclasses constructed, to wrap
arithmetic that is already native.

## Which arithmetic the quote path runs

Not the exact one. `exact_probe._price` prefers `get_dy_fast`, and the
integer form is what *admits* a pool -- `stable_params` compares it against
the chain wei for wei -- rather than what prices a route. Pricing runs
thousands of times and ranks candidates that differ by basis points; the float
form was measured on 263 mainnet stableswaps to be out by at most 5.4e-4 bp.

Which makes `get_dy` the wrong baseline for a port of the hot path, and the
first numbers here were against it:

| family | python exact | python fast | rust U256 | against fast |
|---|---|---|---|---|
| stableswap | 11.30 us | 4.12 | 1.13 | 3.6x |
| twocrypto | 26.60 | 10.06 | 2.87 | 3.5x |
| tricrypto | 27.47 | 12.14 | 4.01 | 3.0x |

So the U256 port is worth about 3.5x on the quote path, not the sixfold it
looked like. The stableswap `f64` probe ran at 0.07 us -- **59x** against the
4.12 the path really pays -- which is where the hot side belongs.

Two ports, two jobs, and they do not compete:

* **`f64`** prices routes. Hot, 43% of a quote, ranking only, and it has a
  documented error budget to hit: 5.4e-4 bp.
* **`U256`** admits pools. Warm-time, once per pool, and wei-exact against the
  chain is the entire point of it -- a bound that disagrees with the chain by
  a wei is a bound that reverts.

The 1,156 vectors belong to the second and stay there. The first needs its own
acceptance test, against the float path's tolerance rather than equality.

## What the float path's error actually is

The drift is measured on `dy` -- the delta, not the invariant -- and it is
**wei quantisation, not floating-point divergence**. `dy = xp[j] - y - 1`
differences two numbers of the pool's magnitude to get one of the trade's, so
a relative error in `y` reaches `dy` multiplied by `y/dy`, which is the
inverse of the trade's share of the pool. The measured drift follows that
exactly, a decade per decade:

| dx / balance | stableswap median | worst | twocrypto worst | tricrypto worst |
|---|---|---|---|---|
| 1e-9 | 1.99e-3 bp | 5.03e+1 | 8.19e-3 | 3.08e-2 |
| 1e-6 | 1.60e-6 | 4.56e+0 | 1.21e-5 | 1.33e-5 |
| 1e-4 | 1.00e-7 | 5.76e-1 | 1.94e-7 | 2.52e-7 |
| 1e-2 | 6.78e-8 | 5.80e-3 | 9.83e-10 | 1.35e-9 |

The worst column is small outputs, not bad arithmetic: 5.03e+01 bp is one wei
on an output of 199 units, 1.00 bp is two wei on 19,933. And the router
declines to ask there -- `probe.py` floors every probe at
`MIN_OUT_QUANTA = 10,000` output units, because "an answer of `n` units
carries a rounding error of `1/n`".

**But the error is not a wei, and calling it one was wrong.** Over a proper
size range it is a relative error that scales with the *pool's* magnitude:

    absolute error in dy, wei      relative error in y
      p50         11,314,133         p50   2.54e-16     ~1 ulp
      p90     79,299,036,762         p90   9.35e-16
      p100   309,870,984,527         p100  3.50e-09

    dy > 1e18 units: median error 1,244,702,346 wei
    dy <= 1e12 units: median error 0 wei

Two things follow, and the second is the one worth keeping.

`y` converges to about one ulp -- 2.54e-16, well *below* `_FAST_TOL` of 1e-14,
because Newton doubles its digits and overshoots the stopping test. So the
tolerance is not what limits the answer and tightening it would buy nothing.

And `dy = xp[j] - y - 1` inherits `y`'s **absolute** error unreduced. One ulp
of a balance at 1e30 is about 1e14 wei, which is where the p100 comes from.
The transfer is not clean either -- measured against one ulp of `y` it is 4.5
at the median and 7,452 at p90 -- so there is real accumulation through the
subtraction, not just round-off carried over.

The clinching evidence that convergence is not the driver: the pools where `y`
converges *worst* have the *smallest* `dy` errors.

    y relative 3.50e-09  ->  dy out by 2 wei on 24,128,260 units
    y relative 2.21e-09  ->  2 wei
    y relative 4.50e-10  ->  1 wei

Those are the cases where `dy` is a large share of the pool, so `y/dy` is
small and there is nothing to amplify. Convergence and amplification pull in
opposite directions, and amplification wins.

So: the float path is wrong by `y`'s round-off transferred through a
subtraction, which is ~1e-11 of `dy` or better at any size the router probes,
and it is not fixable by iterating harder.

## Where the models are actually called

Counting `probe` alone said the models were ~15 ms of a quote. That was wrong,
and wrong in the direction that matters: two other paths evaluate them and
neither goes through `probe`.

| call | calls | items | ms | share |
|---|---|---|---|---|
| `element_split` | 22 | 924 | 17.89 | 16.8% |
| `quote_routes` | 4 | 39 | 15.21 | 14.3% |
| `probe` | 14 | 1,641 | 13.09 | 12.3% |
| | | | **46.19** | **43.4%** |

**Forty-three per cent of a quote is pool arithmetic**, and at the measured
sixfold that is 46 ms becoming ~7. It is nearly twice what the whole `refine`
and `split` orchestration is worth put together.

It also explains those stages rather than competing with them. `split` spends
2.28 ms searching and about 9.5 quoting the neighbourhood it found; `refine`
spends its probe budget the same way. They are not slow, they are calling
something slow -- so the models are the fix for them too, and porting the
stages first would have moved the cheap half.

## What is wired, and what it is worth

Interleaved in one process, toggling the Rust models at runtime, because
sequential whole-quote totals on this machine swing two to one within a
session:

    verified_out   rust 40237910250133590325   python 40237910250133590325

    rust    min  76.71   median  78.90 ms cpu
    python  min 101.29   median 103.82
    1.32x  ·  24.6 ms saved

The identical `verified_out` is the correctness statement. The route is chosen
and priced through different arithmetic and comes out the same to the wei.

| path | ms | state |
|---|---|---|
| `element_split` | 17.89 | wired -- 1,348 to 84 us a split, 16x |
| `quote_routes` | 15.21 | still Python; needs `walk_route` |
| `probe` | 13.09 | wired -- 8.85 to 3.75 us a probe, 2.36x |

`element_split` paid where `probe` did not, and the reason generalises. A probe
is under a microsecond of arithmetic behind a 1-2 us crossing, so batching
helps and cannot do more than that. An element split is ~100 stateful
exchanges behind one call, so moving the *loop* carries everything with it.
Batch where the work is small; move the loop where it is not.

### Installing it

Neither native extension is in `uv.lock` or `pyproject.toml`. `uv sync` is
therefore destructive and always has been -- it uninstalls `erouter-solve`,
`erouter-evm` and `maturin`, which is `uv sync --dry-run` on a clean tree, not
a consequence of anything here. The Python path still answers without them, so
nothing breaks; it quietly gets slower, which is worse.

Rebuild after touching `rust/`:

    uv pip install --reinstall --no-deps ./rust
    uv pip install --reinstall --no-deps ./rust/evm

## Why `quote_routes` was not ported

It looked like the last model path at 15.21 ms. It is not a model path.

    quote_routes      14.50 ms a quote · 39 routes
      of which revm   11.55 ms · 24 routes (80%)  at 481 us each
      walked here      2.95 ms · 15 routes        at 197 us each

Counting first, as `walk_route` would have needed: of 156 routes over four
quotes, only **56 are walkable in Rust at all**, and they hold 188 of 1,252
legs -- 15% of the work. The 92 that are not are blocked by a single leg kind,
and every one of them by the same one:

    first blocking kind, per route      all legs by kind
      WSTETH_UNWRAP     92                SWAP_CRYPTO    672
                                          SWAP_STABLE    348
                                          WSTETH_UNWRAP   92
                                          WRAP_NATIVE     92

`WSTETH_UNWRAP` resolves to **no model at all** -- not an unported one, none.
So those routes are not slow because Python prices them slowly; they are sent
whole to revm, at 481 us against 197 for a walked one. Porting `walk_route`
would chase the 2.95 ms and leave the 11.55 exactly where it is.

The gap is a missing model, not a missing port. `wrappers.py` already reads
the rate at the warm -- it merges wstETH and stETH into one node and records
that `getStETHByWstETH` is linear to 1.3e-19 across eight decades -- so the
number is in hand and simply never reaches `_resolve_model`. A linear model
for it would take 24 routes off revm, which is worth about 7 ms, against ~2.5
for the walk.

It is a decision rather than a transcription, which is why it is written down
here rather than done. `1.3e-19` is a measurement of linearity, not a proof of
exactness, and the exact models are admitted by reproducing the chain **wei for
wei**. The ERC4626 vaults are the precedent -- they are ratios admitted the
same way -- so the shape exists; whether this one clears that bar is a
question for whoever adds it.

### The models do not change the answer

`verified - modelled` is expected to be **positive**: the quadratic overstates
loss by construction, so the chain beating it is the right sign and the
reverse means a calibration bug. A CLI route once printed it negative, which
is worth recording as checked rather than assumed. The same route, priced four
ways at a pinned block:

| rust models | wstETH | ver - mod |
|---|---|---|
| on | on | +2.27 bp |
| on | off | +2.27 |
| off | on | +2.27 |
| off | off | +2.27 |

Identical in all four, so neither change touches it -- the CLI reading was a
different block, live rather than pinned. What the arms *do* differ by is
small and accounted for: `modelled` by 620,670 wei (1.5e-11, the float drift
feeding back through calibration) and `verified` by 7.76e-10 bp, which is the
wstETH leg no longer going through the EVM.

## Where it ended up

A warm quote, same block, same pair, after the models and the wiring:

| stage | before | after |
|---|---|---|
| candidates | 40.66 | 26.58 |
| refine | 18.63 | 13.81 |
| split | 17.29 | **6.20** |
| **wall total** | **95.37** | **~60-68** |

`split` collapsed because its cost was never the search. It spends 2.28 ms
searching and the rest quoting, and the quoting is models.

What each change was worth, measured interleaved rather than in sequence:

| change | measured |
|---|---|
| Rust pool models (`probe`, `element_split`) | 24.6 ms |
| wstETH modelled instead of executed | 11.5 ms |
| `local_evm` on `erouter_evm` rather than pyrevm | 1.77x on the binding |
| batched calibration | 1.71 ms |

## Moving `refine`, if it is moved

`refine` is the stage worth moving: 13.8 ms, and its named functions account
for 14.6 of its 15.4, so unlike `direct` there is almost no inline glue and
porting the functions *is* moving the stage.

`scripts/dump_refine.py` freezes it -- 840 ladders, 1,374 planned probes, 478
arcs, 3 seeds, every float as an exact bit pattern -- so a port has something
to be held to. A ported stage has no natural acceptance test the way a ported
model does; this is the one it gets.

**It is all or nothing, and that is the finding.** The stage is
`plan_sized -> probe -> collect -> merge -> _recalibrate -> _assemble -> scale
-> seed_subgraph`, and no single piece is worth porting alone:

* `plan_sized` is 3.24 ms over 840 ladders. Porting it alone means marshalling
  those ladders across -- ~10,000 integers each way, 1 to 2 ms -- so it nets
  about a millisecond and adds a module.
* `_recalibrate` is the same shape from the other side. Its remaining 2.80 ms
  is mostly `Ladder.as_float`, marshalling 450 float lists on every quote.

The win is *residency*. If the ladders live on the Rust side for the whole
stage, `as_float` does not get faster, it stops existing, and `plan_sized`,
`collect` and `merge` all read them in place. That is worth on the order of
10 ms and it cannot be had in pieces.

One wrinkle the design has to answer: `probe` in the middle needs the models,
which are resident already, but it also serves vaults, LP tokens and the 1:1
wrappers from Python. So either those port too, or the stage returns its
unservable probes as holes for Python to fill and takes the answers back --
two crossings rather than one, which is still 450 fewer than today.

## The shape of what is left

**21 ms of a ~67 ms quote is the Rust solver** -- 37 `active_set_solve` calls
-- and it does not move without changing the algorithm, which is out of scope.
The rest is Python spread thin:

    optimise_splits    6.70    (of which _ascend 1.03 and probes 0.87 are Rust)
    realize_candidates 4.21
    plan_sized         2.43     _assemble       2.33
    _recalibrate       2.13     seed_subgraph   2.11
    price_legs         1.41     element_of_arcs 1.40
    merge              1.29     conflicting_pools 1.25
    collect            1.02     k_shortest_paths 0.75

Fifteen functions, the largest 6.70 ms and the median about 1.4. Nothing here
is a bottleneck; it is a tail.

**And that list is a third short of the truth.** Accounting for a whole quote
in one run rather than assembling it from several -- timers on, which inflates
the total, so read the shares:

| | ms | share |
|---|---|---|
| Rust solver (`active_set_solve`) | 19.91 | 26% |
| named Python functions | 33.69 | 43% |
| inside a stage, in no named function | 17.06 | 22% |
| outside every stage span | 6.93 | 9% |

Nearly a third is in no function at all. It is the inline bodies of `route()`
and the stage blocks -- loops, comprehensions, dicts and lists built between
the calls, dataclasses constructed to carry a result four lines. Porting
helpers one at a time cannot reach it, however many of them are ported.

Which is `element_split`'s lesson in another form. Moving a *loop* works;
moving a *function* mostly does not. So the options are narrower than "keep
porting": move whole stages with their glue -- which is what the candidate
prototype did, 40.7 to 17.05 ms by relocating the loop rather than a function
inside it -- or accept the ~60-67 ms, of which ~20 is already native.

The rest is where the curve flattens. Some of it is not worth taking at any
price: caching `Ladder.as_float` would save about a millisecond and needs
invalidating at two in-place mutation sites, and a stale ladder calibrates on
old probe data and returns a wrong fit silently. That is the wrong trade, and
noting it here is cheaper than rediscovering it.



The remaining orchestration is **diffuse**, which is the finding that decides
the strategy. Inside `refine`:

```
_recalibrate 3.62   plan_sized 2.52   seed_subgraph 2.06   _assemble 2.03
merge 1.28   collect 1.04   price_impact 0.81   direct_candidates 0.44
```

Nine functions, none dominant, and `split` looks the same -- 18.6 ms of which
the ascent, the one part already in Rust, is 0.95. There is no single function
worth porting. **Port whole stages or nothing**: the candidate prototype took
40.7 ms to 17.05 by moving the loop, not by moving a function out of it.

A whole-pipeline port lands a quote near 25 ms, of which 17 is the solver.

## What 9 ms would take

An arbitrage desk reports 9 ms a quote. Our solver alone, already native, is
20.4 ms -- 2.2x their whole budget -- so it is not this problem solved faster.
`scripts/candidate_sweep.py` measures what the difference buys:

| pair | budget 8 | budget 16 | ms saved by stopping at 8 |
|---|---|---|---|
| USDT>tBTC | 0.00 bp | 0.00 bp | ~0 |
| WETH>USDC | 0.00 | 0.00 | 24 |
| USDC>WETH | 3.45 | 0.34 | 26 |
| crvUSD>WETH | 2.19 | 0.87 | 27 |

On half the pairs the tail of the ballot buys nothing; on the other half it
buys 2.19 to 3.45 bp, which is 22 to 35 dollars on a 100,000 trade for 26 ms
that costs nothing in gas. 9 ms is that trade declined -- the right call for a
bot, the wrong one for a router quoting a user's swap.

## The models, in the browser

The port was built for the PyO3 boundary and reached only that one. The maths
was never the obstacle -- `pools/` has no I/O, no clock and no threading, and
compiles to wasm32 with the solver -- but nothing under `rust/wasm/src/`
referenced it, so the linker stripped every byte of it from the module. The
bundle exported `calibrate`, `cancel`, `find`, `split` and the two classes, and
a browser had no way to price a pool at all.

Wiring it in cost 168 KB of wasm, which is the measure of how much was being
dropped: 1,432,448 bytes to 1,601,003.

**Both bindings now wrap one registry.** `pools::registry::Registry` holds the
models and the arithmetic; `pools/py.rs` and `wasm/src/pools.rs` only marshal.
That is not tidiness. The two boundaries have to price a pool identically --
the browser is meant to answer what the extension answers -- and two
hand-written copies of the construction would have drifted at the first legacy
flag, which is exactly where two deployed generations differ in the last wei.

### What the boundary costs, and one thing it nearly cost

`u128` has no typed array. The first version narrowed `dx` to `u64` to get one,
which caps a probe at 1.8e19 wei -- eighteen tokens at eighteen decimals, a
rounding error rather than a trade -- and the differential test caught it
immediately, quoting 1,426,559,818 against 5,981,843. So amounts cross as the
low and high halves of a `u128` interleaved in a `BigUint64Array`, which is the
same shape the answers come back in, and the alternative of an array of `BigInt`
allocates one object per probe on the path the batch exists to keep cheap.

### Held to the reference, standing rather than once

The 1,156 vectors above were run by hand, once. They are now a test, and the
three arithmetics are held to different contracts because they answer different
questions:

| pair | contract | measured |
|---|---|---|
| Rust exact vs Python exact | equality, wei for wei | 240 of 240 |
| wasm exact vs Python exact | equality, wei for wei | 240 of 240 |
| wasm float vs extension float | equality | 240 of 240 |
| Rust float vs Python float | the quote path's budget | 5.5e-11 bp worst |

The last row is the only one that is not equality, and deliberately: `dy =
xp[j] - y - 1` inherits `y`'s absolute round-off, and the two sides do not have
to reach `y` by the same sequence. 12 of 240 differ, all cryptoswap at a tenth
of the pool, all under 5.5e-11 bp against a budget of 5.4e-4.

That budget is seven orders of magnitude above what the port actually costs, so
on its own it would not notice the port going gradually wrong. The median
carries that: it is exactly zero -- most vectors do agree bit for bit -- so a
median that moves at all means the two sides stopped running the same sequence,
whatever the worst case still says.

The two wasm rows are equality because both targets are the same source through
the same compiler. Python is allowed to differ from both by the budget; wasm
drifting from the extension would mean the browser ranks routes differently
from the CLI, and nothing budgets for that.

### Measuring it, repeatably

`scripts/bench_quote.py --arms` runs this branch's quote path against master's,
interleaved rep by rep, because sequential totals on this machine swing two to
one within a session. Both arms are in one process -- the Rust models toggle on
the client, the batched fit on `pipeline._ACCEL_ON` -- so it is the same
universe, the same block and the same warm, which a second checkout would not
be. The arms are required to agree to the wei; a speed-up that changes the
answer is a bug report.

    arm                        min ms   median
    rust (this branch)          48.76    49.91
    python (master)             60.04    62.28
    speed-up                     1.23x
    saved                       11.28 ms
