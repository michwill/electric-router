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

### The ratios, and what porting them was worth

Vaults, lending wrappers, wstETH and the 1:1 wrappers are now in the registry
too. They are one shape between them -- `dx * num / den`, with a cap -- because
the rounding convention is already resolved into the ratio by whoever admitted
it, so `Vault` and `_OneToOne` are two arms of the same three lines.

**On the quote path this bought nothing measurable, and that was the
expectation.** Counted on a warm quote:

| pair | vaults | 1:1 | probes computed | delegated |
|---|---|---|---|---|
| USDC>WETH | 1 | 1 | 1,741 of 1,743 | 2 |
| crvUSD>sDOLA | 4 | 0 | 1,793 of 1,805 | 6 |

Identical with the ratios on the Rust side and with them left in Python. They
were never holes -- Python priced them without a request -- so this moves a
handful of multiplies across a boundary, not a route off the chain. The arms
agree: 1.22x, against 1.25x and 1.23x measured before it, which is the same
number three times at three blocks.

It is worth doing anyway, for the reason the port now exists: a leg kind left
in Python is a leg kind the browser cannot price at all, and correctness at the
boundary is not a performance question. What it removes is a dependency, and
the measurement above is here so nobody later reads the change as a speed-up
that failed to show.

**What is left in Python on the probe path** is the LP tokens -- `_Withdraw`
over `StableSwapLP` and the crypto LP, and `_Deposit` over the deposit model.
Those are a genuinely different shape rather than a missing transcription: they
move the invariant and the token supply together, and the deposit direction is
admitted on its own evidence because a pool whose withdrawal does not reproduce
may still deposit exactly.

### The LP arcs, and a bug found by porting them

These are the one family that is a different shape rather than a
transcription. A swap holds `D` and asks what one balance becomes; a deposit
and a withdrawal *move* `D`, so they need `get_y_D` -- balance `i` when `D` is
reduced to a target -- which the Rust side did not have at all. That, plus `D`
over balances that are not the pool's own, is most of what the port is.

Both directions are separate registry entries rather than one model with a
flag, because a pool may reproduce one and not the other: a withdrawal that
does not match the chain does not condemn the deposit beside it.

**Porting them turned up a live bug in the reference.** `_Withdraw.get_dy_fast`
called `calc_withdraw_one_coin_fast`, which `StableSwapLP` has and
`TricryptoLP` does not -- so a `WITHDRAW_CRYPTO` probe raised `AttributeError`,
which `probe` does not catch and a quote does not survive. It is on master too,
not something the port introduced; the port is simply what made someone run
that path. Fixed by answering in integers where an LP has no float form, which
is the rule `_price` already applies to a model with no fast path at all.

**And again it bought nothing measurable, for a reason worth writing down.**
LP arcs are 1.7% of a quote's probes:

| kind | probes | share |
|---|---|---|
| SWAP_STABLE | 1,146 | 65.2% |
| SWAP_CRYPTO | 531 | 30.2% |
| WSTETH_UNWRAP | 50 | 2.8% |
| DEPOSIT_FIXED | 15 | 0.9% |
| WITHDRAW_STABLE | 15 | 0.9% |
| WRAP_NATIVE | 2 | 0.1% |

Pool arithmetic is ~43% of a quote, so 1.7% of it is under a percent of the
whole -- below what the interleaved arms can resolve, and the arms agree: 1.22x,
the same figure measured before the ratios and before these.

**What it does finish is the probe path.** Every kind in that table now has a
Rust model, so a probe no longer needs Python for any leg the router generates.
The two probes still delegated are pools whose *swap* model was never admitted
-- no port would help them, because there is nothing to port.

The vector count is the other half of that claim: 411, from 6 stableswaps, 6
twocryptos, 2 tricryptos, 7 ratios, 12 stableswap LP directions and 2 tricrypto
LPs, each at several sizes in every direction. Exact against Python and exact
against the extension, in both arithmetics.

## Is the boundary the thing to remove?

It is worth asking, because the obvious next move after porting every model is
to stop crossing at all and let a quote live on the Rust side. So: what does
the crossing actually cost?

**Measured, with nothing behind it.** A 1:1 wrapper does no arithmetic, so a
batch of them prices the boundary and nothing else:

| batch | us a call | us an item |
|---|---|---|
| 1 | 1.076 | 1.0757 |
| 10 | 2.175 | 0.2175 |
| 100 | 14.713 | 0.1471 |
| 1000 | 138.521 | 0.1385 |

Which separates into a fixed **0.94 us a crossing** and **0.138 us an item**
of marshalling. A quote makes 287 crossings and batches 1,637 items, so:

    fixed        0.269 ms
    marshalling  0.225 ms
    border       0.494 ms   of a ~78 ms quote -- 0.6%

**So the border is not the cost, and removing it buys half a millisecond.**
That is a change from when this document first counted: it was 726 crossings,
452 of them `calibrate`, each building an argument list and a dataclass around
arithmetic that was already native. Batching the models and the fit is what
fixed it, and it is fixed -- `calibrate_many` is one crossing now, and the
1,637 model evaluations are 14.

What is left is not boundary, it is Python doing work:

| | ms | share |
|---|---|---|
| `candidates` | 39.45 | 50.5% |
| `refine` | 12.16 | 15.6% |
| `direct` | 4.21 | 5.4% |
| `split` | 3.75 | 4.8% |
| `solve` | 3.70 | 4.7% |
| `scout` | 3.21 | 4.1% |
| everything else | ~6 | 7.7% |

And `candidates` is mostly not Python either. Of its 39.45 ms, **30.19 is 36
`active_set_solve` calls from `candidates.resolve`** at 839 us each -- already
native, and the document's standing position is that they do not move without
changing the algorithm. Four more come from `_quote` itself.

    solver (already Rust)   ~34 ms   42%
    Python orchestration    ~40 ms   50%
    the boundary             0.5 ms   0.6%

**So "let it all live in Rust" is the right instinct for the right reason and
the wrong one for the obvious reason.** It is not worth doing to stop crossing;
it is worth doing because 40 ms of a quote is Python arranging work rather than
doing it, and because a stage left in Python is a stage the browser cannot run
at all. The ceiling is roughly 78 ms to somewhere near 45 -- the solver does
not move, so it sets the floor.

That also says where to start, and it is not the biggest number. `refine` is
12 ms, its named functions account for nearly all of it, and
`scripts/dump_refine.py` already freezes it -- 840 ladders, 1,374 planned
probes, 478 arcs, every float as an exact bit pattern -- so a port has an
acceptance test waiting. `candidates` is half the quote but three quarters of
it is the solver, so moving the stage buys the 9 ms around it, not the 39.

## Moving `refine`, moved

The ladders now live on the Rust side for the whole stage. They are handed
over once at the warm and forked per quote; `plan_sized`, `collect`, `merge`
and the fit all read them where they are. `Ladder.as_float` does not get
faster, it stops being called -- and so does the conversion behind it, because
`calibrate_many` was rebuilding every list as `float` after `as_float` had
already divided it.

Identity stayed in Python. A `Probe` is addressed by a pool, a kind and two
coin indices, none of which is arithmetic, so the Rust side holds slots and
the caller maps them back to arcs. Three crossings a quote: what is missing,
what the chain answered, the fits.

**It is worth 3.15 ms, not the ten this document estimated.** Interleaved,
resident against Python, same session and same block:

    arm                         refine ms   quote ms
    resident (rust)                  9.30      76.15
    python                          12.45      79.13
    refine saved                     3.15
    quote saved                                 2.99

Both arms return the same `verified_out` to the wei, and all 478 arc fits come
back identical -- `a`, `B` and `cap` alike.

The estimate predates `calibrate_many`, which had already taken the largest
piece: 452 crossings became one, so what was left to win was the marshalling
around the fit rather than the marshalling *and* the fit.

But the honest reason is narrower, and it is visible in the parts. **The
ladder bookkeeping moved; the per-probe Python loops did not.**

| before | ms | after | ms | saved |
|---|---|---|---|---|
| `plan_sized` | 2.41 | `resident.plan` | 2.12 | 0.29 |
| `collect` + `merge` | 2.20 | `resident.absorb` | 0.81 | 1.39 |
| `_recalibrate` | 2.59 | `_recalibrate_resident` | 1.11 | 1.48 |
| | | | | **3.16** |

Which is the 3.15 the arms measured, from the other side. `plan` barely moved
because what it costs is not the planning -- Rust does that now -- but building
1,380 `Probe` dataclasses to describe the answer. `absorb` still walks 1,380
result objects to read a status and a value off each, and
`_recalibrate_resident` still walks 478 arcs to write nine fields onto each.

*(An earlier draft of this section blamed the gap on measurement overhead from
wrapping the parts. That was checked and is wrong: 942 wrapped calls a quote
cost nothing measurable -- interleaved, wrapped against not, the difference was
-2.5 ms, which is noise. The parts were telling the truth; they were simply not
all that was there.)*

So refine's remaining 9.3 ms is: `seed_subgraph` 3.92 and `_assemble` 2.68,
which are graph operations shared with the coarse pass and not refine's to
move, and 4.04 of Python loops over probes and arcs. The probes are the larger
half and the more portable: 1,380 objects built to carry six fields each,
crossing into a client whose arithmetic is already native.

### The probes, without the objects

The half of refine's remaining Python that was refine's to move was the loop
building `Probe` objects. Measured on its own: **885 ns to build one, against
42 for the plain tuple underneath it** -- a dataclass with `slots`, `frozen`
and a validating `__post_init__` -- and refine builds 1,380 a quote, then takes
a `Quote` object back for each.

So `ExactQuoterClient` grew a columnar twin. `probe_columns` takes six lists
and answers three: values, a status code per probe, and the names those codes
index. Zero means a value, so the common case carries no name at all. Objects
are still built for the holes, because the inner client speaks `Probe` and a
hole is two of 1,380 on a warm mainnet quote.

Interleaved, columns against objects, both on the resident path:

    arm                         refine ms   quote ms
    columns                          4.47      58.67
    Probe objects                    5.97      60.60
    refine saved                     1.50
    quote saved                                 1.94

Same answer to the wei. The two spellings have to agree on all three paths at
once -- the batch Rust serves, the Python fallback for a model the batch
declines, and the delegation for a pool with no model -- and
`test_probe_columns.py` reaches all three, with a test that asserts the vectors
reach them rather than trusting that they do.

**Refine, end to end: 12.45 ms to 4.47.** The ladders stopped moving, and then
the probes stopped being objects. What is left in it is `seed_subgraph` and
`_assemble`, which belong to the graph rather than to refine, and the fit's
walk over 478 arcs to write nine fields onto each -- which cannot go while the
arcs are Python objects.

The whole-quote arms, for the record: **1.25x**, from 1.22 before this stage
moved.

## `candidates` cannot be ported, and the measurement says why

It is half the quote and the obvious next stage. It is also **92% a solver
that is already native**, which is the whole finding:

| piece | calls | ms |
|---|---|---|
| `solve_arrays` | 41 | 22.86 |
| `cancel_cycles` | 25 | 0.91 |
| `conflicting_pools` | 36 | 0.83 |
| `prune_dust` | 11 | 0.55 |
| `problem_for` | 164 | 0.21 |
| `_signature`, `repair_order`, `keep_only`, `carries`, `_pool_of` | | <0.35 |
| **`generate`** | 1 | **24.84** |

`solve_arrays` and `cancel_cycles` are Rust already. What is left to move is
about 1.4 ms, and no arrangement of it changes a stage whose cost is 41 solves
at 557 us. Porting `generate` would be a portability job -- real, for the
browser -- but it is not a performance one, and this document should not let
anyone believe otherwise by pointing at 50%.

### What was there was repeated work, not Python

`conflicting_pools` is asked thirty-six times a quote about arcs that do not
change between calls. It was lowering every active arc's address each time,
and re-deriving the element from the same indices each time -- 98 calls to
`element_of_arcs` where there are 19 distinct sets.

Lowering the addresses once and memoising the element check on the index tuple
is worth, interleaved and with the same answer to the wei:

    arm                           min ms    median
    cached                         56.42     57.37
    recomputed                     58.18     59.53
    saved                           1.76 ms

`conflicting_pools` itself halves, 1.61 ms to 0.83, and `element_of_arcs` drops
from 98 calls to 19. Both caches are optional arguments: without them the
function is its own reference, which is what the tests compare against.

**Three findings in a row now have this shape** -- the ladders that were
rebuilt every quote, the 1,380 `Probe` objects, and these thirty-six repeated
groupings. None of them was Python being slow at arithmetic. All of them were
the same answer computed again, and a port would have carried the repetition
across the boundary with it.

The whole-quote arms: **1.27x**, from 1.22 before refine moved.

## `realize_candidates` and `k_shortest_paths`, and the thing behind them

Neither was what it looked like, and looking is what found the real one.

**`k_shortest_paths` is not worth porting.** Its 1.35 ms is 124 `spfa` calls,
and the crossing to reach them is 0.12 ms of that -- the other 1.02 is the
search itself, at 8.2 us over 477 arcs, already native. `weights` is never
passed on this pair, so the array copy the wrapper would make never happens.
Moving Yen's loop into Rust would save the crossings and leave the searches,
which is the same shape of mistake as porting a stage to chase its cheap half.

**`realize_candidates` is 2.66 ms and 2.39 of it is `realize`** -- but ranking
what runs inside it turned up something that is not in `realize` at all:

    tricrypto.newton_y_fast   2,688 calls
    nodes.rate               16,266 calls

`newton_y_fast` is the Python tricrypto invariant. It should not be running: the
models were ported. It runs because `walk_route` prices legs one at a time
through `_price`, and `_price` goes to the Python model -- the batch never sees
them, because a walk is sequential and each leg's size is the last leg's answer.

    model                     calls       ms   us each
    Tricrypto                   450     4.75     10.55
    StableSwap                  345     1.52      4.40
    Vault                       109     0.09      0.83
    _OneToOne                   109     0.04      0.41
    Twocrypto                     3     0.04     11.74
    _price total               1016     6.43     10.7%

**Ten and a half per cent of a quote, in Python arithmetic that had a native
form all along.** A batch of one still pays the whole 1.07 us crossing, and it
is still the cheaper side wherever the arithmetic is dearer than that -- which
is every family that iterates. Interleaved:

    arm                           min ms    median
    resident models                49.93     51.18
    python models                  54.99     55.63
    saved                           5.06 ms

`_price` now runs zero times on a warm mainnet quote. It stays as the fallback
for a model the batch declines, which is what keeps it the reference.

### The arms had to give up exact equality, and why that is not a retreat

The two arms now run *different float arithmetic* on the leg-pricing path, so
they are held to the float path's budget -- 5.4e-4 bp -- instead of to the wei.
Measured where the drift appears at all it is **1.2e-11 bp**, and the route,
the legs and the bps are identical; at most blocks the arms still agree
exactly. The budget is seven orders above the drift, so what it catches is a
regression and not round-off.

That is the same contract the models have always had, arriving one path later:
integers admit a pool, floats rank a route.

**The quote: 1.41x, from 1.27x.**

## Where the Rust/Python drift actually came from

The float arms were held to a 5.4e-4 bp budget on the assumption that the two
languages simply compute floats differently. **They do not**, and conflating
three different drifts is what made it look like they did. Measured on the same
411 vectors:

| comparison | worst bp | vectors differing |
|---|---|---|
| python float vs python **exact** | 8.012e-02 | 163 / 411 |
| rust float vs rust **exact** | 8.012e-02 | 163 / 411 |
| rust float vs python float | 5.515e-11 | 12 / 411 |
| rust exact vs python exact | 0 | 0 / 411 |

The first two are **the same number**, because the float form is the same
approximation in both languages -- that is the error the quote path really
carries, and it belongs to the algorithm rather than to the port. The port's
own contribution was nine orders smaller, and it was not arithmetic either.

All twelve were cryptoswap; no stableswap, and not even the twocrypto variant
that runs the stableswap invariant. The Newton loops are line for line
identical -- same expressions, same order, same `1e-14` tolerance. **The
difference was in the inputs.**

    python   v / 10**18        one rounding, of the exact quotient
    rust     f64::from(v) / 1e18   rounds v to f64, then divides -- two

A pool balance is around 1e24, which passes `2^53` by about 77,000x, so the
conversion has already lost bits before the division runs. On real twocrypto
state the two spellings land 1 ULP apart:

    v = 1195163862946386689613
      python 1195.1638629463866
      rust   1195.1638629463869

One ULP in, and `dy = xp[j] - y - 1` carries a relative error in `y` out
multiplied by `y/dy`. That is the same amplification this document already
describes for the float path itself -- it just had a second, avoidable source
feeding it.

`pools::scaled` divides the way Python does, and **the two float paths now
agree on all 411 vectors**. `test_the_float_port_is_exact_too` asserts equality
rather than a budget, and the arms are back to requiring the wei.

Which leaves the error budget where it belongs. The quote path carries 8e-2 bp
of float-versus-exact approximation, on both sides equally, by choice; the port
carries none.

## What the browser can do, and what is left

The ladders were added with a PyO3 binding and no wasm one -- the same hole
`pools` had, repeated inside one session, and found by reading again. Both are
closed now, and `test_bindings_match.py` reads the two binding files and
compares what they name, so the third time it will fail a build instead. It is
a structural test on purpose: the failure mode is silent, nothing errors, the
browser simply cannot do something the extension can.

**Reachable from JavaScript today**, each with a differential test that holds
it to the extension: the solver (`Problem`, `SolveResult`), `calibrate`,
`cancelCycles`, `findCycle`, `splitAscend`, the EVM, every pool model through
`Pools` -- the three swap families, vaults, lending wrappers, wstETH, the 1:1
wraps and both LP directions -- the refine stage's `Ladders`, which plans,
merges and fits without the ladders crossing, and now `Graph` and `NodeMap`.

### What is ported, and what it cost to get right

`types`, `graph`, `nodes`, `multiport`, `realize`, `gas`, `risk`, `candidates`,
`verify`, `pipeline`'s stage logic, `curves`, `prices`, `slippage`, `refit`,
`keccak`, `codec` and `routecall` are done; `seed` gained Yen's
`k_shortest_paths` and `split` its `split_groups`. They went in
that order because it is the dependency order: everything above takes an
`ArcArrays`, a rate, or a `PoolArc`.

The differential tests are bit-exact rather than tolerant, and that is what
found every one of the following. None of them would have failed a test with a
tolerance in it, and none would have raised in production.

**`laplacian` assembles in four passes, not one.** The reference makes four
`np.add.at` calls -- every head diagonal, then every tail diagonal, then every
`(head, tail)`, then every `(tail, head)`. Folding them per-arc is the obvious
transcription and sums the off-diagonal in a different order where a node pair
carries arcs both ways: `p1, p2, p3` instead of `p1, p3, p2`. Float addition is
not associative. Two of forty seeds found it.

**`ceiling_conductance` maps every non-finite `G` to `inf` before its
minimum**, so `-inf` and `NaN` both take the ceiling. A bare `min` leaves them
alone.

**`int / int` is correctly rounded in CPython, and three roundings are not.**
`f64::from(a) / f64::from(b)` is two conversions and a division. The quotient
sets `theta`, `share_of_node`, a conversion `rate`, and -- through
`round(BPS * share / total)` -- the `bps` a leg is emitted with. A last-bit
difference in the last of those is different calldata. `pools::divided` takes
the quotient with 55 bits of headroom and folds the remainder back as a sticky
low bit, so one rounding decides it; 3,200 random 256-bit pairs agree exactly.

**`round()` is ties-to-even and `f64::round` is not.** Two equal branches out
of one node put `BPS * share / total` exactly on a half every time.

**`to_canonical` is unbounded Python `int`.** A `U256` multiply is not, and at
`2**200` it was quietly wrapping. The product now widens to 512 bits and a
quotient past `2**256` is refused rather than wrapped -- the mirror's one
deliberate divergence, and it is loud.

Three smaller mirrorings: duplicate groups key on `round(a, 12)`, which is
decimal ties-to-even and not an `f64` operation; `np.median` averages the
middle pair; and the reference's error strings are contract, so `pyfmt` spells
a float the way CPython does. One of those strings was leaking a numpy repr --
`[np.int64(0)]` -- fixed on the Python side rather than reproduced here.

`realize` is the first ported stage whose output is an *artefact* rather than a
number, and its three ordering rules each exist because of a measured failure:
a capped arc must never sweep (9,960 USDC into a vault whose `maxDeposit` is
1,142), one fill per spoke rather than one per arc drawing on it, and two arcs
on one spoke are one `bps` group. Each has a case in
`test_realize_differential.py`. Every ordered map on the Rust side is a `Vec`
of pairs rather than a `HashMap`, because the reference iterates its dicts and
the order decides which leg sweeps.

**One documented approximation.** `route_conductance` solves a small dense
system; the reference uses numpy's LAPACK and the port uses `lu.rs`. Same
algorithm, different implementation, so that one comparison carries a relative
tolerance. Everything else in these suites is exact.

### The solver drifted from its reference, and the ballot found it

Differing `candidates` surfaced a bug in the *already-ported* solver, not in
the new code: `core/solve.py` perturbs a cycling basis (`PERTURB_ROUNDS`,
`PERTURB_SCALE`) and `rust/src/solve.rs` had no such step. Its own comment says
why the gap went unnoticed -- the port got the same effect *by accident*, from
the rounding its rank-1 Cholesky update leaves in `u`. Luck is not a mechanism.
The perturbation is now ported and deliberate.

That closed half of it, and the other half turned out not to be a porting
error at all.

### Backward stability is not enough when a threshold reads the answer

The remaining divergence was on a restricted re-solve warm-started from a
nearly-disjoint active set: the two took different pivot sequences, both ways
-- the port cycling to PARTIAL where the reference converged in four, the
reference giving up at 58 where the port converged at 54.

Traced pivot by pivot, the fork is one number. Both admit arc 15, then arc 10;
at `A = {5, 10, 15}` the reference reads `psi[10] = -4.11e-11` and the port
reads `-1.4986e-9`, against `TOL = 1e-9`. So the reference keeps the arc and
admits arc 3; the port drops it. Everything after that is two solvers solving
different problems.

Solved in exact rationals, the truth is `psi[10] = +1.03e-10` -- *positive*,
and neither of them got the sign right. `cond(L)` is 1.6e4, which is nothing.
The mechanism is the multiplication, not the conditioning: `psi = G (u_tau -
u_sig - eps)` is a small difference of larger numbers times a conductance
running to 1e8, so a residual of `9.3e-10` in `u` -- exactly what backward
stability promises and no more -- arrives in `psi` at about `TOL`, the very
threshold the drop rule reads. numpy happened to leave zero there. The rank-1
Cholesky chain happened to leave 9.3e-10. Neither is wrong; the algorithm was
asking a backward-stable solve for a forward-accurate answer.

**Fix: iterative refinement** (`REFINE_STEPS = 3`), in both implementations.
Correct the solve against its own residual, stop as soon as a step fails to
shrink it. Restricted re-solves that disagree went **2 of 63 to 0 of 120**, and
it is *faster* -- 5.4 to ~4.8 us/solve, because an accurate `u` wastes fewer
pivots than the refinement costs. The old `xfail` is now a passing regression
test.

Two notes on where it sits and what it costs:

* It runs **after** the rank-1 drift guard, not before. Refining first would
  mend the answer and leave the factor drifting, so `REFACTOR_RESIDUAL` would
  never fire and the drift would compound. The guard prices the raw solve; the
  refinement then cleans up whatever the guard let stand.
* Mirroring it into `core/linalg.py` changes no measured result -- numpy's
  residual on these systems is already exactly zero, so `refine` returns on its
  first iteration. It is there because the two implementations must describe
  one algorithm, not because the reference needed it.

One visible consequence: on seed 3 the port now reaches the same ballot in 37
pivots where the reference spends 39, having skipped a drop-and-re-admit pair
the reference still pays for. `test_candidates_differential` keeps `solves`,
`skipped` and `skipped_wide` exact and bounds `pivots`, because `pivots` counts
steps of the numerical path and the two kernels are different by design.

### The over-constrained pins: two bugs and one decision

The last divergence was on §6.3's pin sweep, which forces an arc to carry a
multiple of what the relaxation gave it. Six of 112 pinned re-solves ended
somewhere different, in both directions. Refinement moved it by exactly
nothing -- which was the clue, because it meant accuracy was not the problem.

Traced pivot by pivot against the reference, it was three separate things, and
only the last was a judgement call.

**1. A Cholesky factor reused across a basis it does not describe.** Two paths
move the active set by more than one arc: the reconnect admits a whole path at
once, and the reseed replaces the set outright. Both cleared the *pending*
rank-1 term. Neither cleared the factor. With `keep` unchanged and the factor
still `k * k`, nothing left in the rebuild test could tell that it factorised a
different matrix -- so the next solve ran against a stale one. The drift guard
could not catch it either: it only prices a factor that came from an update,
and this one had not been updated at all. A `basis_dirty` flag now says so.
Measured at the point of failure: `u` off by 7e-2 on a solution of size 8e-3.

**2. A perturbation that reached half the problem.** On a cycling basis the
solver shifts every arc's cost by a distinct vanishing amount. The port wrote
those shifts into a local `eps`, which `psi` and `rho` read -- but built the
right-hand side from `arcs.eps`, the original. So the perturbation moved the
drop rule and not the solve, and the two answered different problems. Caught by
noticing that the reference's `u` moved between two solves of an identical
basis and the port's was bit-identical: the reference had perturbed and the
port had not.

**3. Ties decided by rounding.** With both bugs fixed, four of 220 remained.
Two arcs carrying the same flow score *identically* in exact arithmetic; what
separated them was `psi = -0.499999999999865` against `-0.499999999998981`,
a part in 1e12. Both implementations picked with `>`, which is a faithful
mirror -- and which lets the last bits of the linear solve choose the pivot.

That one needed a decision rather than a fix, because it is a real degeneracy
and not a porting error. `PIVOT_TIE = 1e-9`, relative: candidates within that
of the best count as tied and the lowest index wins, which is what the
docstring already claimed for exact ties. Relative with no absolute floor,
because the four pivot categories score in two different units and a floor big
enough to cover `psi ~ 1` would swallow distinct candidates in `rho`.

**Result: 0 of 676** pinned re-solves over 120 universes disagree, at
`maxit = 60` and at 600. Was 6 of 112. The `xfail` is now a regression test.

It is also *faster*, which was not the point but follows from the first bug:
the same 105 pinned solves cost 7.62 ms before and 6.81 ms after (-11%) while
taking 6% *more* pivots -- 1.58 to 1.33 us/pivot. A stale factor makes every
downstream refinement step work and sometimes fails Cholesky into the LU
fallback; a correct one converges immediately.

One consequence worth recording: the reference's pivot counts moved. Seed 3's
ballot took 39 pivots and now takes 37, meeting the port, because `PIVOT_TIE`
stops it from chasing a tie the other way. `test_candidates_differential` is
back to asserting `pivots` equality exactly, which it had been loosened to a
bound to accommodate.

The candidate universes are chosen to be well conditioned -- `a < 1` so every
`eps` is positive and no arbitrage cycles exist, with a real spread between
parallel arcs so no ties are left for a pivot rule to break arbitrarily -- so
that test measures the generator rather than re-measuring the solver.

### What `verify` and `pipeline` are, now that the chain is not in them

`verify.py` holds a `QuoterClient` and puts every candidate out in one
`quote_routes` at the pinned block. `pipeline.py` is the same thing one layer
up: `prepare` probes, `route` runs the quote loop, `_price_once` calls the
quoter, the scout batches probes.

Neither's I/O is ported, and that is the boundary rather than a shortfall.
Chain I/O is the host's -- a browser has its own RPC, the CLI has `transport`.
What is ported is everything that turns a chain answer into a decision, so a
host that can fetch can route. `ready()` says which candidates need pricing and
`verify()` folds back what came in; `Stages` runs the six things a quote does
between fetches.

Those six, in the order a quote runs them: reduce the universe
(`prune_dead_end_nodes`, `restrict_to_component`), assemble the graph
(`clamp_unphysical_depth`, `assemble`), check what the solve returned (the
`kcl_*` family), read the flow back out (`realised_delta`, `realised_theta`),
rank (`scout_priority`, `gas_cost`), and layer the pricing walk
(`pricing_layers`).

Two of them decide what the router can *see*, which is why they are compared
in both directions. `prune_dead_end_nodes` drops a node touched by one pool, on
structure rather than a list of names: wrong one way and the long tail of
single-pool tokens is back on the ballot, wrong the other and `HLX -> USDC` --
a fair question whose single pool is the answer -- returns no route.
`clamp_unphysical_depth` decides which arcs are bottomless, and an arc wrongly
clamped is one the solve will happily fill.

**Two approximations, both documented where they are.** `route_conductance`
and `achievable_kcl` each solve a small dense system, and the port cannot reach
for LAPACK: the first uses `lu.rs` against numpy's LU, the second a cyclic
Jacobi against `np.linalg.cond`'s SVD. Same quantities, different
implementations, so those two comparisons carry a relative tolerance.
`achievable_kcl` feeds a safety factor of 100, so what has to agree is the
scale. Everything else in these suites is exact -- including `quantum`, where
`powi` drifted a ULP at 24 decimals and `powf` does not.

### The pricing tables, and one measured limit

`curves`, `prices`, `slippage` and `refit` are the tables a route is priced
*against*, and each fails quietly in its own way -- a curve that evaluates
differently is a different split, a frame that fits differently moves `eps` and
`G` for every arc, a budget divided differently ships a minimum-out that either
reverts on any movement or protects nothing.

`refit` splits at the chain the way `verify` does: `plan` says what to quote,
`apply` folds the answers back, `rebuild` recomputes the arrays, and the caller
probes and re-solves between them. Its two floors are ported with it, which
matters -- without `REFIT_MIN_FRACTION` a refit at a realised delta of 3 USDC
replaced a fit made at a million and clamped the best pool for the pair to a
cap of 3.

**A gap in `Arcs`, found by porting `refit`.** The arc builder never carried
`calib_delta` or `decimals_out`. Nothing errored: `REFIT_MIN_FRACTION` compares
against a `calib_delta` that was always zero, so the guard was silently off on
the Rust side, and the quote scale was always 18 decimals. Both bindings carry
them now.

**One measured limit, in `curves::sizes`.** `np.geomspace` raises its interior
through `np.power`, a vectorised loop that is not correctly rounded; two nodes
in twenty-four land one ULP from libm's `pow`. That is 1.8e-16 on a probe size
of 1.8e17 -- below the pool's own integer quantum and below what the fit
reading the ladder can resolve. So that one comparison is relative while the
ladder's *shape* -- node count, strictly increasing, exact at the top -- stays
exact. Chasing it further would be chasing an implementation detail of numpy's
SIMD loop, which is not stable across its own releases.

### Calldata, which was on the wrong list

`codec` and `routecall` were filed under "the chain" and should not have been.
Neither touches a network: one is ABI encoding, the other turns a realised
route into the bytes `ElectricRouter.execute` takes. A browser that can pick a
route but not build a transaction has not got what it needs, so they are
ported, with `keccak` under them.

Nothing in that suite carries a tolerance, and it should not: a word one bit
out is a different pool, a different coin, or a minimum rate that reverts an
honest trade or admits a sandwich. What is compared is the packing (nine fields
in one word, against a layout `ElectricRouter.vy` also knows), the fractions
(`Leg.bps` is a share of what a node *held*; the contract wants a share of what
is standing there now), the minimum rates under both the fee rule and a
caller-named budget, and the calldata bytes themselves. The reference's codec
is itself held to `eth_abi`, so matching it matches the real encoder
transitively; keccak is checked against published vectors instead, because both
sides could be wrong the same way -- Ethereum uses original Keccak padding
(`0x01`), and `sha3_256` would produce plausible selectors no contract answers
to.

**A gap in `Route`, found by porting `routecall`.** The realised route had no
way to carry `verified_in`, `verified_out`, `fee_floor` or `fee_frac` across a
binding, and `min_rates` reads all four. A route encoded without them is
bounded off the *model* rather than off what the chain said -- tens of basis
points out on a cryptoswap leg, and bounding on what the leg pays rather than
on the least the pool can charge is what hands a sandwich the gap. Both
bindings carry them now, and the suite runs every route case twice, modelled
and chain-priced.

**One narrowing, stated rather than silent.** The port holds signed ABI
integers in `i128`, so `int144` and wider are refused at parse time. Python
integers are unbounded, so the reference carries `int256` for free; Curve's ABI
uses `int128` for coin indices and nothing wider.

**What has no Rust form:**

| module | lines | what it is |
|---|---|---|
| `pipeline`'s I/O half | ~1,400 | `prepare`, `route`, the probe and quote loops |
| `probe`, `quoter`, `transport`, `evm` | — | the chain itself |

Everything that decides is ported, everything that *encodes* is ported, and so
now is the model-free floor. What is left is fetching. The renderers stay the
CLI's.

### Where a warm quote goes, with everything that is wired turned on

Measured at block 25,870,667-82, USDC -> WETH at $100k, `EROUTER_ACCEL=1`,
private endpoint, min of the reps.

    arms          rust 110.44 ms  python 156.31 ms  1.42x, 45.86 ms saved
                  verified_out identical to the wei
                  (1.22x before generation was wired in; the absolute times
                   drifted with the universe between sessions, so the ratio is
                   what compares, and only within one interleaved run)

    stages        candidates 72.2%   refine 6.5%   solve 5.0%   direct 3.1%
                  realize 1.6%   legs 1.3%   seed 1.3%   graph 1.2%

    crossings     404 per quote, 56.11 ms inside Rust (74.0%)
                  of which solve_arrays 52.65 ms over 67 calls
                  everything else that crosses:      3.45 ms
                  Python around it:                 19.74 ms (26.0%)

    solves        65 per quote, 60.13 ms, 71.3% of the whole quote
                  61 of them asked for by `candidates.resolve`

**Two thirds of a warm quote is already native, and it is all one function.**
`solve_arrays` is 71% of the quote; the crossings that are not it cost 3.45 ms
in total, across 337 calls. The boundary is not what is left to win.

**What `EROUTER_ACCEL=1` actually turns on**, which is less than the port:
`solve`, `calibrate`, `seed`'s shortest path, `realize`, `split`, and
`pipeline`'s resident pool models with the batched refit, and `candidates`'
whole generation. That is seven places.
The modules ported after them -- `prices`, `slippage`, `refit`, `routecall`,
`verify`, `multiport`, `graph`, `nodes`, `curves`, `gas`, `risk` and `naive` --
have Rust twins under differential test and **no caller in the Python quote
path**. They are reachable from wasm, where the whole thing is
Rust, and directly through `erouter_solve`. So the 1.22x above is the six, not
the port; the rest of the port is portability, exactly as this document has
said, and the number that bounds what wiring it would buy is the 19.74 ms of
Python around the crossings -- not the 72% in `candidates`, which is already
native solving.

### Wiring generation in, and what it was actually worth

`candidates.generate` was the one stage both hot and fully ported, so it got a
caller. `Graph.from_arrays` is new and is the reason it works: by the ballot,
`g` has been scaled, clamped and merged, and `Graph.build` would re-derive `G`
from `a`, `B` and `nu` and answer a different question -- `g_scale` alone moves
every conductance. `flagged` crosses with it, not as a nicety: the pin sweep is
*defined* over the flagged active arcs, and defaulting it to false deleted the
family outright. A test caught that, not a review.

Measured interleaved, everything else held on:

    rust generation      90.26 ms      python generation   96.81 ms
    1.07x, 6.55 ms saved, verified_out identical to the wei

**1.07x, not the 1.42x** the arms now report for the branch as a whole. That
is the prediction this document made a section earlier holding up: the 72% in
`candidates` was already native solving, and what wiring the generator could
reach was the Python loop *around* it. 6.55 ms of ~19.7 ms.

The boundary is not where it went, either:

    graph      0.07 ms   0.1%      arcs (477 adds)  0.29 ms   0.3%
    generate  53.03 ms  53.5%      of a 99.2 ms quote

Marshalling the whole universe across costs 0.36 ms once per quote. Generation
is now a single call holding 53% of the quote, where it was 61 separate solve
crossings with Python between them.

**A measurement bug worth recording, because it read as a result.** `arms`
fakes master by turning accelerators off, and its `toggle` names the modules to
turn off one by one. `candidates` was added and `toggle` was not, so both arms
ran the Rust generator and the run reported 1.32x -- a number that survives
turning the thing being measured off. `toggle` now names it, and the docstring
says why every future accelerator has to be added there too. The failure is
silent by construction: an arm that is not master still produces a plausible
ratio.

### A second pair, and where the 1.42x actually comes from

`crvUSD -> sDOLA` at $100 is nothing like `USDC -> WETH` at $100k: a hundred
dollars through a two-hop stable route, `candidates` 62% of the quote rather
than 72%. Pinned at block 25,871,000, interleaved:

    rust (this branch)   67.42 ms      python (master)   95.70 ms
    1.42x, 28.28 ms saved, verified_out identical to the wei

The same ratio as the other pair, which is worth more than either number alone:
the win is not an artefact of one route's shape.

It does not come from where the recent work went, though. Stage by stage
against master at that block:

    refine       15.24 -> 4.41 ms     the resident pool models
    candidates   58.28 -> 49.69 ms    generation in Rust

Generation -- the thing wired in most recently -- is 7.8 ms of the 28, and on
its own is 1.12x (measured by toggling `candidates._ACCEL_ON` rep by rep in one
process). The larger single piece is `refine`, which the resident models and
the batched refit cut by 3.5x. That is the same lesson as §the wiring: the
stage that *looks* expensive is mostly native solving already, and the wins are
in the Python around it.

### The warm-up sweep is the network, and nothing else

The one thing the EVM measurement left open. Same process, same endpoint,
block 25,870,860, 383 pools / 856 arcs / 368 exact models, caches warm so
nothing was re-fetched:

    warm phase         wall ms    cpu ms  cpu/wall
    block                121.3       0.9      0.01
    caches                90.2      89.8      1.00
    code                  56.2      56.1      1.00
    storage            10130.2     138.1      0.01
    universe               4.1       4.1      0.99
    pools               3511.2    1426.8      0.41
    wrappers            1028.7      98.8      0.10
    arcs                  86.2      86.1      1.00
    models              1637.0    1133.1      0.69
    ------------------------------------------------
    warm total         16665.5    3034.1      0.18

**18% CPU.** The storage sweep alone is ten seconds of wall for a tenth of a
second of work -- 1% -- and it is the single largest phase. Making every
remaining line of Python free would take 16.7 s to 13.6 s.

`refresh`, which is what runs on a timer once warm, is worse: **10.9 s wall,
0.60 s CPU, 6%**. It re-reads every loaded slot at the newest block and that
is all it does.

So there is nothing here for the port, and the earlier note that the EVM
measurement covered only the warm path is now closed rather than open: revm
inside would not help the sweep either, because the sweep is not executing --
it is waiting for a socket.

### Turning the accelerator on makes the *warm* slower

Which is worth stating plainly, because it is the opposite of the direction
everything else in this document points.

    accel      warm cpu     warm quote
    on          3034 ms       87.06 ms
    off         2569 ms      403.46 ms
    delta       +465 ms      -316 ms

`prepare` builds `_Resident` -- the Rust-side pool models -- only when the flag
is on, and that build is warm-time work the quote then reads for free. So the
accelerator does not remove work, it *moves* it: 465 ms more in the sweep,
316 ms less in every quote after. It pays for itself in **1.5 quotes**, and the
router is not built to quote once.

The 4.6x here is against pure numpy, not against master: this arm has the
solver in Python too, where `master` has had it in Rust for some time. Against
master the figure is the arms' 1.42x. Both are real; they answer different
questions, and neither should be quoted without saying which.

### The EVM is 0.3% of a warm quote

`erouter_evm` is revm, so it could move inside the solver crate and stop
crossing at all. Measured before deciding, per warm quote:

    call                  2 crossings    0.21 ms    0.3%

Two. The exact pool models answer from resident Rust state, so the sweep is
what makes their getters free (§ the exact-model warm) and the EVM is left
holding almost nothing on this path. There is no round-trip count to optimise
away, and folding revm in would buy at most 0.21 ms while making the crate
depend on an EVM to do arithmetic. It stays outside.

That was a statement about the warm path only when it was written; the sweep
is measured above and is 18% CPU, so it does not change the answer.

### The floor, and what porting a function that quotes looks like

`direct_candidates` is pure and crossed as one call. `two_step_candidates` did
not, and it is the first thing ported here that *calls the chain from inside
itself* -- two batched probe rounds, with ranking between them.

Wrapping that would have meant a callback across the boundary twice per quote.
Instead it splits at the rounds into three stateless stages -- plan the first
round, rank it and plan the second, build the chains -- and the caller does the
quoting. That is the shape a browser needs anyway, where the chain is always on
the far side of the boundary, and it makes the intermediate state inspectable
rather than trapped in a closure.

Two things came out of the porting rather than the design:

* **A repeated coin has to keep its first position and its last index.** The
  reference builds `{coin.address: k}` and iterates it, so a pool listing the
  same token twice yields one entry, positioned where it first appeared and
  numbered where it last did. A `HashMap` gets both halves wrong.
* **An unknown token is an error, not an empty floor.** `nodes.node` raises in
  the reference. The port returned an empty list at first, which is worse than
  it sounds: the floor exists precisely to be the answer when everything else
  came back empty, so "no route" is the one thing it must never say by
  accident. Both sides now refuse.

30 differential tests, including one that drives both targets from one fake
chain keyed by the probe rather than by call order -- a harness answering
positionally could not tell whether the two asked the same questions in the
same sequence, which after a split into stages is the only thing worth
checking.

**Four approximations, each documented where it is.** `route_conductance`,
`achievable_kcl` and the reference-price fit each solve a small dense system
and the two sides solve it differently -- `lu.rs` and a cyclic Jacobi here
against numpy's LU and SVD there. Not because LAPACK is out of reach: it binds
from Rust, and `faer` and `nalgebra` are pure-Rust and would cross to wasm.
`rust/README.md` rules it out on its own terms, and it would not buy exactness
anyway -- numpy here is linked against OpenBLAS 0.3.34 built `DYNAMIC_ARCH`
with `MAX_THREADS=64`, so `np.linalg.solve` is not one fixed sequence of
operations even between two machines. The tolerance is the price of the
determinism rule, and the *reference* is the less reproducible of the two.
Plus the `geomspace` ULP in `curves::sizes`. Everything else is exact.

The Jacobi in `achievable_kcl` was measured rather than assumed: 1.07 ms at
`n = 50` against numpy's 0.096, on a path that runs *only* when a quote is
already failing; and 2.4e-5 relative over the whole range the graph admits,
against a safety factor of 100. It stays.

None of this shows in the arms. `candidates` is 92% a native solver, `realize`
is 2.4 ms, and the Python pipeline still calls Python at every stage -- nothing
in this work changed the hot path. This is the portability half of the goal,
and the measurements in this document are the reason to say so plainly rather
than let a 1.4x figure imply otherwise.


### A refusal that was a SIGABRT, and what unwinding it costs

Python signals "this pool will not do that trade" by raising out of an
`ArithmeticError`; the port signals it by returning `None`. Neither is a crash,
and the router is built on that -- an arc that refuses is an arc the route does
not take.

`[profile.release]` said `panic = "abort"`, carried in from the first commit of
the port and never revisited. That turns every panic in 16,000 lines of Rust
into `SIGABRT`: not an exception, not a failed quote, the whole CPython process
gone with no traceback. PyO3 converts a panic into `PanicException` at each
binding, but only one it is allowed to catch, and `abort` never lets it.

The exposure is not the `unwrap()`s, which are countable -- 24 outside the
tests, and all 24 are gone. It is that `ruint` panics on a subtraction that
borrows and on a product that wraps, **regardless of `overflow-checks`**, and
the pool models are ~550 arithmetic sites of exactly that kind on state read
off a chain. Proving each one total is not a thing that stays proved.

Two reachable ones, found rather than reasoned about:

* **`Curve.at(NaN)`.** `v <= 0.0` admits a NaN, `v >= x[-1]` rejects it, so it
  reached a bisection that cannot order it. In the reference that is an
  `IndexError`; here it was the abort. Both now spell the tail test
  `not v < x[-1]`, which sends a NaN down the branch where it falls out as zero
  and is the same comparison for every number that compares.
* **A coin the pool does not have.** Twocrypto, tricrypto and both LP models
  have always refused an index out of range. Stableswap did not -- it indexed
  `xp` straight, which is an `IndexError` in the reference and an index off the
  end of a `Vec` in the port. Found by fuzzing the registry over 4,000 hostile
  states; it is the sort of index a stale pool record produces.

So both halves: the models refuse where they can, and `Registry::price_one`
catches an unwind and returns `None` where they cannot, which is the same
refusal the caller already handles. The default panic hook still prints, so a
model that reaches the net is a bug report rather than a silence -- and
`test_bad_input_never_aborts` asserts on an empty stderr, not merely on
survival.

**What it costs**, both arms built from the same source and interleaved so the
governor cannot pick a winner (§ the min-not-median rule above):

| | `abort` | `unwind` |
|---|---|---|
| 72 pinned solves, 3,576 pivots | 3.632 ms | 3.835 ms (+5.6%) |
| 1,680 prices, float invariant | 1.205 ms | 1.203 ms |
| 1,680 prices, exact invariant | 3.399 ms | 3.399 ms |
| extension | 2,014,376 B | 2,252,096 B (+11.8%) |
| wasm module | 2,302,052 B | 2,302,052 B |

Landing pads cost in the pivot loop and nowhere else that was measured -- the
pool models, `catch_unwind` seam included, do not move at all. The solve is
~23% of a warm quote, so the end-to-end price is under 2%.

**The browser has no such net.** `wasm32-unknown-unknown` cannot unwind, so the
profile is ignored there -- the module comes out the same size and
`catch_unwind` never catches. A panic traps the instance and poisons it. That
is the argument for the guards being in the models rather than only at the
boundary: they are the only half of this that crosses.

### Then fuzzing the boundary, which found a class the models did not have

The models were the half that could be reasoned about. The bindings were not,
and asking whether the job was finished is what turned up the rest: driving all
207 entry points with arguments they have no answer for produced **37 panics in
372,600 calls**, none of them an `unwrap()` and none in the pool arithmetic.

Two shapes, and every instance is one or the other:

* **Parallel arrays that are not parallel.** `check_pair_drops(eps_f, eps_r)`,
  `round_stats(before, after)`, `b_change`, `ceiling_conductance(g, flagged)`,
  `cancel_cycles(tau, sig, psi)`. The reference subtracts two numpy arrays or
  zips `strict=True` and refuses a mismatch; a `Vec` either reads past the
  shorter one or stops at it and answers for the longer, and the second is
  worse than the panic because it is a number.
* **An index that is not an index.** `tau`/`sig` cross as plain integers and
  the core indexes them straight into an `n_nodes`-long vector; `Problem.solve`,
  `Problem.shortest_path`, `NodeMap.node_symbol` and `find_cycle` all took one
  from the caller and used it. `IndexError` in the reference.

Both are now refused by three helpers at the boundary (`same_length`,
`arc_nodes`, `node` -- `src/py.rs`, and the twins in `wasm/src/guard.rs`),
checked **once per problem rather than once per solve**: a quote solves the
same arcs ~45 times and the arrays do not change between them, so the cost is
one pass over `m` at construction and nothing in the pivot loop.

869,400 calls across 24 seeds now, zero panics. A bounded version of the sweep
is in the suite; the deep one is not, because at 25 calls per target it finds
nothing and at 200 it is not a unit test.

**The wasm bindings needed the same work separately**, and that is the part
worth remembering: the guards inside the models are shared code and crossed for
free, but a check written in `slippage_py.rs` protects only CPython. The
browser had none of them, on the target where a panic is unrecoverable.

### Two more, from asking whether that was all of it

It was not, and both of the remaining ones came from widening the sweep rather
than from reading more code.

**A third of the surface was never entered.** `Route`, `Ballot`, `Stages`,
`Refit` and `RouteCall` do not exist until a quote has been through them, so
the first sweep called every method on them with a receiver PyO3 rejects and
learned nothing. Building the five properly turned up `component_of`, which
indexes its mask by each arc's endpoint while sizing it from a separate
`n_nodes` the caller also supplies -- reached through
`Stages.restrict_to_component`, and fixed in the core rather than at the
binding so `pipeline` and the browser get it too.

**`unwind` does not save you from an allocation.** `n_nodes = 10**18` sizes a
`vec![false; n]`, Rust's allocator cannot serve it, and `handle_alloc_error`
calls `abort()` -- the same dead process as before, with the panic strategy
making no difference at all. Two of the sixteen fuzz seeds died this way.
`numpy` raises `MemoryError` for the same input, so once again the refusal is
the mirror. The bindings now cap `n_nodes` at 4,194,304 and the one bound that
squares -- `laplacian`'s `keep`, which is a dense `k x k` under a cubic LU --
at 8,192. Both are orders past anything measured and both are cheap: one
comparison per call, not per pivot.

Standing at 662,400 calls over sixteen seeds with the pipelined fixtures, plus
106,200 through the wasm module and 15,000 through the EVM extension: no panic,
no abort, no trap. The suite carries a bounded version, and asserts the fixture
set still holds all five pipelined classes -- a constructor that changes shape
would drop one silently and take its methods back out of the sweep.

## A trade smaller than the ladder, and the fee that was not there

`USDT -> trUSD` at 0.001 USDT did not return a bad route. It returned no route:

    flow conservation is violated by 2.496e-03 of the routed value at frxUSD
    (1 arc(s) in, 3 out) (achievable at this conditioning: 2.229e-04)

Which reads as a solver failure and is not one. The refine pass sizes its
probes off the trade -- `TRADE_GRID` is 5%, 10% and 20% of it -- and
`plan_sized` floored those only at one millionth of a unit of the *input*
token. So a trade of 0.001 USDT asked its pools for probes 2.1e+05 below the
smallest node their coarse ladder already held -- five orders at the median,
eight at the worst -- and `a = y0/x0` is read off exactly that node. An answer
of `n` units of the output token carries a rounding error of `1/n`;
`eps = 1 - a nu_sig/nu_tau` inherits it whole, and a part in ten thousand of
`a` is a basis point of fee that is not there.

Measured at block 25,891,322, over the arcs the graph is assembled from. The
"after" column is what the refine pass switched off entirely also produces, to
the digit, at every size:

| | before | after |
|---|---|---|
| median `G` at 0.001 USDT, value units | 2.30e+06 | 7.23e+06 |
| median `G` at 1e-6 USDT | 3.66e+06 | 7.23e+06 |
| most negative `eps` at 0.001 USDT | -75.5 bp | -75.6 bp |
| most negative `eps` at 1e-6 USDT | -309 bp | -75.6 bp |
| largest \|psi\| in the flow, 0.001 USDT | 2.6e+04 | 4.2e+03 |
| largest \|psi\| in the flow, 10 USDT | 4.2e+03 | 4.2e+03 |

The last pair is the mechanism. A negative-`eps` cycle is sized by `|eps| G`,
which has nothing to do with the trade, so the flow the solve is conditioned on
is an absolute quantity while §12.4's residual is normalised by `Psi` -- 4.2e+03
of circulation at every size, against a demand of 0.001. Cut the conductances
by three and the circulation grows sixfold; divide that by the trade and the
gate refuses a quote whose arithmetic was fine.

The floor is now the smallest size the ladder has already answered at, which is
the resolution `a` is already standing on. `plan_refine` gained the
`reserve_out` argument it had been dropping, for the same reason: it re-asks
sizes on an arc `plan_grid` had already floored.

KCL residual, `USDT -> trUSD`, same block, before and after:

| input | before | after |
|---|---|---|
| 1e-6 USDT | 6.5e+00 fail | 1.7e-02 fail |
| 1e-5 | 3.2e+00 fail | 1.7e-03 fail |
| 1e-4 | 7.0e-03 fail | 1.2e-04 ok |
| 1e-3 | 2.5e-03 fail | 9.0e-06 ok |
| 1e-2 | 7.9e-05 ok | 1.5e-06 ok |
| 1e-1 | 4.6e-06 ok | 2.6e-07 ok |
| 1 | 2.1e-08 ok | 2.5e-08 ok |
| 10 | 3.8e-09 ok | 4.6e-09 ok |

**It costs the trades it is not about nothing at all.** From a tenth of a
dollar up the answer is unchanged to the wei -- where the floor still lifts a
size it lifts it onto a node the ladder already holds, so nothing is re-asked
and nothing is re-fitted, and above ~$1,000 the trade-sized nodes are the
larger ones anyway, which is what the refine pass was for. Identical from 0.1
USDT to 1,000,000 USDT on `USDT -> trUSD`, identical at
every size that quoted before on `USDC -> WETH` and `WETH -> WBTC`, and
`bench_quote --arms` still agrees to the wei across the boundary. The share of
refine probes the floor lifts decays exactly as that says it should -- all
1,413 at 1e-4 USDT, 441 at 1,000, 11 at 100,000, none at 1,000,000 -- and so
does the lift: 2.1e+05 at 0.001 USDT, 226 at 1, 4.4 at 1,000, 1.8 at 100,000.

What is left below 1e-5 USDT is not the probes. The measured `eps` is then the
same to the digit as with the refine pass switched off entirely, and the
residual still grows as `1/Psi` because `_achievable_kcl` bounds the solve's
error relative to `Psi` while the flow it actually solved for is `|eps| G`.
Scaling that bound by `||psi||_inf / Psi` would be the correct statement of
what a backward-stable solve can deliver here, and it would also stop the gate
from seeing conjured flow at exactly the sizes where circulation dwarfs the
trade. Left alone: refusing a trade of one hundredth of a cent is the right
answer, and this is the message saying so, badly.
