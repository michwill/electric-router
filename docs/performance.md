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

The worst column looks alarming and is not. Every one of those cases is one or
two wei:

    5.03e+01 bp   =  1 wei on an output of 199 units
    5.21e+00      =  1 wei on 1,918
    1.00e+00      =  2 wei on 19,933

534 of 2,656 samples differ by a single wei. So the figure is not a property
of the arithmetic, it is a property of how many units the answer has -- and
the router already declines to ask at sizes where that is few. `probe.py`
floors every probe at `MIN_OUT_QUANTA = 10,000` output units, because "an
answer of `n` units carries a rounding error of `1/n`". Above that floor the
worst case over all three families is **1.00 bp, and it is two wei**; below
it, 50 bp on 199 units.

Which is the answer to whether the float path is safe to price with: it is
wrong by a couple of wei, always, and what that is worth in basis points is
decided by the size, not by `f64`.

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

## The shape of what is left

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
