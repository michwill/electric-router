# erouter-solve

The router's hot path in Rust: `active_set_solve` and the primitives it runs
once per pivot (component reachability, Laplacian assembly, dense LU).

Written to run in three places without changing:

* **CPython**, as a PyO3 extension (`--features python`), used by
  `erouter.core.accel` when it imports.
* **Pyodide**, the same extension built for `wasm32-unknown-emscripten`, which
  is what the Flet frontend loads.
* **A bare wasm module** for a Web Worker, via the `rlib` and a thin
  `wasm-bindgen` wrapper, for a frontend that would rather not go through
  Python at all.

Consequences for how it is written, all of them load-bearing:

* **No I/O, no threads, no clock, no allocator tricks.**  A Web Worker has no
  filesystem and no `std::time`; anything that touches them will compile on the
  host and fail in the browser.
* **No BLAS or LAPACK.**  The dense solves are n ~ 50 (median measured on
  mainnet) and at that size a hand-written LU with partial pivoting beats the
  cost of binding a Fortran library that has no wasm build.  It is ~40 lines
  and it is in `lu.rs`.
* **Deterministic.**  No parallelism and no floating-point reassociation, so a
  quote is reproducible across the three targets.  The Python implementation
  stays the reference; `tests/test_accel_differential.py` differs them, and
  both are differed against OSQP (§13.3).

`erouter.core` must remain importable with nothing but numpy, so this is never
a hard dependency: `accel.py` tries the import and the pure-Python path answers
when it is absent.

## Status

**Opt-in, not yet the default.**  Set `EROUTER_ACCEL=1` to route through it.

What is established: it reproduces `core/solve.py` to 1e-12 on six shaped
problems and twelve fuzzed graphs, honours pins and forbidden arcs identically,
and matches OSQP's objective on all of them (`tests/test_accel_differential.py`).
The whole Python suite passes with it enabled.

What is not: on real quotes it still diverges where the solve does not converge
cleanly.  `USDC->WETH $20M` returned 9,052 WETH through Python and 4,681
through Rust.  Those quotes reach `maxit`, cycle under Bland's rule and return
`PARTIAL` -- paths a clean synthetic problem never takes, which is why the unit
differential passed while the real quote did not.  **The lesson is about the
tests, not the port**: a differential over problems that converge in a handful
of pivots cannot cover a solver whose interesting behaviour is what it does
when it fails to converge.

It is also not yet faster end to end -- 0.9x on four of five real quotes.  The
solve is about a quarter of a warm quote, and marshalling the arrays across the
boundary costs roughly what the arithmetic saves.  The 20x seen on a 20-node
synthetic problem is real and irrelevant: the measured per-pivot system is
n = 49 median, and the crossing happens 45 times a quote.

### What to do next, in order

1. **Force the hard paths in the differential.**  Small `maxit`, graphs built
   to cycle, `partial_ok=True`.  Until those are covered the port cannot be
   trusted, and with them the remaining divergence is findable in minutes.
2. **Then fix the divergence**, which is somewhere in the cycling and
   exhaustion branches -- the same region where a phantom `CLEANUP_ROUNDS`
   phase (reverted in the Python, ported from memory) was already found and
   removed.
3. **Then reconsider the boundary.**  Lists cost; numpy's buffer protocol would
   avoid a copy, at the price of the numpy C API and a harder Pyodide build.
   Or move more of the pipeline across -- `calibrate` is another ~25% and has
   no such marshalling problem, since it is called far less often.

### Browser build

Not attempted yet.  The wheel here is `abi3` for CPython; Pyodide needs the
same crate built for `wasm32-unknown-emscripten` against Pyodide's own ABI,
which needs `rustup` (absent on this machine) to add the target.  The crate is
written for it -- no I/O, no threads, no clock, no BLAS -- and the `rlib` also
allows a `wasm-bindgen` wrapper for a Worker that skips Python entirely.
