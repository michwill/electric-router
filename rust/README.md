# erouter-solve

The router in Rust. It began as the hot path -- `active_set_solve` and the
primitives it runs once per pivot -- and now carries the stages between the
fetches too: graph assembly and node merging, multi-port elements, `realize`,
the candidate ballot with `verify`/`gas`/`risk`, the pricing tables
(`curves`, `prices`, `slippage`, `refit`), calldata (`keccak`, `codec`,
`routecall`), and `pipeline` over the top.

What is deliberately *not* here is anything that talks to a chain: `evm.py`,
`quoter.py`, `transport.py`. Those are I/O, and the rules below forbid it.

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
* **A panic must stay a panic.**  `[profile.release]` sets `panic = "unwind"`,
  which is what lets PyO3 hand CPython an exception instead of `SIGABRT`: with
  `abort`, one `U256` subtraction that borrows takes the whole process down.
  It costs 5.6% of the pivot loop and nothing on the pool models.  **wasm32
  cannot unwind**, so the browser build has no net at all and a panic poisons
  the module -- which is why the refusals live inside the models, where they
  cross, rather than only at the bindings.  `docs/performance.md` has the
  numbers and `tests/test_bad_input_never_aborts.py` holds the line.
* **Deterministic.**  No parallelism and no floating-point reassociation, so a
  quote is reproducible across the three targets.  The Python implementation
  stays the reference; `tests/test_accel_differential.py` differs them, and
  both are differed against OSQP (§13.3).

`erouter.core` must remain importable with nothing but numpy, so this is never
a hard dependency: `accel.py` tries the import and the pure-Python path answers
when it is absent.

## Status

**Opt-in, not yet the default.** Set `EROUTER_ACCEL=1` to route through it.

What is established. It reproduces `core/solve.py` on every differential in
`tests/`, which now includes the paths that matter rather than only the clean
ones: the degenerate tail (`test_the_degenerate_tail_agrees` -- 216 solves,
two-thirds returning PARTIAL, exact on reason and flow at budgets of 12, 40 and
600) and §6.3's over-constrained pin sweep
(`test_over_constrained_pins_agree` -- 72 pinned re-solves over 12 universes;
the sweep that fixed it ran 676 over 120 and is recorded in
`docs/performance.md`).  It matches OSQP's objective, and
`test_wasm_differential` holds the browser build to **byte equality** with the
native one.

End to end at a pinned block, toggling the Rust models in one process:
**1.32x**, 78.90 ms against 103.82 median, with `verified_out` identical to the
wei. See `docs/performance.md` for where that came from and what it cost.

What is not established. The reason the default has not flipped is no longer a
known divergence -- it is that the evidence is synthetic. The claim this
crate was gated on ("the quote lands hundreds of bp apart") was measured by
replaying 94 problems off live quotes at theta in the hundreds of percent, and
the three defects behind it are fixed and covered. Repeating that replay is
what remains, and it needs a chain.

### What to do next, in order

1. **Repeat the live replay** that produced the original divergence claim, and
   flip the default if it agrees. Everything below it is smaller.
2. **`WSTETH_UNWRAP` has no model at all** -- not an unported one, none. It is
   the first blocking leg on 92 of 156 routes, which is why they go whole to
   revm at 481 us instead of being walked at 197. `wrappers.py` already reads
   the rate at the warm and records `getStETHByWstETH` as linear to 1.3e-19
   across eight decades, so the number is in hand. Whether that clears the
   wei-for-wei bar the other exact models are admitted by is a decision, not a
   transcription; `docs/performance.md` states the case.
3. **Then `walk_route`**, which is worth ~2.5 ms and is mostly blocked by the
   above -- 56 of 156 routes are walkable in Rust at all today.
4. **Then the boundary.** Lists cost; numpy's buffer protocol would avoid a
   copy, at the price of the numpy C API and a harder Pyodide build.

Still in Python: `schema.py`, `poolfee.py`, the two renderers, and the
planning half of `probe.py` -- `ladders.rs` holds the ladders and `plan_sized`,
but `plan_grid`, `plan_refine` and `plan_deltas` are not across, and
`pipeline.py` imports them. None of it is on the measured hot path. Whether a
browser quote can be driven end to end without them has not been tried, so
treat that as open rather than answered.

### Browser build

Built, and it is the `wasm-bindgen` route rather than the Pyodide wheel.

A `wasm32-unknown-emscripten` extension would have to match the Emscripten
version Pyodide was built with *and* be built with a pyo3 that supports its
CPython -- Pyodide 314.0.3 is CPython 3.14, which pyo3 0.23 does not target.
A `wasm-bindgen` module has neither constraint: it is the same Rust for the
browser's own target, and `src/erouter/wasm/` presents the PyO3 spelling to
Python, so `core/accel.py` never learns which one answered.

    ./scripts/build_wasm.sh          # -> rust/wasm/pkg/

2.30 MB, 806 kB gzipped, carrying the solver *and* the EVM (see `evm/`).
`tests/test_wasm_differential.py` differs it against the native extension and
requires **byte equality** -- same crate, same compiler (`rust-toolchain.toml`
pins 1.96.1), and `sqrt` is the only non-trivially-rounded operation in the
solver, which IEEE-754 requires be correctly rounded on both targets.  The
hard paths are in there too: `maxit` exhaustion, cycling under Bland's rule,
`partial_ok`.  Those are what the earlier port got wrong while passing every
clean problem.

### Toolchain

The system `rust-bin` has x86_64 std only, so this needs rustup, and a
`wasm-bindgen` CLI whose version *equals* the crate's -- it refuses a mismatch,
which is why `wasm/Cargo.toml` pins it with `=`.

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.96.1
    ~/.cargo/bin/rustup target add wasm32-unknown-unknown
    curl -L https://github.com/wasm-bindgen/wasm-bindgen/releases/download/0.2.127/\
    wasm-bindgen-0.2.127-x86_64-unknown-linux-musl.tar.gz \
        | tar xz -C ~/.local/bin --strip-components=1 --wildcards '*/wasm-bindgen'
    uv tool install maturin==1.14.1

Nothing needs root and nothing replaces `/usr/bin/cargo`.
