//! The pool models: what a quote is actually made of.
//!
//! Two arithmetics, two jobs, and they do not compete.
//!
//! **`U256` admits a pool.** The integer form *is* the contract, wei for wei,
//! and that is what makes the admission gate mean anything: a model is trusted
//! only when it reproduces the chain exactly, and a wrong rate shows up as a
//! one-wei disagreement. Runs once per pool at the warm.
//!
//! **`f64` prices with it.** A quote evaluates these thousands of times to
//! rank candidates that differ by basis points, where the last wei buys
//! nothing. Measured against the integer answers over 1,156 vectors on 159
//! mainnet pools, it is out by ~1e-11 of `dy` at any size the router probes --
//! and that residual is `y`'s round-off carried through `dy = xp[j] - y - 1`,
//! not a convergence failure, so it does not improve by iterating harder.
//!
//! Only the invariant iteration has a float form. The fee, the price scale and
//! the precisions stay integer on both paths, which is why the two share one
//! set of vectors: the float path hands `y` back and the same wrapper finishes
//! the quote.
//!
//! No I/O, no clock, no threading -- these compile to wasm32 alongside the
//! solver, which is the whole reason they live here rather than in a crate of
//! their own.

pub mod cryptoswap;
#[cfg(feature = "python")]
pub mod py;
pub mod lp;
pub mod prims;
pub mod registry;
pub mod stableswap;
pub mod tricrypto;
pub mod twocrypto;
