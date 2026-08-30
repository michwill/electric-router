//! The quadratic-flow router (spec §5.4), in Rust.
//!
//! The goal is a router that runs without Python at all -- in a browser, in a
//! JS runtime, anywhere a wasm module loads -- so this is growing upward from
//! the QP rather than staying at it.  Settled so far: the active-set solve and
//! its per-pivot primitives, the pool models, the refine stage's ladders, and
//! now the graph assembly and the node map that feed them.  Still Python-only:
//! candidate generation, realisation, verification and the pipeline that
//! drives them.  `docs/performance.md` keeps the running inventory.
//!
//! `core/*.py` stays the reference.  Every change lands there first and is
//! mirrored here -- which is why the differential tests compare bit for bit
//! rather than within a tolerance.
//!
//! It runs unchanged in CPython, in Pyodide, and in a bare Web Worker, which
//! is why there is no I/O, no threading, no clock and no BLAS anywhere in it.
//! See `README.md` for what each of those would break.

// `!(x > 0.0)` rather than `x <= 0.0`, seven times over: the negation rejects
// NaN and the comparison admits it, and every one of those sites is a guard
// whose whole job is to refuse one.  Clippy reads it as a style slip; it is the
// difference between a `Singular` error and a factorisation full of NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod candidates;
pub mod chol;
pub mod curves;
pub mod gas;
pub mod graph;
pub mod multiport;
pub mod nodes;
pub mod prices;
pub mod pyfmt;
pub mod refit;
pub mod risk;
pub mod realize;
pub mod types;
pub mod verify;
pub mod ladders;
pub mod pipeline;
pub mod pools;
pub mod lu;
pub mod slippage;
pub mod solve;

#[cfg(feature = "python")]
mod candidates_py;
#[cfg(feature = "python")]
mod curves_py;
#[cfg(feature = "python")]
mod graph_py;
#[cfg(feature = "python")]
mod ladders_py;
#[cfg(feature = "python")]
mod multiport_py;
#[cfg(feature = "python")]
mod nodes_py;
#[cfg(feature = "python")]
mod realize_py;
#[cfg(feature = "python")]
mod pipeline_py;
#[cfg(feature = "python")]
mod prices_py;
#[cfg(feature = "python")]
mod refit_py;
#[cfg(feature = "python")]
mod slippage_py;
#[cfg(feature = "python")]
mod py;
pub mod calibrate;
pub mod seed;
pub mod cycles;
pub mod split;
