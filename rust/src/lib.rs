//! The quadratic-flow router's hot path (spec §5.4), in Rust.
//!
//! Scope is deliberately narrow: the active-set QP and the primitives it runs
//! once per pivot.  Everything above it -- candidate generation, calibration,
//! realisation, verification -- stays in Python, where it is still changing.
//! This is the part that is numerically settled and measured to dominate.
//!
//! It runs unchanged in CPython, in Pyodide, and in a bare Web Worker, which
//! is why there is no I/O, no threading, no clock and no BLAS anywhere in it.
//! See `README.md` for what each of those would break.

// `!(x > 0.0)` rather than `x <= 0.0`, seven times over: the negation rejects
// NaN and the comparison admits it, and every one of those sites is a guard
// whose whole job is to refuse one.  Clippy reads it as a style slip; it is the
// difference between a `Singular` error and a factorisation full of NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod chol;
pub mod ladders;
pub mod pools;
pub mod lu;
pub mod solve;

#[cfg(feature = "python")]
mod ladders_py;
#[cfg(feature = "python")]
mod py;
pub mod calibrate;
pub mod seed;
pub mod cycles;
pub mod split;
