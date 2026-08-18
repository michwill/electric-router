//! PyO3 bindings.  Behind the `python` feature so the same core compiles to a
//! bare wasm module with no Python in it at all.
//!
//! The boundary is one call per *solve*, not per pivot.  A quote runs ~45
//! solves and ~2,250 pivots, so crossing per pivot would pay the FFI cost 50x
//! more often for no benefit -- and it would leave the pivot loop in Python,
//! which is the part that costs.
//!
//! Arrays come in as plain Python sequences of floats and ints rather than as
//! numpy buffers.  That keeps the extension free of the numpy C API, which is
//! what lets the same wheel work under Pyodide, where matching numpy's ABI is
//! the fiddliest part of shipping a native module.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::solve::{active_set_solve, Arcs, Options};

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (tau, sig, g, eps, cap, n_nodes, src, dst, psi_total,
                    a0=None, forbidden=None, pinned=None, tol=None, maxit=600,
                    min_flow=0.0, gas_cost=0.0, partial_ok=false))]
fn solve<'py>(
    py: Python<'py>,
    tau: Vec<i64>,
    sig: Vec<i64>,
    g: Vec<f64>,
    eps: Vec<f64>,
    cap: Vec<f64>,
    n_nodes: usize,
    src: usize,
    dst: usize,
    psi_total: f64,
    a0: Option<Vec<bool>>,
    forbidden: Option<Vec<bool>>,
    pinned: Option<Vec<(usize, f64)>>,
    tol: Option<f64>,
    maxit: u32,
    min_flow: f64,
    gas_cost: f64,
    partial_ok: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let arcs = Arcs { tau: &tau, sig: &sig, g: &g, eps: &eps, cap: &cap, n_nodes };
    let opt = Options {
        tol: tol.unwrap_or(crate::solve::TOL),
        maxit,
        min_flow,
        gas_cost,
        partial_ok,

    };
    let pins = pinned.unwrap_or_default();
    // The solve is pure arithmetic and can be long; let other Python threads
    // run while it does.
    let out = py.allow_threads(|| {
        active_set_solve(
            &arcs, src, dst, psi_total,
            a0.as_deref(), forbidden.as_deref(), &pins, &opt,
        )
    });

    let d = PyDict::new(py);
    d.set_item("psi", out.psi)?;
    d.set_item("u", out.u)?;
    d.set_item("active", out.active)?;
    d.set_item("upper", out.upper)?;
    d.set_item("psi_upper", out.psi_upper)?;
    d.set_item("rho", out.rho)?;
    d.set_item("pivots", out.pivots)?;
    d.set_item("feasible", out.stop.feasible())?;
    d.set_item("reason", out.stop.reason())?;
    Ok(d)
}

#[pymodule]
fn erouter_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add("__doc__", "The router's active-set QP, in Rust.")?;
    Ok(())
}
