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
use pyo3::types::{PyDict, PyTuple};

use crate::solve::{active_set_solve, Arcs, Options};

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (tau, sig, g, eps, cap, n_nodes, src, dst, psi_total,
                    a0=None, forbidden=None, pinned=None, tol=None, maxit=600,
                    min_flow=0.0, gas_cost=0.0, partial_ok=false, rank1=None))]
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
    rank1: Option<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let arcs = Arcs { tau: &tau, sig: &sig, g: &g, eps: &eps, cap: &cap, n_nodes };
    let opt = Options {
        tol: tol.unwrap_or(crate::solve::TOL),
        maxit,
        min_flow,
        gas_cost,
        partial_ok,
        rank1: rank1.unwrap_or(true),

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

    pack(py, out)
}



/// The graph, held on the Rust side across a quote's many solves.
///
/// A quote runs 45-106 solves over *the same* arcs -- only the warm start, the
/// forbidden mask and the pins change between them.  Marshalling `tau`, `sig`,
/// `G`, `eps` and `cap` on every call meant crossing the boundary with tens of
/// thousands of Python floats per quote to hand over data that had not
/// changed, which cost about what the Rust arithmetic saved.
///
/// Building the problem once turns that into one crossing, and leaves only the
/// per-solve masks -- which are small, and genuinely different each time.
#[pyclass]
pub struct Problem {
    tau: Vec<i64>,
    sig: Vec<i64>,
    g: Vec<f64>,
    eps: Vec<f64>,
    cap: Vec<f64>,
    n_nodes: usize,
}

#[pymethods]
impl Problem {
    #[new]
    fn new(tau: Vec<i64>, sig: Vec<i64>, g: Vec<f64>, eps: Vec<f64>,
           cap: Vec<f64>, n_nodes: usize) -> Self {
        Problem { tau, sig, g, eps, cap, n_nodes }
    }

    #[getter]
    fn m(&self) -> usize {
        self.tau.len()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, psi_total, a0=None, forbidden=None, pinned=None,
                        tol=None, maxit=600, min_flow=0.0, gas_cost=0.0,
                        partial_ok=false, rank1=None))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
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
        rank1: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let arcs = Arcs {
            tau: &self.tau, sig: &self.sig, g: &self.g,
            eps: &self.eps, cap: &self.cap, n_nodes: self.n_nodes,
        };
        let opt = Options {
            tol: tol.unwrap_or(crate::solve::TOL),
            maxit, min_flow, gas_cost, partial_ok,
            rank1: rank1.unwrap_or(true),
        };
        let pins = pinned.unwrap_or_default();
        let out = py.allow_threads(|| {
            active_set_solve(&arcs, src, dst, psi_total,
                             a0.as_deref(), forbidden.as_deref(), &pins, &opt)
        });
        pack(py, out)
    }
}

/// One arc's ladder, fitted.  Returns the fields of `core.calibrate.Calibration`
/// as a tuple, in declaration order, so the Python side builds the dataclass
/// and keeps its own postconditions.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (deltas, quotes, delta_bar=None, structural_flag=false,
                    drift_tol=crate::calibrate::DRIFT_TOL, cap=None,
                    f_at_cap=None, quantum=0.0))]
fn calibrate<'py>(
    py: Python<'py>,
    deltas: Vec<f64>,
    quotes: Vec<f64>,
    delta_bar: Option<f64>,
    structural_flag: bool,
    drift_tol: f64,
    cap: Option<f64>,
    f_at_cap: Option<f64>,
    quantum: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let got = crate::calibrate::calibrate(
        &deltas, &quotes, delta_bar, structural_flag, drift_tol, cap, f_at_cap, quantum,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?;
    PyTuple::new(
        py,
        [
            got.a.into_pyobject(py)?.into_any(),
            got.b.into_pyobject(py)?.into_any(),
            got.cap.into_pyobject(py)?.into_any(),
            got.clamped.into_pyobject(py)?.to_owned().into_any(),
            got.convex_flag.into_pyobject(py)?.to_owned().into_any(),
            got.flag.as_str().into_pyobject(py)?.into_any(),
            got.drift.into_pyobject(py)?.into_any(),
            got.eta.into_pyobject(py)?.into_any(),
            got.split_hint.into_pyobject(py)?.to_owned().into_any(),
            got.calib_delta.into_pyobject(py)?.into_any(),
            got.tangent_delta.into_pyobject(py)?.into_any(),
            got.note.into_pyobject(py)?.into_any(),
        ],
    )
}

fn pack<'py>(py: Python<'py>, out: crate::solve::Solution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("psi", out.psi)?;
    d.set_item("u", out.u)?;
    d.set_item("active", out.active)?;
    d.set_item("upper", out.upper)?;
    d.set_item("psi_upper", out.psi_upper)?;
    d.set_item("rho", out.rho)?;
    d.set_item("pivots", out.pivots)?;
    d.set_item("chol_failures", out.chol_failures)?;
    d.set_item("keep_changes", out.keep_changes)?;
    d.set_item("refits", out.refits)?;
    d.set_item("timings", out.timings.to_vec())?;
    d.set_item("feasible", out.stop.feasible())?;
    d.set_item("reason", out.stop.reason())?;
    Ok(d)
}

#[pymodule]
fn erouter_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_class::<Problem>()?;
    m.add("__doc__", "The router's active-set QP, in Rust.")?;
    Ok(())
}
