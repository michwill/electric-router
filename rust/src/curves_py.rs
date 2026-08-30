//! Sampled leg curves, across the PyO3 boundary.
//!
//! The curve itself stays on the Rust side once built: the split search calls
//! `at` millions of times, and `split.rs` exists because a crossing per call
//! would have lost to the arithmetic.

use crate::curves::{self, Curve as Inner};
use pyo3::prelude::*;

fn err(e: curves::CurveError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

/// One leg's output as a function of its input, through sampled points.
#[pyclass]
pub struct Curve {
    pub(crate) inner: Inner,
}

#[pymethods]
impl Curve {
    /// Build the interpolant of `x / f(x)` through the probes.
    #[staticmethod]
    fn fit(deltas: Vec<f64>, quotes: Vec<f64>) -> PyResult<Curve> {
        Ok(Curve { inner: curves::fit(&deltas, &quotes).map_err(err)? })
    }

    /// `f(x) = rate * x`, for a leg that is a conversion rather than a trade.
    #[staticmethod]
    fn linear(rate: f64) -> PyResult<Curve> {
        Ok(Curve { inner: curves::linear(rate).map_err(err)? })
    }

    /// Log-spaced integer probe sizes up to `top`, strictly increasing.
    #[staticmethod]
    #[pyo3(signature = (top, nodes=curves::NODES, span=curves::SPAN))]
    fn sizes(top: f64, nodes: usize, span: f64) -> Vec<u64> {
        curves::sizes(top, nodes, span)
    }

    /// Scalar evaluation.
    fn at(&self, v: f64) -> f64 {
        self.inner.at(v)
    }

    /// The same, over a batch -- one crossing rather than one per point.
    fn many(&self, v: Vec<f64>) -> Vec<f64> {
        v.into_iter().map(|one| self.inner.at(one)).collect()
    }

    /// Estimated interpolation error at `v`, in basis points, from the data.
    fn error_bp_at(&self, v: f64) -> f64 {
        self.inner.error_bp_at(v)
    }

    #[getter]
    fn top(&self) -> f64 {
        self.inner.top()
    }

    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    #[getter]
    fn u(&self) -> Vec<f64> {
        self.inner.u.clone()
    }

    #[getter]
    fn slope(&self) -> Vec<f64> {
        self.inner.slope.clone()
    }

    #[getter]
    fn rate0(&self) -> f64 {
        self.inner.rate0
    }

    #[getter]
    fn tail(&self) -> f64 {
        self.inner.tail
    }
}
