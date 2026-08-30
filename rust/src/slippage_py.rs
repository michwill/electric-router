//! Dividing a slippage budget, across the PyO3 boundary.
//!
//! Every entry point takes the realised route, because the network the budget
//! is spent across *is* the leg list.

use crate::realize_py::Route;
use crate::slippage;
use pyo3::prelude::*;

/// The potential each leg drops, or `None` if the network will not solve.
#[pyfunction]
pub fn drops(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64,
) -> Option<Vec<f64>> {
    slippage::drops(&route.inner, &resistance, total)
}

/// The most any one path spends.
#[pyfunction]
pub fn longest(route: PyRef<'_, Route>, spend: Vec<f64>) -> f64 {
    slippage::longest(&route.inner, &spend)
}

/// What each leg is owed however the rest of the route is scaled.
#[pyfunction]
#[pyo3(signature = (raw, floor=None))]
pub fn backstops(raw: Vec<f64>, floor: Option<Vec<f64>>) -> Vec<f64> {
    slippage::backstops(&raw, floor.as_deref())
}

/// Split `total` between the legs, as fractions rather than basis points.
#[pyfunction]
#[pyo3(signature = (route, resistance, total, backstop=None))]
pub fn divide(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64,
    backstop: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    slippage::divide(&route.inner, &resistance, total, backstop.as_deref())
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Raise a leg the network runs backwards to `floor`, leaving the rest.
#[pyfunction]
pub fn widen(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64, spend: Vec<f64>,
    floor: f64,
) -> Vec<f64> {
    slippage::widen(&route.inner, &resistance, total, &spend, floor)
}
