//! Dividing a slippage budget, across the PyO3 boundary.
//!
//! Every entry point takes the realised route, because the network the budget
//! is spent across *is* the leg list.

use crate::realize_py::Route;
use crate::slippage;
use pyo3::prelude::*;

use crate::py::same_length;

/// The potential each leg drops, or `None` if the network will not solve.
#[pyfunction]
pub fn drops(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64,
) -> PyResult<Option<Vec<f64>>> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    Ok(slippage::drops(&route.inner, &resistance, total))
}

/// The most any one path spends.
#[pyfunction]
pub fn longest(route: PyRef<'_, Route>, spend: Vec<f64>) -> PyResult<f64> {
    same_length("spend", spend.len(), route.inner.legs.len())?;
    Ok(slippage::longest(&route.inner, &spend))
}

/// What each leg is owed however the rest of the route is scaled.
#[pyfunction]
#[pyo3(signature = (raw, floor=None))]
pub fn backstops(raw: Vec<f64>, floor: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    if let Some(given) = floor.as_deref() {
        // The reference indexes `floor[k]` over `raw`, so a short one is an
        // `IndexError` there and would be a panic here.
        same_length("floor", given.len(), raw.len())?;
    }
    Ok(slippage::backstops(&raw, floor.as_deref()))
}

/// Split `total` between the legs, as fractions rather than basis points.
#[pyfunction]
#[pyo3(signature = (route, resistance, total, backstop=None))]
pub fn divide(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64,
    backstop: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    if let Some(given) = backstop.as_deref() {
        same_length("backstop", given.len(), route.inner.legs.len())?;
    }
    slippage::divide(&route.inner, &resistance, total, backstop.as_deref())
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Raise a leg the network runs backwards to `floor`, leaving the rest.
#[pyfunction]
pub fn widen(
    route: PyRef<'_, Route>, resistance: Vec<f64>, total: f64, spend: Vec<f64>,
    floor: f64,
) -> PyResult<Vec<f64>> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    same_length("spend", spend.len(), route.inner.legs.len())?;
    Ok(slippage::widen(&route.inner, &resistance, total, &spend, floor))
}
