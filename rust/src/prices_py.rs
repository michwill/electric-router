//! The reference frame, across the PyO3 boundary.
//!
//! One crossing per quote: `nu` is fitted once and then read by every stage.

use crate::prices;
use pyo3::prelude::*;

fn err(e: prices::PriceError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

/// Mute arcs whose own reverse direction contradicts them.
///
/// `keys[k]` is `(pool, i, j)`; the partner is `(pool, j, i)`. Pairing on node
/// indices instead is wrong and quietly so.
#[pyfunction]
pub fn price_fit_weights(
    keys: Vec<(String, i32, i32)>, a: Vec<f64>, w: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if keys.len() != a.len() || keys.len() != w.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys, a and w must be the same length"));
    }
    Ok(prices::price_fit_weights(&keys, &a, &w))
}

/// Fit `nu` with `nu[numeraire] == 1`.
#[pyfunction]
pub fn reference_prices(
    tau: Vec<i64>, sig: Vec<i64>, a: Vec<f64>, w: Vec<f64>, n_nodes: usize,
    numeraire: usize,
) -> PyResult<Vec<f64>> {
    prices::reference_prices(&tau, &sig, &a, &w, n_nodes, numeraire).map_err(err)
}

/// Residuals `r_p = z_sig - z_tau + log a_p`.
#[pyfunction]
pub fn dislocations(
    tau: Vec<i64>, sig: Vec<i64>, a: Vec<f64>, nu: Vec<f64>,
) -> Vec<f64> {
    prices::dislocations(&tau, &sig, &a, &nu)
}

/// Fee-free mid price implied by the two one-sided quotes.
#[pyfunction]
pub fn pool_mid(a_forward: f64, a_reverse: f64) -> f64 {
    prices::pool_mid(a_forward, a_reverse)
}

/// Measured effective retention, `sqrt(a_f * a_r)` (§2.6).
#[pyfunction]
pub fn gamma_live(a_forward: Vec<f64>, a_reverse: Vec<f64>) -> Vec<f64> {
    a_forward
        .iter()
        .zip(a_reverse.iter())
        .map(|(&f, &r)| prices::gamma_live(f, r))
        .collect()
}

/// Indices where `eps_f + eps_r <= tol` -- a spurious negative 2-cycle.
#[pyfunction]
#[pyo3(signature = (eps_forward, eps_reverse, tol=0.0))]
pub fn check_pair_drops(
    eps_forward: Vec<f64>, eps_reverse: Vec<f64>, tol: f64,
) -> Vec<usize> {
    prices::check_pair_drops(&eps_forward, &eps_reverse, tol)
}
