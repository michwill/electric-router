//! The reference frame, in the browser.
//!
//! The twin of `src/prices_py.rs`. `price_fit_weights` takes its keys as three
//! parallel arrays, because a tuple has no typed-array form.

use erouter_solve::prices;
use wasm_bindgen::prelude::*;

fn keys_of(pools: &[String], i: &[i32], j: &[i32]) -> Vec<(String, i32, i32)> {
    pools
        .iter()
        .zip(i.iter())
        .zip(j.iter())
        .map(|((p, &a), &b)| (p.clone(), a, b))
        .collect()
}

/// Mute arcs whose own reverse direction contradicts them.
#[wasm_bindgen(js_name = priceFitWeights)]
pub fn price_fit_weights(
    pools: Vec<String>, i: Vec<i32>, j: Vec<i32>, a: Vec<f64>, w: Vec<f64>,
) -> Result<Vec<f64>, JsValue> {
    if pools.len() != i.len() || pools.len() != j.len()
        || pools.len() != a.len() || pools.len() != w.len()
    {
        return Err(JsError::new("keys, a and w must be the same length").into());
    }
    Ok(prices::price_fit_weights(&keys_of(&pools, &i, &j), &a, &w))
}

/// Fit `nu` with `nu[numeraire] == 1`.
#[wasm_bindgen(js_name = referencePrices)]
pub fn reference_prices(
    tau: Vec<i64>, sig: Vec<i64>, a: Vec<f64>, w: Vec<f64>, n_nodes: usize,
    numeraire: usize,
) -> Result<Vec<f64>, JsValue> {
    prices::reference_prices(&tau, &sig, &a, &w, n_nodes, numeraire)
        .map_err(|e| JsError::new(&e.0).into())
}

/// Residuals `r_p = z_sig - z_tau + log a_p`.
#[wasm_bindgen]
pub fn dislocations(tau: Vec<i64>, sig: Vec<i64>, a: Vec<f64>, nu: Vec<f64>) -> Vec<f64> {
    prices::dislocations(&tau, &sig, &a, &nu)
}

/// Fee-free mid price implied by the two one-sided quotes.
#[wasm_bindgen(js_name = poolMid)]
pub fn pool_mid(a_forward: f64, a_reverse: f64) -> f64 {
    prices::pool_mid(a_forward, a_reverse)
}

/// Measured effective retention, `sqrt(a_f * a_r)` (§2.6).
#[wasm_bindgen(js_name = gammaLive)]
pub fn gamma_live(a_forward: Vec<f64>, a_reverse: Vec<f64>) -> Vec<f64> {
    a_forward
        .iter()
        .zip(a_reverse.iter())
        .map(|(&f, &r)| prices::gamma_live(f, r))
        .collect()
}

/// Indices where `eps_f + eps_r <= tol` -- a spurious negative 2-cycle.
#[wasm_bindgen(js_name = checkPairDrops)]
pub fn check_pair_drops(
    eps_forward: Vec<f64>, eps_reverse: Vec<f64>, tol: Option<f64>,
) -> Vec<u32> {
    prices::check_pair_drops(&eps_forward, &eps_reverse, tol.unwrap_or(0.0))
        .into_iter()
        .map(|k| k as u32)
        .collect()
}
