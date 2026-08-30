//! Dividing a slippage budget, in the browser.
//!
//! The twin of `src/slippage_py.rs`. `drops` returns an empty array where the
//! reference returns `None`: the network not solving is a fallback the caller
//! handles, not an error.

use crate::realize::Route;
use erouter_solve::slippage;
use wasm_bindgen::prelude::*;

/// The potential each leg drops, or an empty array if the network will not
/// solve -- `dropsSolved` says which.
#[wasm_bindgen]
pub fn drops(route: &Route, resistance: Vec<f64>, total: f64) -> Vec<f64> {
    slippage::drops(&route.inner, &resistance, total).unwrap_or_default()
}

#[wasm_bindgen(js_name = dropsSolved)]
pub fn drops_solved(route: &Route, resistance: Vec<f64>, total: f64) -> bool {
    slippage::drops(&route.inner, &resistance, total).is_some()
}

/// The most any one path spends.
#[wasm_bindgen]
pub fn longest(route: &Route, spend: Vec<f64>) -> f64 {
    slippage::longest(&route.inner, &spend)
}

/// What each leg is owed however the rest of the route is scaled.
#[wasm_bindgen]
pub fn backstops(raw: Vec<f64>, floor: Option<Vec<f64>>) -> Vec<f64> {
    slippage::backstops(&raw, floor.as_deref())
}

/// Split `total` between the legs, as fractions rather than basis points.
#[wasm_bindgen]
pub fn divide(
    route: &Route, resistance: Vec<f64>, total: f64, backstop: Option<Vec<f64>>,
) -> Result<Vec<f64>, JsValue> {
    slippage::divide(&route.inner, &resistance, total, backstop.as_deref())
        .map_err(|e| JsError::new(&e).into())
}

/// Raise a leg the network runs backwards to `floor`, leaving the rest.
#[wasm_bindgen]
pub fn widen(
    route: &Route, resistance: Vec<f64>, total: f64, spend: Vec<f64>, floor: f64,
) -> Vec<f64> {
    slippage::widen(&route.inner, &resistance, total, &spend, floor)
}
