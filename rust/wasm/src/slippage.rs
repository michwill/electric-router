//! Dividing a slippage budget, in the browser.
//!
//! The twin of `src/slippage_py.rs`. `drops` returns an empty array where the
//! reference returns `None`: the network not solving is a fallback the caller
//! handles, not an error.

use crate::realize::Route;
use erouter_solve::slippage;
use wasm_bindgen::prelude::*;

/// What `zip(..., strict=True)` raises in the reference, thrown at JS.
///
/// The twin of `same_length` in `src/slippage_py.rs`, and it has to say the
/// same thing: a caller who drives this module instead of the extension is
/// entitled to the same refusal, not a quietly truncated answer.
fn same_length(what: &str, got: usize, want: usize) -> Result<(), JsValue> {
    if got == want {
        return Ok(());
    }
    Err(JsError::new(&format!("{what} has {got} value(s) for {want} leg(s)")).into())
}

/// The potential each leg drops, or an empty array if the network will not
/// solve -- `dropsSolved` says which.
#[wasm_bindgen]
pub fn drops(
    route: &Route, resistance: Vec<f64>, total: f64,
) -> Result<Vec<f64>, JsValue> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    Ok(slippage::drops(&route.inner, &resistance, total).unwrap_or_default())
}

#[wasm_bindgen(js_name = dropsSolved)]
pub fn drops_solved(
    route: &Route, resistance: Vec<f64>, total: f64,
) -> Result<bool, JsValue> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    Ok(slippage::drops(&route.inner, &resistance, total).is_some())
}

/// The most any one path spends.
#[wasm_bindgen]
pub fn longest(route: &Route, spend: Vec<f64>) -> Result<f64, JsValue> {
    same_length("spend", spend.len(), route.inner.legs.len())?;
    Ok(slippage::longest(&route.inner, &spend))
}

/// What each leg is owed however the rest of the route is scaled.
#[wasm_bindgen]
pub fn backstops(
    raw: Vec<f64>, floor: Option<Vec<f64>>,
) -> Result<Vec<f64>, JsValue> {
    if let Some(given) = floor.as_deref() {
        same_length("floor", given.len(), raw.len())?;
    }
    Ok(slippage::backstops(&raw, floor.as_deref()))
}

/// Split `total` between the legs, as fractions rather than basis points.
#[wasm_bindgen]
pub fn divide(
    route: &Route, resistance: Vec<f64>, total: f64, backstop: Option<Vec<f64>>,
) -> Result<Vec<f64>, JsValue> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    if let Some(given) = backstop.as_deref() {
        same_length("backstop", given.len(), route.inner.legs.len())?;
    }
    slippage::divide(&route.inner, &resistance, total, backstop.as_deref())
        .map_err(|e| JsError::new(&e).into())
}

/// Raise a leg the network runs backwards to `floor`, leaving the rest.
#[wasm_bindgen]
pub fn widen(
    route: &Route, resistance: Vec<f64>, total: f64, spend: Vec<f64>, floor: f64,
) -> Result<Vec<f64>, JsValue> {
    same_length("resistance", resistance.len(), route.inner.legs.len())?;
    same_length("spend", spend.len(), route.inner.legs.len())?;
    Ok(slippage::widen(&route.inner, &resistance, total, &spend, floor))
}
