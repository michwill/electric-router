//! Calldata, in the browser.
//!
//! The twin of `src/routecall_py.rs`. This is the last step: a browser that
//! reaches here has a transaction to sign without Python having run at all.

use crate::realize::Route;
use erouter_solve::routecall::{self, EncodeStr, Policy, RouteCall as Inner};
use wasm_bindgen::prelude::*;

fn err(e: routecall::EncodingError) -> JsValue {
    JsError::new(&e.0).into()
}

/// A ready-to-send call, and what it is and is not protecting.
#[wasm_bindgen]
pub struct RouteCall {
    inner: Inner,
}

#[wasm_bindgen]
impl RouteCall {
    /// Encode `route` for `ElectricRouter.execute`.
    #[wasm_bindgen(js_name = encodeRoute)]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_route(
        route: &Route, receiver: &str, set_approvals: Option<bool>,
        min_out: Option<String>, amount_in: Option<String>,
        quoted_out: Option<String>, volatile: Option<Vec<String>>,
        naming: Option<String>, allow_unbounded: Option<bool>,
        fee_share: Option<f64>, floor_bp: Option<f64>,
        volatile_floor_bp: Option<f64>, slippage_bp: Option<f64>,
    ) -> Result<RouteCall, JsValue> {
        let loose = volatile.unwrap_or_default();
        let naming = naming.unwrap_or_else(|| "needed".to_string());
        let min_out = min_out.unwrap_or_else(|| "0".to_string());
        let opts = EncodeStr {
            receiver,
            set_approvals: set_approvals.unwrap_or(true),
            min_out: &min_out,
            amount_in: amount_in.as_deref(),
            quoted_out: quoted_out.as_deref(),
            naming: &naming,
            allow_unbounded: allow_unbounded.unwrap_or(false),
            volatile: &loose,
            fee_share: fee_share.unwrap_or(routecall::FEE_SHARE),
            floor_bp: floor_bp.unwrap_or(routecall::FLOOR_BP),
            volatile_floor_bp: volatile_floor_bp.unwrap_or(routecall::VOLATILE_FLOOR_BP),
            slippage_bp,
        };
        Ok(RouteCall {
            inner: routecall::encode_route_str(&route.inner, &opts).map_err(err)?,
        })
    }

    /// The shortest entry point that still expresses this call.
    pub fn calldata(&self, sender: Option<String>) -> Result<Vec<u8>, JsValue> {
        self.inner
            .calldata(sender.as_deref().unwrap_or(""))
            .map_err(err)
    }

    #[wasm_bindgen(getter, js_name = amountIn)]
    pub fn amount_in(&self) -> String {
        self.inner.amount_in_str()
    }

    #[wasm_bindgen(getter)]
    pub fn pools(&self) -> Vec<String> {
        self.inner.pools.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Vec<String> {
        self.inner.params_str()
    }

    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn receiver(&self) -> String {
        self.inner.receiver.clone()
    }

    #[wasm_bindgen(getter, js_name = minOut)]
    pub fn min_out(&self) -> String {
        self.inner.min_out_str()
    }

    #[wasm_bindgen(getter, js_name = tokenIn)]
    pub fn token_in(&self) -> String {
        self.inner.token_in.clone()
    }

    #[wasm_bindgen(getter, js_name = tokenOut)]
    pub fn token_out(&self) -> String {
        self.inner.token_out.clone()
    }

    #[wasm_bindgen(getter, js_name = guaranteedOut)]
    pub fn guaranteed_out(&self) -> String {
        self.inner.guaranteed_out_str()
    }

    #[wasm_bindgen(getter, js_name = quotedOut)]
    pub fn quoted_out(&self) -> String {
        self.inner.quoted_out_str()
    }

    #[wasm_bindgen(getter)]
    pub fn unbounded(&self) -> Vec<u32> {
        self.inner.unbounded.iter().map(|&k| k as u32).collect()
    }

    #[wasm_bindgen(getter, js_name = toleranceBp)]
    pub fn tolerance_bp(&self) -> f64 {
        self.inner.tolerance_bp()
    }

    /// The pool each step names; the numbers are in `stepNumbers` and the two
    /// 256-bit fields in `stepFractions` and `stepRates`.
    pub fn steps(&self) -> Result<Vec<String>, JsValue> {
        Ok(self.inner.steps_str().map_err(err)?.into_iter().map(|s| s.0).collect())
    }

    #[wasm_bindgen(js_name = stepNumbers)]
    pub fn step_numbers(&self) -> Result<Vec<i32>, JsValue> {
        Ok(self
            .inner
            .steps_str()
            .map_err(err)?
            .into_iter()
            .flat_map(|s| [s.1 as i32, s.2, s.3, s.4, s.7 as i32, s.8 as i32])
            .collect())
    }

    #[wasm_bindgen(js_name = stepFractions)]
    pub fn step_fractions(&self) -> Result<Vec<String>, JsValue> {
        Ok(self.inner.steps_str().map_err(err)?.into_iter().map(|s| s.5).collect())
    }

    #[wasm_bindgen(js_name = stepRates)]
    pub fn step_rates(&self) -> Result<Vec<String>, JsValue> {
        Ok(self.inner.steps_str().map_err(err)?.into_iter().map(|s| s.6).collect())
    }
}

/// Each leg's share of the balance standing at its source when it runs.
#[wasm_bindgen]
pub fn fractions(route: &Route) -> Result<Vec<String>, JsValue> {
    Ok(routecall::fractions(&route.inner)
        .map_err(err)?
        .iter()
        .map(|v| v.to_string())
        .collect())
}

/// The minimum rate per leg; `minRatesUnbounded` says which bound nothing.
#[wasm_bindgen(js_name = minRates)]
pub fn min_rates(
    route: &Route, volatile: Option<Vec<String>>, fee_share: Option<f64>,
    floor_bp: Option<f64>, volatile_floor_bp: Option<f64>, slippage_bp: Option<f64>,
) -> Result<Vec<String>, JsValue> {
    Ok(rates_of(route, volatile, fee_share, floor_bp, volatile_floor_bp, slippage_bp)?.0)
}

#[wasm_bindgen(js_name = minRatesUnbounded)]
pub fn min_rates_unbounded(
    route: &Route, volatile: Option<Vec<String>>, fee_share: Option<f64>,
    floor_bp: Option<f64>, volatile_floor_bp: Option<f64>, slippage_bp: Option<f64>,
) -> Result<Vec<u32>, JsValue> {
    Ok(rates_of(route, volatile, fee_share, floor_bp, volatile_floor_bp, slippage_bp)?
        .1
        .iter()
        .map(|&k| k as u32)
        .collect())
}

fn rates_of(
    route: &Route, volatile: Option<Vec<String>>, fee_share: Option<f64>,
    floor_bp: Option<f64>, volatile_floor_bp: Option<f64>, slippage_bp: Option<f64>,
) -> Result<(Vec<String>, Vec<usize>), JsValue> {
    let loose = volatile.unwrap_or_default();
    let policy = Policy {
        volatile: &loose,
        fee_share: fee_share.unwrap_or(routecall::FEE_SHARE),
        floor_bp: floor_bp.unwrap_or(routecall::FLOOR_BP),
        volatile_floor_bp: volatile_floor_bp.unwrap_or(routecall::VOLATILE_FLOOR_BP),
        slippage_bp,
    };
    routecall::min_rates_str(&route.inner, &policy).map_err(err)
}

/// How far below its quote the automatic rule lets each leg land.
#[wasm_bindgen]
pub fn tolerances(
    route: &Route, volatile: Option<Vec<String>>, fee_share: Option<f64>,
    floor_bp: Option<f64>, volatile_floor_bp: Option<f64>,
) -> Vec<f64> {
    let loose = volatile.unwrap_or_default();
    routecall::tolerances(
        &route.inner, &loose,
        fee_share.unwrap_or(routecall::FEE_SHARE),
        floor_bp.unwrap_or(routecall::FLOOR_BP),
        volatile_floor_bp.unwrap_or(routecall::VOLATILE_FLOOR_BP),
    )
}

/// The room each leg needs for what moves under it, whatever it charges.
#[wasm_bindgen(js_name = movementFloors)]
pub fn movement_floors(
    route: &Route, volatile: Option<Vec<String>>, floor_bp: Option<f64>,
    volatile_floor_bp: Option<f64>,
) -> Vec<f64> {
    let loose = volatile.unwrap_or_default();
    routecall::movement_floors(
        &route.inner, &loose,
        floor_bp.unwrap_or(routecall::FLOOR_BP),
        volatile_floor_bp.unwrap_or(routecall::VOLATILE_FLOOR_BP),
    )
}

/// What the bounds promise; `walkBoundsFloors` is the per-leg minimum.
#[wasm_bindgen(js_name = walkBounds)]
pub fn walk_bounds(
    route: &Route, fracs: Vec<String>, rates: Vec<String>,
) -> Result<String, JsValue> {
    Ok(walked(route, fracs, rates)?.0)
}

#[wasm_bindgen(js_name = walkBoundsFloors)]
pub fn walk_bounds_floors(
    route: &Route, fracs: Vec<String>, rates: Vec<String>,
) -> Result<Vec<String>, JsValue> {
    Ok(walked(route, fracs, rates)?.1)
}

fn walked(
    route: &Route, fracs: Vec<String>, rates: Vec<String>,
) -> Result<(String, Vec<String>), JsValue> {
    routecall::walk_bounds_str(&route.inner, &fracs, &rates).map_err(err)
}

/// One packed leg, as a word.
#[wasm_bindgen(js_name = packStep)]
#[allow(clippy::too_many_arguments)]
pub fn pack_step(
    kind: u8, i: i32, j: i32, n: i32, frac: &str, min_rate: &str, in_ref: usize,
    out_ref: usize,
) -> Result<String, JsValue> {
    routecall::pack_step_str(kind, i, j, n, frac, min_rate, in_ref, out_ref)
        .map_err(err)
}

/// The inverse, refusing a word with reserved bits set. Returns
/// `[kind, i, j, n, in_ref, out_ref]`; the two wide fields are in
/// `unpackStepFrac` and `unpackStepRate`.
#[wasm_bindgen(js_name = unpackStep)]
pub fn unpack_step(word: &str) -> Result<Vec<i32>, JsValue> {
    let s = routecall::unpack_step_str(word).map_err(err)?;
    Ok(vec![s.0 as i32, s.1, s.2, s.3, s.6 as i32, s.7 as i32])
}

#[wasm_bindgen(js_name = unpackStepFrac)]
pub fn unpack_step_frac(word: &str) -> Result<String, JsValue> {
    Ok(routecall::unpack_step_str(word).map_err(err)?.4)
}

#[wasm_bindgen(js_name = unpackStepRate)]
pub fn unpack_step_rate(word: &str) -> Result<String, JsValue> {
    Ok(routecall::unpack_step_str(word).map_err(err)?.5)
}
