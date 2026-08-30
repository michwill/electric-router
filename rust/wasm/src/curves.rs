//! Sampled leg curves, in the browser.
//!
//! The twin of `src/curves_py.rs`. The curve stays on the Rust side once
//! built, for the reason `split.rs` gives: a crossing per evaluation would
//! have lost to the arithmetic.

use erouter_solve::curves::{self, Curve as Inner};
use wasm_bindgen::prelude::*;

fn err(e: curves::CurveError) -> JsValue {
    JsError::new(&e.0).into()
}

/// One leg's output as a function of its input, through sampled points.
#[wasm_bindgen]
pub struct Curve {
    inner: Inner,
}

#[wasm_bindgen]
impl Curve {
    /// Build the interpolant of `x / f(x)` through the probes.
    pub fn fit(deltas: Vec<f64>, quotes: Vec<f64>) -> Result<Curve, JsValue> {
        Ok(Curve { inner: curves::fit(&deltas, &quotes).map_err(err)? })
    }

    /// `f(x) = rate * x`, for a leg that is a conversion rather than a trade.
    pub fn linear(rate: f64) -> Result<Curve, JsValue> {
        Ok(Curve { inner: curves::linear(rate).map_err(err)? })
    }

    /// Log-spaced integer probe sizes up to `top`, strictly increasing.
    pub fn sizes(top: f64, nodes: Option<usize>, span: Option<f64>) -> Vec<u64> {
        curves::sizes(top, nodes.unwrap_or(curves::NODES), span.unwrap_or(curves::SPAN))
    }

    /// Scalar evaluation.
    pub fn at(&self, v: f64) -> f64 {
        self.inner.at(v)
    }

    /// The same, over a batch -- one crossing rather than one per point.
    pub fn many(&self, v: Vec<f64>) -> Vec<f64> {
        v.into_iter().map(|one| self.inner.at(one)).collect()
    }

    /// Estimated interpolation error at `v`, in basis points, from the data.
    #[wasm_bindgen(js_name = errorBpAt)]
    pub fn error_bp_at(&self, v: f64) -> f64 {
        self.inner.error_bp_at(v)
    }

    #[wasm_bindgen(getter)]
    pub fn top(&self) -> f64 {
        self.inner.top()
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn u(&self) -> Vec<f64> {
        self.inner.u.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn slope(&self) -> Vec<f64> {
        self.inner.slope.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rate0(&self) -> f64 {
        self.inner.rate0
    }

    #[wasm_bindgen(getter)]
    pub fn tail(&self) -> f64 {
        self.inner.tail
    }
}
