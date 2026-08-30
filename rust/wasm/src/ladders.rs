//! The resident ladders' exports.
//!
//! One for one with `rust/src/ladders_py.rs`, for the reason every pair here
//! shares: the shim that stands in for `erouter_solve` in the browser presents
//! this surface to `core/pipeline.py`, which is not allowed to know which half
//! answered.
//!
//! `u128` has no typed array, so deltas cross as low and high halves
//! interleaved in a `BigUint64Array` -- the same shape `Pools::price` uses, and
//! for the same reason: a probe size past 1.8e19 wei is eighteen tokens at
//! eighteen decimals, which is a rounding error rather than a trade.

use erouter_solve::ladders::{Ladders as Inner, Meta, Plan};
use wasm_bindgen::prelude::*;

/// What is still to ask for: a slot and a size per probe.
#[wasm_bindgen]
pub struct PlanOut {
    slot: Vec<u32>,
    delta: Vec<u64>,
}

#[wasm_bindgen]
impl PlanOut {
    #[wasm_bindgen(getter)]
    pub fn slot(&self) -> Vec<u32> {
        self.slot.clone()
    }

    /// Low and high halves of each `u128`, interleaved.
    #[wasm_bindgen(getter)]
    pub fn delta(&self) -> Vec<u64> {
        self.delta.clone()
    }
}

/// One arc's fit, in the twelve-field order the Python side rebuilds from.
#[wasm_bindgen]
pub struct FitOut {
    a: f64,
    b: f64,
    cap: f64,
    clamped: bool,
    convex_flag: bool,
    flag: String,
    drift: f64,
    eta: f64,
    calib_delta: f64,
}

#[wasm_bindgen]
impl FitOut {
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 { self.a }
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 { self.b }
    #[wasm_bindgen(getter)]
    pub fn cap(&self) -> f64 { self.cap }
    #[wasm_bindgen(getter)]
    pub fn clamped(&self) -> bool { self.clamped }
    #[wasm_bindgen(getter, js_name = convexFlag)]
    pub fn convex_flag(&self) -> bool { self.convex_flag }
    #[wasm_bindgen(getter)]
    pub fn flag(&self) -> String { self.flag.clone() }
    #[wasm_bindgen(getter)]
    pub fn drift(&self) -> f64 { self.drift }
    #[wasm_bindgen(getter)]
    pub fn eta(&self) -> f64 { self.eta }
    #[wasm_bindgen(getter, js_name = calibDelta)]
    pub fn calib_delta(&self) -> f64 { self.calib_delta }
}

fn halves(v: u128) -> (u64, u64) {
    (v as u64, (v >> 64) as u64)
}

fn whole(lo: u64, hi: u64) -> u128 {
    u128::from(lo) | (u128::from(hi) << 64)
}

/// The ladders a quote refines, held here for the whole stage.
#[wasm_bindgen]
pub struct Ladders {
    inner: Inner,
}

#[wasm_bindgen]
impl Ladders {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Inner::new() }
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Register one arc's coarse ladder. Returns its slot.
    pub fn add(&mut self, decimals_in: u32, decimals_out: u32, reserve_in_lo: u64,
               reserve_in_hi: u64, deltas: &[u64], quotes: &[u64], attempted: u32)
        -> Result<usize, JsError> {
        if deltas.len() != quotes.len() || deltas.len() % 2 != 0 {
            return Err(JsError::new(
                "deltas and quotes must be the same even length: lo/hi pairs"));
        }
        let d: Vec<u128> = deltas.chunks(2).map(|c| whole(c[0], c[1])).collect();
        let q: Vec<u128> = quotes.chunks(2).map(|c| whole(c[0], c[1])).collect();
        Ok(self.inner.add(
            Meta { decimals_in, decimals_out,
                   reserve_in: whole(reserve_in_lo, reserve_in_hi) },
            d, q, attempted))
    }

    /// A quote's own copy of the warm ladders.
    pub fn fork(&self) -> Ladders {
        Ladders { inner: self.inner.fork() }
    }

    /// What is still missing. `want` is lo/hi pairs; `spans` bounds each slot's
    /// sizes within it, so a ragged input crosses as two arrays.
    #[wasm_bindgen(js_name = planSized)]
    pub fn plan_sized(&self, slots: &[u32], want: &[u64], spans: &[u32])
        -> Result<PlanOut, JsError> {
        if spans.len() != slots.len() + 1 {
            return Err(JsError::new("spans must bound every slot: len(slots) + 1"));
        }
        let sizes: Vec<u128> = want.chunks(2).map(|c| whole(c[0], c[1])).collect();
        let halved: Vec<u32> = spans.iter().map(|v| v / 2).collect();
        let plan = self.inner.plan_sized(slots, &sizes, &halved);
        let mut delta = Vec::with_capacity(plan.delta.len() * 2);
        for v in &plan.delta {
            let (lo, hi) = halves(*v);
            delta.push(lo);
            delta.push(hi);
        }
        Ok(PlanOut { slot: plan.slot, delta })
    }

    /// Fold what the chain answered into the ladders.
    ///
    /// `status` is 0 for a value and otherwise a one-based index into `names`,
    /// so a refusal keeps its name without a string per probe.
    pub fn absorb(&mut self, slots: Vec<u32>, deltas: &[u64], values: &[u64],
                  status: &[u8], names: Vec<String>) -> Result<(), JsError> {
        let d: Vec<u128> = deltas.chunks(2).map(|c| whole(c[0], c[1])).collect();
        let v: Vec<u128> = values.chunks(2).map(|c| whole(c[0], c[1])).collect();
        let plan = Plan { slot: slots, delta: d };
        self.inner.absorb(&plan, &v, status, &names).map_err(|e| JsError::new(&e))
    }

    /// Fit the named ladders, in the caller's order. A slot whose ladder is too
    /// short, or whose fit refused, answers `undefined`.
    pub fn recalibrate(&self, slots: &[u32], drift_tol: f64) -> Vec<JsValue> {
        self.inner.recalibrate(slots, drift_tol).into_iter().map(|got| {
            match got {
                None => JsValue::UNDEFINED,
                Some(f) => JsValue::from(FitOut {
                    a: f.a, b: f.b, cap: f.cap, clamped: f.clamped,
                    convex_flag: f.convex_flag, flag: f.flag.as_str().to_string(),
                    drift: f.drift, eta: f.eta, calib_delta: f.calib_delta,
                }),
            }
        }).collect()
    }

    /// One ladder's measurements, as lo/hi pairs.
    pub fn points(&self, slot: usize) -> Vec<u64> {
        let deltas = self.inner.deltas_of(slot).unwrap_or(&[]);
        let quotes = self.inner.quotes_of(slot).unwrap_or(&[]);
        let mut out = Vec::with_capacity((deltas.len() + quotes.len()) * 2);
        for v in deltas.iter().chain(quotes.iter()) {
            let (lo, hi) = halves(*v);
            out.push(lo);
            out.push(hi);
        }
        out
    }

    pub fn attempted(&self, slot: usize) -> u32 {
        self.inner.attempted_of(slot)
    }

    /// One ladder's refusals, as `name` then `count` in one flat array.
    ///
    /// Flat rather than an object because a `HashMap` has no order and the
    /// caller only ever reads it to print; two arrays would cross twice.
    pub fn failures(&self, slot: usize) -> Vec<JsValue> {
        let mut out = Vec::new();
        if let Some(got) = self.inner.failures_of(slot) {
            let mut pairs: Vec<(&String, &u32)> = got.iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            for (name, count) in pairs {
                out.push(JsValue::from_str(name));
                out.push(JsValue::from_f64(f64::from(*count)));
            }
        }
        out
    }
}

impl Default for Ladders {
    fn default() -> Self {
        Self::new()
    }
}
