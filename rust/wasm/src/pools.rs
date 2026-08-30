//! The pool models' exports.
//!
//! One for one with `rust/src/pools/py.rs`, and for the same reason the
//! solver's exports are: the shim that stands in for `erouter_solve` in the
//! browser presents this surface to `chain/exact_probe.py`, which is not
//! allowed to know which half answered.
//!
//! Neither binding prices anything. Both wrap `pools::registry::Registry`, so
//! the arithmetic, the legacy flags and the refusals have one definition; a
//! second hand-written copy would drift at the first flag, and the flags are
//! where two deployed generations differ in the last wei.
//!
//! Amounts cross as `BigInt`, which is what a `u128` is on this side, because
//! a balance does not survive an `f64` and does not fit a `u64` either: 1.8e19
//! wei is eighteen tokens at eighteen decimals, which is not a trade size but
//! a rounding error. The batch answers into two buffers -- values and a mask --
//! rather than an array of nullable numbers: the same "plain buffers, no
//! serde" boundary the solver uses, and it keeps a refusal from costing an
//! object per probe.

use erouter_solve::pools::registry::Registry;
use wasm_bindgen::prelude::*;

/// The general element search's answer: the two shares and what they pay.
#[wasm_bindgen]
pub struct BestSplitOut {
    #[wasm_bindgen(readonly)]
    pub a: i64,
    #[wasm_bindgen(readonly)]
    pub b: i64,
    #[wasm_bindgen(readonly)]
    pub payout: f64,
}

/// Ports as the registry takes them, from the two arrays a typed array can
/// carry.
fn pairs(coins: &[i32], bps: &[i64]) -> Result<Vec<(i32, i64)>, JsValue> {
    if coins.len() != bps.len() {
        return Err(JsError::new("each side needs one share per coin").into());
    }
    Ok(coins.iter().zip(bps.iter()).map(|(&c, &b)| (c, b)).collect())
}

/// A batch's answers: `values[k]` is meaningful where `ok[k]` is 1.
///
/// Getters rather than fields so the buffers are only materialised as JS
/// arrays when the caller asks for them.
#[wasm_bindgen]
pub struct PriceResult {
    values: Vec<u64>,
    ok: Vec<u8>,
}

#[wasm_bindgen]
impl PriceResult {
    /// The low and high halves of each `u128`, interleaved, so the whole batch
    /// crosses as one `BigUint64Array` rather than as `n` `BigInt`s. A caller
    /// that wants the number reassembles it as `lo | hi << 64n`.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<u64> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ok(&self) -> Vec<u8> {
        self.ok.clone()
    }
}

/// The best two-way split, in basis points of `dx`.
#[wasm_bindgen]
pub struct SplitOut {
    a: u16,
    b: u16,
}

#[wasm_bindgen]
impl SplitOut {
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> u16 {
        self.a
    }

    #[wasm_bindgen(getter)]
    pub fn b(&self) -> u16 {
        self.b
    }
}

/// The models a session has admitted, by index.
#[wasm_bindgen]
pub struct Pools {
    inner: Registry,
}

#[wasm_bindgen]
impl Pools {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Registry::new() }
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Add a stableswap, in both arithmetics. Returns its index.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = addStableswap)]
    pub fn add_stableswap(
        &mut self, balances: Vec<String>, rates: Vec<String>, amp: &str,
        fee: &str, offpeg_fee_multiplier: &str, a_precision: &str,
        fee_on_xp: bool, subtract_one: bool, admin_fee: Option<String>,
    ) -> Result<usize, JsError> {
        self.inner
            .add_stableswap(&balances, &rates, amp, fee, offpeg_fee_multiplier,
                            a_precision, fee_on_xp, subtract_one,
                            admin_fee.as_deref())
            .map_err(|e| JsError::new(&e))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = addTwocrypto)]
    pub fn add_twocrypto(
        &mut self, balances: Vec<String>, precisions: Vec<String>,
        price_scale: &str, d: &str, amp: &str, gamma: &str, mid_fee: &str,
        out_fee: &str, fee_gamma: &str, stable: bool, v21: bool,
        legacy_fee: bool, legacy_pool: bool, legacy_mul2: bool,
    ) -> Result<usize, JsError> {
        self.inner
            .add_twocrypto(&balances, &precisions, price_scale, d, amp, gamma,
                           mid_fee, out_fee, fee_gamma, stable, v21, legacy_fee,
                           legacy_pool, legacy_mul2)
            .map_err(|e| JsError::new(&e))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = addTricrypto)]
    pub fn add_tricrypto(
        &mut self, balances: Vec<String>, precisions: Vec<String>,
        price_scale: Vec<String>, d: &str, amp: &str, gamma: &str,
        mid_fee: &str, out_fee: &str, fee_gamma: &str, legacy: bool,
        a_multiplier: &str,
    ) -> Result<usize, JsError> {
        self.inner
            .add_tricrypto(&balances, &precisions, &price_scale, d, amp, gamma,
                           mid_fee, out_fee, fee_gamma, legacy, a_multiplier)
            .map_err(|e| JsError::new(&e))
    }

    /// Add a linear conversion: a vault, a lending wrapper or wstETH.
    ///
    /// `cap` of zero means no limit.
    #[wasm_bindgen(js_name = addVault)]
    pub fn add_vault(&mut self, num: &str, den: &str, cap: &str)
        -> Result<usize, JsError> {
        self.inner.add_vault(num, den, cap).map_err(|e| JsError::new(&e))
    }

    /// Add a 1:1 wrapper. It holds nothing, so one entry serves every leg.
    #[wasm_bindgen(js_name = addOneToOne)]
    pub fn add_one_to_one(&mut self) -> usize {
        self.inner.add_one_to_one()
    }

    /// Add a stableswap LP, in one direction: `deposit` picks which.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = addStableLp)]
    pub fn add_stable_lp(
        &mut self, balances: Vec<String>, rates: Vec<String>, amp: &str,
        fee: &str, offpeg_fee_multiplier: &str, a_precision: &str,
        fee_on_xp: bool, subtract_one: bool, total_supply: &str, deposit: bool,
        admin_fee: Option<String>,
    ) -> Result<usize, JsError> {
        self.inner
            .add_stable_lp(&balances, &rates, amp, fee, offpeg_fee_multiplier,
                           a_precision, fee_on_xp, subtract_one,
                           admin_fee.as_deref(), total_supply, deposit)
            .map_err(|e| JsError::new(&e))
    }

    /// Add a tricrypto LP's withdrawal arc. Deposits are exact through the
    /// pool's own getter, so there is no direction to choose.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = addTricryptoLp)]
    pub fn add_tricrypto_lp(
        &mut self, balances: Vec<String>, precisions: Vec<String>,
        price_scale: Vec<String>, d: &str, amp: &str, gamma: &str,
        mid_fee: &str, out_fee: &str, fee_gamma: &str, legacy: bool,
        a_multiplier: &str, total_supply: &str,
    ) -> Result<usize, JsError> {
        self.inner
            .add_tricrypto_lp(&balances, &precisions, &price_scale, d, amp,
                              gamma, mid_fee, out_fee, fee_gamma, legacy,
                              a_multiplier, total_supply)
            .map_err(|e| JsError::new(&e))
    }

    /// One unit of an element through one pool: what each output port pays,
    /// as decimal strings.
    ///
    /// `lp` names the pool's LP model where the element has an LP port, and is
    /// `undefined` otherwise. Ports cross as parallel coin and share arrays,
    /// which is the shape `Element` reads back.
    #[wasm_bindgen(js_name = elementEvaluate)]
    #[allow(clippy::too_many_arguments)]
    pub fn element_evaluate(
        &self, which: usize, lp: Option<usize>, n_coins: i32,
        in_coins: Vec<i32>, in_bps: Vec<i64>,
        out_coins: Vec<i32>, out_bps: Vec<i64>, dx: &str,
    ) -> Result<Vec<String>, JsValue> {
        let (inputs, outputs) = (pairs(&in_coins, &in_bps)?, pairs(&out_coins, &out_bps)?);
        self.inner
            .element_evaluate_str(which, lp, n_coins, &inputs, &outputs, dx)
            .map_err(|e| JsError::new(&e.0).into())
    }

    /// The best two-way split, as `[first bps, second bps]` with the payout in
    /// `payout`.
    ///
    /// `weights` values each output port's token in one denominator -- the
    /// payout is `float(amount * weight) / 1e18` -- because the ports pay
    /// different tokens and only the caller knows what they are worth.
    #[wasm_bindgen(js_name = elementBestSplit)]
    #[allow(clippy::too_many_arguments)]
    pub fn element_best_split(
        &self, which: usize, lp: Option<usize>, n_coins: i32,
        in_coins: Vec<i32>, in_bps: Vec<i64>,
        out_coins: Vec<i32>, out_bps: Vec<i64>, dx: &str, weights: Vec<String>,
    ) -> Result<BestSplitOut, JsValue> {
        let (inputs, outputs) = (pairs(&in_coins, &in_bps)?, pairs(&out_coins, &out_bps)?);
        let (a, b, payout) = self
            .inner
            .element_best_split_str(which, lp, n_coins, &inputs, &outputs, dx, &weights)
            .map_err(|e| -> JsValue { JsError::new(&e.0).into() })?;
        Ok(BestSplitOut { a, b, payout })
    }

    /// The best two-way split of `dx` across two output coins, or `undefined`
    /// where this family has no search and the caller must run its own.
    #[wasm_bindgen(js_name = elementSplit)]
    /// `dx` as the low and high halves of a `u128`, for the reason `price`
    /// gives.
    pub fn element_split(&self, which: usize, i: u8, j1: u8, j2: u8,
                         dx_lo: u64, dx_hi: u64) -> Option<SplitOut> {
        let dx = u128::from(dx_lo) | (u128::from(dx_hi) << 64);
        self.inner
            .element_split(which, i, j1, j2, dx)
            .map(|(a, b)| SplitOut { a, b })
    }

    /// Price a whole batch, in the order asked.
    ///
    /// `dx` is the low and high halves of each `u128`, interleaved, the same
    /// shape the answers come back in. There is no `BigUint128Array`, and the
    /// alternatives are worse: `u64` would cap a probe at eighteen tokens at
    /// eighteen decimals, and an array of `BigInt` allocates one object per
    /// probe on the path this batch exists to keep cheap.
    pub fn price(
        &self, which: &[u32], i: &[u8], j: &[u8], dx: &[u64], fast: bool,
    ) -> Result<PriceResult, JsError> {
        if which.len() != i.len() || which.len() != j.len()
            || dx.len() != which.len() * 2
        {
            return Err(JsError::new(
                "which/i/j must be the same length, and dx twice it"));
        }
        let mut values = Vec::with_capacity(which.len() * 2);
        let mut ok = Vec::with_capacity(which.len());
        for k in 0..which.len() {
            let amount = u128::from(dx[2 * k]) | (u128::from(dx[2 * k + 1]) << 64);
            match self.inner.price_one(which[k] as usize, i[k], j[k], amount,
                                       fast) {
                Some(v) => {
                    values.push(v as u64);
                    values.push((v >> 64) as u64);
                    ok.push(1);
                }
                None => {
                    values.push(0);
                    values.push(0);
                    ok.push(0);
                }
            }
        }
        Ok(PriceResult { values, ok })
    }
}

impl Default for Pools {
    fn default() -> Self {
        Self::new()
    }
}
