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
