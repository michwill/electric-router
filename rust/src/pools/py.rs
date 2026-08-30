//! The pool models, held on this side of the boundary.
//!
//! Pricing crosses once per *batch*, not once per pool. A quote evaluates
//! these ~1,600 times and the arithmetic is 0.08 to 0.84 us; a crossing is 1
//! to 2. Handing them over one at a time would spend more on the trip than on
//! the answer, so the models are built once at the warm and live here, and a
//! batch names them by index.
//!
//! Nothing is priced here. `registry::Registry` holds the models and the
//! arithmetic, and the wasm module wraps the same type -- see its docstring
//! for why the two bindings must not each keep their own copy.

use crate::pools::registry::Registry;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn err(e: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e)
}

/// The models a session has admitted, by index.
#[pyclass]
pub struct Pools {
    inner: Registry,
}

#[pymethods]
impl Pools {
    #[new]
    fn new() -> Self {
        Self { inner: Registry::new() }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Add a stableswap, in both arithmetics. Returns its index.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (balances, rates, amp, fee, offpeg_fee_multiplier,
                        a_precision, fee_on_xp, subtract_one, admin_fee=None))]
    fn add_stableswap(&mut self, balances: Vec<String>, rates: Vec<String>,
                      amp: &str, fee: &str, offpeg_fee_multiplier: &str,
                      a_precision: &str, fee_on_xp: bool, subtract_one: bool,
                      admin_fee: Option<&str>) -> PyResult<usize> {
        self.inner.add_stableswap(&balances, &rates, amp, fee,
                                  offpeg_fee_multiplier, a_precision, fee_on_xp,
                                  subtract_one, admin_fee).map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_twocrypto(&mut self, balances: Vec<String>, precisions: Vec<String>,
                     price_scale: &str, d: &str, amp: &str, gamma: &str,
                     mid_fee: &str, out_fee: &str, fee_gamma: &str,
                     stable: bool, v21: bool, legacy_fee: bool,
                     legacy_pool: bool, legacy_mul2: bool) -> PyResult<usize> {
        self.inner.add_twocrypto(&balances, &precisions, price_scale, d, amp,
                                 gamma, mid_fee, out_fee, fee_gamma, stable,
                                 v21, legacy_fee, legacy_pool, legacy_mul2)
            .map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tricrypto(&mut self, balances: Vec<String>, precisions: Vec<String>,
                     price_scale: Vec<String>, d: &str, amp: &str, gamma: &str,
                     mid_fee: &str, out_fee: &str, fee_gamma: &str,
                     legacy: bool, a_multiplier: &str) -> PyResult<usize> {
        self.inner.add_tricrypto(&balances, &precisions, &price_scale, d, amp,
                                 gamma, mid_fee, out_fee, fee_gamma, legacy,
                                 a_multiplier).map_err(err)
    }

    /// Add a linear conversion: a vault, a lending wrapper or wstETH.
    ///
    /// `cap` of zero means no limit. Crosses as decimal strings like every
    /// other parameter -- a vault's `totalAssets` does not fit a `u64`.
    fn add_vault(&mut self, num: &str, den: &str, cap: &str) -> PyResult<usize> {
        self.inner.add_vault(num, den, cap).map_err(err)
    }

    /// Add a 1:1 wrapper. It holds nothing, so one entry serves every leg.
    fn add_one_to_one(&mut self) -> usize {
        self.inner.add_one_to_one()
    }

    /// Add a stableswap LP, in one direction: `deposit` picks which.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (balances, rates, amp, fee, offpeg_fee_multiplier,
                        a_precision, fee_on_xp, subtract_one, total_supply,
                        deposit, admin_fee=None))]
    fn add_stable_lp(&mut self, balances: Vec<String>, rates: Vec<String>,
                     amp: &str, fee: &str, offpeg_fee_multiplier: &str,
                     a_precision: &str, fee_on_xp: bool, subtract_one: bool,
                     total_supply: &str, deposit: bool,
                     admin_fee: Option<&str>) -> PyResult<usize> {
        self.inner.add_stable_lp(&balances, &rates, amp, fee,
                                 offpeg_fee_multiplier, a_precision, fee_on_xp,
                                 subtract_one, admin_fee, total_supply,
                                 deposit).map_err(err)
    }

    /// Add a tricrypto LP's withdrawal arc. Deposits are exact through the
    /// pool's own getter, so there is no direction to choose.
    #[allow(clippy::too_many_arguments)]
    fn add_tricrypto_lp(&mut self, balances: Vec<String>,
                        precisions: Vec<String>, price_scale: Vec<String>,
                        d: &str, amp: &str, gamma: &str, mid_fee: &str,
                        out_fee: &str, fee_gamma: &str, legacy: bool,
                        a_multiplier: &str, total_supply: &str)
        -> PyResult<usize> {
        self.inner.add_tricrypto_lp(&balances, &precisions, &price_scale, d,
                                    amp, gamma, mid_fee, out_fee, fee_gamma,
                                    legacy, a_multiplier, total_supply)
            .map_err(err)
    }

    /// The best two-way split of `dx` across two output coins.
    ///
    /// One call rather than a batch: the caller asks once per element, and
    /// what is behind it is ~100 stateful exchanges, so the crossing is
    /// nothing beside it.
    /// One unit of an element through one pool: what each output port pays.
    ///
    /// `lp` names the pool's LP model where the element has an LP port, and
    /// is `None` otherwise. Amounts come back as decimal strings, which is
    /// what every other 256-bit answer in this binding does.
    #[pyo3(signature = (which, lp, n_coins, inputs, outputs, dx))]
    fn element_evaluate(
        &self, which: usize, lp: Option<usize>, n_coins: i32,
        inputs: Vec<(i32, i64)>, outputs: Vec<(i32, i64)>, dx: &str,
    ) -> PyResult<Vec<String>> {
        self.inner
            .element_evaluate_str(which, lp, n_coins, &inputs, &outputs, dx)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))
    }

    /// The best two-way split, as `(first bps, second bps, payout)`.
    ///
    /// `weights` values each output port's token in one denominator -- the
    /// payout is `float(amount * weight) / 1e18` -- because the ports pay
    /// different tokens and only the caller knows what they are worth.
    #[pyo3(signature = (which, lp, n_coins, inputs, outputs, dx, weights))]
    #[allow(clippy::too_many_arguments)]
    fn element_best_split(
        &self, which: usize, lp: Option<usize>, n_coins: i32,
        inputs: Vec<(i32, i64)>, outputs: Vec<(i32, i64)>, dx: &str,
        weights: Vec<String>,
    ) -> PyResult<(i64, i64, f64)> {
        self.inner
            .element_best_split_str(which, lp, n_coins, &inputs, &outputs, dx, &weights)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))
    }

    fn element_split(&self, which: usize, i: u8, j1: u8, j2: u8, dx: u128)
        -> Option<(u16, u16)> {
        self.inner.element_split(which, i, j1, j2, dx)
    }

    /// Price a whole batch. `None` where the pool would refuse.
    ///
    /// Amounts cross as `u128` rather than as decimal strings, which the pool
    /// parameters use. That is not an inconsistency: a parameter crosses once
    /// per pool at the warm where the formatting is free, and an amount
    /// crosses on every probe, where `int(str)` either way was most of what
    /// the batch cost.
    fn price<'py>(&self, py: Python<'py>, which: Vec<usize>, i: Vec<u8>,
                  j: Vec<u8>, dx: Vec<u128>, fast: bool)
        -> PyResult<Bound<'py, PyList>> {
        let out = self.inner.price(&which, &i, &j, &dx, fast).map_err(err)?;
        PyList::new(py, out)
    }
}
