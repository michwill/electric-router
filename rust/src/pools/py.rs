//! The pool models, held on this side of the boundary.
//!
//! Pricing crosses once per *batch*, not once per pool. A quote evaluates
//! these ~1,600 times and the arithmetic is 0.08 to 0.84 us; a crossing is 1
//! to 2. Handing them over one at a time would spend more on the trip than on
//! the answer, so the models are built once at the warm and live here, and a
//! batch names them by index.

use crate::pools::{stableswap, tricrypto, twocrypto};
use pyo3::prelude::*;
use pyo3::types::PyList;
use ruint::aliases::U256;

/// One pool, in whichever family it belongs to.
enum Model {
    Stable(Box<stableswap::Pool>, Box<stableswap::fast::Pool>),
    Two(Box<twocrypto::Pool>),
    Tri(Box<tricrypto::Pool>),
}

fn big(s: &str) -> PyResult<U256> {
    s.parse::<U256>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("not a u256: {s}")))
}

fn bigs(v: &[String]) -> PyResult<Vec<U256>> {
    v.iter().map(|s| big(s)).collect()
}

/// The models a session has admitted, by index.
#[pyclass]
pub struct Pools {
    models: Vec<Model>,
}

#[pymethods]
impl Pools {
    #[new]
    fn new() -> Self {
        Self { models: Vec::new() }
    }

    fn __len__(&self) -> usize {
        self.models.len()
    }

    /// Add a stableswap, in both arithmetics. Returns its index.
    ///
    /// Integers cross as decimal strings: a balance does not fit a `u64`, and
    /// round-tripping one through a float would defeat the point of the exact
    /// path. Once per pool at the warm, so the cost is nothing.
    #[allow(clippy::too_many_arguments)]
    fn add_stableswap(&mut self, balances: Vec<String>, rates: Vec<String>,
                      amp: &str, fee: &str, offpeg_fee_multiplier: &str,
                      a_precision: &str, fee_on_xp: bool, subtract_one: bool,
                      admin_fee: Option<&str>) -> PyResult<usize> {
        let exact = stableswap::Pool {
            balances: bigs(&balances)?,
            rates: bigs(&rates)?,
            amp: big(amp)?,
            fee: big(fee)?,
            offpeg_fee_multiplier: big(offpeg_fee_multiplier)?,
            a_precision: big(a_precision)?,
            fee_on_xp,
            subtract_one,
            // Absent where the pool never told us, and `exchange` then
            // refuses rather than guessing.
            admin_fee: match admin_fee {
                Some(v) => Some(big(v)?),
                None => None,
            },
        };
        // `xp` and the inverse rates are constants of a pool frozen at a
        // block, so they are taken once here rather than per call.
        let xp: Vec<f64> = exact.xp().iter().map(|v| f64::from(*v)).collect();
        let rates_f: Vec<f64> = exact.rates.iter().map(|r| f64::from(*r)).collect();
        let inv: Vec<f64> = rates_f.iter()
            .map(|r| if *r == 0.0 { 0.0 } else { 1e18 / r }).collect();
        let fast = stableswap::fast::Pool {
            xp,
            rates: rates_f,
            inv_rates: inv,
            amp: f64::from(exact.amp),
            fee: f64::from(exact.fee),
            offpeg_fee_multiplier: f64::from(exact.offpeg_fee_multiplier),
            a_precision: f64::from(exact.a_precision),
            fee_on_xp,
            subtract_one,
        };
        self.models.push(Model::Stable(Box::new(exact), Box::new(fast)));
        Ok(self.models.len() - 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_twocrypto(&mut self, balances: Vec<String>, precisions: Vec<String>,
                     price_scale: &str, d: &str, amp: &str, gamma: &str,
                     mid_fee: &str, out_fee: &str, fee_gamma: &str,
                     stable: bool, v21: bool, legacy_fee: bool,
                     legacy_pool: bool, legacy_mul2: bool) -> PyResult<usize> {
        let b = bigs(&balances)?;
        let p = bigs(&precisions)?;
        self.models.push(Model::Two(Box::new(twocrypto::Pool {
            balances: [b[0], b[1]], precisions: [p[0], p[1]],
            price_scale: big(price_scale)?, d: big(d)?, amp: big(amp)?,
            gamma: big(gamma)?, mid_fee: big(mid_fee)?, out_fee: big(out_fee)?,
            fee_gamma: big(fee_gamma)?, stable, v21, legacy_fee, legacy_pool,
            legacy_mul2,
        })));
        Ok(self.models.len() - 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tricrypto(&mut self, balances: Vec<String>, precisions: Vec<String>,
                     price_scale: Vec<String>, d: &str, amp: &str, gamma: &str,
                     mid_fee: &str, out_fee: &str, fee_gamma: &str,
                     legacy: bool, a_multiplier: &str) -> PyResult<usize> {
        let b = bigs(&balances)?;
        let p = bigs(&precisions)?;
        let s = bigs(&price_scale)?;
        self.models.push(Model::Tri(Box::new(tricrypto::Pool {
            balances: [b[0], b[1], b[2]], precisions: [p[0], p[1], p[2]],
            price_scale: [s[0], s[1]], d: big(d)?, amp: big(amp)?,
            gamma: big(gamma)?, mid_fee: big(mid_fee)?, out_fee: big(out_fee)?,
            fee_gamma: big(fee_gamma)?, legacy, a_multiplier: big(a_multiplier)?,
        })));
        Ok(self.models.len() - 1)
    }

    /// The best two-way split of `dx` across two output coins.
    ///
    /// One call rather than a batch: the caller asks once per element, and
    /// what is behind it is ~100 stateful exchanges, so the crossing is
    /// nothing beside it.
    fn element_split(&self, which: usize, i: u8, j1: u8, j2: u8, dx: u128)
        -> Option<(u16, u16)> {
        match self.models.get(which) {
            Some(Model::Stable(exact, _)) => stableswap::best_split(
                exact, i as usize, j1 as usize, j2 as usize, U256::from(dx)),
            _ => None,
        }
    }

    /// Price a whole batch. `None` where the pool would refuse.
    ///
    /// `fast` picks the float invariant, which is what a quote wants; the
    /// exact one is for the gate.
    ///
    /// Amounts cross as `u128` rather than as decimal strings, which the pool
    /// parameters use. That is not an inconsistency: a parameter crosses once
    /// per pool at the warm where the formatting is free, and an amount
    /// crosses on every probe, where `int(str)` either way was most of what
    /// the batch cost. `u128` reaches 3.4e38 -- a token with 18 decimals and
    /// 1e20 units -- and the caller keeps anything above it on the Python
    /// path rather than truncating it.
    fn price<'py>(&self, py: Python<'py>, which: Vec<usize>, i: Vec<u8>,
                  j: Vec<u8>, dx: Vec<u128>, fast: bool)
        -> PyResult<Bound<'py, PyList>> {
        if which.len() != i.len() || which.len() != j.len() || which.len() != dx.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "which/i/j/dx must be the same length"));
        }
        // Answers come back as `u128` too. A `dy` past that is a token with
        // more units than exist; `None` sends it to Python rather than
        // wrapping it.
        let mut out: Vec<Option<u128>> = Vec::with_capacity(which.len());
        for k in 0..which.len() {
            let Some(model) = self.models.get(which[k]) else {
                out.push(None);
                continue;
            };
            let amount = U256::from(dx[k]);
            let (a, b) = (i[k] as usize, j[k] as usize);
            let got = match model {
                Model::Stable(exact, quick) => {
                    if fast {
                        quick.get_dy(a, b, f64::from(amount))
                            .and_then(stableswap::to_u256)
                    } else {
                        exact.get_dy(a, b, amount)
                    }
                }
                Model::Two(pool) => {
                    if fast { pool.get_dy_fast(a, b, amount) }
                    else { pool.get_dy(a, b, amount) }
                }
                Model::Tri(pool) => {
                    if fast { pool.get_dy_fast(a, b, amount) }
                    else { pool.get_dy(a, b, amount) }
                }
            };
            out.push(got.and_then(|v| u128::try_from(v).ok()));
        }
        PyList::new(py, out)
    }
}
