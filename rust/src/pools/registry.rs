//! The admitted models, held by index, with no binding in sight.
//!
//! Both boundaries wrap this rather than restating it. That is not tidiness:
//! `py.rs` and the wasm module have to price a pool *identically*, because the
//! differential tests compare them and the browser is meant to answer what the
//! extension answers. Two hand-written copies of the construction would drift
//! at the first legacy flag, and the flags are exactly where the families
//! differ in the last wei.
//!
//! Integers arrive as decimal strings. A balance does not fit a `u64` and
//! round-tripping one through an `f64` would defeat the exact path; the
//! formatting is paid once per pool at the warm, where it costs nothing.

use crate::pools::{stableswap, tricrypto, twocrypto};
use ruint::aliases::U256;

/// One pool, in whichever family it belongs to.
///
/// Stableswap carries both arithmetics because it is the only family whose
/// float form is a separate structure; the other two switch on a flag.
enum Model {
    Stable(Box<stableswap::Pool>, Box<stableswap::fast::Pool>),
    Two(Box<twocrypto::Pool>),
    Tri(Box<tricrypto::Pool>),
}

fn big(s: &str) -> Result<U256, String> {
    s.parse::<U256>().map_err(|_| format!("not a u256: {s}"))
}

fn bigs(v: &[String]) -> Result<Vec<U256>, String> {
    v.iter().map(|s| big(s)).collect()
}

fn at(v: &[U256], n: usize, what: &str) -> Result<(), String> {
    if v.len() < n {
        return Err(format!("{what} needs {n} entries, got {}", v.len()));
    }
    Ok(())
}

/// What an amount may be and still cross as a `u128`: 3.4e38, which is a token
/// with 18 decimals and 1e20 units. A caller keeps anything larger on its own
/// side rather than seeing it truncated.
pub const MAX_AMOUNT: u128 = u128::MAX;

#[derive(Default)]
pub struct Registry {
    models: Vec<Model>,
}

impl Registry {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_stableswap(
        &mut self, balances: &[String], rates: &[String], amp: &str, fee: &str,
        offpeg_fee_multiplier: &str, a_precision: &str, fee_on_xp: bool,
        subtract_one: bool, admin_fee: Option<&str>,
    ) -> Result<usize, String> {
        let exact = stableswap::Pool {
            balances: bigs(balances)?,
            rates: bigs(rates)?,
            amp: big(amp)?,
            fee: big(fee)?,
            offpeg_fee_multiplier: big(offpeg_fee_multiplier)?,
            a_precision: big(a_precision)?,
            fee_on_xp,
            subtract_one,
            // Absent where the pool never told us, and `exchange` then refuses
            // rather than guessing.
            admin_fee: match admin_fee {
                Some(v) => Some(big(v)?),
                None => None,
            },
        };
        // `xp` and the inverse rates are constants of a pool frozen at a block,
        // so they are taken once here rather than per call.
        let xp: Vec<f64> = exact.xp().iter().map(|v| f64::from(*v)).collect();
        let rates_f: Vec<f64> = exact.rates.iter().map(|r| f64::from(*r)).collect();
        let inv: Vec<f64> = rates_f
            .iter()
            .map(|r| if *r == 0.0 { 0.0 } else { 1e18 / r })
            .collect();
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
    pub fn add_twocrypto(
        &mut self, balances: &[String], precisions: &[String], price_scale: &str,
        d: &str, amp: &str, gamma: &str, mid_fee: &str, out_fee: &str,
        fee_gamma: &str, stable: bool, v21: bool, legacy_fee: bool,
        legacy_pool: bool, legacy_mul2: bool,
    ) -> Result<usize, String> {
        let b = bigs(balances)?;
        let p = bigs(precisions)?;
        at(&b, 2, "balances")?;
        at(&p, 2, "precisions")?;
        self.models.push(Model::Two(Box::new(twocrypto::Pool {
            balances: [b[0], b[1]],
            precisions: [p[0], p[1]],
            price_scale: big(price_scale)?,
            d: big(d)?,
            amp: big(amp)?,
            gamma: big(gamma)?,
            mid_fee: big(mid_fee)?,
            out_fee: big(out_fee)?,
            fee_gamma: big(fee_gamma)?,
            stable,
            v21,
            legacy_fee,
            legacy_pool,
            legacy_mul2,
        })));
        Ok(self.models.len() - 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tricrypto(
        &mut self, balances: &[String], precisions: &[String],
        price_scale: &[String], d: &str, amp: &str, gamma: &str, mid_fee: &str,
        out_fee: &str, fee_gamma: &str, legacy: bool, a_multiplier: &str,
    ) -> Result<usize, String> {
        let b = bigs(balances)?;
        let p = bigs(precisions)?;
        let s = bigs(price_scale)?;
        at(&b, 3, "balances")?;
        at(&p, 3, "precisions")?;
        at(&s, 2, "price_scale")?;
        self.models.push(Model::Tri(Box::new(tricrypto::Pool {
            balances: [b[0], b[1], b[2]],
            precisions: [p[0], p[1], p[2]],
            price_scale: [s[0], s[1]],
            d: big(d)?,
            amp: big(amp)?,
            gamma: big(gamma)?,
            mid_fee: big(mid_fee)?,
            out_fee: big(out_fee)?,
            fee_gamma: big(fee_gamma)?,
            legacy,
            a_multiplier: big(a_multiplier)?,
        })));
        Ok(self.models.len() - 1)
    }

    /// The best two-way split of `dx` across two output coins.
    ///
    /// Stableswap only: the other families have no `best_split`, and a caller
    /// that gets `None` runs its own search.
    pub fn element_split(
        &self, which: usize, i: u8, j1: u8, j2: u8, dx: u128,
    ) -> Option<(u16, u16)> {
        match self.models.get(which) {
            Some(Model::Stable(exact, _)) => stableswap::best_split(
                exact, i as usize, j1 as usize, j2 as usize, U256::from(dx),
            ),
            _ => None,
        }
    }

    /// Price one probe. `None` where the pool would refuse, or where the answer
    /// does not fit a `u128` -- a `dy` past that is a token with more units than
    /// exist, and refusing sends it back to the reference path rather than
    /// wrapping it.
    ///
    /// `fast` picks the float invariant, which is what a quote wants; the exact
    /// one is for the admission gate.
    pub fn price_one(&self, which: usize, i: u8, j: u8, dx: u128, fast: bool) -> Option<u128> {
        let model = self.models.get(which)?;
        let amount = U256::from(dx);
        let (a, b) = (i as usize, j as usize);
        let got = match model {
            Model::Stable(exact, quick) => {
                if fast {
                    quick.get_dy(a, b, f64::from(amount)).and_then(stableswap::to_u256)
                } else {
                    exact.get_dy(a, b, amount)
                }
            }
            Model::Two(pool) => {
                if fast { pool.get_dy_fast(a, b, amount) } else { pool.get_dy(a, b, amount) }
            }
            Model::Tri(pool) => {
                if fast { pool.get_dy_fast(a, b, amount) } else { pool.get_dy(a, b, amount) }
            }
        };
        got.and_then(|v| u128::try_from(v).ok())
    }

    /// Price a whole batch, in the order asked.
    ///
    /// A quote evaluates these ~1,600 times and the arithmetic is 0.08 to 0.84
    /// us against a crossing of 1 to 2, so the batch is the unit that makes the
    /// boundary worth crossing at all.
    pub fn price(
        &self, which: &[usize], i: &[u8], j: &[u8], dx: &[u128], fast: bool,
    ) -> Result<Vec<Option<u128>>, String> {
        if which.len() != i.len() || which.len() != j.len() || which.len() != dx.len() {
            return Err("which/i/j/dx must be the same length".to_string());
        }
        Ok((0..which.len())
            .map(|k| self.price_one(which[k], i[k], j[k], dx[k], fast))
            .collect())
    }
}
