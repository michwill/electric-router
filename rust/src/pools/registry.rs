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

use crate::pools::lp::{StableLp, TriLp};
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
    /// An ERC4626 vault, a lending wrapper or wstETH: `dx * num / den`, one
    /// ratio for the whole range. The rounding convention -- whether the vault
    /// carries OpenZeppelin's virtual offset -- is already resolved into `num`
    /// and `den` by the time one of these is built, so there is nothing to
    /// choose here.
    Vault { num: U256, den: U256, cap: U256 },
    /// A wrapped native, or a stake that mints one for one. `RouteQuoter.vy`
    /// answers `dx` for these with no call at all.
    OneToOne,
    /// An LP burn: `dx` is the LP amount and `j` the coin to receive, which is
    /// what `calc_withdraw_one_coin` takes.
    StableWithdraw(Box<StableLp>),
    /// A single-sided deposit: `i` is the coin paid in, `dx` the amount.
    StableDeposit(Box<StableLp>),
    /// A tricrypto LP burn. Withdrawal only -- its deposits are already exact
    /// through the pool's own getter.
    TriWithdraw(Box<TriLp>),
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

    /// The two arithmetics of one stableswap, built together.
    ///
    /// Shared with the LP arcs, which need the same pool: `xp` and the inverse
    /// rates are constants of a pool frozen at a block, so they are taken once
    /// here rather than per call, and a second construction would have to
    /// agree with this one exactly.
    #[allow(clippy::too_many_arguments)]
    fn stableswap_pair(
        &self, balances: &[String], rates: &[String], amp: &str, fee: &str,
        offpeg_fee_multiplier: &str, a_precision: &str, fee_on_xp: bool,
        subtract_one: bool, admin_fee: Option<&str>,
    ) -> Result<(stableswap::Pool, stableswap::fast::Pool), String> {
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
        Ok((exact, fast))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_stableswap(
        &mut self, balances: &[String], rates: &[String], amp: &str, fee: &str,
        offpeg_fee_multiplier: &str, a_precision: &str, fee_on_xp: bool,
        subtract_one: bool, admin_fee: Option<&str>,
    ) -> Result<usize, String> {
        let (exact, fast) = self.stableswap_pair(
            balances, rates, amp, fee, offpeg_fee_multiplier, a_precision,
            fee_on_xp, subtract_one, admin_fee)?;
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

    /// One tricrypto pool, shared with its LP arc.
    #[allow(clippy::too_many_arguments)]
    fn tricrypto_pool(
        &self, balances: &[String], precisions: &[String],
        price_scale: &[String], d: &str, amp: &str, gamma: &str, mid_fee: &str,
        out_fee: &str, fee_gamma: &str, legacy: bool, a_multiplier: &str,
    ) -> Result<tricrypto::Pool, String> {
        let b = bigs(balances)?;
        let p = bigs(precisions)?;
        let s = bigs(price_scale)?;
        at(&b, 3, "balances")?;
        at(&p, 3, "precisions")?;
        at(&s, 2, "price_scale")?;
        Ok(tricrypto::Pool {
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tricrypto(
        &mut self, balances: &[String], precisions: &[String],
        price_scale: &[String], d: &str, amp: &str, gamma: &str, mid_fee: &str,
        out_fee: &str, fee_gamma: &str, legacy: bool, a_multiplier: &str,
    ) -> Result<usize, String> {
        let pool = self.tricrypto_pool(
            balances, precisions, price_scale, d, amp, gamma, mid_fee, out_fee,
            fee_gamma, legacy, a_multiplier)?;
        self.models.push(Model::Tri(Box::new(pool)));
        Ok(self.models.len() - 1)
    }

    /// Add a linear conversion. `cap` of zero means the vault takes any size.
    ///
    /// `cap` is what the vault will actually accept; the preview call does not
    /// apply it, which is why a route can be quoted through a throttle it
    /// cannot pass. Over it the answer is zero rather than a refusal -- a
    /// refusal sends the leg to the chain, and this one is known to fail.
    pub fn add_vault(&mut self, num: &str, den: &str, cap: &str) -> Result<usize, String> {
        self.models.push(Model::Vault {
            num: big(num)?,
            den: big(den)?,
            cap: big(cap)?,
        });
        Ok(self.models.len() - 1)
    }

    /// Add a 1:1 wrapper. It holds nothing, so one entry serves every leg.
    pub fn add_one_to_one(&mut self) -> usize {
        self.models.push(Model::OneToOne);
        self.models.len() - 1
    }

    /// Add a stableswap LP, in one direction.
    ///
    /// The two directions are separate entries rather than one model with a
    /// flag, because the caller already resolves them to separate arcs and a
    /// pool may reproduce one and not the other: a withdrawal that does not
    /// match the chain does not condemn the deposit beside it.
    #[allow(clippy::too_many_arguments)]
    pub fn add_stable_lp(
        &mut self, balances: &[String], rates: &[String], amp: &str, fee: &str,
        offpeg_fee_multiplier: &str, a_precision: &str, fee_on_xp: bool,
        subtract_one: bool, admin_fee: Option<&str>, total_supply: &str,
        deposit: bool,
    ) -> Result<usize, String> {
        let (exact, fast) = self.stableswap_pair(
            balances, rates, amp, fee, offpeg_fee_multiplier, a_precision,
            fee_on_xp, subtract_one, admin_fee)?;
        let lp = Box::new(StableLp { pool: exact, fast, total_supply: big(total_supply)? });
        self.models.push(if deposit {
            Model::StableDeposit(lp)
        } else {
            Model::StableWithdraw(lp)
        });
        Ok(self.models.len() - 1)
    }

    /// Add a tricrypto LP's withdrawal arc.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tricrypto_lp(
        &mut self, balances: &[String], precisions: &[String],
        price_scale: &[String], d: &str, amp: &str, gamma: &str, mid_fee: &str,
        out_fee: &str, fee_gamma: &str, legacy: bool, a_multiplier: &str,
        total_supply: &str,
    ) -> Result<usize, String> {
        let pool = self.tricrypto_pool(
            balances, precisions, price_scale, d, amp, gamma, mid_fee, out_fee,
            fee_gamma, legacy, a_multiplier)?;
        self.models.push(Model::TriWithdraw(Box::new(TriLp {
            pool,
            total_supply: big(total_supply)?,
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
            // No float form: a ratio is one multiply and one divide, and there
            // is no invariant to iterate, so `fast` has nothing to select.
            // The coin indices carry no meaning -- these have one pair.
            Model::Vault { num, den, cap } => {
                if den.is_zero() || num.is_zero() {
                    // An empty vault is a refusal, not a zero: the reference
                    // raises here, and the leg belongs on the chain.
                    return None;
                }
                if amount.is_zero() || (!cap.is_zero() && amount > *cap) {
                    return Some(0);
                }
                // `dx * num` reaches 3.4e68 for the widest inputs either side
                // admits, which U256 holds -- but a vault reporting a nonsense
                // ratio should go to the chain rather than wrap silently.
                amount.checked_mul(*num).map(|v| v / *den)
            }
            Model::OneToOne => Some(amount),
            // `j` is the coin to receive and `amount` the LP burned, which is
            // the order `calc_withdraw_one_coin` takes them in.
            Model::StableWithdraw(lp) => {
                if fast {
                    lp.calc_withdraw_one_coin_fast(amount, b)
                        .and_then(stableswap::to_u256)
                } else {
                    lp.calc_withdraw_one_coin(amount, b)
                }
            }
            // `i` is the coin paid in. The deposit is priced by what
            // `add_liquidity` mints, not by `calc_token_amount`: the getter is
            // fee-free on the legacy pools by its own admission, so quoting a
            // deposit with it promises a mint the deposit does not pay.
            Model::StableDeposit(lp) => {
                if a >= lp.n() {
                    return None;
                }
                let mut amounts = vec![U256::ZERO; lp.n()];
                amounts[a] = amount;
                if fast {
                    lp.calc_token_amount_charged_fast(&amounts)
                        .and_then(stableswap::to_u256)
                } else {
                    lp.calc_token_amount_charged(&amounts)
                }
            }
            // No float form: only the stableswap LP has one, and the reference
            // answers in integers where a model has none.
            Model::TriWithdraw(lp) => lp.calc_withdraw_one_coin(amount, b),
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
