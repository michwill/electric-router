//! What a deposit mints and a withdrawal returns.
//!
//! The invariant is the one `stableswap` and `tricrypto` already reproduce;
//! what is new is that `D` moves. That is the whole difference between these
//! and a swap, and it is why they need `solve_y_d` -- `get_y_D`, balance `i`
//! when `D` is reduced to a target -- rather than the `solve_y` a trade uses.
//!
//! Two conventions the deployed source settles, and both are easy to assume
//! wrongly. `calc_token_amount` takes **no fee at all** on the legacy pools --
//! its own docstring calls it "needed to prevent front-running, not for
//! precise calculations" -- so a deposit is priced by what `add_liquidity`
//! mints instead. And `calc_withdraw_one_coin` charges `fee * N / (4 * (N-1))`
//! on each coin's *imbalance* against the ideal, not on the output, and then
//! withdraws one wei less "to account for rounding errors".
//!
//! Tricrypto is withdrawal-only. Its deposits are already exact through the
//! pool's own `calc_token_amount`, which charges the same fee `add_liquidity`
//! does, so there is nothing for a model to correct.

use crate::pools::{stableswap, tricrypto};
use ruint::aliases::U256;

fn precision() -> U256 {
    U256::from(10u64).pow(U256::from(18u64))
}

fn fee_denominator() -> U256 {
    U256::from(10u64).pow(U256::from(10u64))
}

const FEE_DENOMINATOR_F: f64 = 1e10;
const PRECISION_F: f64 = 1e18;

/// A stableswap pool's LP arcs, in both arithmetics.
///
/// `fast` is the same pool's float form, kept beside the exact one for the
/// same reason the swap models keep theirs: the rates and their inverses are
/// constants of a pool frozen at a block.
pub struct StableLp {
    pub pool: stableswap::Pool,
    pub fast: stableswap::fast::Pool,
    pub total_supply: U256,
}

impl StableLp {
    pub fn n(&self) -> usize {
        self.pool.balances.len()
    }

    /// `xp` for balances that are not the pool's own.
    fn xp_of(&self, balances: &[U256]) -> Vec<U256> {
        let p = precision();
        balances
            .iter()
            .zip(self.pool.rates.iter())
            .map(|(b, r)| *b * *r / p)
            .collect()
    }

    fn xp_of_fast(&self, balances: &[f64]) -> Vec<f64> {
        balances
            .iter()
            .zip(self.fast.rates.iter())
            .map(|(b, r)| *b * *r / PRECISION_F)
            .collect()
    }

    /// The imbalance fee: `fee * N / (4 * (N - 1))`, and zero for a one-coin
    /// pool, where there is no imbalance to charge for.
    fn imbalance_fee(&self) -> U256 {
        let n = self.n();
        if n <= 1 {
            return U256::ZERO;
        }
        self.pool.fee * U256::from(n as u64) / (U256::from(4u64) * U256::from((n - 1) as u64))
    }

    /// Coin `i` returned for burning `token_amount` of LP.
    pub fn calc_withdraw_one_coin(&self, token_amount: U256, i: usize) -> Option<U256> {
        let n = self.n();
        if self.total_supply.is_zero() || i >= n {
            return None;
        }
        let fee = self.imbalance_fee();
        let xp = self.pool.xp();
        let d0 = self.pool.d(&xp)?;
        if d0.is_zero() {
            return None;
        }
        let d1 = d0 - token_amount.checked_mul(d0)? / self.total_supply;
        let new_y = stableswap::solve_y_d(self.pool.amp, self.pool.a_precision, &xp, d1, i, n)?;

        let mut reduced = xp.clone();
        for (j, r) in reduced.iter_mut().enumerate().take(n) {
            let scaled = xp[j].checked_mul(d1)? / d0;
            // Coin `i` is measured against what the solve says it becomes; the
            // others against what they were. Two different expectations, one
            // subtraction.
            let expected = if j == i {
                if scaled > new_y { scaled - new_y } else { return None }
            } else {
                xp[j] - scaled
            };
            let charge = fee.checked_mul(expected)? / fee_denominator();
            if charge > *r {
                return None;
            }
            *r -= charge;
        }
        let solved = stableswap::solve_y_d(
            self.pool.amp, self.pool.a_precision, &reduced, d1, i, n)?;
        if reduced[i] <= solved {
            return None;
        }
        let dy = reduced[i] - solved;
        // One wei less, as the pool does, and back out of `xp` space. A `dy`
        // of zero would go negative here; the reference lets it, and the
        // answer is a refusal either way, so this declines rather than
        // pretending a `U256` can hold it.
        if dy.is_zero() {
            return None;
        }
        Some((dy - U256::from(1u64)) * precision() / self.pool.rates[i])
    }

    /// `calc_withdraw_one_coin`, with the invariants solved in floats.
    ///
    /// The imbalance fee stays the contract's own integer expression: it is
    /// three operations, not a loop, so there is nothing to gain by moving it.
    pub fn calc_withdraw_one_coin_fast(&self, token_amount: U256, i: usize) -> Option<f64> {
        let n = self.n();
        if self.total_supply.is_zero() || i >= n {
            return None;
        }
        let fee = f64::from(self.imbalance_fee());
        let xp = &self.fast.xp;
        let amp = self.fast.amp;
        let ap = self.fast.a_precision;
        let d0 = stableswap::fast::d_raw(xp, amp, ap)?;
        let d1 = d0 - f64::from(token_amount) * d0 / f64::from(self.total_supply);
        let new_y = stableswap::solve_y_d_fast(amp, ap, xp, d1, i, n)?;

        let mut reduced = xp.clone();
        for (j, r) in reduced.iter_mut().enumerate().take(n) {
            let expected = if j == i {
                xp[j] * d1 / d0 - new_y
            } else {
                xp[j] - xp[j] * d1 / d0
            };
            *r -= fee * expected / FEE_DENOMINATOR_F;
        }
        let dy = reduced[i] - stableswap::solve_y_d_fast(amp, ap, &reduced, d1, i, n)?;
        Some((dy - 1.0) * PRECISION_F / self.fast.rates[i])
    }

    /// What a deposit actually mints -- the imbalance fee included.
    ///
    /// `calc_token_amount` is the getter, fee-free on the legacy pools, so it
    /// over-states every deposit it is asked about. This is the number
    /// `add_liquidity` returns, and it needs no `admin_fee`: the DAO's share
    /// changes what the pool keeps, never what the depositor is handed.
    pub fn calc_token_amount_charged(&self, amounts: &[U256]) -> Option<U256> {
        let n = self.n();
        if self.total_supply.is_zero() || amounts.len() != n {
            return None;
        }
        let d0 = self.pool.d(&self.pool.xp())?;
        if d0.is_zero() {
            return None;
        }
        let new: Vec<U256> = self.pool.balances.iter()
            .zip(amounts.iter())
            .map(|(b, a)| *b + *a)
            .collect();
        let d1 = self.pool.d(&self.xp_of(&new))?;
        if d1 <= d0 {
            return None;
        }
        let fee = self.imbalance_fee();
        let mut priced = new.clone();
        for (k, v) in priced.iter_mut().enumerate().take(n) {
            let ideal = d1.checked_mul(self.pool.balances[k])? / d0;
            let difference = if ideal > new[k] { ideal - new[k] } else { new[k] - ideal };
            let charged = fee.checked_mul(difference)? / fee_denominator();
            if charged > *v {
                return None;
            }
            *v -= charged;
        }
        let d2 = self.pool.d(&self.xp_of(&priced))?;
        if d2 <= d0 {
            return None;
        }
        Some(self.total_supply.checked_mul(d2 - d0)? / d0)
    }

    /// `calc_token_amount_charged`, with the three invariants in floats.
    pub fn calc_token_amount_charged_fast(&self, amounts: &[U256]) -> Option<f64> {
        let n = self.n();
        if self.total_supply.is_zero() || amounts.len() != n {
            return None;
        }
        let amp = self.fast.amp;
        let ap = self.fast.a_precision;
        let d0 = stableswap::fast::d_raw(&self.fast.xp, amp, ap)?;
        let balances: Vec<f64> = self.pool.balances.iter().map(|b| f64::from(*b)).collect();
        let new: Vec<f64> = balances.iter()
            .zip(amounts.iter())
            .map(|(b, a)| *b + f64::from(*a))
            .collect();
        let d1 = stableswap::fast::d_raw(&self.xp_of_fast(&new), amp, ap)?;
        if d1 <= d0 {
            return None;
        }
        let fee = f64::from(self.imbalance_fee()) / FEE_DENOMINATOR_F;
        let priced: Vec<f64> = new.iter()
            .zip(balances.iter())
            .map(|(v, b)| *v - fee * (d1 * *b / d0 - *v).abs())
            .collect();
        let d2 = stableswap::fast::d_raw(&self.xp_of_fast(&priced), amp, ap)?;
        Some((d2 - d0) * f64::from(self.total_supply) / d0)
    }
}

/// A tricrypto pool's withdrawal arc, to the wei.
///
/// **The admin fee claim is not modelled here.**
/// `remove_liquidity_one_coin` runs `_claim_admin_fees()` before it prices
/// `dy`, and that is a state change the pool makes to itself rather than part
/// of the withdrawal arithmetic. It is corrected where it applies to the
/// probed and verified paths alike. This reproduces `calc_withdraw_one_coin`,
/// which is the thing it can be checked against.
pub struct TriLp {
    pub pool: tricrypto::Pool,
    pub total_supply: U256,
}

impl TriLp {
    pub fn calc_withdraw_one_coin(&self, token_amount: U256, i: usize) -> Option<U256> {
        if i >= 3 || self.total_supply.is_zero() || token_amount > self.total_supply {
            return None;
        }
        if token_amount.is_zero() {
            return Some(U256::ZERO);
        }
        let p = &self.pool;
        if p.balances.iter().any(|b| b.is_zero()) || p.d.is_zero() {
            return None;
        }

        // `price_scale_i` is read *before* `xp[i]` is overwritten, so for
        // `i > 0` it carries `precisions[i]` and not the scaled balance.
        // Following the source literally matters: the two differ by the
        // balance itself.
        let mut xp = p.precisions;
        let mut price_scale_i = precision() * p.precisions[0];
        xp[0] *= p.balances[0];
        for k in 1..3 {
            let scale = p.price_scale[k - 1];
            if i == k {
                price_scale_i = scale * xp[i];
            }
            xp[k] = xp[k] * p.balances[k] * scale / precision();
        }

        let mut d = p.d;
        // The fee is charged on a deliberately imprecise post-withdrawal `xp`:
        // the pool says so in as many words, because it only wants the fee to
        // rise with imbalance, not to be exact. A withdrawal too large for the
        // correction to fit keeps the maximum fee, which is what stops the
        // subtraction underflowing.
        let mut imprecise = xp;
        let correction = xp[i] * U256::from(3u64) * token_amount / self.total_supply;
        let mut fee = p.out_fee;
        if correction < imprecise[i] {
            imprecise[i] -= correction;
            fee = p.fee_of(&imprecise)?;
        }

        let d_delta = token_amount * d / self.total_supply;
        let d_fee = fee * d_delta / (U256::from(2u64) * fee_denominator()) + U256::from(1u64);
        if d_delta < d_fee {
            return None;
        }
        d -= d_delta - d_fee;
        let y = tricrypto::get_y(p.amp, p.gamma, &xp, d, i)?.0;
        if xp[i] <= y {
            return None;
        }
        Some((xp[i] - y) * precision() / price_scale_i)
    }
}
