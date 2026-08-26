//! `Twocrypto.get_dy` -- the wrapper around whichever invariant the pool has.
//!
//! Three backends, chosen by what the pool *is* rather than by what it is
//! called: the FX Swap's stableswap iteration, the optimized cubic, and the
//! inline Newton of the generation before the maths moved into a `MATH`
//! contract. All three are already ported and verified; this is the price
//! scale, the precisions and the fee around them.
//!
//! The flags are not configuration. Each names a way two deployed generations
//! differ in the last wei, and every one of them was found by reading a
//! deployed contract rather than a repository copy.

use crate::cryptoswap::{get_y, newton_y};
use crate::stableswap::solve_y_raw;
use ruint::aliases::U256;

const N_COINS: u64 = 2;
const A_MULTIPLIER: u64 = 10_000;

fn e(n: u64) -> U256 {
    U256::from(10u64).pow(U256::from(n))
}

pub struct Pool {
    pub balances: [U256; 2],
    pub precisions: [U256; 2],
    pub price_scale: U256,
    pub d: U256,
    pub amp: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub stable: bool,
    pub v21: bool,
    pub legacy_fee: bool,
    pub legacy_pool: bool,
    pub legacy_mul2: bool,
}

impl Pool {
    /// `_fee`, on the balances *after* the trade.
    ///
    /// Two versions are deployed and they are not algebraically equal, so
    /// which one a pool implements is established rather than assumed. The
    /// difference is about a part in ten million of the output -- small enough
    /// to look like a rounding bug and far too large to be one.
    fn fee(&self, xp: &[U256; 2]) -> Option<U256> {
        let p = e(18);
        let fee_p = e(10);
        let total = xp[0] + xp[1];
        if total.is_zero() {
            return None;
        }
        let n = U256::from(N_COINS);
        let k = p * n.pow(n) * xp[0] / total * xp[1] / total;
        if self.legacy_fee {
            // No clamp in this one, and the denominator can collapse.
            if self.fee_gamma + p <= k {
                return None;
            }
            let f = self.fee_gamma * p / (self.fee_gamma + p - k);
            return Some((self.mid_fee * f + self.out_fee * (p - f)) / p);
        }
        let denom = self.fee_gamma * k / p + p;
        if denom <= k {
            return None;
        }
        let b = self.fee_gamma * k / (denom - k);
        let fee = (self.mid_fee * b + self.out_fee * (p - b)) / p;
        let min_fee = fee_p / U256::from(10u64) / U256::from(10_000u64);
        Some(fee.clamp(min_fee, fee_p))
    }

    /// Whichever `y` this pool's maths solves for.
    fn y(&self, xp: &[U256; 2], j: usize) -> Option<U256> {
        if self.legacy_pool {
            // `lim_mul` is the fixed 100e18 of that era.
            return newton_y(self.amp, self.gamma, xp, self.d, j,
                            U256::from(100u64) * e(18), true, self.legacy_mul2);
        }
        if !self.stable {
            return get_y(self.amp, self.gamma, xp, self.d, j, self.v21)
                .map(|(y, _)| y);
        }
        // `StableswapMath.get_y`: the stableswap iteration at A_MULTIPLIER.
        // `i` and `j` are the other way round from `solve_y`'s signature --
        // there `i` is the coin whose balance is known.
        let other = 1 - j;
        solve_y_raw(self.amp, U256::from(A_MULTIPLIER), xp, self.d, other, j,
                    xp[other])
    }

    /// Exactly what the pool's `get_dy(i, j, dx)` returns on chain.
    pub fn get_dy(&self, i: usize, j: usize, dx: U256) -> Option<U256> {
        if i == j || i >= 2 || j >= 2 {
            return None;
        }
        if dx.is_zero() {
            return Some(U256::ZERO);
        }
        if self.balances[0].is_zero() || self.balances[1].is_zero() || self.d.is_zero() {
            return None;
        }
        let p = e(18);

        // The inline generation's `price_scale` already carries
        // `precisions[1]`, so `dy` comes back divided by the product in one
        // step rather than by both in sequence. The same value in exact
        // arithmetic; a different wei in integer arithmetic.
        let scale = if self.legacy_pool {
            self.price_scale * self.precisions[1]
        } else {
            self.price_scale
        };
        let mut raw = self.balances;
        raw[i] += dx;
        let mut xp = [
            raw[0] * self.precisions[0],
            if self.legacy_pool {
                raw[1] * scale / p
            } else {
                raw[1] * self.price_scale * self.precisions[1] / p
            },
        ];

        let y = self.y(&xp, j)?;
        if y >= xp[j] {
            return None;
        }
        let mut dy = xp[j] - y - U256::from(1u64);
        xp[j] = y;
        if self.legacy_pool {
            if j > 0 {
                dy = dy * p / scale;
            } else {
                dy /= self.precisions[0];
            }
        } else {
            if j > 0 {
                dy = dy * p / self.price_scale;
            }
            dy /= self.precisions[j];
        }

        // The fee reads the post-trade balances, which is why `xp[j]` is
        // assigned above rather than after this.
        let fee = self.fee(&xp)?;
        Some(dy - fee * dy / e(10))
    }
}
