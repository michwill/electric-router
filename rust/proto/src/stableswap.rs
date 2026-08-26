//! `StableSwap.get_dy`, wei for wei, in the contracts' own integer width.
//!
//! `U256` rather than `u128` is not caution: `d_p * d` reaches 1e60 on a deep
//! pool and `(multiplier - FEE_DENOMINATOR) * 4 * xpi * xpj` reaches 1e70. The
//! contracts do the same arithmetic in `uint256` and this has to agree with
//! them to the last wei, so it uses the same width and the same order of
//! operations -- `d_p = d_p * d / (x * n)` accumulates a different rounding
//! than the algebraically equal closed form, and that difference is visible.

use ruint::aliases::U256;

const MAX_ITER: usize = 255;

pub struct Pool {
    pub balances: Vec<U256>,
    pub rates: Vec<U256>,
    pub amp: U256,
    pub fee: U256,
    pub offpeg_fee_multiplier: U256,
    pub a_precision: U256,
    pub fee_on_xp: bool,
    pub subtract_one: bool,
}

fn precision() -> U256 {
    U256::from(10u64).pow(U256::from(18u64))
}

fn fee_denominator() -> U256 {
    U256::from(10u64).pow(U256::from(10u64))
}

fn close(a: U256, b: U256) -> bool {
    let gap = if a > b { a - b } else { b - a };
    gap <= U256::from(1u64)
}

impl Pool {
    pub fn xp(&self) -> Vec<U256> {
        let p = precision();
        self.balances.iter().zip(self.rates.iter())
            .map(|(b, r)| *b * *r / p).collect()
    }

    /// `D`, by the contracts' own Newton iteration.
    pub fn d(&self, xp: &[U256]) -> Option<U256> {
        let n = U256::from(xp.len() as u64);
        let s: U256 = xp.iter().fold(U256::ZERO, |acc, x| acc + *x);
        if s.is_zero() {
            return Some(U256::ZERO);
        }
        let ann = self.amp * n;
        if ann <= self.a_precision {
            return None;
        }
        let one = U256::from(1u64);
        let mut d = s;
        for _ in 0..MAX_ITER {
            let mut d_p = d;
            for x in xp {
                if x.is_zero() {
                    return None;
                }
                d_p = d_p * d / (*x * n);
            }
            let prev = d;
            let numer = (ann * s / self.a_precision + d_p * n) * d;
            let denom = (ann - self.a_precision) * d / self.a_precision + (n + one) * d_p;
            if denom.is_zero() {
                return None;
            }
            d = numer / denom;
            if close(d, prev) {
                return Some(d);
            }
        }
        None
    }

    /// The `j` balance restoring the invariant when `i` holds `x`.
    pub fn solve_y(&self, xp: &[U256], d: U256, i: usize, j: usize, x: U256)
        -> Option<U256> {
        let len = xp.len();
        let n = U256::from(len as u64);
        let ann = self.amp * n;
        if ann.is_zero() {
            return None;
        }
        let mut c = d;
        let mut s = U256::ZERO;
        for (k, item) in xp.iter().enumerate().take(len) {
            let below = if k == i { x } else if k != j { *item } else { continue };
            if below.is_zero() {
                return None;
            }
            s += below;
            c = c * d / (below * n);
        }
        c = c * d * self.a_precision / (ann * n);
        let b = s + d * self.a_precision / ann;
        let two = U256::from(2u64);
        let mut y = d;
        for _ in 0..MAX_ITER {
            let prev = y;
            let denom = two * y + b;
            if denom <= d {
                return None;
            }
            y = (y * y + c) / (denom - d);
            if close(y, prev) {
                return Some(y);
            }
        }
        None
    }

    fn dynamic_fee(&self, xpi: U256, xpj: U256) -> U256 {
        let fd = fee_denominator();
        if self.offpeg_fee_multiplier <= fd {
            return self.fee;
        }
        let xps2 = (xpi + xpj) * (xpi + xpj);
        if xps2.is_zero() {
            return self.fee;
        }
        let four = U256::from(4u64);
        self.offpeg_fee_multiplier * self.fee
            / ((self.offpeg_fee_multiplier - fd) * four * xpi * xpj / xps2 + fd)
    }

    /// Exactly what `get_dy(i, j, dx)` returns on chain.
    pub fn get_dy(&self, i: usize, j: usize, dx: U256) -> Option<U256> {
        if dx.is_zero() {
            return Some(U256::ZERO);
        }
        let p = precision();
        let fd = fee_denominator();
        let xp = self.xp();
        let d = self.d(&xp)?;
        let x = xp[i] + dx * self.rates[i] / p;
        let y = self.solve_y(&xp, d, i, j, x)?;
        let sub = if self.subtract_one { U256::from(1u64) } else { U256::ZERO };
        // `xp[j] - y - 1` is signed in Python and unsigned here; a pool that
        // cannot pay is a zero quote either way, not a wrapped one.
        if xp[j] <= y + sub {
            return Some(U256::ZERO);
        }
        let raw = xp[j] - y - sub;
        let two = U256::from(2u64);
        if self.fee_on_xp {
            let fee = self.dynamic_fee((xp[i] + x) / two, (xp[j] + y) / two);
            return Some((raw - raw * fee / fd) * p / self.rates[j]);
        }
        let out = raw * p / self.rates[j];
        Some(out - out * self.fee / fd)
    }
}
