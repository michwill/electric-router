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

/// The same algebra in `f64`, for the places that only need to choose.
///
/// A quote uses the exact form for the *bound* -- a minimum rate is a promise
/// about a number nothing on chain will re-check -- and could use this one for
/// the search, which only has to rank. Whether that trade is worth making is a
/// question about how much of a quote is spent here at all.
pub mod fast {
    //! The arithmetic a quote actually runs.
    //!
    //! The integer form above *is* the contract, wei for wei, and that is what
    //! makes the admission gate meaningful: a pool is trusted only when it
    //! reproduces the chain exactly. But a quote prices thousands of times and
    //! ranks candidates that differ by basis points, where the last wei buys
    //! nothing -- measured over 1,052 samples on 263 mainnet stableswaps, the
    //! float form is out by a median of 2e-9 bp and a worst case of 5.4e-4.
    //!
    //! So: integers decide whether a model may be used, floats price with it.

    /// Rates and their inverses are constants of a pool frozen at a block, so
    /// they are taken once rather than divided by on every call.
    pub struct Pool {
        pub xp: Vec<f64>,
        pub rates: Vec<f64>,
        pub inv_rates: Vec<f64>,
        pub amp: f64,
        pub fee: f64,
        pub offpeg_fee_multiplier: f64,
        pub a_precision: f64,
        pub fee_on_xp: bool,
        pub subtract_one: bool,
    }

    const PRECISION: f64 = 1e18;
    const FEE_DENOMINATOR: f64 = 1e10;
    /// Relative, not absolute. The integer form stops at `|y - prev| <= 1`
    /// because a wei is a wei; in `f64` at 1e30 an ULP is about 1e14, so the
    /// same test is either never true or trivially true and the iteration runs
    /// its cap and returns whatever it reached. Matches Python's `_FAST_TOL`.
    const FAST_TOL: f64 = 1e-14;
    /// Newton doubles its digits a step, so a solve that has not converged in
    /// forty is not going to: measured over 2,240 solves on 268 mainnet pools,
    /// the worst took 12 and none reached the 255 the contract allows.
    const GIVE_UP: usize = 40;
    /// What to fall back to when it does not. A tolerance that cannot be met
    /// is a statement about the pool's conditioning, not a reason to fail a
    /// quote that only needed less precision.
    const LOOSE_TOL: f64 = 1e-9;

    impl Pool {
        pub fn d(&self) -> Option<f64> {
            self.d_within(FAST_TOL).or_else(|| self.d_within(LOOSE_TOL))
        }

        fn d_within(&self, tol: f64) -> Option<f64> {
            let n = self.xp.len() as f64;
            let s: f64 = self.xp.iter().sum();
            if s == 0.0 {
                return Some(0.0);
            }
            let ann = self.amp * n;
            let mut d = s;
            for _ in 0..GIVE_UP {
                let mut d_p = d;
                for x in &self.xp {
                    if *x <= 0.0 {
                        return None;
                    }
                    d_p = d_p * d / (*x * n);
                }
                let prev = d;
                d = (ann * s / self.a_precision + d_p * n) * d
                    / ((ann - self.a_precision) * d / self.a_precision + (n + 1.0) * d_p);
                if (d - prev).abs() <= tol * d {
                    return Some(d);
                }
            }
            None
        }

        pub fn solve_y(&self, d: f64, i: usize, j: usize, x: f64) -> Option<f64> {
            self.y_within(d, i, j, x, FAST_TOL)
                .or_else(|| self.y_within(d, i, j, x, LOOSE_TOL))
        }

        fn y_within(&self, d: f64, i: usize, j: usize, x: f64, tol: f64) -> Option<f64> {
            let len = self.xp.len();
            let n = len as f64;
            let ann = self.amp * n;
            let mut c = d;
            let mut s = 0.0;
            for (k, item) in self.xp.iter().enumerate().take(len) {
                let below = if k == i { x } else if k != j { *item } else { continue };
                if below <= 0.0 {
                    return None;
                }
                s += below;
                c = c * d / (below * n);
            }
            c = c * d * self.a_precision / (ann * n);
            let b = s + d * self.a_precision / ann;
            let mut y = d;
            for _ in 0..GIVE_UP {
                let prev = y;
                y = (y * y + c) / (2.0 * y + b - d);
                if (y - prev).abs() <= tol * y {
                    return Some(y);
                }
            }
            None
        }

        /// `dynamic_fee`, without squaring a 1e24 integer into a 1e48 one.
        fn dynamic_fee(&self, xpi: f64, xpj: f64) -> f64 {
            if self.offpeg_fee_multiplier <= FEE_DENOMINATOR {
                return self.fee;
            }
            let total = xpi + xpj;
            if total <= 0.0 {
                return self.fee;
            }
            let balanced = 4.0 * xpi * xpj / (total * total);
            self.offpeg_fee_multiplier * self.fee
                / ((self.offpeg_fee_multiplier - FEE_DENOMINATOR) * balanced
                   + FEE_DENOMINATOR)
        }

        pub fn get_dy(&self, i: usize, j: usize, dx: f64) -> Option<f64> {
            if dx <= 0.0 {
                return Some(0.0);
            }
            let d = self.d()?;
            let x = self.xp[i] + dx * self.rates[i] / PRECISION;
            let y = self.solve_y(d, i, j, x)?;
            let raw = self.xp[j] - y - if self.subtract_one { 1.0 } else { 0.0 };
            if raw <= 0.0 {
                return Some(0.0);
            }
            // Truncated, as Python's `int(...)` truncates: the caller is
            // handed wei and a fractional wei is not one.
            if self.fee_on_xp {
                let fee = self.dynamic_fee((self.xp[i] + x) * 0.5, (self.xp[j] + y) * 0.5);
                return Some(((raw - raw * fee / FEE_DENOMINATOR) * self.inv_rates[j]).trunc());
            }
            let out = raw * self.inv_rates[j];
            Some((out - out * self.fee / FEE_DENOMINATOR).trunc())
        }
    }
}

/// `solve_y` over explicit parameters rather than a pool.
///
/// The FX Swap is a stableswap invariant wearing cryptoswap's machinery, and
/// it calls this at `A_MULTIPLIER` where stableswap-ng calls it at
/// `A_PRECISION` -- which is the only difference between them, and is already
/// a parameter.
pub fn solve_y_raw(amp: U256, a_precision: U256, xp: &[U256], d: U256,
                   i: usize, j: usize, x: U256) -> Option<U256> {
    let pool = Pool {
        balances: vec![],
        rates: vec![],
        amp,
        fee: U256::ZERO,
        offpeg_fee_multiplier: U256::ZERO,
        a_precision,
        fee_on_xp: false,
        subtract_one: false,
    };
    pool.solve_y(xp, d, i, j, x)
}
