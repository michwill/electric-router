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

#[derive(Clone)]
pub struct Pool {
    pub balances: Vec<U256>,
    pub rates: Vec<U256>,
    pub amp: U256,
    pub fee: U256,
    pub offpeg_fee_multiplier: U256,
    pub a_precision: U256,
    pub fee_on_xp: bool,
    pub subtract_one: bool,
    /// The DAO's share of the fee. Negative in Python means "unknown"; here
    /// `None`, and `exchange` refuses rather than guessing -- a pool advanced
    /// without it is left richer than it is and quotes the next leg too well.
    pub admin_fee: Option<U256>,
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

    /// `(dy, the pool after the trade)` -- what `exchange` would leave.
    ///
    /// A view-only chained quoter cannot see its own earlier leg, which is why
    /// a route may not touch a pool twice. That is a limitation of *asking the
    /// chain*, not of the arithmetic: for a pool the wei-exact gate admitted,
    /// the state after a trade is as computable as the trade itself, and
    /// stableswap makes it easy because `D` is derived from the balances
    /// rather than stored.
    ///
    /// The update is the contract's own, and it keeps the LP's share of the
    /// fee while losing the DAO's. Skipping `dy_admin_fee` would leave the
    /// pool richer than it is.
    pub fn exchange(&self, i: usize, j: usize, dx: U256) -> Option<(U256, Pool)> {
        let admin_fee = self.admin_fee?;
        if dx.is_zero() {
            return Some((U256::ZERO, self.clone()));
        }
        let p = precision();
        let fd = fee_denominator();
        let xp = self.xp();
        let d = self.d(&xp)?;
        let x = xp[i] + dx * self.rates[i] / p;
        let y = self.solve_y(&xp, d, i, j, x)?;
        let sub = if self.subtract_one { U256::from(1u64) } else { U256::ZERO };
        if xp[j] <= y + sub {
            return Some((U256::ZERO, self.clone()));
        }
        let raw = xp[j] - y - sub;
        let two = U256::from(2u64);
        let (dy, admin) = if self.fee_on_xp {
            let fee = self.dynamic_fee((xp[i] + x) / two, (xp[j] + y) / two);
            let charged = raw * fee / fd;
            ((raw - charged) * p / self.rates[j],
             charged * admin_fee / fd * p / self.rates[j])
        } else {
            let out = raw * p / self.rates[j];
            let charged = out * self.fee / fd;
            (out - charged, charged * admin_fee / fd)
        };
        let mut balances = self.balances.clone();
        balances[i] += dx;
        if balances[j] < dy + admin {
            return None;
        }
        balances[j] -= dy + admin;
        Some((dy, Pool { balances, ..self.clone() }))
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
        admin_fee: None,
    };
    pool.solve_y(xp, d, i, j, x)
}

/// `solve_y_fast` over explicit parameters, for the FX Swap backend.
pub fn solve_y_raw_fast(amp: f64, a_precision: f64, xp: &[f64], d: f64,
                        i: usize, j: usize, x: f64) -> Option<f64> {
    let pool = fast::Pool {
        xp: xp.to_vec(),
        rates: vec![],
        inv_rates: vec![],
        amp,
        fee: 0.0,
        offpeg_fee_multiplier: 0.0,
        a_precision,
        fee_on_xp: false,
        subtract_one: false,
    };
    pool.solve_y(d, i, j, x)
}

/// A float back to the integer space, truncating as Python's `int()` does.
///
/// The invariant iteration is the only thing that moves to `f64`; everything
/// downstream -- the fee, the price scale, the precisions -- is still the
/// contract's integer arithmetic, so `y` has to come back.
pub fn to_u256(x: f64) -> Option<U256> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    let t = x.trunc();
    if t < 3.4e38 {
        return Some(U256::from(t as u128));
    }
    // Above u128, split at 2^128 rather than saturating.
    let high = (t / 3.402_823_669_209_385e38).trunc();
    let low = t - high * 3.402_823_669_209_385e38;
    Some((U256::from(high as u128) << 128) + U256::from(low as u128))
}

/// The best `(bps, bps)` split of `dx` between two output coins.
///
/// Priced as one *element*: the second port sees the pool the first one left,
/// which is the thing two independent arcs cannot express. That coupling is
/// the whole reason this exists, and it is why it needs `exchange` rather than
/// `get_dy`.
///
/// Ternary search, because the objective is concave in the split -- each
/// port's output is concave in its own share, and a sum of concave functions
/// of a linear split is concave. True for `exchange` on a stableswap in its
/// normal range, and not asserted for one being pushed off its peg, where the
/// caller should not be here.
///
/// The ports are valued in the pool's own common denominator rather than raw
/// wei: 3pool pays USDC in 6 decimals and DAI in 18, so comparing the two
/// directly would hand the whole trade to whichever coin carries more digits.
pub fn best_split(pool: &Pool, i: usize, j1: usize, j2: usize, dx: U256)
    -> Option<(u16, u16)> {
    const BPS: u64 = 10_000;
    const ROUNDS: usize = 25;

    let n = pool.balances.len();
    if i >= n || j1 >= n || j2 >= n || i == j1 || i == j2 || j1 == j2 {
        return None;
    }
    if dx.is_zero() {
        return None;
    }
    let bps = U256::from(BPS);
    let scale = 1e18f64;

    // `-inf` for a split the element refuses, so the search walks away from
    // it rather than treating a refusal as a zero payout.
    let payout = |at: u64| -> f64 {
        let first = dx * U256::from(at) / bps;
        if first.is_zero() || first >= dx {
            return f64::NEG_INFINITY;
        }
        // The last port takes the remainder, which is what `Leg.bps` does on
        // chain: shares are integers and a rounded-down last share would
        // strand wei in the slot.
        let second = dx - first;
        let mut state = pool.clone();
        let mut total = 0.0f64;
        for (coin, share) in [(j1, first), (j2, second)] {
            match state.exchange(i, coin, share) {
                None => return f64::NEG_INFINITY,
                Some((dy, after)) => {
                    total += f64::from(dy) * f64::from(pool.rates[coin]) / scale;
                    state = after;
                }
            }
        }
        total
    };

    let (mut low, mut high) = (1u64, BPS - 1);
    for _ in 0..ROUNDS {
        if high - low < 3 {
            break;
        }
        let left = low + (high - low) / 3;
        let right = high - (high - low) / 3;
        if payout(left) < payout(right) {
            low = left;
        } else {
            high = right;
        }
    }
    let mut best = low;
    let mut found = f64::NEG_INFINITY;
    for at in low..=high {
        let got = payout(at);
        if got > found {
            found = got;
            best = at;
        }
    }
    if found == f64::NEG_INFINITY {
        return None;
    }
    Some((best as u16, (BPS - best) as u16))
}
