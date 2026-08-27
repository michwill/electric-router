//! `CurveTricryptoMathOptimized.get_y` and the quote around it, three coins.
//!
//! Structurally the twocrypto cubic's sibling and not its copy: a different
//! `a`, a different `delta1`, a coarser divider ladder, and an
//! `additional_prec` rescale that has no counterpart there. `_cbrt` is
//! identical, so it is shared rather than transcribed twice.

use crate::pools::prims::{cbrt, isqrt, I256};
use crate::pools::stableswap::to_u256;
use ruint::aliases::U256;

const N_COINS: u64 = 3;
const A_MULTIPLIER: u64 = 10_000;

fn e(n: u64) -> U256 {
    U256::from(10u64).pow(U256::from(n))
}

fn pos(v: U256) -> I256 {
    I256::pos(v)
}

/// `K = prod(x) / (sum(x)/N)**N`, regulated by `fee_gamma`.
pub fn reduction_coefficient(x: &[U256; 3], fee_gamma: U256) -> Option<U256> {
    let p = e(18);
    let n = U256::from(N_COINS);
    let s = x[0] + x[1] + x[2];
    if s.is_zero() {
        return None;
    }
    let mut k = p * n * x[0] / s;
    k = k * n * x[1] / s;
    k = k * n * x[2] / s;
    if !fee_gamma.is_zero() {
        if fee_gamma + p <= k {
            return None;
        }
        k = fee_gamma * p / (fee_gamma + p - k);
    }
    Some(k)
}

/// `(y, K0_prev)`. `Some((0, _))` marks the branch that hands to `newton_y`.
pub fn get_y(ann: U256, gamma: U256, x: &[U256; 3], d: U256, i: usize)
    -> Option<(U256, I256)> {
    let p = e(18);
    let one = U256::from(1u64);
    let n3 = U256::from(N_COINS * N_COINS * N_COINS);
    let min_a = n3 * U256::from(A_MULTIPLIER) / U256::from(100u64);
    let max_a = U256::from(1000u64) * U256::from(A_MULTIPLIER) * n3;
    if !(ann > min_a - one && ann < max_a + one) {
        return None;
    }
    if !(gamma > e(10) - one && gamma < U256::from(5u64) * e(16) + one) {
        return None;
    }
    if !(d > e(17) - one && d < e(15) * e(18) + one) {
        return None;
    }
    for (k, item) in x.iter().enumerate() {
        if k == i {
            continue;
        }
        let frac = *item * p / d;
        if !(frac > e(16) - one && frac < e(20) + one) {
            return None;
        }
    }

    let (j, k) = match i {
        0 => (1usize, 2usize),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (x_j, x_k) = (x[j], x[k]);
    let gamma2 = gamma * gamma;
    let am = U256::from(A_MULTIPLIER);
    let s729 = U256::from(729u64);

    let mut a = pos(e(36) / U256::from(27u64));
    // The product below runs to the top of the width before it is divided
    // back down, so it is checked rather than wrapped.
    let heavy = pos(d.checked_mul(d)? / x_j)
        .checked_mul(pos(gamma2))?
        .checked_mul(pos(ann))?;
    let mut b = pos(e(36) / U256::from(9u64))
        .add(pos(U256::from(2u64) * e(18) * gamma / U256::from(27u64)))
        .sub(heavy.sdiv(pos(s729))?.sdiv(pos(am))?.sdiv(pos(x_k))?);
    let sum_less_d = pos(x_j + x_k).sub(pos(d));
    let mut c = pos(e(36) / U256::from(9u64))
        .add(pos(gamma * (gamma + U256::from(4u64) * e(18)) / U256::from(27u64)))
        .add(pos(gamma2).checked_mul(sum_less_d)?.sdiv(pos(d))?
             .checked_mul(pos(ann))?
             .sdiv(pos(U256::from(27u64)))?.sdiv(pos(am))?);
    let mut dd = pos((p + gamma) * (p + gamma) / U256::from(27u64));

    if b.is_zero() {
        return None;
    }
    let three = I256::from_i64(3);
    let d0 = three.checked_mul(a)?.checked_mul(c)?.sdiv(b)?.sub(b).mag;

    let mut divider = one;
    for (bound, value) in [(48u64, 30u64), (44, 26), (40, 22), (36, 18),
                           (32, 14), (28, 10), (24, 6), (20, 2)] {
        if d0 > e(bound) {
            divider = e(value);
            break;
        }
    }
    let div = pos(divider);

    // Two rescales, not one: the second normalises `a` against `b` before the
    // divider, and which way round depends on which is larger.
    if a.mag > b.mag {
        let extra = pos(a.sdiv(b)?.mag);
        a = a.checked_mul(extra)?.sdiv(div)?;
        b = b.checked_mul(extra)?.sdiv(div)?;
        c = c.checked_mul(extra)?.sdiv(div)?;
        dd = dd.checked_mul(extra)?.sdiv(div)?;
    } else {
        let extra = pos(b.sdiv(a)?.mag);
        if extra.is_zero() {
            return None;
        }
        a = a.sdiv(extra)?.sdiv(div)?;
        b = b.sdiv(extra)?.sdiv(div)?;
        c = c.sdiv(extra)?.sdiv(div)?;
        dd = dd.sdiv(extra)?.sdiv(div)?;
    }
    if a.is_zero() || b.is_zero() {
        return None;
    }

    let three_ac = three.checked_mul(a)?.checked_mul(c)?;
    let delta0 = three_ac.sdiv(b)?.sub(b);
    let delta1 = three.checked_mul(three_ac)?.sdiv(b)?
        .sub(I256::from_i64(2).mul(b))
        .sub(I256::from_i64(27).checked_mul(a)?.checked_mul(a)?
             .sdiv(b)?.checked_mul(dd)?.sdiv(b)?);

    let sqrt_arg = delta1.checked_mul(delta1)?
        .add(I256::from_i64(4).checked_mul(delta0)?.checked_mul(delta0)?
             .sdiv(b)?.checked_mul(delta0)?);
    if sqrt_arg.is_negative() || sqrt_arg.is_zero() {
        // Not a failure: the contract takes this branch too, and on tricrypto
        // it is reached by real pool state rather than only in theory.
        return newton_y(ann, gamma, x, d, i, false, U256::from(A_MULTIPLIER))
            .map(|y| (y, I256::zero()));
    }
    let sqrt_val = isqrt(sqrt_arg.mag);
    let two = U256::from(2u64);

    // `b >= 0` here, where the twocrypto cubic tests `b > 0`; zero is already
    // refused above, so the two agree, but the source says what it says.
    let b_cbrt = I256::new(b.is_negative(), cbrt(b.mag)?);
    let second_cbrt = if !delta1.is_negative() && !delta1.is_zero() {
        pos(cbrt((delta1.mag + sqrt_val) / two)?)
    } else {
        // `-cbrt(-(delta1 - sqrt) / 2)`, and negating a negative `delta1`
        // adds its magnitude.
        I256::new(true, cbrt((delta1.mag + sqrt_val) / two)?)
    };

    let c1 = b_cbrt.checked_mul(b_cbrt)?.sdiv(pos(p))?
        .checked_mul(second_cbrt)?.sdiv(pos(p))?;
    if c1.is_zero() {
        return None;
    }

    let root_k0 = b.add(b.checked_mul(delta0)?.sdiv(c1)?).sub(c1).sdiv(three)?;
    let root = pos(d.checked_mul(d)? / U256::from(27u64) / x_k)
        .checked_mul(pos(d))?.sdiv(pos(x_j))?
        .checked_mul(root_k0)?.sdiv(a)?;
    if root.is_negative() || root.is_zero() {
        return None;
    }

    let frac = root.mag * p / d;
    if !(frac >= e(16) - one && frac < e(20) + one) {
        return None;
    }
    Some((root.mag, pos(p).checked_mul(root_k0)?.sdiv(a)?))
}

pub struct Pool {
    pub balances: [U256; 3],
    pub precisions: [U256; 3],
    pub price_scale: [U256; 2],
    pub d: U256,
    pub amp: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub legacy: bool,
    pub a_multiplier: U256,
}

impl Pool {
    fn fee(&self, xp: &[U256; 3]) -> Option<U256> {
        let p = e(18);
        let f = reduction_coefficient(xp, self.fee_gamma)?;
        Some((self.mid_fee * f + self.out_fee * (p - f)) / p)
    }

    /// Exactly what the pool's `get_dy(i, j, dx)` returns on chain.
    pub fn get_dy(&self, i: usize, j: usize, dx: U256) -> Option<U256> {
        self.quote(i, j, dx, false)
    }

    /// `get_dy`, with the invariant in floating point.
    pub fn get_dy_fast(&self, i: usize, j: usize, dx: U256) -> Option<U256> {
        self.quote(i, j, dx, true)
    }

    fn quote(&self, i: usize, j: usize, dx: U256, fast: bool) -> Option<U256> {
        if i == j || i >= 3 || j >= 3 {
            return None;
        }
        if dx.is_zero() {
            return Some(U256::ZERO);
        }
        if self.balances.iter().any(|b| b.is_zero()) || self.d.is_zero() {
            return None;
        }
        let p = e(18);
        let mut raw = self.balances;
        raw[i] += dx;
        let mut xp = [raw[0] * self.precisions[0], U256::ZERO, U256::ZERO];
        for k in 0..2 {
            xp[k + 1] = raw[k + 1] * self.price_scale[k] * self.precisions[k + 1] / p;
        }

        // The legacy generation goes through Newton rather than the cubic,
        // and applies the input bound the optimized math dropped.
        let y = if fast {
            // The range checks the contract makes on A, gamma and D are
            // comparisons against the integers it holds, so they are made
            // here rather than re-derived in floating point.
            let one = U256::from(1u64);
            if !self.legacy {
                let n3 = U256::from(N_COINS * N_COINS * N_COINS);
                let am = U256::from(A_MULTIPLIER);
                if !(self.amp > n3 * am / U256::from(100u64) - one
                     && self.amp < U256::from(1000u64) * am * n3 + one) {
                    return None;
                }
                if !(self.gamma > e(10) - one
                     && self.gamma < U256::from(5u64) * e(16) + one) {
                    return None;
                }
                if !(self.d > e(17) - one && self.d < e(15) * e(18) + one) {
                    return None;
                }
            }
            let f = 1e18f64;
            let xpf = [f64::from(xp[0]) / f, f64::from(xp[1]) / f,
                       f64::from(xp[2]) / f];
            let got = newton_y_fast(
                f64::from(self.amp) / f64::from(self.a_multiplier),
                f64::from(self.gamma) / f, &xpf, f64::from(self.d) / f, j,
                self.legacy)?;
            to_u256(got * f)?
        } else if self.legacy {
            newton_y(self.amp, self.gamma, &xp, self.d, j, true, self.a_multiplier)?
        } else {
            get_y(self.amp, self.gamma, &xp, self.d, j)?.0
        };
        if y.is_zero() || y >= xp[j] {
            return None;
        }
        let mut dy = xp[j] - y - U256::from(1u64);
        xp[j] = y;
        if j > 0 {
            dy = dy * p / self.price_scale[j - 1];
        }
        dy /= self.precisions[j];

        let fee = self.fee(&xp)?;
        Some(dy - fee * dy / e(10))
    }
}

const MAX_ITER: usize = 255;

/// The fallback `get_y` takes when the discriminant is not positive.
///
/// Unlike twocrypto's, this one fires on real state -- 4 of 180 vectors from
/// 15 mainnet pools -- so the cubic alone is not a complete port.
///
/// `check_inputs` adds the bound the 2021 pools apply before iterating, which
/// the optimized math dropped; it is a refusal the pool makes, so a model of
/// one has to make it too. `a_multiplier` is not constant across generations:
/// tricrypto2 and the optimized math use 10,000 and the original 2021 pools
/// use 100, and taking the wrong one quotes the pool about twice wrong --
/// close enough to look like a rounding problem and not be one.
pub fn newton_y(ann: U256, gamma: U256, x: &[U256; 3], d: U256, i: usize,
                check_inputs: bool, a_multiplier: U256) -> Option<U256> {
    let p = e(18);
    let one = U256::from(1u64);
    let n = U256::from(N_COINS);
    let am = U256::from(A_MULTIPLIER);
    let n3 = U256::from(N_COINS * N_COINS * N_COINS);
    let min_a = n3 * am / U256::from(100u64);
    let max_a = U256::from(1000u64) * am * n3;
    let lo_a = min_a * a_multiplier / am;
    let hi_a = max_a * a_multiplier / am;
    if !(ann > lo_a - one && ann < hi_a + one) {
        return None;
    }
    if !(gamma > e(10) - one && gamma < U256::from(5u64) * e(16) + one) {
        return None;
    }
    if !(d > e(17) - one && d < e(15) * e(18) + one) {
        return None;
    }
    if check_inputs {
        for (k, item) in x.iter().enumerate() {
            if k == i {
                continue;
            }
            let frac = *item * p / d;
            if !(frac > e(16) - one && frac < e(20) + one) {
                return None;
            }
        }
    }

    let mut others: Vec<U256> = x.iter().enumerate()
        .filter(|(k, _)| *k != i).map(|(_, v)| *v).collect();
    others.sort_unstable();
    others.reverse();

    let e14 = e(14);
    let convergence_limit = (others[0] / e14).max(d / e14).max(U256::from(100u64));

    let mut y = d / n;
    let mut k0_i = p;
    let mut s_i = U256::ZERO;
    // `for j in 2..=N` walks the sorted list from its smallest end.
    for j in 2..=(N_COINS as usize) {
        let item = others[N_COINS as usize - j];
        if item.is_zero() {
            return None;
        }
        y = y * d / (item * n);
        s_i += item;
    }
    for item in others.iter().take(N_COINS as usize - 1) {
        k0_i = k0_i * *item * n / d;
    }

    let two = U256::from(2u64);
    for _ in 0..MAX_ITER {
        let y_prev = y;
        if y.is_zero() {
            return None;
        }
        let k0 = k0_i * y * n / d;
        let s = s_i + y;
        if k0.is_zero() {
            return None;
        }

        let g = gamma + p;
        let g1k0 = if g > k0 { g - k0 + one } else { k0 - g + one };

        let mul1 = p * d / gamma * g1k0 / gamma * g1k0 * a_multiplier / ann;
        let mul2 = p + two * p * k0 / g1k0;

        let mut yfprime = p * y + s * mul2 + mul1;
        let dyfprime = d * mul2;
        if yfprime < dyfprime {
            y = y_prev / two;
            continue;
        }
        yfprime -= dyfprime;
        let fprime = yfprime / y;
        if fprime.is_zero() {
            return None;
        }

        let mut y_minus = mul1 / fprime;
        let y_plus = (yfprime + p * d) / fprime + y_minus * p / k0;
        y_minus += p * s / fprime;
        y = if y_plus < y_minus { y_prev / two } else { y_plus - y_minus };

        let diff = if y > y_prev { y - y_prev } else { y_prev - y };
        if diff < convergence_limit.max(y / e14) {
            let frac = y * p / d;
            if !(frac > e(16) - one && frac < e(20) + one) {
                return None;
            }
            return Some(y);
        }
    }
    None
}

/// `newton_y`, in dollars, three coins.
///
/// `a` is `ann / a_multiplier` and `gamma` is `gamma / 1e18`. The range checks
/// on `A`, `gamma` and `D` stay with the caller, which still holds them as the
/// integers the contract compares.
pub fn newton_y_fast(a: f64, gamma: f64, x: &[f64; 3], d: f64, i: usize,
                     check_inputs: bool) -> Option<f64> {
    const N: f64 = 3.0;
    const FAST_TOL: f64 = 1e-14;
    const GIVE_UP: usize = 60;

    let mut others: Vec<f64> = x.iter().enumerate()
        .filter(|(k, _)| *k != i).map(|(_, v)| *v).collect();
    others.sort_by(|p, q| q.partial_cmp(p).unwrap());
    if others[others.len() - 1] <= 0.0 || d <= 0.0 || a <= 0.0 || gamma <= 0.0 {
        return None;
    }
    if check_inputs {
        for (k, item) in x.iter().enumerate() {
            if k != i && !(0.01 <= *item / d && *item / d <= 100.0) {
                return None;
            }
        }
    }

    let mut y = d / N;
    let mut s_i = 0.0;
    for j in 2..=3usize {
        let item = others[3 - j];
        y = y * d / (item * N);
        s_i += item;
    }
    let mut k0_i = 1.0;
    for item in others.iter().take(2) {
        k0_i = k0_i * *item * N / d;
    }

    for _ in 0..GIVE_UP {
        let y_prev = y;
        if y <= 0.0 {
            return None;
        }
        let k0 = k0_i * y * N / d;
        let s = s_i + y;
        if k0 <= 0.0 {
            return None;
        }
        let g1k0 = (gamma + 1.0 - k0).abs();
        if g1k0 <= 0.0 {
            return None;
        }
        let mul1 = d * g1k0 * g1k0 / (gamma * gamma * a);
        let mul2 = 1.0 + 2.0 * k0 / g1k0;
        let mut yfprime = y + s * mul2 + mul1;
        let dyfprime = d * mul2;
        if yfprime < dyfprime {
            y = y_prev * 0.5;
            continue;
        }
        yfprime -= dyfprime;
        let fprime = yfprime / y;
        if fprime <= 0.0 {
            return None;
        }
        let mut y_minus = mul1 / fprime;
        let y_plus = (yfprime + d) / fprime + y_minus / k0;
        y_minus += s / fprime;
        y = if y_plus < y_minus { y_prev * 0.5 } else { y_plus - y_minus };

        if (y - y_prev).abs() < FAST_TOL * y {
            return Some(y);
        }
    }
    None
}
