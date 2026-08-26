//! `CurveTwocryptoMathOptimized.get_y` -- the cubic, in signed 256-bit.
//!
//! Every intermediate here fits `uint256` on valid input because the contract
//! computes the same ones and would revert otherwise; that is the reason to
//! use this width rather than a wider one. The signs are the interesting part:
//! `b`, `c`, `dd`, both deltas and `root` all go negative, and `sdiv`
//! truncates toward zero where Python's `//` floors. They agree only when the
//! operands share a sign, which here they often do not.

use crate::prims::{cbrt, isqrt, I256};
use ruint::aliases::U256;

const N_COINS: u64 = 2;

fn e(n: u64) -> U256 {
    U256::from(10u64).pow(U256::from(n))
}

fn precision() -> U256 {
    e(18)
}

fn max_gamma_small() -> U256 {
    U256::from(2u64) * e(16)
}

pub struct Bounds {
    pub min_a: U256,
    pub max_a: U256,
    pub min_gamma: U256,
    pub max_gamma: U256,
}

pub fn bounds(v21: bool) -> Bounds {
    // `A` is `A * N**N * A_MULTIPLIER`, so the limits carry both factors.
    let n2 = U256::from(N_COINS * N_COINS);
    Bounds {
        min_a: n2 * U256::from(10_000u64) / U256::from(100u64),
        max_a: n2 * U256::from(10_000u64) * U256::from(1000u64),
        min_gamma: e(10),
        max_gamma: if v21 { U256::from(199u64) * e(15) } else { U256::from(2u64) * e(15) },
    }
}

/// `(y, K0_prev)`. `None` where the contract raises; `Some((_, 0))` is the
/// branch that hands over to `newton_y`, which is not ported here -- measured,
/// it never fires on the state 27 mainnet pools are actually in.
pub fn get_y(ann: U256, gamma: U256, x: &[U256], d: U256, i: usize, v21: bool)
    -> Option<(U256, I256)> {
    let lim = bounds(v21);
    let one = U256::from(1u64);
    if !(ann > lim.min_a - one && ann < lim.max_a + one) {
        return None;
    }
    if !(gamma > lim.min_gamma - one && gamma < lim.max_gamma + one) {
        return None;
    }
    if !(d > e(17) - one && d < e(15) * e(18) + one) {
        return None;
    }

    let p = precision();
    let mut lim_mul = U256::from(100u64) * p;
    if v21 && gamma > max_gamma_small() {
        lim_mul = lim_mul * max_gamma_small() / gamma;
    }

    let x_j = x[1 - i];
    if x_j.is_zero() {
        return None;
    }
    let gamma2 = gamma * gamma;

    let k0_i = p * U256::from(N_COINS) * x_j / d;
    if v21 {
        if !(e(36) / lim_mul <= k0_i && k0_i <= lim_mul) {
            return None;
        }
    } else {
        let lo = e(16) * U256::from(N_COINS);
        let hi = e(20) * U256::from(N_COINS);
        if !(k0_i > lo - one && k0_i < hi + one) {
            return None;
        }
    }

    let ann_gamma2 = ann * gamma2;
    let four_hundred_m = U256::from(400_000_000u64);
    let pos = I256::pos;

    let mut a = pos(e(32));
    let mut b = pos(d * ann_gamma2 / four_hundred_m / x_j)
        .sub(pos(e(32) * U256::from(3u64)))
        .sub(pos(U256::from(2u64) * gamma * e(14)));
    let mut c = pos(e(32) * U256::from(3u64))
        .add(pos(U256::from(4u64) * gamma * e(14)))
        .add(pos(gamma2 / e(4)))
        .add(pos(U256::from(4u64) * ann_gamma2 / four_hundred_m * x_j / d))
        .sub(pos(U256::from(4u64) * ann_gamma2 / four_hundred_m));
    let mut dd = I256::new(true, (p + gamma) * (p + gamma) / e(4));

    if b.is_zero() {
        return None;
    }
    let three = I256::from_i64(3);
    let deltas = |a: I256, b: I256, c: I256, dd: I256| -> Option<(I256, I256)> {
        let delta0 = three.mul(a).mul(c).sdiv(b)?.sub(b);
        let twenty_seven = I256::from_i64(27);
        let delta1 = three.mul(delta0).add(b)
            .sub(twenty_seven.mul(a).mul(a).sdiv(b)?.mul(dd).sdiv(b)?);
        Some((delta0, delta1))
    };
    let (delta0, delta1) = deltas(a, b, c, dd)?;

    // Scale everything down so the cubes below stay inside the width.
    let threshold = delta0.mag.min(delta1.mag).min(a.mag);
    let mut divider = one;
    for (bound, value) in [(48u64, 30u64), (46, 28), (44, 26), (42, 24), (40, 22),
                           (38, 20), (36, 18), (34, 16), (32, 14), (30, 12),
                           (28, 10), (26, 8), (24, 6), (20, 2)] {
        if threshold > e(bound) {
            divider = e(value);
            break;
        }
    }
    let scale = pos(divider);
    a = a.sdiv(scale)?;
    b = b.sdiv(scale)?;
    c = c.sdiv(scale)?;
    dd = dd.sdiv(scale)?;
    if b.is_zero() || a.is_zero() {
        return None;
    }
    let (delta0, delta1) = deltas(a, b, c, dd)?;

    let sqrt_arg = delta1.mul(delta1)
        .add(I256::from_i64(4).mul(delta0).mul(delta0).sdiv(b)?.mul(delta0));
    if sqrt_arg.is_negative() || sqrt_arg.is_zero() {
        // The contract falls through to `newton_y` here rather than failing.
        return Some((U256::ZERO, I256::zero()));
    }
    let sqrt_val = isqrt(sqrt_arg.mag);

    let b_cbrt = I256::new(b.is_negative(), cbrt(b.mag)?);
    let two = U256::from(2u64);
    // `cbrt((delta1 + sqrt) / 2)` when `delta1 > 0`, else
    // `-cbrt((sqrt - delta1) / 2)` -- and subtracting a negative `delta1` is
    // adding its magnitude, which is why both arms read as a sum.
    let second_cbrt = if delta1.is_negative() || delta1.is_zero() {
        I256::new(true, cbrt((sqrt_val + delta1.mag) / two)?)
    } else {
        I256::pos(cbrt((delta1.mag + sqrt_val) / two)?)
    };

    let c1 = b_cbrt.mul(b_cbrt).sdiv(pos(p))?.mul(second_cbrt).sdiv(pos(p))?;
    if c1.is_zero() {
        return None;
    }

    let root = pos(p).mul(c1)
        .sub(pos(p).mul(b))
        .sub(pos(p).mul(b).sdiv(c1)?.mul(delta0))
        .sdiv(three.mul(a))?;
    let y_signed = pos(d).mul(pos(d)).sdiv(pos(x_j))?
        .mul(root).sdiv(I256::from_i64(4))?.sdiv(pos(p))?;
    if y_signed.is_negative() || y_signed.is_zero() {
        return None;
    }
    let y = y_signed.mag;

    let frac = y * p / d;
    let lo = e(36) / U256::from(N_COINS) / lim_mul;
    let hi = lim_mul / U256::from(N_COINS);
    if !(lo <= frac && frac <= hi) {
        return None;
    }
    Some((y, root))
}
