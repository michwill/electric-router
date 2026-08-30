//! The pool models: what a quote is actually made of.
//!
//! Two arithmetics, two jobs, and they do not compete.
//!
//! **`U256` admits a pool.** The integer form *is* the contract, wei for wei,
//! and that is what makes the admission gate mean anything: a model is trusted
//! only when it reproduces the chain exactly, and a wrong rate shows up as a
//! one-wei disagreement. Runs once per pool at the warm.
//!
//! **`f64` prices with it.** A quote evaluates these thousands of times to
//! rank candidates that differ by basis points, where the last wei buys
//! nothing. Measured against the integer answers over 1,156 vectors on 159
//! mainnet pools, it is out by ~1e-11 of `dy` at any size the router probes --
//! and that residual is `y`'s round-off carried through `dy = xp[j] - y - 1`,
//! not a convergence failure, so it does not improve by iterating harder.
//!
//! Only the invariant iteration has a float form. The fee, the price scale and
//! the precisions stay integer on both paths, which is why the two share one
//! set of vectors: the float path hands `y` back and the same wrapper finishes
//! the quote.
//!
//! No I/O, no clock, no threading -- these compile to wasm32 alongside the
//! solver, which is the whole reason they live here rather than in a crate of
//! their own.

pub mod cryptoswap;
#[cfg(feature = "python")]
pub mod py;
pub mod lp;
pub mod prims;
pub mod registry;
pub mod stableswap;
pub mod tricrypto;
pub mod twocrypto;

/// `v / 10^d` as a double, the way Python's `int / int` computes it.
///
/// `f64::from(v) / 1e18` rounds **twice**: a pool balance is past `2^53`, so
/// the conversion loses bits before the division ever runs. Measured on real
/// twocrypto state that is 1 ULP -- and `dy = xp[j] - y - 1` multiplies a
/// relative error in `y` by `y/dy` on the way out, which is how one ULP in the
/// input reached 5.5e-11 bp of a quote and made the two float paths disagree
/// on 12 of 411 vectors. With this they agree on all 411.
///
/// Three cases, because no single spelling is right at every magnitude:
///
/// * `v < 2^53`: the numerator is exact, so one division is one rounding.
/// * `v < 10^d` but past `2^53`: shift the mantissa full before dividing,
///   since the quotient alone would carry no bits.
/// * otherwise: split. The quotient of a balance by `1e18` is small enough to
///   be exact and the remainder contributes under one unit, so its own
///   rounding lands below the result's last bit.
///
/// Exact against Python over 20,000 random values a decade across `1` to
/// `1e30`. Past `1e30` the quotient itself passes `2^53` and about 1.5% of
/// values land 2 ULP out; no pool state reaches there, and `scaled` is still
/// nearer than the spelling it replaced.
/// `a / b` for two integers, correctly rounded -- which is what CPython's
/// `int / int` gives and what `f64::from(a) / f64::from(b)` does not.
///
/// Two conversions and a division are three roundings; this is one. It shows
/// up wherever a wei-scale ratio becomes a float and is then compared:
/// `theta = amount / reserve`, `share_of_node = take / base`, and the `bps`
/// a leg is emitted with. A last-bit difference in the last of those is a
/// different `bps` in the calldata.
///
/// The quotient is taken with 55 bits of headroom and the remainder folded
/// back as a sticky low bit, so the single rounding below sees everything the
/// exact quotient would have. Scaling by a power of two afterwards is exact.
pub fn divided(a: ruint::aliases::U256, b: ruint::aliases::U256) -> f64 {
    use ruint::aliases::{U256, U512};
    if b.is_zero() {
        return if a.is_zero() { f64::NAN } else { f64::INFINITY };
    }
    if a.is_zero() {
        return 0.0;
    }
    let wide = |v: U256| U512::from_limbs_slice(v.as_limbs());
    // Enough bits that the two below the f64 mantissa are the guard and round
    // bits, and the third is free to carry the sticky flag.
    let shift: i32 = 55 + b.bit_len() as i32 - a.bit_len() as i32;
    let (num, den) = if shift >= 0 {
        (wide(a) << (shift as usize), wide(b))
    } else {
        (wide(a), wide(b) << ((-shift) as usize))
    };
    let mut q = num / den;
    if !(num % den).is_zero() {
        // Sticky: the discarded tail is non-zero, so this is above the tie
        // rather than on it. `q` has at least 55 bits, so its lowest is free.
        q |= U512::from(1u8);
    }
    // `as f64` on an integer rounds to nearest, ties to even -- the one
    // rounding this whole function is allowed.
    let value = q.to::<u128>() as f64;
    scale_by_two(value, -shift)
}

/// `x * 2^n`, exactly, without asking `powi` to reach a subnormal in one step.
fn scale_by_two(x: f64, n: i32) -> f64 {
    let mut value = x;
    let mut left = n;
    while left > 1000 {
        value *= 2f64.powi(1000);
        left -= 1000;
    }
    while left < -1000 {
        value *= 2f64.powi(-1000);
        left += 1000;
    }
    value * 2f64.powi(left)
}

pub fn scaled(v: ruint::aliases::U256, d: u32) -> f64 {
    use ruint::aliases::U256;
    let p = U256::from(10u64).pow(U256::from(d));
    if p.is_zero() {
        return 0.0;
    }
    let q = v / p;
    if q.is_zero() {
        if v < U256::from(1u64 << 53) {
            return f64::from(v) / 10f64.powi(d as i32);
        }
        return f64::from((v << 64) / p) * 2f64.powi(-64);
    }
    f64::from(q) + f64::from(v % p) / 10f64.powi(d as i32)
}
