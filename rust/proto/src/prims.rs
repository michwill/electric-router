//! The primitives the cryptoswap cubic is built on, in the EVM's arithmetic.
//!
//! Each of these is a place where a faithful port and a correct one differ.
//! `cbrt` runs *seven* Newton steps and stops -- the count is part of the
//! answer, so iterating to convergence returns a different number than the
//! contract does. `sdiv` truncates toward zero, where Rust's `/` on integers
//! happens to agree but Python's `//` does not. And the seed multiplies
//! modulo 2^256, so it wraps rather than saturating.

use ruint::aliases::U256;

/// A signed 256-bit value as sign and magnitude.
///
/// `ruint` has no signed type and I would rather not depend on one
/// transitively through revm: the only signed operations the cubic needs are
/// add, subtract, multiply and EVM division, and sign-magnitude makes each of
/// them say what it means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct I256 {
    pub neg: bool,
    pub mag: U256,
}

impl I256 {
    pub fn zero() -> Self {
        Self { neg: false, mag: U256::ZERO }
    }

    pub fn pos(mag: U256) -> Self {
        Self { neg: false, mag }
    }

    pub fn new(neg: bool, mag: U256) -> Self {
        // There is one zero, not two: a negative zero would break `is_zero`
        // comparisons and every sign test downstream of them.
        Self { neg: neg && !mag.is_zero(), mag }
    }

    pub fn from_i64(v: i64) -> Self {
        Self::new(v < 0, U256::from(v.unsigned_abs()))
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_zero()
    }

    pub fn is_negative(&self) -> bool {
        self.neg && !self.mag.is_zero()
    }

    pub fn add(self, other: Self) -> Self {
        if self.neg == other.neg {
            return Self::new(self.neg, self.mag + other.mag);
        }
        if self.mag >= other.mag {
            Self::new(self.neg, self.mag - other.mag)
        } else {
            Self::new(other.neg, other.mag - self.mag)
        }
    }

    pub fn sub(self, other: Self) -> Self {
        self.add(Self::new(!other.neg, other.mag))
    }

    pub fn mul(self, other: Self) -> Self {
        Self::new(self.neg != other.neg, self.mag * other.mag)
    }

    /// Signed division truncating toward zero, as the EVM does it.
    pub fn sdiv(self, other: Self) -> Option<Self> {
        if other.is_zero() {
            return None;
        }
        Some(Self::new(self.neg != other.neg, self.mag / other.mag))
    }
}

/// `_snekmate_log_2(x, False)` -- the index of the top set bit, 0 for 0.
pub fn log2(x: U256) -> u32 {
    if x.is_zero() { 0 } else { 255 - x.leading_zeros() as u32 }
}

/// `_cbrt`: seeded from log2, then seven unrolled Newton steps.
pub fn cbrt(x: U256) -> Option<U256> {
    let e18 = U256::from(10u64).pow(U256::from(18u64));
    let e36 = e18 * e18;
    // The two thresholds are the contract's, and they decide how far the
    // input is scaled up before the iteration sees it.
    let big: U256 = "115792089237316195423570985008687907853269".parse().unwrap();
    let xx = if x >= big * e18 {
        x
    } else if x >= big {
        x * e18
    } else {
        x * e36
    };

    let log2x = log2(xx);
    let remainder = log2x % 3;
    // `pow(2, log2x // 3, 2**256)` and `pow(1260, remainder, 2**256)`: modular,
    // so both wrap. `U256::pow` wraps too, which is the same arithmetic.
    let two_pow = U256::from(2u64).pow(U256::from(log2x / 3));
    let mul = U256::from(1260u64).pow(U256::from(remainder));
    let div = U256::from(1000u64).pow(U256::from(remainder));
    let mut a = two_pow.wrapping_mul(mul) / div;

    for _ in 0..7 {
        if a.is_zero() {
            return None;
        }
        a = (U256::from(2u64) * a + xx / (a * a)) / U256::from(3u64);
    }

    // The scale-back that pairs with the scale-up above. Easy to miss and
    // silent when missed: the answer comes out a factor of 1e6 or 1e12 small
    // on exactly the inputs a deep pool reaches, and looks entirely plausible.
    Some(if x >= big * e18 {
        a * U256::from(10u64).pow(U256::from(12u64))
    } else if x >= big {
        a * U256::from(10u64).pow(U256::from(6u64))
    } else {
        a
    })
}

/// Integer square root, floor.
pub fn isqrt(x: U256) -> U256 {
    if x < U256::from(2u64) {
        return x;
    }
    // Newton from a power-of-two seed above the root; converges downward and
    // the guard is the classic one -- stop when the next step stops shrinking.
    let mut guess = U256::from(1u64) << ((log2(x) / 2) + 1);
    loop {
        let next = (guess + x / guess) >> 1;
        if next >= guess {
            return guess;
        }
        guess = next;
    }
}

/// Same as `cbrt`, printing what each stage produced.
pub fn cbrt_trace(x: U256) {
    let e18 = U256::from(10u64).pow(U256::from(18u64));
    let e36 = e18 * e18;
    let big: U256 = "115792089237316195423570985008687907853269".parse().unwrap();
    let (branch, xx) = if x >= big * e18 {
        ("x", x)
    } else if x >= big {
        ("x*1e18", x * e18)
    } else {
        ("x*1e36", x * e36)
    };
    let log2x = log2(xx);
    let remainder = log2x % 3;
    let two_pow = U256::from(2u64).pow(U256::from(log2x / 3));
    let mul = U256::from(1260u64).pow(U256::from(remainder));
    let div = U256::from(1000u64).pow(U256::from(remainder));
    let seed = two_pow.wrapping_mul(mul) / div;
    println!("    branch {branch} · xx {xx} · log2 {log2x} · rem {remainder}");
    println!("    two_pow {two_pow} · seed {seed}");
}
