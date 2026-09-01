//! One leg's output as a function of its input (mirror of `core/curves.py`).
//!
//! Sampled rather than modelled: the split search needs `f(x)` at whatever
//! size a weight sweep lands on, and a pool's true curve is only knowable by
//! asking it. So the probes are interpolated, and the interpolation is done in
//! `u = x / f(x)` rather than in `f` -- `u` is nearly linear over a decade
//! where `f` is not, so the same nodes buy far more accuracy.
//!
//! `Curve::at` is the hot one: the optimiser calls it millions of times, and
//! `split.rs` exists because moving it alone would have lost to the crossing
//! cost.

use std::fmt;

/// Nodes per leg, and the span they cover below the leg's maximum input.
///
/// A leg's input moves over decades as its weight sweeps the simplex, and what
/// the optimiser needs resolved is the marginal rate -- a relative quantity --
/// so the grid is geometric.
pub const NODES: usize = 24;
pub const SPAN: f64 = 4096.0;

/// What the reference raises as `CurveError`, a `ValueError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveError(pub String);

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One leg's output as a function of its input, through sampled points.
///
/// `u = x / f(x)` on the probe sizes, interpolated linearly; `rate0` closes
/// the gap below the first probe and `tail` extrapolates past the last.
pub struct Curve {
    pub x: Vec<f64>,
    pub u: Vec<f64>,
    pub slope: Vec<f64>,
    pub rate0: f64,
    pub tail: f64,
}

impl Curve {
    #[inline]
    pub fn at(&self, v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let x = &self.x;
        if v <= x[0] {
            return v * self.rate0;
        }
        // `!(v < last)`, not `v >= last`: the negation sends a NaN size down the
        // tail branch, where it falls out as 0.0 rather than into a search that
        // cannot order it.  Identical for every number that compares.
        let inverse = if !(v < x[x.len() - 1]) {
            self.u[self.u.len() - 1] + (v - x[x.len() - 1]) * self.tail
        } else {
            // `bisect_right(x, v) - 1`, and the branch above has ruled out
            // both ends, so the index is in range.  `total_cmp` rather than
            // `partial_cmp().unwrap()`: `fit` refuses a non-finite probe, so
            // the two agree on every curve that exists, and this one cannot
            // panic if that ever stops being true.
            let k = match x.binary_search_by(|p| p.total_cmp(&v)) {
                // An exact hit: `bisect_right` returns the slot *after* the
                // last equal element, so walk forward over any duplicates.
                Ok(mut hit) => {
                    while hit + 1 < x.len() && x[hit + 1] <= v {
                        hit += 1;
                    }
                    hit
                }
                Err(insert) => insert - 1,
            };
            self.u[k] + (v - x[k]) * self.slope[k]
        };
        if inverse > 0.0 {
            v / inverse
        } else {
            0.0
        }
    }
}

impl Curve {
    pub fn top(&self) -> f64 {
        self.x[self.x.len() - 1]
    }

    /// Estimated interpolation error at `v`, in basis points, from the data.
    ///
    /// Linear interpolation of `u` is off by about `h^2 |u''| / 8`, and a
    /// relative error in `u` is one in `f = x/u`. `u''` comes from the
    /// neighbouring secants, so this costs no extra probes -- which is what
    /// lets the caller refine only the legs that need it.
    pub fn error_bp_at(&self, v: f64) -> f64 {
        let (x, slope, u) = (&self.x, &self.slope, &self.u);
        if v <= x[0] {
            return 0.0;
        }
        if !(v < x[x.len() - 1]) || slope.len() < 2 {
            return f64::INFINITY; // extrapolating, NaN, or too few nodes to tell
        }
        let k = bisect_right(x, v) - 1;
        let mid = k.clamp(1, slope.len() - 1);
        let second = 2.0 * (slope[mid] - slope[mid - 1]) / (x[mid + 1] - x[mid - 1]);
        let h = x[k + 1] - x[k];
        let here = u[k] + (v - x[k]) * slope[k];
        if !(here > 0.0) {
            return f64::INFINITY;
        }
        0.125 * h * h * second.abs() / here * 10_000.0
    }
}

/// `bisect_right`: the index after the last element `<= v`.
fn bisect_right(x: &[f64], v: f64) -> usize {
    let (mut lo, mut hi) = (0usize, x.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if v < x[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Build the interpolant of `x / f(x)` through the probes.
pub fn fit(deltas: &[f64], quotes: &[f64]) -> Result<Curve, CurveError> {
    if deltas.len() != quotes.len() {
        return Err(CurveError("sizes and quotes must be 1-D and the same length".into()));
    }
    if deltas.len() < 2 {
        return Err(CurveError(format!(
            "need at least 2 probes, got {}",
            deltas.len()
        )));
    }
    if !deltas.windows(2).all(|w| w[1] - w[0] > 0.0) {
        return Err(CurveError("probe sizes must be strictly increasing".into()));
    }
    if !deltas.iter().all(|&v| v > 0.0) {
        return Err(CurveError("probe sizes must be positive".into()));
    }
    if !quotes.iter().all(|&v| v > 0.0) {
        return Err(CurveError("quotes must be positive; drop failed probes first".into()));
    }

    let u: Vec<f64> = deltas.iter().zip(quotes).map(|(x, y)| x / y).collect();
    let slope: Vec<f64> = (1..u.len())
        .map(|k| (u[k] - u[k - 1]) / (deltas[k] - deltas[k - 1]))
        .collect();
    // Never extrapolate a *falling* `u`. A final secant with increasing
    // returns -- a dynamic fee shrinking with size, §11.2's CryptoSwap-NG case
    // -- would drive `u` to zero and then negative, and `f = x/u` through the
    // roof and then off a cliff. Holding `u` flat continues the last average
    // rate instead, which is §2.3's chord: an over-estimate, hence the side
    // that cannot prune the true optimum.
    let tail = slope[slope.len() - 1].max(0.0);
    Ok(Curve {
        x: deltas.to_vec(),
        u,
        slope,
        rate0: quotes[0] / deltas[0],
        tail,
    })
}

/// `f(x) = rate * x`, for a leg that is a conversion rather than a trade.
pub fn linear(rate: f64) -> Result<Curve, CurveError> {
    if !(rate > 0.0) {
        return Err(CurveError(format!(
            "rate must be positive, got {}",
            crate::pyfmt::float(rate)
        )));
    }
    Ok(Curve {
        x: vec![1.0, 2.0],
        u: vec![1.0 / rate, 1.0 / rate],
        slope: vec![0.0],
        rate0: rate,
        tail: 0.0,
    })
}

/// Log-spaced integer probe sizes up to `top`, strictly increasing.
///
/// A node that would round onto its predecessor is dropped rather than sent --
/// two equal sizes are a zero denominator in the secant, and on a 2-decimal
/// token the bottom of the ladder collides hard.
pub fn sizes(top: f64, nodes: usize, span: f64) -> Vec<u64> {
    if top < 2.0 {
        return Vec::new();
    }
    let lo = (top / span).max(1.0);
    let mut out: Vec<u64> = Vec::new();
    for value in geomspace(lo, top, nodes) {
        let node = (value as u64).max(1);
        if out.last().is_some_and(|&last| node <= last) {
            continue;
        }
        out.push(node);
    }
    out
}

/// `np.geomspace`: a linear space in `log10`, raised, with the endpoints
/// written back exactly.
///
/// numpy computes the endpoints from the logs like every other point and then
/// overwrites them with the originals, so the two ends are exact and the
/// interior is not.
///
/// **The interior can differ from the reference by one ULP.** `np.power` is a
/// vectorised loop rather than libm's `pow` and is not correctly rounded;
/// measured over the ladders this module builds, two nodes in twenty-four
/// land one ULP from `powf` -- 1.8e-16 on a probe size of 1.8e17, which is
/// far below the pool's own integer quantum and below anything the fit that
/// consumes the ladder can resolve. Chasing it exactly would be chasing an
/// implementation detail of numpy's SIMD loop, which is not stable across its
/// own releases either.
fn geomspace(lo: f64, hi: f64, nodes: usize) -> Vec<f64> {
    if nodes == 0 {
        return Vec::new();
    }
    if nodes == 1 {
        return vec![lo];
    }
    let (log_lo, log_hi) = (lo.log10(), hi.log10());
    let step = (log_hi - log_lo) / (nodes - 1) as f64;
    let mut out: Vec<f64> = (0..nodes)
        .map(|k| 10f64.powf(log_lo + step * k as f64))
        .collect();
    out[0] = lo;
    out[nodes - 1] = hi;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_linear_curve_is_its_rate() {
        let curve = linear(2.0).unwrap();
        assert_eq!(curve.at(5.0), 10.0);
        assert_eq!(curve.at(0.0), 0.0);
        assert!(linear(0.0).is_err());
        assert!(linear(f64::NAN).is_err());
    }

    #[test]
    fn a_fit_passes_through_its_probes() {
        let deltas = [1.0, 2.0, 4.0, 8.0];
        let quotes = [0.99, 1.96, 3.88, 7.60];
        let curve = fit(&deltas, &quotes).unwrap();
        for (x, y) in deltas.iter().zip(quotes.iter()) {
            assert!((curve.at(*x) - y).abs() < 1e-12, "{x}");
        }
        assert_eq!(curve.top(), 8.0);
    }

    #[test]
    fn a_falling_u_is_never_extrapolated() {
        // Increasing returns in the last secant: `u` would fall through zero
        // and `f = x/u` off a cliff. The tail holds it flat instead.
        let curve = fit(&[1.0, 2.0, 4.0], &[0.9, 1.85, 3.9]).unwrap();
        assert_eq!(curve.tail, 0.0);
        let far = curve.at(1e6);
        assert!(far.is_finite() && far > 0.0);
        // Flat `u` continues the last *average* rate, `y[-1] / x[-1]`.
        assert!((far / 1e6 - 3.9 / 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_fit_refuses_what_it_cannot_interpolate() {
        assert!(fit(&[1.0], &[1.0]).is_err());
        assert!(fit(&[1.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(fit(&[2.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(fit(&[1.0, 2.0], &[1.0, 0.0]).is_err());
        assert!(fit(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn a_node_that_rounds_onto_its_predecessor_is_dropped() {
        // A tight span at the bottom collides hard once rounded to integers.
        let got = sizes(100.0, NODES, SPAN);
        assert!(got.windows(2).all(|w| w[1] > w[0]), "{got:?}");
        assert_eq!(*got.last().unwrap(), 100);
        assert!(sizes(1.0, NODES, SPAN).is_empty());
    }

    #[test]
    fn the_error_estimate_costs_no_extra_probes() {
        let curve = fit(&[1.0, 2.0, 4.0, 8.0], &[0.99, 1.96, 3.88, 7.60]).unwrap();
        assert_eq!(curve.error_bp_at(0.5), 0.0);
        assert_eq!(curve.error_bp_at(100.0), f64::INFINITY);
        assert!(curve.error_bp_at(3.0) > 0.0);
    }
}
