//! `_recalibrate`, on the other side of the boundary, held to Python's answers.
//!
//! A ported model has a natural acceptance test -- wei-exact or not. A ported
//! stage does not, so it is given one: every arc Python re-fitted, with the
//! ladder it read and the calibration it produced, compared field by field.
//!
//! What this is really testing is not the fit. `calibrate` has been Rust for a
//! while and both sides call the same function. It is the *batching* -- 452
//! crossings and 452 dataclasses become one call and a slice -- and whether
//! the rescale, the cap rule and the flags survive the move. Those are the
//! parts a port drops silently.

use erouter_solve::calibrate::{calibrate, Flag};

/// One arc's re-fit, as `_recalibrate` performs it.
pub struct Refit {
    pub a: f64,
    pub b: f64,
    pub cap: f64,
    pub clamped: bool,
    pub convex_flag: bool,
    pub flag: Flag,
    pub drift: f64,
    pub eta: f64,
    pub calib_delta: f64,
}

/// `rescale` -- a fit is in raw token units, an arc is in canonical ones.
///
/// `B` carries `sigma/tau^2`, so `tau` scales it twice. Getting this wrong is
/// invisible on a pair whose rates are both 1, which is most of them.
fn rescale(a: f64, b: f64, rate_in: f64, rate_out: f64) -> (f64, f64) {
    (a * rate_out / rate_in, b * rate_out / (rate_in * rate_in))
}

/// The whole stage in one crossing: every ladder in, every fit out.
pub fn recalibrate(rows: &[Input]) -> Vec<Option<Refit>> {
    rows.iter().map(|row| {
        let fit = calibrate(&row.deltas, &row.quotes, None, false,
                            row.drift_tol, None, None, row.quantum).ok()?;
        let (a, b) = rescale(fit.a, fit.b, row.rate_in, row.rate_out);
        // **A cap only ever tightens.** A fit can discover a wall the ladder
        // walked into; it cannot know about a capacity the curve does not
        // show -- a vault that answers `previewDeposit` linearly at every
        // probe size and then refuses the deposit. Assigning rather than
        // taking the minimum erased a limit read off the chain.
        let fitted = if fit.cap.is_finite() {
            fit.cap * row.rate_in
        } else {
            f64::INFINITY
        };
        Some(Refit {
            a,
            b,
            cap: row.cap_before.min(fitted),
            clamped: fit.clamped,
            convex_flag: fit.convex_flag,
            flag: fit.flag,
            drift: fit.drift,
            eta: fit.eta,
            calib_delta: fit.calib_delta,
        })
    }).collect()
}

pub struct Input {
    pub deltas: Vec<f64>,
    pub quotes: Vec<f64>,
    pub quantum: f64,
    pub drift_tol: f64,
    pub rate_in: f64,
    pub rate_out: f64,
    pub cap_before: f64,
}
