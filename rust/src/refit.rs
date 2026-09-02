//! Refit at the realised size and re-solve (spec §8, §7 rule 3).
//!
//! The mirror of `core/refit.py`, split at the chain.
//!
//! The first solve calibrates every arc at a *guessed* size -- `d_bar` is
//! bootstrapped from TVL before anything is known about the split. Once the
//! optimum says how much each arc actually carries, two more quotes per active
//! arc re-anchor the model there:
//!
//!     quote f(delta_p) and f(1.01 delta_p)
//!     B_p <- 2 (a_p delta_p - f(delta_p)) / delta_p^2       (M2, at the real size)
//!     recompute G, eps ; re-solve warm-started
//!     stop when max|d psi| / Psi < 1e-4
//!
//! `a` deliberately stays the tangent from the probe grid: it is what `eps` is
//! built from, it is stable, and re-deriving it from a large trade would fold
//! price impact into what is supposed to be the zero-size marginal rate.
//!
//! **The loop is the host's.** `plan` says what to quote, `apply` folds the
//! answers back, `rebuild` recomputes the arrays -- and between them the caller
//! probes and re-solves, because both of those touch the chain and the solver
//! respectively. Converges in two rounds essentially always, because `B_p`
//! varies slowly with `delta_p`.

use crate::graph::{arc_params, ceiling_conductance, ArcArrays};
use crate::nodes::NodeMap;
use crate::pools::scaled;
use crate::types::{PoolArc, Probe, TypeError};
use ruint::aliases::U256;

/// §7 rule 3's `1.01 * delta`.
pub const SLOPE_STEP: f64 = 0.01;
/// §8's stopping rule on `max|d psi| / Psi`.
pub const CONVERGED: f64 = 1e-4;

/// How much of `a * delta` the secant's numerator must be before it is a
/// measurement rather than rounding.
///
/// `B = 2(a d - f(d)) / d^2` differences two numbers that agree to within a
/// few basis points, then divides by `d^2`. `a` is a *fitted* tangent, so the
/// numerator carries an error of order `sigma_a * d` and the error in `B`
/// falls only as `1/d`. Below some size the numerator is entirely that error
/// -- including its sign, which is how a refit at a realised delta of 3 USDC
/// replaced a fit made at a million, got `B < 0`, and clamped the best pool
/// for the pair to a cap of 3 USDC.
///
/// `a` is good to something like 1e-7 relative; this asks an order of
/// magnitude of headroom on top. Failing the test is not an error: it means
/// this probe knows less than the ladder fit already in hand.
pub const SECANT_REL_FLOOR: f64 = 1e-6;

/// How far below the size an arc was last calibrated at the refit may
/// re-anchor.
///
/// The floor above catches a numerator lost in `a`'s own error. It does not
/// catch the other end of the same problem: at dust sizes the *pool's* integer
/// arithmetic stops being meaningful, and then the numerator is large in
/// relative terms while being nonsense. Measured on crvUSD/USDT at block
/// 25,769,383 -- a $45M pool, ladder-fitted `B = 4.4e-11` at a delta of 3.9M
/// -- a realised delta of 0.4 USDT quoted an output well below `a * delta`,
/// and the secant read `B = 1.73`, an implied depth of about half a dollar.
///
/// §8 exists to re-anchor a guessed size onto the realised one. Three orders
/// below the existing fit is not re-anchoring, it is extrapolating outside the
/// measured range with an error term that has grown a thousandfold. Checked
/// before the probes are planned, so it costs nothing rather than an RPC round
/// trip.
pub const REFIT_MIN_FRACTION: f64 = 1e-3;

/// What one round of the refit did.
#[derive(Debug, Clone, PartialEq)]
pub struct RefitRound {
    pub quoted: usize,
    pub reflagged: usize,
    /// Arcs whose realised size was too small for the secant to resolve their
    /// curvature; they keep the ladder's fit. See `SECANT_REL_FLOOR`.
    pub unresolved: usize,
    pub max_delta_psi: f64,
    pub max_b_change: f64,
    pub converged: bool,
}

impl Default for RefitRound {
    fn default() -> Self {
        Self {
            quoted: 0,
            reflagged: 0,
            unresolved: 0,
            max_delta_psi: f64::INFINITY,
            max_b_change: 0.0,
            converged: false,
        }
    }
}

/// One arc's place in the round: which arc, where its two probes sit in the
/// batch, and the canonical size they were taken at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Planned {
    pub arc: usize,
    pub offset: usize,
    pub delta_canonical: f64,
}

/// What to quote, and for whom.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    pub probes: Vec<Probe>,
    pub plan: Vec<Planned>,
    /// Arcs skipped before the probes were planned -- see `REFIT_MIN_FRACTION`.
    pub unresolved: usize,
}

fn probe_pair(arc: &PoolArc, delta_raw: u128) -> Result<Vec<Probe>, TypeError> {
    let bumped = ((delta_raw as f64) * (1.0 + SLOPE_STEP)) as u128;
    Ok(vec![
        Probe::new(arc.pool.clone(), arc.kind, arc.i, arc.j, arc.n_coins, delta_raw)?,
        Probe::new(
            arc.pool.clone(), arc.kind, arc.i, arc.j, arc.n_coins,
            bumped.max(delta_raw + 1),
        )?,
    ])
}

/// Which arcs to re-quote, and at what sizes.
///
/// Nothing here touches the chain: the caller takes `plan.probes` to whatever
/// quoter it has and brings the answers back to `apply`.
pub fn plan(
    arcs: &[PoolArc],
    psi: &[f64],
    nu: &[f64],
    nodes: &NodeMap,
) -> Result<Plan, TypeError> {
    let mut out = Plan::default();
    for k in 0..psi.len() {
        if psi[k] <= 0.0 {
            continue;
        }
        let arc = &arcs[k];
        let delta_canonical = psi[k] / nu[arc.tau];
        let delta_token = delta_canonical / nodes.rate(&arc.token_in);
        let raw = delta_token * scaled(U256::from(10u64).pow(U256::from(arc.decimals_in)), 0);
        if !(raw >= 1.0) {
            continue;
        }
        let delta_raw = raw as u128;
        if arc.calib_delta > 0.0
            && delta_canonical < REFIT_MIN_FRACTION * arc.calib_delta
        {
            // Outside the range this arc was measured over. Keep the fit it
            // already has, and do not spend the two probes finding that out.
            out.unresolved += 1;
            continue;
        }
        out.plan.push(Planned { arc: k, offset: out.probes.len(), delta_canonical });
        out.probes.extend(probe_pair(arc, delta_raw)?);
    }
    Ok(out)
}

/// One probe's answer: what the pool said, or that it refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Answer {
    pub ok: bool,
    pub value: u128,
}

/// Fold the answers back onto the arcs. Returns `(quoted, reflagged,
/// unresolved)`, the last including whatever `plan` already skipped.
pub fn apply(
    arcs: &mut [PoolArc],
    planned: &Plan,
    answers: &[Answer],
    nodes: &NodeMap,
) -> (usize, usize, usize) {
    let mut quoted = 0usize;
    let mut reflagged = 0usize;
    let mut unresolved = planned.unresolved;

    for entry in &planned.plan {
        let arc = &mut arcs[entry.arc];
        let (at_delta, at_bumped) = (answers[entry.offset], answers[entry.offset + 1]);
        if !at_delta.ok || at_delta.value == 0 {
            continue;
        }
        let rate_out = nodes.rate(&arc.token_out);
        let scale_out = scaled(U256::from(10u64).pow(U256::from(arc.decimals_out)), 0);
        let f_delta = scaled(U256::from(at_delta.value), 0) / scale_out * rate_out;
        // (M2) at the realised size: match the true curve at 0 and at delta.
        let signal = arc.a * entry.delta_canonical - f_delta;
        if signal.abs() <= SECANT_REL_FLOOR * arc.a * entry.delta_canonical {
            // Below what this secant can resolve. The ladder fit already on
            // the arc was made at a size where the curvature was measurable,
            // so leave it alone rather than replacing a measurement with its
            // own rounding error.
            unresolved += 1;
            continue;
        }
        let mut b = 2.0 * signal / (entry.delta_canonical * entry.delta_canonical);
        quoted += 1;

        if at_bumped.ok && at_bumped.value > at_delta.value {
            let f_bumped = scaled(U256::from(at_bumped.value), 0) / scale_out * rate_out;
            let slope = (f_bumped - f_delta) / (entry.delta_canonical * SLOPE_STEP);
            // The marginal rate must fall with size. If it rises here, the arc
            // has increasing returns exactly where the route wants to use it
            // -- inadmissible in a convex program, so clamp and flag (§11.2).
            //
            // Compared against the same floor: `a` is fitted, so a slope above
            // it by less than `a`'s own accuracy is not increasing returns. A
            // 1e-9 tolerance here was two orders tighter than `a` is true, and
            // every arc that tripped it was clamped on the strength of it.
            if slope > arc.a * (1.0 + SECANT_REL_FLOOR) {
                b = 0.0;
                if !arc.convex_flag {
                    reflagged += 1;
                }
                arc.convex_flag = true;
            }
        }

        if b <= 0.0 {
            // The zero-curvature limit, with the mandatory cap (§2.3 rule 2).
            arc.b = 0.0;
            arc.clamped = true;
            arc.convex_flag = true;
            arc.cap = arc.cap.min(entry.delta_canonical);
        } else {
            arc.b = b;
            arc.clamped = false;
        }
        arc.calib_delta = entry.delta_canonical;
    }
    (quoted, reflagged, unresolved)
}

/// Recompute G and eps in place after a refit, keeping the same indexing.
pub fn rebuild(g: &mut ArcArrays, arcs: &[PoolArc], nu: &[f64]) {
    let a: Vec<f64> = arcs.iter().map(|arc| arc.a).collect();
    let b: Vec<f64> = arcs.iter().map(|arc| arc.b).collect();
    // `arc_params` refuses a negative curvature, which `apply` cannot produce
    // -- it clamps to zero -- so this cannot fail from here.
    let Ok((mut conductance, eps)) = arc_params(&g.tau, &g.sig, &a, &b, nu) else {
        return;
    };
    let flagged: Vec<bool> = arcs.iter().map(|arc| arc.convex_flag).collect();
    ceiling_conductance(&mut conductance, &flagged, crate::graph::CEILING_FACTOR);
    g.a = a;
    g.b = b;
    g.g = conductance.iter().map(|v| v / g.g_scale).collect();
    g.eps = eps;
    g.clamped = arcs.iter().map(|arc| arc.clamped).collect();
    g.flagged = flagged;
    g.cap = arcs
        .iter()
        .map(|arc| {
            if arc.cap.is_finite() {
                nu[arc.tau] * arc.cap / g.g_scale
            } else {
                f64::INFINITY
            }
        })
        .collect();
}

/// How far the flow moved, and whether that is close enough to stop.
pub fn round_stats(before: &[f64], after: &[f64], psi_total: f64) -> (f64, bool) {
    let moved = if psi_total > 0.0 {
        (0..before.len())
            .map(|k| (after[k] - before[k]).abs())
            .fold(0.0f64, f64::max)
            / psi_total
    } else {
        0.0
    };
    (moved, moved < CONVERGED)
}

/// The largest relative move in `B` across a round.
pub fn b_change(before: &[f64], after: &[f64]) -> f64 {
    // Zipped, not indexed: the reference subtracts two numpy arrays and would
    // refuse a mismatch outright, which is what the binding does.
    before
        .iter()
        .zip(after.iter())
        .map(|(b, a)| (a - b).abs() / b.abs().max(1e-30))
        .fold(0.0f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ArcKind;

    fn arc(a: f64, b: f64, calib: f64) -> PoolArc {
        let mut built = PoolArc::new(
            "id".into(), "0xpool".into(), ArcKind::SwapStable, 0, 1, 2,
            "0xin".into(), "0xout".into(), 0, 1,
        );
        built.a = a;
        built.b = b;
        built.calib_delta = calib;
        built.decimals_in = 18;
        built.decimals_out = 18;
        built
    }

    fn nodes() -> NodeMap {
        let mut map = NodeMap::new();
        map.add_token("0xin", "IN", 18);
        map.add_token("0xout", "OUT", 18);
        map
    }

    #[test]
    fn each_arc_gets_two_probes_one_percent_apart() {
        let arcs = [arc(1.0, 1e-6, 0.0)];
        let got = plan(&arcs, &[1000.0], &[1.0, 1.0], &nodes()).unwrap();
        assert_eq!(got.probes.len(), 2);
        assert_eq!(got.plan[0].arc, 0);
        assert!((got.plan[0].delta_canonical - 1000.0).abs() < 1e-9);
        let (small, big) = (got.probes[0].dx, got.probes[1].dx);
        assert!(big > small);
        assert!(((big as f64) / (small as f64) - 1.01).abs() < 1e-9);
    }

    #[test]
    fn a_size_far_below_the_existing_fit_is_not_re_anchored() {
        // The crvUSD/USDT case: a fit made at 3.9M, a realised delta of 0.4.
        let arcs = [arc(1.0, 4.4e-11, 3.9e6)];
        let got = plan(&arcs, &[0.4], &[1.0, 1.0], &nodes()).unwrap();
        assert!(got.probes.is_empty());
        assert_eq!(got.unresolved, 1);
    }

    #[test]
    fn a_numerator_lost_in_the_tangents_own_error_leaves_the_fit_alone() {
        let mut arcs = [arc(1.0, 4.4e-11, 1.0)];
        let planned = plan(&arcs, &[1000.0], &[1.0, 1.0], &nodes()).unwrap();
        // `f(delta)` exactly `a * delta`: the secant has nothing to say.
        let unit = 10u128.pow(18);
        let answers = [
            Answer { ok: true, value: 1000 * unit },
            Answer { ok: true, value: 1010 * unit },
        ];
        let (quoted, _, unresolved) = apply(&mut arcs, &planned, &answers, &nodes());
        assert_eq!(quoted, 0);
        assert_eq!(unresolved, 1);
        assert_eq!(arcs[0].b, 4.4e-11, "the ladder's fit stands");
    }

    #[test]
    fn a_measurable_curvature_re_anchors_b() {
        let mut arcs = [arc(1.0, 1e-30, 1.0)];
        let planned = plan(&arcs, &[1000.0], &[1.0, 1.0], &nodes()).unwrap();
        // 1% below `a * delta` at 1000: B = 2 * 10 / 1e6 = 2e-5.
        let unit = 10u128.pow(18);
        let answers = [
            Answer { ok: true, value: 990 * unit },
            Answer { ok: true, value: 999 * unit },
        ];
        let (quoted, reflagged, _) = apply(&mut arcs, &planned, &answers, &nodes());
        assert_eq!((quoted, reflagged), (1, 0));
        assert!((arcs[0].b - 2e-5).abs() < 1e-12, "{}", arcs[0].b);
        assert!(!arcs[0].clamped);
        assert_eq!(arcs[0].calib_delta, 1000.0);
    }

    #[test]
    fn increasing_returns_at_the_realised_size_are_clamped_and_flagged() {
        let mut arcs = [arc(1.0, 1e-6, 1.0)];
        let planned = plan(&arcs, &[1000.0], &[1.0, 1.0], &nodes()).unwrap();
        // The bumped quote pays *more* per unit than `a`: a rising marginal
        // rate, which a convex program cannot admit.
        let unit = 10u128.pow(18);
        let answers = [
            Answer { ok: true, value: 990 * unit },
            Answer { ok: true, value: 1005 * unit },
        ];
        let (quoted, reflagged, _) = apply(&mut arcs, &planned, &answers, &nodes());
        assert_eq!((quoted, reflagged), (1, 1));
        assert_eq!(arcs[0].b, 0.0);
        assert!(arcs[0].clamped && arcs[0].convex_flag);
        // The zero-curvature limit carries its mandatory cap.
        assert_eq!(arcs[0].cap, 1000.0);
    }

    #[test]
    fn a_refused_probe_leaves_its_arc_untouched() {
        let mut arcs = [arc(1.0, 1e-6, 1.0)];
        let planned = plan(&arcs, &[1000.0], &[1.0, 1.0], &nodes()).unwrap();
        let answers = [Answer { ok: false, value: 0 }, Answer { ok: false, value: 0 }];
        assert_eq!(apply(&mut arcs, &planned, &answers, &nodes()), (0, 0, 0));
        assert_eq!(arcs[0].b, 1e-6);
    }

    #[test]
    fn convergence_is_a_move_relative_to_the_trade() {
        assert_eq!(round_stats(&[1.0], &[1.0], 1.0), (0.0, true));
        let (moved, converged) = round_stats(&[1.0], &[1.5], 1.0);
        assert_eq!(moved, 0.5);
        assert!(!converged);
        assert_eq!(round_stats(&[1.0], &[2.0], 0.0), (0.0, true));
    }
}
