//! The reference frame: what everything is worth, in one numeraire.
//!
//! The mirror of `core/prices.py`.
//!
//! §4 fits log-prices by weighted least squares over the arcs, which is the
//! same Laplacian again -- `nu` is the potential whose differences best match
//! the measured marginal rates. It matters far more than it looks: `nu` sets
//! `eps` and `G` for *every* arc, so one arc claiming WETH is worth 296 crvUSD
//! drags the whole frame, not just itself.

use crate::graph::{component_of, laplacian};

/// A pool's two directions must agree on the price to within fees: `a_f * a_r`
/// is `Gamma_live^2` (§2.6), just under 1 for any symmetric-fee CFMM.
///
/// Far below it means the two probes landed in different regimes and neither
/// `a` is a marginal rate -- mainnet LLAMMA markets, banded and quoting from
/// whichever band is live, measure 0.000616 and 0.002516 against 0.999+ for
/// every ordinary pool. Such an arc is still perfectly routable; it just must
/// not vote on what anything is worth.
pub const ROUND_TRIP_FLOOR: f64 = 0.5;

/// Muted, not removed: `reference_prices` requires strictly positive weights,
/// and a vanishing one is the same thing numerically while keeping the arc in
/// the system of equations that connects the graph.
pub const MUTED_WEIGHT: f64 = 1e-12;

/// What the reference raises as `ValueError` or `SingularSystem`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriceError(pub String);

/// Mute arcs whose own reverse direction contradicts them.
///
/// `keys[k]` is `(pool, i, j)` and the partner is `(pool, j, i)`. Pairing on
/// node indices instead is wrong and quietly so: a dozen pools join USDC and
/// USDT, so a healthy arc gets matched against some *other* pool's reverse and
/// muted for its neighbour's sins.
///
/// Adding 11 LLAMMA markets moved crvUSD 36% and USDC 27% away from parity
/// before this existed.
pub fn price_fit_weights(keys: &[(String, i32, i32)], a: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = w.to_vec();
    for k in 0..keys.len() {
        if a[k] <= 0.0 {
            continue;
        }
        let (pool, i, j) = &keys[k];
        // The partner's *first* entry, which is what a dict built by
        // enumeration keeps -- a later duplicate does not replace it.
        let back = keys
            .iter()
            .enumerate()
            .find(|(at, (p, bi, bj))| a[*at] > 0.0 && p == pool && bi == j && bj == i)
            .map(|(at, _)| a[at]);
        if let Some(back) = back {
            if a[k] * back < ROUND_TRIP_FLOOR {
                out[k] = MUTED_WEIGHT;
            }
        }
    }
    out
}

/// Fit `nu` with `nu[numeraire] == 1`.
///
/// `a` must be strictly positive. A zero marginal rate is not a cheap pool but
/// a broken probe: `log 0` would NaN the whole reference frame, and 6% of arcs
/// fail their smallest probe on mainnet.
pub fn reference_prices(
    tau: &[i64],
    sig: &[i64],
    a: &[f64],
    w: &[f64],
    n_nodes: usize,
    numeraire: usize,
) -> Result<Vec<f64>, PriceError> {
    if !a.is_empty() && !a.iter().all(|&v| v > 0.0) {
        let bad = argmin(a);
        return Err(PriceError(format!(
            "arc {bad} has a={}; reference prices need a > 0 \
             (a failed probe must drop the arc, not enter the fit as zero)",
            crate::pyfmt::float(a[bad])
        )));
    }
    if !w.is_empty() && !w.iter().all(|&v| v > 0.0) {
        return Err(PriceError("reference-price weights must be positive".into()));
    }
    if tau.is_empty() {
        return Ok(vec![1.0; n_nodes]);
    }

    let mut z = vec![0.0f64; n_nodes];
    // Only the numeraire's component is determined; anything disconnected
    // keeps nu = 1 and will be dropped later for want of a route.
    let comp = component_of(numeraire, tau, sig, n_nodes);
    let keep: Vec<usize> = (0..n_nodes).filter(|&v| comp[v] && v != numeraire).collect();
    if !keep.is_empty() {
        let log_a: Vec<f64> = a.iter().map(|v| v.ln()).collect();
        let mut rhs = vec![0.0f64; n_nodes];
        // Two sweeps, in the reference's order: `np.subtract.at` then
        // `np.add.at`. Float addition is not associative and these overlap.
        for p in 0..tau.len() {
            rhs[sig[p] as usize] -= w[p] * log_a[p];
        }
        for p in 0..tau.len() {
            rhs[tau[p] as usize] += w[p] * log_a[p];
        }
        let mut matrix = laplacian(tau, sig, w, n_nodes, &keep);
        let mut b: Vec<f64> = keep.iter().map(|&v| rhs[v]).collect();
        crate::lu::solve_in_place(&mut matrix, &mut b, keep.len())
            .map_err(|_| PriceError("reference-price fit is singular".into()))?;
        for (k, &v) in keep.iter().enumerate() {
            z[v] = b[k];
        }
    }
    Ok(z.into_iter().map(f64::exp).collect())
}

/// Residuals `r_p = z_sig - z_tau + log a_p`.
///
/// Large `|r_p|` flags a stale pool or a genuine arbitrage, so these are worth
/// surfacing rather than discarding.
pub fn dislocations(tau: &[i64], sig: &[i64], a: &[f64], nu: &[f64]) -> Vec<f64> {
    (0..tau.len())
        .map(|p| nu[sig[p] as usize].ln() - nu[tau[p] as usize].ln() + a[p].ln())
        .collect()
}

/// Fee-free mid price implied by the two one-sided quotes.
pub fn pool_mid(a_forward: f64, a_reverse: f64) -> f64 {
    (a_forward / a_reverse).sqrt()
}

/// Measured effective retention, `sqrt(a_f * a_r)` (§2.6).
///
/// Reads the pool's *current* fee off two tiny probes -- no fee parameters, no
/// ABI knowledge of the fee law. For a fixed-fee pool it must equal `1 - fee`
/// to full precision, so a deviation means the probe pipeline is broken; for a
/// dynamic-fee pool it is the live value and its drift is observable.
pub fn gamma_live(a_forward: f64, a_reverse: f64) -> f64 {
    (a_forward * a_reverse).sqrt()
}

/// Indices where `eps_f + eps_r <= tol` -- a spurious negative 2-cycle.
///
/// Round-tripping a pool always loses (`a_f a_r = Gamma^2 < 1`), but the
/// *linearised* drops are frame-dependent and their sum falls below zero when
/// `nu` is far enough off for that pair. Violation means `nu` is inconsistent
/// with the pool, not that arbitrage exists.
pub fn check_pair_drops(eps_forward: &[f64], eps_reverse: &[f64], tol: f64) -> Vec<usize> {
    (0..eps_forward.len())
        .filter(|&k| eps_forward[k] + eps_reverse[k] <= tol)
        .collect()
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (k, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return k;
        }
        if v < values[best] {
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_of_rates_composes_into_one_frame() {
        // 0 -> 1 at 2.0 and 1 -> 2 at 3.0, so node 2 is worth 6 of node 0.
        let nu = reference_prices(&[0, 1], &[1, 2], &[2.0, 3.0], &[1.0, 1.0], 3, 0)
            .unwrap();
        assert!((nu[0] - 1.0).abs() < 1e-12);
        assert!((nu[0] / nu[1] - 2.0).abs() < 1e-9, "{nu:?}");
        assert!((nu[0] / nu[2] - 6.0).abs() < 1e-9, "{nu:?}");
    }

    #[test]
    fn a_disconnected_node_keeps_parity() {
        let nu = reference_prices(&[0], &[1], &[2.0], &[1.0], 3, 0).unwrap();
        assert_eq!(nu[2], 1.0);
    }

    #[test]
    fn a_failed_probe_must_drop_the_arc_rather_than_enter_as_zero() {
        let err = reference_prices(&[0], &[1], &[0.0], &[1.0], 2, 0).unwrap_err();
        assert!(err.0.contains("reference prices need a > 0"));
        let err = reference_prices(&[0], &[1], &[1.0], &[0.0], 2, 0).unwrap_err();
        assert!(err.0.contains("weights must be positive"));
        // No arcs at all is parity, not an error.
        assert_eq!(reference_prices(&[], &[], &[], &[], 3, 0).unwrap(), vec![1.0; 3]);
    }

    #[test]
    fn an_arc_its_own_reverse_contradicts_is_muted() {
        // A banded market: the round trip is nowhere near 1.
        let keys = vec![
            ("0xllamma".to_string(), 0, 1),
            ("0xllamma".to_string(), 1, 0),
            ("0xhealthy".to_string(), 0, 1),
            ("0xhealthy".to_string(), 1, 0),
        ];
        let a = [0.0006, 0.0025, 0.9995, 0.9994];
        let w = [1.0, 1.0, 1.0, 1.0];
        let got = price_fit_weights(&keys, &a, &w);
        assert_eq!(got[0], MUTED_WEIGHT);
        assert_eq!(got[1], MUTED_WEIGHT);
        assert_eq!(got[2], 1.0);
        assert_eq!(got[3], 1.0);
    }

    #[test]
    fn a_healthy_arc_is_not_muted_for_its_neighbours_sins() {
        // Two pools joining the same pair. Pairing on nodes rather than on the
        // pool would mute the healthy one against the banded one's reverse.
        let keys = vec![
            ("0xllamma".to_string(), 0, 1),
            ("0xhealthy".to_string(), 1, 0),
        ];
        let a = [0.0006, 0.9994];
        assert_eq!(price_fit_weights(&keys, &a, &[1.0, 1.0]), vec![1.0, 1.0]);
    }

    #[test]
    fn the_round_trip_reads_the_fee_without_a_fee_parameter() {
        // A 4 bp symmetric fee: both directions retain 0.9996.
        assert!((gamma_live(0.9996, 0.9996) - 0.9996).abs() < 1e-15);
        // And the mid is 1 when the two sides agree.
        assert!((pool_mid(0.9996, 0.9996) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_spurious_negative_two_cycle_is_reported_by_index() {
        assert_eq!(check_pair_drops(&[0.001, 0.001], &[-0.002, 0.001], 0.0), vec![0]);
        assert_eq!(check_pair_drops(&[0.001], &[0.001], 0.0), Vec::<usize>::new());
    }

    #[test]
    fn a_dislocation_is_zero_where_the_frame_fits() {
        let nu = reference_prices(&[0, 1], &[1, 2], &[2.0, 3.0], &[1.0, 1.0], 3, 0)
            .unwrap();
        let got = dislocations(&[0, 1], &[1, 2], &[2.0, 3.0], &nu);
        assert!(got.iter().all(|v| v.abs() < 1e-9), "{got:?}");
    }
}
