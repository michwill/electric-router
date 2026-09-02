//! Value-coordinate graph assembly (spec §3.1, §9.5-9.7).
//!
//! The mirror of `core/graph.py`. Everything here is struct-of-arrays already
//! on the Python side, which is why this is the one module that ports rather
//! than gets rewritten: the reference never had an object to lose.
//!
//! Working in *value* rather than token units is what makes the dual Hessian a
//! plain graph Laplacian instead of a gain-graph one, and it is also why arc
//! conductance is direction-symmetric while `a` and `B` are wildly asymmetric.

use crate::pyfmt::sci;
use std::fmt;

/// §9.6 an arc that cannot carry meaningful flow only adds pivots
pub const DUST_FLOOR: f64 = 1e-6;
/// §9.7 clamped (B=0) arcs would otherwise carry G = inf
pub const CEILING_FACTOR: f64 = 1e3;
pub const MAX_CONDITION: f64 = 1e12;
/// What the adaptive dust floor aims at, with headroom below MAX_CONDITION.
pub const TARGET_CONDITION: f64 = 1e11;
/// Beyond this, the spread is not a wide universe -- it is a bug.
pub const PATHOLOGICAL_CONDITION: f64 = 1e15;

/// The smallest the scaled demand may become. See `scale`.
pub const MIN_SCALED_PSI: f64 = 1e-6;

/// A refusal that the reference raises as `ValueError`. The text is part of
/// the mirror: `tests/test_graph_conditioning.py` matches on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphError(pub String);

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

type Result<T> = std::result::Result<T, GraphError>;

/// Why an arc is not in the solved index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dropped {
    Dust,
    Merged,
}

impl Dropped {
    pub fn name(self) -> &'static str {
        match self {
            Dropped::Dust => "DUST",
            Dropped::Merged => "MERGED",
        }
    }
}

/// (M3) conductance and (M4) forward drop, in value coordinates.
///
///     G_p = nu_tau * a_p / B_p        value scale x token-space conductance
///     eps_p = 1 - a_p * nu_sig / nu_tau
///
/// `eps` may be negative: that is a favourably dislocated pool, an EMF, and it
/// is exactly how arbitrage enters the routing problem.
///
/// B == 0 is the admissible zero-curvature limit (§2.3), giving G = inf here.
/// B < 0 is *not* admissible and is rejected loudly rather than turned into a
/// negative resistor.
pub fn arc_params(
    tau: &[i64],
    sig: &[i64],
    a: &[f64],
    b: &[f64],
    nu: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if let Some(bad) = argmin_negative(b) {
        return Err(GraphError(format!(
            "negative curvature reached the graph (arc {bad}, B={}). \
             calibrate() must clamp B to 0; a negative G makes the Laplacian \
             indefinite and voids the certificate (§11.2).",
            sci(b[bad], 3)
        )));
    }
    let m = tau.len();
    let mut g = Vec::with_capacity(m);
    let mut eps = Vec::with_capacity(m);
    for p in 0..m {
        let nu_tau = nu[tau[p] as usize];
        let nu_sig = nu[sig[p] as usize];
        g.push(if b[p] > 0.0 { nu_tau * a[p] / b[p] } else { f64::INFINITY });
        eps.push(1.0 - a[p] * nu_sig / nu_tau);
    }
    Ok((g, eps))
}

/// `np.argmin(B)` if any B is negative. NaN never compares less, which mirrors
/// `np.any(B < 0)` refusing to see it.
fn argmin_negative(b: &[f64]) -> Option<usize> {
    if !b.iter().any(|&v| v < 0.0) {
        return None;
    }
    // numpy's argmin over a NaN-free-enough array: first index of the minimum.
    let mut best = 0usize;
    for (k, &v) in b.iter().enumerate() {
        if v.is_nan() {
            return Some(k); // numpy propagates NaN to argmin
        }
        if v < b[best] {
            best = k;
        }
    }
    Some(best)
}

/// §2.3 rule (3) / §9.7 -- clamp in G-space, never by flooring B.
///
/// Only clamped arcs (`G = inf`) are normally affected: every finite arc is by
/// definition at or below the maximum. Do **not** lower this ceiling to bound
/// the condition number -- flattening real conductances makes every deep pool
/// look identical. Bound the spread from below instead (see `build`).
pub fn ceiling_conductance(g: &mut [f64], flagged: &[bool], factor: f64) {
    let mut reference = f64::NEG_INFINITY;
    let mut any = false;
    for (k, &v) in g.iter().enumerate() {
        // `get`: the binding refuses a mismatch, and `build` passes two fields
        // of one struct, so an unflagged default is only ever the fallback.
        if v.is_finite() && !flagged.get(k).copied().unwrap_or(false) {
            any = true;
            if v > reference {
                reference = v;
            }
        }
    }
    let ceiling = factor * if any { reference } else { 1.0 };
    for v in g.iter_mut() {
        // `np.where(isfinite(G), G, inf)` first: -inf and NaN both become inf
        // and then take the ceiling, which is not what a bare `min` would do.
        let w = if v.is_finite() { *v } else { f64::INFINITY };
        *v = w.min(ceiling);
    }
}

/// Solver input. Index space is post-dust, post-duplicate-merge.
#[derive(Debug, Clone, Default)]
pub struct ArcArrays {
    pub tau: Vec<i64>,
    pub sig: Vec<i64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub g: Vec<f64>,
    pub eps: Vec<f64>,
    pub cap: Vec<f64>,
    pub flagged: Vec<bool>,
    pub clamped: Vec<bool>,
    pub n_nodes: usize,
    pub g_scale: f64,
    /// Set when the dust floor had to be backed off for connectivity and the
    /// resulting spread exceeds MAX_CONDITION. Non-fatal, but worth surfacing.
    pub ill_conditioned: f64,
    /// index -> original arc indices (a merged duplicate group has several)
    pub sources: Vec<Vec<usize>>,
    /// Original index -> why it is gone, in the order the reference's dict
    /// gained its keys.
    pub dropped: Vec<(usize, Dropped)>,
}

impl ArcArrays {
    pub fn m(&self) -> usize {
        self.tau.len()
    }

    pub fn condition(&self) -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut any = false;
        for &v in &self.g {
            if v > 0.0 {
                any = true;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if any {
            hi / lo
        } else {
            1.0
        }
    }
}

/// Everything `build` takes past the five arrays and the demand.
pub struct BuildOptions<'a> {
    pub cap: Option<&'a [f64]>,
    pub flagged: Option<&'a [bool]>,
    pub clamped: Option<&'a [bool]>,
    pub n_nodes: Option<usize>,
    pub dust_floor: f64,
    pub ceiling_factor: f64,
    pub merge_duplicates: bool,
    /// `(src, dst)`: the pair the dust floor must not disconnect.
    pub require: Option<(usize, usize)>,
}

impl Default for BuildOptions<'_> {
    fn default() -> Self {
        Self {
            cap: None,
            flagged: None,
            clamped: None,
            n_nodes: None,
            dust_floor: DUST_FLOOR,
            ceiling_factor: CEILING_FACTOR,
            merge_duplicates: true,
            require: None,
        }
    }
}

/// Assemble solver arrays, in the order §9.5-9.7 requires.
///
/// Dust first (so the ceiling reference is meaningful), then duplicate merge,
/// then the conductance ceiling, then the invariants.
pub fn build(
    tau: &[i64],
    sig: &[i64],
    a: &[f64],
    b: &[f64],
    nu: &[f64],
    psi: f64,
    opts: &BuildOptions<'_>,
) -> Result<ArcArrays> {
    let m = tau.len();
    let n = match opts.n_nodes {
        Some(v) => v,
        None => {
            let hi = tau.iter().chain(sig.iter()).copied().max().unwrap_or(-1);
            (hi + 1) as usize
        }
    };

    let mut cap: Vec<f64> = match opts.cap {
        Some(v) => v.to_vec(),
        None => vec![f64::INFINITY; m],
    };
    let flagged: Vec<bool> = match opts.flagged {
        Some(v) => v.to_vec(),
        None => vec![false; m],
    };
    let clamped: Vec<bool> = match opts.clamped {
        Some(v) => v.to_vec(),
        None => b.iter().map(|&v| v == 0.0).collect(),
    };

    // §12.4: a zero-curvature arc has no self-limiting term, so without a
    // finite cap a negative-eps cycle gives unbounded flow. Fail here, not in
    // the solve.
    let unbounded: Vec<usize> = (0..m)
        .filter(|&k| clamped[k] && !cap[k].is_finite())
        .collect();
    if !unbounded.is_empty() {
        return Err(GraphError(format!(
            "clamped arcs {} have no finite cap; flow would be unbounded (§2.3 rule 2)",
            format_index_list(&unbounded)
        )));
    }

    let (mut g, eps) = arc_params(tau, sig, a, b, nu)?;

    let mut sources: Vec<Vec<usize>> = (0..m).map(|k| vec![k]).collect();
    let mut dropped: Vec<(usize, Dropped)> = Vec::new();

    // --- §9.6 dust ------------------------------------------------------
    //
    // §9.6's floor is `1e-6 * Psi`. On a real universe that is not enough on
    // its own: genuine conductances span 4e10 on Ethereum, so `max/min` can
    // breach §12.4's 1e12 bound without anything being wrong.
    //
    // Raise the floor rather than lower the ceiling.
    let base_floor = opts.dust_floor * psi;
    let mut floor = base_floor;

    let mut finite_max = f64::NEG_INFINITY;
    let mut any_finite = false;
    let mut any_infinite = false;
    // Measured over the arcs that will survive the floor, not over every arc:
    // a dust pool almost entirely on one side genuinely quotes a huge rate,
    // and it is about to be dropped anyway.
    let mut usable_lo = f64::INFINITY;
    let mut usable_hi = f64::NEG_INFINITY;
    let mut usable_n = 0usize;
    for &v in &g {
        if v.is_finite() {
            any_finite = true;
            if v > finite_max {
                finite_max = v;
            }
            if v > 0.0 && v >= base_floor {
                usable_n += 1;
                usable_lo = usable_lo.min(v);
                usable_hi = usable_hi.max(v);
            }
        } else {
            // `(~np.isfinite(G)).any()` -- NaN counts, the same as an inf.
            any_infinite = true;
        }
    }
    if usable_n > 1 {
        let raw_spread = usable_hi / usable_lo;
        if raw_spread > PATHOLOGICAL_CONDITION {
            // No real universe looks like this: the widest genuine spread
            // measured on Ethereum is ~4e10. A spread of 1e15+ means B was
            // floored instead of G being ceilinged.
            return Err(GraphError(format!(
                "max(G)/min(G) = {} before flooring; \
                 something is being clamped in the wrong space (§9.7)",
                sci(raw_spread, 3)
            )));
        }
    }
    if any_finite {
        // Aim at the spread that will exist *after* the ceiling runs: a
        // clamped arc is lifted to `ceiling_factor * max`, so budgeting
        // against the pre-ceiling maximum alone leaves the assertion tripping
        // on real data.
        let mut top = finite_max;
        if any_infinite {
            top *= opts.ceiling_factor;
        }
        floor = floor.max(top / TARGET_CONDITION);
    }

    // Conditioning must never cost connectivity: back the floor off until the
    // nodes we have to route between are still joined. A badly conditioned
    // solve is recoverable; a graph with no path is not.
    let mut keep: Vec<bool> = g.iter().map(|&v| !(v < floor)).collect();
    let mut backed_off = false;
    if let Some((src, dst)) = opts.require {
        while floor > base_floor {
            let (live_tau, live_sig) = live_arcs(tau, sig, &keep);
            if component_of(dst, &live_tau, &live_sig, n)[src] {
                break;
            }
            floor /= 10.0;
            backed_off = true;
            let cut = floor.max(base_floor);
            keep = g.iter().map(|&v| !(v < cut)).collect();
        }
    }
    for (k, alive) in keep.iter().enumerate() {
        if !alive {
            dropped.push((k, Dropped::Dust));
        }
    }

    // --- §9.5 duplicates, as parallel resistors -------------------------
    if opts.merge_duplicates {
        let mut groups: std::collections::HashMap<(i64, i64, u64, u64), usize> =
            std::collections::HashMap::new();
        for k in 0..m {
            if !keep[k] {
                continue;
            }
            let key = (tau[k], sig[k], round12_bits(a[k]), round12_bits(b[k]));
            match groups.get(&key) {
                Some(&head) => {
                    g[head] += g[k];
                    cap[head] += cap[k];
                    sources[head].push(k);
                    keep[k] = false;
                    dropped.push((k, Dropped::Merged));
                }
                None => {
                    groups.insert(key, k);
                }
            }
        }
    }

    let idx: Vec<usize> = (0..m).filter(|&k| keep[k]).collect();
    let mut arrays = ArcArrays {
        tau: idx.iter().map(|&k| tau[k]).collect(),
        sig: idx.iter().map(|&k| sig[k]).collect(),
        a: idx.iter().map(|&k| a[k]).collect(),
        b: idx.iter().map(|&k| b[k]).collect(),
        g: idx.iter().map(|&k| g[k]).collect(),
        eps: idx.iter().map(|&k| eps[k]).collect(),
        cap: idx.iter().map(|&k| cap[k]).collect(),
        flagged: idx.iter().map(|&k| flagged[k]).collect(),
        clamped: idx.iter().map(|&k| clamped[k]).collect(),
        n_nodes: n,
        g_scale: 1.0,
        ill_conditioned: 0.0,
        sources: idx.iter().map(|&k| sources[k].clone()).collect(),
        dropped,
    };

    // --- §9.7 ceiling, after the dust floor -----------------------------
    ceiling_conductance(&mut arrays.g, &arrays.flagged, opts.ceiling_factor);

    // --- §12.4 invariants -----------------------------------------------
    if arrays.m() > 0 && !arrays.g.iter().all(|&v| v > 0.0) {
        let bad = argmin(&arrays.g);
        return Err(GraphError(format!(
            "arc {bad} has G={}; Laplacian would not be PSD",
            sci(arrays.g[bad], 3)
        )));
    }
    let condition = arrays.condition();
    if condition >= MAX_CONDITION {
        if !backed_off {
            return Err(GraphError(format!(
                "max(G)/min(G) = {} >= {}; \
                 something is being clamped in the wrong space (§9.7)",
                sci(condition, 3),
                sci(MAX_CONDITION, 0)
            )));
        }
        // The dust floor was lowered above precisely to keep `src` joined to
        // `dst`. Failing here would contradict the rule that produced the
        // state: a badly conditioned solve is recoverable and the §12.4 KCL
        // residual check adjudicates it, whereas a graph with no path is
        // simply no route.
        arrays.ill_conditioned = condition;
    }
    Ok(arrays)
}

/// §9.1 -- (P) is homogeneous in (G, Psi), so normalise G by its median.
///
/// `u`, `eps` and `rho` are dimensionless and unchanged; only psi rescales.
/// Uniform scaling cannot change the Laplacian's condition number, so what
/// this buys is magnitude, not conditioning -- and a normalisation that puts
/// `G` near 1 by putting `Psi` under `TOL` has bought nothing and lost the
/// route. Where the two pull apart, the demand wins.
pub fn scale(arrays: &mut ArcArrays, psi: f64) -> f64 {
    let mut positive: Vec<f64> = arrays.g.iter().copied().filter(|&v| v > 0.0).collect();
    let mut s = if positive.is_empty() { 1.0 } else { median(&mut positive) };
    if !s.is_finite() || s <= 0.0 {
        s = 1.0;
    }
    if psi > 0.0 && psi.is_finite() {
        s = s.min(psi / MIN_SCALED_PSI);
    }
    for v in arrays.g.iter_mut() {
        *v /= s;
    }
    for v in arrays.cap.iter_mut() {
        *v /= s;
    }
    arrays.g_scale = s;
    psi / s
}

/// `np.median`: the mean of the two middle elements on an even count, which is
/// not the same number as either of them and does show up in the arms.
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    values.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        values[n / 2]
    } else {
        // numpy averages as `(lo + hi) / 2`, not `lo + (hi - lo) / 2`.
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

// ------------------------------------------------------------ topology

/// L = B^T diag(G) B restricted to `keep`, assembled in O(nnz), row-major.
///
/// Built directly on the *kept* index space rather than as an n x n matrix
/// that is then sliced. The active set is usually a handful of nodes out of
/// ~300, so allocating the full matrix every pivot dominated the solve.
///
/// An arc with exactly one endpoint kept still contributes its diagonal term:
/// that is what grounds the system at `dst`.
pub fn laplacian(tau: &[i64], sig: &[i64], g: &[f64], n: usize, keep: &[usize]) -> Vec<f64> {
    let size = keep.len();
    let mut position = vec![-1i64; n];
    for (slot, &node) in keep.iter().enumerate() {
        position[node] = slot as i64;
    }
    let mut matrix = vec![0.0f64; size * size];
    // Pass by pass, in the reference's order. `np.add.at` accumulates every
    // head contribution before any tail one, and float addition is not
    // associative: folding them per-arc instead gives a diagonal that differs
    // in its last bits, which a differential test reads as a port bug because
    // that is exactly what it would be.
    for p in 0..tau.len() {
        let head = position[tau[p] as usize];
        if head >= 0 {
            let h = head as usize;
            matrix[h * size + h] += g[p];
        }
    }
    for p in 0..sig.len() {
        let tail = position[sig[p] as usize];
        if tail >= 0 {
            let t = tail as usize;
            matrix[t * size + t] += g[p];
        }
    }
    // Four passes, not three: the reference writes `(head, tail)` for every
    // arc and only then `(tail, head)`. Where a node pair carries arcs in both
    // directions the two orders differ -- `[h][t]` gains p1, p3, p2 there and
    // p1, p2, p3 here -- and the sum lands on a different last bit.
    for p in 0..tau.len() {
        let head = position[tau[p] as usize];
        let tail = position[sig[p] as usize];
        if head >= 0 && tail >= 0 {
            matrix[head as usize * size + tail as usize] -= g[p];
        }
    }
    for p in 0..tau.len() {
        let head = position[tau[p] as usize];
        let tail = position[sig[p] as usize];
        if head >= 0 && tail >= 0 {
            matrix[tail as usize * size + head as usize] -= g[p];
        }
    }
    matrix
}

/// Nodes reachable from `root` over the given (undirected) arcs.
///
/// §9.4: `L_A > 0` iff every free node connects to `dst` through the active
/// set. §14's reference listing deletes only `dst`, which produces a singular
/// factorisation the first time a pivot orphans a leaf -- so this is
/// recomputed every pivot rather than once.
pub fn component_of(root: usize, tau: &[i64], sig: &[i64], n: usize) -> Vec<bool> {
    let mut seen = vec![false; n];
    // A root that is not a node reaches nothing, which is what the empty mask
    // says.  `pipeline::restrict_to_component` reaches this with a `dst_node`
    // it took from a caller, so the guard belongs here and not at one binding.
    if root >= n {
        return seen;
    }
    seen[root] = true;
    loop {
        let mut grew = false;
        for p in 0..tau.len() {
            let (h, t) = (tau[p] as usize, sig[p] as usize);
            // An arc naming a node the mask does not cover joins nothing in
            // it.  A negative `tau` wraps to a huge `usize`, which this
            // catches too.
            if h >= n || t >= n {
                continue;
            }
            if seen[h] != seen[t] {
                seen[h] = true;
                seen[t] = true;
                grew = true;
            }
        }
        if !grew {
            return seen;
        }
    }
}

fn live_arcs(tau: &[i64], sig: &[i64], keep: &[bool]) -> (Vec<i64>, Vec<i64>) {
    let mut lt = Vec::new();
    let mut ls = Vec::new();
    for k in 0..tau.len() {
        if keep[k] {
            lt.push(tau[k]);
            ls.push(sig[k]);
        }
    }
    (lt, ls)
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

/// `list(np.flatnonzero(...))` as Python prints it.
fn format_index_list(indices: &[usize]) -> String {
    let inner: Vec<String> = indices.iter().map(|k| k.to_string()).collect();
    format!("[{}]", inner.join(", "))
}

/// `round(x, 12)` as a comparable key.
///
/// The reference groups duplicates on `round(float(a), 12)`, which is decimal
/// rounding, ties to even -- not anything `f64` does natively. Formatting to
/// twelve places and reading the result back is the same operation CPython
/// performs, and it agrees on the ties. The bits rather than the float so the
/// key can be hashed, and so `-0.0` and `0.0` do not land in two buckets.
fn round12_bits(x: f64) -> u64 {
    if !x.is_finite() {
        return x.to_bits();
    }
    let rounded: f64 = format!("{x:.12}").parse().unwrap_or(x);
    if rounded == 0.0 {
        0.0f64.to_bits()
    } else {
        rounded.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductance_is_tvl_over_four_for_constant_product() {
        // a = 1, B = 4/TVL, nu = 1 gives G = TVL/4.
        let (g, eps) = arc_params(&[0], &[1], &[1.0], &[4.0 / 1000.0], &[1.0, 1.0]).unwrap();
        assert!((g[0] - 250.0).abs() < 1e-9);
        assert_eq!(eps[0], 0.0);
    }

    #[test]
    fn negative_curvature_is_refused() {
        let err = arc_params(&[0], &[1], &[1.0], &[-1.0], &[1.0, 1.0]).unwrap_err();
        assert!(err.0.contains("negative curvature"));
    }

    #[test]
    fn duplicates_add_like_parallel_resistors() {
        let opts = BuildOptions { n_nodes: Some(2), ..Default::default() };
        let out = build(
            &[0, 0],
            &[1, 1],
            &[1.0, 1.0],
            &[1e-3, 1e-3],
            &[1.0, 1.0],
            1.0,
            &opts,
        )
        .unwrap();
        assert_eq!(out.m(), 1);
        assert_eq!(out.sources[0], vec![0, 1]);
        assert!((out.g[0] - 2000.0).abs() < 1e-9);
        assert_eq!(out.dropped, vec![(1, Dropped::Merged)]);
    }

    #[test]
    fn clamped_without_a_cap_is_unbounded() {
        let opts = BuildOptions { n_nodes: Some(2), ..Default::default() };
        let err = build(&[0], &[1], &[1.0], &[0.0], &[1.0, 1.0], 1.0, &opts).unwrap_err();
        assert!(err.0.contains("unbounded"));
    }

    #[test]
    fn round12_matches_cpython_on_ties() {
        // round(0.1234567890125, 12) == 0.123456789012 -- the stored double is
        // just under the tie, so it goes down, not to even.
        assert_eq!(round12_bits(0.1234567890125), 0.123456789012f64.to_bits());
        assert_eq!(round12_bits(0.1234567890135), 0.123456789014f64.to_bits());
        assert_eq!(round12_bits(-0.0), round12_bits(0.0));
    }

    #[test]
    fn median_averages_the_middle_pair() {
        let mut v = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut v), 2.5);
    }
}
