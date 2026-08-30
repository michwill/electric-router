//! The stages a quote runs, in order (mirror of `core/pipeline.py`).
//!
//! **What is here and what is not.** `pipeline.py` is two things wound
//! together: the arithmetic that decides, and the I/O that feeds it.
//! `prepare` probes the chain, `route` runs the quote loop, `_price_once`
//! calls the quoter, the scout batches probes. None of that is ported, for the
//! reason `verify` gives -- chain I/O is the host's, and a browser has its own
//! RPC. What is ported is every stage that turns chain answers into a
//! decision, so a host that can fetch can route.
//!
//! In order, as a quote runs them:
//!
//! 1. reduce the universe -- `prune_dead_end_nodes`, `restrict_to_component`;
//! 2. assemble the graph -- `clamp_unphysical_depth`, `assemble`;
//! 3. solve, and check what came back -- the `kcl_*` family;
//! 4. read the flow back out -- `realised_delta`, `realised_theta`;
//! 5. rank -- `scout_priority`, `gas_cost`;
//! 6. price the legs -- `pricing_layers`.
//!
//! The counters and warnings are part of the answer rather than logging: a
//! quote reports how much of the universe it dropped and why, and the
//! reference's `RouteResult` carries them out.

use crate::gas::min_useful_flow;
use crate::graph::{self, ArcArrays, BuildOptions, MAX_CONDITION};
use crate::nodes::NodeMap;
use crate::pyfmt;
use crate::realize::RealizedRoute;
use crate::types::{ArcKind, PoolArc};

/// A quote's counters and warnings -- the reference's `RouteResult`, minus
/// everything the chain fills in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// Insertion-ordered, as the reference's dict is: the renderer prints them
    /// in the order the stages set them.
    pub counters: Vec<(String, i64)>,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn set(&mut self, name: &str, value: i64) {
        match self.counters.iter_mut().find(|(k, _)| k == name) {
            Some(entry) => entry.1 = value,
            None => self.counters.push((name.to_string(), value)),
        }
    }

    pub fn get(&self, name: &str) -> Option<i64> {
        self.counters.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
    }
}

// ---------------------------------------------------- 1. the universe

/// Drop arcs into nodes no route can pass *through*.
///
/// A node that is neither endpoint has to be entered through one pool and left
/// through another. Decision 3 gives a route at most one arc per pool, and for
/// a two-coin pool the only other coin is the one the flow just arrived from,
/// so a second arc of that pool is where it came from rather than onward. A
/// node touched by exactly one pool can therefore only ever be an endpoint --
/// not "is unlikely to help", cannot appear.
///
/// The same holds where a single pool has three coins and could technically
/// serve both hops: `A -> v -> B` inside one pool is dominated by `A -> B`
/// inside it, since the pool prices the pair directly.
///
/// This is what keeps the long tail of single-pool tokens out of the search on
/// structure rather than by a list of names. Measured on mainnet, HLX, CXD,
/// FIDU and STG each sit in exactly one pool, and the two-hop floor was
/// ranking them above crvUSD because their tokens are numerous.
///
/// Iterated, because removing a node can leave its neighbour with one pool.
/// Endpoints are never pruned: quoting `HLX -> USDC` is a fair question and
/// its single pool is the answer.
pub fn prune_dead_end_nodes(
    arcs: &[PoolArc],
    src_node: usize,
    dst_node: usize,
    report: &mut Report,
) -> Vec<PoolArc> {
    let mut live: Vec<PoolArc> = arcs.to_vec();
    for _ in 0..arcs.len() + 1 {
        // node -> the distinct pools touching it
        let mut touching: Vec<(usize, Vec<String>)> = Vec::new();
        let mut note = |node: usize, pool: &str, touching: &mut Vec<(usize, Vec<String>)>| {
            match touching.iter_mut().find(|(n, _)| *n == node) {
                Some((_, pools)) => {
                    if !pools.iter().any(|p| p == pool) {
                        pools.push(pool.to_string());
                    }
                }
                None => touching.push((node, vec![pool.to_string()])),
            }
        };
        for arc in &live {
            let pool = arc.pool.to_ascii_lowercase();
            note(arc.tau, &pool, &mut touching);
            note(arc.sigma, &pool, &mut touching);
        }
        let dead: Vec<usize> = touching
            .iter()
            .filter(|(node, pools)| {
                *node != src_node && *node != dst_node && pools.len() < 2
            })
            .map(|(node, _)| *node)
            .collect();
        if dead.is_empty() {
            break;
        }
        live.retain(|a| !dead.contains(&a.tau) && !dead.contains(&a.sigma));
    }
    report.set("arcs_dead_end", (arcs.len() - live.len()) as i64);
    live
}

/// Keep only what `dst` can be reached from, in either direction.
pub fn restrict_to_component(
    arcs: &[PoolArc],
    dst_node: usize,
    n_nodes: usize,
    report: &mut Report,
) -> Vec<PoolArc> {
    let tau: Vec<i64> = arcs.iter().map(|a| a.tau as i64).collect();
    let sig: Vec<i64> = arcs.iter().map(|a| a.sigma as i64).collect();
    let reachable = graph::component_of(dst_node, &tau, &sig, n_nodes);
    let keep: Vec<PoolArc> = arcs
        .iter()
        .filter(|a| reachable[a.tau] && reachable[a.sigma])
        .cloned()
        .collect();
    report.set("arcs_unreachable", (arcs.len() - keep.len()) as i64);
    report.set("nodes_reachable", reachable.iter().filter(|&&v| v).count() as i64);
    keep
}

// ----------------------------------------------------- 2. the graph

/// A pool cannot be meaningfully deeper than this multiple of its own input
/// reserve. For constant product `G = nu*y0/2` exactly; a stableswap at the
/// peg is far deeper, but not by twelve orders of magnitude.
pub const DEPTH_LIMIT: f64 = 1e4;

/// Treat immeasurably small curvature as the zero-curvature limit.
///
/// A `B` implying a conductance far beyond the pool's own reserves is not a
/// very deep pool but a curvature below the quotes' integer noise floor.
/// Clamping to `B = 0` with a cap is the admissible limit (§2.3) and keeps the
/// arc bottomless only up to the size actually probed.
pub fn clamp_unphysical_depth(arcs: &mut [PoolArc], nu: &[f64], nodes: &NodeMap) -> usize {
    let mut clamped = 0usize;
    for arc in arcs.iter_mut() {
        if arc.b <= 0.0 || arc.reserve_in == 0 {
            continue;
        }
        let reserve_canonical = crate::pools::divided(
            ruint::aliases::U256::from(arc.reserve_in),
            ruint::aliases::U256::from(10u64).pow(ruint::aliases::U256::from(arc.decimals_in)),
        ) * nodes.rate(&arc.token_in);
        let limit = nu[arc.tau] * reserve_canonical * DEPTH_LIMIT;
        let conductance = nu[arc.tau] * arc.a / arc.b;
        if limit > 0.0 && conductance > limit {
            arc.b = 0.0;
            arc.clamped = true;
            arc.convex_flag = true;
            let probed = if arc.calib_delta > 0.0 { arc.calib_delta } else { reserve_canonical };
            arc.cap = arc.cap.min(probed);
            clamped += 1;
        }
    }
    clamped
}

/// Build the solver arrays from the current calibration.
///
/// Called twice, coarse then refined, because `build` drops dust and can merge
/// duplicates -- the arc list has to be re-aligned to whatever survived.
pub fn assemble(
    arcs: &[PoolArc],
    nu: &[f64],
    psi_total: f64,
    nodes: &NodeMap,
    src_node: usize,
    dst_node: usize,
    report: &mut Report,
) -> Result<(Vec<PoolArc>, ArcArrays), graph::GraphError> {
    let mut arcs: Vec<PoolArc> = arcs.to_vec();
    let bottomless = clamp_unphysical_depth(&mut arcs, nu, nodes);
    if bottomless > 0 {
        report.set("arcs_clamped_as_bottomless", bottomless as i64);
    }

    // `cap` is a bound on *value* flow, so convert from canonical token units.
    let caps: Vec<f64> = arcs
        .iter()
        .map(|a| if a.cap.is_finite() { nu[a.tau] * a.cap } else { a.cap })
        .collect();
    let tau: Vec<i64> = arcs.iter().map(|a| a.tau as i64).collect();
    let sig: Vec<i64> = arcs.iter().map(|a| a.sigma as i64).collect();
    let a: Vec<f64> = arcs.iter().map(|x| x.a).collect();
    let b: Vec<f64> = arcs.iter().map(|x| x.b).collect();
    let flagged: Vec<bool> = arcs.iter().map(|x| x.convex_flag).collect();
    let clamped: Vec<bool> = arcs.iter().map(|x| x.clamped).collect();
    let opts = BuildOptions {
        cap: Some(&caps),
        flagged: Some(&flagged),
        clamped: Some(&clamped),
        n_nodes: Some(nodes.n_nodes()),
        merge_duplicates: false,
        require: Some((src_node, dst_node)),
        ..Default::default()
    };
    let g = graph::build(&tau, &sig, &a, &b, nu, psi_total, &opts)?;

    let mut kept: Vec<PoolArc> = g.sources.iter().map(|group| arcs[group[0]].clone()).collect();
    report.set(
        "arcs_dropped_dust",
        g.dropped.iter().filter(|(_, r)| *r == graph::Dropped::Dust).count() as i64,
    );
    for (k, arc) in kept.iter_mut().enumerate() {
        arc.g = g.g[k];
        arc.eps = g.eps[k];
    }
    if g.ill_conditioned != 0.0 {
        report.set("condition", g.ill_conditioned as i64);
        report.warnings.push(format!(
            "conductance spread is {}, past §12.4's {} bound: the dust floor \
             was backed off to keep the pair connected at all. The route is \
             still checked on-chain, but treat the modelled split as approximate",
            pyfmt::sci(g.ill_conditioned, 2),
            pyfmt::sci(MAX_CONDITION, 0)
        ));
    }
    warn_pair_drops(&kept, report);
    Ok((kept, g))
}

/// §2.6: `eps_f + eps_r <= 0` means `nu` is inconsistent with that pool.
///
/// It manufactures a two-arc negative cycle that does not exist. Round-tripping
/// a pool always loses (`a_f a_r = Gamma^2 < 1`), but the *linearised* drops
/// are frame-dependent and their sum falls below zero when `nu` is far enough
/// off for that pair.
pub fn warn_pair_drops(arcs: &[PoolArc], report: &mut Report) {
    // Insertion-ordered and last-wins, as the reference's dict is.
    let mut forward: Vec<((String, i32, i32), usize)> = Vec::new();
    for (k, arc) in arcs.iter().enumerate() {
        let key = (arc.pool.to_ascii_lowercase(), arc.i, arc.j);
        match forward.iter_mut().find(|(existing, _)| *existing == key) {
            Some(entry) => entry.1 = k,
            None => forward.push((key, k)),
        }
    }
    let mut violations = 0i64;
    let mut any = false;
    for ((pool, i, j), k) in &forward {
        if i >= j {
            continue;
        }
        let mirror = (pool.clone(), *j, *i);
        let Some((_, r)) = forward.iter().find(|(key, _)| *key == mirror) else {
            continue;
        };
        any = true;
        if arcs[*k].eps + arcs[*r].eps <= 0.0 {
            violations += 1;
        }
    }
    if !any {
        return;
    }
    report.set("eps_pair_violations", violations);
    if violations > 0 {
        report.warnings.push(format!(
            "{violations} pool(s) have eps_f + eps_r <= 0: the reference price \
             is inconsistent with them (spurious negative 2-cycle, §2.6)"
        ));
    }
}

/// Link each arc to its opposite and record the fee the pair measures.
///
/// `gamma_live = sqrt(a_f * a_r)` reads a pool's *current* retention off two
/// tiny probes, with no fee parameter and no ABI knowledge of the fee law
/// (§2.6). The node-merge rescaling cancels in the product, so it is the same
/// number in canonical coordinates as in the pool's own.
///
/// Only same-kind pairs qualify: a deposit's opposite is a withdrawal, and
/// round-tripping those measures two fees plus an imbalance, not one fee.
pub fn pair_directions(arcs: &mut [PoolArc]) -> usize {
    let ids: Vec<String> = arcs.iter().map(|a| a.id.clone()).collect();
    let wanted: Vec<String> = arcs
        .iter()
        .map(|a| format!("{}:{}:{}>{}", a.pool.to_ascii_lowercase(), a.kind.code(), a.j, a.i))
        .collect();
    let mut paired = 0usize;
    for k in 0..arcs.len() {
        let Some(other) = ids.iter().position(|id| *id == wanted[k]) else {
            continue;
        };
        arcs[k].reverse_id = Some(ids[other].clone());
        if arcs[k].a > 0.0 && arcs[other].a > 0.0 {
            arcs[k].gamma_live = (arcs[k].a * arcs[other].a).sqrt();
            paired += 1;
        }
    }
    paired
}

// ------------------------------------------------------- 3. the check

pub const KCL_RELATIVE: f64 = 1e-8;
pub const KCL_ABSOLUTE: f64 = 1e-9;
pub const EPS: f64 = 2.220446049250313e-16;
/// The gap between `k * eps` and the error actually realised, measured at
/// 0.04x to 33x. Even at `k = 1e12` this stays two orders below conjured flow.
pub const KCL_CONDITION_SAFETY: f64 = 100.0;

/// How much KCL slop is floating-point noise rather than a bug.
///
/// The solve runs in units scaled by `g_scale`, so a tolerance expressed purely
/// as a fraction of `Psi` tightens without limit as the trade shrinks and
/// rejects small trades outright. Allowing both terms stays far tighter than
/// the failure this catches, since conjured flow is `O(Psi)`.
pub fn kcl_tolerance(psi_total: f64, g_scale: f64) -> f64 {
    KCL_RELATIVE + KCL_ABSOLUTE * g_scale / psi_total.max(1e-30)
}

/// The residual, and *where* it is -- worst node, and its live arc counts.
///
/// The node is what makes a refusal diagnosable: flow leaving a node with none
/// arriving is conjured flow, a different bug from a conditioning failure.
pub struct KclDetail {
    pub residual: f64,
    pub node: i64,
    pub arcs_in: usize,
    pub arcs_out: usize,
}

pub fn kcl_detail(
    g: &ArcArrays,
    psi: &[f64],
    src: usize,
    dst: usize,
    psi_total: f64,
) -> KclDetail {
    if psi_total <= 0.0 {
        return KclDetail { residual: 0.0, node: -1, arcs_in: 0, arcs_out: 0 };
    }
    let mut net = vec![0.0f64; g.n_nodes];
    for p in 0..psi.len() {
        net[g.tau[p] as usize] += psi[p];
    }
    // A second pass, not one fused loop: `np.add.at` then `np.subtract.at` is
    // two sweeps and float addition is not associative.
    for p in 0..psi.len() {
        net[g.sig[p] as usize] -= psi[p];
    }
    let mut want = vec![0.0f64; g.n_nodes];
    want[src] += psi_total;
    want[dst] -= psi_total;

    let err: Vec<f64> = (0..g.n_nodes).map(|v| (net[v] - want[v]).abs() / psi_total).collect();
    let mut worst = 0usize;
    for v in 0..g.n_nodes {
        if err[v].is_nan() {
            worst = v;
            break;
        }
        if err[v] > err[worst] {
            worst = v;
        }
    }
    // `tau` is an arc's origin and `sig` its head, so counting `tau == worst`
    // counts what *leaves* the node.
    let arcs_out = (0..psi.len())
        .filter(|&p| g.tau[p] as usize == worst && psi[p] > 0.0)
        .count();
    let arcs_in = (0..psi.len())
        .filter(|&p| g.sig[p] as usize == worst && psi[p] > 0.0)
        .count();
    KclDetail { residual: err[worst], node: worst as i64, arcs_in, arcs_out }
}

/// `||B^T psi - s_hat||_inf / Psi` -- Kirchhoff's current law (§12.4).
pub fn kcl_residual(g: &ArcArrays, psi: &[f64], src: usize, dst: usize, psi_total: f64) -> f64 {
    kcl_detail(g, psi, src, dst, psi_total).residual
}

/// The KCL residual a backward-stable solve could deliver on this graph.
///
/// `k * eps` floors any residual computed from a solve of condition number
/// `k`. Returns 0 when there is nothing to condition, leaving the caller's flat
/// tolerance in charge.
pub fn achievable_kcl(g: &ArcArrays, active: &[bool], dst: usize) -> f64 {
    // The *active set*, not the arcs that ended up carrying flow: the solve
    // factorises the Laplacian of everything in `A`, and that is the system
    // whose conditioning limited `u`.
    let live: Vec<usize> = (0..active.len()).filter(|&k| active[k]).collect();
    if live.is_empty() {
        return 0.0;
    }
    let tau: Vec<i64> = live.iter().map(|&k| g.tau[k]).collect();
    let sig: Vec<i64> = live.iter().map(|&k| g.sig[k]).collect();
    let conductance: Vec<f64> = live.iter().map(|&k| g.g[k]).collect();
    let comp = graph::component_of(dst, &tau, &sig, g.n_nodes);
    let keep: Vec<usize> = (0..g.n_nodes).filter(|&v| comp[v] && v != dst).collect();
    if keep.is_empty() {
        return 0.0;
    }
    let matrix = graph::laplacian(&tau, &sig, &conductance, g.n_nodes, &keep);
    let Some(kappa) = condition_number(&matrix, keep.len()) else {
        return 0.0;
    };
    if !kappa.is_finite() {
        return 0.0;
    }
    KCL_CONDITION_SAFETY * kappa * EPS
}

/// `np.linalg.cond` in the 2-norm: the ratio of the extreme singular values.
///
/// The Laplacian restricted to `keep` is symmetric positive definite, so its
/// singular values are its eigenvalues and a symmetric eigensolver is enough.
///
/// Written here rather than bound because `README.md` rules out BLAS and
/// LAPACK: no wasm build, and it would not buy an exact match anyway --
/// numpy's OpenBLAS is built `DYNAMIC_ARCH` and threaded, so it is not one
/// fixed sequence of operations.
///
/// **Measured against `np.linalg.cond`, 172 matrices, `k` from 1 to 1e16:**
/// exact to 2.4e-5 relative over the whole range the graph admits
/// (`MAX_CONDITION` is 1e12), degrading to 29% past 1e14 -- where the small
/// eigenvalues are at the double-precision noise floor and neither
/// implementation is authoritative. Against a safety factor of 100, that is
/// noise.
///
/// It is 11x slower than numpy at `n = 50` (1.07 ms against 0.096), which does
/// not matter here and would if this moved: `_achievable_kcl` is called *only*
/// on the path about to fail the KCL check, never on a quote that is working.
fn condition_number(matrix: &[f64], n: usize) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let mut eigen = symmetric_eigenvalues(matrix, n)?;
    eigen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, hi) = (eigen[0].abs(), eigen[n - 1].abs());
    if lo == 0.0 {
        return Some(f64::INFINITY);
    }
    Some(hi / lo)
}

/// Eigenvalues of a symmetric matrix, by unshifted cyclic Jacobi.
///
/// Slow and unconditionally convergent, which is the right trade on a path
/// that only runs when a quote is already failing: it cannot diverge the way
/// an iteration with a shift strategy can.
///
/// The sweep cap is a backstop rather than the mechanism. Measured over 172
/// Laplacians the worst was 21 sweeps, on a `k = 1e16` matrix; the ordinary
/// case is five to nine.
fn symmetric_eigenvalues(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = matrix.to_vec();
    if a.iter().any(|v| !v.is_finite()) {
        return None;
    }
    for _ in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    Some((0..n).map(|k| a[k * n + k]).collect())
}

// -------------------------------------------------- 4. reading it back

/// This arc's realised input, in its own token's raw units.
pub fn realised_delta(arc: &PoolArc, psi_value: f64, nu: &[f64], nodes: &NodeMap) -> f64 {
    let price = nu[arc.tau];
    let rate = nodes.rate(&arc.token_in);
    if price <= 0.0 || rate <= 0.0 || psi_value <= 0.0 {
        return 0.0;
    }
    psi_value / price / rate * pow10(arc.decimals_in)
}

/// §12.1's `theta_p` for every arc carrying flow.
pub fn realised_theta(
    arcs: &[PoolArc],
    psi: &[f64],
    nu: &[f64],
    nodes: &NodeMap,
    active: &[usize],
) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for &k in active {
        let arc = &arcs[k];
        if arc.reserve_in == 0 {
            continue;
        }
        let delta = realised_delta(arc, psi[k], nu, nodes);
        if delta > 0.0 {
            out.push((k, delta / crate::pools::scaled(ruint::aliases::U256::from(arc.reserve_in), 0)));
        }
    }
    out
}

/// One unit of the output token, in the human units calibration fits in.
///
/// Curve quotes integers, so this is the resolution of every ladder
/// measurement.
pub fn quantum(decimals_out: u32) -> f64 {
    // `powf`, not `powi`: the reference is `10.0 ** -d`, which goes through
    // libm's correctly-rounded `pow`, while `powi` is a multiplication chain
    // that drifts -- at 24 decimals it lands one ULP out.
    10f64.powf(-(decimals_out as f64))
}

/// `10 ** d` as `float(10**d)` reads it: one rounding of the exact integer.
fn pow10(d: u32) -> f64 {
    crate::pools::scaled(
        ruint::aliases::U256::from(10u64).pow(ruint::aliases::U256::from(d)),
        0,
    )
}

// ------------------------------------------------------- 5. the rank

/// Output-token wei per 1 ETH, for costing gas. 0 when ETH is unpriced.
pub const WETH: &str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";

pub fn dst_per_eth(nodes: &NodeMap, nu: &[f64], dst_token: &str) -> f64 {
    if !nodes.has(WETH) {
        return 0.0;
    }
    let (Some(weth), Some(dst)) = (nodes.node(WETH), nodes.node(dst_token)) else {
        return 0.0;
    };
    let (weth_value, dst_value) = (nu[weth], nu[dst]);
    if weth_value <= 0.0 || dst_value <= 0.0 {
        return 0.0;
    }
    weth_value / dst_value * pow10(nodes.decimals(dst_token))
}

/// One leg's gas, in the solver's scaled value units. 0 disables it.
pub fn gas_cost(
    nodes: &NodeMap,
    nu: &[f64],
    dst_token: &str,
    gas_price_wei: i64,
    g_scale: f64,
) -> f64 {
    if gas_price_wei <= 0 || g_scale <= 0.0 {
        return 0.0;
    }
    let per_eth = dst_per_eth(nodes, nu, dst_token) / pow10(nodes.decimals(dst_token));
    min_useful_flow(gas_price_wei, per_eth, ArcKind::SwapStable) / g_scale
}

/// How promising this candidate is as a scout entrant, or 0 to skip it.
///
/// Two things decide it. First, there has to be something to re-split:
/// `split::scout` drops any plan whose legs form no split group, because there
/// are no weights to move. Choosing entrants on anything else spends slots in
/// the shared batch on plans it will throw away -- measured on
/// crvUSD -> sDOLA at $2M, six entrants picked by leg count yielded **one**
/// usable plan.
///
/// Second, among those, how much the topology could carry if it *were* split
/// properly. That is `route_conductance`: the route read as a resistor network
/// with `1/TVL` per pool, so series hops add resistance and parallel branches
/// add conductance. It rewards branching and depth together, which leg count
/// only gestured at.
pub fn scout_priority(route: &RealizedRoute) -> f64 {
    let legs: Vec<crate::types::Leg> = route.legs.iter().map(|rl| rl.leg.clone()).collect();
    if route.legs.is_empty() || crate::split::split_groups(&legs).is_empty() {
        return 0.0;
    }
    crate::realize::route_conductance(route)
}

// ------------------------------------------------------ 6. the walk

/// Leg indices grouped so no leg in a group feeds another in it.
///
/// Legs arrive topologically ordered, so a group closes as soon as one draws
/// on a slot the group has just filled. Depth is what costs round trips here,
/// not leg count: a five-leg route with two branches is two batches.
pub fn pricing_layers(route: &RealizedRoute) -> Vec<Vec<usize>> {
    let mut layers: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut filled: Vec<i32> = Vec::new();
    for (k, realized) in route.legs.iter().enumerate() {
        if filled.contains(&realized.leg.src_slot) {
            layers.push(std::mem::take(&mut current));
            filled.clear();
        }
        current.push(k);
        if !filled.contains(&realized.leg.dst_slot) {
            filled.push(realized.leg.dst_slot);
        }
    }
    if !current.is_empty() {
        layers.push(current);
    }
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ArcKind;

    fn arc(id: &str, pool: &str, tau: usize, sigma: usize) -> PoolArc {
        let mut a = PoolArc::new(
            id.into(), pool.into(), ArcKind::SwapStable, 0, 1, 2,
            format!("0x{tau:040x}"), format!("0x{sigma:040x}"), tau, sigma,
        );
        a.a = 0.999;
        a.b = 1e-6;
        a
    }

    #[test]
    fn a_node_in_one_pool_can_only_be_an_endpoint() {
        // 0 -> 1 through two pools, and 1 -> 2 through one: node 2 is a dead
        // end unless it is an endpoint.
        let arcs = vec![
            arc("a", "0xp1", 0, 1),
            arc("b", "0xp2", 0, 1),
            arc("c", "0xp3", 1, 2),
        ];
        let mut report = Report::default();
        let live = prune_dead_end_nodes(&arcs, 0, 1, &mut report);
        assert_eq!(live.len(), 2);
        assert_eq!(report.get("arcs_dead_end"), Some(1));
        // Asked for `0 -> 2`, the single pool is the answer rather than dust.
        let mut report = Report::default();
        assert_eq!(prune_dead_end_nodes(&arcs, 0, 2, &mut report).len(), 3);
    }

    #[test]
    fn pruning_iterates_until_it_stops_changing() {
        // A chain of single-pool hops peels one node at a time.
        let arcs = vec![
            arc("a", "0xp1", 0, 1),
            arc("b", "0xp2", 0, 1),
            arc("c", "0xp3", 1, 2),
            arc("d", "0xp4", 2, 3),
        ];
        let mut report = Report::default();
        assert_eq!(prune_dead_end_nodes(&arcs, 0, 1, &mut report).len(), 2);
    }

    #[test]
    fn an_immeasurably_deep_pool_becomes_the_zero_curvature_limit() {
        let mut arcs = vec![arc("a", "0xp", 0, 1)];
        arcs[0].b = 1e-30;
        arcs[0].reserve_in = 10u128.pow(24);
        arcs[0].decimals_in = 18;
        let mut nodes = NodeMap::new();
        nodes.add_token(&arcs[0].token_in, "IN", 18);
        nodes.add_token(&arcs[0].token_out, "OUT", 18);
        assert_eq!(clamp_unphysical_depth(&mut arcs, &[1.0, 1.0], &nodes), 1);
        assert_eq!(arcs[0].b, 0.0);
        assert!(arcs[0].clamped && arcs[0].convex_flag);
        // Bottomless only up to the size actually probed.
        assert_eq!(arcs[0].cap, 1e6);
    }

    #[test]
    fn a_measurable_pool_is_left_alone() {
        let mut arcs = vec![arc("a", "0xp", 0, 1)];
        arcs[0].b = 1e-6;
        arcs[0].reserve_in = 10u128.pow(24);
        let mut nodes = NodeMap::new();
        nodes.add_token(&arcs[0].token_in, "IN", 18);
        nodes.add_token(&arcs[0].token_out, "OUT", 18);
        assert_eq!(clamp_unphysical_depth(&mut arcs, &[1.0, 1.0], &nodes), 0);
        assert_eq!(arcs[0].b, 1e-6);
    }

    #[test]
    fn the_tolerance_does_not_tighten_without_limit_on_a_small_trade() {
        // A tolerance that is purely relative rejects small trades outright.
        assert!(kcl_tolerance(1e-9, 1.0) > kcl_tolerance(1.0, 1.0));
        assert_eq!(kcl_tolerance(1.0, 0.0), KCL_RELATIVE);
    }

    #[test]
    fn conjured_flow_names_the_node_it_came_from() {
        // Arc 0 delivers 0 -> 1, which is the trade; arc 1 leaves node 2
        // carrying flow nothing fed it. Conjured flow always imbalances two
        // nodes -- here 2 and 3 -- and `argmax` takes the first, which is the
        // one it came *from*. That is the useful end to name.
        let g = ArcArrays {
            tau: vec![0, 2],
            sig: vec![1, 3],
            g: vec![1.0, 1.0],
            eps: vec![0.0, 0.0],
            cap: vec![f64::INFINITY; 2],
            n_nodes: 4,
            ..Default::default()
        };
        let got = kcl_detail(&g, &[1.0, 0.5], 0, 1, 1.0);
        assert_eq!(got.node, 2);
        assert_eq!(got.arcs_out, 1);
        assert_eq!(got.arcs_in, 0);
        assert!((got.residual - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_clean_flow_has_no_residual() {
        let g = ArcArrays {
            tau: vec![0],
            sig: vec![1],
            g: vec![1.0],
            eps: vec![0.0],
            cap: vec![f64::INFINITY],
            n_nodes: 2,
            ..Default::default()
        };
        assert_eq!(kcl_residual(&g, &[1.0], 0, 1, 1.0), 0.0);
    }

    #[test]
    fn the_condition_number_of_a_known_matrix() {
        // diag(1, 100): eigenvalues 1 and 100, so cond = 100.
        let got = condition_number(&[1.0, 0.0, 0.0, 100.0], 2).unwrap();
        assert!((got - 100.0).abs() < 1e-9);
        // A 2x2 Laplacian of one arc with G = 4, grounded: [[4]] -> cond 1.
        assert!((condition_number(&[4.0], 1).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_condition_number_holds_up_where_the_caller_uses_it() {
        // This runs only when the KCL check is failing, which is where `k` is
        // large -- so the claim worth pinning is at the graph's own ceiling,
        // `MAX_CONDITION = 1e12`, not at 100.
        //
        // A rotated diagonal, so the answer is known rather than compared
        // against another implementation. The rotation is dense, which is what
        // makes it a real test of the sweep rather than of the diagonal.
        for &target in &[1e6f64, 1e9, 1e12] {
            let n = 8usize;
            let lambda: Vec<f64> =
                (0..n).map(|k| target.powf(k as f64 / (n - 1) as f64)).collect();
            // Householder from a fixed vector: Q = I - 2vv^T / v.v.
            let v: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
            let vv: f64 = v.iter().map(|x| x * x).sum();
            let q: Vec<f64> = (0..n * n)
                .map(|at| {
                    let (r, c) = (at / n, at % n);
                    f64::from(r == c) - 2.0 * v[r] * v[c] / vv
                })
                .collect();
            // A = Q diag(lambda) Q^T, symmetric by construction.
            let mut a = vec![0.0f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    a[r * n + c] =
                        (0..n).map(|k| q[r * n + k] * lambda[k] * q[c * n + k]).sum();
                }
            }
            let got = condition_number(&a, n).unwrap();
            assert!(
                (got / target - 1.0).abs() < 1e-4,
                "k = {target:e}: got {got:e}"
            );
        }
    }

    #[test]
    fn a_pair_whose_drops_sum_to_nothing_is_flagged() {
        let mut arcs = vec![arc("f", "0xp", 0, 1), arc("r", "0xp", 1, 0)];
        arcs[0].i = 0;
        arcs[0].j = 1;
        arcs[1].i = 1;
        arcs[1].j = 0;
        arcs[0].eps = 0.001;
        arcs[1].eps = -0.002;
        let mut report = Report::default();
        warn_pair_drops(&arcs, &mut report);
        assert_eq!(report.get("eps_pair_violations"), Some(1));
        assert_eq!(report.warnings.len(), 1);
        // A healthy pair says so and warns about nothing.
        arcs[1].eps = 0.001;
        let mut report = Report::default();
        warn_pair_drops(&arcs, &mut report);
        assert_eq!(report.get("eps_pair_violations"), Some(0));
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn a_pair_measures_the_fee_without_reading_one() {
        let mut arcs = vec![
            {
                let mut a = arc("0xp:0:0>1", "0xp", 0, 1);
                a.a = 0.999;
                a
            },
            {
                let mut a = arc("0xp:0:1>0", "0xp", 1, 0);
                a.i = 1;
                a.j = 0;
                a.a = 0.997;
                a
            },
        ];
        assert_eq!(pair_directions(&mut arcs), 2);
        assert_eq!(arcs[0].reverse_id.as_deref(), Some("0xp:0:1>0"));
        assert!((arcs[0].gamma_live - (0.999f64 * 0.997).sqrt()).abs() < 1e-15);
        // The same number from either side.
        assert_eq!(arcs[0].gamma_live, arcs[1].gamma_live);
    }
}
