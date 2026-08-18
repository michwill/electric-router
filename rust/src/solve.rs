//! `active_set_solve` (spec §5.4), ported from `core/solve.py`.
//!
//! A faithful port, not an improvement: the Python stays the reference and the
//! two are differed against each other and against OSQP.  Where a choice in
//! the original looks arbitrary it is reproduced exactly, because the pivot
//! sequence decides which of several optimal bases is reached, and a quote has
//! to be the same answer on every target.
//!
//! Three details carried over deliberately:
//!
//! * ties in the steepest-edge rule go to the **lowest index**, matching
//!   numpy's `argmax`;
//! * the component containing `dst` is recomputed **every pivot** (§9.4), not
//!   once -- the first pivot that orphans a leaf otherwise yields a singular
//!   factorisation;
//! * `psi` is zeroed below `tol` only on the arcs of the *cycle* being
//!   settled, never globally -- zeroing a negative arc elsewhere strands its
//!   flow, which is a bug this project has already paid for once.

use crate::lu;
use std::collections::HashSet;

pub const TOL: f64 = 1e-9;
pub const CYCLE_PATIENCE: u32 = 3;

/// Why a solve stopped, in the same words the Python uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    Optimal,
    Partial,
    NoConvergence(u32),
    Cycling(u32),
    SrcDetached,
    PinDetached,
    Singular(usize),
}

impl Stop {
    pub fn reason(&self) -> String {
        match self {
            Stop::Optimal => String::new(),
            Stop::Partial => "PARTIAL".into(),
            Stop::NoConvergence(n) => format!("no convergence in {n} pivots"),
            Stop::Cycling(n) => {
                format!("no convergence: cycling under Bland's rule after {n} pivots")
            }
            Stop::SrcDetached => "src not connected to dst through the active set".into(),
            Stop::PinDetached => "a pinned arc is detached from the active network".into(),
            Stop::Singular(c) => format!("singular Laplacian: zero pivot at {c}"),
        }
    }
    pub fn feasible(&self) -> bool {
        matches!(self, Stop::Optimal | Stop::Partial)
    }
}

/// The graph the solve runs over.  Value semantics: the caller owns the arrays.
pub struct Arcs<'a> {
    pub tau: &'a [i64],
    pub sig: &'a [i64],
    pub g: &'a [f64],
    pub eps: &'a [f64],
    pub cap: &'a [f64],
    pub n_nodes: usize,
}

impl Arcs<'_> {
    pub fn m(&self) -> usize {
        self.tau.len()
    }
}

pub struct Options {
    pub tol: f64,
    pub maxit: u32,
    pub min_flow: f64,
    pub gas_cost: f64,
    pub partial_ok: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tol: TOL,
            maxit: 600,
            min_flow: 0.0,
            gas_cost: 0.0,
            partial_ok: false,
        }
    }
}

pub struct Solution {
    pub psi: Vec<f64>,
    pub u: Vec<f64>,
    pub active: Vec<bool>,
    pub upper: Vec<bool>,
    pub psi_upper: Vec<f64>,
    pub rho: Vec<f64>,
    pub pivots: u32,
    pub stop: Stop,
}

/// Nodes reachable from `root` over the active arcs, undirected (§9.4).
fn component_of(root: usize, arcs: &Arcs, active: &[bool], out: &mut [bool]) {
    out.iter_mut().for_each(|v| *v = false);
    if arcs.n_nodes == 0 {
        return;
    }
    out[root] = true;
    // Iterate to a fixed point.  The arc count is small and this is
    // branch-free; a queue would be asymptotically better and slower here.
    loop {
        let mut grew = false;
        for p in 0..arcs.m() {
            if !active[p] {
                continue;
            }
            let (a, b) = (arcs.tau[p] as usize, arcs.sig[p] as usize);
            if out[a] != out[b] {
                out[a] = true;
                out[b] = true;
                grew = true;
            }
        }
        if !grew {
            return;
        }
    }
}

/// Most-violating candidate; ties to the lowest index, as numpy's argmax does.
fn steepest(mask: &[bool], score: &[f64]) -> usize {
    let mut best = usize::MAX;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &on) in mask.iter().enumerate() {
        if on && score[i] > best_score {
            best = i;
            best_score = score[i];
        }
    }
    best
}

/// Bland's rule: lowest index. Guarantees termination once a basis repeats.
fn bland(mask: &[bool]) -> usize {
    mask.iter().position(|&on| on).unwrap_or(usize::MAX)
}

pub fn active_set_solve(
    arcs: &Arcs,
    src: usize,
    dst: usize,
    psi_total: f64,
    a0: Option<&[bool]>,
    forbidden: Option<&[bool]>,
    pinned: &[(usize, f64)],
    opt: &Options,
) -> Solution {
    let m = arcs.m();
    let n = arcs.n_nodes;
    let forbid: Vec<bool> = match forbidden {
        Some(f) => f.to_vec(),
        None => vec![false; m],
    };

    let mut s_hat = vec![0.0; n];
    s_hat[src] += psi_total;
    s_hat[dst] -= psi_total;

    let mut active = vec![false; m];
    let mut upper = vec![false; m];
    let mut psi_upper = vec![0.0; m];
    let mut is_pinned = vec![false; m];
    for &(arc, value) in pinned {
        upper[arc] = true;
        psi_upper[arc] = value;
        is_pinned[arc] = true;
    }
    if let Some(seed) = a0 {
        for (i, &on) in seed.iter().enumerate() {
            if on {
                active[i] = true;
            }
        }
    }
    for i in 0..m {
        active[i] &= !forbid[i] && !upper[i];
    }
    if !active.iter().any(|&v| v) {
        // §5.4 warm start: every admissible arc active is the pure (f', f'')
        // answer, exact in the small-trade limit.
        for i in 0..m {
            active[i] = !forbid[i] && !upper[i];
        }
    }

    let mut psi = vec![0.0; m];
    let mut u = vec![0.0; n];
    let mut rho = vec![0.0; m];
    let mut comp = vec![false; n];
    let mut pivots = 0u32;
    // Bit-packed and hashed, not a list of vectors compared one by one.
    // Python keeps `(A.tobytes(), U.tobytes())` in a set; a linear scan over
    // full boolean vectors is O(pivots^2 * m), which at 300 nodes and 900 arcs
    // cost more than the whole solve it was guarding.
    let words = m.div_ceil(64);
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut use_bland = false;
    let mut cycles = 0u32;
    let mut reseeded = false;

    for _ in 0..opt.maxit {
        component_of(dst, arcs, &active, &mut comp);
        if !comp[src] && psi_total != 0.0 {
            // A disconnected active set is a starting point, not a verdict:
            // the arc joining src to dst may simply have saturated into `U`.
            let mut candidates = vec![false; m];
            let mut any = false;
            for i in 0..m {
                candidates[i] = !forbid[i] && !upper[i];
                any |= candidates[i];
            }
            if !reseeded && any && candidates != active {
                active = candidates;
                reseeded = true;
                continue;
            }
            return Solution {
                psi: vec![0.0; m], u: vec![0.0; n], active, upper, psi_upper,
                rho: vec![0.0; m], pivots, stop: Stop::SrcDetached,
            };
        }

        let mut rhs = s_hat.clone();
        for p in 0..m {
            if active[p] {
                let flow = arcs.g[p] * arcs.eps[p];
                rhs[arcs.tau[p] as usize] += flow;
                rhs[arcs.sig[p] as usize] -= flow;
            }
        }
        let mut stray: Vec<usize> = Vec::new();
        for p in 0..m {
            if upper[p] {
                rhs[arcs.tau[p] as usize] -= psi_upper[p];
                rhs[arcs.sig[p] as usize] += psi_upper[p];
                if !comp[arcs.tau[p] as usize] || !comp[arcs.sig[p] as usize] {
                    stray.push(p);
                }
            }
        }
        if !stray.is_empty() {
            // A caller's pin is the candidate's question, so a detached one is
            // fatal.  One that merely saturated was never asked for: release
            // it, which is the pivot the loop would have taken had it looked.
            let loose: Vec<usize> =
                stray.iter().copied().filter(|&p| !is_pinned[p]).collect();
            if !loose.is_empty() {
                for p in loose {
                    upper[p] = false;
                    psi_upper[p] = 0.0;
                }
                pivots += 1;
                continue;
            }
            return Solution {
                psi: vec![0.0; m], u: vec![0.0; n], active, upper, psi_upper,
                rho: vec![0.0; m], pivots, stop: Stop::PinDetached,
            };
        }

        // Solve on the kept nodes only: `dst` is grounded, and anything
        // outside `dst`'s component is not in the system at all.
        let keep: Vec<usize> = (0..n).filter(|&i| comp[i] && i != dst).collect();
        u.iter_mut().for_each(|v| *v = 0.0);
        if !keep.is_empty() {
            let k = keep.len();
            let mut index = vec![usize::MAX; n];
            for (slot, &node) in keep.iter().enumerate() {
                index[node] = slot;
            }
            let mut mat = vec![0.0; k * k];
            for p in 0..m {
                if !active[p] {
                    continue;
                }
                let (a, b) = (index[arcs.tau[p] as usize], index[arcs.sig[p] as usize]);
                let g = arcs.g[p];
                if a != usize::MAX {
                    mat[a * k + a] += g;
                }
                if b != usize::MAX {
                    mat[b * k + b] += g;
                }
                if a != usize::MAX && b != usize::MAX {
                    mat[a * k + b] -= g;
                    mat[b * k + a] -= g;
                }
            }
            let mut vec_b: Vec<f64> = keep.iter().map(|&node| rhs[node]).collect();
            if let Err(e) = lu::solve_in_place(&mut mat, &mut vec_b, k) {
                return Solution {
                    psi: vec![0.0; m], u: vec![0.0; n], active, upper, psi_upper,
                    rho: vec![0.0; m], pivots, stop: Stop::Singular(e.column),
                };
            }
            for (slot, &node) in keep.iter().enumerate() {
                u[node] = vec_b[slot];
            }
        }

        for p in 0..m {
            psi[p] = if upper[p] { psi_upper[p] } else { 0.0 };
        }
        for p in 0..m {
            if active[p] {
                psi[p] = arcs.g[p]
                    * (u[arcs.tau[p] as usize] - u[arcs.sig[p] as usize] - arcs.eps[p]);
            }
        }
        // §9.4: nodes outside `dst`'s component carry no flow by construction.
        for p in 0..m {
            if !comp[arcs.tau[p] as usize] || !comp[arcs.sig[p] as usize] {
                psi[p] = 0.0;
            }
        }
        for p in 0..m {
            rho[p] = u[arcs.tau[p] as usize] - u[arcs.sig[p] as usize] - arcs.eps[p];
        }

        let mut signature = vec![0u64; 2 * words];
        for p in 0..m {
            if active[p] {
                signature[p / 64] |= 1u64 << (p % 64);
            }
            if upper[p] {
                signature[words + p / 64] |= 1u64 << (p % 64);
            }
        }
        if seen.contains(&signature) {
            if use_bland {
                cycles += 1;
                if cycles >= CYCLE_PATIENCE {
                    if opt.partial_ok {
                        for v in psi.iter_mut() {
                            if v.abs() < opt.tol {
                                *v = 0.0;
                            }
                        }
                        return Solution {
                            psi, u, active, upper, psi_upper, rho, pivots,
                            stop: Stop::Partial,
                        };
                    }
                    return Solution {
                        psi, u, active, upper, psi_upper, rho, pivots,
                        stop: Stop::Cycling(pivots),
                    };
                }
            }
            use_bland = true;
        }
        seen.insert(signature);

        let mut mask = vec![false; m];
        let mut score = vec![0.0; m];

        // 1. an arc carrying negative flow leaves the basis
        let mut any = false;
        for p in 0..m {
            mask[p] = active[p] && psi[p] < -opt.tol;
            score[p] = -psi[p];
            any |= mask[p];
        }
        if any {
            let j = if use_bland { bland(&mask) } else { steepest(&mask, &score) };
            active[j] = false;
            pivots += 1;
            continue;
        }

        // 2. an arc past its cap moves to the upper bound
        any = false;
        for p in 0..m {
            mask[p] = active[p] && psi[p] > arcs.cap[p] + opt.tol;
            score[p] = psi[p] - arcs.cap[p];
            any |= mask[p];
        }
        if any {
            let j = if use_bland { bland(&mask) } else { steepest(&mask, &score) };
            active[j] = false;
            upper[j] = true;
            psi_upper[j] = arcs.cap[j];
            pivots += 1;
            continue;
        }

        // 3. an arc outside the basis that wants in
        any = false;
        for p in 0..m {
            let free = !active[p] && !upper[p] && !forbid[p];
            let mut want = free && rho[p] > opt.tol;
            if want && opt.min_flow > 0.0 {
                want = arcs.g[p] * rho[p] > opt.min_flow;
            }
            if want && opt.gas_cost > 0.0 {
                // What admitting it is worth: at reduced cost `rho` it settles
                // at `psi = G rho` and the objective falls by `G rho^2 / 2`.
                want = 0.5 * arcs.g[p] * rho[p] * rho[p] > opt.gas_cost;
            }
            mask[p] = want;
            score[p] = rho[p];
            any |= want;
        }
        if any {
            let j = if use_bland { bland(&mask) } else { steepest(&mask, &score) };
            active[j] = true;
            pivots += 1;
            continue;
        }

        // 4. an arc held at its cap that would rather come back
        any = false;
        for p in 0..m {
            mask[p] = upper[p] && rho[p] < -opt.tol && !is_pinned[p];
            score[p] = -rho[p];
            any |= mask[p];
        }
        if any {
            let j = if use_bland { bland(&mask) } else { steepest(&mask, &score) };
            upper[j] = false;
            active[j] = true;
            pivots += 1;
            continue;
        }

        for v in psi.iter_mut() {
            if v.abs() < opt.tol {
                *v = 0.0;
            }
        }
        return Solution {
            psi, u, active, upper, psi_upper, rho, pivots, stop: Stop::Optimal,
        };
    }

    // Out of pivots.  Every iterate satisfies conservation exactly -- `u`
    // solves the Laplacian with the conservation right-hand side -- so only
    // optimality is incomplete, and the caller decides whether that is usable.
    if !opt.partial_ok {
        return Solution {
            psi, u, active, upper, psi_upper, rho, pivots,
            stop: Stop::NoConvergence(opt.maxit),
        };
    }
    for v in psi.iter_mut() {
        if v.abs() < opt.tol {
            *v = 0.0;
        }
    }
    Solution { psi, u, active, upper, psi_upper, rho, pivots, stop: Stop::Partial }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arcs<'a>(
        tau: &'a [i64], sig: &'a [i64], g: &'a [f64], eps: &'a [f64], cap: &'a [f64],
        n: usize,
    ) -> Arcs<'a> {
        Arcs { tau, sig, g, eps, cap, n_nodes: n }
    }

    #[test]
    fn a_single_arc_carries_the_whole_trade() {
        let (tau, sig) = ([0i64], [1i64]);
        let (g, eps, cap) = ([1.0], [0.0], [f64::INFINITY]);
        let a = arcs(&tau, &sig, &g, &eps, &cap, 2);
        let out = active_set_solve(&a, 0, 1, 0.25, None, None, &[], &Options::default());
        assert!(out.stop.feasible());
        assert!((out.psi[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn parallel_arcs_split_in_proportion_to_conductance() {
        let (tau, sig) = ([0i64, 0], [1i64, 1]);
        let (g, eps, cap) = ([1.0, 2.0], [0.0, 0.0], [f64::INFINITY; 2]);
        let a = arcs(&tau, &sig, &g, &eps, &cap, 2);
        let out = active_set_solve(&a, 0, 1, 1.0, None, None, &[], &Options::default());
        assert!(out.stop.feasible());
        assert!((out.psi[1] / out.psi[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn the_diode_keeps_a_dearer_arc_shut_until_it_pays() {
        // eps 0 against eps 0.5: at a small trade the dear arc stays at zero.
        let (tau, sig) = ([0i64, 0], [1i64, 1]);
        let (g, eps, cap) = ([1.0, 1.0], [0.0, 0.5], [f64::INFINITY; 2]);
        let a = arcs(&tau, &sig, &g, &eps, &cap, 2);
        let small = active_set_solve(&a, 0, 1, 0.25, None, None, &[], &Options::default());
        assert_eq!(small.psi[1], 0.0, "the dear arc opened too early");
        let large = active_set_solve(&a, 0, 1, 4.0, None, None, &[], &Options::default());
        assert!(large.psi[1] > 0.0, "the dear arc never opened");
    }

    #[test]
    fn a_cap_binds_and_the_rest_goes_elsewhere() {
        let (tau, sig) = ([0i64, 0], [1i64, 1]);
        let (g, eps) = ([1.0, 1.0], [0.0, 0.1]);
        let cap = [0.2, f64::INFINITY];
        let a = arcs(&tau, &sig, &g, &eps, &cap, 2);
        let out = active_set_solve(&a, 0, 1, 1.0, None, None, &[], &Options::default());
        assert!(out.stop.feasible());
        assert!((out.psi[0] - 0.2).abs() < 1e-9, "the cap did not bind");
        assert!((out.psi[0] + out.psi[1] - 1.0).abs() < 1e-9, "flow was lost");
    }

    #[test]
    fn conservation_holds_on_a_network() {
        let (tau, sig) = ([0i64, 0, 1, 2, 0], [1i64, 2, 3, 3, 3]);
        let g = [1.0, 0.5, 1.0, 0.33, 2.0];
        let eps = [0.0005, 0.0010, 0.0005, 0.0002, 0.004];
        let cap = [f64::INFINITY; 5];
        let a = arcs(&tau, &sig, &g, &eps, &cap, 4);
        let out = active_set_solve(&a, 0, 3, 2.0, None, None, &[], &Options::default());
        assert!(out.stop.feasible());
        let mut net = vec![0.0; 4];
        for p in 0..5 {
            net[tau[p] as usize] += out.psi[p];
            net[sig[p] as usize] -= out.psi[p];
        }
        assert!((net[0] - 2.0).abs() < 1e-9, "source does not emit Psi");
        assert!((net[3] + 2.0).abs() < 1e-9, "sink does not absorb Psi");
        assert!(net[1].abs() < 1e-9 && net[2].abs() < 1e-9, "flow stranded mid-route");
    }

    #[test]
    fn a_pinned_arc_stays_where_it_was_put() {
        let (tau, sig) = ([0i64, 0], [1i64, 1]);
        let (g, eps, cap) = ([1.0, 1.0], [0.0, 0.0], [f64::INFINITY; 2]);
        let a = arcs(&tau, &sig, &g, &eps, &cap, 2);
        let out = active_set_solve(&a, 0, 1, 1.0, None, None, &[(0, 0.3)],
                                   &Options::default());
        assert!(out.stop.feasible());
        assert!((out.psi[0] - 0.3).abs() < 1e-12, "the pin moved");
        assert!((out.psi[1] - 0.7).abs() < 1e-9, "the rest did not follow");
    }
}
