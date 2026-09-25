//! Routing as a nonlinear circuit: Newton on node prices (mirror of
//! `core/circuit.py`).
//!
//! Unknowns are node prices, the destination held at 1. Each arc is a
//! two-terminal device trading the `d` that maximises its value plus a
//! barrier, so the routing dual is convex: its gradient is the KCL residual,
//! its Hessian a weighted Laplacian of rank-one stamps. The barriers are
//! SPICE's gmin and step down over three stages. The Laplacian's pattern is
//! the graph's, fixed for the quote, so it is ordered once (`sparse`) and only
//! refactored per iteration.
//!
//! The reference solves its Newton step densely; this one does not, so the two
//! agree to rounding rather than bit for bit.

use crate::candidates::{
    conflicting_pools, keep_only, legs_of, port_ids, repair_order, Candidate, CandidateSet,
    MIN_FLOW_FRACTION,
};
use crate::cycles::cancel_cycles;
use crate::realize::prune_dust;
use crate::sparse::Symbolic;
use crate::types::{ArcKind, PoolArc};
use std::collections::HashSet;

pub const ARMIJO: f64 = 1e-4;
pub const MAX_ITER: usize = 60;
pub const MU_START: f64 = 1e-2;
pub const MU_END: f64 = 1e-10;
pub const STAGES: usize = 3;
pub const RESUME_MU: f64 = 1e-7;
pub const RESUME_STAGES: usize = 2;
pub const TOL: f64 = 1e-9;
pub const INNER: usize = 80;
pub const GMIN: f64 = 1e-14;
pub const LINEAR: f64 = 1e-6;
pub const MAX_ROUNDS: usize = 12;
/// `realize.DUST_SHARE` and the tolerance `prune_dust`/`cancel_cycles` take.
const DUST_SHARE: f64 = 1e-4;
const FLOW_TOL: f64 = 1e-12;

fn is_conversion(kind: ArcKind) -> bool {
    matches!(kind, ArcKind::WrapNative | ArcKind::UnwrapNative | ArcKind::WstethWrap
        | ArcKind::WstethUnwrap | ArcKind::Erc4626Deposit | ArcKind::Erc4626Redeem)
}

/// Every arc as a device: `a d - B d^2/2`, or `a d / (1 + k d)` with
/// `k = B / 2a` for a v3/v4 tick range, on `[0, cap]`.
#[derive(Debug, Clone)]
pub struct Devices {
    pub tau: Vec<usize>,
    pub sig: Vec<usize>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub cap: Vec<f64>,
    pub exact: Vec<bool>,
    pub n: usize,
    pub d: Vec<f64>,
}

impl Devices {
    fn live(&self, k: usize) -> bool {
        self.cap[k] > 0.0
    }
}

/// The arcs as devices, in canonical units, for a trade worth `v0`.
pub fn devices(arcs: &[PoolArc], n_nodes: usize, nu0: &[f64], v0: f64) -> Devices {
    let m = arcs.len();
    let (mut a, mut b, mut cap, mut exact) =
        (vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![false; m]);
    for (k, arc) in arcs.iter().enumerate() {
        a[k] = arc.a;
        let whole = v0 / nu0[arc.tau].max(1e-300);
        let floor = 1e-12 * arc.a.max(1e-300);
        b[k] = arc.b.max(floor).max(2.0 * LINEAR * arc.a / whole);
        // A v2 pair is one range of the exact curve at any size; its cap bounded
        // only the tangent quadratic (`circuit.devices`).
        let pair = arc.kind == ArcKind::SwapUniv2;
        exact[k] = pair
            || matches!(arc.kind, ArcKind::SwapUniv3 | ArcKind::SwapUniv4) && arc.parallel;
        let c = if pair {
            let reserve = arc.reserve_in as f64 / 10f64.powi(arc.decimals_in as i32) * arc.rate_in;
            if reserve > 0.0 { (2.0 * whole).min(crate::realize::PAIR_DEPTH * reserve) } else { 2.0 * whole }
        } else if exact[k] {
            arc.cap
        } else {
            arc.cap.min(a[k] / b[k])
        };
        cap[k] = if c.is_finite() && a[k] > 0.0 { c } else { 0.0 };
    }
    let d = cap.iter().map(|c| c * 1e-6).collect();
    Devices {
        tau: arcs.iter().map(|x| x.tau).collect(),
        sig: arcs.iter().map(|x| x.sigma).collect(),
        a, b, cap, exact, n: n_nodes, d,
    }
}

/// f' and f'' at d for arc k.
fn slopes(dev: &Devices, k: usize, d: f64) -> (f64, f64) {
    let (a, b) = (dev.a[k], dev.b[k]);
    if dev.exact[k] {
        let kk = b / (2.0 * a);
        let x = 1.0 + kk * d;
        (a / (x * x), -2.0 * a * kk / (x * x * x))
    } else {
        (a - b * d, -b)
    }
}

fn value(dev: &Devices, k: usize, d: f64) -> f64 {
    let (a, b) = (dev.a[k], dev.b[k]);
    if dev.exact[k] {
        a * d / (1.0 + b / (2.0 * a.max(1e-300)) * d)
    } else {
        a * d - 0.5 * b * d * d
    }
}

/// Every live arc's trade at prices `nu`: `(d, f(d), f'(d), dF/dd)`.
pub fn respond(dev: &Devices, nu: &[f64], t: f64, d0: Option<&[f64]>)
    -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = dev.a.len();
    let start = d0.unwrap_or(&dev.d);
    let mut d = vec![0.0; m];
    let mut f = vec![0.0; m];
    let mut f1 = vec![0.0; m];
    let mut big_d = vec![0.0; m];
    for k in 0..m {
        let (nu_in, nu_out) = (nu[dev.tau[k]], nu[dev.sig[k]]);
        if !dev.live(k) {
            let (s1, s2) = slopes(dev, k, 0.0);
            f1[k] = s1;
            big_d[k] = nu_out * s2 - 1.0;
            continue;
        }
        let cap = dev.cap[k];
        let mut lo = (t / (2.0 * nu_in.max(1e-300))).min(cap * 0.5);
        let mut hi = cap;
        let residual = |x: f64| {
            let (s1, _) = slopes(dev, k, x);
            nu_out * s1 - nu_in + t / x - t / (cap - x)
        };
        // Start from the warm point or the barrier-free answer, whichever the
        // equation likes better: from the warm point alone every arc in the
        // leak regime walked ~12 steps when t fell a hundredfold.
        let mut x = start[k].max(lo).min(cap * (1.0 - 1e-12));
        if !(x > lo && x < hi) {
            x = (lo * hi).sqrt();
        }
        let guess = barrier_free(dev, k, nu_in, nu_out, t);
        if guess > lo && guess < hi && residual(guess).abs() < residual(x).abs() {
            x = guess;
        }
        for _ in 0..INNER {
            let (s1, s2) = slopes(dev, k, x);
            let (pay, leak, wall) = (nu_out * s1, t / x, t / (cap - x));
            let big_f = pay - nu_in + leak - wall;
            // Zero to rounding: F is a difference of terms of order one, and in
            // the leak regime its derivative is t/d small, so a step test at
            // 1e-14 could never pass and every such arc ran to INNER.
            if big_f.abs() <= 8.0 * f64::EPSILON * (pay.abs() + nu_in + leak + wall) {
                break;
            }
            let df = nu_out * s2 - t / (x * x) - t / ((cap - x) * (cap - x));
            if big_f > 0.0 { lo = x; } else { hi = x; }
            let mut new = x * (-big_f / (df * x)).clamp(-50.0, 50.0).exp();
            let bad = !(new > lo && new < hi);
            if bad {
                new = (lo * hi).sqrt();
            }
            let done = (!bad && (new / x).ln().abs() <= 1e-14) || hi <= lo * (1.0 + 1e-15);
            x = new;
            if done {
                break;
            }
        }
        d[k] = x;
        let (s1, s2) = slopes(dev, k, x);
        f1[k] = s1;
        f[k] = value(dev, k, x);
        big_d[k] = nu_out * s2 - (t / (x * x) + t / ((cap - x) * (cap - x)));
    }
    (d, f, f1, big_d)
}

/// The grounded system: every node but `dst`, ordered once.
struct System {
    /// Node -> row, `None` for `dst`.
    row: Vec<Option<usize>>,
    sym: Symbolic,
    /// Per arc: slots of (tau,tau), (sig,sig), (tau,sig) in the value array.
    slot_tt: Vec<Option<usize>>,
    slot_ss: Vec<Option<usize>>,
    slot_ts: Vec<Option<usize>>,
    slot_diag: Vec<usize>,
}

fn system(arcs: &[PoolArc], n_nodes: usize, dst: usize) -> System {
    let mut row = vec![None; n_nodes];
    let mut node_of = Vec::new();
    for (v, r) in row.iter_mut().enumerate() {
        if v != dst {
            *r = Some(node_of.len());
            node_of.push(v);
        }
    }
    let edges: Vec<(usize, usize)> = arcs.iter()
        .filter_map(|a| match (row[a.tau], row[a.sigma]) {
            (Some(i), Some(j)) => Some((i, j)),
            _ => None,
        })
        .collect();
    let sym = Symbolic::new(node_of.len(), &edges);
    let pick = |x: usize, y: usize| match (row[x], row[y]) {
        (Some(i), Some(j)) => sym.slot(i, j),
        _ => None,
    };
    let slot_tt = arcs.iter().map(|a| pick(a.tau, a.tau)).collect();
    let slot_ss = arcs.iter().map(|a| pick(a.sigma, a.sigma)).collect();
    let slot_ts = arcs.iter().map(|a| pick(a.tau, a.sigma)).collect();
    let slot_diag = (0..node_of.len())
        .map(|r| sym.slot(r, r).expect("the diagonal is in the pattern"))
        .collect();
    System { row, sym, slot_tt, slot_ss, slot_ts, slot_diag }
}

/// Where arc k settles as `t -> 0`: at `f'(d) = nu_in / nu_out` when that is
/// inside the range, else a leak of `t / gap` off whichever end it rests on.
fn barrier_free(dev: &Devices, k: usize, nu_in: f64, nu_out: f64, t: f64) -> f64 {
    let (a, b, cap) = (dev.a[k], dev.b[k], dev.cap[k]);
    if !(nu_out > 0.0) {
        return t / nu_in.max(1e-300);
    }
    let r = nu_in / nu_out;
    if r >= a {
        return t / (nu_in - nu_out * a).max(1e-300);
    }
    let inside = if dev.exact[k] {
        let kk = b / (2.0 * a);
        ((a / r).sqrt() - 1.0) / kk
    } else {
        (a - r) / b
    };
    if inside < cap {
        return inside;
    }
    let (edge, _) = slopes(dev, k, cap);
    cap - t / (nu_out * edge - nu_in).abs().max(1e-300)
}

struct Problem<'a> {
    dev: &'a Devices,
    q: f64,
    src: usize,
    dst: usize,
    sys: &'a System,
}

type Response = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn objective(p: &Problem, nu: &[f64], t: f64, kappa: f64, d0: &[f64]) -> (f64, Response) {
    let dev = p.dev;
    let (d, f, f1, big_d) = respond(dev, nu, t, Some(d0));
    let mut arcs = 0.0;
    for k in 0..d.len() {
        if dev.live(k) {
            let barrier = d[k].max(1e-300).ln() + (dev.cap[k] - d[k]).max(1e-300).ln();
            arcs += nu[dev.sig[k]] * f[k] - nu[dev.tau[k]] * d[k] + t * barrier;
        }
    }
    let prices: f64 = (0..dev.n).filter(|&v| v != p.dst).map(|v| nu[v].ln()).sum();
    (p.q * nu[p.src] + arcs - kappa * prices, (d, f, f1, big_d))
}

pub struct Solved {
    pub nu: Vec<f64>,
    pub delta: Vec<f64>,
    pub out: Vec<f64>,
    pub iterations: usize,
    pub residual: f64,
}

fn logspace(from: f64, to: f64, n: usize) -> Vec<f64> {
    let (a, b) = (from.log10(), to.log10());
    (0..n).map(|k| 10f64.powf(if n > 1 { a + (b - a) * k as f64 / (n - 1) as f64 } else { a }))
        .collect()
}

fn solve(p: &Problem, dev_d: &mut Vec<f64>, nu0: &[f64], nu_start: Option<&[f64]>,
         mu_start: f64, stages: usize) -> Solved {
    let dev = p.dev;
    let n = dev.n;
    let scale: Vec<f64> = nu0.iter().map(|&v| if v > 0.0 { v } else { 1.0 }).collect();
    let mut nu: Vec<f64> = nu_start.unwrap_or(nu0).to_vec();
    let v0 = p.q * nu0[p.src];
    let m_live = (0..dev.a.len()).filter(|&k| dev.live(k)).count().max(1) as f64;
    let (mut total, mut residual) = (0usize, f64::INFINITY);
    let schedule = logspace(mu_start, MU_END, stages);
    let mut t = 0.0;
    let mut work = dev.clone();
    for (stage, &mu) in schedule.iter().enumerate() {
        t = mu * v0 / m_live;
        let kappa = mu * v0 / n as f64;
        let stage_tol = if stage == schedule.len() - 1 { TOL } else { TOL.max(10.0 * mu) };
        let mut its = 0usize;
        let mut g_cur: Option<f64> = None;
        // The accepted trial's response is the next iterate's: same prices, same t.
        let mut held: Option<Response> = None;
        for _ in 0..MAX_ITER {
            its += 1;
            let (d, f, f1, big_d) = match held.take() {
                Some(r) => r,
                None => {
                    work.d.clone_from(dev_d);
                    respond(&work, &nu, t, None)
                }
            };
            *dev_d = d.clone();
            // Gradient: the KCL residual in tokens, less the price barrier.
            let mut grad = vec![0.0; n];
            for k in 0..d.len() {
                if dev.live(k) {
                    grad[dev.tau[k]] -= d[k];
                    grad[dev.sig[k]] += f[k];
                }
            }
            grad[p.src] += p.q;
            for v in 0..n {
                if v != p.dst {
                    grad[v] -= kappa / nu[v];
                }
            }
            residual = (0..n).filter(|&v| v != p.dst)
                .map(|v| (grad[v] * nu[v]).abs()).fold(0.0, f64::max) / v0;
            if residual < stage_tol {
                break;
            }
            // Hessian in the scaled prices, stamped into the fixed pattern.
            let mut ax = vec![0.0; p.sys.sym.nnz()];
            for k in 0..d.len() {
                if !dev.live(k) {
                    continue;
                }
                let w = -1.0 / big_d[k];
                let (i, j) = (dev.tau[k], dev.sig[k]);
                if let Some(s) = p.sys.slot_tt[k] { ax[s] += w * scale[i] * scale[i]; }
                if let Some(s) = p.sys.slot_ss[k] { ax[s] += w * f1[k] * f1[k] * scale[j] * scale[j]; }
                if let Some(s) = p.sys.slot_ts[k] { ax[s] -= w * f1[k] * scale[i] * scale[j]; }
            }
            let mut max_diag = 0.0f64;
            for v in 0..n {
                if let Some(r) = p.sys.row[v] {
                    let s = p.sys.slot_diag[r];
                    ax[s] += kappa / (nu[v] * nu[v]) * scale[v] * scale[v];
                    max_diag = max_diag.max(ax[s]);
                }
            }
            let shunt = GMIN * max_diag.max(1e-300);
            for r in 0..p.sys.sym.n {
                ax[p.sys.slot_diag[r]] += shunt;
            }
            let mut step = vec![0.0; p.sys.sym.n];
            for v in 0..n {
                if let Some(r) = p.sys.row[v] {
                    step[r] = -grad[v] * scale[v];
                }
            }
            match p.sys.sym.factor(&ax) {
                Some(fac) => p.sys.sym.solve(&fac, &mut step),
                None => {} // a non-positive pivot: steepest descent, as the reference falls back
            }
            let mut direction = vec![0.0; n];
            for v in 0..n {
                if let Some(r) = p.sys.row[v] {
                    direction[v] = step[r] * scale[v];
                }
            }
            let g0 = match g_cur {
                Some(g) => g,
                None => objective(p, &nu, t, kappa, dev_d).0,
            };
            let mut trial_response: Option<Response> = None;
            let slope: f64 = (0..n).map(|v| grad[v] * direction[v]).sum();
            let mut s = 1.0f64;
            for v in 0..n {
                if direction[v] < 0.0 {
                    s = s.min(0.99 * (-nu[v] / direction[v]));
                }
            }
            let mut accepted = false;
            while s > 1e-12 {
                let trial: Vec<f64> = (0..n).map(|v| nu[v] + s * direction[v]).collect();
                let (g1, r1) = objective(p, &trial, t, kappa, dev_d);
                if g1 <= g0 + ARMIJO * s * slope {
                    dev_d.clone_from(&r1.0);
                    trial_response = Some(r1);
                    g_cur = Some(g1);
                    accepted = true;
                    break;
                }
                s *= 0.5;
            }
            if !accepted {
                g_cur = None;
            }
            for v in 0..n {
                nu[v] += s * direction[v];
            }
            held = if accepted { trial_response } else { None };
        }
        total += its;
    }
    work.d.clone_from(dev_d);
    let (d, f, _, _) = respond(&work, &nu, t, None);
    *dev_d = d.clone();
    Solved { nu, delta: d, out: f, iterations: total, residual }
}

/// The quoter's ABI capacity, `circuit.candidates`'s default leg budget.
pub const MAX_LEGS: usize = 128;

/// What the circuit needs beyond the graph.
#[derive(Debug, Clone)]
pub struct CircuitOptions {
    pub advanceable: Option<HashSet<String>>,
    pub leg_cost_bp: f64,
    /// One unit of gas in canonical destination units.
    pub per_gas: f64,
    /// Each arc's gas, as `verify`'s table would charge its leg.
    pub gas: Vec<f64>,
    /// The most legs a candidate may realise as.
    pub max_legs: usize,
}

impl Default for CircuitOptions {
    fn default() -> Self {
        CircuitOptions {
            advanceable: None,
            leg_cost_bp: 0.0,
            per_gas: 0.0,
            gas: Vec::new(),
            max_legs: MAX_LEGS,
        }
    }
}

/// The circuit's answer, as a ballot of two for `verify` to adjudicate
/// (`circuit.candidates`). `psi_total` and the candidates are in the graph's
/// scaled value units.
#[allow(clippy::too_many_arguments)]
pub fn candidates(arcs: &[PoolArc], n_nodes: usize, g_scale: f64, nu: &[f64],
                  src: usize, dst: usize, psi_total: f64, opts: &CircuitOptions) -> CandidateSet {
    let v = psi_total * g_scale;
    let q = v / nu[src];
    let nu0: Vec<f64> = nu.iter().map(|x| x / nu[dst]).collect();
    let mut dev = devices(arcs, n_nodes, &nu0, q * nu0[src]);
    let port = port_ids(arcs);
    let tau64: Vec<i64> = arcs.iter().map(|a| a.tau as i64).collect();
    let sig64: Vec<i64> = arcs.iter().map(|a| a.sigma as i64).collect();

    let sys = system(arcs, n_nodes, dst);

    let mut out = CandidateSet::default();
    let flow = |res: &Solved| -> Vec<f64> {
        (0..arcs.len()).map(|k| {
            let psi = res.delta[k] * nu[arcs[k].tau] / g_scale;
            if psi >= MIN_FLOW_FRACTION * psi_total { psi } else { 0.0 }
        }).collect()
    };

    let mut dev_d = dev.d.clone();
    let first = {
        let p = Problem { dev: &dev, q, src, dst, sys: &sys };
        solve(&p, &mut dev_d, &nu0, None, MU_START, STAGES)
    };
    out.solves += 1;
    out.pivots += first.iterations;

    let settle = |dev: &mut Devices, dev_d: &mut Vec<f64>, out: &mut CandidateSet,
                      mut res: Solved, prune: bool| -> (Solved, Vec<f64>) {
        let mut psi = flow(&res);
        for rounds in 0..MAX_ROUNDS {
            let mut banned = vec![false; arcs.len()];
            let clash = conflicting_pools(arcs, &psi, psi_total, None, None, opts.advanceable.as_ref());
            if !clash.is_empty() {
                keep_only(&mut banned, &repair_order(&clash, &psi, Some(&port)), 0, &[], Some(&port));
            }
            if prune && rounds == 0 {
                for (k, b) in unearned(dev, &res, arcs, &psi, &port, dst, opts).into_iter().enumerate() {
                    banned[k] |= b;
                }
            }
            if !banned.iter().any(|&b| b) {
                banned = over_budget(dev, &res, arcs, &psi, &port, opts.max_legs);
            }
            if !banned.iter().any(|&b| b) {
                break;
            }
            for (k, &b) in banned.iter().enumerate() {
                if b {
                    dev.cap[k] = 0.0;
                }
            }
            let p = Problem { dev, q, src, dst, sys: &sys };
            res = solve(&p, dev_d, &nu0, Some(&res.nu), RESUME_MU, RESUME_STAGES);
            out.solves += 1;
            out.pivots += res.iterations;
            psi = flow(&res);
        }
        (res, psi)
    };

    let (settled, psi) = settle(&mut dev, &mut dev_d, &mut out, first, false);
    offer(&mut out, &tau64, &sig64, n_nodes, psi, src, dst, "circuit");
    let (saved, warm) = (dev.cap.clone(), dev_d.clone());
    let (_, psi) = settle(&mut dev, &mut dev_d, &mut out, settled, true);
    offer(&mut out, &tau64, &sig64, n_nodes, psi, src, dst, "circuit, pruned");
    dev.cap = saved;
    dev.d = warm;
    out
}

/// Each port's surplus at the solved prices, by its head arc, in the order
/// the ports first carry flow.
fn earned(dev: &Devices, res: &Solved, psi: &[f64], port: &[usize]) -> Vec<(usize, f64)> {
    let mut earned: Vec<(usize, f64)> = Vec::new();
    for k in 0..psi.len() {
        if psi[k] > 0.0 {
            let surplus = if dev.live(k) {
                res.nu[dev.sig[k]] * res.out[k] - res.nu[dev.tau[k]] * res.delta[k]
            } else {
                0.0
            };
            match earned.iter_mut().find(|(h, _)| *h == port[k]) {
                Some(e) => e.1 += surplus,
                None => earned.push((port[k], surplus)),
            }
        }
    }
    earned
}

/// Arcs of every port whose surplus is below its leg charge.
fn unearned(dev: &Devices, res: &Solved, arcs: &[PoolArc], psi: &[f64], port: &[usize],
            dst: usize, opts: &CircuitOptions) -> Vec<bool> {
    let m = arcs.len();
    let mut premium = 0.0;
    for k in 0..m {
        if dev.sig[k] == dst {
            premium += res.out[k];
        }
    }
    premium *= opts.leg_cost_bp / 1e4;
    let mut banned = vec![false; m];
    for (head, got) in earned(dev, res, psi, port) {
        let arc = &arcs[head];
        let gas = opts.gas.get(head).copied().unwrap_or(0.0) * opts.per_gas;
        let charge = if is_conversion(arc.kind) { gas } else { premium.max(gas) };
        if got < charge {
            for k in 0..m {
                if port[k] == head {
                    banned[k] = true;
                }
            }
        }
    }
    banned
}

/// The legs this flow realises as, conversions included (`circuit._legs`):
/// ports, plus a wrap or unwrap for each token past the first a node uses.
fn legs(arcs: &[PoolArc], psi: &[f64]) -> usize {
    let mut tokens: Vec<(usize, HashSet<String>)> = Vec::new();
    let mut add = |node: usize, token: &str| {
        let token = token.to_ascii_lowercase();
        match tokens.iter_mut().find(|(n, _)| *n == node) {
            Some((_, set)) => {
                set.insert(token);
            }
            None => tokens.push((node, HashSet::from([token]))),
        }
    };
    for (arc, &f) in arcs.iter().zip(psi) {
        if f > 0.0 {
            add(arc.tau, &arc.token_in);
            add(arc.sigma, &arc.token_out);
        }
    }
    legs_of(arcs, psi, None) + tokens.iter().map(|(_, set)| set.len() - 1).sum::<usize>()
}

/// The weakest ports, one per leg over `max_legs` (`circuit._over_budget`).
fn over_budget(dev: &Devices, res: &Solved, arcs: &[PoolArc], psi: &[f64], port: &[usize],
               max_legs: usize) -> Vec<bool> {
    let mut banned = vec![false; arcs.len()];
    let excess = legs(arcs, psi).saturating_sub(max_legs);
    if excess == 0 {
        return banned;
    }
    let mut ranked = earned(dev, res, psi, port);
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    for &(head, _) in ranked.iter().take(excess) {
        for k in 0..arcs.len() {
            if port[k] == head {
                banned[k] = true;
            }
        }
    }
    banned
}

#[allow(clippy::too_many_arguments)]
fn offer(out: &mut CandidateSet, tau: &[i64], sig: &[i64], n_nodes: usize, psi: Vec<f64>,
         src: usize, dst: usize, label: &str) {
    let (psi, _) = cancel_cycles(tau, sig, &psi, FLOW_TOL, n_nodes);
    let (psi, _) = prune_dust(tau, sig, &psi, src, dst, DUST_SHARE, FLOW_TOL);
    if !psi.iter().any(|&x| x > 0.0) {
        return;
    }
    let same = |c: &Candidate| c.psi.len() == psi.len()
        && c.psi.iter().zip(&psi).all(|(a, b)| (a - b).abs() <= 1e-12 * b.abs());
    if out.candidates.iter().any(same) {
        return;
    }
    let n_arcs = psi.iter().filter(|&&x| x > 0.0).count();
    out.candidates.push(Candidate::new(label.to_string(), psi, false, "", "circuit", n_arcs, 0.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn arc(k: usize, pool: usize, tau: usize, sigma: usize, a: f64, b: f64, cap: f64,
           kind: ArcKind, i: i32, j: i32, n: i32, parallel: bool) -> PoolArc {
        let mut x = PoolArc::new(format!("{pool}:{k}"), format!("0x{pool:040x}"), kind, i, j, n,
                                 format!("in{k}"), format!("out{k}"), tau, sigma);
        x.a = a;
        x.b = b;
        x.cap = cap;
        x.parallel = parallel;
        x
    }

    fn swap(k: usize, tau: usize, sigma: usize, a: f64, b: f64, cap: f64) -> PoolArc {
        arc(k, k + 1, tau, sigma, a, b, cap, ArcKind::SwapStable, 0, 1, 2, false)
    }

    fn run(arcs: &[PoolArc], q: f64, n: usize, src: usize, dst: usize) -> Solved {
        let nu0 = vec![1.0; n];
        let dev = devices(arcs, n, &nu0, q);
        let sys = system(arcs, n, dst);
        let p = Problem { dev: &dev, q, src, dst, sys: &sys };
        let mut d = dev.d.clone();
        solve(&p, &mut d, &nu0, None, MU_START, STAGES)
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn one_arc_carries_the_whole_trade() {
        let res = run(&[swap(0, 0, 1, 1.0, 1e-3, f64::INFINITY)], 100.0, 2, 0, 1);
        assert!(close(res.delta[0], 100.0, 1e-8), "{}", res.delta[0]);
        assert!(close(res.out[0], 100.0 - 0.5e-3 * 1e4, 1e-8));
        assert!(res.residual < TOL);
    }

    #[test]
    fn parallel_arcs_settle_at_one_marginal_rate() {
        let res = run(&[swap(0, 0, 1, 1.0, 1e-3, f64::INFINITY),
                        swap(1, 0, 1, 0.99, 2e-3, f64::INFINITY)], 100.0, 2, 0, 1);
        let m0 = 1.0 - 1e-3 * res.delta[0];
        let m1 = 0.99 - 2e-3 * res.delta[1];
        assert!((m0 - m1).abs() < 1e-9, "{m0} vs {m1}");
    }

    #[test]
    fn a_cheap_capped_arc_saturates() {
        let res = run(&[swap(0, 0, 1, 1.0, 1e-6, 30.0),
                        swap(1, 0, 1, 0.95, 1e-3, f64::INFINITY)], 100.0, 2, 0, 1);
        assert!(close(res.delta[0], 30.0, 1e-6) && close(res.delta[1], 70.0, 1e-6));
    }

    #[test]
    fn a_tick_range_is_its_exact_curve() {
        let tick = arc(0, 1, 0, 1, 2.0, 2.0 * 2.0 * 0.01, 1e9, ArcKind::SwapUniv3, 0, 1, 2, true);
        let res = run(&[tick], 50.0, 2, 0, 1);
        assert!(close(res.out[0], 2.0 * 50.0 / 1.5, 1e-8), "{}", res.out[0]);
    }

    #[test]
    fn a_dead_end_token_does_not_stall_the_solve() {
        let res = run(&[swap(0, 0, 1, 1.0, 1e-3, f64::INFINITY),
                        swap(1, 0, 2, 1.0, 1e-3, f64::INFINITY)], 100.0, 3, 0, 1);
        assert!(res.residual < TOL && res.iterations < 60, "{} after {}", res.residual, res.iterations);
        assert!(res.delta[1] < 1e-6 * res.delta[0]);
    }

    #[test]
    fn a_pool_is_not_counted_twice_across_two_ports() {
        let tri = 9;
        let arcs = vec![
            arc(0, tri, 0, 1, 1.0, 2e-3, f64::INFINITY, ArcKind::SwapCrypto, 0, 1, 3, false),
            arc(1, tri, 0, 2, 1.0, 2e-3, f64::INFINITY, ArcKind::SwapCrypto, 0, 2, 3, false),
            swap(2, 1, 3, 1.0, 1e-4, f64::INFINITY),
            swap(3, 2, 3, 1.0, 1e-4, f64::INFINITY),
            swap(4, 0, 3, 0.98, 1e-3, f64::INFINITY),
        ];
        let opts = CircuitOptions { advanceable: Some(HashSet::new()), ..Default::default() };
        let got = candidates(&arcs, 4, 1.0, &[1.0; 4], 0, 3, 100.0, &opts);
        assert!(!got.candidates.is_empty());
        for c in &got.candidates {
            assert!(c.psi[0] == 0.0 || c.psi[1] == 0.0);
        }
    }

    #[test]
    fn the_pruned_candidate_drops_a_branch_worth_less_than_its_leg() {
        let arcs = vec![swap(0, 0, 1, 1.0, 1e-4, f64::INFINITY),
                        swap(1, 0, 1, 0.995, 1e-4, f64::INFINITY)];
        let opts = CircuitOptions { leg_cost_bp: 5.0, gas: vec![0.0; 2], ..Default::default() };
        let got = candidates(&arcs, 2, 1.0, &[1.0; 2], 0, 1, 100.0, &opts);
        let labels: Vec<&str> = got.candidates.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, ["circuit", "circuit, pruned"]);
        assert!(got.candidates[0].psi[1] > 0.0 && got.candidates[1].psi[1] == 0.0);
    }

    #[test]
    fn a_route_over_the_leg_limit_loses_its_weakest_ports() {
        // Six pools on one pair of tokens, so no conversion is counted.
        let arcs: Vec<PoolArc> = (0..6).map(|k| {
            let mut x = swap(k, 0, 1, 1.0 - 0.002 * k as f64, 1e-3, f64::INFINITY);
            x.token_in = "0xa".into();
            x.token_out = "0xb".into();
            x
        }).collect();
        let free = candidates(&arcs, 2, 1.0, &[1.0; 2], 0, 1, 100.0, &CircuitOptions::default());
        assert_eq!(free.candidates[0].psi.iter().filter(|&&x| x > 0.0).count(), 6);
        let opts = CircuitOptions { max_legs: 3, ..Default::default() };
        let got = candidates(&arcs, 2, 1.0, &[1.0; 2], 0, 1, 100.0, &opts);
        assert!(!got.candidates.is_empty());
        for c in &got.candidates {
            let live: Vec<usize> = (0..6).filter(|&k| c.psi[k] > 0.0).collect();
            assert_eq!(live, [0, 1, 2]);
        }
    }

    #[test]
    fn a_node_drawing_on_two_of_its_tokens_needs_a_conversion() {
        let mut arcs = vec![swap(0, 0, 1, 1.0, 1e-3, f64::INFINITY),
                            swap(1, 0, 1, 1.0, 1e-3, f64::INFINITY)];
        for x in arcs.iter_mut() {
            x.token_out = "0xb".into();
        }
        arcs[0].token_in = "0xa".into();
        arcs[1].token_in = "0xa2".into();
        assert_eq!(legs(&arcs, &[1.0, 1.0]), 3);
        assert_eq!(legs(&arcs, &[1.0, 0.0]), 1);
    }
}
