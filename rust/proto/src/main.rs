//! The candidate stage with no Python in it, on a problem Python froze.
//!
//! `scripts/dump_quote.py` writes one prepared quote -- the solver's own index
//! space, the base solution, and every candidate solve exactly as it was asked
//! -- and this replays it. The measurement it exists for:
//!
//!   candidates   40.7 ms in Python, of which 18.1 is already this solver
//!   realize      ~18 ms of that, once per candidate
//!
//! So the question is not whether Rust is faster at the solve; it already does
//! the solve. It is what the ~22 ms of Python around it costs when it is not
//! Python. Anything this prints that is close to 18 ms means the solve count is
//! the problem and porting buys nothing.

mod stableswap;

use erouter_solve::cycles::cancel_cycles;
use erouter_solve::solve::{active_set_solve, Arcs, Options};
use std::time::Instant;

const DUST: f64 = 1e-12;

fn floats(v: &serde_json::Value, key: &str) -> Vec<f64> {
    v[key].as_array().unwrap().iter()
        .map(|x| x.as_f64().unwrap_or(f64::INFINITY)).collect()
}

fn ints(v: &serde_json::Value, key: &str) -> Vec<i64> {
    v[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn mask(v: &serde_json::Value) -> Option<Vec<bool>> {
    v.as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0) != 0).collect())
}

/// Kahn over the arcs carrying flow. The order legs must execute in.
fn topological(tau: &[i64], sig: &[i64], live: &[usize], n_nodes: usize)
    -> Option<Vec<usize>> {
    let mut indeg = vec![0usize; n_nodes];
    for &k in live {
        indeg[sig[k] as usize] += 1;
    }
    let mut queue: Vec<usize> = (0..n_nodes).filter(|&n| indeg[n] == 0).collect();
    let mut order = Vec::with_capacity(live.len());
    let mut seen = vec![false; live.len()];
    while let Some(node) = queue.pop() {
        for (idx, &k) in live.iter().enumerate() {
            if seen[idx] || tau[k] as usize != node {
                continue;
            }
            seen[idx] = true;
            order.push(k);
            let head = sig[k] as usize;
            indeg[head] -= 1;
            if indeg[head] == 0 {
                queue.push(head);
            }
        }
    }
    if order.len() == live.len() { Some(order) } else { None }
}

/// What `realize` computes: a flow becomes ordered legs against slots, with
/// each leg's input the share of the balance standing where it starts.
fn realize(tau: &[i64], sig: &[i64], psi: &[f64], n_nodes: usize, amount_in: f64)
    -> usize {
    let (flow, _removed) = cancel_cycles(tau, sig, psi, DUST, n_nodes);
    let live: Vec<usize> = (0..flow.len()).filter(|&k| flow[k] > DUST).collect();
    if live.is_empty() {
        return 0;
    }
    let order = match topological(tau, sig, &live, n_nodes) {
        Some(o) => o,
        None => return 0,
    };
    // Slots: one per node the route actually touches, in first-seen order.
    let mut slot_of = vec![usize::MAX; n_nodes];
    let mut slots = 0usize;
    for &k in &order {
        for node in [tau[k] as usize, sig[k] as usize] {
            if slot_of[node] == usize::MAX {
                slot_of[node] = slots;
                slots += 1;
            }
        }
    }
    // Forward simulate, the way the contract walks it: a leg takes its share
    // of what stands at its source, and pays it into its destination.
    let mut balance = vec![0.0f64; slots];
    let mut outflow = vec![0.0f64; n_nodes];
    for &k in &order {
        outflow[tau[k] as usize] += flow[k];
    }
    balance[slot_of[tau[order[0]] as usize]] = amount_in;
    let mut legs = 0usize;
    for &k in &order {
        let from = slot_of[tau[k] as usize];
        let to = slot_of[sig[k] as usize];
        let total = outflow[tau[k] as usize];
        let share = if total > 0.0 { flow[k] / total } else { 0.0 };
        let dx = balance[from] * share;
        balance[from] -= dx;
        balance[to] += dx;
        legs += 1;
    }
    legs
}

/// Replay real pools and their Python answers: exactness first, then speed.
fn pools_bench(path: &str, reps: usize) {
    use ruint::aliases::U256;
    let raw = std::fs::read_to_string(path).expect("read pools");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse pools");
    let specs = v.as_array().unwrap();

    let big = |x: &serde_json::Value| -> U256 {
        x.as_str().unwrap().parse::<U256>().unwrap()
    };
    let mut pools = Vec::new();
    for spec in specs {
        let pool = stableswap::Pool {
            balances: spec["balances"].as_array().unwrap().iter().map(big).collect(),
            rates: spec["rates"].as_array().unwrap().iter().map(big).collect(),
            amp: big(&spec["amp"]),
            fee: big(&spec["fee"]),
            offpeg_fee_multiplier: big(&spec["offpeg_fee_multiplier"]),
            a_precision: big(&spec["a_precision"]),
            fee_on_xp: spec["fee_on_xp"].as_bool().unwrap(),
            subtract_one: spec["subtract_one"].as_bool().unwrap(),
        };
        let vectors: Vec<(usize, usize, U256, U256)> = spec["vectors"].as_array()
            .unwrap().iter().map(|t| (
                t[0].as_u64().unwrap() as usize,
                t[1].as_u64().unwrap() as usize,
                big(&t[2]), big(&t[3]))).collect();
        pools.push((pool, vectors));
    }

    let mut n = 0usize;
    let mut wrong = 0usize;
    let mut none = 0usize;
    for (pool, vectors) in &pools {
        for (i, j, dx, want) in vectors {
            n += 1;
            match pool.get_dy(*i, *j, *dx) {
                Some(got) if got == *want => {}
                Some(_) => wrong += 1,
                None => none += 1,
            }
        }
    }
    println!("{} pools · {} vectors · {} wrong · {} would not converge",
             pools.len(), n, wrong, none);

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        let mut sink = U256::ZERO;
        for (pool, vectors) in &pools {
            for (i, j, dx, _) in vectors {
                if let Some(got) = pool.get_dy(*i, *j, *dx) {
                    sink += got;
                }
            }
        }
        std::hint::black_box(sink);
        best = best.min(start.elapsed().as_secs_f64() * 1e6);
    }
    println!("rust get_dy: {best:.0} us for {n} calls = {:.2} us each", best / n as f64);
    println!("python was 9.30 us each");
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: proto <quote.json>");
    if path.ends_with("pools.json") {
        let reps: usize = std::env::args().nth(2)
            .and_then(|s| s.parse().ok()).unwrap_or(20);
        pools_bench(&path, reps);
        return;
    }
    let reps: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(20);
    let raw = std::fs::read_to_string(&path).expect("read dump");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse dump");

    let g = &v["graph"];
    let tau = ints(g, "tau");
    let sig = ints(g, "sig");
    let gg = floats(g, "G");
    let eps = floats(g, "eps");
    let cap = floats(g, "cap");
    let n_nodes = g["n_nodes"].as_u64().unwrap() as usize;
    let arcs = Arcs { tau: &tau, sig: &sig, g: &gg, eps: &eps, cap: &cap, n_nodes };

    let calls = v["solve_calls"].as_array().unwrap();
    let cands = v["candidates"].as_array().unwrap();
    let amount_in = v["amount_in"].as_f64().unwrap();
    println!("{} arcs · {} nodes · {} solves · {} candidates · {} reps",
             tau.len(), n_nodes, calls.len(), cands.len(), reps);

    // --- the solves, replayed exactly as Python asked for them
    let mut pivots = 0u64;
    let mut best_solve = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        let mut p = 0u64;
        for c in calls {
            let a0 = mask(&c["a0"]);
            let forbidden = mask(&c["forbidden"]);
            let pinned: Vec<(usize, f64)> = c["pinned"].as_array().unwrap().iter()
                .map(|x| (x[0].as_u64().unwrap() as usize, x[1].as_f64().unwrap()))
                .collect();
            let opt = Options {
                min_flow: c["min_flow"].as_f64().unwrap(),
                gas_cost: c["gas_cost"].as_f64().unwrap(),
                maxit: c["maxit"].as_u64().unwrap() as u32,
                partial_ok: c["partial_ok"].as_bool().unwrap(),
                ..Options::default()
            };
            let s = active_set_solve(
                &arcs,
                c["src"].as_u64().unwrap() as usize,
                c["dst"].as_u64().unwrap() as usize,
                c["psi_total"].as_f64().unwrap(),
                a0.as_deref(), forbidden.as_deref(), &pinned, &opt);
            p += s.pivots as u64;
        }
        best_solve = best_solve.min(start.elapsed().as_secs_f64() * 1e3);
        pivots = p;
    }

    // --- realising every candidate
    let mut best_real = f64::INFINITY;
    let mut legs = 0usize;
    for _ in 0..reps {
        let start = Instant::now();
        let mut n = 0usize;
        for c in cands {
            let psi = floats(c, "psi");
            n += realize(&tau, &sig, &psi, n_nodes, amount_in);
        }
        best_real = best_real.min(start.elapsed().as_secs_f64() * 1e3);
        legs = n;
    }

    println!("\n  solves    {best_solve:8.2} ms   {pivots} pivots");
    println!("  realize   {best_real:8.2} ms   {legs} legs over {} candidates",
             cands.len());
    println!("  ------------------------------");
    println!("  total     {:8.2} ms", best_solve + best_real);
    println!("\n  against Python: solves 18.1 ms · realize ~18 ms · stage 40.7 ms");
}
