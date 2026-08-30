//! Shortest `src -> dst` path by arc length, tolerating negative arcs (§5.3).
//!
//! A port of `core/seed.py`'s `spfa`.  Bellman-Ford with a FIFO queue, not
//! Dijkstra: `eps_p` can be negative -- a favourably dislocated pool is an EMF
//! -- and Dijkstra needs non-negative weights.
//!
//! The Python is already careful (plain lists, a `deque`, no numpy in the
//! loop) and still costs 0.3 ms a call because the loop runs ~300k times a
//! route.  That is the shape a compiled port helps most: no clever
//! arithmetic, just an enormous number of very cheap steps.

/// Relaxation depth for the seed searches. Real Curve routes are 1-3 hops
/// (mainnet universe: mean 1.97, max 3), so this is generous -- but it turns
/// each search from "iterate to a fixed point over 301 nodes" into a bounded
/// sweep. Safe in a way bounding the solve would not be: §5.5's certificate
/// prices out *all* m arcs, so a path the seed misses is still found by column
/// generation.
pub const MAX_HOPS: usize = 8;

/// Where the walk back from `dst` ended up.
pub struct Path {
    pub arcs: Vec<usize>,
    pub length: f64,
    pub found: bool,
    /// Set instead of `arcs` when the parent chain re-enters a node: `dist`
    /// keeps falling around a loop rather than reaching `src`.
    pub negative_cycle: Vec<usize>,
}

impl Path {
    fn empty() -> Self {
        Path { arcs: Vec::new(), length: f64::INFINITY, found: false,
               negative_cycle: Vec::new() }
    }
}

/// CSR adjacency by tail node: `arcs[starts[v]..starts[v + 1]]` leave `v`.
pub struct Adjacency {
    pub starts: Vec<usize>,
    pub arcs: Vec<usize>,
}

pub fn build_adjacency(tau: &[i64], n_nodes: usize) -> Adjacency {
    let m = tau.len();
    let mut counts = vec![0usize; n_nodes + 1];
    for &t in tau {
        counts[t as usize + 1] += 1;
    }
    for v in 0..n_nodes {
        counts[v + 1] += counts[v];
    }
    let starts = counts.clone();
    let mut cursor = counts;
    let mut arcs = vec![0usize; m];
    for (p, &t) in tau.iter().enumerate() {
        let slot = cursor[t as usize];
        arcs[slot] = p;
        cursor[t as usize] += 1;
    }
    Adjacency { starts, arcs }
}

#[allow(clippy::too_many_arguments)]
pub fn spfa(
    tau: &[i64],
    sig: &[i64],
    cost: &[f64],
    n_nodes: usize,
    adj: &Adjacency,
    src: usize,
    dst: usize,
    banned_arcs: &[bool],
    banned_nodes: &[bool],
    max_hops: usize,
) -> Path {
    let n = n_nodes;
    let mut dist = vec![f64::INFINITY; n];
    let mut parent = vec![-1i64; n];
    let mut hops = vec![0usize; n];
    let mut in_queue = vec![false; n];

    // Depth-bounding already forces termination; this is a backstop against a
    // pathology in the bound itself, not the mechanism.
    let mut budget = (8 * max_hops * n) as i64;

    dist[src] = 0.0;
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(src);
    in_queue[src] = true;

    while let Some(node) = queue.pop_front() {
        if budget <= 0 {
            break;
        }
        in_queue[node] = false;
        let depth = hops[node] + 1;
        if depth > max_hops {
            continue;
        }
        let base = dist[node];
        for k in adj.starts[node]..adj.starts[node + 1] {
            let arc = adj.arcs[k];
            if banned_arcs[arc] {
                continue;
            }
            let head = sig[arc] as usize;
            if banned_nodes[head] {
                continue;
            }
            let candidate = base + cost[arc];
            if candidate < dist[head] - 1e-15 {
                dist[head] = candidate;
                parent[head] = arc as i64;
                hops[head] = depth;
                budget -= 1;
                if !in_queue[head] {
                    queue.push_back(head);
                    in_queue[head] = true;
                }
            }
        }
    }

    if !dist[dst].is_finite() {
        return Path::empty();
    }

    // Walk the parent pointers back.  A negative cycle shows up here and only
    // here, and detecting it on the walk costs nothing.
    let mut arcs: Vec<usize> = Vec::new();
    let mut seen_at = vec![usize::MAX; n];
    let mut node = dst;
    while node != src {
        if seen_at[node] != usize::MAX {
            let mut cycle: Vec<usize> = arcs[seen_at[node]..].to_vec();
            cycle.reverse();
            return Path { arcs: Vec::new(), length: f64::INFINITY, found: false,
                          negative_cycle: cycle };
        }
        seen_at[node] = arcs.len();
        let arc = parent[node];
        if arc < 0 {
            return Path::empty();
        }
        arcs.push(arc as usize);
        node = tau[arc as usize] as usize;
    }
    arcs.reverse();
    let length = dist[dst];
    Path { arcs, length, found: true, negative_cycle: Vec::new() }
}

/// Yen's algorithm over `eps`, returning arc-index paths.
///
/// Only ever returns genuine `src -> dst` paths. A negative cycle is not one:
/// conflating the two lets a caller warm-start the solver from a loop that
/// does not contain `src`, which then reports the graph as disconnected.
pub fn k_shortest_paths(
    tau: &[i64], sig: &[i64], eps: &[f64], n_nodes: usize,
    src: usize, dst: usize, k: usize, adj: &Adjacency,
) -> Vec<Vec<usize>> {
    let m = tau.len();
    let none = vec![false; m];
    let no_nodes = vec![false; n_nodes];
    let run = |cost: &[f64], from: usize, banned_arcs: &[bool], banned_nodes: &[bool]| {
        spfa(tau, sig, cost, n_nodes, adj, from, dst, banned_arcs, banned_nodes, MAX_HOPS)
    };

    let mut first = run(eps, src, &none, &no_nodes);
    // §5.3: "shift by min(0, min_p eps_p) and correct". Shifting changes a
    // path's cost per hop rather than preserving the order, so the result is
    // only approximately cheapest -- fine, since seed quality affects rounds.
    let shifted: Option<Vec<f64>> = if first.negative_cycle.is_empty() {
        None
    } else {
        let floor = eps.iter().copied().fold(0.0f64, f64::min);
        let weights: Vec<f64> = eps.iter().map(|v| v - floor).collect();
        first = run(&weights, src, &none, &no_nodes);
        Some(weights)
    };
    let cost: &[f64] = shifted.as_deref().unwrap_or(eps);
    if !first.found {
        return Vec::new();
    }

    let mut accepted: Vec<Vec<usize>> = vec![first.arcs.clone()];
    // Yen's B-list. A `BinaryHeap` would need the reverse of Python's ordering
    // and its tie-break is not specified; this is a handful of entries and the
    // pop is the same `(cost, path)` minimum `heapq` takes -- lexicographic on
    // the path where the costs tie, which decides which of two equal-length
    // routes reaches the ballot.
    let mut candidates: Vec<(f64, Vec<usize>)> = Vec::new();
    let mut seen: Vec<Vec<usize>> = vec![first.arcs];

    while accepted.len() < k {
        let previous = accepted[accepted.len() - 1].clone();
        for i in 0..previous.len() {
            let root = &previous[..i];
            let spur_node = tau[previous[i]] as usize;

            let mut banned_arcs = vec![false; m];
            for path in &accepted {
                if path.len() > i && &path[..i] == root {
                    banned_arcs[path[i]] = true;
                }
            }
            let mut banned_nodes = vec![false; n_nodes];
            for &arc in root {
                banned_nodes[tau[arc] as usize] = true;
            }

            let spur = run(cost, spur_node, &banned_arcs, &banned_nodes);
            if !spur.found {
                continue;
            }
            let mut whole = root.to_vec();
            whole.extend(spur.arcs);
            if seen.contains(&whole) {
                continue;
            }
            seen.push(whole.clone());
            // Scored on `eps`, not on the shifted weights: the shift is only
            // there to make the search terminate.
            let length: f64 = whole.iter().map(|&a| eps[a]).sum();
            candidates.push((length, whole));
        }

        if candidates.is_empty() {
            break;
        }
        let mut best = 0usize;
        for at in 1..candidates.len() {
            if less(&candidates[at], &candidates[best]) {
                best = at;
            }
        }
        accepted.push(candidates.remove(best).1);
    }

    accepted
}

/// `(cost, path) < (cost, path)` the way a tuple compares in Python: the
/// number first, then the list, element by element and then by length.
fn less(a: &(f64, Vec<usize>), b: &(f64, Vec<usize>)) -> bool {
    if a.0 != b.0 {
        return a.0 < b.0;
    }
    a.1 < b.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tau: &[i64], sig: &[i64], cost: &[f64], n: usize, src: usize, dst: usize) -> Path {
        let adj = build_adjacency(tau, n);
        spfa(tau, sig, cost, n, &adj, src, dst,
             &vec![false; tau.len()], &vec![false; n], 8)
    }

    #[test]
    fn it_takes_the_cheaper_of_two_lanes() {
        let got = run(&[0, 0], &[1, 1], &[2.0, 1.0], 2, 0, 1);
        assert!(got.found);
        assert_eq!(got.arcs, vec![1]);
        assert!((got.length - 1.0).abs() < 1e-15);
    }

    #[test]
    fn it_chains_hops() {
        let got = run(&[0, 1], &[1, 2], &[1.0, 1.0], 3, 0, 2);
        assert_eq!(got.arcs, vec![0, 1]);
    }

    #[test]
    fn a_negative_arc_is_taken_rather_than_diverging() {
        let got = run(&[0, 0, 1], &[2, 1, 2], &[0.0, -1.0, -1.0], 3, 0, 2);
        assert!(got.found, "dijkstra would have stopped at the zero-cost arc");
        assert_eq!(got.arcs, vec![1, 2]);
    }

    #[test]
    fn an_unreachable_target_is_not_found() {
        let got = run(&[0], &[1], &[1.0], 3, 0, 2);
        assert!(!got.found && got.negative_cycle.is_empty());
    }
}
