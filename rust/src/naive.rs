//! The model-free candidate floor: one-leg and two-leg chains.
//!
//! These two generators exist so a router is never beaten by a swap anyone
//! could find by inspection. They depend on no part of the model -- not the
//! probe grid, the calibration, the price fit or the solver -- so a dropped arc
//! cannot lose sight of an obvious direct route.
//!
//! The reference has `two_step_candidates` call the quoter twice from inside
//! one function. That cannot cross to wasm, where the chain is on the other
//! side of the boundary, so it is split at its two probe rounds into three
//! stateless stages: plan the first round, rank it and plan the second, build
//! the candidates. The caller does the quoting in between. Same arithmetic,
//! same order, no I/O in here.

use crate::candidates::Candidate;
use crate::nodes::NodeMap;
use crate::types::{ArcKind, PoolArc, Probe, TypeError};
use ruint::aliases::U256;

/// The slice of a `PoolSpec` these generators read.
///
/// Not the whole thing: the rest of `PoolSpec` is the fetch layer's, and
/// nothing here is worth dragging it across the boundary for.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolFacts {
    pub address: String,
    pub name: String,
    /// `None` where no swap dialect is resolved, which is the reference's
    /// `swap_kind is None` -- the pool is skipped, not defaulted.
    pub kind: Option<ArcKind>,
    /// The pool's own coins, lowercase, in index order.
    pub coins: Vec<String>,
    pub decimals: Vec<u32>,
    pub balances: Vec<u128>,
    pub tvl_usd: f64,
}

impl PoolFacts {
    pub fn n_coins(&self) -> i32 {
        self.coins.len() as i32
    }

    fn balance(&self, i: usize) -> u128 {
        self.balances.get(i).copied().unwrap_or(0)
    }

    fn decimals_at(&self, i: usize) -> u32 {
        self.decimals.get(i).copied().unwrap_or(18)
    }

    /// `{coin.address.lower(): k}` built the way the reference builds it: a
    /// repeated address keeps its **first** position and its **last** index,
    /// because that is what a dict comprehension does. Two pools in the
    /// universe list the same token twice, so this is load-bearing.
    fn coin_index(&self) -> Vec<(&str, usize)> {
        let mut out: Vec<(&str, usize)> = Vec::with_capacity(self.coins.len());
        for (k, coin) in self.coins.iter().enumerate() {
            match out.iter_mut().find(|(addr, _)| *addr == coin.as_str()) {
                Some(slot) => slot.1 = k,
                None => out.push((coin.as_str(), k)),
            }
        }
        out
    }
}

/// `nodes.node(token)`, which *raises* in the reference rather than returning
/// a sentinel. A token the map has never seen is a caller error at every call
/// site the floor has, so it is one here too -- returning an empty floor
/// instead would look like "no route" and be indistinguishable from one.
fn node_of(nodes: &NodeMap, token: &str) -> Result<usize, TypeError> {
    nodes
        .node(token)
        .ok_or_else(|| TypeError(format!("token is not in the node map: {token}")))
}

fn ratio(nu: &[f64], tau: usize, sigma: usize) -> f64 {
    match (nu.get(tau), nu.get(sigma)) {
        (Some(&t), Some(&s)) if s != 0.0 => t / s,
        _ => 1.0,
    }
}

/// One-leg candidates through every pool holding both tokens.
///
/// `amount_in` is not a parameter here and is not one in the reference either:
/// which pools hold the pair does not depend on the size.
pub fn direct_candidates(
    pools: &[PoolFacts],
    nodes: &NodeMap,
    nu: &[f64],
    src_token: &str,
    dst_token: &str,
) -> Result<(Vec<Candidate>, Vec<PoolArc>), TypeError> {
    let (src_node, dst_node) = (node_of(nodes, src_token)?, node_of(nodes, dst_token)?);
    let mut out = Vec::new();
    let mut made = Vec::new();
    for pool in pools {
        let Some(kind) = pool.kind else { continue };
        let index = pool.coin_index();
        for &(token_in, i) in &index {
            if nodes.node(token_in) != Some(src_node) {
                continue;
            }
            for &(token_out, j) in &index {
                if i == j || nodes.node(token_out) != Some(dst_node) {
                    continue;
                }
                let mut arc = PoolArc::new(
                    format!("direct:{}:{i}>{j}", pool.address.to_lowercase()),
                    pool.address.clone(),
                    kind,
                    i as i32,
                    j as i32,
                    pool.n_coins(),
                    token_in.to_string(),
                    token_out.to_string(),
                    src_node,
                    dst_node,
                );
                arc.a = ratio(nu, src_node, dst_node);
                arc.b = 0.0;
                arc.reserve_in = pool.balance(i);
                arc.decimals_in = pool.decimals_at(i);
                arc.decimals_out = pool.decimals_at(j);
                arc.tvl_usd = pool.tvl_usd;
                arc.note = pool.name.clone();
                made.push(arc);
                out.push(Candidate::naive(
                    format!("direct {}", truncate(&pool.name, 22)),
                    vec![1.0],
                    "DIRECT",
                    1,
                ));
            }
        }
    }
    Ok((out, made))
}

/// `pool.name[:22]`, which is a slice of *characters* in Python.
fn truncate(name: &str, n: usize) -> String {
    name.chars().take(n).collect()
}

/// An arc built for realisation only -- never calibrated, never solved.
fn synthetic_arc(
    pool: &PoolFacts,
    i: usize,
    j: usize,
    nu: &[f64],
    tau: usize,
    sigma: usize,
) -> PoolArc {
    let mut arc = PoolArc::new(
        format!("naive:{}:{i}>{j}", pool.address.to_lowercase()),
        pool.address.clone(),
        pool.kind.unwrap_or(ArcKind::SwapStable),
        i as i32,
        j as i32,
        pool.n_coins(),
        pool.coins[i].clone(),
        pool.coins[j].clone(),
        tau,
        sigma,
    );
    arc.a = ratio(nu, tau, sigma);
    arc.b = 0.0;
    arc.reserve_in = pool.balance(i);
    arc.decimals_in = pool.decimals_at(i);
    arc.decimals_out = pool.decimals_at(j);
    arc.tvl_usd = pool.tvl_usd;
    arc.note = pool.name.clone();
    arc
}

/// Which tradeable pools hold which node, at which coin slots.
///
/// Resolved once. Round B otherwise walks the whole universe for each
/// intermediate token it considers -- 11,610 pool scans for a question that
/// does not depend on the token being asked about.
struct Index {
    /// Indices into the caller's `pools`, tradeable ones only.
    tradeable: Vec<usize>,
    /// Per tradeable pool, `node -> coin slots`, in first-seen order.
    slots_of: Vec<Vec<(usize, Vec<usize>)>>,
    /// `node -> tradeable pool positions`, in first-seen order.
    holders: Vec<(usize, Vec<usize>)>,
}

fn slots<'a>(per_node: &'a [(usize, Vec<usize>)], node: usize) -> &'a [usize] {
    per_node
        .iter()
        .find(|(id, _)| *id == node)
        .map_or(&[][..], |(_, v)| v.as_slice())
}

impl Index {
    fn build(pools: &[PoolFacts], nodes: &NodeMap) -> Self {
        let mut tradeable = Vec::new();
        let mut slots_of: Vec<Vec<(usize, Vec<usize>)>> = Vec::new();
        let mut holders: Vec<(usize, Vec<usize>)> = Vec::new();
        for (position, pool) in pools.iter().enumerate() {
            if pool.kind.is_none() {
                continue;
            }
            let slot = tradeable.len();
            tradeable.push(position);
            let mut per_node: Vec<(usize, Vec<usize>)> = Vec::new();
            for (k, coin) in pool.coins.iter().enumerate() {
                let Some(node) = nodes.node(coin) else { continue };
                match per_node.iter_mut().find(|(id, _)| *id == node) {
                    Some(entry) => entry.1.push(k),
                    None => per_node.push((node, vec![k])),
                }
            }
            for (node, _) in &per_node {
                match holders.iter_mut().find(|(id, _)| id == node) {
                    Some(entry) => entry.1.push(slot),
                    None => holders.push((*node, vec![slot])),
                }
            }
            slots_of.push(per_node);
        }
        Self { tradeable, slots_of, holders }
    }

    fn held_by(&self, node: usize) -> &[usize] {
        slots(&self.holders, node)
    }
}

/// One `src -> M` hop the first round is asking about.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstHop {
    /// Index into the caller's `pools`.
    pub pool: usize,
    pub i: usize,
    pub j: usize,
    pub middle: usize,
}

/// One `M -> dst` hop the second round is asking about.
#[derive(Debug, Clone, PartialEq)]
pub struct SecondHop {
    pub pool: usize,
    pub i: usize,
    pub j: usize,
    pub middle: usize,
}

/// The best first hop into a middle, carried between rounds.
#[derive(Debug, Clone, PartialEq)]
pub struct BestFirst {
    pub middle: usize,
    pub canonical: u128,
    pub pool: usize,
    pub i: usize,
    pub j: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanA {
    pub probes: Vec<Probe>,
    pub hops: Vec<FirstHop>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanB {
    pub probes: Vec<Probe>,
    pub hops: Vec<SecondHop>,
    /// Every middle's winning first hop, not just the ranked ones: stage three
    /// looks the first leg up here by middle.
    pub best_first: Vec<BestFirst>,
}

/// Round A: `src -> M`, for every middle that can reach `dst` at all.
///
/// The reachability filter runs *before* the ranking, not after. Round A used
/// to quote every token one hop from the source and keep the best `3 * limit`
/// by output. Two things went wrong together. The output is
/// `to_canonical_wei`, a count of tokens rather than a value, so a token
/// trading at a fraction of a cent sorts above every stable simply by being
/// numerous -- on USDC -> sUSDe at $1,000 the top of the list was CXD at
/// 2,513,355 units. And the cut was taken without asking where any of them
/// could go next: of 52 middles quoted, three could reach sUSDe and all three
/// were cut, so the two-hop floor came back empty for a pair with an obvious
/// two-hop route.
pub fn two_step_plan_first(
    pools: &[PoolFacts],
    nodes: &NodeMap,
    src_token: &str,
    dst_token: &str,
    amount_in: u128,
) -> Result<PlanA, TypeError> {
    let empty = PlanA { probes: Vec::new(), hops: Vec::new() };
    let (src_node, dst_node) = (node_of(nodes, src_token)?, node_of(nodes, dst_token)?);
    let index = Index::build(pools, nodes);

    let reaching: Vec<usize> = index
        .holders
        .iter()
        .filter(|(node, held)| {
            *node != src_node
                && *node != dst_node
                && held.iter().any(|&slot| !slots(&index.slots_of[slot], dst_node).is_empty())
        })
        .map(|(node, _)| *node)
        .collect();
    if reaching.is_empty() {
        return Ok(empty);
    }

    let mut probes = Vec::new();
    let mut hops = Vec::new();
    for &slot in index.held_by(src_node) {
        let position = index.tradeable[slot];
        let pool = &pools[position];
        for &i in slots(&index.slots_of[slot], src_node) {
            for (j, coin) in pool.coins.iter().enumerate() {
                if j == i {
                    continue;
                }
                let Some(middle) = nodes.node(coin) else { continue };
                if !reaching.contains(&middle) {
                    continue;
                }
                probes.push(Probe::new(
                    pool.address.clone(),
                    pool.kind.expect("tradeable"),
                    i as i32,
                    j as i32,
                    pool.n_coins(),
                    amount_in,
                )?);
                hops.push(FirstHop { pool: position, i, j, middle });
            }
        }
    }
    if probes.is_empty() {
        return Ok(empty);
    }
    Ok(PlanA { probes, hops })
}

/// What a hop into this middle is worth, rather than how many tokens it made.
///
/// Comparing `to_canonical_wei` across middles compares token counts, and a
/// token trading at a fraction of a cent wins every such comparison by
/// arithmetic alone -- CXD at 2,513,355 units above crvUSD at 1,000, on pools
/// of $30k and $400M. `nu` is the reference price fit and exists precisely to
/// make quantities in different tokens comparable.
fn worth(nodes: &NodeMap, nu: &[f64], best: &BestFirst) -> f64 {
    let canonical = nodes.canonical_of.get(best.middle);
    let decimals = canonical.map_or(18, |token| nodes.decimals(token));
    let units = best.canonical as f64 / 10f64.powi(decimals as i32);
    match nu.get(best.middle) {
        Some(&price) => units * price,
        None => units,
    }
}

/// Rank round A's answers and plan round B: `M -> dst`.
///
/// `quotes[k]` is the output of `plan.probes[k]`, `None` where the chain
/// refused. The ranking only has to be roughly right -- the quoter decides the
/// amounts -- and the reachability filter above means the cut rarely binds.
pub fn two_step_rank(
    pools: &[PoolFacts],
    nodes: &NodeMap,
    nu: &[f64],
    plan: &PlanA,
    quotes: &[Option<u128>],
    dst_token: &str,
    limit: usize,
) -> Result<PlanB, TypeError> {
    let empty = PlanB { probes: Vec::new(), hops: Vec::new(), best_first: Vec::new() };
    let dst_node = node_of(nodes, dst_token)?;
    let index = Index::build(pools, nodes);

    // Insertion-ordered, because the reference iterates a dict: the first
    // middle to reach a given value keeps it, and ties never displace.
    let mut best_first: Vec<BestFirst> = Vec::new();
    for (hop, quote) in plan.hops.iter().zip(quotes.iter()) {
        let Some(value) = *quote else { continue };
        if value == 0 {
            continue;
        }
        let pool = &pools[hop.pool];
        let Some(canonical) = nodes.to_canonical_wei(&pool.coins[hop.j], U256::from(value))
        else {
            continue;
        };
        let canonical: u128 = canonical.try_into().unwrap_or(u128::MAX);
        match best_first.iter_mut().find(|b| b.middle == hop.middle) {
            Some(best) if canonical > best.canonical => {
                *best = BestFirst {
                    middle: hop.middle,
                    canonical,
                    pool: hop.pool,
                    i: hop.i,
                    j: hop.j,
                };
            }
            Some(_) => {}
            None => best_first.push(BestFirst {
                middle: hop.middle,
                canonical,
                pool: hop.pool,
                i: hop.i,
                j: hop.j,
            }),
        }
    }
    if best_first.is_empty() {
        return Ok(empty);
    }

    let mut ranked: Vec<&BestFirst> = best_first.iter().collect();
    // Stable, so equal worth keeps insertion order -- `sorted` in the
    // reference is stable too, and on a key of `-worth`.
    ranked.sort_by(|a, b| {
        let (x, y) = (-worth(nodes, nu, a), -worth(nodes, nu, b));
        x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(3 * limit);

    let mut probes = Vec::new();
    let mut hops = Vec::new();
    for best in ranked {
        for &slot in index.held_by(best.middle) {
            let position = index.tradeable[slot];
            let pool = &pools[position];
            for &i in slots(&index.slots_of[slot], best.middle) {
                let start = nodes
                    .from_canonical_wei(&pool.coins[i], U256::from(best.canonical))
                    .and_then(|v| u128::try_from(v).ok())
                    .unwrap_or(0);
                if start == 0 {
                    continue;
                }
                for &j in slots(&index.slots_of[slot], dst_node) {
                    if i == j {
                        continue;
                    }
                    probes.push(Probe::new(
                        pool.address.clone(),
                        pool.kind.expect("tradeable"),
                        i as i32,
                        j as i32,
                        pool.n_coins(),
                        start,
                    )?);
                    hops.push(SecondHop { pool: position, i, j, middle: best.middle });
                }
            }
        }
    }
    if probes.is_empty() {
        return Ok(empty);
    }
    Ok(PlanB { probes, hops, best_first })
}

/// Build the chains from round B's answers, best `limit` by final output.
pub fn two_step_build(
    pools: &[PoolFacts],
    nodes: &NodeMap,
    nu: &[f64],
    plan: &PlanB,
    quotes: &[Option<u128>],
    src_token: &str,
    dst_token: &str,
    limit: usize,
) -> Result<(Vec<Candidate>, Vec<Vec<PoolArc>>), TypeError> {
    let (src_node, dst_node) = (node_of(nodes, src_token)?, node_of(nodes, dst_token)?);

    let mut best_chain: Vec<(usize, u128, usize, usize, usize)> = Vec::new();
    for (hop, quote) in plan.hops.iter().zip(quotes.iter()) {
        let Some(value) = *quote else { continue };
        if value == 0 {
            continue;
        }
        let pool = &pools[hop.pool];
        let Some(out) = nodes.to_canonical_wei(&pool.coins[hop.j], U256::from(value)) else {
            continue;
        };
        let out: u128 = out.try_into().unwrap_or(u128::MAX);
        match best_chain.iter_mut().find(|(middle, ..)| *middle == hop.middle) {
            Some(entry) if out > entry.1 => {
                *entry = (hop.middle, out, hop.pool, hop.i, hop.j);
            }
            Some(_) => {}
            None => best_chain.push((hop.middle, out, hop.pool, hop.i, hop.j)),
        }
    }

    let mut ranked = best_chain;
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.truncate(limit);

    let mut out = Vec::new();
    let mut made = Vec::new();
    for (middle, _value, position2, i2, j2) in ranked {
        let Some(first) = plan.best_first.iter().find(|b| b.middle == middle) else {
            continue;
        };
        let pool1 = &pools[first.pool];
        let pool2 = &pools[position2];
        made.push(vec![
            synthetic_arc(pool1, first.i, first.j, nu, src_node, middle),
            synthetic_arc(pool2, i2, j2, nu, middle, dst_node),
        ]);
        out.push(Candidate::naive(
            format!("2-hop via {}", nodes.symbol(&pool1.coins[first.j])),
            vec![1.0, 1.0],
            "TWO_STEP",
            2,
        ));
    }
    Ok((out, made))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes_with(tokens: &[(&str, &str, u32)]) -> NodeMap {
        let mut nodes = NodeMap::new();
        for (address, symbol, decimals) in tokens {
            nodes.add_token(address, symbol, *decimals);
        }
        nodes
    }

    fn pool(address: &str, coins: &[&str], decimals: &[u32]) -> PoolFacts {
        PoolFacts {
            address: address.to_string(),
            name: format!("pool {address}"),
            kind: Some(ArcKind::SwapStable),
            coins: coins.iter().map(|c| c.to_string()).collect(),
            decimals: decimals.to_vec(),
            balances: vec![1_000_000; coins.len()],
            tvl_usd: 1.0,
        }
    }

    #[test]
    fn a_pool_holding_both_tokens_gives_one_direct_arc_per_direction() {
        let nodes = nodes_with(&[("0xa", "A", 18), ("0xb", "B", 6)]);
        let pools = [pool("0xp", &["0xa", "0xb"], &[18, 6])];
        let (cands, arcs) =
            direct_candidates(&pools, &nodes, &[1.0, 1.0], "0xa", "0xb").unwrap();
        assert_eq!(arcs.len(), 1);
        assert_eq!(cands.len(), 1);
        assert_eq!(arcs[0].id, "direct:0xp:0>1");
        assert_eq!(arcs[0].decimals_in, 18);
        assert_eq!(arcs[0].decimals_out, 6);
        assert_eq!(cands[0].reason, "DIRECT");
    }

    #[test]
    fn a_pool_without_a_dialect_is_skipped_not_defaulted() {
        let nodes = nodes_with(&[("0xa", "A", 18), ("0xb", "B", 18)]);
        let mut pools = [pool("0xp", &["0xa", "0xb"], &[18, 18])];
        pools[0].kind = None;
        let (cands, arcs) =
            direct_candidates(&pools, &nodes, &[1.0, 1.0], "0xa", "0xb").unwrap();
        assert!(cands.is_empty() && arcs.is_empty());
    }

    #[test]
    fn a_repeated_coin_keeps_its_first_position_and_last_index() {
        // What `{c.address.lower(): k for k, c in enumerate(coins)}` does, and
        // the reason this is a Vec of pairs rather than a map.
        let p = pool("0xp", &["0xa", "0xb", "0xa"], &[18, 18, 18]);
        assert_eq!(p.coin_index(), vec![("0xa", 2), ("0xb", 1)]);
    }

    #[test]
    fn a_middle_that_cannot_reach_dst_is_never_probed() {
        let nodes = nodes_with(&[
            ("0xa", "A", 18), ("0xb", "B", 18), ("0xm", "M", 18), ("0xz", "Z", 18),
        ]);
        let pools = [
            pool("0xp1", &["0xa", "0xm"], &[18, 18]),  // src -> M
            pool("0xp2", &["0xm", "0xb"], &[18, 18]),  // M -> dst
            pool("0xp3", &["0xa", "0xz"], &[18, 18]),  // src -> Z, a dead end
        ];
        let plan = two_step_plan_first(&pools, &nodes, "0xa", "0xb", 1_000).unwrap();
        assert_eq!(plan.hops.len(), 1);
        assert_eq!(plan.hops[0].middle, nodes.node("0xm").unwrap());
    }

    #[test]
    fn with_no_middle_reaching_dst_the_floor_is_empty_rather_than_wrong() {
        let nodes = nodes_with(&[("0xa", "A", 18), ("0xb", "B", 18), ("0xz", "Z", 18)]);
        let pools = [pool("0xp", &["0xa", "0xz"], &[18, 18])];
        let plan = two_step_plan_first(&pools, &nodes, "0xa", "0xb", 1_000).unwrap();
        assert!(plan.probes.is_empty());
    }

    #[test]
    fn ranking_is_by_value_not_by_token_count() {
        // The CXD lesson: 2.5e6 units of a cheap token must not outrank 1e3 of
        // a dear one.  Same decimals, so only `nu` separates them.
        let nodes = nodes_with(&[
            ("0xa", "A", 18), ("0xb", "B", 18),
            ("0xcheap", "CHEAP", 18), ("0xdear", "DEAR", 18),
        ]);
        let cheap = nodes.node("0xcheap").unwrap();
        let dear = nodes.node("0xdear").unwrap();
        let mut nu = vec![1.0; 4];
        nu[cheap] = 1e-6;
        nu[dear] = 1.0;
        let a = BestFirst { middle: cheap, canonical: 2_513_355, pool: 0, i: 0, j: 1 };
        let b = BestFirst { middle: dear, canonical: 1_000, pool: 0, i: 0, j: 1 };
        assert!(worth(&nodes, &nu, &b) > worth(&nodes, &nu, &a));
    }
}
