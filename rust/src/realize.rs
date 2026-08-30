//! Turn a solved value flow into an ordered list of executable legs (§5.6).
//!
//! The mirror of `core/realize.py`.
//!
//! Two things make this more than bookkeeping.
//!
//! **Do not undo the netting.** The solution is an *edge* flow, so a pool that
//! two decomposed paths both traverse already carries the net amount.
//! Re-expanding into paths and quoting each separately double-counts that
//! pool's impact, which is a common source of failed quotes. So legs are
//! emitted per arc, in topological order, and paths are reconstructed
//! afterwards for display only.
//!
//! **Merged nodes need conversion legs.** Routing treats ETH and WETH as one
//! node, but a pool holds one or the other. Each node uses its canonical token
//! as a hub: arriving non-canonical tokens convert in, departing ones convert
//! out. On a node whose arcs all use the canonical token -- the common case --
//! no conversion is emitted at all.
//!
//! Every ordered map here is a `Vec` of pairs rather than a `HashMap`: the
//! reference iterates its dicts and the order decides which leg sweeps, so
//! insertion order is part of the answer rather than an implementation detail.

use crate::multiport::{element_from, MultiPortError};
use crate::nodes::{Conversion, NodeMap};
use crate::pools::{divided, scaled};
use crate::types::{ArcKind, Leg, PoolArc, TypeError};
use ruint::aliases::U256;
use std::fmt;

pub const BPS: i64 = 10_000;

/// A branch carrying less than this share of what leaves its node cannot
/// change the answer, but it can still destroy it: measured on rETH->WETH, a
/// branch holding 7e-6 of one node fed six more legs, each carrying an amount
/// that rounded to zero, and one of them quoting zero aborted the whole route.
///
/// `MIN_FLOW_FRACTION` in candidates does not catch this: it screens an arc's
/// flow against the *whole trade*, while a branch can be a meaningful share of
/// Psi and still be a rounding error at the node it leaves from.
pub const DUST_SHARE: f64 = 1e-4;

/// Paths are display-only, and there can be exponentially many of them: the
/// walk below is a DAG enumeration, so a route 16 legs deep branching three
/// ways at each node has tens of millions. §11.6 warns that path enumeration
/// is exponential, and it is just as true in the presentation layer as in the
/// solver.
pub const MAX_DISPLAY_PATHS: usize = 64;

/// The leg kinds whose effect on a pool the models can reproduce. A swap moves
/// two balances by amounts `exchange` computes exactly, and a deposit is
/// `add_liquidity` -- which charges the imbalance fee `calc_token_amount`
/// explicitly does not, and keeps all but the DAO's share.
///
/// A *withdrawal* is still not here: `remove_liquidity_one_coin`'s effect on
/// the supply has not been read off the deployed source, and guessing it is
/// what this list exists to prevent.
pub const ADVANCEABLE: [ArcKind; 4] = [
    ArcKind::SwapStable,
    ArcKind::DepositFixed,
    ArcKind::DepositDyn,
    ArcKind::DepositFixedNoflag,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationError(pub String);

impl fmt::Display for RealizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<TypeError> for RealizationError {
    fn from(e: TypeError) -> Self {
        RealizationError(e.0)
    }
}

type Result<T> = std::result::Result<T, RealizationError>;

#[derive(Debug, Clone, PartialEq)]
pub struct RealizedLeg {
    pub leg: Leg,
    pub kind: ArcKind,
    pub target: String,
    pub token_in: String,
    pub token_out: String,
    /// raw wei, modelled
    pub amount_in: U256,
    /// raw wei, modelled
    pub amount_out: U256,
    pub share_of_node: f64,
    pub arc_id: Option<String>,
    pub pool_name: String,
    pub eps: f64,
    pub impact_frac: f64,
    pub theta: f64,
    pub psi: f64,
    /// The arc's own input reserve, kept so `forward_simulate` can refresh
    /// `theta` after it rescales the amounts. Without it `theta` describes the
    /// flow the arc was realised at rather than the one being quoted.
    pub reserve_in: U256,
    /// The most this leg's arc will take, in `token_in` wei. The solve honours
    /// it; everything that re-weights legs afterwards works from quotes, and a
    /// view does not have to refuse what the pool will.
    pub cap_in: f64,
    /// The pool's TVL, for `route_conductance`. Deliberately the pool's own
    /// size rather than anything fitted.
    pub tvl_usd: f64,
    /// The size `verified_out` was measured at. Zero where nothing priced it.
    pub verified_in: U256,
    /// What this leg's own pool says it pays at this size, at the pinned
    /// block. `amount_out` is the quadratic's answer and is a *choice*.
    pub verified_out: U256,
    /// The least this leg's pool can charge. NaN where no model could say.
    pub fee_floor: f64,
    /// What this leg's own size pays in fees, read back out of the pool's
    /// exact model. NaN where no model could price the leg.
    pub fee_frac: f64,
    /// The pool's measured retention, `sqrt(a_forward * a_reverse)`. NaN where
    /// the opposite direction was not calibrated.
    pub gamma_live: f64,
    /// False when the arc behind this leg carries no calibration -- the
    /// model-free `direct`/`two-step` candidates, built at `psi = 1` with
    /// `B = 0`. Their `eps` and `impact_frac` are placeholders.
    pub modelled: bool,
}

impl RealizedLeg {
    fn blank(leg: Leg, kind: ArcKind, target: String) -> Self {
        Self {
            leg,
            kind,
            target,
            token_in: String::new(),
            token_out: String::new(),
            amount_in: U256::ZERO,
            amount_out: U256::ZERO,
            share_of_node: 1.0,
            arc_id: None,
            pool_name: String::new(),
            eps: 0.0,
            impact_frac: 0.0,
            theta: 0.0,
            psi: 0.0,
            reserve_in: U256::ZERO,
            cap_in: f64::INFINITY,
            tvl_usd: 0.0,
            verified_in: U256::ZERO,
            verified_out: U256::ZERO,
            fee_floor: f64::NAN,
            fee_frac: f64::NAN,
            gamma_live: f64::NAN,
            modelled: true,
        }
    }

    /// A leg the *merge* emitted, which is the one that is free.
    ///
    /// `is_conversion` is a kind, and a mint arc into an unmerged vault shares
    /// it: same `ERC4626_DEPOSIT`, but a rate, a cap and a gas cost. Only a
    /// merge comes from a `Conversion`, and only those legs have no arc.
    pub fn is_merge(&self) -> bool {
        self.is_conversion() && self.arc_id.is_none()
    }

    /// A leg the node merge emitted, not a pool the solver chose.
    ///
    /// Exactly the six kinds `Conversion::forward_kind`/`reverse_kind`
    /// produce. With the wstETH pair missing, its legs were rescaled as swaps
    /// from an `amount_out` no calibration set -- zero -- and a route through
    /// wstETH lost that branch: 42.18 realised, 33.98 modelled.
    pub fn is_conversion(&self) -> bool {
        matches!(
            self.kind,
            ArcKind::WrapNative
                | ArcKind::UnwrapNative
                | ArcKind::WstethWrap
                | ArcKind::WstethUnwrap
                | ArcKind::Erc4626Deposit
                | ArcKind::Erc4626Redeem
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealizedRoute {
    pub legs: Vec<RealizedLeg>,
    /// token -> slot, in the order the slots were opened.
    pub slots: Vec<(String, usize)>,
    pub dst_slot: usize,
    pub src_token: String,
    pub dst_token: String,
    pub amount_in: U256,
    pub modelled_out: U256,
    pub node_of_slot: Vec<(usize, usize)>,
    pub potentials: Vec<(usize, f64)>,
    pub paths: Vec<Vec<String>>,
    pub warnings: Vec<String>,
}

impl RealizedRoute {
    pub fn wire_legs(&self) -> Vec<Leg> {
        self.legs.iter().map(|rl| rl.leg.clone()).collect()
    }

    pub fn pools_used(&self) -> Vec<String> {
        let mut seen: Vec<String> = self
            .legs
            .iter()
            .filter(|rl| !rl.is_conversion())
            .map(|rl| rl.target.to_ascii_lowercase())
            .collect();
        seen.sort();
        seen.dedup();
        seen
    }

    /// The first leg carrying more than its arc will take, if any.
    ///
    /// Read rather than stored, so it describes the amounts as they stand
    /// after whatever last re-weighted them.
    pub fn over_capacity(&self) -> Option<&RealizedLeg> {
        self.legs.iter().find(|rl| over(rl.amount_in, rl.cap_in))
    }

    fn slot_of(&self, token: &str) -> Option<usize> {
        self.slots.iter().find(|(k, _)| k == token).map(|(_, v)| *v)
    }
}

/// `amount > cap`, exactly, the way Python compares an `int` with a `float`.
///
/// `n > c` is `n > floor(c)` whether or not `c` is a whole number, so the
/// comparison never has to round `n` down to an `f64` first -- which on a
/// wei-scale amount is the difference between honouring a cap and breaching
/// it by a few units.
fn over(amount: U256, cap: f64) -> bool {
    if cap.is_nan() {
        return false;
    }
    if cap == f64::INFINITY {
        return false;
    }
    if cap < 0.0 {
        return true;
    }
    match crate::pools::stableswap::to_u256(cap.floor()) {
        Some(floor) => amount > floor,
        // Past 2^256 there is no amount that could exceed it.
        None => false,
    }
}

// ------------------------------------------------------------- topology

/// Kahn's algorithm over the active arcs. Refuses a cycle.
pub fn topological_nodes(tau: &[i64], sig: &[i64], n_nodes: usize) -> Result<Vec<usize>> {
    let mut indegree = vec![0i64; n_nodes];
    for &s in sig {
        indegree[s as usize] += 1;
    }
    let mut queue: std::collections::VecDeque<usize> = (0..n_nodes)
        .filter(|&k| indegree[k] == 0)
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(n_nodes);
    let mut remaining = indegree;
    while let Some(node) = queue.pop_front() {
        order.push(node);
        // Arcs in index order, which is what `np.flatnonzero` gives.
        for p in 0..tau.len() {
            if tau[p] as usize != node {
                continue;
            }
            let head = sig[p] as usize;
            remaining[head] -= 1;
            if remaining[head] == 0 {
                queue.push_back(head);
            }
        }
    }
    if order.len() != n_nodes {
        return Err(RealizationError(
            "the active arcs contain a cycle; flow cannot be ordered for execution".into(),
        ));
    }
    Ok(order)
}

/// Live arcs reachable from `src` and co-reachable to `dst`.
fn on_a_path(
    tau: &[i64],
    sig: &[i64],
    live: &[bool],
    n: usize,
    src: usize,
    dst: usize,
) -> Vec<bool> {
    let idx: Vec<usize> = (0..live.len()).filter(|&k| live[k]).collect();
    let sweep = |seed: usize, frm: &[i64], to: &[i64]| -> Vec<bool> {
        let mut seen = vec![false; n];
        seen[seed] = true;
        for _ in 0..n {
            let next: Vec<usize> = idx
                .iter()
                .filter(|&&k| seen[frm[k] as usize])
                .map(|&k| to[k] as usize)
                .collect();
            if next.is_empty() || next.iter().all(|&v| seen[v]) {
                break;
            }
            for v in next {
                seen[v] = true;
            }
        }
        seen
    };
    let forward = sweep(src, tau, sig);
    let backward = sweep(dst, sig, tau);
    (0..live.len())
        .map(|k| live[k] && forward[tau[k] as usize] && backward[sig[k] as usize])
        .collect()
}

/// Drop branches too small to matter, and whatever they were feeding.
///
/// Two rules, applied together until the flow stops changing:
///
/// * a branch carrying less than `share` of its node's outflow is dust;
/// * an arc no longer on any src->dst path goes with it, which is what removes
///   the orphaned tail rather than leaving legs quoting on nothing.
///
/// The dropped value is not lost. The quoter splits a node by share of the
/// balance actually sitting in its slot and the last leg of a group sweeps the
/// remainder, so removing a branch hands its flow to its siblings -- and at
/// the optimum those siblings are priced within a hair of it (§6.1).
///
/// This *does* leave KCL violated by up to `share` at the pruned node, which
/// is deliberate and belongs strictly after §12.4's check: the invariant is
/// about what the solver produced, and this is about what the quoter can
/// survive. Returns the pruned flow and the number of arcs cut.
pub fn prune_dust(
    tau: &[i64],
    sig: &[i64],
    psi: &[f64],
    src: usize,
    dst: usize,
    share: f64,
    tol: f64,
) -> (Vec<f64>, usize) {
    let mut flow = psi.to_vec();
    if flow.is_empty() {
        return (flow, 0);
    }
    let hi = tau
        .iter()
        .chain(sig.iter())
        .map(|&v| v as usize)
        .chain([src, dst])
        .max()
        .unwrap_or(0);
    let n = hi + 1;
    let mut removed = 0usize;
    for _ in 0..flow.len() + 1 {
        let live: Vec<bool> = flow.iter().map(|&v| v > tol).collect();
        if !live.iter().any(|&v| v) {
            break;
        }
        let mut out_total = vec![0.0f64; n];
        for k in 0..flow.len() {
            if live[k] {
                out_total[tau[k] as usize] += flow[k];
            }
        }
        let reachable = on_a_path(tau, sig, &live, n, src, dst);
        let doomed: Vec<bool> = (0..flow.len())
            .map(|k| {
                live[k] && (flow[k] < share * out_total[tau[k] as usize] || !reachable[k])
            })
            .collect();
        if !doomed.iter().any(|&v| v) {
            break;
        }
        for k in 0..flow.len() {
            if doomed[k] {
                flow[k] = 0.0;
                removed += 1;
            }
        }
    }
    // Pruning may not decide there is no route. If the trade no longer reaches
    // the destination, the criterion was wrong for this flow and the original
    // stands -- a reverting candidate is adjudicated by the quoter, an empty
    // one never gets that far.
    let arrives = (0..flow.len()).any(|k| sig[k] as usize == dst && flow[k] > tol);
    if !arrives {
        return (psi.to_vec(), 0);
    }
    (flow, removed)
}

// -------------------------------------------------------------- realise

/// `round(x)` as CPython does it: ties to even, not away from zero.
///
/// `bps` is set from `round(BPS * share / total)`, and a split that lands
/// exactly on a half is not rare -- two equal branches out of one node put it
/// there every time.
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// `int(x)` on a float: truncate toward zero, and refuse what Python refuses.
fn to_int(x: f64) -> Result<U256> {
    if x.is_nan() {
        return Err(RealizationError("cannot convert float NaN to integer".into()));
    }
    if x.is_infinite() {
        return Err(RealizationError("cannot convert float infinity to integer".into()));
    }
    let truncated = x.trunc();
    if truncated <= 0.0 {
        return Ok(U256::ZERO);
    }
    crate::pools::stableswap::to_u256(truncated)
        .ok_or_else(|| RealizationError("amount does not fit in 256 bits".into()))
}

/// `10 ** d` as a float, the way `float(10**d)` reads it: one rounding of the
/// exact integer, which is what the reference multiplies by.
fn pow10(d: u32) -> f64 {
    scaled(U256::from(10u64).pow(U256::from(d)), 0)
}

/// The accumulator this token's balance lands in.
///
/// Aliases share one. Two addresses over a single balance -- gnosis's two EURe
/// contracts report the same `totalSupply` to the wei and the same `balanceOf`
/// for every holder -- have no conversion leg between them, because there is
/// nothing to execute. Giving them a slot each meant the legs delivered into
/// one and the route read the other. Only aliases collapse; a vault or a
/// native wrapper still needs its own slot, because a leg converts between
/// them.
fn slot(route: &mut RealizedRoute, nodes: &NodeMap, token: &str) -> Result<usize> {
    let mut key = token.to_ascii_lowercase();
    if let Some(conversion) = nodes.conversion.get(&key) {
        if conversion.is_alias() {
            key = conversion.canonical.to_ascii_lowercase();
        }
    }
    if let Some(found) = route.slot_of(&key) {
        return Ok(found);
    }
    let next = route.slots.len();
    route.slots.push((key.clone(), next));
    let node = nodes
        .node(&key)
        .ok_or_else(|| RealizationError(format!("{key} is not in the graph")))?;
    route.node_of_slot.push((next, node));
    Ok(next)
}

/// What each item of an emission group carries.
#[derive(Clone)]
enum Item {
    /// One arc, drawing straight from the hub.
    Arc(usize),
    /// One fill into a spoke token, standing for every arc that draws on it.
    Spoke(String),
}

/// Build the executable leg list from a solved flow.
///
/// `arcs` and `psi` are parallel and already restricted to the arcs carrying
/// flow. `psi` is value; `delta = psi / nu[tau]` converts back to canonical
/// token units, and the node map converts those to the pool's actual token.
pub fn realize(
    arcs: &[PoolArc],
    psi: &[f64],
    nu: &[f64],
    nodes: &NodeMap,
    src_token: &str,
    dst_token: &str,
    amount_in: U256,
    potentials: Option<&[f64]>,
) -> Result<RealizedRoute> {
    let mut route = RealizedRoute {
        src_token: src_token.to_ascii_lowercase(),
        dst_token: dst_token.to_ascii_lowercase(),
        amount_in,
        ..Default::default()
    };
    if arcs.is_empty() {
        return Err(RealizationError("no arcs carry flow".into()));
    }

    let tau: Vec<i64> = arcs.iter().map(|a| a.tau as i64).collect();
    let sig: Vec<i64> = arcs.iter().map(|a| a.sigma as i64).collect();
    let mut touched: Vec<usize> = tau
        .iter()
        .chain(sig.iter())
        .map(|&v| v as usize)
        .collect();
    touched.sort_unstable();
    touched.dedup();
    let index = |node: usize| touched.iter().position(|&v| v == node).unwrap() as i64;
    let local_tau: Vec<i64> = tau.iter().map(|&t| index(t as usize)).collect();
    let local_sig: Vec<i64> = sig.iter().map(|&s| index(s as usize)).collect();
    let order = topological_nodes(&local_tau, &local_sig, touched.len())?;
    let node_order: Vec<usize> = order.iter().map(|&k| touched[k]).collect();

    slot(&mut route, nodes, src_token)?; // slot 0 is always the input
    if let Some(values) = potentials {
        route.potentials = touched.iter().map(|&n| (n, values[n])).collect();
    }

    // --- amounts --------------------------------------------------------
    let mut deltas: Vec<U256> = Vec::with_capacity(arcs.len());
    let mut outs: Vec<U256> = Vec::with_capacity(arcs.len());
    for (arc, &flow) in arcs.iter().zip(psi.iter()) {
        let delta_canonical = flow / nu[arc.tau];
        let delta_token = delta_canonical / nodes.rate(&arc.token_in);
        deltas.push(to_int(delta_token * pow10(nodes.decimals(&arc.token_in)))?);
        // (M1) is only valid on [0, a/B], where f_hat' hits zero; beyond that
        // the model turns *decreasing*, so it is a hard box constraint rather
        // than something to watch. Clipping keeps a solver excursion from
        // showing up as a negative -- and therefore zero -- output.
        let domain = if arc.b > 0.0 { arc.a / arc.b } else { f64::INFINITY };
        let d = delta_canonical.min(domain);
        let model = arc.a * d - 0.5 * arc.b * d * d;
        let out_token = model.max(0.0) / nodes.rate(&arc.token_out);
        outs.push(to_int(out_token * pow10(nodes.decimals(&arc.token_out)))?);
    }

    let mut by_source: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, arc) in arcs.iter().enumerate() {
        match by_source.iter_mut().find(|(node, _)| *node == arc.tau) {
            Some((_, group)) => group.push(k),
            None => by_source.push((arc.tau, vec![k])),
        }
    }

    // --- emit -----------------------------------------------------------
    let destination = nodes
        .node(dst_token)
        .ok_or_else(|| RealizationError(format!("{dst_token} is not in the graph")))?;
    let dst_lower = dst_token.to_ascii_lowercase();
    let src_node = nodes
        .node(src_token)
        .ok_or_else(|| RealizationError(format!("{src_token} is not in the graph")))?;
    let reused = reused_pools(arcs);

    for &node in &node_order {
        let canonical = nodes.canonical_of[node].clone();
        let outgoing: Vec<usize> = by_source
            .iter()
            .find(|(n, _)| *n == node)
            .map(|(_, g)| g.clone())
            .unwrap_or_default();

        let mut incoming_tokens: Vec<String> = arcs
            .iter()
            .filter(|a| a.sigma == node)
            .map(|a| a.token_out.to_ascii_lowercase())
            .collect();
        if node == src_node {
            incoming_tokens.push(src_token.to_ascii_lowercase());
        }
        incoming_tokens.sort();
        incoming_tokens.dedup();

        // (0) which member of this node should actually hold the balance?
        //
        // The canonical token is an arbitrary label -- what matters is which
        // member the legs want. Defaulting to it round-trips a node whose flow
        // both arrives *and* leaves as the same non-canonical token, two legs
        // and two lots of integer rounding to end where it started.
        //
        // Only when the whole node agrees, and only when everything arriving
        // can reach the new hub in one hop -- every conversion is defined
        // against the canonical, so a second non-canonical arrival would need
        // two. Left alone for the destination, whose tail is keyed to the
        // canonical, and for mixed nodes, where funnelling through one slot is
        // what makes the `bps` split well defined.
        let mut hub = canonical.clone();
        if !outgoing.is_empty() && node != destination {
            let mut wanted: Vec<String> = outgoing
                .iter()
                .map(|&k| arcs[k].token_in.to_ascii_lowercase())
                .collect();
            wanted.sort();
            wanted.dedup();
            if wanted.len() == 1 {
                let only = &wanted[0];
                if *only != canonical
                    && nodes.conversion.contains_key(only)
                    && incoming_tokens
                        .iter()
                        .all(|t| t == only || *t == canonical)
                {
                    hub = only.clone();
                }
            }
        }

        // (1) fold everything held at this node into the hub
        for token in &incoming_tokens {
            if *token == hub {
                continue;
            }
            // The destination has no outgoing arcs, so skipping it here left a
            // route that ends in native ETH depositing into the ETH slot while
            // the caller asked for WETH -- the quoter then reads the WETH
            // slot, finds nothing, and the whole candidate reads as
            // "reverted".
            if outgoing.is_empty() && node != destination {
                continue;
            }
            // Flow that already arrives *as the token the caller asked for* is
            // finished. Folding it into the hub only for the destination tail
            // below to convert it straight back is two legs and two lots of
            // integer rounding to end where it started.
            if node == destination && outgoing.is_empty() && *token == dst_lower {
                continue;
            }
            // Every conversion is defined against the canonical, so folding
            // *into* a non-canonical hub runs the same one backwards.
            let (found, forward) = if *token == canonical {
                (nodes.conversion.get(&hub), false)
            } else {
                (nodes.conversion.get(token), true)
            };
            let Some(conversion) = found else { continue };
            if conversion.is_alias() {
                continue; // an alias is already the same balance; see nodes.rs
            }
            let conversion = conversion.clone();
            let (src, dst) = (slot(&mut route, nodes, token)?, slot(&mut route, nodes, &hub)?);
            route.legs.push(conversion_leg(nodes, &conversion, forward, src, dst, 0)?);
        }

        if outgoing.is_empty() {
            continue;
        }

        // (2) everything leaving the hub, as one contiguous group
        let total: U256 = outgoing.iter().map(|&k| deltas[k]).fold(U256::ZERO, |a, b| a + b);
        if total.is_zero() {
            continue;
        }
        let hub_slot = slot(&mut route, nodes, &hub)?;
        let mut spokes: Vec<(usize, usize)> = Vec::new(); // (arc index, spoke slot)
        let mut hub_arcs: Vec<usize> = Vec::new();
        let mut by_token: Vec<(String, Vec<usize>)> = Vec::new(); // spoke token -> arcs
        for &k in &outgoing {
            let token_in = arcs[k].token_in.to_ascii_lowercase();
            // Compare *slots*, not addresses. An alias is a second address
            // over one balance, so `slot` deliberately collapses it onto the
            // canonical, and an arc drawing on the alias is already drawing on
            // the hub. Testing the address instead sent it down the spoke
            // path, where the conversion leg it then built moved slot 0 to
            // slot 0 and `Leg` refused it outright.
            let its_slot = slot(&mut route, nodes, &token_in)?;
            if its_slot == hub_slot {
                hub_arcs.push(k);
            } else {
                spokes.push((k, its_slot));
                match by_token.iter_mut().find(|(t, _)| *t == token_in) {
                    Some((_, group)) => group.push(k),
                    None => by_token.push((token_in, vec![k])),
                }
            }
        }

        // One fill per spoke, not one per arc drawing on it. Every arc wanting
        // scrvUSD used to get its own crvUSD -> scrvUSD wrap: measured on
        // crvUSD -> sDOLA at $100,000, four deposits at one ratio into one
        // slot where a single deposit of the total pays the same 80,140.808884
        // -- three redundant vault calls, and three of the caller's 32 legs,
        // for nothing. The draw side has always grouped this way; see (3).
        //
        // Keyed on the *token* rather than the slot it lands in, because the
        // conversion is a property of the token: two of them sharing a slot
        // are two different calls and still need a leg each.
        let mut group: Vec<Item> = hub_arcs.iter().map(|&k| Item::Arc(k)).collect();
        group.extend(by_token.iter().map(|(t, _)| Item::Spoke(t.clone())));

        let behind = |item: &Item| -> Vec<usize> {
            match item {
                Item::Arc(k) => vec![*k],
                Item::Spoke(t) => by_token
                    .iter()
                    .find(|(name, _)| name == t)
                    .map(|(_, g)| g.clone())
                    .unwrap_or_default(),
            }
        };

        // The last leg of a group sweeps -- `bps == 0` takes whatever is left,
        // so no dust strands in the slot -- and that makes the order
        // load-bearing twice over.
        //
        // **A capped arc must never be last.** Its `cap` is honoured in the
        // solve and there is no room for it in the calldata, so being the
        // sweeper hands it the remainder whatever the solve decided. Measured
        // on USDT -> ZCHF at $10,000: the USD3 vault arc holds `cap = 5.0e-05`
        // and `clamped`, the solve gave it nothing, and coming last out of the
        // USDC slot handed it 99.7% of the trade -- 9,960 USDC into a vault
        // whose `maxDeposit` is 1,142.
        //
        // **A pool entered twice needs the legs we cannot advance past last**,
        // which is what lets the gnosis split swap through the 3pool and then
        // deposit into it. Where the two fight, the cap wins: emitting a
        // reentry in the wrong order costs a candidate, because
        // `check_one_arc_per_pool` refuses it, and a capped sweeper costs a
        // reverted route.
        let absorbs_remainder = |item: &Item| -> bool {
            // A fill hands its remainder on to whatever draws on it, so it may
            // sweep only when every one of those can take it.
            behind(item).iter().all(|&k| !arcs[k].cap.is_finite())
        };
        let late = |item: &Item| -> bool {
            !reused.is_empty()
                && behind(item).iter().any(|&k| {
                    reused.contains(&arcs[k].pool.to_ascii_lowercase())
                        && !ADVANCEABLE.contains(&arcs[k].kind)
                })
        };
        // Stable, as Python's `list.sort` is: two items with the same key keep
        // the order the hub and spoke lists put them in.
        group.sort_by_key(|item| (absorbs_remainder(item), late(item)));
        // Nothing here can take the remainder, so nobody sweeps and the
        // rounding dust stays in the slot. A few wei stranded is a cost; a leg
        // that sweeps past its cap is a route that does not run.
        let sweeper: i64 = if group.iter().any(absorbs_remainder) {
            group.len() as i64 - 1
        } else {
            -1
        };

        for (position, item) in group.iter().enumerate() {
            let carried = behind(item);
            let share: U256 = carried.iter().map(|&k| deltas[k]).fold(U256::ZERO, |a, b| a + b);
            let bps = if position as i64 == sweeper {
                0
            } else {
                share_bps(share, total)
            };
            match item {
                Item::Spoke(token_in) => {
                    let conversion = nodes
                        .conversion
                        .get(token_in)
                        .ok_or_else(|| {
                            RealizationError(format!("{token_in} has no conversion"))
                        })?
                        .clone();
                    let dst = slot(&mut route, nodes, token_in)?;
                    route
                        .legs
                        .push(conversion_leg(nodes, &conversion, false, hub_slot, dst, bps)?);
                }
                Item::Arc(k) => {
                    let dst = slot(&mut route, nodes, &arcs[*k].token_out)?;
                    route.legs.push(arc_leg(
                        &arcs[*k], hub_slot, dst, bps, deltas[*k], outs[*k], psi[*k],
                        ratio(deltas[*k], total),
                    )?);
                }
            }
        }

        // (3) arcs that had to draw from a spoke. One group per spoke *slot*,
        // not one per arc.
        //
        // The quoter groups by contiguous `src_slot`, so two arcs drawing from
        // the same spoke are one `bps` group however they were emitted. Giving
        // each of them `bps = 0` put two sweepers in that group: the first
        // took the whole slot and the second was left with nothing to trade,
        // which is a leg that can never do anything. Measured on
        // crvUSD -> sDOLA at $2M, a candidate carried `SaveDola` and
        // `LlamaThena` both sweeping slot 1.
        let mut by_spoke: Vec<(usize, Vec<usize>)> = Vec::new();
        for &(k, spoke) in &spokes {
            match by_spoke.iter_mut().find(|(s, _)| *s == spoke) {
                Some((_, group)) => group.push(k),
                None => by_spoke.push((spoke, vec![k])),
            }
        }
        for (spoke, ks) in by_spoke.iter_mut() {
            // The sweeper goes last and must be able to absorb the remainder,
            // the same rule the hub group above follows and for the same
            // reason -- which is `not isfinite`, the uncapped arc, and this
            // sorted the other way round: `isfinite` ascending puts the
            // *capped* one last and hands it the whole slot, the very thing
            // the USD3 measurement above is about.
            ks.sort_by_key(|&k| !arcs[k].cap.is_finite());
            let drawn: U256 = ks.iter().map(|&k| deltas[k]).fold(U256::ZERO, |a, b| a + b);
            for (position, &k) in ks.iter().enumerate() {
                let bps = if position == ks.len() - 1 || drawn.is_zero() {
                    0
                } else {
                    share_bps(deltas[k], drawn)
                };
                let dst = slot(&mut route, nodes, &arcs[k].token_out)?;
                route.legs.push(arc_leg(
                    &arcs[k], *spoke, dst, bps, deltas[k], outs[k], psi[k],
                    ratio(deltas[k], total),
                )?);
            }
        }
    }

    // --- destination ----------------------------------------------------
    let dst_node = destination;
    let dst_canonical = nodes.canonical_of[dst_node].clone();
    // Only convert out of the hub if anything actually landed in it. Arrivals
    // already in `dst_token` bypass the fold above, so when *every* leg pays
    // the requested token the hub is empty and this leg would move zero --
    // harmless in the quoter, but it still spends one of the caller's legs.
    let arriving: Vec<String> = arcs
        .iter()
        .filter(|a| a.sigma == dst_node)
        .map(|a| a.token_out.to_ascii_lowercase())
        .collect();
    let hub_has_balance = arriving.iter().any(|token| {
        *token == dst_canonical || (*token != dst_lower && nodes.conversion.contains_key(token))
    });
    if dst_lower != dst_canonical && hub_has_balance {
        // An alias needs no leg and cannot have one: `slot` already put both
        // addresses in one accumulator, so this would move a slot to itself.
        if let Some(conversion) = nodes.conversion.get(&dst_lower).cloned() {
            if !conversion.is_alias() {
                let src = slot(&mut route, nodes, &dst_canonical)?;
                let dst = slot(&mut route, nodes, dst_token)?;
                route.legs.push(conversion_leg(nodes, &conversion, false, src, dst, 0)?);
            }
        }
    }
    route.dst_slot = slot(&mut route, nodes, dst_token)?;
    route.modelled_out = forward_simulate(&mut route, nodes);
    route.paths = decompose(&mut route);
    Ok(route)
}

/// `max(1, min(BPS - 1, round(BPS * share / total)))`.
fn share_bps(share: U256, total: U256) -> i64 {
    // `BPS * share` is an integer on the reference side and the division that
    // follows is the correctly-rounded one, so the multiply happens first and
    // in 256 bits -- not as a float scaling of a ratio.
    let rounded = py_round(divided(U256::from(BPS as u64) * share, total));
    (rounded as i64).clamp(1, BPS - 1)
}

fn ratio(part: U256, total: U256) -> f64 {
    divided(part, total)
}

#[allow(clippy::too_many_arguments)]
fn arc_leg(
    arc: &PoolArc,
    src: usize,
    dst: usize,
    bps: i64,
    amount_in: U256,
    amount_out: U256,
    psi: f64,
    share: f64,
) -> Result<RealizedLeg> {
    let impact = if arc.g > 0.0 { psi / (2.0 * arc.g) } else { 0.0 };
    let reserve_in = U256::from(arc.reserve_in);
    let theta = if reserve_in.is_zero() {
        0.0
    } else {
        divided(amount_in, reserve_in)
    };
    let leg = Leg::new(
        arc.pool.clone(), arc.kind, arc.i, arc.j, arc.n_coins,
        src as i32, dst as i32, bps as i32,
    )?;
    Ok(RealizedLeg {
        token_in: arc.token_in.clone(),
        token_out: arc.token_out.clone(),
        amount_in,
        amount_out,
        share_of_node: share,
        cap_in: if arc.cap.is_finite() {
            arc.cap * pow10(arc.decimals_in)
        } else {
            f64::INFINITY
        },
        arc_id: Some(arc.id.clone()),
        pool_name: arc.note.clone(),
        eps: arc.eps,
        impact_frac: impact,
        theta,
        psi,
        reserve_in,
        tvl_usd: arc.tvl_usd,
        gamma_live: arc.gamma_live,
        modelled: arc.g > 0.0,
        ..RealizedLeg::blank(leg, arc.kind, arc.pool.clone())
    })
}

/// `forward` means token -> canonical.
fn conversion_leg(
    nodes: &NodeMap,
    conversion: &Conversion,
    forward: bool,
    src: usize,
    dst: usize,
    bps: i64,
) -> Result<RealizedLeg> {
    let kind = if forward {
        conversion.forward_kind()
    } else {
        conversion.reverse_kind()
    };
    let (token_in, token_out) = if forward {
        (conversion.token.clone(), conversion.canonical.clone())
    } else {
        (conversion.canonical.clone(), conversion.token.clone())
    };
    let target = if conversion.target.is_empty() {
        token_out.clone()
    } else {
        conversion.target.clone()
    };
    let leg = Leg::new(
        target.clone(), kind, 0, 1, 2, src as i32, dst as i32, bps as i32,
    )?;
    Ok(RealizedLeg {
        pool_name: format!("{} -> {}", nodes.symbol(&token_in), nodes.symbol(&token_out)),
        token_in,
        token_out,
        ..RealizedLeg::blank(leg, kind, target)
    })
}

/// The route between two tokens of the *same* node: the conversion itself.
///
/// Merging is what lets the solve treat crvUSD and scrvUSD, or ETH and WETH,
/// as one place -- and it also means there is no arc between them and nothing
/// for the graph to find. Asking for one used to be an error, which is right
/// about the model and wrong about the question: crvUSD -> scrvUSD is a real
/// trade, it just happens to be a deposit rather than a swap.
///
/// Every conversion is defined against the node's canonical token, so the
/// answer is at most two legs -- in to the canonical, out to the target -- and
/// often one. An ALIAS pair emits none: holding one is holding the other.
pub fn conversion_route(
    nodes: &NodeMap,
    src_token: &str,
    dst_token: &str,
    amount_in: U256,
) -> Result<RealizedRoute> {
    let src = src_token.to_ascii_lowercase();
    let dst = dst_token.to_ascii_lowercase();
    let canonical = nodes
        .canonical(&src)
        .ok_or_else(|| RealizationError(format!("{src} is not in the graph")))?
        .to_string();
    let mut legs: Vec<RealizedLeg> = Vec::new();
    let mut at = 0usize;
    let mut token = src.clone();

    for (target, forward) in [(canonical.clone(), true), (dst.clone(), false)] {
        if token == target {
            continue;
        }
        let key = if forward { &token } else { &target };
        match nodes.conversion.get(key) {
            Some(conversion) if !conversion.is_alias() => {
                legs.push(conversion_leg(nodes, conversion, forward, at, at + 1, 0)?);
                at += 1;
                token = target;
            }
            _ => {
                token = target;
            }
        }
    }

    // Both tokens are the same node by construction; the renderer labels each
    // bus from this, and without it every slot reads as node 0 -- which drew
    // "crvUSD = DAI/sDAI" over a crvUSD -> scrvUSD deposit.
    let node = nodes
        .node(&src)
        .ok_or_else(|| RealizationError(format!("{src} is not in the graph")))?;
    let empty = legs.is_empty();
    let mut route = RealizedRoute {
        legs,
        dst_slot: at,
        src_token: src.clone(),
        dst_token: dst.clone(),
        amount_in,
        slots: if empty {
            vec![(src, 0)]
        } else {
            vec![(src, 0), (dst, at)]
        },
        node_of_slot: (0..=at).map(|k| (k, node)).collect(),
        ..Default::default()
    };
    route.modelled_out = if empty {
        amount_in
    } else {
        forward_simulate(&mut route, nodes)
    };
    Ok(route)
}

/// Replay the legs the way the quoter will, to fill in modelled amounts.
///
/// The *routing* matches the contract exactly -- same group snapshot, same
/// `bps`-of-base arithmetic, same order -- so the split the diagram shows is
/// the split the quoter will be asked to confirm.
///
/// The *amounts* are the model's, not the chain's, and deliberately so. Each
/// leg keeps the ratio its arc was calibrated at, rescaled linearly to
/// whatever arrives, which is a straight line through a curve: worst measured
/// drift against a stateful walk of the same legs is 2.78 bp.
///
/// Pricing these legs from the exact models instead would destroy the one
/// number that makes the loss ledger worth reading: `verified - modelled`
/// measures the model, so a `modelled_out` computed from the exact models
/// would report its own accuracy as zero.
pub fn forward_simulate(route: &mut RealizedRoute, nodes: &NodeMap) -> U256 {
    let mut balances: Vec<(i32, U256)> = vec![(0, route.amount_in)];
    let get = |balances: &Vec<(i32, U256)>, slot: i32| -> U256 {
        balances
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, v)| *v)
            .unwrap_or(U256::ZERO)
    };
    let set = |balances: &mut Vec<(i32, U256)>, slot: i32, value: U256| {
        match balances.iter_mut().find(|(s, _)| *s == slot) {
            Some(entry) => entry.1 = value,
            None => balances.push((slot, value)),
        }
    };
    let mut current: Option<i32> = None;
    let mut base = U256::ZERO;
    let bps = U256::from(BPS as u64);

    for realized in route.legs.iter_mut() {
        let src = realized.leg.src_slot;
        if Some(src) != current {
            current = Some(src);
            base = get(&balances, src);
        }
        let available = get(&balances, src);
        let take = if realized.leg.bps == 0 {
            available
        } else {
            (base * U256::from(realized.leg.bps as u64) / bps).min(available)
        };
        if take.is_zero() {
            continue;
        }
        // Same reason `theta` is refreshed below: the split optimiser retunes
        // `bps` and re-walks, and a share left at the solve's `psi` then names
        // the split before tuning.
        realized.share_of_node = if base.is_zero() {
            1.0
        } else {
            divided(take, base)
        };
        // A merge is a kind *and* a `Conversion`. A mint arc into a vault
        // nothing merged shares the kind and has no conversion, and passing
        // its amount through 1:1 priced a 1.0334 vault at par -- 3.3% out, in
        // the diagram and in `modelled_out`, while the quote itself was right.
        let conversion = if realized.is_conversion() {
            nodes
                .conversion
                .get(&realized.token_in.to_ascii_lowercase())
                .or_else(|| nodes.conversion.get(&realized.token_out.to_ascii_lowercase()))
        } else {
            None
        };
        let produced = match conversion {
            Some(conversion) => {
                // Direction from the conversion, not from a list of kinds: it
                // names the two itself, so a kind added later cannot be
                // missed.
                let moved = if realized.kind == conversion.forward_kind() {
                    conversion.to_canonical(take)
                } else {
                    conversion.from_canonical(take)
                };
                let moved = moved.unwrap_or(U256::ZERO);
                realized.amount_in = take;
                realized.amount_out = moved;
                moved
            }
            None => {
                // keep the modelled ratio, rescaled to whatever actually
                // arrives
                let moved = if realized.amount_in.is_zero() {
                    U256::ZERO
                } else {
                    mul_div(realized.amount_out, take, realized.amount_in)
                };
                realized.amount_in = take;
                realized.amount_out = moved;
                if !realized.reserve_in.is_zero() {
                    // The amounts just moved; `theta` describes them or it
                    // describes nothing. A model-free candidate is realised at
                    // `psi = 1` -- under a token of flow -- so a stale `theta`
                    // reads 0.00% on a leg taking several times the pool.
                    realized.theta = divided(take, realized.reserve_in);
                }
                moved
            }
        };
        set(&mut balances, src, available - take);
        let landed = get(&balances, realized.leg.dst_slot) + produced;
        set(&mut balances, realized.leg.dst_slot, landed);
    }
    get(&balances, route.dst_slot as i32)
}

/// `a * b // c`, with the product taken in 512 bits.
///
/// `amount_out * take` is routinely past 2^256 -- two wei-scale numbers
/// multiplied -- and the reference does it in Python `int`, which has no
/// ceiling.
fn mul_div(a: U256, b: U256, c: U256) -> U256 {
    use ruint::aliases::U512;
    if c.is_zero() {
        return U256::ZERO;
    }
    let wide = |v: U256| U512::from_limbs_slice(v.as_limbs());
    let quotient = wide(a) * wide(b) / wide(c);
    let limbs = quotient.as_limbs();
    if limbs[4..].iter().any(|&limb| limb != 0) {
        return U256::MAX;
    }
    U256::from_limbs_slice(&limbs[..4])
}

/// Flow decomposition, for display only.
///
/// Deliberately derived *after* the legs: the legs are the truth, and paths
/// are a reading of them. Doing it the other way round is what double-counts
/// shared pools.
///
/// Bounded at `MAX_DISPLAY_PATHS`: nobody reads the 65th path, and the legs --
/// which are the executable artefact -- are unaffected by the cut.
pub fn decompose(route: &mut RealizedRoute) -> Vec<Vec<String>> {
    let mut outgoing: Vec<(i32, Vec<usize>)> = Vec::new();
    for (k, realized) in route.legs.iter().enumerate() {
        match outgoing.iter_mut().find(|(s, _)| *s == realized.leg.src_slot) {
            Some((_, group)) => group.push(k),
            None => outgoing.push((realized.leg.src_slot, vec![k])),
        }
    }
    let mut paths: Vec<Vec<String>> = Vec::new();
    walk(route, &outgoing, 0, &[], 0, &mut paths);
    if paths.len() >= MAX_DISPLAY_PATHS {
        route.warnings.push(format!(
            "path list truncated at {MAX_DISPLAY_PATHS}; the legs are complete \
             and are what executes"
        ));
    }
    paths
}

fn walk(
    route: &RealizedRoute,
    outgoing: &[(i32, Vec<usize>)],
    slot: i32,
    trail: &[String],
    depth: usize,
    paths: &mut Vec<Vec<String>>,
) {
    if depth > 16 || paths.len() >= MAX_DISPLAY_PATHS {
        return;
    }
    let legs = outgoing.iter().find(|(s, _)| *s == slot).map(|(_, g)| g);
    let empty = legs.is_none_or(|g| g.is_empty());
    if empty || slot == route.dst_slot as i32 {
        if !trail.is_empty() {
            paths.push(trail.to_vec());
        }
        return;
    }
    for &k in legs.unwrap() {
        if paths.len() >= MAX_DISPLAY_PATHS {
            return;
        }
        let realized = &route.legs[k];
        let label = match realized.arc_id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            // `target[:10]` on characters, the way Python slices a `str`.
            _ => format!(
                "{}:{}",
                realized.kind.name(),
                realized.target.chars().take(10).collect::<String>()
            ),
        };
        let mut next: Vec<String> = trail.to_vec();
        next.push(label);
        walk(route, outgoing, realized.leg.dst_slot, &next, depth + 1, paths);
    }
}

/// Pools carrying more than one of these arcs.
fn reused_pools(arcs: &[PoolArc]) -> Vec<String> {
    let mut seen: Vec<(String, usize)> = Vec::new();
    for arc in arcs {
        let key = arc.pool.to_ascii_lowercase();
        match seen.iter_mut().find(|(p, _)| *p == key) {
            Some(entry) => entry.1 += 1,
            None => seen.push((key, 1)),
        }
    }
    seen.into_iter().filter(|(_, n)| *n > 1).map(|(p, _)| p).collect()
}

/// Decision 3: a pool appears once, or its legs form one multi-port element.
///
/// A view-only chained quoter cannot see its own earlier leg, so two arcs of
/// one pool priced independently are priced against a state neither will see.
/// The old exemption let them through when every leg but the last was
/// `ADVANCEABLE`, which bought a walk that advances the pool but still models
/// the arcs as two independent resistors -- separate `psi^2/2G` terms, no
/// cross-term.
///
/// An **element** is the same trade with the pool appearing once, so the
/// coupling *is* the advancing state. Admissibility is then structural rather
/// than a rule to remember (`multiport.rs`).
///
/// Returns the pool addresses whose legs are not an admissible element.
pub fn check_one_arc_per_pool(route: &RealizedRoute) -> Vec<String> {
    let mut order: Vec<(String, Vec<usize>)> = Vec::new();
    for (k, realized) in route.legs.iter().enumerate() {
        if realized.is_conversion() {
            continue;
        }
        let key = realized.target.to_ascii_lowercase();
        match order.iter_mut().find(|(p, _)| *p == key) {
            Some(entry) => entry.1.push(k),
            None => order.push((key, vec![k])),
        }
    }
    let mut bad: Vec<String> = Vec::new();
    for (pool, legs) in &order {
        if legs.len() < 2 {
            continue;
        }
        let first = &route.legs[legs[0]];
        let triples: Vec<(ArcKind, i32, i32)> = legs
            .iter()
            .map(|&k| {
                let rl = &route.legs[k];
                (rl.kind, rl.leg.i, rl.leg.j)
            })
            .collect();
        if element_of_legs(&first.target, first.leg.n, &triples).is_err() {
            bad.push(pool.clone());
        }
    }
    bad.sort();
    bad
}

fn element_of_legs(
    target: &str,
    n: i32,
    triples: &[(ArcKind, i32, i32)],
) -> std::result::Result<(), MultiPortError> {
    element_from(target, n, triples).map(|_| ())
}

/// The route as a resistor network: `1/TVL` per pool, src to dst.
///
/// The same reading the rest of the router uses, applied to a whole candidate
/// rather than one arc. Series hops add resistance and parallel branches add
/// conductance, so a topology that splits across deep pools scores above one
/// that funnels everything through a thin series chain.
///
/// **TVL, not the fitted `G`.** The scout exists because the model's split is
/// not to be trusted on wide topologies; ranking those candidates by a number
/// the same model produced would inherit the error being corrected. The pool's
/// own size is independent of it.
///
/// Node merges are shorts (`eps = 0`, `G = infinity`, §3.1), so their slots
/// are joined rather than given an edge. Returns 0 when src cannot reach dst
/// through pools with a size to speak of.
pub fn route_conductance(route: &RealizedRoute) -> f64 {
    if route.legs.is_empty() {
        return 0.0;
    }
    let mut parent: Vec<(i32, i32)> = Vec::new();
    fn find(parent: &mut Vec<(i32, i32)>, x: i32) -> i32 {
        if !parent.iter().any(|(k, _)| *k == x) {
            parent.push((x, x));
        }
        let mut at = x;
        loop {
            let up = parent.iter().find(|(k, _)| *k == at).map(|(_, v)| *v).unwrap();
            if up == at {
                return at;
            }
            let grand = {
                if !parent.iter().any(|(k, _)| *k == up) {
                    parent.push((up, up));
                }
                parent.iter().find(|(k, _)| *k == up).map(|(_, v)| *v).unwrap()
            };
            if let Some(entry) = parent.iter_mut().find(|(k, _)| *k == at) {
                entry.1 = grand;
            }
            at = grand;
        }
    }

    let mut slots: Vec<i32> = vec![0, route.dst_slot as i32];
    for realized in &route.legs {
        slots.push(realized.leg.src_slot);
        slots.push(realized.leg.dst_slot);
    }
    for realized in &route.legs {
        if realized.is_conversion() {
            let a = find(&mut parent, realized.leg.src_slot);
            let b = find(&mut parent, realized.leg.dst_slot);
            if a != b {
                if let Some(entry) = parent.iter_mut().find(|(k, _)| *k == a) {
                    entry.1 = b;
                }
            }
        }
    }
    let src = find(&mut parent, 0);
    let dst = find(&mut parent, route.dst_slot as i32);
    if src == dst {
        return f64::INFINITY; // nothing but merges between the two ends
    }

    let mut nodes: Vec<i32> = slots.iter().map(|&s| find(&mut parent, s)).collect();
    nodes.sort_unstable();
    nodes.dedup();
    let position = |node: i32, nodes: &[i32]| nodes.iter().position(|&v| v == node).unwrap();
    let n = nodes.len();
    let mut laplacian = vec![0.0f64; n * n];
    for realized in &route.legs {
        if realized.is_conversion() || realized.tvl_usd <= 0.0 {
            continue;
        }
        let a = position(find(&mut parent, realized.leg.src_slot), &nodes);
        let b = position(find(&mut parent, realized.leg.dst_slot), &nodes);
        if a == b {
            continue;
        }
        laplacian[a * n + a] += realized.tvl_usd;
        laplacian[b * n + b] += realized.tvl_usd;
        laplacian[a * n + b] -= realized.tvl_usd;
        laplacian[b * n + a] -= realized.tvl_usd;
    }

    // Ground the destination and inject a unit current at the source: the
    // potential left at the source *is* the effective resistance.
    let grounded = position(dst, &nodes);
    let keep: Vec<usize> = (0..n).filter(|&k| k != grounded).collect();
    if keep.is_empty() {
        return 0.0;
    }
    let at_src = keep.iter().position(|&k| k == position(src, &nodes)).unwrap();
    let mut matrix = vec![0.0f64; keep.len() * keep.len()];
    for (r, &row) in keep.iter().enumerate() {
        for (c, &col) in keep.iter().enumerate() {
            matrix[r * keep.len() + c] = laplacian[row * n + col];
        }
    }
    let mut rhs = vec![0.0f64; keep.len()];
    rhs[at_src] = 1.0;
    if crate::lu::solve_in_place(&mut matrix, &mut rhs, keep.len()).is_err() {
        return 0.0; // src and dst are in different components
    }
    let resistance = rhs[at_src];
    if resistance > 0.0 {
        1.0 / resistance
    } else {
        0.0
    }
}

pub fn max_theta(route: &RealizedRoute) -> f64 {
    route
        .legs
        .iter()
        .filter(|rl| !rl.is_conversion())
        .map(|rl| rl.theta)
        .fold(0.0f64, |a, b| if b > a { b } else { a })
}

/// Realised loss against a frictionless trade at the reference price.
pub fn total_loss_bp(route: &RealizedRoute, price_out_per_in: f64) -> f64 {
    if route.amount_in.is_zero() || price_out_per_in <= 0.0 {
        return f64::NAN;
    }
    let ideal = scaled(route.amount_in, 0) * price_out_per_in;
    if ideal <= 0.0 {
        return f64::NAN;
    }
    (1.0 - scaled(route.modelled_out, 0) / ideal) * 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kahn_orders_a_chain_and_refuses_a_cycle() {
        assert_eq!(topological_nodes(&[0, 1], &[1, 2], 3).unwrap(), vec![0, 1, 2]);
        let err = topological_nodes(&[0, 1], &[1, 0], 2).unwrap_err();
        assert!(err.0.contains("contain a cycle"));
    }

    #[test]
    fn rounding_is_ties_to_even() {
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(-0.5), -0.0);
        assert_eq!(py_round(2.4999), 2.0);
    }

    #[test]
    fn a_cap_is_compared_exactly() {
        // 1e24 as an f64 is not 10**24, so a rounded comparison gets this
        // wrong in one direction or the other.
        let amount = U256::from(10u64).pow(U256::from(24u64));
        assert!(!over(amount, f64::INFINITY));
        assert!(over(amount, 1.0));
        assert!(!over(U256::from(5u8), 5.5));
        assert!(over(U256::from(6u8), 5.5));
    }

    #[test]
    fn a_branch_under_the_dust_share_goes_and_takes_its_tail() {
        // 0 -> 1 carries everything; 0 -> 2 -> 3 carries 1e-6 of it and 3 is
        // not the destination, so the whole branch goes.
        let tau = [0i64, 0, 2];
        let sig = [1i64, 2, 3];
        let psi = [1.0f64, 1e-6, 1e-6];
        let (flow, removed) = prune_dust(&tau, &sig, &psi, 0, 1, DUST_SHARE, 1e-12);
        assert_eq!(flow, vec![1.0, 0.0, 0.0]);
        assert_eq!(removed, 2);
    }

    #[test]
    fn pruning_may_not_decide_there_is_no_route() {
        // Every arc is dust against itself, so pruning would empty the flow;
        // the original stands instead.
        let tau = [0i64];
        let sig = [1i64];
        let psi = [1e-30f64];
        let (flow, removed) = prune_dust(&tau, &sig, &psi, 0, 1, 1.0, 1e-40);
        assert_eq!(flow, vec![1e-30]);
        assert_eq!(removed, 0);
    }
}
