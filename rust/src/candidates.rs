//! Candidate generation (spec §6).
//!
//! The mirror of `core/candidates.py`.
//!
//! The solved flow is the optimum of a *relaxation*, and three things make it
//! unfit to quote directly:
//!
//! * clamped arcs look bottomless, so §2.3 predicts they are preferentially
//!   filled rather than probed away;
//! * reference prices are fitted, so real dislocations read as free money and
//!   the solver spreads across dozens of arcs chasing them;
//! * a pool may end up carrying flow on two arcs, which a view-only quoter
//!   cannot evaluate because it cannot see its own earlier leg.
//!
//! So the model chooses *which pools*, and the on-chain quote chooses
//! *between candidates*. Every generator below is a cheap re-solve -- a rank-1
//! change to the active set -- and the multicall adjudicates.
//!
//! Priority order matters when the budget truncates. The pin sweep outranks
//! the drop candidates because §13.1's chord regression is explicit that the
//! active set is *identical* across the endpoint allocations, so no
//! drop-an-arc candidate can find the interior optimum.

use crate::graph::ArcArrays;
use crate::multiport::element_of_arcs;
use crate::realize::{prune_dust, RealizedRoute, DUST_SHARE};
use crate::seed::{build_adjacency, k_shortest_paths};
use crate::solve::{active_set_solve, Arcs, Options, Solution};
use crate::types::PoolArc;
use crate::cycles;

/// Sparsification levels. The relaxation routinely activates dozens of arcs
/// chasing fitted dislocations; these ask "what if you only had k pools?" and
/// double as §11.1's gas-sparsification candidates.
pub const TOP_K: [usize; 7] = [1, 2, 3, 4, 6, 8, 12];
pub const PIN_LADDER: [f64; 7] = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
/// An arc carrying less than this fraction of the trade cannot change the
/// outcome, but chasing it costs pivots. Candidates are heuristics adjudicated
/// by the quoter, so they are solved to this screen; the certified base solve
/// is not.
pub const MIN_FLOW_FRACTION: f64 = 1e-4;
/// A candidate is realised into the quoter's slot accumulator, one slot per
/// distinct token it touches. A support wider than that cannot be priced no
/// matter how good it is.
///
/// Predicting it from the relaxation's own width does not work: forbidding an
/// arc makes the re-solve find a different and sometimes far narrower support,
/// and skipping those families cost 5.65 bp on a $100 swap. So a family is
/// stopped only once it has actually produced this many unrealisable
/// candidates in a row.
pub const WIDE_STREAK: usize = 2;
/// Candidates are heuristics; stopping early yields a feasible flow, not a
/// broken one, and the quoter is what decides between them anyway.
pub const CANDIDATE_PIVOTS: u32 = 60;
/// Repair rounds per candidate. Three was enough while the repair only ever
/// made one choice; branching to the next arc down spends a round each time it
/// has to back out, and the deepest measured chain is three bans then two
/// backtracks.
pub const REPAIR_ROUNDS: usize = 6;

/// Flow below this share of the trade is not a decision the solve made -- it
/// is the residue of a pivot, and it differs in the last bits between one
/// linear kernel and another. Testing `psi > 0` therefore made *membership*
/// itself kernel-dependent: an arc carrying 1e-18 counted as active under LU
/// and not under Cholesky, which changed which pools conflict, which changed
/// the repair candidates, which changed the ballot -- measured, by 72 bp.
pub const ACTIVE_FLOOR: f64 = 1e-12;

/// Arcs the solve actually routed through, as a boolean mask.
pub fn carries(psi: &[f64], psi_total: f64) -> Vec<bool> {
    let floor = (ACTIVE_FLOOR * psi_total.abs()).max(0.0);
    psi.iter().map(|&v| v > floor).collect()
}

#[derive(Debug, Clone, Default)]
pub struct Candidate {
    pub label: String,
    pub psi: Vec<f64>,
    pub certificate: bool,
    pub reason: String,
    pub kind: String,
    pub n_arcs: usize,
    pub modelled_loss: f64,
    pub route: Option<RealizedRoute>,
    pub verified_out: Option<u128>,
    pub status: String,
    pub note: String,
    pub rank: Option<usize>,
    pub gas: i64,
    /// P(every leg's minimum-out holds until inclusion); 1.0 until priced.
    pub survival: f64,
}

impl Candidate {
    fn new(label: String, psi: Vec<f64>, certificate: bool, reason: &str,
           kind: &str, n_arcs: usize, modelled_loss: f64) -> Self {
        Self {
            label, psi, certificate,
            reason: reason.to_string(),
            kind: kind.to_string(),
            n_arcs, modelled_loss,
            route: None,
            verified_out: None,
            status: "pending".to_string(),
            note: String::new(),
            rank: None,
            gas: 0,
            survival: 1.0,
        }
    }

    pub fn ok(&self) -> bool {
        self.status == "ok" && self.verified_out.is_some()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CandidateSet {
    pub candidates: Vec<Candidate>,
    pub skipped: usize,
    /// How much solving this generation asked for. Surfaced because it is what
    /// separates "the build got slower" from "this block is harder": the same
    /// pair and size runs 48 solves at one block and 113 at another.
    pub solves: usize,
    pub pivots: usize,
    /// Node count of the relaxation when it was too wide for any perturbation
    /// of it to be realisable; 0 when the pin/drop/repair families ran
    /// normally.
    pub skipped_wide: usize,
}

impl CandidateSet {
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// The winner, by the rank `verify` assigned -- which is *not* simply the
    /// largest output: outputs within a hair of each other are the same
    /// answer, and the cheaper route to execute wins (§11.1).
    pub fn best(&self) -> Option<&Candidate> {
        let ranked: Vec<&Candidate> = self
            .candidates
            .iter()
            .filter(|c| c.ok() && c.rank.is_some())
            .collect();
        if !ranked.is_empty() {
            // `min` keeps the first of a tie, as Python's does.
            return ranked.into_iter().reduce(|a, b| if b.rank < a.rank { b } else { a });
        }
        self.candidates
            .iter()
            .filter(|c| c.ok())
            .reduce(|a, b| {
                if b.verified_out.unwrap_or(0) > a.verified_out.unwrap_or(0) { b } else { a }
            })
    }
}

/// Active set plus split rounded to 10 bp -- §6.2's dedup key.
fn signature(psi: &[f64], tol: f64) -> Vec<(usize, f64)> {
    let total: f64 = psi.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut out: Vec<(usize, f64)> = (0..psi.len())
        .filter(|&k| psi[k] > tol)
        .map(|k| (k, round3(psi[k] / total)))
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// `round(x, 3)`: decimal, ties to even, the way `graph` keys its duplicates.
fn round3(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.3}").parse().unwrap_or(x)
}

fn pool_of(arcs: &[PoolArc]) -> Vec<String> {
    arcs.iter().map(|a| a.pool.to_ascii_lowercase()).collect()
}

/// Each conflicting pool's arcs, the one carrying most first.
///
/// Decision 3 allows a pool one arc per route, so a conflict is a choice of
/// which to keep. Largest-first is the order to try them in, not the answer.
pub fn repair_order(
    conflicts: &[(String, Vec<usize>)],
    psi: &[f64],
) -> Vec<(String, Vec<usize>)> {
    conflicts
        .iter()
        .map(|(pool, indices)| {
            let mut sorted = indices.clone();
            // Stable and descending, which `sorted(key=lambda k: -psi[k])` is.
            sorted.sort_by(|&a, &b| {
                psi[b].partial_cmp(&psi[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            (pool.clone(), sorted)
        })
        .collect()
}

/// Ban every arc of each conflicting pool but the one at `rank`.
///
/// `rank = 0` is the greedy choice; higher ranks are the rest of the branch. A
/// rank past the end clamps, so a caller sweeping ranks cannot fall off.
/// Returns whether anything was newly banned -- nothing banned means the
/// repair has no move left to make and the caller must stop rather than loop.
pub fn keep_only(
    banned: &mut [bool],
    ordered: &[(String, Vec<usize>)],
    rank: usize,
    pinned: &[(usize, f64)],
) -> bool {
    let mut applied = false;
    for (_, indices) in ordered {
        let keep = indices[rank.min(indices.len() - 1)];
        for &k in indices {
            let is_pinned = pinned.iter().any(|&(at, _)| at == k);
            if k != keep && !is_pinned && !banned[k] {
                banned[k] = true;
                applied = true;
            }
        }
    }
    applied
}

/// Pools carrying flow on more than one arc whose arcs are not one element.
///
/// The same rule `check_one_arc_per_pool` applies to realised legs, asked here
/// of the arcs -- a coin holds at most one port, so `#in + #out <= N` and a
/// 2-coin pool cannot be entered twice. Order does not exist yet at this stage
/// and the rule does not need it: admissibility is a property of which ports
/// are used, not of the sequence.
///
/// `pools` and `cache` are the same answer asked twice. `generate` calls this
/// thirty-six times a quote against arcs that do not change, so lowering every
/// active arc's address and re-deriving the element from the same indices is
/// work the second call already knows the answer to.
pub fn conflicting_pools(
    arcs: &[PoolArc],
    psi: &[f64],
    psi_total: f64,
    pools: Option<&[String]>,
    cache: Option<&mut Vec<(Vec<usize>, bool)>>,
) -> Vec<(String, Vec<usize>)> {
    let owned;
    let lowered = match pools {
        Some(v) => v,
        None => {
            owned = pool_of(arcs);
            &owned
        }
    };
    let live: Vec<bool> = if psi_total != 0.0 {
        carries(psi, psi_total)
    } else {
        psi.iter().map(|&v| v > 0.0).collect()
    };
    // Insertion-ordered, as the reference's dict is. Bounded by the arc list
    // as well as the flow: a caller that hands over a `psi` of the wrong
    // length gets the arcs it named rather than a panic, which in wasm would
    // take the whole instance down.
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for k in 0..psi.len().min(lowered.len()) {
        if !live[k] {
            continue;
        }
        match groups.iter_mut().find(|(p, _)| *p == lowered[k]) {
            Some((_, idx)) => idx.push(k),
            None => groups.push((lowered[k].clone(), vec![k])),
        }
    }
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let mut store = cache;
    for (pool, idx) in groups {
        if idx.len() < 2 {
            continue;
        }
        if let Some(table) = store.as_deref_mut() {
            if let Some((_, clashes)) = table.iter().find(|(key, _)| *key == idx) {
                if *clashes {
                    out.push((pool, idx));
                }
                continue;
            }
        }
        let members: Vec<PoolArc> = idx.iter().map(|&k| arcs[k].clone()).collect();
        let clashes = element_of_arcs(&members).is_err();
        if clashes {
            out.push((pool, idx.clone()));
        }
        if let Some(table) = store.as_deref_mut() {
            table.push((idx, clashes));
        }
    }
    out
}

/// Re-order paths so each one brings pools the earlier ones did not.
///
/// The cheapest path stays first -- it is `C_*`, what a caller with no
/// appetite for splitting gets. After that, greedily take whichever remaining
/// path adds the most unseen pools, breaking ties toward the cheaper one, so
/// the cumulative unions grow in *venues* rather than in count. Ordering only.
pub fn by_new_pools(paths: &[Vec<usize>], pools: &[String]) -> Vec<Vec<usize>> {
    if paths.len() < 3 {
        return paths.to_vec();
    }
    let pool_sets: Vec<Vec<String>> = paths
        .iter()
        .map(|path| {
            let mut names: Vec<String> = path.iter().map(|&a| pools[a].clone()).collect();
            names.sort();
            names.dedup();
            names
        })
        .collect();
    let mut order = vec![0usize];
    let mut seen = pool_sets[0].clone();
    let mut remaining: Vec<usize> = (1..paths.len()).collect();
    while !remaining.is_empty() {
        // `max(key=(new pools, -i))` -- more new pools first, then the earlier
        // (cheaper) path.
        let mut best = 0usize;
        let mut best_key = (usize::MAX, 0i64);
        for (at, &i) in remaining.iter().enumerate() {
            let new = pool_sets[i].iter().filter(|p| !seen.contains(p)).count();
            let key = (new, -(i as i64));
            if best_key.0 == usize::MAX || key > (best_key.0, best_key.1) {
                best = at;
                best_key = key;
            }
        }
        let pick = remaining.remove(best);
        order.push(pick);
        for pool in &pool_sets[pick] {
            if !seen.contains(pool) {
                seen.push(pool.clone());
            }
        }
    }
    order.into_iter().map(|i| paths[i].clone()).collect()
}

/// The `k` levels to spend `budget` candidates on, across the whole ladder.
///
/// Taking `top_k` in order spends everything on its dense low end -- with
/// `(1, 2, 3, 4, 6, 8, 12)` and room for four, the widest union ever built is
/// four paths. Sampling evenly keeps `1` and reaches `12`.
pub fn spread(top_k: &[usize], budget: usize) -> Vec<usize> {
    let mut levels: Vec<usize> = top_k.to_vec();
    levels.sort_unstable();
    levels.dedup();
    if budget >= levels.len() || budget <= 1 {
        return levels;
    }
    let step = (levels.len() - 1) as f64 / (budget - 1) as f64;
    let mut picked: Vec<usize> = (0..budget)
        .map(|i| levels[py_round(i as f64 * step) as usize])
        .collect();
    picked.sort_unstable();
    picked.dedup();
    picked
}

/// `round(x)` as CPython does it: ties to even.
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

/// Everything `generate` takes past the graph and the base solve.
pub struct GenerateOptions {
    pub base_certificate: bool,
    pub max_candidates: usize,
    pub top_k: Vec<usize>,
    pub gas_floor: f64,
    pub max_legs: usize,
    pub max_slots: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            base_certificate: false,
            max_candidates: 20,
            top_k: TOP_K.to_vec(),
            gas_floor: 0.0,
            // The quoter's ABI capacity; the caller supplies its own.
            max_legs: 32,
            max_slots: 8,
        }
    }
}

/// What an element pricer answers: the two flows to pin, or nothing.
pub type ElementSplit<'a> = &'a dyn Fn(&PoolArc, &PoolArc, f64, f64) -> Option<(f64, f64)>;

struct Generator<'a> {
    g: &'a ArcArrays,
    arcs: &'a [PoolArc],
    src: usize,
    dst: usize,
    psi_total: f64,
    opts: &'a GenerateOptions,
    out: CandidateSet,
    seen: Vec<Vec<(usize, f64)>>,
    streak: Vec<(String, usize)>,
    pools: Vec<String>,
    /// Whether a set of arc indices forms one element -- a property of the
    /// arcs rather than of the solve that landed on them, so it is asked once
    /// per set.
    elements: Vec<(Vec<usize>, bool)>,
    warm: Vec<bool>,
}

impl Generator<'_> {
    fn arc_view(&self) -> Arcs<'_> {
        Arcs {
            tau: &self.g.tau,
            sig: &self.g.sig,
            g: &self.g.g,
            eps: &self.g.eps,
            cap: &self.g.cap,
            n_nodes: self.g.n_nodes,
        }
    }

    /// Has this family produced only unrealisable candidates lately?
    fn exhausted(&self, kind: &str) -> bool {
        self.streak
            .iter()
            .find(|(k, _)| k == kind)
            .is_some_and(|(_, n)| *n >= WIDE_STREAK)
    }

    fn bump(&mut self, kind: &str, to: Option<usize>) {
        match self.streak.iter_mut().find(|(k, _)| k == kind) {
            Some(entry) => entry.1 = to.unwrap_or(entry.1 + 1),
            None => self.streak.push((kind.to_string(), to.unwrap_or(1))),
        }
    }

    /// Distinct nodes carrying flow -- a lower bound on realised slots.
    fn width(&self, psi: &[f64]) -> usize {
        let live = carries(psi, self.psi_total);
        if !live.iter().any(|&v| v) {
            return 0;
        }
        let mut nodes: Vec<i64> = Vec::new();
        for k in 0..psi.len() {
            if live[k] {
                nodes.push(self.g.tau[k]);
                nodes.push(self.g.sig[k]);
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    }

    fn add(&mut self, psi: Vec<f64>, label: String, kind: &str, certificate: bool,
           reason: &str) -> bool {
        let (psi, _) = cycles::cancel_cycles(&self.g.tau, &self.g.sig, &psi, 1e-12,
                                             self.g.n_nodes);
        // Before the dedup signature, so two candidates differing only by a
        // dust branch collapse into one instead of spending two verify slots.
        let (psi, _) = prune_dust(&self.g.tau, &self.g.sig, &psi, self.src, self.dst,
                                  DUST_SHARE, 1e-12);
        let active = psi.iter().filter(|&&v| v > 0.0).count();
        if active == 0 {
            return false;
        }
        let key = signature(&psi, 1e-12);
        if key.is_empty() || self.seen.contains(&key) {
            return false;
        }
        self.seen.push(key);
        let loss: f64 = (0..psi.len())
            .map(|k| self.g.eps[k] * psi[k])
            .sum::<f64>()
            + (0..psi.len())
                .map(|k| if self.g.g[k] > 0.0 { psi[k] * psi[k] / (2.0 * self.g.g[k]) } else { 0.0 })
                .sum::<f64>();
        self.out.candidates.push(Candidate::new(
            label, psi, certificate, reason, kind, active, loss,
        ));
        true
    }

    /// Re-solve, then repair pool conflicts rather than discarding them.
    ///
    /// Decision 3 allows a pool at most one arc per route, and the Laplacian
    /// knows nothing about that. Repairing in place -- keep one arc, forbid
    /// its siblings, solve again -- turns what would be a wasted candidate
    /// into a usable one, and every generator gets it for free.
    ///
    /// **Which arc to keep is a branch, not a guess.** Keeping the one
    /// carrying most is the right first try, but when the sibling it bans is
    /// the only thing joining src to dst in this restricted subgraph, the
    /// re-solve comes back "src not connected" and a candidate that had
    /// *already solved* is thrown away. Measured on crvUSD -> sDOLA at $2M:
    /// four candidates died exactly this way, and the router fell back to
    /// dumping the whole trade through one pool at 212% of its reserve, 706 bp
    /// behind the route the branch finds.
    ///
    /// So on an infeasible repair, put the bans back and keep the next arc
    /// down.
    fn resolve(&mut self, forbidden: Vec<bool>, label: String, kind: &str,
               pinned: &[(usize, f64)]) -> bool {
        let mut banned = forbidden;
        // (bans before the repair, arcs per conflicting pool, which one we kept)
        let mut undo: Option<(Vec<bool>, Vec<(String, Vec<usize>)>, usize)> = None;
        let mut solution: Option<Solution> = None;
        let mut settled = false;
        for _ in 0..REPAIR_ROUNDS {
            // One warm-started active-set solve, not column generation. The
            // base solve already priced out all m arcs, so a candidate is a
            // small perturbation of a known optimum.
            self.out.solves += 1;
            let options = Options {
                min_flow: MIN_FLOW_FRACTION * self.psi_total,
                // §11.1: gas cannot enter the objective without making the
                // program mixed-integer, but it bounds it from outside. An arc
                // carrying less value than its leg costs to execute cannot pay
                // for itself even if it were pure profit.
                gas_cost: self.opts.gas_floor,
                maxit: CANDIDATE_PIVOTS,
                partial_ok: true,
                ..Default::default()
            };
            let got = active_set_solve(
                &self.arc_view(), self.src, self.dst, self.psi_total,
                Some(&self.warm), Some(&banned), pinned, &options,
            );
            self.out.pivots += got.pivots as usize;
            if !got.stop.feasible() {
                let Some((before, ordered, rank)) = undo.take() else {
                    return false;
                };
                let rank = rank + 1;
                let deepest = ordered.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
                if rank >= deepest {
                    return false;
                }
                let mut restored = before.clone();
                keep_only(&mut restored, &ordered, rank, pinned);
                banned = restored;
                undo = Some((before, ordered, rank));
                continue;
            }
            let conflicts = conflicting_pools(
                self.arcs, &got.psi, 0.0, Some(&self.pools), Some(&mut self.elements),
            );
            if conflicts.is_empty() {
                solution = Some(got);
                settled = true;
                break;
            }
            let ordered = repair_order(&conflicts, &got.psi);
            let before = banned.clone();
            if !keep_only(&mut banned, &ordered, 0, pinned) {
                return false;
            }
            undo = Some((before, ordered, 0));
            solution = Some(got);
        }
        // Python's `for ... else`: falling out of the loop without breaking is
        // a repair that never settled, and the candidate is dropped.
        if !settled {
            return false;
        }
        let solution = solution.expect("a settled repair has a solution");
        // Two ways to be unrealisable, and both are known before realising:
        // more distinct tokens than the quoter has slots, or more arcs than
        // the caller will accept legs (each arc is at least one leg).
        let support = solution.psi.iter().filter(|&&v| v > 0.0).count();
        if self.width(&solution.psi) > self.opts.max_slots || support > self.opts.max_legs {
            // Solved, and unrealisable. Adding it would spend a realise and a
            // slot in the verification batch to learn what the node count
            // already said.
            self.out.skipped += 1;
            self.bump(kind, None);
            return false;
        }
        self.bump(kind, Some(0));
        self.add(solution.psi, label, kind, false, "RESTRICTED")
    }
}

/// Generate the ballot: every cheap re-solve worth putting to a quote.
pub fn generate(
    g: &ArcArrays,
    arcs: &[PoolArc],
    src: usize,
    dst: usize,
    psi_total: f64,
    base: &Solution,
    opts: &GenerateOptions,
    element_split: Option<ElementSplit<'_>>,
) -> CandidateSet {
    let pools = pool_of(arcs);
    let mut ballot = Generator {
        g, arcs, src, dst, psi_total, opts,
        out: CandidateSet::default(),
        seen: Vec::new(),
        streak: Vec::new(),
        pools: pools.clone(),
        elements: Vec::new(),
        warm: Vec::new(),
    };

    // 1. the relaxation itself
    ballot.add(base.psi.clone(), "C0 full".into(), "base", opts.base_certificate, "");

    let base_live = carries(&base.psi, psi_total);
    let base_active: Vec<usize> = (0..base.psi.len()).filter(|&k| base_live[k]).collect();
    // Warm-start from the *circulation-free* support. The raw optimum carries
    // flow on arcs that only exist to go round a negative-eps loop; they are
    // cancelled before execution anyway, and leaving them in the start set
    // makes every candidate re-solve churn through them again.
    let (acyclic, _) = cycles::cancel_cycles(&g.tau, &g.sig, &base.psi, 1e-12, g.n_nodes);
    let mut warm: Vec<bool> = acyclic.iter().map(|&v| v > 0.0).collect();
    if !warm.iter().any(|&v| v) {
        warm = base_live.clone();
    }
    ballot.warm = warm;

    // Per-family budgets. Ordering alone is not enough: with many flagged arcs
    // the pin sweep alone is 3 x 7 = 21 candidates, which used to consume the
    // whole budget before sparsification ran. Every un-sparsified candidate
    // inherits the relaxation's sprawl and blows the quoter's leg limit -- on
    // a 5M swap, *all twenty* came back too_long and the router fell back to a
    // single pool.
    let sparse_budget = 6.max((opts.max_candidates as f64 * 0.45) as usize);
    let pin_budget = 4.max((opts.max_candidates as f64 * 0.30) as usize);
    // Split the sparse budget between its two families. They answer different
    // questions -- a union of the cheapest *paths*, versus the *pools* the
    // relaxation actually put flow through -- and one shared counter let the
    // first starve the second. Measured on USDC->sUSDS 1M, the pool family's
    // winner was never generated and the route came out 48.8 bp short of it.
    let path_budget = 3.max(sparse_budget / 2);

    // 2. sparsification, over the k cheapest *paths* rather than the k largest
    //    arcs. Restricting to arbitrary arcs usually leaves src and dst
    //    disconnected and the re-solve is infeasible; a union of shortest
    //    paths is connected by construction. `k = 1` is §6.2's `C_*`.
    let mut made_paths = 0usize;
    let adjacency = build_adjacency(&g.tau, g.n_nodes);
    let want = opts.top_k.iter().copied().max().unwrap_or(6);
    let paths = k_shortest_paths(&g.tau, &g.sig, &g.eps, g.n_nodes, src, dst, want,
                                 &adjacency);
    // Yen's returns *near-duplicates*: the same route with one hop swapped, in
    // eps order. Taking them in that order makes each union differ from the
    // last by a single arc, so a budget of four buys four nested sets covering
    // the same handful of pools. Measured on USDC -> crvUSD at $5M, the 9th
    // path -- the one through 3pool that Curve's own router takes -- was never
    // on the ballot.
    let paths = by_new_pools(&paths, &pools);
    let levels = spread(&opts.top_k, path_budget);
    let mut union: Vec<usize> = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let k = index + 1;
        for &a in path {
            if !union.contains(&a) {
                union.push(a);
            }
        }
        if !levels.contains(&k) {
            continue;
        }
        let mut forbidden = vec![true; g.m()];
        for &at in &union {
            forbidden[at] = false;
        }
        let label = if k == 1 {
            "C* best single path".to_string()
        } else {
            format!("top {k} paths")
        };
        if ballot.resolve(forbidden, label, "sparse", &[]) {
            made_paths += 1;
        }
        if made_paths >= path_budget {
            break;
        }
    }

    // 2b. keep only the pools the relaxation liked best, but let the solver
    //     use any arc of those pools so it can still find a connected route.
    let mut order = base_active.clone();
    order.sort_by(|&a, &b| {
        base.psi[b].partial_cmp(&base.psi[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranked_pools: Vec<String> = Vec::new();
    for &k in &order {
        if !ranked_pools.contains(&pools[k]) {
            ranked_pools.push(pools[k].clone());
        }
    }
    let pool_budget = 3.max(sparse_budget.saturating_sub(made_paths));
    let mut made_pools = 0usize;
    for &k in &opts.top_k {
        if k >= ranked_pools.len() || made_pools >= pool_budget {
            continue;
        }
        let keep = &ranked_pools[..k];
        let forbidden: Vec<bool> = pools.iter().map(|p| !keep.contains(p)).collect();
        let label = format!("top {k} pool{}", if k > 1 { "s" } else { "" });
        if ballot.resolve(forbidden, label, "sparse", &[]) {
            made_pools += 1;
        }
    }

    ballot.out.skipped_wide = ballot.width(&base.psi);

    // 3. pin sweep on every active flagged arc (§6.3). Still ahead of the drop
    //    candidates, because no drop candidate can find a chord interior.
    //
    //    Arcs of a **re-entered pool** join the sweep, and this is the whole
    //    treatment reentry gets in the solver: two arcs of one pool are two
    //    independent resistors here, both calibrated at a state neither will
    //    see. So do what §6.3 already does for a chord -- stop trusting the
    //    model for the *allocation*, sweep it, and let a real quote adjudicate.
    let mut made = 0usize;
    let mut swept: Vec<usize> = base_active.iter().copied().filter(|&k| g.flagged[k]).collect();
    let mut live_pools: Vec<String> = base_active.iter().map(|&k| pools[k].clone()).collect();
    live_pools.sort();
    live_pools.dedup();
    // Every arc of a pool the route touches, not only the ones carrying flow:
    // the sweep wants the co-active ones, and the element generator below
    // wants the idle siblings too.
    let mut by_pool: Vec<(String, Vec<usize>)> = Vec::new();
    for k in 0..arcs.len() {
        if !live_pools.contains(&pools[k]) {
            continue;
        }
        match by_pool.iter_mut().find(|(p, _)| *p == pools[k]) {
            Some((_, group)) => group.push(k),
            None => by_pool.push((pools[k].clone(), vec![k])),
        }
    }
    let mut active_by_pool: Vec<(String, Vec<usize>)> = Vec::new();
    for &k in &base_active {
        match active_by_pool.iter_mut().find(|(p, _)| *p == pools[k]) {
            Some((_, group)) => group.push(k),
            None => active_by_pool.push((pools[k].clone(), vec![k])),
        }
    }
    for (_, shared) in &active_by_pool {
        if shared.len() > 1 {
            for &k in shared {
                if !swept.contains(&k) {
                    swept.push(k);
                }
            }
        }
    }
    swept.sort_unstable();
    for &arc_index in swept.iter().take(3) {
        let star = base.psi[arc_index];
        let mut stop = false;
        for &step in PIN_LADDER.iter() {
            let pin = (star * step).min(g.cap[arc_index]);
            if step > 0.0 && pin <= 0.0 {
                continue;
            }
            let label = format!(
                "pin {} x{}",
                truncate(&arcs[arc_index].note, 18),
                crate::pyfmt::general(step)
            );
            if ballot.resolve(vec![false; g.m()], label, "pin", &[(arc_index, pin)]) {
                made += 1;
            }
            if made >= pin_budget || ballot.out.len() >= opts.max_candidates
                || ballot.exhausted("pin")
            {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
    }

    // 3b. multi-port elements (docs/multi-port-elements.md, step 2).
    //
    //     Where one pool pays two ports out of one coin, the split between
    //     them is not something the model can rank. The sweep above brackets
    //     that; an element *solves* it, so its answer is pinned as one more
    //     candidate and ranked on measured output like everything else.
    //
    //     Pinned rather than forced: `resolve` takes these as `forced_upper`,
    //     so the solver may take less than the element asked for.
    //
    //     Pairs are proposed **speculatively**, not read off the base solve.
    //     Gating on "the solver already went through this pool twice" makes
    //     the generator unable to reach the case it was built for: gnosis
    //     WXDAI -> EURe runs 100% down one arm, so no pool is co-active, so no
    //     element is offered, so the second arm is never priced.
    if let Some(price) = element_split {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (_, shared) in &by_pool {
            let live: Vec<usize> =
                shared.iter().copied().filter(|k| base_active.contains(k)).collect();
            for &k1 in &live {
                for &k2 in shared {
                    if k2 != k1 && arcs[k1].tau == arcs[k2].tau {
                        pairs.push(if k1 < k2 { (k1, k2) } else { (k2, k1) });
                    }
                }
            }
        }
        let mut unique: Vec<(usize, usize)> = Vec::new();
        for pair in pairs {
            if !unique.contains(&pair) {
                unique.push(pair);
            }
        }
        for (k1, k2) in unique {
            if ballot.out.len() >= opts.max_candidates || ballot.exhausted("element") {
                break;
            }
            let Some((psi1, psi2)) = price(&arcs[k1], &arcs[k2], base.psi[k1], base.psi[k2])
            else {
                continue; // a pricer that cannot answer is not an error
            };
            if psi1 <= 0.0 || psi2 <= 0.0 {
                continue;
            }
            let share = psi1 / (psi1 + psi2);
            let label = format!(
                "element {} {}%",
                truncate(&arcs[k1].note, 16),
                py_round(share * 100.0) as i64
            );
            ballot.resolve(vec![false; g.m()], label, "element", &[(k1, psi1), (k2, psi2)]);
        }
    }

    // 4. one arc per pool (decision 3) -- keep the largest, forbid the rest
    let conflicts = conflicting_pools(
        arcs, &base.psi, psi_total, Some(&pools), Some(&mut ballot.elements),
    );
    if !conflicts.is_empty() {
        let mut forbidden = vec![false; g.m()];
        for (_, indices) in &conflicts {
            let keep_index = *indices
                .iter()
                .reduce(|a, b| if base.psi[*b] > base.psi[*a] { b } else { a })
                .unwrap();
            for &k in indices {
                if k != keep_index {
                    forbidden[k] = true;
                }
            }
        }
        let label = format!("repair {} pool conflict(s)", conflicts.len());
        ballot.resolve(forbidden, label, "repair", &[]);

        // There is no "re-enter this pool anyway" candidate any more, and none
        // is needed: `conflicting_pools` only reports a pool whose arcs are
        // not an admissible element, so a legal element was never a conflict.
        let worst = conflicts
            .iter()
            .reduce(|a, b| if b.1.len() > a.1.len() { b } else { a })
            .unwrap()
            .clone();
        let mut by_flow = worst.1.clone();
        by_flow.sort_by(|&a, &b| {
            base.psi[b].partial_cmp(&base.psi[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &keep_index in by_flow.iter().skip(1).take(1) {
            let mut alt = vec![false; g.m()];
            for &k in &worst.1 {
                if k != keep_index {
                    alt[k] = true;
                }
            }
            let label = format!("repair alt {}", truncate(&arcs[keep_index].note, 18));
            ballot.resolve(alt, label, "repair", &[]);
        }
    }

    // 5. drop each active arc in turn (§6.2)
    let room = opts.max_candidates.saturating_sub(ballot.out.len());
    for &k in order.iter().take(room) {
        let mut forbidden = vec![false; g.m()];
        forbidden[k] = true;
        let label = format!("drop {}", truncate(&arcs[k].note, 20));
        ballot.resolve(forbidden, label, "drop", &[]);
        if ballot.out.len() >= opts.max_candidates || ballot.exhausted("drop") {
            break;
        }
    }

    ballot.out.candidates.truncate(opts.max_candidates);
    ballot.out
}

/// `text[:n]` on characters, the way Python slices a `str`.
fn truncate(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_active_floor_is_relative_to_the_trade() {
        // 1e-18 of a unit trade is a pivot residue, not a decision.
        assert_eq!(carries(&[1e-18, 0.5], 1.0), vec![false, true]);
        // The same absolute number is a real branch of a tiny trade.
        assert_eq!(carries(&[1e-18, 1e-15], 1e-9), vec![true, true]);
    }

    #[test]
    fn the_ladder_is_sampled_across_its_whole_range() {
        // Taking `top_k` in order would spend a budget of four on 1, 2, 3, 4.
        assert_eq!(spread(&TOP_K, 4), vec![1, 3, 6, 12]);
        assert_eq!(spread(&TOP_K, 1), TOP_K.to_vec());
        assert_eq!(spread(&TOP_K, 99), TOP_K.to_vec());
    }

    #[test]
    fn keep_only_clamps_a_rank_past_the_end() {
        let ordered = vec![("0xp".to_string(), vec![0usize, 1])];
        let mut banned = vec![false; 3];
        assert!(keep_only(&mut banned, &ordered, 9, &[]));
        // Rank 9 clamps to the last, so arc 0 is banned and arc 1 kept.
        assert_eq!(banned, vec![true, false, false]);
        // Nothing new to ban: the caller must stop rather than loop.
        assert!(!keep_only(&mut banned, &ordered, 9, &[]));
    }

    #[test]
    fn a_pinned_arc_is_never_banned() {
        let ordered = vec![("0xp".to_string(), vec![0usize, 1])];
        let mut banned = vec![false; 2];
        assert!(!keep_only(&mut banned, &ordered, 0, &[(1, 1.0)]));
        assert_eq!(banned, vec![false, false]);
    }

    #[test]
    fn paths_are_ordered_by_the_pools_they_bring() {
        let pools: Vec<String> = ["a", "a", "b", "c"].iter().map(|s| s.to_string()).collect();
        // Path 0 is cheapest and stays first; of the rest, the one adding the
        // most unseen pools comes next.
        let paths = vec![vec![0usize], vec![1], vec![2, 3]];
        assert_eq!(by_new_pools(&paths, &pools), vec![vec![0], vec![2, 3], vec![1]]);
    }

    #[test]
    fn gas_bounds_the_program_from_outside() {
        // The floor is sound rather than heuristic: a leg carrying less than
        // it costs cannot pay for itself.
        let floor = crate::gas::min_useful_flow(1_000_000_000, 3000.0, crate::types::ArcKind::SwapStable);
        assert!(floor > 0.0);
    }
}
