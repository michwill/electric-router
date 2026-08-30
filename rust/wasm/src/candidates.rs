//! The ballot, in the browser.
//!
//! The twin of `src/candidates_py.rs`. Two shape differences, both forced by
//! what a typed array can carry: a pair crosses as two parallel arrays, and
//! the element pricer is a `js_sys::Function` rather than a callable.

use erouter_solve::candidates::{self, CandidateSet, GenerateOptions};
use erouter_solve::gas::GasTable;
use erouter_solve::risk::RiskTable;
use erouter_solve::solve::{Solution, Stop};
use erouter_solve::types::{ArcKind, PoolArc};
use erouter_solve::verify::{self, VerifyOptions};
use wasm_bindgen::prelude::*;

use crate::graph::Graph;
use crate::nodes::NodeMap;
use crate::realize::Arcs;

fn kind_of(code: u8) -> Result<ArcKind, JsValue> {
    ArcKind::from_code(code).ok_or_else(|| JsError::new(&format!("no such kind: {code}")).into())
}

/// The gas and risk tables, together.
#[wasm_bindgen]
#[derive(Default)]
pub struct Tables {
    pub(crate) gas: GasTable,
    pub(crate) risk: RiskTable,
    /// Whether the risk term applies at all. The reference takes
    /// `risk_table=None` to mean "do not price survival", which is not the
    /// same as a table with no entries.
    pub(crate) risk_on: bool,
}

#[wasm_bindgen]
impl Tables {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Gas measured for one direction of one pool, or `(-1, -1)` for the pool.
    #[wasm_bindgen(js_name = setLegGas)]
    pub fn set_leg_gas(
        &mut self, target: &str, kind: u8, i: i32, j: i32, gas: i64,
    ) -> Result<(), JsValue> {
        self.gas.set_leg(target, kind_of(kind)?, i, j, gas);
        Ok(())
    }

    /// The measured median for a kind -- what a wholly new pool is priced at.
    #[wasm_bindgen(js_name = setKindGas)]
    pub fn set_kind_gas(&mut self, kind: u8, gas: i64) -> Result<(), JsValue> {
        self.gas.set_kind(kind_of(kind)?, gas);
        Ok(())
    }

    /// P(this leg's minimum-out trips before inclusion).
    #[wasm_bindgen(js_name = setRisk)]
    pub fn set_risk(&mut self, target: &str, i: i32, j: i32, risk: f64) {
        self.risk.set(target, i, j, risk);
        self.risk_on = true;
    }

    /// What an arc nobody has measured is charged. Turns the risk term on.
    #[wasm_bindgen(setter, js_name = defaultRisk)]
    pub fn set_default_risk(&mut self, risk: f64) {
        self.risk.default = risk;
        self.risk_on = true;
    }

    /// Price survival with an empty table, as distinct from leaving the term
    /// out.
    #[wasm_bindgen(js_name = enableRisk)]
    pub fn enable_risk(&mut self) {
        self.risk_on = true;
    }

    #[wasm_bindgen(js_name = gasOf)]
    pub fn gas_of(&self, kind: u8, target: &str, i: i32, j: i32) -> Result<i64, JsValue> {
        Ok(self.gas.gas(kind_of(kind)?, target, i, j))
    }

    #[wasm_bindgen(js_name = riskOf)]
    pub fn risk_of(&self, kind: u8, target: &str, i: i32, j: i32) -> Result<f64, JsValue> {
        Ok(self.risk.of(kind_of(kind)?, target, i, j))
    }
}

/// The candidates one generation produced, and what ranking made of them.
#[wasm_bindgen]
pub struct Ballot {
    inner: CandidateSet,
}

#[wasm_bindgen]
impl Ballot {
    /// Generate the ballot: every cheap re-solve worth putting to a quote.
    ///
    /// `elementSplit(a, b, psiA, psiB)` is handed the two arcs' *indices* and
    /// returns `[psiA, psiB]`, or anything falsy to decline. It lives with the
    /// caller because pricing needs a pool model this crate does not hold.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        graph: &Graph, arcs: &Arcs, src: usize, dst: usize, psi_total: f64,
        base_psi: Vec<f64>, base_certificate: Option<bool>,
        max_candidates: Option<usize>, top_k: Option<Vec<usize>>,
        gas_floor: Option<f64>, max_legs: Option<usize>, max_slots: Option<usize>,
        element_split: Option<js_sys::Function>,
    ) -> Ballot {
        let opts = GenerateOptions {
            base_certificate: base_certificate.unwrap_or(false),
            max_candidates: max_candidates.unwrap_or(20),
            top_k: top_k.unwrap_or_else(|| candidates::TOP_K.to_vec()),
            gas_floor: gas_floor.unwrap_or(0.0),
            max_legs: max_legs.unwrap_or(32),
            max_slots: max_slots.unwrap_or(8),
        };
        let base = Solution { psi: base_psi, ..empty_solution() };
        let members = &arcs.inner;
        // A pricer that throws is not an error: the reference swallows it and
        // moves on, because a model that cannot answer has nothing to say
        // about this pair.
        let call = |a: &PoolArc, b: &PoolArc, pa: f64, pb: f64| -> Option<(f64, f64)> {
            let handler = element_split.as_ref()?;
            let got = handler
                .apply(
                    &JsValue::NULL,
                    &js_sys::Array::of4(
                        &JsValue::from_f64(index_of(members, a) as f64),
                        &JsValue::from_f64(index_of(members, b) as f64),
                        &JsValue::from_f64(pa),
                        &JsValue::from_f64(pb),
                    ),
                )
                .ok()?;
            let pair = js_sys::Array::from(&got);
            if pair.length() != 2 {
                return None;
            }
            Some((pair.get(0).as_f64()?, pair.get(1).as_f64()?))
        };
        let split: Option<candidates::ElementSplit<'_>> =
            if element_split.is_some() { Some(&call) } else { None };
        let inner = candidates::generate(
            &graph.inner, members, src, dst, psi_total, &base, &opts, split,
        );
        Ballot { inner }
    }

    #[wasm_bindgen(js_name = length)]
    pub fn len(&self) -> usize {
        self.inner.candidates.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.candidates.is_empty()
    }

    // -- the pieces generation is built from --------------------------------

    /// Arcs the solve actually routed through.
    pub fn carries(psi: Vec<f64>, psi_total: f64) -> Vec<u8> {
        candidates::carries(&psi, psi_total).into_iter().map(u8::from).collect()
    }

    /// The `k` levels to spend `budget` candidates on, across the ladder.
    pub fn spread(top_k: Vec<usize>, budget: usize) -> Vec<u32> {
        candidates::spread(&top_k, budget).into_iter().map(|v| v as u32).collect()
    }

    /// Yen's algorithm over `eps`. Paths cross flat, with `pathSpans` bounding
    /// each one.
    #[wasm_bindgen(js_name = kShortestPaths)]
    pub fn k_shortest_paths(graph: &Graph, src: usize, dst: usize, k: usize) -> Vec<u32> {
        flatten(&Self::paths_of(graph, src, dst, k))
    }

    #[wasm_bindgen(js_name = kShortestPathSpans)]
    pub fn k_shortest_path_spans(
        graph: &Graph, src: usize, dst: usize, k: usize,
    ) -> Vec<u32> {
        spans(&Self::paths_of(graph, src, dst, k))
    }

    /// Re-order paths so each one brings pools the earlier ones did not.
    #[wasm_bindgen(js_name = byNewPools)]
    pub fn by_new_pools(
        paths: Vec<u32>, path_spans: Vec<u32>, pools: Vec<String>,
    ) -> Vec<u32> {
        flatten(&candidates::by_new_pools(&unflatten(&paths, &path_spans), &pools))
    }

    #[wasm_bindgen(js_name = byNewPoolsSpans)]
    pub fn by_new_pools_spans(
        paths: Vec<u32>, path_spans: Vec<u32>, pools: Vec<String>,
    ) -> Vec<u32> {
        spans(&candidates::by_new_pools(&unflatten(&paths, &path_spans), &pools))
    }

    /// Pools carrying flow on more than one arc whose arcs are not one
    /// element. The addresses come back here and the indices in
    /// `conflictingArcs`.
    #[wasm_bindgen(js_name = conflictingPools)]
    pub fn conflicting_pools(
        arcs: &Arcs, psi: Vec<f64>, psi_total: Option<f64>,
    ) -> Result<Vec<String>, JsValue> {
        Ok(Self::conflicts(arcs, &psi, psi_total)?.into_iter().map(|(p, _)| p).collect())
    }

    #[wasm_bindgen(js_name = conflictingArcs)]
    pub fn conflicting_arcs(
        arcs: &Arcs, psi: Vec<f64>, psi_total: Option<f64>,
    ) -> Result<Vec<u32>, JsValue> {
        let found = Self::conflicts(arcs, &psi, psi_total)?;
        Ok(flatten(&found.into_iter().map(|(_, v)| v).collect::<Vec<_>>()))
    }

    #[wasm_bindgen(js_name = conflictingSpans)]
    pub fn conflicting_spans(
        arcs: &Arcs, psi: Vec<f64>, psi_total: Option<f64>,
    ) -> Result<Vec<u32>, JsValue> {
        let found = Self::conflicts(arcs, &psi, psi_total)?;
        Ok(spans(&found.into_iter().map(|(_, v)| v).collect::<Vec<_>>()))
    }

    /// Each conflicting pool's arcs, the one carrying most first.
    #[wasm_bindgen(js_name = repairOrder)]
    pub fn repair_order(
        pools: Vec<String>, arcs: Vec<u32>, arc_spans: Vec<u32>, psi: Vec<f64>,
    ) -> Vec<u32> {
        let groups = named(&pools, &unflatten(&arcs, &arc_spans));
        flatten(
            &candidates::repair_order(&groups, &psi)
                .into_iter()
                .map(|(_, v)| v)
                .collect::<Vec<_>>(),
        )
    }

    /// Ban every arc of each conflicting pool but the one at `rank`. Returns
    /// the mask; `keepOnlyApplied` says whether anything was newly banned.
    #[wasm_bindgen(js_name = keepOnly)]
    pub fn keep_only(
        banned: Vec<u8>, pools: Vec<String>, arcs: Vec<u32>, arc_spans: Vec<u32>,
        rank: usize, pinned: Option<Vec<u32>>,
    ) -> Vec<u8> {
        Self::keep_only_inner(banned, pools, arcs, arc_spans, rank, pinned).0
    }

    #[wasm_bindgen(js_name = keepOnlyApplied)]
    pub fn keep_only_applied(
        banned: Vec<u8>, pools: Vec<String>, arcs: Vec<u32>, arc_spans: Vec<u32>,
        rank: usize, pinned: Option<Vec<u32>>,
    ) -> bool {
        Self::keep_only_inner(banned, pools, arcs, arc_spans, rank, pinned).1
    }

    // -- what generation produced -----------------------------------------

    pub fn labels(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.label.clone()).collect()
    }

    pub fn kinds(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.kind.clone()).collect()
    }

    pub fn reasons(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.reason.clone()).collect()
    }

    pub fn certificates(&self) -> Vec<u8> {
        self.inner.candidates.iter().map(|c| u8::from(c.certificate)).collect()
    }

    #[wasm_bindgen(js_name = nArcs)]
    pub fn n_arcs(&self) -> Vec<u32> {
        self.inner.candidates.iter().map(|c| c.n_arcs as u32).collect()
    }

    #[wasm_bindgen(js_name = modelledLoss)]
    pub fn modelled_loss(&self) -> Vec<f64> {
        self.inner.candidates.iter().map(|c| c.modelled_loss).collect()
    }

    /// One candidate's flow, over the graph's arc index space.
    pub fn psi(&self, at: usize) -> Vec<f64> {
        self.inner.candidates.get(at).map(|c| c.psi.clone()).unwrap_or_default()
    }

    pub fn statuses(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.status.clone()).collect()
    }

    pub fn notes(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.note.clone()).collect()
    }

    /// `0` where the candidate has no rank, which is what `None` means here.
    pub fn ranks(&self) -> Vec<u32> {
        self.inner.candidates.iter().map(|c| c.rank.unwrap_or(0) as u32).collect()
    }

    pub fn gas(&self) -> Vec<f64> {
        self.inner.candidates.iter().map(|c| c.gas as f64).collect()
    }

    pub fn survival(&self) -> Vec<f64> {
        self.inner.candidates.iter().map(|c| c.survival).collect()
    }

    /// `-1` where nothing has been quoted.
    #[wasm_bindgen(js_name = verifiedOut)]
    pub fn verified_out(&self) -> Vec<String> {
        self.inner
            .candidates
            .iter()
            .map(|c| c.verified_out.map_or("-1".to_string(), |v| v.to_string()))
            .collect()
    }

    pub fn legs(&self) -> Vec<u32> {
        self.inner
            .candidates
            .iter()
            .map(|c| c.route.as_ref().map_or(0, |r| r.legs.len()) as u32)
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn solves(&self) -> usize {
        self.inner.solves
    }

    #[wasm_bindgen(getter)]
    pub fn pivots(&self) -> usize {
        self.inner.pivots
    }

    #[wasm_bindgen(getter)]
    pub fn skipped(&self) -> usize {
        self.inner.skipped
    }

    #[wasm_bindgen(getter, js_name = skippedWide)]
    pub fn skipped_wide(&self) -> usize {
        self.inner.skipped_wide
    }

    /// The winner's index, or `undefined`.
    pub fn best(&self) -> Option<usize> {
        let target = self.inner.best()?;
        self.inner.candidates.iter().position(|c| std::ptr::eq(c, target))
    }

    // -- realisation and ranking ------------------------------------------

    /// Turn each candidate's flow into legs, marking the ones that cannot be.
    #[wasm_bindgen(js_name = realizeCandidates)]
    #[allow(clippy::too_many_arguments)]
    pub fn realize_candidates(
        &mut self, arcs: &Arcs, nu: Vec<f64>, nodes: &NodeMap, src_token: &str,
        dst_token: &str, amount_in: &str, potentials: Option<Vec<f64>>,
        max_legs: Option<usize>, max_slots: Option<usize>,
    ) -> Result<(), JsValue> {
        let amount = amount_in
            .parse()
            .map_err(|_| JsError::new(&format!("not a u256: {amount_in}")))?;
        verify::realize_candidates(
            &mut self.inner, &arcs.inner, &nu, &nodes.inner, src_token, dst_token,
            amount, potentials.as_deref(), max_legs.unwrap_or(32),
            max_slots.unwrap_or(8),
        );
        Ok(())
    }

    /// Which candidates need quoting, in the order the answers must come back.
    pub fn ready(&self) -> Vec<u32> {
        verify::ready(&self.inner).into_iter().map(|v| v as u32).collect()
    }

    /// Fold one batch of quotes back in, then rank everything. `at` and
    /// `quotes` are parallel; a quote of `"0"` is a revert.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &mut self, at: Vec<u32>, quotes: Vec<String>, tables: Option<Tables>,
        gas_price_wei: Option<f64>, dst_wei_per_eth: Option<f64>,
        revert_cost_bp: Option<f64>, leg_cost_bp: Option<f64>,
    ) -> Result<(), JsValue> {
        if at.len() != quotes.len() {
            return Err(JsError::new("at and quotes must be the same length").into());
        }
        let mut folded = Vec::with_capacity(at.len());
        for (k, value) in at.iter().zip(quotes.iter()) {
            let parsed = value
                .parse::<u128>()
                .map_err(|_| JsError::new(&format!("not a u128: {value}")))?;
            folded.push((*k as usize, parsed));
        }
        let fallback = Tables::default();
        let held = tables.unwrap_or(fallback);
        let opts = VerifyOptions {
            gas_price_wei: gas_price_wei.unwrap_or(0.0) as i64,
            dst_wei_per_eth: dst_wei_per_eth.unwrap_or(0.0),
            gas_table: &held.gas,
            risk_table: held.risk_on.then_some(&held.risk),
            revert_cost_bp: revert_cost_bp.unwrap_or(erouter_solve::risk::REVERT_COST_BP),
            leg_cost_bp: leg_cost_bp.unwrap_or(verify::LEG_COST_BP),
        };
        verify::verify(&mut self.inner, &folded, &opts);
        Ok(())
    }

    /// One candidate's realised legs: the targets, with the numbers in
    /// `wireNumbers`.
    #[wasm_bindgen(js_name = wireLegs)]
    pub fn wire_legs(&self, at: usize) -> Vec<String> {
        match self.inner.candidates.get(at).and_then(|c| c.route.as_ref()) {
            Some(route) => route.wire_legs().iter().map(|leg| leg.target.clone()).collect(),
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = wireNumbers)]
    pub fn wire_numbers(&self, at: usize) -> Vec<i32> {
        match self.inner.candidates.get(at).and_then(|c| c.route.as_ref()) {
            Some(route) => route
                .wire_legs()
                .iter()
                .flat_map(|leg| {
                    [leg.kind.code() as i32, leg.i, leg.j, leg.n,
                     leg.src_slot, leg.dst_slot, leg.bps]
                })
                .collect(),
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = dstSlot)]
    pub fn dst_slot(&self, at: usize) -> usize {
        self.inner
            .candidates
            .get(at)
            .and_then(|c| c.route.as_ref())
            .map_or(0, |r| r.dst_slot)
    }
}

impl Ballot {
    fn paths_of(graph: &Graph, src: usize, dst: usize, k: usize) -> Vec<Vec<usize>> {
        let g = &graph.inner;
        let adjacency = erouter_solve::seed::build_adjacency(&g.tau, g.n_nodes);
        erouter_solve::seed::k_shortest_paths(
            &g.tau, &g.sig, &g.eps, g.n_nodes, src, dst, k, &adjacency,
        )
    }

    fn conflicts(
        arcs: &Arcs, psi: &[f64], psi_total: Option<f64>,
    ) -> Result<Vec<(String, Vec<usize>)>, JsValue> {
        if psi.len() != arcs.inner.len() {
            return Err(JsError::new(&format!(
                "psi has {} entries and there are {} arcs",
                psi.len(),
                arcs.inner.len()
            ))
            .into());
        }
        Ok(candidates::conflicting_pools(
            &arcs.inner, psi, psi_total.unwrap_or(0.0), None, None,
        ))
    }

    fn keep_only_inner(
        banned: Vec<u8>, pools: Vec<String>, arcs: Vec<u32>, arc_spans: Vec<u32>,
        rank: usize, pinned: Option<Vec<u32>>,
    ) -> (Vec<u8>, bool) {
        let mut mask: Vec<bool> = banned.into_iter().map(|b| b != 0).collect();
        let groups = named(&pools, &unflatten(&arcs, &arc_spans));
        let pins: Vec<(usize, f64)> = pinned
            .unwrap_or_default()
            .into_iter()
            .map(|k| (k as usize, 0.0))
            .collect();
        let applied = candidates::keep_only(&mut mask, &groups, rank, &pins);
        (mask.into_iter().map(u8::from).collect(), applied)
    }
}

fn flatten(groups: &[Vec<usize>]) -> Vec<u32> {
    groups.iter().flat_map(|g| g.iter().map(|&v| v as u32)).collect()
}

fn spans(groups: &[Vec<usize>]) -> Vec<u32> {
    let mut out = vec![0u32];
    let mut total = 0u32;
    for group in groups {
        total += group.len() as u32;
        out.push(total);
    }
    out
}

fn unflatten(flat: &[u32], spans: &[u32]) -> Vec<Vec<usize>> {
    (0..spans.len().saturating_sub(1))
        .map(|k| {
            flat[spans[k] as usize..spans[k + 1] as usize]
                .iter()
                .map(|&v| v as usize)
                .collect()
        })
        .collect()
}

fn named(pools: &[String], groups: &[Vec<usize>]) -> Vec<(String, Vec<usize>)> {
    pools.iter().cloned().zip(groups.iter().cloned()).collect()
}

/// A `Solution` carrying nothing but the flow, which is all `generate` reads
/// of the base solve.
fn empty_solution() -> Solution {
    Solution {
        psi: Vec::new(),
        u: Vec::new(),
        active: Vec::new(),
        upper: Vec::new(),
        psi_upper: Vec::new(),
        rho: Vec::new(),
        pivots: 0,
        stop: Stop::Optimal,
        chol_failures: 0,
        keep_changes: 0,
        refits: 0,
        timings: [0; 7],
    }
}

/// Which arc this is, so the callback is handed an index rather than a copy of
/// thirty fields.
fn index_of(arcs: &[PoolArc], arc: &PoolArc) -> usize {
    arcs.iter().position(|a| std::ptr::eq(a, arc)).unwrap_or(0)
}
