//! The ballot, across the PyO3 boundary.
//!
//! Generation and ranking are one class because they are one object: `verify`
//! writes `status`, `rank`, `gas` and `survival` back onto the candidates
//! `generate` produced. Splitting them would mean handing the whole set across
//! twice.
//!
//! The quotes are an argument rather than a call. Chain I/O is the host's --
//! see `verify.rs` -- so `ready()` says which candidates need pricing and
//! `verify()` takes what came back.

use crate::candidates::{self, CandidateSet, GenerateOptions};
use crate::gas::GasTable;
use crate::graph_py::Graph;
use crate::nodes_py::NodeMap;
use crate::realize_py::Arcs;
use crate::risk::RiskTable;
use crate::solve::Solution;
use crate::types::PoolArc;
use crate::verify::{self, VerifyOptions};
use pyo3::prelude::*;

/// The gas and risk tables, together: `verify` needs both or neither, and a
/// caller with no measurements passes an empty one rather than `None`.
#[pyclass]
#[derive(Default)]
pub struct Tables {
    pub(crate) gas: GasTable,
    pub(crate) risk: RiskTable,
    /// Whether the risk term applies at all. The reference takes
    /// `risk_table=None` to mean "do not price survival", which is not the
    /// same as a table with no entries -- that one charges every arc the
    /// default.
    pub(crate) risk_on: bool,
}

#[pymethods]
impl Tables {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Gas measured for one direction of one pool, or `(-1, -1)` for the pool.
    fn set_leg_gas(&mut self, target: &str, kind: u8, i: i32, j: i32, gas: i64) -> PyResult<()> {
        self.gas.set_leg(target, kind_of(kind)?, i, j, gas);
        Ok(())
    }

    /// The measured median for a kind -- what a wholly new pool is priced at.
    fn set_kind_gas(&mut self, kind: u8, gas: i64) -> PyResult<()> {
        self.gas.set_kind(kind_of(kind)?, gas);
        Ok(())
    }

    /// P(this leg's minimum-out trips before inclusion).
    fn set_risk(&mut self, target: &str, i: i32, j: i32, risk: f64) {
        self.risk.set(target, i, j, risk);
        self.risk_on = true;
    }

    /// What an arc nobody has measured is charged. Turns the risk term on.
    #[setter]
    fn set_default_risk(&mut self, risk: f64) {
        self.risk.default = risk;
        self.risk_on = true;
    }

    /// Price survival with an empty table, which charges every pool arc the
    /// default -- as distinct from leaving the term out.
    fn enable_risk(&mut self) {
        self.risk_on = true;
    }

    fn gas_of(&self, kind: u8, target: &str, i: i32, j: i32) -> PyResult<i64> {
        Ok(self.gas.gas(kind_of(kind)?, target, i, j))
    }

    fn risk_of(&self, kind: u8, target: &str, i: i32, j: i32) -> PyResult<f64> {
        Ok(self.risk.of(kind_of(kind)?, target, i, j))
    }
}

fn kind_of(code: u8) -> PyResult<crate::types::ArcKind> {
    crate::types::ArcKind::from_code(code)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no such kind: {code}")))
}

/// The candidates one generation produced, and what ranking made of them.
#[pyclass]
pub struct Ballot {
    inner: CandidateSet,
}

#[pymethods]
impl Ballot {
    /// Generate the ballot: every cheap re-solve worth putting to a quote.
    ///
    /// `element_split(arc_a, arc_b, psi_a, psi_b) -> (psi_a, psi_b) | None`
    /// prices a two-port element. It lives with the caller because pricing
    /// needs a pool model and this module holds `a` and `B`, not a
    /// `StableSwap`. Absent, that family does not run.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (graph, arcs, src, dst, psi_total, base_psi, *,
                        base_certificate=false, max_candidates=20, top_k=None,
                        gas_floor=0.0, max_legs=32, max_slots=8,
                        element_split=None))]
    fn generate(
        graph: PyRef<'_, Graph>, arcs: PyRef<'_, Arcs>, src: usize, dst: usize,
        psi_total: f64, base_psi: Vec<f64>, base_certificate: bool,
        max_candidates: usize, top_k: Option<Vec<usize>>, gas_floor: f64,
        max_legs: usize, max_slots: usize, element_split: Option<Py<PyAny>>,
    ) -> PyResult<Ballot> {
        let opts = GenerateOptions {
            base_certificate,
            max_candidates,
            top_k: top_k.unwrap_or_else(|| candidates::TOP_K.to_vec()),
            gas_floor,
            max_legs,
            max_slots,
        };
        let base = Solution { psi: base_psi, ..empty_solution() };
        let members = &arcs.inner;
        // The callback runs under the GIL the caller already holds, and a
        // pricer that raises is not an error -- the reference swallows it and
        // moves on, because a model that cannot answer is a model that has
        // nothing to say about this pair.
        let call = |a: &PoolArc, b: &PoolArc, pa: f64, pb: f64| -> Option<(f64, f64)> {
            let handler = element_split.as_ref()?;
            Python::with_gil(|py| {
                let got = handler.bind(py).call1((
                    index_of(members, a), index_of(members, b), pa, pb,
                ));
                match got {
                    Ok(value) if !value.is_none() => value.extract::<(f64, f64)>().ok(),
                    _ => None,
                }
            })
        };
        let split: Option<candidates::ElementSplit<'_>> =
            if element_split.is_some() { Some(&call) } else { None };
        let inner = candidates::generate(
            &graph.inner, members, src, dst, psi_total, &base, &opts, split,
        );
        Ok(Ballot { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.candidates.len()
    }

    // -- the pieces generation is built from --------------------------------
    //
    // Exposed because they are what a divergence localises to. `generate`
    // holds the cache and the budget, so a mismatch in the ballot says only
    // "somewhere in here"; these say where.

    /// Arcs the solve actually routed through.
    #[staticmethod]
    fn carries(psi: Vec<f64>, psi_total: f64) -> Vec<bool> {
        candidates::carries(&psi, psi_total)
    }

    /// The `k` levels to spend `budget` candidates on, across the ladder.
    #[staticmethod]
    fn spread(top_k: Vec<usize>, budget: usize) -> Vec<usize> {
        candidates::spread(&top_k, budget)
    }

    /// Yen's algorithm over `eps`, returning arc-index paths.
    #[staticmethod]
    fn k_shortest_paths(
        graph: PyRef<'_, Graph>, src: usize, dst: usize, k: usize,
    ) -> Vec<Vec<usize>> {
        let g = &graph.inner;
        let adjacency = crate::seed::build_adjacency(&g.tau, g.n_nodes);
        crate::seed::k_shortest_paths(&g.tau, &g.sig, &g.eps, g.n_nodes, src, dst, k,
                                      &adjacency)
    }

    /// Re-order paths so each one brings pools the earlier ones did not.
    #[staticmethod]
    fn by_new_pools(paths: Vec<Vec<usize>>, pools: Vec<String>) -> Vec<Vec<usize>> {
        candidates::by_new_pools(&paths, &pools)
    }

    /// Pools carrying flow on more than one arc whose arcs are not one element.
    #[staticmethod]
    #[pyo3(signature = (arcs, psi, psi_total=0.0))]
    fn conflicting_pools(
        arcs: PyRef<'_, Arcs>, psi: Vec<f64>, psi_total: f64,
    ) -> PyResult<Vec<(String, Vec<usize>)>> {
        if psi.len() != arcs.inner.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "psi has {} entries and there are {} arcs",
                psi.len(), arcs.inner.len()
            )));
        }
        Ok(candidates::conflicting_pools(&arcs.inner, &psi, psi_total, None, None))
    }

    /// Each conflicting pool's arcs, the one carrying most first.
    #[staticmethod]
    fn repair_order(
        conflicts: Vec<(String, Vec<usize>)>, psi: Vec<f64>,
    ) -> Vec<(String, Vec<usize>)> {
        candidates::repair_order(&conflicts, &psi)
    }

    /// Ban every arc of each conflicting pool but the one at `rank`.
    #[staticmethod]
    #[pyo3(signature = (banned, ordered, rank, pinned=None))]
    fn keep_only(
        banned: Vec<bool>, ordered: Vec<(String, Vec<usize>)>, rank: usize,
        pinned: Option<Vec<usize>>,
    ) -> (Vec<bool>, bool) {
        let mut banned = banned;
        let pins: Vec<(usize, f64)> =
            pinned.unwrap_or_default().into_iter().map(|k| (k, 0.0)).collect();
        let applied = candidates::keep_only(&mut banned, &ordered, rank, &pins);
        (banned, applied)
    }

    // -- what generation produced -----------------------------------------

    fn labels(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.label.clone()).collect()
    }

    fn kinds(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.kind.clone()).collect()
    }

    fn reasons(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.reason.clone()).collect()
    }

    fn certificates(&self) -> Vec<bool> {
        self.inner.candidates.iter().map(|c| c.certificate).collect()
    }

    fn n_arcs(&self) -> Vec<usize> {
        self.inner.candidates.iter().map(|c| c.n_arcs).collect()
    }

    fn modelled_loss(&self) -> Vec<f64> {
        self.inner.candidates.iter().map(|c| c.modelled_loss).collect()
    }

    /// One candidate's flow, over the graph's arc index space.
    fn psi(&self, at: usize) -> Vec<f64> {
        self.inner.candidates.get(at).map(|c| c.psi.clone()).unwrap_or_default()
    }

    fn statuses(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.status.clone()).collect()
    }

    fn notes(&self) -> Vec<String> {
        self.inner.candidates.iter().map(|c| c.note.clone()).collect()
    }

    /// `0` where the candidate has no rank, which is what `None` means here.
    fn ranks(&self) -> Vec<usize> {
        self.inner.candidates.iter().map(|c| c.rank.unwrap_or(0)).collect()
    }

    fn gas(&self) -> Vec<i64> {
        self.inner.candidates.iter().map(|c| c.gas).collect()
    }

    fn survival(&self) -> Vec<f64> {
        self.inner.candidates.iter().map(|c| c.survival).collect()
    }

    /// `-1` where nothing has been quoted.
    fn verified_out(&self) -> Vec<String> {
        self.inner
            .candidates
            .iter()
            .map(|c| c.verified_out.map_or("-1".to_string(), |v| v.to_string()))
            .collect()
    }

    fn legs(&self) -> Vec<usize> {
        self.inner
            .candidates
            .iter()
            .map(|c| c.route.as_ref().map_or(0, |r| r.legs.len()))
            .collect()
    }

    #[getter]
    fn solves(&self) -> usize {
        self.inner.solves
    }

    #[getter]
    fn pivots(&self) -> usize {
        self.inner.pivots
    }

    #[getter]
    fn skipped(&self) -> usize {
        self.inner.skipped
    }

    #[getter]
    fn skipped_wide(&self) -> usize {
        self.inner.skipped_wide
    }

    /// The winner's index, or `None`.
    fn best(&self) -> Option<usize> {
        let target = self.inner.best()?;
        self.inner.candidates.iter().position(|c| std::ptr::eq(c, target))
    }

    // -- realisation and ranking ------------------------------------------

    /// Turn each candidate's flow into legs, marking the ones that cannot be.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (arcs, nu, nodes, src_token, dst_token, amount_in, *,
                        potentials=None, max_legs=32, max_slots=8))]
    fn realize_candidates(
        &mut self, arcs: PyRef<'_, Arcs>, nu: Vec<f64>, nodes: PyRef<'_, NodeMap>,
        src_token: &str, dst_token: &str, amount_in: &str,
        potentials: Option<Vec<f64>>, max_legs: usize, max_slots: usize,
    ) -> PyResult<()> {
        let amount = amount_in.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("not a u256: {amount_in}"))
        })?;
        verify::realize_candidates(
            &mut self.inner, &arcs.inner, &nu, &nodes.inner, src_token, dst_token,
            amount, potentials.as_deref(), max_legs, max_slots,
        );
        Ok(())
    }

    /// Which candidates need quoting, in the order the answers must come back.
    fn ready(&self) -> Vec<usize> {
        verify::ready(&self.inner)
    }

    /// Fold one batch of quotes back in, then rank everything.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (quotes, *, tables=None, gas_price_wei=0, dst_wei_per_eth=0.0,
                        revert_cost_bp=crate::risk::REVERT_COST_BP,
                        leg_cost_bp=verify::LEG_COST_BP))]
    fn verify(
        &mut self, quotes: Vec<(usize, u128)>, tables: Option<PyRef<'_, Tables>>,
        gas_price_wei: i64, dst_wei_per_eth: f64, revert_cost_bp: f64,
        leg_cost_bp: f64,
    ) {
        let fallback = GasTable::new();
        let gas = tables.as_ref().map_or(&fallback, |t| &t.gas);
        let risk = tables.as_ref().and_then(|t| t.risk_on.then_some(&t.risk));
        let opts = VerifyOptions {
            gas_price_wei,
            dst_wei_per_eth,
            gas_table: gas,
            risk_table: risk,
            revert_cost_bp,
            leg_cost_bp,
        };
        verify::verify(&mut self.inner, &quotes, &opts);
    }

    /// One candidate's realised legs, as the wire artefact.
    fn wire_legs(&self, at: usize) -> Vec<(String, u8, i32, i32, i32, i32, i32, i32)> {
        let Some(route) = self.inner.candidates.get(at).and_then(|c| c.route.as_ref()) else {
            return Vec::new();
        };
        route
            .wire_legs()
            .iter()
            .map(|leg| {
                (leg.target.clone(), leg.kind.code(), leg.i, leg.j, leg.n,
                 leg.src_slot, leg.dst_slot, leg.bps)
            })
            .collect()
    }

    fn dst_slot(&self, at: usize) -> usize {
        self.inner
            .candidates
            .get(at)
            .and_then(|c| c.route.as_ref())
            .map_or(0, |r| r.dst_slot)
    }
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
        stop: crate::solve::Stop::Optimal,
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
