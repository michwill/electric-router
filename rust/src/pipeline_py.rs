//! The quote's stages, across the PyO3 boundary.
//!
//! One class holding the arcs and the report, because the stages run in order
//! and each one hands the next a reduced arc list. Handing that list back and
//! forth as thirty-field objects per stage would be the whole cost of the
//! boundary; here it stays put and only the counters cross.
//!
//! The chain is not in here. See `pipeline.rs` for what that leaves out.

use crate::graph_py::Graph;
use crate::nodes_py::NodeMap;
use crate::pipeline::{self, Report};
use crate::realize_py::{Arcs, Route};
use pyo3::prelude::*;

/// The stages of one quote, over one arc list.
#[pyclass]
pub struct Stages {
    arcs: Vec<crate::types::PoolArc>,
    report: Report,
}

#[pymethods]
impl Stages {
    #[new]
    fn new(arcs: PyRef<'_, Arcs>) -> Self {
        Stages { arcs: arcs.inner.clone(), report: Report::default() }
    }

    fn __len__(&self) -> usize {
        self.arcs.len()
    }

    /// The arcs still standing, by their ids -- what each stage left behind.
    fn arc_ids(&self) -> Vec<String> {
        self.arcs.iter().map(|a| a.id.clone()).collect()
    }

    /// `(name, value)` in the order the stages set them.
    fn counters(&self) -> Vec<(String, i64)> {
        self.report.counters.clone()
    }

    fn warnings(&self) -> Vec<String> {
        self.report.warnings.clone()
    }

    // -- 1. the universe ---------------------------------------------------

    /// Drop arcs into nodes no route can pass *through*.
    fn prune_dead_end_nodes(&mut self, src_node: usize, dst_node: usize) {
        self.arcs =
            pipeline::prune_dead_end_nodes(&self.arcs, src_node, dst_node, &mut self.report);
    }

    /// Keep only what `dst` can be reached from, in either direction.
    fn restrict_to_component(&mut self, dst_node: usize, n_nodes: usize) {
        self.arcs =
            pipeline::restrict_to_component(&self.arcs, dst_node, n_nodes, &mut self.report);
    }

    // -- 2. the graph ------------------------------------------------------

    /// Treat immeasurably small curvature as the zero-curvature limit.
    /// Returns how many arcs were clamped.
    fn clamp_unphysical_depth(&mut self, nu: Vec<f64>, nodes: PyRef<'_, NodeMap>) -> usize {
        pipeline::clamp_unphysical_depth(&mut self.arcs, &nu, &nodes.inner)
    }

    /// Build the solver arrays from the current calibration. The arc list is
    /// re-aligned to whatever survived, and `G`/`eps` written back onto it.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &mut self, nu: Vec<f64>, psi_total: f64, nodes: PyRef<'_, NodeMap>,
        src_node: usize, dst_node: usize,
    ) -> PyResult<Graph> {
        let (kept, g) = pipeline::assemble(
            &self.arcs, &nu, psi_total, &nodes.inner, src_node, dst_node, &mut self.report,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?;
        self.arcs = kept;
        Ok(Graph { inner: g })
    }

    /// §2.6: `eps_f + eps_r <= 0` means `nu` is inconsistent with that pool.
    fn warn_pair_drops(&mut self) {
        pipeline::warn_pair_drops(&self.arcs, &mut self.report);
    }

    /// Link each arc to its opposite and record the fee the pair measures.
    fn pair_directions(&mut self) -> usize {
        pipeline::pair_directions(&mut self.arcs)
    }

    /// `sqrt(a_f * a_r)` per arc, NaN where the opposite was not calibrated.
    fn gamma_live(&self) -> Vec<f64> {
        self.arcs.iter().map(|a| a.gamma_live).collect()
    }

    /// `""` where an arc has no opposite.
    fn reverse_ids(&self) -> Vec<String> {
        self.arcs
            .iter()
            .map(|a| a.reverse_id.clone().unwrap_or_default())
            .collect()
    }

    /// What `assemble` and the clamp left on each arc: `a, B, cap, G, eps`,
    /// flat, plus the two flags.
    fn arc_numbers(&self) -> Vec<f64> {
        self.arcs
            .iter()
            .flat_map(|a| [a.a, a.b, a.cap, a.g, a.eps])
            .collect()
    }

    fn arc_flags(&self) -> Vec<(bool, bool)> {
        self.arcs.iter().map(|a| (a.clamped, a.convex_flag)).collect()
    }

    // -- 4. reading it back -------------------------------------------------

    /// This arc's realised input, in its own token's raw units.
    fn realised_delta(&self, at: usize, psi_value: f64, nu: Vec<f64>,
                      nodes: PyRef<'_, NodeMap>) -> f64 {
        pipeline::realised_delta(&self.arcs[at], psi_value, &nu, &nodes.inner)
    }

    /// §12.1's `theta_p` for every arc carrying flow.
    fn realised_theta(&self, psi: Vec<f64>, nu: Vec<f64>, nodes: PyRef<'_, NodeMap>,
                      active: Vec<usize>) -> Vec<(usize, f64)> {
        pipeline::realised_theta(&self.arcs, &psi, &nu, &nodes.inner, &active)
    }
}

// -- 3. the check ----------------------------------------------------------

/// How much KCL slop is floating-point noise rather than a bug.
#[pyfunction]
pub fn kcl_tolerance(psi_total: f64, g_scale: f64) -> f64 {
    pipeline::kcl_tolerance(psi_total, g_scale)
}

/// `||B^T psi - s_hat||_inf / Psi`, and *where* it is: the residual, the worst
/// node, and that node's live in and out arc counts.
#[pyfunction]
pub fn kcl_detail(
    graph: PyRef<'_, Graph>, psi: Vec<f64>, src: usize, dst: usize, psi_total: f64,
) -> (f64, i64, usize, usize) {
    let got = pipeline::kcl_detail(&graph.inner, &psi, src, dst, psi_total);
    (got.residual, got.node, got.arcs_in, got.arcs_out)
}

/// The KCL residual a backward-stable solve could deliver on this graph.
#[pyfunction]
pub fn achievable_kcl(graph: PyRef<'_, Graph>, active: Vec<bool>, dst: usize) -> f64 {
    pipeline::achievable_kcl(&graph.inner, &active, dst)
}

// -- 5. the rank -----------------------------------------------------------

/// Output-token wei per 1 ETH, for costing gas. 0 when ETH is unpriced.
#[pyfunction]
pub fn dst_per_eth(nodes: PyRef<'_, NodeMap>, nu: Vec<f64>, dst_token: &str) -> f64 {
    pipeline::dst_per_eth(&nodes.inner, &nu, dst_token)
}

/// One leg's gas, in the solver's scaled value units. 0 disables it.
#[pyfunction]
pub fn gas_cost(
    nodes: PyRef<'_, NodeMap>, nu: Vec<f64>, dst_token: &str, gas_price_wei: i64,
    g_scale: f64,
) -> f64 {
    pipeline::gas_cost(&nodes.inner, &nu, dst_token, gas_price_wei, g_scale)
}

/// How promising this candidate is as a scout entrant, or 0 to skip it.
#[pyfunction]
pub fn scout_priority(route: PyRef<'_, Route>) -> f64 {
    pipeline::scout_priority(&route.inner)
}

/// Indices of each contiguous run of legs leaving one slot, where it splits.
#[pyfunction]
pub fn split_groups(route: PyRef<'_, Route>) -> Vec<Vec<usize>> {
    let legs: Vec<crate::types::Leg> =
        route.inner.legs.iter().map(|rl| rl.leg.clone()).collect();
    crate::split::split_groups(&legs)
}

// -- 6. the walk -----------------------------------------------------------

/// Leg indices grouped so no leg in a group feeds another in it.
#[pyfunction]
pub fn pricing_layers(route: PyRef<'_, Route>) -> Vec<Vec<usize>> {
    pipeline::pricing_layers(&route.inner)
}

/// One unit of the output token, in the human units calibration fits in.
#[pyfunction]
pub fn quantum(decimals_out: u32) -> f64 {
    pipeline::quantum(decimals_out)
}
