//! The quote's stages, in the browser.
//!
//! The twin of `src/pipeline_py.rs`. Counters cross as two parallel arrays for
//! the usual reason: a pair has no typed-array form.

use erouter_solve::pipeline::{self, Report};
use erouter_solve::types::PoolArc;
use wasm_bindgen::prelude::*;

use crate::graph::Graph;
use crate::nodes::NodeMap;
use crate::realize::{Arcs, Route};

/// The stages of one quote, over one arc list.
#[wasm_bindgen]
pub struct Stages {
    arcs: Vec<PoolArc>,
    report: Report,
}

#[wasm_bindgen]
impl Stages {
    #[wasm_bindgen(constructor)]
    pub fn new(arcs: &Arcs) -> Self {
        Stages { arcs: arcs.inner.clone(), report: Report::default() }
    }

    #[wasm_bindgen(js_name = length)]
    pub fn len(&self) -> usize {
        self.arcs.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.arcs.is_empty()
    }

    /// The arcs still standing, by their ids -- what each stage left behind.
    #[wasm_bindgen(js_name = arcIds)]
    pub fn arc_ids(&self) -> Vec<String> {
        self.arcs.iter().map(|a| a.id.clone()).collect()
    }

    /// Counter names, in the order the stages set them; the numbers are in
    /// `counterValues`.
    pub fn counters(&self) -> Vec<String> {
        self.report.counters.iter().map(|(k, _)| k.clone()).collect()
    }

    #[wasm_bindgen(js_name = counterValues)]
    pub fn counter_values(&self) -> Vec<f64> {
        self.report.counters.iter().map(|&(_, v)| v as f64).collect()
    }

    pub fn warnings(&self) -> Vec<String> {
        self.report.warnings.clone()
    }

    // -- 1. the universe ---------------------------------------------------

    /// Drop arcs into nodes no route can pass *through*.
    #[wasm_bindgen(js_name = pruneDeadEndNodes)]
    pub fn prune_dead_end_nodes(&mut self, src_node: usize, dst_node: usize) {
        self.arcs =
            pipeline::prune_dead_end_nodes(&self.arcs, src_node, dst_node, &mut self.report);
    }

    /// Keep only what `dst` can be reached from, in either direction.
    #[wasm_bindgen(js_name = restrictToComponent)]
    pub fn restrict_to_component(&mut self, dst_node: usize, n_nodes: usize) {
        self.arcs =
            pipeline::restrict_to_component(&self.arcs, dst_node, n_nodes, &mut self.report);
    }

    // -- 2. the graph ------------------------------------------------------

    /// Treat immeasurably small curvature as the zero-curvature limit.
    #[wasm_bindgen(js_name = clampUnphysicalDepth)]
    pub fn clamp_unphysical_depth(&mut self, nu: Vec<f64>, nodes: &NodeMap) -> usize {
        pipeline::clamp_unphysical_depth(&mut self.arcs, &nu, &nodes.inner)
    }

    /// Build the solver arrays from the current calibration.
    pub fn assemble(
        &mut self, nu: Vec<f64>, psi_total: f64, nodes: &NodeMap, src_node: usize,
        dst_node: usize,
    ) -> Result<Graph, JsValue> {
        let (kept, g) = pipeline::assemble(
            &self.arcs, &nu, psi_total, &nodes.inner, src_node, dst_node, &mut self.report,
        )
        .map_err(|e| JsError::new(&e.0))?;
        self.arcs = kept;
        Ok(Graph { inner: g })
    }

    /// §2.6: `eps_f + eps_r <= 0` means `nu` is inconsistent with that pool.
    #[wasm_bindgen(js_name = warnPairDrops)]
    pub fn warn_pair_drops(&mut self) {
        pipeline::warn_pair_drops(&self.arcs, &mut self.report);
    }

    /// Link each arc to its opposite and record the fee the pair measures.
    #[wasm_bindgen(js_name = pairDirections)]
    pub fn pair_directions(&mut self) -> usize {
        pipeline::pair_directions(&mut self.arcs)
    }

    /// `sqrt(a_f * a_r)` per arc, NaN where the opposite was not calibrated.
    #[wasm_bindgen(js_name = gammaLive)]
    pub fn gamma_live(&self) -> Vec<f64> {
        self.arcs.iter().map(|a| a.gamma_live).collect()
    }

    /// `""` where an arc has no opposite.
    #[wasm_bindgen(js_name = reverseIds)]
    pub fn reverse_ids(&self) -> Vec<String> {
        self.arcs
            .iter()
            .map(|a| a.reverse_id.clone().unwrap_or_default())
            .collect()
    }

    /// `a, B, cap, G, eps` per arc, flat.
    #[wasm_bindgen(js_name = arcNumbers)]
    pub fn arc_numbers(&self) -> Vec<f64> {
        self.arcs.iter().flat_map(|a| [a.a, a.b, a.cap, a.g, a.eps]).collect()
    }

    /// `clamped, convex_flag` per arc, interleaved.
    #[wasm_bindgen(js_name = arcFlags)]
    pub fn arc_flags(&self) -> Vec<u8> {
        self.arcs
            .iter()
            .flat_map(|a| [u8::from(a.clamped), u8::from(a.convex_flag)])
            .collect()
    }

    // -- 4. reading it back -------------------------------------------------

    /// This arc's realised input, in its own token's raw units.
    #[wasm_bindgen(js_name = realisedDelta)]
    pub fn realised_delta(
        &self, at: usize, psi_value: f64, nu: Vec<f64>, nodes: &NodeMap,
    ) -> f64 {
        pipeline::realised_delta(&self.arcs[at], psi_value, &nu, &nodes.inner)
    }

    /// §12.1's `theta_p` for every arc carrying flow: the arc indices, with
    /// the values in `realisedThetaValues`.
    #[wasm_bindgen(js_name = realisedTheta)]
    pub fn realised_theta(
        &self, psi: Vec<f64>, nu: Vec<f64>, nodes: &NodeMap, active: Vec<u32>,
    ) -> Vec<u32> {
        self.theta(&psi, &nu, nodes, &active).into_iter().map(|(k, _)| k as u32).collect()
    }

    #[wasm_bindgen(js_name = realisedThetaValues)]
    pub fn realised_theta_values(
        &self, psi: Vec<f64>, nu: Vec<f64>, nodes: &NodeMap, active: Vec<u32>,
    ) -> Vec<f64> {
        self.theta(&psi, &nu, nodes, &active).into_iter().map(|(_, v)| v).collect()
    }
}

impl Stages {
    fn theta(
        &self, psi: &[f64], nu: &[f64], nodes: &NodeMap, active: &[u32],
    ) -> Vec<(usize, f64)> {
        let live: Vec<usize> = active.iter().map(|&k| k as usize).collect();
        pipeline::realised_theta(&self.arcs, psi, nu, &nodes.inner, &live)
    }
}

// -- 3. the check ----------------------------------------------------------

/// How much KCL slop is floating-point noise rather than a bug.
#[wasm_bindgen(js_name = kclTolerance)]
pub fn kcl_tolerance(psi_total: f64, g_scale: f64) -> f64 {
    pipeline::kcl_tolerance(psi_total, g_scale)
}

/// `[residual, worst node, live arcs in, live arcs out]`.
#[wasm_bindgen(js_name = kclDetail)]
pub fn kcl_detail(
    graph: &Graph, psi: Vec<f64>, src: usize, dst: usize, psi_total: f64,
) -> Vec<f64> {
    let got = pipeline::kcl_detail(&graph.inner, &psi, src, dst, psi_total);
    vec![got.residual, got.node as f64, got.arcs_in as f64, got.arcs_out as f64]
}

/// The KCL residual a backward-stable solve could deliver on this graph.
#[wasm_bindgen(js_name = achievableKcl)]
pub fn achievable_kcl(graph: &Graph, active: Vec<u8>, dst: usize) -> f64 {
    let live: Vec<bool> = active.into_iter().map(|v| v != 0).collect();
    pipeline::achievable_kcl(&graph.inner, &live, dst)
}

// -- 5. the rank -----------------------------------------------------------

/// Output-token wei per 1 ETH, for costing gas. 0 when ETH is unpriced.
#[wasm_bindgen(js_name = dstPerEth)]
pub fn dst_per_eth(nodes: &NodeMap, nu: Vec<f64>, dst_token: &str) -> f64 {
    pipeline::dst_per_eth(&nodes.inner, &nu, dst_token)
}

/// One leg's gas, in the solver's scaled value units. 0 disables it.
#[wasm_bindgen(js_name = gasCost)]
pub fn gas_cost(
    nodes: &NodeMap, nu: Vec<f64>, dst_token: &str, gas_price_wei: f64, g_scale: f64,
) -> f64 {
    pipeline::gas_cost(&nodes.inner, &nu, dst_token, gas_price_wei as i64, g_scale)
}

/// How promising this candidate is as a scout entrant, or 0 to skip it.
#[wasm_bindgen(js_name = scoutPriority)]
pub fn scout_priority(route: &Route) -> f64 {
    pipeline::scout_priority(&route.inner)
}

/// Indices of each contiguous run of legs leaving one slot, where it splits.
/// Flat, with `splitGroupSpans` bounding each run.
#[wasm_bindgen(js_name = splitGroups)]
pub fn split_groups(route: &Route) -> Vec<u32> {
    flatten(&groups_of(route))
}

#[wasm_bindgen(js_name = splitGroupSpans)]
pub fn split_group_spans(route: &Route) -> Vec<u32> {
    spans(&groups_of(route))
}

// -- 6. the walk -----------------------------------------------------------

/// Leg indices grouped so no leg in a group feeds another in it.
#[wasm_bindgen(js_name = pricingLayers)]
pub fn pricing_layers(route: &Route) -> Vec<u32> {
    flatten(&pipeline::pricing_layers(&route.inner))
}

#[wasm_bindgen(js_name = pricingLayerSpans)]
pub fn pricing_layer_spans(route: &Route) -> Vec<u32> {
    spans(&pipeline::pricing_layers(&route.inner))
}

/// One unit of the output token, in the human units calibration fits in.
#[wasm_bindgen]
pub fn quantum(decimals_out: u32) -> f64 {
    pipeline::quantum(decimals_out)
}

fn groups_of(route: &Route) -> Vec<Vec<usize>> {
    let legs: Vec<erouter_solve::types::Leg> =
        route.inner.legs.iter().map(|rl| rl.leg.clone()).collect();
    erouter_solve::split::split_groups(&legs)
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
