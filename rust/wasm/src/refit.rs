//! The refit, in the browser.
//!
//! The twin of `src/refit_py.rs`. Probes cross as parallel arrays and answers
//! as an `ok` mask beside the values, for the usual reason: a tuple has no
//! typed-array form.

use crate::graph::Graph;
use crate::nodes::NodeMap;
use crate::realize::Arcs;
use erouter_solve::refit::{self, Answer, Plan};
use wasm_bindgen::prelude::*;

/// What one round asked for, and for whom.
#[wasm_bindgen]
pub struct Refit {
    inner: Plan,
    arcs: Vec<erouter_solve::types::PoolArc>,
}

#[wasm_bindgen]
impl Refit {
    /// Which arcs to re-quote, and at what sizes.
    pub fn plan(
        arcs: &Arcs, psi: Vec<f64>, nu: Vec<f64>, nodes: &NodeMap,
    ) -> Result<Refit, JsValue> {
        let inner = refit::plan(&arcs.inner, &psi, &nu, &nodes.inner)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Refit { inner, arcs: arcs.inner.clone() })
    }

    /// The pool each probe targets; the numbers are in `probeNumbers` and the
    /// sizes, which are wei-scale, in `probeSizes` as decimal strings.
    pub fn probes(&self) -> Vec<String> {
        self.inner.probes.iter().map(|p| p.pool.clone()).collect()
    }

    /// `kind, i, j, n` per probe, flat.
    #[wasm_bindgen(js_name = probeNumbers)]
    pub fn probe_numbers(&self) -> Vec<i32> {
        self.inner
            .probes
            .iter()
            .flat_map(|p| [p.kind.code() as i32, p.i, p.j, p.n])
            .collect()
    }

    #[wasm_bindgen(js_name = probeSizes)]
    pub fn probe_sizes(&self) -> Vec<String> {
        self.inner.probes.iter().map(|p| p.dx.to_string()).collect()
    }

    /// The arc each planned entry belongs to; `plannedOffsets` and
    /// `plannedSizes` carry the rest.
    pub fn planned(&self) -> Vec<u32> {
        self.inner.plan.iter().map(|p| p.arc as u32).collect()
    }

    #[wasm_bindgen(js_name = plannedOffsets)]
    pub fn planned_offsets(&self) -> Vec<u32> {
        self.inner.plan.iter().map(|p| p.offset as u32).collect()
    }

    #[wasm_bindgen(js_name = plannedSizes)]
    pub fn planned_sizes(&self) -> Vec<f64> {
        self.inner.plan.iter().map(|p| p.delta_canonical).collect()
    }

    /// Arcs skipped before the probes were planned.
    #[wasm_bindgen(getter)]
    pub fn unresolved(&self) -> usize {
        self.inner.unresolved
    }

    /// Fold the answers back onto the arcs. Returns `[quoted, reflagged,
    /// unresolved]`.
    pub fn apply(
        &mut self, ok: Vec<u8>, values: Vec<String>, nodes: &NodeMap,
    ) -> Result<Vec<u32>, JsValue> {
        if ok.len() != self.inner.probes.len() || values.len() != ok.len() {
            return Err(JsError::new("one answer per probe").into());
        }
        let mut folded = Vec::with_capacity(ok.len());
        for (flag, value) in ok.iter().zip(values.iter()) {
            let parsed = value
                .parse::<u128>()
                .map_err(|_| JsError::new(&format!("not a u128: {value}")))?;
            folded.push(Answer { ok: *flag != 0, value: parsed });
        }
        let (quoted, reflagged, unresolved) =
            refit::apply(&mut self.arcs, &self.inner, &folded, &nodes.inner);
        Ok(vec![quoted as u32, reflagged as u32, unresolved as u32])
    }

    /// What the refit left on each arc: `a, B, cap, calib_delta`, flat.
    #[wasm_bindgen(js_name = arcNumbers)]
    pub fn arc_numbers(&self) -> Vec<f64> {
        self.arcs.iter().flat_map(|a| [a.a, a.b, a.cap, a.calib_delta]).collect()
    }

    /// `clamped, convex_flag` per arc, interleaved.
    #[wasm_bindgen(js_name = arcFlags)]
    pub fn arc_flags(&self) -> Vec<u8> {
        self.arcs
            .iter()
            .flat_map(|a| [u8::from(a.clamped), u8::from(a.convex_flag)])
            .collect()
    }

    /// Recompute G and eps in place, keeping the same indexing.
    pub fn rebuild(&self, graph: &mut Graph, nu: Vec<f64>) {
        refit::rebuild(&mut graph.inner, &self.arcs, &nu);
    }
}

/// How far the flow moved, and whether that is close enough to stop:
/// `[moved, converged]`.
#[wasm_bindgen(js_name = roundStats)]
pub fn round_stats(before: Vec<f64>, after: Vec<f64>, psi_total: f64) -> Vec<f64> {
    let (moved, converged) = refit::round_stats(&before, &after, psi_total);
    vec![moved, f64::from(u8::from(converged))]
}

/// The largest relative move in `B` across a round.
#[wasm_bindgen(js_name = bChange)]
pub fn b_change(before: Vec<f64>, after: Vec<f64>) -> f64 {
    refit::b_change(&before, &after)
}
