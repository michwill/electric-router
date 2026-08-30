//! The value-coordinate graph, in the browser.
//!
//! The twin of `src/graph_py.rs`, over the same `erouter_solve::graph`. The
//! arrays are typed arrays rather than lists, which is the only difference
//! that matters: `Vec<bool>` has no typed-array form, so the flags cross as
//! `Uint8Array` with the usual 0/1 convention.

use erouter_solve::graph::{
    self, ArcArrays, BuildOptions, CEILING_FACTOR, DUST_FLOOR, MAX_CONDITION,
};
use wasm_bindgen::prelude::*;

fn err(e: graph::GraphError) -> JsValue {
    JsError::new(&e.0).into()
}

fn flags(raw: Option<Vec<u8>>) -> Option<Vec<bool>> {
    raw.map(|v| v.into_iter().map(|b| b != 0).collect())
}

/// Solver input, assembled and owned on the Rust side.
#[wasm_bindgen]
pub struct Graph {
    pub(crate) inner: ArcArrays,
}

#[wasm_bindgen]
impl Graph {
    /// A graph from arrays that are already built, rather than from `a`, `B`
    /// and `nu`.
    ///
    /// `build` derives `G` and `eps`, then scales, clamps and merges. A
    /// caller that already has those -- the ballot's, which runs after all of
    /// it -- would answer a different question by re-deriving them, because
    /// `g_scale` alone moves every conductance. Generation reads `tau`,
    /// `sig`, `G`, `eps`, `cap`, `flagged` and `n_nodes` and nothing else.
    ///
    /// `flagged` is not optional in practice: the pin sweep is *defined* over
    /// the flagged active arcs, so an empty one does not degrade the ballot,
    /// it deletes a family from it.
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        tau: Vec<i64>,
        sig: Vec<i64>,
        g: Vec<f64>,
        eps: Vec<f64>,
        cap: Vec<f64>,
        flagged: Vec<u8>,
        n_nodes: usize,
    ) -> Result<Graph, JsValue> {
        let m = tau.len();
        if sig.len() != m || g.len() != m || eps.len() != m || cap.len() != m
            || flagged.len() != m
        {
            return Err(JsError::new(
                "tau, sig, G, eps, cap and flagged must be the same length",
            )
            .into());
        }
        Ok(Graph {
            inner: erouter_solve::graph::ArcArrays {
                tau,
                sig,
                a: vec![0.0; m],
                b: vec![0.0; m],
                g,
                eps,
                cap,
                flagged: flagged.iter().map(|&v| v != 0).collect(),
                clamped: vec![false; m],
                n_nodes,
                g_scale: 1.0,
                ill_conditioned: 0.0,
                sources: (0..m).map(|k| vec![k]).collect(),
                dropped: Vec::new(),
            },
        })
    }

    /// Assemble solver arrays (§9.5-9.7). `require` is `[src, dst]`: the pair
    /// the dust floor may not disconnect, or `undefined` for no constraint.
    #[wasm_bindgen(js_name = build)]
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        tau: Vec<i64>,
        sig: Vec<i64>,
        a: Vec<f64>,
        b: Vec<f64>,
        nu: Vec<f64>,
        psi: f64,
        cap: Option<Vec<f64>>,
        flagged: Option<Vec<u8>>,
        clamped: Option<Vec<u8>>,
        n_nodes: Option<usize>,
        dust_floor: Option<f64>,
        ceiling_factor: Option<f64>,
        merge_duplicates: Option<bool>,
        require: Option<Vec<u32>>,
    ) -> Result<Graph, JsValue> {
        let flagged = flags(flagged);
        let clamped = flags(clamped);
        let opts = BuildOptions {
            cap: cap.as_deref(),
            flagged: flagged.as_deref(),
            clamped: clamped.as_deref(),
            n_nodes,
            dust_floor: dust_floor.unwrap_or(DUST_FLOOR),
            ceiling_factor: ceiling_factor.unwrap_or(CEILING_FACTOR),
            merge_duplicates: merge_duplicates.unwrap_or(true),
            require: require.and_then(|v| match v[..] {
                [src, dst] => Some((src as usize, dst as usize)),
                _ => None,
            }),
        };
        let inner = graph::build(&tau, &sig, &a, &b, &nu, psi, &opts).map_err(err)?;
        Ok(Graph { inner })
    }

    /// (M3) conductance and (M4) forward drop, without assembling anything.
    /// Returns `G` and `eps` concatenated: one allocation, split by `m`.
    #[wasm_bindgen(js_name = arcParams)]
    pub fn arc_params(
        tau: Vec<i64>,
        sig: Vec<i64>,
        a: Vec<f64>,
        b: Vec<f64>,
        nu: Vec<f64>,
    ) -> Result<Vec<f64>, JsValue> {
        let (mut g, eps) = graph::arc_params(&tau, &sig, &a, &b, &nu).map_err(err)?;
        g.extend(eps);
        Ok(g)
    }

    /// §9.7 -- clamp in G-space, never by flooring B.
    #[wasm_bindgen(js_name = ceilingConductance)]
    pub fn ceiling_conductance(g: Vec<f64>, flagged: Vec<u8>, factor: Option<f64>) -> Vec<f64> {
        let mut g = g;
        let flagged: Vec<bool> = flagged.into_iter().map(|b| b != 0).collect();
        graph::ceiling_conductance(&mut g, &flagged, factor.unwrap_or(CEILING_FACTOR));
        g
    }

    /// L = B^T diag(G) B restricted to `keep`, row-major and flat.
    pub fn laplacian(
        tau: Vec<i64>,
        sig: Vec<i64>,
        g: Vec<f64>,
        n: usize,
        keep: Vec<u32>,
    ) -> Vec<f64> {
        let keep: Vec<usize> = keep.into_iter().map(|v| v as usize).collect();
        graph::laplacian(&tau, &sig, &g, n, &keep)
    }

    /// Nodes reachable from `root` over the given (undirected) arcs.
    #[wasm_bindgen(js_name = componentOf)]
    pub fn component_of(root: usize, tau: Vec<i64>, sig: Vec<i64>, n: usize) -> Vec<u8> {
        graph::component_of(root, &tau, &sig, n)
            .into_iter()
            .map(u8::from)
            .collect()
    }

    /// §9.1 -- normalise G by its median. Returns the scaled demand.
    pub fn scale(&mut self, psi: f64) -> f64 {
        graph::scale(&mut self.inner, psi)
    }

    #[wasm_bindgen(js_name = length)]
    pub fn len(&self) -> usize {
        self.inner.m()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.m() == 0
    }

    pub fn condition(&self) -> f64 {
        self.inner.condition()
    }

    #[wasm_bindgen(getter)]
    pub fn tau(&self) -> Vec<i64> {
        self.inner.tau.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn sig(&self) -> Vec<i64> {
        self.inner.sig.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> Vec<f64> {
        self.inner.a.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> Vec<f64> {
        self.inner.b.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn g(&self) -> Vec<f64> {
        self.inner.g.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn eps(&self) -> Vec<f64> {
        self.inner.eps.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn cap(&self) -> Vec<f64> {
        self.inner.cap.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn flagged(&self) -> Vec<u8> {
        self.inner.flagged.iter().copied().map(u8::from).collect()
    }
    #[wasm_bindgen(getter)]
    pub fn clamped(&self) -> Vec<u8> {
        self.inner.clamped.iter().copied().map(u8::from).collect()
    }
    #[wasm_bindgen(getter, js_name = nNodes)]
    pub fn n_nodes(&self) -> usize {
        self.inner.n_nodes
    }
    #[wasm_bindgen(getter, js_name = gScale)]
    pub fn g_scale(&self) -> f64 {
        self.inner.g_scale
    }
    #[wasm_bindgen(getter, js_name = illConditioned)]
    pub fn ill_conditioned(&self) -> f64 {
        self.inner.ill_conditioned
    }

    /// A merged group has several sources, so this crosses flat, with
    /// `sourceSpans` bounding each arc's slice.
    pub fn sources(&self) -> Vec<u32> {
        self.inner
            .sources
            .iter()
            .flat_map(|group| group.iter().map(|&k| k as u32))
            .collect()
    }

    #[wasm_bindgen(js_name = sourceSpans)]
    pub fn source_spans(&self) -> Vec<u32> {
        let mut spans = vec![0u32];
        let mut total = 0u32;
        for group in &self.inner.sources {
            total += group.len() as u32;
            spans.push(total);
        }
        spans
    }

    /// Original arc index -> why it is gone, in insertion order.
    pub fn dropped(&self) -> Vec<u32> {
        self.inner.dropped.iter().map(|&(k, _)| k as u32).collect()
    }

    #[wasm_bindgen(js_name = droppedReason)]
    pub fn dropped_reason(&self) -> Vec<String> {
        self.inner
            .dropped
            .iter()
            .map(|&(_, r)| r.name().to_string())
            .collect()
    }

    #[wasm_bindgen(js_name = maxCondition)]
    pub fn max_condition() -> f64 {
        MAX_CONDITION
    }
}
