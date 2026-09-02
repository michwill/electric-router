//! The value-coordinate graph, across the PyO3 boundary.
//!
//! One crossing per quote, not per pivot: `build` is called once and the
//! arrays come back whole. This is not here for speed -- `graph.build` is a
//! millisecond -- it is here so the assembly rules live in one place and a
//! browser gets them too. See `graph.rs`.

use crate::graph::{
    self, ArcArrays, BuildOptions, CEILING_FACTOR, DUST_FLOOR, MAX_CONDITION,
};
use pyo3::prelude::*;

fn err(e: graph::GraphError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

/// Solver input, assembled and owned on the Rust side.
#[pyclass]
pub struct Graph {
    pub(crate) inner: ArcArrays,
}

#[pymethods]
impl Graph {
    /// Assemble solver arrays (§9.5-9.7). `require` is the `(src, dst)` pair
    /// the dust floor may not disconnect.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tau, sig, a, b, nu, psi, *, cap=None, flagged=None,
                        clamped=None, n_nodes=None, dust_floor=DUST_FLOOR,
                        ceiling_factor=CEILING_FACTOR, merge_duplicates=true,
                        require=None))]
    fn build(
        tau: Vec<i64>,
        sig: Vec<i64>,
        a: Vec<f64>,
        b: Vec<f64>,
        nu: Vec<f64>,
        psi: f64,
        cap: Option<Vec<f64>>,
        flagged: Option<Vec<bool>>,
        clamped: Option<Vec<bool>>,
        n_nodes: Option<usize>,
        dust_floor: f64,
        ceiling_factor: f64,
        merge_duplicates: bool,
        require: Option<(usize, usize)>,
    ) -> PyResult<Graph> {
        let n = n_nodes.unwrap_or_else(|| {
            let hi = tau.iter().chain(sig.iter()).copied().max().unwrap_or(-1);
            (hi + 1).max(0) as usize
        });
        crate::py::arc_nodes(&tau, &sig, n)?;
        for (what, len) in [("a", a.len()), ("b", b.len())] {
            crate::py::same_length(what, len, tau.len())?;
        }
        crate::py::same_length("nu", nu.len(), n)?;
        let opts = BuildOptions {
            cap: cap.as_deref(),
            flagged: flagged.as_deref(),
            clamped: clamped.as_deref(),
            n_nodes,
            dust_floor,
            ceiling_factor,
            merge_duplicates,
            require,
        };
        let inner = graph::build(&tau, &sig, &a, &b, &nu, psi, &opts).map_err(err)?;
        Ok(Graph { inner })
    }

    /// A graph from arrays that are already built, rather than from `a`, `B`
    /// and `nu`.
    ///
    /// `build` derives `G` and `eps`, then scales, clamps and merges. By the
    /// time the pipeline reaches the ballot all of that has happened, and
    /// re-deriving it here would answer a different question -- `g_scale`
    /// alone moves every conductance. Generation reads `tau`, `sig`, `G`,
    /// `eps`, `cap`, `flagged` and `n_nodes` and nothing else, so those are
    /// what cross.
    ///
    /// `flagged` is not optional in practice: the pin sweep is *defined* over
    /// the flagged active arcs, so defaulting it to false does not degrade the
    /// ballot, it deletes a family from it.
    #[staticmethod]
    #[pyo3(signature = (tau, sig, g, eps, cap, flagged, n_nodes))]
    fn from_arrays(
        tau: Vec<i64>,
        sig: Vec<i64>,
        g: Vec<f64>,
        eps: Vec<f64>,
        cap: Vec<f64>,
        flagged: Vec<bool>,
        n_nodes: usize,
    ) -> PyResult<Graph> {
        crate::py::arc_nodes(&tau, &sig, n_nodes)?;
        let m = tau.len();
        if sig.len() != m || g.len() != m || eps.len() != m || cap.len() != m
            || flagged.len() != m
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tau, sig, G, eps, cap and flagged must be the same length",
            ));
        }
        Ok(Graph {
            inner: crate::graph::ArcArrays {
                tau,
                sig,
                a: vec![0.0; m],
                b: vec![0.0; m],
                g,
                eps,
                cap,
                flagged,
                clamped: vec![false; m],
                n_nodes,
                g_scale: 1.0,
                ill_conditioned: 0.0,
                sources: (0..m).map(|k| vec![k]).collect(),
                dropped: Vec::new(),
            },
        })
    }

    /// (M3) conductance and (M4) forward drop, without assembling anything.
    #[staticmethod]
    fn arc_params(
        tau: Vec<i64>,
        sig: Vec<i64>,
        a: Vec<f64>,
        b: Vec<f64>,
        nu: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        graph::arc_params(&tau, &sig, &a, &b, &nu).map_err(err)
    }

    /// §9.7 -- clamp in G-space, never by flooring B.
    #[staticmethod]
    #[pyo3(signature = (g, flagged, factor=CEILING_FACTOR))]
    fn ceiling_conductance(
        g: Vec<f64>, flagged: Vec<bool>, factor: f64,
    ) -> PyResult<Vec<f64>> {
        crate::py::same_length("flagged", flagged.len(), g.len())?;
        let mut g = g;
        graph::ceiling_conductance(&mut g, &flagged, factor);
        Ok(g)
    }

    /// L = B^T diag(G) B restricted to `keep`, row-major and flat.
    #[staticmethod]
    fn laplacian(
        tau: Vec<i64>, sig: Vec<i64>, g: Vec<f64>, n: usize, keep: Vec<usize>,
    ) -> PyResult<Vec<f64>> {
        crate::py::arc_nodes(&tau, &sig, n)?;
        crate::py::same_length("g", g.len(), tau.len())?;
        // `keep` squares: the matrix is `keep.len()^2` doubles, and the LU
        // that reads it is cubic, so the bound is the dense solve's and not
        // the graph's.
        if keep.len() > crate::py::MAX_DENSE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "keep has {} node(s), past the {} this will solve for",
                keep.len(),
                crate::py::MAX_DENSE
            )));
        }
        for &k in &keep {
            crate::py::node("keep", k, n)?;
        }
        Ok(graph::laplacian(&tau, &sig, &g, n, &keep))
    }

    /// Nodes reachable from `root` over the given (undirected) arcs.
    #[staticmethod]
    fn component_of(
        root: usize, tau: Vec<i64>, sig: Vec<i64>, n: usize,
    ) -> PyResult<Vec<bool>> {
        crate::py::arc_nodes(&tau, &sig, n)?;
        crate::py::node("root", root, n)?;
        Ok(graph::component_of(root, &tau, &sig, n))
    }

    /// §9.1 -- normalise G by its median. Returns the scaled demand.
    fn scale(&mut self, psi: f64) -> f64 {
        graph::scale(&mut self.inner, psi)
    }

    fn __len__(&self) -> usize {
        self.inner.m()
    }

    fn condition(&self) -> f64 {
        self.inner.condition()
    }

    #[getter]
    fn tau(&self) -> Vec<i64> {
        self.inner.tau.clone()
    }
    #[getter]
    fn sig(&self) -> Vec<i64> {
        self.inner.sig.clone()
    }
    #[getter]
    fn a(&self) -> Vec<f64> {
        self.inner.a.clone()
    }
    #[getter]
    fn b(&self) -> Vec<f64> {
        self.inner.b.clone()
    }
    #[getter]
    fn g(&self) -> Vec<f64> {
        self.inner.g.clone()
    }
    #[getter]
    fn eps(&self) -> Vec<f64> {
        self.inner.eps.clone()
    }
    #[getter]
    fn cap(&self) -> Vec<f64> {
        self.inner.cap.clone()
    }
    #[getter]
    fn flagged(&self) -> Vec<bool> {
        self.inner.flagged.clone()
    }
    #[getter]
    fn clamped(&self) -> Vec<bool> {
        self.inner.clamped.clone()
    }
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes
    }
    #[getter]
    fn g_scale(&self) -> f64 {
        self.inner.g_scale
    }
    #[getter]
    fn ill_conditioned(&self) -> f64 {
        self.inner.ill_conditioned
    }

    /// A merged group has several sources, so this crosses flat with `spans`
    /// bounding each arc's slice -- the same shape `plan_sized` uses.
    fn sources(&self) -> (Vec<u32>, Vec<u32>) {
        let mut flat = Vec::new();
        let mut spans = vec![0u32];
        for group in &self.inner.sources {
            flat.extend(group.iter().map(|&k| k as u32));
            spans.push(flat.len() as u32);
        }
        (flat, spans)
    }

    /// Original arc index -> why it is gone, in insertion order.
    fn dropped(&self) -> (Vec<u32>, Vec<String>) {
        let index = self.inner.dropped.iter().map(|&(k, _)| k as u32).collect();
        let reason = self
            .inner
            .dropped
            .iter()
            .map(|&(_, r)| r.name().to_string())
            .collect();
        (index, reason)
    }

    #[staticmethod]
    fn max_condition() -> f64 {
        MAX_CONDITION
    }
}
