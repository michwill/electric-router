//! The refit, across the PyO3 boundary.
//!
//! Three calls a round with the chain in between: `plan` says what to quote,
//! `apply` folds the answers back, `rebuild` recomputes the arrays. The loop
//! stays with the caller because two of its steps -- probing and re-solving --
//! are the caller's to make.

use crate::graph_py::Graph;
use crate::nodes_py::NodeMap;
use crate::realize_py::Arcs;
use crate::refit::{self, Answer, Plan};
use pyo3::prelude::*;

/// What one round asked for, and for whom.
#[pyclass]
pub struct Refit {
    inner: Plan,
    arcs: Vec<crate::types::PoolArc>,
}

#[pymethods]
impl Refit {
    /// Which arcs to re-quote, and at what sizes.
    #[staticmethod]
    fn plan(
        arcs: PyRef<'_, Arcs>, psi: Vec<f64>, nu: Vec<f64>, nodes: PyRef<'_, NodeMap>,
    ) -> PyResult<Refit> {
        let inner = refit::plan(&arcs.inner, &psi, &nu, &nodes.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?;
        Ok(Refit { inner, arcs: arcs.inner.clone() })
    }

    /// `(pool, kind, i, j, n, dx)` per probe, in batch order.
    fn probes(&self) -> Vec<(String, u8, i32, i32, i32, u128)> {
        self.inner
            .probes
            .iter()
            .map(|p| (p.pool.clone(), p.kind.code(), p.i, p.j, p.n, p.dx))
            .collect()
    }

    /// `(arc index, probe offset, canonical size)` per planned arc.
    fn planned(&self) -> Vec<(usize, usize, f64)> {
        self.inner
            .plan
            .iter()
            .map(|p| (p.arc, p.offset, p.delta_canonical))
            .collect()
    }

    /// Arcs skipped before the probes were planned.
    #[getter]
    fn unresolved(&self) -> usize {
        self.inner.unresolved
    }

    /// Fold the answers back onto the arcs. `answers` is `(ok, value)` per
    /// probe, in the order `probes` returned them.
    fn apply(
        &mut self, answers: Vec<(bool, u128)>, nodes: PyRef<'_, NodeMap>,
    ) -> PyResult<(usize, usize, usize)> {
        if answers.len() != self.inner.probes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} answers for {} probes", answers.len(), self.inner.probes.len()
            )));
        }
        let folded: Vec<Answer> =
            answers.into_iter().map(|(ok, value)| Answer { ok, value }).collect();
        Ok(refit::apply(&mut self.arcs, &self.inner, &folded, &nodes.inner))
    }

    /// What the refit left on each arc: `a, B, cap, calib_delta`, flat.
    fn arc_numbers(&self) -> Vec<f64> {
        self.arcs
            .iter()
            .flat_map(|a| [a.a, a.b, a.cap, a.calib_delta])
            .collect()
    }

    fn arc_flags(&self) -> Vec<(bool, bool)> {
        self.arcs.iter().map(|a| (a.clamped, a.convex_flag)).collect()
    }

    /// Recompute G and eps in place, keeping the same indexing.
    fn rebuild(&self, graph: &mut Graph, nu: Vec<f64>) {
        refit::rebuild(&mut graph.inner, &self.arcs, &nu);
    }
}

/// How far the flow moved, and whether that is close enough to stop.
#[pyfunction]
pub fn round_stats(before: Vec<f64>, after: Vec<f64>, psi_total: f64) -> (f64, bool) {
    refit::round_stats(&before, &after, psi_total)
}

/// The largest relative move in `B` across a round.
#[pyfunction]
pub fn b_change(before: Vec<f64>, after: Vec<f64>) -> f64 {
    refit::b_change(&before, &after)
}
