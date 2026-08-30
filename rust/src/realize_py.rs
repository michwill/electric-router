//! Realisation, across the PyO3 boundary.
//!
//! An arc has thirty fields and a route has a leg list; neither is a number,
//! so this is the first binding whose job is mostly marshalling. Arcs go in
//! through a builder, one call each, and legs come back as parallel arrays --
//! the same shape `Graph` uses, for the same reason: a list of objects would
//! be one allocation per leg on a path that runs once per candidate.

use crate::nodes_py::NodeMap;
use crate::realize::{self, RealizedRoute};
use crate::types::{ArcKind, PoolArc};
use pyo3::prelude::*;

fn err(e: realize::RealizationError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.0)
}

fn kind_of(code: u8) -> PyResult<ArcKind> {
    ArcKind::from_code(code)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no such kind: {code}")))
}

/// The arcs a route is realised from, built one at a time.
#[pyclass]
#[derive(Default)]
pub struct Arcs {
    pub(crate) inner: Vec<PoolArc>,
}

#[pymethods]
impl Arcs {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// One arc, with the fields realisation actually reads. Returns its index.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, pool, kind, i, j, n_coins, token_in, token_out, tau, sigma,
                        a, b, cap, g, eps, reserve_in, decimals_in, tvl_usd,
                        gamma_live, note="", calib_delta=0.0, decimals_out=18))]
    fn add(
        &mut self, id: &str, pool: &str, kind: u8, i: i32, j: i32, n_coins: i32,
        token_in: &str, token_out: &str, tau: usize, sigma: usize,
        a: f64, b: f64, cap: f64, g: f64, eps: f64, reserve_in: u128,
        decimals_in: u32, tvl_usd: f64, gamma_live: f64, note: &str,
        calib_delta: f64, decimals_out: u32,
    ) -> PyResult<usize> {
        let mut arc = PoolArc::new(
            id.to_string(), pool.to_string(), kind_of(kind)?, i, j, n_coins,
            token_in.to_string(), token_out.to_string(), tau, sigma,
        );
        arc.a = a;
        arc.b = b;
        arc.cap = cap;
        arc.g = g;
        arc.eps = eps;
        arc.reserve_in = reserve_in;
        arc.decimals_in = decimals_in;
        arc.tvl_usd = tvl_usd;
        arc.gamma_live = gamma_live;
        arc.note = note.to_string();
        // The refit reads both: `calib_delta` is what `REFIT_MIN_FRACTION`
        // compares a realised size against, and without it that guard is
        // silently off; `decimals_out` scales the quote it fits against.
        arc.calib_delta = calib_delta;
        arc.decimals_out = decimals_out;
        self.inner.push(arc);
        Ok(self.inner.len() - 1)
    }
}

/// A solved flow, realised into legs.
#[pyclass]
pub struct Route {
    pub(crate) inner: RealizedRoute,
}

#[pymethods]
impl Route {
    /// Build the executable leg list from a solved flow.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (arcs, psi, nu, nodes, src_token, dst_token, amount_in,
                        potentials=None))]
    fn realize(
        arcs: PyRef<'_, Arcs>, psi: Vec<f64>, nu: Vec<f64>, nodes: PyRef<'_, NodeMap>,
        src_token: &str, dst_token: &str, amount_in: &str,
        potentials: Option<Vec<f64>>,
    ) -> PyResult<Route> {
        let amount = amount_in
            .parse()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                format!("not a u256: {amount_in}")))?;
        let inner = realize::realize(
            &arcs.inner, &psi, &nu, &nodes.inner, src_token, dst_token, amount,
            potentials.as_deref(),
        )
        .map_err(err)?;
        Ok(Route { inner })
    }

    /// The route between two tokens of the *same* node: the conversion itself.
    #[staticmethod]
    fn conversion_route(
        nodes: PyRef<'_, NodeMap>, src_token: &str, dst_token: &str, amount_in: &str,
    ) -> PyResult<Route> {
        let amount = amount_in
            .parse()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                format!("not a u256: {amount_in}")))?;
        let inner = realize::conversion_route(&nodes.inner, src_token, dst_token, amount)
            .map_err(err)?;
        Ok(Route { inner })
    }

    /// Kahn's algorithm over the active arcs. Refuses a cycle.
    #[staticmethod]
    fn topological_nodes(tau: Vec<i64>, sig: Vec<i64>, n_nodes: usize) -> PyResult<Vec<usize>> {
        realize::topological_nodes(&tau, &sig, n_nodes).map_err(err)
    }

    /// Drop branches too small to matter, and whatever they were feeding.
    #[staticmethod]
    #[pyo3(signature = (tau, sig, psi, src, dst, share=realize::DUST_SHARE, tol=1e-12))]
    fn prune_dust(
        tau: Vec<i64>, sig: Vec<i64>, psi: Vec<f64>, src: usize, dst: usize,
        share: f64, tol: f64,
    ) -> (Vec<f64>, usize) {
        realize::prune_dust(&tau, &sig, &psi, src, dst, share, tol)
    }

    fn __len__(&self) -> usize {
        self.inner.legs.len()
    }

    // -- the wire artefact ------------------------------------------------

    /// `(target, kind, i, j, n, src_slot, dst_slot, bps)` per leg -- exactly
    /// what the on-chain router executes.
    fn wire_legs(&self) -> Vec<(String, u8, i32, i32, i32, i32, i32, i32)> {
        self.inner
            .wire_legs()
            .iter()
            .map(|leg| {
                (leg.target.clone(), leg.kind.code(), leg.i, leg.j, leg.n,
                 leg.src_slot, leg.dst_slot, leg.bps)
            })
            .collect()
    }

    // -- what each leg carries -------------------------------------------

    fn targets(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.target.clone()).collect()
    }

    fn kinds(&self) -> Vec<u8> {
        self.inner.legs.iter().map(|rl| rl.kind.code()).collect()
    }

    fn tokens_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.token_in.clone()).collect()
    }

    fn tokens_out(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.token_out.clone()).collect()
    }

    fn amounts_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.amount_in.to_string()).collect()
    }

    fn amounts_out(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.amount_out.to_string()).collect()
    }

    fn reserves_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.reserve_in.to_string()).collect()
    }

    /// `""` where the reference carries `None`: a merge leg has no arc.
    fn arc_ids(&self) -> Vec<String> {
        self.inner
            .legs
            .iter()
            .map(|rl| rl.arc_id.clone().unwrap_or_default())
            .collect()
    }

    fn pool_names(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.pool_name.clone()).collect()
    }

    /// `share_of_node, eps, impact_frac, theta, psi, cap_in, tvl_usd,
    /// gamma_live` per leg, flat.
    fn numbers(&self) -> Vec<f64> {
        self.inner
            .legs
            .iter()
            .flat_map(|rl| {
                [rl.share_of_node, rl.eps, rl.impact_frac, rl.theta, rl.psi,
                 rl.cap_in, rl.tvl_usd, rl.gamma_live]
            })
            .collect()
    }

    fn modelled(&self) -> Vec<bool> {
        self.inner.legs.iter().map(|rl| rl.modelled).collect()
    }

    fn is_conversion(&self) -> Vec<bool> {
        self.inner.legs.iter().map(|rl| rl.is_conversion()).collect()
    }

    fn is_merge(&self) -> Vec<bool> {
        self.inner.legs.iter().map(|rl| rl.is_merge()).collect()
    }

    // -- the route ---------------------------------------------------------

    #[getter]
    fn dst_slot(&self) -> usize {
        self.inner.dst_slot
    }

    #[getter]
    fn src_token(&self) -> String {
        self.inner.src_token.clone()
    }

    #[getter]
    fn dst_token(&self) -> String {
        self.inner.dst_token.clone()
    }

    #[getter]
    fn amount_in(&self) -> String {
        self.inner.amount_in.to_string()
    }

    #[getter]
    fn modelled_out(&self) -> String {
        self.inner.modelled_out.to_string()
    }

    fn slots(&self) -> Vec<(String, usize)> {
        self.inner.slots.clone()
    }

    fn node_of_slot(&self) -> Vec<(usize, usize)> {
        self.inner.node_of_slot.clone()
    }

    fn potentials(&self) -> Vec<(usize, f64)> {
        self.inner.potentials.clone()
    }

    fn paths(&self) -> Vec<Vec<String>> {
        self.inner.paths.clone()
    }

    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    fn pools_used(&self) -> Vec<String> {
        self.inner.pools_used()
    }

    /// The index of the first leg over its cap, or `None`.
    fn over_capacity(&self) -> Option<usize> {
        let target = self.inner.over_capacity()?;
        self.inner.legs.iter().position(|rl| std::ptr::eq(rl, target))
    }

    /// The pools whose legs are not an admissible element (decision 3).
    fn check_one_arc_per_pool(&self) -> Vec<String> {
        realize::check_one_arc_per_pool(&self.inner)
    }

    fn route_conductance(&self) -> f64 {
        realize::route_conductance(&self.inner)
    }

    fn max_theta(&self) -> f64 {
        realize::max_theta(&self.inner)
    }

    fn total_loss_bp(&self, price_out_per_in: f64) -> f64 {
        realize::total_loss_bp(&self.inner, price_out_per_in)
    }
}
