//! The model-free floor, across the PyO3 boundary.
//!
//! `two_step_candidates` quotes twice from inside one function in the
//! reference. Here it is three calls with the caller's probing in between, so
//! the plans cross as opaque handles rather than being rebuilt each stage --
//! `PlanA` and `PlanB` go back exactly as they came out.

use crate::naive::{self, PoolFacts as Facts};
use crate::nodes_py::NodeMap;
use crate::realize_py::Arcs;
use crate::types::ArcKind;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn value(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn kind_of(code: Option<u8>) -> PyResult<Option<ArcKind>> {
    match code {
        None => Ok(None),
        Some(c) => ArcKind::from_code(c).map(Some).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("no such kind: {c}"))
        }),
    }
}

/// The pool facts the two generators read, built once and reused.
///
/// Named for the fields rather than the objects: `Pools` is already the pool
/// *models*, and this is the handful of columns the floor needs.
#[pyclass]
#[derive(Default)]
pub struct PoolFacts {
    pub(crate) inner: Vec<Facts>,
}

#[pymethods]
impl PoolFacts {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// One pool. `kind` is `None` where no swap dialect is resolved, which is
    /// the reference's `swap_kind is None`: skipped, never defaulted.
    ///
    /// Balances cross as decimal strings for the same reason every other
    /// integer in these bindings does -- they are `uint256` on the chain and
    /// only accidentally fit anything narrower.
    #[pyo3(signature = (address, name, kind, coins, decimals, balances, tvl_usd=0.0))]
    fn add(
        &mut self, address: &str, name: &str, kind: Option<u8>, coins: Vec<String>,
        decimals: Vec<u32>, balances: Vec<String>, tvl_usd: f64,
    ) -> PyResult<usize> {
        let parsed: Result<Vec<u128>, _> =
            balances.iter().map(|b| b.parse::<u128>()).collect();
        self.inner.push(Facts {
            address: address.to_string(),
            name: name.to_string(),
            kind: kind_of(kind)?,
            coins,
            decimals,
            balances: parsed.map_err(value)?,
            tvl_usd,
        });
        Ok(self.inner.len() - 1)
    }
}

/// `[{label, psi, certificate, kind, reason, n_arcs}]` -- the six fields the
/// reference sets on a model-free candidate, and no more.
fn candidates<'py>(
    py: Python<'py>, made: &[crate::candidates::Candidate],
) -> PyResult<Bound<'py, PyList>> {
    let rows: PyResult<Vec<Bound<'py, PyDict>>> = made
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("label", &c.label)?;
            d.set_item("psi", c.psi.clone())?;
            d.set_item("certificate", c.certificate)?;
            d.set_item("kind", &c.kind)?;
            d.set_item("reason", &c.reason)?;
            d.set_item("n_arcs", c.n_arcs)?;
            Ok(d)
        })
        .collect();
    PyList::new(py, rows?)
}

/// `(pool, kind, i, j, n_coins, dx)` per probe, `dx` a decimal string.
type ProbeRow = (String, u8, i32, i32, i32, String);

fn probe_rows(probes: &[crate::types::Probe]) -> Vec<ProbeRow> {
    probes
        .iter()
        .map(|p| (p.pool.clone(), p.kind.code(), p.i, p.j, p.n, p.dx.to_string()))
        .collect()
}

fn quote_values(quotes: Vec<Option<String>>) -> PyResult<Vec<Option<u128>>> {
    quotes
        .into_iter()
        .map(|q| match q {
            None => Ok(None),
            Some(text) => text.parse::<u128>().map(Some).map_err(value),
        })
        .collect()
}

/// Round A's plan: what to quote, and what each answer will mean.
#[pyclass]
pub struct PlanA {
    inner: naive::PlanA,
}

#[pymethods]
impl PlanA {
    /// The probes to put to the chain, in order.
    fn probes(&self) -> Vec<ProbeRow> {
        probe_rows(&self.inner.probes)
    }

    fn __len__(&self) -> usize {
        self.inner.probes.len()
    }

    /// `(pool, i, j, middle)` per probe -- the hop each answer is about.
    fn hops(&self) -> Vec<(usize, usize, usize, usize)> {
        self.inner.hops.iter().map(|h| (h.pool, h.i, h.j, h.middle)).collect()
    }
}

/// Round B's plan, plus the winning first hop into every middle.
#[pyclass]
pub struct PlanB {
    inner: naive::PlanB,
}

#[pymethods]
impl PlanB {
    fn probes(&self) -> Vec<ProbeRow> {
        probe_rows(&self.inner.probes)
    }

    fn __len__(&self) -> usize {
        self.inner.probes.len()
    }

    fn hops(&self) -> Vec<(usize, usize, usize, usize)> {
        self.inner.hops.iter().map(|h| (h.pool, h.i, h.j, h.middle)).collect()
    }

    /// `(middle, canonical, pool, i, j)`, `canonical` a decimal string.
    fn best_first(&self) -> Vec<(usize, String, usize, usize, usize)> {
        self.inner
            .best_first
            .iter()
            .map(|b| (b.middle, b.canonical.to_string(), b.pool, b.i, b.j))
            .collect()
    }
}

/// One-leg candidates through every pool holding both tokens.
#[pyfunction]
pub fn direct_candidates<'py>(
    py: Python<'py>, pools: PyRef<'_, PoolFacts>, nodes: PyRef<'_, NodeMap>, nu: Vec<f64>,
    src_token: &str, dst_token: &str,
) -> PyResult<(Bound<'py, PyList>, Arcs)> {
    let (made, arcs) =
        naive::direct_candidates(&pools.inner, &nodes.inner, &nu, src_token, dst_token)
            .map_err(value)?;
    Ok((candidates(py, &made)?, Arcs { inner: arcs }))
}

/// Round A: `src -> M`, for every middle that can reach `dst` at all.
#[pyfunction]
pub fn two_step_plan_first(
    pools: PyRef<'_, PoolFacts>, nodes: PyRef<'_, NodeMap>, src_token: &str, dst_token: &str,
    amount_in: &str,
) -> PyResult<PlanA> {
    let amount = amount_in.parse::<u128>().map_err(value)?;
    let inner =
        naive::two_step_plan_first(&pools.inner, &nodes.inner, src_token, dst_token, amount)
            .map_err(value)?;
    Ok(PlanA { inner })
}

/// Rank round A's answers and plan round B. `quotes[k]` answers
/// `plan.probes()[k]`, `None` where the chain refused.
#[pyfunction]
#[pyo3(signature = (pools, nodes, nu, plan, quotes, dst_token, limit=6))]
pub fn two_step_rank(
    pools: PyRef<'_, PoolFacts>, nodes: PyRef<'_, NodeMap>, nu: Vec<f64>,
    plan: PyRef<'_, PlanA>, quotes: Vec<Option<String>>, dst_token: &str, limit: usize,
) -> PyResult<PlanB> {
    let inner = naive::two_step_rank(
        &pools.inner, &nodes.inner, &nu, &plan.inner, &quote_values(quotes)?, dst_token,
        limit,
    )
    .map_err(value)?;
    Ok(PlanB { inner })
}

/// Build the chains from round B's answers, best `limit` by final output.
#[pyfunction]
#[pyo3(signature = (pools, nodes, nu, plan, quotes, src_token, dst_token, limit=6))]
pub fn two_step_build<'py>(
    py: Python<'py>, pools: PyRef<'_, PoolFacts>, nodes: PyRef<'_, NodeMap>, nu: Vec<f64>,
    plan: PyRef<'_, PlanB>, quotes: Vec<Option<String>>, src_token: &str,
    dst_token: &str, limit: usize,
) -> PyResult<(Bound<'py, PyList>, Vec<Arcs>)> {
    let (made, chains) = naive::two_step_build(
        &pools.inner, &nodes.inner, &nu, &plan.inner, &quote_values(quotes)?, src_token,
        dst_token, limit,
    )
    .map_err(value)?;
    let arcs = chains.into_iter().map(|inner| Arcs { inner }).collect();
    Ok((candidates(py, &made)?, arcs))
}
