//! The node map, across the PyO3 boundary.
//!
//! Rates cross as decimal strings, the way every other 256-bit number in these
//! bindings does: a conversion rate is `canonical wei per 10**decimals`, which
//! is exact and routinely past what a float holds. See `nodes.rs`.

use crate::nodes::{self, Conversion, ConversionKind, NodeError, NodeMap as Inner};
use pyo3::prelude::*;

fn value(e: NodeError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.0)
}

fn kind_of(name: &str) -> PyResult<ConversionKind> {
    ConversionKind::parse(name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown kind: {name}")))
}

#[pyclass]
pub struct NodeMap {
    pub(crate) inner: Inner,
}

#[pymethods]
impl NodeMap {
    #[new]
    fn new() -> Self {
        NodeMap { inner: Inner::new() }
    }

    #[pyo3(signature = (address, symbol="", decimals=18))]
    fn add_token(&mut self, address: &str, symbol: &str, decimals: u32) -> usize {
        self.inner.add_token(address, symbol, decimals)
    }

    /// Fold `token` into the node of `canonical`.
    #[pyo3(signature = (kind, token, canonical, rate_num="1", rate_den="1", target=""))]
    fn merge(
        &mut self,
        kind: &str,
        token: &str,
        canonical: &str,
        rate_num: &str,
        rate_den: &str,
        target: &str,
    ) -> PyResult<()> {
        let conversion = Conversion::new(kind_of(kind)?, token, canonical)
            .with_rate_str(rate_num, rate_den)
            .map_err(value)?
            .with_target(target);
        self.inner
            .merge(conversion)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.0))
    }

    fn node(&self, token: &str) -> PyResult<usize> {
        self.inner
            .node(token)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(token.to_string()))
    }

    fn has(&self, token: &str) -> bool {
        self.inner.has(token)
    }

    fn canonical(&self, token: &str) -> PyResult<String> {
        self.inner
            .canonical(token)
            .map(str::to_string)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(token.to_string()))
    }

    fn rate(&self, token: &str) -> f64 {
        self.inner.rate(token)
    }

    fn to_canonical_wei(&self, token: &str, amount: &str) -> PyResult<String> {
        self.inner.to_canonical_wei_str(token, amount).map_err(value)
    }

    fn from_canonical_wei(&self, token: &str, amount: &str) -> PyResult<String> {
        self.inner.from_canonical_wei_str(token, amount).map_err(value)
    }

    fn symbol(&self, token: &str) -> String {
        self.inner.symbol(token)
    }

    fn decimals(&self, token: &str) -> u32 {
        self.inner.decimals(token)
    }

    fn node_symbol(&self, node: usize) -> PyResult<String> {
        crate::py::node("node", node, self.inner.n_nodes())?;
        Ok(self.inner.node_symbol(node))
    }

    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    fn merged_nodes(&self) -> Vec<usize> {
        self.inner.merged_nodes()
    }

    /// The members of one node, in the order they joined it.
    fn tokens_of(&self, node: usize) -> Vec<String> {
        self.inner.tokens_of.get(node).cloned().unwrap_or_default()
    }

    /// `(kind, canonical, rate_num, rate_den, target)`, or `None` where the
    /// token is canonical.
    fn conversion(&self, token: &str) -> Option<(String, String, String, String, String)> {
        let lowered = token.to_ascii_lowercase();
        let found = self
            .inner
            .conversion
            .get(token)
            .or_else(|| self.inner.conversion.get(&lowered))?;
        Some((
            found.kind.as_str().to_string(),
            found.canonical.clone(),
            found.rate_num.to_string(),
            found.rate_den.to_string(),
            found.target.clone(),
        ))
    }

    /// `token -> canonical` and `canonical -> token` as ArcKind codes.
    fn conversion_kinds(&self, token: &str) -> Option<(u8, u8)> {
        let lowered = token.to_ascii_lowercase();
        let found = self
            .inner
            .conversion
            .get(token)
            .or_else(|| self.inner.conversion.get(&lowered))?;
        Some((found.forward_kind().code(), found.reverse_kind().code()))
    }

    fn is_alias(&self, token: &str) -> bool {
        let lowered = token.to_ascii_lowercase();
        self.inner
            .conversion
            .get(token)
            .or_else(|| self.inner.conversion.get(&lowered))
            .is_some_and(Conversion::is_alias)
    }

    /// Re-express an arc's derivatives in canonical units (§3.1).
    #[staticmethod]
    fn rescale(a: f64, b: f64, rate_in: f64, rate_out: f64) -> PyResult<(f64, f64)> {
        nodes::rescale(a, b, rate_in, rate_out)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))
    }
}
