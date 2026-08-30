//! The node map, in the browser.
//!
//! The twin of `src/nodes_py.rs`. Rates and amounts cross as decimal strings
//! for the same reason they do there: a conversion rate is exact 256-bit
//! integer arithmetic, and JS has no other lossless spelling for it that a
//! typed array can carry.

use erouter_solve::nodes::{self, Conversion, ConversionKind, NodeError, NodeMap as Inner};
use wasm_bindgen::prelude::*;

fn err(e: NodeError) -> JsValue {
    JsError::new(&e.0).into()
}

fn kind_of(name: &str) -> Result<ConversionKind, JsValue> {
    ConversionKind::parse(name).ok_or_else(|| JsError::new(&format!("unknown kind: {name}")).into())
}

#[wasm_bindgen]
pub struct NodeMap {
    inner: Inner,
}

impl Default for NodeMap {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl NodeMap {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        NodeMap { inner: Inner::new() }
    }

    #[wasm_bindgen(js_name = addToken)]
    pub fn add_token(&mut self, address: &str, symbol: Option<String>, decimals: Option<u32>) -> usize {
        self.inner
            .add_token(address, symbol.as_deref().unwrap_or(""), decimals.unwrap_or(18))
    }

    /// Fold `token` into the node of `canonical`.
    pub fn merge(
        &mut self,
        kind: &str,
        token: &str,
        canonical: &str,
        rate_num: Option<String>,
        rate_den: Option<String>,
        target: Option<String>,
    ) -> Result<(), JsValue> {
        let conversion = Conversion::new(kind_of(kind)?, token, canonical)
            .with_rate_str(
                rate_num.as_deref().unwrap_or("1"),
                rate_den.as_deref().unwrap_or("1"),
            )
            .map_err(err)?
            .with_target(target.as_deref().unwrap_or(""));
        self.inner.merge(conversion).map_err(err)
    }

    pub fn node(&self, token: &str) -> Result<usize, JsValue> {
        self.inner
            .node(token)
            .ok_or_else(|| JsError::new(token).into())
    }

    pub fn has(&self, token: &str) -> bool {
        self.inner.has(token)
    }

    pub fn canonical(&self, token: &str) -> Result<String, JsValue> {
        self.inner
            .canonical(token)
            .map(str::to_string)
            .ok_or_else(|| JsError::new(token).into())
    }

    pub fn rate(&self, token: &str) -> f64 {
        self.inner.rate(token)
    }

    #[wasm_bindgen(js_name = toCanonicalWei)]
    pub fn to_canonical_wei(&self, token: &str, amount: &str) -> Result<String, JsValue> {
        self.inner.to_canonical_wei_str(token, amount).map_err(err)
    }

    #[wasm_bindgen(js_name = fromCanonicalWei)]
    pub fn from_canonical_wei(&self, token: &str, amount: &str) -> Result<String, JsValue> {
        self.inner.from_canonical_wei_str(token, amount).map_err(err)
    }

    pub fn symbol(&self, token: &str) -> String {
        self.inner.symbol(token)
    }

    pub fn decimals(&self, token: &str) -> u32 {
        self.inner.decimals(token)
    }

    #[wasm_bindgen(js_name = nodeSymbol)]
    pub fn node_symbol(&self, node: usize) -> String {
        self.inner.node_symbol(node)
    }

    #[wasm_bindgen(js_name = nNodes)]
    pub fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[wasm_bindgen(js_name = mergedNodes)]
    pub fn merged_nodes(&self) -> Vec<u32> {
        self.inner.merged_nodes().into_iter().map(|k| k as u32).collect()
    }

    /// The members of one node, in the order they joined it.
    #[wasm_bindgen(js_name = tokensOf)]
    pub fn tokens_of(&self, node: usize) -> Vec<String> {
        self.inner.tokens_of.get(node).cloned().unwrap_or_default()
    }

    /// `[kind, canonical, rate_num, rate_den, target]`, or an empty array
    /// where the token is canonical.
    pub fn conversion(&self, token: &str) -> Vec<String> {
        match self.lookup(token) {
            Some(found) => vec![
                found.kind.as_str().to_string(),
                found.canonical.clone(),
                found.rate_num.to_string(),
                found.rate_den.to_string(),
                found.target.clone(),
            ],
            None => Vec::new(),
        }
    }

    /// `[token -> canonical, canonical -> token]` as ArcKind codes, or empty.
    #[wasm_bindgen(js_name = conversionKinds)]
    pub fn conversion_kinds(&self, token: &str) -> Vec<u8> {
        match self.lookup(token) {
            Some(found) => vec![found.forward_kind().code(), found.reverse_kind().code()],
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = isAlias)]
    pub fn is_alias(&self, token: &str) -> bool {
        self.lookup(token).is_some_and(Conversion::is_alias)
    }

    /// Re-express an arc's derivatives in canonical units (§3.1). Returns
    /// `[a, B]`.
    pub fn rescale(a: f64, b: f64, rate_in: f64, rate_out: f64) -> Result<Vec<f64>, JsValue> {
        let (a, b) = nodes::rescale(a, b, rate_in, rate_out).map_err(err)?;
        Ok(vec![a, b])
    }
}

impl NodeMap {
    fn lookup(&self, token: &str) -> Option<&Conversion> {
        let lowered = token.to_ascii_lowercase();
        self.inner
            .conversion
            .get(token)
            .or_else(|| self.inner.conversion.get(&lowered))
    }
}
