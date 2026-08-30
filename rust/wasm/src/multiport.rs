//! Multi-port elements, in the browser.
//!
//! The twin of `src/multiport_py.rs`. Ports cross as two flat arrays -- coins
//! and shares -- rather than as pairs, because that is what a typed array
//! carries; `Pools.elementEvaluate` takes the same shape.

use erouter_solve::multiport::{self, MultiPort, MultiPortError, Port, BPS, LP};
use erouter_solve::types::ArcKind;
use wasm_bindgen::prelude::*;

fn err(e: MultiPortError) -> JsValue {
    JsError::new(&e.0).into()
}

fn kind_of(code: u8) -> Result<ArcKind, JsValue> {
    ArcKind::from_code(code).ok_or_else(|| JsError::new(&format!("no such kind: {code}")).into())
}

pub(crate) fn ports(coins: &[i32], bps: &[i64]) -> Vec<Port> {
    coins.iter().zip(bps.iter()).map(|(&c, &b)| Port::new(c, b)).collect()
}

/// `inputs -> outputs` through one pool, priced on advancing state.
#[wasm_bindgen]
pub struct Element {
    inner: MultiPort,
}

#[wasm_bindgen]
impl Element {
    /// Ports cross as parallel `coins` and `bps` arrays, `LP` for the LP token.
    #[wasm_bindgen(constructor)]
    pub fn new(
        pool: &str,
        n_coins: i32,
        in_coins: Vec<i32>,
        in_bps: Vec<i64>,
        out_coins: Vec<i32>,
        out_bps: Vec<i64>,
    ) -> Result<Element, JsValue> {
        if in_coins.len() != in_bps.len() || out_coins.len() != out_bps.len() {
            return Err(JsError::new("each side needs one share per coin").into());
        }
        let inner = MultiPort::new(
            pool,
            n_coins,
            ports(&in_coins, &in_bps),
            ports(&out_coins, &out_bps),
        )
        .map_err(err)?;
        Ok(Element { inner })
    }

    /// The element `(kind, i, j)` triples on one pool form, or why they do not.
    #[wasm_bindgen(js_name = fromTriples)]
    pub fn from_triples(
        pool: &str,
        n_coins: i32,
        kinds: Vec<u8>,
        i: Vec<i32>,
        j: Vec<i32>,
    ) -> Result<Element, JsValue> {
        if kinds.len() != i.len() || kinds.len() != j.len() {
            return Err(JsError::new("kinds, i and j must be the same length").into());
        }
        let mut triples = Vec::with_capacity(kinds.len());
        for k in 0..kinds.len() {
            triples.push((kind_of(kinds[k])?, i[k], j[k]));
        }
        let inner = multiport::element_from(pool, n_coins, &triples).map_err(err)?;
        Ok(Element { inner })
    }

    /// `[input port, output port]` for one leg, `LP` for the LP token.
    #[wasm_bindgen(js_name = portsOf)]
    pub fn ports_of(kind: u8, i: i32, j: i32) -> Result<Vec<i32>, JsValue> {
        let (source, sink) = multiport::ports_of(kind_of(kind)?, i, j).map_err(err)?;
        Ok(vec![source, sink])
    }

    #[wasm_bindgen(getter)]
    pub fn pool(&self) -> String {
        self.inner.pool.clone()
    }

    #[wasm_bindgen(getter, js_name = nCoins)]
    pub fn n_coins(&self) -> i32 {
        self.inner.n_coins
    }

    #[wasm_bindgen(getter, js_name = inputCoins)]
    pub fn input_coins(&self) -> Vec<i32> {
        self.inner.inputs.iter().map(|p| p.coin).collect()
    }

    #[wasm_bindgen(getter, js_name = inputBps)]
    pub fn input_bps(&self) -> Vec<i64> {
        self.inner.inputs.iter().map(|p| p.bps).collect()
    }

    #[wasm_bindgen(getter, js_name = outputCoins)]
    pub fn output_coins(&self) -> Vec<i32> {
        self.inner.outputs.iter().map(|p| p.coin).collect()
    }

    #[wasm_bindgen(getter, js_name = outputBps)]
    pub fn output_bps(&self) -> Vec<i64> {
        self.inner.outputs.iter().map(|p| p.bps).collect()
    }

    #[wasm_bindgen(getter)]
    pub fn ports(&self) -> usize {
        self.inner.ports()
    }

    /// A port on the LP token rather than on one of the pool's coins.
    pub fn lp() -> i32 {
        LP
    }

    pub fn bps() -> i64 {
        BPS
    }

    /// The reference exposes `inputs`/`outputs` as pairs; here they are the
    /// four arrays above, and this says so in one place rather than in each
    /// caller.
    #[wasm_bindgen(js_name = inputs)]
    pub fn inputs(&self) -> Vec<i64> {
        interleave(&self.inner.inputs)
    }

    #[wasm_bindgen(js_name = outputs)]
    pub fn outputs(&self) -> Vec<i64> {
        interleave(&self.inner.outputs)
    }
}

fn interleave(ports: &[Port]) -> Vec<i64> {
    ports.iter().flat_map(|p| [p.coin as i64, p.bps]).collect()
}
