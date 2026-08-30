//! The model-free floor, in the browser.
//!
//! The twin of `src/naive_py.rs`. `two_step_candidates` is three calls with
//! the caller's quoting in between for the same reason it is there -- except
//! here the reason is the whole point, because in a browser the chain is
//! always on the far side of the boundary.
//!
//! Structured results cross as `js_sys::Array` rows rather than as objects:
//! the same choice `codec.rs` makes, and it keeps the shape the caller reads
//! next to the code that writes it.

use crate::nodes::NodeMap;
use crate::realize::Arcs;
use erouter_solve::naive::{self, PoolFacts as Facts};
use erouter_solve::types::{ArcKind, TypeError};
use wasm_bindgen::prelude::*;

fn err(e: TypeError) -> JsValue {
    JsError::new(&e.0).into()
}

fn kind_of(code: Option<u8>) -> Result<Option<ArcKind>, JsValue> {
    match code {
        None => Ok(None),
        Some(c) => ArcKind::from_code(c)
            .map(Some)
            .ok_or_else(|| JsError::new(&format!("no such kind: {c}")).into()),
    }
}

fn amount(text: &str) -> Result<u128, JsValue> {
    text.parse::<u128>()
        .map_err(|_| JsError::new(&format!("not an amount: {text}")).into())
}

/// `null` stays `null`: a refused probe is not a zero one.
fn quote_values(quotes: Vec<JsValue>) -> Result<Vec<Option<u128>>, JsValue> {
    quotes
        .into_iter()
        .map(|q| {
            if q.is_null() || q.is_undefined() {
                return Ok(None);
            }
            let text = q.as_string().ok_or_else(|| {
                JsValue::from(JsError::new("a quote is a decimal string or null"))
            })?;
            amount(&text).map(Some)
        })
        .collect()
}

fn candidate_rows(made: &[erouter_solve::candidates::Candidate]) -> js_sys::Array {
    let out = js_sys::Array::new();
    for c in made {
        let psi = js_sys::Array::new();
        for &v in &c.psi {
            psi.push(&JsValue::from_f64(v));
        }
        let row = js_sys::Array::new();
        row.push(&JsValue::from_str(&c.label));
        row.push(&psi);
        row.push(&JsValue::from_bool(c.certificate));
        row.push(&JsValue::from_str(&c.kind));
        row.push(&JsValue::from_str(&c.reason));
        row.push(&JsValue::from_f64(c.n_arcs as f64));
        out.push(&row);
    }
    out
}

fn probe_rows(probes: &[erouter_solve::types::Probe]) -> js_sys::Array {
    let out = js_sys::Array::new();
    for p in probes {
        let row = js_sys::Array::new();
        row.push(&JsValue::from_str(&p.pool));
        row.push(&JsValue::from_f64(p.kind.code() as f64));
        row.push(&JsValue::from_f64(p.i as f64));
        row.push(&JsValue::from_f64(p.j as f64));
        row.push(&JsValue::from_f64(p.n as f64));
        row.push(&JsValue::from_str(&p.dx.to_string()));
        out.push(&row);
    }
    out
}

fn hop_rows(hops: impl Iterator<Item = (usize, usize, usize, usize)>) -> js_sys::Array {
    let out = js_sys::Array::new();
    for (pool, i, j, middle) in hops {
        let row = js_sys::Array::new();
        for v in [pool, i, j, middle] {
            row.push(&JsValue::from_f64(v as f64));
        }
        out.push(&row);
    }
    out
}

/// The pool facts the two generators read, built once and reused.
#[wasm_bindgen]
#[derive(Default)]
pub struct PoolFacts {
    pub(crate) inner: Vec<Facts>,
}

#[wasm_bindgen]
impl PoolFacts {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PoolFacts {
        PoolFacts::default()
    }

    #[wasm_bindgen(js_name = length)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// One pool. `kind` is `undefined` where no swap dialect is resolved --
    /// the reference's `swap_kind is None`, skipped rather than defaulted.
    /// Balances are decimal strings, as every `uint256` is here.
    pub fn add(
        &mut self,
        address: &str,
        name: &str,
        kind: Option<u8>,
        coins: Vec<String>,
        decimals: Vec<u32>,
        balances: Vec<String>,
        tvl_usd: f64,
    ) -> Result<usize, JsValue> {
        let parsed: Result<Vec<u128>, JsValue> =
            balances.iter().map(|b| amount(b)).collect();
        self.inner.push(Facts {
            address: address.to_string(),
            name: name.to_string(),
            kind: kind_of(kind)?,
            coins,
            decimals,
            balances: parsed?,
            tvl_usd,
        });
        Ok(self.inner.len() - 1)
    }
}

/// Round A's plan: what to quote, and what each answer will mean.
#[wasm_bindgen]
pub struct PlanA {
    inner: naive::PlanA,
}

#[wasm_bindgen]
impl PlanA {
    /// `[pool, kind, i, j, nCoins, dx]` per probe, in order.
    pub fn probes(&self) -> js_sys::Array {
        probe_rows(&self.inner.probes)
    }

    #[wasm_bindgen(js_name = length)]
    pub fn length(&self) -> usize {
        self.inner.probes.len()
    }

    /// `[pool, i, j, middle]` per probe -- the hop each answer is about.
    pub fn hops(&self) -> js_sys::Array {
        hop_rows(self.inner.hops.iter().map(|h| (h.pool, h.i, h.j, h.middle)))
    }
}

/// Round B's plan, plus the winning first hop into every middle.
#[wasm_bindgen]
pub struct PlanB {
    inner: naive::PlanB,
}

#[wasm_bindgen]
impl PlanB {
    pub fn probes(&self) -> js_sys::Array {
        probe_rows(&self.inner.probes)
    }

    #[wasm_bindgen(js_name = length)]
    pub fn length(&self) -> usize {
        self.inner.probes.len()
    }

    pub fn hops(&self) -> js_sys::Array {
        hop_rows(self.inner.hops.iter().map(|h| (h.pool, h.i, h.j, h.middle)))
    }

    /// `[middle, canonical, pool, i, j]`, `canonical` a decimal string.
    #[wasm_bindgen(js_name = bestFirst)]
    pub fn best_first(&self) -> js_sys::Array {
        let out = js_sys::Array::new();
        for b in &self.inner.best_first {
            let row = js_sys::Array::new();
            row.push(&JsValue::from_f64(b.middle as f64));
            row.push(&JsValue::from_str(&b.canonical.to_string()));
            for v in [b.pool, b.i, b.j] {
                row.push(&JsValue::from_f64(v as f64));
            }
            out.push(&row);
        }
        out
    }
}

/// `[candidates, arcs]` -- one-leg candidates through every pool holding both
/// tokens.
#[wasm_bindgen(js_name = directCandidates)]
pub fn direct_candidates(
    pools: &PoolFacts,
    nodes: &NodeMap,
    nu: Vec<f64>,
    src_token: &str,
    dst_token: &str,
) -> Result<js_sys::Array, JsValue> {
    let (made, arcs) =
        naive::direct_candidates(&pools.inner, &nodes.inner, &nu, src_token, dst_token)
            .map_err(err)?;
    Ok(js_sys::Array::of2(
        &candidate_rows(&made),
        &JsValue::from(Arcs::from_parts(arcs)),
    ))
}

/// Round A: `src -> M`, for every middle that can reach `dst` at all.
#[wasm_bindgen(js_name = twoStepPlanFirst)]
pub fn two_step_plan_first(
    pools: &PoolFacts,
    nodes: &NodeMap,
    src_token: &str,
    dst_token: &str,
    amount_in: &str,
) -> Result<PlanA, JsValue> {
    let inner = naive::two_step_plan_first(
        &pools.inner, &nodes.inner, src_token, dst_token, amount(amount_in)?,
    )
    .map_err(err)?;
    Ok(PlanA { inner })
}

/// Rank round A's answers and plan round B. `quotes[k]` answers
/// `plan.probes()[k]`, `null` where the chain refused.
#[wasm_bindgen(js_name = twoStepRank)]
pub fn two_step_rank(
    pools: &PoolFacts,
    nodes: &NodeMap,
    nu: Vec<f64>,
    plan: &PlanA,
    quotes: Vec<JsValue>,
    dst_token: &str,
    limit: usize,
) -> Result<PlanB, JsValue> {
    let inner = naive::two_step_rank(
        &pools.inner, &nodes.inner, &nu, &plan.inner, &quote_values(quotes)?, dst_token,
        limit,
    )
    .map_err(err)?;
    Ok(PlanB { inner })
}

/// `[candidates, [arcs, ...]]` -- the chains, best `limit` by final output.
#[wasm_bindgen(js_name = twoStepBuild)]
pub fn two_step_build(
    pools: &PoolFacts,
    nodes: &NodeMap,
    nu: Vec<f64>,
    plan: &PlanB,
    quotes: Vec<JsValue>,
    src_token: &str,
    dst_token: &str,
    limit: usize,
) -> Result<js_sys::Array, JsValue> {
    let (made, chains) = naive::two_step_build(
        &pools.inner, &nodes.inner, &nu, &plan.inner, &quote_values(quotes)?, src_token,
        dst_token, limit,
    )
    .map_err(err)?;
    let arcs = js_sys::Array::new();
    for chain in chains {
        arcs.push(&JsValue::from(Arcs::from_parts(chain)));
    }
    Ok(js_sys::Array::of2(&candidate_rows(&made), &arcs))
}
