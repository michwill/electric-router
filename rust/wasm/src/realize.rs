//! Realisation, in the browser.
//!
//! The twin of `src/realize_py.rs`. The one shape difference is that a pair
//! has no typed-array form, so what PyO3 returns as `[(token, slot)]` comes
//! back here as two parallel arrays.

use crate::nodes::NodeMap;
use erouter_solve::realize::{self, RealizedRoute};
use erouter_solve::types::{ArcKind, PoolArc};
use wasm_bindgen::prelude::*;

fn err(e: realize::RealizationError) -> JsValue {
    JsError::new(&e.0).into()
}

fn kind_of(code: u8) -> Result<ArcKind, JsValue> {
    ArcKind::from_code(code).ok_or_else(|| JsError::new(&format!("no such kind: {code}")).into())
}

/// The arcs a route is realised from, built one at a time.
#[wasm_bindgen]
#[derive(Default)]
pub struct Arcs {
    pub(crate) inner: Vec<PoolArc>,
}

#[wasm_bindgen]
impl Arcs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(js_name = length)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// One arc, with the fields realisation actually reads. Returns its index.
    ///
    /// `reserveIn` crosses as a decimal string: it is a wei-scale reserve and
    /// a `number` would lose its low digits.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self, id: &str, pool: &str, kind: u8, i: i32, j: i32, n_coins: i32,
        token_in: &str, token_out: &str, tau: usize, sigma: usize,
        a: f64, b: f64, cap: f64, g: f64, eps: f64, reserve_in: &str,
        decimals_in: u32, tvl_usd: f64, gamma_live: f64, note: Option<String>,
        calib_delta: Option<f64>, decimals_out: Option<u32>,
    ) -> Result<usize, JsValue> {
        let reserve = reserve_in
            .parse::<u128>()
            .map_err(|_| JsError::new(&format!("not a u128: {reserve_in}")))?;
        let mut arc = PoolArc::new(
            id.to_string(), pool.to_string(), kind_of(kind)?, i, j, n_coins,
            token_in.to_string(), token_out.to_string(), tau, sigma,
        );
        arc.a = a;
        arc.b = b;
        arc.cap = cap;
        arc.g = g;
        arc.eps = eps;
        arc.reserve_in = reserve;
        arc.decimals_in = decimals_in;
        arc.tvl_usd = tvl_usd;
        arc.gamma_live = gamma_live;
        arc.note = note.unwrap_or_default();
        // The refit reads both -- see the PyO3 twin.
        arc.calib_delta = calib_delta.unwrap_or(0.0);
        arc.decimals_out = decimals_out.unwrap_or(18);
        self.inner.push(arc);
        Ok(self.inner.len() - 1)
    }
}

/// A solved flow, realised into legs.
#[wasm_bindgen]
pub struct Route {
    pub(crate) inner: RealizedRoute,
}

#[wasm_bindgen]
impl Route {
    /// Build the executable leg list from a solved flow.
    #[allow(clippy::too_many_arguments)]
    pub fn realize(
        arcs: &Arcs, psi: Vec<f64>, nu: Vec<f64>, nodes: &NodeMap,
        src_token: &str, dst_token: &str, amount_in: &str,
        potentials: Option<Vec<f64>>,
    ) -> Result<Route, JsValue> {
        let amount = amount_in
            .parse()
            .map_err(|_| JsError::new(&format!("not a u256: {amount_in}")))?;
        let inner = realize::realize(
            &arcs.inner, &psi, &nu, &nodes.inner, src_token, dst_token, amount,
            potentials.as_deref(),
        )
        .map_err(err)?;
        Ok(Route { inner })
    }

    /// The route between two tokens of the *same* node: the conversion itself.
    #[wasm_bindgen(js_name = conversionRoute)]
    pub fn conversion_route(
        nodes: &NodeMap, src_token: &str, dst_token: &str, amount_in: &str,
    ) -> Result<Route, JsValue> {
        let amount = amount_in
            .parse()
            .map_err(|_| JsError::new(&format!("not a u256: {amount_in}")))?;
        let inner = realize::conversion_route(&nodes.inner, src_token, dst_token, amount)
            .map_err(err)?;
        Ok(Route { inner })
    }

    /// Kahn's algorithm over the active arcs. Refuses a cycle.
    #[wasm_bindgen(js_name = topologicalNodes)]
    pub fn topological_nodes(
        tau: Vec<i64>, sig: Vec<i64>, n_nodes: usize,
    ) -> Result<Vec<u32>, JsValue> {
        realize::topological_nodes(&tau, &sig, n_nodes)
            .map(|v| v.into_iter().map(|k| k as u32).collect())
            .map_err(err)
    }

    /// Drop branches too small to matter, and whatever they were feeding.
    /// Returns the pruned flow; `prunedCount` says how many arcs went.
    #[wasm_bindgen(js_name = pruneDust)]
    pub fn prune_dust(
        tau: Vec<i64>, sig: Vec<i64>, psi: Vec<f64>, src: usize, dst: usize,
        share: Option<f64>, tol: Option<f64>,
    ) -> PruneOut {
        let (flow, removed) = realize::prune_dust(
            &tau, &sig, &psi, src, dst,
            share.unwrap_or(realize::DUST_SHARE), tol.unwrap_or(1e-12),
        );
        PruneOut { flow, removed }
    }

    #[wasm_bindgen(js_name = length)]
    pub fn len(&self) -> usize {
        self.inner.legs.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.legs.is_empty()
    }

    // -- the wire artefact ------------------------------------------------

    /// `target` per leg, with the numbers in `wireNumbers` -- exactly what the
    /// on-chain router executes.
    #[wasm_bindgen(js_name = wireLegs)]
    pub fn wire_legs(&self) -> Vec<String> {
        self.inner.wire_legs().iter().map(|leg| leg.target.clone()).collect()
    }

    /// `kind, i, j, n, src_slot, dst_slot, bps` per leg, flat.
    #[wasm_bindgen(js_name = wireNumbers)]
    pub fn wire_numbers(&self) -> Vec<i32> {
        self.inner
            .wire_legs()
            .iter()
            .flat_map(|leg| {
                [leg.kind.code() as i32, leg.i, leg.j, leg.n,
                 leg.src_slot, leg.dst_slot, leg.bps]
            })
            .collect()
    }

    // -- what each leg carries -------------------------------------------

    pub fn targets(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.target.clone()).collect()
    }

    pub fn kinds(&self) -> Vec<u8> {
        self.inner.legs.iter().map(|rl| rl.kind.code()).collect()
    }

    #[wasm_bindgen(js_name = tokensIn)]
    pub fn tokens_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.token_in.clone()).collect()
    }

    #[wasm_bindgen(js_name = tokensOut)]
    pub fn tokens_out(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.token_out.clone()).collect()
    }

    #[wasm_bindgen(js_name = amountsIn)]
    pub fn amounts_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.amount_in.to_string()).collect()
    }

    #[wasm_bindgen(js_name = amountsOut)]
    pub fn amounts_out(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.amount_out.to_string()).collect()
    }

    #[wasm_bindgen(js_name = reservesIn)]
    pub fn reserves_in(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.reserve_in.to_string()).collect()
    }

    /// `""` where the reference carries `None`: a merge leg has no arc.
    #[wasm_bindgen(js_name = arcIds)]
    pub fn arc_ids(&self) -> Vec<String> {
        self.inner
            .legs
            .iter()
            .map(|rl| rl.arc_id.clone().unwrap_or_default())
            .collect()
    }

    #[wasm_bindgen(js_name = poolNames)]
    pub fn pool_names(&self) -> Vec<String> {
        self.inner.legs.iter().map(|rl| rl.pool_name.clone()).collect()
    }

    /// `share_of_node, eps, impact_frac, theta, psi, cap_in, tvl_usd,
    /// gamma_live` per leg, flat.
    pub fn numbers(&self) -> Vec<f64> {
        self.inner
            .legs
            .iter()
            .flat_map(|rl| {
                [rl.share_of_node, rl.eps, rl.impact_frac, rl.theta, rl.psi,
                 rl.cap_in, rl.tvl_usd, rl.gamma_live]
            })
            .collect()
    }

    pub fn modelled(&self) -> Vec<u8> {
        self.inner.legs.iter().map(|rl| u8::from(rl.modelled)).collect()
    }

    #[wasm_bindgen(js_name = isConversion)]
    pub fn is_conversion(&self) -> Vec<u8> {
        self.inner.legs.iter().map(|rl| u8::from(rl.is_conversion())).collect()
    }

    #[wasm_bindgen(js_name = isMerge)]
    pub fn is_merge(&self) -> Vec<u8> {
        self.inner.legs.iter().map(|rl| u8::from(rl.is_merge())).collect()
    }

    // -- the route ---------------------------------------------------------

    #[wasm_bindgen(getter, js_name = dstSlot)]
    pub fn dst_slot(&self) -> usize {
        self.inner.dst_slot
    }

    #[wasm_bindgen(getter, js_name = srcToken)]
    pub fn src_token(&self) -> String {
        self.inner.src_token.clone()
    }

    #[wasm_bindgen(getter, js_name = dstToken)]
    pub fn dst_token(&self) -> String {
        self.inner.dst_token.clone()
    }

    #[wasm_bindgen(getter, js_name = amountIn)]
    pub fn amount_in(&self) -> String {
        self.inner.amount_in.to_string()
    }

    #[wasm_bindgen(getter, js_name = modelledOut)]
    pub fn modelled_out(&self) -> String {
        self.inner.modelled_out.to_string()
    }

    /// The slot tokens, in the order the slots were opened; `slotIndices`
    /// carries the numbers, because a pair has no typed-array form.
    pub fn slots(&self) -> Vec<String> {
        self.inner.slots.iter().map(|(token, _)| token.clone()).collect()
    }

    #[wasm_bindgen(js_name = slotIndices)]
    pub fn slot_indices(&self) -> Vec<u32> {
        self.inner.slots.iter().map(|(_, k)| *k as u32).collect()
    }

    #[wasm_bindgen(js_name = nodeOfSlot)]
    pub fn node_of_slot(&self) -> Vec<u32> {
        self.inner
            .node_of_slot
            .iter()
            .flat_map(|&(slot, node)| [slot as u32, node as u32])
            .collect()
    }

    /// The touched nodes, with `potentialValues` alongside.
    pub fn potentials(&self) -> Vec<u32> {
        self.inner.potentials.iter().map(|&(n, _)| n as u32).collect()
    }

    #[wasm_bindgen(js_name = potentialValues)]
    pub fn potential_values(&self) -> Vec<f64> {
        self.inner.potentials.iter().map(|&(_, v)| v).collect()
    }

    /// Each path as one string, its labels joined by `>` -- an array of
    /// arrays has no typed form and the paths are display-only anyway.
    pub fn paths(&self) -> Vec<String> {
        self.inner.paths.iter().map(|p| p.join(">")).collect()
    }

    pub fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    #[wasm_bindgen(js_name = poolsUsed)]
    pub fn pools_used(&self) -> Vec<String> {
        self.inner.pools_used()
    }

    /// The index of the first leg over its cap, or `undefined`.
    #[wasm_bindgen(js_name = overCapacity)]
    pub fn over_capacity(&self) -> Option<usize> {
        let target = self.inner.over_capacity()?;
        self.inner.legs.iter().position(|rl| std::ptr::eq(rl, target))
    }

    /// The pools whose legs are not an admissible element (decision 3).
    #[wasm_bindgen(js_name = checkOneArcPerPool)]
    pub fn check_one_arc_per_pool(&self) -> Vec<String> {
        realize::check_one_arc_per_pool(&self.inner)
    }

    #[wasm_bindgen(js_name = routeConductance)]
    pub fn route_conductance(&self) -> f64 {
        realize::route_conductance(&self.inner)
    }

    #[wasm_bindgen(js_name = maxTheta)]
    pub fn max_theta(&self) -> f64 {
        realize::max_theta(&self.inner)
    }

    #[wasm_bindgen(js_name = totalLossBp)]
    pub fn total_loss_bp(&self, price_out_per_in: f64) -> f64 {
        realize::total_loss_bp(&self.inner, price_out_per_in)
    }
}

/// What `pruneDust` left: the flow, and how many arcs it cut.
#[wasm_bindgen]
pub struct PruneOut {
    flow: Vec<f64>,
    #[wasm_bindgen(readonly, js_name = prunedCount)]
    pub removed: usize,
}

#[wasm_bindgen]
impl PruneOut {
    #[wasm_bindgen(getter)]
    pub fn flow(&self) -> Vec<f64> {
        self.flow.clone()
    }
}
