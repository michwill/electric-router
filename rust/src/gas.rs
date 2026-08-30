//! Execution gas, as a cost and as a pruning bound (spec §11.1).
//!
//! The mirror of `core/gas.py`.
//!
//! §11.1 keeps gas *out of the convex core*, and that is not squeamishness: a
//! fixed cost per arc is a step function, so the moment gas enters the
//! objective the program stops being convex and becomes mixed-integer. The
//! Laplacian structure -- the entire reason this router is fast -- depends on
//! it staying out.
//!
//! But gas can bound the problem from outside without entering it. A leg that
//! carries less *value* than the leg costs to execute cannot pay for itself
//! under any circumstances, which makes
//!
//!     psi_min = gas_leg * gas_price / eth_per_value_unit
//!
//! a **sound** floor rather than a heuristic one. It is also loose,
//! deliberately: the true threshold is higher, but a loose sound bound is
//! worth more than a tight unsound one.
//!
//! The per-kind numbers are curve_solver's, calibrated against a deployed
//! router rather than guessed from quote gas. Quote gas is the wrong measure:
//! `get_dy` is a view call, while `exchange` pays for transfers and storage
//! writes it never touches.

use crate::types::{ArcKind, Leg};
use std::collections::HashMap;

/// Paid once per transaction, not per leg.
pub const TX_BASE: i64 = 71_000;
/// A split plan pays a little extra to distribute the input across branches.
pub const SPLIT_OVERHEAD: i64 = 20_000;

/// What an unrecognised leg is assumed to cost -- the swap figure, so a new arc
/// kind is never accidentally free.
pub const DEFAULT_LEG: i64 = 102_000;

pub fn leg_gas(kind: ArcKind) -> i64 {
    use ArcKind::*;
    match kind {
        SwapStable | SwapCrypto => 102_000,
        DepositFixed | DepositDyn | DepositFixedNoflag => 71_000,
        WithdrawStable | WithdrawCrypto => 107_000,
        Erc4626Deposit | Erc4626Redeem => 102_000,
        // A native wrap is a deposit/withdraw on WETH and nothing else.
        WrapNative | UnwrapNative => 40_000,
        WstethWrap | WstethUnwrap | StakeNative => 60_000,
        // Measured: `cDAI.redeem` of 100k cDAI cost 172,906 -- a lending
        // redemption touches an interest-accrual write a swap never does, so
        // it is dearer than any of them. `facts` replaces this with the real
        // figure per token.
        LendMint => 170_000,
        LendRedeem => 173_000,
    }
}

/// Total gas for one execution plan, base and split overhead included.
pub fn route_gas(kinds: &[ArcKind], legs: Option<usize>) -> i64 {
    let total = TX_BASE + kinds.iter().map(|&k| leg_gas(k)).sum::<i64>();
    let count = legs.unwrap_or(kinds.len());
    if count > 1 {
        total + SPLIT_OVERHEAD
    } else {
        total
    }
}

/// Per-leg gas that was *executed* rather than assumed.
///
/// The flat per-kind figures above are wrong in a biased direction and by
/// different amounts per pool: measured against one block, four pools priced
/// at a flat 102,000 ran from +16% to +53%, and a crypto pool costs a third
/// more than a stable one while the table charges them the same.
///
/// Two things this deliberately does not claim:
///
/// - **A measurement is per direction, not per pool.** USDT's transfer costs
///   more than DAI's, and the same pool differs by ~3,000 gas between its own
///   pairs, so the key carries `(i, j)`. A per-pool figure is accepted as a
///   fallback under `(-1, -1)`.
/// - **Each leg was measured cold, alone.** A later leg in a real route
///   inherits warm accounts and costs less, so a sum over legs is an upper
///   bound. Erring high is the safe direction.
///
/// Lookup walks from the specific to the general: this direction of this pool;
/// any direction of it; the measured median for the kind; the static figure.
#[derive(Debug, Clone, Default)]
pub struct GasTable {
    pub legs: HashMap<(String, u8, i32, i32), i64>,
    pub kinds: HashMap<u8, i64>,
}

impl GasTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_leg(&mut self, target: &str, kind: ArcKind, i: i32, j: i32, gas: i64) {
        self.legs
            .insert((target.to_ascii_lowercase(), kind.code(), i, j), gas);
    }

    pub fn set_kind(&mut self, kind: ArcKind, gas: i64) {
        self.kinds.insert(kind.code(), gas);
    }

    pub fn gas(&self, kind: ArcKind, target: &str, i: i32, j: i32) -> i64 {
        let address = target.to_ascii_lowercase();
        let code = kind.code();
        self.legs
            .get(&(address.clone(), code, i, j))
            // a per-pool figure, if this direction was missed
            .or_else(|| self.legs.get(&(address, code, -1, -1)))
            // a pool we have never executed
            .or_else(|| self.kinds.get(&code))
            .copied()
            .unwrap_or_else(|| leg_gas(kind))
    }

    pub fn len(&self) -> usize {
        self.legs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }
}

/// Total gas for an executable plan, preferring measured per-leg figures.
///
/// Takes `Leg`s rather than kinds, because the measurement is keyed by which
/// pool and which direction -- that is the whole point of measuring.
pub fn plan_gas(legs: &[Leg], table: &GasTable) -> i64 {
    let total = TX_BASE
        + legs
            .iter()
            .map(|leg| table.gas(leg.kind, &leg.target, leg.i, leg.j))
            .sum::<i64>();
    if legs.len() > 1 {
        total + SPLIT_OVERHEAD
    } else {
        total
    }
}

/// Cost of one unit of gas, in the solver's value units.
pub fn value_per_gas(gas_price_wei: i64, value_units_per_eth: f64) -> f64 {
    if gas_price_wei <= 0 || value_units_per_eth <= 0.0 {
        return 0.0;
    }
    gas_price_wei as f64 / 1e18 * value_units_per_eth
}

/// What a route's shape costs to prefer, in the output token.
///
/// Two charges answer the same question, so only the larger is levied. Gas is
/// what a leg costs to execute; `leg_cost_bp` is what it costs in branching
/// risk -- one more pool between signing and landing -- and that risk scales
/// with the trade, which is why the premium is proportional. Charging both
/// double-counted at both ends: at $10k the gas dominates, at $5M the premium
/// does.
///
/// The fixed part of a plan -- the transaction itself, plus the overhead of
/// splitting at all -- is gas with no branching counterpart, so it is charged
/// whole. Conversions carry gas but no premium: a wrap is a leg to the
/// executor and not a choice the router is making (§11.1).
pub fn shape_cost(
    legs: &[Leg],
    is_conversion: &[bool],
    value: f64,
    leg_cost_bp: f64,
    per_gas: f64,
    table: &GasTable,
) -> f64 {
    let per_leg: Vec<i64> = legs
        .iter()
        .map(|leg| table.gas(leg.kind, &leg.target, leg.i, leg.j))
        .collect();
    let fixed = plan_gas(legs, table) - per_leg.iter().sum::<i64>();
    let premium = value * leg_cost_bp / 1e4;
    fixed as f64 * per_gas
        + per_leg
            .iter()
            .zip(is_conversion.iter())
            .map(|(&gas, &conversion)| {
                let branch = if conversion { 0.0 } else { premium };
                branch.max(gas as f64 * per_gas)
            })
            .sum::<f64>()
}

/// The sound floor described above, in value units.
///
/// Returns 0 when gas is disabled, which restores the previous behaviour
/// exactly rather than silently applying some default price.
pub fn min_useful_flow(gas_price_wei: i64, value_units_per_eth: f64, kind: ArcKind) -> f64 {
    leg_gas(kind) as f64 * value_per_gas(gas_price_wei, value_units_per_eth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(kind: ArcKind, target: &str) -> Leg {
        Leg::new(target.into(), kind, 0, 1, 2, 0, 1, 0).unwrap()
    }

    #[test]
    fn a_single_leg_pays_no_split_overhead() {
        assert_eq!(route_gas(&[ArcKind::SwapStable], None), TX_BASE + 102_000);
        assert_eq!(
            route_gas(&[ArcKind::SwapStable, ArcKind::SwapStable], None),
            TX_BASE + 204_000 + SPLIT_OVERHEAD
        );
    }

    #[test]
    fn lookup_walks_from_the_specific_to_the_general() {
        let mut table = GasTable::new();
        assert_eq!(table.gas(ArcKind::SwapStable, "0xp", 0, 1), 102_000);
        table.set_kind(ArcKind::SwapStable, 120_000);
        assert_eq!(table.gas(ArcKind::SwapStable, "0xp", 0, 1), 120_000);
        table.set_leg("0xP", ArcKind::SwapStable, -1, -1, 130_000);
        assert_eq!(table.gas(ArcKind::SwapStable, "0xp", 0, 1), 130_000);
        table.set_leg("0xp", ArcKind::SwapStable, 0, 1, 140_000);
        assert_eq!(table.gas(ArcKind::SwapStable, "0xp", 0, 1), 140_000);
        // A direction that was never measured still gets the pool figure.
        assert_eq!(table.gas(ArcKind::SwapStable, "0xp", 1, 0), 130_000);
    }

    #[test]
    fn only_the_larger_of_the_two_shape_charges_is_levied() {
        let legs = [leg(ArcKind::SwapStable, "0xa"), leg(ArcKind::WrapNative, "0xb")];
        let table = GasTable::new();
        // No gas price: the premium is all there is, and the conversion is
        // exempt from it.
        let premium_only = shape_cost(&legs, &[false, true], 1e6, 1.0, 0.0, &table);
        assert_eq!(premium_only, 1e6 * 1.0 / 1e4);
        // A gas price high enough that gas dominates: the premium disappears
        // into it rather than adding to it.
        let gas_heavy = shape_cost(&legs, &[false, true], 1e6, 1.0, 1.0, &table);
        assert_eq!(
            gas_heavy,
            (TX_BASE + SPLIT_OVERHEAD) as f64 + 102_000.0 + 40_000.0
        );
    }
}
