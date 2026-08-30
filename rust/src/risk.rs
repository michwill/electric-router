//! What a route's own minimum-out costs it.
//!
//! The mirror of `core/risk.py`.
//!
//! Every leg executes with a minimum-out at a fraction of the pool's fee; if
//! the rate moves past it before the transaction lands, the leg reverts and
//! the route with it. So `P(lands) = product over legs of (1 - p_i)`.
//!
//! **A revert does not cost the trade, and that sets the scale of the term.**
//! The gas is spent and the user resubmits, so a failure costs one more
//! transaction plus whatever the price did -- around a basis point, not a
//! hundred. Ranking on `output * P(lands)` prices it as losing the whole
//! notional, wrong by three orders of magnitude, and at measured 20-40% breach
//! rates it pays 17-126 bp for safety. Instead:
//!
//!     output * (1 - P(fails) * REVERT_COST_BP / 1e4) - gas * (1 + P(fails))
//!
//! The gas term carries `(1 + P(fails))` because a failed attempt pays and the
//! retry pays again.
//!
//! Only pool arcs carry risk -- a wrap, stake, lending mint or vault
//! redemption is priced by a rate that accrues slowly and not against us. An
//! unmeasured pool takes `DEFAULT_RISK` rather than zero, so a thin unsampled
//! pool cannot look safer than a deep measured one.

use crate::types::{ArcKind, Leg};
use std::collections::HashMap;

/// Kinds whose output is not a market rate, so no bound of this sort binds.
pub fn is_riskless(kind: ArcKind) -> bool {
    matches!(
        kind,
        ArcKind::WrapNative
            | ArcKind::UnwrapNative
            | ArcKind::WstethWrap
            | ArcKind::WstethUnwrap
            | ArcKind::StakeNative
            | ArcKind::LendMint
            | ArcKind::LendRedeem
            | ArcKind::Erc4626Deposit
            | ArcKind::Erc4626Redeem
    )
}

/// What an arc nobody has measured is assumed to cost, as a probability.
///
/// Small but not zero: zero would say "this pool provably never moves", which
/// is the one thing a missing measurement cannot say, and would make an
/// unprobed pool the cheapest thing in the graph. The measured distribution is
/// sharply bimodal -- nine arcs in ten at the 1e-5 floor, the rest from 1% to
/// 50% -- so neither its median nor its upper decile stands in. 0.2% is what
/// an unknown arc must beat: about 20 bp, more than any tail-end routing gain
/// and less than the cost of dropping a good pool.
pub const DEFAULT_RISK: f64 = 0.002;

/// What one failed attempt costs, as basis points of the trade.
///
/// Gas, plus whatever the price did while the user resubmitted. Set so that
/// the most dangerous arc measured -- 39%, TricryptoUSDC's ETH side -- costs a
/// route 0.4 bp. Anything larger stops being a tie-break and starts buying
/// worse prices.
pub const REVERT_COST_BP: f64 = 1.0;

/// Per-arc probability that a leg's minimum-out trips before inclusion.
///
/// Keyed by direction, like `GasTable` and for the same reason: a pool's own
/// pairs do not behave alike, and the minimum-out is written per leg, so
/// pricing a pool by its worst pair would charge every route for the riskiest
/// thing in it.
#[derive(Debug, Clone)]
pub struct RiskTable {
    pub arcs: HashMap<(String, i32, i32), f64>,
    pub default: f64,
}

impl Default for RiskTable {
    fn default() -> Self {
        Self { arcs: HashMap::new(), default: DEFAULT_RISK }
    }
}

impl RiskTable {
    pub fn new(default: f64) -> Self {
        Self { arcs: HashMap::new(), default }
    }

    pub fn set(&mut self, target: &str, i: i32, j: i32, risk: f64) {
        self.arcs.insert((target.to_ascii_lowercase(), i, j), risk);
    }

    pub fn of(&self, kind: ArcKind, target: &str, i: i32, j: i32) -> f64 {
        if is_riskless(kind) {
            return 0.0;
        }
        let address = target.to_ascii_lowercase();
        let mut got = None;
        if kind.is_swap() {
            // `(i, j)` means coin indices only on a swap. A deposit or a
            // single-coin withdrawal numbers its legs differently, so it takes
            // the pool-level entry, which is the right granularity for it.
            got = self.arcs.get(&(address.clone(), i, j)).copied();
        }
        if got.is_none() {
            got = self.arcs.get(&(address, -1, -1)).copied();
        }
        got.unwrap_or(self.default)
    }

    /// P(every leg stays inside its bound).
    ///
    /// A pool touched twice counts twice: two legs are two separate
    /// minimum-outs, both of which have to hold.
    pub fn survival(&self, legs: &[Leg]) -> f64 {
        legs.iter().fold(1.0, |product, leg| {
            product * (1.0 - self.of(leg.kind, &leg.target, leg.i, leg.j))
        })
    }

    pub fn len(&self) -> usize {
        self.arcs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arcs.is_empty()
    }
}

/// What this route is worth, netting the cost of it not landing.
///
/// `output` and `gas_cost` are both in output-token units. A failure costs
/// `revert_cost_bp` of the trade plus one extra transaction -- not the trade.
pub fn expected_value(
    output: f64,
    survival: f64,
    gas_cost: f64,
    revert_cost_bp: f64,
) -> f64 {
    let failure = (1.0 - survival).clamp(0.0, 1.0);
    output * (1.0 - failure * revert_cost_bp / 1e4) - gas_cost * (1.0 + failure)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(kind: ArcKind, target: &str, i: i32, j: i32) -> Leg {
        Leg::new(target.into(), kind, i, j, 2, 0, 1, 0).unwrap()
    }

    #[test]
    fn a_wrap_carries_no_risk_and_an_unmeasured_pool_carries_the_default() {
        let table = RiskTable::default();
        assert_eq!(table.of(ArcKind::WrapNative, "0xp", 0, 1), 0.0);
        assert_eq!(table.of(ArcKind::SwapStable, "0xp", 0, 1), DEFAULT_RISK);
    }

    #[test]
    fn only_a_swap_is_looked_up_by_coin() {
        let mut table = RiskTable::default();
        table.set("0xp", 0, 1, 0.4);
        table.set("0xp", -1, -1, 0.1);
        assert_eq!(table.of(ArcKind::SwapStable, "0xp", 0, 1), 0.4);
        assert_eq!(table.of(ArcKind::SwapStable, "0xp", 1, 0), 0.1);
        // A withdrawal numbers its legs differently, so it takes the pool
        // figure even where `(i, j)` happens to match a swap entry.
        assert_eq!(table.of(ArcKind::WithdrawStable, "0xp", 0, 1), 0.1);
    }

    #[test]
    fn a_pool_touched_twice_counts_twice() {
        let mut table = RiskTable::default();
        table.set("0xp", -1, -1, 0.5);
        let one = table.survival(&[leg(ArcKind::SwapStable, "0xp", 0, 1)]);
        let two = table.survival(&[
            leg(ArcKind::SwapStable, "0xp", 0, 1),
            leg(ArcKind::SwapStable, "0xp", 1, 0),
        ]);
        assert_eq!(one, 0.5);
        assert_eq!(two, 0.25);
    }

    #[test]
    fn a_failure_costs_a_basis_point_not_the_trade() {
        // 40% breach: a route worth 1e6 loses 40 units, not 400,000.
        let value = expected_value(1e6, 0.6, 0.0, REVERT_COST_BP);
        assert!((1e6 - value - 40.0).abs() < 1e-9);
    }
}
