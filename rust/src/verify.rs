//! Ranking verified candidates (spec §7).
//!
//! The mirror of `core/verify.py`, minus the call itself. The reference holds
//! a `QuoterClient` and puts every candidate out in one `quote_routes` at the
//! pinned block; here the quotes arrive as an argument, because chain I/O is
//! the host's -- a browser has its own RPC and the CLI has `transport`. What
//! is ported is everything that decides *which* candidate wins, which is the
//! part that has to agree.
//!
//! Four rules from §7, all load-bearing here:
//!
//! 1. *Net shared pools before quoting.* Enforced upstream by the edge-flow
//!    formulation, and asserted here: a candidate touching a pool twice is
//!    rejected rather than quoted, because a view-only call cannot see its own
//!    earlier leg.
//! 2. *Pin the block.* All candidates share one transport, hence one block.
//! 3. *Quote at the real size.* The legs carry the amounts the model chose.
//! 4. *Treat a failed quote as arc removal.* A zero comes back as `reverted`
//!    and the candidate is dropped, never as an error.

use crate::candidates::{Candidate, CandidateSet};
use crate::gas::{leg_gas, plan_gas, shape_cost, value_per_gas, GasTable};
use crate::nodes::NodeMap;
use crate::realize::{check_one_arc_per_pool, realize, RealizedRoute};
use crate::risk::{expected_value, RiskTable, REVERT_COST_BP};
use crate::types::{ArcKind, PoolArc};
use ruint::aliases::U256;

/// What counts as "the same answer". `score` nets gas off every candidate
/// before ranking, so the only thing left for a tolerance to absorb is noise
/// in the quotes themselves -- integer rounding, around 1e-12 relative. A flat
/// 5e-6 was instead absorbing a whole basis point of real difference at low
/// gas: measured on WETH->stETH 100 at 0.049 gwei, a 4-leg route lost to a
/// 2-leg one over 0.0415 bp, while the two extra legs cost thirty times less
/// than the gain thrown away. So the tolerance is what one more leg actually
/// costs, in output units, and nothing more; the floor keeps a tie from
/// turning on the last bit.
pub const TIE_FLOOR: f64 = 1e-12;

/// Fraction of the trade re-quoted to find what the same route would have paid
/// if the trade had been small.
///
/// Every leg's share is a fraction of its slot's balance, so scaling the input
/// scales every branch with it: the shape of the route is held fixed and only
/// the size changes, which is exactly the comparison price impact is supposed
/// to make. 5% is small enough that its own impact is a rounding error against
/// the full trade's, and large enough to stay clear of integer dust on a
/// 6-decimal token.
pub const IMPACT_FRACTION: f64 = 0.05;

/// What one leg is charged beyond its gas, in basis points of the trade.
///
/// Gas is a real per-leg price and at any ordinary gas price it decides this
/// on its own: measured on USDC->WETH $10k, the winner is 9 legs at 0.045
/// gwei, 3 at 5 gwei and 1 at 30. This term is for the case gas stops
/// arbitrating, where the relaxation will take a long tail of branches for a
/// fraction of a basis point each.
///
/// The value is the measured knee, swept at 0.045 gwei: 0.02 takes 21 legs off
/// the $10k trade for 0.62 bp and halves USDC->USDT $1M for nothing, while
/// costing zero where the legs are earning their keep. Above 0.1 it starts
/// buying simplicity with real money.
///
/// Proportional rather than absolute, which is the right shape: what avoiding
/// a leg is worth scales with the trade, and so does what the leg earns, so
/// the charge stays self-limiting.
pub const LEG_COST_BP: f64 = 0.02;

/// Turn each candidate's flow into legs, marking the ones that cannot be.
///
/// `max_legs` is what *we* can price -- not what anything can execute. A
/// deployed router has its own, much tighter, limit.
#[allow(clippy::too_many_arguments)]
pub fn realize_candidates(
    candidates: &mut CandidateSet,
    arcs: &[PoolArc],
    nu: &[f64],
    nodes: &NodeMap,
    src_token: &str,
    dst_token: &str,
    amount_in: U256,
    potentials: Option<&[f64]>,
    max_legs: usize,
    max_slots: usize,
) {
    for candidate in candidates.candidates.iter_mut() {
        let active: Vec<usize> =
            (0..candidate.psi.len()).filter(|&k| candidate.psi[k] > 0.0).collect();
        if active.is_empty() {
            candidate.status = "empty".into();
            continue;
        }
        let members: Vec<PoolArc> = active.iter().map(|&k| arcs[k].clone()).collect();
        let flows: Vec<f64> = active.iter().map(|&k| candidate.psi[k]).collect();
        let route = match realize(
            &members, &flows, nu, nodes, src_token, dst_token, amount_in, potentials,
        ) {
            Ok(route) => route,
            Err(e) => {
                candidate.status = "infeasible".into();
                candidate.note = e.0;
                continue;
            }
        };

        let conflicts = check_one_arc_per_pool(&route);
        if !conflicts.is_empty() {
            candidate.status = "conflict".into();
            candidate.note = format!("{} pool(s) used twice", conflicts.len());
            continue;
        }
        if route.slots.len() > max_slots {
            candidate.status = "too_wide".into();
            candidate.note =
                format!("{} tokens > quoter limit {max_slots}", route.slots.len());
            continue;
        }
        if route.legs.len() > max_legs {
            candidate.status = "too_long".into();
            candidate.note = format!("{} legs > limit {max_legs}", route.legs.len());
            continue;
        }
        candidate.route = Some(route);
        candidate.status = "ready".into();
    }
}

/// Everything ranking needs beyond the quotes themselves.
pub struct VerifyOptions<'a> {
    pub gas_price_wei: i64,
    pub dst_wei_per_eth: f64,
    pub gas_table: &'a GasTable,
    /// `None` leaves the survival term out entirely, as the reference does.
    pub risk_table: Option<&'a RiskTable>,
    pub revert_cost_bp: f64,
    pub leg_cost_bp: f64,
}

impl<'a> VerifyOptions<'a> {
    pub fn new(gas_table: &'a GasTable) -> Self {
        Self {
            gas_price_wei: 0,
            dst_wei_per_eth: 0.0,
            gas_table,
            risk_table: None,
            revert_cost_bp: REVERT_COST_BP,
            leg_cost_bp: LEG_COST_BP,
        }
    }
}

/// The candidates that are ready to be quoted, in the order the quotes must
/// come back in.
pub fn ready(candidates: &CandidateSet) -> Vec<usize> {
    (0..candidates.candidates.len())
        .filter(|&k| {
            let c = &candidates.candidates[k];
            c.status == "ready" && c.route.is_some()
        })
        .collect()
}

/// Fold one batch of quotes back in, then rank everything.
///
/// `quotes` is one value per index `ready` returned, in that order -- what the
/// reference gets back from a single `quote_routes` at the pinned block. A
/// zero is a revert, not an error.
///
/// Ranking happens unconditionally: this is called more than once per route
/// (candidates, then the direct floor, then the refit), and returning early
/// when nothing needs quoting used to leave ranks stale from an earlier call.
/// The winner is chosen by rank, so a stale rank silently picks the wrong
/// route.
pub fn verify(candidates: &mut CandidateSet, quotes: &[(usize, u128)],
              opts: &VerifyOptions<'_>) {
    for &(at, value) in quotes {
        let candidate = &mut candidates.candidates[at];
        if value == 0 {
            candidate.status = "reverted".into();
            if candidate.note.is_empty() {
                candidate.note = "quoter returned 0".into();
            }
            continue;
        }
        // A view does not have to refuse what the pool will: a vault past its
        // deposit room quotes the ratio quite happily and then reverts.
        let over = candidate
            .route
            .as_ref()
            .and_then(RealizedRoute::over_capacity)
            .map(|leg| {
                let name = if leg.pool_name.is_empty() {
                    leg.target.chars().take(10).collect::<String>()
                } else {
                    leg.pool_name.clone()
                };
                format!(
                    "{name} takes at most {} and is handed {}",
                    crate::pyfmt::fixed(leg.cap_in, 0),
                    leg.amount_in
                )
            });
        if let Some(note) = over {
            candidate.status = "reverted".into();
            candidate.note = note;
            continue;
        }
        candidate.verified_out = Some(value);
        candidate.status = "ok".into();
    }
    rank(candidates, opts);
}

/// What this candidate is worth, net of what the route costs to attempt.
///
/// Gas, plus the chance one of its minimum-outs trips first, charged at what a
/// resubmission is worth rather than at the trade. Both corrections need the
/// caller to have supplied the means to value them in the output token;
/// neither is an element law and neither may enter the convex core (§11.1).
fn score(candidate: &mut Candidate, opts: &VerifyOptions<'_>) -> f64 {
    let value = candidate.verified_out.unwrap_or(0) as f64;
    let mut gas_cost = 0.0;
    if let Some(route) = candidate.route.as_ref() {
        let legs: Vec<crate::types::Leg> = route.legs.iter().map(|rl| rl.leg.clone()).collect();
        candidate.gas = plan_gas(&legs, opts.gas_table);
        let conversions: Vec<bool> = route.legs.iter().map(|rl| rl.is_conversion()).collect();
        // The greater of the two shape charges, never their sum.
        gas_cost = shape_cost(
            &legs, &conversions, value, opts.leg_cost_bp,
            value_per_gas(opts.gas_price_wei, opts.dst_wei_per_eth), opts.gas_table,
        );
    }
    let (Some(table), Some(route)) = (opts.risk_table, candidate.route.as_ref()) else {
        return value - gas_cost;
    };
    // Every leg carries a minimum-out at a fraction of its pool's fee, so the
    // route lands only if none of those pools moves past its own bound while
    // the user is confirming. `survival` is the chance of that; the cost of
    // the other case is one more transaction and a basis point of price
    // movement, not the trade.
    let legs: Vec<crate::types::Leg> = route.legs.iter().map(|rl| rl.leg.clone()).collect();
    candidate.survival = table.survival(&legs);
    expected_value(value, candidate.survival, gas_cost, opts.revert_cost_bp)
}

fn legs_of(candidate: &Candidate) -> usize {
    candidate.route.as_ref().map_or(1_000, |r| r.legs.len())
}

/// Prefer the simpler route when the outputs are indistinguishable.
///
/// Measured on stablecoin pairs, the relaxation happily takes a 25-leg route
/// to gain 0.02 bp over a 1-leg one, and §11.1 is explicit that a fixed
/// per-arc cost belongs in candidate selection rather than in the convex core.
/// Quantising the score is how a tie becomes visible to the sort at all.
fn rank(candidates: &mut CandidateSet, opts: &VerifyOptions<'_>) {
    for candidate in candidates.candidates.iter_mut() {
        if !candidate.ok() {
            candidate.rank = None; // a stale rank must never survive a re-verify
        }
    }
    let usable: Vec<usize> = (0..candidates.candidates.len())
        .filter(|&k| candidates.candidates[k].ok())
        .collect();
    if usable.is_empty() {
        return;
    }
    // `score` writes `gas` and `survival` back onto the candidate, so it runs
    // once per candidate and the numbers are read from here on.
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(usable.len());
    for &k in &usable {
        let value = score(&mut candidates.candidates[k], opts);
        scores.push((k, value));
    }
    let at = |k: usize| scores.iter().find(|(i, _)| *i == k).map(|(_, v)| *v).unwrap();
    let best_score = scores.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);

    // What one more leg has to earn to be worth taking. Both of its costs are
    // already inside `score` -- gas subtracted, revert risk multiplied in -- so
    // what is left for a tolerance to absorb is one leg's gas.
    let per_leg = leg_gas(ArcKind::SwapStable) as f64 * opts.gas_price_wei as f64 / 1e18
        * opts.dst_wei_per_eth;
    let tolerance = per_leg.max(best_score.abs() * TIE_FLOOR);

    let mut ranked = usable.clone();
    // Stable, as Python's `sorted` is.
    ranked.sort_by(|&a, &b| {
        let key = |k: usize| {
            let value = at(k);
            let bucket = if best_score - value <= tolerance { 0 } else { 1 };
            (bucket, if bucket == 0 { legs_of(&candidates.candidates[k]) } else { 0 }, -value)
        };
        let (ba, la, va) = key(a);
        let (bb, lb, vb) = key(b);
        (ba, la).cmp(&(bb, lb)).then(va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Hard floor: never rank below a plain one-hop swap. The tie-break should
    // already give this -- a direct candidate has one leg, so it wins any tie
    // -- but "the router is never worse than a pool you could find by
    // inspection" is a promise, not an emergent property.
    //
    // Compared on `score`, not on the raw quote, so the floor speaks the same
    // language as the ranking: a single hop through a pool that breaches a
    // quarter of the time can quote the largest number on the page and still
    // be the worse trade.
    let floor = usable
        .iter()
        .copied()
        .filter(|&k| candidates.candidates[k].kind == "direct")
        // `max` keeps the *last* of a tie, which is what Python's does.
        .reduce(|a, b| if at(b) >= at(a) { b } else { a });
    if let Some(floor) = floor {
        if !ranked.is_empty() && at(ranked[0]) < at(floor) {
            ranked.retain(|&k| k != floor);
            ranked.insert(0, floor);
        }
    }

    for (position, &k) in ranked.iter().enumerate() {
        candidates.candidates[k].rank = Some(position + 1);
    }
}

/// How much worse this trade's price is than a small one down the same route.
///
/// Price is input over output, as the trader pays it, and the impact is the
/// difference between the price at full size and at `fraction` of it:
///
///     impact = price(full) / price(small) - 1
///
/// Returns `(impact_bp, reference_in, reference_out)`, or `None` when there is
/// nothing to compare -- a size that rounds to zero, or a quote that reverts
/// at the smaller size, which happens on legs whose pool has a minimum.
///
/// **What this is not.** The reference trade has its own impact, so this
/// understates the true spot-relative figure by roughly `fraction` of it --
/// about 5%, in the same direction for every route, and not corrected for here
/// because correcting would mean assuming a shape for the impact curve.
pub fn price_impact(
    amount_in: U256,
    verified_out: u128,
    reference_in: U256,
    reference_out: u128,
) -> Option<(f64, U256, u128)> {
    if reference_in.is_zero() || amount_in.is_zero() || verified_out == 0 {
        return None;
    }
    if reference_out == 0 {
        return None;
    }
    let rate_small = crate::pools::divided(U256::from(reference_out), reference_in);
    let rate_full = crate::pools::divided(U256::from(verified_out), amount_in);
    if rate_small <= 0.0 || rate_full <= 0.0 {
        return None;
    }
    Some(((rate_small / rate_full - 1.0) * 1e4, reference_in, reference_out))
}

/// `int(amount_in * fraction)` -- the size the reference quote is taken at.
pub fn impact_reference_in(amount_in: U256, fraction: f64) -> U256 {
    let scaled = crate::pools::scaled(amount_in, 0) * fraction;
    crate::pools::stableswap::to_u256(scaled.trunc()).unwrap_or(U256::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::Candidate;

    fn ready_candidate(label: &str, kind: &str, value: u128, legs: usize) -> Candidate {
        let mut candidate = Candidate {
            label: label.into(),
            kind: kind.into(),
            status: "ok".into(),
            verified_out: Some(value),
            survival: 1.0,
            ..Default::default()
        };
        let mut route = RealizedRoute::default();
        for k in 0..legs {
            let leg = crate::types::Leg::new(
                format!("0x{k:040x}"), ArcKind::SwapStable, 0, 1, 2,
                k as i32, k as i32 + 1, 0,
            )
            .unwrap();
            route.legs.push(crate::realize::RealizedLeg {
                token_in: String::new(),
                token_out: String::new(),
                ..leg_of(leg)
            });
        }
        candidate.route = Some(route);
        candidate
    }

    fn leg_of(leg: crate::types::Leg) -> crate::realize::RealizedLeg {
        crate::realize::RealizedLeg {
            kind: leg.kind,
            target: leg.target.clone(),
            leg,
            token_in: String::new(),
            token_out: String::new(),
            amount_in: U256::ZERO,
            amount_out: U256::ZERO,
            share_of_node: 1.0,
            arc_id: None,
            pool_name: String::new(),
            eps: 0.0,
            impact_frac: 0.0,
            theta: 0.0,
            psi: 0.0,
            reserve_in: U256::ZERO,
            cap_in: f64::INFINITY,
            tvl_usd: 0.0,
            verified_in: U256::ZERO,
            verified_out: U256::ZERO,
            fee_floor: f64::NAN,
            fee_frac: f64::NAN,
            gamma_live: f64::NAN,
            modelled: true,
        }
    }

    #[test]
    fn a_tie_goes_to_the_shorter_route() {
        let table = GasTable::new();
        let opts = VerifyOptions::new(&table);
        let mut set = CandidateSet::default();
        // The long one quotes fractionally more and still loses: within the
        // tolerance the two are the same answer, and legs break the tie.
        set.candidates.push(ready_candidate("long", "solve", 1_000_000_001, 9));
        set.candidates.push(ready_candidate("short", "solve", 1_000_000_000, 1));
        rank(&mut set, &opts);
        assert_eq!(set.best().unwrap().label, "short");
    }

    #[test]
    fn a_real_gap_beats_the_shorter_route() {
        let table = GasTable::new();
        let opts = VerifyOptions::new(&table);
        let mut set = CandidateSet::default();
        set.candidates.push(ready_candidate("long", "solve", 2_000_000_000, 9));
        set.candidates.push(ready_candidate("short", "solve", 1_000_000_000, 1));
        rank(&mut set, &opts);
        assert_eq!(set.best().unwrap().label, "long");
    }

    #[test]
    fn the_router_is_never_worse_than_a_pool_you_could_find_by_inspection() {
        // The case the floor exists for, and it is narrow: the direct
        // candidate scores *higher* but carries more legs, so the tie-break --
        // which prefers the shorter route inside the tolerance -- puts the
        // other one first. Only the floor puts it back.
        let table = GasTable::new();
        let mut opts = VerifyOptions::new(&table);
        opts.gas_price_wei = 1_000_000_000;
        opts.dst_wei_per_eth = 1e10;
        let mut set = CandidateSet::default();
        set.candidates.push(ready_candidate("short", "solve", 1_000_000_000, 1));
        set.candidates.push(ready_candidate("plain", "direct", 1_003_000_000, 3));
        rank(&mut set, &opts);
        assert_eq!(set.best().unwrap().label, "plain");
        // And without the floor the shorter one would have taken it: the two
        // are inside one leg's gas of each other, which is what puts them in
        // the same bucket.
        let short = score(&mut set.candidates[0].clone(), &opts);
        let plain = score(&mut set.candidates[1].clone(), &opts);
        assert!(plain > short);
        assert!(plain - short <= leg_gas(ArcKind::SwapStable) as f64 * 10.0);
    }

    #[test]
    fn a_stale_rank_never_survives_a_re_verify() {
        let table = GasTable::new();
        let opts = VerifyOptions::new(&table);
        let mut set = CandidateSet::default();
        set.candidates.push(ready_candidate("one", "solve", 1_000, 1));
        rank(&mut set, &opts);
        assert_eq!(set.candidates[0].rank, Some(1));
        set.candidates[0].status = "reverted".into();
        rank(&mut set, &opts);
        assert_eq!(set.candidates[0].rank, None);
    }
}
