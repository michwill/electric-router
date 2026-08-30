//! Splitting a slippage budget between a route's legs.
//!
//! The mirror of `core/slippage.py`.
//!
//! A route is a resistor network and a budget is a potential to spend across
//! it: Dirichlet at the two terminals, Kirchhoff everywhere else. That is the
//! same Laplacian the solver assembles, on a graph of three or four nodes, and
//! it gives each leg the drop its own depth earns rather than an equal share.
//!
//! The awkward case is a leg the network runs *backwards* -- its drop comes
//! back negative, which is an imbalance rather than a price. Those are held at
//! a floor and everything else is scaled around them, by bisection: the
//! longest path is a maximum over paths rather than a sum, so it is monotone
//! in the scale but not linear in it.

use crate::realize::RealizedRoute;

/// Slot 0 holds the input. `routecall.fractions` seeds its walk the same way.
pub const SOURCE: i32 = 0;

/// A resistance of zero is a short circuit and has no conductance to write.
pub const TINY: f64 = 1e-12;

/// Enough halvings to land the scale factor inside a float's last digits.
pub const BISECTIONS: usize = 60;

/// Every slot the route touches, in the order its legs first name them.
fn slots(route: &RealizedRoute) -> Vec<i32> {
    let mut seen: Vec<i32> = Vec::new();
    for realized in &route.legs {
        for slot in [realized.leg.src_slot, realized.leg.dst_slot] {
            if !seen.contains(&slot) {
                seen.push(slot);
            }
        }
    }
    seen
}

/// The potential each leg drops, or `None` if the network will not solve.
///
/// Dirichlet at the two terminals and Kirchhoff everywhere else, which is the
/// same Laplacian as the solver on a graph of three or four nodes.
pub fn drops(route: &RealizedRoute, resistance: &[f64], total: f64) -> Option<Vec<f64>> {
    let slots = slots(route);
    let index = |slot: i32| slots.iter().position(|&s| s == slot);
    let (source, sink) = (index(SOURCE)?, index(route.dst_slot as i32)?);
    if source == sink {
        return None;
    }
    let n = slots.len();
    let mut matrix = vec![0.0f64; n * n];
    for (realized, &r) in route.legs.iter().zip(resistance.iter()) {
        let a = index(realized.leg.src_slot)?;
        let b = index(realized.leg.dst_slot)?;
        let conductance = 1.0 / r.max(TINY);
        matrix[a * n + a] += conductance;
        matrix[b * n + b] += conductance;
        matrix[a * n + b] -= conductance;
        matrix[b * n + a] -= conductance;
    }
    let free: Vec<usize> = (0..n).filter(|&k| k != source && k != sink).collect();
    let mut potential = vec![0.0f64; n];
    potential[source] = total;
    if !free.is_empty() {
        let mut system = vec![0.0f64; free.len() * free.len()];
        for (r, &row) in free.iter().enumerate() {
            for (c, &col) in free.iter().enumerate() {
                system[r * free.len() + c] = matrix[row * n + col];
            }
        }
        let mut rhs: Vec<f64> =
            free.iter().map(|&row| -matrix[row * n + source] * total).collect();
        crate::lu::solve_in_place(&mut system, &mut rhs, free.len()).ok()?;
        for (k, &slot) in free.iter().enumerate() {
            potential[slot] = rhs[k];
        }
    }
    Some(
        route
            .legs
            .iter()
            .map(|realized| {
                potential[index(realized.leg.src_slot).unwrap()]
                    - potential[index(realized.leg.dst_slot).unwrap()]
            })
            .collect(),
    )
}

/// The most any one path spends. Legs arrive topologically ordered.
pub fn longest(route: &RealizedRoute, spend: &[f64]) -> f64 {
    let mut best: Vec<(i32, f64)> = vec![(SOURCE, 0.0)];
    for (realized, &drop) in route.legs.iter().zip(spend.iter()) {
        let Some(reached) = best.iter().find(|(s, _)| *s == realized.leg.src_slot)
            .map(|&(_, v)| v)
        else {
            continue;
        };
        let head = realized.leg.dst_slot;
        let value = reached + drop;
        match best.iter_mut().find(|(s, _)| *s == head) {
            Some(entry) => entry.1 = entry.1.max(value),
            None => best.push((head, value.max(0.0))),
        }
    }
    best.iter()
        .find(|(s, _)| *s == route.dst_slot as i32)
        .map_or(0.0, |&(_, v)| v)
}

/// What each leg is owed however the rest of the route is scaled.
///
/// Zero for a leg the network drops forwards, which is nearly all of them. A
/// leg it runs backwards is owed the magnitude it came back by, and never less
/// than `floor` -- the automatic rule's own answer, which is what that leg
/// would have shipped with no budget named at all.
pub fn backstops(raw: &[f64], floor: Option<&[f64]>) -> Vec<f64> {
    raw.iter()
        .enumerate()
        .map(|(k, &value)| {
            if value < 0.0 {
                (-value).max(floor.map_or(0.0, |f| f[k]))
            } else {
                0.0
            }
        })
        .collect()
}

/// Split `total` between the legs, as fractions rather than basis points.
///
/// `backstop` is read only where the network runs a leg backwards. Those legs
/// are held at their floor and everything else is scaled around them, by
/// bisection because the longest path is a maximum over paths and not a sum --
/// monotone in the scale, so it converges, and the side it converges from is
/// the one that cannot overspend.
///
/// Falls back to sharing in proportion to `resistance` alone where the network
/// does not solve -- a slot nothing reaches -- and to depth where every
/// resistance is zero. All three are normalised the same way, so the promise a
/// caller reads off `RouteCall.tolerance_bp` holds however it got there.
pub fn divide(
    route: &RealizedRoute,
    resistance: &[f64],
    total: f64,
    backstop: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    if total < 0.0 {
        return Err(format!(
            "a slippage budget cannot be negative, got {}",
            crate::pyfmt::float(total)
        ));
    }
    if route.legs.is_empty() {
        return Ok(Vec::new());
    }
    let raw = drops(route, resistance, total).unwrap_or_else(|| resistance.to_vec());
    let held = backstops(&raw, backstop);
    let mut share: Vec<f64> = raw.iter().map(|&v| v.max(0.0)).collect();
    let mut spent = longest(route, &share);
    if spent <= 0.0 {
        share = vec![1.0; route.legs.len()];
        spent = longest(route, &share);
    }
    if spent <= 0.0 {
        return Ok(held);
    }
    share = share.iter().map(|&v| v * total / spent).collect();
    if !held.iter().any(|&v| v != 0.0) {
        return Ok(share);
    }
    if longest(route, &held) >= total {
        // The floors are the budget on their own. They are what a leg needs to
        // survive movement, so they stand and the total is what it is.
        return Ok(held);
    }
    let (mut low, mut high) = (0.0f64, 1.0f64);
    for _ in 0..BISECTIONS {
        let mid = 0.5 * (low + high);
        if longest(route, &blend(&held, &share, mid)) <= total {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(blend(&held, &share, low))
}

fn blend(held: &[f64], share: &[f64], scale: f64) -> Vec<f64> {
    held.iter()
        .zip(share.iter())
        .map(|(&floor, &value)| floor.max(scale * value))
        .collect()
}

/// Raise a leg the network runs backwards to `floor`, leaving the rest.
///
/// `divide` normalises so that no path can spend more than the budget, which
/// is the right answer when the budget is all the caller said. A caller who
/// names one has said something else as well: that they will accept losing
/// that much on the trade. A bridge leg is the one the automatic rule cannot
/// price -- its drop comes back negative, so it ships at the magnitude of an
/// imbalance rather than at anything the caller chose -- and it is also the
/// leg that reverts on any movement at all. So it is given the figure they
/// named outright.
///
/// That breaks `divide`'s promise on paths through it, deliberately: such a
/// path now spends the budget plus its share of the rest. What stands in for
/// it is `min_out`, the end-to-end bound a named budget also buys.
///
/// Only backwards legs move, and only upwards -- a bridge already granted more
/// than the budget keeps what it had.
pub fn widen(
    route: &RealizedRoute,
    resistance: &[f64],
    total: f64,
    spend: &[f64],
    floor: f64,
) -> Vec<f64> {
    let Some(raw) = drops(route, resistance, total) else {
        return spend.to_vec();
    };
    spend
        .iter()
        .zip(raw.iter())
        .map(|(&value, &backwards)| if backwards < 0.0 { value.max(floor) } else { value })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::RealizedLeg;
    use crate::types::{ArcKind, Leg};

    fn route(hops: &[(i32, i32)], dst_slot: usize) -> RealizedRoute {
        let mut built = RealizedRoute { dst_slot, ..Default::default() };
        for (k, &(src, dst)) in hops.iter().enumerate() {
            let leg = Leg::new(
                format!("0x{k:040x}"), ArcKind::SwapStable, 0, 1, 2, src, dst, 0,
            )
            .unwrap();
            built.legs.push(RealizedLeg {
                kind: leg.kind,
                target: leg.target.clone(),
                leg,
                token_in: String::new(),
                token_out: String::new(),
                amount_in: ruint::aliases::U256::ZERO,
                amount_out: ruint::aliases::U256::ZERO,
                share_of_node: 1.0,
                arc_id: None,
                pool_name: String::new(),
                eps: 0.0,
                impact_frac: 0.0,
                theta: 0.0,
                psi: 0.0,
                reserve_in: ruint::aliases::U256::ZERO,
                cap_in: f64::INFINITY,
                tvl_usd: 0.0,
                verified_in: ruint::aliases::U256::ZERO,
                verified_out: ruint::aliases::U256::ZERO,
                fee_floor: f64::NAN,
                fee_frac: f64::NAN,
                gamma_live: f64::NAN,
                modelled: true,
            });
        }
        built
    }

    #[test]
    fn a_series_chain_splits_by_depth() {
        // Two hops in series: the budget divides in proportion to resistance.
        let chain = route(&[(0, 1), (1, 2)], 2);
        let got = drops(&chain, &[1.0, 3.0], 100.0).unwrap();
        assert!((got[0] - 25.0).abs() < 1e-9, "{got:?}");
        assert!((got[1] - 75.0).abs() < 1e-9, "{got:?}");
        assert!((longest(&chain, &got) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn parallel_branches_each_see_the_whole_drop() {
        // Both legs span the same two slots, so each drops the full budget --
        // the longest *path* is one leg, not two.
        let split = route(&[(0, 1), (0, 1)], 1);
        let got = drops(&split, &[1.0, 3.0], 100.0).unwrap();
        assert!((got[0] - 100.0).abs() < 1e-9);
        assert!((got[1] - 100.0).abs() < 1e-9);
        assert!((longest(&split, &got) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_budget_is_never_overspent_on_any_path() {
        let wide = route(&[(0, 1), (0, 2), (1, 3), (2, 3)], 3);
        let got = divide(&wide, &[1.0, 2.0, 3.0, 0.5], 50.0, None).unwrap();
        assert!(longest(&wide, &got) <= 50.0 + 1e-9, "{got:?}");
        assert!(got.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn a_backwards_leg_is_held_at_its_floor() {
        let raw = [1.0, -4.0, 2.0];
        assert_eq!(backstops(&raw, None), vec![0.0, 4.0, 0.0]);
        // Never less than the automatic rule's own answer.
        assert_eq!(backstops(&raw, Some(&[0.0, 9.0, 0.0])), vec![0.0, 9.0, 0.0]);
    }

    #[test]
    fn a_negative_budget_is_refused() {
        let chain = route(&[(0, 1)], 1);
        assert!(divide(&chain, &[1.0], -1.0, None).is_err());
        // And an empty route spends nothing rather than failing.
        assert_eq!(divide(&RealizedRoute::default(), &[], 1.0, None).unwrap(), Vec::new());
    }

    #[test]
    fn every_resistance_zero_falls_back_to_depth() {
        // Nothing to share in proportion to, so the split is by hop count.
        let chain = route(&[(0, 1), (1, 2)], 2);
        let got = divide(&chain, &[0.0, 0.0], 100.0, None).unwrap();
        assert!((longest(&chain, &got) - 100.0).abs() < 1e-6, "{got:?}");
    }

    #[test]
    fn widen_moves_only_the_backwards_legs() {
        let chain = route(&[(0, 1), (1, 2)], 2);
        let spend = vec![10.0, 20.0];
        // Both drops are forwards here, so nothing moves.
        assert_eq!(widen(&chain, &[1.0, 1.0], 100.0, &spend, 99.0), spend);
    }
}
