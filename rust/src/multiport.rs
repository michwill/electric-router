//! One pool touched once, with several ports.
//!
//! The mirror of `core/multiport.py`.
//!
//! A pool with `N` coins can serve more than two ports in a single
//! interaction:
//!
//!     XDAI -> (3pool) -> 3Crv + USDC.e
//!     DAI  -> (3pool) -> USDC + USDT
//!
//! Modelled as separate arcs those are two entries into one pool, which is two
//! `psi^2/2G` terms with no cross-term and two calibrations against a state
//! only the first of them will see. Modelled as one **element** the pool
//! appears once, so there is no stale second calibration and nothing to
//! cross-couple -- decision 3 is satisfied by construction rather than
//! enforced.
//!
//! This module is the representation and the arithmetic. It does not touch the
//! solve; see `docs/multi-port-elements.md` for why that is a separate step.
//!
//! The port bound is the structure's, not a cap:
//!
//!     #coin-ports in + #coin-ports out <= N
//!
//! because each port occupies a distinct coin. One coin cannot be both an
//! input and an output -- that is a wash -- and two input ports on one coin
//! are just one larger port, so ports map injectively onto coins. The **LP
//! token is not one of the `N`** and so consumes no slot: counting it would
//! reject `add_liquidity` of both coins of a 2-coin pool, which is a real
//! operation.

use crate::pools::lp::StableLp;
use crate::pools::stableswap;
use crate::types::{ArcKind, PoolArc};
use ruint::aliases::U256;
use std::fmt;

/// A port on the LP token rather than on one of the pool's coins.
pub const LP: i32 = -1;

/// Shares are integers, in basis points, and the last port on a side takes the
/// remainder. Not a style choice: `int(amount * 0.25)` on a wei-scale integer
/// silently loses the low digits -- `f64` carries ~15 of them and a 400,000
/// token deposit needs 24 -- so a float split disagrees with the integer one
/// it is meant to stand for. This is also the arithmetic `Leg.bps` already
/// executes, so an element splits the way the router does.
pub const BPS: i64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiPortError(pub String);

impl fmt::Display for MultiPortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn refuse<T>(message: impl Into<String>) -> Result<T, MultiPortError> {
    Err(MultiPortError(message.into()))
}

/// One side of an element: which token, and its share of that side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port {
    /// a coin index, or `LP`
    pub coin: i32,
    /// share of this side's total, in basis points
    pub bps: i64,
}

impl Port {
    pub fn new(coin: i32, bps: i64) -> Self {
        Self { coin, bps }
    }
}

/// `inputs -> outputs` through one pool, priced on advancing state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiPort {
    pub pool: String,
    pub n_coins: i32,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

impl MultiPort {
    /// The reference's `__post_init__`, which is the whole of the structure's
    /// rule. Constructing is the only way in, so a `MultiPort` that exists is
    /// admissible.
    pub fn new(
        pool: impl Into<String>,
        n_coins: i32,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Self, MultiPortError> {
        if inputs.is_empty() || outputs.is_empty() {
            return refuse("an element needs a port on each side");
        }
        let seen: Vec<i32> = inputs
            .iter()
            .chain(outputs.iter())
            .map(|p| p.coin)
            .filter(|&c| c != LP)
            .collect();
        for &coin in &seen {
            if !(0..n_coins).contains(&coin) {
                return refuse(format!("coin {coin} out of range"));
            }
        }
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != seen.len() {
            // The injection that makes the bound below the right one: a coin
            // on both sides is a wash, and twice on one side is one port.
            return refuse("a coin may hold at most one port");
        }
        if seen.len() > n_coins.max(0) as usize {
            return refuse(format!(
                "{} coin-ports on a {n_coins}-coin pool",
                seen.len()
            ));
        }
        if inputs.iter().filter(|p| p.coin == LP).count() > 1
            || outputs.iter().filter(|p| p.coin == LP).count() > 1
        {
            return refuse("one LP port per side");
        }
        for (side, name) in [(&inputs, "inputs"), (&outputs, "outputs")] {
            if side.iter().any(|p| !(0 < p.bps && p.bps <= BPS)) {
                return refuse(format!("{name}: shares must be in (0, {BPS}]"));
            }
            if side.iter().map(|p| p.bps).sum::<i64>() != BPS {
                return refuse(format!("{name}: shares must sum to {BPS}"));
            }
        }
        Ok(Self { pool: pool.into(), n_coins, inputs, outputs })
    }

    pub fn ports(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }
}

/// `(port, amount)` per port, in integers, last one taking the rest.
fn split(ports: &[Port], total: U256) -> Vec<(Port, U256)> {
    let bps = U256::from(BPS as u64);
    let mut left = total;
    let mut out = Vec::with_capacity(ports.len());
    for (k, port) in ports.iter().enumerate() {
        let share = if k == ports.len() - 1 {
            left
        } else {
            total * U256::from(port.bps as u64) / bps
        };
        left -= share;
        out.push((*port, share));
    }
    out
}

/// What one evaluation left behind: the outputs, and the pool and LP model as
/// the last leg through them ended.
pub struct Evaluated {
    pub outputs: Vec<U256>,
    pub pool: stableswap::Pool,
    pub lp: Option<StableLp>,
}

/// `(outputs, pool after, lp after)` for one unit of `element`.
///
/// `lp` is the pool's `StableLp`, or `None` when the element has no LP port.
/// Every leg is priced against the pool as the previous leg left it, which is
/// what makes this one element rather than several arcs: the coupling *is* the
/// advancing state.
///
/// Two shapes are served today, which are the ones that arise:
///
/// * **one in, many out** -- split the input by the output weights, then one
///   `exchange` per coin port and one `add_liquidity` for an LP port;
/// * **many in, one out** -- an `add_liquidity` whose amounts vector is the
///   input weights, which is a single call and needs no ordering.
///
/// The general `j`-in-`k`-out case needs a pairing rule between the sides and
/// is deliberately refused rather than guessed at.
pub fn evaluate(
    element: &MultiPort,
    pool: &stableswap::Pool,
    lp: Option<&StableLp>,
    amount_in: U256,
) -> Result<Evaluated, MultiPortError> {
    if amount_in.is_zero() {
        return refuse("nothing to route");
    }
    if element.inputs.len() > 1 && element.outputs.len() > 1 {
        return refuse("many-in many-out needs a pairing rule");
    }

    if element.inputs.len() > 1 {
        // k in, 1 out. Only the LP token can absorb several coins at once.
        let out = element.outputs[0];
        let (Some(model), true) = (lp, out.coin == LP) else {
            return refuse("several inputs pay only an LP port");
        };
        let mut amounts = vec![U256::ZERO; element.n_coins.max(0) as usize];
        for (port, share) in split(&element.inputs, amount_in) {
            if port.coin == LP {
                return refuse("an LP input cannot join a deposit");
            }
            amounts[port.coin as usize] = share;
        }
        let (minted, after) = model
            .add_liquidity(&amounts)
            .ok_or_else(|| MultiPortError("the deposit does not price".into()))?;
        return Ok(Evaluated {
            outputs: vec![minted],
            pool: after.pool.clone(),
            lp: Some(after),
        });
    }

    // 1 in, k out.
    let source = element.inputs[0];
    let mut pool = pool.clone();
    let mut lp = lp.map(|model| model.clone_model());
    let mut outs: Vec<U256> = Vec::with_capacity(element.outputs.len());
    for (port, share) in split(&element.outputs, amount_in) {
        if share.is_zero() {
            return refuse("a port was allocated nothing");
        }
        if port.coin == LP {
            if lp.is_none() || source.coin == LP {
                return refuse("no LP model for an LP port");
            }
            let mut amounts = vec![U256::ZERO; element.n_coins.max(0) as usize];
            amounts[source.coin as usize] = share;
            let (minted, after) = lp
                .as_ref()
                .unwrap()
                .add_liquidity(&amounts)
                .ok_or_else(|| MultiPortError("the deposit does not price".into()))?;
            pool = after.pool.clone();
            lp = Some(after);
            outs.push(minted);
        } else if source.coin == LP {
            let Some(model) = lp.as_ref() else {
                return refuse("no LP model for an LP input");
            };
            outs.push(
                model
                    .calc_withdraw_one_coin(share, port.coin as usize)
                    .ok_or_else(|| MultiPortError("the withdrawal does not price".into()))?,
            );
        } else {
            let (dy, after) = pool
                .exchange(source.coin as usize, port.coin as usize, share)
                .ok_or_else(|| MultiPortError("the swap does not price".into()))?;
            pool = after;
            if let Some(model) = lp.as_ref() {
                lp = Some(model.with_pool(pool.clone()));
            }
            outs.push(dy);
        }
    }
    Ok(Evaluated { outputs: outs, pool, lp })
}

/// The two-port split that maximises what the element pays out.
///
/// `value(port_index, amount) -> f64` prices each port's token, because the
/// ports pay different tokens and only the caller knows what they are worth.
///
/// This is the piece the pin sweep approximates. §6.3 sweeps an allocation
/// over `psi* x {0, 1/8, 1/4, 1/2, 1, 2, 4}` because the model underneath it
/// cannot be trusted; here the element's own arithmetic *is* trustworthy -- it
/// advances the pool between legs, matching execution to within a wei on the
/// shapes tested -- so the split can be optimised rather than bracketed.
///
/// Ternary search, because the objective is concave in the split: each port's
/// output is concave in its own share, and a sum of concave functions of a
/// linear split is concave. That holds for `exchange` and `add_liquidity` on a
/// stableswap in its normal range; it is not asserted for a pool being pushed
/// off its peg, where §2.5's split arcs apply and the caller should not be
/// here. Falls back to the best grid point if the search leaves the feasible
/// interior.
///
/// Only two ports. Three needs a simplex search and no caller wants one yet.
pub fn best_split(
    element: &MultiPort,
    pool: &stableswap::Pool,
    lp: Option<&StableLp>,
    amount_in: U256,
    value: impl Fn(usize, U256) -> f64,
    grid: usize,
) -> Result<(MultiPort, f64), MultiPortError> {
    if element.inputs.len() != 1 || element.outputs.len() != 2 {
        return refuse("best_split is for one-in two-out elements");
    }

    let payout = |bps: i64| -> f64 {
        let Ok(at) = at(element, bps) else {
            return f64::NEG_INFINITY;
        };
        match evaluate(&at, pool, lp, amount_in) {
            Err(_) => f64::NEG_INFINITY,
            Ok(done) => done
                .outputs
                .iter()
                .enumerate()
                .map(|(k, out)| value(k, *out))
                .sum(),
        }
    };

    let (mut low, mut high) = (1i64, BPS - 1);
    for _ in 0..grid {
        if high - low < 3 {
            break;
        }
        let left = low + (high - low) / 3;
        let right = high - (high - low) / 3;
        if payout(left) < payout(right) {
            low = left;
        } else {
            high = right;
        }
    }
    // `max(range, key=...)` keeps the first of a tie, which is what a plain
    // `>` comparison does here.
    let mut best = low;
    let mut found = f64::NEG_INFINITY;
    for candidate in low..=high {
        let got = payout(candidate);
        if got > found {
            found = got;
            best = candidate;
        }
    }
    if found == f64::NEG_INFINITY {
        return refuse("no feasible split");
    }
    Ok((at(element, best)?, found))
}

/// `element` with its two output ports split `bps` / `BPS - bps`.
fn at(element: &MultiPort, bps: i64) -> Result<MultiPort, MultiPortError> {
    let (first, second) = (element.outputs[0], element.outputs[1]);
    MultiPort::new(
        element.pool.clone(),
        element.n_coins,
        element.inputs.clone(),
        vec![Port::new(first.coin, bps), Port::new(second.coin, BPS - bps)],
    )
}

/// `(input port, output port)` for one leg, `LP` for the LP token.
///
/// A withdrawal is absent from what an element can *advance* for the same
/// reason `realize::ADVANCEABLE` omits it: `remove_liquidity_one_coin`'s
/// effect on the supply has not been read off the deployed source, and an
/// element prices on advancing state or not at all.
pub fn ports_of(kind: ArcKind, i: i32, j: i32) -> Result<(i32, i32), MultiPortError> {
    let name = kind.name();
    if name.starts_with("SWAP") {
        return Ok((i, j));
    }
    if name.starts_with("DEPOSIT") {
        return Ok((i, LP));
    }
    if name.starts_with("WITHDRAW") {
        return Ok((LP, j));
    }
    refuse(format!("{name} is not a port of a pool"))
}

/// The same rule on `PoolArc`s, before realisation orders them into legs.
pub fn element_of_arcs(arcs: &[PoolArc]) -> Result<MultiPort, MultiPortError> {
    let Some(first) = arcs.first() else {
        return refuse("no arcs");
    };
    let triples: Vec<(ArcKind, i32, i32)> =
        arcs.iter().map(|a| (a.kind, a.i, a.j)).collect();
    element_from(&first.pool, first.n_coins, &triples)
}

/// The element `(kind, i, j)` triples on one pool form, or why they do not.
///
/// This is what replaces the re-entry exemption. Two arcs of one pool used to
/// be admitted when every leg but the last could be advanced past, which
/// priced them as two independent resistors -- no cross-term, each calibrated
/// at a state the other never sees. An element is the same trade with the pool
/// appearing *once*, so the coupling is the advancing state rather than
/// something the model has to carry separately, and the port bound is
/// structural: `#coin-ports in + #coin-ports out <= N`, because a coin holds
/// at most one port. A 2-coin pool therefore admits exactly one in and one out
/// -- it cannot be re-entered at all, which is the whole point.
///
/// Shares are left at whatever the legs already carry; `best_split` is what
/// chooses them. Here the question is only whether the shape is admissible.
pub fn element_from(
    pool: &str,
    n_coins: i32,
    triples: &[(ArcKind, i32, i32)],
) -> Result<MultiPort, MultiPortError> {
    let pool = pool.to_ascii_lowercase();
    // Insertion-ordered and deduped, as `dict.fromkeys` is: several legs
    // drawing on one coin are *one* input port, which is what makes 1-in-k-out
    // a single element. A coin appearing on both sides is a wash and
    // `MultiPort` refuses it there, along with the `#in + #out <= N` bound.
    let mut ins: Vec<i32> = Vec::new();
    let mut outs: Vec<i32> = Vec::new();
    for &(kind, i, j) in triples {
        let (source, sink) = ports_of(kind, i, j)?;
        if !ins.contains(&source) {
            ins.push(source);
        }
        if !outs.contains(&sink) {
            outs.push(sink);
        }
    }
    // An LP *input* would be several `calc_withdraw_one_coin` against one
    // state: `evaluate` cannot advance a withdrawal, because its effect on the
    // supply has not been read off the deployed source. Pricing two burns
    // against the same supply is the error an element exists to prevent, so
    // refuse it here rather than answer confidently.
    if ins.contains(&LP) && outs.len() > 1 {
        return refuse("a multi-output withdrawal cannot be advanced");
    }
    // Built first, so the injection and the `#in + #out <= N` bound answer
    // before the shape does -- "a coin may hold at most one port" is the useful
    // thing to say about a re-entered 2-coin pool.
    let element = MultiPort::new(pool, n_coins, even(&ins), even(&outs))?;
    if ins.len() > 1 && outs.len() > 1 {
        return refuse("many-in many-out needs a pairing rule");
    }
    // One leg per port on the many side. Two arcs sharing *both* ports dedupe
    // to a 1-in-1-out and would read as admissible; they are not an element
    // but §9.5's duplicate pair, which belongs in the graph as parallel
    // resistors and in a route as one leg.
    if triples.len() != ins.len().max(outs.len()) {
        return refuse(format!(
            "{} legs over {}-in {}-out: duplicate ports are a parallel pair, \
             not an element",
            triples.len(),
            ins.len(),
            outs.len()
        ));
    }
    Ok(element)
}

/// Equal shares over `coins`, the last taking the remainder.
///
/// A placeholder: `element_from` answers whether the *shape* is admissible,
/// and `best_split` is what chooses the shares. Even is the honest starting
/// point -- it asserts nothing about the split the solver wanted.
fn even(coins: &[i32]) -> Vec<Port> {
    let n = coins.len() as i64;
    if n == 0 {
        return Vec::new();
    }
    let each = BPS / n;
    coins
        .iter()
        .enumerate()
        .map(|(k, &coin)| {
            Port::new(coin, if (k as i64) < n - 1 { each } else { BPS - each * (n - 1) })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_in(coin: i32) -> Vec<Port> {
        vec![Port::new(coin, BPS)]
    }

    #[test]
    fn a_coin_may_hold_at_most_one_port() {
        let err = MultiPort::new("0xp", 3, one_in(0), vec![Port::new(0, BPS)]).unwrap_err();
        assert_eq!(err.0, "a coin may hold at most one port");
    }

    #[test]
    fn a_two_coin_pool_cannot_be_re_entered() {
        // Two swaps through a 2-coin pool: three distinct coin-ports over two
        // coins, which the injection refuses before the shape is considered.
        let err = element_from(
            "0xp",
            2,
            &[
                (ArcKind::SwapStable, 0, 1),
                (ArcKind::SwapStable, 1, 0),
            ],
        )
        .unwrap_err();
        assert_eq!(err.0, "a coin may hold at most one port");
    }

    #[test]
    fn the_lp_token_consumes_no_slot() {
        // add_liquidity of both coins of a 2-coin pool is a real operation.
        let element = element_from(
            "0xp",
            2,
            &[
                (ArcKind::DepositFixed, 0, 0),
                (ArcKind::DepositFixed, 1, 0),
            ],
        )
        .unwrap();
        assert_eq!(element.inputs, vec![Port::new(0, 5_000), Port::new(1, 5_000)]);
        assert_eq!(element.outputs, vec![Port::new(LP, BPS)]);
    }

    #[test]
    fn duplicate_ports_are_a_parallel_pair_not_an_element() {
        let err = element_from(
            "0xp",
            3,
            &[
                (ArcKind::SwapStable, 0, 1),
                (ArcKind::SwapStable, 0, 1),
            ],
        )
        .unwrap_err();
        assert!(err.0.contains("duplicate ports are a parallel pair"));
    }

    #[test]
    fn a_multi_output_withdrawal_is_refused() {
        let err = element_from(
            "0xp",
            3,
            &[
                (ArcKind::WithdrawStable, 0, 0),
                (ArcKind::WithdrawStable, 0, 1),
            ],
        )
        .unwrap_err();
        assert_eq!(err.0, "a multi-output withdrawal cannot be advanced");
    }

    #[test]
    fn shares_split_with_the_remainder_last() {
        let ports = vec![Port::new(0, 3_333), Port::new(1, 3_333), Port::new(2, 3_334)];
        let got: Vec<u128> = split(&ports, U256::from(100u8))
            .iter()
            .map(|(_, v)| v.to::<u128>())
            .collect();
        // 33 + 33 + the rest, which is 34 rather than a rounded-down 33.
        assert_eq!(got, vec![33, 33, 34]);
        assert_eq!(got.iter().sum::<u128>(), 100);
    }

    #[test]
    fn even_shares_put_the_remainder_on_the_last_port() {
        assert_eq!(
            even(&[0, 1, 2]),
            vec![Port::new(0, 3_333), Port::new(1, 3_333), Port::new(2, 3_334)]
        );
    }

    #[test]
    fn a_kind_that_is_not_a_pool_port_is_refused() {
        let err = ports_of(ArcKind::WrapNative, 0, 1).unwrap_err();
        assert_eq!(err.0, "WRAP_NATIVE is not a port of a pool");
        assert_eq!(ports_of(ArcKind::DepositDyn, 2, 9).unwrap(), (2, LP));
        assert_eq!(ports_of(ArcKind::WithdrawCrypto, 9, 2).unwrap(), (LP, 2));
    }
}
