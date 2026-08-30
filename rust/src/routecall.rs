//! Turn a realised route into the calldata `ElectricRouter.execute` takes.
//!
//! The mirror of `core/routecall.py`. Two conversions happen here, and both
//! are places the router differs from what the model produced.
//!
//! **Fractions become fractions of what is left.** `Leg.bps` is a share of the
//! balance a node held when its group of outgoing legs opened; the router asks
//! instead for a share of the balance standing there right now, so a 50/50
//! split is `50%` then `100%`. The second form cannot be starved by a first
//! leg that returned less than modelled, which is the only reason to prefer
//! it.
//!
//! **Every leg gets its own minimum rate.** A single end-to-end bound lets a
//! route give everything away in one pool and win it back in another -- the
//! shape of a sandwich. The bound is a fraction of the fee that pool is
//! measured to be charging *on this trade*, and what it buys is a ceiling on
//! how much a sandwich can take: the front-run can only be as large as the
//! bound will still settle.

use crate::codec::{encode_call, Value};
use crate::realize::{RealizedLeg, RealizedRoute};
use crate::slippage::{divide, widen};
use crate::types::ArcKind;
use ruint::aliases::U256;
use std::fmt;

/// `1e18`, the fixed-point one the contract reads fractions and rates in.
pub fn one() -> U256 {
    U256::from(10u64).pow(U256::from(18u64))
}

/// Curve's sentinel for native ETH.
pub const NATIVE: &str = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

pub const MAX_LEGS: usize = 32;
pub const MAX_TOKENS: usize = 31;

// Packing, low bit first. Must match `contracts/ElectricRouter.vy`.
pub const FRAC_BITS: usize = 60;
pub const RATE_SHIFT: usize = 60;
pub const RATE_BITS: usize = 128;
pub const I_SHIFT: usize = 188;
pub const J_SHIFT: usize = 192;
pub const N_SHIFT: usize = 196;
pub const KIND_SHIFT: usize = 200;
pub const IN_REF_SHIFT: usize = 205;
pub const OUT_REF_SHIFT: usize = 210;
pub const RESERVED_SHIFT: usize = 215;

pub fn max_rate() -> U256 {
    (U256::from(1u8) << RATE_BITS) - U256::from(1u8)
}

/// Vyper writes one entry point per default argument, so a call that wants
/// none of the trailing three can be sent as the shortest signature that still
/// says what it means. Three words of calldata is real money on an L2, and
/// Curve's lending callbacks pass the whole thing through as `bytes`.
pub const SIGNATURES: [&str; 4] = [
    "execute(uint256,address[],uint256[],bool)",
    "execute(uint256,address[],uint256[],bool,address[])",
    "execute(uint256,address[],uint256[],bool,address[],address)",
    "execute(uint256,address[],uint256[],bool,address[],address,uint256)",
];

/// How much of a pool's own fee a sandwich is allowed to take before the leg
/// refuses -- and, measured, almost exactly how much one does take when it
/// can.
pub const FEE_SHARE: f64 = 0.2;

/// A floor on the tolerance for a pair whose price genuinely moves between the
/// quote and the block it lands in, in bp. It is a real slippage allowance and
/// it is a real cost: measured against the deployed TricryptoUSDC, the fee
/// rule alone grants 0.60 bp and a sandwich nets 0.52 bp of the trade at
/// $15,000, while 5 bp lets the same attack take about eight times that.
pub const VOLATILE_FLOOR_BP: f64 = 5.0;

/// A floor on that tolerance, in bp, and not slippage at all: room for the wei
/// of rounding a wrap or a rebasing token loses on the way through, and the
/// only thing standing under a leg whose fee could not be measured.
pub const FLOOR_BP: f64 = 0.1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingError(pub String);

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

type Result<T> = std::result::Result<T, EncodingError>;

/// One packed leg, as the contract unpacks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub pool: String,
    pub kind: ArcKind,
    pub i: i32,
    pub j: i32,
    pub n: i32,
    pub frac: U256,
    pub min_rate: U256,
    /// 0 means "read the token off the pool"; otherwise an index into the
    /// call's `tokens`, one-based so that zero can be the sentinel.
    pub in_ref: usize,
    pub out_ref: usize,
}

impl Step {
    pub fn pack(&self) -> Result<U256> {
        if self.frac.is_zero() || self.frac > one() {
            return Err(EncodingError(format!(
                "frac {} outside (0, 1e18]",
                self.frac
            )));
        }
        if self.min_rate > max_rate() {
            return Err(EncodingError(format!(
                "min_rate {} does not fit in {RATE_BITS} bits",
                self.min_rate
            )));
        }
        for (name, value, limit) in [
            ("i", self.i as i64, 15i64),
            ("j", self.j as i64, 15),
            ("n", self.n as i64, 15),
            ("kind", self.kind.code() as i64, 31),
            ("in_ref", self.in_ref as i64, MAX_TOKENS as i64),
            ("out_ref", self.out_ref as i64, MAX_TOKENS as i64),
        ] {
            if !(0..=limit).contains(&value) {
                return Err(EncodingError(format!("{name}={value} outside 0..{limit}")));
            }
        }
        Ok(self.frac
            | (self.min_rate << RATE_SHIFT)
            | (U256::from(self.i as u64) << I_SHIFT)
            | (U256::from(self.j as u64) << J_SHIFT)
            | (U256::from(self.n as u64) << N_SHIFT)
            | (U256::from(self.kind.code()) << KIND_SHIFT)
            | (U256::from(self.in_ref as u64) << IN_REF_SHIFT)
            | (U256::from(self.out_ref as u64) << OUT_REF_SHIFT))
    }
}

pub fn unpack(word: U256, pool: &str) -> Result<Step> {
    if !(word >> RESERVED_SHIFT).is_zero() {
        return Err(EncodingError("reserved bits set".into()));
    }
    let field = |shift: usize, mask: u64| ((word >> shift) & U256::from(mask)).to::<u64>();
    let code = field(KIND_SHIFT, 31) as u8;
    let kind = ArcKind::from_code(code)
        .ok_or_else(|| EncodingError(format!("no such kind: {code}")))?;
    Ok(Step {
        pool: pool.to_string(),
        kind,
        i: field(I_SHIFT, 15) as i32,
        j: field(J_SHIFT, 15) as i32,
        n: field(N_SHIFT, 15) as i32,
        frac: word & ((U256::from(1u8) << FRAC_BITS) - U256::from(1u8)),
        min_rate: (word >> RATE_SHIFT) & max_rate(),
        in_ref: field(IN_REF_SHIFT, 31) as usize,
        out_ref: field(OUT_REF_SHIFT, 31) as usize,
    })
}

/// A ready-to-send call, and what it is and is not protecting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RouteCall {
    pub amount_in: U256,
    pub pools: Vec<String>,
    pub params: Vec<U256>,
    pub tokens: Vec<String>,
    pub set_approvals: bool,
    pub receiver: String,
    pub min_out: U256,
    /// What the caller has to hold and approve, and what comes back. Neither
    /// appears in the calldata -- the router reads them -- so they are carried
    /// here rather than made the caller's problem to re-derive.
    pub token_in: String,
    pub token_out: String,
    /// What the per-leg bounds alone promise, against what the route was
    /// quoted at. The gap between them is the slippage the caller is granting.
    pub guaranteed_out: U256,
    /// The chained walk's figure where there was one, the model's otherwise --
    /// the number the user was shown, so the number the tolerance is against.
    pub quoted_out: U256,
    /// Legs whose bound imposes no floor at all.
    pub unbounded: Vec<usize>,
}

impl RouteCall {
    /// The shortest entry point that still expresses this call.
    ///
    /// An empty `receiver` means "whoever sends it", which is the contract's
    /// own default and therefore a word that does not have to be sent. Naming
    /// `sender` buys the same saving for a call that pays its own sender.
    pub fn calldata(&self, sender: &str) -> Result<Vec<u8>> {
        let pays_the_sender = self.receiver.is_empty()
            || (!sender.is_empty()
                && self.receiver.to_ascii_lowercase() == sender.to_ascii_lowercase());
        if self.receiver.is_empty() && !self.min_out.is_zero() {
            return Err(EncodingError(
                "min_out needs a receiver: a call cannot default the one and \
                 send the other"
                    .into(),
            ));
        }
        let args = [
            Value::Uint(self.amount_in),
            Value::Array(self.pools.iter().map(|p| Value::Address(p.clone())).collect()),
            Value::Array(self.params.iter().map(|p| Value::Uint(*p)).collect()),
            Value::Bool(self.set_approvals),
            Value::Array(self.tokens.iter().map(|t| Value::Address(t.clone())).collect()),
            Value::Address(self.receiver.clone()),
            Value::Uint(self.min_out),
        ];
        let mut keep = 7usize;
        if self.min_out.is_zero() {
            keep = 6;
            if pays_the_sender {
                keep = 5;
                if self.tokens.is_empty() {
                    keep = 4;
                }
            }
        }
        encode_call(SIGNATURES[keep - 4], &args[..keep])
            .map_err(|e| EncodingError(e.0))
    }

    /// How far below the quote the route may land without reverting.
    pub fn tolerance_bp(&self) -> f64 {
        if self.quoted_out.is_zero() {
            return 0.0;
        }
        (1.0 - crate::pools::divided(self.guaranteed_out, self.quoted_out)) * 1e4
    }

    pub fn steps(&self) -> Result<Vec<Step>> {
        self.pools
            .iter()
            .zip(self.params.iter())
            .map(|(pool, word)| unpack(*word, pool))
            .collect()
    }
}

// ------------------------------------------------------------- fractions

/// Each leg's share of the balance standing at its source when it runs.
///
/// Taken from the modelled amounts rather than from `Leg.bps`, so the split is
/// whatever the solver actually chose and not a basis-point rounding of it.
/// The last leg out of a node always takes everything, which is what stops
/// dust accumulating a node at a time.
pub fn fractions(route: &RealizedRoute) -> Result<Vec<U256>> {
    let mut balances: Vec<(i32, U256)> = vec![(0, route.amount_in)];
    let get = |balances: &Vec<(i32, U256)>, slot: i32| -> U256 {
        balances.iter().find(|(s, _)| *s == slot).map_or(U256::ZERO, |&(_, v)| v)
    };
    let set = |balances: &mut Vec<(i32, U256)>, slot: i32, value: U256| {
        match balances.iter_mut().find(|(s, _)| *s == slot) {
            Some(entry) => entry.1 = value,
            None => balances.push((slot, value)),
        }
    };
    let mut out = Vec::with_capacity(route.legs.len());
    for (k, realized) in route.legs.iter().enumerate() {
        let src = realized.leg.src_slot;
        let have = get(&balances, src);
        let last = !route.legs[k + 1..].iter().any(|later| later.leg.src_slot == src);
        let (frac, take) = if last || realized.amount_in >= have {
            (one(), have)
        } else if have.is_zero() {
            return Err(EncodingError(format!(
                "leg {k} spends slot {src}, which nothing has filled"
            )));
        } else {
            (realized.amount_in * one() / have, realized.amount_in)
        };
        out.push(frac);
        set(&mut balances, src, have - take);
        let dst = realized.leg.dst_slot;
        let landed = get(&balances, dst) + leg_out(realized);
        set(&mut balances, dst, landed);
    }
    Ok(out)
}

/// What this leg produces: its own pool's answer, or the model's.
///
/// The two differ by tens of basis points on a cryptoswap leg, which does not
/// matter for a share of a node -- both branches out of it move together --
/// and matters entirely for a bound on the leg itself.
pub fn leg_out(realized: &RealizedLeg) -> U256 {
    if realized.verified_out.is_zero() {
        realized.amount_out
    } else {
        realized.verified_out
    }
}

/// The size this leg's output was measured at.
///
/// `leg_out` is the pool's own answer where there is one, and it was quoted at
/// the leg's real input rather than its modelled one. Dividing that by the
/// modelled input would inflate the rate by exactly the gap the measurement
/// exists to close.
pub fn leg_in(realized: &RealizedLeg) -> U256 {
    if realized.verified_in.is_zero() {
        realized.amount_in
    } else {
        realized.verified_in
    }
}

// ------------------------------------------------------------- min rates

fn usable(value: f64) -> bool {
    value.is_finite() && (0.0..1.0).contains(&value)
}

/// What this leg pays in fees, at its own size where that is known.
///
/// `fee_frac` is the pool's own model charging the real trade; `gamma_live` is
/// two tiny probes measuring the marginal fee. They agree on a fixed-fee pool
/// and diverge on a dynamic one exactly when it matters -- the trade that
/// skews the pool is the trade the fee climbs for.
pub fn leg_fee(realized: &RealizedLeg) -> f64 {
    if usable(realized.fee_frac) {
        return realized.fee_frac;
    }
    let gamma = realized.gamma_live;
    if !gamma.is_finite() || !(gamma > 0.0 && gamma <= 1.0) {
        return 0.0;
    }
    1.0 - gamma
}

/// What the minimum rate is set from: the least this pool can charge.
///
/// Not what the leg pays. A sandwich front-runs and unwinds in small, balanced
/// trades and is charged near `mid_fee`, while the leg it wraps pays the
/// dynamic fee at its own size -- measured on TricryptoUSDC, 3 bp against
/// 13 bp. Bounding on the larger of the two hands the attacker the gap:
/// against the deployed pool on a fork, it cost the victim 2.72 bp instead of
/// 0.60 bp for the same attack.
pub fn bounding_fee(realized: &RealizedLeg) -> f64 {
    if usable(realized.fee_floor) {
        return realized.fee_floor;
    }
    leg_fee(realized)
}

/// The room each leg needs for what moves under it, whatever it charges.
///
/// `volatile_floor_bp` where the pair genuinely moves between the quote and
/// the block it lands in, and the wei a wrap or a rebasing token rounds away
/// everywhere else. Distinct from the fee rule above it, which is a ceiling on
/// what a sandwich can take rather than an allowance for movement -- so this
/// is the part no caller-named budget gets to take back.
pub fn movement_floors(
    route: &RealizedRoute,
    volatile: &[String],
    floor_bp: f64,
    volatile_floor_bp: f64,
) -> Vec<f64> {
    let loose: Vec<String> = volatile.iter().map(|a| a.to_ascii_lowercase()).collect();
    route
        .legs
        .iter()
        .map(|realized| {
            let bp = if loose.contains(&realized.target.to_ascii_lowercase()) {
                volatile_floor_bp
            } else {
                floor_bp
            };
            bp / 1e4
        })
        .collect()
}

/// How far below its quote the automatic rule lets each leg land.
///
/// A fifth of the least that pool can charge, over `movement_floors`. These
/// are also the weights a caller-named budget is divided by, which is what
/// keeps the two rules the same shape.
pub fn tolerances(
    route: &RealizedRoute,
    volatile: &[String],
    fee_share: f64,
    floor_bp: f64,
    volatile_floor_bp: f64,
) -> Vec<f64> {
    let floors = movement_floors(route, volatile, floor_bp, volatile_floor_bp);
    route
        .legs
        .iter()
        .zip(floors.iter())
        .map(|(realized, &floor)| (fee_share * bounding_fee(realized)).max(floor).min(1.0))
        .collect()
}

/// Everything `min_rates` takes past the route.
pub struct Policy<'a> {
    pub volatile: &'a [String],
    pub fee_share: f64,
    pub floor_bp: f64,
    pub volatile_floor_bp: f64,
    /// Replaces the whole rule with a total the caller names.
    pub slippage_bp: Option<f64>,
}

impl Default for Policy<'_> {
    fn default() -> Self {
        Self {
            volatile: &[],
            fee_share: FEE_SHARE,
            floor_bp: FLOOR_BP,
            volatile_floor_bp: VOLATILE_FLOOR_BP,
            slippage_bp: None,
        }
    }
}

/// `(min_rate per leg, indices the bound does not really cover)`.
///
/// `slippage_bp` replaces the whole rule with a total the caller names,
/// divided between the legs as voltage divides between resistors: in
/// proportion to the fees along a series route, and once rather than once per
/// branch on one that splits. `movement_floors` is the one thing the budget
/// cannot take back.
///
/// A tolerance finer than one unit of the output token cannot be expressed as
/// a rate -- `min_rate` is `out * 1e18 // in`, so one unit is `1/out` of it,
/// and a leg making a few thousand units quantises harder than the room the
/// fee rule asks for. What ships then is the tightest rate that still leaves a
/// whole unit: the finest tolerance the token has, and never less than one.
///
/// Binding at the quote itself does not survive contact. The route's own
/// arithmetic moves a leg by a wei with no market behind it -- a downstream
/// `dx` is a 60-bit fraction of a standing balance, and the sweeper takes
/// whatever earlier divisions stranded in the slot -- so a leg with no room
/// reverts on rounding. Measured: the tBTC -> USDT dust route tripped its own
/// bound at some blocks and not others, on a fork, where nothing moved at all.
pub fn min_rates(route: &RealizedRoute, policy: &Policy<'_>) -> Result<(Vec<U256>, Vec<usize>)> {
    let mut grant = tolerances(
        route, policy.volatile, policy.fee_share, policy.floor_bp,
        policy.volatile_floor_bp,
    );
    if let Some(slippage_bp) = policy.slippage_bp {
        // Movement is not slippage, so it is not the budget's to spend: a leg
        // keeps its floor however little the caller granted. The automatic
        // tolerances go in twice -- as the weights, and as the backstop under
        // a leg the network runs backwards, which divides to nothing.
        let floors = movement_floors(
            route, policy.volatile, policy.floor_bp, policy.volatile_floor_bp,
        );
        let budget = slippage_bp / 1e4;
        let share = divide(route, &grant, budget, Some(&grant))
            .map_err(EncodingError)?;
        // The bridge leg, and only it, is then raised to the whole budget: the
        // division leaves it at an imbalance the caller never chose, and it is
        // the leg that reverts on any movement at all.
        let share = widen(route, &grant, budget, &share, budget);
        grant = share
            .iter()
            .zip(floors.iter())
            .map(|(&value, &floor)| value.max(floor).min(1.0))
            .collect();
    }

    let billion = U256::from(1_000_000_000u64);
    let mut rates: Vec<U256> = Vec::with_capacity(route.legs.len());
    let mut unbounded: Vec<usize> = Vec::new();
    for (k, realized) in route.legs.iter().enumerate() {
        let (want, have) = (leg_out(realized), leg_in(realized));
        if have.is_zero() || want.is_zero() {
            rates.push(U256::ZERO);
            unbounded.push(k);
            continue;
        }
        let share = U256::from(py_round(grant[k] * 1e9).max(0.0) as u64);
        let rate = want * one() / have;
        let mut bound = rate - rate * share / billion;
        // A tolerance worth less than one unit of the output token has no rate
        // to be written as, so grant the unit: the tightest rate whose floor is
        // a whole unit under the quote.
        if want * share < billion {
            bound = (want * one() - U256::from(1u8)) / have;
        }
        if bound > max_rate() {
            return Err(EncodingError(format!(
                "leg {k} on {} has a raw-unit rate of {rate}, beyond what \
                 {RATE_BITS} bits can bound",
                realized.target
            )));
        }
        rates.push(bound);
        if (have * bound / one()).is_zero() {
            unbounded.push(k);
        }
    }
    Ok((rates, unbounded))
}

/// `round(x)` as CPython does it: ties to even.
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// `(what the bounds promise, the minimum output each leg enforces)`.
///
/// The router's own arithmetic, run with `dy = dx * min_rate / 1e18` at every
/// step. Worth computing twice over: the bounds compound, so the total is a
/// number a caller should see before signing rather than discover afterwards.
pub fn walk_bounds(
    route: &RealizedRoute,
    fracs: &[U256],
    rates: &[U256],
) -> (U256, Vec<U256>) {
    let mut balances: Vec<(i32, U256)> = vec![(0, route.amount_in)];
    let get = |balances: &Vec<(i32, U256)>, slot: i32| -> U256 {
        balances.iter().find(|(s, _)| *s == slot).map_or(U256::ZERO, |&(_, v)| v)
    };
    let set = |balances: &mut Vec<(i32, U256)>, slot: i32, value: U256| {
        match balances.iter_mut().find(|(s, _)| *s == slot) {
            Some(entry) => entry.1 = value,
            None => balances.push((slot, value)),
        }
    };
    let mut floors = Vec::with_capacity(route.legs.len());
    for (k, realized) in route.legs.iter().enumerate() {
        let (src, dst) = (realized.leg.src_slot, realized.leg.dst_slot);
        let dx = get(&balances, src) * fracs[k] / one();
        let floor = dx * rates[k] / one();
        floors.push(floor);
        let left = get(&balances, src) - dx;
        set(&mut balances, src, left);
        let landed = get(&balances, dst) + floor;
        set(&mut balances, dst, landed);
    }
    (get(&balances, route.dst_slot as i32), floors)
}

/// What the per-leg bounds alone promise, if every leg pays its minimum.
pub fn guaranteed_out(route: &RealizedRoute, fracs: &[U256], rates: &[U256]) -> U256 {
    walk_bounds(route, fracs, rates).0
}

// --------------------------------------------------------- token naming

/// How the contract works out a leg's token when the caller does not name it.
/// `Getter` is the one that can fail -- fourteen mainnet pools keep their LP
/// token elsewhere and expose no getter for it at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Derive {
    Coin,
    Getter,
    Target,
    Native,
}

fn derive_rule(kind: ArcKind) -> Option<(Derive, Derive)> {
    use ArcKind::*;
    use Derive::*;
    Some(match kind {
        SwapStable | SwapCrypto => (Coin, Coin),
        DepositFixed | DepositDyn | DepositFixedNoflag => (Coin, Getter),
        WithdrawStable | WithdrawCrypto => (Getter, Coin),
        Erc4626Deposit => (Getter, Target),
        Erc4626Redeem => (Target, Getter),
        WstethWrap => (Getter, Target),
        WstethUnwrap => (Target, Getter),
        LendMint => (Getter, Target),
        LendRedeem => (Target, Getter),
        WrapNative | StakeNative => (Native, Target),
        UnwrapNative => (Target, Native),
    })
}

/// What to put in `tokens`, which is a straight trade of calldata against gas.
///
/// `Needed` names only what the router cannot read for itself -- LP tokens,
/// vault assets, lending underlyings -- so a route made of swaps names
/// nothing. `None` names nothing at all and trusts every getter, which is the
/// shortest calldata there is. `All` names everything, which is the cheapest
/// to execute and the longest to send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Naming {
    Needed,
    None,
    All,
}

impl Naming {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "needed" => Naming::Needed,
            "none" => Naming::None,
            "all" => Naming::All,
            _ => return None,
        })
    }
}

fn must_name(rule: Derive, token: &str, target: &str, naming: Naming) -> bool {
    if naming == Naming::All {
        return true;
    }
    match rule {
        Derive::Target => token.to_ascii_lowercase() != target.to_ascii_lowercase(),
        Derive::Native => token.to_ascii_lowercase() != NATIVE,
        Derive::Getter => naming != Naming::None,
        // a pool coin: `i` and `j` already say which one
        Derive::Coin => false,
    }
}

// -------------------------------------------------------------- the call

/// Everything `encode_route` takes past the route.
pub struct Encode<'a> {
    pub receiver: &'a str,
    pub set_approvals: bool,
    pub min_out: U256,
    pub amount_in: Option<U256>,
    pub quoted_out: Option<U256>,
    pub naming: Naming,
    pub allow_unbounded: bool,
    pub policy: Policy<'a>,
}

impl<'a> Encode<'a> {
    pub fn new(receiver: &'a str) -> Self {
        Self {
            receiver,
            set_approvals: true,
            min_out: U256::ZERO,
            amount_in: None,
            quoted_out: None,
            naming: Naming::Needed,
            allow_unbounded: false,
            policy: Policy::default(),
        }
    }
}

/// Encode `route` for `ElectricRouter.execute`.
///
/// Tokens a leg does not name are read off the pool at execution time, which
/// is what makes `i` and `j` binding rather than advisory: the caller cannot
/// point a minimum rate at a token the pool does not hold there.
pub fn encode_route(route: &RealizedRoute, opts: &Encode<'_>) -> Result<RouteCall> {
    if route.legs.is_empty() {
        return Err(EncodingError(
            "route has no legs; an alias pair has nothing to execute".into(),
        ));
    }
    if route.legs.len() > MAX_LEGS {
        return Err(EncodingError(format!(
            "{} legs, the router takes {MAX_LEGS}",
            route.legs.len()
        )));
    }
    if route.legs[route.legs.len() - 1].leg.dst_slot != route.dst_slot as i32 {
        return Err(EncodingError(
            "the last leg does not produce the destination token".into(),
        ));
    }

    let fracs = fractions(route)?;
    let (rates, unbounded) = min_rates(route, &opts.policy)?;
    if !unbounded.is_empty() && !opts.allow_unbounded {
        let legs: Vec<String> = unbounded
            .iter()
            .map(|&k| {
                format!(
                    "{k} on {} ({} in, {} out)",
                    route.legs[k].target,
                    route.legs[k].amount_in,
                    leg_out(&route.legs[k])
                )
            })
            .collect();
        return Err(EncodingError(format!(
            "leg(s) {} produce too little for a minimum rate to bound -- the \
             floor it imposes rounds to nothing, so the check is vacuous. A leg \
             worth that little is not worth executing; re-solve without it, or \
             pass allow_unbounded to ship it unprotected",
            legs.join(", ")
        )));
    }
    // Both bounds are enforced, so the guarantee is the stronger one. It
    // matters where a bridge leg was widened past the budget: the per-leg walk
    // then reads worse than the caller can actually lose.
    let promised = walk_bounds(route, &fracs, &rates).0.max(opts.min_out);

    let mut tokens: Vec<String> = Vec::new();
    let mut steps: Vec<Step> = Vec::with_capacity(route.legs.len());
    for (k, realized) in route.legs.iter().enumerate() {
        let Some(rule) = derive_rule(realized.kind) else {
            return Err(EncodingError(format!(
                "leg {k}: {} is not executable",
                realized.kind.name()
            )));
        };
        let target = &realized.target;
        let mut reference = |token: &str, tokens: &mut Vec<String>| -> Result<usize> {
            let key = token.to_ascii_lowercase();
            if let Some(at) = tokens.iter().position(|t| *t == key) {
                return Ok(at + 1);
            }
            if tokens.len() >= MAX_TOKENS {
                return Err(EncodingError(format!(
                    "route names more than {MAX_TOKENS} tokens"
                )));
            }
            tokens.push(key);
            Ok(tokens.len())
        };
        let in_ref = if must_name(rule.0, &realized.token_in, target, opts.naming) {
            reference(&realized.token_in, &mut tokens)?
        } else {
            0
        };
        let out_ref = if must_name(rule.1, &realized.token_out, target, opts.naming) {
            reference(&realized.token_out, &mut tokens)?
        } else {
            0
        };
        steps.push(Step {
            pool: target.clone(),
            kind: realized.kind,
            i: realized.leg.i,
            j: realized.leg.j,
            n: realized.leg.n,
            frac: fracs[k],
            min_rate: rates[k],
            in_ref,
            out_ref,
        });
    }

    let params: Result<Vec<U256>> = steps.iter().map(Step::pack).collect();
    Ok(RouteCall {
        amount_in: opts.amount_in.unwrap_or(route.amount_in),
        pools: steps.iter().map(|s| s.pool.clone()).collect(),
        params: params?,
        tokens,
        set_approvals: opts.set_approvals,
        receiver: opts.receiver.to_string(),
        min_out: opts.min_out,
        token_in: route.legs[0].token_in.to_ascii_lowercase(),
        token_out: route.legs[route.legs.len() - 1].token_out.to_ascii_lowercase(),
        guaranteed_out: promised,
        quoted_out: opts.quoted_out.unwrap_or(route.modelled_out),
        unbounded,
    })
}

// ------------------------------------------------- the string forms
//
// Every 256-bit field crosses a binding as a decimal string, and parsing it
// lives here rather than in each binding: one spelling of the refusal, and no
// binding has to name `U256`.

fn amount(text: &str) -> Result<U256> {
    text.parse()
        .map_err(|_| EncodingError(format!("not a u256: {text}")))
}

fn amounts(texts: &[String]) -> Result<Vec<U256>> {
    texts.iter().map(|t| amount(t)).collect()
}

/// Everything `encode_route` takes, as the bindings carry it.
pub struct EncodeStr<'a> {
    pub receiver: &'a str,
    pub set_approvals: bool,
    pub min_out: &'a str,
    pub amount_in: Option<&'a str>,
    pub quoted_out: Option<&'a str>,
    pub naming: &'a str,
    pub allow_unbounded: bool,
    pub volatile: &'a [String],
    pub fee_share: f64,
    pub floor_bp: f64,
    pub volatile_floor_bp: f64,
    pub slippage_bp: Option<f64>,
}

impl<'a> EncodeStr<'a> {
    pub fn new(receiver: &'a str) -> Self {
        Self {
            receiver,
            set_approvals: true,
            min_out: "0",
            amount_in: None,
            quoted_out: None,
            naming: "needed",
            allow_unbounded: false,
            volatile: &[],
            fee_share: FEE_SHARE,
            floor_bp: FLOOR_BP,
            volatile_floor_bp: VOLATILE_FLOOR_BP,
            slippage_bp: None,
        }
    }
}

pub fn encode_route_str(route: &RealizedRoute, opts: &EncodeStr<'_>) -> Result<RouteCall> {
    let naming = Naming::parse(opts.naming).ok_or_else(|| {
        EncodingError(format!(
            "naming must be one of needed/none/all, got {:?}",
            opts.naming
        ))
    })?;
    let encode = Encode {
        receiver: opts.receiver,
        set_approvals: opts.set_approvals,
        min_out: amount(opts.min_out)?,
        amount_in: opts.amount_in.map(amount).transpose()?,
        quoted_out: opts.quoted_out.map(amount).transpose()?,
        naming,
        allow_unbounded: opts.allow_unbounded,
        policy: Policy {
            volatile: opts.volatile,
            fee_share: opts.fee_share,
            floor_bp: opts.floor_bp,
            volatile_floor_bp: opts.volatile_floor_bp,
            slippage_bp: opts.slippage_bp,
        },
    };
    encode_route(route, &encode)
}

pub fn fractions_str(route: &RealizedRoute) -> Result<Vec<String>> {
    Ok(fractions(route)?.iter().map(|v| v.to_string()).collect())
}

pub fn min_rates_str(
    route: &RealizedRoute, policy: &Policy<'_>,
) -> Result<(Vec<String>, Vec<usize>)> {
    let (rates, unbounded) = min_rates(route, policy)?;
    Ok((rates.iter().map(|v| v.to_string()).collect(), unbounded))
}

pub fn walk_bounds_str(
    route: &RealizedRoute, fracs: &[String], rates: &[String],
) -> Result<(String, Vec<String>)> {
    let (promised, floors) = walk_bounds(route, &amounts(fracs)?, &amounts(rates)?);
    Ok((promised.to_string(), floors.iter().map(|v| v.to_string()).collect()))
}

/// `(kind, i, j, n, frac, min_rate, in_ref, out_ref)` packed into one word.
#[allow(clippy::too_many_arguments)]
pub fn pack_step_str(
    kind: u8, i: i32, j: i32, n: i32, frac: &str, min_rate: &str, in_ref: usize,
    out_ref: usize,
) -> Result<String> {
    let kind = ArcKind::from_code(kind)
        .ok_or_else(|| EncodingError(format!("no such kind: {kind}")))?;
    let step = Step {
        pool: String::new(), kind, i, j, n,
        frac: amount(frac)?, min_rate: amount(min_rate)?, in_ref, out_ref,
    };
    Ok(step.pack()?.to_string())
}

/// The inverse, refusing a word with reserved bits set.
#[allow(clippy::type_complexity)]
pub fn unpack_step_str(
    word: &str,
) -> Result<(u8, i32, i32, i32, String, String, usize, usize)> {
    let step = unpack(amount(word)?, "")?;
    Ok((step.kind.code(), step.i, step.j, step.n, step.frac.to_string(),
        step.min_rate.to_string(), step.in_ref, step.out_ref))
}

impl RouteCall {
    /// `(pool, kind, i, j, n, frac, min_rate, in_ref, out_ref)` per leg.
    #[allow(clippy::type_complexity)]
    pub fn steps_str(
        &self,
    ) -> Result<Vec<(String, u8, i32, i32, i32, String, String, usize, usize)>> {
        Ok(self
            .steps()?
            .iter()
            .map(|s| {
                (s.pool.clone(), s.kind.code(), s.i, s.j, s.n,
                 s.frac.to_string(), s.min_rate.to_string(), s.in_ref, s.out_ref)
            })
            .collect())
    }

    pub fn amount_in_str(&self) -> String {
        self.amount_in.to_string()
    }

    pub fn params_str(&self) -> Vec<String> {
        self.params.iter().map(|p| p.to_string()).collect()
    }

    pub fn min_out_str(&self) -> String {
        self.min_out.to_string()
    }

    pub fn guaranteed_out_str(&self) -> String {
        self.guaranteed_out.to_string()
    }

    pub fn quoted_out_str(&self) -> String {
        self.quoted_out.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step() -> Step {
        Step {
            pool: "0xpool".into(),
            kind: ArcKind::SwapStable,
            i: 0,
            j: 1,
            n: 2,
            frac: one(),
            min_rate: U256::from(12345u32),
            in_ref: 0,
            out_ref: 3,
        }
    }

    #[test]
    fn a_step_round_trips_through_its_word() {
        let packed = step().pack().unwrap();
        assert_eq!(unpack(packed, "0xpool").unwrap(), step());
    }

    #[test]
    fn the_reserved_bits_are_refused() {
        let packed = step().pack().unwrap() | (U256::from(1u8) << RESERVED_SHIFT);
        assert!(unpack(packed, "0xpool").is_err());
    }

    #[test]
    fn a_fraction_outside_the_unit_interval_is_refused() {
        let mut bad = step();
        bad.frac = U256::ZERO;
        assert!(bad.pack().is_err());
        bad.frac = one() + U256::from(1u8);
        assert!(bad.pack().is_err());
        bad.frac = one();
        assert!(bad.pack().is_ok());
    }

    #[test]
    fn a_rate_past_its_field_is_refused() {
        let mut bad = step();
        bad.min_rate = max_rate() + U256::from(1u8);
        assert!(bad.pack().is_err());
        bad.min_rate = max_rate();
        assert!(bad.pack().is_ok());
    }

    #[test]
    fn a_token_reference_past_the_table_is_refused() {
        let mut bad = step();
        bad.out_ref = MAX_TOKENS + 1;
        assert!(bad.pack().is_err());
    }

    #[test]
    fn the_shortest_entry_point_that_still_says_it() {
        // No min_out, no receiver, no tokens: the four-argument form.
        let pool = format!("0x{}", "aa".repeat(20));
        let token = format!("0x{}", "bb".repeat(20));
        let someone = format!("0x{}", "cc".repeat(20));
        let mut call = RouteCall {
            amount_in: U256::from(1u8),
            pools: vec![pool],
            params: vec![step().pack().unwrap()],
            set_approvals: true,
            ..Default::default()
        };
        assert_eq!(call.calldata("").unwrap()[..4], crate::codec::selector(SIGNATURES[0]));
        call.tokens = vec![token];
        assert_eq!(call.calldata("").unwrap()[..4], crate::codec::selector(SIGNATURES[1]));
        call.receiver = someone.clone();
        assert_eq!(call.calldata("").unwrap()[..4], crate::codec::selector(SIGNATURES[2]));
        // A call that pays its own sender does not need to say so.
        assert_eq!(
            call.calldata(&someone.to_uppercase().replace("0X", "0x")).unwrap()[..4],
            crate::codec::selector(SIGNATURES[1])
        );
        call.min_out = U256::from(5u8);
        assert_eq!(call.calldata("").unwrap()[..4], crate::codec::selector(SIGNATURES[3]));
    }

    #[test]
    fn min_out_without_a_receiver_is_refused() {
        let call = RouteCall {
            amount_in: U256::from(1u8),
            min_out: U256::from(1u8),
            ..Default::default()
        };
        assert!(call.calldata("").is_err());
    }

    #[test]
    fn a_swap_names_no_tokens_and_a_vault_names_one() {
        // `i` and `j` already say which coin, so a swap needs no table.
        assert!(!must_name(Derive::Coin, "0xa", "0xp", Naming::Needed));
        // A vault's own share token is the target, so it need not be named.
        assert!(!must_name(Derive::Target, "0xp", "0xP", Naming::Needed));
        // Its asset cannot be read without a getter, so it is.
        assert!(must_name(Derive::Getter, "0xa", "0xp", Naming::Needed));
        // ...unless the caller says to trust every getter.
        assert!(!must_name(Derive::Getter, "0xa", "0xp", Naming::None));
        // `All` names everything, including what `i` and `j` would have said.
        assert!(must_name(Derive::Coin, "0xa", "0xp", Naming::All));
    }

    #[test]
    fn rounding_is_ties_to_even() {
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
    }
}
