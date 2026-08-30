//! The values the router passes around (mirror of `core/types.py`).
//!
//! These are the first ported things that are not numbers: an arc has an
//! address, a kind and a pair of coin indices, and every stage above the
//! solver is addressed by them. Porting them here is what lets `realize`,
//! `verify` and `candidates` follow.
//!
//! `Probe` and `Leg` validate in their constructors for the reason the
//! reference does: a swap with `i == j` reverts on every real pool, and left
//! to itself the failure is invisible -- the quoter returns 0, the route is
//! dropped as "unroutable", and nothing says why.

use std::fmt;

/// Which call a leg makes. The discriminants are the reference's, and they
/// reach the on-chain router as calldata, so they are not free to renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ArcKind {
    SwapStable = 0,          // get_dy(int128,int128,uint256)
    SwapCrypto = 1,          // get_dy(uint256,uint256,uint256)
    DepositFixed = 2,        // calc_token_amount(uint256[N],bool)
    DepositDyn = 3,          // calc_token_amount(uint256[],bool)
    DepositFixedNoflag = 4,  // calc_token_amount(uint256[N])
    WithdrawStable = 5,      // calc_withdraw_one_coin(uint256,int128)
    WithdrawCrypto = 6,      // calc_withdraw_one_coin(uint256,uint256)
    Erc4626Deposit = 7,      // previewDeposit(uint256)
    Erc4626Redeem = 8,       // previewRedeem(uint256)
    WrapNative = 9,          // 1:1, no call
    UnwrapNative = 10,       // 1:1, no call
    WstethUnwrap = 11,       // getStETHByWstETH(uint256)
    WstethWrap = 12,         // getWstETHByStETH(uint256)
    StakeNative = 13,        // native -> LST at 1:1 (Lido submit, frxETHMinter)
    // 14 is deliberately absent. It was `SWAP_UNDERLYING` on an abandoned
    // branch, and `data/facts` still records that survey under it.
    LendMint = 15,           // underlying -> cToken, at `exchangeRateStored`
    LendRedeem = 16,         // cToken -> underlying, at `exchangeRateStored`
}

impl ArcKind {
    pub fn from_code(code: u8) -> Option<Self> {
        use ArcKind::*;
        Some(match code {
            0 => SwapStable,
            1 => SwapCrypto,
            2 => DepositFixed,
            3 => DepositDyn,
            4 => DepositFixedNoflag,
            5 => WithdrawStable,
            6 => WithdrawCrypto,
            7 => Erc4626Deposit,
            8 => Erc4626Redeem,
            9 => WrapNative,
            10 => UnwrapNative,
            11 => WstethUnwrap,
            12 => WstethWrap,
            13 => StakeNative,
            15 => LendMint,
            16 => LendRedeem,
            _ => return None,
        })
    }

    pub fn code(self) -> u8 {
        self as u8
    }

    /// The reference's `.name`, which is what error messages and the wire
    /// format both use.
    pub fn name(self) -> &'static str {
        use ArcKind::*;
        match self {
            SwapStable => "SWAP_STABLE",
            SwapCrypto => "SWAP_CRYPTO",
            DepositFixed => "DEPOSIT_FIXED",
            DepositDyn => "DEPOSIT_DYN",
            DepositFixedNoflag => "DEPOSIT_FIXED_NOFLAG",
            WithdrawStable => "WITHDRAW_STABLE",
            WithdrawCrypto => "WITHDRAW_CRYPTO",
            Erc4626Deposit => "ERC4626_DEPOSIT",
            Erc4626Redeem => "ERC4626_REDEEM",
            WrapNative => "WRAP_NATIVE",
            UnwrapNative => "UNWRAP_NATIVE",
            WstethUnwrap => "WSTETH_UNWRAP",
            WstethWrap => "WSTETH_WRAP",
            StakeNative => "STAKE_NATIVE",
            LendMint => "LEND_MINT",
            LendRedeem => "LEND_REDEEM",
        }
    }

    /// A lending wrapper leg, which is not a swap and not a merge.
    ///
    /// Not a merge because the two directions differ: Compound V2 answers
    /// "mint is paused" and redeems fine, Aave V2 freezes reserves the same
    /// way. A node merge is symmetric and could not say that.
    pub fn is_lending(self) -> bool {
        matches!(self, ArcKind::LendMint | ArcKind::LendRedeem)
    }

    pub fn is_swap(self) -> bool {
        matches!(self, ArcKind::SwapStable | ArcKind::SwapCrypto)
    }

    pub fn is_deposit(self) -> bool {
        matches!(
            self,
            ArcKind::DepositFixed | ArcKind::DepositDyn | ArcKind::DepositFixedNoflag
        )
    }

    pub fn is_withdraw(self) -> bool {
        matches!(self, ArcKind::WithdrawStable | ArcKind::WithdrawCrypto)
    }

    /// Wrap/ERC4626 legs are linear and stateless from the router's view.
    pub fn touches_pool_state(self) -> bool {
        self.is_swap() || self.is_deposit() || self.is_withdraw()
    }
}

impl fmt::Display for ArcKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Which index type a pool's swap ABI uses. Never inferred from one probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    Stable,
    Crypto,
}

impl Dialect {
    pub fn as_str(self) -> &'static str {
        match self {
            Dialect::Stable => "int128",
            Dialect::Crypto => "uint256",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FlagReason {
    #[default]
    None,
    DividedDiff,
    Structural,
    Clamped,
    Both,
}

impl FlagReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FlagReason::None => "NONE",
            FlagReason::DividedDiff => "DIVIDED_DIFF",
            FlagReason::Structural => "STRUCTURAL",
            FlagReason::Clamped => "CLAMPED",
            FlagReason::Both => "BOTH",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "NONE" => FlagReason::None,
            "DIVIDED_DIFF" => FlagReason::DividedDiff,
            "STRUCTURAL" => FlagReason::Structural,
            "CLAMPED" => FlagReason::Clamped,
            "BOTH" => FlagReason::Both,
            _ => return None,
        })
    }
}

/// What the reference raises as `ValueError` from a constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError(pub String);

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn check_indices(kind: ArcKind, i: i32, j: i32) -> Result<(), TypeError> {
    if kind.is_swap() && i == j {
        return Err(TypeError(format!("{} needs i != j (got i=j={i})", kind.name())));
    }
    Ok(())
}

/// One (pool, direction, size) quote request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Probe {
    pub pool: String,
    pub kind: ArcKind,
    pub i: i32,
    pub j: i32,
    pub n: i32,
    pub dx: u128,
}

impl Probe {
    pub fn new(
        pool: String,
        kind: ArcKind,
        i: i32,
        j: i32,
        n: i32,
        dx: u128,
    ) -> Result<Self, TypeError> {
        check_indices(kind, i, j)?;
        Ok(Self { pool, kind, i, j, n, dx })
    }
}

/// One executable step of a route.
///
/// `bps` is a fraction of the *current* balance at `src_slot`, snapshotted
/// when the group of legs leaving that slot opens. `bps == 0` means "take
/// whatever is left", which is how the last leg out of a node avoids dust.
/// This is what the on-chain router will execute, not a quoting convenience.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Leg {
    pub target: String,
    pub kind: ArcKind,
    pub i: i32,
    pub j: i32,
    pub n: i32,
    pub src_slot: i32,
    pub dst_slot: i32,
    pub bps: i32,
}

impl Leg {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: String,
        kind: ArcKind,
        i: i32,
        j: i32,
        n: i32,
        src_slot: i32,
        dst_slot: i32,
        bps: i32,
    ) -> Result<Self, TypeError> {
        check_indices(kind, i, j)?;
        if src_slot == dst_slot {
            return Err(TypeError(format!(
                "leg must move between slots (got {src_slot})"
            )));
        }
        if !(0..=10_000).contains(&bps) {
            return Err(TypeError(format!("bps out of range: {bps}")));
        }
        Ok(Self { target, kind, i, j, n, src_slot, dst_slot, bps })
    }
}

/// Provenance for a calibration: every number the fit was derived from.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProbeLadder {
    pub deltas: Vec<u128>,
    /// `None` where the chain refused; the fit skips those, it does not zero
    /// them.
    pub quotes: Vec<Option<u128>>,
    pub reserve_in: u128,
    pub decimals_in: u32,
    pub decimals_out: u32,
    pub block: u64,
}

/// One direction of one pool interaction (spec §15).
///
/// Always per-direction, never per-pool: `B` differs between directions by
/// three powers of the price even on a plain constant-product pool, and on
/// dynamic-fee pools exactly one of the pair carries CONVEX_FLAG.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolArc {
    pub id: String,
    pub pool: String,
    pub kind: ArcKind,
    pub i: i32,
    pub j: i32,
    pub n_coins: i32,
    pub token_in: String,
    pub token_out: String,
    /// canonical node index, post-merge
    pub tau: usize,
    pub sigma: usize,

    // calibration, in canonical (post node-merge) units
    pub a: f64,
    pub b: f64,
    pub cap: f64,
    pub calib_delta: f64,

    // §2.3 / §12.2 diagnostics
    pub convex_flag: bool,
    pub clamped: bool,
    pub flag_reason: FlagReason,
    pub drift: f64,
    pub eta: f64,
    pub asym: f64,
    pub gamma_live: f64,

    // derived (§3.1 M3/M4), filled by graph::build
    pub g: f64,
    pub eps: f64,

    // node-merge rescaling, so realize() can invert it exactly
    pub rate_in: f64,
    pub rate_out: f64,

    pub reserve_in: u128,
    pub decimals_in: u32,
    pub decimals_out: u32,
    pub tvl_usd: f64,
    pub reverse_id: Option<String>,
    pub ladder: Option<ProbeLadder>,
    pub note: String,
}

impl PoolArc {
    /// The ten fields the reference has no default for; everything else takes
    /// the dataclass default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        pool: String,
        kind: ArcKind,
        i: i32,
        j: i32,
        n_coins: i32,
        token_in: String,
        token_out: String,
        tau: usize,
        sigma: usize,
    ) -> Self {
        Self {
            id,
            pool,
            kind,
            i,
            j,
            n_coins,
            token_in,
            token_out,
            tau,
            sigma,
            a: 0.0,
            b: 0.0,
            cap: f64::INFINITY,
            calib_delta: 0.0,
            convex_flag: false,
            clamped: false,
            flag_reason: FlagReason::None,
            drift: 0.0,
            eta: f64::NAN,
            asym: f64::NAN,
            gamma_live: f64::NAN,
            g: 0.0,
            eps: 0.0,
            rate_in: 1.0,
            rate_out: 1.0,
            reserve_in: 0,
            decimals_in: 18,
            decimals_out: 18,
            tvl_usd: 0.0,
            reverse_id: None,
            ladder: None,
            note: String::new(),
        }
    }

    pub fn resistance(&self) -> f64 {
        if self.g <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.g
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_swap_onto_itself_is_refused() {
        let err = Probe::new("0xpool".into(), ArcKind::SwapStable, 1, 1, 2, 10).unwrap_err();
        assert_eq!(err.0, "SWAP_STABLE needs i != j (got i=j=1)");
        // A withdrawal legitimately names one coin twice.
        assert!(Probe::new("0xpool".into(), ArcKind::WithdrawStable, 1, 1, 2, 10).is_ok());
    }

    #[test]
    fn a_leg_must_move_between_slots() {
        let err = Leg::new("0xt".into(), ArcKind::SwapStable, 0, 1, 2, 3, 3, 0).unwrap_err();
        assert_eq!(err.0, "leg must move between slots (got 3)");
        let err = Leg::new("0xt".into(), ArcKind::SwapStable, 0, 1, 2, 0, 1, 10_001).unwrap_err();
        assert_eq!(err.0, "bps out of range: 10001");
    }

    #[test]
    fn fourteen_is_not_a_kind() {
        assert_eq!(ArcKind::from_code(14), None);
        assert_eq!(ArcKind::from_code(15), Some(ArcKind::LendMint));
        assert_eq!(ArcKind::LendRedeem.code(), 16);
    }

    #[test]
    fn kind_predicates_match_the_reference() {
        assert!(ArcKind::SwapCrypto.touches_pool_state());
        assert!(ArcKind::WithdrawCrypto.touches_pool_state());
        assert!(!ArcKind::WrapNative.touches_pool_state());
        assert!(!ArcKind::Erc4626Redeem.touches_pool_state());
        assert!(ArcKind::LendMint.is_lending());
    }
}
