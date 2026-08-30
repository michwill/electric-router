//! Node merging: tokens that are the same thing at zero cost (spec §3.1).
//!
//! The mirror of `core/nodes.py`.
//!
//! A wrapped native (WETH/ETH) and a linear ERC4626 vault (scrvUSD/crvUSD) are
//! *zero-resistance elements*. In value coordinates a linear arc has `eps = 0`
//! and `G = inf` -- a short circuit -- so the two tokens are literally the same
//! node.
//!
//! Merging rather than adding an arc is not a nicety:
//!
//! * it avoids a degenerate `B = 0`, `eps_f + eps_r = 0` arc pair, which would
//!   violate §2.6's `eps_f + eps_r > 0` guard and §12.4's `clamped => cap < inf`;
//! * it connects halves of the graph that are otherwise nearly disjoint --
//!   eight Ethereum pools hold native ETH under the `0xEeee...` sentinel
//!   (including the $77M ETH/stETH pool) while 63 hold WETH.
//!
//! The conversion itself is materialised only when a route is emitted, as a leg.
//!
//! **Merging is allowlist-gated, never inferred.** Linearity of
//! `convertToAssets` is necessary and nowhere near sufficient: of 31 linear
//! ERC4626 tokens on Ethereum, `pufETH` reports `asset = WETH` with zero
//! linearity error and redeems through a *withdrawal queue*, `sUSDe` has a
//! 7-day cooldown, and `sfrxUSD` has `maxDeposit == 0`. Merging any of those
//! would declare the vault equal to its asset at NAV and mint the market
//! discount out of thin air.

use crate::types::ArcKind;
use ruint::aliases::{U256, U512};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionKind {
    /// 1:1, ERC20 <-> native
    NativeWrap,
    /// shares <-> assets at the vault's rate
    Erc4626,
    /// Lido's wstETH: a native-wrapper shape but rate-bearing, and it predates
    /// ERC4626 so it exposes its own getters. A second token of this shape is
    /// the signal to generalise, as curve_solver's AmountCall* does.
    Wsteth,
    /// Two addresses over *one* balance, not two assets that convert. Gnosis
    /// EURe is the case: v1 and v2 report identical `totalSupply` and
    /// identical `balanceOf` for every holder, to the wei -- holding one *is*
    /// holding the other, and modelled as separate nodes they split one market
    /// in half.
    ///
    /// It merges like any other conversion and realises like none of them: the
    /// rate is exactly 1 and no leg is emitted, because there is nothing to
    /// call.
    Alias,
}

impl ConversionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ConversionKind::NativeWrap => "NATIVE_WRAP",
            ConversionKind::Erc4626 => "ERC4626",
            ConversionKind::Wsteth => "WSTETH",
            ConversionKind::Alias => "ALIAS",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "NATIVE_WRAP" => ConversionKind::NativeWrap,
            "ERC4626" => ConversionKind::Erc4626,
            "WSTETH" => ConversionKind::Wsteth,
            "ALIAS" => ConversionKind::Alias,
            _ => return None,
        })
    }
}

/// How to get from `token` to the canonical token of its node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    pub kind: ConversionKind,
    /// the non-canonical side
    pub token: String,
    /// the node's canonical token
    pub canonical: String,
    /// Exact integer rate: canonical wei per 10**decimals of `token`.
    pub rate_num: U256,
    pub rate_den: U256,
    /// contract to call (WETH, or the vault)
    pub target: String,
}

impl Conversion {
    pub fn new(kind: ConversionKind, token: &str, canonical: &str) -> Self {
        Self {
            kind,
            token: token.to_string(),
            canonical: canonical.to_string(),
            rate_num: U256::from(1u8),
            rate_den: U256::from(1u8),
            target: String::new(),
        }
    }

    pub fn with_rate(mut self, num: U256, den: U256) -> Self {
        self.rate_num = num;
        self.rate_den = den;
        self
    }

    /// The rate as the bindings carry it: two decimal strings.
    ///
    /// Parsing lives here rather than in each binding for the reason the pool
    /// models do -- one spelling of the refusal, and no binding has to name
    /// `U256` to hold a rate that does not fit an `f64`.
    pub fn with_rate_str(self, num: &str, den: &str) -> Result<Self, NodeError> {
        let den = parse_u256(den, "rate_den")?;
        if den.is_zero() {
            // The reference would raise `ZeroDivisionError` on the first
            // conversion instead, which is a long way from where the bad rate
            // was set.
            return Err(NodeError("rate_den must not be zero".to_string()));
        }
        Ok(self.with_rate(parse_u256(num, "rate_num")?, den))
    }

    pub fn with_target(mut self, target: &str) -> Self {
        self.target = target.to_string();
        self
    }

    /// Canonical units per unit of `token`, in human terms.
    pub fn rate(&self) -> f64 {
        // Both sides through the same correctly-rounded conversion the pool
        // models use, so a 256-bit rate does not lose its low bits on the way
        // to an `f64` the way `as` would.
        crate::pools::scaled(self.rate_num, 0) / crate::pools::scaled(self.rate_den, 0)
    }

    pub fn to_canonical(&self, amount: U256) -> Option<U256> {
        mul_div(amount, self.rate_num, self.rate_den)
    }

    pub fn from_canonical(&self, amount: U256) -> Option<U256> {
        mul_div(amount, self.rate_den, self.rate_num)
    }

    /// Nothing to execute: the two addresses share a balance.
    pub fn is_alias(&self) -> bool {
        self.kind == ConversionKind::Alias
    }

    /// token -> canonical.
    pub fn forward_kind(&self) -> ArcKind {
        match self.kind {
            // The canonical side of a native pair is the wrapped ERC20, so
            // going from native to canonical is a wrap.
            ConversionKind::NativeWrap => ArcKind::WrapNative,
            ConversionKind::Wsteth => ArcKind::WstethUnwrap, // wstETH -> stETH
            _ => ArcKind::Erc4626Redeem,                     // shares -> assets
        }
    }

    /// canonical -> token.
    pub fn reverse_kind(&self) -> ArcKind {
        match self.kind {
            ConversionKind::NativeWrap => ArcKind::UnwrapNative,
            ConversionKind::Wsteth => ArcKind::WstethWrap, // stETH -> wstETH
            _ => ArcKind::Erc4626Deposit,                  // assets -> shares
        }
    }
}

/// A refusal the reference raises as `KeyError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeError(pub String);

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Token address -> graph node, plus how to convert between members.
#[derive(Debug, Clone, Default)]
pub struct NodeMap {
    pub node_of: HashMap<String, usize>,
    pub tokens_of: Vec<Vec<String>>,
    pub canonical_of: Vec<String>,
    pub conversion: HashMap<String, Conversion>,
    pub symbol_of: HashMap<String, String>,
    pub decimals_of: HashMap<String, u32>,
    pub rejected: Vec<(String, String)>,
}

impl NodeMap {
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------- building

    pub fn add_token(&mut self, address: &str, symbol: &str, decimals: u32) -> usize {
        let key = address.to_ascii_lowercase();
        if !self.node_of.contains_key(&key) {
            let node = self.tokens_of.len();
            self.node_of.insert(key.clone(), node);
            self.tokens_of.push(vec![key.clone()]);
            self.canonical_of.push(key.clone());
        }
        if !symbol.is_empty() {
            self.symbol_of.entry(key.clone()).or_insert_with(|| symbol.to_string());
        }
        self.decimals_of.entry(key.clone()).or_insert(decimals);
        self.node_of[&key]
    }

    /// Fold `conversion.token` into the node of `conversion.canonical`.
    pub fn merge(&mut self, conversion: Conversion) -> Result<(), NodeError> {
        let token = conversion.token.to_ascii_lowercase();
        let canonical = conversion.canonical.to_ascii_lowercase();
        let target = match self.node_of.get(&canonical) {
            Some(&v) => v,
            None => {
                return Err(NodeError(format!(
                    "canonical token {canonical} is not in the graph"
                )))
            }
        };
        if self.node_of.get(&token) == Some(&target) {
            self.conversion.insert(token, conversion);
            return Ok(());
        }

        match self.node_of.get(&token).copied() {
            Some(old) if old != target => {
                let members = std::mem::take(&mut self.tokens_of[old]);
                for member in members {
                    self.node_of.insert(member.clone(), target);
                    if !self.tokens_of[target].contains(&member) {
                        self.tokens_of[target].push(member);
                    }
                }
            }
            _ => {
                self.node_of.insert(token.clone(), target);
                self.tokens_of[target].push(token.clone());
            }
        }
        self.conversion.insert(token, conversion);
        Ok(())
    }

    // -------------------------------------------------------------- lookup

    // Every one of these tries the address as given before lowering it, and
    // the difference is not cosmetic: `node` and `has` are called 32,000 times
    // in a single route, almost always with an address that came from a
    // `PoolSpec` and is already lowercase. Lowering it again allocates a
    // string per call to look up the same key. The fallback stays for the
    // symbol resolver and the CLI, which do hand checksummed addresses in and
    // simply pay for it.

    pub fn node(&self, token: &str) -> Option<usize> {
        match self.node_of.get(token) {
            Some(&v) => Some(v),
            None => self.node_of.get(&token.to_ascii_lowercase()).copied(),
        }
    }

    pub fn has(&self, token: &str) -> bool {
        self.node_of.contains_key(token)
            || self.node_of.contains_key(&token.to_ascii_lowercase())
    }

    pub fn canonical(&self, token: &str) -> Option<&str> {
        self.node(token).map(|k| self.canonical_of[k].as_str())
    }

    fn conversion_of(&self, token: &str) -> Option<&Conversion> {
        match self.conversion.get(token) {
            Some(v) => Some(v),
            None => self.conversion.get(&token.to_ascii_lowercase()),
        }
    }

    /// Canonical units per unit of `token` (1.0 for a canonical token).
    pub fn rate(&self, token: &str) -> f64 {
        self.conversion_of(token).map_or(1.0, |c| c.rate())
    }

    pub fn to_canonical_wei(&self, token: &str, amount: U256) -> Option<U256> {
        match self.conversion_of(token) {
            Some(c) => c.to_canonical(amount),
            None => Some(amount),
        }
    }

    pub fn from_canonical_wei(&self, token: &str, amount: U256) -> Option<U256> {
        match self.conversion_of(token) {
            Some(c) => c.from_canonical(amount),
            None => Some(amount),
        }
    }

    pub fn to_canonical_wei_str(&self, token: &str, amount: &str) -> Result<String, NodeError> {
        let amount = parse_u256(amount, "amount")?;
        self.to_canonical_wei(token, amount)
            .map(|v| v.to_string())
            .ok_or_else(|| overflowed(token, amount))
    }

    pub fn from_canonical_wei_str(&self, token: &str, amount: &str) -> Result<String, NodeError> {
        let amount = parse_u256(amount, "amount")?;
        self.from_canonical_wei(token, amount)
            .map(|v| v.to_string())
            .ok_or_else(|| overflowed(token, amount))
    }

    pub fn symbol(&self, token: &str) -> String {
        if let Some(found) = self.symbol_of.get(token) {
            return found.clone();
        }
        match self.symbol_of.get(&token.to_ascii_lowercase()) {
            Some(found) => found.clone(),
            // `token[:10]` on a str, not on bytes: every address is ASCII, but
            // a symbol resolver can hand this a name that is not.
            None => token.chars().take(10).collect(),
        }
    }

    pub fn decimals(&self, token: &str) -> u32 {
        match self.decimals_of.get(token) {
            Some(&v) => v,
            None => self
                .decimals_of
                .get(&token.to_ascii_lowercase())
                .copied()
                .unwrap_or(18),
        }
    }

    /// A label for the merged node, e.g. `ETH/WETH`.
    pub fn node_symbol(&self, node: usize) -> String {
        let members: Vec<&String> = self.tokens_of[node]
            .iter()
            .filter(|t| self.symbol_of.contains_key(*t))
            .collect();
        if members.is_empty() {
            return format!("node{node}");
        }
        let canonical = &self.canonical_of[node];
        let mut ordered: Vec<&String> = vec![canonical];
        ordered.extend(members.into_iter().filter(|m| *m != canonical));
        // `dict.fromkeys` on the *symbols*, not the addresses: two members
        // sharing a symbol print once.
        let mut seen: Vec<&str> = Vec::new();
        for member in ordered {
            if let Some(symbol) = self.symbol_of.get(member) {
                if !seen.contains(&symbol.as_str()) {
                    seen.push(symbol.as_str());
                }
            }
        }
        seen.join("/")
    }

    pub fn n_nodes(&self) -> usize {
        self.tokens_of.len()
    }

    pub fn merged_nodes(&self) -> Vec<usize> {
        (0..self.tokens_of.len())
            .filter(|&k| self.tokens_of[k].len() > 1)
            .collect()
    }
}

/// `amount * num // den`, with the product taken in 512 bits.
///
/// The reference does this in Python `int`, which has no ceiling. A `U256`
/// multiply does, and 256 bits is not enough to hold `amount * rate_num` for
/// an amount anywhere near the top of the range -- so the intermediate widens
/// and only the quotient comes back. `None` where even the quotient will not
/// fit, which is not a number of wei any ERC20 could hold; refusing is the
/// mirror's one deliberate divergence, and it is loud rather than a wrap.
fn mul_div(amount: U256, num: U256, den: U256) -> Option<U256> {
    if den.is_zero() {
        return None;
    }
    let wide = |v: U256| U512::from_limbs_slice(v.as_limbs());
    let quotient = wide(amount) * wide(num) / wide(den);
    // Back through the limbs too: ruint has no `TryFrom` between two widths,
    // and the top four limbs being clear is exactly the question being asked.
    let limbs = quotient.as_limbs();
    if limbs[4..].iter().any(|&limb| limb != 0) {
        return None;
    }
    Some(U256::from_limbs_slice(&limbs[..4]))
}

fn overflowed(token: &str, amount: U256) -> NodeError {
    NodeError(format!(
        "converting {amount} of {token} does not fit in 256 bits"
    ))
}

/// A 256-bit integer as the bindings spell it: base ten, no sign, no prefix.
pub fn parse_u256(text: &str, what: &str) -> Result<U256, NodeError> {
    text.parse::<U256>()
        .map_err(|_| NodeError(format!("{what}: {text:?}")))
}

/// Re-express an arc's derivatives in canonical units.
///
///     a_canonical = a * R_out / R_in
///     B_canonical = B * R_out / R_in^2
///
/// `B` has units of output per input *squared*, so the input rate enters
/// twice. Getting this wrong is silent: the arc still solves, just with a
/// conductance off by the price ratio.
pub fn rescale(
    a: f64,
    b: f64,
    rate_in: f64,
    rate_out: f64,
) -> Result<(f64, f64), NodeError> {
    if rate_in <= 0.0 || rate_out <= 0.0 {
        return Err(NodeError(format!(
            "conversion rates must be positive ({}, {})",
            crate::pyfmt::float(rate_in),
            crate::pyfmt::float(rate_out)
        )));
    }
    Ok((a * rate_out / rate_in, b * rate_out / (rate_in * rate_in)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weth() -> &'static str {
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    }
    fn eth() -> &'static str {
        "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    }

    #[test]
    fn a_native_pair_becomes_one_node() {
        let mut map = NodeMap::new();
        map.add_token(weth(), "WETH", 18);
        map.add_token(eth(), "ETH", 18);
        assert_eq!(map.n_nodes(), 2);
        map.merge(Conversion::new(ConversionKind::NativeWrap, eth(), weth()))
            .unwrap();
        assert_eq!(map.node(eth()), map.node(weth()));
        assert_eq!(map.merged_nodes(), vec![0]);
        assert_eq!(map.node_symbol(0), "WETH/ETH");
        assert_eq!(map.rate(eth()), 1.0);
    }

    #[test]
    fn merging_an_unknown_canonical_is_refused() {
        let mut map = NodeMap::new();
        map.add_token(eth(), "ETH", 18);
        let err = map
            .merge(Conversion::new(ConversionKind::NativeWrap, eth(), weth()))
            .unwrap_err();
        assert!(err.0.contains("is not in the graph"));
    }

    #[test]
    fn a_vault_rate_is_exact_in_wei_and_approximate_in_float() {
        // 1 share = 1.05 assets, as the vault reports it.
        let c = Conversion::new(ConversionKind::Erc4626, "0xshare", "0xasset")
            .with_rate(U256::from(1_050_000_000_000_000_000u128), U256::from(10u128).pow(U256::from(18)));
        assert_eq!(c.to_canonical(U256::from(1_000_000u128)), Some(U256::from(1_050_000u128)));
        // Truncating, not rounding -- the vault's own arithmetic.
        assert_eq!(c.from_canonical(U256::from(1_050_000u128)), Some(U256::from(1_000_000u128)));
        // The product needs 512 bits long before the quotient needs 257.
        let big = U256::from(1u8) << 200;
        assert_eq!(c.to_canonical(big), Some(big * U256::from(105u8) / U256::from(100u8)));
        assert!((c.rate() - 1.05).abs() < 1e-15);
        assert_eq!(c.forward_kind(), ArcKind::Erc4626Redeem);
        assert_eq!(c.reverse_kind(), ArcKind::Erc4626Deposit);
    }

    #[test]
    fn an_alias_emits_nothing() {
        let c = Conversion::new(ConversionKind::Alias, "0xeure2", "0xeure1");
        assert!(c.is_alias());
        assert_eq!(c.rate(), 1.0);
    }

    #[test]
    fn rescaling_squares_the_input_rate() {
        let (a, b) = rescale(2.0, 3.0, 2.0, 5.0).unwrap();
        assert_eq!(a, 5.0);
        assert_eq!(b, 3.75);
        assert!(rescale(1.0, 1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn merging_two_populated_nodes_moves_every_member() {
        let mut map = NodeMap::new();
        map.add_token("0xa", "A", 18);
        map.add_token("0xb", "B", 18);
        map.add_token("0xc", "C", 18);
        map.merge(Conversion::new(ConversionKind::Alias, "0xc", "0xb")).unwrap();
        // Now fold b's whole node (b and c) into a's.
        map.merge(Conversion::new(ConversionKind::Alias, "0xb", "0xa")).unwrap();
        assert_eq!(map.node("0xc"), map.node("0xa"));
        assert_eq!(map.tokens_of[0].len(), 3);
        assert!(map.tokens_of[1].is_empty());
    }
}
