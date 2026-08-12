"""Per-chain constants.  Committed -- no secrets here, only addresses.

`api_name` is what the Curve Prices v2 API calls the chain.  Note Gnosis is
served as **"xdai"**, not "gnosis"; `/v2/pools/chains/` returns exactly:
ethereum 1, optimism 10, bsc 56, xdai 100, polygon 137, sonic 146,
fraxtal 252, arbitrum 42161, base 8453, hyperliquid 999.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Some Curve pools hold native ETH directly and use this placeholder as the coin
# address -- the $77M ETH/stETH pool is one.  Merging it with the wrapped token
# is what connects those pools to the rest of the graph.
NATIVE_SENTINEL = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"


@dataclass(frozen=True, slots=True)
class Chain:
    name: str
    chain_id: int
    api_name: str
    rpc_attr: str  # attribute name in networks.py
    native_symbol: str
    wrapped: str  # wrapped-native ERC20
    # ERC4626 vaults that may be MERGED with their asset into one graph node.
    # Allowlist, never auto-discovery: linearity of convertToAssets is necessary
    # and nowhere near sufficient.  pufETH is perfectly linear and redeems
    # through a withdrawal queue; sUSDe has a 7-day cooldown; sfrxUSD has
    # maxDeposit == 0.  Merging any of those mints value from nothing.
    # Every entry must pass an executed deposit/redeem on a fork (see tests).
    # A deployed RouteQuoter, if there is one.  With it, quoting is a plain
    # `eth_call` to a known address; without it the runtime bytecode rides
    # along as a state override on every call (7,486 bytes, ~14% of a cold
    # route's upload).  A deployed quoter also takes boa out of the request
    # path, which is what the browser build needs.
    quoter: str = ""
    erc4626_allowlist: tuple[str, ...] = ()
    # (token, canonical) pairs handled by ConversionKind.WSTETH.  Kept off the
    # ERC4626 list because wstETH predates that standard and exposes its own
    # getters, not convertToAssets.
    wsteth_pairs: tuple[tuple[str, str], ...] = ()
    # One-way instant conversions: `(token_in, token_out, kind, target,
    # cap_selector)`.  Never merges -- minting is instant but redemption is a
    # queue or a cooldown, and a merge would let the router unstake for free.
    stake_arcs: tuple[tuple[str, str, str, str, str], ...] = ()
    # Vaults whose *deposit* is instant while redemption is gated.  Modelled as
    # a one-way ERC4626_DEPOSIT arc rather than dropped: sUSDe's 7-day cooldown
    # and pufETH's withdrawal queue make them unmergeable, not unroutable.
    oneway_vaults: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


SCRVUSD = "0x0655977FEb2f289A4aB78af67BAB0d17aAb84367"
WSTETH = "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"
STETH = "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84"
SUSDE = "0x9D39A5DE30e57443BfF2A8307A4256c8797A3497"
# Merged only after an executed deposit->redeem round trip on a fork returned
# to the wei (tests/forked/test_vault_roundtrip.py).  Static checks are not
# enough: sUSDe and pufETH pass linearity and maxDeposit and are traps.
SDOLA = "0xb45ad160634c528Cc3D2926d9807104FA3157305"
SDAI = "0x83F20F44975D03b1b09e64809B757c47f942BEeA"
SUSDS = "0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD"
SFRXETH = "0xac3E018457B222d93114458476f3E3416Abbe38F"
SGHO = "0xE1753F2E00940cC31213dD92013CF019dfe4ca1D"
STUSDS = "0x99cD4EC3F88a45940936f469E4Bb72a2a701EeB9"
# Round-trip clean but only $3.0M and $1.1M deep.  A merge asserts unbounded
# depth in both directions, which those cannot honour at routing sizes, so
# they stay out until there is a capped bidirectional element to hold them.
# cvcrvUSD = 0xcea18a8752bb7e7817f9ae7565328fe415c0f2ca
# sUSG     = 0xf17d6f98a5c6eaa99d149079984119e0a4ef6900
PUFETH = "0xD9A442856C234a39a81a089C06451EBAa4306a72"
CRVUSD = "0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E"

CHAINS: dict[str, Chain] = {
    "ethereum": Chain(
        name="ethereum",
        chain_id=1,
        api_name="ethereum",
        rpc_attr="NETWORK",
        native_symbol="ETH",
        wrapped="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        erc4626_allowlist=(SCRVUSD, SDOLA, SDAI, SUSDS, SFRXETH, SGHO, STUSDS),
        wsteth_pairs=((WSTETH, STETH),),
        stake_arcs=(
            # Lido: 1 ETH -> 1 stETH, capped by the daily staking limit.
            (NATIVE_SENTINEL, STETH, "STAKE_NATIVE", STETH, "getCurrentStakeLimit()"),
        ),
        oneway_vaults=(SUSDE, PUFETH),
    ),
    "arbitrum": Chain(
        name="arbitrum",
        chain_id=42161,
        api_name="arbitrum",
        rpc_attr="ARBITRUM",
        native_symbol="ETH",
        wrapped="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    ),
    "optimism": Chain(
        name="optimism",
        chain_id=10,
        api_name="optimism",
        rpc_attr="OPTIMISM",
        native_symbol="ETH",
        wrapped="0x4200000000000000000000000000000000000006",
    ),
    "base": Chain(
        name="base",
        chain_id=8453,
        api_name="base",
        rpc_attr="BASE",
        native_symbol="ETH",
        wrapped="0x4200000000000000000000000000000000000006",
    ),
    "gnosis": Chain(
        name="gnosis",
        chain_id=100,
        api_name="xdai",  # NB: the API does not call this "gnosis"
        rpc_attr="GNOSIS",
        native_symbol="XDAI",
        wrapped="0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d",  # WXDAI
    ),
    "polygon": Chain(
        name="polygon",
        chain_id=137,
        api_name="polygon",
        rpc_attr="POLYGON",
        native_symbol="POL",
        wrapped="0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WPOL (ex-WMATIC)
    ),
    "fraxtal": Chain(
        name="fraxtal",
        chain_id=252,
        api_name="fraxtal",
        rpc_attr="FRAXTAL",
        native_symbol="frxETH",
        wrapped="0xFC00000000000000000000000000000000000006",  # wfrxETH
    ),
    "sonic": Chain(
        name="sonic",
        chain_id=146,
        api_name="sonic",
        rpc_attr="SONIC",
        native_symbol="S",
        wrapped="0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38",  # wS
    ),
}

BY_ID: dict[int, Chain] = {c.chain_id: c for c in CHAINS.values()}


def get(name_or_id: str | int) -> Chain:
    if isinstance(name_or_id, int):
        return BY_ID[name_or_id]
    key = str(name_or_id).lower()
    if key in CHAINS:
        return CHAINS[key]
    if key.isdigit():
        return BY_ID[int(key)]
    raise KeyError(f"unknown chain {name_or_id!r}; known: {', '.join(CHAINS)}")
