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
    # Served by `api2.curve.finance` rather than the Prices API -- a factory
    # deployment with no trade indexing.  See `dev/lite.py`; the only chain in
    # both lists is sonic, where Prices wins because it has the whole
    # deployment and Lite reports a fraction of it.
    lite: bool = False
    quoter: str = ""
    # Pools that quote fine and cannot be traded, so no probe should be spent
    # on them and no route should reach them.  Not the same as the reserve
    # check, which catches accounting that has come adrift from the tokens
    # actually held: these hold what they claim, and the failure is one layer
    # down, in a protocol that will no longer accept a deposit.  Only executing
    # finds them, so they are recorded once rather than rediscovered.
    blacklist: tuple[str, ...] = ()
    # Pools worth re-testing on every `facts` build, whether or not they are
    # currently routable: deprecated lending protocols whose `exchange` quotes
    # but whose `exchange_underlying` cannot deposit.  Kept explicit because
    # there are few of them and they change on a protocol's schedule, not a
    # block's -- and because a blacklisted pool builds no arcs, so nothing else
    # would ever look at it again.  Re-testing is what lets one come back.
    watch: tuple[str, ...] = ()
    # (wrapper, underlying, family) -- lending tokens whose mint and redeem are
    # tested on every `facts` build.  What survives becomes an arc; see
    # `wrappers.build_lending_arcs`.
    wrappers: tuple[tuple[str, str, str], ...] = ()
    # Tokens that hold a peg.  Used only to decide how much a routing gain has
    # to be worth before another leg is taken: a pair of these moves 0.17 bp
    # over a thousand blocks, where ETH moves 125, so a gain worth chasing on
    # one is noise on the other.  Membership is not measurable by execution the
    # way redemption is -- it is a claim about what a token is for.
    stables: tuple[str, ...] = ()
    # A committed endpoint, for a checkout with no `networks.py`.
    #
    # Deliberately public, and safe to be: the key serves reads only --
    # `eth_getStorageAt`, `eth_getCode`, `eth_getBalance`, `eth_getBlockByNumber`,
    # `eth_blockNumber`, `eth_createAccessList`, `eth_gasPrice` -- plus
    # `eth_call` restricted to the quoter address above, which is a stateless
    # view contract that can move nothing.  `eth_simulateV1` is off: it is not
    # address-scoped, and with state overrides it can fabricate balances and
    # execute against them, which is strictly more than `eth_call` allows.
    #
    # Everything the router needs runs on this: the local EVM holds the pools'
    # code and storage and computes `get_dy` in-process, so quoting needs no
    # execution rights at all.
    public_rpc: str = ""
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
        quoter="0x977ACB8f30412278B33fA7457dcd667613f6CB93",
        public_rpc=(
            "https://lb.drpc.live/ethereum/"
            "AskGI4lH8UlFtIRsb5UfRvXOC_8-l9AR8YojRoYgFhqK"
        ),
        stables=(
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",   # DAI
            CRVUSD,
            "0x853d955aCEf822Db058eb8505911ED77F175b99e",   # FRAX
            "0x4c9EDD5852cd905f086C759E8383e09bff1E68B3",   # USDe
            "0xdC035D45d973E3EC169d2276DDab16f1e407384F",   # USDS
            "0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f",   # GHO
        ),
        wrappers=(
            ("0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643",   # cDAI
             "0x6B175474E89094C44Da98b954EedeAC495271d0F", "ctoken"),
            ("0x39AA39c021dfbaE8faC545936693aC917d5E7563",   # cUSDC
             "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "ctoken"),
        ),
        watch=(
            "0xDeBF20617708857ebe4F679508E7b7863a8A8EeE",   # aDAI/aUSDC/aUSDT
            "0xA2B47E3D5c44877cca798226B7B8118F9BFb7A56",   # cDAI/cUSDC
            "0x52EA46506B9CC5Ef470C5bf89f17Dc28bB35D85C",   # cDAI/cUSDC/USDT
        ),
        blacklist=(
            # Curve.fi aDAI/aUSDC/aUSDT.  Aave V2's reserves are frozen, so the
            # deposit inside `exchange_underlying` reverts while
            # `get_dy_underlying` -- which only runs the invariant over the
            # pool's own balances and never touches Aave -- answers happily.
            # Verified on a fork: quotes 1,875.46 USDC for 1,875 USDT, reverts
            # on execution.  Curve's own solver excludes it for the same reason
            # (curve_solver 0511e53, "Exclude frozen Aave V2 pool").
            "0xDeBF20617708857ebe4F679508E7b7863a8A8EeE",
        ),
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
    "bsc": Chain(
        name="bsc",
        chain_id=56,
        api_name="bsc",
        rpc_attr="BSC",
        native_symbol="BNB",
        wrapped="0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
    ),
    # --- Curve Lite deployments ------------------------------------------
    #
    # Small enough that the mainnet $10,000 pool floor would empty them, so
    # `lite.LITE_MIN_TVL` is what the loader uses for these instead.
    #
    # Every wrapped native below was derived from the chain's own pool data --
    # the coin whose symbol is W + the native symbol the API reports -- and
    # then confirmed by asking the token for its own `symbol()`.  Guessing
    # these fails silently: a wrong address breaks the native/ERC20 merge and
    # the router simply never finds routes through it.
    #
    # Two Lite deployments are deliberately absent.  **fantom** is in
    # wind-down: 164 of its 321 pools answer neither ABI spelling and probing
    # them takes 19 minutes.  **kava** resolves a dialect for none of its 19
    # pools, so it yields zero arcs -- there is nothing there to route.
    "avalanche": Chain(
        name="avalanche",
        chain_id=43114,
        api_name="avalanche",
        rpc_attr="AVALANCHE",
        native_symbol="AVAX",
        wrapped="0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
        lite=True,
    ),
    "etherlink": Chain(
        name="etherlink",
        chain_id=42793,
        api_name="etherlink",
        rpc_attr="ETHERLINK",
        native_symbol="XTZ",
        wrapped="0xc9B53AB2679f573e480d01e0f49e2B5CFB7a3EAb",
        lite=True,   # ~$9.1M
    ),
    "monad": Chain(
        name="monad",
        chain_id=143,
        api_name="monad",
        rpc_attr="MONAD",
        native_symbol="MON",
        wrapped="0x3bD359c1119Da7dA1d913d1C4d2B7c461115433A",
        lite=True,   # ~$7.5M
    ),
    "plasma": Chain(
        name="plasma",
        chain_id=9745,
        api_name="plasma",
        rpc_attr="PLASMA",
        native_symbol="XPL",
        wrapped="0x6100E367285b01F48D07953803A2d8dCA5D19873",
        lite=True,   # ~$3.8M
    ),
    "xlayer": Chain(
        name="xlayer",
        chain_id=196,
        api_name="xlayer",
        rpc_attr="XLAYER",
        native_symbol="OKB",
        wrapped="0xe538905cf8410324e03A5A23C1c177a474D59b2b",
        lite=True,   # ~$3.6M
    ),
    "unichain": Chain(
        name="unichain",
        chain_id=130,
        api_name="unichain",
        rpc_attr="UNICHAIN",
        native_symbol="ETH",
        wrapped="0x4200000000000000000000000000000000000006",
        lite=True,   # ~$572k
    ),
    "robinhood": Chain(
        name="robinhood",
        chain_id=4663,
        api_name="robinhood",
        rpc_attr="ROBINHOOD",
        native_symbol="ETH",
        wrapped="0x0bD7d308f8e1639fAb988df18a8011f41EacaD73",
        lite=True,   # ~$414k
    ),
    "celo": Chain(
        name="celo",
        chain_id=42220,
        api_name="celo",
        rpc_attr="CELO",
        native_symbol="CELO",
        # CELO is natively an ERC20 at this address -- there is no
        # separate wrapped token, so the merge is the identity.
        wrapped="0x471EcE3750Da237f93B8E339c536989b8978a438",
        lite=True,   # ~$259k
    ),
    "tac": Chain(
        name="tac",
        chain_id=239,
        api_name="tac",
        rpc_attr="TAC",
        native_symbol="TAC",
        wrapped="0xB63B9f0eb4A6E6f191529D71d4D88cc8900Df2C9",
        lite=True,   # ~$171k
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
