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
    #: This chain cannot be read at all until `RouteQuoter` is deployed on it,
    #: because it rejects `eth_call` state overrides.  Distinct from an empty
    #: `quoter`, which merely means quoting pays to ship the bytecode.
    needs_quoter: bool = False
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
    # (token_a, token_b, adapter) -- a contract that converts between two
    # tokens 1:1 in both directions, holding a reserve of one and minting the
    # other.  Not a merge: the redeem side is bounded by what the adapter
    # actually holds, and a merge asserts there is no bound.
    #
    # Gnosis's USDC transmuter is the case: `USDC.e -> USDC` at exactly 1:1 for
    # as long as its 10.04M USDC lasts.  Our router priced that hop through
    # pools at 99,448 per 100,000 -- 55 bp -- while curve_solver, which
    # configures the adapter, returned 100,000.000000.
    transmuters: tuple[tuple[str, str, str], ...] = ()
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

#: RouteQuoter, at the same address on every chain.
#
# Deployed through the canonical CREATE2 proxy with salt
# keccak("erouter.RouteQuoter.v1"), so the address is a function of the
# initcode alone -- see `scripts/deploy_quoter.py --create2`.  One address is
# not a tidiness argument: a scoped RPC key gates `eth_call` by target address
# and holds ten of them, against fifteen chains, so per-chain addresses cannot
# all be whitelisted at once.
#
# Changing RouteQuoter.vy moves this address.  The salt carries the version so
# that move is deliberate rather than a whitelist entry quietly describing
# something that is no longer deployed.
QUOTER = "0xd0f20bd005df7657b6192873ca7e2b84f7d11e49"

#: The scoped drpc key, committed on purpose.
#
# It cannot do anything a public node could not: reads, plus `eth_call`
# restricted to `QUOTER` above -- a stateless view contract that can move
# nothing.  Measured on every chain: a direct `eth_call` to a pool answers
# HTTP 403 "address 0x...", while the same call through the quoter returns a
# quote.  `eth_sendRawTransaction` is not served at all, which is why
# deployments still go through `networks.py`.
#
# Committing it is the point.  A checkout with no `networks.py` routes on
# fifteen chains, and the browser build ships this URL rather than asking the
# user for one.
SCOPED_KEY = "AskGI4lH8UlFtIRsb5UfRvXOC_8-l9AR8YojRoYgFhqK"


def scoped_rpc(network: str) -> str:
    """The scoped endpoint for a drpc network name.

    drpc puts the network in the path, so one key serves them all; the segment
    is the chain's own name on every chain we carry.
    """
    return f"https://lb.drpc.live/{network}/{SCOPED_KEY}"

CHAINS: dict[str, Chain] = {
    "ethereum": Chain(
        name="ethereum",
        chain_id=1,
        api_name="ethereum",
        rpc_attr="NETWORK",
        native_symbol="ETH",
        wrapped="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        erc4626_allowlist=(SCRVUSD, SDOLA, SDAI, SUSDS, SFRXETH, SGHO, STUSDS),
        quoter=QUOTER,
        public_rpc=scoped_rpc("ethereum"),
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
        quoter=QUOTER,
        public_rpc=scoped_rpc("arbitrum"),
    ),
    "optimism": Chain(
        name="optimism",
        chain_id=10,
        api_name="optimism",
        rpc_attr="OPTIMISM",
        native_symbol="ETH",
        wrapped="0x4200000000000000000000000000000000000006",
        quoter=QUOTER,
        public_rpc=scoped_rpc("optimism"),
    ),
    "base": Chain(
        name="base",
        chain_id=8453,
        api_name="base",
        rpc_attr="BASE",
        native_symbol="ETH",
        wrapped="0x4200000000000000000000000000000000000006",
        quoter=QUOTER,
        public_rpc=scoped_rpc("base"),
    ),
    "gnosis": Chain(
        name="gnosis",
        chain_id=100,
        api_name="xdai",  # NB: the API does not call this "gnosis"
        rpc_attr="GNOSIS",
        native_symbol="XDAI",
        wrapped="0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d",
        transmuters=(
            # USDC (native) <-> USDC.e (bridged), 1:1 both ways.
            ("0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83",
             "0x2a22f9c3b484c3629090FeED35F17Ff8F88f76F0",
             "0x0392A2F5Ac47388945D8c84212469F545fAE52B2"),
        ),  # WXDAI
        quoter=QUOTER,
        public_rpc=scoped_rpc("gnosis"),
    ),
    "polygon": Chain(
        name="polygon",
        chain_id=137,
        api_name="polygon",
        rpc_attr="POLYGON",
        native_symbol="POL",
        wrapped="0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WPOL (ex-WMATIC)
        quoter=QUOTER,
        public_rpc=scoped_rpc("polygon"),
    ),
    "fraxtal": Chain(
        name="fraxtal",
        chain_id=252,
        api_name="fraxtal",
        rpc_attr="FRAXTAL",
        native_symbol="frxETH",
        wrapped="0xFC00000000000000000000000000000000000006",  # wfrxETH
        quoter=QUOTER,
        public_rpc=scoped_rpc("fraxtal"),
    ),
    "bsc": Chain(
        name="bsc",
        chain_id=56,
        api_name="bsc",
        rpc_attr="BSC",
        native_symbol="BNB",
        wrapped="0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
        quoter=QUOTER,
        public_rpc=scoped_rpc("bsc"),
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
    # Six Lite deployments are deliberately absent, each measured:
    #
    # * **fantom** is in wind-down -- 164 of its 321 pools answer neither ABI
    #   spelling and probing them takes 19 minutes.
    # * **kava** resolves a dialect for none of its 19 pools, so it yields zero
    #   arcs.  There is nothing there to route.
    # * **unichain** and **robinhood** are almost entirely scam and dust pools.
    #   Unichain has one real venue -- Lido's wstETH secondary market -- and
    #   everything else is sub-$100 or fake; robinhood's are worse, being built
    #   to move a price oracle, so quoting them would either revert or rug the
    #   caller.  Their universes trip the §9.7 conditioning guard, and that is
    #   the guard working: `fly` pairs a token with 1.1e17 units against SUSHI,
    #   so `a` spans 1e24 within one pool and no value coordinates can hold
    #   both.  Nothing here needed fixing -- the answer to a universe of scams
    #   is not to route it.  Curve is expected to drop robinhood's from the API,
    #   at which point it can come back.
    # **etherlink** and **tac** are present but carry `needs_quoter`: they
    # reject `eth_call` state overrides outright (HTTP 400), and every batched
    # read goes through the quoter, which off mainnet rides along as an
    # override.  So balances come back as zeros and the universe looks like a
    # quiet chain rather than an unreadable one.  They are listed because
    # `deploy_quoter.py` has to be able to name them -- deploying is the fix --
    # and flagged so nothing else pretends they work first.
    "avalanche": Chain(
        name="avalanche",
        chain_id=43114,
        api_name="avalanche",
        rpc_attr="AVALANCHE",
        native_symbol="AVAX",
        wrapped="0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
        lite=True,
        quoter=QUOTER,
        public_rpc=scoped_rpc("avalanche"),
    ),
    "monad": Chain(
        name="monad",
        chain_id=143,
        api_name="monad",
        rpc_attr="MONAD",
        native_symbol="MON",
        wrapped="0x3bD359c1119Da7dA1d913d1C4d2B7c461115433A",
        lite=True,   # ~$7.5M
        quoter=QUOTER,
        public_rpc=scoped_rpc("monad"),
    ),
    "plasma": Chain(
        name="plasma",
        chain_id=9745,
        api_name="plasma",
        rpc_attr="PLASMA",
        native_symbol="XPL",
        wrapped="0x6100E367285b01F48D07953803A2d8dCA5D19873",
        lite=True,   # ~$3.8M
        quoter=QUOTER,
        public_rpc=scoped_rpc("plasma"),
    ),
    "xlayer": Chain(
        name="xlayer",
        chain_id=196,
        api_name="xlayer",
        rpc_attr="XLAYER",
        native_symbol="OKB",
        wrapped="0xe538905cf8410324e03A5A23C1c177a474D59b2b",
        lite=True,   # ~$3.6M
        quoter=QUOTER,
        public_rpc=scoped_rpc("xlayer"),
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
        quoter=QUOTER,
        public_rpc=scoped_rpc("celo"),
    ),
    # Needs `RouteQuoter` deployed before anything here can read it; see
    # `needs_quoter`.  Its `debug_traceCall` works, so once the quoter exists
    # the local EVM can still be warmed by the tracer and tac should behave
    # like any other chain.
    "tac": Chain(
        name="tac",
        chain_id=239,
        api_name="tac",
        rpc_attr="TAC",
        native_symbol="TAC",
        wrapped="0xB63B9f0eb4A6E6f191529D71d4D88cc8900Df2C9",  # WTAC, verified
        lite=True,
        needs_quoter=True,
        quoter=QUOTER,
        public_rpc=scoped_rpc("tac"),
    ),
    # Also needs the quoter -- and even then stays wire-only: HTTP 500 from
    # `eth_createAccessList` in every shape and `debug_traceCall` output that
    # does not decode (`Json_encoding.Cannot_destruct`; it is a Tezos EVM), so
    # there is no path to a local state cache at all.
    "etherlink": Chain(
        name="etherlink",
        chain_id=42793,
        api_name="etherlink",
        rpc_attr="ETHERLINK",
        native_symbol="XTZ",
        wrapped="0xc9B53AB2679f573e480d01e0f49e2B5CFB7a3EAb",  # WXTZ, verified
        lite=True,
        needs_quoter=True,
        public_rpc=scoped_rpc("etherlink"),
    ),
    "sonic": Chain(
        name="sonic",
        chain_id=146,
        api_name="sonic",
        rpc_attr="SONIC",
        native_symbol="S",
        wrapped="0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38",  # wS
        quoter=QUOTER,
        public_rpc=scoped_rpc("sonic"),
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
