"""Build the node map: native wrappers and validated ERC4626 vaults.

Discovery is automatic and produces a *report*; merging is gated on the
per-chain allowlist in `chains.py`.  Every allowlisted vault is still checked
at runtime, because an allowlist entry can go stale -- a vault can pause
deposits or grow a queue long after it was added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import encode_call
from ..core.nodes import Conversion, ConversionKind, NodeMap
from ..core.pools import PoolSpec
from ..core.quoter import QuoterClient
from ..core.transport import Call, Status
from ..core.types import ArcKind, PoolArc
from .chains import NATIVE_SENTINEL, Chain

# A vault must accept at least this multiple of the trade before it can be
# treated as a zero-resistance short circuit rather than a capped arc.
MIN_DEPOSIT_HEADROOM = 100
LINEARITY_TOL = 1e-9


@dataclass(slots=True)
class VaultReport:
    token: str
    symbol: str
    asset: str = ""
    decimals: int = 18
    rate_num: int = 0
    rate_den: int = 1
    max_deposit: int = 0
    linearity_error: float = 0.0
    merged: bool = False
    reason: str = ""


@dataclass(slots=True)
class WrapperReport:
    native_merged: list[tuple[str, str]] = field(default_factory=list)
    vaults: list[VaultReport] = field(default_factory=list)

    @property
    def merged_vaults(self) -> list[VaultReport]:
        return [v for v in self.vaults if v.merged]

    @property
    def rejected_vaults(self) -> list[VaultReport]:
        return [v for v in self.vaults if not v.merged and v.reason]


def merge_candidates(chain: Chain, facts=None) -> list[str]:
    """Which vaults are worth *trying* to merge, from measurement where there is
    any.

    R5 gates merging on an allowlist because linearity says nothing about
    whether you can get back out: pufETH reports `asset = WETH`, converts with
    no error and caps at `2**256-1`, yet redeems through a queue.  Merging it
    would declare it equal to WETH at NAV and mint the market's discount from
    nothing.  `maxRedeem` cannot answer either -- it is per-owner and returns 0
    for an address holding no shares.

    Executing both directions can answer it, and `data/facts` now does, so the
    list stops being a judgement and becomes a measurement.  It is still an
    allowlist -- a vault has to earn its way on by redeeming, not merely by
    failing to look suspicious -- but nobody maintains it, and a vault that
    stops honouring redemptions falls off at the next facts build rather than
    when someone notices.

    The hand-written list stays as an addition, not a replacement: a vault the
    harness could not test is absent from the facts, and absent must not mean
    refused.  Measurement can only widen this, never narrow it below what a
    human vouched for.

    And `oneway_vaults` always wins.  A measured redemption is one redemption,
    of one size, at one block, by one holder -- it is not proof that the exit
    is open.  pufETH proves the point: this probe found its `redeem` answering
    and would have merged it, which is precisely the case R5 exists to forbid,
    because the way out is a withdrawal queue and a merge claims there is none.
    Measurement may say "this looks fine"; only a human may say "and I know
    why it is fine".  So the deny list is not a fallback for what measurement
    misses, it is a veto over what measurement is not competent to judge.
    """
    hand = [v.lower() for v in chain.erc4626_allowlist]
    wrappers = getattr(facts, "wrappers", None) or {}
    measured = [
        address for address, entry in wrappers.items()
        if entry.get("mint") is True and entry.get("redeem") is True
    ]
    denied = {v.lower() for v in getattr(chain, "oneway_vaults", ())}
    denied |= {token.lower() for token, _, _, _, _ in getattr(chain, "stake_arcs", ())}
    return sorted((set(hand) | set(measured)) - denied)


def build_node_map(
    pools: list[PoolSpec],
    chain: Chain,
    client: QuoterClient,
    *,
    value_wei: int = 0,
    facts=None,
) -> tuple[NodeMap, WrapperReport]:
    """Every pool coin as a node, with wrappers and vaults merged in.

    `value_wei` is the size of the trade being routed, used only to decide
    whether a vault's deposit cap is large enough to call it unbounded.
    `facts` supplies measured mint/redeem verdicts; see `merge_candidates`.
    """
    nodes = NodeMap()
    report = WrapperReport()

    for pool in pools:
        for coin in pool.coins:
            nodes.add_token(coin.address, coin.symbol, coin.decimals)

    # LP tokens are deliberately *not* registered here.  A pool's LP is a node
    # only when some other pool trades it -- 3Crv, crvFRAX, sbtcCrv, gnosis's
    # x3CRV -- and `build_arcs` then finds it already present.
    #
    # Registering all of them instead makes an arc pair per coin for every pool
    # on the chain, mostly to tokens nothing else trades: 759 arcs became 2,020
    # on mainnet, and crvUSD -> sDOLA at $2M went from 1,419,819 sDOLA to
    # 1,136,396 in 2.2x the time.  A dead-end node cannot improve a route, but
    # it can and did drag the reference-price fit and the solver with it.

    # --- native wrapper: 1:1, no probing needed --------------------------
    wrapped = chain.wrapped.lower()
    sentinel = NATIVE_SENTINEL.lower()
    if nodes.has(sentinel):
        if not nodes.has(wrapped):
            nodes.add_token(wrapped, f"W{chain.native_symbol}", 18)
        # The wrapped ERC20 is canonical because most pools hold it; only the
        # handful of sentinel pools then need a conversion leg.
        nodes.merge(
            Conversion(
                kind=ConversionKind.NATIVE_WRAP,
                token=sentinel,
                canonical=wrapped,
                rate_num=1,
                rate_den=1,
                target=chain.wrapped,
            )
        )
        nodes.symbol_of.setdefault(sentinel, chain.native_symbol)
        report.native_merged.append((chain.native_symbol, f"W{chain.native_symbol}"))

    # --- wstETH: rate-bearing, but otherwise a native wrapper ------------
    pairs = [
        (t.lower(), c.lower())
        for t, c in chain.wsteth_pairs
        if nodes.has(t.lower()) and nodes.has(c.lower())
    ]
    if pairs:
        _merge_wsteth(nodes, report, pairs, client)

    # --- ERC4626: allowlist, then verify ---------------------------------
    candidates = [v for v in merge_candidates(chain, facts) if nodes.has(v)]
    if candidates:
        _merge_vaults(nodes, report, candidates, client, value_wei)

    return nodes, report


def _merge_wsteth(
    nodes: NodeMap,
    report: WrapperReport,
    pairs: list[tuple[str, str]],
    client: QuoterClient,
) -> None:
    """Merge wstETH into stETH -- lossless, unbounded, and exactly linear.

    Measured on mainnet at block 25,734,769: the rate is 1.241440951 stETH per
    wstETH, `getStETHByWstETH` is linear to 1.3e-19 across eight decades, and a
    1 stETH round trip loses 1 wei to integer rounding.  There is no deposit
    cap, no withdrawal queue and no cooldown -- the three things that disqualify
    pufETH and sUSDe under R5 -- so it is a short circuit in value coordinates.

    Not merging it is expensive rather than merely incomplete: without the
    merge, wstETH cannot reach the deep ETH/stETH pool, and 50 wstETH -> WETH
    quoted 10.7% below NAV against curve_solver, which does model the wrap.
    """
    calls: list[Call] = []
    for token, _canonical in pairs:
        unit = 10 ** nodes.decimals(token)
        calls.extend(
            [
                Call(token, encode_call("getStETHByWstETH(uint256)", unit)),
                Call(token, encode_call("getStETHByWstETH(uint256)", unit * 1_000_000)),
            ]
        )

    answers = client.raw(calls)
    for k, (token, canonical) in enumerate(pairs):
        one_ans, million_ans = answers[2 * k], answers[2 * k + 1]
        entry = VaultReport(token=token, symbol=nodes.symbol(token), asset=canonical)
        report.vaults.append(entry)

        if one_ans.status is not Status.VALUE or one_ans.uint() == 0:
            entry.reason = "getStETHByWstETH returned nothing"
            continue
        unit = 10 ** nodes.decimals(token)
        one = one_ans.uint()
        entry.rate_num, entry.rate_den = one, unit
        entry.decimals = nodes.decimals(token)

        million = million_ans.uint() if million_ans.status is Status.VALUE else 0
        if million == 0:
            entry.reason = "rate is not linear (no answer at scale)"
            continue
        entry.linearity_error = abs(million - one * 1_000_000) / max(million, 1)
        if entry.linearity_error > LINEARITY_TOL:
            entry.reason = f"non-linear ({entry.linearity_error:.2e})"
            continue

        nodes.merge(
            Conversion(
                kind=ConversionKind.WSTETH,
                token=token,
                canonical=canonical,
                rate_num=one,
                rate_den=unit,
                target=token,
            )
        )
        entry.merged = True


def _merge_vaults(
    nodes: NodeMap,
    report: WrapperReport,
    vaults: list[str],
    client: QuoterClient,
    value_wei: int,
) -> None:
    calls: list[Call] = []
    for vault in vaults:
        decimals = nodes.decimals(vault)
        unit = 10**decimals
        calls.extend(
            [
                Call(vault, encode_call("asset()")),
                Call(vault, encode_call("decimals()")),
                Call(vault, encode_call("maxDeposit(address)", "0x" + "11" * 20)),
                Call(vault, encode_call("convertToAssets(uint256)", unit)),
                Call(vault, encode_call("convertToAssets(uint256)", unit * 1_000_000)),
            ]
        )
    answers = client.raw(calls)

    for k, vault in enumerate(vaults):
        chunk = answers[5 * k : 5 * k + 5]
        entry = VaultReport(token=vault, symbol=nodes.symbol(vault))
        report.vaults.append(entry)

        if chunk[0].status is not Status.VALUE:
            entry.reason = "no asset()"
            continue
        asset = "0x" + chunk[0].data[-20:].hex()
        entry.asset = asset
        if chunk[1].status is Status.VALUE:
            entry.decimals = int(chunk[1].uint())
        unit = 10**entry.decimals

        if not nodes.has(asset):
            entry.reason = "asset is not a routable token"
            continue
        if chunk[3].status is not Status.VALUE or chunk[3].uint() == 0:
            entry.reason = "convertToAssets returned nothing"
            continue

        one, million = chunk[3].uint(), chunk[4].uint() if chunk[4].status is Status.VALUE else 0
        entry.rate_num, entry.rate_den = one, unit
        entry.max_deposit = chunk[2].uint() if chunk[2].status is Status.VALUE else 0

        # Linear across six decades?  Necessary, and nowhere near sufficient.
        if million == 0:
            entry.reason = "convertToAssets is not linear (no answer at scale)"
            continue
        entry.linearity_error = abs(million - one * 1_000_000) / max(million, 1)
        if entry.linearity_error > LINEARITY_TOL:
            entry.reason = f"non-linear ({entry.linearity_error:.2e})"
            continue

        # Deposit headroom.  NB `maxRedeem(addr)` is per-owner and returns 0
        # for an address holding no shares, so it cannot be used here; the
        # redeem side is covered by the executed round trip in CI.
        needed = MIN_DEPOSIT_HEADROOM * max(value_wei, unit)
        if entry.max_deposit < needed:
            entry.reason = f"deposit cap {entry.max_deposit} below {needed}"
            continue

        nodes.merge(
            Conversion(
                kind=ConversionKind.ERC4626,
                token=vault,
                canonical=asset,
                rate_num=entry.rate_num,
                rate_den=entry.rate_den,
                target=vault,
            )
        )
        entry.merged = True


def discover_vaults(
    nodes: NodeMap, client: QuoterClient, tokens: list[str] | None = None
) -> list[VaultReport]:
    """Report every ERC4626-looking token, merged or not.

    Diagnostic only: nothing here changes the node map.  It exists so the
    allowlist can be reviewed against what is actually out there.
    """
    targets = [t.lower() for t in (tokens or list(nodes.node_of))]
    calls = [Call(t, encode_call("asset()")) for t in targets]
    answers = client.raw(calls)

    found = []
    for token, answer in zip(targets, answers, strict=True):
        if answer.status is not Status.VALUE:
            continue
        asset = "0x" + answer.data[-20:].hex()
        if int(asset, 16) == 0:
            continue
        found.append(
            VaultReport(
                token=token,
                symbol=nodes.symbol(token),
                asset=asset,
                reason="discovered, not allowlisted",
            )
        )
    return found


#: Whose deposit room to ask about.  `maxDeposit` is per-owner, so it needs
#: *someone*; nobody in particular is the honest choice, and a vault that gates
#: by allowlist then reads as closed, which for routing purposes it is.
DEPOSITOR = "0x" + "11" * 20


def mintable_vaults(nodes: NodeMap, client: QuoterClient,
                    chain_id: int | None = None) -> list[str]:
    """Every token in the graph that mints itself from another token in it.

    Minting is not the same claim as merging, and only the second one needs a
    human.  A merge asserts the exit is open and unbounded, which is why R5
    keeps it on an allowlist -- pufETH is perfectly linear and redeems through
    a queue.  A *deposit* arc asserts only what the vault will do on request:
    take the asset, hand back shares, at a rate the quoter reads and the
    on-chain verification adjudicates.  So this one can be discovered.

    Which matters, because the hand-written list is short and the universes are
    not.  Measured on mainnet: nine vaults modelled, sixteen ignored, eleven of
    those mintable from an asset already in the graph -- ynUSDx, ynRWAx and
    srRoyUSDC from USDC, ynETHx from WETH, and so on.  Fraxtal's sdUSD was the
    whole of that chain's vault coverage, missing.  A token nothing mints can
    only be reached by buying it in a pool, which is how crvUSD -> sDOLA at $5M
    lost 16% to a route that minted instead of bought.

    A vault already merged with its asset falls out downstream rather than
    here: the two share a node, and the caller skips an arc whose ends are the
    same node.

    `asset()` cannot change, so the answer is cached per chain and a warm
    universe costs nothing.
    """
    from .cache import TokenFactsCache

    facts = TokenFactsCache()
    known = facts.load(chain_id) if chain_id is not None else {}
    tokens = [t.lower() for t in nodes.node_of]
    ask = [t for t in tokens if "asset" not in known.get(t, {})]

    learned: dict[str, dict] = {}
    if ask:
        answers = client.raw([Call(t, encode_call("asset()")) for t in ask])
        for token, answer in zip(ask, answers, strict=True):
            asset = ""
            if answer.status is Status.VALUE and len(answer.data) >= 32:
                word = answer.data[-20:].hex()
                if int(word, 16):
                    asset = "0x" + word
            learned[token] = {"asset": asset}
        if chain_id is not None:
            facts.save(chain_id, learned)

    out = []
    for token in tokens:
        asset = (learned.get(token) or known.get(token) or {}).get("asset") or ""
        if asset and nodes.has(asset):
            out.append(token)
    return out


def build_stake_arcs(
    nodes: NodeMap, chain: Chain, client: QuoterClient
) -> list[PoolArc]:
    """One-way instant conversions, as capped linear arcs.

    A merge is bidirectional by definition -- one node, zero resistance both
    ways -- so minting cannot be one.  Lido pays 1 stETH per ETH instantly but
    withdrawal is a queue; sUSDe mints on demand but redemption has a seven-day
    cooldown; pufETH the same with a queue.  Merging any of them would let the
    router unstake for free and emit routes that cannot execute.

    They are exactly the §2.3 clamped-arc shape instead: `a` = the mint rate,
    `B = 0` (genuinely linear, not a chord approximation), and a finite cap.
    The cap is not decoration -- an uncapped linear arc with `eps < 0` gives
    unbounded flow (§2.3 rule 2), and `eps < 0` is precisely what happens when
    the token trades above NAV, which is when minting is attractive.

    Measured on the ETH/stETH pool at block 25,734,769: the pool beats minting
    by 0.7 bp at 1 ETH and 0.2 bp at 1,000, then loses by 2.1 bp at 5,000,
    48.5 bp at 20,000 and 7,803 bp at 100,000.  Minting is a hard floor at
    exactly 1:1, and it binds at the sizes where the quadratic model is least
    trustworthy.

    `convex_flag` stays False: these arcs are exactly linear, so §5.5's
    certificate is still valid over them.  `clamped` is True because the cap
    invariant must apply.
    """
    arcs: list[PoolArc] = []
    calls: list[Call] = []
    native: list[tuple] = []
    for token_in, token_out, kind, target, cap_sig in chain.stake_arcs:
        if not (nodes.has(token_in.lower()) and nodes.has(token_out.lower())):
            continue
        native.append((token_in.lower(), token_out.lower(), kind, target.lower(), cap_sig))
        calls.append(Call(target, encode_call(cap_sig)))

    vaults = sorted(
        {v.lower() for v in chain.oneway_vaults if nodes.has(v.lower())}
        | set(mintable_vaults(nodes, client, chain.chain_id))
    )
    for vault in vaults:
        unit = 10 ** nodes.decimals(vault)
        calls.extend([
            Call(vault, encode_call("asset()")),
            Call(vault, encode_call("convertToAssets(uint256)", unit)),
            Call(vault, encode_call("totalAssets()")),
            Call(vault, encode_call("maxDeposit(address)", DEPOSITOR)),
        ])

    if not calls:
        return arcs
    answers = client.raw(calls)

    for k, (token_in, token_out, kind, target, _sig) in enumerate(native):
        limit = answers[k]
        if limit.status is not Status.VALUE or limit.uint() == 0:
            continue
        tau, sigma = nodes.node(token_in), nodes.node(token_out)
        if tau == sigma:
            continue
        arcs.append(PoolArc(
            id=f"stake:{target}:{token_in[:10]}>{token_out[:10]}",
            pool=target, kind=ArcKind[kind], i=0, j=0, n_coins=0,
            token_in=token_in, token_out=token_out, tau=tau, sigma=sigma,
            a=1.0, B=0.0, cap=limit.uint() / 10 ** nodes.decimals(token_in),
            clamped=True, convex_flag=False,
            decimals_in=nodes.decimals(token_in),
            decimals_out=nodes.decimals(token_out),
            tvl_usd=0.0,  # deliberately no weight in the §4 price fit: the
                          # reference price should come from markets, not the
                          # mint rate, which is an upper bound on the token
            note=f"mint {nodes.symbol(token_out)}",
        ))

    base = len(native)
    for k, vault in enumerate(vaults):
        asset_ans, rate_ans, total_ans, room_ans = answers[base + 4 * k : base + 4 * k + 4]
        if asset_ans.status is not Status.VALUE or rate_ans.status is not Status.VALUE:
            continue
        asset = "0x" + asset_ans.data[-20:].hex()
        if not nodes.has(asset):
            continue
        # `convertToAssets` was handed one whole share, so its answer is one
        # share's worth of *asset*, in the asset's decimals -- not the vault's.
        # Dividing by the vault's unit is only right when the two agree, which
        # every hand-listed vault happened to do and no discovered one need:
        # ynUSDx is 18-decimal shares over 6-decimal USDC, and read that way it
        # priced at 917,882,555,091 shares per USDC, which the solver reads as
        # free money.
        rate = rate_ans.uint() / 10 ** nodes.decimals(asset)  # assets per share
        if rate <= 0:
            continue
        tau, sigma = nodes.node(asset), nodes.node(vault)
        if tau == sigma:
            continue
        # `maxDeposit` is routinely 2**256-1 on these, which is not a capacity
        # the model may use.  `totalAssets` is a real, finite, on-chain measure
        # that scales with the protocol, and it is conservative.
        total = total_ans.uint() if total_ans.status is Status.VALUE else 0
        if total == 0:
            continue
        # A vault that will not take a deposit right now is not an arc.
        # Discovery finds those -- USD3 answers `maxDeposit` with zero -- where
        # a hand-written list would simply not have named them.
        room = room_ans.uint() if room_ans.status is Status.VALUE else 0
        if room == 0:
            continue
        total = min(total, room)
        arcs.append(PoolArc(
            id=f"mint:{vault}:{asset[:10]}>{vault[:10]}",
            pool=vault, kind=ArcKind.ERC4626_DEPOSIT, i=0, j=0, n_coins=0,
            token_in=asset, token_out=vault, tau=tau, sigma=sigma,
            a=1.0 / rate, B=0.0,
            cap=total / 10 ** nodes.decimals(asset),
            clamped=True, convex_flag=False,
            decimals_in=nodes.decimals(asset),
            decimals_out=nodes.decimals(vault),
            tvl_usd=0.0,
            note=f"mint {nodes.symbol(vault)}",
        ))
    return arcs


def build_lending_arcs(
    nodes: NodeMap, chain: Chain, client: QuoterClient, facts=None
) -> list[PoolArc]:
    """Leaving a lending wrapper, where the protocol still allows it.

    Curve's lending pools trade wrapped tokens -- cDAI against cUSDC -- and the
    underlying only reachable through `exchange_underlying`, which on both
    surviving deployments reverts: Compound V2 answers "mint is paused" and
    Aave V2's reserves are frozen.  Doing the conversion *outside* the pool is
    the same trade in three steps, and the step that fails is the deposit, not
    the withdrawal.

    So a redemption is its own arc.  That makes `USDT -> cDAI -> DAI` routable
    through a pool whose own `exchange_underlying` cannot run, and it is why
    these are arcs rather than node merges: a merge is symmetric and would
    claim the mint direction too.

    Linear, like the mint arcs above -- `underlying = cTokens * rate` with the
    rate carrying the decimal difference -- so `B = 0` and the §5.5 certificate
    survives them.  The cap is the market's own cash: redemption pays out of
    it, and an uncapped linear arc with `eps < 0` gives unbounded flow (§2.3
    rule 2).

    `facts` is the committed capability table.  A direction absent from it is
    not built, which is how a paused mint stays out of the graph without a
    blacklist and how it would return on its own if the protocol reopened.
    """
    wrappers = getattr(facts, "wrappers", {}) or {}
    if not wrappers:
        return []

    known = [(a.lower(), entry) for a, entry in wrappers.items()
             if nodes.has(a.lower()) and (entry.get("redeem") or entry.get("mint"))]
    if not known:
        return []

    calls: list[Call] = []
    for address, _ in known:
        calls.extend([
            Call(address, encode_call("underlying()")),
            Call(address, encode_call("exchangeRateStored()")),
            Call(address, encode_call("getCash()")),
        ])
    answers = client.raw(calls)

    arcs: list[PoolArc] = []
    for k, (address, entry) in enumerate(known):
        under_ans, rate_ans, cash_ans = answers[3 * k : 3 * k + 3]
        if under_ans.status is not Status.VALUE or rate_ans.status is not Status.VALUE:
            continue
        underlying = "0x" + under_ans.data[-20:].hex()
        if not nodes.has(underlying):
            continue
        rate = rate_ans.uint()
        cash = cash_ans.uint() if cash_ans.status is Status.VALUE else 0
        if rate == 0 or cash == 0:
            continue
        wrapped_dec = nodes.decimals(address)
        under_dec = nodes.decimals(underlying)
        tau, sigma = nodes.node(address), nodes.node(underlying)
        if tau == sigma:
            continue

        if entry.get("redeem"):
            # `a` is in value coordinates: how much underlying one wrapped
            # token buys, both sides measured in their own units.
            per_token = rate / 10 ** 18 * 10 ** wrapped_dec / 10 ** under_dec
            arcs.append(PoolArc(
                id=f"redeem:{address}:{address[:10]}>{underlying[:10]}",
                pool=address, kind=ArcKind.LEND_REDEEM, i=0, j=0, n_coins=0,
                token_in=address, token_out=underlying, tau=tau, sigma=sigma,
                a=per_token, B=0.0,
                # Only what the market can actually pay out today.
                cap=cash / 10 ** under_dec / max(per_token, 1e-30),
                clamped=True, convex_flag=False,
                decimals_in=wrapped_dec, decimals_out=under_dec,
                tvl_usd=0.0,
                note=f"redeem {nodes.symbol(address)}",
            ))
        if entry.get("mint"):
            arcs.append(PoolArc(
                id=f"lend:{address}:{underlying[:10]}>{address[:10]}",
                pool=address, kind=ArcKind.LEND_MINT, i=0, j=0, n_coins=0,
                token_in=underlying, token_out=address, tau=sigma, sigma=tau,
                a=10 ** 18 / rate * 10 ** under_dec / 10 ** wrapped_dec, B=0.0,
                cap=float("inf") if False else cash / 10 ** under_dec,
                clamped=True, convex_flag=False,
                decimals_in=under_dec, decimals_out=wrapped_dec,
                tvl_usd=0.0,
                note=f"mint {nodes.symbol(address)}",
            ))
    return arcs
