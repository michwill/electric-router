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


def build_node_map(
    pools: list[PoolSpec],
    chain: Chain,
    client: QuoterClient,
    *,
    value_wei: int = 0,
) -> tuple[NodeMap, WrapperReport]:
    """Every pool coin as a node, with wrappers and vaults merged in.

    `value_wei` is the size of the trade being routed, used only to decide
    whether a vault's deposit cap is large enough to call it unbounded.
    """
    nodes = NodeMap()
    report = WrapperReport()

    for pool in pools:
        for coin in pool.coins:
            nodes.add_token(coin.address, coin.symbol, coin.decimals)

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
    candidates = [v.lower() for v in chain.erc4626_allowlist if nodes.has(v.lower())]
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

    vaults = [v.lower() for v in chain.oneway_vaults if nodes.has(v.lower())]
    for vault in vaults:
        unit = 10 ** nodes.decimals(vault)
        calls.extend([
            Call(vault, encode_call("asset()")),
            Call(vault, encode_call("convertToAssets(uint256)", unit)),
            Call(vault, encode_call("totalAssets()")),
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
        asset_ans, rate_ans, total_ans = answers[base + 3 * k : base + 3 * k + 3]
        if asset_ans.status is not Status.VALUE or rate_ans.status is not Status.VALUE:
            continue
        asset = "0x" + asset_ans.data[-20:].hex()
        if not nodes.has(asset):
            continue
        unit = 10 ** nodes.decimals(vault)
        rate = rate_ans.uint() / unit  # assets per share
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
