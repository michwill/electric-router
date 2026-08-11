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

    # --- ERC4626: allowlist, then verify ---------------------------------
    candidates = [v.lower() for v in chain.erc4626_allowlist if nodes.has(v.lower())]
    if candidates:
        _merge_vaults(nodes, report, candidates, client, value_wei)

    return nodes, report


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
