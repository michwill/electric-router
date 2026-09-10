"""Reading v4 pool state, through `StateView` rather than out of storage.

v3's reader goes at the pool's own slots with `eth_getStorageAt`, which the
committed endpoint refuses -- hence `--univ3` needing `--private`.  v4 cannot be
read that way at all: every pool lives in one singleton behind `extsload`, so
there are no per-pool slots to name.  What it has instead is `StateView`, a
periphery contract exposing the same four things as ordinary `eth_call`s:

    getSlot0(poolId)        -> sqrtPriceX96, tick, protocolFee, lpFee
    getLiquidity(poolId)    -> liquidity
    getTickBitmap(poolId, word)
    getTickLiquidity(poolId, tick) -> liquidityGross, liquidityNet

Which is a straight improvement: no storage reads means no `--private`, and the
shape is otherwise v3's, so `univ3.arcs` takes the result unchanged.

`lpFee` from `getSlot0` is the pool's *current* fee.  For the pools this venue
admits it always equals the `PoolKey`'s, because a static fee cannot be
overridden -- `OVERRIDE_FEE_FLAG` works only on dynamic-fee pools, and those are
tier 3 and never reach here.  It is read anyway and checked, because a silent
disagreement would mean the tier gate had let something through.
"""

from __future__ import annotations

from ..core.codec import decode, encode_call
from .univ3 import PoolState, Tick

#: Uniswap's `StateView`, per chain.  Periphery, so it is not the singleton and
#: not deterministic across chains the way `ElectricRouter` is.
STATE_VIEW = {
    "ethereum": "0x7ffe42c4a5deea5b0fec41c94c136cf115597227",
    "arbitrum": "0x76fd297e2d437cd7f76d50f01afe6160f86e9990",
    "optimism": "0xc18a3169788f4f75a170290584eca6395c75ecdb",
    "base": "0xa3c0c9b65bad0b08107aa264b0f3db444b867a71",
    "polygon": "0x5ea1bd7974c8a611cbab0bdcafcb1d9cc9b3ba5a",
    "bsc": "0xd13dd3d6e93f276fafc9db9e6bb47c1180aee0c4",
    "avalanche": "0xc3c9e198c735a4b97e3e683f391ccbdd60b69286",
}


def _call(view: str, sig: str, *args) -> tuple:
    return ("eth_call", [{"to": view, "data": "0x" + encode_call(sig, *args).hex()}])


def _at(call: tuple, block: str) -> tuple:
    method, params = call
    return (method, [params[0], block])


def read_pools(rpc, view: str, pools: dict, block: int, *, ticks: int = 16,
               words: int = 3):
    """`poolId -> (PoolState, [Tick])` for every pool that answered.

    `pools` maps a `PoolId` to `(key, decimals0, decimals1)`, where `key` is a
    `univ4.PoolKey`.  Three round trips whatever the pool count, the same shape
    v3 reads in: state, then bitmap words, then one call per initialized tick.
    """
    at = hex(block)
    ids = list(pools)
    if not ids:
        return {}
    raw_ids = {pid: bytes.fromhex(pid[2:]) for pid in ids}
    got = rpc.fetch_multi(
        [_at(_call(view, "getSlot0(bytes32)", raw_ids[p]), at) for p in ids]
        + [_at(_call(view, "getLiquidity(bytes32)", raw_ids[p]), at) for p in ids],
        concurrent=True)

    n = len(ids)
    state: dict[str, PoolState] = {}
    for k, pid in enumerate(ids):
        slot0, liq = got[k], got[k + n]
        if not (isinstance(slot0, str) and isinstance(liq, str)):
            continue
        try:
            sqrt_price, tick, _protocol, lp_fee = decode(
                ["uint160", "int24", "uint24", "uint24"],
                bytes.fromhex(slot0[2:]))
            liquidity = decode(["uint128"], bytes.fromhex(liq[2:]))[0]
        except (ValueError, IndexError):
            continue
        if not sqrt_price:
            continue                      # never initialised
        key, dec0, dec1 = pools[pid]
        if lp_fee != key.fee:
            # A static fee cannot be overridden, so this can only mean the tier
            # gate let a dynamic-fee pool through.  Refuse it rather than build
            # an arc priced at a fee the next swap will not charge.
            continue
        state[pid] = PoolState(
            sqrt_price_x96=sqrt_price, tick=tick, liquidity=liquidity,
            tick_spacing=key.tick_spacing, fee=key.fee,
            decimals0=dec0, decimals1=dec1)

    asks, index = [], []
    for pid, st in state.items():
        centre = (st.tick // st.tick_spacing) >> 8
        for w in range(centre - words // 2, centre + words // 2 + 1):
            asks.append(_at(_call(view, "getTickBitmap(bytes32,int16)",
                                  raw_ids[pid], w), at))
            index.append((pid, w))
    bitmaps = rpc.fetch_multi(asks, concurrent=True) if asks else []

    live: dict[str, list[int]] = {}
    for (pid, word), raw in zip(index, bitmaps, strict=True):
        if not isinstance(raw, str) or len(raw) < 3:
            continue
        bits = int(raw, 16)
        if not bits:
            continue
        spacing = state[pid].tick_spacing
        for bit in range(256):
            if bits >> bit & 1:
                live.setdefault(pid, []).append(((word << 8) + bit) * spacing)

    asks, index = [], []
    for pid, found in live.items():
        here = state[pid].tick
        # Nearest first, both sides: a walk crosses the near ones or nothing.
        for tick in sorted(found, key=lambda t: abs(t - here))[:2 * ticks]:
            asks.append(_at(_call(view, "getTickLiquidity(bytes32,int24)",
                                  raw_ids[pid], tick), at))
            index.append((pid, tick))
    nets = rpc.fetch_multi(asks, concurrent=True) if asks else []

    out: dict[str, tuple] = {}
    ticks_of: dict[str, list[Tick]] = {}
    for (pid, tick), raw in zip(index, nets, strict=True):
        if not isinstance(raw, str) or len(raw) < 3:
            continue
        try:
            _gross, net = decode(["uint128", "int128"], bytes.fromhex(raw[2:]))
        except (ValueError, IndexError):
            continue
        ticks_of.setdefault(pid, []).append(Tick(index=tick, liquidity_net=net))
    for pid, st in state.items():
        out[pid] = (st, sorted(ticks_of.get(pid, []), key=lambda t: t.index))
    return out
