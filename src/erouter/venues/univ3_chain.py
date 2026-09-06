"""Reading Uniswap v3 pools off a chain, and nothing else.

Kept apart from `univ3`, which is arithmetic and imports no transport: the
model is testable without a socket and this is not.

Only one word per tick is fetched.  A tick entry occupies four, but
`prototype_univ3_fee_slots.py` establishes on the local EVM -- with a control
that moves -- that `feeGrowthOutside0/1X128` and the `tickCumulative` word never
reach the price.  On v3 they cannot: fees accrue to positions rather than
joining the reserves, so the curve never sees them.  Curve is the opposite,
which is why nothing like this applies there.
"""

from __future__ import annotations

from ..core.codec import decode, encode_call
from ..core.keccak import keccak256
from .univ3 import PoolState, Tick

TICKS_SLOT, BITMAP_SLOT = 5, 6
SLOT0, LIQUIDITY_SLOT = 0, 4


def _word(v: int) -> bytes:
    return (v & (2**256 - 1)).to_bytes(32, "big")


def tick_key(tick: int) -> int:
    """Where a tick's entry lives.  Derivable, so it is never stored."""
    return int.from_bytes(keccak256(_word(tick) + _word(TICKS_SLOT)), "big")


def bitmap_key(pos: int) -> int:
    return int.from_bytes(keccak256(_word(pos) + _word(BITMAP_SLOT)), "big")


def _liquidity_net(raw: bytes) -> int:
    """The signed half of the first word: `liquidityNet` sits above `Gross`."""
    net = int.from_bytes(raw[:16], "big")
    return net - 2**128 if net >= 2**127 else net


def read_pools(rpc, pools: dict, block: int, *, ticks: int = 16,
               words: int = 3, batch: int = 3000):
    """`pool -> (PoolState, [Tick])` for every pool that answered.

    `pools` maps address to `(token0, token1, fee, decimals0, decimals1)`.
    Three round trips whatever the pool count: the immutables, the bitmap
    words, then one word per initialized tick.
    """
    at = hex(block)
    addrs = list(pools)
    got = rpc.fetch_multi(
        [("eth_getStorageAt", [p, hex(SLOT0), at]) for p in addrs]
        + [("eth_getStorageAt", [p, hex(LIQUIDITY_SLOT), at]) for p in addrs]
        + [("eth_call", [{"to": p, "data": "0x" + encode_call(
            "tickSpacing()").hex()}, at]) for p in addrs], concurrent=True)

    n = len(addrs)
    state: dict[str, PoolState] = {}
    for k, pool in enumerate(addrs):
        slot0, liq, spacing = got[k], got[k + n], got[k + 2 * n]
        if not all(isinstance(v, str) for v in (slot0, liq, spacing)):
            continue
        raw = bytes.fromhex(slot0[2:])
        # slot0 is packed little-end-first: sqrtPriceX96 in the low 160 bits,
        # then the current tick as an int24 above it.
        packed = int.from_bytes(raw, "big")
        tick = (packed >> 160) & 0xFFFFFF
        _t0, _t1, fee, dec0, dec1 = pools[pool]
        state[pool] = PoolState(
            sqrt_price_x96=packed & (2**160 - 1),
            tick=tick - 2**24 if tick >= 2**23 else tick,
            liquidity=int(liq, 16),
            tick_spacing=decode(["int24"], bytes.fromhex(spacing[2:]))[0],
            fee=fee, decimals0=dec0, decimals1=dec1)

    asks, index = [], []
    for pool, st in state.items():
        centre = (st.tick // st.tick_spacing) >> 8
        for pos in range(centre - words, centre + words + 1):
            asks.append(("eth_getStorageAt", [pool, hex(bitmap_key(pos)), at]))
            index.append((pool, pos))
    found: dict[str, list[int]] = {}
    for start in range(0, len(asks), batch):
        answers = rpc.fetch_multi(asks[start:start + batch], concurrent=True)
        for (pool, pos), raw in zip(index[start:start + batch], answers,
                                    strict=True):
            if not isinstance(raw, str):
                continue
            bits = int(raw, 16)
            spacing = state[pool].tick_spacing
            found.setdefault(pool, []).extend(
                ((pos << 8) + b) * spacing for b in range(256) if bits >> b & 1)

    # Only the nearest, and only as many as the bank will use: a pool holds
    # thousands and the arcs are truncated anyway.
    asks, index = [], []
    for pool, all_ticks in found.items():
        here = state[pool].tick
        near = (sorted((t for t in all_ticks if t <= here), reverse=True)[:ticks]
                + sorted(t for t in all_ticks if t > here)[:ticks])
        for t in near:
            asks.append(("eth_getStorageAt", [pool, hex(tick_key(t)), at]))
            index.append((pool, t))
    out: dict[str, list[Tick]] = {}
    for start in range(0, len(asks), batch):
        answers = rpc.fetch_multi(asks[start:start + batch], concurrent=True)
        for (pool, t), raw in zip(index[start:start + batch], answers,
                                  strict=True):
            if isinstance(raw, str):
                out.setdefault(pool, []).append(
                    Tick(t, _liquidity_net(bytes.fromhex(raw[2:]))))

    return {p: (state[p], out[p]) for p in out if p in state}
