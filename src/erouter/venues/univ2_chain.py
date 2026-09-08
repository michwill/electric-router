"""Reading Uniswap v2 pairs off a chain, and nothing else.

Kept apart from `univ2`, which is arithmetic and imports no transport, for the
reason `univ3_chain` is: the model stays testable without a socket.

**One `eth_call` per pair, and that is the whole read.**  `getReserves()`
returns both sides and a timestamp in a single word-triple, and a pair holds
nothing else the price depends on -- no ticks, no bitmap, no liquidity slot.
So unlike v3 this needs no `eth_getStorageAt` and works against the committed
scoped endpoint, which is the difference between a venue that needs
`--private` and one that does not.

The fee is *not* read.  Uniswap's own factory hard-codes 30 bp with no getter,
and the forks that changed it did so in bytecode rather than storage, so a fee
is a fact about a deployment that belongs in the census beside the pair.
"""

from __future__ import annotations

from ..core.codec import encode_call
from .univ2 import DEFAULT_FEE_BPS, PairState

#: `getReserves()` packs `uint112, uint112, uint32` into one word, low first.
_RESERVE_MASK = (1 << 112) - 1


def decode_reserves(raw: bytes) -> tuple[int, int]:
    """`(reserve0, reserve1)` from the returned word triple.

    Vyper and Solidity both pad each return value to a word, so the two
    reserves arrive as separate 32-byte slots rather than packed -- reading it
    as the *storage* layout would give one enormous number and one zero, which
    is a plausible-looking pair rather than an error.

    A short answer raises for the same reason.  `eth_call` to an address with
    no code returns `0x`, and reading that as a pool holding nothing says the
    pair is dead when the truth is that nothing was asked.  The census counts
    the two separately.
    """
    if len(raw) < 64:
        raise ValueError(f"getReserves() answered {len(raw)} bytes, not 64")
    return (int.from_bytes(raw[:32], "big") & _RESERVE_MASK,
            int.from_bytes(raw[32:64], "big") & _RESERVE_MASK)


def read_pairs(rpc, pairs: dict, block: int):
    """`pair -> PairState` for every pair that answered.

    `pairs` maps address to `(token0, token1, fee_bps, decimals0, decimals1)`,
    which is the census row.  One round trip whatever the pair count.

    A pair that does not answer, or answers with a side at zero, is simply
    absent: it has no arc to give and saying so here beats building one from a
    rate no trade can realise.
    """
    at = hex(block)
    addrs = list(pairs)
    if not addrs:
        return {}
    call = "0x" + encode_call("getReserves()").hex()
    got = rpc.fetch_multi(
        [("eth_call", [{"to": p, "data": call}, at]) for p in addrs],
        concurrent=True)
    out: dict[str, PairState] = {}
    for pool, answer in zip(addrs, got, strict=True):
        if not answer or not isinstance(answer, str):
            continue
        try:
            raw = bytes.fromhex(answer[2:])
        except ValueError:
            continue
        try:
            reserve0, reserve1 = decode_reserves(raw)
        except ValueError:
            continue
        row = pairs[pool]
        state = PairState(
            reserve0=reserve0, reserve1=reserve1,
            decimals0=int(row[3]), decimals1=int(row[4]),
            fee_bps=int(row[2]) if len(row) > 2 and row[2] else DEFAULT_FEE_BPS)
        if state.live:
            out[pool.lower()] = state
    return out
