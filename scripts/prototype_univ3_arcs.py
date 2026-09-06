#!/usr/bin/env python3
"""Model a real Uniswap v3 pool as parallel tick-arcs, and diff it against the
pool's own quoter.

`erouter.venues.univ3` builds one diode+resistor+cap per tick range straight
off the pool's state.  Nothing is probed and nothing is fitted, so the only
question this asks is whether the arithmetic is right -- and `QuoterV2` is the
adjudicator, at the same block.

    uv run python scripts/prototype_univ3_arcs.py [--pool 0x..] [--fee 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erouter.chain import chains as chain_table
from erouter.core.codec import decode, encode_call
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, JsonRpcTransport
from erouter.venues.univ3 import PoolState, Tick, arcs, capacity, output

QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
#: USDC/WETH 0.05%, the deepest v3 pool on mainnet.
DEFAULT_POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--pool", default=DEFAULT_POOL)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--window", type=int, default=8,
                   help="tick-bitmap words to scan either side of spot")
    p.add_argument("--max-ticks", type=int, default=64)
    p.add_argument("--private", action="store_true", default=True)
    args = p.parse_args(argv)

    chain = chain_table.CHAINS[args.chain]
    url = config.rpc_url(chain.rpc_attr) if args.private else chain.public_rpc
    rpc = JsonRpcTransport(url, chain_id=chain.chain_id)
    block = hex(args.block or rpc.block)
    pool = args.pool
    # The endpoint's own ceiling rather than the transport's 500 default:
    # `fetch_multi` chunks at `batch_size` and Erigon refuses an oversized
    # batch whole, which reads here as every tick being unreadable.  The same
    # reconciliation `dev/rpc.AsyncTransport` does for the session.
    rpc.batch_size = max(
        rpc.probe_batch_limit(("eth_getStorageAt", [pool, "0x" + "00" * 32, block])),
        BATCH_FLOOR)

    def call(to: str, data: bytes) -> bytes:
        got = rpc.fetch("eth_call", [{"to": to, "data": "0x" + data.hex()}, block])
        return bytes.fromhex(got[2:])

    slot0 = decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
                   call(pool, encode_call("slot0()")))
    liquidity = int.from_bytes(call(pool, encode_call("liquidity()")), "big")
    spacing = decode(["int24"], call(pool, encode_call("tickSpacing()")))[0]
    fee = decode(["uint24"], call(pool, encode_call("fee()")))[0]
    token0 = "0x" + call(pool, encode_call("token0()")).hex()[-40:]
    token1 = "0x" + call(pool, encode_call("token1()")).hex()[-40:]
    dec0 = decode(["uint8"], call(token0, encode_call("decimals()")))[0]
    dec1 = decode(["uint8"], call(token1, encode_call("decimals()")))[0]

    state = PoolState(sqrt_price_x96=slot0[0], tick=slot0[1], liquidity=liquidity,
                      tick_spacing=spacing, fee=fee, decimals0=dec0, decimals1=dec1)

    # Initialized ticks, from the bitmap: one word covers 256 tick-slots, so a
    # handful of words either side of spot is the whole plausible range and a
    # few round trips rather than thousands.
    compressed = state.tick // spacing
    word = compressed >> 8
    words = list(range(word - args.window, word + args.window + 1))
    got = rpc.fetch_multi(
        [("eth_call", [{"to": pool, "data": "0x" + encode_call(
            "tickBitmap(int16)", w).hex()}, block]) for w in words],
        concurrent=True)
    initialized: list[int] = []
    refused = 0
    for w, raw in zip(words, got, strict=True):
        if isinstance(raw, Exception):
            refused += 1
            continue
        bits = int(raw, 16)
        for bit in range(256):
            if bits >> bit & 1:
                initialized.append(((w << 8) + bit) * spacing)
    if refused:
        print(f"  ! {refused} bitmap word(s) unreadable")

    # Only the ticks the walk can reach.  A deep pool has thousands
    # initialized within the window and asking for all of them is both slow and
    # pointless: the bank is truncated at `max_ticks` a side anyway.
    reach = args.max_ticks + 2
    below = sorted((t for t in initialized if t <= state.tick), reverse=True)[:reach]
    above = sorted(t for t in initialized if t > state.tick)[:reach]
    wanted = sorted(set(below) | set(above))

    got = rpc.fetch_multi(
        [("eth_call", [{"to": pool, "data": "0x" + encode_call(
            "ticks(int24)", t).hex()}, block]) for t in wanted],
        concurrent=True)
    ticks: list[Tick] = []
    failed: list[str] = []
    for index, raw in zip(wanted, got, strict=True):
        if isinstance(raw, Exception):
            failed.append(f"tick {index}: {str(raw)[:50]}")
            continue
        net = decode(["uint128", "int128"], bytes.fromhex(raw[2:])[:64])[1]
        ticks.append(Tick(index=index, liquidity_net=net))
    if failed:
        print(f"  ! {len(failed)} tick read(s) refused, first: {failed[0]}")

    print(f"pool    {pool}   block {int(block, 16):,}")
    print(f"tokens  {token0} ({dec0}) / {token1} ({dec1})")
    print(f"state   tick {state.tick:,}  spacing {spacing}  fee {fee/1e4:.2f}%  "
          f"L {liquidity:,}")
    print(f"ticks   {len(ticks)} read of {len(initialized):,} initialized "
          f"in +/-{args.window} bitmap word(s)\n")

    for zero_for_one in (True, False):
        bank = arcs(state, ticks, zero_for_one=zero_for_one,
                    max_ticks=args.max_ticks, linear=args.linear)
        if not bank:
            print("no arcs in this direction")
            continue
        tin, tout = (token0, token1) if zero_for_one else (token1, token0)
        din = dec0 if zero_for_one else dec1
        room = capacity(bank)
        print(f"{'token0 -> token1' if zero_for_one else 'token1 -> token0'}: "
              f"{len(bank)} arc(s), capacity {room:,.2f}")
        print(f"{'trade in':>16}{'quoter out':>20}{'model out':>20}{'bp':>10}{'arcs':>6}")
        for frac in (1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.9):
            dx = room * frac
            units = int(dx * 10**din)
            if units <= 0:
                continue
            data = encode_call(
                "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
                (tin, tout, units, fee, 0))
            try:
                truth = decode(["uint256", "uint160", "uint32", "uint256"],
                               call(QUOTER_V2, data))[0]
            except Exception as exc:
                print(f"{dx:>16,.2f}{'quoter refused':>20}   {str(exc)[:34]}")
                continue
            dout = dec1 if zero_for_one else dec0
            true_out = truth / 10**dout
            model = output(bank, dx)
            bp = (model - true_out) / true_out * 1e4 if true_out else float("nan")
            live = sum(1 for a in bank if a.cap > 0 and a.a > 0)
            print(f"{dx:>16,.2f}{true_out:>20,.6f}{model:>20,.6f}{bp:>+10.3f}{live:>6}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
