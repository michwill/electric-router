#!/usr/bin/env python3
"""Does a v3 quote depend on `feeGrowthOutside`?

Two of the four words a tick occupies are `feeGrowthOutside0X128` and
`feeGrowthOutside1X128`.  `Tick.cross` reads and rewrites them, but the swap's
output is a function of `liquidityNet` alone -- so a cache that omits them
should quote identically, and omitting them halves the tick payload.

"Should" is the word that has cost this project three bugs in a week, so this
asks the EVM.  A slot the local EVM does not hold reads as zero, which is what
absence *is*, so zeroing them here is the same experiment as never fetching
them -- and the local EVM is where it matters, because that is what quotes.

    uv run python scripts/prototype_univ3_fee_slots.py [--pool 0x..]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erouter.chain import chains as chain_table
from erouter.chain.localevm import LocalEvm
from erouter.core.codec import decode, encode_call
from erouter.core.keccak import keccak256
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, JsonRpcTransport

QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
DEFAULT_POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"   # USDC/WETH 0.05%
TICKS_SLOT, BITMAP_SLOT = 5, 6


def word(v: int) -> bytes:
    return (v & (2**256 - 1)).to_bytes(32, "big")


def tick_key(tick: int) -> int:
    return int.from_bytes(keccak256(word(tick) + word(TICKS_SLOT)), "big")


class _Rpc:
    """`fill` wants the async shape; this command has the blocking one."""

    def __init__(self, transport) -> None:
        self._t = transport
        self.chain_id = transport.chain_id
        self.batch_size = transport.batch_size
        self.max_streams = transport.max_streams

    async def batch(self, requests):
        return self._t.fetch_multi(list(requests), concurrent=True)

    async def call(self, method, params):
        got = self._t.fetch_multi([(method, params)])[0]
        if isinstance(got, Exception):
            raise got
        return got


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--pool", default=DEFAULT_POOL)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--sizes", default="1000,100000,1000000,10000000")
    args = p.parse_args(argv)

    import erouter_evm

    chain = chain_table.CHAINS[args.chain]
    transport = JsonRpcTransport(config.rpc_url(chain.rpc_attr),
                                 chain_id=chain.chain_id)
    transport.batch_size = max(
        transport.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)
    block = args.block or transport.block
    rpc = _Rpc(transport)

    def onchain(to, data):
        return bytes.fromhex(transport.fetch(
            "eth_call", [{"to": to, "data": "0x" + data.hex()}, hex(block)])[2:])

    token0 = "0x" + onchain(args.pool, encode_call("token0()")).hex()[-40:]
    token1 = "0x" + onchain(args.pool, encode_call("token1()")).hex()[-40:]
    fee = decode(["uint24"], onchain(args.pool, encode_call("fee()")))[0]
    dec0 = decode(["uint8"], onchain(token0, encode_call("decimals()")))[0]

    evm = LocalEvm(erouter_evm.Evm("Osaka", chain.chain_id), chain.chain_id, block)
    sizes = [int(float(v) * 10**dec0) for v in args.sizes.split(",")]

    def quote_all():
        out = []
        for amount in sizes:
            data = encode_call(
                "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
                (token0, token1, amount, fee, 0))
            try:
                got = evm.call(QUOTER_V2, data)
                out.append(decode(["uint256", "uint160", "uint32", "uint256"],
                                  got)[0] if len(got) >= 128 else None)
            except Exception:
                out.append(None)
        return out

    warm = asyncio.run(evm.fill(rpc, quote_all, block=hex(block)))
    if any(v is None for v in warm):
        print(f"  ! the warm did not answer every size: {warm}")
    print(f"pool {args.pool}  block {block:,}")
    print(f"warm: {evm.stats.rounds} round(s), {evm.stats.fetched} account(s), "
          f"{sum(len(v) for v in evm.learned.values()):,} slot(s), "
          f"unreadable {evm.stats.unreadable}")

    # Which of the learned slots are feeGrowthOutside: for every tick the walk
    # could have touched, words +1 and +2 of its entry.
    pool_slots = evm.learned.get(args.pool.lower(), set())
    spacing = decode(["int24"], onchain(args.pool, encode_call("tickSpacing()")))[0]
    tick_now = decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8",
                       "bool"], onchain(args.pool, encode_call("slot0()")))[1]
    base = (tick_now // spacing) * spacing
    fee_slots = set()
    for step in range(-4096, 4097):
        key = tick_key(base + step * spacing)
        fee_slots |= {key + 1, key + 2}
    target = sorted(pool_slots & fee_slots)
    print(f"of {len(pool_slots):,} pool slot(s) read, {len(target):,} are "
          f"feeGrowthOutside\n")

    if not target:
        print("nothing to zero: the walk crossed no initialized tick at these sizes")
        return 0

    evm.apply_storage([(args.pool, s, 0) for s in target])
    after = quote_all()

    print(f"{'trade in':>18}{'with fee slots':>26}{'zeroed':>26}{'same':>7}")
    same = True
    for amount, before, now in zip(sizes, warm, after, strict=True):
        ok = before == now
        same &= ok
        print(f"{amount / 10**dec0:>18,.0f}{before!s:>26}{now!s:>26}"
              f"{'yes' if ok else 'NO':>7}")
    print(f"\n{'identical' if same else 'THE QUOTE MOVED'} across "
          f"{len(sizes)} size(s)")

    # The control, and the test is void without it.  "Zeroing changed nothing"
    # and "zeroing did nothing" print the same, so zero a word the swap must
    # depend on -- `liquidityNet`, at offset 0 of the same tick entries -- and
    # require that the quote moves.
    net_slots = {tick_key(base + step * spacing) for step in range(-4096, 4097)}
    control = sorted(pool_slots & net_slots)
    evm.apply_storage([(args.pool, s, 0) for s in control])
    moved = quote_all()
    changed = sum(1 for a, b in zip(after, moved, strict=True) if a != b)
    print(f"control: zeroing {len(control):,} liquidityNet word(s) moved "
          f"{changed} of {len(sizes)} quote(s)")
    for amount, before, now in zip(sizes, after, moved, strict=True):
        if before != now:
            print(f"  {amount / 10**dec0:>14,.0f}  {before} -> {now}")
    if not changed:
        print("  ! the control did not move: apply_storage is not taking effect,"
              " so the result above proves nothing")
        return 2
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
