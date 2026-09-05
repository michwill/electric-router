#!/usr/bin/env python3
"""How much of Uniswap v3 a router could actually use.

Measured on ethereum at block 25,913,519:

    PoolCreated                     72,734
    liquidity() > 0                 42,645   58.6%
    distinct tokens, live pools     41,323   (the Curve node map holds 309)

**The filter that matters is liquidity, not token membership.**  TVL is
concentrated to the point of absurdity -- proxied as twice the quote asset each
pool holds, priced from v3 itself:

    floor        pools   cumulative TVL
    $10         15,934      993,000,000
    $1,000       2,893      991,000,000
    $10,000      1,208      985,000,000     99.2% of all v3 liquidity
    $100,000       384      958,000,000
    $1,000,000      87      868,000,000

So Curve's own $10k floor admits **1,208 v3 pools** and keeps 99.2% of the
depth; dropping to $10 multiplies the pool count thirteenfold for 0.8% more.

An earlier cut of this asked instead how many pools have *both* coins already in
the node map -- 327, or 0.8% -- and that is the answer to a different question.
It matters only if reference prices have to come from somewhere else, and they
do not: §4 fits log-prices by weighted least squares over the arcs that reach a
token, so any pool adjacent to the priced component prices its own far side.
95.8% of live v3 pools have exactly one coin already priced, which makes them
adjacent by construction.  Pricing is not the boundary; liquidity is.

The state cost is smaller than the pool count suggests.  Sampled at 1% of each
pool's own holdings, a quote touches 13 slots at the median and 41 at p90.  The
tail is real though: one pool wanted 3,481.

    uv run python scripts/v3_census.py [--sample 120]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.core.codec import encode_call
from erouter.core.keccak import keccak256
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, JsonRpcTransport

FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
DEPLOY = 12_369_621
QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
CALLER = "0x" + "11" * 20
TOPIC = "0x" + keccak256(b"PoolCreated(address,address,uint24,int24,address)").hex()
LIQUIDITY = "0x1a686502"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--span", type=int, default=10_000,
                   help="log window; endpoints commonly cap this at 10,000")
    p.add_argument("--sample", type=int, default=120,
                   help="usable pools to price the state cost from")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    chain = chain_table.CHAINS[args.chain]
    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), chain_id=chain.chain_id)
    head = rpc.block
    # The endpoint's ceiling, not the transport's default -- see `ac31895`.
    rpc.batch_size = max(rpc.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)

    spans = [(lo, min(lo + args.span - 1, head))
             for lo in range(DEPLOY, head + 1, args.span)]
    print(f"{chain.name} head {head:,}: {len(spans):,} log window(s)")

    pools: dict[str, tuple[str, str, int]] = {}
    refused = 0
    t0 = time.perf_counter()
    for start in range(0, len(spans), 400):
        batch = spans[start:start + 400]
        got = rpc.fetch_multi([("eth_getLogs", [{
            "fromBlock": hex(lo), "toBlock": hex(hi),
            "address": FACTORY, "topics": [TOPIC]}]) for lo, hi in batch],
            concurrent=True)
        for raw in got:
            if isinstance(raw, Exception):
                refused += 1
                continue
            for log in raw:
                pools["0x" + log["data"][2:][64:128][-40:]] = (
                    "0x" + log["topics"][1][-40:], "0x" + log["topics"][2][-40:],
                    int(log["topics"][3], 16))
        print(f"  {start + len(batch):>5}/{len(spans)}  {len(pools):,} pools",
              flush=True)

    # A refused window is a silently missing slice of the universe, and a short
    # census looks exactly like a small one.  Say so rather than tallying it.
    if refused:
        print(f"\n  ! {refused} window(s) refused -- census is INCOMPLETE")
    print(f"\nscanned in {time.perf_counter() - t0:,.0f} s")
    print(f"PoolCreated: {len(pools):,}")
    for fee, n in sorted(collections.Counter(f for _, _, f in pools.values()).items()):
        print(f"  fee {fee / 1e4:>5.2f}%   {n:>7,}")

    addrs = list(pools)
    live: dict[str, int] = {}
    for start in range(0, len(addrs), 4000):
        chunk = addrs[start:start + 4000]
        got = rpc.fetch_multi(
            [("eth_call", [{"to": a, "data": LIQUIDITY}, hex(head)]) for a in chunk],
            concurrent=True)
        for a, raw in zip(chunk, got, strict=True):
            if isinstance(raw, str) and len(raw) > 2 and int(raw, 16) > 0:
                live[a] = int(raw, 16)
    print(f"\nliquidity() > 0: {len(live):,} "
          f"({len(live) / max(len(pools), 1) * 100:.1f}%)")

    universe = UniverseCache().get(chain.chain_id, args.min_tvl, allow_stale=True)
    known = {c["address"].lower()
             for pool in (universe or []) for c in pool.get("coins", [])}
    both = [p for p in live
            if live and pools[p][0].lower() in known and pools[p][1].lower() in known]
    one = [p for p in live
           if (pools[p][0].lower() in known) ^ (pools[p][1].lower() in known)]
    tokens = {t.lower() for p in live for t in pools[p][:2]}
    print(f"\nnode map holds {len(known):,} token(s); live v3 pools hold "
          f"{len(tokens):,}")
    print(f"  both coins priced   {len(both):>7,}")
    print(f"  one coin priced     {len(one):>7,}")
    print(f"  neither             {len(live) - len(both) - len(one):>7,}")

    sizes = []
    for start in range(0, min(args.sample, len(both)), 200):
        chunk = both[start:start + 200]
        bal = rpc.fetch_multi([("eth_call", [{"to": pools[p][0], "data": "0x"
            + encode_call("balanceOf(address)", p).hex()}, hex(head)])
            for p in chunk], concurrent=True)
        ask = [(p, int(int(raw, 16) * 0.01)) for p, raw in zip(chunk, bal, strict=True)
               if isinstance(raw, str) and int(raw, 16) > 0]
        got = rpc.fetch_multi([("eth_createAccessList", [{
            "from": CALLER, "to": QUOTER, "data": "0x" + encode_call(
                "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
                (pools[p][0], pools[p][1], amount, pools[p][2], 0)).hex()},
            hex(head)]) for p, amount in ask], concurrent=True)
        for raw in got:
            if not isinstance(raw, Exception):
                sizes.append(sum(len(e.get("storageKeys") or [])
                                 for e in (raw.get("accessList") or [])))
    if sizes:
        sizes.sort()
        print(f"\nstate per usable pool, from {len(sizes)} sampled at 1% of holdings:")
        print(f"  median {statistics.median(sizes):>6,.0f}   "
              f"p90 {sizes[int(len(sizes) * 0.9)]:>6,}   max {max(sizes):>6,}")
        print(f"  all {len(both):,} pools: ~{statistics.median(sizes) * len(both):,.0f} "
              f"slots (curve today: 6,925)")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({p: [*pools[p], live[p]] for p in both}, handle)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
