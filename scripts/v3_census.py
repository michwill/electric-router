#!/usr/bin/env python3
"""How much of Uniswap v3 a router could use, and what holding it would cost.

Measured on ethereum at block 25,913,519.

**Census.**  72,734 pools created, 42,645 with liquidity, 41,323 distinct tokens
against the Curve node map's 309.  The filter that binds is liquidity, not token
membership, and it is brutal:

    floor        pools   cumulative TVL
    $10         15,934      993,000,000
    $1,000       2,893      991,000,000
    $10,000      1,208      985,000,000     99.2% of all v3 liquidity
    $100,000       384      958,000,000
    $1,000,000      87      868,000,000

Curve's own $10k floor admits 1,208 pools and keeps 99.2% of the depth; dropping
to $10 multiplies the count thirteenfold for 0.8% more.

An earlier cut asked how many pools have *both* coins already in the node map --
327 -- and that answers a different question.  It would matter only if reference
prices had to come from somewhere else, and they do not: §4 fits log-prices by
weighted least squares over the arcs that reach a token, and 95.8% of live v3
pools have one coin priced already, so they are adjacent to the priced component
by construction.

**Sizing.**  Everything below is msgpack then zstd, at the level the state
cache uses.  Per-pool figures are what scale; the totals depend on `--floor`.

    current cache (1,537,769 B)     bytecode 1,466,015    95.3%
                                    slot keys   71,566    10.3 B/slot

The two things that could have made v3 expensive both collapse.

    bytecode        60 runtimes, one length, 77-317 bytes apart
                    raw 1,328,520  apart 584,172  together 16,147   82x
                    marginal cost of one more pool: 109 B

    slot keys       32-byte hashes    2,152 B/pool
                    tick indices        169 B/pool    12.7x smaller

Slot keys are derivable -- a tick's four words are `keccak(tick, 5) + 0..3` --
so storing indices instead of hashes is the encoding to use.  The current format
has no such notion because Curve pools have no derivable slot families.

Values are the remaining term, and a third of them are not needed:
`prototype_univ3_fee_slots.py` shows on the local EVM that
`feeGrowthOutside0/1X128` never reach the price, so two words per tick suffice.

    values          all four words          7,286 B/pool
                    without feeGrowth       4,786 B/pool    34% off

    diff            1.0% of slots over 300 blocks (~1 h), 100 B/pool
                    -- a sample drawn by TVL skews quiet

At a $10k floor (1,208 pools) that is roughly 132 K of bytecode, 199 K of slot
keys and 5.6 MB of values, against a committed cache of 1.5 MB today.  The cache
grows about 20%; the values are what a server would serve, and the diff is what
it would actually send.

    uv run python scripts/v3_census.py [--arm census|sizing] [--pools out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import msgpack

from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.chain.statecache import StateCache, _zstd_compress
from erouter.core.codec import decode, encode_call
from erouter.core.keccak import keccak256
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, JsonRpcTransport

FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
DEPLOY = 12_369_621
QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
CALLER = "0x" + "11" * 20
CREATED = "0x" + keccak256(b"PoolCreated(address,address,uint24,int24,address)").hex()
LIQUIDITY = "0x1a686502"
TICKS_SLOT, BITMAP_SLOT = 5, 6

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDC_WETH = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
WBTC_WETH = "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD"
#: Priced at a dollar.  Only a bucket boundary rests on this.
STABLES = {
    USDC: 6, "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,
    "0xdc035d45d973e3ec169d2276ddab16f1e407384f": 18,
    "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e": 18,
    "0x853d955acef822db058eb8505911ed77f175b99e": 18,
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8": 6,
}


def word(v: int) -> bytes:
    return (v & (2**256 - 1)).to_bytes(32, "big")


def tick_key(tick: int) -> int:
    return int.from_bytes(keccak256(word(tick) + word(TICKS_SLOT)), "big")


def bitmap_key(pos: int) -> int:
    return int.from_bytes(keccak256(word(pos) + word(BITMAP_SLOT)), "big")


def packed(obj) -> tuple[int, int]:
    """msgpack, then zstd at the level the state cache uses."""
    blob = msgpack.packb(obj, use_bin_type=True)
    return len(blob), len(_zstd_compress(blob))


def connect(chain):
    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), chain_id=chain.chain_id)
    # The endpoint's ceiling, not the transport's default -- see `ac31895`.
    rpc.batch_size = max(rpc.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)
    return rpc


def discover(rpc, head: int, span: int) -> dict[str, tuple[str, str, int]]:
    """Every pool the factory ever made."""
    spans = [(lo, min(lo + span - 1, head)) for lo in range(DEPLOY, head + 1, span)]
    print(f"scanning {len(spans):,} log window(s) of {span:,} blocks")
    pools: dict[str, tuple[str, str, int]] = {}
    refused = 0
    t0 = time.perf_counter()
    for start in range(0, len(spans), 400):
        batch = spans[start:start + 400]
        got = rpc.fetch_multi([("eth_getLogs", [{
            "fromBlock": hex(lo), "toBlock": hex(hi),
            "address": FACTORY, "topics": [CREATED]}]) for lo, hi in batch],
            concurrent=True)
        for raw in got:
            if isinstance(raw, Exception):
                refused += 1
                continue
            for log in raw:
                pools["0x" + log["data"][2:][64:128][-40:]] = (
                    "0x" + log["topics"][1][-40:], "0x" + log["topics"][2][-40:],
                    int(log["topics"][3], 16))
        print(f"  {start + len(batch):>5}/{len(spans)}  {len(pools):,}", flush=True)
    # A refused window is a silently missing slice, and a short census looks
    # exactly like a small one.  Say so rather than tallying it.
    if refused:
        print(f"  ! {refused} window(s) refused -- census is INCOMPLETE")
    print(f"scanned in {time.perf_counter() - t0:,.0f} s: {len(pools):,} pools")
    return pools


def live_pools(rpc, head: int, pools) -> list[str]:
    out = []
    addrs = list(pools)
    for start in range(0, len(addrs), 4000):
        chunk = addrs[start:start + 4000]
        got = rpc.fetch_multi(
            [("eth_call", [{"to": a, "data": LIQUIDITY}, hex(head)]) for a in chunk],
            concurrent=True)
        out += [a for a, raw in zip(chunk, got, strict=True)
                if isinstance(raw, str) and len(raw) > 2 and int(raw, 16) > 0]
    return out


def quote_prices(rpc, head: int) -> dict[str, tuple[int, float]]:
    """WETH and WBTC off v3 itself, so nothing external is needed."""
    def sqrt_price(pool):
        raw = rpc.fetch("eth_call", [{"to": pool, "data": "0x" + encode_call(
            "slot0()").hex()}, hex(head)])
        return decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8",
                       "bool"], bytes.fromhex(raw[2:]))[0] / 2**96

    weth = 1.0 / (sqrt_price(USDC_WETH) ** 2 * 10 ** (6 - 18))
    wbtc = sqrt_price(WBTC_WETH) ** 2 * 10 ** (8 - 18) * weth
    print(f"WETH ${weth:,.0f}   WBTC ${wbtc:,.0f}")
    price = {WETH: (18, weth), WBTC: (8, wbtc)}
    price.update({a: (d, 1.0) for a, d in STABLES.items()})
    return price


def value_of(rpc, head: int, pools, live, price) -> dict[str, float]:
    """TVL as twice the quote asset held: the usual proxy, and enough to bucket."""
    ask = []
    for pool in live:
        for token in (pools[pool][0].lower(), pools[pool][1].lower()):
            if token in price:
                ask.append((pool, token))
                break
    out: dict[str, float] = {}
    for start in range(0, len(ask), 4000):
        chunk = ask[start:start + 4000]
        got = rpc.fetch_multi([("eth_call", [{"to": token, "data": "0x" + encode_call(
            "balanceOf(address)", pool).hex()}, hex(head)])
            for pool, token in chunk], concurrent=True)
        for (pool, token), raw in zip(chunk, got, strict=True):
            if isinstance(raw, str) and len(raw) > 2:
                dec, usd = price[token]
                out[pool] = 2 * int(raw, 16) / 10**dec * usd
    return out


def tick_state(rpc, head: int, pools: list[str], want: int, words: int):
    """`pool -> (bitmap words, nearest initialized ticks)`, both sides of spot.

    Kept apart from the slot keys they imply: a tick index is a small signed
    number and a slot key is a 256-bit hash, and telling them apart afterwards
    is how a negative tick gets asked for as a slot.
    """
    meta = rpc.fetch_multi(
        [("eth_call", [{"to": p, "data": "0x" + encode_call("slot0()").hex()},
                       hex(head)]) for p in pools]
        + [("eth_call", [{"to": p, "data": "0x" + encode_call("tickSpacing()").hex()},
                         hex(head)]) for p in pools], concurrent=True)
    spots = {}
    for k, pool in enumerate(pools):
        if isinstance(meta[k], Exception) or isinstance(meta[k + len(pools)], Exception):
            continue
        spots[pool] = (
            decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
                   bytes.fromhex(meta[k][2:]))[1],
            decode(["int24"], bytes.fromhex(meta[k + len(pools)][2:]))[0])

    asks, index = [], []
    for pool, (tick, spacing) in spots.items():
        centre = (tick // spacing) >> 8
        for pos in range(centre - words, centre + words + 1):
            asks.append(("eth_getStorageAt", [pool, hex(bitmap_key(pos)), hex(head)]))
            index.append((pool, pos, spacing))
    got = rpc.fetch_multi(asks, concurrent=True)

    held: dict[str, list[int]] = {}
    found: dict[str, list[int]] = {}
    for (pool, pos, spacing), raw in zip(index, got, strict=True):
        if not isinstance(raw, str):
            continue
        held.setdefault(pool, []).append(pos)
        found.setdefault(pool, [])
        bits = int(raw, 16)
        found[pool] += [((pos << 8) + b) * spacing for b in range(256) if bits >> b & 1]

    out = {}
    for pool, ticks in found.items():
        tick, _ = spots[pool]
        below = sorted((t for t in ticks if t <= tick), reverse=True)[:want]
        above = sorted(t for t in ticks if t > tick)[:want]
        out[pool] = (held[pool], sorted(below + above))
    return out


def census(rpc, chain, args) -> dict[str, float]:
    head = rpc.block
    pools = discover(rpc, head, args.span)
    for fee, n in sorted(collections.Counter(f for _, _, f in pools.values()).items()):
        print(f"  fee {fee / 1e4:>5.2f}%   {n:>7,}")

    live = live_pools(rpc, head, pools)
    print(f"\nliquidity() > 0: {len(live):,} "
          f"({len(live) / max(len(pools), 1) * 100:.1f}%)")

    tvl = value_of(rpc, head, pools, live, quote_prices(rpc, head))
    print(f"\npriced {len(tvl):,} of {len(live):,} live pools "
          f"({len(tvl) / max(len(live), 1) * 100:.1f}% hold a quote asset)\n")
    print(f"{'floor':>13}{'pools':>10}{'cumulative TVL':>20}")
    for floor in (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        keep = [v for v in tvl.values() if v >= floor]
        print(f"${floor:>12,}{len(keep):>10,}{sum(keep):>19,.0f}")

    universe = UniverseCache().get(chain.chain_id, args.min_tvl, allow_stale=True)
    known = {c["address"].lower()
             for pool in (universe or []) for c in pool.get("coins", [])}
    tokens = {t.lower() for p in live for t in pools[p][:2]}
    both = sum(1 for p in live
               if pools[p][0].lower() in known and pools[p][1].lower() in known)
    one = sum(1 for p in live
              if (pools[p][0].lower() in known) ^ (pools[p][1].lower() in known))
    print(f"\nnode map {len(known):,} token(s); live v3 pools {len(tokens):,}")
    print(f"  both coins priced {both:>7,}   one {one:>7,}   "
          f"neither {len(live) - both - one:>7,}")
    print("  (membership is not the filter -- §4 prices the far side of any pool"
          " adjacent to the frame)")

    if args.pools:
        keep = {p: v for p, v in tvl.items() if v >= args.floor}
        with open(args.pools, "w") as handle:
            json.dump({p: [*pools[p], keep[p]] for p in keep}, handle)
        print(f"\nwrote {len(keep):,} pool(s) above ${args.floor:,.0f} "
              f"to {args.pools}")
    return tvl


def sizing(rpc, chain, args) -> None:
    head = rpc.block
    blob = Path(__file__).resolve().parents[1] / "data" / "evm-state" / \
        f"{chain.name.lower()}.msgpack"
    cache = StateCache.from_bytes(chain.chain_id, blob.read_bytes())
    slots = cache.slots()
    n_slots = sum(len(v) for v in slots.values())
    code = {k: bytes.fromhex(v) if isinstance(v, str) else v
            for k, v in cache.code.items()}
    keys = {a.encode(): [int(s).to_bytes(32, "big") for s in sorted(sl)]
            for a, sl in slots.items()}

    print(f"=== the cache as it stands ({blob.stat().st_size:,} B on disk)")
    print(f"{'part':<24}{'msgpack':>12}{'+zstd':>11}{'per slot':>11}")
    for name, obj, n in (("bytecode", code, 0), ("slot keys", keys, n_slots)):
        raw, comp = packed(obj)
        each = f"{comp / n:>10.1f}B" if n else ""
        print(f"{name:<24}{raw:>12,}{comp:>11,}{each}")
    print(f"bytecode is {packed(code)[1] / blob.stat().st_size * 100:.1f}% of it\n")

    if args.pools and Path(args.pools).exists():
        with open(args.pools) as handle:
            chosen = list(json.load(handle))
    else:
        pools = discover(rpc, head, args.span)
        live = live_pools(rpc, head, pools)
        tvl = value_of(rpc, head, pools, live, quote_prices(rpc, head))
        chosen = [p for p, v in tvl.items() if v >= args.floor]
    target = len(chosen)
    sample = chosen[:args.sample]
    print(f"=== v3, {target:,} pool(s) above ${args.floor:,.0f}, "
          f"{len(sample)} sampled\n")

    got = rpc.fetch_multi([("eth_getCode", [p, hex(head)]) for p in sample],
                          concurrent=True)
    runtimes = {p: bytes.fromhex(raw[2:]) for p, raw in zip(sample, got, strict=True)
                if isinstance(raw, str) and len(raw) > 2}
    first = next(iter(runtimes.values()))
    apart = sum(packed({p.encode(): v})[1] for p, v in runtimes.items())
    together = packed({p.encode(): v for p, v in runtimes.items()})[1]
    one = packed({next(iter(runtimes)).encode(): first})[1]
    marginal = (together - one) / max(len(runtimes) - 1, 1)
    print(f"bytecode: {len(runtimes)} runtime(s), "
          f"{len({len(v) for v in runtimes.values()})} distinct length(s)")
    print(f"  raw {sum(len(v) for v in runtimes.values()):>12,}   apart {apart:>10,}"
          f"   together {together:>8,}   "
          f"{sum(len(v) for v in runtimes.values()) / together:>4,.0f}x")
    print(f"  marginal per pool {marginal:>7,.0f} B"
          f"   x{target:,} -> {(one + marginal * (target - 1)) / 1024:>8,.0f} K\n")

    state = tick_state(rpc, head, sample, args.ticks, args.words)
    n_ticks = sum(len(t) for _w, t in state.values())
    hashes, derived, flat = {}, {}, []
    for pool, (words_, ticks) in state.items():
        keyset = ([0, 1, 2, 4] + [bitmap_key(p) for p in words_]
                  + [tick_key(t) + off for t in ticks for off in range(4)])
        hashes[pool.encode()] = sorted(word(k) for k in keyset)
        derived[pool.encode()] = [words_, ticks]
        flat += [(pool, k) for k in sorted(set(keyset))]
    print(f"slots: {n_ticks:,} initialized tick(s) kept, {len(flat):,} slot(s) "
          f"({len(flat) / max(len(state), 1):,.0f}/pool)")
    print(f"{'encoding':<24}{'msgpack':>12}{'+zstd':>11}{'per pool':>11}"
          f"{'x' + format(target, ',') :>12}")
    for name, obj in (("32-byte slot keys", hashes), ("tick indices", derived)):
        raw, comp = packed(obj)
        print(f"{name:<24}{raw:>12,}{comp:>11,}{comp / len(state):>10,.0f}B"
              f"{comp / len(state) * target / 1024:>11,.0f}K")

    def read(at: str):
        out = {}
        for start in range(0, len(flat), 3000):
            chunk = flat[start:start + 3000]
            got = rpc.fetch_multi([("eth_getStorageAt", [p, hex(k), at])
                                   for p, k in chunk], concurrent=True)
            for (p, k), raw in zip(chunk, got, strict=True):
                if isinstance(raw, str):
                    out[(p, k)] = bytes.fromhex(raw[2:])
        return out

    now, before = read(hex(head)), read(hex(head - args.blocks))
    # `feeGrowthOutside` never reaches the price -- proved on the local EVM by
    # `prototype_univ3_fee_slots.py` -- so two of a tick's four words can go.
    fee_words = {tick_key(t) + off
                 for _w, ticks in state.values() for t in ticks for off in (1, 2)}
    lean = {k: v for k, v in now.items() if k[1] not in fee_words}
    moved = [k for k in now if before.get(k) != now[k]]

    print(f"\n{'values':<24}{'msgpack':>12}{'+zstd':>11}{'per pool':>11}"
          f"{'x' + format(target, ',') :>12}")
    for name, obj in (("all four words", now), ("without feeGrowthOutside", lean)):
        rows = [[a.encode(), word(k), v] for (a, k), v in obj.items()]
        raw, comp = packed(rows)
        print(f"{name:<24}{raw:>12,}{comp:>11,}{comp / len(state):>10,.0f}B"
              f"{comp / len(state) * target / 1024:>11,.0f}K")

    rows = [[a.encode(), word(k), now[(a, k)]] for a, k in moved]
    print(f"\n{args.blocks} blocks: {len(moved):,} of {len(now):,} slot(s) moved "
          f"({len(moved) / max(len(now), 1) * 100:.1f}%)")
    print(f"  diff {packed(rows)[1]:,} B for {len(state)} pool(s)"
          f"  ->  {packed(rows)[1] / len(state) * target / 1024:,.0f} K for {target:,}")
    print("  (a sample drawn by TVL skews quiet; an active pool moves slot0 "
          "every swap)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--arm", default="census", choices=("census", "sizing"))
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--floor", type=float, default=10_000.0,
                   help="TVL floor for the pools sizing is measured over")
    p.add_argument("--span", type=int, default=10_000,
                   help="log window; endpoints commonly cap this at 10,000")
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--ticks", type=int, default=64, help="initialized ticks a side")
    p.add_argument("--words", type=int, default=3, help="bitmap words a side")
    p.add_argument("--blocks", type=int, default=300, help="diff window")
    p.add_argument("--pools", default="", help="write, or read, the pool list")
    args = p.parse_args(argv)

    chain = chain_table.CHAINS[args.chain]
    rpc = connect(chain)
    print(f"{chain.name} head {rpc.block:,}\n")
    (census if args.arm == "census" else sizing)(rpc, chain, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
