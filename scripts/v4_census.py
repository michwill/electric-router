#!/usr/bin/env python3
"""Which Uniswap v4 pools this router will model, and what they hold.

Simpler than v2's or v3's in one way and harder in another.

Simpler: v4 keeps every pool in one singleton, so discovery is a log scan of a
single address rather than a factory walk.  One `Initialize` event names the
whole `PoolKey`.

Harder: most of what it finds is unroutable, and the filter that matters is not
depth.  A hook's permissions are the low fourteen bits of its own address, so
`venues.univ4.tier` sorts the census with no call and no state -- and only tiers
0 and 1 are kept.  Measured on ethereum at block 25,943,016, over all 132,609
pools ever initialised, by quoting a real $10,000 swap through `V4Quoter`:

    tier                        pools   quoted   usable <100bp   median
    0 no hook                 100,765   19,081             373    9,992 bp
    1 hook off the swap path    1,678    1,349             729       68 bp
    2 runs on swap              1,893      264               2    7,598 bp
    3 dynamic fee               1,753      701               9    6,762 bp
    3 returns delta            26,520   15,694              64    8,008 bp

Tier 1 is the best tier there is despite being 1.3% of the pools, and tier 3 is
refused for a stronger reason than being unmodellable: 1,413 of those quote
*better than spot*, the worst by 2.1e49 bp.

Depth is measured rather than inferred, for a reason worth recording.  v4 is a
singleton, so `balanceOf(pool)` -- which is how `v3_census.py` values a pool --
does not exist here.  Virtual reserves from `L` and `sqrtP` are the obvious
substitute and they are wrong: they describe a full-range position, so a
concentrated one reports depth no chain has, by a factor that varies with how
concentrated it is.  Two attempts at that produced $5.04e27 and a distribution
clipped at whatever cap it was given.  So this asks the pool instead, through
`V4Quoter`, and records what a $10,000 swap actually costs.

Usage:

    uv run python scripts/v4_census.py --chain ethereum
    uv run python scripts/v4_census.py --chain ethereum --floor 25000

Needs `PoolManager`, `StateView` and `V4Quoter` reachable: the committed
endpoint allowlists by address and 403s all three, so this wants `--private` or
those three whitelisted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import erouter  # noqa: F401,E402  -- pins BLAS before numpy
from erouter.chain import chains as chain_table  # noqa: E402
from erouter.core.codec import decode, encode_call  # noqa: E402
from erouter.core.keccak import keccak256  # noqa: E402
from erouter.dev import config  # noqa: E402
from erouter.dev.rpc import BATCH_FLOOR, JsonRpcTransport  # noqa: E402
from erouter.venues import univ4  # noqa: E402
from erouter.venues.univ4_chain import STATE_VIEW  # noqa: E402

#: The singleton, per chain.
POOL_MANAGER = {
    "ethereum": "0x000000000004444c5dc75cB358380D2e3dE08A90",
    "arbitrum": "0x360e68faccca8ca495c1b759fd9eee466db9fb32",
    "optimism": "0x9a13f98cb987694c9f086b1f5eb990eea8264ec3",
    "base": "0x498581ff718922c3f8e6a244956af099b2652b2b",
    "polygon": "0x67366782805870060151383f4bbff9dab53e5cd6",
    "bsc": "0x28e2ea090877bf75740558f6bfb36a5ffee9e9df",
    "avalanche": "0x06380c0e0912312b5150364b9dc4542ba0dbbc85",
}
#: Uniswap's own simulator.  Non-view -- it unlocks the manager and reverts its
#: way back -- so it goes through `eth_call`.
QUOTER = {
    "ethereum": "0x52f0e24d1c21c8a0cb1e5a5dd6198556bd9e1203",
    "arbitrum": "0x3972c00f7ed4885e145823eb7c655375d275a1c5",
    "optimism": "0x1f3131a13296fb91c90870043742c3cdbff1a8d7",
    "base": "0x0d5e0f971ed27fbff6c2837bf31316121532048d",
    "polygon": "0xb3d5c3dfc3a7aebff71895a7191796bffc2c81b9",
    "bsc": "0x9f75dd27d6664c475b90e105573e550ff69437b0",
    "avalanche": "0xbe40675bb704506a3c2ccfb762dcfd1e979845c2",
}
INITIALIZE = ("Initialize(bytes32,address,address,uint24,int24,address,"
              "uint160,int24)")
QSIG = ("quoteExactInputSingle(((address,address,uint24,int24,address),"
        "bool,uint128,bytes))")

NATIVE = "0x" + "00" * 20
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
STABLES = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
}
#: What the depth probe sends.  Big enough that a one-unit rounding is noise --
#: an early version used a thousandth of a token, which for a six-decimal one is
#: 1,000 raw units where ±1 is 10 bp, and measured its own rounding.
NOTIONAL_USD = 10_000.0


def discover(rpc, manager: str, head: int, deploy: int, span: int):
    """Every `PoolKey` ever initialised, and the windows that would not answer.

    Refused windows are re-asked halved, for the reason `v2_census.py` gives:
    a refusal is a silent hole, and it is not a random one.  Log density tracks
    launchpad activity, so the windows that refuse are the busy ones -- dropping
    them cost a third of an earlier run of this and moved the returns-delta
    share from 11.3% to 20.0%.
    """
    topic = "0x" + keccak256(INITIALIZE.encode()).hex()

    def ask(windows):
        got = rpc.fetch_multi([("eth_getLogs", [{
            "fromBlock": hex(a), "toBlock": hex(b),
            "address": manager, "topics": [topic]}]) for a, b in windows],
            concurrent=True)
        found, refused = {}, []
        for (a, b), raw in zip(windows, got, strict=True):
            if isinstance(raw, Exception) or not isinstance(raw, list):
                refused.append((a, b))
                continue
            for log in raw:
                d = log["data"][2:]
                w = [d[i:i + 64] for i in range(0, len(d), 64)]
                spacing = int(w[1], 16)
                found[log["topics"][1]] = [
                    "0x" + log["topics"][2][-40:], "0x" + log["topics"][3][-40:],
                    int(w[0], 16),
                    spacing - 2**24 if spacing >= 2**23 else spacing,
                    "0x" + w[2][-40:]]
        return found, refused

    spans = [(b, min(b + span - 1, head)) for b in range(deploy, head + 1, span)]
    pools, refused = {}, []
    started = time.monotonic()
    for start in range(0, len(spans), 24):
        found, no = ask(spans[start:start + 24])
        pools.update(found)
        refused += no
        print(f"  {min(start + 24, len(spans)):>5}/{len(spans)} windows  "
              f"{len(pools):,} pools  {time.monotonic() - started:,.0f}s"
              + (f"  ({len(no)} refused)" if no else ""), flush=True)
    for _round in range(6):
        if not refused:
            break
        halved = []
        for a, b in refused:
            mid = (a + b) // 2
            halved += [(a, mid), (mid + 1, b)] if mid > a else [(a, b)]
        print(f"  re-asking {len(refused):,} refused as {len(halved):,} narrower",
              flush=True)
        found, refused = ask(halved)
        pools.update(found)
        print(f"    {len(found):,} pool(s); {len(refused)} still refused", flush=True)
    return pools, refused


def priced_side(key: univ4.PoolKey, price: dict):
    """`(token, zero_for_one, decimals)` for a side worth sending, or `None`."""
    for token, zero in ((key.currency0, True), (key.currency1, False)):
        if token in price:
            return token, zero, price[token][0]
    return None


def measure(rpc, quoter: str, view: str, keys: dict, price: dict, block: int):
    """`poolId -> depth in dollars`, from what a $10,000 swap really costs.

    Asked rather than inferred.  A pool that reverts, or that quotes better than
    spot, is left out entirely -- the second is a hook paying the swapper, which
    is the shape that empties a relaxation into it.
    """
    at = hex(block)
    ids = list(keys)
    spot: dict[str, float] = {}
    for start in range(0, len(ids), 3000):
        part = ids[start:start + 3000]
        got = rpc.fetch_multi([("eth_call", [{"to": view, "data": "0x" + encode_call(
            "getSlot0(bytes32)", bytes.fromhex(p[2:])).hex()}, at])
            for p in part], concurrent=True)
        for pid, raw in zip(part, got, strict=True):
            if isinstance(raw, str) and len(raw) >= 66:
                sqrt_price = int(raw[2:66], 16)
                if sqrt_price:
                    spot[pid] = (sqrt_price / 2**96) ** 2
        print(f"  slot0 {min(start + 3000, len(ids)):>7,}/{len(ids):,}", flush=True)

    plans = []
    for pid, key in keys.items():
        if pid not in spot:
            continue
        side = priced_side(key, price)
        if side is None:
            continue
        _token, zero, decimals = side
        usd = price[_token][1]
        amount = int(NOTIONAL_USD / usd * 10 ** decimals)
        if 0 < amount < 2**128:
            plans.append((pid, zero, amount))

    depth: dict[str, float] = {}
    for start in range(0, len(plans), 1500):
        part = plans[start:start + 1500]
        got = rpc.fetch_multi([("eth_call", [{"to": quoter, "data": "0x" + encode_call(
            QSIG, (keys[pid].as_tuple(), zero, amount, b"")).hex()}, at])
            for pid, zero, amount in part], concurrent=True)
        for (pid, zero, amount), raw in zip(part, got, strict=True):
            if not isinstance(raw, str) or len(raw) < 130:
                continue
            out = decode(["uint256", "uint256"], bytes.fromhex(raw[2:]))[0]
            if out <= 0:
                continue
            want = spot[pid] if zero else 1.0 / spot[pid]
            if want <= 0:
                continue
            slip = 1 - (out / amount) / want
            if slip <= 0:
                # Better than spot.  A hook paying the swapper, or a price that
                # is not real; either way not a depth.
                continue
            if slip >= 1:
                depth[pid] = 0.0
                continue
            # `dx (1 - slip) / slip`, doubled for the far side, which is the
            # same "twice the quote asset" convention `v3_census.py` uses.
            #
            # Not `dx / slip`.  That is the small-slip limit of this, and it
            # cannot report less than the notional however shallow the pool is
            # -- so a first run of this put every pool that quoted at all above
            # $10,000 and kept 20,494 of them, which is 655,808 arcs.
            depth[pid] = 2 * NOTIONAL_USD * (1 - slip) / slip
        print(f"  quote {min(start + 1500, len(plans)):>7,}/{len(plans):,}", flush=True)
    return depth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--span", type=int, default=10_000)
    ap.add_argument("--floor", type=float, default=10_000.0,
                    help="keep pools whose measured depth is at least this")
    ap.add_argument("--deploy", type=int, default=0,
                    help="first block to scan; found by bisection when 0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rpc-url", default=None,
                    help="an endpoint to use instead of the chain's, for when "
                         "the committed one has not whitelisted v4's three "
                         "addresses.  Pass it, do not commit it")
    args = ap.parse_args()

    manager = POOL_MANAGER.get(args.chain)
    view, quoter = STATE_VIEW.get(args.chain), QUOTER.get(args.chain)
    if not (manager and view and quoter):
        raise SystemExit(f"no v4 deployment recorded for {args.chain}")

    chain = chain_table.CHAINS[args.chain]
    url = (args.rpc_url or (config.rpc_url(chain.rpc_attr) if args.private
                            else chain.public_rpc))
    rpc = JsonRpcTransport(url, chain_id=chain.chain_id)
    rpc.batch_size = max(rpc.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)
    head = rpc.block
    print(f"{chain.name} · block {head:,} · manager {manager} · "
          f"batch {rpc.batch_size}")

    deploy = args.deploy
    if not deploy:
        lo, hi = 1, head
        while lo < hi:
            mid = (lo + hi) // 2
            code = rpc.fetch("eth_getCode", [manager, hex(mid)])
            if len(code) > 2:
                hi = mid
            else:
                lo = mid + 1
        deploy = lo
        print(f"  manager first has code at block {deploy:,}")

    pools, refused = discover(rpc, manager, head, deploy, args.span)
    print(f"\n{len(pools):,} pool(s) ever initialised")
    if refused:
        print(f"  ! {len(refused)} window(s) still refused after narrowing -- "
              f"not writing a partial census.  First few: "
              f"{[f'{a}-{b}' for a, b in refused[:5]]}")
        return 1

    tiers = Counter()
    keys: dict[str, univ4.PoolKey] = {}
    for pid, row in pools.items():
        key = univ4.PoolKey.from_row(row)
        tiers[key.tier] += 1
        if key.routable:
            keys[pid] = key
    print(f"{'tier':<28}{'pools':>10}")
    for t, name in ((0, "0 no hook"), (1, "1 hook off the swap path"),
                    (2, "2 runs on swap"), (3, "3 returns delta or dynamic")):
        print(f"{name:<28}{tiers[t]:>10,}")
    print(f"\n{len(keys):,} routable (tiers 0 and 1)")

    price = {NATIVE: (18, 0.0), WETH: (18, 0.0), WBTC: (8, 0.0)}
    price.update({a: (d, 1.0) for a, d in STABLES.items()})
    weth, wbtc = _quote_assets(rpc, quoter, keys, price, head)
    price[NATIVE] = price[WETH] = (18, weth)
    price[WBTC] = (8, wbtc)
    print(f"  WETH ${weth:,.2f}   WBTC ${wbtc:,.2f}")

    print(f"\nmeasuring depth on {len(keys):,} pool(s)")
    depth = measure(rpc, quoter, view, keys, price, head)
    print(f"  {len(depth):,} quoted a real price")

    print(f"\n{'floor':>13}{'pools':>10}{'arcs':>9}")
    for step in (0, 1_000, 10_000, 100_000, 1_000_000):
        keep = [v for v in depth.values() if v >= step]
        # Two directions, up to `DEFAULT_TICKS` arcs each: v4 is far denser per
        # pool than v2, so the arc budget binds long before the pool count does.
        print(f"{'$' + format(step, ',') :>13}{len(keep):>10,}{len(keep) * 32:>9,}")

    out = {pid: [*pools[pid], round(depth[pid], 2)]
           for pid in keys if depth.get(pid, 0.0) >= args.floor}
    where = Path(args.out) if args.out else ROOT / "data" / "univ4" / f"{args.chain}.json"
    print(f"\n{len(out):,} pool(s) at or above ${args.floor:,.0f}")
    if not out:
        print(f"  ! nothing to write; {where} left alone")
        return 1
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True))
    print(f"  written to {where.relative_to(ROOT)} ({where.stat().st_size:,} B)")
    return 0


def _quote_assets(rpc, quoter: str, keys: dict, price: dict, block: int):
    """WETH and WBTC in dollars, from the deepest stable pool holding each."""
    at = hex(block)
    out = []
    for token, decimals in ((WETH, 18), (WBTC, 8)):
        found = 0.0
        for key in keys.values():
            pair = {key.currency0, key.currency1}
            if token not in pair and not (token is WETH and NATIVE in pair):
                continue
            stable = next((s for s in STABLES if s in pair), None)
            if stable is None:
                continue
            zero = key.currency0 in (token, NATIVE)
            amount = 10 ** decimals
            data = "0x" + encode_call(
                QSIG, (key.as_tuple(), zero, amount, b"")).hex()
            try:
                raw = rpc.fetch("eth_call", [{"to": quoter, "data": data}, at])
                got = decode(["uint256", "uint256"], bytes.fromhex(raw[2:]))[0]
            except Exception:
                continue
            if got > 0:
                found = got / 10 ** STABLES[stable]
                break
        out.append(found)
    return out[0], out[1]


if __name__ == "__main__":
    raise SystemExit(main())
