#!/usr/bin/env python3
"""Which Uniswap v2 pairs a router could use, written where the venue reads it.

The shape of the problem is the opposite of v3's.  There, 72,734 pools were
discovered and the filter that bound was liquidity.  Here the factory has made
hundreds of thousands of pairs, almost all of them empty forever, and reading
`getReserves()` for each is the expensive step -- so the census filters *before*
it reads rather than after.

**The filter is a quote asset on one side.**  A pair with no priced token cannot
be valued without pricing its other side first, and a token whose only market is
one dead pair is not one the router can reach anyway.  §4 fits log-prices over
the arcs that reach a token, so a pair adjacent to the priced component is what
is useful and the rest is noise -- the same argument `v3_census.py` makes about
its 95.8%.

**TVL is exact here rather than estimated.**  Both sides of a constant product
are equal in value by construction, so one priced side doubled *is* the pool's
value, with no second price to fetch and no assumption to record.

Written to `data/univ2/<chain>.json` as `pair -> [token0, token1, fee_bps,
tvl_usd]`, which is what `venues.univ2_chain.read_pairs` takes a row of.

    uv run python scripts/v2_census.py --private
    uv run python scripts/v2_census.py --private --floor 50000 --out data/univ2/ethereum.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import erouter  # noqa: F401,E402  -- pins BLAS before numpy
from erouter.chain import chains as chain_table  # noqa: E402
from erouter.core.codec import encode_call  # noqa: E402
from erouter.core.keccak import keccak256  # noqa: E402
from erouter.dev import config  # noqa: E402
from erouter.dev.rpc import JsonRpcTransport  # noqa: E402
from erouter.venues.univ2 import DEFAULT_FEE_BPS  # noqa: E402
from erouter.venues.univ2_chain import decode_reserves  # noqa: E402

#: Uniswap v2 on ethereum, and the block it was deployed at.
#:
#: **Every Uniswap v2 pair charges 30 bp.**  It is hard-coded in the pair as
#: `997/1000` with no getter and no tiers -- the protocol-fee switch takes a
#: share of that 30 bp rather than changing it.  So `--fee-bps` exists for
#: *forks*: the same event signature and bytecode deployed at 25 bp reads as a
#: Uniswap pair to every call this makes, and a census that assumed otherwise
#: would misprice every leg of it in one direction.
FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
DEPLOY = 10_000_835
CREATED = "0x" + keccak256(b"PairCreated(address,address,address,uint256)").hex()

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
#: Priced at a dollar.  Only a floor boundary rests on this, and a stable that
#: has actually broken is a pool nobody should be routing through anyway.
STABLES = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
}


def discover(rpc, head: int, span: int, factory: str, deploy: int,
             cache: Path | None = None) -> tuple[dict, int]:
    """Every pair the factory ever made, as `pair -> (token0, token1)`.

    A refused window is a silently missing slice and a short census looks
    exactly like a small one, so refusals are counted and said aloud rather
    than tallied into the total.

    **Resumable.**  Enumerating 1,594 windows takes over an hour against a
    rate-limiting endpoint, and the first two attempts here died -- one to a
    100,000-block span the node would not serve, one to a timeout at window
    1,272 of 1,594 with 281,167 pairs already found and nothing written.  So
    progress is written as it goes and a restart picks up from the last window
    finished.  The cache is keyed by `(factory, span)`: a different scan is a
    different file, not a resume.
    """
    spans = [(lo, min(lo + span - 1, head))
             for lo in range(deploy, head + 1, span)]
    pairs: dict[str, tuple[str, str]] = {}
    refused = 0
    done = 0
    if cache is not None and cache.exists():
        held = json.loads(cache.read_text())
        if held.get("factory") == factory and held.get("span") == span:
            pairs = {k: tuple(v) for k, v in held["pairs"].items()}
            done = int(held.get("done", 0))
            refused = int(held.get("refused", 0))
            print(f"  resuming at window {done}/{len(spans)} with "
                  f"{len(pairs):,} pair(s)")
    started = time.monotonic()
    for start in range(done, len(spans), 24):
        window = spans[start:start + 24]
        got = rpc.fetch_multi([("eth_getLogs", [{
            "fromBlock": hex(lo), "toBlock": hex(hi),
            "address": factory, "topics": [CREATED]}]) for lo, hi in window],
            concurrent=True)
        for raw in got:
            if isinstance(raw, Exception) or not isinstance(raw, list):
                refused += 1
                continue
            for log in raw:
                # `data` is (address pair, uint256 allPairsLength); the pair is
                # the first word, and the tokens are indexed.
                pairs["0x" + log["data"][2:][24:64]] = (
                    "0x" + log["topics"][1][-40:], "0x" + log["topics"][2][-40:])
        print(f"  {min(start + 24, len(spans)):>5}/{len(spans)} windows  "
              f"{len(pairs):,} pairs  {time.monotonic() - started:,.0f}s",
              flush=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "factory": factory, "span": span,
                "done": min(start + 24, len(spans)), "refused": refused,
                "pairs": {k: list(v) for k, v in pairs.items()}}))
    return pairs, refused


def quoted_side(token0: str, token1: str):
    """`(index, decimals, price)` of a priced side, or `None`.

    One side is enough: a constant product holds equal value each way.
    """
    for index, token in enumerate((token0, token1)):
        if token in STABLES:
            return index, STABLES[token], 1.0
        if token == WETH:
            return index, 18, None      # priced below, once
        if token == WBTC:
            return index, 8, None
    return None


def decimals_for(rpc, tokens, head: int) -> dict:
    call = "0x" + encode_call("decimals()").hex()
    got = rpc.fetch_multi(
        [("eth_call", [{"to": t, "data": call}, hex(head)]) for t in tokens],
        concurrent=True)
    out = {}
    for token, answer in zip(tokens, got, strict=True):
        try:
            out[token] = int(answer, 16)
        except (TypeError, ValueError):
            continue
    return out


def read_reserves(rpc, addrs, head: int, chunk: int = 2000) -> dict:
    call = "0x" + encode_call("getReserves()").hex()
    at = hex(head)
    out = {}
    started = time.monotonic()
    for start in range(0, len(addrs), chunk):
        part = addrs[start:start + chunk]
        got = rpc.fetch_multi(
            [("eth_call", [{"to": p, "data": call}, at]) for p in part],
            concurrent=True)
        for pool, answer in zip(part, got, strict=True):
            if isinstance(answer, Exception) or not isinstance(answer, str):
                continue
            try:
                out[pool] = decode_reserves(bytes.fromhex(answer[2:]))
            except ValueError:
                continue
        print(f"  {min(start + chunk, len(addrs)):>7,}/{len(addrs):,} read  "
              f"{time.monotonic() - started:,.0f}s", flush=True)
    return out


def emit(valued: dict, meta: dict, args) -> int:
    """The floor table, the cut, and the write.

    `valued` is every priced pair's TVL, floor or no floor, because the floor
    is the parameter that decides whether this venue helps the solve or wrecks
    it and it should be chosen off a table rather than guessed.  Most of the
    factory is pairs seeded once and abandoned, so **arc count is a cliff
    rather than a cost**: measured elsewhere in this router, a graph taken from
    450 arcs to 7,486 stopped converging at all and every large loss sat on a
    `PARTIAL` solve.
    """
    def shown(path: Path) -> str:
        """`--out` is allowed to point anywhere, so this cannot assume it does
        not.  It used to throw after the file was written, which made a good
        run exit non-zero with the census sitting on disk."""
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    print(f"\n{'floor':>13}{'pairs':>10}{'arcs':>9}{'cumulative TVL':>20}")
    for step in (0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        kept = [v for v in valued.values() if v >= step]
        print(f"{'$' + format(step, ',') :>13}{len(kept):>10,}{len(kept) * 2:>9,}"
              f"{'$' + format(round(sum(kept)), ','):>20}")

    out = {pool: [meta[pool][0], meta[pool][1], meta[pool][2], round(tvl, 2)]
           for pool, tvl in valued.items() if tvl >= args.floor}
    print(f"\n{len(out):,} pair(s) at or above ${args.floor:,.0f}")
    print(f"  holding ${sum(row[3] for row in out.values()):,.0f} between them")
    where = Path(args.out) if args.out else ROOT / "data" / "univ2" / f"{args.chain}.json"
    if not out:
        # An empty census is worse than none: `Univ2.load` would find a file,
        # read no pairs from it, and the venue would be silently absent -- which
        # is the failure `--univ3` shipped with and took a day to notice.  The
        # run above already said why it is empty.
        print(f"  ! nothing to write; {shown(where)} left alone")
        return 1
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True))
    print(f"  written to {shown(where)} ({where.stat().st_size:,} B)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--span", type=int, default=10_000,
                    help="log window; endpoints commonly cap this at 10,000")
    ap.add_argument("--floor", type=float, default=10_000.0,
                    help="keep pairs holding at least this much, in USD")
    ap.add_argument("--fee-bps", type=int, default=DEFAULT_FEE_BPS)
    ap.add_argument("--factory", default=FACTORY)
    ap.add_argument("--deploy", type=int, default=DEPLOY)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None,
                    help="where to keep discovery progress so a killed run "
                         "resumes; a full scan is over an hour")
    ap.add_argument("--from", dest="source", default=None,
                    help="re-cut an existing census at a different --floor "
                         "instead of scanning; the reserve read is the hour, "
                         "and the TVL it found is already in the file")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many pairs pass the quote filter")
    args = ap.parse_args()

    if args.source:
        # Only ever a narrowing: the file holds what passed its own floor, so
        # a lower one here cannot invent the pairs it already dropped.  Said
        # plainly rather than silently producing a census missing its tail.
        held = json.loads(Path(args.source).read_text())
        print(f"re-cutting {len(held):,} pair(s) from {args.source}")
        return emit({p: row[3] for p, row in held.items()},
                    {p: (row[0], row[1], row[2]) for p, row in held.items()},
                    args)

    chain = chain_table.CHAINS[args.chain]
    url = config.rpc_url(chain.rpc_attr) if args.private else chain.public_rpc
    rpc = JsonRpcTransport(url, chain_id=chain.chain_id)
    head = rpc.block
    print(f"{chain.name} · block {head:,} · factory {args.factory}")

    cache = Path(args.cache) if args.cache else None
    pairs, refused = discover(rpc, head, args.span, args.factory, args.deploy,
                              cache)
    print(f"\n{len(pairs):,} pair(s) ever created")
    if refused:
        # Refusing to continue rather than censusing a slice of the factory: a
        # short census and a small one look identical downstream, and the pairs
        # missing from it are missing silently for as long as the file lives.
        print(f"  ! {refused} log window(s) refused -- not writing a partial "
              f"census.  A smaller --span is the usual fix; endpoints commonly "
              f"cap it at 10,000 blocks.")
        return 1

    wanted = {p: v for p, v in pairs.items() if quoted_side(*v) is not None}
    print(f"{len(wanted):,} with a priced side ({len(wanted) / max(len(pairs), 1):.1%})")
    if args.limit:
        wanted = dict(list(wanted.items())[:args.limit])

    # WETH and WBTC in dollars, from a stable pair of each.  One read, and the
    # floor is the only thing that rests on it.
    prices = {}
    for token, name in ((WETH, "WETH"), (WBTC, "WBTC")):
        best = None
        for pool, (t0, t1) in wanted.items():
            if token in (t0, t1) and (t0 in STABLES or t1 in STABLES):
                best = (pool, t0, t1)
                break
        if best is None:
            print(f"  ! no {name}/stable pair; pairs holding it are dropped")
            continue
        got = read_reserves(rpc, [best[0]], head)
        if best[0] not in got:
            continue
        r0, r1 = got[best[0]]
        pool, t0, t1 = best
        stable, other = (t1, t0) if t1 in STABLES else (t0, t1)
        s_res, o_res = (r1, r0) if t1 in STABLES else (r0, r1)
        prices[token] = ((s_res / 10 ** STABLES[stable])
                         / (o_res / 10 ** (18 if other == WETH else 8)))
        print(f"  {name} ${prices[token]:,.2f}")

    addrs = list(wanted)
    print(f"\nreading reserves for {len(addrs):,} pair(s)")
    reserves = read_reserves(rpc, addrs, head)
    print(f"  {len(reserves):,} answered")

    out: dict[str, list] = {}
    valued: dict[str, float] = {}
    for pool, (t0, t1) in wanted.items():
        got = reserves.get(pool)
        if not got:
            continue
        side = quoted_side(t0, t1)
        if side is None:
            continue
        index, decimals, price = side
        if price is None:
            token = (t0, t1)[index]
            price = prices.get(token)
            if price is None:
                continue
        held = got[index] / 10 ** decimals * price
        tvl = 2.0 * held
        valued[pool] = tvl
        if tvl < args.floor:
            continue
        out[pool] = [t0, t1, args.fee_bps, round(tvl, 2)]

    return emit(valued, {p: (t0, t1, args.fee_bps)
                         for p, (t0, t1) in wanted.items()}, args)


if __name__ == "__main__":
    raise SystemExit(main())
