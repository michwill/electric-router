"""Does adding a venue ever cost the answer, and is the answer reproducible?

Adding a source of liquidity must never make a quote worse.  That is a promise
the router makes structurally -- `generate` runs a sub-ballot for the incumbent
so its answer stays on the table -- and this is how the promise gets checked.

**Tokens come from the venue's own census**, not a hand-picked majors list.  The
sweep that shipped `--univ3` covered five majors, reported "0 worse", and was
hiding a **-4805 bp** regression on `FRAX -> WBTC` at $1M: tokens only Uniswap
could reach were never priced by the frame, so their arcs read as 5.4e7 of free
value and wrecked the base solve.  A token list that cannot reach the venue's
own pools cannot find that.

**Every case is measured until it repeats.**  Each reading is
`curve -> venue -> curve`, and the third quote is a contamination control: a
session warms its ladders as it goes, so if the two curve-only readings of one
case disagree, that case is thrown out rather than counted.  That control is
necessary and *not sufficient* -- it catches drift inside a case and misses
drift across the session.  Twice this cost a day: `DAI -> WBTC` reported
-1604 bp and would not reproduce, and `PYUSD -> FRAX` at $1M read **-38.47 bp
once and +12.20 bp another time** on the same commit and block.  So a case is
re-measured until two readings agree to `--agree-bp`, and one that never agrees
is reported as unstable instead of becoming a finding.

Usage:

    uv run python scripts/venue_sweep.py --block 25925722 --private

`--private` is needed for Uniswap: its tick reads are `eth_getStorageAt`, which
a scoped endpoint refuses.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import erouter_evm  # noqa: E402

import erouter  # noqa: F401,E402  -- pins BLAS to one thread, before numpy
from erouter.chain import chains as chain_table  # noqa: E402
from erouter.chain.cache import UniverseCache  # noqa: E402
from erouter.chain.session import RouterSession  # noqa: E402
from erouter.core.types import ArcKind  # noqa: E402
from erouter.dev import config  # noqa: E402
from erouter.dev.rpc import BATCH_FLOOR, AsyncTransport, JsonRpcTransport  # noqa: E402
from erouter.dev.universe import load_pools  # noqa: E402
from erouter.venues.univ3_session import Univ3  # noqa: E402

#: Notionals in USD.  Priced per token by probing, so "1e6" is the same trade
#: whichever coin it starts from -- a sweep in token units compares a $1k WBTC
#: trade against a $1M USDC one and calls the difference a venue effect.
NOTIONALS = (1e3, 1e4, 1e5, 1e6)


class _Files:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self, name: str):
        path = self._root / "data" / name
        return path.read_bytes() if path.exists() else None


def session_for(args):
    chain = chain_table.CHAINS[args.chain]
    url = config.rpc_url(chain.rpc_attr) if args.private else chain.public_rpc
    transport = JsonRpcTransport(url, chain_id=chain.chain_id)
    transport.batch_size = max(
        transport.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)
    cache = UniverseCache()
    if cache.get(chain.chain_id, args.min_tvl, allow_stale=True) is None:
        load_pools(chain, min_tvl=args.min_tvl)
    universe = cache.get(chain.chain_id, args.min_tvl, allow_stale=True)
    venue = Univ3.load(ROOT, args.chain)
    if venue is None:
        raise SystemExit(f"no uniswap census for {args.chain}")
    session = RouterSession(
        chain, AsyncTransport(transport),
        erouter_evm.Evm("Osaka", chain.chain_id), _Files(ROOT), universe,
        min_tvl=args.min_tvl, univ3=venue)
    report = asyncio.run(session.warm(block=args.block or transport.block))
    return session, venue, report


def token_set(session, venue, limit):
    """The venue's most connected tokens, priced, most first.

    A token the venue offers but the frame cannot reach has no comparison to
    make -- `curve` refuses the pair -- so the probe failing is how it drops
    out, rather than a list of exclusions that goes stale.
    """
    nodes = session.nodes
    symbols: dict[str, str] = {}
    for pool in session.pools:
        for coin in pool.coins:
            symbols.setdefault(coin.address.lower(), coin.symbol)
    seen: Counter = Counter()
    for meta in venue.pools.values():
        for addr in (meta[0], meta[1]):
            if nodes.has(addr.lower()):
                seen[addr.lower()] += 1
    usdc = next((a for a, n in symbols.items()
                 if n.upper() == "USDC" and nodes.has(a)), None)
    if usdc is None:
        raise SystemExit("no USDC to price against")
    price = {usdc: 1.0}
    tokens = [usdc]
    for cand, _n in seen.most_common():
        if len(tokens) >= limit:
            break
        if cand == usdc:
            continue
        try:
            asyncio.run(session.set_pair(usdc, cand))
            got = session.quote(int(10_000 * 10 ** nodes.decimals(usdc)))
            out = int(got.verified_out or 0) / 10 ** nodes.decimals(cand)
        except Exception:
            continue
        if out > 0:
            price[cand] = 10_000.0 / out
            tokens.append(cand)
    return tokens, price, symbols


def read_once(session, venue, amount):
    """One `curve -> venue -> curve` reading, or `None` if contaminated."""
    def one(with_venue):
        held, session.univ3 = session.univ3, (venue if with_venue else None)
        try:
            got = session.quote(amount)
            legs = got.route.legs if got.route else []
            return (int(got.verified_out or 0),
                    sum(1 for leg in legs if leg.kind is ArcKind.SWAP_UNIV3))
        except Exception:
            return (0, 0)
        finally:
            session.univ3 = held

    base, _ = one(False)
    got, legs = one(True)
    control, _ = one(False)
    if not base or not got or base != control:
        return None
    return ((got / base - 1) * 1e4, base, got, legs)


def measure(session, venue, amount, repeats, agree_bp):
    """Read until two agree, or give up and say so.

    The A/B/A control inside `read_once` catches drift *within* a case; this
    catches drift across the session, which is what turned -38.47 bp into
    +12.20 bp on the same commit and block.
    """
    seen: list = []
    for _ in range(repeats):
        got = read_once(session, venue, amount)
        if got is None:
            continue
        for other in seen:
            if abs(other[0] - got[0]) <= agree_bp:
                return got, [*seen, got], True
        seen.append(got)
    return (seen[-1] if seen else None), seen, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--min-tvl", type=float, default=10_000.0)
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3,
                    help="most readings per case before calling it unstable")
    ap.add_argument("--agree-bp", type=float, default=0.5,
                    help="how close two readings must be to be believed")
    ap.add_argument("--worse-bp", type=float, default=0.5,
                    help="a loss past this is reported as a regression")
    args = ap.parse_args()

    session, venue, report = session_for(args)
    print(f"block {session.block:,} · {report.pools} pools · "
          f"v3 {report.univ3_pools} · {len(venue.arcs):,} tick-arc(s)")
    tokens, price, symbols = token_set(session, venue, args.tokens)
    print(f"tokens: {[symbols.get(a, a[:8]) for a in tokens]}\n")
    nodes = session.nodes

    print(f"{'pair':<18}{'notional':>12}{'curve':>26}{'curve+venue':>26}"
          f"{'delta':>10}{'v3':>4}{'reads':>7}")
    deltas: list[float] = []
    worse: list = []
    unstable: list = []
    for src in tokens:
        for dst in tokens:
            if src == dst:
                continue
            try:
                asyncio.run(session.set_pair(src, dst))
            except Exception:
                continue
            name = f"{symbols.get(src, src[:6])}->{symbols.get(dst, dst[:6])}"
            for usd in NOTIONALS:
                amount = int(usd / price[src] * 10 ** nodes.decimals(src))
                if amount <= 0:
                    continue
                got, reads, agreed = measure(
                    session, venue, amount, args.repeats, args.agree_bp)
                if got is None:
                    print(f"{name:<18}{usd:>12,.0f}   no comparable reading")
                    continue
                delta, base, out, legs = got
                if not agreed:
                    unstable.append((name, usd, [r[0] for r in reads]))
                    print(f"{name:<18}{usd:>12,.0f}{base:>26,}{out:>26,}"
                          f"{delta:>+9.2f}bp{legs:>4}{len(reads):>7}  UNSTABLE "
                          f"{[round(r[0], 2) for r in reads]}")
                    continue
                deltas.append(delta)
                if delta < -args.worse_bp:
                    worse.append((name, usd, delta, legs))
                print(f"{name:<18}{usd:>12,.0f}{base:>26,}{out:>26,}"
                      f"{delta:>+9.2f}bp{legs:>4}{len(reads):>7}")

    if not deltas:
        print("\nnothing measured")
        return 1
    ordered = sorted(deltas)
    better = sum(1 for d in deltas if d > args.agree_bp)
    tied = sum(1 for d in deltas if abs(d) <= args.agree_bp)
    print(f"\n{len(deltas)} case(s) believed: {better} better, {tied} tied, "
          f"{len(deltas) - better - tied} worse")
    print(f"  best {ordered[-1]:+.2f} bp   median "
          f"{statistics.median(ordered):+.2f} bp   worst {ordered[0]:+.2f} bp")
    print(f"\n  worse than {args.worse_bp} bp: {len(worse)}")
    for name, usd, delta, legs in sorted(worse, key=lambda r: r[2]):
        print(f"    {name:<18}${usd:>12,.0f}{delta:>+9.2f}bp   {legs} v3 leg(s)")
    print(f"  unstable, never repeated: {len(unstable)}")
    for name, usd, reads in unstable:
        print(f"    {name:<18}${usd:>12,.0f}   {[round(r, 2) for r in reads]}")
    # An unstable case is not a pass.  It is a case this harness cannot speak
    # about, and quoting the rest as if it could is how -38.47 bp became a
    # finding for half a day.
    return 1 if worse or unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
