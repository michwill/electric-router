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

**Every case is measured until it repeats, in more than one session.**  Each
reading is `curve -> venue -> curve`, and the third quote is a contamination
control: a session warms its ladders as it goes, so if the two curve-only
readings of one case disagree, that case is thrown out rather than counted.

That control is necessary and *not sufficient*, at two levels, and both were
learned by believing a number that was not true.  It catches drift inside a
case and misses drift across the session: `DAI -> WBTC` reported -1604 bp and
would not reproduce, and `PYUSD -> FRAX` at $1M read **-38.47 bp once and
+12.20 bp another time** on the same commit and block.  So a case is
re-measured until two readings agree to `--agree-bp`.

Repeating inside one session is *also* not sufficient, because a session
settles into a state and then reports that state consistently.  `FRAX -> WBTC`
at $1M read **-5088.34 bp twice** in one session -- agreeing to well inside the
tolerance, control passing, reported as believed -- and **+0.00 bp** in the
next session at the same commit and block.  Eleven of 104 comparable cases
moved between two runs, concentrated on the large notionals a venue is judged
by.  So `--sessions` independently built sessions must also agree, and a case
they disagree on is reported as unstable rather than becoming a finding.

Usage:

    uv run python scripts/venue_sweep.py --block 25925722 --private
    uv run python scripts/venue_sweep.py --venue v2 --block 25925722 --private

`--venue` picks what the A/B switches off.  Measuring `v3,v2` together answers
a different question from measuring each alone: two venues can each be neutral
and still cost the answer jointly, by crowding the ballot.

`--private` is needed for Uniswap: its tick reads are `eth_getStorageAt`, which
a scoped endpoint refuses.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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
from erouter.venues.univ2_session import Univ2  # noqa: E402
from erouter.venues.univ3_session import Univ3  # noqa: E402

#: Notionals in USD.  Priced per token by probing, so "1e6" is the same trade
#: whichever coin it starts from -- a sweep in token units compares a $1k WBTC
#: trade against a $1M USDC one and calls the difference a venue effect.
NOTIONALS = (1e3, 1e4, 1e5, 1e6)


@dataclass
class Venue:
    """One venue, and the two things the sweep does to it: switch it off and
    count its legs.

    The A/B is `setattr(session, attr, None)`, which is the same seam the CLI
    flag uses, so an arm with the venue off is a session that never loaded it
    rather than one carrying its arcs and declining to pick them.
    """

    name: str
    attr: str
    kind: ArcKind
    obj: object

    def arc_count(self) -> int:
        return len(self.obj.arcs)


#: Venues the sweep can measure, and what `--venue` accepts.
VENUES = {
    "v3": ("univ3", ArcKind.SWAP_UNIV3, Univ3),
    "v2": ("univ2", ArcKind.SWAP_UNIV2, Univ2),
}


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
    venues, kwargs = [], {}
    for want in args.venue:
        attr, kind, cls = VENUES[want]
        obj = cls.load(ROOT, args.chain)
        if obj is None:
            raise SystemExit(f"no {want} census for {args.chain}")
        venues.append(Venue(want, attr, kind, obj))
        kwargs[attr] = obj
    session = RouterSession(
        chain, AsyncTransport(transport),
        erouter_evm.Evm("Osaka", chain.chain_id), _Files(ROOT), universe,
        min_tvl=args.min_tvl, **kwargs)
    report = asyncio.run(session.warm(block=args.block or transport.block))
    return session, venues, report


def token_set(session, venues, limit):
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
    for venue in venues:
        for meta in venue.obj.pools.values():
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


class Reading(NamedTuple):
    """One believed A/B/A measurement, and the route that produced it.

    The route is kept because the number alone cannot say *why*: a case that
    loses 5,088 bp through one v2 leg and one that loses it by perturbing the
    base solve read identically here, and only the legs tell them apart.
    """

    delta: float
    base: int
    out: int
    legs: int
    route: object = None


def read_once(session, venues, amount):
    """One `curve -> venue -> curve` reading, or `None` if contaminated."""
    kinds = {v.kind for v in venues}

    def one(with_venue):
        held = {v.attr: getattr(session, v.attr) for v in venues}
        for v in venues:
            setattr(session, v.attr, v.obj if with_venue else None)
        try:
            got = session.quote(amount)
            legs = got.route.legs if got.route else []
            return (int(got.verified_out or 0),
                    sum(1 for leg in legs if leg.kind in kinds),
                    got.route)
        except Exception:
            return (0, 0, None)
        finally:
            for attr, was in held.items():
                setattr(session, attr, was)

    base, _, _ = one(False)
    got, legs, route = one(True)
    control, _, _ = one(False)
    if not base or not got or base != control:
        return None
    return Reading((got / base - 1) * 1e4, base, got, legs, route)


def measure(session, venues, amount, repeats, agree_bp):
    """Read until two agree, or give up and say so.

    The A/B/A control inside `read_once` catches drift *within* a case; this
    catches drift across the session, which is what turned -38.47 bp into
    +12.20 bp on the same commit and block.
    """
    seen: list = []
    for _ in range(repeats):
        got = read_once(session, venues, amount)
        if got is None:
            continue
        for other in seen:
            if abs(other.delta - got.delta) <= agree_bp:
                return got, [*seen, got], True
        seen.append(got)
    return (seen[-1] if seen else None), seen, False


def sweep_one(session, venues, pairs, price, symbols, nodes, args):
    """Every case on one session, in the order given.  `case -> Reading`.

    The order is the point.  Sessions that walk the same case list settle the
    same way and then agree with each other for the wrong reason, which is how
    the first version of this cross-check would have certified the very number
    it was written to catch.  The caller hands each session a different order.
    """
    out: dict = {}
    for src, dst in pairs:
        try:
            asyncio.run(session.set_pair(src, dst))
        except Exception:
            continue
        for usd in NOTIONALS:
            amount = int(usd / price[src] * 10 ** nodes.decimals(src))
            if amount <= 0:
                continue
            got, _reads, agreed = measure(
                session, venues, amount, args.repeats, args.agree_bp)
            out[(src, dst, usd)] = got if agreed else None
        print(f"  {symbols.get(src, src[:6])}->{symbols.get(dst, dst[:6])}",
              flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--min-tvl", type=float, default=10_000.0)
    ap.add_argument("--venue", default="v3",
                    help="which venue(s) the A/B switches: v3, v2, or v3,v2 "
                         "to measure them together")
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--sessions", type=int, default=2,
                    help="independent sessions that must agree, each walking "
                         "the pairs in a different order; 1 is half the wall "
                         "clock and has been seen to report a 5,088 bp "
                         "regression that the next session did not")
    ap.add_argument("--repeats", type=int, default=3,
                    help="most readings per case before calling it unstable")
    ap.add_argument("--agree-bp", type=float, default=0.5,
                    help="how close two readings must be to be believed")
    ap.add_argument("--explain-bp", type=float, default=50.0,
                    help="print the venue arm's legs for losses past this")
    ap.add_argument("--worse-bp", type=float, default=0.5,
                    help="a loss past this is reported as a regression")
    args = ap.parse_args()
    args.venue = [v.strip() for v in args.venue.split(",") if v.strip()]
    unknown = [v for v in args.venue if v not in VENUES]
    if unknown or not args.venue:
        raise SystemExit(f"--venue takes {'/'.join(VENUES)}, not {unknown}")

    built = [session_for(args) for _ in range(max(args.sessions, 1))]
    session, venues, report = built[0]
    arms = [(s, v) for s, v, _ in built]
    held = " · ".join(f"{v.name} {v.arc_count():,} arc(s)" for v in venues)
    print(f"block {session.block:,} · {report.pools} pools · {held} · "
          f"{len(arms)} session(s) that must agree")
    tokens, price, symbols = token_set(session, venues, args.tokens)
    print(f"tokens: {[symbols.get(a, a[:8]) for a in tokens]}\n")
    nodes = session.nodes

    legs_col = "+".join(v.name for v in venues)
    pairs = [(s, d) for s in tokens for d in tokens if s != d]

    # Session 0 walks the pairs forward, session 1 backward, and so on.  Two
    # sessions warmed together and marched through the same list in lockstep
    # accumulate the same history, so they settle the same way and agree
    # because they are the same experiment twice -- not because the number is
    # true.  Alternating the order is what makes the second session evidence.
    tables = []
    for n, (arm, arm_venues) in enumerate(arms):
        order = pairs if n % 2 == 0 else list(reversed(pairs))
        print(f"\nsession {n + 1}/{len(arms)}, pairs "
              f"{'forward' if n % 2 == 0 else 'backward'}:")
        tables.append(sweep_one(arm, arm_venues, order, price, symbols,
                                nodes, args))

    print(f"\n{'pair':<18}{'notional':>14}{'curve':>30}{'curve+venue':>30}"
          f"{'delta':>10}{legs_col:>6}")
    deltas: list[float] = []
    worse: list = []
    unstable: list = []
    for src, dst in pairs:
        name = f"{symbols.get(src, src[:6])}->{symbols.get(dst, dst[:6])}"
        for usd in NOTIONALS:
            seen = [table.get((src, dst, usd)) for table in tables]
            if any(r is None for r in seen):
                continue
            spread = max(abs(a.delta - b.delta) for a in seen for b in seen)
            got = seen[0]
            if spread > args.agree_bp:
                unstable.append((name, usd, [r.delta for r in seen]))
                print(f"{name:<18}{usd:>14,.0f}{got.base:>30,}{got.out:>30,}"
                      f"{got.delta:>+9.2f}bp{got.legs:>6}  UNSTABLE "
                      f"{[round(r.delta, 2) for r in seen]}")
                continue
            deltas.append(got.delta)
            if got.delta < -args.worse_bp:
                worse.append((name, usd, got.delta, got.legs, got.route))
            print(f"{name:<18}{usd:>14,.0f}{got.base:>30,}{got.out:>30,}"
                  f"{got.delta:>+9.2f}bp{got.legs:>6}")

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
    for name, usd, delta, legs, route in sorted(worse, key=lambda r: r[2]):
        print(f"    {name:<18}${usd:>12,.0f}{delta:>+9.2f}bp   "
              f"{legs} {legs_col} leg(s)")
        # The legs, for the ones big enough that the number is not the answer.
        if delta < -args.explain_bp and route is not None:
            for leg in route.legs:
                mark = "*" if leg.kind in {v.kind for v in venues} else " "
                # A realised leg names its pool and says what share of the node
                # it took, which is what a route is read by.  `i`/`j` live on
                # `leg.leg` and say much less.
                name = getattr(leg, "pool_name", "") or getattr(
                    leg, "target", "?")[:14]
                share = getattr(leg, "share_of_node", 1.0)
                print(f"      {mark} {leg.kind.name:<16} {name[:26]:<26} "
                      f"{share:>6.1%} of node")
    print(f"  unstable, never repeated: {len(unstable)}")
    for name, usd, reads in unstable:
        print(f"    {name:<18}${usd:>12,.0f}   {[round(r, 2) for r in reads]}")
    # An unstable case is not a pass.  It is a case this harness cannot speak
    # about, and quoting the rest as if it could is how -38.47 bp became a
    # finding for half a day.
    return 1 if worse or unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
