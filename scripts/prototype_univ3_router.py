#!/usr/bin/env python3
"""A router over Curve *and* Uniswap v3, to see what the pair is worth.

The v3 arcs are built from tick state rather than probed, so they join the
pipeline already calibrated: `_recalibrate` keys on the ladder store and a v3
arc has no ladder, so refine and size-check pass over them.  That is the
trade -- much better derivatives, many more pools and many more slots.

Injected at `_assemble`, which is the point where the arc list and the solver
arrays are built together; appending there keeps them aligned, which appending
at `build` does not (`_assemble` realigns its list to whatever `build` kept).
`realize` is wrapped to `collapse` each pool's tick-arcs back into one leg
before the route is drawn.

No execution: `RouteExecutor` has no v3 callback, so a route carrying one is
quoted and not sent.  Verification is skipped for the same reason.

    uv run python scripts/prototype_univ3_router.py --from USDC --to WETH \
        --amount 100000 --pools v3.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.chain.session import RouterSession
from erouter.core import pipeline
from erouter.core.codec import decode, encode_call
from erouter.core.types import ArcKind
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, AsyncTransport, JsonRpcTransport
from erouter.dev.universe import load_pools
from erouter.venues import univ3
from erouter.venues.univ3_chain import read_pools
from erouter.venues.univ3_client import Bank, teach

ROOT = Path(__file__).resolve().parents[1]
QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"


class _Files:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self, name):
        path = self._root / "data" / name
        return path.read_bytes() if path.exists() else None


def resolve(nodes, pools, text: str) -> str:
    if text.startswith("0x") and len(text) == 42:
        return text.lower()
    wanted = text.upper()
    best, tvl = None, -1.0
    for pool in pools:
        for coin in pool.coins:
            if coin.symbol.upper() == wanted and pool.tvl_usd > tvl:
                best, tvl = coin.address.lower(), pool.tvl_usd
    if best is None:
        raise SystemExit(f"no token {text!r} in the universe")
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--from", dest="src", default="USDC")
    p.add_argument("--to", dest="dst", default="WETH")
    p.add_argument("--amount", type=float, default=100_000)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--pools", required=True, help="census --pools output")
    p.add_argument("--floor", type=float, default=10_000.0)
    p.add_argument("--ticks", type=int, default=16, help="tick-arcs a side")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--legs", action="store_true", help="print each route's legs")
    p.add_argument("--max-spread", type=float, default=1e18,
                   help="§9.7 conductance-spread bound, raised for v3")
    args = p.parse_args(argv)

    import json

    import erouter_evm

    chain = chain_table.CHAINS[args.chain]
    transport = JsonRpcTransport(config.rpc_url(chain.rpc_attr),
                                 chain_id=chain.chain_id)
    transport.batch_size = max(
        transport.probe_batch_limit(("eth_blockNumber", [])), BATCH_FLOOR)
    cache = UniverseCache()
    if cache.get(chain.chain_id, args.min_tvl, allow_stale=True) is None:
        load_pools(chain, min_tvl=args.min_tvl)
    universe = cache.get(chain.chain_id, args.min_tvl, allow_stale=True)

    session = RouterSession(chain, AsyncTransport(transport),
                            erouter_evm.Evm("Osaka", chain.chain_id),
                            _Files(ROOT), universe, min_tvl=args.min_tvl)
    block = args.block or transport.block
    asyncio.run(session.warm(block=block))
    nodes = session.nodes
    src = resolve(nodes, session.pools, args.src)
    dst = resolve(nodes, session.pools, args.dst)
    asyncio.run(session.set_pair(src, dst))
    print(f"{chain.name} block {block:,}   {nodes.node_symbol(nodes.node(src))} -> "
          f"{nodes.node_symbol(nodes.node(dst))}   {args.amount:,.0f}\n")

    with open(args.pools) as handle:
        census = json.load(handle)
    # Both coins must already be nodes: a token the frame cannot price has no
    # arc to give, and extending the node map is a separate piece of work.
    wanted = {}
    for pool, row in census.items():
        t0, t1, fee = row[0].lower(), row[1].lower(), row[2]
        tvl = row[3] if len(row) > 3 else 0.0
        if tvl < args.floor or not (nodes.has(t0) and nodes.has(t1)):
            continue
        if nodes.node(t0) == nodes.node(t1):
            continue
        wanted[pool.lower()] = (t0, t1, fee, nodes.decimals(t0), nodes.decimals(t1))
    print(f"v3: {len(wanted):,} of {len(census):,} pool(s) above "
          f"${args.floor:,.0f} have both coins in the node map")

    t0 = time.perf_counter()
    state = read_pools(transport, wanted, block, ticks=args.ticks)
    read_ms = (time.perf_counter() - t0) * 1e3

    extra, banks = [], {}
    for pool, (pool_state, ticks) in state.items():
        t_0, t_1, _fee, _d0, _d1 = wanted[pool]
        made = univ3.pool_arcs(pool, pool_state, ticks, nodes,
                               token0=t_0, token1=t_1, max_ticks=args.ticks,
                               tvl_usd=census[pool][3] if len(census[pool]) > 3
                               else 0.0)
        extra += made
        for zero_for_one in (True, False):
            bank = univ3.arcs(pool_state, ticks, zero_for_one=zero_for_one,
                              max_ticks=args.ticks)
            if bank:
                banks[(pool, *((0, 1) if zero_for_one else (1, 0)))] = bank
    print(f"     {len(state):,} answered, {len(extra):,} tick-arc(s) built, "
          f"read in {read_ms:,.0f} ms")
    if extra:
        import numpy as _np
        a = _np.array([x.a for x in extra])
        b = _np.array([x.B for x in extra])
        cap = _np.array([x.cap for x in extra])
        ratio = _np.where(b > 0, a / _np.where(b > 0, b, 1.0), _np.inf)
        finite = ratio[_np.isfinite(ratio)]
        print(f"     a    {a.min():.3e} .. {a.max():.3e}")
        print(f"     B    {b.min():.3e} .. {b.max():.3e}")
        print(f"     cap  {cap.min():.3e} .. {cap.max():.3e}")
        print(f"     a/B  {finite.min():.3e} .. {finite.max():.3e}"
              f"   spread {finite.max() / max(finite.min(), 1e-300):.2e}")
        for q in (1, 50, 99):
            print(f"       p{q:<3} a/B {_np.percentile(finite, q):.3e}"
                  f"   cap {_np.percentile(cap, q):.3e}")
    print()

    # `verify` re-quotes every candidate to rank it, and a v3 leg has nothing
    # on chain to answer for it.  Teach the walk instead.
    priced = {key: Bank(bank,
                        wanted[key[0]][3] if key[1] == 0 else wanted[key[0]][4],
                        wanted[key[0]][4] if key[1] == 0 else wanted[key[0]][3])
              for key, bank in banks.items()}
    teach(session.client, priced)

    real_assemble = pipeline._assemble
    real_realize = pipeline.realize
    enabled = {"on": False}

    real_build = pipeline.build

    def build(*a, **kw):
        # §9.7's spread bound is calibrated on a Curve-only universe, where
        # 1e15 can only mean a floored `B`.  A v3 tick is nearly linear over
        # its own range, so its `B` is genuinely tiny and 144 pools reach
        # 1.9e15 with nothing floored: the inference breaks, not the rule.
        if enabled["on"]:
            kw.setdefault("max_spread", args.max_spread)
        return real_build(*a, **kw)

    def assemble(arcs, nu, Psi, nodes_, src_node, dst_node, result):
        if enabled["on"] and extra:
            have = {a.id for a in arcs}
            arcs = list(arcs) + [a for a in extra if a.id not in have]
        return real_assemble(arcs, nu, Psi, nodes_, src_node, dst_node, result)

    carried = {"v3": 0, "value": 0.0}

    def realize(live, psi, nu, nodes_, **kw):
        if enabled["on"]:
            carried["v3"] = sum(1 for a, f in zip(live, psi, strict=True)
                                if a.kind is ArcKind.SWAP_UNIV3 and f > 0)
            carried["value"] = sum(float(f) for a, f in zip(live, psi, strict=True)
                                   if a.kind is ArcKind.SWAP_UNIV3 and f > 0)
            live, psi = univ3.collapse(live, psi, nu, nodes_, banks)
            if args.legs:
                for a, f in zip(live, psi, strict=True):
                    if a.kind is ArcKind.SWAP_UNIV3 and f > 0:
                        dx = f / float(nu[a.tau]) / a.rate_in
                        print(f"           collapsed {a.pool[:12]} {a.i}>{a.j}"
                              f"  psi {f:,.2f}  dx {dx:,.4f}"
                              f"  a {a.a:.6g}  cap {a.cap:,.2f}"
                              f"  {a.token_in[:8]}->{a.token_out[:8]}")
        return real_realize(live, psi, nu, nodes_, **kw)

    pipeline.build = build
    pipeline._assemble = assemble
    pipeline.realize = realize

    amount = int(args.amount * 10 ** nodes.decimals(src))
    print(f"{'arm':<10}{'out':>26}{'arcs':>8}{'legs':>6}{'v3 legs':>9}"
          f"{'ms':>9}{'vs curve':>11}")
    base = None
    for label, on in (("curve", False), ("curve+v3", True)):
        enabled["on"] = on
        session.quote(amount)                      # warm the path
        best, result = None, None
        for _ in range(args.reps):
            started = time.perf_counter()
            got = session.quote(amount)
            took = (time.perf_counter() - started) * 1e3
            if best is None or took < best:
                best, result = took, got
        # `modelled_out`, not `verified_out`: a route carrying a v3 leg
        # cannot be re-quoted on chain, so the modelled figure is the
        # only one both arms have.
        out = result.route.modelled_out if result.route else 0
        legs = result.route.legs if result.route else []
        v3 = sum(1 for leg in legs
                 if leg.kind is ArcKind.SWAP_UNIV3) if legs else 0
        if base is None:
            base = out
        delta = (out - base) / base * 1e4 if base else 0.0
        counters = result.counters
        print(f"{label:<10}{out:>26,}{counters.get('arcs_priced_out', 0):>8,}"
              f"{len(legs):>6}{v3:>9}{best:>9.0f}{delta:>+10.2f}bp")
        print(f"           dust dropped {counters.get('arcs_dropped_dust', 0):>5,}"
              f"   v3 arcs with flow {carried['v3'] if on else 0:>4}"
              f"   condition {counters.get('condition', 0):.2e}")
        for note in (result.warnings or [])[:3]:
            if "conductance spread" in note or "dust" in note.lower():
                print(f"           ! {note[:96]}")
        if args.legs:
            for leg in legs:
                print(f"           {leg.kind.name:<14}{leg.pool_name[:22]:<24}"
                      f"{leg.amount_in:>26,} -> {leg.amount_out:>26,}")
        # Every v3 leg against the pool's own quoter, at the same block: the
        # model is the whole claim, so it is checked rather than trusted.
        for leg in legs:
            if leg.kind is not ArcKind.SWAP_UNIV3:
                continue
            t_in, t_out, fee, _d0, _d1 = wanted[leg.target.lower()]
            if leg.leg.i == 1:
                t_in, t_out = t_out, t_in
            data = encode_call(
                "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
                (t_in, t_out, leg.amount_in, fee, 0))
            try:
                truth = decode(["uint256", "uint160", "uint32", "uint256"],
                               bytes.fromhex(transport.fetch("eth_call", [
                                   {"to": QUOTER_V2, "data": "0x" + data.hex()},
                                   hex(block)])[2:]))[0]
            except Exception as exc:
                print(f"           quoter refused: {str(exc)[:60]}")
                continue
            bp = (leg.amount_out - truth) / truth * 1e4 if truth else float("nan")
            print(f"           vs QuoterV2  {truth:>26,}   modelled "
                  f"{bp:+.4f} bp")

    pipeline.build = real_build
    pipeline._assemble, pipeline.realize = real_assemble, real_realize
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
