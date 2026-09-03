"""Every HTTP round trip a warm quote makes, and what it asked for.

`_post_inner` is the one place a request leaves the process, so wrapping it
counts round trips rather than inferring them from a stage label.  A stage
marked `rpc` in `erouter bench` is *allowed* to talk to the network; this says
whether it did.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import sys as _sys
import time
from pathlib import Path

import erouter_evm

from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.chain.session import RouterSession
from erouter.core import candidates as _cand
from erouter.dev import config
from erouter.dev.rpc import AsyncTransport, JsonRpcTransport
from erouter.dev.universe import load_pools

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="src", default="crvUSD")
ap.add_argument("--to", dest="dst", default="sDOLA")
ap.add_argument("--amount", type=float, default=100.0)
ap.add_argument("--chain", default="ethereum")
ap.add_argument("--min-tvl", type=float, default=10_000.0)
ap.add_argument("--reps", type=int, default=8)
# Block-to-block variance on this pair is large -- 64 solves at one
# block and 106 at another -- so any A/B has to pin it.
ap.add_argument("--block", type=int, default=None)
opts = ap.parse_args()


class _Files:
    def __init__(self, root):
        self._root = root

    async def load(self, name):
        path = self._root / "data" / name
        return path.read_bytes() if path.exists() else None


chain = chain_table.get(opts.chain)
rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), chain_id=chain.chain_id)
cache = UniverseCache()
if cache.get(chain.chain_id, opts.min_tvl, allow_stale=True) is None:
    load_pools(chain, min_tvl=opts.min_tvl)
universe = cache.get(chain.chain_id, opts.min_tvl, allow_stale=True)
root = Path(__file__).resolve()
root = Path("/home/michwill/Projects/electric-router")

session = RouterSession(chain, AsyncTransport(rpc), erouter_evm.Evm("Osaka", chain.chain_id),
                        _Files(root), json.loads(json.dumps(universe)),
                        min_tvl=opts.min_tvl)

# --- count every request that actually leaves the process ---------------
posts = collections.Counter()
spent = collections.Counter()
sizes = collections.Counter()
recording = [False]
real_post = JsonRpcTransport._post_inner


def counted(self, payload):
    if not recording[0]:
        return real_post(self, payload)
    try:
        body = json.loads(payload)
    except Exception:
        body = []
    calls = body if isinstance(body, list) else [body]
    label = ",".join(sorted({c.get("method", "?") for c in calls}))
    t0 = time.perf_counter()
    try:
        return real_post(self, payload)
    finally:
        posts[label] += 1
        spent[label] += time.perf_counter() - t0
        sizes[label] += len(calls)


JsonRpcTransport._post_inner = counted

# --- and who asks for each solve --------------------------------------
asked = collections.Counter()
asked_ms = collections.Counter()
real_solve = _cand.active_set_solve


def attributed(g, *a, **kw):
    if not recording[0]:
        return real_solve(g, *a, **kw)
    frame, depth = _sys._getframe(1), 2
    where = f"{frame.f_code.co_filename.split('/')[-1]}:{frame.f_code.co_name}"
    # `resolve` is a closure inside `generate`; walk out of solve.py so the
    # name that appears is the family that wanted the solve.
    while "solve.py" in where and depth < 8:
        frame = _sys._getframe(depth)
        where = f"{frame.f_code.co_filename.split('/')[-1]}:{frame.f_code.co_name}"
        depth += 1
    t0 = time.perf_counter()
    try:
        return real_solve(g, *a, **kw)
    finally:
        asked[where] += 1
        asked_ms[where] += time.perf_counter() - t0


_cand.active_set_solve = attributed

block = opts.block or rpc.block
report = asyncio.run(session.warm(block=block))
print(f"block {report.block:,} · {report.pools} pools · {report.arcs} arcs "
      f"· solver {session.solver}")

# resolve the pair by symbol the way the CLI does
def address_of(symbol: str) -> str:
    if symbol.startswith("0x"):
        return symbol
    for spec in session.pools:
        for coin in spec.coins:
            if coin.symbol.lower() == symbol.lower():
                return coin.address
    for node in range(session.nodes.n_nodes):
        for token in session.nodes.tokens_of(node):
            if session.nodes.symbol(token).lower() == symbol.lower():
                return token
    raise SystemExit(f"no token called {symbol}")


src, dst = address_of(opts.src), address_of(opts.dst)
asyncio.run(session.set_pair(src, dst))
decimals = session.nodes.decimals(src)
amount = int(opts.amount * 10 ** decimals)

session.quote(amount)          # settle: caches, models, the first solve
posts.clear()
spent.clear()
sizes.clear()

recording[0] = True
wall, cpu = [], []
stages: dict[str, list[float]] = {}
arms: dict[bool, list[float]] = {True: [], False: []}
arm_stages: dict[bool, dict[str, list[float]]] = {True: {}, False: {}}
can_toggle = hasattr(_cand, "_ACCEL_ON")
for rep in range(opts.reps * (2 if can_toggle else 1)):
    on = (rep % 2 == 0) if can_toggle else True
    if can_toggle:
        _cand._ACCEL_ON = on
    w0, c0 = time.perf_counter(), time.process_time()
    result = session.quote(amount)
    took = (time.perf_counter() - w0) * 1e3
    arms[on].append(took)
    for name, value in (result.timings or {}).items():
        arm_stages[on].setdefault(name, []).append(value)
    if on:
        wall.append(took)
        cpu.append((time.process_time() - c0) * 1e3)
        for name, value in (result.timings or {}).items():
            stages.setdefault(name, []).append(value)
if can_toggle:
    _cand._ACCEL_ON = True
recording[0] = False

n = opts.reps
print(f"\n{opts.src} -> {opts.dst}, {opts.amount:g} units "
      f"({amount} wei, {decimals} decimals)")
print(f"{'':<28}{'min ms':>10}{'median':>10}")
print(f"{'warm quote wall':<28}{min(wall):>10.2f}{statistics.median(wall):>10.2f}")
print(f"{'warm quote cpu':<28}{min(cpu):>10.2f}{statistics.median(cpu):>10.2f}")
print(f"{'cpu/wall at the min':<28}{min(cpu) / min(wall):>10.2f}")

total_posts = sum(posts.values())
print(f"\nHTTP round trips per warm quote: {total_posts / n:.2f}")
if total_posts:
    print(f"\n{'method(s)':<44}{'per quote':>11}{'calls':>8}{'ms':>9}")
    for label, count in posts.most_common():
        print(f"{label[:43]:<44}{count / n:>11.2f}"
              f"{sizes[label] / n:>8.1f}{spent[label] * 1e3 / n:>9.2f}")
    print(f"\n{'network total':<44}{'':>11}{'':>8}"
          f"{sum(spent.values()) * 1e3 / n:>9.2f}")
else:
    print("  none -- the warm quote never leaves the process")
if stages:
    print(f"\n{'stage':<22}{'min ms':>10}{'median':>10}{'share':>9}")
    ranked = sorted(stages.items(), key=lambda kv: -min(kv[1]))
    for name, values in ranked:
        share = min(values) / min(wall) * 100
        print(f"{name:<22}{min(values):>10.2f}"
              f"{statistics.median(values):>10.2f}{share:>8.1f}%")
    accounted = sum(min(v) for v in stages.values())
    print("-" * 51)
    print(f"{'accounted':<22}{accounted:>10.2f}{'':>10}"
          f"{accounted / min(wall) * 100:>8.1f}%")
    print(f"{'unclocked':<22}{min(wall) - accounted:>10.2f}{'':>10}"
          f"{(min(wall) - accounted) / min(wall) * 100:>8.1f}%")

if asked:
    print(f"\n{'who asks for the solve':<34}{'per quote':>11}{'ms':>9}")
    for name, count in asked.most_common():
        print(f"{name[:33]:<34}{count / n:>11.1f}{asked_ms[name] * 1e3 / n:>9.2f}")

interesting = ("candidate_solves", "candidate_pivots", "candidates",
               "candidates_quoted", "cg_rounds", "arcs_planned",
               "arcs_calibrated", "probes", "quotes")
counters = {k: v for k, v in (result.counters or {}).items()
            if k in interesting or "solve" in k or "round" in k}
if counters:
    print("\nwork")
    for name, value in sorted(counters.items()):
        print(f"  {name:<26}{value:>10}")

if can_toggle and arms[False]:
    print(f"\n{'generation':<26}{'min ms':>10}{'median':>10}")
    for label, on in (("rust", True), ("python", False)):
        v = arms[on]
        print(f"{label:<26}{min(v):>10.2f}{statistics.median(v):>10.2f}")
    print(f"{'speed-up':<26}{min(arms[False]) / min(arms[True]):>9.2f}x")
    print(f"{'saved':<26}{min(arms[False]) - min(arms[True]):>10.2f} ms")
    for name in ("candidates", "refine"):
        a = arm_stages[True].get(name)
        b = arm_stages[False].get(name)
        if a and b:
            print(f"  {name:<24}{min(a):>10.2f}{min(b):>10.2f}   (rust/python)")

print(f"\nverified_out {result.verified_out}")
