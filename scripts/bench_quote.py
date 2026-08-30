"""Where a warm quote's CPU goes, measured so a few milliseconds can be seen.

Three rules, each learned by getting it wrong first:

* **min, not median.**  Sequential arms on this machine drift 30 ms between
  them and whichever ran second lost, whichever it was.  The min is the least
  contaminated sample; the median measures the machine.
* **CPU, not wall.**  `process_time` drops the scheduler.  The ratio is printed
  so a run with network still in it is obvious rather than silently averaged in.
* **ablation, not attribution.**  cProfile charges per call, so a function
  called 26,000 times reads several times its cost -- `copy.copy` looked like
  8% of a quote and is 2%.  Every number here comes from a span around real
  work or from a counter, never from a profiler's share.

    python scripts/bench_quote.py --block N --reps 25
    python scripts/bench_quote.py --arms        # this branch against master
    python scripts/bench_quote.py --solves      # who asks for each solve
    python scripts/bench_quote.py --boundary    # FFI crossing vs Rust compute
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.chain.session import RouterSession
from erouter.core import accel
from erouter.dev import config
from erouter.dev.rpc import JsonRpcTransport
from erouter.dev.universe import load_pools

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


class _Rpc:
    """The session's transport, over the blocking one."""

    batch_size = 100

    def __init__(self, transport) -> None:
        self._t = transport
        self.chain_id = transport.chain_id

    async def batch(self, requests):
        return self._t.fetch_multi(list(requests), concurrent=True)

    async def call(self, method, params):
        got = self._t.fetch_multi([(method, params)])[0]
        if isinstance(got, Exception):
            raise got
        return got


class _Files:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self, name):
        path = self._root / "data" / name
        return path.read_bytes() if path.exists() else None


def warm(args):
    import erouter_evm

    chain = chain_table.CHAINS[args.chain]
    url = config.rpc_url(chain.rpc_attr) if args.private else chain.public_rpc
    transport = JsonRpcTransport(url, chain_id=chain.chain_id)
    cache = UniverseCache()
    if cache.get(chain.chain_id, args.min_tvl, allow_stale=True) is None:
        load_pools(chain, min_tvl=args.min_tvl)
    universe = cache.get(chain.chain_id, args.min_tvl, allow_stale=True)
    root = Path(__file__).resolve().parents[1]
    session = RouterSession(chain, _Rpc(transport),
                            erouter_evm.Evm("Osaka", chain.chain_id),
                            _Files(root), universe, min_tvl=args.min_tvl)
    asyncio.run(session.warm(block=args.block or transport.block))
    asyncio.run(session.set_pair(args.src, args.dst))
    for _ in range(3):
        session.quote(args.amount)
    return session


def stages(session, args):
    """Per-stage cost, from the pipeline's own spans rather than a profiler."""
    seen: dict[str, list[float]] = {}
    wall, cpu = [], []
    for _ in range(args.reps):
        w0, c0 = time.perf_counter(), time.process_time()
        result = session.quote(args.amount)
        wall.append((time.perf_counter() - w0) * 1e3)
        cpu.append((time.process_time() - c0) * 1e3)
        for name, ms in (result.timings or {}).items():
            seen.setdefault(name, []).append(ms)

    print(f"\n{'stage':<22}{'min ms':>9}{'median':>9}{'share':>9}")
    floor = min(wall)
    rows = sorted(((min(v), statistics.median(v), k) for k, v in seen.items()),
                  reverse=True)
    for lo, mid, name in rows:
        print(f"{name:<22}{lo:>9.2f}{mid:>9.2f}{lo / floor * 100:>8.1f}%")
    print("-" * 49)
    print(f"{'accounted':<22}{sum(r[0] for r in rows):>9.2f}")
    print(f"{'wall total':<22}{min(wall):>9.2f}{statistics.median(wall):>9.2f}")
    print(f"{'cpu total':<22}{min(cpu):>9.2f}{statistics.median(cpu):>9.2f}")
    print(f"\ncpu/wall {min(cpu) / min(wall):.3f} at the min "
          f"-- 1.0 means nothing waited on a socket")


_ACCEL_ENV = os.environ.get("EROUTER_ACCEL", "") == "1"


def arms(session, args):
    """This branch's quote path against master's, interleaved.

    The arms alternate rep by rep rather than running in sequence, because
    sequential totals on this machine swing two to one within a session and
    whichever ran second lost.  Two things separate them, and both are what
    the port added to the quote path: the Rust pool models, held on the
    client, and the batched fit `_recalibrate` reads off `pipeline._ACCEL_ON`.
    Turning both off is master's arithmetic in this process, which is a fairer
    comparison than a second checkout -- same universe, same block, same warm.

    The outputs are compared, not only the times.  A speed-up that changes the
    answer is a bug report, so the arms are required to agree to the wei.
    """
    from erouter.core import pipeline

    def toggle(on):
        session.client._native_pools = None if on else False
        pipeline._ACCEL_ON = on and _ACCEL_ENV

    cpu = {True: [], False: []}
    got = {}
    for rep in range(args.reps * 2):
        on = rep % 2 == 0
        toggle(on)
        session.quote(args.amount)
        c0 = time.process_time()
        result = session.quote(args.amount)
        cpu[on].append((time.process_time() - c0) * 1e3)
        got.setdefault(on, result.verified_out)
    toggle(True)

    fast, slow = min(cpu[True]), min(cpu[False])
    print(f"\n{'arm':<24}{'min ms':>9}{'median':>9}")
    print(f"{'rust (this branch)':<24}{fast:>9.2f}"
          f"{statistics.median(cpu[True]):>9.2f}")
    print(f"{'python (master)':<24}{slow:>9.2f}"
          f"{statistics.median(cpu[False]):>9.2f}")
    print("-" * 42)
    print(f"{'speed-up':<24}{slow / fast:>8.2f}x")
    print(f"{'saved':<24}{slow - fast:>9.2f} ms")

    agree = got[True] == got[False]
    print(f"\nverified_out  rust   {got[True]}")
    print(f"              python {got[False]}")
    print("same to the wei" if agree
          else "*** ARMS DISAGREE -- the port is wrong ***")
    return 0 if agree else 1


def solves(session, args):
    """Who asks for each solve.  The solver is already native; the count is not."""
    original = accel.solve_arrays
    count: collections.Counter = collections.Counter()
    cost: collections.Counter = collections.Counter()

    def traced(g, *a, **kw):
        frame, depth = sys._getframe(1), 2
        where = f"{frame.f_code.co_filename.split('/')[-1]}:{frame.f_code.co_name}"
        while "solve.py" in where and depth < 8:
            frame = sys._getframe(depth)
            where = f"{frame.f_code.co_filename.split('/')[-1]}:{frame.f_code.co_name}"
            depth += 1
        start = time.perf_counter()
        try:
            return original(g, *a, **kw)
        finally:
            count[where] += 1
            cost[where] += time.perf_counter() - start

    accel.solve_arrays = traced
    try:
        session.quote(args.amount)
        count.clear(), cost.clear()
        c0 = time.process_time()
        for _ in range(args.reps):
            session.quote(args.amount)
        total = (time.process_time() - c0) * 1e3 / args.reps
        print(f"\nquote {total:.1f} ms cpu\n"
              f"{'caller':<44}{'solves':>8}{'ms':>9}{'us each':>9}")
        for who, n in count.most_common():
            ms = cost[who] * 1e3 / args.reps
            print(f"{who:<44}{n // args.reps:>8}{ms:>9.2f}"
                  f"{ms / (n / args.reps) * 1000:>9.0f}")
        done = sum(cost.values()) * 1e3 / args.reps
        print(f"{'':<44}{sum(count.values()) // args.reps:>8}{done:>9.2f}"
              f"{done / total * 100:>8.1f}%")
    finally:
        accel.solve_arrays = original


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None, help="pin, for a comparable run")
    p.add_argument("--from", dest="src", default=USDC)
    p.add_argument("--to", dest="dst", default=WETH)
    p.add_argument("--amount", type=int, default=100_000 * 10**6, help="raw units")
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--arms", action="store_true",
                   help="this branch against master, interleaved")
    p.add_argument("--solves", action="store_true", help="who asks for each solve")
    args = p.parse_args()

    session = warm(args)
    print(f"block {session.block:,}  ·  accel {accel.available()}  ·  "
          f"n={args.reps}")
    if args.arms:
        return arms(session, args)
    if args.solves:
        solves(session, args)
    else:
        stages(session, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
