#!/usr/bin/env python3
"""Put one real v3 pool's tick-arcs into a real graph, and see what it costs.

The tick decomposition adds K parallel arcs between a single pair of nodes,
which is a different growth mode from adding K more pools: parallel arcs share
both endpoints, so `laplacian` -- sized by *nodes* -- does not grow at all, and
`np.add.at` simply sums their conductances into the same entries.  What can
grow is the active set and the pivot count, and those are what this measures.

Injected at `pipeline.build` so the arcs travel the whole pipeline -- scale,
seed, solve, candidates, realise -- rather than a bench harness's idea of it.

    uv run python scripts/prototype_univ3_in_graph.py [--amount 100000]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from erouter.chain import chains as chain_table
from erouter.chain.cache import UniverseCache
from erouter.chain.session import RouterSession
from erouter.core import pipeline
from erouter.core.codec import decode, encode_call
from erouter.core.graph import scale
from erouter.core.nodes import rescale
from erouter.core.seed import seed_subgraph
from erouter.dev import config
from erouter.dev.rpc import BATCH_FLOOR, AsyncTransport, JsonRpcTransport
from erouter.dev.universe import load_pools
from erouter.venues.univ3 import PoolState, Tick, capacity
from erouter.venues.univ3 import arcs as tick_arcs

ROOT = Path(__file__).resolve().parents[1]
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
V3_POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"   # USDC/WETH 0.05%


class _Files:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self, name):
        path = self._root / "data" / name
        return path.read_bytes() if path.exists() else None


def read_pool(rpc, pool: str, block: str, window: int, want: int):
    """`PoolState` and the nearest initialized ticks, both directions."""
    def call(to, data):
        return bytes.fromhex(rpc.fetch(
            "eth_call", [{"to": to, "data": "0x" + data.hex()}, block])[2:])

    slot0 = decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
                   call(pool, encode_call("slot0()")))
    state = PoolState(
        sqrt_price_x96=slot0[0], tick=slot0[1],
        liquidity=int.from_bytes(call(pool, encode_call("liquidity()")), "big"),
        tick_spacing=decode(["int24"], call(pool, encode_call("tickSpacing()")))[0],
        fee=decode(["uint24"], call(pool, encode_call("fee()")))[0],
        decimals0=6, decimals1=18)

    word = (state.tick // state.tick_spacing) >> 8
    words = range(word - window, word + window + 1)
    got = rpc.fetch_multi([("eth_call", [{"to": pool, "data": "0x" + encode_call(
        "tickBitmap(int16)", w).hex()}, block]) for w in words], concurrent=True)
    init: list[int] = []
    for w, raw in zip(words, got, strict=True):
        if isinstance(raw, Exception):
            raise SystemExit(f"bitmap word {w} unreadable: {str(raw)[:60]}")
        bits = int(raw, 16)
        init += [((w << 8) + b) * state.tick_spacing
                 for b in range(256) if bits >> b & 1]

    near = (sorted((t for t in init if t <= state.tick), reverse=True)[:want]
            + sorted(t for t in init if t > state.tick)[:want])
    got = rpc.fetch_multi([("eth_call", [{"to": pool, "data": "0x" + encode_call(
        "ticks(int24)", t).hex()}, block]) for t in sorted(set(near))],
        concurrent=True)
    ticks = []
    for index, raw in zip(sorted(set(near)), got, strict=True):
        if isinstance(raw, Exception):
            raise SystemExit(f"tick {index} unreadable: {str(raw)[:60]}")
        ticks.append(Tick(index, decode(["uint128", "int128"],
                                        bytes.fromhex(raw[2:])[:64])[1]))
    return state, ticks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--amount", type=float, default=100_000)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--banks", default="0,1,4,16,64")
    args = p.parse_args(argv)

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
    asyncio.run(session.set_pair(USDC, WETH))
    nodes = session.nodes

    state, ticks = read_pool(transport, V3_POOL, hex(block), args.window, 70)
    bank = tick_arcs(state, ticks, zero_for_one=True, max_ticks=64)
    print(f"v3 pool {V3_POOL}  tick {state.tick:,}  "
          f"{len(bank)} arc(s), capacity {capacity(bank):,.0f} USDC")

    tau_i, sig_i = nodes.node(USDC), nodes.node(WETH)
    rate_in, rate_out = nodes.rate(USDC), nodes.rate(WETH)
    print(f"nodes: {nodes.node_symbol(tau_i)} [{tau_i}] -> "
          f"{nodes.node_symbol(sig_i)} [{sig_i}]\n")

    # Capture what `_assemble` hands `build`, then drive the solve directly.
    # Injecting into the arrays alone is not enough for the full pipeline --
    # `_assemble` realigns its Python arc list to whatever `build` kept -- and
    # the cost being measured here is the solver's, not realisation's.
    real_build = pipeline.build
    captured: dict = {}

    def capture(tau, sig, a, B, nu, Psi, **kw):
        captured.update(tau=tau, sig=sig, a=a, B=B, nu=nu, Psi=Psi, kw=kw)
        return real_build(tau, sig, a, B, nu, Psi, **kw)

    pipeline.build = capture
    session.quote(int(args.amount * 10 ** nodes.decimals(USDC)))
    pipeline.build = real_build
    if not captured:
        raise SystemExit("no graph captured")

    src_node, dst_node = nodes.node(USDC), nodes.node(WETH)
    nu = captured["nu"]
    Psi = captured["Psi"]

    print(f"{'v3 arcs':>8}{'graph arcs':>12}{'nodes kept':>12}{'active':>8}"
          f"{'solve ms':>10}{'pivots':>9}{'loss bp':>12}{'v3 share':>12}")
    for k in (int(v) for v in args.banks.split(",")):
        chosen = bank[:k]
        tau, sig = captured["tau"], captured["sig"]
        a, B = captured["a"], captured["B"]
        kw = dict(captured["kw"])
        if chosen:
            n = len(chosen)
            ca, cb, cc = [], [], []
            for arc in chosen:
                a_c, b_c = rescale(arc.a, arc.B, rate_in, rate_out)
                ca.append(a_c)
                cb.append(b_c)
                cc.append(float(nu[src_node]) * arc.cap * rate_in)
            tau = np.concatenate([tau, np.full(n, src_node, np.int64)])
            sig = np.concatenate([sig, np.full(n, dst_node, np.int64)])
            a = np.concatenate([a, np.array(ca)])
            B = np.concatenate([B, np.array(cb)])
            for name, extra in (("cap", np.array(cc)),
                                ("flagged", np.zeros(n, bool)),
                                ("clamped", np.zeros(n, bool))):
                if kw.get(name) is not None:
                    kw[name] = np.concatenate([np.asarray(kw[name]), extra])

        g = real_build(tau, sig, a, B, nu, Psi, **kw)
        g, psi_scaled = scale(g, Psi)
        seed = seed_subgraph(g, src_node, dst_node, k=8)
        t0 = time.perf_counter()
        report = pipeline.solve(g, src_node, dst_node, psi_scaled,
                                seed=seed, max_rounds=6)
        solve_ms = (time.perf_counter() - t0) * 1e3
        A = np.asarray(report.solution.A)
        live = np.flatnonzero(A)
        kept = len(set(g.tau[live]) | set(g.sig[live])) if live.size else 0
        # In bp, the way `pipeline` reports it: the objective is in scaled
        # units and `scale` normalises by median(G), which the injected arcs
        # move -- so the raw number is not comparable row to row.  This is the
        # whole objective, circulation included, so it says the optimum
        # improved and not what a route would have paid.
        loss_bp = report.solution.objective(g) * g.g_scale / Psi * 1e4
        psi = report.solution.psi
        share = float(psi[-k:].sum() / psi.sum() * 100) if k and psi.sum() else 0.0
        print(f"{k:>8}{g.m:>12,}{kept:>12,}{live.size:>8,}{solve_ms:>10.1f}"
              f"{report.solution.pivots:>9,}{loss_bp:>12.3f}{share:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
