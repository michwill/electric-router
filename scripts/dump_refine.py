"""Freeze the `refine` stage's inputs and its answers, for a differential.

The stage is `plan_sized -> probe -> collect -> merge -> _recalibrate ->
_assemble -> scale -> seed_subgraph`, and its named functions account for 14.6
of its 15.4 ms -- unlike `direct`, there is almost no inline glue, so porting
the functions *is* moving the stage.

The design the port wants is the ladders staying on the Rust side for the
whole stage. `_recalibrate` currently marshals 450 float lists across on every
quote through `Ladder.as_float`, and that is now the larger half of what it
costs; if the ladders never cross, that disappears rather than getting faster.

What is recorded here is everything crossing the stage boundary, and every
number it writes: the arcs before and after, the graph arrays it assembles,
and the seed. Floats cross as exact bit patterns -- decimal text loses a ULP
through serde_json, and `calibrate` runs divided differences that amplify one
ULP into 2.7e-7 relative, so a differential shipping decimals cannot tell a
real disagreement from its own transport.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from bench_quote import USDC, WETH, warm

from erouter.core import pipeline


def bits(value) -> str:
    """A float as its exact bit pattern. See the module docstring."""
    return str(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def arc_row(arc) -> dict:
    return {
        "id": arc.id, "tau": int(arc.tau), "sigma": int(arc.sigma),
        "a": bits(arc.a), "B": bits(arc.B), "cap": bits(arc.cap),
        "G": bits(arc.G), "eps": bits(arc.eps),
        "clamped": bool(arc.clamped), "convex_flag": bool(arc.convex_flag),
        "flag_reason": str(arc.flag_reason.value),
        "drift": bits(arc.drift), "eta": bits(arc.eta),
        "calib_delta": bits(arc.calib_delta),
        "decimals_in": int(arc.decimals_in), "token_in": arc.token_in,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--from", dest="src", default=USDC)
    p.add_argument("--to", dest="dst", default=WETH)
    p.add_argument("--amount", type=int, default=100_000 * 10**6)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--out", default="/tmp/refine.json")
    args = p.parse_args()

    session = warm(args)
    captured: dict = {}

    # Every stage function, with what it was handed and what it produced.
    plan_sized, collect, merge = pipeline.plan_sized, pipeline.collect, pipeline.merge
    recalibrate, assemble = pipeline._recalibrate, pipeline._assemble
    subgraph = pipeline.seed_subgraph

    def watched_plan(ladders, sizes):
        captured["sizes"] = {k: [str(v) for v in vs] for k, vs in sizes.items()}
        captured["ladders_before"] = [
            {"id": lad.arc.id,
             "deltas": [str(d) for d in lad.deltas],
             "quotes": [str(q) for q in lad.quotes],
             "decimals_in": int(lad.arc.decimals_in),
             "decimals_out": int(lad.arc.decimals_out),
             "reserve_in": str(lad.arc.reserve_in)}
            for lad in ladders]
        got = plan_sized(ladders, sizes)
        captured["planned"] = [
            {"pool": pr.pool, "kind": int(pr.kind), "i": pr.i, "j": pr.j,
             "n": pr.n, "dx": str(pr.dx)} for pr in got.probes]
        return got

    def watched_recal(arcs, ladders, nodes):
        captured["arcs_before"] = [arc_row(a) for a in arcs]
        captured["rates"] = {a.token_in: bits(nodes.rate(a.token_in)) for a in arcs}
        captured["rates"].update(
            {a.token_out: bits(nodes.rate(a.token_out)) for a in arcs})
        captured["ladders_after"] = [
            {"id": lad.arc.id,
             "deltas": [str(d) for d in lad.deltas],
             "quotes": [str(q) for q in lad.quotes]}
            for lad in ladders]
        got = recalibrate(arcs, ladders, nodes)
        captured["arcs_recalibrated"] = [arc_row(a) for a in arcs]
        return got

    def watched_assemble(arcs, nu, Psi, nodes, src_node, dst_node, result):
        captured["nu"] = [bits(v) for v in np.asarray(nu, float)]
        captured["Psi"] = bits(Psi)
        captured["src_node"], captured["dst_node"] = int(src_node), int(dst_node)
        out_arcs, g = assemble(arcs, nu, Psi, nodes, src_node, dst_node, result)
        captured["arcs_after"] = [arc_row(a) for a in out_arcs]
        captured["graph"] = {
            "tau": [int(v) for v in np.asarray(g.tau)],
            "sig": [int(v) for v in np.asarray(g.sig)],
            "G": [bits(v) for v in np.asarray(g.G, float)],
            "eps": [bits(v) for v in np.asarray(g.eps, float)],
            "cap": [bits(v) for v in np.asarray(g.cap, float)],
            "n_nodes": int(g.n_nodes),
            "sources": [list(map(int, s)) for s in g.sources],
            "dropped": {str(k): v for k, v in g.dropped.items()},
        }
        return out_arcs, g

    def watched_subgraph(g, src, dst, *, k):
        got = subgraph(g, src, dst, k=k)
        captured.setdefault("seeds", []).append(
            {"k": int(k), "mask": [int(bool(v)) for v in np.asarray(got)]})
        return got

    pipeline.plan_sized = watched_plan
    pipeline._recalibrate = watched_recal
    pipeline._assemble = watched_assemble
    pipeline.seed_subgraph = watched_subgraph
    try:
        asyncio.run(session.set_pair(args.src, args.dst))
        session.quote(args.amount)          # settle
        captured.clear()
        session.quote(args.amount)
    finally:
        pipeline.plan_sized, pipeline.collect, pipeline.merge = plan_sized, collect, merge
        pipeline._recalibrate, pipeline._assemble = recalibrate, assemble
        pipeline.seed_subgraph = subgraph

    if not math.isfinite(1.0):        # pragma: no cover - keeps math imported
        raise SystemExit(1)
    Path(args.out).write_text(json.dumps(captured))
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out}  {size / 1e6:.2f} MB")
    print(f"  ladders {len(captured.get('ladders_before') or [])}"
          f" · planned probes {len(captured.get('planned') or [])}"
          f" · arcs {len(captured.get('arcs_before') or [])}"
          f" -> {len(captured.get('arcs_after') or [])}"
          f" · seeds {len(captured.get('seeds') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
