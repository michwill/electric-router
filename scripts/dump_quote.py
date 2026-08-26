"""Freeze one prepared quote to JSON, so Rust can be handed the same problem.

The point is a like-for-like benchmark: the candidate stage is 40 ms of which
18 is already native, and the only way to know what the rest costs without
Python is to run the identical graph, the identical solves and the identical
realisation on the other side of the boundary.

What goes in the file is the solver's own index space -- post-dust,
post-duplicate-merge -- plus the base solution to warm-start from and the
per-arc metadata `realize` needs to build legs.  Nothing derived: whatever
Rust recomputes has to be recomputed, or the comparison is a lie.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from bench_quote import USDC, WETH, warm


def _mask(a0, m: int):
    """`A0` as a length-m boolean mask, however the caller spelled it."""
    arr = np.asarray(a0)
    if arr.dtype == bool and arr.size == m:
        return arr
    out = np.zeros(m, bool)
    out[arr.astype(np.int64)] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--from", dest="src", default=USDC)
    p.add_argument("--to", dest="dst", default=WETH)
    p.add_argument("--amount", type=int, default=100_000 * 10**6)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--out", default="/tmp/quote.json")
    p.add_argument("--reps", type=int, default=1)
    args = p.parse_args()

    session = warm(args)

    # Every candidate solve, exactly as it was asked -- the warm start, the
    # bans and the pins.  A prototype that guessed at these would be measuring
    # a different problem and reporting it as this one.
    from erouter.core import candidates as cand_mod
    asked: list[dict] = []
    original = cand_mod.active_set_solve

    def recording(g, src, dst, psi_total, **kw):
        a0 = kw.get("A0")
        forbidden = kw.get("forbidden")
        pinned = kw.get("forced_upper")
        asked.append({
            "src": int(src), "dst": int(dst), "psi_total": float(psi_total),
            "a0": (None if a0 is None else
                   np.asarray(_mask(a0, len(g.tau)), bool).astype(int).tolist()),
            "forbidden": (None if forbidden is None else
                          np.asarray(forbidden, bool).astype(int).tolist()),
            "pinned": ([] if not pinned else
                       [[int(k), float(v)] for k, v in pinned.items()]),
            "min_flow": float(kw.get("min_flow", 0.0)),
            "gas_cost": float(kw.get("gas_cost", 0.0)),
            "maxit": int(kw.get("maxit", 600)),
            "partial_ok": bool(kw.get("partial_ok", False)),
        })
        return original(g, src, dst, psi_total, **kw)

    cand_mod.active_set_solve = recording
    try:
        result = session.quote(args.amount)
    finally:
        cand_mod.active_set_solve = original
    scratch = getattr(result, "scratch", None) or result
    g = getattr(scratch, "graph", None) or getattr(result, "graph", None)
    if g is None:
        # The pipeline keeps it on the report; fall back to the arcs it kept.
        print("no ArcArrays on the result; dumping what is reachable",
              file=sys.stderr)

    arcs = result.arcs
    report = getattr(result, "report", None)
    solution = getattr(report, "solution", None) if report is not None else None
    payload: dict = {
        "block": session.block,
        "amount_in": args.amount,
        "n_arcs": len(arcs),
        "arcs": [
            {
                "id": a.id, "pool": a.pool, "kind": int(a.kind),
                "i": a.i, "j": a.j, "n_coins": a.n_coins,
                "token_in": a.token_in, "token_out": a.token_out,
                "tau": a.tau, "sigma": a.sigma,
                "G": float(a.G), "eps": float(a.eps),
                "cap": (None if not np.isfinite(a.cap) else float(a.cap)),
                "a": float(a.a), "B": float(a.B),
            }
            for a in arcs
        ],
    }
    if g is not None:
        payload["graph"] = {
            "tau": np.asarray(g.tau, np.int64).tolist(),
            "sig": np.asarray(g.sig, np.int64).tolist(),
            "G": np.asarray(g.G, float).tolist(),
            "eps": np.asarray(g.eps, float).tolist(),
            "cap": [None if not np.isfinite(c) else float(c)
                    for c in np.asarray(g.cap, float)],
            "n_nodes": int(g.n_nodes),
            "g_scale": float(g.g_scale),
        }
    if solution is not None:
        psi = np.asarray(solution.psi, float)
        payload["base_psi"] = psi.tolist()
        if g is not None:
            # The terminals and the flow total, recovered from the solution
            # rather than guessed: `Psi` is the net value leaving the source in
            # the solver's own scaled units, which is what a re-solve wants.
            nodes = result.nodes
            src_node = int(nodes.node(result.src_token))
            dst_node = int(nodes.node(result.dst_token))
            tau = np.asarray(g.tau, np.int64)
            sig = np.asarray(g.sig, np.int64)
            out = float(psi[tau == src_node].sum())
            back = float(psi[sig == src_node].sum())
            payload["src_node"] = src_node
            payload["dst_node"] = dst_node
            payload["psi_total"] = out - back

    cands = getattr(result, "candidates", None)
    if cands is not None:
        payload["candidates"] = [
            {"label": c.label, "psi": np.asarray(c.psi, float).tolist()}
            for c in cands.candidates
        ]
        payload["solves"] = int(cands.solves)

    payload["solve_calls"] = asked
    Path(args.out).write_text(json.dumps(payload))
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out}  {size / 1e6:.2f} MB  ·  {len(arcs)} arcs"
          f"  ·  {len(payload.get('candidates') or [])} candidates"
          f"  ·  {payload.get('solves', '?')} solves"
          f"  ·  {len(asked)} recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
