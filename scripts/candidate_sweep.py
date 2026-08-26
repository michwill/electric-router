"""What the marginal candidate is worth, in basis points and milliseconds.

The candidate stage is 40.7 ms of a 95 ms quote, and 17 of that is the solver
running 37 times -- once per candidate family, plus repairs.  A Rust port of
everything around it lands the stage at 17.05 ms, so the solve count is the
floor.  This asks the other question: how much output do the later candidates
actually buy, if any.

Answered per pair, because it is a property of the pair rather than of the
router: a route with one obvious path has nothing for the twelfth candidate to
find, and a branchy one may.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_quote import warm

from erouter.core import pipeline

PAIRS = {
    "USDC>WETH": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                  "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 100_000 * 10**6),
    "USDT>tBTC": ("0xdac17f958d2ee523a2206206994597c13d831ec7",
                  "0x18084fba666a33d37592fa2633fd49a74dd93a88", 200_000 * 10**6),
    "crvUSD>WETH": ("0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",
                    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 500_000 * 10**18),
    "WETH>USDC": ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                  "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 40 * 10**18),
}
BUDGETS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)


def one(session, src, dst, amount, budget, reps):
    """Quote at this budget; return (out, ms, solves, candidates)."""
    best, out, solves, made = 1e9, None, 0, 0
    for _ in range(reps):
        start = time.process_time()
        result = pipeline.route(
            session.pools, session.nodes, session.client,
            src_token=src, dst_token=dst, amount_in=int(amount),
            prepared=session.prepared, extra_arcs=session.stake_arcs,
            max_legs=session.max_legs, gas_price_wei=session.gas_price_wei,
            gas_table=session.gas_table, risk_table=session.risk_table,
            max_candidates=budget,
        )
        best = min(best, (time.process_time() - start) * 1e3)
        out = result.verified_out
        solves = result.counters.get("candidate_solves", 0)
        made = result.counters.get("candidates", 0)
    return out, best, solves, made


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain", default="ethereum")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--min-tvl", type=float, default=10_000.0)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--pair", default=None, help="only this one")
    args = p.parse_args()

    args.src, args.dst = PAIRS["USDC>WETH"][:2]
    args.amount = PAIRS["USDC>WETH"][2]
    session = warm(args)
    print(f"block {session.block:,}  ·  reps {args.reps}\n")

    for name, (src, dst, amount) in PAIRS.items():
        if args.pair and args.pair != name:
            continue
        import asyncio
        asyncio.run(session.set_pair(src, dst))
        rows = []
        for budget in BUDGETS:
            rows.append((budget, *one(session, src, dst, amount, budget, args.reps)))
        best_out = max((r[1] or 0) for r in rows)
        print(f"{name}   best {best_out / 1e18 if best_out > 1e12 else best_out:,.6f}")
        print(f"  {'budget':>7}{'made':>6}{'solves':>8}{'ms':>9}{'bp behind best':>16}")
        for budget, out, ms, solves, made in rows:
            behind = ((best_out - (out or 0)) / best_out * 1e4) if best_out else 0.0
            print(f"  {budget:>7}{made:>6}{solves:>8}{ms:>9.1f}{behind:>15.2f}")
        # Where the last real gain came from.
        gains = [(b, ((rows[k][1] or 0) - (rows[k - 1][1] or 0)) / best_out * 1e4)
                 for k, (b, *_) in enumerate(rows) if k]
        useful = [b for b, g in gains if g > 0.01]
        print(f"  last budget that bought more than 0.01 bp: "
              f"{max(useful) if useful else BUDGETS[0]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
