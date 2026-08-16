"""Route a real pair on every chain, through the CLI, and report what happened.

"Supported" is a claim about a chain's endpoint, its API and its pools all
working together, and the only way to hold it is to route.  So this picks a
pair the chain actually has -- the two deepest coins of its deepest pool,
sized to a fraction of the reserve so impact stays sane -- and shells out to
`erouter route` exactly as a user would, one chain at a time.

It drives the CLI rather than the library on purpose.  Everything this is meant
to catch (a missing `--chain` entry, a symbol that will not resolve, an
argparse crash, a chain with no state cache) lives in the CLI and is invisible
to a library-level test.

    uv run python scripts/route_sweep.py [chain ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erouter.core.codec import selector
from erouter.dev import chains as chain_table
from erouter.dev.cli import _rpc_url
from erouter.dev.lite import LITE_MIN_TVL
from erouter.dev.rpc import JsonRpcTransport
from erouter.dev.universe import load_pools

#: Trade size as a fraction of the source coin's reserve.  Small enough that
#: any pool can serve it, large enough to be a trade rather than a dust probe.
SIZE_OF_RESERVE = 0.001
TIMEOUT_S = 600


def reserves(rpc, pool) -> list[int]:
    """What each coin contract says the pool holds.

    `PoolSpec.balances` is empty at load time -- it is filled by the probe
    stage, which is exactly the machinery this script is trying not to build --
    and sizing every chain at "0.001 token" instead would be dust in one pool
    and the whole reserve in another.  A plain `balanceOf` per coin needs only
    the transport, and reads the ETH sentinel as zero, which ranks it out.
    """
    calls = [("eth_call", [{"to": coin.address,
                            "data": "0x" + selector("balanceOf(address)").hex()
                                    + pool.address[2:].rjust(64, "0")},
                           rpc.pin.hex_block])
             for coin in pool.coins]
    out = []
    for answer in rpc.fetch_multi(calls):
        try:
            out.append(int(answer, 16) if isinstance(answer, str) and len(answer) > 2 else 0)
        except (TypeError, ValueError):
            out.append(0)
    return out


def pick_pair(chain, floor: float, rpc):
    """The deepest pool's two largest coins, and a size its reserve can serve."""
    load = load_pools(chain, min_tvl=floor)
    for pool in sorted(load.pools, key=lambda p: -p.tvl_usd)[:5]:
        if len(pool.coins) < 2:
            continue
        held = pool.balances or reserves(rpc, pool)
        ranked = sorted(
            ((bal, coin) for bal, coin in zip(held, pool.coins, strict=False) if bal > 0),
            key=lambda pair: -(pair[0] / 10 ** pair[1].decimals),
        )
        if len(ranked) < 2:
            continue
        (src_bal, src), (_, dst) = ranked[0], ranked[1]
        amount = src_bal * SIZE_OF_RESERVE / 10 ** src.decimals
        if amount <= 0:
            continue
        return pool, src, dst, amount
    return None, None, None, 0.0


def route(name: str, chain, floor: float, rpc) -> dict:
    pool, src, dst, amount = pick_pair(chain, floor, rpc)
    if pool is None:
        return {"chain": name, "status": "no pair", "note": "no pool with two funded coins"}
    out = Path(tempfile.mkdtemp()) / "route.json"
    cmd = ["uv", "run", "erouter", "route",
           "--chain", name, "--from", src.address, "--to", dst.address,
           "--amount", f"{amount:.6f}", "--min-tvl", str(floor), "--json", str(out)]
    started = time.perf_counter()
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"chain": name, "status": "timeout", "pair": f"{src.symbol}->{dst.symbol}"}
    ms = (time.perf_counter() - started) * 1000
    row = {"chain": name, "pair": f"{src.symbol}->{dst.symbol}",
           "amount": amount, "ms": ms, "exit": done.returncode}
    if not out.exists():
        tail = [ln for ln in done.stdout.splitlines() + done.stderr.splitlines() if ln.strip()]
        row.update(status="failed", note=tail[-1][:70] if tail else "no output")
        return row
    result = json.loads(out.read_text())
    res = result.get("result", {})
    diag = result.get("diagnostics", {})
    impact = res.get("price_impact") or {}
    row.update(status="ok" if res.get("amount_out") else "no output",
               out=res.get("amount_out_human") or res.get("amount_out"),
               verified=bool(res.get("verified")),
               certificate=bool(res.get("certificate")),
               legs=len(result.get("legs") or []),
               pools=diag.get("pools") or diag.get("arcs_calibrated"),
               impact_bp=impact.get("bp"))
    return row


def main(argv: list[str]) -> int:
    wanted = argv or list(chain_table.CHAINS)
    rows = []
    for name in wanted:
        chain = chain_table.CHAINS[name]
        floor = LITE_MIN_TVL if chain.lite else 10_000.0
        try:
            rpc = JsonRpcTransport(_rpc_url(chain, argparse.Namespace(rpc=None)))
            row = route(name, chain, floor, rpc)
        except Exception as exc:  # a chain that cannot even list its pools
            row = {"chain": name, "status": "error", "note": str(exc)[:70]}
        rows.append(row)
        print(f"  {row['chain']:<11} {row.get('status','?'):<9} "
              f"{row.get('pair','')[:26]:<26} {row.get('ms', 0):>7.0f} ms  "
              f"{row.get('note','')}", flush=True)

    print(f"\n{'chain':<11} {'status':<9} {'pair':<26} {'legs':>4} {'pools':>6} "
          f"{'ver':>4} {'cert':>5} {'impact':>7} {'ms':>7}")
    print("-" * 92)
    for row in rows:
        impact = row.get("impact_bp")
        shown = f"{impact:.2f}" if impact is not None else "-"
        print(f"{row['chain']:<11} {row.get('status','?'):<9} {row.get('pair','')[:26]:<26} "
              f"{row.get('legs', 0):>4} {row.get('pools') or 0:>6} "
              f"{'y' if row.get('verified') else '-':>4} "
              f"{'y' if row.get('certificate') else '-':>5} {shown:>7} "
              f"{row.get('ms', 0):>7.0f}")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n{ok}/{len(rows)} chains routed")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
