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

from erouter.core.codec import encode_call
from erouter.core.quoter import QuoterClient
from erouter.core.transport import Call
from erouter.dev import chains as chain_table
from erouter.dev.cli import _rpc_url
from erouter.dev.lite import LITE_MIN_TVL
from erouter.dev.rpc import JsonRpcTransport
from erouter.dev.universe import load_pools

#: Trade size as a fraction of the source coin's reserve.  Small enough that
#: any pool can serve it, large enough to be a trade rather than a dust probe.
SIZE_OF_RESERVE = 0.001
TIMEOUT_S = 600

#: Named cases, on top of the pair discovered per chain.
#
# The discovered pair is one hop through the deepest pool at a thousandth of
# its reserve, which is a liveness check and nothing more.  It cannot catch a
# regression in *routing*, and did not: enumerating LP arcs for every pool cost
# crvUSD -> sDOLA at $2M twenty per cent of its output while every row here
# stayed green, because none of them was big enough to need more than one pool.
#
# So each entry names a trade whose answer depends on the router choosing well.
# Keyed by chain; a symbol that does not resolve on the day is skipped rather
# than failing the sweep, since the universes move.
NAMED_CASES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "ethereum": (
        ("crvUSD", "sDOLA", "2000000"),    # multi-hop, and the regression above
        ("USDC", "WETH", "1000000"),       # the deepest pair on the chain
        ("USDC", "GUSD", "10000"),         # 2-decimal output, via a 3Crv deposit
    ),
    "gnosis": (
        ("XDAI", "EURe", "1000"),          # native in, LP deposit mid-route
    ),
}


def reserves(rpc, chain, pool) -> list[int]:
    """What each coin contract says the pool holds, asked through the quoter.

    `PoolSpec.balances` is empty at load time -- it is filled by the probe
    stage, which is exactly the machinery this script is trying not to build --
    and sizing every chain at "0.001 token" instead would be dust in one pool
    and the whole reserve in another.

    These went out as direct `eth_call`s until the scoped endpoint became the
    default and answered every one of them HTTP 403, which is what it is for.
    Routing them through the quoter's `raw_batch` is what the router itself
    does, so the harness now needs exactly the rights production needs -- and
    would catch it if that stopped being true.
    """
    calls = [Call(coin.address, encode_call("balanceOf(address)", pool.address))
             for coin in pool.coins]
    answers = QuoterClient(rpc, chain.quoter).raw(calls)
    return [answer.uint_or(0) or 0 for answer in answers]


def pick_pair(chain, floor: float, rpc):
    """The deepest pool's two largest coins, and a size its reserve can serve."""
    load = load_pools(chain, min_tvl=floor)
    for pool in sorted(load.pools, key=lambda p: -p.tvl_usd)[:5]:
        if len(pool.coins) < 2:
            continue
        held = pool.balances or reserves(rpc, chain, pool)
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


def named(name: str, chain, floor: float, src: str, dst: str, amount: str) -> dict:
    """One case from `NAMED_CASES`, run through the CLI like any other."""
    out = Path(tempfile.mkdtemp()) / "route.json"
    cmd = ["uv", "run", "erouter", "route", "--chain", name,
           "--from", src, "--to", dst, "--amount", amount,
           "--min-tvl", str(floor), "--json", str(out)]
    started = time.perf_counter()
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"chain": name, "status": "timeout", "pair": f"{src}->{dst}"}
    row = {"chain": name, "pair": f"{src}->{dst} {amount}",
           "ms": (time.perf_counter() - started) * 1000, "exit": done.returncode}
    if not out.exists():
        tail = [ln for ln in done.stdout.splitlines() + done.stderr.splitlines() if ln.strip()]
        row.update(status="failed", note=tail[-1][:70] if tail else "no output")
        return row
    result = json.loads(out.read_text())
    res = result.get("result", {})
    impact = res.get("price_impact") or {}
    row.update(status="ok" if res.get("amount_out") else "no output",
               out=res.get("amount_out_human") or res.get("amount_out"),
               verified=bool(res.get("verified")),
               certificate=bool(res.get("certificate")),
               legs=len(result.get("legs") or []),
               impact_bp=impact.get("bp"))
    return row


def main(argv: list[str]) -> int:
    wanted = argv or list(chain_table.CHAINS)
    rows = []
    for name in wanted:
        chain = chain_table.CHAINS[name]
        floor = LITE_MIN_TVL if chain.lite else 10_000.0
        try:
            rpc = JsonRpcTransport(_rpc_url(chain, argparse.Namespace(rpc=None)),
                                   chain_id=chain.chain_id)
            row = route(name, chain, floor, rpc)
        except Exception as exc:  # a chain that cannot even list its pools
            row = {"chain": name, "status": "error", "note": str(exc)[:70]}
        rows.append(row)
        for src, dst, amount in NAMED_CASES.get(name, ()):
            try:
                rows.append(named(name, chain, floor, src, dst, amount))
            except Exception as exc:
                rows.append({"chain": name, "status": "error",
                             "pair": f"{src}->{dst}", "note": str(exc)[:70]})
        for shown in rows[-1 - len(NAMED_CASES.get(name, ())):]:
            print(f"  {shown['chain']:<11} {shown.get('status','?'):<9} "
                  f"{shown.get('pair','')[:30]:<30} {shown.get('ms', 0):>7.0f} ms  "
                  f"{shown.get('note','')}", flush=True)

    print(f"\n{'chain':<11} {'status':<9} {'pair':<30} {'legs':>4} {'pools':>6} "
          f"{'ver':>4} {'cert':>5} {'impact':>7} {'ms':>7}")
    print("-" * 92)
    for row in rows:
        impact = row.get("impact_bp")
        shown = f"{impact:.2f}" if impact is not None else "-"
        print(f"{row['chain']:<11} {row.get('status','?'):<9} {row.get('pair','')[:30]:<30} "
              f"{row.get('legs', 0):>4} {row.get('pools') or 0:>6} "
              f"{'y' if row.get('verified') else '-':>4} "
              f"{'y' if row.get('certificate') else '-':>5} {shown:>7} "
              f"{row.get('ms', 0):>7.0f}")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n{ok}/{len(rows)} chains routed")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
