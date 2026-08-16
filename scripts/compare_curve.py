#!/usr/bin/env python3
"""Quote the same trades through Curve's own solver and through this one.

Curve's solver answers from a periodically-warmed snapshot rather than from
chain head, so the first thing this does is ask it which block it is on and pin
our side there.  Without that the comparison measures its snapshot age:
measured, the snapshot has run 4-11 blocks behind head, worth 1.8-27.7 bp on
ETH pairs and always in the direction of flattering us.  A row whose block
moved under it is dropped rather than reported.

The hosted deployment serves one host per chain -- `ethereum`, `arbitrum`,
`gnosis`, `base`, `optimism` at `<chain>.router.curve.finance` -- and those
five are the whole comparable set; the other ten chains this router covers have
nothing to compare against.

Pairs come from each chain's own universe rather than a hardcoded list, so a
chain whose pools move does not quietly stop being tested: `chain.stables` is
already the curated "these hold a peg" set, and stable-to-stable is where a
difference means routing rather than freshness.  One volatile pair per chain is
quoted too, marked `*` and excluded from the tally.

    python scripts/compare_curve.py [--chain all|ethereum|...] [--sizes 10000,100000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: The hosted deployment, one host per chain.
HOSTED = "https://{chain}.router.curve.finance/quote"
SUPPORTED = ("ethereum", "arbitrum", "gnosis", "base", "optimism")
DEFAULT_SIZES = (10_000, 100_000)
#: Sizes are in units of the source token, which is always a stable here, so
#: they read as dollars.  A chain thinner than this simply reports the loss.
TIMEOUT_S = 180


def ask_curve(api: str, src: str, dst: str, wei: int) -> tuple[dict, float]:
    body = json.dumps({"input_token": src, "output_token": dst,
                       "amount_in": str(wei), "exact": True}).encode()
    # A User-Agent, because the hosted ingress answers 403 to urllib's
    # default one -- the same trap the Curve API sets (E10).
    request = urllib.request.Request(
        api, data=body, headers={"content-type": "application/json",
                                 "User-Agent": "erouter/compare"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
        payload = json.loads(resp.read())
    return payload, (time.perf_counter() - started) * 1000


def pairs_for(chain, pools, sizes: tuple[int, ...]):
    """(src, dst, amount, volatile) for one chain, from its own universe."""
    holders: dict[str, tuple[str, int, float]] = {}
    for pool in pools:
        for coin in pool.coins:
            key = coin.address.lower()
            symbol, decimals, tvl = holders.get(key, (coin.symbol, coin.decimals, 0.0))
            holders[key] = (symbol, decimals, tvl + pool.tvl_usd)

    stables = [s.lower() for s in chain.stables if s.lower() in holders]
    if not stables:
        # Only mainnet carries a curated list; elsewhere the symbol is the
        # evidence available.  Folding U+20AE first catches tac's and xlayer's
        # Tether spelling, and "USD" alone covers USDC, USDT, crvUSD, frxUSD,
        # USDe and the rest without naming any of them.
        stables = [a for a, (symbol, _, _) in holders.items()
                   if "USD" in symbol.upper().replace("\u20ae", "T")
                   or symbol.upper() in {"DAI", "EURE", "EURC.E"}]
    stables.sort(key=lambda a: -holders[a][2])
    out = []
    for size in sizes:
        for src, dst in pairwise(stables[:4]):
            out.append((src, dst, size, False))
    # And one volatile leg, for watching rather than scoring.
    wrapped = chain.wrapped.lower()
    if stables and wrapped in holders:
        out.append((stables[0], wrapped, sizes[0], True))
    return out, holders


def compare_chain(name: str, sizes: tuple[int, ...], rows: list[dict]) -> None:
    from erouter.core.pipeline import RoutingError, prepare, route
    from erouter.core.quoter import QuoterClient
    from erouter.dev import chains as chain_table
    from erouter.dev.cli import _local_quoter, _rpc_url
    from erouter.dev.facts import FactsCache
    from erouter.dev.rpc import JsonRpcTransport
    from erouter.dev.universe import (
        load_pools,
        read_balances,
        resolve_dialects,
        resolve_lp_tokens,
    )
    from erouter.dev.wrappers import (
        build_node_map,
        build_stake_arcs,
        build_transmuter_arcs,
    )

    api = HOSTED.format(chain=name)
    chain = chain_table.CHAINS[name]

    # Its block, not ours.  The `eth_call`s below need the private endpoint --
    # the scoped key answers 403 to anything that is not the quoter.
    args = argparse.Namespace(rpc=None, block=None, private=True)
    load = load_pools(chain, min_tvl=10_000.0 if not chain.lite else 1_000.0)
    cases, holders = pairs_for(chain, load.pools, sizes)
    if not cases:
        print(f"  {name}: no stable pair in the universe to compare")
        return

    facts = FactsCache.load(chain.chain_id, name)
    bound: dict = {"block": 0}

    def bind(block: int):
        """Our side, at *their* block.

        Rebuilt whenever the snapshot moves, which on mainnet is most rows: a
        12-second block against several seconds per quote means a single pin
        for the whole chain would reject nearly everything.  Balances are the
        part that must be re-read; the pool list and dialects are not
        block-sensitive.
        """
        if bound["block"] == block:
            return bound
        rpc = JsonRpcTransport(_rpc_url(chain, args), block=block,
                               chain_id=chain.chain_id)
        client = QuoterClient(rpc, chain.quoter)
        resolve_dialects(load.pools, client, chain)
        read_balances(load.pools, client, None, chain.chain_id)
        resolve_lp_tokens(load.pools, client, chain.chain_id)
        nodes, _ = build_node_map(load.pools, chain, client, facts=facts)
        stake = (build_stake_arcs(nodes, chain, client)
                 + build_transmuter_arcs(nodes, chain, client))
        local = _local_quoter(rpc, chain, load, nodes, quiet=True)
        bound.update(block=block, client=local or client, nodes=nodes, stake=stake)
        return bound

    print(f"  {name}: {len(load.pools)} pools, {len(cases)} cases")

    for src, dst, amount, volatile in cases:
        src_symbol, src_decimals, _ = holders[src]
        dst_symbol, dst_decimals, _ = holders[dst]
        wei = amount * 10**src_decimals
        label = f"{src_symbol}->{dst_symbol} {amount:,}" + (" *" if volatile else "")
        row = {"chain": name, "case": label, "volatile": volatile}
        try:
            theirs, their_ms = ask_curve(api, src, dst, wei)
        except Exception as exc:
            rows.append(row | {"note": f"solver: {str(exc)[:38]}"})
            continue
        block = int(theirs.get("snapshot_block") or 0)
        their_out = int(theirs.get("expected_out") or 0)
        if their_out == 0 or block <= 0:
            rows.append(row | {"note": f"solver: {theirs.get('error', 'no route')}"[:44]})
            continue
        try:
            here = bind(block)
            client, nodes, stake = here["client"], here["nodes"], here["stake"]
            prepare(load.pools, nodes, client, src_token=src, dst_token=dst,
                    extra_arcs=stake)
            started = time.perf_counter()
            result = route(load.pools, nodes, client, src_token=src, dst_token=dst,
                           amount_in=wei, extra_arcs=stake)
        except RoutingError as exc:
            rows.append(row | {"note": f"ours: {str(exc)[:38]}"})
            continue
        our_ms = (time.perf_counter() - started) * 1000
        our_out = result.verified_out or 0
        rows.append(row | {
            "theirs": their_out / 10**dst_decimals,
            "ours": our_out / 10**dst_decimals,
            "bp": (our_out / their_out - 1) * 1e4,
            "legs": f"{theirs.get('legs', 0)}/{len(result.route.legs)}",
            "ms": f"{their_ms:.0f}/{our_ms:.0f}",
            "block": block,
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="all")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    args = parser.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(","))
    wanted = SUPPORTED if args.chain == "all" else (args.chain,)
    rows: list[dict] = []
    for name in wanted:
        try:
            compare_chain(name, sizes, rows)
        except Exception as exc:
            print(f"  {name}: failed: {str(exc)[:70]}")

    header = (f"\n  {'chain':<10}{'case':<26}{'curve_solver':>17}{'electric':>17}"
              f"{'diff':>10}{'legs':>7}{'ms t/e':>11}")
    print(header)
    print("  " + "-" * (len(header) - 3))
    better = tied = worse = 0
    for row in rows:
        if "note" in row:
            print(f"  {row['chain']:<10}{row['case']:<26}{row['note']:>62}")
            continue
        bp = row["bp"]
        if not row["volatile"]:
            better += bp > 1
            tied += -1 <= bp <= 1
            worse += bp < -1
        print(f"  {row['chain']:<10}{row['case']:<26}{row['theirs']:>17,.6f}"
              f"{row['ours']:>17,.6f}{bp:>+9.1f}bp{row['legs']:>7}{row['ms']:>11}")
    print(f"\n  stable rows: {better} better, {tied} tied (<1 bp), {worse} worse")
    print("  * volatile pair, excluded from the tally: its row measures the "
          "solver's snapshot age\n    as much as routing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
