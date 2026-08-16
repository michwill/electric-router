#!/usr/bin/env python3
"""Quote the same trades through Curve's own solver and through this one.

Curve's solver (the `curve_solver` repo, run as `curve-solver-api`) answers
from a periodically-warmed snapshot rather than from chain head, so the first
thing this does is ask it which block it is on and pin our side there.  Without
that the comparison measures its snapshot age: measured, the snapshot has run
4-11 blocks behind head, worth 1.8-27.7 bp on ETH pairs and always in the
direction of flattering us.  If its block moves mid-run the run is refused
rather than reported.

Two more things the numbers depend on:

* **Wait for it to warm.**  Its bootstrap takes ~58 s and answers HTTP 503
  throughout; a cold solver quotes materially worse and inflates every row.
* **Use the private endpoint.**  The comparison needs `eth_call` against
  arbitrary pools, which the scoped key refuses by design, so this reads
  `networks.py` rather than the committed URL.

Only stable-to-stable and pegged pairs support a claim about routing quality;
the volatile rows are printed because they are worth watching, not because a
few bp there means anything.

    python scripts/compare_curve.py [--api URL] [--gwei N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_API = "http://127.0.0.1:38071/quote"

#: (symbol, address, decimals) for everything the cases below name.
TOKENS = {
    "USDC": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6),
    "USDT": ("0xdac17f958d2ee523a2206206994597c13d831ec7", 6),
    "DAI": ("0x6b175474e89094c44da98b954eedeac495271d0f", 18),
    "crvUSD": ("0xf939e0a03fb07f59a73314e73794be0e57ac1b4e", 18),
    "WETH": ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18),
    "stETH": ("0xae7ab96520de3a18e5e111b5eaab095312d7fe84", 18),
    "wstETH": ("0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0", 18),
    "rETH": ("0xae78736cd615f374d3085123a210448e74fc6393", 18),
    "sDOLA": ("0xb45ad160634c528cc3d2926d9807104fa3157305", 18),
    "sDAI": ("0x83f20f44975d03b1b09e64809b757c47f942beea", 18),
    "sUSDS": ("0xa3931d71877c0e7a3148cb7eb4463524fec27fbd", 18),
    "3Crv": ("0x6c3f90f043a72fa612cbac8115ee7e52bde6e490", 18),
    "GUSD": ("0x056fd409e1d7a124bd7017459dfea2f387b6d5cd", 2),
}

#: `stale` marks a pair whose price moves between the solver's snapshot and
#: chain head, so its row measures freshness as much as routing.
CASES = [
    ("USDC", "USDT", 100_000, False),
    ("USDC", "USDT", 5_000_000, False),
    ("DAI", "USDC", 100_000, False),
    ("USDC", "crvUSD", 250_000, False),
    ("crvUSD", "sDOLA", 2_000_000, False),
    ("crvUSD", "sDOLA", 5_000_000, False),
    ("USDC", "sDAI", 1_000_000, False),
    ("USDC", "sUSDS", 1_000_000, False),
    ("USDC", "3Crv", 1_000_000, False),
    ("USDC", "GUSD", 10_000, False),
    ("stETH", "WETH", 50, True),
    ("wstETH", "WETH", 50, True),
    ("rETH", "WETH", 50, True),
    ("USDC", "WETH", 100_000, True),
    ("WETH", "USDC", 30, True),
]


def ask_curve(api: str, src: str, dst: str, wei: int) -> tuple[dict, float]:
    body = json.dumps({
        "input_token": TOKENS[src][0], "output_token": TOKENS[dst][0],
        "amount_in": str(wei), "exact": True,
    }).encode()
    request = urllib.request.Request(
        api, data=body, headers={"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as resp:
        payload = json.loads(resp.read())
    return payload, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--gwei", type=float, default=None,
                        help="gas price; default: whatever the solver quotes")
    args = parser.parse_args()

    from erouter.core.pipeline import RoutingError, prepare, route
    from erouter.dev import chains, config
    from erouter.dev.boa_host import quoter_client
    from erouter.dev.probe_cache import CachedQuoterClient
    from erouter.dev.rpc import JsonRpcTransport
    from erouter.dev.universe import (
        load_pools,
        read_balances,
        resolve_dialects,
        resolve_lp_tokens,
    )
    from erouter.dev.wrappers import build_node_map, build_stake_arcs

    # Its block, not ours: a probe quote reports the snapshot it answered from.
    probe, _ = ask_curve(args.api, "USDC", "USDT", 100 * 10**6)
    block = int(probe["snapshot_block"])
    gwei = args.gwei if args.gwei is not None else float(probe["gas_price_gwei"])
    print(f"  curve_solver snapshot block {block:,}, gas {gwei:.4f} gwei")

    chain = chains.get("ethereum")
    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), block=block,
                           chain_id=chain.chain_id)
    client = CachedQuoterClient(quoter_client(rpc, chain), chain.chain_id, block)
    load = load_pools(chain, min_tvl=10_000.0)
    resolve_dialects(load.pools, client, chain)
    read_balances(load.pools, client, None, chain.chain_id)
    resolve_lp_tokens(load.pools, client, chain.chain_id)
    nodes, _ = build_node_map(load.pools, chain, client)
    stake = build_stake_arcs(nodes, chain, client)
    print(f"  our universe {len(load.pools)} pools at the same block\n")

    header = (f"  {'case':<24}{'curve_solver':>18}{'electric':>18}{'diff':>11}"
              f"{'legs t/e':>10}{'ms t/e':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    better = tied = worse = 0
    stale_rows: list[str] = []
    for src, dst, amount, stale in CASES:
        wei = int(amount * 10 ** TOKENS[src][1])
        decimals = TOKENS[dst][1]
        label = f"{src}->{dst} {amount:,}" + (" *" if stale else "")
        try:
            theirs, their_ms = ask_curve(args.api, src, dst, wei)
        except Exception as exc:
            print(f"  {label:<24} curve_solver: {str(exc)[:44]}")
            continue
        if int(theirs["snapshot_block"]) != block:
            print(f"\n  ! the solver moved to block {theirs['snapshot_block']} "
                  f"mid-run; refusing to report a comparison across two blocks")
            return 1
        their_out = int(theirs.get("expected_out") or 0)
        their_legs = int(theirs.get("legs") or 0)

        prepare(load.pools, nodes, client, src_token=TOKENS[src][0],
                dst_token=TOKENS[dst][0], extra_arcs=stake)
        started = time.perf_counter()
        try:
            result = route(load.pools, nodes, client, src_token=TOKENS[src][0],
                           dst_token=TOKENS[dst][0], amount_in=wei,
                           extra_arcs=stake, gas_price_wei=int(gwei * 1e9))
        except RoutingError as exc:
            print(f"  {label:<24} ours: RoutingError {str(exc)[:40]}")
            continue
        our_ms = (time.perf_counter() - started) * 1000
        our_out = result.verified_out or 0
        our_legs = len(result.route.legs)
        bp = (our_out / their_out - 1) * 1e4 if their_out else 0.0
        if not stale:
            better += bp > 1
            tied += -1 <= bp <= 1
            worse += bp < -1
        else:
            stale_rows.append(f"{label} {bp:+.1f} bp")
        print(f"  {label:<24}{their_out / 10**decimals:>18,.6f}"
              f"{our_out / 10**decimals:>18,.6f}{bp:>+10.1f}bp"
              f"{f'{their_legs}/{our_legs}':>10}"
              f"{f'{their_ms:.0f}/{our_ms:.0f}':>12}")

    print(f"\n  stale-insensitive rows: {better} better, {tied} tied (<1 bp), "
          f"{worse} worse")
    print("  * marks a pair whose price moves between the solver's snapshot "
          "and head;\n    those rows measure freshness as much as routing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
