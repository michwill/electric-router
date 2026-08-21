"""Route a spread of pairs, then execute every one through `ElectricRouter`.

`tests/forked/test_router_execution.py` runs four pairs and asserts.  This runs
many and reports, which is the other half: the assertions cannot tell you what
fraction of the venue actually executes, or which leg kinds have never been
through the contract at all.

The bounds are on.  A route that quotes and then trips its own minimum rate is
the thing worth finding, so nothing here relaxes them.

    uv run python scripts/fork_execute_routes.py --private
"""

from __future__ import annotations

import argparse
import time

from erouter.core.pipeline import RoutingError, route
from erouter.core.pools import parse_universe, volatile_pools
from erouter.core.routecall import EncodingError, encode_route
from erouter.dev import chains as chain_table
from erouter.dev import config
from erouter.dev.boa_host import override_client
from erouter.dev.cli import _token_holders
from erouter.dev.crypto_lp_params import build_exact_crypto_lp
from erouter.dev.curve_api import CurveApi
from erouter.dev.exact_probe import ExactQuoterClient
from erouter.dev.executor import fork
from erouter.dev.lp_params import build_exact_lp
from erouter.dev.probe_cache import CachedQuoterClient
from erouter.dev.router import deploy, send
from erouter.dev.rpc import JsonRpcTransport
from erouter.dev.stable_params import build_exact_pools
from erouter.dev.tricrypto_params import build_exact_tricrypto
from erouter.dev.twocrypto_params import build_exact_twocrypto
from erouter.dev.universe import read_balances, resolve_dialects, resolve_lp_tokens
from erouter.dev.wrappers import build_node_map

#: `(from, to, human amount)`, chosen to reach different leg kinds rather than
#: to flatter the router: native both ways, a wrapper, a vault, an LP token,
#: a currency pair, and the plain swaps.
PAIRS = [
    ("USDC", "WETH", 250_000),
    ("WETH", "USDC", 100),
    ("ETH", "USDC", 100),
    ("USDC", "ETH", 250_000),
    ("USDC", "USDT", 1_000_000),
    ("DAI", "USDC", 500_000),
    ("USDT", "DAI", 250_000),
    ("crvUSD", "WETH", 500_000),
    ("crvUSD", "sDOLA", 500_000),
    ("USDC", "crvUSD", 250_000),
    ("USDC", "wstETH", 100_000),
    ("wstETH", "USDC", 30),
    ("stETH", "USDC", 30),
    ("USDC", "scrvUSD", 100_000),
    ("USDC", "3Crv", 100_000),
    ("3Crv", "USDT", 100_000),
    ("WBTC", "USDC", 2),
    ("USDC", "WBTC", 100_000),
    ("USDC", "EURS", 50_000),
    ("USDC", "tBTC", 100_000),
]


def resolve(pools, symbol: str, chain) -> tuple[str, int]:
    """Symbol -> (address, decimals), by depth, as the CLI resolves it."""
    if symbol.upper() == chain.native_symbol.upper():
        return chain_table.NATIVE_SENTINEL.lower(), 18
    best: dict[str, tuple[str, int, float]] = {}
    for pool in pools:
        for coin in pool.coins:
            key = coin.symbol.upper()
            _, _, tvl = best.get(key, ("", 0, 0.0))
            if tvl < pool.tvl_usd:
                best[key] = (coin.address.lower(), coin.decimals, pool.tvl_usd)
        if pool.lp_token:
            key = pool.name.split()[-1].upper()
            best.setdefault(key, (pool.lp_token.lower(), pool.lp_decimals,
                                  pool.tvl_usd))
    hit = best.get(symbol.upper())
    if hit is None:
        raise KeyError(symbol)
    return hit[0], hit[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="ethereum")
    parser.add_argument("--block", default="latest")
    parser.add_argument("--min-tvl", type=float, default=10_000.0)
    parser.add_argument("--private", action="store_true",
                        help="use networks.py; a fork needs it")
    args = parser.parse_args()

    chain = chain_table.get(args.chain)
    url = config.rpc_url(chain.rpc_attr) if args.private else chain.public_rpc
    rpc = JsonRpcTransport(url, block=args.block, chain_id=chain.chain_id)
    base = CachedQuoterClient(override_client(rpc), rpc.chain_id, rpc.block)

    started = time.monotonic()
    specs = parse_universe(CurveApi().list_pools(chain.chain_id,
                                                 min_tvl=args.min_tvl))
    resolve_dialects(specs, base, chain)
    read_balances(specs, base, None, chain.chain_id, token_client=base)
    resolve_lp_tokens(specs, base, chain.chain_id, token_client=base)
    nodes, _ = build_node_map(specs, chain, base)
    # The LP models matter here: a legacy pool s own calc_token_amount omits
    # the fee add_liquidity charges, so a deposit quoted from the chain is
    # quoted too high and trips its own bound on the way out.
    stable = build_exact_pools(specs, base)
    crypto = build_exact_tricrypto(specs, base)
    with_lp = [p for p in specs if p.lp_token]
    client = ExactQuoterClient(
        base, stable, build_exact_twocrypto(specs, base), crypto,
        lp=build_exact_lp(with_lp, stable, base),
        crypto_lp=build_exact_crypto_lp(with_lp, crypto, base))
    loose = volatile_pools(specs, chain.stables + chain.forex)
    print(f"{chain.name} block {rpc.block:,}, {len(specs)} pools, "
          f"warmed in {time.monotonic() - started:.0f}s\n")

    quoted = {}
    for src_sym, dst_sym, human in PAIRS:
        label = f"{src_sym}->{dst_sym}"
        try:
            src, decimals = resolve(specs, src_sym, chain)
            dst, _ = resolve(specs, dst_sym, chain)
        except KeyError as exc:
            quoted[label] = f"unknown symbol {exc}"
            continue
        try:
            quoted[label] = route(specs, nodes, client, src_token=src,
                                  dst_token=dst,
                                  amount_in=int(human * 10**decimals))
        except RoutingError as exc:
            quoted[label] = f"no route: {exc}"

    fork(url, rpc.block)
    router = deploy()
    print(f"{'pair':<18}{'legs':>5}{'kinds':<34}{'drift bp':>10}{'gas':>11}  verdict")

    ok = failed = skipped = 0
    kinds_seen: set[str] = set()
    for src_sym, dst_sym, _ in PAIRS:
        label = f"{src_sym}->{dst_sym}"
        result = quoted[label]
        if isinstance(result, str):
            print(f"{label:<18}{'':>5}{result[:60]}")
            skipped += 1
            continue
        kinds = sorted({leg.kind.name for leg in result.route.legs})
        kinds_seen |= set(kinds)
        short = ",".join(k.split("_")[0].lower()[:6] for k in kinds)[:32]
        try:
            call = encode_route(result.route, receiver="0x" + "11" * 20,
                                volatile=loose, quoted_out=result.verified_out)
        except EncodingError as exc:
            print(f"{label:<18}{len(result.route.legs):>5}{short:<34}"
                  f"{'':>10}{'':>11}  cannot encode: {exc}")
            failed += 1
            continue
        report = send(call, router=router, quoted_out=result.verified_out,
                      wrapped=chain.wrapped, expect_block=rpc.block,
                      holders=_token_holders(specs, call.token_in,
                                             avoid=result.route.pools_used))
        if not report.ok:
            print(f"{label:<18}{len(result.route.legs):>5}{short:<34}"
                  f"{'':>10}{'':>11}  FAILED: {report.error[:40]}")
            failed += 1
            continue
        ok += 1
        print(f"{label:<18}{len(result.route.legs):>5}{short:<34}"
              f"{report.drift_bp:>+10.4f}{report.gas:>11,}  ok"
              f"{'  ' + '; '.join(report.warnings)[:40] if report.warnings else ''}")

    print(f"\n{ok} executed, {failed} failed, {skipped} not routed")
    print(f"leg kinds exercised: {', '.join(sorted(kinds_seen))}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
