"""`erouter` command line.

Phase 0 ships `doctor`, which turns every environment assumption the design
rests on into a runtime check: state-override support (decides whether the
quoter needs deploying), `debug_traceCall` (decides boa's fork prefetch), batch
support, and the Curve API behind its User-Agent requirement.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import chains as chain_table
from . import config
from .curve_api import CurveApi, CurveApiError
from .rpc import JsonRpcTransport, RpcError

OK = "\x1b[32m✔\x1b[0m"
BAD = "\x1b[31m✘\x1b[0m"
WARN = "\x1b[33m•\x1b[0m"


def _mark(value: bool) -> str:
    return OK if value else BAD


def cmd_doctor(args: argparse.Namespace) -> int:
    names = [args.chain] if args.chain else list(chain_table.CHAINS)
    if not config.have_networks():
        print(f"{BAD} networks.py not found -- copy networks.example.py and fill it in")
        return 4

    failures = 0
    for name in names:
        chain = chain_table.get(name)
        print(f"\n\x1b[1m{chain.name}\x1b[0m (chain_id {chain.chain_id})")
        try:
            url = config.rpc_url(chain.rpc_attr)
        except KeyError as exc:
            print(f"  {WARN} {exc}")
            continue

        started = time.monotonic()
        try:
            rpc = JsonRpcTransport(url, block=args.block)
        except (RpcError, KeyError) as exc:
            print(f"  {BAD} unreachable: {exc}")
            failures += 1
            continue
        latency = (time.monotonic() - started) * 1000

        if rpc.chain_id != chain.chain_id:
            print(f"  {BAD} chain id mismatch: node says {rpc.chain_id}, table says {chain.chain_id}")
            failures += 1

        try:
            client = rpc.fetch("web3_clientVersion", [])
        except RpcError:
            client = "unknown"

        override = rpc.supports_state_override()
        trace = rpc.supports_debug_trace()
        batch = rpc.supports_batching()

        print(f"  {OK} reachable          block {rpc.block:,}  ({latency:.0f} ms)  {client}")
        print(f"  {_mark(override)} state override     " + (
            "quoter runs via eth_call, no deployment"
            if override
            else "quoter must be deployed or run under boa.fork"
        ))
        print(f"  {_mark(trace)} debug_traceCall    " + (
            "prestateTracer served -> boa fork prefetch ON"
            if trace
            else "not served -> boa fork prefetch OFF (use the override path)"
        ))
        print(f"  {_mark(batch)} JSON-RPC batching  " + ("" if batch else "falls back to serial"))
        if not override and not trace:
            failures += 1

    # --- Curve API ---------------------------------------------------------
    print("\n\x1b[1mCurve Prices API\x1b[0m")
    api = CurveApi()
    try:
        known = api.chains()
        print(f"  {OK} reachable          {len(known)} chains: {', '.join(sorted(known))}")
    except CurveApiError as exc:
        print(f"  {BAD} unreachable        {exc}")
        return 4

    for name in names:
        chain = chain_table.get(name)
        if chain.api_name not in known:
            print(f"  {WARN} {chain.name:<10} not served by the API (api_name={chain.api_name!r})")
            continue
        try:
            pools = api.list_pools(chain.chain_id, min_tvl=args.min_tvl)
        except CurveApiError as exc:
            print(f"  {BAD} {chain.name:<10} {exc}")
            failures += 1
            continue
        coins = sum(len(p.get("coins") or []) for p in pools)
        arcs = sum(len(p.get("coins") or []) * (len(p.get("coins") or []) - 1) for p in pools)
        print(
            f"  {OK} {chain.name:<10} {len(pools):>4} pools  "
            f"{coins:>5} coin slots  ~{arcs:>5} swap arcs  (min_tvl ${args.min_tvl:,.0f})"
        )

    return 5 if failures else 0


def cmd_pools(args: argparse.Namespace) -> int:
    """Load the universe and resolve every pool's ABI dialect."""
    from .boa_host import override_client
    from .rpc import JsonRpcTransport
    from .universe import count_swap_arcs, load_pools, resolve_dialects

    chain = chain_table.get(args.chain)
    try:
        load = load_pools(chain, min_tvl=args.min_tvl, refresh=args.refresh,
                          pool_filters=args.pool_filters,
                          llamma=args.llamma)
    except CurveApiError as exc:
        print(f"{BAD} {exc}")
        return 4
    for warning in load.warnings:
        print(f"{WARN} {warning}")

    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), block=args.block)
    client = override_client(rpc)
    audit = resolve_dialects(load.pools, client, chain, use_cache=not args.refresh)

    source = load.source + (f" ({load.age / 60:.0f} min old)" if load.source != "api" else "")
    print(
        f"\n\x1b[1m{chain.name}\x1b[0m  block {rpc.block:,}  "
        f"min_tvl ${args.min_tvl:,.0f}  universe from {source}"
    )
    print(
        f"  {len(load.pools):>4} pools   {count_swap_arcs(load.pools):>5} swap arcs   "
        f"{sum(p.n_coins for p in load.pools):>5} coin slots"
    )
    print(
        f"  dialects: {audit.resolved} resolved "
        f"({audit.from_probe} probed, {audit.from_cache} cached) in {audit.seconds:.2f}s"
    )
    print(f"  {audit.empty_returndata} probes returned EMPTY data (the silent-zero trap)")

    if audit.mistyped:
        print(f"\n  {WARN} {len(audit.mistyped)} pool(s) the API mis-types:")
        for pool, claimed in audit.mistyped:
            print(
                f"      {pool.name[:34]:<34} {pool.address}  "
                f"type={pool.pool_type} says {claimed.name}, answers {pool.dialect.name}"
            )
    if audit.no_answer:
        print(f"\n  {WARN} {len(audit.no_answer)} pool(s) answered neither spelling:")
        for pool in audit.no_answer[:10]:
            print(f"      {pool.name[:34]:<34} {pool.address}  type={pool.pool_type}")
    if audit.unresolved:
        print(f"\n  {BAD} {len(audit.unresolved)} pool(s) with an unknown dialect:")
        for pool in audit.unresolved[:10]:
            print(f"      {pool.name[:34]:<34} {pool.address}  type={pool.pool_type}")

    if args.type or args.token:
        print()
        for pool in load.pools:
            if args.type and chain_table and args.type.lower() not in pool.pool_type.lower():
                continue
            symbols = [c.symbol for c in pool.coins]
            if args.token and args.token.upper() not in [s.upper() for s in symbols]:
                continue
            print(
                f"  {pool.address}  {pool.name[:30]:<30} {pool.pool_type:<18} "
                f"{'/'.join(symbols):<28} ${pool.tvl_usd:>13,.0f}  "
                f"{pool.dialect.name if pool.dialect else '?'}"
            )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Run the geometric grid on one pool and print every §12.2 number."""
    from ..core.calibrate import CalibrationError, asym, calibrate, peg_boundary
    from ..core.prices import gamma_live
    from ..core.probe import collect, plan_grid
    from .boa_host import override_client
    from .rpc import JsonRpcTransport
    from .universe import arc_refs, load_pools, read_balances, resolve_dialects

    chain = chain_table.get(args.chain)
    load = load_pools(chain, min_tvl=args.min_tvl)
    wanted = args.pool.lower()
    pools = [p for p in load.pools if p.address.lower() == wanted]
    if not pools:
        print(f"{BAD} pool {args.pool} not in the universe at min_tvl ${args.min_tvl:,.0f}")
        return 2

    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), block=args.block)
    client = override_client(rpc)
    resolve_dialects(pools, client, chain)
    read_balances(pools, client)

    pool = pools[0]
    print(f"\n\x1b[1m{pool.name}\x1b[0m  {pool.address}")
    print(
        f"  type {pool.pool_type}   dialect {pool.dialect.name if pool.dialect else '?'}"
        f" ({pool.note})   block {rpc.block:,}   TVL ${pool.tvl_usd:,.0f}"
    )
    for k, coin in enumerate(pool.coins):
        scale = 10**coin.decimals
        print(
            f"    coin {k}  {coin.symbol:<10} {coin.address}  "
            f"dec {coin.decimals:<3} balance {pool.balances[k] / scale:,.4f}"
        )

    refs = arc_refs(pools)
    if args.pair:
        i, j = (int(x) for x in args.pair.split(","))
        refs = [r for r in refs if r.i == i and r.j == j]
    plan = plan_grid(refs)
    ladders = collect(plan, client.probe(plan.probes))

    fits: dict[tuple[int, int], object] = {}
    print(f"\n  {len(plan)} probes over {len(ladders)} directions\n")
    for ladder in ladders:
        arc = ladder.arc
        label = f"{pool.coins[arc.i].symbol} -> {pool.coins[arc.j].symbol}"
        deltas, quotes = ladder.as_float()
        status = f"{len(ladder.deltas)}/{ladder.attempted} probes"
        if ladder.failures:
            status += "  " + " ".join(f"{k}x{v}" for k, v in ladder.failures.items())
        print(f"  \x1b[1m{label:<22}\x1b[0m {status}")
        if not ladder.ok:
            print(f"    {WARN} too few successful probes to calibrate")
            continue
        for d, q in zip(deltas, quotes, strict=True):
            print(f"      dx {d:>18,.6f}   dy {q:>18,.6f}   rate {q / d:.9f}")
        try:
            fit = calibrate(deltas, quotes)
        except CalibrationError as exc:
            print(f"    {BAD} {exc}")
            continue
        fits[(arc.i, arc.j)] = fit
        flag = f"{BAD} {fit.flag_reason.value}" if fit.convex_flag else f"{OK} concave"
        print(
            f"    a {fit.a:.10f}   B {fit.B:.6e}   {flag}"
            f"{'  CLAMPED cap=' + format(fit.cap, ',.4f') if fit.clamped else ''}"
        )
        print(
            f"    drift {fit.drift:+.4f}   eta {fit.eta:.4f}   "
            f"calib_delta {fit.calib_delta:,.4f}"
            f"{'   ' + WARN + ' split_hint' if fit.split_hint else ''}"
            f"{'   ' + WARN + ' COARSE_TANGENT' if ladder.coarse_tangent else ''}"
        )
        boundary = peg_boundary(deltas, quotes)
        if boundary is not None:
            print(f"    {WARN} curvature jump at dx {boundary:,.4f} -- split the arc (§2.5)")

    print("\n  \x1b[1mdirection checks (§12.2c)\x1b[0m")
    for (i, j), fit in sorted(fits.items()):
        reverse = fits.get((j, i))
        if reverse is None or j < i:
            continue
        pair = f"{pool.coins[i].symbol}/{pool.coins[j].symbol}"
        product = fit.a * reverse.a
        gamma = float(gamma_live(fit.a, reverse.a))
        mark = OK if product < 1.0 else BAD
        print(
            f"    {pair:<20} a_f*a_r {product:.10f} {mark}   "
            f"gamma_live {gamma:.8f}  (fee {1 - gamma:.6%})   "
            f"ASYM {asym(fit.a, reverse.a, fit.B, reverse.B):+.6f}"
        )
    return 0


def _ledger(result, nodes, dst, in_human, out_human) -> dict:
    """Modelled loss against the reference price, and the verified figure."""
    ledger = {
        "fee_bp": result.fee_bp,
        "impact_bp": result.impact_bp,
        "total_bp": result.fee_bp + result.impact_bp,
    }
    price = result.price_out_per_in
    if price > 0 and in_human > 0:
        ideal = in_human * price
        modelled = result.route.modelled_out / 10 ** nodes.decimals(dst)
        ledger["total_bp"] = (1 - modelled / ideal) * 10_000
        if result.verified_out is not None:
            ledger["verified_bp"] = (1 - out_human / ideal) * 10_000
            ledger["model_delta_bp"] = ledger["total_bp"] - ledger["verified_bp"]
    return ledger


def _resolve_token(nodes, symbol_or_address: str, pools) -> str:
    """Address, or the highest-TVL token with that symbol."""
    if symbol_or_address.startswith("0x") and len(symbol_or_address) == 42:
        return symbol_or_address.lower()
    wanted = symbol_or_address.upper()
    tvl: dict[str, float] = {}
    for pool in pools:
        for coin in pool.coins:
            if coin.symbol.upper() == wanted:
                key = coin.address.lower()
                tvl[key] = tvl.get(key, 0.0) + pool.tvl_usd
    if not tvl:
        raise KeyError(f"no token with symbol {symbol_or_address!r} in the universe")
    ranked = sorted(tvl.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1:
        alternatives = ", ".join(f"{a} (${v:,.0f})" for a, v in ranked[1:4])
        print(
            f"{WARN} symbol {symbol_or_address} is ambiguous, using {ranked[0][0]} "
            f"(${ranked[0][1]:,.0f}); also: {alternatives}"
        )
    return ranked[0][0]


def cmd_route(args: argparse.Namespace) -> int:
    from decimal import Decimal

    from ..core.pipeline import RoutingError, route
    from .boa_host import override_client
    from .probe_cache import CachedQuoterClient
    from .rpc import JsonRpcTransport
    from .universe import load_pools, read_balances, resolve_dialects
    from .wrappers import build_node_map, build_stake_arcs

    chain = chain_table.get(args.chain)
    started = time.monotonic()
    try:
        load = load_pools(chain, min_tvl=args.min_tvl, refresh=args.refresh,
                          pool_filters=args.pool_filters,
                          llamma=args.llamma)
    except CurveApiError as exc:
        print(f"{BAD} {exc}")
        return 4

    rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), block=args.block)
    client = override_client(rpc)
    resolve_dialects(load.pools, client, chain, use_cache=not args.refresh)
    read_balances(load.pools, client)
    if not args.no_cache:
        # Probe results are a pure function of (pool state at the pinned block,
        # size), so memoising them is exact.  A second route in the same block
        # then costs no probing at all.
        client = CachedQuoterClient(client, chain.chain_id, rpc.block)

    try:
        src = _resolve_token(None, args.src, load.pools)
        dst = _resolve_token(None, args.dst, load.pools)
    except KeyError as exc:
        print(f"{BAD} {exc}")
        return 2

    nodes, wrappers = build_node_map(load.pools, chain, client)
    stake_arcs = build_stake_arcs(nodes, chain, client)
    # Gas is priced by default.  Leaving it at zero made every route look free
    # to branch, which is exactly backwards for the small trades where an extra
    # leg costs more than it saves.
    gas_price_wei = (
        int(float(args.gas_price) * 1e9) if args.gas_price is not None
        else rpc.gas_price()
    )
    args.gas_price_wei = gas_price_wei
    if not nodes.has(src) or not nodes.has(dst):
        print(f"{BAD} token not routable in this universe")
        return 2

    if args.amount is None and not args.amount_wei:
        return _interactive(args, chain, rpc, client, nodes, wrappers, load, src, dst,
                            stake_arcs)

    amount_in = int(Decimal((args.amount or "1").replace("_", "")) * 10 ** nodes.decimals(src))
    if args.amount_wei:
        amount_in = int(args.amount_wei)

    try:
        result = route(
            load.pools, nodes, client,
            src_token=src, dst_token=dst, amount_in=amount_in,
            verify_on_chain=not args.no_verify,
            max_candidates=args.candidates,
            gas_price_wei=gas_price_wei,
            refit_rounds=args.refit,
            extra_arcs=stake_arcs,
            max_legs=args.max_legs,
        )
    except RoutingError as exc:
        print(f"{BAD} no route: {exc}")
        return 2

    return _present(result, args, chain, rpc, nodes, wrappers, load,
                    src, dst, amount_in, started)


def _interactive(args, chain, rpc, client, nodes, wrappers, load, src, dst,
                 stake_arcs=None) -> int:
    """Quote sizes as they are typed, reusing everything that does not depend on one.

    The expensive half of a route -- probing every arc and fitting reference
    prices -- is a function of the block and the pair, never of the amount, so
    it is paid once here.  What is left per keystroke is the solve, which also
    starts from the previous size's active set: the KKT system is affine in
    `Psi` within one active set, so a nearby amount usually keeps the same arcs
    conducting and converges almost immediately.
    """
    from decimal import Decimal, InvalidOperation

    from ..core.pipeline import RoutingError, prepare, route

    symbol_src, symbol_dst = nodes.symbol(src), nodes.symbol(dst)
    print(f"  {chain.name} · block {rpc.block:,} · {symbol_src} -> {symbol_dst}")
    print("  preparing (probing arcs, fitting reference prices)...", flush=True)
    started = time.monotonic()
    try:
        prepared = prepare(load.pools, nodes, client, src_token=src, dst_token=dst,
                           extra_arcs=stake_arcs)
    except RoutingError as exc:
        print(f"{BAD} no route: {exc}")
        return 2
    prep_warnings = set(prepared.warnings)
    print(
        f"  ready in {(time.monotonic() - started) * 1000:.0f} ms "
        f"({len(prepared.arcs)} arcs calibrated"
        + (f", {len(prep_warnings)} dropped" if prep_warnings else "")
        + f").  Type an amount in {symbol_src}; blank line or 'q' to quit.\n"
    )

    while True:
        try:
            raw = input(f"  {symbol_src} amount> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw or raw.lower() in {"q", "quit", "exit"}:
            return 0
        text = raw.replace("_", "").replace(",", "")
        multiplier = 1
        if text and text[-1].lower() in "kmb":
            multiplier = {"k": 10**3, "m": 10**6, "b": 10**9}[text[-1].lower()]
            text = text[:-1]
        try:
            amount_in = int(Decimal(text) * multiplier * 10 ** nodes.decimals(src))
        except (InvalidOperation, ValueError):
            print(f"  {BAD} not a number: {raw!r}")
            continue
        if amount_in <= 0:
            print(f"  {BAD} amount must be positive")
            continue

        started = time.monotonic()
        try:
            result = route(
                load.pools, nodes, client,
                src_token=src, dst_token=dst, amount_in=amount_in,
                verify_on_chain=not args.no_verify,
                max_candidates=args.candidates,
                gas_price_wei=getattr(args, "gas_price_wei", 0),
                refit_rounds=args.refit,
                prepared=prepared,
                max_legs=args.max_legs,
            )
        except RoutingError as exc:
            print(f"  {BAD} no route: {exc}")
            continue
        _present(result, args, chain, rpc, nodes, wrappers, load,
                 src, dst, amount_in, started, lean=True, suppress=prep_warnings)


def _present(result, args, chain, rpc, nodes, wrappers, load,
             src, dst, amount_in, started, *, lean: bool = False,
             suppress: set[str] | None = None) -> int:
    """Draw one route and, if asked, write its JSON.

    `lean` is the interactive view: the amount and the route, nothing else.
    Repeating the loss ledger, the diagnostics table and the candidate list
    after every keystroke buries the one thing being compared between sizes.
    Warnings stay -- those are the ones worth interrupting for -- but the ones
    raised while *preparing* are size-independent and identical on every quote,
    so `suppress` carries them and they are reported once at startup instead.
    """
    import json

    from ..core.render_text import render
    from ..core.rendermodel import build_diagram
    from ..core.schema import to_json
    from ..core.verify import summary as candidate_summary

    elapsed = (time.monotonic() - started) * 1000
    result.warnings.extend(load.warnings)
    for entry in wrappers.rejected_vaults:
        result.warnings.append(f"vault {entry.symbol} not merged: {entry.reason}")

    delivered = result.verified_out or result.route.modelled_out
    out_human = delivered / 10 ** nodes.decimals(dst)
    in_human = amount_in / 10 ** nodes.decimals(src)
    price = in_human / out_human if out_human else 0.0
    diagram = build_diagram(
        result.route,
        nodes,
        title=(
            f"{in_human:,.6f} {nodes.symbol(src)}  ->  {out_human:,.6f} {nodes.symbol(dst)}"
            f"        {price:,.4f} {nodes.symbol(src)}/{nodes.symbol(dst)}"
        ),
        subtitle=(
            f"{chain.name} · block {rpc.block:,} · {elapsed:.0f} ms · "
            f"{result.counters.get('active_arcs', 0)} arcs of "
            f"{result.counters.get('arcs_priced_out', 0)} priced out"
        ),
        certificate=result.certificate,
        certificate_reason=result.certificate_reason,
        pool_names=result.pool_names,
        ledger=None if lean else _ledger(result, nodes, dst, in_human, out_human),
        diagnostics={} if lean else {
            "pools": result.counters.get("pools", 0),
            "arcs calibrated": result.counters.get("arcs_calibrated", 0),
            "probes": result.counters.get("probes", 0),
            "pivots / CG": f"{result.counters.get('pivots', 0)} / {result.counters.get('cg_rounds', 0)}",
            "probe ms": f"{result.timings.get('probe', 0):.0f}",
            "solve ms": f"{result.timings.get('solve', 0):.1f}",
        },
        warnings=[w for w in result.warnings if not suppress or w not in suppress][:8],
        verified_out=result.verified_out,
    )
    if result.candidates and not lean:
        diagram.candidates = candidate_summary(
            result.candidates, nodes.decimals(dst), limit=args.candidates
        )
    print(render(diagram, unicode=not args.ascii, color=not args.no_color,
                 legend=not lean))

    if args.json:
        payload = to_json(
            result, chain=chain.name, chain_id=chain.chain_id, block=rpc.block,
            candidates=diagram.candidates, verified_out=result.verified_out,
        )
        text = json.dumps(payload, indent=2)
        if args.json == "-":
            print(text)
        else:
            with open(args.json, "w") as handle:
                handle.write(text)
            print(f"\n  wrote {args.json}")

    if args.strict and not result.certificate:
        return 5
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="erouter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check node and API capabilities")
    doctor.add_argument("--chain", help="only this chain (default: all configured)")
    doctor.add_argument("--block", default="latest", help="block to pin (default: latest)")
    doctor.add_argument("--min-tvl", type=float, default=10_000.0)
    doctor.set_defaults(func=cmd_doctor)

    pools = sub.add_parser("pools", help="list the universe and audit ABI dialects")
    pools.add_argument("--chain", default="ethereum")
    pools.add_argument("--block", default="latest")
    pools.add_argument("--min-tvl", type=float, default=10_000.0)
    pools.add_argument("--refresh", action="store_true", help="bypass caches")
    pools.add_argument("--type", help="only pools whose pool_type contains this")
    pools.add_argument("--token", help="only pools holding this symbol")
    pools.set_defaults(func=cmd_pools)

    probe = sub.add_parser("probe", help="run the probe grid on one pool")
    probe.add_argument("--pool", required=True)
    probe.add_argument("--chain", default="ethereum")
    probe.add_argument("--block", default="latest")
    probe.add_argument("--min-tvl", type=float, default=10_000.0)
    probe.add_argument("--pair", help="only this direction, e.g. 0,1")
    probe.set_defaults(func=cmd_probe)

    route_cmd = sub.add_parser("route", help="compute and draw an optimal route")
    route_cmd.add_argument("--from", dest="src", required=True, help="symbol or address")
    route_cmd.add_argument("--to", dest="dst", required=True, help="symbol or address")
    route_cmd.add_argument("--amount", default=None,
                       help="in human units, e.g. 1_000_000; omit for an interactive session")
    route_cmd.add_argument("--amount-wei", help="exact integer input, overrides --amount")
    route_cmd.add_argument("--chain", default="ethereum")
    route_cmd.add_argument("--block", default="latest")
    route_cmd.add_argument("--min-tvl", type=float, default=10_000.0)
    route_cmd.add_argument("--refresh", action="store_true")
    route_cmd.add_argument("--json", help="write JSON here, or '-' for stdout")
    route_cmd.add_argument("--ascii", action="store_true", help="no box-drawing characters")
    route_cmd.add_argument("--no-color", action="store_true")
    route_cmd.add_argument("--strict", action="store_true", help="exit 5 without a certificate")
    route_cmd.add_argument("--no-verify", action="store_true",
                           help="skip on-chain verification (modelled numbers only)")
    route_cmd.add_argument("--candidates", type=int, default=20)
    route_cmd.add_argument(
        "--gas-price", default=None,
        help="gwei; defaults to the node's live price. 0 disables gas costing",
    )
    route_cmd.add_argument("--no-cache", action="store_true", help="re-probe every arc")
    route_cmd.add_argument("--refit", type=int, default=2, help="§8 refit rounds (0 = off)")
    route_cmd.add_argument(
        "--max-legs", type=int, default=32,
        help="reject routes with more legs than this. The quoter can price 128; "
             "the default is what an executor might plausibly run. Going wide is "
             "worth ~5 bp below ~4 gwei and a loss above it",
    )
    route_cmd.add_argument(
        "--llamma", action="store_true",
        help="include crvUSD/lending LLAMMA markets. EXPERIMENTAL: they are a "
             "banded AMM and calibrate badly from a single band, which today "
             "makes routes materially worse",
    )
    route_cmd.add_argument(
        "--pool-filters", action="store_true",
        help="drop pools on Curve's pool_filters list (an extra request; "
             "/v2/pools already excludes them, so this only guards a stale cache)",
    )
    route_cmd.set_defaults(func=cmd_route)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
