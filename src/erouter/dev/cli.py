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


def _rpc_url(chain, args) -> str:
    """Where to reach the chain: the flag, then `networks.py`, then the table.

    `--rpc-url` wins so an endpoint can be tried without editing the gitignored
    `networks.py` that holds real keys.  The table's `public_rpc` is last, and
    exists so a fresh checkout with no `networks.py` still routes.
    """
    override = getattr(args, "rpc_url", None)
    if override:
        return override
    try:
        return config.rpc_url(chain.rpc_attr)
    except (KeyError, FileNotFoundError, ImportError):
        if chain.public_rpc:
            return chain.public_rpc
        raise


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
            rpc = JsonRpcTransport(url, block=args.block, chain_id=chain.chain_id)
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
    from .boa_host import quoter_client
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

    rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                           chain_id=chain.chain_id)
    client = quoter_client(rpc, chain)
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
    from .boa_host import quoter_client
    from .rpc import JsonRpcTransport
    from .universe import arc_refs, load_pools, read_balances, resolve_dialects

    chain = chain_table.get(args.chain)
    load = load_pools(chain, min_tvl=args.min_tvl)
    wanted = args.pool.lower()
    pools = [p for p in load.pools if p.address.lower() == wanted]
    if not pools:
        print(f"{BAD} pool {args.pool} not in the universe at min_tvl ${args.min_tvl:,.0f}")
        return 2

    rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                           chain_id=chain.chain_id)
    client = quoter_client(rpc, chain)
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


def _local_quoter(rpc, chain, load, nodes, *, quiet: bool = False):
    """A quoter backed by an in-process EVM, or None if that is not available.

    The committed cache says which slots each pool reads and holds its code, so
    the only per-block traffic is reading current values -- about a second for
    the whole universe.  A pool the cache has never seen is discovered here and
    written back, so keeping up with a moving universe costs an access list for
    what moved rather than for everything.
    """
    try:
        from .local_evm import LocalEvm
        from .state_cache import StateCache
    except ImportError as exc:
        if not quiet:
            print(f"  {WARN} local EVM unavailable ({exc}); quoting over the wire")
        return None

    from ..core.pipeline import build_arcs
    from .boa_host import quoter_client

    cache = StateCache.load(chain.chain_id, chain.name.lower())
    if not cache.accounts:
        if not quiet:
            print(f"  {WARN} no state cache for {chain.name}; run `erouter warmcache`")
        return None
    try:
        evm = LocalEvm(rpc, cache=cache)
        stats = evm.prime()
        fresh = cache.unknown(p.address for p in load.pools)
        # Re-list every arc, not only the new pools.  `prime` refreshes values
        # and `unknown` finds new pools, but neither sees a pool that has begun
        # reading a slot it did not read before -- an oracle round that
        # advanced, a band that moved.  Those slots then read as zero, which is
        # not a small error: it gives an arc the wrong `a`, `B` and `cap`, and
        # the solve built on it violates flow conservation outright.  Measured
        # at a block 10,802 after the cache was built, 18 such slots turned
        # USDC->sUSDS 3M from a route into a hard failure, and recovering them
        # brought every size back to within 0.2 bp of the wire path.
        #
        # One access-list pass over 887 arcs, ~1.0-1.4 s, concurrent.  It has
        # to happen *before* routing: a route that fails produces no output to
        # compare, so the after-the-fact check downstream never fires.
        refs, _ = build_arcs(load.pools, nodes)
        learned = evm.refresh_arcs(refs, quoter_client(rpc, chain).address) if refs else 0
        if fresh:
            cache.learn_pools(p.address for p in load.pools)
        if fresh or learned:
            cache.save()
    except Exception as exc:  # a cold cache must degrade, never abort a route
        if not quiet:
            print(f"  {WARN} local EVM warm failed ({str(exc)[:70]}); quoting over the wire")
        return None

    if not quiet:
        extra = f", {len(fresh)} new pool(s)" if fresh else ""
        extra += f", {learned} stale slot(s) recovered" if learned else ""
        print(f"  local EVM: {stats.slots:,} slots in {stats.ms:,.0f} ms"
              f" ({stats.accounts} accounts{extra})")
    return quoter_client(evm, chain)


# A local quote this far from the chain's own answer means the prefetched state
# is stale, not that the route is bad.  Well under the basis point a route is
# decided by, and well above integer rounding.
STALE_TOL_BP = 0.5


def _confirm_against_chain(result, rpc, chain, evm, nodes, pools, quiet=False):
    """Quote the chosen legs on-chain and say so if the local EVM disagreed.

    The local EVM is exact when its state is complete, so a disagreement is a
    statement about the prefetch, not about the route.  One `eth_call` to the
    quoter -- the only address a scoped key need allow -- turns a silent few
    basis points into something that repairs itself.
    """
    from ..core.pipeline import build_arcs
    from .boa_host import quoter_client

    route = result.route
    if route is None or evm is None:
        return result
    legs = [rl.leg for rl in route.legs]
    amounts, slots = [result.amount_in], [route.dst_slot]
    try:
        onchain = quoter_client(rpc, chain).quote_routes([legs], amounts, slots)[0]
    except Exception as exc:
        if not quiet:
            print(f"  {WARN} could not confirm on-chain ({str(exc)[:50]})")
        return result
    mine = result.verified_out or 0
    if not (onchain and mine):
        return result
    gap = (mine / onchain - 1) * 10_000
    if abs(gap) <= STALE_TOL_BP:
        return result

    used = {rl.target.lower() for rl in route.legs}
    refs, _ = build_arcs([p for p in pools if p.address.lower() in used], nodes)
    learned = evm.refresh_arcs(refs, quoter_client(rpc, chain).address) if refs else 0
    if not quiet:
        print(f"  {WARN} local state was {gap:+.2f} bp off the chain; "
              f"re-listed {len(refs)} arcs, {learned} slot(s) were missing"
              f"{' -- routing again' if learned else ''}")
    return result if not learned else None  # None == repaired, route again


def cmd_route(args: argparse.Namespace) -> int:
    from decimal import Decimal

    from ..core.pipeline import RoutingError, route
    from .boa_host import quoter_client
    from .probe_cache import CachedQuoterClient
    from .rpc import JsonRpcTransport
    from .universe import (
        check_reserves_are_real,
        load_pools,
        read_balances,
        resolve_dialects,
    )
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

    rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                           chain_id=chain.chain_id)
    client = quoter_client(rpc, chain)
    resolve_dialects(load.pools, client, chain, use_cache=not args.refresh)
    read_balances(load.pools, client)
    for warning in check_reserves_are_real(load.pools, client, rpc):
        print(f"{WARN} {warning}")
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
    evm = None
    if args.local:
        local = _local_quoter(rpc, chain, load, nodes)
        if local is not None:
            client, evm = local, getattr(local, "transport", None)
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

    if _confirm_against_chain(result, rpc, chain, evm, nodes, load.pools) is None:
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
            print(f"{BAD} no route after refresh: {exc}")
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
        if _confirm_against_chain(result, rpc, chain,
                                  getattr(client, "transport", None),
                                  nodes, load.pools) is None:
            result = route(load.pools, nodes, client, src_token=src, dst_token=dst,
                           amount_in=amount_in, verify_on_chain=not args.no_verify,
                           max_candidates=args.candidates,
                           gas_price_wei=getattr(args, "gas_price_wei", 0),
                           refit_rounds=args.refit, extra_arcs=stake_arcs,
                           max_legs=args.max_legs, prepared=prepared)
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


# `prepare` is a parent span wrapping these, so counting both double-counts.
_PREPARE_CHILDREN = ("arcs", "probe", "calibrate", "component", "prices")
# Stages whose cost is a network round trip rather than arithmetic.
_RPC_STAGES = {"probe", "refine", "verify", "direct", "refit", "split"}
_STAGE_ORDER = (
    "arcs", "probe", "calibrate", "component", "prices",
    "graph", "seed", "refine", "solve", "candidates", "direct", "verify",
    "refit", "split", "realize",
)


def _stage_table(title: str, wall_ms: float, timings: dict[str, float], width: int = 34):
    """One phase's stage breakdown, split into network and compute."""
    print(f"\n  {title}   {wall_ms:,.0f} ms wall")
    print(f"  {'stage':<12}{'ms':>9}{'share':>8}  where")
    rpc = cpu = 0.0
    for name in _STAGE_ORDER:
        ms = timings.get(name)
        if not ms:
            continue
        where = "rpc" if name in _RPC_STAGES else "cpu"
        if where == "rpc":
            rpc += ms
        else:
            cpu += ms
        bar = "#" * max(1, int(ms / max(wall_ms, 1e-9) * width))
        print(f"  {name:<12}{ms:>9,.0f}{ms / wall_ms * 100:>7.1f}%  {where}  {bar}")
    other = wall_ms - rpc - cpu
    print(f"  {'-' * 46}")
    for label, value in (("network", rpc), ("compute", cpu), ("unclocked", other)):
        print(f"  {label:<12}{value:>9,.0f}{value / wall_ms * 100:>7.1f}%")
    return rpc, cpu


def _hot_functions(fn, limit: int = 12, reference_ms: float | None = None) -> None:
    """Function-level self time inside one route.

    Stage timings say *which phase*; this says *which line*.  cProfile inflates
    the total, and by how much depends on the call count and the machine -- it
    has been 3x and 15x on the same code -- so the inflation is measured
    against the un-profiled wall and printed, rather than asserted.  Only the
    shares are meaningful.
    """
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    stats = pstats.Stats(profiler)
    total = stats.total_tt or 1e-9
    rows = sorted(
        ((entry[2], key) for key, entry in stats.stats.items()),
        key=lambda pair: -pair[0],
    )[:limit]
    inflation = (
        f", {total * 1000 / reference_ms:.0f}x the un-profiled {reference_ms:,.0f} ms"
        if reference_ms else ""
    )
    print(f"\n  HOT FUNCTIONS   self time in one route "
          f"(profiled {total * 1000:,.0f} ms{inflation} — read the shares, not the times)")
    print(f"  {'share':>7}{'calls':>9}  where")
    for self_time, (path, line, name) in rows:
        where = f"{path.rsplit('/', 1)[-1]}:{line}({name})"
        calls = stats.stats[(path, line, name)][0]
        print(f"  {self_time / total * 100:>6.1f}%{calls:>9,}  {where[:62]}")


def cmd_bench(args: argparse.Namespace) -> int:
    """Where a route's time goes, cold and warm.

    Two phases, because they have opposite shapes and different audiences.
    *Cold* is an app's first quote: probing the universe dominates and it is
    almost entirely network, so it moves with your connection, not with the
    solver.  *Warm* is every quote after that in a session -- `prepare()`
    reused, probes cached -- and it is almost entirely compute.  A session pays
    cold once and warm per keystroke.
    """
    import statistics
    import time as _time
    from decimal import Decimal

    from ..core.pipeline import RoutingError, prepare, route
    from .boa_host import quoter_client
    from .probe_cache import CachedQuoterClient
    from .rpc import JsonRpcTransport
    from .universe import load_pools, read_balances, resolve_dialects
    from .wrappers import build_node_map, build_stake_arcs

    chain = chain_table.get(args.chain)
    try:
        rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                           chain_id=chain.chain_id)
    except Exception as exc:
        print(f"{BAD} node unreachable: {exc}")
        return 4

    # Baseline the link before anything else, so a slow network is visible as a
    # number rather than as a mysteriously large `probe` stage.
    rpc.stats.reset()
    for _ in range(5):
        rpc.fetch("eth_chainId", [])
    link_ms = rpc.stats.mean_ms

    try:
        load = load_pools(chain, min_tvl=args.min_tvl, llamma=args.llamma)
    except CurveApiError as exc:
        print(f"{BAD} {exc}")
        return 4
    raw = quoter_client(rpc, chain)
    cached = CachedQuoterClient(raw, chain.chain_id, rpc.block)
    uncached = CachedQuoterClient(raw, chain.chain_id, rpc.block, enabled=False)
    resolve_dialects(load.pools, cached, chain)
    read_balances(load.pools, cached)
    nodes, _ = build_node_map(load.pools, chain, cached)
    stake = build_stake_arcs(nodes, chain, cached)

    try:
        src = _resolve_token(None, args.src, load.pools)
        dst = _resolve_token(None, args.dst, load.pools)
    except KeyError as exc:
        print(f"{BAD} {exc}")
        return 2
    amount = int(Decimal(args.amount.replace("_", "")) * 10 ** nodes.decimals(src))
    gas_price = (
        int(float(args.gas_price) * 1e9) if args.gas_price is not None else rpc.gas_price()
    )

    print(f"\n  {chain.name} · block {rpc.block:,} · {len(load.pools)} pools · "
          f"gas {gas_price / 1e9:.4f} gwei")
    print(f"  {nodes.symbol(src)} -> {nodes.symbol(dst)}, "
          f"{Decimal(args.amount.replace('_', '')):,} in, {args.reps} reps (min)")
    print(f"  link: {link_ms:.1f} ms per JSON-RPC round trip  "
          f"({'local' if link_ms < 15 else 'remote — expect the cold phase to scale with this'})")

    kw = {
        "src_token": src, "dst_token": dst, "amount_in": amount,
        "extra_arcs": stake, "gas_price_wei": gas_price, "max_legs": args.max_legs,
        "optimise_split": not args.no_split,
    }

    def phase(title, client, prepared):
        try:
            route(load.pools, nodes, client, **kw, prepared=prepared)
        except RoutingError as exc:
            print(f"{BAD} no route: {exc}")
            return None
        runs = []
        for _ in range(args.reps):
            started = _time.perf_counter()
            result = route(load.pools, nodes, client, **kw, prepared=prepared)
            runs.append((_time.perf_counter() - started, result))
        wall, best = min(runs, key=lambda pair: pair[0])
        spread = statistics.pstdev([r * 1000 for r, _ in runs]) if len(runs) > 1 else 0.0
        rpc_ms, cpu_ms = _stage_table(title, wall * 1000, best.timings)
        if spread > wall * 1000 * 0.15:
            print(f"  {WARN} run-to-run spread {spread:.0f} ms — machine is busy")
        return wall * 1000, rpc_ms, cpu_ms, best

    cold = phase("COLD   first quote: probes the universe", uncached, None)
    if args.profile and cold:
        _hot_functions(lambda: route(load.pools, nodes, uncached, **kw),
                       reference_ms=cold[0])
    warm_state = prepare(load.pools, nodes, cached, src_token=src, dst_token=dst,
                         extra_arcs=stake)
    warm = phase("WARM   in-session: prepare() reused, probes cached", cached, warm_state)
    if args.profile and warm:
        _hot_functions(lambda: route(load.pools, nodes, cached, **kw, prepared=warm_state),
                       reference_ms=warm[0])
    if cold and warm:
        print(f"\n  cold {cold[0]:,.0f} ms ({cold[1] / cold[0] * 100:.0f}% network)"
              f"   ->   warm {warm[0]:,.0f} ms ({warm[2] / warm[0] * 100:.0f}% compute)"
              f"   {cold[0] / warm[0]:.1f}x")
        counters = warm[3].counters
        print(f"  {counters.get('arcs_calibrated', 0)} arcs · "
              f"{counters.get('active_arcs', 0)} active · "
              f"{counters.get('cg_rounds', 0)} CG rounds · "
              f"{counters.get('candidates', 0)} candidates "
              f"({counters.get('candidates_quoted', 0)} quoted)")
        print(f"  split: {counters.get('split_calls', 0)} round trips · "
              f"{counters.get('split_evaluations', 0)} quotes · "
              f"{counters.get('split_gain_bp', 0.0):+.2f} bp")
    return 0


def cmd_warmcache(args: argparse.Namespace) -> int:
    """Learn every pool's storage layout and bytecode, and commit the result.

    Two of the three costs of warming a local EVM answer questions that do not
    change between blocks -- which slots a pool reads, and the code that reads
    them.  This resolves them once so a checkout starts warm and a session pays
    only for the storage sweep.

    Incremental by construction: a pool already in the cache is not listed
    again, so adding a newly deployed pool costs an access list for that pool
    rather than for the universe.
    """
    import time as _time

    from ..core.pipeline import prepare
    from .boa_host import quoter_client
    from .local_evm import LocalEvm, Recorder
    from .rpc import JsonRpcTransport
    from .state_cache import StateCache
    from .universe import load_pools, read_balances, resolve_dialects
    from .wrappers import build_node_map, build_stake_arcs

    chain = chain_table.get(args.chain)
    try:
        rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                           chain_id=chain.chain_id)
    except Exception as exc:
        print(f"{BAD} node unreachable: {exc}")
        return 4

    cache = StateCache.load(chain.chain_id, chain.name.lower())
    before = cache.stats()
    print(f"\n  {chain.name} · block {rpc.block:,}")
    print(f"  cache: {before.accounts} accounts, {before.slots:,} slots, "
          f"{before.code_blobs} code blobs, {before.pools_known} pools known")

    load = load_pools(chain, min_tvl=args.min_tvl, llamma=args.llamma)
    # Everything through the recorder, universe setup included.  Node merges
    # and stake arcs read ERC4626 vaults and wrappers that no swap probe ever
    # touches, and a cache without them cannot reach the tokens behind them --
    # `sDOLA is not reachable from any pool`, learned the hard way.
    recorder = Recorder(rpc)
    setup = quoter_client(recorder, chain)
    resolve_dialects(load.pools, setup, chain)
    read_balances(load.pools, setup)
    nodes, _ = build_node_map(load.pools, chain, setup)
    stake = build_stake_arcs(nodes, chain, setup)

    # A LLAMMA reads a different set of band slots as the price moves, so its
    # layout is a function of state and cannot be cached.  Recorded as volatile
    # rather than silently cached wrong.
    volatile = [p.address for p in load.pools if (p.pool_type or "").lower() == "llamma"]
    cache.mark_volatile(volatile)

    fresh = cache.unknown(p.address for p in load.pools)
    print(f"  universe: {len(load.pools)} pools, {len(fresh)} to learn "
          f"({len(volatile)} volatile)")
    if not fresh:
        print(f"  {OK} nothing to do")
        return 0

    started = _time.perf_counter()
    try:
        prepare(load.pools, nodes, setup,
                src_token=_resolve_token(nodes, args.src, load.pools),
                dst_token=_resolve_token(nodes, args.dst, load.pools),
                extra_arcs=stake)
    except Exception as exc:
        print(f"{BAD} could not probe the universe: {exc}")
        return 4
    probing = (_time.perf_counter() - started) * 1000
    print(f"  probed in {probing:,.0f} ms, {len(recorder.calls)} distinct quoter calls")

    evm = LocalEvm(rpc, cache=cache)
    stats = evm.warm(list(recorder.calls))
    cache.learn_pools(p.address for p in load.pools)
    cache.save()
    after = cache.stats()
    size = cache.path.stat().st_size if cache.path.exists() else 0
    print(f"  learned in {stats.ms:,.0f} ms "
          f"(lists {stats.list_ms:,.0f} · code {stats.code_ms:,.0f} · "
          f"storage {stats.storage_ms:,.0f})")
    print(f"  cache: {after.accounts} accounts (+{after.accounts - before.accounts}), "
          f"{after.slots:,} slots, {after.code_blobs} code blobs")
    print(f"  {OK} wrote {cache.path} ({size / 1024:,.0f} KiB)")
    for line in stats.errors[:5]:
        print(f"  {WARN} {line}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="erouter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    warm = sub.add_parser("warmcache",
                          help="learn pool storage layouts and code; writes data/evm-state")
    warm.add_argument("--chain", default="ethereum")
    warm.add_argument("--block", default="latest")
    warm.add_argument("--min-tvl", type=float, default=10_000.0)
    warm.add_argument("--llamma", action="store_true")
    warm.add_argument("--from", dest="src", default="USDC")
    warm.add_argument("--to", dest="dst", default="WETH")
    warm.add_argument(
        "--rpc-url", default=None,
        help="override the endpoint from networks.py (for trying a hosted RPC)")
    warm.set_defaults(func=cmd_warmcache)

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

    bench = sub.add_parser("bench", help="where a route's time goes, cold and warm")
    bench.add_argument("--from", dest="src", default="USDC")
    bench.add_argument("--to", dest="dst", default="WETH")
    bench.add_argument("--amount", default="100000", help="in human units")
    bench.add_argument("--chain", default="ethereum")
    bench.add_argument("--block", default="latest")
    bench.add_argument("--min-tvl", type=float, default=10_000.0)
    bench.add_argument("--reps", type=int, default=4, help="timed runs per phase; min wins")
    bench.add_argument("--gas-price", default=None, help="gwei; defaults to the live price")
    bench.add_argument("--max-legs", type=int, default=32)
    bench.add_argument("--llamma", action="store_true")
    bench.add_argument(
        "--no-split", action="store_true",
        help="skip the §7 split-ratio optimisation, to price what it costs",
    )
    bench.add_argument(
        "--profile", action="store_true",
        help="also print function-level self time inside a warm route",
    )
    bench.add_argument(
        "--rpc-url", default=None,
        help="override the endpoint from networks.py (for trying a hosted RPC)")
    bench.set_defaults(func=cmd_bench)

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
    route_cmd.add_argument(
        "--no-local", dest="local", action="store_false",
        help="quote over the wire instead of in an in-process EVM primed from "
             "data/evm-state (the default; falls back to the wire on its own)",
    )
    route_cmd.add_argument(
        "--rpc-url", default=None,
        help="override the endpoint from networks.py (for trying a hosted RPC)")
    route_cmd.set_defaults(local=True)
    route_cmd.set_defaults(func=cmd_route)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
