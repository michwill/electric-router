"""`erouter` command line.

Phase 0 ships `doctor`, which turns every environment assumption the design
rests on into a runtime check: state-override support (decides whether the
quoter needs deploying), `debug_traceCall` (decides boa's fork prefetch), batch
support, and the Curve API behind its User-Agent requirement.
"""

from __future__ import annotations

import os

# Before numpy loads, and therefore before anything that imports it.
#
# The solve is thousands of *tiny* factorisations -- §9.4 restricts each pivot
# to its own connected component, so `n` is 5-10 -- and one route can make
# 4,713 calls into `numpy.linalg.solve`.  At that size OpenBLAS's threading is
# pure overhead: the arithmetic is nanoseconds and the thread handoff is not.
# Measured on the solve stage alone, five reps, minimum, eight threads against
# one: USDC->USDT 103 vs 55 ms, USDC->WETH 50 vs 35, stETH->WETH 106 vs 53,
# USDC->CRV 232 vs 135.  Twice as fast in all four -- but that is 35-135 ms of
# routes taking 600-11,400 ms, so end to end it disappears into noise.  The
# reproducibility below is the reason this is on by default, not the speed.
#
# It also makes the answer reproducible.  A threaded reduction sums in whatever
# order the threads finish, so the §12.4 flow-conservation residual moves
# between runs; on that pair it straddled the tolerance and the route failed
# outright with eight threads while succeeding with one.  A router whose answer
# depends on how busy the machine is cannot be verified against anything.
#
# Set EROUTER_BLAS_THREADS to override -- the §4 price fit is one dense solve
# at n~300 per block and is the only part that could want more.
_THREADS = os.environ.get("EROUTER_BLAS_THREADS", "1")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from . import chains as chain_table  # noqa: E402
from . import config  # noqa: E402
from .curve_api import CurveApi, CurveApiError  # noqa: E402
from .facts import FactsCache  # noqa: E402
from .rpc import JsonRpcTransport, RpcError  # noqa: E402

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
                          pool_filters=getattr(args, "pool_filters", False),
                          llamma=getattr(args, "llamma", False))
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
        f"min_tvl ${_floor_shown(chain, args.min_tvl)}  universe from {source}"
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
            if args.token and _fold(args.token) not in [_fold(s) for s in symbols]:
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


def _floor_shown(chain, asked: float) -> str:
    """The floor that was really applied, which is not always the one asked for.

    A Curve Lite deployment is smaller than the $10,000 default -- fantom's 321
    pools come to $0.9M between them -- so `lite.list_pools` drops the floor to
    zero rather than return an empty universe.  Printing the requested figure
    there would describe a filter that did not run.
    """
    from .lite import LITE_MIN_TVL

    if getattr(chain, "lite", False):
        return f"{min(asked, LITE_MIN_TVL):,.0f} (lite)"
    return f"{asked:,.0f}"


def _certificate_note(result) -> str | None:
    """Why there is no proof, and what the relaxation's own bound was.

    "RESTRICTED" on its own reads worse than it is.  It says the winning
    candidate was a restriction of the full program -- almost always because
    `C0`, the only candidate that can carry the certificate, put flow through
    one pool twice and a view-only chained quote cannot see its own earlier
    leg.  The gap is the §5.5 bound on what the relaxation still had on the
    table, which is the number that says whether the missing proof matters.

    It bounds the *relaxation*, not the executed route: the winner is a
    different candidate, quoted on-chain.  So it is reported as what it is.
    """
    reason = result.certificate_reason
    if reason is None:
        return None
    gap = result.counters.get("optimality_gap_bp")
    if gap is None:
        return reason
    return f"{reason} · relaxation gap {gap:.3f} bp"


def _ledger(result, nodes, dst, in_human, out_human) -> dict:
    """Modelled loss against the reference price, and the verified figure."""
    ledger = {
        "fee_bp": result.fee_bp,
        "impact_bp": result.impact_bp,
        "total_bp": result.fee_bp + result.impact_bp,
    }
    if result.price_impact_bp is not None:
        ledger["price_impact_bp"] = result.price_impact_bp
        ledger["impact_fraction"] = result.impact_fraction
    price = result.price_out_per_in
    if price > 0 and in_human > 0:
        ideal = in_human * price
        modelled = result.route.modelled_out / 10 ** nodes.decimals(dst)
        ledger["total_bp"] = (1 - modelled / ideal) * 10_000
        if result.verified_out is not None:
            ledger["verified_bp"] = (1 - out_human / ideal) * 10_000
            ledger["model_delta_bp"] = ledger["total_bp"] - ledger["verified_bp"]
    return ledger


#: Symbol glyphs nobody types, folded to the letter they stand for.
#
# Tether's ticker is spelled with U+20AE TETHER SIGN on several chains -- tac
# calls it `USD₮` and xlayer `USD₮0` -- and `--from USDT` there fails with "no
# token with symbol 'USDT' in the universe" while the pool it wants is right
# there and named `USDT/WTAC`.  NFKC does not fold this codepoint, so the map is
# explicit.  A chain carrying both spellings is not a problem: they resolve to
# one symbol and the existing TVL ranking picks between them out loud.
GLYPH_FOLD = {"₮": "T"}


def _fold(symbol: str) -> str:
    """A symbol as the user would have typed it: upper case, ASCII tickers."""
    for glyph, letter in GLYPH_FOLD.items():
        symbol = symbol.replace(glyph, letter)
    return symbol.upper()


def _resolve_token(nodes, symbol_or_address: str, pools) -> str:
    """Address, or the highest-TVL token with that symbol."""
    if symbol_or_address.startswith("0x") and len(symbol_or_address) == 42:
        return symbol_or_address.lower()
    wanted = _fold(symbol_or_address)
    tvl: dict[str, float] = {}
    for pool in pools:
        for coin in pool.coins:
            if _fold(coin.symbol) == wanted:
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


def _local_quoter(rpc, chain, load, nodes, *, quiet: bool = False,
                  fresh_quoter: bool = False):
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
    if fresh_quoter:
        # The local EVM loads the *deployed* quoter's code from the state
        # cache, so a kind added since that deployment is priced as a revert --
        # which reads as "this leg cannot be traded" and is silently routed
        # around.  Injecting the compiled runtime instead makes new kinds
        # usable here before they exist on chain.  Only ever local: the wire
        # path still talks to what is actually deployed.
        from .boa_host import override_client

        return override_client(evm)
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
    from .wrappers import build_lending_arcs, build_node_map, build_stake_arcs

    chain = chain_table.get(args.chain)
    started = time.monotonic()
    try:
        load = load_pools(chain, min_tvl=args.min_tvl, refresh=args.refresh,
                          pool_filters=getattr(args, "pool_filters", False),
                          llamma=getattr(args, "llamma", False))
    except CurveApiError as exc:
        print(f"{BAD} {exc}")
        return 4

    gas_table, gas_measured = _gas_table(chain, args, load.pools)
    risk_table, risk_measured = _risk_table(chain, args)
    route_opts = _route_options(args)
    if gas_measured:
        print(f"  gas: {gas_measured:,} legs priced from measured execution")
    elif not getattr(args, "static_gas", False):
        print(f"  {WARN} gas: no measurements; using the assumed per-kind table"
              f" (run `erouter gascal`)")
    if risk_measured:
        print(f"  risk: {risk_measured:,} pools priced by how often their own "
              f"minimum-out would trip")
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

    nodes, wrappers = build_node_map(
        load.pools, chain, client,
        facts=FactsCache.load(chain.chain_id, chain.name.lower()))
    stake_arcs = build_stake_arcs(nodes, chain, client)
    # Leaving a lending wrapper, where `data/facts` says the protocol still
    # allows it.  Rides with the stake arcs: same shape, same treatment.
    stake_arcs = stake_arcs + build_lending_arcs(
        nodes, chain, client, FactsCache.load(chain.chain_id, chain.name.lower()))
    evm = None
    if args.local:
        local = _local_quoter(rpc, chain, load, nodes,
                              fresh_quoter=getattr(args, 'fresh_quoter', False))
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
            gas_table=gas_table,
            risk_table=risk_table,
            **route_opts,
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
                gas_table=gas_table,
                risk_table=risk_table,
                **route_opts,
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

    gas_table, _ = _gas_table(chain, args, load.pools)
    risk_table, _ = _risk_table(chain, args)
    route_opts = _route_options(args)
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
                gas_table=gas_table,
                risk_table=risk_table,
                **route_opts,
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
                           max_legs=args.max_legs, prepared=prepared,
                           gas_table=gas_table,
                           risk_table=risk_table, **route_opts)
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
            # Next to the price, because it qualifies the price -- and in the
            # header rather than the loss ledger, which the compact view drops.
            + ("" if result.price_impact_bp is None
               else f"  ·  impact {result.price_impact_bp:,.2f} bp")
        ),
        subtitle=(
            f"{chain.name} · block {rpc.block:,} · {elapsed:.0f} ms · "
            f"{result.counters.get('active_arcs', 0)} arcs of "
            f"{result.counters.get('arcs_priced_out', 0)} priced out"
        ),
        certificate=result.certificate,
        certificate_reason=_certificate_note(result),
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
    from .wrappers import build_lending_arcs, build_node_map, build_stake_arcs

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
    gas_table, gas_measured = _gas_table(chain, args, load.pools)
    risk_table, _ = _risk_table(chain, args)
    route_opts = _route_options(args)
    if gas_measured:
        print(f"  gas: {gas_measured:,} legs priced from measured execution")
    elif not getattr(args, "static_gas", False):
        print(f"  {WARN} gas: no measurements; using the assumed per-kind table"
              f" (run `erouter gascal`)")
    raw = quoter_client(rpc, chain)
    cached = CachedQuoterClient(raw, chain.chain_id, rpc.block)
    uncached = CachedQuoterClient(raw, chain.chain_id, rpc.block, enabled=False)
    resolve_dialects(load.pools, cached, chain)
    read_balances(load.pools, cached)
    nodes, _ = build_node_map(load.pools, chain, cached,
                              facts=FactsCache.load(chain.chain_id, chain.name.lower()))
    stake = build_stake_arcs(nodes, chain, cached)
    # Leaving a lending wrapper, where `data/facts` says the protocol still
    # allows it.  Rides with the stake arcs: same shape, same treatment.
    stake = stake + build_lending_arcs(
        nodes, chain, cached, FactsCache.load(chain.chain_id, chain.name.lower()))

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
        "gas_table": gas_table, "risk_table": risk_table, **route_opts,
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





#: Multiples of a "unit" trade.  Three decades, because gas is size-dependent:
#: a crypto pool rebalances inside `exchange` at some sizes and not others, and
#: the dearest sample is the one we keep.
GASCAL_SIZES = (0.1, 1.0, 10.0)
#: One unit, by decimals: $10k of a stablecoin, 10 of an 18-decimal asset.
GASCAL_UNIT = {6: 10_000, 8: 1, 18: 10}
#: Enough of the graph to reach every arc kind -- swaps of both dialects, a
#: vault deposit and redeem, a wrap and an unwrap.
GASCAL_TOKENS = ("USDC", "USDT", "DAI", "crvUSD", "WETH", "stETH", "wstETH",
                 "sDAI", "sUSDS", "CRV", "sDOLA", "rETH")


def _gascal_pairs(nodes, pools, args):
    """Every ordered pair among the tokens this chain can resolve."""
    if getattr(args, "pairs", None):
        wanted = [tuple(p.split("-", 1)) for p in args.pairs]
    else:
        wanted = [(a, b) for a in GASCAL_TOKENS for b in GASCAL_TOKENS if a != b]
    out, seen = [], set()
    for src_symbol, dst_symbol in wanted:
        try:
            src = _resolve_token(nodes, src_symbol, pools)
            dst = _resolve_token(nodes, dst_symbol, pools)
        except KeyError:
            continue          # a token this chain does not have is not an error
        if src == dst or (src, dst) in seen:
            continue
        seen.add((src, dst))
        decimals = nodes.decimals(src)
        out.append((src, dst, GASCAL_UNIT.get(decimals, 10) * 10 ** decimals))
    return out[:args.max_pairs]



def _holder(holders, token, avoid):
    """The richest holder of `token` that is not the pool we are about to trade on.

    A ranked list rather than the maximum, because for several tokens the
    largest holder *is* the pool under test -- stETH's biggest pool holder is
    the ETH/stETH pool itself -- and taking only the top one silently gave up
    on exactly the legs the fallback exists to rescue.
    """
    for address, _ in holders.get(token.lower(), ()):
        if address.lower() != avoid.lower():
            return address
    return ""



#: `ArcKind.SWAP_UNDERLYING` on the branch that adds it.  Reserved here so the
#: survey's findings do not collide with the pool's own wrapped-coin arcs.
UNDERLYING_KIND = 14


def _underlying_swap(i: int, j: int, dx: int) -> bytes:
    """`exchange_underlying` -- the call a lending pool cannot always honour."""
    from ..core.keccak import keccak256
    from .gas_probe import _pad

    return (keccak256(b"exchange_underlying(int128,int128,uint256,uint256)")[:4]
            + _pad(i) + _pad(j) + _pad(dx) + _pad(0))


def _underlying_of(evm, pool: str, index: int) -> str:
    """The underlying token a lending pool's coin `index` wraps, or ""."""
    from ..core.keccak import keccak256
    from .gas_probe import CALLER, _pad

    for signature in (b"underlying_coins(int128)", b"underlying_coins(uint256)"):
        try:
            out = evm.message_call(caller=CALLER, to=pool,
                                   calldata=keccak256(signature)[:4] + _pad(index))
        except Exception:
            continue          # Aave answers uint256 and reverts on int128 (E6)
        raw = bytes(out)
        if len(raw) >= 32 and int.from_bytes(raw, "big"):
            return "0x" + raw[-20:].hex()
    return ""


def cmd_gascal(args: argparse.Namespace) -> int:
    """Measure what our own routes cost to execute, and commit the answer.

    Driven by realised routes rather than a synthetic ladder, because gas is
    state-dependent in ways a ladder cannot reach: a crypto pool may rebalance
    inside `exchange` and cost tens of thousands more, and only at some sizes.
    The legs a route actually chose, at the sizes it chose, are the sample that
    matters.

    Everything runs in revm against local state, so a full pass costs no round
    trips beyond the ones routing already pays.
    """
    import time as _time

    from ..core.pipeline import RoutingError, prepare, route
    from .boa_host import quoter_client
    from .facts import FactsCache
    from .gas_probe import CALLER as GASCAL_CALLER
    from .gas_probe import Funder, measure_legs
    from .local_evm import LocalEvm
    from .probe_cache import CachedQuoterClient
    from .rpc import JsonRpcTransport
    from .state_cache import StateCache
    from .universe import load_pools, read_balances, resolve_dialects
    from .wrappers import build_lending_arcs, build_node_map, build_stake_arcs

    chain = chain_table.get(args.chain)
    try:
        rpc = JsonRpcTransport(_rpc_url(chain, args), block=args.block,
                               chain_id=chain.chain_id)
    except Exception as exc:
        print(f"{BAD} node unreachable: {exc}")
        return 4

    load = load_pools(chain, min_tvl=args.min_tvl)
    setup = CachedQuoterClient(quoter_client(rpc, chain), chain.chain_id, rpc.block)
    resolve_dialects(load.pools, setup, chain)
    read_balances(load.pools, setup)
    nodes, _ = build_node_map(load.pools, chain, setup,
                              facts=FactsCache.load(chain.chain_id, chain.name.lower()))
    stake = build_stake_arcs(nodes, chain, setup)
    # Leaving a lending wrapper, where `data/facts` says the protocol still
    # allows it.  Rides with the stake arcs: same shape, same treatment.
    stake = stake + build_lending_arcs(
        nodes, chain, setup, FactsCache.load(chain.chain_id, chain.name.lower()))

    quoting = LocalEvm(rpc, cache=StateCache.load(chain.chain_id, chain.name.lower()))
    quoting.prime()
    client = quoter_client(quoting, chain)

    pairs = _gascal_pairs(nodes, load.pools, args)
    print(f"  {chain.name} · block {rpc.block:,} · {len(pairs)} pairs x "
          f"{len(GASCAL_SIZES)} sizes")

    legs: dict = {}
    routed = 0
    started = _time.perf_counter()
    for src, dst, unit in pairs:
        try:
            prepared = prepare(load.pools, nodes, client, src_token=src,
                               dst_token=dst, extra_arcs=stake)
        except RoutingError:
            continue
        for scale in GASCAL_SIZES:
            try:
                result = route(load.pools, nodes, client, src_token=src,
                               dst_token=dst, amount_in=int(unit * scale),
                               extra_arcs=stake, gas_price_wei=args.gas_price_wei,
                               prepared=prepared, max_legs=args.max_legs)
            except RoutingError:
                continue
            routed += 1
            for leg in result.route.legs:
                key = (leg.target.lower(), leg.leg.kind, leg.leg.i, leg.leg.j)
                if key not in legs or leg.amount_in > legs[key][1]:
                    legs[key] = (leg.token_in, leg.amount_in)
    print(f"  {routed} routes -> {len(legs)} distinct legs "
          f"in {_time.perf_counter() - started:.0f}s")

    # Who holds each token, for legs whose input cannot be conjured by writing
    # a slot.  A pool's own reserves are exactly the account we need, and the
    # universe already read them.
    holders: dict[str, list[tuple[str, int]]] = {}
    for pool in load.pools:
        # `held`, never `balances`.  The latter is the pool's own accounting,
        # which is a claim rather than a holding -- the same fiction
        # `check_reserves_are_real` exists to catch.  Borrowing against it
        # picks an address that reports reserves and owns nothing, and the
        # transfer then fails for a reason that has nothing to do with the
        # token being tested: eight of eight sampled holders had recorded
        # reserves and a `balanceOf` of zero.
        for coin, held in zip(pool.coins, pool.held, strict=False):
            if held > 0:
                holders.setdefault(coin.address.lower(), []).append((pool.address, held))
    for ranked in holders.values():   # richest first, so a swap of any size funds
        ranked.sort(key=lambda kv: -kv[1])

    executing = LocalEvm(rpc, strict=False)   # may fetch whatever execution touches
    started = _time.perf_counter()
    got = measure_legs(
        executing._evm,
        [(t, k, i, j, token, amount, _holder(holders, token, t))
         for (t, k, i, j), (token, amount) in legs.items()],
        funder=Funder(executing._evm),
    )
    print(f"  measured {len(got['legs'])}/{len(legs)} legs "
          f"in {_time.perf_counter() - started:.0f}s")

    from ..core.pools import registry_key
    from ..core.types import ArcKind  # noqa: F401

    cache = FactsCache.load(chain.chain_id, chain.name.lower())
    changed = cache.learn(
        got["legs"], block=rpc.block,
        classes={p.address.lower(): registry_key(p.pool_type) for p in load.pools},
    )

    # --- what quotes but cannot be traded ---------------------------------
    #
    # No second execution: the gas pass above already ran every leg for real,
    # and a revert there is exactly the evidence wanted here.  Splitting them
    # would double the work and let the two answers drift.
    #
    # A leg that could not be funded is *untested*, not broken.  Conflating
    # them would delete good pools whose input token cannot be conjured, which
    # is a far worse failure than the revert being guarded against.
    if not args.skip_executability:
        broken, healed = {}, []
        for miss in got["failed"]:
            if miss.note.startswith("reverted"):
                broken[cache.key(miss.target, miss.kind, miss.i, miss.j)] = (
                    miss.note.removeprefix("reverted: ").strip() or "reverted")
        measured = {cache.key(t, k, i, j) for (t, k, i, j) in got["legs"]}
        healed = [key for key in cache.broken if key in measured]
        marked = cache.learn_broken(broken)
        cleared = cache.forget_broken(healed)
        print(f"  executability: {len(broken)} reverted, "
              f"{len(got['legs'])} settled  ({marked} newly broken, "
              f"{cleared} recovered)")
        for key, reason in sorted(broken.items())[:8]:
            print(f"      {key:<46} {reason}")

    # --- the survey: protocols that quote and cannot be traded -------------
    #
    # The pass above only sees legs a route chose, so it can verify what we
    # execute but never discover a pool we already refuse -- a blacklisted pool
    # builds no arcs and is never reached again.  These are checked directly,
    # every build, which is also what lets a pool come back if a protocol is
    # unpaused.
    if not args.skip_executability and getattr(chain, "watch", ()):
        from .executability import revert_reason

        found, recovered = {}, []
        for address in chain.watch:
            for i, j in ((0, 1), (1, 0)):
                # 14 is the reserved `SWAP_UNDERLYING` id.  Recording these
                # under SWAP_STABLE collided with the pool's own
                # wrapped-coin arcs, which share (i, j) and are healthy.
                key = cache.key(address, UNDERLYING_KIND, i, j)
                snapshot = executing._evm.snapshot()
                try:
                    token = _underlying_of(executing._evm, address, i)
                    if not token:
                        continue
                    if not Funder(executing._evm).fund(token, address, 10_000 * 10 ** 18):
                        continue
                    executing._evm.message_call(
                        caller=GASCAL_CALLER, to=address,
                        calldata=_underlying_swap(i, j, 10_000 * 10 ** 18))
                    recovered.append(key)
                except Exception as exc:
                    found[key] = revert_reason(exc)
                finally:
                    with contextlib.suppress(Exception):
                        executing._evm.revert(snapshot)
        marked = cache.learn_broken(found)
        cleared = cache.forget_broken(recovered)
        print(f"  watched pools: {len(found)} of {len(chain.watch) * 2} directions "
              f"revert ({marked} new, {cleared} recovered)")
        for key, reason in sorted(found.items()):
            print(f"      {key:<46} {reason}")

    # --- can each wrapper still be entered, and still be left? -------------
    #
    # A property of the token, not of a pool or a swap, so every coin of every
    # pool is asked both ways rather than a list someone maintains.
    #
    # Under boa rather than revm, and only here.  revm refuses a caller that
    # has code (EIP-3607) and every holder of these tokens is a pool, so
    # borrowing from one is impossible there -- which left 19 redemptions
    # untestable and no way to reach them.  `boa.env.prank` has no such rule.
    # It is slower, which is why quoting does not use it, but it costs 71s once
    # per build and takes redemption coverage from 11 of 30 to all of them.
    if not args.skip_executability:
        import boa

        from .executability import discover_wrappers, probe_wrappers_by_prank

        found = discover_wrappers(load.pools, setup)
        richest = {token: ranked[0][0] for token, ranked in holders.items() if ranked}
        boa.fork(_rpc_url(chain, args), block_identifier=rpc.block)
        boa.env._fork_try_prefetch_state = True
        # Not `got`: that name holds `measure_legs`' result and is read again
        # below.  This is the second time shadowing it has crashed the command
        # after the cache was already written, which makes the failure look
        # like the probe never ran.
        able = probe_wrappers_by_prank(found, richest)
        tally = {"mint": 0, "redeem": 0, "refused": 0, "untested": 0}
        for capability in able:
            for way in ("mint", "redeem"):
                state = getattr(capability, way)
                tally[way if state else ("refused" if state is False else "untested")] += 1
            cache.learn_wrapper(
                capability.address, mint=capability.mint, redeem=capability.redeem,
                note=capability.notes.get("redeem") or capability.notes.get("mint") or "")
        print(f"  wrappers: {len(found)} token(s) wrap another -- "
              f"{tally['mint']} mintable, {tally['redeem']} redeemable, "
              f"{tally['refused']} refused, {tally['untested']} untested")

    # --- the survey: protocols that quote and cannot be traded -------------
    #
    # The pass above only sees legs a route chose, so it can verify what we
    # execute but never discover a pool we already refuse -- a blacklisted pool
    # builds no arcs and is never reached again.  These are checked directly,
    # every build, which is also what lets a pool come back if a protocol is
    # unpaused.
    if not args.skip_executability and getattr(chain, "watch", ()):
        from .executability import revert_reason

        found, recovered = {}, []
        for address in chain.watch:
            for i, j in ((0, 1), (1, 0)):
                # 14 is the reserved `SWAP_UNDERLYING` id.  Recording these
                # under SWAP_STABLE collided with the pool's own
                # wrapped-coin arcs, which share (i, j) and are healthy.
                key = cache.key(address, UNDERLYING_KIND, i, j)
                snapshot = executing._evm.snapshot()
                try:
                    token = _underlying_of(executing._evm, address, i)
                    if not token:
                        continue
                    if not Funder(executing._evm).fund(token, address, 10_000 * 10 ** 18):
                        continue
                    executing._evm.message_call(
                        caller=GASCAL_CALLER, to=address,
                        calldata=_underlying_swap(i, j, 10_000 * 10 ** 18))
                    recovered.append(key)
                except Exception as exc:
                    found[key] = revert_reason(exc)
                finally:
                    with contextlib.suppress(Exception):
                        executing._evm.revert(snapshot)
        marked = cache.learn_broken(found)
        cleared = cache.forget_broken(recovered)
        print(f"  watched pools: {len(found)} of {len(chain.watch) * 2} directions "
              f"revert ({marked} new, {cleared} recovered)")
        for key, reason in sorted(found.items()):
            print(f"      {key:<46} {reason}")

    # --- the survey: protocols that quote and cannot be traded -------------
    #
    # The pass above only sees legs a route chose, so it can verify what we
    # execute but never discover a pool we already refuse -- a blacklisted pool
    # builds no arcs and is never reached again.  These are checked directly,
    # every build, which is also what lets a pool come back if a protocol is
    # unpaused.
    if not args.skip_executability and getattr(chain, "watch", ()):
        from .executability import revert_reason

        found, recovered = {}, []
        for address in chain.watch:
            for i, j in ((0, 1), (1, 0)):
                # 14 is the reserved `SWAP_UNDERLYING` id.  Recording these
                # under SWAP_STABLE collided with the pool's own
                # wrapped-coin arcs, which share (i, j) and are healthy.
                key = cache.key(address, UNDERLYING_KIND, i, j)
                snapshot = executing._evm.snapshot()
                try:
                    token = _underlying_of(executing._evm, address, i)
                    if not token:
                        continue
                    if not Funder(executing._evm).fund(token, address, 10_000 * 10 ** 18):
                        continue
                    executing._evm.message_call(
                        caller=GASCAL_CALLER, to=address,
                        calldata=_underlying_swap(i, j, 10_000 * 10 ** 18))
                    recovered.append(key)
                except Exception as exc:
                    found[key] = revert_reason(exc)
                finally:
                    with contextlib.suppress(Exception):
                        executing._evm.revert(snapshot)
        marked = cache.learn_broken(found)
        cleared = cache.forget_broken(recovered)
        print(f"  watched pools: {len(found)} of {len(chain.watch) * 2} directions "
              f"revert ({marked} new, {cleared} recovered)")
        for key, reason in sorted(found.items()):
            print(f"      {key:<46} {reason}")

    # --- can a lending wrapper still be entered, and still be left? ---------
    #
    # Both directions attempted separately, because on a deprecated protocol
    # they genuinely differ.  What comes out of this gates `build_lending_arcs`:
    # a direction absent here is never built, so a paused mint stays out of the
    # graph without anyone maintaining a list, and returns on its own if the
    # protocol reopens.
    if not args.skip_executability and getattr(chain, "wrappers", ()):
        from .executability import try_wrapper

        for token, underlying, family in chain.wrappers:
            # Not `got`: that name holds `measure_legs`' result, which is read
            # again below.  Shadowing it crashed the command after the cache
            # had already been written, so the failure looked like the probe
            # never running.
            able = try_wrapper(executing._evm, Funder(executing._evm), token=token,
                               underlying=underlying, family=family,
                               amount=10_000 * 10 ** 8)
            note = able.notes.get("mint") or able.notes.get("redeem") or ""
            if cache.learn_wrapper(able.address, mint=able.mint, redeem=able.redeem,
                                   note=note):
                def word(state):
                    return "yes" if state else ("no" if state is False else "untested")
                print(f"      {able.address[:12]} {family:<8} "
                      f"mint {word(able.mint):<8} redeem {word(able.redeem):<8} "
                      f"{note[:40]}")

    # --- what each pair does on its own ------------------------------------
    #
    # The floor a routing gain must clear is the pair's own movement.  Measured
    # from a pool holding both tokens, which is the rate itself -- an earlier
    # version stored one price per token and divided, and each token was priced
    # in whatever its own deepest pool paired it with, so the ratio was
    # meaningless.  One series per arc, one request per block.
    # Gated on its own flag, not on `--skip-executability`.  These two answer
    # unrelated questions and age at different rates: what a pool costs to
    # execute holds for months, while how far its rate moves against its own
    # bound is this week's market.
    if not args.skip_drift:
        from .drift import SAMPLE_BLOCKS, SAMPLE_FRACTION, sample_rates

        deepest: dict[str, tuple] = {}
        for pool in load.pools:
            kind = pool.swap_kind
            if kind is None or not pool.balances:
                continue
            for i, j in pool.swap_pairs():
                if i >= len(pool.balances) or pool.balances[i] <= 0:
                    continue
                # Every pool holding the pair, not just the deepest: the
                # busiest one is what reveals the movement, and it is not
                # always the biggest.
                # Canonical, not raw.  ETH and WETH are one node, so pools
                # quoting each of them against stETH answer the same question;
                # keyed raw they split into two keys over disjoint pools and
                # disagreed -- 0.0533 bp against 1.2915 for the same pair.
                key = (f"{nodes.canonical(pool.coins[i].address)}"
                       f"|{nodes.canonical(pool.coins[j].address)}"
                       f"@{pool.address.lower()}")
                dx = max(int(pool.balances[i] * SAMPLE_FRACTION), 1)
                deepest[key] = (key, pool.address, int(kind), i, j, pool.n_coins,
                                dx, pool.tvl_usd)
        arcs = [entry[:-1] for entry in deepest.values()]
        _quoter = quoter_client(rpc, chain)
        _overrides = getattr(_quoter, "overrides", None)
        rates = sample_rates(rpc, _quoter.address, arcs, overrides=_overrides)
        moved = cache.learn_prices(rates)
        print(f"  drift: {len(rates)} pair rate(s) sampled at {len(SAMPLE_BLOCKS)} "
              f"blocks ({moved} changed)")

        # --- how often each pool's own minimum-out would trip ---------------
        #
        # The same arcs, resampled at a minute apart instead of hours, scored
        # against the bound the executor will really set: 20% of the pool's fee,
        # floored for the pools on the exception list.  One request per block
        # for the whole universe, so the second sweep costs what the first did.
        #
        # The list comes from the *four-hour* series above rather than from
        # this one, and that ordering matters: what it asks is what the pair
        # does, and half an hour of a quiet pool cannot answer it.
        from .revert_risk import (
            FINE_BLOCKS,
            breach_risk,
            read_fees,
            wide_bound_pools,
        )

        client = quoter_client(rpc, chain)
        fees = read_fees(client, load.pools)
        fine = sample_rates(rpc, client.address, arcs, blocks=FINE_BLOCKS,
                            overrides=getattr(client, "overrides", None))
        # Scored twice against the same samples.  The first pass puts every
        # pool on its fee-derived bound, which is the evidence for whether that
        # bound is survivable; the list is drawn from it and from the pair
        # drift; the second pass is the answer, against the bound the executor
        # will really set.
        tight = breach_risk(fine, fees, arcs)
        wide = wide_bound_pools(rates, fees, tight)
        changed_wide = cache.learn_wide_bounds(wide)
        by_address = {p.address.lower(): (p.name or p.address)[:20]
                      for p in load.pools}
        print(f"  wide bounds: {len(wide)} pool(s) cannot execute against a "
              f"fraction of their own fee ({changed_wide} changed)")
        for address, entry in sorted(wide.items(), key=lambda kv: -kv[1]["tight_p"])[:6]:
            print(f"      {by_address.get(address, address[:12]):<22}"
                  f"fee {entry['fee_bp']:>6.2f} bp   pair moves "
                  f"{entry['drift_bp']:>8.2f} bp   tight bound trips "
                  f"{entry['tight_p'] * 100:>5.1f}%")
        risk = breach_risk(fine, fees, arcs, wide=set(wide))
        changed_risk = cache.learn_breach(risk)
        if risk:
            worst = sorted(risk.items(), key=lambda kv: -kv[1]["p"])[:3]
            median_p = sorted(e["p"] for e in risk.values())[len(risk) // 2]
            named = ", ".join(
                f"{by_address.get(key.split(':')[0], key[:10])} {e['p'] * 100:.1f}%"
                for key, e in worst)
            print(f"  risk: {len(risk)} arc(s) priced over "
                  f"{len(FINE_BLOCKS)} samples ({changed_risk} changed); "
                  f"median {median_p * 100:.2f}%, worst {named}")
        else:
            print(f"  {WARN} risk: no pool answered fee() with a usable series")

    if not args.dry_run:
        cache.save()
    stats = cache.stats()
    print(f"  {OK} {changed} gas figure(s) changed; {stats['legs']:,} legs, "
          f"{stats['broken']} broken, {stats['classes']} class defaults")
    for kind, count in sorted(stats["by_kind"].items(), key=lambda kv: -kv[1]):
        print(f"      {kind:<22}{count:>5}")

    unfunded = [m for m in got["failed"] if not m.note.startswith("reverted")]
    if unfunded:
        print(f"  {WARN} {len(unfunded)} leg(s) untested -- their input could not be "
              f"funded, so they keep the assumed gas and no verdict:")
        for miss in unfunded[:6]:
            print(f"      {miss.target[:12]} {miss.kind.name:<18} {miss.note[:46]}")
    if args.dry_run:
        print(f"  {WARN} --dry-run: nothing written")
    return 0



def _route_options(args=None) -> dict:
    """The `route()` knobs the CLI exposes, as kwargs -- and only the ones the
    user actually set.

    An unset knob is left out entirely rather than defaulted here, so
    `core.risk.REVERT_COST_BP` and `core.verify.IMPACT_FRACTION` stay the
    single source of truth for their own values instead of being copied.
    """
    options: dict = {}
    got = getattr(args, "revert_cost_bp", None)
    if got is not None:
        options["revert_cost_bp"] = float(got)
    got = getattr(args, "leg_cost_bp", None)
    if got is not None:
        options["leg_cost_bp"] = float(got)
    if getattr(args, "no_impact", False):
        options["measure_impact"] = False
    return options


def _risk_table(chain, args=None):
    """Per-pool minimum-out risk, measured, or nothing at all.

    Every leg executes with a minimum-out at a fraction of its pool's fee, so
    the route lands only if none of those pools moves past its own bound in the
    minute or two before inclusion.  `core/risk.py` turns that into the
    quantity worth ranking on -- expected output rather than quoted output --
    and this hands it the measured probabilities.

    Returning `None` when nothing has been measured is deliberate.  An empty
    table is not a table of zeros: it would price every pool at the default and
    charge a long route several percent on the strength of no evidence.  A
    *partial* table is different -- there the default is filling a gap in a real
    measurement, which is what it is for.

    This replaces a per-pair `min_gain_bp` floor derived from drift.  That
    number was sound as a measurement and wrong as an instrument: it could only
    say "long routes are suspect", where the pool it should have been indicting
    was TriCRV specifically.  The drift series is still collected.
    """
    from .facts import FactsCache

    if getattr(args, "no_risk", False):
        return None, 0
    cache = FactsCache.load(chain.chain_id, chain.name.lower())
    if not cache.breach:
        return None, 0
    return cache.risk_table(), len(cache.breach)


def _gas_table(chain, args, pools=None):
    """Measured execution gas, or the static table when none was measured.

    Off by default is the wrong default here: the figures are committed, so a
    checkout has them, and routing with gas we invented when gas we measured is
    sitting in the repo would be strictly worse.  `--static-gas` opts out, for
    comparing against the old behaviour.
    """
    from ..core.gas import STATIC
    from .facts import FactsCache

    if getattr(args, "static_gas", False):
        return STATIC, 0
    cache = FactsCache.load(chain.chain_id, chain.name.lower())
    return cache.table(pools), len(cache.legs)

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
    from .wrappers import build_lending_arcs, build_node_map, build_stake_arcs

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
    nodes, _ = build_node_map(load.pools, chain, setup,
                              facts=FactsCache.load(chain.chain_id, chain.name.lower()))
    stake = build_stake_arcs(nodes, chain, setup)
    # Leaving a lending wrapper, where `data/facts` says the protocol still
    # allows it.  Rides with the stake arcs: same shape, same treatment.
    stake = stake + build_lending_arcs(
        nodes, chain, setup, FactsCache.load(chain.chain_id, chain.name.lower()))

    # A LLAMMA reads a different set of band slots as the price moves, so its
    # layout is a function of state and cannot be cached.  Recorded as volatile
    # rather than silently cached wrong.
    volatile = [p.address for p in load.pools if (p.pool_type or "").lower() == "llamma"]
    cache.mark_volatile(volatile)

    fresh = cache.unknown(p.address for p in load.pools)
    # The quoter is not a pool, so `unknown` never asks about it -- but the
    # local EVM reads *its* code from this cache too, and a redeployment gives
    # it an address nothing here has seen.  Reporting "nothing to do" while the
    # contract every quote goes through is missing is the wrong answer, so it
    # is checked explicitly.  (In practice a route recovers on its own:
    # `refresh_arcs` passes the quoter to `warm`, which learns it.  This is so
    # the cache is complete before one is run, not after.)
    quoter = (getattr(chain, "quoter", "") or "").lower()
    quoter_known = not quoter or quoter in cache.code_of
    if not quoter_known:
        print(f"  {WARN} quoter {quoter[:12]} is not in the cache -- learning it")
    print(f"  universe: {len(load.pools)} pools, {len(fresh)} to learn "
          f"({len(volatile)} volatile)")
    if not fresh and quoter_known:
        print(f"  {OK} nothing to do")
        return 0

    started = _time.perf_counter()
    # Only the quoter path needs a route to record.  Off mainnet the recorded
    # calls are discarded anyway (see below), and running `prepare` for them
    # meant demanding a src and dst pair that most chains do not have: of 17,
    # thirteen failed here on "no token with symbol 'WETH'" or "no path from
    # USDC to WETH" -- gnosis, polygon, bsc, sonic, avalanche, etherlink,
    # monad, plasma, xlayer, robinhood, celo, tac, fraxtal.  The universe is
    # what is being cached, not a route through it.
    # Record a real route where one can be had.  It covers call shapes
    # `warm_arcs` does not -- ERC4626 reads, wrappers, stake arcs -- so it is
    # worth having, but it is not worth *requiring*: the pair is hardcoded to
    # USDC/WETH and most chains have neither.  Deploying tac's quoter put it
    # back on this path and it failed with "no token with symbol 'USDC'", which
    # is a caching command refusing to cache over a token it was never asked
    # about.
    recorded = False
    if chain.quoter:
        try:
            src_token = _resolve_token(nodes, args.src, load.pools)
            dst_token = _resolve_token(nodes, args.dst, load.pools)
        except Exception as exc:
            print(f"  {WARN} {str(exc)[:60]}; caching the universe instead of a route")
        else:
            try:
                prepare(load.pools, nodes, setup, src_token=src_token,
                        dst_token=dst_token, extra_arcs=stake)
                recorded = True
            except Exception as exc:
                print(f"  {WARN} could not probe {args.src}->{args.dst} ({str(exc)[:40]}); "
                      f"caching the universe instead")
    probing = (_time.perf_counter() - started) * 1000
    print(f"  probed in {probing:,.0f} ms, {len(recorder.calls)} distinct quoter calls")

    evm = LocalEvm(rpc, cache=cache)
    if recorded:
        stats = evm.warm(list(recorder.calls))
    else:
        # The recorded calls all target the quoter, and off mainnet the quoter
        # is not deployed -- it rides along as an `eth_call` state override,
        # which `eth_createAccessList` cannot accept.  Those requests execute
        # against an address with no code and return an empty list: measured
        # on arbitrum, 34 pools yielded 1 account and 0 slots.  Ask the pools
        # for their own `get_dy` instead; it reads the same storage.
        from ..core.pipeline import build_arcs as _build_arcs

        refs, _ = _build_arcs(load.pools, nodes)
        stats = evm.warm_arcs(refs, "")
    learned = cache.stats().slots
    if not learned:
        # Recording the pools as known would make the next run skip them, so a
        # warm that learned nothing would cache its own failure -- which is
        # exactly what happened here before the direct path existed.
        print(f"  {WARN} learned no slots; not marking these pools as known")
    else:
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

    gascal = sub.add_parser(
        "facts",
        help="probe what does not change between blocks -- execution gas, and "
             "which arcs quote but revert; writes data/facts")
    gascal.add_argument("--chain", default="ethereum")
    gascal.add_argument("--block", default="latest")
    gascal.add_argument("--min-tvl", type=float, default=10_000.0)
    gascal.add_argument("--rpc-url", default=None)
    gascal.add_argument("--max-legs", type=int, default=32)
    gascal.add_argument("--max-pairs", type=int, default=40)
    gascal.add_argument("--gas-price-wei", type=int, default=int(0.1e9))
    gascal.add_argument(
        "--pairs", nargs="*", default=None, metavar="SRC-DST",
        help="measure only these pairs, e.g. USDC-WETH stETH-WETH")
    gascal.add_argument("--dry-run", action="store_true",
                        help="measure and report, but do not write the cache")
    gascal.add_argument(
        "--skip-executability", action="store_true",
        help="only re-measure gas, leaving the broken-arc list as it stands")
    gascal.add_argument(
        "--skip-drift", action="store_true",
        help="leave the rate series and per-arc minimum-out risk as they stand")
    gascal.set_defaults(func=cmd_gascal)

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
    bench.add_argument(
        "--static-gas", action="store_true",
        help="price legs from the assumed per-kind table instead of the "
             "measured figures in data/gas -- for comparing against the old "
             "behaviour")
    bench.add_argument(
        "--no-risk", action="store_true",
        help="rank on quoted output alone, ignoring per-pool minimum-out risk")
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
    route_cmd.add_argument(
        "--fresh-quoter", action="store_true",
        help="run the compiled quoter in the local EVM instead of the deployed "
             "one, so leg kinds newer than the deployment can be priced. The "
             "wire path is unaffected and still sees what is on chain")
    route_cmd.add_argument(
        "--static-gas", action="store_true",
        help="price legs from the assumed per-kind table instead of the "
             "measured figures in data/gas -- for comparing against the old "
             "behaviour")
    route_cmd.add_argument(
        "--no-risk", action="store_true",
        help="rank on quoted output alone, ignoring how often each pool's own "
             "minimum-out would trip before the route lands")
    route_cmd.add_argument(
        "--leg-cost-bp", type=float, default=None,
        help="what one leg is charged beyond its gas, in basis points of the "
             "trade (default 0.02). Only bites when gas is near zero; see "
             "scripts/leg_cost_frontier.py")
    route_cmd.add_argument(
        "--no-impact", action="store_true",
        help="skip the price-impact measurement, which re-quotes the finished "
             "route at 5% of the size -- 3 ms against a local EVM, one more "
             "round trip over the wire")
    route_cmd.add_argument(
        "--revert-cost-bp", type=float, default=None,
        help="what one failed attempt costs, in basis points of the trade "
             "(default 1.0): gas plus whatever the price did while the user "
             "resubmitted. Raise it to buy safety with price")
    route_cmd.set_defaults(func=cmd_route)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
