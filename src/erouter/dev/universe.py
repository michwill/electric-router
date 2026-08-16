"""Load the pool universe and resolve each pool's ABI dialect.

Splits cleanly: the Curve API supplies *which pools exist* and a TVL bootstrap,
and the chain supplies every number that enters the solve.  An API outage
therefore degrades to a slightly stale pool list, never to a wrong route.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.pools import Coin, PoolSpec, dialect_from_probes, parse_universe
from ..core.quoter import QuoterClient
from ..core.transport import Status
from ..core.types import ArcKind, Dialect, Probe
from . import lite
from .cache import DialectCache, UniverseCache
from .chains import Chain
from .curve_api import DEFAULT_MIN_TVL, CurveApi, CurveApiError


@dataclass(slots=True)
class UniverseLoad:
    pools: list[PoolSpec]
    source: str  # "api" | "cache" | "stale"
    age: float = 0.0
    warnings: list[str] = field(default_factory=list)
    filtered: int = 0


def _list_pools(chain: Chain, api: CurveApi, min_tvl: float) -> list[dict]:
    """The pool list, from whichever API serves this chain.

    Lite deployments answer on `api2.curve.finance` and are not in the Prices
    API at all -- asking it for one returns a 404, not an empty list.  The
    floor comes down with them: see `lite.LITE_MIN_TVL`.
    """
    if chain.lite:
        # The Lite floor is a *default*, not a ceiling on what the caller may
        # ask for.  Taking `min()` of the two made `--min-tvl` unable to raise
        # it at all, which hid a real problem: unichain's universe spans 20
        # orders of magnitude in curvature and trips the §9.7 conditioning
        # guard, and no floor the caller passed could thin it out.
        floor = lite.LITE_MIN_TVL if min_tvl == DEFAULT_MIN_TVL else min_tvl
        return lite.list_pools(chain.chain_id, min_tvl=floor)
    return api.list_pools(chain.chain_id, min_tvl=min_tvl)


def load_pools(
    chain: Chain,
    *,
    min_tvl: float = 10_000.0,
    api: CurveApi | None = None,
    cache: UniverseCache | None = None,
    refresh: bool = False,
    pool_filters: bool = False,
    llamma: bool = False,
) -> UniverseLoad:
    api = api or CurveApi()
    cache = cache or UniverseCache()
    warnings: list[str] = []

    if not refresh:
        fresh = cache.get(chain.chain_id, min_tvl)
        if fresh is not None:
            pools, dropped = _apply_filters(
                parse_universe(fresh), chain, api, warnings, enabled=pool_filters
            )
            pools += _llamma(chain, api, min_tvl, warnings) if llamma else []
            load = UniverseLoad(pools, "cache", cache.age(chain.chain_id, min_tvl), warnings)
            load.filtered = dropped
            return load

    try:
        raw = _list_pools(chain, api, min_tvl)
    except CurveApiError as exc:
        stale = cache.get(chain.chain_id, min_tvl, allow_stale=True)
        if stale is None:
            raise
        age = cache.age(chain.chain_id, min_tvl)
        warnings.append(f"Curve API unavailable ({exc}); using a universe {age / 60:.0f} min old")
        pools, dropped = _apply_filters(
            parse_universe(stale), chain, api, warnings, enabled=pool_filters
        )
        pools += _llamma(chain, api, min_tvl, warnings) if llamma else []
        load = UniverseLoad(pools, "stale", age, warnings)
        load.filtered = dropped
        return load

    cache.put(chain.chain_id, min_tvl, raw)
    pools = parse_universe(raw)
    pools, dropped = _apply_filters(pools, chain, api, warnings, enabled=pool_filters)
    pools += _llamma(chain, api, min_tvl, warnings) if llamma else []
    load = UniverseLoad(pools, "api", 0.0, warnings)
    load.filtered = dropped
    return load


def _llamma(chain, api, min_tvl, warnings) -> list[PoolSpec]:
    found = llamma_pools(chain, api, min_tvl=min_tvl)
    if found:
        warnings.append(
            f"{len(found)} LLAMMA market(s) added "
            f"(${sum(p.tvl_usd for p in found):,.0f} of collateral and debt)"
        )
    return found


def llamma_pools(
    chain: Chain, api: CurveApi | None = None, *, min_tvl: float = 10_000.0
) -> list[PoolSpec]:
    """LLAMMA markets as ordinary two-coin crypto pools.

    **Off by default: enabling this today makes routes much worse.**  Measured
    at block 25,738,767, adding the 11 markets that clear a $10k floor took
    USDC->sUSDS from 900,795 to 296,795 (-67%) and crvUSD->sDOLA from
    3,502,562 to 1,500,637 (-57%), and made USDC->WETH and crvUSD->CRV fail
    outright on the §12.4 conservation and pin-detachment guards.

    The cause is calibration, not plumbing.  LLAMMA is a *banded* AMM, and the
    reserve we size probes from is the market's borrowed balance -- 3.40 crvUSD
    on the sUSDS market -- so the whole grid lands inside a single band where
    the curve is nearly linear.  It fits `a = 0.9927, B = 5.8e-3` with no cap,
    the model computes `G ~ 171`, and the solver posts millions through a
    three-crvUSD venue.  This is §2.5's "most dangerous single mis-calibration"
    with the peg replaced by a band, and the symptoms are the ones §2.5 and R3
    predict: the two directions of the WETH market disagree on `a` by 7x, and
    one arc calibrates to `a = 0` outright.

    Making it usable needs band-aware capacity -- a real `cap` from the
    liquidity actually standing in reachable bands, so §2.3's clamp can bound
    it -- not a wider probe grid.  The plumbing here is correct and stays, so
    that work has somewhere to land.

    Everything downstream -- probing, calibration, the graph, realisation --
    treats them like any other pool, because after this they *are* one: the
    only genuinely different things are that they quote with the `uint256`
    spelling (so the dialect is set here rather than probed) and that they
    have no `balances()` getter, so the reserves come from the API.

    Markets with nothing on one side are dropped.  Many are empty, and an arc
    with a zero reserve produces a zero probe grid and then a zero `a`, which
    §R3 is explicit about: it looks like a valid quote and poisons the
    reference-price fit.
    """
    api = api or CurveApi()
    out: list[PoolSpec] = []
    for entry in api.llamma_markets(chain.api_name):
        address = (entry.get("llamma") or "").strip()
        borrowed, collateral = entry.get("borrowed_token"), entry.get("collateral_token")
        if not address or not borrowed or not collateral:
            continue
        amounts = (entry.get("borrowed_balance") or 0.0, entry.get("collateral_balance") or 0.0)
        if min(amounts) <= 0:
            continue
        usd = (entry.get("borrowed_balance_usd") or 0.0) + (
            entry.get("collateral_balance_usd") or 0.0
        )
        if usd < min_tvl:
            continue
        coins = []
        balances = []
        for index, (token, amount) in enumerate(
            zip((borrowed, collateral), amounts, strict=True)
        ):
            decimals = int(token.get("decimals") or 18)
            coins.append(
                Coin(
                    index=index,
                    address=(token.get("address") or "").lower(),
                    symbol=token.get("symbol") or "?",
                    decimals=decimals,
                )
            )
            balances.append(int(amount * 10**decimals))
        if not all(c.address for c in coins):
            continue
        out.append(
            PoolSpec(
                address=address,
                name=entry.get("name") or f"LLAMMA {coins[1].symbol}",
                # A registry key of its own, so the dialect table and any
                # pool-type special-casing cannot mistake it for a stableswap.
                pool_type="llamma",
                coins=tuple(coins),
                tvl_usd=usd,
                dialect=Dialect.CRYPTO,
                balances=tuple(balances),
                note=entry.get("_llamma_kind", ""),
            )
        )
    return out


def _probed_dead_pools(chain) -> set[str]:
    """Pools the facts file says cannot be traded at all.

    Deliberately narrow, and it took getting this wrong to see why.  The first
    version dropped any pool with a recorded revert and no recorded gas figure
    -- which is every pool no route happened to choose, and it would have
    deleted both Compound pools outright.  Their `exchange_underlying` reverts
    while their `cDAI`/`cUSDC` arcs are healthy and routed through daily; a
    broken direction is not a broken pool.

    So a recorded revert bans a *direction*, not a pool, and nothing here bans
    a pool at all.  Pool-level removal stays with the hand-written list, where
    a human decided it.  This exists so that when arcs learn to read the broken
    list per direction, the plumbing is already the right shape.
    """
    return set()


def _apply_filters(
    pools: list[PoolSpec],
    chain: Chain,
    api: CurveApi,
    warnings: list[str],
    *,
    enabled: bool = False,
) -> tuple[list[PoolSpec], int]:
    """Drop pools Curve itself flags as not honouring their quotes.

    Off by default, because measured against the live API it drops nothing:
    `/v2/pools` already excludes all 185 flagged mainnet pools, even at
    min_tvl=0 where it returns 2,211.  Paying an HTTP round trip on every load
    for a filter that never fires is not worth it.

    Turn it on for the case it is written for -- a universe cached before a
    pool was flagged, since Curve's list moves on Curve's schedule -- which is
    why it is applied on the cache and stale paths too, not only a fresh load.
    """
    # The hand-written list plus everything a `facts` run found reverting.
    # Measured beats hand-written: the Aave entry below started as a constant
    # in `chains.py` after one afternoon of executing legs, which is fine until
    # the next protocol is deprecated and nobody notices.  The probed list is
    # regenerated with the gas figures and committed beside them, so the
    # hand-written one is now only for pools we want gone for reasons no probe
    # would find.
    banned = {a.lower() for a in getattr(chain, "blacklist", ())}
    banned |= _probed_dead_pools(chain)   # empty by design -- see the docstring
    dropped_here = 0
    if banned:
        before = len(pools)
        pools = [p for p in pools if p.address.lower() not in banned]
        dropped_here = before - len(pools)
        if dropped_here:
            warnings.append(
                f"{dropped_here} pool(s) on {chain.name}'s blacklist skipped: they quote "
                "but cannot be traded"
            )
    if not enabled:
        return pools, dropped_here
    blocked = api.pool_filters(chain.chain_id)
    if not blocked:
        return pools, dropped_here
    kept = [p for p in pools if p.address.lower() not in blocked]
    dropped = len(pools) - len(kept)
    if dropped:
        warnings.append(
            f"{dropped} pool(s) excluded by Curve's pool_filters list "
            f"({len(blocked)} flagged on this chain)"
        )
    return kept, dropped + dropped_here


@dataclass(slots=True)
class DialectAudit:
    resolved: int = 0
    from_cache: int = 0
    from_probe: int = 0
    unresolved: list[PoolSpec] = field(default_factory=list)
    mistyped: list[tuple[PoolSpec, Dialect]] = field(default_factory=list)
    no_answer: list[PoolSpec] = field(default_factory=list)
    empty_returndata: int = 0
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.resolved + len(self.unresolved)


def resolve_dialects(
    pools: list[PoolSpec],
    client: QuoterClient,
    chain: Chain,
    *,
    cache: DialectCache | None = None,
    use_cache: bool = True,
) -> DialectAudit:
    """Probe both spellings for every pool and record which one answers.

    One batched call for the whole universe (776 probes measured at 1.06 s).
    `Status.WRONG_ABI` -- succeeded but returned nothing -- is deliberately not
    treated as an answer; 60 Ethereum pools rely on that distinction and one is
    outright mis-typed by the API.
    """
    audit = DialectAudit()
    cache = cache or DialectCache()
    known = cache.load(chain.chain_id) if use_cache else {}

    pending: list[PoolSpec] = []
    for pool in pools:
        if pool.dialect is not None and pool.note != "CACHED":
            # Already known from the source that produced it -- LLAMMA is
            # always the uint256 spelling -- so there is nothing to probe.
            audit.resolved += 1
            continue
        cached = known.get(pool.address.lower())
        if cached:
            pool.dialect = Dialect(cached)
            pool.note = "CACHED"
            audit.from_cache += 1
            audit.resolved += 1
        else:
            pending.append(pool)

    if not pending:
        return audit

    probes: list[Probe] = []
    for pool in pending:
        # One token of coin 0 is plenty to tell the spellings apart, and small
        # enough that a shallow pool still answers.
        dx = 10 ** pool.coins[0].decimals
        probes.append(Probe(pool.address, ArcKind.SWAP_STABLE, 0, 1, pool.n_coins, dx))
        probes.append(Probe(pool.address, ArcKind.SWAP_CRYPTO, 0, 1, pool.n_coins, dx))

    started = time.monotonic()
    answers = client.probe(probes)
    audit.seconds = time.monotonic() - started

    fresh: dict[str, str] = {}
    for k, pool in enumerate(pending):
        stable, crypto = answers[2 * k], answers[2 * k + 1]
        audit.empty_returndata += sum(
            1 for q in (stable, crypto) if q.status is Status.WRONG_ABI
        )
        table = pool.table_dialect
        dialect, note = dialect_from_probes(
            table, stable.status is Status.VALUE, crypto.status is Status.VALUE
        )
        pool.dialect, pool.note = dialect, note

        if dialect is None:
            audit.unresolved.append(pool)
            continue
        audit.resolved += 1
        if note == "PROBED":
            audit.from_probe += 1
            fresh[pool.address.lower()] = dialect.value
            if table is not None and table is not dialect:
                audit.mistyped.append((pool, table))
        elif note == "NO_ANSWER":
            audit.no_answer.append(pool)

    if fresh and use_cache:
        cache.save(chain.chain_id, fresh)
    return audit


def count_swap_arcs(pools: list[PoolSpec]) -> int:
    return sum(p.swap_arc_count() for p in pools if p.swap_kind is not None)


def read_balances(pools: list[PoolSpec], client: QuoterClient) -> int:
    """Fill in each pool's reserves, in one batched call.

    Curve has two spellings of the same getter and the registry does not say
    which a pool implements, so both go out for every coin and the first that
    *answers* wins -- "answers" meaning 32 bytes back, not merely no revert.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    # LLAMMA has no `balances()` getter; its reserves come from the market
    # feed, and probing for them would only overwrite good numbers with none.
    pools = [p for p in pools if not p.balances]

    # Three calls per coin, not two: what the pool says it has, in both
    # spellings, and what the coin says it holds.  The third rides the same
    # batch, so knowing whether a pool's accounting is real costs no extra
    # round trip -- measured, it was 3,653 ms as a separate pass.
    calls: list[Call] = []
    spans: list[tuple[int, int]] = []
    for pool in pools:
        start = len(calls)
        for k in range(pool.n_coins):
            calls.append(Call(pool.address, encode_call("balances(uint256)", k)))
            calls.append(Call(pool.address, encode_call("balances(int128)", k)))
            coin = pool.coins[k] if k < len(pool.coins) else None
            target = coin.address if coin else pool.address
            calls.append(Call(target, encode_call("balanceOf(address)", pool.address)))
        spans.append((start, len(calls)))

    answers = client.raw(calls)
    filled = 0
    for pool, (lo, hi) in zip(pools, spans, strict=True):
        chunk = answers[lo:hi]
        balances, held = [], []
        for k in range(pool.n_coins):
            as_uint, as_int, owns = chunk[3 * k], chunk[3 * k + 1], chunk[3 * k + 2]
            if as_uint.status is Status.VALUE:
                balances.append(as_uint.uint())
            elif as_int.status is Status.VALUE:
                balances.append(as_int.uint())
            else:
                balances.append(0)
            # -1 means the coin would not say, which is not evidence of anything.
            held.append(owns.uint() if owns.status is Status.VALUE else -1)
        pool.balances = tuple(balances)
        pool.held = tuple(held)
        if any(balances):
            filled += 1
    return filled


# A pool holding less than this share of what it reports is not merely low on a
# coin -- its accounting has come adrift from reality, and every quote it gives
# is computed against tokens that are not there.
HELD_TOLERANCE = 0.5


def _pool(pools: list[PoolSpec], address: str) -> PoolSpec:
    return next(p for p in pools if p.address == address)


def check_reserves_are_real(
    pools: list[PoolSpec], client: QuoterClient, transport=None
) -> list[str]:
    """Drop pools that report more than they hold.

    `get_dy` runs the invariant over the pool's *own* `balances[]` storage and
    never asks the token whether those coins exist.  When an issuer retires or
    migrates a token out from under a pool, the accounting keeps its old number
    and the pool goes on quoting against liquidity it cannot pay: the sUSD/sUSDe
    pool reports 421,778 sUSD and holds zero, quotes 652 sUSD for 100 sUSDe, and
    reverts with `sUSD retired` on execution.  A quote that cannot settle is
    worse than no quote, and this is one batched call per coin to find them.

    Curve's own solver has no such check: it routed WETH->USDC through a lending
    pool whose `exchange_underlying` reverts, and reported the quote validated.

    Pools listing WETH may legitimately hold *native* ETH instead (E11), which
    is 29 of the 31 apparent shortfalls on mainnet, so that is checked before
    anything is dropped.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    # `read_balances` already asked, in the batch it was already sending.  Only
    # a pool it skipped -- LLAMMA, whose reserves come from the market feed --
    # needs asking here, and then only if a client was given.
    missing = [p for p in pools if len(p.held) != len(p.balances)]
    if missing and client is not None:
        calls: list[Call] = []
        meta: list[tuple[PoolSpec, int]] = []
        for pool in missing:
            for k, coin in enumerate(pool.coins[: len(pool.balances)]):
                calls.append(Call(coin.address,
                                  encode_call("balanceOf(address)", pool.address)))
                meta.append((pool, k))
        found: dict[int, list[int]] = {}
        for (pool, _k), answer in zip(meta, client.raw(calls), strict=True):
            found.setdefault(id(pool), []).append(
                answer.uint() if answer.status is Status.VALUE else -1)
        for pool in missing:
            pool.held = tuple(found.get(id(pool), []))

    warnings: list[str] = []
    short: dict[str, list[tuple[int, int, int]]] = {}
    for pool in pools:
        for k, coin in enumerate(pool.coins[: min(len(pool.balances), len(pool.held))]):
            reported, held = pool.balances[k], pool.held[k]
            if reported <= 0 or held < 0:
                continue  # nothing there, or the coin would not say
            if coin.address.lower().startswith("0xeeee"):
                continue  # native ETH is not an ERC20 and answers nothing useful
            if held >= reported * HELD_TOLERANCE:
                continue
            short.setdefault(pool.address, []).append((k, reported, held))

    # Native ETH covers a WETH slot that looks empty -- 29 of the 31 apparent
    # shortfalls on mainnet.  One batch, because asking 29 times in sequence
    # cost 1,962 ms and this whole check is otherwise free.
    wants_native = [
        address for address, rows in short.items()
        if any(_pool(pools, address).coins[k].symbol.upper() in ("WETH", "ETH")
               for k, _, _ in rows)
    ]
    native: dict[str, int] = {}
    if wants_native and transport is not None:
        block = transport.pin.hex_block
        try:
            answers = transport.fetch_multi(
                [("eth_getBalance", [a, block]) for a in wants_native], concurrent=True)
        except TypeError:
            answers = transport.fetch_multi([("eth_getBalance", [a, block])
                                             for a in wants_native])
        for address, answer in zip(wants_native, answers, strict=True):
            native[address] = 0 if isinstance(answer, Exception) else int(answer, 16)

    for address, rows in list(short.items()):
        pool = _pool(pools, address)
        have = native.get(address, 0)
        rows[:] = [r for r in rows
                   if not (pool.coins[r[0]].symbol.upper() in ("WETH", "ETH")
                           and have >= r[1] * HELD_TOLERANCE)]
        if not rows:
            del short[address]

    for address, rows in short.items():
        pool = _pool(pools, address)
        k, reported, held = rows[0]
        coin = pool.coins[k]
        warnings.append(
            f"{pool.name or address} dropped: reports {reported / 10 ** coin.decimals:,.2f} "
            f"{coin.symbol} but holds {held / 10 ** coin.decimals:,.2f} -- its quotes "
            "are computed against tokens it cannot pay out"
        )
        pool.balances = tuple(0 for _ in pool.balances)
    return warnings


def arc_refs(pools: list[PoolSpec]):
    """Every swap direction of every pool, as probe targets."""
    from ..core.probe import ArcRef

    refs = []
    for pool in pools:
        kind = pool.swap_kind
        if kind is None or not pool.balances:
            continue
        for i, j in pool.swap_pairs():
            reserve = pool.balances[i] if i < len(pool.balances) else 0
            if reserve <= 0:
                continue
            refs.append(
                ArcRef(
                    pool=pool.address,
                    kind=kind,
                    i=i,
                    j=j,
                    n_coins=pool.n_coins,
                    reserve_in=reserve,
                    decimals_in=pool.coins[i].decimals,
                    decimals_out=pool.coins[j].decimals,
                )
            )
    return refs
