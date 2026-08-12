"""Load the pool universe and resolve each pool's ABI dialect.

Splits cleanly: the Curve API supplies *which pools exist* and a TVL bootstrap,
and the chain supplies every number that enters the solve.  An API outage
therefore degrades to a slightly stale pool list, never to a wrong route.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.pools import PoolSpec, dialect_from_probes, parse_universe
from ..core.quoter import QuoterClient
from ..core.transport import Status
from ..core.types import ArcKind, Dialect, Probe
from .cache import DialectCache, UniverseCache
from .chains import Chain
from .curve_api import CurveApi, CurveApiError


@dataclass(slots=True)
class UniverseLoad:
    pools: list[PoolSpec]
    source: str  # "api" | "cache" | "stale"
    age: float = 0.0
    warnings: list[str] = field(default_factory=list)
    filtered: int = 0


def load_pools(
    chain: Chain,
    *,
    min_tvl: float = 10_000.0,
    api: CurveApi | None = None,
    cache: UniverseCache | None = None,
    refresh: bool = False,
) -> UniverseLoad:
    api = api or CurveApi()
    cache = cache or UniverseCache()
    warnings: list[str] = []

    if not refresh:
        fresh = cache.get(chain.chain_id, min_tvl)
        if fresh is not None:
            pools, dropped = _apply_filters(parse_universe(fresh), chain, api, warnings)
            load = UniverseLoad(pools, "cache", cache.age(chain.chain_id, min_tvl), warnings)
            load.filtered = dropped
            return load

    try:
        raw = api.list_pools(chain.chain_id, min_tvl=min_tvl)
    except CurveApiError as exc:
        stale = cache.get(chain.chain_id, min_tvl, allow_stale=True)
        if stale is None:
            raise
        age = cache.age(chain.chain_id, min_tvl)
        warnings.append(f"Curve API unavailable ({exc}); using a universe {age / 60:.0f} min old")
        pools, dropped = _apply_filters(parse_universe(stale), chain, api, warnings)
        load = UniverseLoad(pools, "stale", age, warnings)
        load.filtered = dropped
        return load

    cache.put(chain.chain_id, min_tvl, raw)
    pools = parse_universe(raw)
    pools, dropped = _apply_filters(pools, chain, api, warnings)
    load = UniverseLoad(pools, "api", 0.0, warnings)
    load.filtered = dropped
    return load


def _apply_filters(
    pools: list[PoolSpec], chain: Chain, api: CurveApi, warnings: list[str]
) -> tuple[list[PoolSpec], int]:
    """Drop pools Curve itself flags as not honouring their quotes.

    Applied after the TVL floor and outside the universe cache, because the
    list changes on Curve's schedule rather than ours -- a pool can be flagged
    long after it was cached, and continuing to route through it because of a
    stale snapshot is the whole failure this prevents.
    """
    blocked = api.pool_filters(chain.chain_id)
    if not blocked:
        return pools, 0
    kept = [p for p in pools if p.address.lower() not in blocked]
    dropped = len(pools) - len(kept)
    if dropped:
        warnings.append(
            f"{dropped} pool(s) excluded by Curve's pool_filters list "
            f"({len(blocked)} flagged on this chain)"
        )
    return kept, dropped


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

    calls: list[Call] = []
    spans: list[tuple[int, int]] = []
    for pool in pools:
        start = len(calls)
        for k in range(pool.n_coins):
            calls.append(Call(pool.address, encode_call("balances(uint256)", k)))
            calls.append(Call(pool.address, encode_call("balances(int128)", k)))
        spans.append((start, len(calls)))

    answers = client.raw(calls)
    filled = 0
    for pool, (lo, hi) in zip(pools, spans, strict=True):
        chunk = answers[lo:hi]
        balances = []
        for k in range(pool.n_coins):
            as_uint, as_int = chunk[2 * k], chunk[2 * k + 1]
            if as_uint.status is Status.VALUE:
                balances.append(as_uint.uint())
            elif as_int.status is Status.VALUE:
                balances.append(as_int.uint())
            else:
                balances.append(0)
        pool.balances = tuple(balances)
        if any(balances):
            filled += 1
    return filled


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
