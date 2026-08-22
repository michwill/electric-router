"""Load the pool universe and resolve each pool's ABI dialect.

Splits cleanly: the Curve API supplies *which pools exist* and a TVL bootstrap,
and the chain supplies every number that enters the solve.  An API outage
therefore degrades to a slightly stale pool list, never to a wrong route.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from ..core.pools import Coin, PoolSpec, dialect_from_probes, parse_universe
from ..core.quoter import QuoterClient
from ..core.transport import Status
from ..core.types import ArcKind, Dialect, Probe
from . import lite
from .cache import DialectCache, TokenFactsCache, UniverseCache
from .chains import Chain
from .curve_api import DEFAULT_MIN_TVL, CurveApi, CurveApiError


@dataclass(slots=True)
class UniverseLoad:
    pools: list[PoolSpec]
    source: str  # "api" | "cache" | "stale" | "pinned"
    age: float = 0.0
    warnings: list[str] = field(default_factory=list)
    filtered: int = 0

    @property
    def fingerprint(self) -> str:
        """Which pool set this is, in eight characters.

        The pool list comes from an API on a five-minute TTL, so it is the one
        input to a quote that `--block` does not pin.  Two runs that disagree are
        only comparable if this matches.
        """
        import hashlib

        joined = "".join(sorted(p.address.lower() for p in self.pools))
        return hashlib.sha256(joined.encode()).hexdigest()[:8]


def _list_pools(chain: Chain, api: CurveApi, min_tvl: float) -> list[dict]:
    """The pool list, from whichever API serves this chain.

    Lite deployments answer on `api2.curve.finance` and are not in the Prices
    API at all -- asking it for one returns a 404, not an empty list.  The
    floor comes down with them: see `lite.LITE_MIN_TVL`.
    """
    if chain.lite:
        # The Lite floor is a *default*, not a ceiling on what the caller may
        # ask for.  Taking `min()` of the two made `--min-tvl` unable to raise it
        # at all, which hid unichain tripping the §9.7 conditioning guard with no
        # floor the caller passed able to thin it out.
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
    pin: bool = False,
) -> UniverseLoad:
    api = api or CurveApi()
    cache = cache or UniverseCache()
    warnings: list[str] = []

    if pin:
        # Whatever is on disk, however old, and never a request.  A pinned
        # block with an unpinned pool list is not a pinned quote: the list is
        # what decides which pools exist and what the price fit is weighted by.
        held = cache.get(chain.chain_id, min_tvl, allow_stale=True)
        if held is not None:
            pools, dropped = _apply_filters(
                parse_universe(held), chain, api, warnings, enabled=pool_filters
            )
            pools += _llamma(chain, api, min_tvl, warnings) if llamma else []
            load = UniverseLoad(pools, "pinned",
                                cache.age(chain.chain_id, min_tvl), warnings)
            load.filtered = dropped
            return load
        warnings.append("no cached universe to pin to; fetching one")

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

    **Off by default: enabling this today makes routes much worse.**  Measured at
    block 25,738,767, adding the 11 markets that clear a $10k floor cost
    USDC->sUSDS 67% and crvUSD->sDOLA 57%, and made USDC->WETH and crvUSD->CRV
    fail the §12.4 conservation and pin-detachment guards outright.

    The cause is calibration, not plumbing.  LLAMMA is a *banded* AMM, and the
    reserve we size probes from is the market's borrowed balance -- 3.40 crvUSD
    on the sUSDS market -- so the whole grid lands inside a single band where the
    curve is nearly linear.  It fits no cap, the model computes `G ~ 171`, and the
    solver posts millions through a three-crvUSD venue.  This is §2.5's "most
    dangerous single mis-calibration" with the peg replaced by a band.

    Making it usable needs band-aware capacity -- a real `cap` from the liquidity
    actually standing in reachable bands, so §2.3's clamp can bound it -- not a
    wider probe grid.  The plumbing here is correct and stays, so that work has
    somewhere to land.

    Everything downstream treats them like any other pool, because after this they
    *are* one: the only differences are that they quote with the `uint256`
    spelling (so the dialect is set here rather than probed) and that they have no
    `balances()` getter, so the reserves come from the API.

    Markets with nothing on one side are dropped: an arc with a zero reserve
    produces a zero probe grid and then a zero `a`, which looks like a valid quote
    and poisons the reference-price fit (§R3).
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

    Deliberately narrow.  A recorded revert bans a *direction*, not a pool: both
    Compound pools have a reverting `exchange_underlying` and healthy
    `cDAI`/`cUSDC` arcs routed through daily, and the first version of this would
    have deleted them outright.

    Nothing here bans a pool.  Pool-level removal stays with the hand-written
    list, where a human decided it.
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
    min_tvl=0.  Turn it on for the case it is written for -- a universe cached
    before a pool was flagged -- which is why it is applied on the cache and
    stale paths too, not only a fresh load.
    """
    # The hand-written list plus everything a `facts` run found reverting.
    # Measured beats hand-written: the probed list is regenerated with the gas
    # figures and committed beside them, so the hand-written one is now only for
    # pools we want gone for reasons no probe would find.
    banned = {a.lower() for a in getattr(chain, "blacklist", ())}
    banned |= _probed_dead_pools(chain)   # empty by design -- see the docstring
    dropped_here = 0
    if banned:
        before = len(pools)
        pools = [p for p in pools if p.address.lower() not in banned]
        dropped_here = before - len(pools)
        if dropped_here:
            warnings.append(
                f"{dropped_here} pool(s) on {chain.name}'s blacklist skipped: a quote "
                "that cannot execute, or a pool with no solvable invariant"
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
    #: One line per pool whose coin list the API over-reported; see
    #: `resolve_coin_counts`.  Surfaced rather than counted, because a pool
    #: whose N moved is a pool whose arcs and invariant both moved with it.
    coin_notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.resolved + len(self.unresolved)


def resolve_lp_tokens(pools: list[PoolSpec], client: QuoterClient,
                      chain_id: int | None = None,
                      token_client: QuoterClient | None = None) -> int:
    """Find each pool's LP token, so deposits and withdrawals can be arcs.

    The API reports `lp_token_address` for none of them, so it is read in one
    batch, from four questions asked of every pool:

        token()        50 mainnet pools, the older crypto ones
        lp_token()     12, the older stable ones
        totalSupply()  309, the factory-ng pools, which *are* their own LP
        base_pool      the rest, by inference (below)

    The fourth is the interesting one.  Fourteen mainnet pools expose no getter
    at all, and they include 3pool, whose LP token connects 25 meta-tokens to
    everything else.  But a base pool's LP *is* a coin of every metapool built on
    it, and the API does report `base_pool`, so a metapool naming this pool hands
    over the address as its own `coins[1]`.

    Curve's MetaRegistry would answer all of them and is not used: it exists on
    ethereum and polygon and on none of the other thirteen chains.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    # A pool's LP token cannot change, so it is read once per chain.  Supply
    # does change, and is read every time -- it sizes withdrawal probes.
    facts = TokenFactsCache()
    known = facts.load(chain_id) if chain_id is not None else {}
    learned: dict[str, dict] = {}

    ask = [p for p in pools if "lp_token" not in known.get(p.address.lower(), {})]
    calls: list[Call] = []
    for pool in ask:
        calls += [
            Call(pool.address, encode_call("token()")),
            Call(pool.address, encode_call("lp_token()")),
            Call(pool.address, encode_call("totalSupply()")),
        ]
    answers = client.raw(calls)
    found: dict[str, tuple[str, int]] = {}
    for k, pool in enumerate(ask):
        token, lp_token, _ = answers[3 * k : 3 * k + 3]
        address = ""
        for answer in (token, lp_token):
            if answer.status is Status.VALUE and answer.uint():
                address = "0x" + f"{answer.uint():040x}"
                break
        found[pool.address.lower()] = (address, 18)

    # A metapool's second coin is its base pool's LP token.
    from_base: dict[str, tuple[str, int]] = {}
    for pool in pools:
        if pool.base_pool and len(pool.coins) > 1:
            from_base[pool.base_pool.lower()] = (
                pool.coins[1].address, pool.coins[1].decimals
            )

    # Supply, for every pool: the batch above only covered the ones whose LP
    # token was unknown.
    supplies = client.raw([Call(p.address, encode_call("totalSupply()")) for p in pools])

    resolved = 0
    for k, pool in enumerate(pools):
        supply = supplies[k]
        cached = known.get(pool.address.lower(), {})
        address = cached.get("lp_token", "")
        # A cached zero is not a fact, it is the shape of a missed read: the
        # local EVM holds pool storage, and an LP token that lives at its own
        # address answers `decimals()` out of storage it does not have.  Zero
        # decimals is legal for an ERC20 and has never been legal for a Curve LP
        # token, so it is read again rather than believed.
        decimals = int(cached.get("lp_decimals") or 18)
        if not address and pool.address.lower() in found:
            address, decimals = found[pool.address.lower()]
        if not address and supply.status is Status.VALUE and supply.uint() > 0:
            address = pool.address  # the pool is its own ERC20
        if not address:
            address, decimals = from_base.get(pool.address.lower(), ("", 18))
        if not address:
            continue
        learned.setdefault(pool.address.lower(), {})["lp_token"] = address.lower()
        pool.lp_token = address.lower()
        pool.lp_decimals = decimals
        pool.lp_supply = supply.uint() if supply.status is Status.VALUE else 0
        resolved += 1

    # Supply for an LP token that lives at its own address, which the batch
    # above asked of the *pool* -- the same account for factory-ng pools and a
    # different one for the rest.
    separate = [p for p in pools
                if p.lp_token and p.lp_token != p.address.lower() and not p.lp_supply]
    if separate:
        # An LP token at its own address is an ERC20, not a pool, so these two
        # reads are the ones the local EVM cannot serve -- same split as
        # `read_balances`, and for the same reason.
        reader = token_client or client
        extra = reader.raw([Call(p.lp_token, encode_call("totalSupply()")) for p in separate]
                           + [Call(p.lp_token, encode_call("decimals()")) for p in separate])
        for idx, pool in enumerate(separate):
            supply, digits = extra[idx], extra[len(separate) + idx]
            if supply.status is Status.VALUE:
                pool.lp_supply = supply.uint()
            if digits.status is Status.VALUE and 1 <= digits.uint() <= 36:
                pool.lp_decimals = digits.uint()
    for pool in pools:
        if pool.lp_token:
            learned.setdefault(pool.address.lower(), {})["lp_decimals"] = pool.lp_decimals
    if chain_id is not None and learned:
        facts.save(chain_id, learned)
    return resolved


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

    Coin counts are settled first, because a probe of coin `j` is meaningless
    until `j` is known to be a coin -- see `resolve_coin_counts`.  Here rather
    than in each caller: fourteen places resolve a universe and all of them
    come through this one.
    """
    audit = DialectAudit()
    audit.coin_notes = resolve_coin_counts(pools, client)
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


#: How far past the API's count to look.  Nothing needs it -- the API has never
#: reported *fewer* coins than a pool has, on any chain measured -- but a silent
#: truncation would be the one failure this cannot detect on its own.
COIN_PROBE_SLACK = 2


def resolve_coin_counts(pools: list[PoolSpec], client: QuoterClient) -> list[str]:
    """Truncate each pool's coins to the ones the pool itself reports.

    The Prices listing appends a lending pool's *underlying* view to its coins
    and marks it nowhere: `is_metapool` is false, `base_pool` is null, and
    `pool_index` just counts on past the real coins.  So `cDAI/cUSDC/USDT`
    arrives with five coins and answers `coins(3)` with a revert, and the four
    ethereum and three polygon pools shaped like that were contributing 19 arcs
    naming indices that do not exist.  They quote `REVERTED` and are dropped,
    which is why nothing has broken -- but the guard is an accident, and the
    padded coin list also fails `all(balances)`, which is what kept those pools
    from ever reaching the exact-model gate.

    A metapool's tail is handled in `PoolSpec.from_api`, which keeps the first
    two; this is the case that has no flag to key on.

    N is not cosmetic -- it is in the stableswap invariant and in the
    `uint256[N]` an `add_liquidity` sends -- so it comes from the pool.  Both
    spellings are asked: the older lending pools index with `int128` and answer
    every `coins(uint256)` with a revert, which would otherwise read as a pool
    with no coins at all.  Returns a note per pool that changed.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    wanted = [p for p in pools if p.coins]
    if not wanted:
        return []
    plan = [(spec, sig, k)
            for spec in wanted
            for sig in ("coins(uint256)", "coins(int128)")
            for k in range(len(spec.coins) + COIN_PROBE_SLACK)]
    answers = client.raw([Call(spec.address, encode_call(sig, k))
                          for spec, sig, k in plan])

    seen: dict[str, dict[str, dict[int, str]]] = {}
    for (spec, sig, k), got in zip(plan, answers, strict=True):
        ok = bool(got.ok and got.data and len(got.data) >= 32)
        seen.setdefault(spec.address.lower(), {}).setdefault(sig, {})[k] = (
            ("0x" + got.data[-20:].hex()).lower() if ok else "")

    notes: list[str] = []
    for spec in wanted:
        by_sig = seen.get(spec.address.lower(), {})
        best = 0
        for answered in by_sig.values():
            n = 0
            while answered.get(n):
                n += 1
            best = max(best, n)
        if best == 0 or best == len(spec.coins):
            # No answer at all leaves the listing alone: a pool that will not
            # say is not evidence that the listing is wrong, and shrinking on
            # silence would delete real coins.
            continue
        if best > len(spec.coins):
            notes.append(f"{spec.name or spec.address}: pool reports {best} coins, "
                         f"the listing had {len(spec.coins)} -- kept the listing")
            continue
        notes.append(f"{spec.name or spec.address}: {len(spec.coins)} coins in the "
                     f"listing, {best} on the pool -- dropped the underlying view")
        spec.coins = spec.coins[:best]
        if spec.balances:
            spec.balances = spec.balances[:best]
        if getattr(spec, "held", None):
            spec.held = spec.held[:best]
    return notes


def count_swap_arcs(pools: list[PoolSpec]) -> int:
    return sum(p.swap_arc_count() for p in pools if p.swap_kind is not None)


def read_balances(pools: list[PoolSpec], client: QuoterClient,
                  report: list[str] | None = None,
                  chain_id: int | None = None,
                  token_client: QuoterClient | None = None) -> int:
    """Fill in each pool's reserves, in one batched call.

    Curve has two spellings of the same getter and the registry does not say
    which a pool implements, so both go out for every coin and the first that
    *answers* wins -- "answers" meaning 32 bytes back, not merely no revert.

    **The coin's own `decimals()` rides along, and it wins.**  The API is a
    directory, not an oracle: it omits `decimals` altogether on its newer
    registries -- every `twocryptong` and `stableswapng` entry measured -- and a
    missing value defaults to 18.  That is not a small error: gnosis USDC.e really
    has 6, so every amount through that pool came out 1e12 wrong.  It costs one
    more call per coin in a batch already going out, and `decimals` never changes.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    # LLAMMA has no `balances()` getter; its reserves come from the market
    # feed, and probing for them would only overwrite good numbers with none.
    pools = [p for p in pools if not p.balances]

    # Three calls per coin, not two: what the pool says it has, in both spellings,
    # and what the coin says it holds.  The third rides the same batch, so knowing
    # whether a pool's accounting is real costs no extra round trip.
    facts = TokenFactsCache()
    known = facts.load(chain_id) if chain_id is not None else {}
    learned: dict[str, dict] = {}

    # Two of the four calls per coin go to the **coin**, not the pool, and that
    # distinction decides which client may answer them.  A caller reading pools
    # out of a warmed local EVM has the pools' storage and not the tokens': an
    # unloaded account answers `balanceOf` with zero, which reads as a pool
    # holding nothing, and `check_reserves_are_real` then drops it as insolvent.
    # So the token calls go to `token_client` -- the wire, when there is one --
    # while the pool calls stay wherever the caller pointed them.
    tokens = token_client if token_client is not None else client
    calls: list[Call] = []
    token_calls: list[Call] = []
    spans: list[tuple[int, int]] = []
    for pool in pools:
        start = len(calls)
        for k in range(pool.n_coins):
            calls.append(Call(pool.address, encode_call("balances(uint256)", k)))
            calls.append(Call(pool.address, encode_call("balances(int128)", k)))
            coin = pool.coins[k] if k < len(pool.coins) else None
            target = coin.address if coin else pool.address
            token_calls.append(Call(target, encode_call("balanceOf(address)",
                                                        pool.address)))
            # Only for a coin whose decimals nobody has read yet.  A placeholder
            # keeps the stride so the unpacking below stays simple.
            #
            # Ask whether the *fact* is known, not whether the address is: other
            # passes cache other things about the same token -- an ERC4626
            # `asset`, a pool's `lp_token` -- so an address can be present with no
            # `decimals` in it, and testing membership then skips the read and
            # finds nothing to use.  It fails silent and sticky, and this is also
            # what heals a cache an older build poisoned.
            unknown = coin is not None and "decimals" not in known.get(
                coin.address.lower(), {})
            token_calls.append(Call(target, encode_call("decimals()")) if unknown
                               else Call(target, b""))
        spans.append((start, len(calls)))

    pool_answers = client.raw(calls)
    token_answers = tokens.raw(token_calls) if token_calls else []
    # Re-interleave, so everything below reads as it did: pool, pool, coin, coin.
    answers = []
    for k in range(len(spans)):
        lo, hi = spans[k]
        for c in range((hi - lo) // 2):
            answers.append(pool_answers[lo + 2 * c])
            answers.append(pool_answers[lo + 2 * c + 1])
            answers.append(token_answers[lo + 2 * c])
            answers.append(token_answers[lo + 2 * c + 1])
    spans = [(2 * lo, 2 * hi) for lo, hi in spans]
    filled = 0
    for pool, (lo, hi) in zip(pools, spans, strict=True):
        chunk = answers[lo:hi]
        balances, held, coins = [], [], list(pool.coins)
        for k in range(pool.n_coins):
            as_uint, as_int, owns, digits = chunk[4 * k : 4 * k + 4]
            if as_uint.status is Status.VALUE:
                balances.append(as_uint.uint())
            elif as_int.status is Status.VALUE:
                balances.append(as_int.uint())
            else:
                balances.append(0)
            # -1 means the coin would not say, which is not evidence of anything.
            held.append(owns.uint() if owns.status is Status.VALUE else -1)
            # A token that will not answer keeps whatever the API claimed;
            # one that answers is the authority on its own decimals.
            said = None
            if k < len(coins):
                cached = known.get(coins[k].address.lower())
                if cached is not None and "decimals" in cached:
                    said = int(cached["decimals"])
                elif digits.status is Status.VALUE:
                    said = digits.uint()
                    # A zero is far likelier to be a missed read than a real
                    # token: an account the local EVM has not loaded answers
                    # every getter with zero, and zero decimals is legal enough
                    # for an ERC20 that nothing downstream would question it.
                    # Refusing it keeps the API's guess, which gets re-checked,
                    # rather than caching a wrong fact for good.
                    if 1 <= said <= 36:
                        learned[coins[k].address.lower()] = {"decimals": said}
            if said is not None and 1 <= said <= 36 and said != coins[k].decimals:
                if report is not None:
                    report.append(
                        f"{coins[k].symbol} ({coins[k].address}) has "
                        f"{said} decimals, not the {coins[k].decimals} the "
                        f"API reports; using {said}"
                    )
                coins[k] = replace(coins[k], decimals=said)
        pool.coins = tuple(coins)
        pool.balances = tuple(balances)
        pool.held = tuple(held)
        if any(balances):
            filled += 1
    if chain_id is not None and learned:
        facts.save(chain_id, learned)
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
    pool reports 421,778 sUSD, holds zero, quotes 652 sUSD for 100 sUSDe, and
    reverts with `sUSD retired` on execution.  A quote that cannot settle is worse
    than no quote, and this is one batched call per coin to find them.

    Pools listing WETH may legitimately hold *native* ETH instead (E11), which is
    29 of the 31 apparent shortfalls on mainnet, so that is checked first.
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

    # A pool that holds nothing cannot trade, whatever an index says it is worth.
    # The check below only looks at pools reporting *more* than they hold, so one
    # reporting zero is never short and never flagged -- it is simply empty, and
    # it stays in the universe on the strength of a TVL figure that came from an
    # API rather than from the chain.  Measured on `WETH/USDC Curve StableNG`
    # (0x68A67937620aF5DbF4093D0022B79CC1e92060f2), listed at $37,798.92 with both
    # `balances()` zero and `get_virtual_price()` reverting, while the legacy
    # api.curve.finance reports the same pool as `usdTotal: 0`.
    #
    # It could not have produced a quote -- the invariant needs every balance
    # positive -- so this costs no output.  What it costs is arcs enumerated,
    # probes planned and sent, and a pool in every count.  Balances are read at
    # the pinned block; membership should follow them.
    for pool in pools:
        if not pool.balances or any(b > 0 for b in pool.balances):
            continue
        warnings.append(
            f"{pool.name or pool.address} holds nothing at this block "
            f"(listed at ${pool.tvl_usd:,.0f}): no arcs built from it"
        )

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


def check_the_invariant_answers(
    pools: list[PoolSpec], client: QuoterClient
) -> list[str]:
    """Find pools with no `D` at the numbers they hold, for the blacklist.

    A pool that cannot publish `get_virtual_price()` cannot quote anything.
    `WETH/yETH` is the expensive case: 43,294 wei of WETH against a coin whose
    supply is 2.35e56, carried at $2.1M because the index prices that coin.

    **The second call is the whole point.** Every LLAMMA reverts here and every
    LLAMMA quotes -- it has no `D` by construction -- so the first call alone
    drops $24M of crvUSD liquidity on mainnet.  Not the "huge supply" tell
    either: `BABYPEPE/SPANK` holds 6.4e12 tokens and trades.

    Off the route path deliberately: `get_virtual_price` solves the invariant,
    so 386 of them cost 800 ms against the reserve check's 181 ms, to save the
    ~3.5 ms of probes the grid spends reaching the same conclusion.  The local
    EVM cannot stand in -- it reverts on 17 healthy pools it holds no base pool
    or oracle for.  `scripts/find_broken_pools.py` runs it; the finding goes in
    `chains.py`.  Zeroing balances is how a drop is expressed, as above.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    warnings: list[str] = []
    live = [p for p in pools
            if p.balances and len(p.balances) >= 2 and any(b > 0 for b in p.balances)]
    if not live or client is None:
        return warnings

    answers = client.raw([Call(p.address, encode_call("get_virtual_price()"))
                          for p in live])
    silent = [pool for pool, answer in zip(live, answers, strict=True)
              if answer.status is Status.REVERTED]
    if not silent:
        return warnings

    # From the fullest side, so a failure is the arithmetic and not an empty
    # reserve.
    probes = []
    for pool in silent:
        i = max(range(len(pool.balances)), key=lambda k: pool.balances[k])
        j = next(k for k in range(len(pool.balances)) if k != i)
        probes.append(Probe(pool.address, pool.swap_kind or ArcKind.SWAP_STABLE,
                            i, j, len(pool.coins),
                            max(1, int(pool.balances[i] * 0.001))))
    for pool, quote in zip(silent, client.probe(probes), strict=True):
        if quote.ok and quote.value > 0:
            continue                       # a LLAMMA, or anything else without a D
        warnings.append(
            f"{pool.name or pool.address} dropped: neither its virtual price nor "
            f"its own get_dy can be computed from the balances it holds "
            f"(listed at ${pool.tvl_usd:,.0f})"
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


#: The Twocrypto allowlist stores its own on/off switch under the zero address.
DEPOSIT_GATE_FLAG = "0x" + "00" * 20


def resolve_deposit_gates(pools: list[PoolSpec], client: QuoterClient) -> int:
    """Mark pools whose `add_liquidity` is allowlisted, in one batched read.

    Twocrypto carries `lp_allowlist`, and when the switch under the zero address
    is on, `add_liquidity` asserts `lp_allowlist[msg.sender]` and refuses
    everyone else.  `calc_token_amount` carries no such check, so the arc quotes
    a number nobody outside the list can get -- measured on the four Yield Basis
    pools, $245M of TVL between them, every deposit reverting with "!wl".

    A flag an admin can flip, so it is read per block rather than remembered.
    Only the deposit is gated; swaps and withdrawals stay.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call

    if not pools:
        return 0
    calls = [Call(p.address, encode_call("lp_allowlist(address)", DEPOSIT_GATE_FLAG))
             for p in pools]
    gated = 0
    for pool, answer in zip(pools, client.raw(calls), strict=True):
        # A pool without the getter answers empty, which is not a "no" -- but it
        # is the absence of an allowlist, which is the same thing here.
        pool.deposit_gated = bool(answer.ok and answer.uint_or(0))
        gated += pool.deposit_gated
    return gated
