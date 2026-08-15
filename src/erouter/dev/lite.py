"""Curve Lite: the small deployments, on a different API.

The Prices API indexes trades, so it can report volume and TVL and history.
**Curve Lite** is the other kind of deployment -- factory contracts and a
gauge, without any of that indexing -- and it is served by
`api2.curve.finance`, a separate service with a different shape.

**It cannot replace the main API**, which was worth checking rather than
assuming: `get_pools/1` and `get_pools/42161` both answer 404.  `get_platforms`
lists 26 deployments and only one of them, sonic, is also a Prices chain --
where Prices wins, because it serves sonic as a full deployment while Lite
reports $0.2M across its 173 pools.  So this is a second source for the chains
the first one does not have, not a migration.

Two differences that matter to the router:

* **The floor has to come down.**  A whole Lite deployment is smaller than the
  $10,000 pool floor mainnet uses -- fantom's 321 pools come to $0.9M between
  them, with one pool over $10k, and sonic's Lite side to $0.2M.  Applying the
  usual floor would return an empty universe, which reads as "chain not
  supported" rather than "chain is small".
* **No paging.**  The whole chain arrives in one response, which is why there
  is no `min_tvl` in the request: it is filtered here.

The field names differ throughout and are translated to what
`PoolSpec.from_api` reads, so nothing downstream needs to know which API a
pool came from.
"""

from __future__ import annotations

from typing import Any

from .curve_api import CurveApiError, _get

LITE_API = "https://api2.curve.finance"

#: The list floor for a Lite chain.  Zero, where the main chains use $10,000:
#: see the module docstring -- the same cut would empty the universe.
LITE_MIN_TVL = 0.0

#: Deployments that are not real chains.  `get_platforms` mixes testnets and
#: devnets in with the rest, and a router that offers to quote on `bsc_testnet`
#: is offering something nobody wants.
NOT_MAINNET = ("sepolia", "testnet", "devnet:", "megaeth", "expchain")


def _is_mainnet(name: str) -> bool:
    lowered = name.lower()
    return not any(mark in lowered for mark in NOT_MAINNET)


def platforms() -> dict[str, dict[str, Any]]:
    """Every Lite deployment: name -> its metadata, testnets excluded.

    Failure is not fatal to the caller: a chain served by the Prices API does
    not need this at all, so an outage here should cost the Lite chains and
    nothing else.
    """
    try:
        payload = _get(f"{LITE_API}/get_platforms")
    except CurveApiError:
        return {}
    data = payload.get("data") or {}
    meta = data.get("platforms_metadata") or {}
    return {
        name: meta.get(name, {})
        for name in (data.get("platforms") or {})
        if _is_mainnet(name)
    }


def chain_ids() -> dict[str, int]:
    """Lite deployment name -> chain id, for the ones that report one."""
    out: dict[str, int] = {}
    for name, meta in platforms().items():
        chain_id = meta.get("chain_id")
        if chain_id is not None:
            out[name] = int(chain_id)
    return out


def _coin(raw: dict) -> dict:
    """One coin, in the shape `Coin.from_api` reads.

    `decimals` arrives as a string here and as a number on the Prices API, so
    it is normalised rather than left for the parser to guess.
    """
    return {
        "address": raw.get("address") or "",
        "symbol": raw.get("symbol") or "",
        "decimals": int(raw.get("decimals") or 18),
        "pool_balance": raw.get("pool_balance"),
        "usd_price": raw.get("usd_price"),
    }


def _pool(raw: dict) -> dict:
    """One pool, translated into what `PoolSpec.from_api` reads.

    The names differ on almost every field that matters: `registry_id` for the
    pool type, `tvl` for `tvl_usd`, `is_meta_pool` for `is_metapool`.  A pool
    the API marks `is_broken` is dropped by the caller, not renamed here.
    """
    return {
        "address": raw.get("address") or "",
        "name": raw.get("name") or raw.get("symbol") or "",
        "pool_type": raw.get("registry_id") or "",
        "coins": [_coin(c) for c in (raw.get("coins") or [])],
        "tvl_usd": float(raw.get("tvl") or 0.0),
        "is_metapool": bool(raw.get("is_meta_pool")),
        "base_pool": raw.get("base_pool") or "",
        "lp_token_address": raw.get("lp_token_address") or "",
    }


def list_pools(chain_id: int, *, min_tvl: float = LITE_MIN_TVL) -> list[dict]:
    """Every usable pool on a Lite chain, in one request.

    Broken pools are dropped here rather than downstream: the API knows which
    of its own pools cannot be read, and a pool that cannot be read is worse
    than a pool that is absent -- it costs a probe to discover and shows up as
    a mysteriously unroutable token.
    """
    payload = _get(f"{LITE_API}/get_pools/{chain_id}")
    data = payload.get("data") or {}
    raw_pools = data.get("pool_data") or []
    out = []
    for raw in raw_pools:
        if raw.get("is_broken"):
            continue
        if float(raw.get("tvl") or 0.0) < min_tvl:
            continue
        translated = _pool(raw)
        if translated["address"] and len(translated["coins"]) >= 2:
            out.append(translated)
    out.sort(key=lambda entry: -entry["tvl_usd"])
    return out
