"""Brute-force baselines the router must never lose to.

Deliberately dumb: no model, no calibration, no solver, no reference prices --
just quote every pool and every two-pool chain and take the best.  That makes
them an independent check rather than a re-run of the same logic, which is the
whole point of a differential test.

Both are *lower bounds* on what is achievable.  The router splits, so it should
usually beat them; it must never be worse, because anyone comparing by hand
would find these in a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

from erouter.core.nodes import NodeMap
from erouter.core.pools import PoolSpec
from erouter.core.types import Probe


@dataclass(frozen=True, slots=True)
class Baseline:
    amount_out: int = 0
    label: str = ""

    @property
    def found(self) -> bool:
        return self.amount_out > 0


def _slots(pool: PoolSpec, nodes: NodeMap, node: int) -> list[int]:
    """Coin indices of `pool` that belong to graph node `node`."""
    return [
        k
        for k, coin in enumerate(pool.coins)
        if nodes.has(coin.address) and nodes.node(coin.address) == node
    ]


def naive_direct(
    pools: list[PoolSpec], nodes: NodeMap, client, src: str, dst: str, amount: int
) -> Baseline:
    """Best single-pool swap, over every pool holding both tokens."""
    src_node, dst_node = nodes.node(src), nodes.node(dst)
    probes: list[Probe] = []
    labels: list[str] = []
    for pool in pools:
        if pool.swap_kind is None:
            continue
        for i in _slots(pool, nodes, src_node):
            for j in _slots(pool, nodes, dst_node):
                if i == j:
                    continue
                probes.append(
                    Probe(pool.address, pool.swap_kind, i, j, pool.n_coins, amount)
                )
                labels.append(f"{pool.name} [{i}>{j}]")
    if not probes:
        return Baseline()
    best = Baseline()
    for quote, label in zip(client.probe(probes), labels, strict=True):
        if quote.ok and quote.value > best.amount_out:
            best = Baseline(quote.value, label)
    return best


def naive_two_step(
    pools: list[PoolSpec],
    nodes: NodeMap,
    client,
    src: str,
    dst: str,
    amount: int,
    *,
    max_intermediates: int = 40,
) -> Baseline:
    """Best `src -> M -> dst` chain, one pool per hop.

    Two batched rounds: quote every `src -> M`, keep the best amount per
    intermediate, then quote every `M -> dst` from it.  Amounts crossing a
    merged node are converted at the node's exact rate, so an ETH-paying pool
    can feed a WETH-taking one.
    """
    src_node, dst_node = nodes.node(src), nodes.node(dst)

    # --- round A: src -> M ------------------------------------------------
    probes: list[Probe] = []
    meta: list[tuple[int, str, str]] = []  # (node, token, label)
    for pool in pools:
        if pool.swap_kind is None:
            continue
        for i in _slots(pool, nodes, src_node):
            for j, coin in enumerate(pool.coins):
                if j == i or not nodes.has(coin.address):
                    continue
                middle = nodes.node(coin.address)
                if middle in (src_node, dst_node):
                    continue
                probes.append(
                    Probe(pool.address, pool.swap_kind, i, j, pool.n_coins, amount)
                )
                meta.append((middle, coin.address.lower(), f"{pool.name} [{i}>{j}]"))
    if not probes:
        return Baseline()

    # Best arrival per intermediate *node*, held in canonical units so the
    # second hop can start from whichever token it needs.
    best_mid: dict[int, tuple[int, str]] = {}
    for quote, (middle, token, label) in zip(client.probe(probes), meta, strict=True):
        if not quote.ok or quote.value <= 0:
            continue
        canonical = nodes.to_canonical_wei(token, quote.value)
        if canonical > best_mid.get(middle, (0, ""))[0]:
            best_mid[middle] = (canonical, label)
    if not best_mid:
        return Baseline()

    ranked = sorted(best_mid.items(), key=lambda kv: -kv[1][0])[:max_intermediates]

    # --- round B: M -> dst ------------------------------------------------
    probes = []
    tails: list[tuple[str, str]] = []  # (output token, label)
    for middle, (canonical, first) in ranked:
        for pool in pools:
            if pool.swap_kind is None:
                continue
            for i in _slots(pool, nodes, middle):
                token = pool.coins[i].address
                start = nodes.from_canonical_wei(token, canonical)
                if start <= 0:
                    continue
                for j in _slots(pool, nodes, dst_node):
                    if i == j:
                        continue
                    probes.append(
                        Probe(pool.address, pool.swap_kind, i, j, pool.n_coins, start)
                    )
                    tails.append(
                        (pool.coins[j].address, f"{first} -> {pool.name} [{i}>{j}]")
                    )
    if not probes:
        return Baseline()

    best = Baseline()
    for quote, (token, label) in zip(client.probe(probes), tails, strict=True):
        if not quote.ok:
            continue
        # The second hop pays out that pool's own token, which may be a sibling
        # of the destination on a merged node -- ETH when WETH was asked for.
        # Express it in the token the caller actually wants.
        value = nodes.from_canonical_wei(dst, nodes.to_canonical_wei(token, quote.value))
        if value > best.amount_out:
            best = Baseline(value, label)
    return best


def naive_best(
    pools: list[PoolSpec], nodes: NodeMap, client, src: str, dst: str, amount: int
) -> tuple[Baseline, Baseline]:
    return (
        naive_direct(pools, nodes, client, src, dst, amount),
        naive_two_step(pools, nodes, client, src, dst, amount),
    )
