"""The direct-swap floor and the probe cache.

The floor exists because of a real failure: on a run where many probes timed
out, only 276 of 766 arcs calibrated, no good path survived, and the router
returned **13,700 USDC for 100,000 DAI** -- and verified it honestly, because
every candidate it generated really was that bad.

A router must never be beaten by a swap anyone could find by inspection, and
the way to guarantee that is a candidate that does not depend on the probe
grid, the calibration, the price fit or the solver.
"""

from __future__ import annotations

import numpy as np

from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.pipeline import direct_candidates
from erouter.core.pools import Coin, PoolSpec
from erouter.core.types import ArcKind, Dialect

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


def pool(address, coins, dialect=Dialect.STABLE, name="pool"):
    spec = PoolSpec(
        address=address, name=name, pool_type="main",
        coins=tuple(Coin(a, s, d, k) for k, (a, s, d) in enumerate(coins)),
        tvl_usd=1e6,
    )
    spec.dialect = dialect
    spec.balances = tuple(10**18 for _ in coins)
    return spec


def nodes_for(pools, merge_eth=False):
    nodes = NodeMap()
    for spec in pools:
        for coin in spec.coins:
            nodes.add_token(coin.address, coin.symbol, coin.decimals)
    if merge_eth:
        nodes.add_token(WETH, "WETH", 18)
        nodes.merge(Conversion(ConversionKind.NATIVE_WRAP, ETH, WETH, 1, 1, target=WETH))
    return nodes


def test_one_candidate_per_pool_holding_both_tokens():
    pools = [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6), (USDT, "USDT", 6)]),
        pool("0x" + "22" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)]),
        pool("0x" + "33" * 20, [(USDT, "USDT", 6), (WETH, "WETH", 18)]),  # irrelevant
    ]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)

    candidates, arcs = direct_candidates(pools, nodes, nu, DAI, USDC, 10**18)
    assert len(candidates) == len(arcs) == 2
    assert {a.pool for a in arcs} == {"0x" + "11" * 20, "0x" + "22" * 20}
    for arc in arcs:
        assert arc.token_in.lower() == DAI
        assert arc.token_out.lower() == USDC
        assert arc.i != arc.j
    assert all(c.kind == "direct" for c in candidates)


def test_the_floor_does_not_depend_on_calibration():
    """No probe, no ladder, no `a`/`B` fit -- that is the entire point."""
    pools = [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)])]
    nodes = nodes_for(pools)
    _, arcs = direct_candidates(pools, nodes, np.ones(nodes.n_nodes), DAI, USDC, 10**18)
    assert arcs[0].B == 0.0  # never fitted
    assert arcs[0].kind is ArcKind.SWAP_STABLE  # dialect still respected


def test_merged_tokens_are_matched_through_their_node():
    """A pool holding native ETH still counts as a direct WETH route."""
    pools = [pool("0x" + "44" * 20, [(USDC, "USDC", 6), (ETH, "ETH", 18)])]
    nodes = nodes_for(pools, merge_eth=True)
    candidates, arcs = direct_candidates(
        pools, nodes, np.ones(nodes.n_nodes), USDC, WETH, 10**6
    )
    assert len(candidates) == 1
    assert arcs[0].token_out.lower() == ETH  # the pool's own token
    assert arcs[0].sigma == nodes.node(WETH)  # but the graph node is shared


def test_no_direct_pool_means_no_floor_not_an_error():
    pools = [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)])]
    nodes = nodes_for(pools)
    nodes.add_token(WETH, "WETH", 18)
    candidates, arcs = direct_candidates(
        pools, nodes, np.ones(nodes.n_nodes), DAI, WETH, 10**18
    )
    assert candidates == [] and arcs == []


def test_a_pool_with_the_pair_in_both_directions_yields_one_arc_each_way():
    pools = [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)])]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)
    forward, _ = direct_candidates(pools, nodes, nu, DAI, USDC, 10**18)
    reverse, _ = direct_candidates(pools, nodes, nu, USDC, DAI, 10**6)
    assert len(forward) == len(reverse) == 1


# ------------------------------------------------------------- probe cache


def test_probe_cache_round_trips(tmp_path):
    from erouter.core.transport import Status
    from erouter.core.types import Probe
    from erouter.dev.cache import Cache
    from erouter.dev.probe_cache import CachedQuoterClient

    class Counter:
        def __init__(self):
            self.calls = 0

        def probe(self, probes):
            from erouter.core.quoter import Quote

            self.calls += 1
            return [Quote(Status.VALUE, 100 + k) for k in range(len(probes))]

    inner = Counter()
    client = CachedQuoterClient(inner, 1, 123, cache=Cache(tmp_path))
    probes = [Probe("0x" + "11" * 20, ArcKind.SWAP_STABLE, 0, 1, 2, 10**18)]

    first = client.probe(probes)
    second = client.probe(probes)
    assert inner.calls == 1  # the second is served from disk
    assert [q.value for q in first] == [q.value for q in second]
    assert [q.status for q in first] == [q.status for q in second]
    assert client.stats.hits == 1 and client.stats.misses == 1


def test_probe_cache_is_keyed_by_block_and_by_batch(tmp_path):
    from erouter.core.quoter import Quote
    from erouter.core.transport import Status
    from erouter.core.types import Probe
    from erouter.dev.cache import Cache
    from erouter.dev.probe_cache import CachedQuoterClient, digest

    class Counter:
        def __init__(self):
            self.calls = 0

        def probe(self, probes):
            self.calls += 1
            return [Quote(Status.VALUE, 1) for _ in probes]

    inner = Counter()
    cache = Cache(tmp_path)
    probes = [Probe("0x" + "11" * 20, ArcKind.SWAP_STABLE, 0, 1, 2, 10**18)]
    other = [Probe("0x" + "11" * 20, ArcKind.SWAP_STABLE, 0, 1, 2, 2 * 10**18)]

    assert digest(probes) != digest(other)  # size is part of the key

    CachedQuoterClient(inner, 1, 100, cache=cache).probe(probes)
    CachedQuoterClient(inner, 1, 101, cache=cache).probe(probes)  # a new block
    assert inner.calls == 2

    CachedQuoterClient(inner, 1, 100, cache=cache).probe(probes)  # hit
    assert inner.calls == 2
