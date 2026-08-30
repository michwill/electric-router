"""The model-free floor, Python against Rust.

`direct_candidates` is pure and ports as one call.  `two_step_candidates` is
not: it quotes twice from inside one function, and the chain is on the far side
of the boundary in the browser.  So the port splits it at its two probe rounds
into three stateless stages and the caller does the quoting between them.

That split is the thing most likely to be wrong, so these tests drive *both*
sides from one deterministic fake quoter and compare what comes out: the same
probes, in the same order, and the same chains ranked the same way.
"""

from __future__ import annotations

import erouter_solve
import numpy as np
import pytest

from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.pipeline import direct_candidates, two_step_candidates
from erouter.core.pools import Coin, PoolSpec
from erouter.core.types import Dialect, Probe

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"
CRVUSD = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
CHEAP = "0x" + "cc" * 20


def pool(address, coins, dialect=Dialect.STABLE, name="pool", tvl=1e6):
    spec = PoolSpec(
        address=address, name=name, pool_type="main",
        coins=tuple(Coin(a, s, d, k) for k, (a, s, d) in enumerate(coins)),
        tvl_usd=tvl,
    )
    spec.dialect = dialect
    spec.balances = tuple(10 ** 18 for _ in coins)
    return spec


def facts_of(pools):
    """The same universe, on the Rust side."""
    facts = erouter_solve.PoolFacts()
    for spec in pools:
        kind = spec.swap_kind
        facts.add(
            spec.address, spec.name,
            None if kind is None else kind.value,
            [c.address.lower() for c in spec.coins],
            [c.decimals for c in spec.coins],
            [str(b) for b in spec.balances],
            spec.tvl_usd,
        )
    return facts


def nodes_for(pools, merge_eth=False):
    nodes = NodeMap()
    for spec in pools:
        for coin in spec.coins:
            nodes.add_token(coin.address, coin.symbol, coin.decimals)
    if merge_eth:
        nodes.add_token(WETH, "WETH", 18)
        nodes.merge(Conversion(ConversionKind.NATIVE_WRAP, ETH, WETH, 1, 1, target=WETH))
    return nodes


def rust_nodes(nodes, pools, merge_eth=False):
    out = erouter_solve.NodeMap()
    for spec in pools:
        for coin in spec.coins:
            out.add_token(coin.address, coin.symbol, coin.decimals)
    if merge_eth:
        out.add_token(WETH, "WETH", 18)
        out.merge("NATIVE_WRAP", ETH, WETH, "1", "1", WETH)
    return out


class FakeQuoter:
    """Deterministic, and it records what it was asked.

    A hash of the probe, so the answer depends on the pool and the amount but
    not on the order -- which is what lets the two sides be compared at all.
    Some probes refuse, because the ranking has to survive that.
    """

    def __init__(self, refuse=()):
        self.seen: list[Probe] = []
        self.refuse = set(refuse)

    @staticmethod
    def value_for(probe) -> int | None:
        pool, i, j, dx = probe
        seed = abs(hash((pool.lower(), i, j))) % 997 + 3
        return dx * seed // 1000

    def answer(self, pool, i, j, dx):
        if pool.lower() in self.refuse:
            return None
        return self.value_for((pool, i, j, dx))

    # -- the Python side's interface --------------------------------------
    def probe(self, probes):
        self.seen.extend(probes)
        out = []
        for p in probes:
            value = self.answer(p.pool, p.i, p.j, p.dx)
            out.append(
                type("Q", (), {"ok": value is not None, "value": value or 0})()
            )
        return out

    # -- and the Rust side's, which is the same answers as strings ---------
    def rows(self, probe_rows):
        out = []
        for pool, _kind, i, j, _n, dx in probe_rows:
            value = self.answer(pool, i, j, int(dx))
            out.append(None if value is None else str(value))
        return out


def run_rust(pools, nodes, nu, src, dst, quoter, limit=6, merge_eth=False):
    """The three stages, with the quoting the caller has to do between them."""
    facts = facts_of(pools)
    rnodes = rust_nodes(nodes, pools, merge_eth)
    plan_a = erouter_solve.two_step_plan_first(facts, rnodes, src, dst, str(10 ** 18))
    if len(plan_a) == 0:
        return [], [], plan_a, None
    plan_b = erouter_solve.two_step_rank(
        facts, rnodes, list(nu), plan_a, quoter.rows(plan_a.probes()), dst, limit
    )
    if len(plan_b) == 0:
        return [], [], plan_a, plan_b
    cands, chains = erouter_solve.two_step_build(
        facts, rnodes, list(nu), plan_b, quoter.rows(plan_b.probes()),
        src, dst, limit,
    )
    return cands, chains, plan_a, plan_b


# ----------------------------------------------------------- direct


DIRECT_CASES = {
    "one pool, one direction": (
        [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)])], DAI, USDC),
    "three coins, still one pair": (
        [pool("0x" + "11" * 20,
              [(DAI, "DAI", 18), (USDC, "USDC", 6), (USDT, "USDT", 6)])], DAI, USDC),
    "two pools hold the pair": (
        [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)]),
         pool("0x" + "22" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)])], DAI, USDC),
    "a long name gets cut at 22": (
        [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDC, "USDC", 6)],
              name="a name comfortably past twenty-two characters")], DAI, USDC),
    "crypto dialect": (
        [pool("0x" + "11" * 20, [(WETH, "WETH", 18), (USDC, "USDC", 6)],
              dialect=Dialect.CRYPTO)], WETH, USDC),
}


@pytest.mark.parametrize("case", sorted(DIRECT_CASES))
def test_direct_candidates_agree(case):
    pools, src, dst = DIRECT_CASES[case]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)

    want_c, want_a = direct_candidates(pools, nodes, nu, src, dst, 10 ** 18)
    got_c, got_a = erouter_solve.direct_candidates(
        facts_of(pools), rust_nodes(nodes, pools), list(nu), src, dst
    )

    assert len(got_c) == len(want_c)
    assert len(got_a) == len(want_a)
    for want, got in zip(want_c, got_c, strict=True):
        assert got["label"] == want.label
        assert got["reason"] == want.reason
        assert got["kind"] == want.kind
        assert got["n_arcs"] == want.n_arcs
        assert got["certificate"] == want.certificate
        assert got["psi"] == list(want.psi)
    for k, want in enumerate(want_a):
        got = got_a.row(k)
        assert got["id"] == want.id
        assert got["pool"] == want.pool
        assert got["kind"] == want.kind.value
        assert (got["i"], got["j"]) == (want.i, want.j)
        assert got["n_coins"] == want.n_coins
        assert (got["tau"], got["sigma"]) == (want.tau, want.sigma)
        assert got["a"] == want.a
        assert got["b"] == want.B == 0.0
        assert got["reserve_in"] == str(want.reserve_in)
        assert got["decimals_in"] == want.decimals_in
        assert got["decimals_out"] == want.decimals_out
        assert got["note"] == want.note


def test_a_merged_token_is_matched_through_its_node_on_both_sides():
    """A pool holding native ETH is still a direct WETH route."""
    pools = [pool("0x" + "44" * 20, [(USDC, "USDC", 6), (ETH, "ETH", 18)])]
    nodes = nodes_for(pools, merge_eth=True)
    nu = np.ones(nodes.n_nodes)

    want_c, want_a = direct_candidates(pools, nodes, nu, USDC, WETH, 10 ** 6)
    got_c, got_a = erouter_solve.direct_candidates(
        facts_of(pools), rust_nodes(nodes, pools, merge_eth=True), list(nu), USDC, WETH
    )
    assert len(want_c) == len(got_c) == 1
    row = got_a.row(0)
    assert row["token_out"] == want_a[0].token_out
    assert row["sigma"] == want_a[0].sigma == nodes.node(WETH)


# ----------------------------------------------------------- two-step


TWO_STEP_CASES = {
    "one middle, one chain": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
    ],
    "two middles compete": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
        pool("0x" + "33" * 20, [(DAI, "DAI", 18), (USDT, "USDT", 6)]),
        pool("0x" + "44" * 20, [(USDT, "USDT", 6), (USDC, "USDC", 6)]),
    ],
    "a dead-end middle is never probed": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
        pool("0x" + "55" * 20, [(DAI, "DAI", 18), (WETH, "WETH", 18)]),
    ],
    "two pools for the same second hop": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
        pool("0x" + "66" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
    ],
    "three coins in one pool": [
        pool("0x" + "11" * 20,
             [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18), (USDT, "USDT", 6)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
        pool("0x" + "77" * 20, [(USDT, "USDT", 6), (USDC, "USDC", 6)]),
    ],
    "no two-hop route at all": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (WETH, "WETH", 18)]),
        pool("0x" + "aa" * 20, [(WETH, "WETH", 18), (USDC, "USDC", 6)]),
    ],
    "an untradeable pool in the middle": [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)]),
        pool("0x" + "22" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)]),
        pool("0x" + "88" * 20, [(DAI, "DAI", 18), (USDT, "USDT", 6)], dialect=None),
        pool("0x" + "99" * 20, [(USDT, "USDT", 6), (USDC, "USDC", 6)]),
    ],
}


def compare_two_step(pools, src, dst, quoter_py, quoter_rs, limit=6):
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)
    want_c, want_chains = two_step_candidates(
        pools, nodes, nu, quoter_py, src, dst, 10 ** 18, limit=limit
    )
    got_c, got_chains, plan_a, plan_b = run_rust(
        pools, nodes, nu, src, dst, quoter_rs, limit=limit
    )

    # The probes, in order: the split is only faithful if both sides ask the
    # chain exactly the same questions.
    asked = [(p.pool.lower(), p.kind.value, p.i, p.j, p.n, p.dx) for p in quoter_py.seen]
    rust_asked = [(p[0].lower(), p[1], p[2], p[3], p[4], int(p[5]))
                  for p in plan_a.probes()]
    if plan_b is not None:
        rust_asked += [(p[0].lower(), p[1], p[2], p[3], p[4], int(p[5]))
                       for p in plan_b.probes()]
    assert rust_asked == asked

    assert len(got_c) == len(want_c)
    for want, got in zip(want_c, got_c, strict=True):
        assert got["label"] == want.label
        assert got["reason"] == want.reason == "TWO_STEP"
        assert got["n_arcs"] == want.n_arcs == 2
        assert got["psi"] == list(want.psi)
    assert len(got_chains) == len(want_chains)
    for want, got in zip(want_chains, got_chains, strict=True):
        assert len(got) == len(want) == 2
        for k, arc in enumerate(want):
            row = got.row(k)
            assert row["id"] == arc.id
            assert row["pool"] == arc.pool
            assert (row["i"], row["j"]) == (arc.i, arc.j)
            assert (row["tau"], row["sigma"]) == (arc.tau, arc.sigma)
            assert row["a"] == arc.a
            assert row["token_in"] == arc.token_in
            assert row["token_out"] == arc.token_out


@pytest.mark.parametrize("case", sorted(TWO_STEP_CASES))
def test_two_step_candidates_agree(case):
    pools = TWO_STEP_CASES[case]
    compare_two_step(pools, DAI, USDC, FakeQuoter(), FakeQuoter())


@pytest.mark.parametrize("case", sorted(TWO_STEP_CASES))
def test_two_step_agrees_when_the_chain_refuses(case):
    """A refused probe must drop its hop on both sides, not zero it."""
    pools = TWO_STEP_CASES[case]
    refuse = {"0x" + "22" * 20}
    compare_two_step(pools, DAI, USDC, FakeQuoter(refuse), FakeQuoter(refuse))


@pytest.mark.parametrize("limit", [1, 2, 6])
def test_the_limit_cuts_the_same_chains(limit):
    pools = TWO_STEP_CASES["two middles compete"]
    compare_two_step(pools, DAI, USDC, FakeQuoter(), FakeQuoter(), limit=limit)


def test_ranking_goes_by_value_not_by_token_count():
    """The CXD lesson, end to end.

    A middle trading at a fraction of a cent produces enormously more *units*
    than a dear one.  Ranking by `to_canonical_wei` puts it first; ranking
    through `nu` does not.  Both sides have to make the same call, so this
    pins the one place where `nu` enters the floor at all.
    """
    pools = [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (CHEAP, "CHEAP", 18)], tvl=3e4),
        pool("0x" + "22" * 20, [(CHEAP, "CHEAP", 18), (USDC, "USDC", 6)], tvl=3e4),
        pool("0x" + "33" * 20, [(DAI, "DAI", 18), (CRVUSD, "crvUSD", 18)], tvl=4e8),
        pool("0x" + "44" * 20, [(CRVUSD, "crvUSD", 18), (USDC, "USDC", 6)], tvl=4e8),
    ]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)
    nu[nodes.node(CHEAP)] = 1e-6

    py, rs = FakeQuoter(), FakeQuoter()
    want_c, _ = two_step_candidates(pools, nodes, nu, py, DAI, USDC, 10 ** 18, limit=6)
    got_c, _, plan_a, plan_b = run_rust(pools, nodes, nu, DAI, USDC, rs, limit=6)

    assert [c["label"] for c in got_c] == [c.label for c in want_c]
    assert [(p.pool.lower(), p.dx) for p in py.seen] == [
        (p[0].lower(), int(p[5]))
        for p in list(plan_a.probes()) + list(plan_b.probes())
    ]


# --------------------------------------------------- the contract at the edge


def test_a_token_no_pool_holds_is_an_error_on_both_sides():
    """`nodes.node` raises in the reference, so the port must not shrug.

    An empty floor and an unknown token look identical to a caller if the
    second is reported as the first -- and the floor exists precisely to be
    the answer when everything else came back empty, so "no route" is the one
    thing it must never say by accident.
    """
    pools = [pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDT, "USDT", 6)])]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)

    with pytest.raises(KeyError):
        direct_candidates(pools, nodes, nu, DAI, USDC, 10 ** 18)
    with pytest.raises(ValueError, match="not in the node map"):
        erouter_solve.direct_candidates(
            facts_of(pools), rust_nodes(nodes, pools), list(nu), DAI, USDC
        )

    with pytest.raises(KeyError):
        two_step_candidates(pools, nodes, nu, FakeQuoter(), DAI, USDC, 10 ** 18)
    with pytest.raises(ValueError, match="not in the node map"):
        erouter_solve.two_step_plan_first(
            facts_of(pools), rust_nodes(nodes, pools), DAI, USDC, str(10 ** 18)
        )


def test_a_reachable_dst_with_no_two_hop_route_is_empty_not_an_error():
    """The other side of the same line: known tokens, genuinely no chain."""
    pools = [
        pool("0x" + "11" * 20, [(DAI, "DAI", 18), (USDT, "USDT", 6)]),
        pool("0x" + "22" * 20, [(USDC, "USDC", 6), (CRVUSD, "crvUSD", 18)]),
    ]
    nodes = nodes_for(pools)
    nu = np.ones(nodes.n_nodes)
    want_c, want_chains = two_step_candidates(
        pools, nodes, nu, FakeQuoter(), DAI, USDC, 10 ** 18
    )
    got_c, got_chains, _, _ = run_rust(pools, nodes, nu, DAI, USDC, FakeQuoter())
    assert want_c == [] and want_chains == []
    assert got_c == [] and got_chains == []
