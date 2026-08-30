"""The Rust stages must decide what `core/pipeline.py`'s stages decide.

`pipeline.py` is two things wound together: the arithmetic that decides, and
the I/O that feeds it.  Only the first half is ported -- chain I/O is the
host's -- so this compares the stages a quote runs between fetches, in the
order it runs them:

1. reduce the universe;
2. assemble the graph;
3. check what the solve came back with;
4. read the flow out of it;
5. rank;
6. layer the pricing walk.

Two of these decide what the router can *see*.  `prune_dead_end_nodes` drops a
node touched by one pool, and it does so on structure rather than on a list of
names: get it wrong in one direction and the long tail of single-pool tokens is
back on the ballot, wrong in the other and `HLX -> USDC` -- a fair question
whose single pool is the answer -- returns no route at all.
`clamp_unphysical_depth` decides which arcs are bottomless, and an arc wrongly
clamped is one the solve will happily fill.

The KCL family is the *refusal*, so it is compared including where it points:
"flow leaving a node nothing fed" is a different bug from a conditioning
failure, and the node index is what tells them apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import pipeline
from erouter.core.accel import available
from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.split import split_groups
from erouter.core.types import ArcKind, PoolArc

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
TOKENS = [
    (WETH, "WETH", 18),
    ("0x" + "22" * 20, "USDC", 6),
    ("0x" + "33" * 20, "USDT", 6),
    ("0x" + "44" * 20, "crvUSD", 18),
    ("0x" + "55" * 20, "HLX", 18),
]


def build_nodes():
    """The same node map on both sides, built by the same calls."""
    import erouter_solve

    reference, ported = NodeMap(), erouter_solve.NodeMap()
    for address, symbol, decimals in TOKENS:
        reference.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)
    # One merged pair, so `rate` is not 1.0 everywhere and the unit
    # conversions in the clamp and in `realised_delta` are exercised.
    share = "0x" + "66" * 20
    reference.add_token(share, "scrvUSD", 18)
    ported.add_token(share, "scrvUSD", 18)
    reference.merge(Conversion(ConversionKind.ERC4626, share, TOKENS[3][0],
                               11 * 10**17, 10**18, target=share))
    ported.merge("ERC4626", share, TOKENS[3][0], str(11 * 10**17), str(10**18), share)
    return reference, ported


def universe(seed: int, nodes: NodeMap):
    """Arcs with parallel pools, a dead-end token and an unphysical depth."""
    rng = np.random.default_rng(seed)
    n = len(TOKENS)
    arcs: list[PoolArc] = []
    for tail in range(n - 1):
        for head in range(n - 1):
            if tail == head:
                continue
            for copy in range(1 + int(rng.integers(0, 2))):
                pool = f"0x{tail:02x}{head:02x}{copy:02x}" + "cd" * 17
                arcs.append(PoolArc(
                    id=f"{pool.lower()}:0:{tail}>{head}",
                    pool=pool, kind=ArcKind.SWAP_STABLE, i=tail, j=head,
                    n_coins=4,
                    token_in=TOKENS[tail][0], token_out=TOKENS[head][0],
                    tau=nodes.node(TOKENS[tail][0]),
                    sigma=nodes.node(TOKENS[head][0]),
                    a=float(rng.uniform(0.980, 0.999)),
                    B=float(np.exp(rng.uniform(math.log(1e-7), math.log(1e-4)))),
                    reserve_in=int(rng.uniform(1e20, 1e24)),
                    decimals_in=TOKENS[tail][2], decimals_out=TOKENS[head][2],
                    tvl_usd=float(rng.uniform(1e6, 1e8)),
                    note=f"pool {tail}{head}{copy}",
                ))
    # HLX sits in exactly one pool: a dead end unless it is an endpoint.
    lonely = "0x" + "99" * 20
    arcs.append(PoolArc(
        id=f"{lonely}:0:1>4", pool=lonely, kind=ArcKind.SWAP_STABLE, i=0, j=1,
        n_coins=2, token_in=TOKENS[1][0], token_out=TOKENS[4][0],
        tau=nodes.node(TOKENS[1][0]), sigma=nodes.node(TOKENS[4][0]),
        a=0.99, B=1e-5, reserve_in=10**22, decimals_in=6, decimals_out=18,
        tvl_usd=4e4, note="HLX pool",
    ))
    # One arc whose curvature is below the quotes' noise floor.
    arcs[0].B = 1e-30
    # And one pair, so `pair_directions` has something to link.
    twin = arcs[0]
    arcs.append(PoolArc(
        id=f"{twin.pool.lower()}:0:{twin.j}>{twin.i}",
        pool=twin.pool, kind=ArcKind.SWAP_STABLE, i=twin.j, j=twin.i,
        n_coins=4, token_in=twin.token_out, token_out=twin.token_in,
        tau=twin.sigma, sigma=twin.tau, a=0.997, B=twin.B,
        reserve_in=10**22, decimals_in=twin.decimals_out,
        decimals_out=twin.decimals_in, tvl_usd=twin.tvl_usd, note="twin",
    ))
    return arcs


def ported_arcs(arcs):
    import erouter_solve

    built = erouter_solve.Arcs()
    for arc in arcs:
        built.add(arc.id, arc.pool, int(arc.kind), arc.i, arc.j, arc.n_coins,
                  arc.token_in, arc.token_out, arc.tau, arc.sigma,
                  arc.a, arc.B, arc.cap, arc.G, arc.eps, arc.reserve_in,
                  arc.decimals_in, arc.tvl_usd, arc.gamma_live, arc.note)
    return built


def stages(arcs):
    import erouter_solve

    return erouter_solve.Stages(ported_arcs(arcs))


class Result:
    """Enough of `RouteResult` for the stages to write their counters into."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.warnings: list[str] = []


SEEDS = list(range(8))


# ----------------------------------------------------- 1. the universe


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_dead_end_nodes_are_pruned(seed):
    reference, _ = build_nodes()
    arcs = universe(seed, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])

    result = Result()
    want = pipeline._prune_dead_end_nodes(arcs, src, dst, result)
    got = stages(arcs)
    got.prune_dead_end_nodes(src, dst)

    assert got.arc_ids() == [a.id for a in want]
    assert dict(got.counters())["arcs_dead_end"] == result.counters["arcs_dead_end"]


def test_a_single_pool_token_survives_as_an_endpoint():
    """`HLX -> USDC` is a fair question and its one pool is the answer."""
    reference, _ = build_nodes()
    arcs = universe(0, reference)
    src, dst = reference.node(TOKENS[4][0]), reference.node(TOKENS[1][0])

    result = Result()
    want = pipeline._prune_dead_end_nodes(arcs, src, dst, result)
    got = stages(arcs)
    got.prune_dead_end_nodes(src, dst)
    assert got.arc_ids() == [a.id for a in want]
    assert any(a.token_out.lower() == TOKENS[4][0] for a in want)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_component_is_kept(seed):
    reference, _ = build_nodes()
    arcs = universe(seed, reference)
    dst = reference.node(TOKENS[3][0])

    result = Result()
    want = pipeline._restrict_to_component(arcs, dst, reference.n_nodes, result)
    got = stages(arcs)
    got.restrict_to_component(dst, reference.n_nodes)

    assert got.arc_ids() == [a.id for a in want]
    assert dict(got.counters()) == {
        "arcs_unreachable": result.counters["arcs_unreachable"],
        "nodes_reachable": result.counters["nodes_reachable"],
    }


# ------------------------------------------------------- 2. the graph


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_arcs_are_clamped_as_bottomless(seed):
    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    nu = np.array([1.0, 2600.0, 2600.0, 2600.0, 1e6, 2600.0][:reference.n_nodes])

    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    want = pipeline._clamp_unphysical_depth(mine, nu, reference)
    got = stages(arcs)
    assert got.clamp_unphysical_depth([float(v) for v in nu], ported_nodes) == want
    assert want > 0, "the fixture is supposed to contain one"

    numbers = np.array(got.arc_numbers()).reshape(-1, 5)
    expected = np.array([[a.a, a.B, a.cap, a.G, a.eps] for a in mine])
    agree = (numbers == expected) | (np.isnan(numbers) & np.isnan(expected))
    assert agree.all(), (numbers[~agree], expected[~agree])
    assert got.arc_flags() == [(a.clamped, a.convex_flag) for a in mine]


@pytest.mark.parametrize("seed", SEEDS)
def test_assembly_agrees_arc_for_arc(seed):
    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])
    nu = np.ones(reference.n_nodes)
    Psi = 1.0

    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    result = Result()
    kept, g = pipeline._assemble(mine, nu, Psi, reference, src, dst, result)

    got = stages(arcs)
    ported_g = got.assemble([float(v) for v in nu], Psi, ported_nodes, src, dst)

    assert got.arc_ids() == [a.id for a in kept]
    assert list(ported_g.g) == list(g.G)
    assert list(ported_g.eps) == list(g.eps)
    assert list(ported_g.cap) == list(g.cap)
    assert ported_g.ill_conditioned == g.ill_conditioned
    assert dict(got.counters()) == result.counters
    assert got.warnings() == result.warnings

    numbers = np.array(got.arc_numbers()).reshape(-1, 5)
    expected = np.array([[a.a, a.B, a.cap, a.G, a.eps] for a in kept])
    agree = (numbers == expected) | (np.isnan(numbers) & np.isnan(expected))
    assert agree.all(), (numbers[~agree], expected[~agree])


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_pairs_are_linked_and_flagged(seed):
    reference, _ = build_nodes()
    arcs = universe(seed, reference)

    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    want = pipeline.pair_directions(mine)
    got = stages(arcs)
    assert got.pair_directions() == want
    assert want > 0, "the fixture is supposed to contain a pair"

    assert got.reverse_ids() == [a.reverse_id or "" for a in mine]
    gammas, expected = np.array(got.gamma_live()), np.array([a.gamma_live for a in mine])
    agree = (gammas == expected) | (np.isnan(gammas) & np.isnan(expected))
    assert agree.all()


def test_a_spurious_negative_two_cycle_is_flagged_on_both_sides():
    """§2.6: `eps_f + eps_r <= 0` means `nu` is inconsistent with that pool."""
    reference, _ = build_nodes()
    arcs = universe(0, reference)
    # Force the pair's drops to sum to nothing.
    forward = arcs[0]
    reverse = next(a for a in arcs if a.note == "twin")
    forward.eps, reverse.eps = 0.001, -0.002

    result = Result()
    pipeline._warn_pair_drops(arcs, result)
    got = stages(arcs)
    got.warn_pair_drops()

    assert dict(got.counters())["eps_pair_violations"] == \
        result.counters["eps_pair_violations"]
    assert got.warnings() == result.warnings
    assert result.warnings, "the fixture is supposed to trip it"


# ------------------------------------------------------- 3. the check


@pytest.mark.parametrize("Psi", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("g_scale", [0.0, 1.0, 1e8])
def test_the_kcl_tolerance_agrees(Psi, g_scale):
    import erouter_solve

    assert erouter_solve.kcl_tolerance(Psi, g_scale) == \
        pipeline._kcl_tolerance(Psi, g_scale)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_kcl_residual_and_where_it_is_agree(seed):
    import erouter_solve

    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])
    nu = np.ones(reference.n_nodes)

    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    _, g = pipeline._assemble(mine, nu, 1.0, reference, src, dst, Result())
    ported = stages(arcs)
    ported_g = ported.assemble([float(v) for v in nu], 1.0, ported_nodes, src, dst)

    rng = np.random.default_rng(seed + 400)
    for _ in range(6):
        psi = rng.random(g.m) * (rng.random(g.m) > 0.5)
        for Psi in (0.0, 1.0):
            want = pipeline._kcl_detail(g, psi, src, dst, Psi)
            assert erouter_solve.kcl_detail(ported_g, list(psi), src, dst, Psi) == want


def test_conjured_flow_is_told_apart_from_a_conditioning_failure():
    """The node index is what distinguishes them, so it is compared."""
    import erouter_solve

    reference, ported_nodes = build_nodes()
    arcs = universe(0, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])
    nu = np.ones(reference.n_nodes)
    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    _, g = pipeline._assemble(mine, nu, 1.0, reference, src, dst, Result())
    ported = stages(arcs)
    ported_g = ported.assemble([float(v) for v in nu], 1.0, ported_nodes, src, dst)

    # One arc carrying flow nothing fed it.
    psi = np.zeros(g.m)
    orphan = next(k for k in range(g.m) if g.tau[k] != src)
    psi[orphan] = 0.5
    want = pipeline._kcl_detail(g, psi, src, dst, 1.0)
    assert erouter_solve.kcl_detail(ported_g, list(psi), src, dst, 1.0) == want
    assert want[0] > 0.0, "the fixture is supposed to conjure flow"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_achievable_kcl_floor_agrees_in_magnitude(seed):
    """A different eigensolver, so this is an order-of-magnitude claim.

    The reference calls `np.linalg.cond`, which is LAPACK's SVD; the port runs
    a cyclic Jacobi, because `rust/README.md` rules out BLAS and LAPACK -- no
    wasm build, and at `n ~ 50` a hand-written kernel wins anyway.  Binding it
    would not make this exact regardless: numpy's OpenBLAS is `DYNAMIC_ARCH`
    and threaded, so it is not one fixed sequence of operations.

    Same quantity, and it feeds a safety factor of 100 -- so what has to agree
    is the scale, which is what the caller compares its residual against.
    """
    import erouter_solve

    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])
    nu = np.ones(reference.n_nodes)
    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    _, g = pipeline._assemble(mine, nu, 1.0, reference, src, dst, Result())
    ported = stages(arcs)
    ported_g = ported.assemble([float(v) for v in nu], 1.0, ported_nodes, src, dst)

    rng = np.random.default_rng(seed + 800)
    for _ in range(4):
        active = rng.random(g.m) > 0.4
        want = pipeline._achievable_kcl(g, active, dst)
        got = erouter_solve.achievable_kcl(ported_g, [bool(v) for v in active], dst)
        if want == 0.0:
            assert got == 0.0
            continue
        assert got == pytest.approx(want, rel=1e-6)

    # Nothing active leaves the caller's flat tolerance in charge.
    assert erouter_solve.achievable_kcl(ported_g, [False] * g.m, dst) == 0.0
    assert pipeline._achievable_kcl(g, np.zeros(g.m, bool), dst) == 0.0


# -------------------------------------------------- 4. reading it back


@pytest.mark.parametrize("seed", SEEDS)
def test_the_realised_flow_reads_the_same(seed):
    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    nu = np.array([1.0, 2600.0, 2600.0, 2600.0, 1e6, 2600.0][:reference.n_nodes])
    ported = stages(arcs)

    rng = np.random.default_rng(seed + 200)
    psi = rng.random(len(arcs)) * (rng.random(len(arcs)) > 0.3)
    active = [k for k in range(len(arcs)) if psi[k] > 0]

    for k in range(len(arcs)):
        assert ported.realised_delta(k, float(psi[k]), [float(v) for v in nu],
                                     ported_nodes) == \
            pipeline._realised_delta(arcs[k], float(psi[k]), nu, reference)

    want = pipeline._realised_theta(arcs, psi, nu, reference, active)
    got = ported.realised_theta([float(v) for v in psi], [float(v) for v in nu],
                                ported_nodes, active)
    assert dict(got) == want


@pytest.mark.parametrize("decimals", [0, 6, 8, 18, 24])
def test_the_quantum_agrees(decimals):
    import erouter_solve

    assert erouter_solve.quantum(decimals) == pipeline._quantum(decimals)


# --------------------------------------------------------- 5. the rank


@pytest.mark.parametrize("gas_price", [0, 45_000_000, 30_000_000_000])
def test_the_gas_floor_agrees(gas_price):
    import erouter_solve

    reference, ported_nodes = build_nodes()
    nu = np.array([1.0, 1 / 2600, 1 / 2600, 1 / 2600, 1e-6, 1 / 2600][:reference.n_nodes])
    for dst in (TOKENS[1][0], TOKENS[3][0]):
        assert erouter_solve.dst_per_eth(ported_nodes, [float(v) for v in nu], dst) == \
            pipeline._dst_per_eth(reference, nu, dst)
        for g_scale in (0.0, 1.0, 7.6e7):
            assert erouter_solve.gas_cost(ported_nodes, [float(v) for v in nu], dst,
                                          gas_price, g_scale) == \
                pipeline._gas_cost(reference, nu, dst, gas_price, g_scale)


def test_an_unpriced_eth_disables_the_gas_term_on_both_sides():
    import erouter_solve

    reference, ported = NodeMap(), erouter_solve.NodeMap()
    for address, symbol, decimals in TOKENS[1:]:
        reference.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)
    nu = np.ones(reference.n_nodes)
    assert erouter_solve.dst_per_eth(ported, [float(v) for v in nu], TOKENS[1][0]) == \
        pipeline._dst_per_eth(reference, nu, TOKENS[1][0]) == 0.0


# ---------------------------------------------------------- 6. the walk


def route_for(seed):
    """A realised route on both sides, to layer and rank."""
    import erouter_solve

    reference, ported_nodes = build_nodes()
    arcs = universe(seed, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[3][0])
    nu = np.ones(reference.n_nodes)
    mine = [PoolArc(**{f.name: getattr(a, f.name)
                       for f in a.__dataclass_fields__.values()}) for a in arcs]
    kept, _ = pipeline._assemble(mine, nu, 1.0, reference, src, dst, Result())

    # A flow that fans out of the source and rejoins, so there is a split to
    # find and more than one pricing layer.
    picked = [k for k, a in enumerate(kept) if a.tau == src][:2]
    picked += [k for k, a in enumerate(kept)
               if a.tau == kept[picked[0]].sigma and a.sigma == dst][:1]
    if len(picked) < 3:
        pytest.skip("no fan-out in this universe")
    flows = [1.0] * len(picked)

    from erouter.core.realize import realize

    want = realize([kept[k] for k in picked], np.array(flows), nu, reference,
                   src_token=TOKENS[0][0], dst_token=TOKENS[3][0],
                   amount_in=10**18)
    built = erouter_solve.Arcs()
    for k in picked:
        a = kept[k]
        built.add(a.id, a.pool, int(a.kind), a.i, a.j, a.n_coins, a.token_in,
                  a.token_out, a.tau, a.sigma, a.a, a.B, a.cap, a.G, a.eps,
                  a.reserve_in, a.decimals_in, a.tvl_usd, a.gamma_live, a.note)
    got = erouter_solve.Route.realize(built, flows, [float(v) for v in nu],
                                      ported_nodes, TOKENS[0][0], TOKENS[3][0],
                                      str(10**18), None)
    return want, got


@pytest.mark.parametrize("seed", SEEDS)
def test_the_pricing_walk_batches_the_same_way(seed):
    """Depth is what costs round trips, so the layering is the cost model."""
    import erouter_solve

    want, got = route_for(seed)
    assert erouter_solve.pricing_layers(got) == pipeline._pricing_layers(want.legs)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_split_groups_and_scout_priority_agree(seed):
    import erouter_solve

    want, got = route_for(seed)
    assert erouter_solve.split_groups(got) == \
        split_groups([rl.leg for rl in want.legs])
    # A different LU under `route_conductance`, so this is not bit-exact --
    # see `test_realize_differential`.
    assert erouter_solve.scout_priority(got) == \
        pytest.approx(pipeline.scout_priority(want), rel=1e-12)


def test_a_route_with_nothing_to_re_split_scores_zero_on_both_sides():
    """`split.scout` drops a plan whose legs form no split group."""
    import erouter_solve

    reference, ported_nodes = build_nodes()
    arcs = universe(0, reference)
    src, dst = reference.node(TOKENS[0][0]), reference.node(TOKENS[1][0])
    nu = np.ones(reference.n_nodes)
    single = next(a for a in arcs if a.tau == src and a.sigma == dst)

    from erouter.core.realize import realize

    want = realize([single], np.array([1.0]), nu, reference,
                   src_token=TOKENS[0][0], dst_token=TOKENS[1][0],
                   amount_in=10**18)
    built = erouter_solve.Arcs()
    built.add(single.id, single.pool, int(single.kind), single.i, single.j,
              single.n_coins, single.token_in, single.token_out, single.tau,
              single.sigma, single.a, single.B, single.cap, single.G, single.eps,
              single.reserve_in, single.decimals_in, single.tvl_usd,
              single.gamma_live, single.note)
    got = erouter_solve.Route.realize(built, [1.0], [float(v) for v in nu],
                                      ported_nodes, TOKENS[0][0], TOKENS[1][0],
                                      str(10**18), None)
    assert erouter_solve.scout_priority(got) == pipeline.scout_priority(want) == 0.0
