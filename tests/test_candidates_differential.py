"""The Rust ballot must be the ballot `core/candidates.py` and `verify.py` build.

This is the stage where a divergence changes *which route the user gets*, and
it does so without anything looking wrong: a candidate that fails to generate
is simply absent, and a rank one place out picks a different route with a
perfectly plausible number beside it.

Two halves.

**Generation** is a sequence of re-solves whose order is the answer -- the
budget truncates, so a family that runs late may not run at all, and §13.1 is
explicit that the pin sweep has to outrank the drop candidates.  So the whole
ballot is compared: labels in order, flows arc for arc, and the solve and pivot
counters, which are what say the two took the same path through the generator
rather than arriving at the same place by luck.

**Ranking** is compared over hand-set quotes, because that isolates it from the
chain: the two implementations see identical numbers and must order them
identically, including the two rules that are not "largest wins" -- the tie
goes to the shorter route, and a direct candidate is a floor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import graph
from erouter.core.accel import available
from erouter.core.candidates import (
    ACTIVE_FLOOR,
    TOP_K,
    _by_new_pools,
    _pool_of,
    _spread,
    carries,
    conflicting_pools,
    generate,
    keep_only,
    repair_order,
)
from erouter.core.gas import GasTable
from erouter.core.nodes import NodeMap
from erouter.core.risk import RiskTable
from erouter.core.seed import k_shortest_paths
from erouter.core.solve import active_set_solve
from erouter.core.types import ArcKind, PoolArc
from erouter.core.verify import realize_candidates, verify

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

TOKENS = [
    ("0x" + "11" * 20, "AAA", 18),
    ("0x" + "22" * 20, "BBB", 18),
    ("0x" + "33" * 20, "CCC", 18),
    ("0x" + "44" * 20, "DDD", 18),
    ("0x" + "55" * 20, "EEE", 18),
]


# --------------------------------------------------------------- universes


def universe(seed: int):
    """A small graph with parallel pools, a hub and a re-entered pool."""
    rng = np.random.default_rng(seed)
    n = len(TOKENS)
    nodes = NodeMap()
    ported = _ported_nodes()
    for address, symbol, decimals in TOKENS:
        nodes.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)

    arcs: list[PoolArc] = []
    # Every ordered pair gets a pool or two, so the generator has paths,
    # parallel venues and a re-entrant pool to repair around.
    for tail in range(n):
        for head in range(n):
            if tail == head:
                continue
            for copy in range(1 + int(rng.integers(0, 2))):
                pool = f"0x{tail:02x}{head:02x}{copy:02x}" + "ab" * 17
                # Deliberately easy instances, and the reason is worth
                # stating: `a < 1` everywhere makes every `eps` positive, so
                # there are no arbitrage cycles, and a real spread between
                # parallel arcs leaves no ties for a pivot rule to break
                # arbitrarily. Degeneracy is where the two *solvers* take
                # different pivot sequences -- a difference that belongs to
                # `solve`, is reproduced in `test_solve_differential`, and
                # would otherwise be re-measured here as though it were a
                # generator bug. This test is about the generator.
                a = float(rng.uniform(0.980, 0.999))
                B = float(np.exp(rng.uniform(math.log(1e-7), math.log(1e-5))))
                arcs.append(PoolArc(
                    id=f"{tail}>{head}#{copy}",
                    pool=pool, kind=ArcKind.SWAP_STABLE, i=0, j=1, n_coins=2,
                    token_in=TOKENS[tail][0], token_out=TOKENS[head][0],
                    tau=tail, sigma=head, a=a, B=B,
                    reserve_in=10**24, decimals_in=18, decimals_out=18,
                    tvl_usd=float(rng.uniform(1e6, 1e8)),
                    gamma_live=a, note=f"pool {tail}{head}{copy}",
                ))
    # One pool entered twice, on two different coin pairs of the same address:
    # a conflict the repair family exists for.
    twin = arcs[0].pool
    arcs.append(PoolArc(
        id="twin", pool=twin, kind=ArcKind.SWAP_STABLE, i=0, j=1, n_coins=2,
        token_in=TOKENS[1][0], token_out=TOKENS[0][0],
        tau=1, sigma=0, a=0.999, B=2e-6,
        reserve_in=10**24, decimals_in=18, decimals_out=18,
        tvl_usd=5e6, gamma_live=0.999, note="twin",
    ))
    # A three-coin pool paying two ports out of one coin, which is what the
    # element family is for: same address, same `tau`, two different `sigma`.
    element_pool = "0x" + "ee" * 20
    for sink, j in ((2, 1), (3, 2)):
        arcs.append(PoolArc(
            id=f"element>{sink}", pool=element_pool, kind=ArcKind.SWAP_STABLE,
            i=0, j=j, n_coins=3,
            token_in=TOKENS[0][0], token_out=TOKENS[sink][0],
            tau=0, sigma=sink, a=0.998, B=3e-6,
            reserve_in=10**24, decimals_in=18, decimals_out=18,
            tvl_usd=2e7, gamma_live=0.998, note=f"element {sink}",
        ))

    tau = np.array([arc.tau for arc in arcs], np.int64)
    sig = np.array([arc.sigma for arc in arcs], np.int64)
    a = np.array([arc.a for arc in arcs], float)
    B = np.array([arc.B for arc in arcs], float)
    nu = np.ones(n)
    Psi = 1.0
    g = graph.build(tau, sig, a, B, nu, Psi, n_nodes=n, merge_duplicates=False)
    for k, arc in enumerate(arcs):
        arc.G, arc.eps = float(g.G[k]), float(g.eps[k])
    return g, arcs, nodes, ported, nu, Psi


def _ported_nodes():
    import erouter_solve

    return erouter_solve.NodeMap()


def ported_graph(g):
    import erouter_solve

    return erouter_solve.Graph.build(
        [int(v) for v in g.tau], [int(v) for v in g.sig],
        [float(v) for v in g.a], [float(v) for v in g.B],
        # `arc_params` is re-derived from `nu`, so the ones that built `g`.
        [1.0] * g.n_nodes, 1.0,
        cap=[float(v) for v in g.cap],
        flagged=[bool(v) for v in g.flagged], n_nodes=g.n_nodes,
        merge_duplicates=False,
    )


def ported_arcs(arcs):
    import erouter_solve

    built = erouter_solve.Arcs()
    for arc in arcs:
        built.add(arc.id, arc.pool, int(arc.kind), arc.i, arc.j, arc.n_coins,
                  arc.token_in, arc.token_out, arc.tau, arc.sigma,
                  arc.a, arc.B, arc.cap, arc.G, arc.eps, arc.reserve_in,
                  arc.decimals_in, arc.tvl_usd, arc.gamma_live, arc.note)
    return built


SEEDS = list(range(10))

#: Universes where the *solver* -- not the generator -- takes a different pivot
#: sequence, so the ballots would legitimately differ. Empty, and deliberately
#: kept: the universes above are chosen to be well conditioned precisely so
#: this stays empty, and a seed appearing here later is the signal that a
#: change made them degenerate again rather than that the generator broke.
#: The solver difference itself is reproduced in `test_solve_differential`.
SOLVER_DIVERGES: set[int] = set()


def skip_where_the_solver_diverges(seed):
    if seed in SOLVER_DIVERGES:
        pytest.skip("the ported solver diverges on this universe; see "
                    "test_solve_differential")


# ------------------------------------------------------------- the helpers


@pytest.mark.parametrize("seed", SEEDS)
def test_carries_agrees(seed):
    """Membership must not be kernel-dependent -- §6's 72 bp lesson."""
    import erouter_solve

    rng = np.random.default_rng(seed)
    psi = np.concatenate([rng.random(6), [0.0, 1e-18, ACTIVE_FLOOR, 2 * ACTIVE_FLOOR]])
    for psi_total in (1.0, 1e-9, 1e6):
        want = [bool(v) for v in carries(psi, psi_total)]
        assert erouter_solve.Ballot.carries(list(psi), psi_total) == want


@pytest.mark.parametrize("budget", range(1, 10))
def test_spread_agrees(budget):
    """What keeps the path family off the dense low end of the ladder."""
    import erouter_solve

    assert erouter_solve.Ballot.spread(list(TOP_K), budget) == sorted(_spread(TOP_K, budget))


@pytest.mark.parametrize("seed", SEEDS)
def test_k_shortest_paths_agree(seed):
    """Yen's, arc for arc: the unions the sparse family is built from."""
    import erouter_solve

    g = universe(seed)[0]
    ported = ported_graph(g)
    for k in (1, 3, 6, 12):
        want = [[int(a) for a in path] for path in k_shortest_paths(g, 0, len(TOKENS) - 1, k=k)]
        assert erouter_solve.Ballot.k_shortest_paths(ported, 0, len(TOKENS) - 1, k) == want


@pytest.mark.parametrize("seed", SEEDS)
def test_by_new_pools_agrees(seed):
    """Ordering only, but it decides which unions the budget reaches."""
    import erouter_solve

    g, arcs = universe(seed)[:2]
    pools = _pool_of(arcs)
    paths = [[int(a) for a in path]
             for path in k_shortest_paths(g, 0, len(TOKENS) - 1, k=12)]
    want = [[int(a) for a in path] for path in _by_new_pools(paths, pools)]
    got = erouter_solve.Ballot.by_new_pools(paths, [str(p) for p in pools])
    assert got == want


@pytest.mark.parametrize("seed", SEEDS)
def test_conflicting_pools_agree(seed):
    """The rule that decides which candidates need repairing at all."""
    import erouter_solve

    g, arcs = universe(seed)[:2]
    ported = ported_arcs(arcs)
    rng = np.random.default_rng(seed + 500)
    for _ in range(20):
        psi = rng.random(g.m) * (rng.random(g.m) > 0.4)
        for psi_total in (0.0, 1.0):
            want = conflicting_pools(arcs, psi, psi_total)
            got = erouter_solve.Ballot.conflicting_pools(ported, list(psi), psi_total)
            assert dict(got) == {k: list(v) for k, v in want.items()}
            assert [p for p, _ in got] == list(want), "insertion order decides the repair"


@pytest.mark.parametrize("seed", SEEDS)
def test_repair_order_and_keep_only_agree(seed):
    """Which arc a repair keeps, and which it bans."""
    import erouter_solve

    g, arcs = universe(seed)[:2]
    rng = np.random.default_rng(seed + 700)
    for _ in range(10):
        psi = rng.random(g.m) * (rng.random(g.m) > 0.3)
        conflicts = conflicting_pools(arcs, psi, 1.0)
        if not conflicts:
            continue
        want_order = repair_order(conflicts, psi)
        got_order = erouter_solve.Ballot.repair_order(
            [(p, list(v)) for p, v in conflicts.items()], list(psi))
        assert {p: list(v) for p, v in got_order} == {p: list(v) for p, v in want_order.items()}
        for rank in range(4):
            banned = np.zeros(g.m, bool)
            applied = keep_only(banned, want_order, rank)
            got_banned, got_applied = erouter_solve.Ballot.keep_only(
                [False] * g.m, got_order, rank)
            assert got_banned == [bool(v) for v in banned]
            assert got_applied == applied


def test_a_mismatched_flow_is_refused_rather_than_crashing():
    """A panic would take a wasm instance down, so the binding checks."""
    import erouter_solve

    g, arcs, *_ = universe(0)
    with pytest.raises(ValueError, match="psi has"):
        erouter_solve.Ballot.conflicting_pools(ported_arcs(arcs), [0.0] * (g.m + 1))


# ------------------------------------------------------------- generation


def both_ballots(seed, **kw):
    import erouter_solve

    g, arcs, nodes, ported_nodes, nu, Psi = universe(seed)
    src, dst = 0, len(TOKENS) - 1
    base = active_set_solve(g, src, dst, Psi)
    want = generate(g, arcs, src, dst, Psi, base, **kw)
    got = erouter_solve.Ballot.generate(
        ported_graph(g), ported_arcs(arcs), src, dst, Psi,
        [float(v) for v in base.psi], **kw)
    return g, arcs, nodes, ported_nodes, nu, Psi, want, got


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_candidates_are_generated_in_the_same_order(seed):
    skip_where_the_solver_diverges(seed)
    *_, want, got = both_ballots(seed)

    assert got.labels() == [c.label for c in want.candidates]
    assert got.kinds() == [c.kind for c in want.candidates]
    assert got.reasons() == [c.reason for c in want.candidates]
    assert got.certificates() == [c.certificate for c in want.candidates]
    assert got.n_arcs() == [c.n_arcs for c in want.candidates]

    # The *flows* inherit the solver's own kernel difference and carry its
    # tolerance -- `test_solve_differential` allows `1e-6 * Psi` between one
    # linear algebra kernel and another, and a candidate is a re-solve. What
    # must be exact is everything structural above: which candidates exist, in
    # what order, over which arcs. That is what decides the ballot.
    for k, candidate in enumerate(want.candidates):
        assert np.allclose(got.psi(k), candidate.psi, atol=1e-6), (k, candidate.label)
    assert np.allclose(got.modelled_loss(),
                       [c.modelled_loss for c in want.candidates], rtol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_generator_took_the_same_path_not_just_the_same_place(seed):
    skip_where_the_solver_diverges(seed)
    """`solves` and `pivots` say how the ballot was reached, not what it is."""
    *_, want, got = both_ballots(seed)
    assert got.solves == want.solves
    assert got.pivots == want.pivots
    assert got.skipped == want.skipped
    assert got.skipped_wide == want.skipped_wide


@pytest.mark.parametrize("budget", [1, 3, 6, 12, 20])
def test_a_truncated_budget_keeps_the_same_families(budget):
    """Ordering is the answer when the budget bites."""
    *_, want, got = both_ballots(3, max_candidates=budget)
    assert got.labels() == [c.label for c in want.candidates]
    assert len(got) <= budget


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_a_gas_floor_prunes_the_same_arcs(seed):
    skip_where_the_solver_diverges(seed)
    *_, want, got = both_ballots(seed, gas_floor=1e-3)
    assert got.labels() == [c.label for c in want.candidates]
    for k, candidate in enumerate(want.candidates):
        assert np.allclose(got.psi(k), candidate.psi, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_a_tight_leg_limit_skips_the_same_candidates(seed):
    skip_where_the_solver_diverges(seed)
    *_, want, got = both_ballots(seed, max_legs=3)
    assert got.labels() == [c.label for c in want.candidates]
    assert got.skipped == want.skipped


def test_an_element_pricer_is_offered_the_same_pairs():
    """The callback is the caller's, so both must ask it the same questions."""
    import erouter_solve

    # Seed 0 is one where a pool pays two ports out of one coin *and* the
    # solve puts flow through it, which is what the family needs.
    g, arcs, _, _, _, Psi = universe(0)
    src, dst = 0, len(TOKENS) - 1
    base = active_set_solve(g, src, dst, Psi)

    asked_reference: list[tuple] = []
    asked_ported: list[tuple] = []

    def reference_split(a, b, pa, pb):
        asked_reference.append((a.id, b.id, pa, pb))
        return (pa * 0.6 + 1e-9, pb * 0.4 + 1e-9)

    def ported_split(ka, kb, pa, pb):
        asked_ported.append((arcs[ka].id, arcs[kb].id, pa, pb))
        return (pa * 0.6 + 1e-9, pb * 0.4 + 1e-9)

    want = generate(g, arcs, src, dst, Psi, base, element_split=reference_split)
    got = erouter_solve.Ballot.generate(
        ported_graph(g), ported_arcs(arcs), src, dst, Psi,
        [float(v) for v in base.psi], element_split=ported_split)

    assert asked_ported == asked_reference
    assert asked_reference, "the element family never ran"
    assert got.labels() == [c.label for c in want.candidates]


# ------------------------------------------------------ realise and rank


def realized(seed, **kw):
    """Both ballots, realised into legs."""
    _, arcs, nodes, ported_nodes, nu, _, want, got = both_ballots(seed, **kw)
    amount_in = 10**18
    realize_candidates(want, arcs, nu, nodes,
                       src_token=TOKENS[0][0], dst_token=TOKENS[-1][0],
                       amount_in=amount_in)
    got.realize_candidates(ported_arcs(arcs), [float(v) for v in nu], ported_nodes,
                           TOKENS[0][0], TOKENS[-1][0], str(amount_in))
    return want, got


@pytest.mark.parametrize("seed", SEEDS)
def test_realisation_marks_the_same_candidates(seed):
    skip_where_the_solver_diverges(seed)
    want, got = realized(seed)
    assert got.statuses() == [c.status for c in want.candidates]
    assert got.notes() == [c.note for c in want.candidates]
    assert got.legs() == [len(c.route.legs) if c.route else 0
                          for c in want.candidates]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_candidates_are_put_up_for_quoting(seed):
    skip_where_the_solver_diverges(seed)
    want, got = realized(seed)
    ready = [k for k, c in enumerate(want.candidates)
             if c.status == "ready" and c.route]
    assert got.ready() == ready


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gas_price", [0, 45_000_000, 5_000_000_000])
def test_ranking_agrees(seed, gas_price):
    skip_where_the_solver_diverges(seed)
    """Hand-set quotes, so this isolates the ranking from the chain."""
    import erouter_solve

    want, got = realized(seed)
    ready = got.ready()
    if not ready:
        pytest.skip("nothing realised for this universe")

    rng = np.random.default_rng(seed + 900)
    # Quotes clustered tightly, which is where the tie-break decides.
    quotes = [int(1e21 * (1.0 + rng.normal(0.0, 1e-6))) for _ in ready]

    class Fake:
        def quote_routes(self, legs, amounts, slots):
            assert len(legs) == len(quotes), (len(legs), len(quotes))
            return quotes

    verify(want, Fake(), amount_in=10**18, gas_price_wei=gas_price,
           dst_wei_per_eth=3e14)
    tables = erouter_solve.Tables()
    got.verify(list(zip(ready, quotes, strict=True)), tables=tables,
               gas_price_wei=gas_price, dst_wei_per_eth=3e14)

    assert got.statuses() == [c.status for c in want.candidates]
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert got.gas() == [c.gas for c in want.candidates]
    assert [int(v) for v in got.verified_out()] == [
        -1 if c.verified_out is None else c.verified_out for c in want.candidates]
    best = want.best
    assert got.best() == (want.candidates.index(best) if best else None)


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_ranking_agrees_with_a_risk_table(seed):
    skip_where_the_solver_diverges(seed)
    """The survival term, which multiplies rather than subtracts."""
    import erouter_solve

    want, got = realized(seed)
    ready = got.ready()
    if not ready:
        pytest.skip("nothing realised for this universe")
    quotes = [int(1e21 + k) for k in range(len(ready))]

    class Fake:
        def quote_routes(self, legs, amounts, slots):
            assert len(legs) == len(quotes), (len(legs), len(quotes))
            return quotes

    risk = RiskTable(default=0.01)
    verify(want, Fake(), amount_in=10**18, gas_price_wei=45_000_000,
           dst_wei_per_eth=3e14, gas_table=GasTable(), risk_table=risk)

    tables = erouter_solve.Tables()
    tables.default_risk = 0.01
    got.verify(list(zip(ready, quotes, strict=True)), tables=tables,
               gas_price_wei=45_000_000, dst_wei_per_eth=3e14)

    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert got.survival() == [c.survival for c in want.candidates]


def test_a_zero_quote_is_a_revert_on_both_sides():

    want, got = realized(2)
    ready = got.ready()
    if not ready:
        pytest.skip("nothing realised for this universe")
    quotes = [0] * len(ready)

    class Fake:
        def quote_routes(self, legs, amounts, slots):
            assert len(legs) == len(quotes), (len(legs), len(quotes))
            return quotes

    verify(want, Fake(), amount_in=10**18)
    got.verify(list(zip(ready, quotes, strict=True)))

    assert got.statuses() == [c.status for c in want.candidates]
    assert got.notes() == [c.note for c in want.candidates]
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert got.best() is None and want.best is None


def test_a_stale_rank_never_survives_a_re_verify_on_either_side():
    """Ranking runs unconditionally, so a rank cannot outlive its quote.

    The reference re-ranks on every call even when nothing needs quoting,
    because it is called three times a route -- candidates, then the direct
    floor, then the refit -- and an early return used to leave ranks from the
    call before. A stale rank silently picks the wrong route.
    """

    want, got = realized(0)
    ready = got.ready()
    if len(ready) < 2:
        pytest.skip("needs two candidates")
    good = [int(1e21 + k) for k in range(len(ready))]

    class Fake:
        def __init__(self, values):
            self.values = values

        def quote_routes(self, legs, amounts, slots):
            return self.values

    verify(want, Fake(good), amount_in=10**18)
    got.verify(list(zip(ready, good, strict=True)))
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert any(got.ranks())

    # A second call with nothing new to quote: both re-rank rather than
    # returning early, and both keep the same answer.
    verify(want, Fake([]), amount_in=10**18)
    got.verify([])
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert any(got.ranks())

    # Now every candidate reverts: every rank goes with it.
    for candidate in want.candidates:
        if candidate.status == "ok":
            candidate.status = "reverted"
            candidate.verified_out = None
    verify(want, Fake([]), amount_in=10**18)
    got.verify([(k, 0) for k in ready])
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    assert not any(got.ranks())


# ---------------------------------------------------------- the tie rules


def test_the_tie_goes_to_the_shorter_route_on_both_sides():
    """A 25-leg route gaining 0.02 bp over a 1-leg one is the same answer."""

    want, got = realized(0)
    ready = got.ready()
    if len(ready) < 2:
        pytest.skip("needs two candidates")
    legs = got.legs()
    # The longest realised candidate quotes fractionally more than the rest.
    longest = max(ready, key=lambda k: legs[k])
    quotes = [int(1e21) + (1 if k == longest else 0) for k in ready]

    class Fake:
        def quote_routes(self, legs, amounts, slots):
            assert len(legs) == len(quotes), (len(legs), len(quotes))
            return quotes

    verify(want, Fake(), amount_in=10**18)
    got.verify(list(zip(ready, quotes, strict=True)))
    assert got.ranks() == [c.rank or 0 for c in want.candidates]
    # And the winner is not simply the largest number on the page.
    winner = got.best()
    assert legs[winner] <= legs[longest]
