"""Routing as a nonlinear circuit: the solve, its barriers, and its ballot."""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import circuit, graph
from erouter.core.candidates import conflicting_pools
from erouter.core.types import ArcKind, PoolArc

POOL = ["0x" + f"{k:02x}" * 20 for k in range(1, 9)]


def arc(k, pool, tau, sigma, *, a=1.0, B=1e-3, cap=math.inf, kind=ArcKind.SWAP_STABLE,
        i=0, j=1, n=2, parallel=False):
    return PoolArc(id=f"{pool}:{k}", pool=pool, kind=kind, i=i, j=j, n_coins=n,
                   token_in=f"0xin{k}", token_out=f"0xout{k}", tau=tau, sigma=sigma,
                   a=a, B=B, cap=cap, parallel=parallel, note=f"pool{k}")


def solve(arcs, Q, *, n=None, src=0, dst=1):
    n = n or max(max(x.tau, x.sigma) for x in arcs) + 1
    nu0 = np.ones(n)
    dev = circuit.devices(arcs, n, nu0, Q)
    return circuit.solve(dev, Q, src, dst, nu0), dev


def test_one_arc_carries_the_whole_trade():
    res, _ = solve([arc(0, POOL[0], 0, 1, a=1.0, B=1e-3)], 100.0)
    assert res.delta[0] == pytest.approx(100.0, rel=1e-8)
    assert res.out[0] == pytest.approx(100.0 - 0.5e-3 * 100.0**2, rel=1e-8)
    assert res.residual < circuit.TOL


def test_parallel_arcs_settle_at_one_marginal_rate():
    res, _ = solve([arc(0, POOL[0], 0, 1, a=1.0, B=1e-3),
                    arc(1, POOL[1], 0, 1, a=0.99, B=2e-3)], 100.0)
    marginals = [1.0 - 1e-3 * res.delta[0], 0.99 - 2e-3 * res.delta[1]]
    assert marginals[0] == pytest.approx(marginals[1], abs=1e-9)
    assert res.delta.sum() == pytest.approx(100.0, rel=1e-9)


def test_a_cheap_capped_arc_saturates_and_the_rest_goes_elsewhere():
    res, _ = solve([arc(0, POOL[0], 0, 1, a=1.0, B=1e-6, cap=30.0),
                    arc(1, POOL[1], 0, 1, a=0.95, B=1e-3)], 100.0)
    assert res.delta[0] == pytest.approx(30.0, rel=1e-6)
    assert res.delta[1] == pytest.approx(70.0, rel=1e-6)


def test_a_tick_range_is_priced_by_its_exact_curve():
    """`a d / (1 + k d)`, k = B / 2a -- not the tangent quadratic."""
    tick = arc(0, POOL[0], 0, 1, a=2.0, B=2 * 2.0 * 0.01, cap=1e9,
               kind=ArcKind.SWAP_UNIV3, parallel=True)
    res, _ = solve([tick], 50.0)
    assert res.out[0] == pytest.approx(2.0 * 50 / 1.5, rel=1e-8)


def test_a_token_nothing_will_buy_does_not_stall_the_solve():
    """Without the price barrier its price raced to zero and every Newton step
    was cut to 1% of the one before."""
    arcs = [arc(0, POOL[0], 0, 1, a=1.0, B=1e-3),
            arc(1, POOL[1], 0, 2, a=1.0, B=1e-3)]       # node 2: a dead end
    res, _ = solve(arcs, 100.0, n=3)
    assert res.residual < circuit.TOL
    assert res.iterations < 60
    assert res.delta[1] < 1e-6 * res.delta[0]


def test_a_linear_arc_with_no_cap_keeps_the_hessian_usable():
    """A clamped arc (B = 0, no cap) put a conductance near 1e12 in the matrix;
    the trade-relative floor keeps it linear to 1e-6 and solvable."""
    arcs = [arc(0, POOL[0], 0, 1, a=1.0, B=0.0),
            arc(1, POOL[1], 0, 1, a=0.999, B=1e-3)]
    res, _ = solve(arcs, 100.0)
    assert res.residual < circuit.TOL
    assert res.delta[0] == pytest.approx(100.0, rel=1e-3)


def build(arcs, n):
    tau = np.array([x.tau for x in arcs], np.int64)
    sig = np.array([x.sigma for x in arcs], np.int64)
    return graph.build(tau, sig, np.array([x.a for x in arcs]), np.array([x.B for x in arcs]),
                       np.ones(n), 1.0, n_nodes=n, merge_duplicates=False)


def test_a_pool_is_not_counted_twice_across_two_ports():
    """TriCRV fed crvUSD and WETH at once, each port its whole depth: the model
    said 580 WETH for a trade the chain paid 434 for.  A pool that cannot be
    priced across two ports keeps one."""
    tri = POOL[0]
    arcs = [
        arc(0, tri, 0, 1, a=1.0, B=2e-3, kind=ArcKind.SWAP_CRYPTO, i=0, j=1, n=3),
        arc(1, tri, 0, 2, a=1.0, B=2e-3, kind=ArcKind.SWAP_CRYPTO, i=0, j=2, n=3),
        arc(2, POOL[1], 1, 3, a=1.0, B=1e-4),
        arc(3, POOL[2], 2, 3, a=1.0, B=1e-4),
        arc(4, POOL[3], 0, 3, a=0.98, B=1e-3),
    ]
    g = build(arcs, 4)
    got = circuit.candidates(g, arcs, np.ones(4), 0, 3, 100.0, advanceable=frozenset())
    assert got.candidates
    for c in got.candidates:
        assert not conflicting_pools(arcs, c.psi, 100.0, advanceable=frozenset())
        assert c.psi[0] == 0.0 or c.psi[1] == 0.0
    # Where the pool can be priced across both ports, the solve uses both.
    both = circuit.candidates(g, arcs, np.ones(4), 0, 3, 100.0, advanceable=frozenset({tri}))
    assert both.candidates[0].psi[0] > 0 and both.candidates[0].psi[1] > 0


def test_the_pruned_candidate_drops_a_branch_worth_less_than_its_leg():
    """A side branch earning a sliver of surplus is not worth 0.02 bp of the
    trade; the pruned candidate goes without it, and verify chooses."""
    arcs = [
        arc(0, POOL[0], 0, 1, a=1.0, B=1e-4),
        arc(1, POOL[1], 0, 1, a=0.995, B=1e-4),          # takes a quarter, earns a sliver
    ]
    g = build(arcs, 2)
    got = circuit.candidates(g, arcs, np.ones(2), 0, 1, 100.0, leg_cost_bp=5.0)
    labels = [c.label for c in got.candidates]
    assert labels == ["circuit", "circuit, pruned"]
    plain, pruned = got.candidates
    assert plain.psi[1] > 0 and pruned.psi[1] == 0.0


def shared(k, pool, *, a, B, token_in="0xa", token_out="0xb", tau=0, sigma=1):
    """An arc on the node pair's own tokens, so no conversion is counted."""
    return PoolArc(id=f"{pool}:{k}", pool=pool, kind=ArcKind.SWAP_STABLE, i=0, j=1,
                   n_coins=2, token_in=token_in, token_out=token_out, tau=tau,
                   sigma=sigma, a=a, B=B, note=f"pool{k}")


def test_a_route_over_the_leg_limit_loses_its_weakest_ports():
    """`verify` refuses a route over the limit whole, so the circuit cuts it
    to fit, keeping the ports that earn most."""
    arcs = [shared(k, POOL[k], a=1.0 - 0.002 * k, B=1e-3) for k in range(6)]
    g = build(arcs, 2)
    free = circuit.candidates(g, arcs, np.ones(2), 0, 1, 100.0)
    assert np.count_nonzero(free.candidates[0].psi) == 6
    got = circuit.candidates(g, arcs, np.ones(2), 0, 1, 100.0, max_legs=3)
    assert got.candidates
    for c in got.candidates:
        assert np.count_nonzero(c.psi) <= 3
        if not c.label.startswith("circuit, top"):
            assert set(np.flatnonzero(c.psi)) == {0, 1, 2}


def test_the_pools_carrying_most_are_offered_alone():
    """The solve spreads a trade over every arc that pays at all: CRV->WETH
    $10k took 16 legs where three paid more.  The k pools it put most through,
    solved alone, are on the ballot too, and verify weighs their gas."""
    arcs = [shared(0, POOL[0], a=1.0, B=1e-5)]
    arcs += [shared(k, POOL[k], a=0.9999, B=1e-3) for k in range(1, 6)]
    g = build(arcs, 2)
    got = circuit.candidates(g, arcs, np.ones(2), 0, 1, 100.0)
    assert np.count_nonzero(got.candidates[0].psi) > 1
    by = {c.label: c for c in got.candidates}
    assert set(np.flatnonzero(by["circuit, top 1 pool"].psi)) == {0}


def test_a_pool_is_kept_with_every_port_it_has():
    """TriCRV fed crvUSD on to WETH; kept alone, it has to be free to sell
    straight to WETH instead, or the ballot's three-leg route is out of reach."""
    tri = POOL[0]
    arcs = [
        arc(0, tri, 0, 1, a=1.0, B=2e-3, kind=ArcKind.SWAP_CRYPTO, i=0, j=1, n=3),
        arc(1, tri, 0, 2, a=1.01, B=2e-3, kind=ArcKind.SWAP_CRYPTO, i=0, j=2, n=3),
        arc(2, POOL[1], 2, 1, a=1.0, B=1e-4),
    ]
    g = build(arcs, 3)
    got = circuit.candidates(g, arcs, np.ones(3), 0, 1, 100.0, advanceable=frozenset())
    assert got.candidates[0].psi[1] > 0 and got.candidates[0].psi[0] == 0.0
    by = {c.label: c for c in got.candidates}
    assert set(np.flatnonzero(by["circuit, top 1 pool"].psi)) == {0}


def test_pools_the_trade_does_not_fit_are_not_offered():
    """WETH->USDC $10k's compact candidates ran through a v3 bank reaching
    2.1 wstETH, and realisation put the whole 2.7 through it."""
    arcs = [shared(0, POOL[0], a=1.0, B=1e-5), shared(1, POOL[1], a=0.99, B=1e-3)]
    arcs[0].cap = 60.0
    arcs[1].cap = 45.0
    g = build(arcs, 2)
    got = circuit.candidates(g, arcs, np.ones(2), 0, 1, 100.0)
    assert got.candidates[0].psi[0] > got.candidates[0].psi[1] > 0
    assert "circuit, top 1 pool" not in [c.label for c in got.candidates]


def test_a_node_drawing_on_two_of_its_tokens_needs_a_conversion():
    arcs = [shared(0, POOL[0], a=1.0, B=1e-3),
            shared(1, POOL[1], a=1.0, B=1e-3, token_in="0xa2")]
    assert circuit._legs(arcs, np.array([1.0, 1.0])) == 3
    assert circuit._legs(arcs, np.array([1.0, 0.0])) == 1


def test_a_v2_pair_is_its_exact_curve_and_its_cap_does_not_bind():
    """`a d / (1 + k d)` with `k = B / 2a` is the pair itself at any size; the
    cap bounded only the tangent quadratic, which fell 1% short on CRV/WETH and
    kept the circuit off a pair a re-split then loaded 3.5x deeper."""
    pair = arc(0, POOL[0], 0, 1, a=2.0, B=2 * 2.0 * 0.01, cap=10.0, kind=ArcKind.SWAP_UNIV2)
    res, dev = solve([pair], 50.0)
    assert dev.exact[0] and dev.cap[0] > 50.0
    assert res.delta[0] == pytest.approx(50.0, rel=1e-8)
    assert res.out[0] == pytest.approx(2.0 * 50 / 1.5, rel=1e-8)


def test_a_v2_pair_reaches_as_far_as_a_bank():
    """Open-ended, a pair soaked up what the model could not place elsewhere
    -- CRV->WETH $10M lost 308 bp -- where a bank's reach lets it spill."""
    from erouter.core.cryptobank import DEPTH

    pair = PoolArc(id="pair", pool=POOL[0], kind=ArcKind.SWAP_UNIV2, i=0, j=1, n_coins=2,
                   token_in="0xa", token_out="0xb", tau=0, sigma=1, a=1.0, B=2e-3,
                   cap=2.5, reserve_in=10 * 10**18, decimals_in=18)
    dev = circuit.devices([pair], 2, np.ones(2), 1000.0)
    assert dev.cap[0] == pytest.approx(DEPTH * 10)
