"""The Rust circuit must hand `verify` the ballot `core/circuit.py` does.

The reference solves each Newton step densely and the port with a sparse
factorisation ordered once, so the two agree to rounding rather than bit for
bit: the same candidates, on the same arcs, carrying the same flows to 1e-6.
What would matter is a different support -- a pool kept by one side and
repaired away by the other -- and that is compared exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core import accel, circuit
from erouter.core.accel import available
from erouter.core.candidates import _from_ballot
from erouter.core.quoter import MAX_LEGS
from erouter.core.types import ArcKind, PoolArc
from test_candidates_differential import SEEDS, universe

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")


def both(g, arcs, nu, src, dst, Psi, **kw):
    want = circuit.candidates(g, arcs, nu, src, dst, Psi, **kw)
    gas = [0.0] * len(arcs)
    got = _from_ballot(accel.circuit(
        arcs, g.n_nodes, g.g_scale, nu, src, dst, Psi,
        advanceable=kw.get("advanceable"), leg_cost_bp=kw.get("leg_cost_bp", 0.0),
        per_gas=kw.get("per_gas", 0.0), gas=gas, max_legs=kw.get("max_legs", MAX_LEGS)))
    return want, got


def same(want, got):
    assert [c.label for c in got.candidates] == [c.label for c in want.candidates]
    for w, h in zip(want.candidates, got.candidates, strict=True):
        assert np.array_equal(w.psi > 0, h.psi > 0), (w.label, np.flatnonzero((w.psi > 0) != (h.psi > 0)))
        scale = max(float(w.psi.max()), 1e-300)
        assert np.allclose(h.psi, w.psi, rtol=1e-6, atol=1e-9 * scale), w.label


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("advanceable", [None, frozenset()])
def test_the_ballot_agrees(seed, advanceable):
    g, arcs = universe(seed)[:2]
    nu = np.ones(g.n_nodes)
    want, got = both(g, arcs, nu, 0, g.n_nodes - 1, 1e3 / g.g_scale,
                     advanceable=advanceable, leg_cost_bp=2.0)
    assert want.candidates
    same(want, got)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_route_is_cut_to_the_same_leg_budget(seed):
    """Which ports go is decided by surplus, so a port whose surplus the two
    sides round differently would be cut on one and kept on the other."""
    g, arcs = universe(seed)[:2]
    nu = np.ones(g.n_nodes)
    src, dst, Psi = 0, g.n_nodes - 1, 1e3 / g.g_scale
    free = circuit.candidates(g, arcs, nu, src, dst, Psi)
    budget = max(1, circuit._legs(arcs, free.candidates[0].psi) // 2)
    want, got = both(g, arcs, nu, src, dst, Psi, max_legs=budget)
    assert all(circuit._legs(arcs, c.psi) <= budget for c in want.candidates)
    same(want, got)


def test_a_tick_bank_is_solved_on_its_exact_curve_on_both_sides():
    """v3 pieces are hyperbolas, not quadratics; a port that got this wrong
    would still converge, to a different split."""
    from erouter.core import graph

    arcs = [PoolArc(id=f"t#{k}", pool="0x" + "cc" * 20, kind=ArcKind.SWAP_UNIV3,
                    i=0, j=1, n_coins=2, token_in="0xa", token_out="0xb", tau=0, sigma=1,
                    a=1.0 - 0.01 * k, B=2e-4, cap=300.0, parallel=True)
            for k in range(3)]
    arcs.append(PoolArc(id="plain", pool="0x" + "dd" * 20, kind=ArcKind.SWAP_STABLE,
                        i=0, j=1, n_coins=2, token_in="0xa", token_out="0xb",
                        tau=0, sigma=1, a=0.985, B=1e-4))
    tau = np.array([x.tau for x in arcs])
    sig = np.array([x.sigma for x in arcs])
    g = graph.build(tau, sig, np.array([x.a for x in arcs]), np.array([x.B for x in arcs]),
                    np.ones(2), 1.0, n_nodes=2, merge_duplicates=False)
    want, got = both(g, arcs, np.ones(2), 0, 1, 800.0 / g.g_scale)
    same(want, got)


def test_a_v2_pair_is_solved_on_its_exact_curve_on_both_sides():
    from erouter.core import graph

    # 80 raw at 1.1 reaches 4 x 88 = 352, short of the 800 traded: it binds.
    arcs = [PoolArc(id="pair", pool="0x" + "ee" * 20, kind=ArcKind.SWAP_UNIV2, i=0, j=1,
                    n_coins=2, token_in="0xa", token_out="0xb", tau=0, sigma=1,
                    a=0.99, B=2e-4, cap=100.0, reserve_in=80 * 10**18, decimals_in=18,
                    rate_in=1.1),
            PoolArc(id="plain", pool="0x" + "dd" * 20, kind=ArcKind.SWAP_STABLE, i=0, j=1,
                    n_coins=2, token_in="0xa", token_out="0xb", tau=0, sigma=1,
                    a=0.985, B=1e-4)]
    tau = np.array([x.tau for x in arcs])
    sig = np.array([x.sigma for x in arcs])
    g = graph.build(tau, sig, np.array([x.a for x in arcs]), np.array([x.B for x in arcs]),
                    np.ones(2), 1.0, n_nodes=2, merge_duplicates=False)
    want, got = both(g, arcs, np.ones(2), 0, 1, 800.0 / g.g_scale)
    assert 100.0 < want.candidates[0].psi[0] * g.g_scale <= 352.0 * (1 + 1e-6)
    same(want, got)
