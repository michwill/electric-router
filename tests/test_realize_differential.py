"""The Rust realisation must emit the leg list `core/realize.py` emits.

This is the first ported stage whose output is an *artefact* rather than a
number: an ordered list of legs that the on-chain router executes verbatim.
So the comparison is the whole list, leg for leg and field for field, and the
ordering rules are the point of the test rather than a side effect of it.

Three of those rules are load-bearing and each was written down after a
measured failure, so each gets a case here:

* **a capped arc must never sweep.**  `bps == 0` takes whatever is left, and
  handing that to an arc with a finite cap published a route that could not be
  run -- 9,960 USDC into a vault whose `maxDeposit` is 1,142.
* **one fill per spoke, not one per arc drawing on it.**  Four deposits at one
  ratio into one slot pay what a single deposit of the total pays, and spend
  three of the caller's legs for nothing.
* **two arcs drawing on the same spoke are one `bps` group.**  Giving each
  `bps = 0` puts two sweepers in one group and the second is left with nothing
  to trade.

A divergence in any of them is silent: the legs still typecheck, the route
still renders, and it reverts on chain or quotes the wrong split.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core.accel import available
from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.realize import (
    check_one_arc_per_pool,
    conversion_route,
    max_theta,
    prune_dust,
    realize,
    route_conductance,
    topological_nodes,
    total_loss_bp,
)
from erouter.core.types import ArcKind, PoolArc

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
CRVUSD = "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e"
SCRVUSD = "0x0655977feb2f289a4ab78af67bab0d17aab84367"

POOL_A = "0x" + "a1" * 20
POOL_B = "0x" + "b2" * 20
POOL_C = "0x" + "c3" * 20
POOL_D = "0x" + "d4" * 20


# --------------------------------------------------------------- fixtures


def build_nodes():
    """The same node map on both sides, built by the same calls."""
    import erouter_solve

    tokens = [(WETH, "WETH", 18), (USDC, "USDC", 6), (USDT, "USDT", 6),
              (CRVUSD, "crvUSD", 18)]
    merges = [
        ("NATIVE_WRAP", ETH, WETH, 1, 1, WETH),
        ("ERC4626", SCRVUSD, CRVUSD, 11 * 10**17, 10**18, SCRVUSD),
    ]
    reference, ported = NodeMap(), erouter_solve.NodeMap()
    for address, symbol, decimals in tokens:
        reference.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)
    for address, symbol, decimals in [(ETH, "ETH", 18), (SCRVUSD, "scrvUSD", 18)]:
        reference.add_token(address, symbol, decimals)
        ported.add_token(address, symbol, decimals)
    for kind, token, canonical, num, den, target in merges:
        reference.merge(Conversion(ConversionKind(kind), token, canonical,
                                   num, den, target=target))
        ported.merge(kind, token, canonical, str(num), str(den), target)
    return reference, ported


def make_arc(pool, token_in, token_out, nodes, *, i=0, j=1, a=1.0, B=1e-9,
             cap=math.inf, reserve=10**24, tvl=1e7, kind=ArcKind.SWAP_STABLE,
             decimals_in=18, note=""):
    return PoolArc(
        id=f"{pool[:6]}:{i}>{j}:{token_in[:6]}",
        pool=pool, kind=kind, i=i, j=j, n_coins=2,
        token_in=token_in, token_out=token_out,
        tau=nodes.node(token_in), sigma=nodes.node(token_out),
        a=a, B=B, cap=cap, G=1.0 / max(B, 1e-30), eps=1.0 - a,
        reserve_in=reserve, decimals_in=decimals_in,
        decimals_out=18, tvl_usd=tvl, gamma_live=a, note=note,
    )


def port_arcs(arcs):
    """The same arcs on the Rust side."""
    import erouter_solve

    built = erouter_solve.Arcs()
    for arc in arcs:
        built.add(arc.id, arc.pool, int(arc.kind), arc.i, arc.j, arc.n_coins,
                  arc.token_in, arc.token_out, arc.tau, arc.sigma,
                  arc.a, arc.B, arc.cap, arc.G, arc.eps, arc.reserve_in,
                  arc.decimals_in, arc.tvl_usd, arc.gamma_live, arc.note)
    return built


def both(arcs, psi, nu, reference_nodes, ported_nodes, *, src, dst,
         amount_in, potentials=None):
    import erouter_solve

    want = realize(arcs, np.asarray(psi, float), np.asarray(nu, float),
                   reference_nodes, src_token=src, dst_token=dst,
                   amount_in=amount_in,
                   potentials=None if potentials is None
                   else np.asarray(potentials, float))
    got = erouter_solve.Route.realize(
        port_arcs(arcs), [float(v) for v in psi], [float(v) for v in nu],
        ported_nodes, src, dst, str(amount_in),
        None if potentials is None else [float(v) for v in potentials])
    return want, got


def same_route(want, got):
    """Every field of every leg, plus everything the route carries."""
    assert len(got) == len(want.legs), (
        [(rl.target[:8], rl.leg.src_slot, rl.leg.dst_slot, rl.leg.bps)
         for rl in want.legs])

    wire = [(rl.leg.target, int(rl.leg.kind), rl.leg.i, rl.leg.j, rl.leg.n,
             rl.leg.src_slot, rl.leg.dst_slot, rl.leg.bps) for rl in want.legs]
    assert got.wire_legs() == wire

    assert got.targets() == [rl.target for rl in want.legs]
    # `Vec<u8>` crosses as `bytes`, which is a PyO3 convention rather than a
    # difference in the answer.
    assert list(got.kinds()) == [int(rl.kind) for rl in want.legs]
    assert got.tokens_in() == [rl.token_in for rl in want.legs]
    assert got.tokens_out() == [rl.token_out for rl in want.legs]
    assert [int(v) for v in got.amounts_in()] == [rl.amount_in for rl in want.legs]
    assert [int(v) for v in got.amounts_out()] == [rl.amount_out for rl in want.legs]
    assert [int(v) for v in got.reserves_in()] == [rl.reserve_in for rl in want.legs]
    assert got.arc_ids() == [rl.arc_id or "" for rl in want.legs]
    assert got.pool_names() == [rl.pool_name for rl in want.legs]
    assert got.modelled() == [rl.modelled for rl in want.legs]
    assert got.is_conversion() == [rl.is_conversion for rl in want.legs]
    assert got.is_merge() == [rl.is_merge for rl in want.legs]

    numbers = np.array(got.numbers()).reshape(-1, 8) if want.legs else np.zeros((0, 8))
    expected = np.array([[rl.share_of_node, rl.eps, rl.impact_frac, rl.theta,
                          rl.psi, rl.cap_in, rl.tvl_usd, rl.gamma_live]
                         for rl in want.legs]) if want.legs else np.zeros((0, 8))
    agree = (numbers == expected) | (np.isnan(numbers) & np.isnan(expected))
    assert agree.all(), (numbers[~agree], expected[~agree])

    assert dict(got.slots()) == want.slots
    assert dict(got.node_of_slot()) == want.node_of_slot
    assert got.dst_slot == want.dst_slot
    assert got.src_token == want.src_token
    assert got.dst_token == want.dst_token
    assert int(got.amount_in) == want.amount_in
    assert int(got.modelled_out) == want.modelled_out
    assert got.paths() == want.paths
    assert got.warnings() == want.warnings
    assert got.pools_used() == want.pools_used
    assert got.check_one_arc_per_pool() == check_one_arc_per_pool(want)
    assert got.max_theta() == max_theta(want)


# ---------------------------------------------------------------- topology


TOPOLOGIES = [
    ([0, 1], [1, 2], 3),
    ([0, 0, 1, 2], [1, 2, 3, 3], 4),
    ([0], [1], 2),
    ([1, 0], [2, 1], 3),
]


@pytest.mark.parametrize("tau,sig,n", TOPOLOGIES)
def test_topological_nodes_agree(tau, sig, n):
    import erouter_solve

    want = topological_nodes(np.array(tau), np.array(sig), n)
    assert erouter_solve.Route.topological_nodes(tau, sig, n) == want


def test_a_cycle_is_refused_on_both_sides():
    import erouter_solve

    from erouter.core.realize import RealizationError

    with pytest.raises(RealizationError, match="contain a cycle"):
        topological_nodes(np.array([0, 1]), np.array([1, 0]), 2)
    with pytest.raises(RuntimeError, match="contain a cycle"):
        erouter_solve.Route.topological_nodes([0, 1], [1, 0], 2)


PRUNINGS = [
    ([0, 0, 2], [1, 2, 3], [1.0, 1e-6, 1e-6], 0, 1),
    ([0, 0], [1, 1], [0.5, 0.5], 0, 1),
    ([0, 0, 1, 2], [1, 2, 3, 3], [0.9, 0.1, 0.9, 0.1], 0, 3),
    ([0, 0, 1, 2], [1, 2, 3, 3], [1.0, 1e-9, 1.0, 1e-9], 0, 3),
    ([0], [1], [1e-30], 0, 1),
    ([0, 1], [1, 2], [1.0, 1.0], 0, 2),
]


@pytest.mark.parametrize("tau,sig,psi,src,dst", PRUNINGS)
def test_prune_dust_agrees(tau, sig, psi, src, dst):
    import erouter_solve

    flow, removed = prune_dust(np.array(tau), np.array(sig), np.array(psi),
                               src, dst)
    got_flow, got_removed = erouter_solve.Route.prune_dust(tau, sig, psi, src, dst)
    assert got_flow == list(flow)
    assert got_removed == removed


# ----------------------------------------------------------- realisation


def test_a_plain_two_hop_route_agrees():
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    want, got = both(arcs, [1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=CRVUSD,
                     amount_in=10**18)
    same_route(want, got)


def test_a_split_across_two_pools_agrees():
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, WETH, USDC, reference, a=2999.0, B=2e-3),
        make_arc(POOL_C, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    want, got = both(arcs, [0.6, 0.4, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=CRVUSD,
                     amount_in=5 * 10**18)
    same_route(want, got)


def test_an_even_split_lands_on_the_same_bps():
    """`round(BPS * share / total)` sits exactly on a half here.

    Python rounds ties to even and Rust's `f64::round` does not, so two equal
    branches out of one node is where the two would first disagree.
    """
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, WETH, USDC, reference, a=3000.0, B=1e-3),
    ]
    want, got = both(arcs, [1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=USDC,
                     amount_in=2 * 10**18)
    same_route(want, got)
    assert [rl.leg.bps for rl in want.legs] == [5000, 0]


def test_a_merged_node_emits_the_same_conversion_legs():
    """ETH and WETH are one node; a pool holds one or the other."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, ETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    want, got = both(arcs, [1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=CRVUSD,
                     amount_in=10**18)
    same_route(want, got)
    assert any(rl.is_conversion for rl in want.legs)


def test_the_destination_tail_agrees():
    """A route asked for scrvUSD ends in a deposit out of the crvUSD hub."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    want, got = both(arcs, [1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=SCRVUSD,
                     amount_in=10**18)
    same_route(want, got)
    assert want.legs[-1].kind is ArcKind.ERC4626_DEPOSIT


def test_a_capped_arc_never_sweeps_on_either_side():
    """The USD3 measurement: `bps == 0` past a cap is a route that reverts."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        # Capped, and emitted last -- so only the ordering rule keeps it from
        # sweeping.
        make_arc(POOL_B, WETH, USDC, reference, a=2900.0, B=2e-3, cap=5e-5),
    ]
    want, got = both(arcs, [0.9, 0.1], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=USDC,
                     amount_in=10**18)
    same_route(want, got)
    capped = [rl for rl in want.legs if math.isfinite(rl.cap_in)]
    assert capped and all(rl.leg.bps != 0 for rl in capped)


def test_one_fill_per_spoke_on_both_sides():
    """Four arcs wanting scrvUSD share one crvUSD -> scrvUSD deposit."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, USDC, CRVUSD, reference, a=1.0, B=1e-9),
        *[make_arc(pool, SCRVUSD, USDT, reference, a=1.0, B=1e-9)
          for pool in (POOL_B, POOL_C, POOL_D)],
    ]
    want, got = both(arcs, [3.0, 1.0, 1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=USDC, dst=USDT,
                     amount_in=3 * 10**6)
    same_route(want, got)
    fills = [rl for rl in want.legs if rl.kind is ArcKind.ERC4626_DEPOSIT]
    assert len(fills) == 1, "one fill per spoke, not one per arc"


def test_two_arcs_on_one_spoke_are_one_bps_group():
    """Two sweepers in one group leaves the second with nothing to trade."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, USDC, CRVUSD, reference, a=1.0, B=1e-9),
        make_arc(POOL_B, SCRVUSD, USDT, reference, a=1.0, B=1e-9),
        make_arc(POOL_C, SCRVUSD, USDT, reference, a=0.999, B=2e-9),
    ]
    want, got = both(arcs, [2.0, 1.0, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=USDC, dst=USDT,
                     amount_in=2 * 10**6)
    same_route(want, got)
    spoke = [rl for rl in want.legs if rl.token_in.lower() == SCRVUSD]
    assert sum(1 for rl in spoke if rl.leg.bps == 0) == 1


def test_potentials_come_back_the_same():
    reference, ported = build_nodes()
    arcs = [make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3)]
    potentials = np.arange(reference.n_nodes, dtype=float) * 0.5
    want, got = both(arcs, [1.0], np.ones(reference.n_nodes), reference, ported,
                     src=WETH, dst=USDC, amount_in=10**18, potentials=potentials)
    same_route(want, got)
    assert dict(got.potentials()) == want.potentials


def test_a_non_unit_nu_agrees():
    """`delta = psi / nu[tau]` is where value becomes token units."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    nu = np.array([2600.0, 1.0, 1.0, 1.0, 2600.0, 1.0][:reference.n_nodes])
    want, got = both(arcs, [1500.0, 1500.0], nu, reference, ported,
                     src=WETH, dst=CRVUSD, amount_in=10**18)
    same_route(want, got)


def test_no_arcs_is_refused_on_both_sides():
    import erouter_solve

    from erouter.core.realize import RealizationError

    reference, ported = build_nodes()
    with pytest.raises(RealizationError, match="no arcs carry flow"):
        realize([], np.array([]), np.ones(reference.n_nodes), reference,
                src_token=WETH, dst_token=USDC, amount_in=10**18)
    with pytest.raises(RuntimeError, match="no arcs carry flow"):
        erouter_solve.Route.realize(erouter_solve.Arcs(), [], [1.0], ported,
                                    WETH, USDC, "1")


# ------------------------------------------------------ conversion routes


@pytest.mark.parametrize("src,dst", [
    (CRVUSD, SCRVUSD), (SCRVUSD, CRVUSD), (ETH, WETH), (WETH, ETH),
])
def test_conversion_routes_agree(src, dst):
    import erouter_solve

    reference, ported = build_nodes()
    want = conversion_route(reference, src_token=src, dst_token=dst,
                            amount_in=10**18)
    got = erouter_solve.Route.conversion_route(ported, src, dst, str(10**18))
    same_route(want, got)


# ------------------------------------------------------------- the scout


def test_route_conductance_agrees():
    """Solved with a different LU than numpy's, so this is not bit-exact."""
    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3, tvl=5e7),
        make_arc(POOL_B, WETH, USDC, reference, a=2999.0, B=2e-3, tvl=2e7),
        make_arc(POOL_C, USDC, CRVUSD, reference, a=1.0, B=1e-9, tvl=9e7),
    ]
    want, got = both(arcs, [0.6, 0.4, 1.0], np.ones(reference.n_nodes),
                     reference, ported, src=WETH, dst=CRVUSD,
                     amount_in=5 * 10**18)
    expected = route_conductance(want)
    assert got.route_conductance() == pytest.approx(expected, rel=1e-12)


def test_total_loss_bp_agrees():
    reference, ported = build_nodes()
    arcs = [make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3)]
    want, got = both(arcs, [1.0], np.ones(reference.n_nodes), reference, ported,
                     src=WETH, dst=USDC, amount_in=10**18)
    for price in (3000e-12, 1.0, 0.0, -1.0):
        one, two = got.total_loss_bp(price), total_loss_bp(want, price)
        assert (one == two) or (math.isnan(one) and math.isnan(two)), price
