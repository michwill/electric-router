"""The four pricing modules, mirrored: curves, prices, slippage, refit.

These are the tables a route is priced *against*, and each one fails quietly
in its own way.

`curves` is the interpolant the split search runs over -- millions of
evaluations, so a divergence in `at` shows up as a different split rather than
as an error.  Its tail rule is the interesting part: a final secant with
increasing returns would drive `u` through zero and `f = x/u` off a cliff, so
it is held flat instead.

`prices` fits `nu`, which sets `eps` and `G` for *every* arc -- so one arc
lying about a price drags the whole frame, and `price_fit_weights` is what
mutes it.  Pairing on node indices instead of on the pool is wrong and quietly
so: a dozen pools join USDC and USDT.

`slippage` divides a budget across a resistor network.  Get it wrong and a
route ships with a minimum-out that either reverts on any movement or protects
nothing.

`refit` re-anchors `B` at the realised size, and its two floors exist because
without them a refit at 3 USDC replaced a fit made at a million and clamped the
best pool for the pair to a cap of 3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import curves, prices, slippage
from erouter.core import refit as refit_mod
from erouter.core.accel import available
from erouter.core.nodes import NodeMap
from erouter.core.realize import realize
from erouter.core.types import ArcKind, PoolArc

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")


def native():
    import erouter_solve

    return erouter_solve


# ------------------------------------------------------------- the curves


LADDERS = [
    ([1.0, 2.0, 4.0, 8.0], [0.99, 1.96, 3.88, 7.60]),
    ([1e3, 1e4, 1e5, 1e6], [999.0, 9980.0, 99000.0, 940000.0]),
    ([1.0, 2.0], [1.0, 1.999]),
    # Increasing returns in the last secant, which the tail must not follow.
    ([1.0, 2.0, 4.0], [0.9, 1.85, 3.9]),
    ([1e-6, 1e-5, 1e-4], [0.9e-6, 0.99e-5, 0.999e-4]),
]


@pytest.mark.parametrize("deltas,quotes", LADDERS,
                         ids=[f"n{len(d)}" for d, _ in LADDERS])
def test_a_fitted_curve_evaluates_the_same_everywhere(deltas, quotes):
    want = curves.fit(np.array(deltas), np.array(quotes))
    got = native().Curve.fit(deltas, quotes)

    assert got.x == list(want.x)
    assert got.u == list(want.u)
    assert got.slope == list(want.slope)
    assert got.rate0 == want.rate0
    assert got.tail == want.tail
    assert got.top == want.top

    # Across the probes, between them, below the first and past the last --
    # every branch of `at`, including the one the tail rule guards.
    probes = np.concatenate([
        np.array(deltas),
        np.array(deltas[:-1]) * 1.5,
        [deltas[0] * 0.1, deltas[0] * 0.5, deltas[-1] * 2, deltas[-1] * 1e3, 0.0, -1.0],
    ])
    assert got.many([float(v) for v in probes]) == [want.at(float(v)) for v in probes]
    for v in probes:
        assert got.at(float(v)) == want.at(float(v))


@pytest.mark.parametrize("deltas,quotes", LADDERS,
                         ids=[f"n{len(d)}" for d, _ in LADDERS])
def test_the_error_estimate_agrees(deltas, quotes):
    want = curves.fit(np.array(deltas), np.array(quotes))
    got = native().Curve.fit(deltas, quotes)
    for v in [*deltas, deltas[0] * 0.5, deltas[-1] * 2, (deltas[0] + deltas[1]) / 2]:
        one, two = got.error_bp_at(float(v)), want.error_bp_at(float(v))
        assert one == two or (math.isinf(one) and math.isinf(two)), v


@pytest.mark.parametrize("rate", [1.0, 0.9996, 3210.5, 1e-9])
def test_a_linear_curve_agrees(rate):
    want, got = curves.linear(rate), native().Curve.linear(rate)
    for v in (0.0, 1.0, 1e6, 1e-9):
        assert got.at(v) == want.at(v)


def test_a_fit_refuses_the_same_ladders():
    bad = [
        ([1.0], [1.0]),
        ([1.0, 1.0], [1.0, 1.0]),
        ([2.0, 1.0], [1.0, 1.0]),
        ([1.0, 2.0], [1.0, 0.0]),
        ([-1.0, 2.0], [1.0, 1.0]),
    ]
    for deltas, quotes in bad:
        with pytest.raises(curves.CurveError):
            curves.fit(np.array(deltas), np.array(quotes))
        with pytest.raises(ValueError):
            native().Curve.fit(deltas, quotes)
    with pytest.raises(curves.CurveError):
        curves.linear(0.0)
    with pytest.raises(ValueError):
        native().Curve.linear(0.0)


@pytest.mark.parametrize("top", [0.0, 1.0, 2.0, 100.0, 1e6, 1e18, 12345.678])
def test_the_probe_ladder_agrees(top):
    """A node that rounds onto its predecessor is a zero denominator.

    Sizes are compared relatively rather than exactly, and only here.
    `np.geomspace` raises its interior through `np.power`, a vectorised loop
    that is not correctly rounded -- measured, two nodes in twenty-four land
    one ULP from libm's `pow`.  That is 1.8e-16 on a probe of 1.8e17: below
    the pool's own integer quantum, and below what the fit reading the ladder
    can resolve.  The *shape* -- how many nodes, strictly increasing, ending
    on `top` -- is compared exactly, because that is what the rule is about.
    """
    for nodes in (2, 5, 24):
        for span in (10.0, 4096.0):
            got = native().Curve.sizes(top, nodes, span)
            want = curves.sizes(top, nodes=nodes, span=span)
            assert len(got) == len(want), (got, want)
            assert got == sorted(set(got)), "strictly increasing"
            if want:
                assert got[-1] == want[-1], "the top is exact on both sides"
            assert np.allclose(got, want, rtol=1e-15, atol=0), (got, want)
    assert len(native().Curve.sizes(top)) == len(curves.sizes(top))


# ------------------------------------------------------------- the frame


def price_universe(seed: int):
    rng = np.random.default_rng(seed)
    n = 6
    tau, sig, a, keys = [], [], [], []
    for tail in range(n):
        for head in range(n):
            if tail == head or rng.random() < 0.4:
                continue
            pool = f"0xp{tail}{head}"
            tau.append(tail)
            sig.append(head)
            a.append(float(rng.uniform(0.9, 1.1)))
            keys.append((pool, tail, head))
            # And the reverse, so the round-trip rule has pairs to read.
            tau.append(head)
            sig.append(tail)
            a.append(float(1.0 / a[-1] * rng.uniform(0.998, 0.9999)))
            keys.append((pool, head, tail))
    return (np.array(tau, np.int64), np.array(sig, np.int64), np.array(a),
            keys, n)


@pytest.mark.parametrize("seed", range(8))
def test_the_reference_frame_agrees(seed):
    tau, sig, a, _, n = price_universe(seed)
    w = np.ones(len(a))
    want = prices.reference_prices(tau, sig, a, w, n, 0)
    got = native().reference_prices([int(v) for v in tau], [int(v) for v in sig],
                                    [float(v) for v in a], [float(v) for v in w], n, 0)
    # A different LU than numpy's, so this is not bit-exact -- see
    # `test_realize_differential`. The frame is a ratio, and what matters is
    # that every price agrees to far more digits than a quote can carry.
    assert np.allclose(got, want, rtol=1e-9), (np.array(got) / want - 1.0)


@pytest.mark.parametrize("seed", range(8))
def test_a_contradicted_arc_is_muted_the_same_way(seed):
    _, _, a, keys, _ = price_universe(seed)
    # Break one pair badly, the way a banded LLAMMA market reads.
    a = a.copy()
    a[0], a[1] = 0.000616, 0.002516
    w = np.ones(len(a))
    want = prices.price_fit_weights(keys, a, w)
    got = native().price_fit_weights(keys, [float(v) for v in a],
                                     [float(v) for v in w])
    assert got == list(want)
    assert want[0] == prices.MUTED_WEIGHT, "the fixture is supposed to mute it"


def test_a_healthy_arc_is_not_muted_for_its_neighbours_sins():
    """A dozen pools join USDC and USDT; pairing on nodes would mute them all."""
    keys = [("0xllamma", 0, 1), ("0xhealthy", 1, 0)]
    a = np.array([0.000616, 0.9994])
    w = np.ones(2)
    want = prices.price_fit_weights(keys, a, w)
    got = native().price_fit_weights(keys, [float(v) for v in a], [1.0, 1.0])
    assert got == list(want) == [1.0, 1.0]


def test_the_frame_refuses_the_same_inputs():
    with pytest.raises(ValueError, match="need a > 0"):
        prices.reference_prices(np.array([0]), np.array([1]), np.array([0.0]),
                                np.array([1.0]), 2, 0)
    with pytest.raises(ValueError, match="need a > 0"):
        native().reference_prices([0], [1], [0.0], [1.0], 2, 0)
    with pytest.raises(ValueError, match="weights must be positive"):
        prices.reference_prices(np.array([0]), np.array([1]), np.array([1.0]),
                                np.array([0.0]), 2, 0)
    with pytest.raises(ValueError, match="weights must be positive"):
        native().reference_prices([0], [1], [1.0], [0.0], 2, 0)
    # No arcs is parity, not an error.
    assert native().reference_prices([], [], [], [], 3, 0) == \
        list(prices.reference_prices(np.array([], np.int64), np.array([], np.int64),
                                     np.array([]), np.array([]), 3, 0))


@pytest.mark.parametrize("seed", range(8))
def test_dislocations_and_the_round_trip_readings_agree(seed):
    tau, sig, a, _, n = price_universe(seed)
    nu = prices.reference_prices(tau, sig, a, np.ones(len(a)), n, 0)
    want = prices.dislocations(tau, sig, a, nu)
    got = native().dislocations([int(v) for v in tau], [int(v) for v in sig],
                                [float(v) for v in a], [float(v) for v in nu])
    assert np.allclose(got, want, rtol=0, atol=1e-12)

    forward, reverse = a[0::2], a[1::2]
    assert native().gamma_live([float(v) for v in forward],
                               [float(v) for v in reverse]) == \
        list(prices.gamma_live(forward, reverse))
    for f, r in zip(forward, reverse, strict=True):
        assert native().pool_mid(float(f), float(r)) == prices.pool_mid(f, r)


@pytest.mark.parametrize("seed", range(6))
def test_check_pair_drops_agrees(seed):
    rng = np.random.default_rng(seed)
    forward = rng.normal(0.0, 0.01, 30)
    reverse = rng.normal(0.0, 0.01, 30)
    for tol in (0.0, -1e-6, 1e-3):
        assert native().check_pair_drops([float(v) for v in forward],
                                         [float(v) for v in reverse], tol) == \
            [int(v) for v in prices.check_pair_drops(forward, reverse, tol)]


# ---------------------------------------------------------- the slippage


TOKENS = ["0x" + f"{k:02x}" * 20 for k in range(6)]


def slippage_route(hops, dst_slot):
    """A realised route on both sides, over the given slot hops."""
    nodes, ported = NodeMap(), native().NodeMap()
    for k, address in enumerate(TOKENS):
        nodes.add_token(address, f"T{k}", 18)
        ported.add_token(address, f"T{k}", 18)

    arcs, flows = [], []
    for k, (src, dst) in enumerate(hops):
        arcs.append(PoolArc(
            id=f"a{k}", pool=f"0xpool{k}", kind=ArcKind.SWAP_STABLE, i=0, j=1,
            n_coins=2, token_in=TOKENS[src], token_out=TOKENS[dst],
            tau=nodes.node(TOKENS[src]), sigma=nodes.node(TOKENS[dst]),
            a=0.999, B=1e-6, G=1e6, eps=0.001,
            reserve_in=10**24, decimals_in=18, decimals_out=18, tvl_usd=1e7,
        ))
        flows.append(1.0)

    nu = np.ones(nodes.n_nodes)
    want = realize(arcs, np.array(flows), nu, nodes,
                   src_token=TOKENS[hops[0][0]], dst_token=TOKENS[dst_slot],
                   amount_in=10**18)
    built = native().Arcs()
    for a in arcs:
        built.add(a.id, a.pool, int(a.kind), a.i, a.j, a.n_coins, a.token_in,
                  a.token_out, a.tau, a.sigma, a.a, a.B, a.cap, a.G, a.eps,
                  a.reserve_in, a.decimals_in, a.tvl_usd, a.gamma_live, a.note)
    got = native().Route.realize(built, flows, [float(v) for v in nu], ported,
                                 TOKENS[hops[0][0]], TOKENS[dst_slot],
                                 str(10**18), None)
    return want, got


SHAPES = [
    ([(0, 1)], 1),
    ([(0, 1), (1, 2)], 2),
    ([(0, 1), (0, 1)], 1),
    ([(0, 1), (0, 2), (1, 3), (2, 3)], 3),
    ([(0, 1), (1, 2), (2, 3), (0, 3)], 3),
]


@pytest.mark.parametrize("hops,dst", SHAPES, ids=[f"{len(h)}legs" for h, _ in SHAPES])
@pytest.mark.parametrize("total", [0.0, 1e-4, 0.005, 1.0])
def test_the_budget_divides_the_same_way(hops, dst, total):
    want, got = slippage_route(hops, dst)
    resistance = [1.0 + 0.5 * k for k in range(len(hops))]

    raw_want = slippage.drops(want, resistance, total)
    raw_got = native().drops(got, resistance, total)
    if raw_want is None:
        assert raw_got is None
    else:
        # A different LU than numpy's; the network is three or four nodes, so
        # the two agree to far more than a basis point.
        assert np.allclose(raw_got, raw_want, rtol=1e-12, atol=1e-15)

    share_want = slippage.divide(want, resistance, total)
    share_got = native().divide(got, resistance, total)
    assert np.allclose(share_got, share_want, rtol=1e-9, atol=1e-15)
    assert native().longest(got, share_got) == pytest.approx(
        slippage.longest(want, list(share_want)), rel=1e-9, abs=1e-15)


def test_a_backwards_leg_is_held_at_its_floor_on_both_sides():
    raw = [1.0, -4.0, 2.0]
    assert native().backstops(raw) == list(slippage.backstops(raw, None))
    floor = [0.0, 9.0, 0.0]
    assert native().backstops(raw, floor) == list(slippage.backstops(raw, floor))


def test_a_negative_budget_is_refused_on_both_sides():
    want, got = slippage_route([(0, 1)], 1)
    with pytest.raises(ValueError, match="cannot be negative"):
        slippage.divide(want, [1.0], -1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        native().divide(got, [1.0], -1.0)


@pytest.mark.parametrize("hops,dst", SHAPES, ids=[f"{len(h)}legs" for h, _ in SHAPES])
def test_widen_moves_the_same_legs(hops, dst):
    want, got = slippage_route(hops, dst)
    resistance = [1.0] * len(hops)
    spend = [0.001 * (k + 1) for k in range(len(hops))]
    assert native().widen(got, resistance, 0.01, spend, 0.05) == \
        list(slippage.widen(want, resistance, 0.01, spend, 0.05))


# ------------------------------------------------------------- the refit


class FakeQuoter:
    """Answers the two probes a refit asks for, from a supplied curve."""

    def __init__(self, f):
        self.f = f
        self.seen: list = []

    def probe(self, probes):
        from erouter.core.transport import Status

        class Answer:
            def __init__(self, ok, value):
                self.ok, self.value = ok, value

        self.seen = list(probes)
        assert Status is not None
        return [Answer(*self.f(p.dx)) for p in probes]


def refit_arcs_fixture(a=1.0, B=1e-6, calib=0.0):
    nodes, ported = NodeMap(), native().NodeMap()
    nodes.add_token(TOKENS[0], "IN", 18)
    nodes.add_token(TOKENS[1], "OUT", 18)
    ported.add_token(TOKENS[0], "IN", 18)
    ported.add_token(TOKENS[1], "OUT", 18)
    arc = PoolArc(
        id="a0", pool="0xpool", kind=ArcKind.SWAP_STABLE, i=0, j=1, n_coins=2,
        token_in=TOKENS[0], token_out=TOKENS[1], tau=0, sigma=1,
        a=a, B=B, calib_delta=calib,
        reserve_in=10**24, decimals_in=18, decimals_out=18,
    )
    return nodes, ported, [arc]


def ported_of(arcs):
    built = native().Arcs()
    for a in arcs:
        built.add(a.id, a.pool, int(a.kind), a.i, a.j, a.n_coins, a.token_in,
                  a.token_out, a.tau, a.sigma, a.a, a.B, a.cap, a.G, a.eps,
                  a.reserve_in, a.decimals_in, a.tvl_usd, a.gamma_live, a.note,
                  a.calib_delta, a.decimals_out)
        built_last = a
    assert built_last is not None
    return built


CASES = [
    # (a, B, calib_delta, psi, quote scale) -- measurable, unmeasurable, and
    # increasing returns.
    (1.0, 1e-30, 1.0, 1000.0, 0.99),
    (1.0, 4.4e-11, 3.9e6, 0.4, 0.99),        # far below the existing fit
    (1.0, 4.4e-11, 1.0, 1000.0, 1.0),        # nothing for the secant to see
    (1.0, 1e-6, 1.0, 1000.0, 0.995),
]


@pytest.mark.parametrize("a,B,calib,psi,scale", CASES,
                         ids=["measurable", "too-small", "flat", "mild"])
def test_the_refit_plans_and_applies_the_same(a, B, calib, psi, scale):
    nodes, ported, arcs = refit_arcs_fixture(a, B, calib)
    nu = np.array([1.0, 1.0])

    def quote(dx):
        return True, int(dx * scale)

    mine = [PoolArc(**{f.name: getattr(x, f.name)
                       for f in x.__dataclass_fields__.values()}) for x in arcs]
    client = FakeQuoter(quote)
    want = refit_mod.refit_arcs(
        None, mine, np.array([psi]), nu, client,
        rate_in=lambda arc: nodes.rate(arc.token_in),
        rate_out=lambda arc: nodes.rate(arc.token_out),
    )

    planned = native().Refit.plan(ported_of(arcs), [psi], [1.0, 1.0], ported)
    probes = planned.probes()
    assert [p[5] for p in probes] == [p.dx for p in client.seen]
    answers = [quote(dx) for _, _, _, _, _, dx in probes]
    got = planned.apply(answers, ported)

    assert got == want
    numbers = np.array(planned.arc_numbers()).reshape(-1, 4)
    expected = np.array([[x.a, x.B, x.cap, x.calib_delta] for x in mine])
    agree = (numbers == expected) | (np.isnan(numbers) & np.isnan(expected))
    assert agree.all(), (numbers[~agree], expected[~agree])
    assert planned.arc_flags() == [(x.clamped, x.convex_flag) for x in mine]


def test_increasing_returns_are_clamped_on_both_sides():
    nodes, ported, arcs = refit_arcs_fixture(a=1.0, B=1e-6, calib=1.0)
    nu = np.array([1.0, 1.0])
    unit = 10**18

    def quote(dx):
        # The bumped quote pays more per unit than `a`: a rising marginal rate.
        return (True, 990 * unit) if dx == 1000 * unit else (True, 1005 * unit)

    mine = [PoolArc(**{f.name: getattr(x, f.name)
                       for f in x.__dataclass_fields__.values()}) for x in arcs]
    want = refit_mod.refit_arcs(
        None, mine, np.array([1000.0]), nu, FakeQuoter(quote),
        rate_in=lambda arc: nodes.rate(arc.token_in),
        rate_out=lambda arc: nodes.rate(arc.token_out),
    )
    planned = native().Refit.plan(ported_of(arcs), [1000.0], [1.0, 1.0], ported)
    answers = [quote(dx) for _, _, _, _, _, dx in planned.probes()]
    assert planned.apply(answers, ported) == want == (1, 1, 0)
    assert mine[0].B == 0.0 and mine[0].clamped
    assert planned.arc_flags() == [(True, True)]


def test_a_refused_probe_leaves_its_arc_untouched_on_both_sides():
    _, ported, arcs = refit_arcs_fixture(a=1.0, B=1e-6, calib=1.0)
    planned = native().Refit.plan(ported_of(arcs), [1000.0], [1.0, 1.0], ported)
    answers = [(False, 0)] * len(planned.probes())
    assert planned.apply(answers, ported) == (0, 0, 0)
    assert np.array(planned.arc_numbers()).reshape(-1, 4)[0][1] == 1e-6


@pytest.mark.parametrize("psi_total", [0.0, 1.0, 1e6])
def test_the_convergence_test_agrees(psi_total):
    before = np.array([1.0, 2.0, 3.0])
    for after in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.00001], [1.0, 2.0, 9.0]):
        moved = (float(np.max(np.abs(np.array(after) - before)) / psi_total)
                 if psi_total > 0 else 0.0)
        assert native().round_stats(list(before), after, psi_total) == \
            (moved, moved < refit_mod.CONVERGED)

    want = float(np.max(np.abs(np.array([1.0, 2.0, 9.0]) - before)
                        / np.maximum(np.abs(before), 1e-30)))
    assert native().b_change(list(before), [1.0, 2.0, 9.0]) == want
