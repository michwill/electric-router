"""Pricing a node that only a venue reaches.

`Univ2`/`Univ3` used to require *both* coins of a pair to be priced by the
frame, which is a sufficient guard against `nu` defaulting to 1.0 and not a
necessary one.  On ethereum it discarded 6,416 of 6,452 census pairs and
$617.8M of the $697.7M they hold: an X/USDT pair was dropped even though USDT
is exactly the bridge that makes X routable as `X -> USDT -> crvUSD`.

The fix is to price X rather than refuse it.  §4 minimises
`w (z_tau - z_sig - log a)^2` over `z = log nu`, so these tests pin that the
derived price is what that objective wants, and that nothing Curve priced moves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core.pipeline import admissible_late_arcs
from erouter.core.types import ArcKind, PoolArc


def _arc(id_, tau, sigma, venue="", a=1.0, tvl=1_000.0):
    return PoolArc(
        id=id_, pool="0x" + id_.replace(":", "").ljust(40, "0")[:40],
        kind=ArcKind.SWAP_STABLE, i=0, j=1, n_coins=2,
        token_in="0xin", token_out="0xout", tau=tau, sigma=sigma,
        a=a, B=1.0, venue=venue, tvl_usd=tvl,
    )


def test_a_node_only_the_venue_reaches_is_priced_from_its_arcs():
    """The whole point: X is one bridge from routable, so route it."""
    frame = [_arc("curve:0", 0, 1)]              # nodes 0 and 1 are priced
    nu = np.array([1.0, 2.0, 1.0])              # node 2 is the default
    bridge = _arc("v2:0", 2, 1, "uniswap v2", a=3.0)   # X -> node 1
    fresh, refused, out, bridged = admissible_late_arcs(frame, [bridge], nu, bridgeable=(2,))
    assert [a.id for a in fresh] == ["v2:0"], "the bridge arc is admitted"
    assert refused == 0 and bridged == 1
    # §4 wants nu_tau = a * nu_sig for an arc it fits perfectly.
    assert out[2] == pytest.approx(3.0 * 2.0)


def test_both_directions_land_the_arc_on_the_pool_s_own_fee():
    """One direction alone would price the arc as lossless.

    Fitting `nu_tau = a nu_sig` from a single arc gives it `eps = 0` -- a free
    edge into a pool that charges 30 bp.  With both directions the price lands
    at `sqrt(a_f / a_r)` and each arc gets `eps = 1 - sqrt(a_f a_r)`.
    """
    fee = 0.003
    spot = 5.0
    forward = _arc("v2:f", 2, 1, "uniswap v2", a=spot * (1 - fee))
    reverse = _arc("v2:r", 1, 2, "uniswap v2", a=(1 / spot) * (1 - fee))
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    _fresh, _refused, out, _b = admissible_late_arcs(
        frame, [forward, reverse], nu, bridgeable=(2,))
    assert out[2] == pytest.approx(spot, rel=1e-12), "the mid price, not a side"
    for arc in (forward, reverse):
        eps = 1 - arc.a * out[arc.sigma] / out[arc.tau]
        assert eps == pytest.approx(fee, rel=1e-9), "the pool's own fee"


def test_nothing_the_frame_priced_is_moved():
    """`late_arcs` exists so a venue cannot outvote Curve on a shared price.

    Bridging must not smuggle that back in: the new node is solved for with
    every priced node held fixed.
    """
    frame = [_arc("curve:0", 0, 1), _arc("curve:1", 1, 2)]
    nu = np.array([1.0, 7.0, 13.0, 1.0])
    before = nu.copy()
    _f, _r, out, _b = admissible_late_arcs(
        frame, [_arc("v2:0", 3, 1, "uniswap v2", a=2.0)], nu, bridgeable=(3,))
    assert np.array_equal(out[:3], before[:3]), "Curve's prices are untouched"
    assert out[3] != 1.0, "and the new node is no longer the default"
    assert nu is not out, "the caller's array is not mutated in place"


def test_an_arc_with_neither_end_priced_is_still_refused():
    """There is no anchor to price it against, and this does one round."""
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0, 1.0])
    fresh, refused, _out, bridged = admissible_late_arcs(
        frame, [_arc("v2:0", 2, 3, "uniswap v2")], nu, bridgeable=(2, 3))
    assert fresh == [] and refused == 1 and bridged == 0


def test_a_failed_probe_prices_nothing():
    """`a = 0` would put `log 0` into the fit and NaN the whole frame.

    6% of arcs fail their smallest probe on mainnet, which is why
    `reference_prices` refuses a zero rather than treating it as a cheap pool.
    """
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    fresh, refused, out, bridged = admissible_late_arcs(
        frame, [_arc("v2:0", 2, 1, "uniswap v2", a=0.0)], nu, bridgeable=(2,))
    assert bridged == 0 and fresh == [] and refused == 1
    assert np.isfinite(out).all()


def test_the_deeper_pool_wins_when_two_bridges_disagree():
    """Two venues pricing the same new token is a weighted fit, not a race."""
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    thin = _arc("v2:0", 2, 1, "uniswap v2", a=1.0, tvl=1.0)
    deep = _arc("v3:0", 2, 1, "uniswap v3", a=100.0, tvl=1e9)
    _f, _r, out, _b = admissible_late_arcs(
        frame, [thin, deep], nu, bridgeable=(2,))
    assert out[2] > 50.0, "the $1e9 pool should dominate a $1 one"
    # Exactly the weighted mean of the logs the two arcs ask for.
    want = math.exp((1.0 * math.log(1.0) + 1e9 * math.log(100.0)) / (1.0 + 1e9))
    assert out[2] == pytest.approx(want)


def test_the_old_two_value_call_still_works():
    """Callers with no frame prices to hand cannot bridge anything anyway."""
    frame = [_arc("curve:0", 0, 1)]
    fresh, refused = admissible_late_arcs(
        frame, [_arc("v2:0", 0, 1, "uniswap v2"), _arc("v2:1", 0, 9, "x")])
    assert [a.id for a in fresh] == ["v2:0"] and refused == 1


def test_a_pool_s_tick_arcs_price_from_the_top_of_the_book():
    """32 tick arcs are one curve cut up, not 32 observations of a price.

    Every piece past the first is worse than spot by construction, so the mean
    of their logs sits below the market -- which leaves the near ticks looking
    profitable and the far ones dreadful.  Measured on `PEPE -> crvUSD`, the
    solver smeared the trade across the book and the modelled route needed 197
    legs against a 32-leg ceiling.
    """
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    spot = 100.0
    book = [
        PoolArc(
            id=f"v3:{n}", pool="0x" + "ab" * 20, kind=ArcKind.SWAP_UNIV3,
            i=0, j=1, n_coins=2, token_in="0xin", token_out="0xout",
            tau=2, sigma=1, a=spot / (1.5 ** n), B=1.0,
            venue="uniswap v3", tvl_usd=1_000.0, parallel=True,
        )
        for n in range(8)
    ]
    _f, _r, out, bridged = admissible_late_arcs(
        frame, book, nu, bridgeable=(2,))
    assert bridged == 1
    assert out[2] == pytest.approx(spot), "the marginal price, not the mean"
    assert out[2] > spot / 1.5, "and well above what averaging the book gives"


def test_two_pools_still_both_vote_for_the_same_token():
    """Collapsing to the top of the book is per pool, not per token."""
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    a = PoolArc(
        id="v3:a", pool="0x" + "aa" * 20, kind=ArcKind.SWAP_UNIV3, i=0, j=1,
        n_coins=2, token_in="0xin", token_out="0xout", tau=2, sigma=1,
        a=10.0, B=1.0, venue="uniswap v3", tvl_usd=1.0, parallel=True)
    b = PoolArc(
        id="v3:b", pool="0x" + "bb" * 20, kind=ArcKind.SWAP_UNIV3, i=0, j=1,
        n_coins=2, token_in="0xin", token_out="0xout", tau=2, sigma=1,
        a=1000.0, B=1.0, venue="uniswap v3", tvl_usd=1e9, parallel=True)
    _f, _r, out, _b = admissible_late_arcs(
        frame, [a, b], nu, bridgeable=(2,))
    assert out[2] > 900.0, "the $1e9 pool dominates, but both were counted"
    assert out[2] < 1000.0, "the $1 pool still moved it a little"


def test_connectivity_may_see_the_late_arcs_but_the_fit_never_holds_them():
    """`prepare` takes them to know what is connected, and returns none of them.

    This is the other half of the `late_arcs` bargain, and the half that a
    signature change could quietly break: the arcs get a say in *what is
    reachable* and no say at all in what anything is worth.
    """
    from erouter.core.pipeline import RouteResult, _restrict_to_component

    result = RouteResult(src_token="0xa", dst_token="0xb", nodes=None)
    frame = [_arc("curve:0", 0, 1)]
    late = [_arc("v2:0", 1, 2, "uniswap v2")]      # node 2 hangs off the venue

    kept = _restrict_to_component(frame, 2, 3, result, joining=late)
    assert [a.id for a in kept] == ["curve:0"], "the frame arc survives"
    assert all(a.venue == "" for a in kept), "and no venue arc is returned"

    # Without them node 2 is in no component reachable from the frame at all.
    assert _restrict_to_component(frame, 2, 3, result) == []


def test_a_node_with_one_curve_pool_and_one_pair_can_be_passed_through():
    """`_prune_dead_end_nodes` counts pools, and a venue pair is a pool.

    A node entered through Curve and left through Uniswap is a perfectly good
    hop.  Counting only Curve's pools deletes it before the arc that completes
    it has joined -- and dropping the prune entirely instead leaves every real
    dead end in the graph, which took a modelled route to 197 legs.
    """
    from erouter.core.pipeline import RouteResult, _prune_dead_end_nodes

    result = RouteResult(src_token="0xa", dst_token="0xb", nodes=None)
    frame = [_arc("curve:0", 0, 1), _arc("curve:1", 1, 2)]
    frame[1].pool = "0x" + "dd" * 20              # a second, distinct pool
    late = [_arc("v2:0", 2, 3, "uniswap v2")]
    late[0].pool = "0x" + "ee" * 20

    # Routing 0 -> 3: node 2 has one Curve pool and one pair, so it is passable.
    kept = _prune_dead_end_nodes(frame, 0, 3, result, joining=late)
    assert {a.id for a in kept} == {"curve:0", "curve:1"}

    # Without the venue counted, node 2 is a dead end and the hop is deleted.
    assert len(_prune_dead_end_nodes(frame, 0, 3, result)) < 2


def test_a_node_nobody_asked_for_is_not_priced_even_where_it_could_be():
    """Being in the node map and being priced by the frame are different.

    RSR and XYO are in the map with no Curve arc touching them.  Pricing every
    node the frame missed would admit arcs that have always been refused, on
    every route, whether or not anyone asked to reach them -- measured at
    1.03 bp on `USDC -> WETH` alone.  `venues/bridge.py` decides who gets a
    node, this decides who gets a price, and they must agree.
    """
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0, 1.0])
    reach_2 = _arc("v2:0", 2, 1, "uniswap v2", a=3.0)
    reach_3 = _arc("v2:1", 3, 1, "uniswap v2", a=5.0)
    fresh, refused, out, bridged = admissible_late_arcs(
        frame, [reach_2, reach_3], nu, bridgeable=(2,))
    assert bridged == 1 and [a.id for a in fresh] == ["v2:0"]
    assert refused == 1, "node 3 was never asked for"
    assert out[3] == 1.0, "and keeps the default rather than a derived price"


def test_naming_no_node_bridges_nothing():
    """The default, so a caller that has not thought about it changes nothing."""
    frame = [_arc("curve:0", 0, 1)]
    nu = np.array([1.0, 1.0, 1.0])
    fresh, refused, _out, bridged = admissible_late_arcs(
        frame, [_arc("v2:0", 2, 1, "uniswap v2", a=3.0)], nu)
    assert bridged == 0 and fresh == [] and refused == 1
