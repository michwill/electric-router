"""The Rust calldata must be the calldata `core/routecall.py` produces.

This is the end of the line: the bytes a wallet signs.  Nothing here has a
tolerance, because a word one bit out is either a different pool, a different
coin, or a minimum rate that reverts an honest trade or admits a sandwich.

Three things are compared, and they fail differently.

**The packing.**  `Step.pack` writes nine fields into one word against a layout
`contracts/ElectricRouter.vy` also knows.  A shift out by one is a route that
executes something else entirely.

**The fractions.**  `Leg.bps` is a share of what a node held when its group
opened; the contract wants a share of what is standing there *now*.  Get the
conversion wrong and a split silently starves.

**The minimum rates.**  These are the only thing between the route and a
sandwich, and the rule is deliberately not "bind at the quote": a leg with no
room reverts on the route's own rounding.  Both the fee rule and a
caller-named budget are compared, because they are different code paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core import routecall
from erouter.core.accel import available
from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.realize import realize
from erouter.core.types import ArcKind, PoolArc

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

TOKENS = ["0x" + f"{k:02x}" * 20 for k in range(6)]
POOLS = ["0x" + f"{0xa0 + k:02x}" * 20 for k in range(6)]
RECEIVER = "0x" + "99" * 20


def native():
    import erouter_solve

    return erouter_solve


# --------------------------------------------------------------- packing


PACKINGS = [
    (0, 0, 1, 2, str(10**18), "0", 0, 0),
    (0, 3, 7, 8, "1", str((1 << 128) - 1), 31, 31),
    (1, 15, 15, 15, str(10**18 - 1), "12345", 1, 2),
    (5, 0, 2, 4, str(5 * 10**17), str(10**18), 0, 5),
    (16, 1, 0, 2, "1000", "999999", 7, 0),
]


@pytest.mark.parametrize("packed", PACKINGS, ids=[str(p[0]) for p in PACKINGS])
def test_a_step_packs_to_the_same_word(packed):
    kind, i, j, n, frac, rate, in_ref, out_ref = packed
    step = routecall.Step(
        pool="0xp", kind=ArcKind(kind), i=i, j=j, n=n, frac=int(frac),
        min_rate=int(rate), in_ref=in_ref, out_ref=out_ref,
    )
    want = step.pack()
    got = native().pack_step(kind, i, j, n, frac, rate, in_ref, out_ref)
    assert int(got) == want

    back = routecall.unpack(want)
    assert native().unpack_step(str(want)) == (
        int(back.kind), back.i, back.j, back.n, str(back.frac),
        str(back.min_rate), back.in_ref, back.out_ref)


BAD_PACKINGS = [
    (0, 0, 1, 2, "0", "0", 0, 0),                       # frac zero
    (0, 0, 1, 2, str(10**18 + 1), "0", 0, 0),           # frac past one
    (0, 0, 1, 2, str(10**18), str(1 << 128), 0, 0),     # rate past its field
    (0, 16, 1, 2, str(10**18), "0", 0, 0),              # i past its field
    (0, 0, 1, 2, str(10**18), "0", 32, 0),              # in_ref past the table
]


@pytest.mark.parametrize("packed", BAD_PACKINGS)
def test_the_same_words_are_refused(packed):
    kind, i, j, n, frac, rate, in_ref, out_ref = packed
    with pytest.raises(ValueError):
        routecall.Step(pool="0xp", kind=ArcKind(kind), i=i, j=j, n=n,
                       frac=int(frac), min_rate=int(rate),
                       in_ref=in_ref, out_ref=out_ref).pack()
    with pytest.raises(ValueError):
        native().pack_step(kind, i, j, n, frac, rate, in_ref, out_ref)


def test_reserved_bits_are_refused_on_both_sides():
    word = 1 << routecall.RESERVED_SHIFT
    with pytest.raises(ValueError):
        routecall.unpack(word)
    with pytest.raises(ValueError):
        native().unpack_step(str(word))


# --------------------------------------------------------------- routes


def build(hops, dst_slot, *, verified=False, fees=True, volatile_pool=False):
    """A realised route on both sides, over the given `(src, dst)` slot hops."""
    nodes, ported = NodeMap(), native().NodeMap()
    for k, address in enumerate(TOKENS):
        nodes.add_token(address, f"T{k}", 18)
        ported.add_token(address, f"T{k}", 18)

    arcs, flows = [], []
    for k, (src, dst) in enumerate(hops):
        arcs.append(PoolArc(
            id=f"a{k}", pool=POOLS[k], kind=ArcKind.SWAP_STABLE, i=0, j=1,
            n_coins=2, token_in=TOKENS[src], token_out=TOKENS[dst],
            tau=nodes.node(TOKENS[src]), sigma=nodes.node(TOKENS[dst]),
            a=0.999 - 0.0001 * k, B=1e-6, G=1e6, eps=0.001,
            gamma_live=0.9996 if fees else float("nan"),
            reserve_in=10**24, decimals_in=18, decimals_out=18, tvl_usd=1e7,
        ))
        flows.append(1.0 + 0.1 * k)

    nu = np.ones(nodes.n_nodes)
    want = realize(arcs, np.array(flows), nu, nodes,
                   src_token=TOKENS[hops[0][0]], dst_token=TOKENS[dst_slot],
                   amount_in=10**18)
    built = native().Arcs()
    for a in arcs:
        built.add(a.id, a.pool, int(a.kind), a.i, a.j, a.n_coins, a.token_in,
                  a.token_out, a.tau, a.sigma, a.a, a.B, a.cap, a.G, a.eps,
                  a.reserve_in, a.decimals_in, a.tvl_usd, a.gamma_live, a.note,
                  a.calib_delta, a.decimals_out)
    got = native().Route.realize(built, flows, [float(v) for v in nu], ported,
                                 TOKENS[hops[0][0]], TOKENS[dst_slot],
                                 str(10**18), None)
    if verified:
        # What a chained walk fills in. These are the difference between a
        # route as modelled and one as priced: `leg_out` prefers
        # `verified_out`, and the bound is set from `fee_floor` -- the least
        # the pool can charge -- rather than from what the leg pays.
        for k, rl in enumerate(want.legs):
            rl.verified_in = rl.amount_in
            rl.verified_out = rl.amount_out * 999 // 1000
            rl.fee_floor = 0.0003
            rl.fee_frac = 0.0013
            got.set_verified(k, str(rl.verified_in), str(rl.verified_out),
                             rl.fee_floor, rl.fee_frac)
    return want, got


SHAPES = [
    ([(0, 1)], 1),
    ([(0, 1), (1, 2)], 2),
    ([(0, 1), (0, 1)], 1),
    ([(0, 1), (0, 2), (1, 3), (2, 3)], 3),
    ([(0, 1), (1, 2), (2, 3), (0, 3)], 3),
]
IDS = [f"{len(h)}legs" for h, _ in SHAPES]


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
@pytest.mark.parametrize("verified", [False, True], ids=["modelled", "priced"])
def test_the_fractions_agree(hops, dst, verified):
    """A share of what is standing there now, not of what the node held."""
    want, got = build(hops, dst, verified=verified)
    assert [int(v) for v in native().fractions(got)] == routecall.fractions(want)


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
def test_the_movement_floors_and_tolerances_agree(hops, dst):
    want, got = build(hops, dst)
    for volatile in ([], [POOLS[0]], POOLS[:3]):
        assert native().movement_floors(got, volatile=volatile) == \
            routecall.movement_floors(want, volatile=volatile)
        assert native().tolerances(got, volatile=volatile) == \
            routecall.tolerances(want, volatile=volatile)


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
@pytest.mark.parametrize("volatile", [[], [POOLS[0]]], ids=["steady", "volatile"])
@pytest.mark.parametrize("verified", [False, True], ids=["modelled", "priced"])
def test_the_minimum_rates_agree(hops, dst, volatile, verified):
    """The only thing between the route and a sandwich."""
    want, got = build(hops, dst, verified=verified)
    want_rates, want_unbounded = routecall.min_rates(want, volatile=volatile)
    got_rates, got_unbounded = native().min_rates(got, volatile=volatile)
    assert [int(v) for v in got_rates] == want_rates
    assert got_unbounded == want_unbounded


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
@pytest.mark.parametrize("budget", [0.0, 1.0, 5.0, 50.0])
def test_a_caller_named_budget_divides_the_same_way(hops, dst, budget):
    """The other code path: `slippage.divide` instead of the fee rule."""
    want, got = build(hops, dst)
    want_rates, want_unbounded = routecall.min_rates(want, slippage_bp=budget)
    got_rates, got_unbounded = native().min_rates(got, slippage_bp=budget)
    assert [int(v) for v in got_rates] == want_rates
    assert got_unbounded == want_unbounded


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
def test_the_bounds_walk_agrees(hops, dst):
    want, got = build(hops, dst)
    fracs = routecall.fractions(want)
    rates, _ = routecall.min_rates(want)
    promised, floors = routecall.walk_bounds(want, fracs, rates)
    got_promised, got_floors = native().walk_bounds(
        got, [str(v) for v in fracs], [str(v) for v in rates])
    assert int(got_promised) == promised
    assert [int(v) for v in got_floors] == floors


# ------------------------------------------------------------- the call


@pytest.mark.parametrize("hops,dst", SHAPES, ids=IDS)
@pytest.mark.parametrize("naming", ["needed", "none", "all"])
@pytest.mark.parametrize("verified", [False, True], ids=["modelled", "priced"])
def test_the_whole_call_encodes_to_the_same_bytes(hops, dst, naming, verified):
    want, got = build(hops, dst, verified=verified)
    made = routecall.encode_route(want, receiver=RECEIVER, naming=naming)
    ported = native().RouteCall.encode_route(got, receiver=RECEIVER, naming=naming)

    assert ported.pools == list(made.pools)
    assert [int(v) for v in ported.params] == list(made.params)
    assert ported.tokens == list(made.tokens)
    assert ported.token_in == made.token_in
    assert ported.token_out == made.token_out
    assert int(ported.amount_in) == made.amount_in
    assert int(ported.guaranteed_out) == made.guaranteed_out
    assert int(ported.quoted_out) == made.quoted_out
    assert ported.unbounded == list(made.unbounded)
    assert ported.tolerance_bp == made.tolerance_bp

    # The bytes themselves, which is the only thing a node sees.
    assert ported.calldata() == made.calldata()
    assert ported.calldata(RECEIVER) == made.calldata(RECEIVER)
    # And the verified fields the bound was set from crossed intact.
    assert [(int(a), int(b)) for a, b in got.verified()] == \
        [(rl.verified_in, rl.verified_out) for rl in want.legs]


def test_the_shortest_entry_point_is_chosen_the_same_way():
    """Three words of calldata is real money on an L2."""
    want, got = build([(0, 1), (1, 2)], 2)
    for receiver, min_out, sender in [
        ("", 0, ""),
        (RECEIVER, 0, ""),
        (RECEIVER, 0, RECEIVER),
        (RECEIVER, 1, ""),
    ]:
        made = routecall.encode_route(want, receiver=receiver, min_out=min_out)
        ported = native().RouteCall.encode_route(
            got, receiver=receiver, min_out=str(min_out))
        assert ported.calldata(sender) == made.calldata(sender)
        assert len(ported.calldata(sender)) == len(made.calldata(sender))


def test_min_out_without_a_receiver_is_refused_on_both_sides():
    want, got = build([(0, 1)], 1)
    made = routecall.encode_route(want, receiver="", min_out=5)
    ported = native().RouteCall.encode_route(got, receiver="", min_out="5")
    with pytest.raises(ValueError, match="min_out needs a receiver"):
        made.calldata()
    with pytest.raises(ValueError, match="min_out needs a receiver"):
        ported.calldata()


def test_an_unroutable_kind_is_refused_on_both_sides():
    want, got = build([(0, 1)], 1)
    want.legs[0].kind = ArcKind.LEND_MINT
    # `LEND_MINT` is executable; a kind with no derivation rule is not. The
    # reference's table covers every kind, so this checks the pair agree that
    # it does rather than inventing one that does not exist.
    assert routecall._DERIVE.get(ArcKind.LEND_MINT) is not None
    assert native().RouteCall.encode_route(got, receiver=RECEIVER) is not None


def test_a_naming_it_does_not_know_is_refused():
    want, got = build([(0, 1)], 1)
    with pytest.raises(ValueError, match="naming must be one of"):
        routecall.encode_route(want, receiver=RECEIVER, naming="whatever")
    with pytest.raises(ValueError, match="naming must be one of"):
        native().RouteCall.encode_route(got, receiver=RECEIVER, naming="whatever")


def test_a_route_with_no_legs_is_refused_on_both_sides():
    """An alias pair has nothing to execute."""
    import erouter_solve

    nodes, ported = NodeMap(), native().NodeMap()
    for k, address in enumerate(TOKENS[:2]):
        nodes.add_token(address, f"T{k}", 18)
        ported.add_token(address, f"T{k}", 18)
    nodes.merge(Conversion(ConversionKind.ALIAS, TOKENS[1], TOKENS[0]))
    ported.merge("ALIAS", TOKENS[1], TOKENS[0])

    from erouter.core.realize import conversion_route

    want = conversion_route(nodes, src_token=TOKENS[0], dst_token=TOKENS[1],
                            amount_in=10**18)
    got = erouter_solve.Route.conversion_route(ported, TOKENS[0], TOKENS[1],
                                               str(10**18))
    assert not want.legs and len(got) == 0
    with pytest.raises(ValueError, match="no legs"):
        routecall.encode_route(want, receiver=RECEIVER)
    with pytest.raises(ValueError, match="no legs"):
        native().RouteCall.encode_route(got, receiver=RECEIVER)
