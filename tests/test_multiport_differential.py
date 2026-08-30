"""An element in Rust must be the element `core/multiport.py` describes.

Two halves, and they fail differently.

The **shape rules** are pure combinatorics -- the injection of ports onto
coins, the `#in + #out <= N` bound, the LP token consuming no slot -- and a
divergence there admits a route the reference would refuse.  Every refusal is
compared by its message, because the messages are how a caller tells "this is a
parallel pair" from "this is a re-entered pool".

The **arithmetic** is the advancing state, which is the whole reason an element
exists: the second port sees the pool the first one left.  That is compared to
the wei.  A port priced against a stale state is exactly the error the element
was introduced to prevent, and it would show up here as an output that is
slightly too large -- never as an exception.
"""

from __future__ import annotations

import pytest

from erouter.core.accel import available
from erouter.core.multiport import (
    BPS,
    LP,
    MultiPort,
    MultiPortError,
    Port,
    best_split,
    element_from,
    evaluate,
)
from erouter.core.stableswap import StableSwap, StableSwapLP
from erouter.core.types import ArcKind

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")

UNIT = 10**18
ADDRESS = "0x" + "3c" * 20

POOL = StableSwap(
    balances=(1_000_000 * UNIT, 1_000_000 * UNIT, 1_000_000 * UNIT),
    rates=(UNIT, UNIT, UNIT),
    amp=2000, fee=4_000_000, offpeg_fee_multiplier=0,
    a_precision=1, fee_on_xp=False, admin_fee=5_000_000_000,
)
#: Off its peg and with mismatched decimals, which is where a stale state shows.
LOPSIDED = StableSwap(
    balances=(4_000_000 * UNIT, 250_000 * 10**6, 900_000 * UNIT),
    rates=(UNIT, 10**30, UNIT),
    amp=350, fee=3_000_000, offpeg_fee_multiplier=20_000_000_000,
    a_precision=100, fee_on_xp=True, admin_fee=5_000_000_000,
)
SUPPLY = 3_000_000 * UNIT


def native(pool: StableSwap, supply: int | None = None):
    """The same pool on the Rust side, plus its LP model. `(Pools, i, lp)`."""
    import erouter_solve

    pools = erouter_solve.Pools()
    which = pools.add_stableswap(
        [str(b) for b in pool.balances], [str(r) for r in pool.rates],
        str(pool.amp), str(pool.fee), str(pool.offpeg_fee_multiplier),
        str(pool.a_precision), pool.fee_on_xp, pool.subtract_one,
        str(pool.admin_fee),
    )
    lp = None
    if supply is not None:
        lp = pools.add_stable_lp(
            [str(b) for b in pool.balances], [str(r) for r in pool.rates],
            str(pool.amp), str(pool.fee), str(pool.offpeg_fee_multiplier),
            str(pool.a_precision), pool.fee_on_xp, pool.subtract_one,
            str(supply), False, str(pool.admin_fee),
        )
    return pools, which, lp


def element_class():
    import erouter_solve

    return erouter_solve.Element


def flat(ports) -> list[tuple[int, int]]:
    return [(p.coin, p.bps) for p in ports]


# ------------------------------------------------------ the shape rules


SHAPES = [
    # (n_coins, inputs, outputs) -- admissible and not, by turns.
    (3, [(0, BPS)], [(1, BPS)]),
    (3, [(0, BPS)], [(1, 5_000), (2, 5_000)]),
    (3, [(0, 4_000), (1, 6_000)], [(LP, BPS)]),
    (2, [(0, 5_000), (1, 5_000)], [(LP, BPS)]),       # LP consumes no slot
    (3, [(LP, BPS)], [(0, BPS)]),
    (2, [(0, BPS)], [(1, 5_000), (2, 5_000)]),        # coin out of range
    (3, [(0, BPS)], [(0, 5_000), (1, 5_000)]),        # a wash
    (3, [(0, BPS)], []),                               # no port on a side
    (3, [(0, BPS)], [(1, 4_000), (2, 5_000)]),        # shares do not sum
    (3, [(0, BPS)], [(1, 0), (2, BPS)]),              # a zero share
    (3, [(0, BPS)], [(1, 10_001), (2, -1)]),          # out of (0, BPS]
    (3, [(LP, 5_000), (LP, 5_000)], [(0, BPS)]),      # two LP ports one side
    (4, [(0, 5_000), (1, 5_000)], [(2, 5_000), (3, 5_000)]),  # admissible shape
]


@pytest.mark.parametrize("n_coins,inputs,outputs", SHAPES,
                         ids=[f"{n}:{len(i)}->{len(o)}" for n, i, o in SHAPES])
def test_the_shape_rules_agree_refusal_for_refusal(n_coins, inputs, outputs):
    want = None
    try:
        made = MultiPort(pool=ADDRESS, n_coins=n_coins,
                         inputs=tuple(Port(*p) for p in inputs),
                         outputs=tuple(Port(*p) for p in outputs))
    except MultiPortError as e:
        want = str(e)

    got = None
    try:
        ported = element_class()(ADDRESS, n_coins, inputs, outputs)
    except ValueError as e:
        got = str(e)

    assert got == want
    if want is None:
        assert ported.inputs == flat(made.inputs)
        assert ported.outputs == flat(made.outputs)
        assert ported.n_coins == made.n_coins
        assert ported.ports == made.ports
        assert ported.pool == made.pool


TRIPLES = [
    # One pool's legs, as `(kind, i, j)`.
    (3, [(ArcKind.SWAP_STABLE, 0, 1)]),
    (3, [(ArcKind.SWAP_STABLE, 0, 1), (ArcKind.SWAP_STABLE, 0, 2)]),
    (2, [(ArcKind.SWAP_STABLE, 0, 1), (ArcKind.SWAP_STABLE, 1, 0)]),
    (3, [(ArcKind.SWAP_STABLE, 0, 1), (ArcKind.SWAP_STABLE, 0, 1)]),
    (3, [(ArcKind.DEPOSIT_FIXED, 0, 0), (ArcKind.DEPOSIT_FIXED, 1, 0)]),
    (2, [(ArcKind.DEPOSIT_FIXED, 0, 0), (ArcKind.DEPOSIT_FIXED, 1, 0)]),
    (3, [(ArcKind.WITHDRAW_STABLE, 0, 0), (ArcKind.WITHDRAW_STABLE, 0, 1)]),
    (3, [(ArcKind.WITHDRAW_STABLE, 0, 1)]),
    (3, [(ArcKind.WRAP_NATIVE, 0, 1)]),
    (3, [(ArcKind.SWAP_STABLE, 0, 1), (ArcKind.DEPOSIT_FIXED, 2, 0)]),
    (3, [(ArcKind.SWAP_STABLE, 0, 1), (ArcKind.SWAP_STABLE, 2, 1)]),
]


@pytest.mark.parametrize("n_coins,triples", TRIPLES,
                         ids=[f"{n}:{len(t)}" for n, t in TRIPLES])
def test_element_from_agrees_refusal_for_refusal(n_coins, triples):
    want = None
    try:
        made = element_from(ADDRESS, n_coins, triples)
    except MultiPortError as e:
        want = str(e)

    got = None
    try:
        ported = element_class().from_triples(
            ADDRESS, n_coins,
            [int(k) for k, _, _ in triples],
            [i for _, i, _ in triples],
            [j for _, _, j in triples],
        )
    except ValueError as e:
        got = str(e)

    assert got == want, (got, want)
    if want is None:
        assert ported.inputs == flat(made.inputs)
        assert ported.outputs == flat(made.outputs)


@pytest.mark.parametrize("kind", list(ArcKind))
def test_ports_of_agrees_including_what_is_not_a_port(kind):
    from erouter.core.multiport import ports_of

    want = None
    try:
        made = ports_of(kind, 1, 2)
    except MultiPortError as e:
        want = str(e)
    got = None
    try:
        ported = element_class().ports_of(int(kind), 1, 2)
    except ValueError as e:
        got = str(e)
    assert got == want
    if want is None:
        assert ported == made


# ------------------------------------------------- the advancing state


SIZES = [UNIT, 100 * UNIT, 10_000 * UNIT, 250_000 * UNIT]


@pytest.mark.parametrize("dx", SIZES)
@pytest.mark.parametrize("shares", [(5_000, 5_000), (1, 9_999), (7_777, 2_223)])
def test_one_in_two_out_agrees_to_the_wei(dx, shares):
    """The second swap must see the pool the first one left."""
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, shares[0]), Port(2, shares[1])))
    want, _, _ = evaluate(made, POOL, None, dx)

    pools, which, _ = native(POOL)
    got = pools.element_evaluate(which, None, 3, [(0, BPS)],
                                 [(1, shares[0]), (2, shares[1])], str(dx))
    assert [int(v) for v in got] == want


@pytest.mark.parametrize("dx", SIZES)
def test_a_lopsided_pool_agrees_too(dx):
    """Dynamic fees and 6-decimal coins, which is where a stale state shows."""
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, 3_000), Port(2, 7_000)))
    want, _, _ = evaluate(made, LOPSIDED, None, dx)

    pools, which, _ = native(LOPSIDED)
    got = pools.element_evaluate(which, None, 3, [(0, BPS)],
                                 [(1, 3_000), (2, 7_000)], str(dx))
    assert [int(v) for v in got] == want


@pytest.mark.parametrize("dx", SIZES)
def test_a_deposit_out_of_a_swap_agrees(dx):
    """One in, one coin out and one LP out: the mint sees the swapped pool."""
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, 4_000), Port(LP, 6_000)))
    want, _, _ = evaluate(made, POOL, StableSwapLP(pool=POOL, total_supply=SUPPLY), dx)

    pools, which, lp = native(POOL, SUPPLY)
    got = pools.element_evaluate(which, lp, 3, [(0, BPS)],
                                 [(1, 4_000), (LP, 6_000)], str(dx))
    assert [int(v) for v in got] == want


@pytest.mark.parametrize("dx", SIZES)
def test_many_in_one_lp_out_agrees(dx):
    """A single `add_liquidity` whose amounts vector is the input weights."""
    made = MultiPort(pool=ADDRESS, n_coins=3,
                     inputs=(Port(0, 2_500), Port(1, 2_500), Port(2, 5_000)),
                     outputs=(Port(LP, BPS),))
    want, _, _ = evaluate(made, POOL, StableSwapLP(pool=POOL, total_supply=SUPPLY), dx)

    pools, which, lp = native(POOL, SUPPLY)
    got = pools.element_evaluate(
        which, lp, 3, [(0, 2_500), (1, 2_500), (2, 5_000)], [(LP, BPS)], str(dx))
    assert [int(v) for v in got] == want


@pytest.mark.parametrize("dx", SIZES)
def test_an_lp_input_agrees(dx):
    """A burn does not advance -- deliberately -- so both sides must not."""
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(LP, BPS),),
                     outputs=(Port(1, BPS),))
    want, _, _ = evaluate(made, POOL, StableSwapLP(pool=POOL, total_supply=SUPPLY), dx)

    pools, which, lp = native(POOL, SUPPLY)
    got = pools.element_evaluate(which, lp, 3, [(LP, BPS)], [(1, BPS)], str(dx))
    assert [int(v) for v in got] == want


def test_the_advance_is_what_makes_it_an_element():
    """Against two independent swaps at the same shares, not just against Rust.

    If either side priced the second port on the pool as it started, this
    would pass at the wei and mean nothing -- so it is checked here that the
    element genuinely differs from the un-advanced pair.
    """
    dx = 250_000 * UNIT
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, 5_000), Port(2, 5_000)))
    coupled, _, _ = evaluate(made, POOL, None, dx)
    # The same two trades, each against the untouched pool.
    naive = [POOL.get_dy(0, 1, dx // 2), POOL.get_dy(0, 2, dx - dx // 2)]
    assert coupled[0] == naive[0]      # the first leg sees the same state
    assert coupled[1] < naive[1]       # the second does not
    assert naive[1] - coupled[1] > 0


# -------------------------------------------------------- the best split


@pytest.mark.parametrize("dx", SIZES)
@pytest.mark.parametrize("pool", [POOL, LOPSIDED], ids=["even", "lopsided"])
def test_best_split_picks_the_same_bps(dx, pool):
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, 5_000), Port(2, 5_000)))
    rates, coins = pool.rates, (1, 2)
    tuned, payout = best_split(
        made, pool, None, dx,
        lambda k, amount: amount * rates[coins[k]] / 1e18)

    pools, which, _ = native(pool)
    first, second, got = pools.element_best_split(
        which, None, 3, [(0, BPS)], [(1, 5_000), (2, 5_000)], str(dx),
        [str(rates[1]), str(rates[2])])

    assert (first, second) == (tuned.outputs[0].bps, tuned.outputs[1].bps)
    assert got == payout


def test_best_split_is_for_one_in_two_out_only():
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, BPS),))
    with pytest.raises(MultiPortError, match="one-in two-out"):
        best_split(made, POOL, None, UNIT, lambda k, a: float(a))

    pools, which, _ = native(POOL)
    with pytest.raises(ValueError, match="one-in two-out"):
        pools.element_best_split(which, None, 3, [(0, BPS)], [(1, BPS)],
                                 str(UNIT), [str(UNIT)])


def test_the_two_best_splits_agree_with_the_specialised_one():
    """`Pools.element_split` is the swap-only fast path for the same search."""
    dx = 10_000 * UNIT
    pools, which, _ = native(POOL)
    quick = pools.element_split(which, 0, 1, 2, dx)
    first, second, _ = pools.element_best_split(
        which, None, 3, [(0, BPS)], [(1, 5_000), (2, 5_000)], str(dx),
        [str(POOL.rates[1]), str(POOL.rates[2])])
    assert quick == (first, second)


# ------------------------------------------------------------ refusals


def test_nothing_to_route_is_refused_on_both_sides():
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, BPS),))
    with pytest.raises(MultiPortError, match="nothing to route"):
        evaluate(made, POOL, None, 0)

    pools, which, _ = native(POOL)
    with pytest.raises(ValueError, match="nothing to route"):
        pools.element_evaluate(which, None, 3, [(0, BPS)], [(1, BPS)], "0")


def test_many_in_many_out_is_refused_on_both_sides():
    made = MultiPort(pool=ADDRESS, n_coins=4,
                     inputs=(Port(0, 5_000), Port(1, 5_000)),
                     outputs=(Port(2, 5_000), Port(3, 5_000)))
    with pytest.raises(MultiPortError, match="pairing rule"):
        evaluate(made, POOL, None, UNIT)

    pools, which, _ = native(POOL)
    with pytest.raises(ValueError, match="pairing rule"):
        pools.element_evaluate(which, None, 4, [(0, 5_000), (1, 5_000)],
                               [(2, 5_000), (3, 5_000)], str(UNIT))


def test_several_inputs_pay_only_an_lp_port_on_both_sides():
    made = MultiPort(pool=ADDRESS, n_coins=3,
                     inputs=(Port(0, 5_000), Port(1, 5_000)),
                     outputs=(Port(2, BPS),))
    with pytest.raises(MultiPortError, match="several inputs pay only an LP port"):
        evaluate(made, POOL, None, UNIT)

    pools, which, _ = native(POOL)
    with pytest.raises(ValueError, match="several inputs pay only an LP port"):
        pools.element_evaluate(which, None, 3, [(0, 5_000), (1, 5_000)],
                               [(2, BPS)], str(UNIT))


def test_a_port_allocated_nothing_is_refused_on_both_sides():
    """A share that rounds to zero wei, not a zero `bps`."""
    made = MultiPort(pool=ADDRESS, n_coins=3, inputs=(Port(0, BPS),),
                     outputs=(Port(1, 1), Port(2, 9_999)))
    with pytest.raises(MultiPortError, match="allocated nothing"):
        evaluate(made, POOL, None, 100)

    pools, which, _ = native(POOL)
    with pytest.raises(ValueError, match="allocated nothing"):
        pools.element_evaluate(which, None, 3, [(0, BPS)],
                               [(1, 1), (2, 9_999)], "100")
