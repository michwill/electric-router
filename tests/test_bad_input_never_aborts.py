"""A bad number must not take the process with it.

Python signals "this pool will not do that trade" by raising out of an
`ArithmeticError`; Rust signals it by returning `None`.  Neither is a crash,
and the whole router is built on that -- an arc that refuses is an arc the
route does not take.

The port broke the arrangement in a way no differential caught, because every
differential feeds it a pool that works.  `ruint` panics on a subtraction that
borrows and on a product that wraps, always, and indexing a `Vec` past its end
panics too; the release profile said `panic = "abort"`, so each of those was
`SIGABRT` and the whole CPython process went with it.  No traceback, no
`except`, no quote.

So this feeds the compiled side inputs it has no answer for and requires that
it *answers* -- `None`, or an exception a caller can catch.  It is the test
that would have failed before `panic = "unwind"` and the guards that came with
it; `docs/performance.md` records what unwinding costs.
"""

from __future__ import annotations

import pathlib
import random

import pytest

from erouter.core import curves
from erouter.core.stableswap import StableSwap, StableSwapError

erouter_solve = pytest.importorskip("erouter_solve")

E18 = 10**18
#: Values that break a different thing each: nothing, a wei, a whole token, a
#: rate for a 6-decimal coin, and the three magnitudes where `U256` products
#: start to wrap.
EDGES = (0, 1, 2, 10**6, E18, 10**30, 2**128, 2**255, 2**256 - 1)


def _s(v):
    return str(int(v))


def _hostile(rng):
    """One registry holding one pool built out of values no chain would hold."""
    pools = erouter_solve.Pools()
    draw = lambda: rng.choice((*EDGES, rng.randrange(1, 2**200)))  # noqa: E731
    family = rng.choice(("stableswap", "twocrypto", "tricrypto", "vault", "lp"))
    if family == "stableswap":
        which = pools.add_stableswap(
            [_s(draw()), _s(draw())], [_s(draw()), _s(draw())], _s(draw()),
            _s(rng.choice((0, 3 * 10**6, 10**10))), _s(draw()),
            _s(rng.choice((1, 100))), rng.random() < 0.5, rng.random() < 0.5,
            _s(rng.choice((0, 5 * 10**9))))
    elif family == "twocrypto":
        which = pools.add_twocrypto(
            [_s(draw()), _s(draw())], [_s(1), _s(1)], _s(draw()), _s(draw()),
            _s(draw()), _s(draw()), _s(3000000), _s(30000000), _s(10**16),
            rng.random() < 0.5, rng.random() < 0.5, False, False, False)
    elif family == "tricrypto":
        which = pools.add_tricrypto(
            [_s(draw()), _s(draw()), _s(draw())], [_s(1), _s(1), _s(1)],
            [_s(draw()), _s(draw())], _s(draw()), _s(draw()), _s(draw()),
            _s(2999999), _s(80000000), _s(350000000000000), False, _s(10000))
    elif family == "vault":
        which = pools.add_vault(_s(draw()), _s(draw()),
                                _s(rng.choice((0, draw()))))
    else:
        which = pools.add_stable_lp(
            [_s(draw()), _s(draw())], [_s(E18), _s(E18)], _s(draw()),
            _s(3 * 10**6), _s(0), _s(100), True, True, _s(draw()),
            rng.random() < 0.5, _s(0))
    return pools, which


def test_a_hostile_pool_state_answers_instead_of_aborting(capfd):
    """Every model, over states that reach every branch of the arithmetic.

    `capfd` is the real assertion.  A panic caught at the registry's boundary
    still prints through the default hook, so an empty stderr says the guards
    inside the models did the refusing -- not the net underneath them.
    """
    rng = random.Random(20260901)
    answers = 0
    for _ in range(600):
        pools, which = _hostile(rng)
        for i, j in ((0, 1), (1, 0), (0, 0), (0, 2), (2, 0), (3, 0)):
            for dx in (0, 1, E18, 10**30, 2**127 - 1):
                for fast in (True, False):
                    got = pools.price([which], [i], [j], [dx], fast)
                    assert len(got) == 1
                    assert got[0] is None or isinstance(got[0], int)
                    answers += 1
    assert answers == 600 * 6 * 5 * 2
    assert capfd.readouterr().err == "", "a pool model panicked"


@pytest.mark.parametrize("fast", (True, False))
@pytest.mark.parametrize("i, j", ((2, 0), (0, 2), (7, 9)))
def test_a_coin_the_pool_does_not_have_is_a_refusal(i, j, fast):
    """Both sides refuse, in the spelling each uses for a refusal.

    Stableswap was the family that did not check: the other two have always
    refused an index out of range, and this one indexed `xp` straight -- an
    `IndexError` in the reference, an index off the end of a `Vec` in the port.
    """
    balances, rates = (10**18, 10**18), (10**18, 10**18)
    model = StableSwap(
        balances=balances, rates=rates, amp=20000, fee=3 * 10**6,
        offpeg_fee_multiplier=0, a_precision=100, fee_on_xp=True,
        subtract_one=True, admin_fee=5 * 10**9)
    with pytest.raises(StableSwapError):
        (model.get_dy_fast if fast else model.get_dy)(i, j, E18)
    with pytest.raises(StableSwapError):
        model.exchange(i, j, E18)

    pools = erouter_solve.Pools()
    which = pools.add_stableswap(
        [_s(b) for b in balances], [_s(r) for r in rates], _s(20000),
        _s(3 * 10**6), _s(0), _s(100), True, True, _s(5 * 10**9))
    assert pools.price([which], [i], [j], [E18], fast) == [None]


def test_a_size_that_does_not_compare_is_zero_on_both_sides():
    """A NaN input to `Curve.at`, which the split search reaches by arithmetic.

    It used to search for a bracket for it and unwrap the `None` that came
    back from comparing.  Both sides now send it down the tail, where it falls
    out as zero.
    """
    xs, ys = [1.0, 10.0, 100.0, 1000.0], [1.0, 9.9, 98.0, 950.0]
    reference = curves.fit(xs, ys)
    ported = erouter_solve.Curve.fit(xs, ys)
    nan = float("nan")
    assert reference.at(nan) == 0.0
    assert ported.at(nan) == 0.0
    assert reference.error_bp_at(nan) == float("inf")
    assert ported.error_bp_at(nan) == float("inf")
    # And the numbers that do compare are untouched by the change.
    for v in (0.5, 1.0, 50.0, 1000.0, 5000.0):
        assert ported.at(v) == reference.at(v)


def test_the_release_profile_can_still_unwind():
    """`panic = "abort"` would make every case above a `SIGABRT` again.

    A guard rather than a preference: nothing else in the suite fails if this
    flips back, because an aborting interpreter reports no failures at all.
    """
    manifest = pathlib.Path(__file__).resolve().parents[1] / "rust" / "Cargo.toml"
    body = manifest.read_text()
    assert '[profile.release]' in body
    assert 'panic = "unwind"' in body
    assert 'panic = "abort"' not in body
