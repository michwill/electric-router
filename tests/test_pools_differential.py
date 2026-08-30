"""The ported pool models must answer what the Python ones answer.

Python is the reference: a model is admitted by reproducing the chain wei for
wei, and the port is admitted by reproducing Python.  So this compares the two
directly rather than checking either against a stored table -- a table would
freeze whichever was right when it was written, and the point is that the two
stay together as the reference moves.

`docs/performance.md` reports 1,156 such vectors run once, by hand.  This is
the standing version of that run: fewer pools, every legacy flag, and it fails
the build rather than a paragraph.

The vectors are shared with `test_wasm_differential.py`, which puts the same
inputs through the browser's copy, so all three arithmetics are held to one
reference.
"""

from __future__ import annotations

import pytest

from erouter.core.stableswap import StableSwap
from erouter.core.tricrypto import Tricrypto
from erouter.core.twocrypto import Twocrypto

erouter_solve = pytest.importorskip("erouter_solve")

GNOSIS_3POOL = {
    "balances": (142638 * 10**18, 153563 * 10**6, 246110 * 10**6),
    "rates": (10**18, 10**30, 10**30),
    "amp": 200 * 100,
    "fee": 3 * 10**6,
    "a_precision": 100,
    "fee_on_xp": True,
    "admin_fee": 5 * 10**9,
}

YB_WETH = {
    "balances": (1195163862946386689613, 2295927389925329891241),
    "precisions": (1, 1),
    "price_scale": 1000000000000000000,
    "d": 3491091252871716580854,
    "amp": 200000000,
    "gamma": 1000000000000000,
    "mid_fee": 3000000,
    "out_fee": 30000000,
    "fee_gamma": 10000000000000000,
    "stable": True,
}

TRICRYPTO = {
    "balances": (1328923109972, 2051494736, 695461346171780166020),
    "precisions": (10**12, 10**10, 1),
    "price_scale": (64757744418661683527417, 3067521869705084157440),
    "d": 3986769329916000000000000,
    "amp": 1707629,
    "gamma": 11809167828997,
    "mid_fee": 2999999,
    "out_fee": 80000000,
    "fee_gamma": 350000000000000,
}


def _stableswap():
    """The flags that change the last wei, each on its own."""
    for over in ({}, {"fee_on_xp": False}, {"subtract_one": False},
                 {"offpeg_fee_multiplier": 20000000000},
                 {"amp": 20 * 100}, {"fee": 0}):
        yield {**GNOSIS_3POOL, **over}


def _twocrypto():
    for over in ({}, {"stable": False}, {"stable": False, "v21": False},
                 {"stable": False, "legacy_fee": True},
                 {"stable": False, "legacy_pool": True},
                 {"stable": False, "legacy_pool": True, "legacy_mul2": True}):
        yield {**YB_WETH, **over}


def _tricrypto():
    for over in ({}, {"legacy": True, "a_multiplier": 100}):
        yield {**TRICRYPTO, **over}


def _s(v):
    return str(int(v))


def _add(pools, kind, spec):
    """Hand one model over, in the spelling the warm uses."""
    if kind == "stableswap":
        return pools.add_stableswap(
            [_s(b) for b in spec["balances"]], [_s(r) for r in spec["rates"]],
            _s(spec["amp"]), _s(spec["fee"]),
            _s(spec.get("offpeg_fee_multiplier", 0)),
            _s(spec.get("a_precision", 100)), spec.get("fee_on_xp", True),
            spec.get("subtract_one", True),
            _s(spec["admin_fee"]) if spec.get("admin_fee", -1) >= 0 else None)
    if kind == "twocrypto":
        return pools.add_twocrypto(
            [_s(b) for b in spec["balances"]],
            [_s(p) for p in spec["precisions"]], _s(spec["price_scale"]),
            _s(spec["d"]), _s(spec["amp"]), _s(spec["gamma"]),
            _s(spec["mid_fee"]), _s(spec["out_fee"]), _s(spec["fee_gamma"]),
            spec.get("stable", True), spec.get("v21", True),
            spec.get("legacy_fee", False), spec.get("legacy_pool", False),
            spec.get("legacy_mul2", False))
    return pools.add_tricrypto(
        [_s(b) for b in spec["balances"]],
        [_s(p) for p in spec["precisions"]],
        [_s(p) for p in spec["price_scale"]], _s(spec["d"]), _s(spec["amp"]),
        _s(spec["gamma"]), _s(spec["mid_fee"]), _s(spec["out_fee"]),
        _s(spec["fee_gamma"]), spec.get("legacy", False),
        _s(spec.get("a_multiplier", 10000)))


def _python_price(model, i, j, dx, fast):
    """What Python answers, with a refusal spelled the way Rust spells it.

    The two sides express "this pool will not do that trade" differently --
    Python raises out of the family's own error type, the registry returns
    `None` -- so one is translated into the other here rather than the test
    pretending either is wrong.  `ArithmeticError` is the common base of all
    three families' errors.
    """
    try:
        got = model.get_dy_fast(i, j, dx) if fast else model.get_dy(i, j, dx)
    except (ArithmeticError, ValueError):
        return None
    return int(got) if got is not None else None


def _model(kind, spec):
    if kind == "stableswap":
        return StableSwap(**spec)
    if kind == "twocrypto":
        return Twocrypto(**spec)
    return Tricrypto(**spec)


#: Sizes over five decades, so a rounding convention that only shows at one end
#: is caught.  Scaled by the input coin's own magnitude rather than fixed, or a
#: 6-decimal coin and an 18-decimal one would not be asked comparable trades.
SHARES = (10**-6, 10**-4, 10**-2, 10**-1)


def vectors():
    """Every (kind, spec, i, j, dx) the two sides are held to."""
    for kind, specs in (("stableswap", _stableswap()),
                        ("twocrypto", _twocrypto()),
                        ("tricrypto", _tricrypto())):
        for n, spec in enumerate(specs):
            balances = spec["balances"]
            for i in range(len(balances)):
                for j in range(len(balances)):
                    if i == j:
                        continue
                    for share in SHARES:
                        dx = int(balances[i] * share)
                        if dx > 0:
                            yield kind, n, spec, i, j, dx


CASES = list(vectors())


def _ids():
    return [f"{k}{n}-{i}to{j}-{dx:.0e}" for k, n, _s, i, j, dx in CASES]


#: What the float path is allowed to differ by, in bp.  `docs/performance.md`
#: measures the Python float form against its own integer form at 5.4e-4 bp
#: over 263 mainnet stableswaps and budgets the quote path to that; the port is
#: held to the same number rather than a looser one invented here.
FLOAT_BUDGET_BP = 5.4e-4


def _batch(fast):
    """Both sides' answers for every vector, in one pass."""
    pools = erouter_solve.Pools()
    want, which, ii, jj, dd = [], [], [], [], []
    for kind, _n, spec, i, j, dx in CASES:
        want.append(_python_price(_model(kind, spec), i, j, dx, fast))
        which.append(_add(pools, kind, spec))
        ii.append(i)
        jj.append(j)
        dd.append(dx)
    assert len(pools) == len(CASES)
    return want, pools.price(which, ii, jj, dd, fast)


def test_the_exact_port_is_wei_for_wei():
    """The integer path is the contract's own arithmetic, so this is equality.

    Not a tolerance: the exact form *is* what admits a pool -- a model is
    trusted because it reproduces the chain to the wei -- so a port that is one
    wei out is a port that would admit a pool the chain refuses.
    """
    want, rust = _batch(False)
    wrong = [(i, w, r) for i, w, r in zip(_ids(), want, rust, strict=True) if w != r]
    assert not wrong, f"{len(wrong)} of {len(CASES)} disagree: {wrong[:5]}"


def test_the_float_port_stays_inside_the_quote_path_budget():
    """The float path ranks routes; it does not admit pools.

    So the question is not whether the two agree to the wei -- they do not, and
    `docs/performance.md` explains why: `dy = xp[j] - y - 1` inherits `y`'s
    absolute round-off, and the two sides do not have to reach `y` by the same
    sequence.  The question is whether the disagreement stays under what a
    ranking can absorb.  Both sides' refusals must still line up exactly.
    """
    want, rust = _batch(True)
    drift = []
    for name, w, r in zip(_ids(), want, rust, strict=True):
        assert (w is None) == (r is None), f"{name}: one side refused, {w} {r}"
        if w is not None and w != 0:
            drift.append((abs(r - w) / w * 10_000, name))
    worst, where = max(drift)
    assert worst <= FLOAT_BUDGET_BP, (
        f"worst {worst:.3e} bp at {where}, budget {FLOAT_BUDGET_BP:.1e}")

    # The budget is the contract and it is seven orders of magnitude above what
    # this actually costs, so on its own it would not notice a port going
    # gradually wrong.  The median does: most vectors agree bit for bit -- only
    # the large-share cryptoswap ones amplify `y`'s round-off -- so a median
    # that is no longer zero means the two sides stopped running the same
    # sequence, whatever the worst case says.
    median = sorted(bp for bp, _ in drift)[len(drift) // 2]
    assert median == 0.0, f"median drift {median:.3e} bp, expected exact"


def test_a_refusal_is_a_refusal_on_both_sides():
    """An index the registry does not hold answers `None`, never a number."""
    pools = erouter_solve.Pools()
    _add(pools, "stableswap", GNOSIS_3POOL)
    assert pools.price([7], [0], [1], [10**18], True) == [None]


def test_element_split_agrees_with_the_python_search():
    """The one call that moves a loop rather than batching one.

    Stableswap only: the other families have no `best_split`, so the registry
    answers `None` and the caller runs the sweep it would have done anyway.
    """
    from erouter.core.multiport import MultiPort, Port, best_split

    spec = GNOSIS_3POOL
    pools = erouter_solve.Pools()
    _add(pools, "stableswap", spec)
    model = StableSwap(**spec)
    dx = spec["balances"][0] // 100

    got = pools.element_split(0, 0, 1, 2, dx)
    assert got is not None

    element = MultiPort(pool="p", n_coins=3, inputs=(Port(0, 10_000),),
                        outputs=(Port(1, 5_000), Port(2, 5_000)))
    rates, coins = model.rates, (1, 2)
    tuned, _ = best_split(element, model, None, dx,
                          lambda k, amount: amount * rates[coins[k]] / 1e18)
    assert got == (tuned.outputs[0].bps, tuned.outputs[1].bps)
    assert sum(got) == 10_000, "the legs must partition dx"


def test_a_family_without_a_search_declines():
    """Twocrypto has no `best_split`; the registry must say so rather than
    answering for the wrong family."""
    pools = erouter_solve.Pools()
    _add(pools, "twocrypto", YB_WETH)
    assert pools.element_split(0, 0, 1, 1, 10**18) is None
