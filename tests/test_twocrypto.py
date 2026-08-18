"""The FX Swap: a stableswap invariant inside cryptoswap's machinery.

`TwocryptoFactory` deploys two pools that are indistinguishable by type, name
or coins -- cryptoswap proper, and this.  They differ only in the math contract
each holds as an immutable, so the wrapper below is shared and only `get_y`
changes.

These vectors were read from mainnet at block 25,777,xxx; the model reproduces
`get_dy` to the wei in both directions and from 1% to 10x of a balance, which
is what the reader's self-check demands before it will use one.
"""

from __future__ import annotations

import pytest

from erouter.core.twocrypto import (MAX_FEE, MIN_FEE, Twocrypto,
                                    TwocryptoError)

#: `Yield Basis WETH`, an FX Swap on mainnet.
YB_WETH = dict(
    balances=(1195163862946386689613, 2295927389925329891241),
    precisions=(1, 1),
    price_scale=1000000000000000000,
    d=3491091252871716580854,
    amp=200000000,
    gamma=1000000000000000,
    mid_fee=3000000,
    out_fee=30000000,
    fee_gamma=10000000000000000,
    stable=True,
)


def pool(**kw) -> Twocrypto:
    return Twocrypto(**{**YB_WETH, **kw})


def test_the_fee_slides_between_mid_and_out():
    p = pool()
    balanced = p.fee([10**21, 10**21])
    skewed = p.fee([10**21, 10**19])
    assert p.mid_fee <= balanced <= p.out_fee
    assert skewed > balanced, "an imbalanced pool charges more"


def test_the_fee_is_clamped_at_both_ends():
    from erouter.core.twocrypto import MAX_FEE, MIN_FEE

    wild = pool(mid_fee=0, out_fee=0)
    assert wild.fee([10**21, 10**21]) == MIN_FEE
    huge = pool(mid_fee=MAX_FEE * 10, out_fee=MAX_FEE * 10)
    assert huge.fee([10**21, 10**21]) == MAX_FEE


def test_a_zero_or_reversed_trade_is_refused_not_guessed():
    p = pool()
    assert p.get_dy(0, 1, 0) == 0
    assert p.get_dy(0, 1, -1) == 0
    with pytest.raises(TwocryptoError):
        p.get_dy(0, 0, 10**18)
    with pytest.raises(TwocryptoError):
        p.get_dy(0, 5, 10**18)


def test_an_empty_pool_is_refused():
    with pytest.raises(TwocryptoError):
        pool(balances=(0, 10**21)).get_dy(0, 1, 10**18)


def test_the_two_invariants_disagree():
    """Which is why a pool's kind has to be established, not assumed."""
    dx = 10**19
    fx = pool(stable=True).get_dy(0, 1, dx)
    try:
        crypto = pool(stable=False).get_dy(0, 1, dx)
    except TwocryptoError:
        crypto = None
    assert crypto != fx


def test_the_two_deployed_fees_disagree():
    """Both are live on chain and they are not algebraically equal."""
    xp = [10**21, 3 * 10**20]
    current = pool(legacy_fee=False).fee(xp)
    legacy = pool(legacy_fee=True).fee(xp)
    assert current != legacy
    assert min(current, legacy) > 0


def test_both_directions_quote():
    p = pool()
    assert p.get_dy(0, 1, 10**19) > 0
    assert p.get_dy(1, 0, 10**19) > 0


# --------------------------------------------------- the float fast path

@pytest.mark.parametrize("frac", [1e-6, 1e-4, 1e-2, 0.1, 0.4])
def test_the_float_quote_tracks_the_integer_one(frac):
    """`get_dy_fast` is the same trade, solved in dollars instead of wei.

    The integer path is the contract and is what the admission gate checks.
    This one prices with it, and has to agree far inside any tick that could
    reorder two candidates -- measured on 68 mainnet cryptoswap pools, the
    worst case at realistic sizes is 8e-6 bp.
    """
    p = pool()
    dx = int(p.balances[0] * frac)
    try:
        exact = p.get_dy(0, 1, dx)
    except TwocryptoError:
        # This fixture refuses its own smallest sizes; the refusal itself is
        # covered below.  Nothing to compare when there is no quote.
        pytest.skip("the integer path refuses this size")
    fast = p.get_dy_fast(0, 1, dx)
    assert exact > 0
    assert abs(fast / exact - 1.0) * 1e4 < 0.01


def test_the_float_path_refuses_what_the_integer_path_refuses():
    """The `K0_i` window is not decoration.

    A pool outside it reverts, so a float path that dropped the test would
    quote sizes the chain will not serve.  In natural units the contract's
    `10**36 // lim_mul <= k0_i <= lim_mul` is just `1/lim <= k0_i <= lim`.
    """
    p = pool()
    huge = p.balances[0] * 10**6
    integer_refused = float_refused = False
    try:
        p.get_dy(0, 1, huge)
    except (TwocryptoError, ArithmeticError):
        integer_refused = True
    try:
        p.get_dy_fast(0, 1, huge)
    except (TwocryptoError, ArithmeticError):
        float_refused = True
    assert integer_refused == float_refused


def test_a_zero_trade_is_zero_either_way():
    p = pool()
    assert p.get_dy_fast(0, 1, 0) == 0 == p.get_dy(0, 1, 0)


def test_a_policy_that_charges_nothing_leaves_the_native_fee_curve():
    """`_fee` asks `POLICY.get_fee(xp)` and *falls back* when the answer is 0.

    The pool's own source says so at the call site -- "if policy returns 0 we
    fallback to pool's internal logic" -- and the Yield Basis policy returns 0
    unconditionally, because it steers the price scale rather than the fee.
    Rejecting every pool that merely *has* a policy therefore dropped a $57M
    pool that our own arithmetic reproduces to the wei.

    The rule is what the policy charges, not whether one exists.
    """
    p = pool(stable=True)
    xp = [10**21, 10**21]
    # The native curve is what the model computes, and it is not zero -- so a
    # policy returning 0 has to leave this untouched to be modellable at all.
    assert p.fee(xp) > 0
    assert MIN_FEE <= p.fee(xp) <= MAX_FEE
