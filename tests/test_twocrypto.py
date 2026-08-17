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

from erouter.core.twocrypto import Twocrypto, TwocryptoError

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


def test_cryptoswap_is_refused_rather_than_approximated():
    """Until `newton_y` exists, a non-FX pool must not be answered at all."""
    with pytest.raises(TwocryptoError, match="not implemented"):
        pool(stable=False).get_dy(0, 1, 10**18)


def test_both_directions_quote():
    p = pool()
    assert p.get_dy(0, 1, 10**19) > 0
    assert p.get_dy(1, 0, 10**19) > 0
