"""A Compound pool's rate is its cToken's, carried forward to this block.

`_stored_rates` on the deployed pool is `PRECISION_MUL[i] * rate`, where

    rate = exchangeRateStored
    rate += rate * supplyRatePerBlock * (block - accrualBlockNumber) / 1e18

Two things in that are easy to get wrong and both were measured against the
chain before this existed.  `PRECISION_MUL` corrects for the *underlying*
decimals, not the wrapped ones -- cUSDC has eight and USDC six -- and taking the
wrapped ones puts the model 101 bp out.  And the accrual term is not rounding:
skipping it leaves the rate stale by however many blocks since the cToken last
accrued.

With both, the model reproduces cDAI/cUSDC and cDAI/cUSDC/USDT to the wei.
"""

from __future__ import annotations

from erouter.core.codec import encode_call
from erouter.core.pools import PoolSpec
from erouter.core.transport import Answer, Status
from erouter.dev.stable_params import LENDING_PRECISION, _lending_rates

CDAI = "0x" + "c1" * 20
CUSDC = "0x" + "c2" * 20
DAI = "0x" + "d1" * 20
USDC = "0x" + "d2" * 20
POOL = "0x" + "aa" * 20

RATE_DAI = 240_000_000_000_000_000_000_000_000     # ~2.4e26
RATE_USDC = 240_000_000_000_000_000                # ~2.4e17
SUPPLY = 10**9
ACCRUED = 1_000
BLOCK = 1_100


def _word(value: int) -> bytes:
    return value.to_bytes(32, "big")


class Chain:
    """A Compound pool and its two cTokens, answering the getters they have."""

    def __init__(self, *, lending=True, underlying=True):
        self.lending = lending
        self.underlying = underlying

    def raw(self, calls):
        out = []
        for call in calls:
            out.append(self._one(call))
        return out

    def _one(self, call):
        to, data = call.to.lower(), bytes(call.data)
        if to == POOL:
            if not self.underlying:
                return Answer(Status.REVERTED, b"")
            for k, addr in enumerate((DAI, USDC)):
                if data == encode_call("underlying_coins(int128)", k):
                    return Answer(Status.VALUE, _word(int(addr, 16)))
            return Answer(Status.REVERTED, b"")
        if to in (DAI, USDC):
            if data == encode_call("decimals()"):
                return Answer(Status.VALUE, _word(18 if to == DAI else 6))
            return Answer(Status.REVERTED, b"")
        if to in (CDAI, CUSDC) and self.lending:
            rate = RATE_DAI if to == CDAI else RATE_USDC
            if data == encode_call("exchangeRateStored()"):
                return Answer(Status.VALUE, _word(rate))
            if data == encode_call("supplyRatePerBlock()"):
                return Answer(Status.VALUE, _word(SUPPLY))
            if data == encode_call("accrualBlockNumber()"):
                return Answer(Status.VALUE, _word(ACCRUED))
        return Answer(Status.REVERTED, b"")


def _pool():
    return PoolSpec.from_api({
        "address": POOL, "name": "cDAI/cUSDC", "pool_type": "main",
        "coins": [{"address": CDAI, "symbol": "cDAI", "decimals": 8, "pool_index": 0},
                  {"address": CUSDC, "symbol": "cUSDC", "decimals": 8, "pool_index": 1}],
    })


def _accrued(rate: int) -> int:
    return rate + rate * SUPPLY * (BLOCK - ACCRUED) // LENDING_PRECISION


def test_the_rate_is_the_ctoken_s_carried_to_this_block():
    rates = _lending_rates([_pool()], Chain(), BLOCK)
    assert rates == {POOL: (_accrued(RATE_DAI), 10**12 * _accrued(RATE_USDC))}


def test_the_multiplier_follows_the_underlying_decimals():
    """cUSDC has eight and USDC six; taking the wrapped ones is 101 bp out."""
    rates = _lending_rates([_pool()], Chain(), BLOCK)[POOL]
    assert rates[1] // _accrued(RATE_USDC) == 10**12, "USDC is 6 decimals, not 8"
    assert rates[0] // _accrued(RATE_DAI) == 1, "DAI is 18 decimals"


def test_the_accrual_term_is_not_skipped():
    """It is interest since the cToken last accrued, not a rounding term."""
    stale = _lending_rates([_pool()], Chain(), ACCRUED)[POOL]
    fresh = _lending_rates([_pool()], Chain(), BLOCK)[POOL]
    assert fresh[0] > stale[0]
    assert stale[0] == RATE_DAI, "at the accrual block there is nothing to add"


def test_a_pool_with_no_underlying_is_not_asked_about_lending():
    """Ordinary pools pay one batch of reverts for the question and no more."""
    assert _lending_rates([_pool()], Chain(underlying=False), BLOCK) == {}


def test_a_pool_whose_coins_do_not_lend_is_left_out():
    """`underlying_coins` alone does not make it a lending pool -- the coin has
    to answer Compound's getters, or the rate is `LENDING_PRECISION` and the
    plain candidate already covers it."""
    assert _lending_rates([_pool()], Chain(lending=False), BLOCK) == {}
