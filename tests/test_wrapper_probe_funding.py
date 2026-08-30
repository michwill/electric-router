"""Finding someone to mint with when the graph knows no holder of the asset.

`no holder` is not a verdict, but it reads like one: nothing downstream can
tell "we could not ask" from "yes".  Nine mainnet vaults sat in that state and
three of them refuse the mint -- NaraUSD in a deposit hook, USPC and tETH on
the asset's own transfer -- so all three built mint arcs that revert.
"""

from __future__ import annotations

import contextlib

from erouter.dev.executability import _funded_depositor

WRAPPER = "0x" + "cc" * 20
ASSET = "0x" + "dd" * 20
FRESH = "0x" + "ee" * 20


class Token:
    def __init__(self, balances, *, transfers=True, decimals=18):
        self.balances = dict(balances)
        self.transfers = transfers
        self._decimals = decimals
        self.sender = None

    def balanceOf(self, who):
        return self.balances.get(who, 0)

    def decimals(self):
        return self._decimals

    def transfer(self, to, value):
        if not self.transfers:
            raise RuntimeError("Unauthorized()")
        self.balances[self.sender] = self.balances.get(self.sender, 0) - value
        self.balances[to] = self.balances.get(to, 0) + value


class Erc20:
    def __init__(self, token):
        self.token = token

    def at(self, address):
        return self.token


class Boa:
    """Just the surface `_funded_depositor` touches."""

    def __init__(self, token, *, deals=True):
        self.token = token
        self.deals = deals
        self.dealt = False
        self.env = self

    def generate_address(self):
        return FRESH

    def set_balance(self, who, value):
        pass

    @contextlib.contextmanager
    def prank(self, who):
        was, self.token.sender = self.token.sender, who
        try:
            yield
        finally:
            self.token.sender = was

    def deal(self, token, who, amount):
        self.dealt = True
        if not self.deals:
            raise RuntimeError("no balance slot found")
        token.balances[who] = amount


def test_the_vault_funds_the_probe_from_its_own_asset():
    token = Token({WRAPPER: 10**21})
    boa = Boa(token)
    assert _funded_depositor(boa, Erc20(token), ASSET, WRAPPER) == FRESH
    assert token.balanceOf(FRESH) > 0
    assert not boa.dealt  # a real transfer is preferred to a written balance


def test_a_refusing_asset_falls_back_to_writing_the_balance():
    # tETH's asset reverts `Unauthorized()` on a plain transfer.  That is a fact
    # about the asset, not an answer about the mint, so it must not end the probe.
    token = Token({WRAPPER: 10**21}, transfers=False)
    boa = Boa(token)
    assert _funded_depositor(boa, Erc20(token), ASSET, WRAPPER) == FRESH
    assert boa.dealt
    assert token.balanceOf(FRESH) == 100 * 10**18


def test_a_vault_holding_nothing_still_gets_a_depositor():
    token = Token({})
    boa = Boa(token)
    assert _funded_depositor(boa, Erc20(token), ASSET, WRAPPER) == FRESH
    assert boa.dealt


def test_no_way_to_fund_is_reported_as_no_depositor():
    # Which leaves the direction untested -- never refused.
    token = Token({}, transfers=False)
    assert _funded_depositor(Boa(token, deals=False), Erc20(token), ASSET, WRAPPER) == ""


def test_decimals_are_the_assets_own():
    token = Token({}, decimals=6)
    boa = Boa(token)
    _funded_depositor(boa, Erc20(token), ASSET, WRAPPER)
    assert token.balanceOf(FRESH) == 100 * 10**6
