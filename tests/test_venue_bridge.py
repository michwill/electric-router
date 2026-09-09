"""Introducing a token the frame has never priced.

`X -> USDT -> crvUSD` was never a topology problem: the graph could always
represent it, but `X` had no node, because the node map is built from Curve's
own pools.  On ethereum that left 6,379 tokens and $617.8M one hop outside.
"""

from __future__ import annotations

from erouter.core.codec import encode_call
from erouter.venues import bridge

USDT = "0x" + "cc" * 20
WETH = "0x" + "ee" * 20
X = "0x" + "11" * 20          # only Uniswap has ever held it
Y = "0x" + "22" * 20          # nothing has


class Nodes:
    """Prices USDT and WETH; `add_token` is the only way in."""

    def __init__(self):
        self.added = []
        self._known = {USDT, WETH}

    def has(self, token):
        return token.lower() in self._known

    def add_token(self, address, symbol="", decimals=18):
        self.added.append((address.lower(), symbol, decimals))
        self._known.add(address.lower())
        return len(self._known)


class Venue:
    def __init__(self, census, floor_usd=10_000.0):
        self.census, self.floor_usd = census, floor_usd


def word(n):
    return n.to_bytes(32, "big").hex()


class Transport:
    """Answers `decimals()` and `symbol()`; anything else is a refusal."""

    def __init__(self, decimals=6, symbol="XCOIN", bytes32=False, fail=False):
        self.decimals, self.symbol, self.bytes32, self.fail = (
            decimals, symbol, bytes32, fail)
        self.calls = 0

    def fetch_multi(self, requests, concurrent=False):
        if self.fail:
            raise RuntimeError("endpoint said no")
        want = "0x" + encode_call("decimals()").hex()
        out = []
        for _m, params in requests:
            self.calls += 1
            if params[0]["data"] == want:
                out.append("0x" + word(self.decimals)
                           if self.decimals is not None else None)
                continue
            raw = self.symbol.encode()
            if self.bytes32:
                out.append("0x" + raw.ljust(32, b"\x00").hex())
            else:
                out.append("0x" + word(32) + word(len(raw))
                           + raw.ljust(32, b"\x00").hex())
        return out


def test_a_token_joined_to_a_priced_coin_is_introduced():
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, USDT, 30, 250_000.0]})]
    got = bridge.introduce(X, nodes, venues, Transport(), 1)
    assert got == "XCOIN"
    assert nodes.added == [(X, "XCOIN", 6)]
    assert nodes.has(X), "and it is now routable"


def test_a_token_joined_only_to_another_unpriced_token_stays_out():
    """There is no anchor, and pricing is one Gauss-Seidel step, not a walk."""
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, Y, 30, 250_000.0]})]
    assert bridge.introduce(X, nodes, venues, Transport(), 1) is None
    assert nodes.added == []


def test_a_bridge_thinner_than_the_floor_does_not_count():
    """The pair's price is the only price the token gets."""
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, USDT, 30, 500.0]})]
    assert bridge.introduce(X, nodes, venues, Transport(), 1) is None


def test_a_token_that_will_not_say_its_decimals_is_refused():
    """Every amount in and out of the pair is scaled by them."""
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, USDT, 30, 250_000.0]})]
    assert bridge.introduce(X, nodes, venues,
                            Transport(decimals=None), 1) is None
    assert bridge.introduce(X, nodes, venues, Transport(fail=True), 1) is None
    assert nodes.added == []


def test_an_absurd_decimals_is_refused():
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, USDT, 30, 250_000.0]})]
    assert bridge.introduce(X, nodes, venues, Transport(decimals=77), 1) is None


def test_a_bytes32_symbol_still_introduces_the_token():
    """Plenty of tokens predate the string return; the symbol is cosmetic."""
    nodes = Nodes()
    venues = [Venue({"0xpool": [X, USDT, 30, 250_000.0]})]
    got = bridge.introduce(X, nodes, venues,
                           Transport(symbol="OLD", bytes32=True), 1)
    assert got == "OLD" and nodes.added[0][2] == 6


def test_a_token_the_frame_already_prices_is_left_alone():
    """Called on every pair change, so it has to be idempotent."""
    nodes = Nodes()
    venues = [Venue({"0xpool": [USDT, WETH, 30, 250_000.0]})]
    transport = Transport()
    assert bridge.introduce(USDT, nodes, venues, transport, 1) is None
    assert nodes.added == [] and transport.calls == 0, "and costs no read"


def test_the_deepest_bridge_is_offered_first():
    nodes = Nodes()
    venues = [Venue({"0xthin": [X, USDT, 30, 20_000.0],
                     "0xdeep": [X, WETH, 30, 900_000.0]})]
    found = bridge.bridges_for(X, nodes, venues)
    assert [p for _v, p, _t in found] == ["0xdeep", "0xthin"]


def test_both_venues_are_searched():
    nodes = Nodes()
    v2 = Venue({"0xa": [X, USDT, 30, 50_000.0]})
    v3 = Venue({"0xb": [X, WETH, 3000, 80_000.0]})
    assert len(bridge.bridges_for(X, nodes, [v2, v3])) == 2
