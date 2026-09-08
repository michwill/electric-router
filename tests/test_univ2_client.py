"""Pricing a v2 leg where the model lives, not where the chain cannot answer.

No deployed quoter knows `SWAP_UNIV2`, so `quote_routes` returns zero for it,
`verify` reads zero as a revert, and every candidate carrying v2 is dropped
before it can win.  That failure is indistinguishable from v2 losing on the
merits, which is why it is worth a test rather than a comment.
"""

from __future__ import annotations

import pytest

from erouter.core.types import ArcKind
from erouter.core.walk import LegUnquotable
from erouter.venues import univ2_client
from erouter.venues.univ2 import PairState, output

PAIR = "0x" + "11" * 20
ELSEWHERE = "0x" + "99" * 20


class Leg:
    def __init__(self, target, kind, i, j):
        self.target, self.kind, self.i, self.j = target, kind, i, j


class Client:
    """A client whose `_quote_leg` answers a sentinel, so delegation shows."""

    def __init__(self):
        self.asked = []

    def _quote_leg(self, leg, dx):
        self.asked.append((leg.kind, dx))
        return 12_345


def state():
    return {PAIR: PairState(1_000 * 10**18, 2_500_000 * 10**6, 18, 6, 30)}


def test_a_v2_leg_is_priced_from_the_reserves_in_hand():
    client = Client()
    univ2_client.teach(client, state())
    got = client._quote_leg(Leg(PAIR, ArcKind.SWAP_UNIV2, 0, 1), 10**18)
    assert got == output(state()[PAIR], True, 10**18)
    assert client.asked == [], "a v2 leg must not fall through to the chain"


def test_the_input_index_picks_the_direction():
    """`i` is the input coin, so `i == 0` is `zero_for_one`.

    Getting this backwards prices every leg at the reciprocal, which on a
    USDC/WETH pair is wrong by six orders and still looks like a number.
    """
    client = Client()
    univ2_client.teach(client, state())
    forward = client._quote_leg(Leg(PAIR, ArcKind.SWAP_UNIV2, 0, 1), 10**18)
    reverse = client._quote_leg(Leg(PAIR, ArcKind.SWAP_UNIV2, 1, 0), 10**6)
    assert forward == output(state()[PAIR], True, 10**18)
    assert reverse == output(state()[PAIR], False, 10**6)
    assert forward > 10**9 and reverse < 10**18


def test_every_other_kind_still_goes_where_it_went():
    client = Client()
    univ2_client.teach(client, state())
    assert client._quote_leg(Leg(PAIR, ArcKind.SWAP_STABLE, 0, 1), 7) == 12_345
    assert client.asked == [(ArcKind.SWAP_STABLE, 7)]


def test_a_pair_not_held_refuses_rather_than_answering_zero():
    """A zero is read as a revert, which would hide the failure it signals."""
    client = Client()
    univ2_client.teach(client, state())
    with pytest.raises(LegUnquotable):
        client._quote_leg(Leg(ELSEWHERE, ArcKind.SWAP_UNIV2, 0, 1), 10**18)


def test_two_venues_taught_to_one_client_each_answer_for_their_own():
    """A session may hold both, and the order they are taught must not matter.

    Each `teach` chains with whatever `_quote_leg` it found, so the second does
    not shadow the first -- which is the failure that would make one venue
    silently unquotable.
    """
    from erouter.venues import univ3_client
    from erouter.venues.univ3 import Arc

    client = Client()
    # `univ3.output` water-fills at a common marginal rate, taking
    # `(a - u) / B` from each arc, so a bank with `B = 0` is degenerate and
    # pays nothing.  Curvature here is a property of the fixture, not a claim.
    banks = {(ELSEWHERE, 0, 1): univ3_client.Bank(
        arcs=[Arc(a=2_500.0, B=1e-3, cap=1e6)], decimals_in=18, decimals_out=6)}
    for teach, payload in ((univ2_client.teach, state()),
                           (univ3_client.teach, banks)):
        teach(client, payload)

    v2 = client._quote_leg(Leg(PAIR, ArcKind.SWAP_UNIV2, 0, 1), 10**18)
    v3 = client._quote_leg(Leg(ELSEWHERE, ArcKind.SWAP_UNIV3, 0, 1), 10**18)
    assert v2 == output(state()[PAIR], True, 10**18)
    assert v3 > 0
    assert client._quote_leg(Leg(PAIR, ArcKind.SWAP_CRYPTO, 0, 1), 3) == 12_345
