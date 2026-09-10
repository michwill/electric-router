"""Reading v4 state, and the fee that reading it wrong hides.

The arc math is v3's and already proven; what is new here is where the numbers
come from.  One of them was very nearly missed: `getSlot0` returns a protocol
fee alongside the LP fee, v4 charges it on top, and ignoring it made every arc
over-promise by exactly that much.

Audited against `V4Quoter` on twelve live pools at a $1,000 notional, the fix
took nineteen of twenty directions to within 0.011 bp, worst over-quote
+0.010 bp.  Before it, four pools drifted +5.007, +1.251, +0.250 and +0.020 bp
-- which were their protocol fees of 500, 125, 25 and 2, exactly.
"""

from __future__ import annotations

import pytest

from erouter.venues.univ4_chain import PIPS, STATE_VIEW, effective_fee


def packed(zero_for_one: int, one_for_zero: int) -> int:
    """`getSlot0`'s protocol fee: low twelve bits are `zeroForOne`."""
    return (one_for_zero << 12) | zero_for_one


def test_no_protocol_fee_leaves_the_lp_fee_alone():
    assert effective_fee(3000, 0, True) == 3000
    assert effective_fee(3000, 0, False) == 3000


@pytest.mark.parametrize("protocol,lp,want", [
    (500, 3000, 3499),        # the +5.007 bp pool; 3498.5 exact, rounded up
    (125, 500, 625),          # the +1.251 bp pool
    (25, 100, 125),           # the +0.250 bp pool
    (2, 10, 12),              # the +0.020 bp pool
])
def test_the_two_fees_compose_rather_than_add(protocol, lp, want):
    """v4 takes the protocol fee from the input before the LP fee sees it.

    Adding them would over-charge by `protocol * lp`, which is small -- and
    small, unexplained and in the wrong direction is the worst kind of error to
    carry, because nothing ever gets big enough to notice.
    """
    got = effective_fee(lp, packed(protocol, protocol), True)
    assert got == want
    assert got <= protocol + lp, "composed, not summed"
    exact = protocol + lp * (PIPS - protocol) / PIPS
    assert got >= exact, "rounded up, so the model never over-promises"


def test_the_two_directions_are_unpacked_separately():
    """They are allowed to differ, so assuming they match is a real bug that
    would show only on the direction nobody happened to audit."""
    raw = packed(zero_for_one=500, one_for_zero=0)
    assert effective_fee(3000, raw, True) == effective_fee(3000, 500, True)
    assert effective_fee(3000, raw, False) == 3000


def test_a_protocol_fee_is_capped_at_a_tenth_of_a_percent():
    """v4's own bound.  A field that big is a misread, not a fee."""
    assert effective_fee(0, packed(4095, 4095), True) == 1_000


def test_every_state_view_address_is_an_address():
    """A typo here reads as a chain with no v4 rather than as an error."""
    for chain, addr in STATE_VIEW.items():
        assert addr.startswith("0x") and len(addr) == 42, chain
        int(addr, 16)


def test_ethereum_state_view_is_the_deployed_one():
    """Pinned: this is what the census and every read go through, and a wrong
    address answers nothing rather than answering wrongly."""
    assert STATE_VIEW["ethereum"] == "0x7ffe42c4a5deea5b0fec41c94c136cf115597227"


def test_the_fee_is_never_rounded_down():
    """Down is an output rounded *up*, which is the direction that breaks the
    certificate.  `round` would have: 3498.5 goes to 3498, not 3499."""
    for lp in range(0, 10_001, 7):
        for protocol in (0, 1, 2, 25, 125, 500, 1_000):
            got = effective_fee(lp, packed(protocol, protocol), True)
            exact = protocol + lp * (PIPS - protocol) / PIPS
            assert got >= exact
            assert got - exact < 1, "and never by more than one hundredth of a bip"
