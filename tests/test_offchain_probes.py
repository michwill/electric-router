"""An off-chain kind must be *probeable*, not merely walkable.

`test_venue_teaches_its_kind.py` holds the venues to teaching `_quote_leg`,
which is what `quote_routes` walks -- so `verify` prices a v2, v3 or v4 leg and
a candidate carrying one can win.  That test passes while `probe` is still
broken, because `probe` does not go through `_quote_leg` at all: it encodes
`SIG_PROBE_BATCH` for the deployed `RouteQuoter`, which has never heard of
kinds 17, 18 or 19 and answers zero for each.

The failure is silent and lands two stages away.  `split` samples each leg's
curve by probing it, `_probe_ladders` wants two usable answers per leg and gets
none, `_trusted_curves` gives up with "a leg would not probe", and `optimise`
falls back to the chained hill-climb -- having already paid for the probes it
then discards.  Measured on ethereum at 25,935,978 with v2 enabled, that cost
`WETH -> WBTC` at $1M 9.2 s of a 13.6 s quote and `ALD -> FRAX` at $100k 3.9 s
of 7.5 s, every v2 leg answering 0 of 24.
"""

from __future__ import annotations

import pytest

from erouter.core.transport import Status
from erouter.core.types import OFF_CHAIN_KINDS, ArcKind, Probe
from erouter.venues.offchain_client import teach_probes

POOL = "0x" + "11" * 20
ON_CHAIN = ArcKind.SWAP_STABLE


class Client:
    """A quoter that answers zero for a kind it does not know, as the real one
    does, and records what it was asked to send."""

    def __init__(self, walkable=True):
        self.sent = []
        self.walkable = walkable

    def probe(self, probes):
        self.sent.append(list(probes))
        return [_Quote(Status.VALUE, 7) for _ in probes]

    def _quote_leg(self, leg, dx):
        if leg.kind in OFF_CHAIN_KINDS and self.walkable:
            return dx * 2
        return 0


class _Quote:
    __slots__ = ("status", "value")

    def __init__(self, status, value):
        self.status, self.value = status, value


def probe_of(kind, dx=100):
    return Probe(pool=POOL, kind=kind, i=0, j=1, n=2, dx=dx)


@pytest.mark.parametrize("kind", sorted(OFF_CHAIN_KINDS, key=int))
def test_an_off_chain_probe_is_answered_locally(kind):
    client = Client()
    teach_probes(client)
    got = client.probe([probe_of(kind)])
    assert [q.value for q in got] == [200], (
        f"{kind.name} probes as zero, which reads as a leg that would not "
        f"probe and sends `split` to the chained search")
    assert client.sent == [], "it should not have reached the deployed quoter"


def test_an_on_chain_probe_still_goes_to_the_quoter():
    client = Client()
    teach_probes(client)
    got = client.probe([probe_of(ON_CHAIN)])
    assert [q.value for q in got] == [7]
    assert len(client.sent) == 1 and len(client.sent[0]) == 1


def test_a_mixed_batch_keeps_its_order():
    """The callers zip answers against the probes they sent, so a batch that
    comes back re-ordered is worse than one that fails."""
    client = Client()
    teach_probes(client)
    probes = [probe_of(ArcKind.SWAP_UNIV2), probe_of(ON_CHAIN, 5),
              probe_of(ArcKind.SWAP_UNIV3), probe_of(ON_CHAIN, 6)]
    got = client.probe(probes)
    assert [q.value for q in got] == [200, 7, 200, 7]
    assert [p.dx for p in client.sent[0]] == [5, 6], "only the on-chain ones"


def test_a_pool_the_venue_does_not_hold_reverts_rather_than_zeroes():
    """`LegUnquotable` means "ask the chain", and a zero `VALUE` would instead
    assert that the pool really pays nothing."""
    client = Client(walkable=False)

    def refuse(leg, dx):
        from erouter.core.walk import LegUnquotable
        raise LegUnquotable(leg.target)

    client._quote_leg = refuse
    teach_probes(client)
    got = client.probe([probe_of(ArcKind.SWAP_UNIV2)])
    assert got[0].status is Status.REVERTED and got[0].value == 0


def test_teaching_order_does_not_matter():
    """`teach_probes` reads `_quote_leg` when it is called, so it may be
    installed before the venue teachers as readily as after."""
    client = Client(walkable=False)
    teach_probes(client)
    client._quote_leg = lambda leg, dx: dx * 3     # a venue, taught after
    assert [q.value for q in client.probe([probe_of(ArcKind.SWAP_UNIV4)])] == [300]
