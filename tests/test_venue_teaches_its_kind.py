"""Every venue that puts arcs in the graph must teach the client its own kind.

This is the third time the same seam would have broken.  `--univ3` shipped with
its arcs never reaching the quoter; `--univ2` shipped teaching a client that was
never passed in; and v4 reuses v3's teacher, which gated on `SWAP_UNIV3` and
would have let a `SWAP_UNIV4` leg fall through.

All three fail identically and invisibly: the deployed quoter has never heard of
the kind, answers zero, `verify` reads zero as a revert, and every candidate
carrying the venue is dropped before it can win.  The output says the venue was
read and how many arcs it built, so it reads as the venue losing on the merits.
"""

from __future__ import annotations

import pytest

from erouter.core.types import ArcKind
from erouter.venues.univ2_client import teach as teach_v2
from erouter.venues.univ3_client import teach as teach_v3


class Leg:
    def __init__(self, kind, target="0x" + "11" * 20, i=0, j=1):
        self.kind, self.target, self.i, self.j = kind, target, i, j


class Client:
    """A quoter that has never heard of a venue kind, which is the real one."""

    def __init__(self):
        self.asked = []

    def _quote_leg(self, leg, dx):
        self.asked.append(leg.kind)
        return 0                      # what a real quoter returns for kind 19


class FakeBank:
    def quote(self, dx):
        return dx * 2


POOL = "0x" + "11" * 20


@pytest.mark.parametrize("kind", [ArcKind.SWAP_UNIV3, ArcKind.SWAP_UNIV4])
def test_a_tick_bank_answers_for_the_kind_it_was_taught(kind):
    """v4's banks are v3's banks: same math, same object, different kind."""
    client = Client()
    teach_v3(client, {(POOL, 0, 1): FakeBank()}, kind)
    assert client._quote_leg(Leg(kind), 100) == 200, (
        f"{kind.name} falls through to a quoter that answers zero, and a zero "
        f"is read as a revert")
    assert client.asked == [], "it should not have reached the real quoter"


def test_a_teacher_does_not_answer_for_a_kind_it_was_not_given():
    """Chaining is how two venues share one client, so each must decline the
    other's legs rather than swallow them."""
    client = Client()
    teach_v3(client, {(POOL, 0, 1): FakeBank()}, ArcKind.SWAP_UNIV4)
    client._quote_leg(Leg(ArcKind.SWAP_UNIV3), 100)
    assert client.asked == [ArcKind.SWAP_UNIV3], "delegated, not answered"


def test_two_venues_taught_to_one_client_each_keep_their_own():
    """The order they are taught in must not matter."""
    client = Client()
    teach_v3(client, {(POOL, 0, 1): FakeBank()}, ArcKind.SWAP_UNIV3)
    teach_v3(client, {(POOL, 0, 1): FakeBank()}, ArcKind.SWAP_UNIV4)
    assert client._quote_leg(Leg(ArcKind.SWAP_UNIV3), 10) == 20
    assert client._quote_leg(Leg(ArcKind.SWAP_UNIV4), 10) == 20
    assert client.asked == []


def test_every_off_chain_kind_has_a_teacher_that_claims_it():
    """A kind in `OFF_CHAIN_KINDS` cannot be sent, so it *must* be quotable
    locally -- otherwise it can only ever be dropped."""
    from erouter.core.types import OFF_CHAIN_KINDS

    claimed = set()
    for kind in OFF_CHAIN_KINDS:
        client = Client()
        if kind is ArcKind.SWAP_UNIV2:
            teach_v2(client, {POOL: object()})
        else:
            teach_v3(client, {(POOL, 0, 1): FakeBank()}, kind)
        # A taught client either answers or raises; what it must not do is
        # quietly return the zero the real quoter would.
        try:
            got = client._quote_leg(Leg(kind), 100)
        except Exception:
            claimed.add(kind)
            continue
        if got != 0 or not client.asked:
            claimed.add(kind)
    assert claimed == set(OFF_CHAIN_KINDS), (
        f"no teacher claims {set(OFF_CHAIN_KINDS) - claimed}; those legs would "
        f"be quoted as zero and read as reverts")


def test_collapse_folds_its_own_kind_and_leaves_the_others():
    """The fourth time this seam broke, and the first time `collapse` did it.

    v4's banks *are* v3's banks -- same object, same math -- so v4 folds through
    `univ3.collapse`.  With the kind hard-coded to `SWAP_UNIV3` that call did
    the wrong thing twice at once: v4's own arcs were left uncollapsed because
    their kind did not match, and v3's arcs were folded against v4's banks,
    which do not hold them.  With both venues live every major pair raised
    `KeyError('0x60594a405d53811d3bc4766596efd80fd545a270', 0, 1)` -- the v3
    DAI/WETH pool -- and dropped out of the sweep's token set entirely.
    """
    from erouter.venues import univ3

    class Arc:
        def __init__(self, kind, pool, i=0, j=1):
            self.kind, self.pool, self.i, self.j = kind, pool, i, j
            self.a = self.B = 0.0
            self.note = self.id = f"{pool}:{i}{j}"
            self.tau, self.sigma = 0, 1
            self.decimals_in = self.decimals_out = 18
            self.parallel = True
            self.rate_in = self.rate_out = 1.0
            self.cap = float("inf")

    v3_arc = Arc(ArcKind.SWAP_UNIV3, "0x" + "a3" * 20)
    v4_arc = Arc(ArcKind.SWAP_UNIV4, "0x" + "a4" * 20)
    # Only v4's bank is on offer, which is exactly the live arrangement when
    # `Univ4.collapse` runs over a graph that also carries v3 arcs.
    banks = {(v4_arc.pool, 0, 1): FakeBank()}

    # The v3 arc carries the flow and the v4 arc none, so a correct fold never
    # reaches a bank at all: v4's arc is its kind but has nothing to fold, and
    # v3's is not its kind.  Hard-coded to `SWAP_UNIV3` this raised `KeyError`
    # on the v3 pool, which is the bug.
    kept, _psi = univ3.collapse([v3_arc, v4_arc], [1.0, 0.0], [1.0, 1.0], None,
                                banks, kind=ArcKind.SWAP_UNIV4)
    kinds = [a.kind for a in kept]
    assert ArcKind.SWAP_UNIV3 in kinds, (
        "a v3 arc must pass through v4's fold untouched, not be looked up in "
        "banks that were never given it")
