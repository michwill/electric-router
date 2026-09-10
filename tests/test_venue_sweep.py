"""The A/B has to switch off the venue it says it is measuring.

`read_once` is the whole experiment: quote without the venue, with it, and
without it again.  If the toggle misses a venue, both arms are the same session
and every case reads exactly +0.00 bp -- a sweep that passes on all counts
while measuring nothing.  That is a more dangerous result than a crash, because
"0 worse" is what the harness is meant to print when the venue is safe.

The venues are faked here on purpose: what breaks is the bookkeeping around the
quote, not the quote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

venue_sweep = pytest.importorskip("venue_sweep")

from erouter.core.types import ArcKind  # noqa: E402


class Leg:
    def __init__(self, kind):
        self.kind = kind


class Route:
    def __init__(self, kinds):
        self.legs = [Leg(k) for k in kinds]


class Quote:
    def __init__(self, out, kinds=()):
        self.verified_out = out
        self.route = Route(kinds)


class Session:
    """Pays `with_venue` when any venue attribute is set, `base` otherwise."""

    def __init__(self, base, with_venue, kinds=(), drift=None):
        self.univ3 = None
        self.univ2 = None
        self._base, self._with, self._kinds = base, with_venue, kinds
        self._drift = list(drift or [])
        self.seen = []

    def quote(self, _amount):
        on = [a for a in ("univ3", "univ2") if getattr(self, a) is not None]
        self.seen.append(tuple(on))
        if on:
            return Quote(self._with, self._kinds)
        if self._drift:
            return Quote(self._drift.pop(0))
        return Quote(self._base)


def venues(*names):
    out = []
    for name in names:
        attr, kind, _cls = venue_sweep.VENUES[name]
        obj = type("Fake", (), {"arcs": [1, 2], "pools": {}})()
        out.append(venue_sweep.Venue(name, attr, kind, obj))
    return out


def test_every_requested_venue_is_switched_off_in_the_base_arm():
    """A toggle that misses one venue makes both arms identical."""
    session = Session(100, 110)
    session.univ3, session.univ2 = "held3", "held2"
    got = venue_sweep.read_once(session, venues("v3", "v2"), 1)
    assert got is not None
    assert session.seen == [(), ("univ3", "univ2"), ()], (
        "arms must be venue-free, both venues, venue-free")


def test_the_venues_are_put_back_after_a_reading():
    """The sweep quotes thousands of times on one session."""
    session = Session(100, 110)
    session.univ3, session.univ2 = "held3", "held2"
    venue_sweep.read_once(session, venues("v3", "v2"), 1)
    assert (session.univ3, session.univ2) == ("held3", "held2")


def test_measuring_one_venue_leaves_the_other_alone():
    """`--venue v2` must not silently also disable v3."""
    session = Session(100, 110)
    session.univ3 = "held3"
    venue_sweep.read_once(session, venues("v2"), 1)
    assert session.seen[0] == ("univ3",), "v3 stays on in the base arm"
    assert session.seen[1] == ("univ3", "univ2")


def test_legs_are_counted_for_every_selected_venue():
    session = Session(100, 110, kinds=(ArcKind.SWAP_UNIV3, ArcKind.SWAP_UNIV2,
                                       ArcKind.SWAP_STABLE))
    got = venue_sweep.read_once(session, venues("v3", "v2"), 1)
    assert got.legs == 2, "one v3 and one v2, and not the Curve leg"

    assert venue_sweep.read_once(session, venues("v2"), 1).legs == 1


def test_a_drifting_control_throws_the_case_out():
    """The A and the A' disagree, so the B between them means nothing."""
    session = Session(100, 110, drift=[100, 104])
    assert venue_sweep.read_once(session, venues("v3"), 1) is None


def test_the_delta_is_basis_points_against_the_venue_free_arm():
    session = Session(10_000, 10_010)
    got = venue_sweep.read_once(session, venues("v3"), 1)
    assert (got.base, got.out) == (10_000, 10_010)
    assert got.delta == pytest.approx(10.0)


def test_a_case_is_believed_once_two_readings_agree():
    session = Session(10_000, 10_010)
    _got, reads, agreed = venue_sweep.measure(
        session, venues("v3"), 1, repeats=3, agree_bp=0.5)
    assert agreed and len(reads) == 2


def test_a_case_that_never_repeats_is_reported_unstable():
    """-38.47 bp then +12.20 bp on the same commit and block, twice over."""
    session = Session(10_000, 10_010)
    moving = iter([10_010, 10_100, 11_000])
    session.quote = lambda _a, s=session: (
        Quote(next(moving)) if not any(
            getattr(s, x) is not None for x in ("univ3", "univ2"))
        else Quote(10_500))
    # Base moves every read, so no two deltas land within the tolerance.
    _got, reads, agreed = venue_sweep.measure(
        session, venues("v3"), 1, repeats=3, agree_bp=0.5)
    assert not agreed
    assert len(reads) <= 3


def test_the_venue_table_names_attributes_the_session_actually_has():
    """`setattr` on a typo is silent: it makes an attribute and changes nothing."""
    import inspect

    from erouter.chain.session import RouterSession
    params = inspect.signature(RouterSession.__init__).parameters
    for name, (attr, _kind, _cls) in venue_sweep.VENUES.items():
        assert attr in params, f"--venue {name} sets session.{attr}, which is gone"


def test_a_reading_keeps_the_route_that_produced_it():
    """The number cannot say why.

    A case that loses through a venue leg and one that loses by perturbing the
    base solve read identically as a delta; only the legs tell them apart, and
    at -5088 bp that distinction is the whole question.
    """
    session = Session(100, 60, kinds=(ArcKind.SWAP_UNIV2, ArcKind.SWAP_STABLE))
    got = venue_sweep.read_once(session, venues("v2"), 1)
    assert got.route is not None
    assert [leg.kind for leg in got.route.legs] == [
        ArcKind.SWAP_UNIV2, ArcKind.SWAP_STABLE]
    assert got.delta < 0 and got.legs == 1


def test_each_session_walks_the_pairs_in_a_different_order():
    """Two sessions marched through the same list settle the same way.

    They then agree with each other for the wrong reason -- the same experiment
    run twice, not a number confirmed.  The first version of this cross-check
    had exactly that hole, and would have certified the 5,088 bp reading it was
    written to catch.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(venue_sweep))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    reversed_calls = [n for n in ast.walk(fn)
                      if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == "reversed"]
    assert reversed_calls, (
        "every session walks the same order; the second one is not evidence")


def test_a_case_one_session_could_not_measure_is_dropped_not_halved():
    """A `None` from either table means the case has no cross-session claim."""
    tables = [{("a", "b", 1e3): venue_sweep.Reading(1.0, 10, 10, 0)},
              {("a", "b", 1e3): None}]
    seen = [tbl.get(("a", "b", 1e3)) for tbl in tables]
    assert any(r is None for r in seen), "the join must skip this case"


def test_the_gas_price_is_pinned_across_every_session():
    """§11.1 ranks gas-aware and `warm` takes whatever was live.

    The block is pinned and this was not, so two runs of the same commit rank
    against different gas -- measured, `FRAX -> WETH` at $100k reads -21.08 bp
    at 0.05 gwei and -0.00 bp at 1.0.  Even the two sessions of one run warm
    minutes apart.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(venue_sweep))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    writes = [n for n in ast.walk(fn)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == "gas_price_wei"
                      for t in n.targets)]
    assert writes, "no session has its gas pinned; runs cannot be compared"


def test_every_venue_offers_its_token_pairs():
    """The sweep must not know how a venue lays out its `pools` rows.

    It did, and read `meta[0]` as an address -- true for v2 and v3, and v4
    names a pool by `PoolKey` because it has no address to name it by.  The
    sweep raised on the first v4 run, before a single case was measured.
    """
    from erouter.venues.univ2_session import Univ2
    from erouter.venues.univ3_session import Univ3
    from erouter.venues.univ4_session import Univ4

    for cls in (Univ2, Univ3, Univ4):
        assert hasattr(cls, "token_pairs"), f"{cls.__name__} has no token_pairs"

    v2 = Univ2({}, )
    v2.pools = {"0xp": ("0xaa", "0xbb", 30, 18, 6)}
    assert v2.token_pairs() == [("0xaa", "0xbb")]

    from erouter.venues.univ4 import PoolKey
    v4 = Univ4({})
    v4.pools = {"0xid": (PoolKey("0xaa", "0xbb", 3000, 60, "0x" + "00" * 20), 18, 6)}
    assert v4.token_pairs() == [("0xaa", "0xbb")]


def test_the_sweep_does_not_reach_into_a_venue_s_rows():
    """A guard on the thing that broke, rather than on the venue that broke it."""
    import ast
    import inspect

    src = inspect.getsource(venue_sweep.token_set)
    tree = ast.parse(src.lstrip())
    text = ast.unparse(tree)
    assert "token_pairs()" in text
    assert "pools.values()" not in text, (
        "the sweep is reading a venue's row layout again")
