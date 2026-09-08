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
    _delta, _base, _out, legs = venue_sweep.read_once(
        session, venues("v3", "v2"), 1)
    assert legs == 2, "one v3 and one v2, and not the Curve leg"

    _d, _b, _o, only_v2 = venue_sweep.read_once(session, venues("v2"), 1)
    assert only_v2 == 1


def test_a_drifting_control_throws_the_case_out():
    """The A and the A' disagree, so the B between them means nothing."""
    session = Session(100, 110, drift=[100, 104])
    assert venue_sweep.read_once(session, venues("v3"), 1) is None


def test_the_delta_is_basis_points_against_the_venue_free_arm():
    session = Session(10_000, 10_010)
    delta, base, out, _legs = venue_sweep.read_once(session, venues("v3"), 1)
    assert (base, out) == (10_000, 10_010)
    assert delta == pytest.approx(10.0)


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
