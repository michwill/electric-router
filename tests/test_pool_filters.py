"""Curve's pool_filters list, applied to a cached universe.

Today `/v2/pools` already excludes every flagged pool, so the filter drops
nothing on a fresh load -- measured at min_tvl=0: 2,211 pools returned, 0
flagged among them.  The case it exists for is the *cached* one: our universe
cache can serve a snapshot taken before a pool was flagged, and Curve's list
moves on Curve's schedule, not ours.  So the filter runs on the cache and
stale paths too, and that is what these tests pin.
"""

from __future__ import annotations

from erouter.dev.universe import _apply_filters


class FakeApi:
    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked
        self.calls = 0

    def pool_filters(self, chain_id: int) -> set[str]:
        self.calls += 1
        return self.blocked


class FakeChain:
    chain_id = 1


class FakePool:
    def __init__(self, address: str) -> None:
        self.address = address


def test_flagged_pools_are_dropped():
    pools = [FakePool("0xAAA"), FakePool("0xBbB"), FakePool("0xCCC")]
    warnings: list[str] = []
    kept, dropped = _apply_filters(pools, FakeChain(), FakeApi({"0xbbb"}), warnings)

    assert dropped == 1
    assert [p.address for p in kept] == ["0xAAA", "0xCCC"]
    assert warnings and "pool_filters" in warnings[0]


def test_matching_is_case_insensitive():
    """The API returns checksummed addresses; our specs are mixed case."""
    pools = [FakePool("0xDeAdBeEf"), FakePool("0xFEED")]
    kept, dropped = _apply_filters(pools, FakeChain(), FakeApi({"0xdeadbeef"}), [])
    assert dropped == 1
    assert [p.address for p in kept] == ["0xFEED"]


def test_an_unreachable_filter_list_does_not_empty_the_universe():
    """Routing unfiltered is worse than routing filtered, and far better than
    not routing.  Every quote is verified on-chain regardless."""
    pools = [FakePool("0xAAA"), FakePool("0xBBB")]
    warnings: list[str] = []
    kept, dropped = _apply_filters(pools, FakeChain(), FakeApi(set()), warnings)

    assert dropped == 0
    assert len(kept) == 2
    assert not warnings
