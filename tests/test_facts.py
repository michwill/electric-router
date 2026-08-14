"""Facts that hold until someone redeploys something.

Gas figures, arcs that quote and then revert, and whether a lending wrapper can
still be entered or only left -- none of it depends on the block, all of it
costs execution to learn, and the route path must never pay for it.  This is
about the storage keeping its promises: that a recovered pool stops being
banned, that a pool with one working direction is not thrown away for the sake
of a broken one, and that a chain stub cannot crash the lookup.
"""

from __future__ import annotations

import json

import pytest

from erouter.core.types import ArcKind
from erouter.dev.executability import revert_reason
from erouter.dev.facts import FactsCache


@pytest.fixture
def cache(tmp_path):
    return FactsCache(chain_id=1, path=tmp_path / "ethereum.json")


def test_broken_directions_survive_a_round_trip(cache):
    cache.learn_broken({cache.key("0xaa", ArcKind.SWAP_STABLE, 0, 1): "mint is paused"})
    cache.save()

    again = FactsCache.load(1, "ethereum", directory=cache.path.parent)
    assert again.is_broken("0xAA", ArcKind.SWAP_STABLE, 0, 1) == "mint is paused"
    assert again.is_broken("0xAA", ArcKind.SWAP_STABLE, 1, 0) == "", "one direction only"


def test_a_pool_that_starts_working_again_stops_being_banned(cache):
    """A protocol can be unpaused.  A stale entry would cost us the pool for
    good, which is worse than the revert it was recorded to avoid."""
    key = cache.key("0xaa", ArcKind.SWAP_STABLE, 0, 1)
    cache.learn_broken({key: "mint is paused"})
    assert cache.forget_broken([key]) == 1
    assert cache.is_broken("0xaa", ArcKind.SWAP_STABLE, 0, 1) == ""


def test_wrapper_capability_is_per_direction(cache):
    """Compound V2's mint is paused while its redeem settles -- the whole
    reason this is stored per direction rather than per protocol."""
    cache.learn_wrapper("0xcDaI", mint=False, redeem=True, note="mint is paused")
    cache.save()

    again = FactsCache.load(1, "ethereum", directory=cache.path.parent)
    entry = again.wrappers["0xcdai"]
    assert entry["mint"] is False
    assert entry["redeem"] is True
    assert entry["note"] == "mint is paused"


def test_the_file_stays_readable(cache):
    """It is committed and reviewed in diffs; that is why it is not gzipped."""
    cache.learn_broken({cache.key("0xaa", ArcKind.SWAP_STABLE, 0, 1): "frozen"})
    cache.save()
    raw = json.loads(cache.path.read_text())
    assert raw["broken"] == {"0xaa:0:0>1": "frozen"}
    assert cache.path.read_text().count("\n") > 3, "indented, not one long line"


def test_nothing_learned_means_nothing_written(cache):
    assert not cache.dirty
    cache.save()
    assert not cache.path.exists()


# --- what the universe does with it ---------------------------------------

class Stub:
    """A chain object with only what `_probed_dead_pools` may assume."""

    def __init__(self, name="ethereum", chain_id=1):
        self.name = name
        self.chain_id = chain_id


def test_a_chain_stub_cannot_crash_the_lookup():
    """`_apply_filters` is called with fakes in other tests, and a missing
    attribute must degrade to "no facts" rather than to a traceback."""
    from erouter.dev.universe import _probed_dead_pools

    assert _probed_dead_pools(object()) == set()
    assert _probed_dead_pools(Stub(name="", chain_id=0)) == set()


def test_a_recorded_revert_never_drops_a_pool(tmp_path, monkeypatch):
    """A broken direction is not a broken pool, and the difference is not
    academic: an earlier version of this dropped any pool with a revert and no
    recorded gas figure, which is every pool no route happened to choose.  It
    would have deleted both Compound pools -- whose `exchange_underlying`
    reverts while their cDAI/cUSDC arcs are routed through daily.

    Pool-level removal belongs to the hand-written blacklist, where a human
    decided it.  Nothing probed removes a pool on its own.
    """
    from erouter.dev import facts as facts_mod
    from erouter.dev.universe import _probed_dead_pools

    cache = FactsCache(chain_id=1, path=tmp_path / "ethereum.json")
    cache.learn_broken({
        cache.key("0xcomp", 14, 0, 1): "reverted without a reason",
        cache.key("0xcomp", 14, 1, 0): "reverted without a reason",
    })
    cache.save()
    monkeypatch.setattr(facts_mod, "DEFAULT_DIR", tmp_path)

    assert _probed_dead_pools(Stub()) == set()


def test_the_underlying_kind_does_not_collide_with_the_pools_own_arcs(tmp_path):
    """`exchange_underlying(0,1)` and `exchange(0,1)` are different calls on the
    same pool -- one reverts on Compound, the other settles.  Recording both
    under one key marked a healthy arc broken."""
    cache = FactsCache(chain_id=1, path=tmp_path / "ethereum.json")
    cache.learn_broken({cache.key("0xcomp", 14, 0, 1): "reverted"})
    assert cache.is_broken("0xcomp", ArcKind.SWAP_STABLE, 0, 1) == "", (
        "the pool's own wrapped-coin arc must be unaffected"
    )
    assert cache.is_broken("0xcomp", 14, 0, 1) == "reverted"


def test_no_facts_at_all_bans_nothing(tmp_path, monkeypatch):
    from erouter.dev import facts as facts_mod
    from erouter.dev.universe import _probed_dead_pools

    monkeypatch.setattr(facts_mod, "DEFAULT_DIR", tmp_path)
    assert _probed_dead_pools(Stub()) == set()


def test_revert_reasons_are_decoded():
    """Compound answers with a Solidity string; the raw blob is unreadable in a
    committed file, and the string is the whole value of recording it."""
    paused = ("Revert { gas_used: 63502, output: 0x08c379a0"
              + "00" * 31 + "20"
              + "00" * 31 + "0e"
              + b"mint is paused".hex() + "00" * 18 + " }")
    assert revert_reason(RuntimeError(paused)) == "mint is paused"
    assert revert_reason(RuntimeError("Revert { gas_used: 1, output: 0x }")) == (
        "reverted without a reason"
    )
