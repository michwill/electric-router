"""The committed storage-layout cache -- no chain.

What is cached has to be exactly the part that does not change between blocks:
which slots a pool reads, and the code that reads them.  Values must never be,
and a pool whose layout is a function of state must be able to opt out.
"""

from __future__ import annotations

import gzip
import json

import pytest

from erouter.dev.state_cache import VERSION, StateCache

POOL = "0x" + "a1" * 20
OTHER = "0x" + "b2" * 20
DEP = "0x" + "c3" * 20


def build(tmp_path) -> StateCache:
    cache = StateCache.load(1, "test", tmp_path)
    cache.learn_slots({POOL: {0, 1, 5}, DEP: {2}})
    cache.learn_code(POOL, b"\x60\x80\x60\x40")
    cache.learn_code(DEP, b"\x60\x80\x60\x40")  # same bytecode, one blob
    cache.learn_funded(POOL, 10 ** 18)
    cache.learn_funded(DEP, 0)
    cache.learn_pools([POOL, DEP])
    return cache


def test_it_round_trips_through_disk(tmp_path):
    build(tmp_path).save()
    back = StateCache.load(1, "test", tmp_path)
    assert back.accounts == {POOL.lower(): {0, 1, 5}, DEP.lower(): {2}}
    assert back.bytecode(POOL) == b"\x60\x80\x60\x40"
    assert back.funded == {POOL.lower()}
    assert back.knows(POOL)


def test_identical_bytecode_is_stored_once(tmp_path):
    cache = build(tmp_path)
    assert cache.stats().accounts == 2
    assert cache.stats().code_blobs == 1, "factory pools share code; storing it twice is waste"


def test_a_new_pool_is_the_only_thing_that_needs_learning(tmp_path):
    cache = build(tmp_path)
    assert cache.unknown([POOL, DEP]) == []
    assert cache.unknown([POOL, OTHER]) == [OTHER.lower()]


def test_a_volatile_pool_is_never_considered_known(tmp_path):
    """A LLAMMA's touched set moves with the price, so its layout cannot cache."""
    cache = build(tmp_path)
    cache.mark_volatile([POOL])
    assert not cache.knows(POOL)
    assert cache.unknown([POOL, DEP]) == [POOL.lower()]
    assert cache.stats().volatile == 1


def test_no_slot_values_are_ever_written(tmp_path):
    """The file is committed; a stored value would be wrong one block later."""
    cache = build(tmp_path)
    cache.save()
    with gzip.open(cache.path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert set(raw) == {"version", "chain_id", "accounts", "code_of", "code",
                        "funded", "pools", "volatile"}
    assert raw["version"] == VERSION
    # `accounts` maps to slot *keys* only.
    assert all(isinstance(k, int) for keys in raw["accounts"].values() for k in keys)


def test_saving_twice_produces_an_identical_file(tmp_path):
    """It is committed, so an unchanged cache must not show up as a diff."""
    cache = build(tmp_path)
    cache.save()
    first = cache.path.read_bytes()
    again = StateCache.load(1, "test", tmp_path)
    again.dirty = True
    again.save()
    assert again.path.read_bytes() == first


def test_a_format_change_re_learns_rather_than_guessing(tmp_path):
    cache = build(tmp_path)
    cache.save()
    with gzip.open(cache.path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["version"] = VERSION + 1
    with gzip.GzipFile(cache.path, "wb", mtime=0) as handle:
        handle.write(json.dumps(raw).encode())
    assert StateCache.load(1, "test", tmp_path).accounts == {}


def test_a_cache_from_another_chain_is_not_used(tmp_path):
    build(tmp_path).save()
    assert StateCache.load(999, "test", tmp_path).accounts == {}


def test_an_unsaved_clean_cache_writes_nothing(tmp_path):
    cache = StateCache.load(1, "test", tmp_path)
    cache.save()
    assert not cache.path.exists()


def test_missing_bytecode_is_absent_not_empty(tmp_path):
    cache = build(tmp_path)
    assert cache.bytecode(OTHER) is None
    with pytest.raises(AssertionError):
        assert cache.bytecode(OTHER) == b""
