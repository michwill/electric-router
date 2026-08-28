"""The state cache is msgpack+zstd, and reads whatever it is handed.

It is fetched over HTTP by the browser, where the old format cost 5.1 MB and
96 ms against 1.5 MB and 32 ms.  lzma would have been smaller still and is a
trap: 120-156 ms in wasm, slower than the gzip it would replace.

Format is decided by the magic bytes, not the name.  `absorb` used to call
`gzip.decompress` unconditionally, so anything else raised `BadGzipFile` out of
the warm -- and `from_bytes` promises a slow warm rather than a failed one.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from erouter.chain.statecache import (
    GZIP_MAGIC,
    LEGACY_SUFFIX,
    SUFFIX,
    VERSION,
    ZSTD_MAGIC,
    StateCache,
)

ADDR = "0x" + "a1" * 20
OTHER = "0x" + "a2" * 20
DIGEST = "0x" + "b1" * 16


def filled(chain_id: int = 1, path=None) -> StateCache:
    cache = StateCache(chain_id=chain_id, path=path or Path("unused") / "x.msgpack")
    cache.accounts = {ADDR: {0, 1, 2**255 + 7}, OTHER: set()}
    cache.code_of = {ADDR: DIGEST}
    cache.code = {DIGEST: "60806040"}
    cache.funded = {ADDR}
    cache.pools = {ADDR, OTHER}
    cache.volatile = {OTHER}
    cache.wrapper_needs = {ADDR: {3}}
    cache.arc_needs = {ADDR: {4, 5}}
    cache.wrapper_sig = "deadbeef"
    cache.dirty = True
    return cache


def legacy_blob(chain_id: int = 1) -> bytes:
    """What the previous format wrote: json, gzipped, slots as integers."""
    payload = {
        "version": VERSION, "chain_id": chain_id,
        "accounts": {ADDR: [0, 1, 2**255 + 7], OTHER: []},
        "code_of": {ADDR: DIGEST}, "code": {DIGEST: "60806040"},
        "funded": [ADDR], "pools": [ADDR, OTHER], "volatile": [OTHER],
        "wrapper_needs": {ADDR: [3]}, "wrapper_sig": "deadbeef",
        "arc_needs": {ADDR: [4, 5]},
    }
    return gzip.compress(json.dumps(payload).encode())


def test_saving_writes_zstd(tmp_path):
    cache = filled()
    cache.path = tmp_path / f"ethereum{SUFFIX}"
    cache.save()
    assert cache.path.read_bytes()[:4] == ZSTD_MAGIC


def test_a_round_trip_keeps_every_field(tmp_path):
    cache = filled()
    cache.path = tmp_path / f"ethereum{SUFFIX}"
    cache.save()
    back = StateCache.load(1, "ethereum", tmp_path)
    assert back.accounts == cache.accounts       # includes a full-width slot key
    assert back.code == cache.code
    assert back.code_of == cache.code_of
    assert back.funded == cache.funded
    assert back.pools == cache.pools
    assert back.volatile == cache.volatile
    assert back.wrapper_needs == cache.wrapper_needs
    assert back.arc_needs == cache.arc_needs
    assert back.wrapper_sig == cache.wrapper_sig


def test_a_256_bit_slot_key_survives(tmp_path):
    # msgpack has no integer wider than 64 bits, which is why keys are bytes.
    cache = filled()
    cache.path = tmp_path / f"ethereum{SUFFIX}"
    cache.save()
    assert 2**255 + 7 in StateCache.load(1, "ethereum", tmp_path).accounts[ADDR]


def test_the_old_format_is_still_read():
    back = StateCache.from_bytes(1, legacy_blob())
    assert back.accounts[ADDR] == {0, 1, 2**255 + 7}
    assert back.code == {DIGEST: "60806040"}
    assert legacy_blob()[:2] == GZIP_MAGIC


def test_a_legacy_file_is_found_and_then_rewritten_in_the_new_format(tmp_path):
    (tmp_path / f"ethereum{LEGACY_SUFFIX}").write_bytes(legacy_blob())
    cache = StateCache.load(1, "ethereum", tmp_path)
    assert cache.accounts[ADDR] == {0, 1, 2**255 + 7}
    cache.dirty = True
    cache.save()
    assert (tmp_path / f"ethereum{SUFFIX}").exists()
    assert (tmp_path / f"ethereum{SUFFIX}").read_bytes()[:4] == ZSTD_MAGIC


def test_the_new_file_wins_when_both_are_there(tmp_path):
    (tmp_path / f"ethereum{LEGACY_SUFFIX}").write_bytes(legacy_blob())
    fresh = filled()
    fresh.code = {DIGEST: "5f5f"}
    fresh.path = tmp_path / f"ethereum{SUFFIX}"
    fresh.save()
    assert StateCache.load(1, "ethereum", tmp_path).code == {DIGEST: "5f5f"}


def test_something_that_is_neither_format_is_a_slow_warm_not_a_crash():
    assert StateCache.from_bytes(1, b"<!doctype html><title>404</title>").accounts == {}


def test_a_truncated_payload_does_not_raise():
    cache = filled()
    good = None
    import msgpack

    import erouter.chain.statecache as sc
    good = sc._zstd_compress(msgpack.packb(sc._to_binary({
        "version": VERSION, "chain_id": 1,
        "accounts": {ADDR: [0]}, "code_of": {}, "code": {},
        "funded": [], "pools": [], "volatile": [],
        "wrapper_needs": {}, "wrapper_sig": "", "arc_needs": {},
    }), use_bin_type=True))
    assert StateCache.from_bytes(1, good[: len(good) // 2]).accounts == {}
    assert cache.chain_id == 1


def test_another_chains_file_is_refused():
    blob_for_one = legacy_blob(chain_id=1)
    assert StateCache.from_bytes(137, blob_for_one).accounts == {}
