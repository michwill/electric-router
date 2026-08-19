"""The committed runtime bytecode is what the source compiles to.

The override path injects this, and it is read from disk rather than compiled
because compiling needs boa and vyper -- neither of which exists under Pyodide,
where the Flet frontend runs.  That makes the committed copy load-bearing: if
`RouteQuoter.vy` changes and the hex does not, every chain without a deployed
quoter silently runs the *old* contract.

This is the test that stops that, and it is the right place for boa: a
developer running the suite has the compiler, the browser does not.
"""

from __future__ import annotations

import hashlib

import pytest

from erouter.dev import boa_host

boa = pytest.importorskip("boa", reason="compiling needs boa; the runtime path does not")


def test_the_committed_runtime_matches_the_source():
    fresh = boa_host._deployer().compiler_data.bytecode_runtime
    assert boa_host.runtime_bytecode() == fresh, (
        "data/quoter/RouteQuoter.runtime.hex is stale -- recompile it, or every "
        "chain without a deployed quoter runs the old contract"
    )


def test_the_recorded_source_hash_matches_the_source():
    """A second lock, so a hand-edited hex file is caught as well."""
    recorded = ""
    for line in boa_host.RUNTIME.read_text().splitlines():
        if line.startswith("# source sha256:"):
            recorded = line.split(":", 1)[1].strip()
    assert recorded, "the hex file should record which source it came from"
    assert recorded == hashlib.sha256(boa_host.CONTRACT.read_bytes()).hexdigest()


def test_reading_the_runtime_does_not_need_boa(monkeypatch):
    """The whole point: the override path must work with no compiler present."""
    import sys

    boa_host.runtime_bytecode.cache_clear()
    monkeypatch.setitem(sys.modules, "boa", None)   # importing it would now fail
    got = boa_host.runtime_bytecode()
    assert len(got) > 1000
    boa_host.runtime_bytecode.cache_clear()
