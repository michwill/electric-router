"""The two chunk sizes have to be the same number.

`RouterSession._batched` hands over `batch_size * max_streams` requests and
expects `max_streams` chunks of `batch_size` to go out; `fetch_multi` splits at
the *transport's* `batch_size`.  When those disagree the session sends batches
the endpoint refuses -- and refuses whole, so a sweep comes back empty rather
than slow.  These pin the reconciliation, not the speed.
"""

from __future__ import annotations

import asyncio

import pytest

from erouter.dev.rpc import BATCH_FLOOR, DEFAULT_BATCH, AsyncTransport


class _Pin:
    hex_block = "0x1"


class _Transport:
    """As much of `JsonRpcTransport` as the adapter touches."""

    pin = _Pin()

    def __init__(self, ceiling=100, *, raises=False):
        self.chain_id = 1
        self.batch_size = DEFAULT_BATCH
        self.max_streams = 8
        self._ceiling = ceiling
        self._raises = raises
        self.probed: list = []

    def probe_batch_limit(self, sample=None):
        self.probed.append(sample)
        if self._raises:
            raise RuntimeError("this endpoint will not say")
        return self._ceiling

    def fetch_multi(self, payloads, *, concurrent=False):
        return [f"0x{k:x}" for k, _ in enumerate(payloads)]


def test_both_sides_take_the_measured_ceiling():
    transport = _Transport(ceiling=100)

    rpc = AsyncTransport(transport)

    assert rpc.batch_size == 100
    assert transport.batch_size == 100, "fetch_multi chunks at this one"
    assert rpc.batch_size == transport.batch_size


def test_the_ceiling_is_measured_with_a_storage_read():
    """A limit learned from `eth_blockNumber` does not survive a sweep."""
    transport = _Transport()

    AsyncTransport(transport)

    assert [method for method, _ in transport.probed] == ["eth_getStorageAt"]


def test_a_roomier_endpoint_keeps_its_room():
    transport = _Transport(ceiling=1000)

    rpc = AsyncTransport(transport)

    assert (rpc.batch_size, transport.batch_size) == (1000, 1000)


def test_an_endpoint_that_will_not_say_keeps_the_floor():
    transport = _Transport(raises=True)

    rpc = AsyncTransport(transport)

    assert (rpc.batch_size, transport.batch_size) == (BATCH_FLOOR, BATCH_FLOOR)


def test_the_streams_are_declared_at_all():
    """Undeclared is what `_batched` reads as one, and one is the serial sweep."""
    rpc = AsyncTransport(_Transport())

    assert rpc.max_streams == 8
    assert AsyncTransport(_Transport(), streams=16).max_streams == 16


def test_the_step_is_a_whole_number_of_chunks():
    """What `_batched` computes, against what `fetch_multi` will do with it."""
    transport = _Transport(ceiling=100)
    rpc = AsyncTransport(transport)

    step = rpc.batch_size * rpc.max_streams
    chunks = -(-step // transport.batch_size)

    assert chunks == rpc.max_streams
    assert step % transport.batch_size == 0


def test_a_failed_call_raises_and_a_failed_batch_does_not():
    """`batch` reports per-request failure in the slot; `call` raises."""
    class _Broken(_Transport):
        def fetch_multi(self, payloads, *, concurrent=False):
            return [RuntimeError("nope") for _ in payloads]

    rpc = AsyncTransport(_Broken())

    got = asyncio.run(rpc.batch([("eth_getStorageAt", [])]))
    assert isinstance(got[0], Exception)
    with pytest.raises(RuntimeError):
        asyncio.run(rpc.call("eth_chainId", []))
