"""JSON-RPC transport over stdlib urllib.

Implements `core.transport.Transport`.  Three things here are load-bearing and
were each established by measurement against the local Erigon node:

1.  **Never send a `gas` field.**  This node runs `rpc.gascap = 0` and answers a
    13,872-call batch when `gas` is omitted, but returns "out of gas" for an
    explicit `gas: 50_000_000` -- which is geth's *default* cap.  Omitting the
    field gets you the node's own limit, whatever it is.
2.  **Batch.**  A sequential `eth_getStorageAt` costs 33.6 ms even locally
    (per-request scheduling, not per-slot work); 200 of them in one JSON-RPC
    batch take 68 ms total.  Everything goes through `fetch_multi`.
3.  **Empty returndata is not a value.**  A Curve pool that lacks a function
    returns `0x` rather than reverting, so `Status.WRONG_ABI` is distinct from
    `Status.REVERTED`.  Six mainnet arcs are mis-typed by the Curve API today
    and are only caught by keeping them apart.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..core.codec import decode, selector
from ..core.transport import Answer, Call, Status

USER_AGENT = "electric-router/0.1"

# JSON-RPC batches are chunked by both count and total calldata size; a chunk
# that fails outright is halved and retried before falling back per call.
DEFAULT_BATCH = 500
MAX_BATCH_BYTES = 16 << 20


class RpcError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


@dataclass(frozen=True, slots=True)
class Pin:
    """A block, resolved exactly once and threaded through every stage.

    Nothing downstream may see "latest": all candidates must be quoted at the
    same block or the winner is noise.
    """

    chain_id: int
    block: int
    url: str

    @property
    def hex_block(self) -> str:
        return hex(self.block)


_ERROR_SEL = selector("Error(string)")
_PANIC_SEL = selector("Panic(uint256)")


def revert_reason(data: Any) -> str:
    """Best-effort decode of revert returndata (same shape as yb-core's helper)."""
    if isinstance(data, str) and data.startswith("0x"):
        try:
            raw = bytes.fromhex(data[2:])
        except ValueError:
            return data
        if raw[:4] == _ERROR_SEL:
            try:
                return str(decode(["string"], raw[4:])[0])
            except Exception:
                return data
        if raw[:4] == _PANIC_SEL:
            try:
                return f"Panic(0x{decode(['uint256'], raw[4:])[0]:x})"
            except Exception:
                return data
        return data
    return "" if data is None else str(data)


class JsonRpcTransport:
    """Read-only JSON-RPC client pinned to a block."""

    def __init__(
        self,
        url: str,
        block: int | str = "latest",
        *,
        timeout: float = 300.0,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self._id = 0
        chain_id = int(self.fetch("eth_chainId", []), 16)
        resolved = (
            int(self.fetch("eth_blockNumber", []), 16) if block == "latest" else int(block)
        )
        self.pin = Pin(chain_id=chain_id, block=resolved, url=url)

    # ---------------------------------------------------------------- Transport

    @property
    def block(self) -> int:
        return self.pin.block

    def gas_price(self) -> int:
        """Live gas price in wei, or 0 if the node will not say."""
        try:
            return int(self.fetch("eth_gasPrice", []), 16)
        except Exception:
            return 0

    @property
    def chain_id(self) -> int:
        return self.pin.chain_id

    def call(self, to: str, data: bytes, *, overrides: dict | None = None) -> bytes:
        params: list[Any] = [self._tx(to, data), self.pin.hex_block]
        if overrides:
            params.append(overrides)
        return _to_bytes(self.fetch("eth_call", params))

    def call_many(self, calls: list[Call], *, overrides: dict | None = None) -> list[Answer]:
        payloads = []
        for c in calls:
            params: list[Any] = [self._tx(c.to, c.data), self.pin.hex_block]
            if overrides:
                params.append(overrides)
            payloads.append(("eth_call", params))
        return [_answer(r) for r in self.fetch_multi(payloads)]

    # -------------------------------------------------------------------- wire

    @staticmethod
    def _tx(to: str, data: bytes) -> dict[str, str]:
        # No "gas" key -- see the module docstring.
        return {"to": to, "data": "0x" + data.hex()}

    def _post(self, body: Any) -> Any:
        payload = json.dumps(body).encode()
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RpcError(f"transport failure: {exc}") from exc

    def fetch(self, method: str, params: list[Any]) -> Any:
        self._id += 1
        result = self._post({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        if isinstance(result, dict) and result.get("error"):
            err = result["error"]
            raise RpcError(err.get("message", str(err)), err.get("code"), err.get("data"))
        return result["result"]

    def fetch_multi(self, payloads: list[tuple[str, list[Any]]]) -> list[Any]:
        """Batched JSON-RPC.  Returns one entry per payload, in order.

        An entry is the raw result, or an `RpcError` for a per-call failure --
        never raised, because a failed quote is arc removal, not an error.
        """
        out: list[Any] = [None] * len(payloads)
        for lo, hi in _chunks(payloads, self.batch_size):
            self._fetch_chunk(payloads, lo, hi, out)
        return out

    def _fetch_chunk(
        self, payloads: list[tuple[str, list[Any]]], lo: int, hi: int, out: list[Any]
    ) -> None:
        body = []
        for i in range(lo, hi):
            self._id += 1
            method, params = payloads[i]
            body.append({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        first_id = body[0]["id"]
        try:
            responses = self._post(body)
            if not isinstance(responses, list):
                raise RpcError(f"expected a batch response, got {type(responses).__name__}")
            by_id = {r.get("id"): r for r in responses}
            for i in range(lo, hi):
                entry = by_id.get(first_id + (i - lo))
                if entry is None:
                    out[i] = RpcError("missing from batch response")
                elif entry.get("error"):
                    err = entry["error"]
                    out[i] = RpcError(
                        err.get("message", str(err)), err.get("code"), err.get("data")
                    )
                else:
                    out[i] = entry.get("result")
        except RpcError as exc:
            if hi - lo == 1:
                out[lo] = exc
                return
            # Halve and retry: an oversized or out-of-gas batch is the common
            # cause, and one bad call must not drop its 499 neighbours.
            mid = (lo + hi) // 2
            self._fetch_chunk(payloads, lo, mid, out)
            self._fetch_chunk(payloads, mid, hi, out)

    # ------------------------------------------------------------ capabilities

    def supports_state_override(self) -> bool:
        """Can we inject the quoter's bytecode instead of deploying it?"""
        scratch = "0x" + "ee" * 20
        # PUSH1 0x2a PUSH1 0x00 MSTORE PUSH1 0x20 PUSH1 0x00 RETURN
        code = "0x602a60005260206000f3"
        try:
            result = self.fetch(
                "eth_call",
                [{"to": scratch, "data": "0x"}, self.pin.hex_block, {scratch: {"code": code}}],
            )
        except RpcError:
            return False
        return _to_bytes(result)[-1:] == b"\x2a"

    def supports_debug_trace(self) -> bool:
        """prestateTracer availability decides boa's fork prefetch setting."""
        try:
            self.fetch(
                "debug_traceCall",
                [
                    {"to": "0x" + "00" * 20, "data": "0x"},
                    self.pin.hex_block,
                    {"tracer": "prestateTracer"},
                ],
            )
        except RpcError:
            return False
        return True

    def supports_batching(self) -> bool:
        results = self.fetch_multi([("eth_chainId", []), ("eth_blockNumber", [])])
        return all(not isinstance(r, RpcError) for r in results)


# ----------------------------------------------------------------- helpers


def _chunks(items: list[Any], size: int):
    """Yield (lo, hi) index ranges bounded by count and serialized size."""
    lo = 0
    while lo < len(items):
        hi = lo
        nbytes = 0
        while hi < len(items) and hi - lo < size:
            nbytes += len(json.dumps(items[hi]))
            if nbytes > MAX_BATCH_BYTES and hi > lo:
                break
            hi += 1
        yield lo, hi
        lo = hi


def _to_bytes(value: Any) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x"):
        return b""
    return bytes.fromhex(value[2:])


def _answer(result: Any) -> Answer:
    if isinstance(result, RpcError):
        message = revert_reason(result.data) or result.message
        if "transport failure" in result.message or "missing from batch" in result.message:
            return Answer(Status.MISSING, message=result.message)
        return Answer(Status.REVERTED, message=message)
    raw = _to_bytes(result)
    if len(raw) == 0:
        return Answer(Status.WRONG_ABI)
    return Answer(Status.VALUE, raw)
