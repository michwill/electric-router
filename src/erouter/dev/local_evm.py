"""A `Transport` that executes locally, against state fetched once (§10).

Everything the router asks a chain is a read at one pinned block, and it asks
the same few hundred arcs over and over -- the probe ladder, the refine pass,
twenty candidates, the split search.  Paying a network round trip for each is
what shapes the algorithms upstream: rationed rounds, batched line searches,
sampled curves instead of exact evaluation.

So fetch the state once and run the EVM here.  Measured on this node, a
`get_dy` costs 30-449 us locally against ~0.8 ms marginal inside a batched
`quote_routes` on the wire, and the round trips go to zero.

**The state comes from `eth_createAccessList`, and that is what makes it
cheap.**  The access list is exactly the accounts and slots the call touches,
so one `eth_getProof` per account pulls all of them in a single reply.  Two
batched round trips warm a whole route.  Measured over four decades of trade
size on stableswap, stableswap-ng, tricrypto and tricrypto-ng, the touched set
does **not** vary with size -- so one list serves every probe of that arc.  It
is still gathered at several sizes and unioned, because "measured on five
pools" is not "true of every pool", and a LLAMMA crossing bands is exactly the
shape that would break it.

**The prefetch is an optimisation, not a correctness requirement.**  Measured
both ways on a deliberately incomplete 3pool: with no `fork_url` the call
*reverted*, and with one set it fetched the missing slots and matched the node
exactly.  So the fork stays wired up as a fallback and an incomplete or stale
slot list costs latency -- ~34 ms per slot, lazily -- rather than an answer.

That matters most for the state a cached slot list cannot predict.  A proxy
upgrade is visible, because the EIP-1967 implementation slot is itself in the
traced set and the per-block value fetch hands over the new address.  But a
stableswap-ng reading a Chainlink aggregator hits `s_transmissions[roundId]`,
and the round advances on every price update, so the *derived* key genuinely
differs between blocks; likewise a LLAMMA whose active band has moved.  Nothing
derivable from the old values detects those.  With the fallback they are simply
a few extra lazy reads; without it, a zero fee or a zero rate is a plausible
wrong number, which is the one outcome worth engineering against.

`strict=True` turns the fallback off, which is how the forked test asserts the
prefetch is actually complete rather than merely load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.transport import Answer, Call, Status

# A caller that is nobody, funded so `msg.sender` balance checks never bind.
CALLER = "0x" + "11" * 20
# Erigon refuses a JSON-RPC batch larger than this by default ("batch limit 100
# exceeded"), and it refuses the *whole* batch -- so a route's worth of proofs
# has to be split or none of it arrives.
BATCH_LIMIT = 100


class LocalEvmError(RuntimeError):
    pass


@dataclass(slots=True)
class WarmStats:
    accounts: int = 0
    slots: int = 0
    list_calls: int = 0
    fetch_calls: int = 0
    round_trips: int = 0
    ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocalEvm:
    """Read-only chain access at a pinned block, executed in-process.

    Implements `core.transport.Transport`, so `QuoterClient` and everything
    above it work unchanged -- including the quoter contract itself, whose
    runtime bytecode is inserted here exactly as an `eth_call` state override
    would inject it on the wire.
    """

    rpc: object
    strict: bool = False
    _evm: object = None
    _loaded: set[str] = field(default_factory=set)
    _listed: set[tuple[str, bytes]] = field(default_factory=set)
    _injected: set[str] = field(default_factory=set)
    _tracing: bool | None = None
    stats: WarmStats = field(default_factory=WarmStats)

    def __post_init__(self) -> None:
        from pyrevm import EVM, BlockEnv

        header = self.rpc.fetch("eth_getBlockByNumber", [self.rpc.pin.hex_block, False])
        self._evm = EVM(
            fork_url=None if self.strict else self.rpc.url,
            fork_block=None if self.strict else str(self.rpc.block),
            tracing=False, spec_id="CANCUN",
        )
        # pyrevm does not take the block env from anywhere, and zero is not a
        # harmless default: 3pool's `A()` ramps off `block.timestamp` and
        # underflows at zero, so every call would revert.
        self._evm.set_block_env(BlockEnv(
            number=int(header["number"], 16),
            timestamp=int(header["timestamp"], 16),
            basefee=int(header.get("baseFeePerGas", "0x0") or "0x0", 16),
            gas_limit=int(header["gasLimit"], 16),
            coinbase=header["miner"],
            prevrandao=bytes.fromhex((header.get("mixHash") or "0x" + "00" * 32)[2:]),
            excess_blob_gas=int(header.get("excessBlobGas", "0x0") or "0x0", 16),
        ))
        self._evm.set_balance(CALLER, 10 ** 24)

    # ------------------------------------------------------------- Transport

    @property
    def block(self) -> int:
        return self.rpc.block

    @property
    def chain_id(self) -> int:
        return self.rpc.chain_id

    def call(self, to: str, data: bytes, *, overrides: dict | None = None) -> bytes:
        self._inject(overrides)
        raw = self._evm.message_call(caller=CALLER, to=to, calldata=bytes(data))
        return bytes(raw)

    def call_many(self, calls: list[Call], *, overrides: dict | None = None) -> list[Answer]:
        self._inject(overrides)
        out: list[Answer] = []
        for one in calls:
            try:
                raw = bytes(self._evm.message_call(
                    caller=CALLER, to=one.to, calldata=bytes(one.data)))
            except Exception as exc:  # revm raises on revert
                out.append(Answer(Status.REVERTED, message=str(exc)[:120]))
                continue
            out.append(Answer(Status.VALUE if raw else Status.WRONG_ABI, raw))
        return out

    def _inject(self, overrides: dict | None) -> None:
        """Honour an `eth_call` code override by inserting the code locally."""
        if not overrides:
            return
        from pyrevm import AccountInfo

        for address, entry in overrides.items():
            code = entry.get("code") if isinstance(entry, dict) else None
            if not code or address in self._injected:
                continue
            raw = bytes.fromhex(code[2:] if code.startswith("0x") else code)
            self._evm.insert_account_info(address, AccountInfo(nonce=0, code=raw))
            self._evm.set_balance(address, 0)
            self._injected.add(address)

    # ----------------------------------------------------------------- warm

    def warm(self, calls: list[Call]) -> WarmStats:
        """Load every account and slot these calls touch.

        Two ways, and the difference is an order of magnitude.  `prestateTracer`
        hands back the accounts, code and storage *values* a call reads, in one
        message -- measured at 0.127 s each, so a route's worth is one batch.
        `eth_createAccessList` only names the slots, and pulling their values
        then costs an `eth_getProof` per account, which is Merkle proof
        generation the caller throws away: measured at 53 ms each, 665 accounts,
        36 seconds.  So trace when the node will, and fall back when it will
        not -- many hosted endpoints serve no `debug_*` at all.

        Calls already seen are skipped, so warming a session again costs nothing.
        """
        import time as _time

        started = _time.perf_counter()
        fresh = [c for c in calls if (c.to.lower(), bytes(c.data)) not in self._listed]
        if not fresh:
            return self.stats
        for one in fresh:
            self._listed.add((one.to.lower(), bytes(one.data)))

        if self._tracing is not False and self._warm_by_trace(fresh):
            self.stats.ms += (_time.perf_counter() - started) * 1000
            self.stats.accounts = len(self._loaded)
            return self.stats
        self._warm_by_proof(fresh)
        self.stats.ms += (_time.perf_counter() - started) * 1000
        self.stats.accounts = len(self._loaded)
        return self.stats

    def _batched(self, payloads: list[tuple[str, list]]) -> list:
        """`fetch_multi`, split to the node's batch ceiling.

        Erigon rejects the *whole* batch over its limit rather than truncating,
        so an unsplit route's worth of requests returns nothing at all.
        """
        out: list = []
        for lo in range(0, len(payloads), BATCH_LIMIT):
            out.extend(self.rpc.fetch_multi(payloads[lo:lo + BATCH_LIMIT]))
            self.stats.round_trips += 1
        return out

    def _warm_by_trace(self, calls: list[Call]) -> bool:
        """prestateTracer: state values straight back, no second fetch."""
        block = self.rpc.pin.hex_block
        payloads = [
            ("debug_traceCall",
             [{"from": CALLER, "to": one.to, "data": "0x" + bytes(one.data).hex()},
              block, {"tracer": "prestateTracer"}])
            for one in calls
        ]
        answers = self._batched(payloads)
        self.stats.list_calls += len(payloads)
        good = 0
        for answer in answers:
            if isinstance(answer, Exception) or not isinstance(answer, dict):
                self.stats.errors.append(f"traceCall: {str(answer)[:100]}")
                continue
            good += 1
            self._install_prestate(answer)
        if not good:
            self._tracing = False
            return False
        self._tracing = True
        return True

    def _install_prestate(self, state: dict) -> None:
        from pyrevm import AccountInfo

        for raw_address, entry in state.items():
            address = raw_address.lower()
            if not isinstance(entry, dict):
                continue
            if address not in self._loaded:
                code = entry.get("code") or ""
                blob = bytes.fromhex(code[2:]) if code.startswith("0x") else b""
                self._evm.insert_account_info(
                    address, AccountInfo(nonce=int(entry.get("nonce", 0) or 0),
                                         code=blob or None))
                self._evm.set_balance(address, int(entry.get("balance", "0x0") or "0x0", 16))
                self._loaded.add(address)
            for key, value in (entry.get("storage") or {}).items():
                self._evm.insert_account_storage(address, int(key, 16), int(value, 16))
                self.stats.slots += 1

    def _warm_by_proof(self, calls: list[Call]) -> None:
        """accessList names the slots; getProof fetches them.  The portable path."""
        block = self.rpc.pin.hex_block
        payloads = [
            ("eth_createAccessList",
             [{"from": CALLER, "to": one.to, "data": "0x" + bytes(one.data).hex()}, block])
            for one in calls
        ]
        touched: dict[str, set[int]] = {}
        for one, answer in zip(calls, self._batched(payloads), strict=True):
            touched.setdefault(one.to.lower(), set())
            if isinstance(answer, Exception) or not isinstance(answer, dict):
                self.stats.errors.append(f"accessList: {str(answer)[:100]}")
                continue
            for entry in answer.get("accessList", []):
                touched.setdefault(entry["address"].lower(), set()).update(
                    int(k, 16) for k in entry["storageKeys"])
        self.stats.list_calls += len(payloads)

        wanted = {a: keys for a, keys in touched.items() if a not in self._loaded or keys}
        if not wanted:
            return
        order = list(wanted)
        answers = self._batched(
            [("eth_getProof", [a, [f"0x{s:064x}" for s in sorted(wanted[a])], block])
             for a in order]
            + [("eth_getCode", [a, block]) for a in order if a not in self._loaded]
        )
        self.stats.fetch_calls += len(answers)
        proofs = answers[:len(order)]
        codes = {a: answers[len(order) + k]
                 for k, a in enumerate(a for a in order if a not in self._loaded)}
        self._install(order, proofs, codes)

    def _install(self, order, proofs, codes) -> None:
        """Insert fetched accounts and slots.  Failures are recorded, not raised."""
        from pyrevm import AccountInfo

        for address, proof in zip(order, proofs, strict=True):
            if isinstance(proof, Exception) or not isinstance(proof, dict):
                self.stats.errors.append(f"getProof {address[:10]}: {str(proof)[:90]}")
                continue
            if address not in self._loaded:
                code = codes.get(address)
                raw = b"" if isinstance(code, Exception) or not code else bytes.fromhex(code[2:])
                self._evm.insert_account_info(
                    address, AccountInfo(nonce=int(proof["nonce"], 16), code=raw or None))
                self._evm.set_balance(address, int(proof["balance"], 16))
                self._loaded.add(address)
            for item in proof.get("storageProof", []):
                self._evm.insert_account_storage(
                    address, int(item["key"], 16), int(item["value"], 16))
                self.stats.slots += 1

    # --------------------------------------------------------------- checks

    def verify_against(self, node, calls: list[Call]) -> list[tuple[Call, int, int]]:
        """Every call, both ways.  Returns the disagreements.

        The prefetch is the one part of this that can be silently incomplete,
        so it gets an explicit check rather than a comment saying it is fine.
        """
        bad: list[tuple[Call, int, int]] = []
        for one in calls:
            try:
                mine = int.from_bytes(self.call(one.to, one.data)[:32], "big")
            except Exception:
                mine = -1
            try:
                theirs = int.from_bytes(node.call(one.to, one.data)[:32], "big")
            except Exception:
                theirs = -1
            if mine != theirs:
                bad.append((one, mine, theirs))
        return bad


@dataclass(slots=True)
class Recorder:
    """A `Transport` that passes through and remembers what was asked.

    The access list has to be built from the calls that will actually be made,
    and the calls the router makes are quoter batches, not bare `get_dy` --
    so rather than re-deriving the contract's dispatch in Python, run once
    against the node and record.  One warm session then serves every later
    quote at that block.
    """

    inner: object
    calls: list[Call] = field(default_factory=list)
    seen: set[tuple[str, bytes]] = field(default_factory=set)

    @property
    def block(self) -> int:
        return self.inner.block

    @property
    def chain_id(self) -> int:
        return self.inner.chain_id

    def _note(self, to: str, data: bytes) -> None:
        key = (to.lower(), bytes(data))
        if key not in self.seen:
            self.seen.add(key)
            self.calls.append(Call(to, bytes(data)))

    def call(self, to: str, data: bytes, *, overrides: dict | None = None) -> bytes:
        self._note(to, data)
        return self.inner.call(to, data, overrides=overrides)

    def call_many(self, calls: list[Call], *, overrides: dict | None = None) -> list[Answer]:
        for one in calls:
            self._note(one.to, one.data)
        return self.inner.call_many(calls, overrides=overrides)

    def call_batch(self, requests, *, to: str, overrides: dict | None = None):
        # `run_batch` looks this up by name, so without it here `__getattr__`
        # hands over the inner transport's and every probe batch -- which is
        # most of what the router asks -- goes unrecorded.
        for data in requests:
            self._note(to, data)
        inner = getattr(self.inner, "call_batch", None)
        if inner is not None:
            return inner(requests, to=to, overrides=overrides)
        out = []
        for data in requests:
            try:
                out.append(self.inner.call(to, data, overrides=overrides))
            except Exception:
                out.append(None)
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)
