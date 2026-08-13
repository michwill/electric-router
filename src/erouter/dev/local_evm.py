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

**Prefetching and falling back are mutually exclusive here, by measurement.**
With `fork_url` set, revm loads an account from the fork *before* accepting an
insert, so every prefetched slot is fetched over the network anyway: warming 13
accounts and 39 slots cost 294 ms strict against 2,340 ms with the fork on, of
which 2,042 ms was the inserts alone.  A fallback therefore does not make the
prefetch safer, it makes it pointless.

So `strict` (the default) bulk-loads and accepts that a missed slot reads as
zero.  Measured on a deliberately incomplete 3pool that *reverted* rather than
returning a number, and a reverted probe is arc removal, which the pipeline
already handles -- but that is the shape of one pool, not a guarantee, and a
missing fee or rate would read as a plausible zero instead.  What makes it safe
is upstream: the router verifies its chosen route on the chain regardless, so a
stale prefetch costs route quality, never a wrong answer.

`strict=False` gives up the bulk load entirely and lets the fork serve every
read lazily at ~34 ms a slot.  Slow, and correct by construction; the mode to
reach for when a quote disagrees with the chain and the question is why.

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
# The storage sweep is a few thousand tiny independent reads, which is a very
# different shape from the probe batches the transport's default is tuned for.
# Measured on 4,136 slots: 4,513 ms on one stream, 1,238 on four, 521 on
# sixteen, 457 on thirty-two.  Sixteen takes 2.4x of the four-stream default;
# doubling again buys 12% more and twice the connections, which a hosted
# endpoint is entitled to object to.
PRIME_STREAMS = 16


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
    list_ms: float = 0.0
    code_ms: float = 0.0
    storage_ms: float = 0.0
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
    strict: bool = True
    cache: object = None
    _evm: object = None
    _loaded: set[str] = field(default_factory=set)
    _slots: dict[str, set[int]] = field(default_factory=dict)
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
        # Discovering the node's batch ceiling by failing into it is fine when
        # chunks go out one at a time -- the first failure teaches the rest.
        # Concurrently it is not: every in-flight chunk fails together, and the
        # halving retries collide.  So start at a ceiling every node accepts.
        limit = getattr(self.rpc, "batch_size", None)
        if isinstance(limit, int) and limit > BATCH_LIMIT:
            self.rpc.batch_size = BATCH_LIMIT
        streams = getattr(self.rpc, "max_streams", None)
        if isinstance(streams, int) and streams < PRIME_STREAMS:
            self.rpc.max_streams = PRIME_STREAMS

    # ------------------------------------------------------------- Transport

    #: Quotes execute in-process, so callers may spend them freely.
    local = True

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

    def prime(self, pools=()) -> WarmStats:
        """Load everything the disk cache already knows, then read values.

        No access lists and no `eth_getCode` for a pool that is already in the
        cache -- its layout and its bytecode are properties of code that does
        not change.  What is left is the storage sweep, which is per-block and
        irreducible, plus the balances of the few accounts that hold any.

        Returns without touching the network for any pool it does not know;
        `warm` discovers those.
        """
        import time as _t

        if self.cache is None:
            return self.stats
        started = _t.perf_counter()
        block = self.rpc.pin.hex_block
        wanted = self.cache.slots()
        if pools:
            reachable = self.cache.unknown(pools)
            self.stats.errors.extend(f"uncached pool {p[:10]}" for p in reachable[:3])
        for address in wanted:
            if address in self._loaded:
                continue
            blob = self.cache.bytecode(address)
            from pyrevm import AccountInfo

            self._evm.insert_account_info(address, AccountInfo(nonce=1, code=blob))
            self._loaded.add(address)
        self._read_values(wanted, block, funded=self.cache.funded)
        self.stats.ms += (_t.perf_counter() - started) * 1000
        self.stats.accounts = len(self._loaded)
        return self.stats

    def _read_values(self, wanted: dict, block: str, funded=None) -> None:
        """The only genuinely per-block traffic: slot values, and live balances."""
        import time as _t

        flat = [(a, slot) for a, slots in wanted.items() for slot in sorted(slots)]
        if flat:
            _mark = _t.perf_counter()
            values = self._batched(
                [("eth_getStorageAt", [a, f"0x{slot:064x}", block]) for a, slot in flat])
            self.stats.storage_ms += (_t.perf_counter() - _mark) * 1000
            self.stats.fetch_calls += len(values)
            for (address, slot), value in zip(flat, values, strict=True):
                if isinstance(value, Exception) or not isinstance(value, str):
                    self.stats.errors.append(f"getStorageAt {address[:10]}: {str(value)[:70]}")
                    continue
                self._evm.insert_account_storage(address, slot, int(value, 16))
                self._slots.setdefault(address, set()).add(slot)
                self.stats.slots += 1
        # Balances only where there is one to have.  Most pools hold no native
        # ETH at all, and the cache remembers which do -- but the ETH/stETH pool
        # does, and a zero there makes `get_dy` answer zero (E11).
        holders = sorted(set(funded or ()) & set(wanted)) if funded is not None else list(wanted)
        if holders:
            _mark = _t.perf_counter()
            got = self._batched([("eth_getBalance", [a, block]) for a in holders])
            self.stats.code_ms += (_t.perf_counter() - _mark) * 1000
            for address, balance in zip(holders, got, strict=True):
                if not isinstance(balance, Exception) and balance:
                    self._evm.set_balance(address, int(balance, 16))

    def warm(self, calls: list[Call]) -> WarmStats:
        """Load every account and slot these calls touch.

        `eth_createAccessList` names the slots and `eth_getStorageAt` reads
        their values, because neither computes anything the caller discards.
        `prestateTracer` returns the same state in one message but re-executes
        the call under a tracer to do it -- over a 600-probe batch that is tens
        of seconds -- so it is the fallback, for nodes that serve `debug_*` but
        not `eth_createAccessList`.

        Calls already seen are skipped, so warming a session again costs nothing.
        """
        import time as _time

        started = _time.perf_counter()
        if not self.strict:
            return self.stats  # every insert would cost a fork read; see above
        fresh = [c for c in calls if (c.to.lower(), bytes(c.data)) not in self._listed]
        if not fresh:
            return self.stats
        for one in fresh:
            self._listed.add((one.to.lower(), bytes(one.data)))

        if not self._warm_by_proof(fresh) and self._tracing is not False:
            # No access list came back at all -- the node may serve `debug_*`
            # instead.  This must not key off "did we load a new account": an
            # incremental warm legitimately adds only slots, and treating that
            # as failure re-traced every quote, which cost 93 s of 105 s.
            self._warm_by_trace(fresh)
        self.stats.ms += (_time.perf_counter() - started) * 1000
        self.stats.accounts = len(self._loaded)
        return self.stats

    def _batched(self, payloads: list[tuple[str, list]]) -> list:
        """`fetch_multi`, split to the node's batch ceiling.

        Erigon rejects the *whole* batch over its limit rather than truncating,
        so an unsplit route's worth of requests returns nothing at all.
        """
        if not payloads:
            return []
        self.stats.round_trips += max(1, (len(payloads) + BATCH_LIMIT - 1) // BATCH_LIMIT)
        try:
            return self.rpc.fetch_multi(payloads, concurrent=True)
        except TypeError:  # a transport that predates the keyword
            out: list = []
            for lo in range(0, len(payloads), BATCH_LIMIT):
                out.extend(self.rpc.fetch_multi(payloads[lo:lo + BATCH_LIMIT]))
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
                self._slots.setdefault(address, set()).add(int(key, 16))
                self.stats.slots += 1

    def _warm_by_proof(self, calls: list[Call]) -> bool:
        """accessList names the slots; getStorageAt fetches their values.

        Not `eth_getProof`: it returns many slots per account in one reply,
        which looks like the efficient choice and is not -- it computes a
        Merkle proof per account that this caller immediately discards.
        Measured on one route's 198 accounts and 1,212 slots: 2,343 ms by
        proof against 1,303 ms by plain storage reads, and the gap widens with
        account count because proofs are per-account while storage reads batch.

        `eth_getCode` is the real cost here -- 6,265 ms for those 198 accounts,
        because it ships whole contracts.  Code is immutable, so it is fetched
        once per address per process and belongs in a disk cache next.
        """
        block = self.rpc.pin.hex_block
        payloads = [
            ("eth_createAccessList",
             [{"from": CALLER, "to": one.to, "data": "0x" + bytes(one.data).hex()}, block])
            for one in calls
        ]
        import time as _t
        _mark = _t.perf_counter()
        listed = self._batched(payloads)
        self.stats.list_ms += (_t.perf_counter() - _mark) * 1000
        touched: dict[str, set[int]] = {}
        served = False
        for one, answer in zip(calls, listed, strict=True):
            touched.setdefault(one.to.lower(), set())
            if isinstance(answer, Exception) or not isinstance(answer, dict):
                self.stats.errors.append(f"accessList: {str(answer)[:100]}")
                continue
            served = True
            for entry in answer.get("accessList", []):
                touched.setdefault(entry["address"].lower(), set()).update(
                    int(k, 16) for k in entry["storageKeys"])
        self.stats.list_calls += len(payloads)
        if not served:
            return False

        # Only what is not already resident.  Without this an incremental warm
        # re-reads every slot of every account it has ever seen, which on five
        # successive quotes was 7,846 reads instead of a few hundred.
        if self.cache is not None:
            self.cache.learn_slots(touched)
        wanted = {}
        for address, keys in touched.items():
            fresh_keys = keys - self._slots.get(address, set())
            if fresh_keys or address not in self._loaded:
                wanted[address] = fresh_keys
        if not wanted:
            return True

        from pyrevm import AccountInfo

        cold = [a for a in wanted if a not in self._loaded]
        # Code only for what the disk cache cannot supply: `eth_getCode` ships
        # whole contracts and was 6,265 ms for 198 accounts, and code does not
        # change, so a committed cache removes this entirely on a warm repo.
        needs_code = [a for a in cold
                      if self.cache is None or self.cache.bytecode(a) is None]
        blobs: dict[str, bytes] = {}
        if needs_code:
            _mark = _t.perf_counter()
            answers = self._batched([("eth_getCode", [a, block]) for a in needs_code])
            self.stats.code_ms += (_t.perf_counter() - _mark) * 1000
            self.stats.fetch_calls += len(answers)
            for address, code in zip(needs_code, answers, strict=True):
                blob = b"" if isinstance(code, Exception) or not code \
                    else bytes.fromhex(code[2:])
                blobs[address] = blob
                if self.cache is not None and blob:
                    self.cache.learn_code(address, blob)
        for address in cold:
            blob = blobs.get(address)
            if blob is None and self.cache is not None:
                blob = self.cache.bytecode(address)
            self._evm.insert_account_info(address, AccountInfo(nonce=1, code=blob or None))
            self._loaded.add(address)

        self._read_values(wanted, block)
        if self.cache is not None:
            for address in cold:
                self.cache.learn_funded(address, self._evm.get_balance(address))
        return True


    def warm_arcs(self, refs, quoter: str, grid=None, per_call: int = 200) -> WarmStats:
        """Discover state for specific arcs -- the ones the cache has not seen.

        A new pool costs an access list over a probe batch covering only its own
        arcs, so keeping up with a moving universe is proportional to what
        moved.  The batch is the same shape the router itself sends, which is
        what makes one list cover every size that arc will ever be quoted at.
        """
        from ..core.codec import encode_call
        from ..core.probe import COARSE_GRID, plan_grid
        from ..core.quoter import SIG_PROBE_BATCH

        plan = plan_grid(list(refs), grid or COARSE_GRID)
        calls = [
            Call(quoter, encode_call(
                SIG_PROBE_BATCH,
                [p.as_tuple() for p in plan.probes[lo:lo + per_call]]))
            for lo in range(0, len(plan.probes), per_call)
        ]
        return self.warm(calls) if calls else self.stats

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
