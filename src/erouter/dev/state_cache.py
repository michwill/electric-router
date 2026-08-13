"""What storage a pool reads, and the code that reads it -- kept on disk.

Warming a local EVM has three costs, and two of them are answers to questions
that do not change between blocks:

    access lists   4,302 ms   *which* slots a pool touches
    code+balance   1,841 ms   the bytecode that touches them
    storage        1,358 ms   what those slots contain *now*

Only the third is per-block.  A contract's storage layout is a property of its
bytecode, so the slot list for a pool is stable until the pool's code changes --
which for a Curve pool is never, since they are not upgradeable.  Cache the
first two and a session's warm becomes one storage sweep.

That makes the file worth committing: a new checkout starts warm, and a new
pool costs an access list for that pool alone rather than for the universe.
It is also why `code` is keyed by hash -- factory-deployed pools share
bytecode, so a few hundred pools resolve to a few dozen blobs.

**Some pools do not get to be cached.**  A LLAMMA reads a different set of band
slots as the price moves, so its "layout" is genuinely a function of state, not
just of code.  Those are listed as volatile and re-discovered every time; the
cost is one access list each, which is what the cache is saving everywhere else.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1
# Committed, so it has to be small and it has to diff.  Slots are integers and
# code is deduplicated by hash, which is what keeps a universe-wide file in the
# hundreds of kilobytes rather than the tens of megabytes.
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "evm-state"


@dataclass(slots=True)
class CacheStats:
    accounts: int = 0
    slots: int = 0
    code_blobs: int = 0
    pools_known: int = 0
    pools_new: int = 0
    volatile: int = 0


@dataclass(slots=True)
class StateCache:
    """Per-chain layout knowledge: which slots, whose code, what is volatile."""

    chain_id: int
    path: Path
    accounts: dict[str, set[int]] = field(default_factory=dict)
    code_of: dict[str, str] = field(default_factory=dict)  # account -> code hash
    code: dict[str, str] = field(default_factory=dict)  # code hash -> hex bytecode
    funded: set[str] = field(default_factory=set)
    pools: set[str] = field(default_factory=set)
    volatile: set[str] = field(default_factory=set)
    dirty: bool = False

    # ------------------------------------------------------------- load/save

    @classmethod
    def load(cls, chain_id: int, name: str, directory: Path | None = None) -> StateCache:
        path = (directory or DEFAULT_DIR) / f"{name}.json.gz"
        cache = cls(chain_id=chain_id, path=path)
        if not path.exists():
            return cache
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("version") != VERSION or raw.get("chain_id") != chain_id:
            return cache  # a stale format is not worth migrating; re-learn it
        cache.accounts = {a: set(s) for a, s in raw.get("accounts", {}).items()}
        cache.code_of = dict(raw.get("code_of", {}))
        cache.code = dict(raw.get("code", {}))
        cache.funded = set(raw.get("funded", []))
        cache.pools = set(raw.get("pools", []))
        cache.volatile = set(raw.get("volatile", []))
        return cache

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": VERSION,
            "chain_id": self.chain_id,
            "accounts": {a: sorted(s) for a, s in sorted(self.accounts.items())},
            "code_of": dict(sorted(self.code_of.items())),
            "code": dict(sorted(self.code.items())),
            "funded": sorted(self.funded),
            "pools": sorted(self.pools),
            "volatile": sorted(self.volatile),
        }
        tmp = self.path.with_suffix(".tmp")
        # `mtime=0` so an unchanged cache produces a byte-identical file: this
        # is committed, and a gzip header timestamp would make every save a diff.
        with gzip.GzipFile(tmp, "wb", mtime=0) as raw:
            raw.write(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        tmp.replace(self.path)
        self.dirty = False

    # ------------------------------------------------------------- questions

    def knows(self, pool: str) -> bool:
        pool = pool.lower()
        return pool in self.pools and pool not in self.volatile

    def unknown(self, pools) -> list[str]:
        """Which of these need an access list -- new, or volatile by nature."""
        return [p.lower() for p in dict.fromkeys(p.lower() for p in pools)
                if not self.knows(p)]

    def slots(self) -> dict[str, set[int]]:
        return {a: set(s) for a, s in self.accounts.items()}

    def bytecode(self, account: str) -> bytes | None:
        blob = self.code.get(self.code_of.get(account.lower(), ""), "")
        return bytes.fromhex(blob) if blob else None

    def stats(self) -> CacheStats:
        return CacheStats(
            accounts=len(self.accounts),
            slots=sum(len(s) for s in self.accounts.values()),
            code_blobs=len(self.code),
            pools_known=len(self.pools - self.volatile),
            volatile=len(self.volatile),
        )

    # --------------------------------------------------------------- updates

    def learn_slots(self, touched: dict[str, set[int]]) -> None:
        for address, keys in touched.items():
            address = address.lower()
            self.accounts.setdefault(address, set()).update(keys)
        self.dirty = True

    def learn_code(self, address: str, blob: bytes) -> None:
        address = address.lower()
        if not blob:
            return
        digest = "0x" + hashlib.sha256(blob).hexdigest()[:32]
        self.code[digest] = blob.hex()
        self.code_of[address] = digest
        self.dirty = True

    def learn_funded(self, address: str, balance: int) -> None:
        address = address.lower()
        if balance > 0:
            self.funded.add(address)
        else:
            self.funded.discard(address)
        self.dirty = True

    def learn_pools(self, pools) -> None:
        self.pools.update(p.lower() for p in pools)
        self.dirty = True

    def mark_volatile(self, pools) -> None:
        fresh = {p.lower() for p in pools} - self.volatile
        if fresh:
            self.volatile.update(fresh)
            self.dirty = True
