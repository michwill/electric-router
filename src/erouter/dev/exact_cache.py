"""Which pools reproduced their own `get_dy`, remembered between runs.

Admitting a pool is expensive and the answer barely changes.  The gate quotes
each candidate six times and compares against arithmetic, which is 2,300 calls
across the universe -- and what it establishes is that a pool's *code*
implements the maths we think it does.  Curve pools are not upgradeable, so
that verdict is good until either side of the comparison changes.

What it saves is the gate and nothing else.  It does *not* let the local EVM
skip those pools' storage: they are computed from that same storage, so
skipping the sweep relocates the read onto the wire rather than removing it --
measured, and it turned startup into minutes.  See `tests/test_startup_cost.py`.

**What voids a verdict.**  Anything that changes either side of the comparison:

* the maths on our side -- hence `fingerprint`, over the source of every module
  that participates.  Edit `core/stableswap.py` and every stableswap verdict on
  every chain is discarded, which is the behaviour you want the day a rounding
  fix changes one pool in a thousand.
* the parameters on the pool's side.  These are *not* cached: `A`, `gamma`, the
  fee terms, the ramp state and the balances are re-read every run, and a pool
  mid-ramp is refused there as before.  The verdict records which invariant and
  which variant matched, never the numbers.

**What it does not protect against** is a pool whose behaviour changes without
its code or its readable parameters changing -- an external fee policy, say.
Those are excluded at build time rather than trusted here: a twocrypto pool
with a `POLICY` contract is refused before the gate, because its fee can vary
with trade size and one probe would agree at the size it was taken.

`trust` takes a `resample` set so a caller can force pools back through the
gate.  Nothing passes one today; it is there for a scheduled audit, and saying
so is better than implying a check that does not run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "exact"

#: Every module whose source decides whether a pool reproduces its own quote:
#: the invariants themselves, and the readers that choose which variant to try.
MATH_SOURCES = (
    ("core", "stableswap.py"),
    ("core", "twocrypto.py"),
    ("core", "tricrypto.py"),
    ("core", "cryptoswap.py"),
    ("dev", "stable_params.py"),
    ("dev", "twocrypto_params.py"),
    ("dev", "tricrypto_params.py"),
)


def math_fingerprint() -> str:
    """A digest of the maths, so editing it discards every stale verdict."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package, name in MATH_SOURCES:
        path = root / package / name
        digest.update(name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            # A missing module is a real difference, not a reason to fall back
            # on a fingerprint that would match a tree that still has it.
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


@dataclass(slots=True)
class ExactCache:
    """Per-chain: pool -> the variant that reproduced it, or nothing yet."""

    chain_id: int
    path: Path
    fingerprint: str = ""
    verdicts: dict[str, dict] = field(default_factory=dict)
    #: Pools checked this run and found *not* to reproduce, with why.  Held
    #: for reporting and **never written**: the reason carries the mismatching
    #: wei, which moves with the block, so persisting it rewrote a committed
    #: file on every route.
    refused: dict[str, str] = field(default_factory=dict)
    #: Pools that failed, keyed by the balances they failed at.
    #:
    #: The reason is not written -- that is what churned the file -- but the
    #: *fact* is, because re-deriving it costs a probe per size per direction
    #: on every run.  Once the verdicts are warm those probes are the entire
    #: remaining gate: measured at 94 of 94, all of them on pools that will
    #: never pass.
    #:
    #: Keyed by balances because that is what makes forgetting automatic.  A
    #: pool that answers nothing is empty or holds dust, and a pool in that
    #: state does not trade, so its balances sit still and the record stays
    #: valid without being refreshed.  The moment someone deposits, the key
    #: stops matching and it is checked again -- which is exactly when the
    #: answer might have changed.  A pool that merely disagreed is covered by
    #: the same key for the same reason, and by the maths fingerprint above
    #: for the other reason it might start passing.
    unquotable: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, chain_id: int, name: str, directory: Path | None = None):
        path = (directory or DEFAULT_DIR) / f"{name}.json"
        current = math_fingerprint()
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls(chain_id=chain_id, path=path, fingerprint=current)
        if blob.get("version") != VERSION or blob.get("fingerprint") != current:
            # Not an error and not worth warning about -- the maths moved, so
            # every pool gets checked again and the file is rewritten.
            return cls(chain_id=chain_id, path=path, fingerprint=current)
        return cls(
            chain_id=chain_id,
            path=path,
            fingerprint=current,
            verdicts={k.lower(): v for k, v in blob.get("verdicts", {}).items()},
            unquotable={k.lower(): v
                        for k, v in blob.get("unquotable", {}).items()},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "version": VERSION,
            "chain_id": self.chain_id,
            "fingerprint": self.fingerprint or math_fingerprint(),
            "verdicts": dict(sorted(self.verdicts.items())),
            "unquotable": dict(sorted(self.unquotable.items())),
        }, indent=1, sort_keys=True) + "\n")

    # ------------------------------------------------------------- reading

    def get(self, pool: str) -> dict | None:
        return self.verdicts.get(pool.lower())

    def record(self, pool: str, variant: dict) -> None:
        key = pool.lower()
        self.verdicts[key] = variant
        self.refused.pop(key, None)
        self.unquotable.pop(key, None)

    def refuse(self, pool: str, why: str, balances=None) -> None:
        key = pool.lower()
        self.refused[key] = why[:80]
        self.verdicts.pop(key, None)
        if balances is not None:
            self.unquotable[key] = balance_key(balances)

    def skip(self, pool: str, balances) -> bool:
        """Whether this pool failed before, in the state it is in now."""
        got = self.unquotable.get(pool.lower())
        return got is not None and got == balance_key(balances)

    def __len__(self) -> int:
        return len(self.verdicts)


def balance_key(balances) -> str:
    """A short digest of a pool's balances, as the state a failure belongs to."""
    raw = ",".join(str(int(b)) for b in (balances or ()))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def trust(out, cache, resample, built, key: str) -> bool:
    """Admit a pool on a remembered verdict.  True if it needs no gate.

    `built` is every variant constructed for every pool this run, as
    `(pool, model, variant)`.  The verdict is matched against those rather than
    used to construct one, which is what makes a stale entry harmless: a
    remembered variant that is no longer on offer -- a pool that stopped
    reporting `stored_rates`, a reader that changed what it builds -- simply
    finds no match, and the pool falls through to the gate as if it had never
    been cached.
    """
    if cache is None or key in resample:
        return False
    verdict = cache.get(key)
    if verdict is None:
        return False
    for pool, model, variant in built:
        if pool.address.lower() == key and variant == verdict:
            out.by_pool[key] = model
            out.trusted += 1
            return True
    return False
