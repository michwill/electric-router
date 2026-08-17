"""Which pools reproduced their own `get_dy`, remembered between runs.

Admitting a pool is expensive and the answer barely changes.  The gate quotes
each candidate six times and compares against arithmetic, which is 2,300 calls
across the universe -- and what it establishes is that a pool's *code*
implements the maths we think it does.  Curve pools are not upgradeable, so
that verdict is good until either side of the comparison changes.

Keeping it matters for more than the calls it saves.  A verdict known *before
anything is warmed* is what lets the local EVM fetch storage for the pools it
still has to execute and skip the ones that will be computed -- which is 68% of
the slots it reads today.  Without it the ordering is circular: you cannot know
a pool is computable until you have run its `get_dy`, and running its `get_dy`
is what the storage is for.

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
Those are excluded at build time rather than trusted here, and a sample is
re-gated every run so that a systematic drift surfaces without waiting for a
route to go wrong.
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
    #: Pools checked and found *not* to reproduce.  Worth keeping too: they are
    #: the ones that must stay warm, and rediscovering that costs the same gate.
    refused: dict[str, str] = field(default_factory=dict)

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
            refused={k.lower(): v for k, v in blob.get("refused", {}).items()},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "version": VERSION,
            "chain_id": self.chain_id,
            "fingerprint": self.fingerprint or math_fingerprint(),
            "verdicts": dict(sorted(self.verdicts.items())),
            "refused": dict(sorted(self.refused.items())),
        }, indent=1, sort_keys=True) + "\n")

    # ------------------------------------------------------------- reading

    def get(self, pool: str) -> dict | None:
        return self.verdicts.get(pool.lower())

    def expected(self, pools) -> set[str]:
        """Addresses this cache expects to model, so warming can skip them.

        Deliberately excludes anything previously refused: those are exactly
        the pools that still have to be executed, and a stale "refused" costs
        a warm we would have done anyway.
        """
        known = self.verdicts
        return {p.address.lower() for p in pools if p.address.lower() in known}

    def record(self, pool: str, variant: dict) -> None:
        key = pool.lower()
        self.verdicts[key] = variant
        self.refused.pop(key, None)

    def refuse(self, pool: str, why: str) -> None:
        key = pool.lower()
        self.refused[key] = why[:80]
        self.verdicts.pop(key, None)

    def __len__(self) -> int:
        return len(self.verdicts)


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
