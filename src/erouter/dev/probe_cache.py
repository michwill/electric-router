"""Per-block probe memoisation.

Wraps a `QuoterClient` so an identical batch of probes at the same block is
answered from disk.  Everything the grid measures is a pure function of
(pool state at that block, size), so caching it is exact rather than an
approximation -- which is why the block must be pinned for this to be sound.

Sits in `dev` and implements the same surface as `QuoterClient`, so `core`
neither knows nor cares: the browser build simply does not wrap it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..core.quoter import Quote, QuoterClient
from ..core.transport import Status
from ..core.types import Probe
from .cache import Cache

_STATUS_BY_NAME = {s.name: s for s in Status}


def digest(probes: list[Probe]) -> str:
    """Stable fingerprint of a probe batch."""
    hasher = hashlib.sha256()
    for probe in probes:
        hasher.update(
            f"{probe.pool.lower()}:{int(probe.kind)}:{probe.i}:{probe.j}:"
            f"{probe.n}:{probe.dx}\n".encode()
        )
    return hasher.hexdigest()[:32]


@dataclass(slots=True)
class ProbeCacheStats:
    hits: int = 0
    misses: int = 0
    probes_served: int = 0

    @property
    def hit(self) -> bool:
        return self.hits > 0 and self.misses == 0


class CachedQuoterClient:
    """A `QuoterClient` whose `probe` results persist across runs."""

    def __init__(
        self,
        client: QuoterClient,
        chain_id: int,
        block: int,
        *,
        cache: Cache | None = None,
        enabled: bool = True,
    ) -> None:
        self.client = client
        self.chain_id = chain_id
        self.block = block
        self.cache = cache or Cache()
        self.enabled = enabled
        self.stats = ProbeCacheStats()

    # -- pass-through -------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.client, name)

    def quote_routes(self, *args, **kwargs):
        return self.client.quote_routes(*args, **kwargs)

    def quote_route(self, *args, **kwargs):
        return self.client.quote_route(*args, **kwargs)

    def raw(self, *args, **kwargs):
        return self.client.raw(*args, **kwargs)

    # -- the cached one -----------------------------------------------------

    def probe(self, probes: list[Probe]) -> list[Quote]:
        if not self.enabled or not probes:
            return self.client.probe(probes)

        parts = (str(self.chain_id), str(self.block), f"probes-{digest(probes)}.json")
        stored = self.cache.read(*parts)
        if stored and len(stored) == len(probes):
            self.stats.hits += 1
            self.stats.probes_served += len(probes)
            return [
                Quote(_STATUS_BY_NAME.get(row[0], Status.MISSING), int(row[1]))
                for row in stored
            ]

        self.stats.misses += 1
        answers = self.client.probe(probes)
        self.cache.write([[q.status.name, str(q.value)] for q in answers], *parts)
        return answers
