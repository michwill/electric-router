"""A quoter that computes stableswap probes instead of asking for them.

Every derivative in this router is measured, which is the right default: a
pool's behaviour is a property of its deployed code, and reading it is the only
way to be sure.  But for stableswap we can do better than measure -- the
invariant is known, the parameters are readable, and `core/stableswap.py`
reproduces `get_dy` to the wei.  Computing it is then both exact at any size
and free, which matters twice over:

* the quadratic stops being asked to describe a curve at 80% of a reserve,
  which is where it produced a candidate returning 0.3% of the input;
* the probes for those pools stop being sent at all, and the probe batch is
  the router's dominant cost.

Everything else -- CryptoSwap, LLAMMA, wrappers, any pool whose parameters did
not reproduce its own quote -- goes to the wire exactly as before.  This is a
front for `QuoterClient`, not a replacement: `quote_routes` is deliberately
*not* intercepted, because the final answer must come from the chain (§7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.quoter import Quote
from ..core.stableswap import StableSwapError
from ..core.transport import Status
from ..core.types import ArcKind


@dataclass(slots=True)
class ExactStats:
    computed: int = 0
    delegated: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.computed + self.delegated + self.failed


class ExactQuoterClient:
    """Wraps a `QuoterClient`, answering what it can from arithmetic."""

    def __init__(self, client, exact, *, enabled: bool = True):
        self.client = client
        self.exact = exact
        self.enabled = enabled
        self.stats = ExactStats()

    # -- pass-through -------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.client, name)

    # -- the one method that changes ---------------------------------------

    def probe(self, probes):
        """Compute the stableswap probes; send the rest.

        Order is preserved by filling the computed slots in place and asking
        the inner client only for the holes, so a caller cannot tell which
        answers came from where -- which is the point, since everything
        downstream is entitled to assume one consistent source.
        """
        if not self.enabled or not self.exact:
            self.stats.delegated += len(probes)
            return self.client.probe(probes)

        out: list[Quote | None] = [None] * len(probes)
        holes: list[int] = []
        for k, probe in enumerate(probes):
            model = self._model_for(probe)
            if model is None:
                holes.append(k)
                continue
            try:
                value = model.get_dy(probe.i, probe.j, probe.dx)
            except (StableSwapError, ZeroDivisionError, ValueError):
                # A size the invariant cannot serve is a real answer -- the
                # pool would revert too -- but a *failure to converge* is not
                # something to guess at, so both go to the wire.
                self.stats.failed += 1
                holes.append(k)
                continue
            self.stats.computed += 1
            out[k] = Quote(Status.VALUE if value > 0 else Status.REVERTED, value)

        if holes:
            self.stats.delegated += len(holes)
            served = self.client.probe([probes[k] for k in holes])
            for k, quote in zip(holes, served, strict=True):
                out[k] = quote
        return [q for q in out if q is not None]

    def _model_for(self, probe):
        if probe.kind is not ArcKind.SWAP_STABLE or probe.dx <= 0:
            return None
        model = self.exact.get(probe.pool)
        if model is None:
            return None
        n = model.n
        if not (0 <= probe.i < n and 0 <= probe.j < n) or probe.i == probe.j:
            return None
        return model
