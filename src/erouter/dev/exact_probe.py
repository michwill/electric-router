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
from ..core.tricrypto import TricryptoError
from ..core.twocrypto import TwocryptoError
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

    def __init__(self, client, exact, twocrypto=None, tricrypto=None, *,
                 enabled: bool = True, models_block: int = 0, rebuild=None):
        self.client = client
        self.exact = exact
        #: Three-coin crypto pools, whose maths is its own module again.
        self.tricrypto = tricrypto
        #: Twocrypto-ng pools whose invariant reproduces their own `get_dy`
        #: -- today the FX Swaps, whose maths is stableswap's.
        self.twocrypto = twocrypto
        self.enabled = enabled
        self.stats = ExactStats()
        #: The block these models were read at.  Not named `block`: this class
        #: forwards unknown attributes to the quoter, and shadowing one it may
        #: itself define would answer a question nobody asked here.
        self.models_block = models_block
        self._rebuild = rebuild

    def refresh_at(self, block: int) -> int:
        """Rebuild the models if the chain moved out from under them.

        Every model freezes the storage it was built from, so it is only valid
        at that block -- correct today because the transport pins one block for
        the life of a process, and wrong the moment a live provider is handed
        in instead.  `pipeline.route` calls this whenever it sees the block
        move; it is a no-op when nothing has.

        A rebuild that raises leaves the previous models in place and says so
        by re-raising: quoting the wrong block is worse than not quoting.
        """
        if not block or block == self.models_block or self._rebuild is None:
            return 0
        exact, twocrypto, tricrypto = self._rebuild(block)
        self.exact, self.twocrypto, self.tricrypto = exact, twocrypto, tricrypto
        self.models_block = block
        self.stats = ExactStats()
        return len(exact) + len(twocrypto or ()) + len(tricrypto or ())

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
            except (StableSwapError, TwocryptoError, TricryptoError,
                    ZeroDivisionError, ValueError):
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

    def computes(self, pool: str) -> bool:
        """Whether a probe on this pool costs arithmetic rather than a request.

        `pipeline` asks so it can re-fit *every* such arc at the trade's own
        size instead of only the ones on its shortlist -- the shortlist is a
        budget for round trips, and these have none.
        """
        if not self.enabled:
            return False
        if self.exact.get(pool) is not None:
            return True
        if self.tricrypto is not None and self.tricrypto.get(pool) is not None:
            return True
        return self.twocrypto is not None and self.twocrypto.get(pool) is not None

    def _model_for(self, probe):
        if probe.dx <= 0:
            return None
        if probe.kind is ArcKind.SWAP_CRYPTO:
            if probe.i == probe.j:
                return None
            if self.tricrypto is not None:
                model = self.tricrypto.get(probe.pool)
                if model is not None:
                    if not (0 <= probe.i < 3 and 0 <= probe.j < 3):
                        return None
                    return model
            if self.twocrypto is None:
                return None
            model = self.twocrypto.get(probe.pool)
            if model is None:
                return None
            if not (0 <= probe.i < 2 and 0 <= probe.j < 2):
                return None
            return model
        if probe.kind is not ArcKind.SWAP_STABLE:
            return None
        model = self.exact.get(probe.pool)
        if model is None:
            return None
        n = model.n
        if not (0 <= probe.i < n and 0 <= probe.j < n) or probe.i == probe.j:
            return None
        return model
