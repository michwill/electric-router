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
front for `QuoterClient`, not a replacement.

`quote_routes` is intercepted too, but only for a route whose *every* leg is a
pool the gate admitted.  That is a narrower claim than it looks: the gate's
whole job is to establish that this arithmetic and the pool's own `get_dy`
return the same integer, so executing such a route is a round trip spent
confirming what is already known.  §7's "the final answer must come from the
chain" is satisfied where it binds -- the transaction carries a minimum-out and
the chain adjudicates it at submission, which is the only verification that
survives contact with a moving block anyway.

A route with one leg this cannot serve goes to the chain whole.  Half an answer
from a model and half from execution would be worse than either, because
nothing downstream could tell which half it was reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.quoter import Quote
from ..core.stableswap import StableSwapError
from ..core.tricrypto import TricryptoError
from ..core.twocrypto import TwocryptoError
from ..core.transport import Status
from ..core.types import ArcKind
from ..core.walk import LegUnquotable, walk_route


@dataclass(slots=True)
class ExactStats:
    computed: int = 0
    delegated: int = 0
    failed: int = 0
    #: Candidate routes verified by walking the models rather than the chain.
    walked: int = 0
    sent_routes: int = 0

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
        return self._model(probe.pool, probe.kind, probe.i, probe.j)

    def _model(self, pool: str, kind, i: int, j: int):
        """The model that can answer for this pool and direction, if any."""
        if kind is ArcKind.SWAP_CRYPTO:
            if i == j:
                return None
            if self.tricrypto is not None:
                model = self.tricrypto.get(pool)
                if model is not None:
                    if not (0 <= i < 3 and 0 <= j < 3):
                        return None
                    return model
            if self.twocrypto is None:
                return None
            model = self.twocrypto.get(pool)
            if model is None:
                return None
            if not (0 <= i < 2 and 0 <= j < 2):
                return None
            return model
        if kind is not ArcKind.SWAP_STABLE:
            return None
        model = self.exact.get(pool)
        if model is None:
            return None
        n = model.n
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return None
        return model

    # -- verification, where every leg is a pool we can evaluate --------------

    def quote_routes(self, routes, amounts_in, dst_slots):
        """Walk the routes we can evaluate; send the rest to the chain.

        Verifying a route used to mean executing it, which is why the local EVM
        warms storage for every pool in the universe.  For a pool admitted by
        the wei-exact gate that is a round trip to be told what the arithmetic
        here already knows, so those routes are walked with `core/walk.py` --
        the same accumulator the contract runs, so the two agree by
        construction rather than by luck.

        A route with a single leg this cannot serve -- a wrapper, an LP
        deposit, a pool still being probed -- goes to the chain whole.  Mixing
        the two inside one route would be worse than either: half the answer
        would come from a model and half from execution, and nothing downstream
        could tell which.
        """
        if not self.enabled:
            return self.client.quote_routes(routes, amounts_in, dst_slots)

        out: list[int | None] = [None] * len(routes)
        holes: list[int] = []
        for k, (legs, amount, dst) in enumerate(
                zip(routes, amounts_in, dst_slots, strict=True)):
            try:
                out[k] = walk_route(legs, amount, dst, self._quote_leg)
            except LegUnquotable:
                holes.append(k)
            except (StableSwapError, TwocryptoError, TricryptoError,
                    ZeroDivisionError, ValueError):
                # The invariant refusing a size is a real answer -- the pool
                # would revert too -- but this is the number the winner is
                # chosen on, so it goes to the chain rather than being guessed.
                holes.append(k)
        self.stats.walked += len(routes) - len(holes)

        if holes:
            self.stats.sent_routes += len(holes)
            served = self.client.quote_routes(
                [routes[k] for k in holes], [amounts_in[k] for k in holes],
                [dst_slots[k] for k in holes])
            for k, value in zip(holes, served, strict=True):
                out[k] = value
        return [0 if v is None else v for v in out]

    def _quote_leg(self, leg, dx: int) -> int:
        model = self._model(leg.target, leg.kind, leg.i, leg.j)
        if model is None:
            raise LegUnquotable(leg.target)
        return model.get_dy(leg.i, leg.j, dx)
