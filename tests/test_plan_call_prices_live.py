"""`plan_call` bounds a leg on the state it is about to enforce the bound on.

The models freeze the storage they were built from, so they are only valid at
the block they were read at.  `plan_call` re-reads the accounts its route
touches at the newest block and dry-runs against *those*, so pricing the legs
from the models sets the bound on one state and enforces it on another.  Every
leg whose rate has fallen further than its tolerance since the warm then
refuses a route the chain would have settled -- and a leg on the volatile floor
is granted 5.00 bp against a pool measured to move a median 3.86 bp in two
minutes, p90 11.72.

Reported from the flet-curve-demo side as a swap that refused to send every
time.  Measured on 0.1 ETH -> DOLA, whose offending leg is TriCRV: with the
models frozen at the warm and the storage walked forward, the dry run refused
with "leg below its minimum rate" at 30 blocks of model age and passed at every
age once the quotes came off the storage instead.

`refresh` gets this right for the route path -- its own comment is "a refresh
that kept them would be self-consistent and wrong" -- and calls `refresh_at`.
Nothing on the planning path does.
"""

from __future__ import annotations

from erouter.chain import session as sess


class Inner:
    """The `QuoterClient` underneath: reads whatever storage the EVM holds."""

    def __init__(self) -> None:
        self.asked: list = []

    def probe(self, probes):
        self.asked.append(list(probes))
        return ["live"] * len(probes)


class Exact:
    """The wrapper, answering from models frozen at `models_block`."""

    def __init__(self, inner) -> None:
        self.client = inner
        self.models_block = 100
        self.asked: list = []

    def probe(self, probes):
        self.asked.append(list(probes))
        return ["frozen"] * len(probes)

    def fee_at(self, pool, kind, i, j, dx):
        return 0.0004

    def fee_floor(self, pool, kind, i, j):
        return 0.0003

    def model_for(self, pool, kind, i, j):
        return ("model", pool)


def test_the_quote_comes_off_storage_and_not_off_the_models():
    inner = Inner()
    exact = Exact(inner)
    live = sess._LiveQuotes(exact)
    assert live.probe(["a", "b"]) == ["live", "live"]
    assert inner.asked == [["a", "b"]]
    assert exact.asked == []


def test_the_fee_still_comes_off_the_models():
    # A fee is a fraction the models compute better than a probe can, and it
    # drifts far more slowly than a rate.  Moving it too would change every
    # leg's tolerance, which is not the bug and not the fix.
    live = sess._LiveQuotes(Exact(Inner()))
    assert live.fee_at("0xpool", None, 0, 1, 10**18) == 0.0004
    assert live.fee_floor("0xpool", None, 0, 1) == 0.0003


def test_a_pool_the_route_touches_twice_keeps_its_model():
    # `model_for` is carried through the route's own earlier leg, which no
    # probe can see: 0.236 bp on ethereum, 115 bp on gnosis.
    live = sess._LiveQuotes(Exact(Inner()))
    assert live.model_for("0xpool", None, 0, 1) == ("model", "0xpool")


def test_anything_else_the_pricer_reaches_for_is_forwarded():
    live = sess._LiveQuotes(Exact(Inner()))
    assert live.models_block == 100
    assert not hasattr(live, "no_such_attribute")


def test_a_session_with_no_exact_models_is_already_live():
    # Without the exact wrapper the client *is* the quoter, which reads the
    # EVM: nothing to redirect, and the shim must not lose the probe.
    inner = Inner()
    live = sess._LiveQuotes(inner)
    assert live.probe(["a"]) == ["live"]
    assert inner.asked == [["a"]]
