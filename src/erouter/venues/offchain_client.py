"""Let a quoter client *probe* an off-chain leg, not merely walk one.

`univ2_client.teach` and `univ3_client.teach` both install on `_quote_leg`,
which is what `quote_routes` walks -- so `verify` prices a v2, v3 or v4 leg
correctly and a candidate carrying one can win.  `probe` is a different path
entirely: it encodes `SIG_PROBE_BATCH` and asks the deployed `RouteQuoter`,
which has never heard of kinds 17, 18 or 19.  Every such probe comes back zero.

Nothing reports that as an error, because a zero probe is indistinguishable
from a pool that refused, so the cost lands somewhere else.  `split` samples
each leg's curve by probing it; `_probe_ladders` needs two usable answers per
leg and gets none; `_trusted_curves` returns "a leg would not probe"; and
`optimise` falls back to the chained hill-climb, which re-quotes whole routes
through the chain instead of composing curves.

Measured on ethereum at block 25,935,978, v2 enabled, before this existed:

    case                split      probes  every v2 leg     mode
    WETH -> WBTC      9,157 ms      1,056   0 of 24 usable  chained
    ALD  -> FRAX      3,928 ms      1,512   0 of 24 usable  chained
    USDC -> WETH         20 ms        288   0 of 24 usable  chained

67% and 53% of those two quotes, and the probes are paid for *first* and then
thrown away.  The curve path is not being refused as inaccurate -- when it does
run its check lands within 0.53 bp -- it is refused for want of an answer that
is pure arithmetic and already implemented.

So: answer them here.  Split a batch by kind, hand the off-chain ones to
whatever `_quote_leg` the venues installed, and forward the rest to the
contract unchanged.  `_quote_leg` is read at call time rather than captured, so
this may be installed before or after the venue teachers -- the same
order-independence they promise each other.
"""

from __future__ import annotations

from ..core.quoter import Quote
from ..core.transport import Status
from ..core.types import OFF_CHAIN_KINDS, Leg, Probe
from ..core.walk import LegUnquotable


def teach_probes(client) -> None:
    """Make `client.probe` answer for the kinds no deployed quoter knows."""
    real = client.probe

    def probe(probes: list[Probe]) -> list[Quote]:
        if not any(p.kind in OFF_CHAIN_KINDS for p in probes):
            return real(probes)
        forwarded = [p for p in probes if p.kind not in OFF_CHAIN_KINDS]
        answers = iter(real(forwarded) if forwarded else [])
        out: list[Quote] = []
        for p in probes:
            if p.kind not in OFF_CHAIN_KINDS:
                out.append(next(answers, Quote(Status.MISSING, 0)))
                continue
            leg = Leg(target=p.pool, kind=p.kind, i=p.i, j=p.j, n=p.n)
            try:
                value = int(client._quote_leg(leg, p.dx))
            except LegUnquotable:
                # A pool the venue does not hold.  `REVERTED` rather than a
                # zero `VALUE`, because the callers read the status and a zero
                # that claims to be a real quote is the failure this removes.
                out.append(Quote(Status.REVERTED, 0))
                continue
            out.append(Quote(Status.VALUE, value) if value > 0
                       else Quote(Status.REVERTED, 0))
        return out

    client.probe = probe
