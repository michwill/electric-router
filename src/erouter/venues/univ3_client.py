"""Let a quoter client price a Uniswap v3 leg from its own tick bank.

`verify` ranks candidates by re-quoting them through `quote_routes`, and a
`SWAP_UNIV3` leg has no on-chain quoter to answer for it: the call returns zero,
`verify` reads zero as a revert, and every candidate carrying v3 is dropped
before it can win.  The solver's chosen flow then never reaches a route.

The fix is not to make the chain answer -- it cannot, there is no deployed
quoter for this -- but to let the walk answer.  `exact_probe.quote_routes`
already walks a route leg by leg through `_quote_leg` rather than sending it,
for pools whose arithmetic reproduces their own `get_dy`; a v3 tick bank is the
same claim by construction, since it *is* the pool's arithmetic.  So the leg is
priced where the model already lives.

A route mixing Curve and v3 is walked in one pass, because `walk_route` neither
knows nor cares which leg came from where.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import ArcKind
from ..core.walk import LegUnquotable
from .univ3 import Arc, output


@dataclass(frozen=True, slots=True)
class Bank:
    """One direction of one pool: the arcs, and the units they are in."""

    arcs: list[Arc]
    decimals_in: int
    decimals_out: int

    def quote(self, dx: int) -> int:
        """Raw wei in, raw wei out -- the shape `walk_route` chains."""
        if dx <= 0:
            return 0
        human = output(self.arcs, dx / 10**self.decimals_in)
        return int(human * 10**self.decimals_out)


def teach(client, banks: dict) -> None:
    """Make `client` able to walk a v3 leg.  Keyed by `(pool, i, j)`.

    Installed on the instance rather than the class: a session holds one client
    and the banks are that session's block, so there is nothing to share and
    something to get wrong by sharing it.
    """
    real = client._quote_leg

    def quote_leg(leg, dx: int) -> int:
        if leg.kind is not ArcKind.SWAP_UNIV3:
            return real(leg, dx)
        bank = banks.get((leg.target.lower(), leg.i, leg.j))
        if bank is None:
            # Never a zero: `verify` reads a zero as a revert, which is exactly
            # the failure this exists to remove, and it would remove it by
            # hiding it.  Say the leg cannot be walked and let the caller send
            # the route to the chain, where it will fail honestly.
            raise LegUnquotable(leg.target)
        return bank.quote(dx)

    client._quote_leg = quote_leg
