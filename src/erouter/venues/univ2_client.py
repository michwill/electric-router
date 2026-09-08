"""Let a quoter client price a Uniswap v2 leg from the reserves it already holds.

The same problem `univ3_client` solves, and the same shape of answer.  `verify`
ranks candidates by re-quoting them through `quote_routes`, and no deployed
quoter answers for a `SWAP_UNIV2` leg: `RouteQuoter` has never heard of the
kind, so the call returns zero, `verify` reads zero as a revert, and every
candidate carrying v2 is dropped before it can win.  The solver's chosen flow
then never reaches a route -- which looks exactly like v2 losing on the merits.

The fix is to let the walk answer.  `exact_probe.quote_routes` already walks a
route leg by leg for pools whose arithmetic reproduces their own getter, and a
constant product with a flat fee *is* that arithmetic -- `univ2.output` is the
contract's `getAmountOut` transcribed, floor for floor.

Simpler than v3's in the one way that matters: there is no bank to install and
no units to carry, because a pair's whole state is two integers and the session
already holds them.  A route mixing Curve, v3 and v2 is walked in one pass,
since `walk_route` neither knows nor cares which leg came from where.
"""

from __future__ import annotations

from ..core.types import ArcKind
from ..core.walk import LegUnquotable
from .univ2 import output


def teach(client, state: dict) -> None:
    """Make `client` able to walk a v2 leg.  `state` is `pair -> PairState`.

    Installed on the instance rather than the class, for the reason
    `univ3_client.teach` gives: a session holds one client and the state is that
    session's block, so there is nothing to share and something to get wrong by
    sharing it.

    Chains with whatever `_quote_leg` is already there, so teaching both venues
    to one client leaves each answering for its own kind and delegating the
    rest -- the order they are taught in does not matter.
    """
    real = client._quote_leg

    def quote_leg(leg, dx: int) -> int:
        if leg.kind is not ArcKind.SWAP_UNIV2:
            return real(leg, dx)
        pair = state.get(leg.target.lower())
        if pair is None:
            # Never a zero.  `verify` reads a zero as a revert, which is the
            # failure this exists to remove, and returning one would remove it
            # by hiding it.  Say the leg cannot be walked and let the caller
            # send the route to the chain, where it fails honestly.
            raise LegUnquotable(leg.target)
        # `i` is the input coin's index in the pair, so `i == 0` is
        # `zero_for_one` -- the same convention `univ2.arcs_for` writes.
        return output(pair, leg.i == 0, dx)

    client._quote_leg = quote_leg
