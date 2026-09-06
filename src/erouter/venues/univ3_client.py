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


#: Uniswap's own quoter, which simulates the swap and reads its answer out of a
#: deliberate revert.  Mainnet; a chain without one cannot be audited and says so.
QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

#: How far a bank may sit from the pool's own quoter before the candidate it
#: priced is refused.  Measured on the winning routes of a 21-case sweep, every
#: v3 leg landed inside 1.7 bp and most inside 0.01, so this is wide enough to
#: pass a healthy bank and far too tight to let a broken one through.
AUDIT_TOLERANCE_BP = 5.0


def _truth_leg(transport, pools: dict, block: int, inner):
    """`inner`, except that a v3 leg is priced by the pool's own quoter."""
    from ..core.codec import decode, encode_call

    def quote_leg(leg, dx: int) -> int:
        if leg.kind is not ArcKind.SWAP_UNIV3:
            return inner(leg, dx)
        row = pools.get(leg.target.lower())
        if row is None or dx <= 0:
            raise LegUnquotable(f"{leg.target}: no quoter row to audit against")
        token_in, token_out, fee = row[0], row[1], row[2]
        if leg.i == 1:
            token_in, token_out = token_out, token_in
        data = encode_call(
            "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
            (token_in, token_out, dx, fee, 0))
        raw = transport.fetch("eth_call", [
            {"to": QUOTER_V2, "data": "0x" + data.hex()}, hex(block)])
        return decode(["uint256", "uint160", "uint32", "uint256"],
                      bytes.fromhex(raw[2:]))[0]

    return quote_leg


def audit(pool_set, client, transport, pools: dict, *, block: int,
          tolerance_bp: float = AUDIT_TOLERANCE_BP) -> list[tuple]:
    """Hold the winner to the pool's own quoter, and refuse it if it disagrees.

    Every other venue in this router is ranked on a number the *chain* produced:
    `verify` re-quotes each candidate through `quote_routes`, and for a Curve leg
    that is an on-chain `get_dy`.  A v3 leg has no deployed quoter the walk can
    chain, so `teach` answers it from the bank -- which means the number the
    winner is ranked on came from the same arithmetic that proposed it.  A bank
    that drifted would produce a confident wrong answer and nothing downstream
    would notice.

    So the published one is checked: re-walk the winner with `QuoterV2` standing
    in for the banks, on the same legs at the same block, and compare with what
    it was ranked on.  Disagreement past `tolerance_bp` clears `verified_out`,
    which is what `CandidateSet.best` reads, so the next candidate is promoted
    and audited in its turn.

    The winner only, and deliberately.  Auditing every candidate is one sequential
    `eth_call` per v3 leg per candidate -- 776 walked routes on one measured quote
    -- where auditing in rank order costs two or three calls and guarantees the
    number that actually leaves the building.  A candidate that never wins is a
    candidate nobody was told about.

    Returns one row per audit performed: `(label, ranked, truth, bp, kept)`.
    """
    from ..core.walk import walk_route

    done: list[tuple] = []
    while True:
        winner = pool_set.best
        if winner is None or winner.route is None:
            return done
        legs = [rl.leg for rl in winner.route.legs]
        if not any(leg.kind is ArcKind.SWAP_UNIV3 for leg in legs):
            return done                      # nothing of ours in it to doubt
        ranked = int(winner.verified_out or 0)
        try:
            truth = walk_route(
                legs, winner.route.amount_in, winner.route.dst_slot,
                _truth_leg(transport, pools, block, client._stateful_leg(legs)))
        except Exception:
            truth = 0
        bp = (ranked / truth - 1) * 1e4 if truth else float("inf")
        kept = bool(truth) and abs(bp) <= tolerance_bp
        done.append((winner.label, ranked, truth, bp, kept))
        if kept:
            return done
        winner.status = "univ3 audit"
        winner.note = (f"bank {ranked:,} against the pool's quoter {truth:,}"
                       if truth else "the pool's quoter would not answer")
        winner.verified_out = None
