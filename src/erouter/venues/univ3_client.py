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
from .univ3 import Arc, exact_output


@dataclass(frozen=True, slots=True)
class Bank:
    """One direction of one pool: the arcs, and the units they are in."""

    arcs: list[Arc]
    decimals_in: int
    decimals_out: int

    def quote(self, dx: int) -> int:
        """Raw wei in, raw wei out -- the shape `walk_route` chains.

        Exact rather than the arcs' water-fill: this is the number a route is
        ranked on, and the audit refuses what the pool's quoter disagrees with.
        """
        if dx <= 0:
            return 0
        human = exact_output(self.arcs, dx / 10**self.decimals_in)
        return int(human * 10**self.decimals_out)


def teach(client, banks: dict, kind: ArcKind = ArcKind.SWAP_UNIV3) -> None:
    """Make `client` able to walk a tick-bank leg.  Keyed by `(pool, i, j)`.

    Installed on the instance rather than the class: a session holds one client
    and the banks are that session's block, so there is nothing to share and
    something to get wrong by sharing it.

    `kind` is a parameter because v4's banks are these banks -- same math, same
    `Bank`, different `ArcKind` -- and a teacher that answers only for
    `SWAP_UNIV3` would let a v4 leg fall through to a quoter that has never
    heard of kind 19.  That returns zero, `verify` reads zero as a revert, and
    every route carrying the venue is dropped before it can win.  Which is how
    `--univ3` shipped, and then `--univ2`, and it is not going to be how v4
    ships.
    """
    real = client._quote_leg

    def quote_leg(leg, dx: int) -> int:
        if leg.kind is not kind:
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


def _quoter_call(leg, dx: int, pools: dict, block: int):
    """The `eth_call` asking QuoterV2 what `leg` pays for `dx`, or `None`."""
    from ..core.codec import encode_call

    row = pools.get(leg.target.lower())
    if row is None or dx <= 0:
        return None
    token_in, token_out, fee = row[0], row[1], row[2]
    if leg.i == 1:
        token_in, token_out = token_out, token_in
    data = encode_call(
        "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
        (token_in, token_out, dx, fee, 0))
    return "eth_call", [{"to": QUOTER_V2, "data": "0x" + data.hex()}, hex(block)]


def _quoted(raw) -> int:
    from ..core.codec import decode

    return decode(["uint256", "uint160", "uint32", "uint256"], bytes.fromhex(raw[2:]))[0]


def _truth_leg(transport, pools: dict, block: int, inner):
    """`inner`, except that a v3 leg is priced by the pool's own quoter."""

    def quote_leg(leg, dx: int) -> int:
        if leg.kind is not ArcKind.SWAP_UNIV3:
            return inner(leg, dx)
        call = _quoter_call(leg, dx, pools, block)
        if call is None:
            raise LegUnquotable(f"{leg.target}: no quoter row to audit against")
        return _quoted(transport.fetch(*call))

    return quote_leg


#: How far the v3 legs' banks may sit from their pools' quoter, summed, for the
#: first batch to stand as it is: 0.1 bp, so the walk it returns is within 0.1 bp
#: of the sequential one and fifty times inside `AUDIT_TOLERANCE_BP`.  Measured on
#: a 10-leg route, legs agreed to 1e-11..3e-9 and one small WBTC leg to 1.1e-6:
#: the pool's integer rounding against the bank's float.
BATCH_AGREE = 1e-5
#: Batches the walk may take to settle before it asks one leg at a time.
BATCH_ROUNDS = 4


def _batched_truth(legs, route, client, transport, pools: dict, block: int):
    """The truth walk in a few round trips rather than one per v3 leg.

    Walk with the banks standing in and ask QuoterV2 about every v3 leg at the
    input it saw, in one batch.  If they agree to `BATCH_AGREE`, that walk is
    the answer.  Otherwise walk again with the quoter's answers wherever a leg
    sees the same input, and ask only about the legs whose input moved: once a
    walk asks nothing new, every v3 leg was priced by its quoter at the input it
    really saw, which is the sequential walk exactly.  One leg 3.8 bp off on
    WETH->WBTC $10M used to send all 21 to the node in series, 2.2 s of 2.9.
    `None` if it does not settle in `BATCH_ROUNDS`; the caller then walks it
    the slow way.
    """
    from ..core.walk import walk_route

    fetch_multi = getattr(transport, "fetch_multi", None)
    bank = getattr(client, "_quote_leg", None)
    if fetch_multi is None or bank is None:
        return None
    walker = getattr(client, "_mixed_leg", client._stateful_leg)
    known: dict[tuple[int, int], int] = {}
    for rounds in range(BATCH_ROUNDS + 1):
        other = walker(legs)
        asked: list[tuple] = []

        def quote_leg(leg, dx: int, other=other, asked=asked) -> int:
            if leg.kind is not ArcKind.SWAP_UNIV3:
                return other(leg, dx)
            got = known.get((id(leg), dx))
            if got is not None:
                return got
            out = bank(leg, dx)
            asked.append((leg, dx, out))
            return out

        walked = walk_route(legs, route.amount_in, route.dst_slot, quote_leg)
        if not walked:
            return None
        if not asked:
            return walked
        if rounds == BATCH_ROUNDS:
            return None
        calls = [_quoter_call(leg, dx, pools, block) for leg, dx, _ in asked]
        if any(c is None for c in calls):
            return None
        drift = 0.0
        for (leg, dx, out), raw in zip(asked, fetch_multi(calls), strict=True):
            if not isinstance(raw, str):
                return None
            truth = _quoted(raw)
            if truth <= 0:
                return None
            known[(id(leg), dx)] = truth
            drift += abs(out / truth - 1)
        if rounds == 0 and drift <= BATCH_AGREE:
            return walked
    return None


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
            truth = _batched_truth(legs, winner.route, client, transport, pools, block)
        except Exception:
            truth = None
        if truth is None:
            try:
                truth = walk_route(
                    legs, winner.route.amount_in, winner.route.dst_slot,
                    _truth_leg(transport, pools, block,
                               getattr(client, "_mixed_leg", client._stateful_leg)(legs)))
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
