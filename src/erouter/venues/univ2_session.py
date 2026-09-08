"""One chain's Uniswap v2 pairs, held across a session.

The same shape as `univ3_session.Univ3`, and shorter by everything a
single-range pool does not need.  There is no bank, so there is no `collapse`
and no second map holding the same pools for a different consumer; there are no
ticks, so a refresh is one `eth_call` per pair rather than three round trips of
storage reads.

What it keeps is the part that matters to the router: `arcs` go in through
`late_arcs`, which puts them in the *graph* and not in the *frame*.  §4 fits
reference prices by weighted least squares over the arcs it is given, and a
venue that joins before the fit votes on every price in it -- measured at
9.50 bp on `crvUSD -> sDOLA` when v3 arcs went in through `extra_arcs`.  v2 is
one arc per direction rather than 32, so it would outvote Curve less loudly and
in the same direction, which is a worse bug than a loud one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.types import PoolArc
from . import univ2
from .univ2_chain import read_pairs

#: Pairs below this hold too little to route through and cost a read each.
DEFAULT_FLOOR_USD = 10_000.0


def census_path(root: Path, chain: str) -> Path:
    return root / "data" / "univ2" / f"{chain}.json"


@dataclass
class Univ2:
    """One chain's v2 pairs, at one block.

    `arcs` is empty until `refresh`, so a session that never warms is a session
    with no v2 in it rather than one that fails.
    """

    census: dict
    floor_usd: float = DEFAULT_FLOOR_USD
    arcs: list[PoolArc] = field(default_factory=list)
    #: `pair -> (token0, token1, fee_bps, decimals0, decimals1)`, which is both
    #: what `read_pairs` takes and what the arc builder needs afterwards.
    pools: dict = field(default_factory=dict)
    #: `pair -> PairState`, kept so a leg can be priced exactly without a
    #: second read.  A v2 pair's whole state is two integers.
    state: dict = field(default_factory=dict)
    block: int = 0
    read_ms: float = 0.0
    #: How many pairs each filter left, so "no v2 here" says which one said no.
    considered: tuple = (0, 0, 0)

    @classmethod
    def load(cls, root: Path, chain: str, **kw) -> Univ2 | None:
        """`None` when the chain has no census, which is not an error.

        Every chain but the one with a census file routes exactly as it did.
        """
        path = census_path(root, chain)
        if not path.exists():
            return None
        census = json.loads(path.read_text())
        if not census:
            # An empty census would make the venue silently absent while every
            # flag and boot line said it was on.  `scripts/v2_census.py` refuses
            # to write one; this refuses to trust one that appeared anyway.
            return None
        return cls(census, **kw)

    def wanted(self, nodes) -> dict:
        """The pairs worth reading: above the floor, and priceable.

        Both coins have to be in the node map already, for the reason
        `Univ3.wanted` gives: a token the frame cannot price has no arc to give,
        and extending the map from a venue's pairs would put tokens in the graph
        nothing else in the router has seen.
        """
        out, above, priceable = {}, 0, 0
        for pool, row in self.census.items():
            token0, token1 = row[0].lower(), row[1].lower()
            fee = int(row[2]) if len(row) > 2 and row[2] else univ2.DEFAULT_FEE_BPS
            tvl = row[3] if len(row) > 3 else 0.0
            if tvl < self.floor_usd:
                continue
            above += 1
            if not (nodes.has(token0) and nodes.has(token1)):
                continue
            priceable += 1
            if nodes.node(token0) == nodes.node(token1):
                continue
            out[pool.lower()] = (token0, token1, fee,
                                 nodes.decimals(token0), nodes.decimals(token1))
        self.considered = (above, priceable, len(out))
        return out

    def refresh(self, transport, nodes, block: int) -> int:
        """Read the reserves and rebuild the arcs.  Returns the pair count."""
        self.pools = self.wanted(nodes)
        if not self.pools:
            self.arcs, self.state, self.block = [], {}, block
            return 0
        started = time.perf_counter()
        state = read_pairs(transport, self.pools, block)
        self.read_ms = (time.perf_counter() - started) * 1e3

        arcs: list[PoolArc] = []
        for pool, pair_state in state.items():
            token0, token1, _fee, _dec0, _dec1 = self.pools[pool]
            row = self.census.get(pool) or self.census.get(pool.lower()) or []
            arcs += univ2.arcs_for(
                pool, pair_state, token0, token1, nodes,
                tvl=row[3] if len(row) > 3 else 0.0)
        self.arcs, self.state, self.block = arcs, state, block
        return len(state)

    def output(self, pool: str, zero_for_one: bool, dx: int) -> int | None:
        """What a pair pays for `dx`, exactly, or `None` if it is not held.

        The whole of v2's pricing.  A tick walk needs a bank and a `collapse` to
        put it back; a constant product needs two integers, so a leg can be
        priced from the state already in hand.
        """
        state = self.state.get(pool.lower())
        if state is None:
            return None
        return univ2.output(state, zero_for_one, dx)
