"""Uniswap v3 as something a `RouterSession` can hold.

The pieces were all here already -- `univ3` builds the arcs, `univ3_chain`
reads the ticks, `univ3_client` prices a leg and audits a winner -- and what was
missing was one object that owns them together and knows the four things
`pipeline.route` wants: the extra arcs, how to collapse them, how to audit the
winner, and that §9.7's spread bound is measuring something else here.

Reading is per *block*, not per quote: 144 pools cost ~2.3 s of tick state and
every quote at that block reuses it.  `refresh` is what a session calls when the
block moves, and it is the only expensive thing in here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..core.graph import PATHOLOGICAL_CONDITION
from ..core.types import PoolArc
from . import univ3
from .univ3_chain import read_pools
from .univ3_client import Bank, audit, teach

#: A v3 tick is nearly linear over its own range, so `a/B` reaches 1.4e11 with
#: nothing floored anywhere and 144 mainnet pools span 2.95e15 between them.
#: §9.7's default reads that as a `B` floored in the wrong space, which on a
#: Curve universe it would be.
SPREAD = 1e18

#: Below this a pool is not worth the two round trips its ticks cost, and it is
#: the floor most aggregators draw.
DEFAULT_FLOOR_USD = 10_000.0

#: Tick-arcs a side.  Sixteen covers the depth any single leg reaches on the
#: pools that pass the floor; past that the arcs are dust the graph drops.
DEFAULT_TICKS = 16


def census_path(root: Path, chain: str) -> Path:
    return root / "data" / "univ3" / f"{chain}.json"


@dataclass
class Univ3:
    """One chain's v3 pools, at one block.

    `arcs` and `banks` are empty until `refresh`, so a session that never warms
    is a session with no v3 in it rather than one that fails.
    """

    census: dict
    floor_usd: float = DEFAULT_FLOOR_USD
    ticks: int = DEFAULT_TICKS
    arcs: list[PoolArc] = field(default_factory=list)
    #: `(pool, i, j) -> [Arc]`, which is what `collapse` sums back together.
    banks: dict = field(default_factory=dict)
    #: The same keys carrying `Bank`, which is what the walk prices a leg from.
    #: Two dicts because the two want different things of one bank, and holding
    #: one and unwrapping it at each call site is how they get confused.
    priced: dict = field(default_factory=dict)
    #: pool -> (token0, token1, fee, decimals0, decimals1), for the audit.
    pools: dict = field(default_factory=dict)
    block: int = 0
    read_ms: float = 0.0
    #: How many pools each filter left, so "no v3 here" says which one said no.
    considered: tuple = (0, 0, 0)

    @classmethod
    def load(cls, root: Path, chain: str, **kw) -> Univ3 | None:
        """`None` when the chain has no census, which is not an error.

        Every chain but the one with a census file routes exactly as it did.
        """
        path = census_path(root, chain)
        if not path.exists():
            return None
        return cls(json.loads(path.read_text()), **kw)

    def wanted(self, nodes) -> dict:
        """The pools worth reading: above the floor, and priceable.

        Both coins have to be in the node map already.  A token the frame cannot
        price has no arc to give -- extending the map from a v3 pool's coins is
        a different piece of work, and doing it silently here would put tokens in
        the graph that nothing else in the router has ever seen.
        """
        out, above, priceable = {}, 0, 0
        for pool, row in self.census.items():
            token0, token1, fee = row[0].lower(), row[1].lower(), row[2]
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

    def token_pairs(self):
        """`(token0, token1)` for every pool this venue holds.

        Here so a caller does not have to know the shape of `pools`, which
        differs per venue: v4 names a pool by `PoolKey` because it has no
        address to name it by.  `venue_sweep.token_set` reached in and read
        `meta[0]` directly, which worked for two venues and raised on the
        third.
        """
        return [(meta[0], meta[1]) for meta in self.pools.values()]

    def refresh(self, transport, nodes, block: int) -> int:
        """Read the tick state and rebuild the arcs.  Returns the pool count."""
        import time

        self.pools = self.wanted(nodes)
        if not self.pools:
            self.arcs, self.banks, self.priced, self.block = [], {}, {}, block
            return 0
        started = time.perf_counter()
        state = read_pools(transport, self.pools, block, ticks=self.ticks)
        self.read_ms = (time.perf_counter() - started) * 1e3

        arcs: list[PoolArc] = []
        banks: dict = {}
        priced: dict = {}
        for pool, (pool_state, ticks) in state.items():
            token0, token1, _fee, dec0, dec1 = self.pools[pool]
            arcs += univ3.pool_arcs(
                pool, pool_state, ticks, nodes, token0=token0, token1=token1,
                max_ticks=self.ticks,
                tvl_usd=self.census[pool][3] if len(self.census[pool]) > 3 else 0.0)
            for zero_for_one in (True, False):
                bank = univ3.arcs(pool_state, ticks, zero_for_one=zero_for_one,
                                  max_ticks=self.ticks)
                if not bank:
                    continue
                i, j = (0, 1) if zero_for_one else (1, 0)
                banks[(pool, i, j)] = bank
                priced[(pool, i, j)] = Bank(
                    bank, dec0 if zero_for_one else dec1,
                    dec1 if zero_for_one else dec0)
        self.arcs, self.banks, self.priced = arcs, banks, priced
        self.block = block
        return len(state)

    # ---------------------------------------------------------- the seams

    def teach(self, client) -> None:
        """Let the walk price a v3 leg from its bank."""
        if self.priced:
            teach(client, self.priced)

    def collapse(self, arcs, psi, nu, nodes):
        return univ3.collapse(arcs, psi, nu, nodes, self.banks)

    def auditor(self, transport):
        """The `audit` callable `pipeline.route` takes, bound to this block."""
        def run(pool_set, client):
            return audit(pool_set, client, transport, self.pools,
                         block=self.block)
        return run

    @property
    def max_spread(self) -> float:
        return SPREAD if self.arcs else PATHOLOGICAL_CONDITION
