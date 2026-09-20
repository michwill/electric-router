"""One chain's Uniswap v4 pools, held across a session.

`Univ3`'s shape, with two differences that matter and one that does not.

A pool is named by its `PoolId` rather than an address, because v4 keeps every
pool in one singleton.  Nothing downstream minds -- `PoolArc.pool` is used as an
identity, for dedup and for Decision 3's one-arc-per-pool rule, and a `PoolId`
is a better identity than an address ever was.

And `wanted` gates on the *hook* before it gates on anything else.  Only tiers 0
and 1 are ever read: a hook that can touch a swap is refused whatever it holds,
because the arcs would be priced on arithmetic the hook is entitled to replace.
`venues/univ4.py` carries the measurements behind that line.

The difference that does not matter is the arc math, which is v3's exactly.
`univ3.pool_arcs` builds them, told which kind and venue to stamp on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.graph import PATHOLOGICAL_CONDITION
from ..core.types import ArcKind, PoolArc
from . import univ3, univ4
from .univ3_client import Bank, teach
from .univ3_session import SPREAD
from .univ4_chain import NATIVE, STATE_VIEW, WRAPPED_NATIVE, read_pools

#: Pools below this hold too little to route through and cost reads each.
DEFAULT_FLOOR_USD = 10_000.0
#: Ticks either side of spot, as v3 reads.
DEFAULT_TICKS = 16
#: The most pools a venue may put in one graph, deepest first.  Arc count is a
#: cliff rather than a cost -- see `univ2_session.MAX_PAIRS` -- and a v4 pool
#: brings up to `2 * DEFAULT_TICKS` arcs rather than two, so this is the tighter
#: budget of the two venues by a wide margin.
MAX_POOLS = 150


def census_path(root: Path, chain: str) -> Path:
    return root / "data" / "univ4" / f"{chain}.json"


@dataclass
class Univ4:
    """One chain's v4 pools, at one block."""

    census: dict
    chain: str = "ethereum"
    floor_usd: float = DEFAULT_FLOOR_USD
    ticks: int = DEFAULT_TICKS
    max_pools: int = MAX_POOLS
    arcs: list[PoolArc] = field(default_factory=list)
    banks: dict = field(default_factory=dict)
    priced: dict = field(default_factory=dict)
    #: `poolId -> (PoolKey, decimals0, decimals1)`, which is what the reader
    #: takes and what the arc builder needs afterwards.
    pools: dict = field(default_factory=dict)
    #: `poolId -> (token0, token1)` as the *node map* names them, which is not
    #: always how the `PoolKey` does: v4 names ether by the zero address and
    #: the map prices the wrapped token.  Kept beside `pools` rather than
    #: rewritten into the key, because the key is what the chain is addressed
    #: by and `as_tuple()` must keep saying what the pool really holds.
    tokens: dict = field(default_factory=dict)
    block: int = 0
    read_ms: float = 0.0
    #: How many pools each filter left, so "no v4 here" says which one said no.
    considered: tuple = (0, 0, 0, 0)

    @classmethod
    def load(cls, root: Path, chain: str, **kw) -> Univ4 | None:
        """`None` when the chain has no census, which is not an error."""
        path = census_path(root, chain)
        if not path.exists():
            return None
        census = json.loads(path.read_text())
        if not census:
            # An empty census would make the venue silently absent while every
            # flag said it was on -- how `--univ3` shipped.
            return None
        kw.setdefault("chain", chain)
        return cls(census, **kw)

    @property
    def state_view(self) -> str | None:
        return STATE_VIEW.get(self.chain)

    def _priced(self, token: str) -> str:
        """The address the node map knows this currency by.

        Ether is the whole of the difference, and it is most of the venue:
        1,932 of ethereum's 2,645 pools name it, and 45 of those clear the
        floor with a priced counterparty -- $390.7M of TVL, including the two
        deepest pools in the census.  All of it was invisible because
        `nodes.has(address(0))` is false.

        Sound for pricing, since wrapping is 1:1.  *Execution* is not the same
        thing -- a native leg moves value rather than transferring an ERC20 --
        and v4 cannot be executed at all until the router has its
        `unlockCallback`, so this widens what is quoted ahead of what can be
        settled, which is already true of every v4 pool.
        """
        if token != NATIVE:
            return token
        return WRAPPED_NATIVE.get(self.chain, NATIVE)

    def wanted(self, nodes) -> dict:
        """The pools worth reading: routable hook, above the floor, priceable.

        Hook first, deliberately.  A tier-2 or tier-3 pool is refused before its
        depth is even considered, because depth is not the question -- whether
        the arithmetic is ours is.
        """
        out, routable, above, priceable = {}, 0, 0, 0
        self.tokens = {}
        for pid, row in self.census.items():
            key = univ4.PoolKey.from_row(row)
            if not key.routable:
                continue
            routable += 1
            tvl = float(row[5]) if len(row) > 5 else 0.0
            if tvl < self.floor_usd:
                continue
            above += 1
            # As the node map names them, so ether resolves to its wrapper.
            tok0, tok1 = self._priced(key.currency0), self._priced(key.currency1)
            if not (nodes.has(tok0) and nodes.has(tok1)):
                continue
            priceable += 1
            if nodes.node(tok0) == nodes.node(tok1):
                continue
            self.tokens[pid] = (tok0, tok1)
            out[pid] = (key, nodes.decimals(tok0), nodes.decimals(tok1))
        if self.max_pools and len(out) > self.max_pools:
            deepest = sorted(out, key=lambda p: -(
                float(self.census[p][5]) if len(self.census[p]) > 5 else 0.0))
            out = {p: out[p] for p in deepest[:self.max_pools]}
        self.considered = (routable, above, priceable, len(out))
        return out

    def token_pairs(self):
        """`(currency0, currency1)` for every pool this venue holds.

        See `Univ2.token_pairs`: the shape of `pools` is this venue's business
        and not its caller's.
        """
        return [self.tokens.get(pid, (key.currency0, key.currency1))
                for pid, (key, _dec0, _dec1) in self.pools.items()]

    def refresh(self, transport, nodes, block: int) -> int:
        """Read the tick state and rebuild the arcs.  Returns the pool count."""
        self.pools = self.wanted(nodes)
        view = self.state_view
        if not self.pools or view is None:
            self.arcs, self.banks, self.priced, self.block = [], {}, {}, block
            return 0
        started = time.perf_counter()
        state = read_pools(transport, view, self.pools, block, ticks=self.ticks)
        self.read_ms = (time.perf_counter() - started) * 1e3

        arcs: list[PoolArc] = []
        banks: dict = {}
        priced: dict = {}
        for pid, ((forward, reverse), ticks) in state.items():
            key, dec0, dec1 = self.pools[pid]
            tok0, tok1 = self.tokens.get(pid, (key.currency0, key.currency1))
            row = self.census.get(pid) or []
            tvl = float(row[5]) if len(row) > 5 else 0.0
            # One call per direction, because the protocol fee is charged per
            # direction and each state carries its own effective fee.
            # `pool_arcs` builds both directions from one state and drops the
            # one that is not this state's, which is the price of reusing it.
            for zero_for_one, st in ((True, forward), (False, reverse)):
                built = univ3.pool_arcs(
                    pid, st, ticks, nodes,
                    token0=tok0, token1=tok1,
                    max_ticks=self.ticks, tvl_usd=tvl,
                    kind=ArcKind.SWAP_UNIV4, venue="uniswap v4",
                    label="Uniswap v4")
                want_i = 0 if zero_for_one else 1
                arcs += [a for a in built if a.i == want_i]
                bank = univ3.arcs(st, ticks, zero_for_one=zero_for_one,
                                  max_ticks=self.ticks)
                if not bank:
                    continue
                i, j = (0, 1) if zero_for_one else (1, 0)
                banks[(pid, i, j)] = bank
                priced[(pid, i, j)] = Bank(
                    bank, dec0 if zero_for_one else dec1,
                    dec1 if zero_for_one else dec0)
        self.arcs, self.banks, self.priced = arcs, banks, priced
        self.block = block
        return len(state)

    # ---------------------------------------------------------- the seams

    def teach(self, client) -> None:
        """Let the walk price a v4 leg from its bank.

        The same teacher v3 uses: the bank is the same object and the walk does
        not care which venue filled it.  `univ3_client.teach` keys on the leg's
        `(target, i, j)`, and a `PoolId` is as good a target as an address.
        """
        if self.priced:
            teach(client, self.priced, ArcKind.SWAP_UNIV4)

    def collapse(self, arcs, psi, nu, nodes):
        # Its own kind: these are v3's banks by shape, not by contents.
        return univ3.collapse(arcs, psi, nu, nodes, self.banks,
                              kind=ArcKind.SWAP_UNIV4)

    @property
    def max_spread(self) -> float:
        return SPREAD if self.arcs else PATHOLOGICAL_CONDITION
