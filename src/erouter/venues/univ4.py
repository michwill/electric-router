"""Uniswap v4: the same tick math as v3, gated on what the pool's hook can do.

The swap arithmetic is v3's exactly -- between two initialized ticks a v4 pool
is a constant product on virtual reserves, so `venues.univ3` builds the arcs and
this module decides *which pools are allowed to have any*.

That decision is unusually cheap, and it is the whole reason v4 is tractable.  A
hook's permissions are the low fourteen bits of its own address; the
`PoolManager` validates them at creation, so what a hook may do is readable from
the `PoolKey` with no call and no state.  Verified against
`Uniswap/v4-core/src/libraries/Hooks.sol`.

Only four of the fourteen bits can touch a swap.  `beforeAddLiquidity`,
`afterRemoveLiquidity`, `beforeDonate` and the rest are not on the swap path at
all -- a hook implementing only those is invisible to a swapper, however
elaborate it is.  So "has a hook" and "cannot be modelled" are very different
sets, and most of v4 is in the first and not the second.

Measured on ethereum at block 25,943,016, all 132,609 pools ever initialised,
by quoting a real $10,000 swap through `V4Quoter` and comparing the realised
rate against spot:

    tier                        pools   quoted   usable <100bp   median cost
    0 no hook                 100,765   19,081             373      9,992 bp
    1 hook off the swap path    1,678    1,349             729         68 bp
    2 runs on swap              1,893      264               2      7,598 bp
    3 dynamic fee               1,753      701               9      6,762 bp
    3 returns delta            26,520   15,694              64      8,008 bp

Three things in that table decided the cut.

**Tier 1 is the best tier there is**, and by a wide margin: 729 usable pools
against tier 0's 373, at a median cost of 68 bp against 9,992.  A hook is a
signal that somebody built the pool deliberately and funded it, where a pool
with no hook at all is what a launchpad mints by the thousand.  Admitting tier 1
roughly triples what v4 is worth to this router.

**Tier 2 is not worth having**: two usable pools out of 1,893, and 84% of them
revert on a plain $10,000 quote.  A hook that runs on the swap path cannot alter
the amounts without a returns-delta bit, so the arithmetic would be ours -- but
there is nothing there to route through.

**Tier 3 is not merely unmodellable, it is poisonous.**  1,413 of the
returns-delta pools quote *better than spot*, the worst by 2.1e49 bp.  A hook
that appears to pay you is exactly the shape that produced `eps = -5.4e7` on an
unpriced node and cost `FRAX -> WBTC` 4,805 bp: the relaxation empties the whole
trade into it.  Excluding these is not caution, it is the difference between a
graph that solves and one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Hook permissions, as the bit each occupies in the hook's own address.
FLAGS: dict[str, int] = {
    "beforeInitialize": 1 << 13,
    "afterInitialize": 1 << 12,
    "beforeAddLiquidity": 1 << 11,
    "afterAddLiquidity": 1 << 10,
    "beforeRemoveLiquidity": 1 << 9,
    "afterRemoveLiquidity": 1 << 8,
    "beforeSwap": 1 << 7,
    "afterSwap": 1 << 6,
    "beforeDonate": 1 << 5,
    "afterDonate": 1 << 4,
    "beforeSwapReturnsDelta": 1 << 3,
    "afterSwapReturnsDelta": 1 << 2,
    "afterAddLiquidityReturnsDelta": 1 << 1,
    "afterRemoveLiquidityReturnsDelta": 1 << 0,
}
ALL_HOOK_MASK = (1 << 14) - 1

#: The hook may *replace* part of the swap's arithmetic, so nothing read from
#: the pool predicts the outcome.
RETURNS_DELTA = FLAGS["beforeSwapReturnsDelta"] | FLAGS["afterSwapReturnsDelta"]
#: The hook runs on the swap path.  Without a returns-delta bit it cannot change
#: the amounts -- but it can revert, and it can move the pool before the swap.
ON_SWAP = FLAGS["beforeSwap"] | FLAGS["afterSwap"]
#: An lp fee of exactly this in the `PoolKey` means the hook sets the fee per
#: swap.  Not a valid static fee: `MAX_LP_FEE` is 1,000,000.
DYNAMIC_FEE = 0x800000

#: Tiers this router will build arcs for.  See the module docstring for why the
#: line is here and not one tier further out.
ROUTABLE = (0, 1)


def permissions(hook: str) -> set[str]:
    """Every callback this hook address is allowed to implement."""
    bits = int(hook, 16) & ALL_HOOK_MASK
    return {name for name, bit in FLAGS.items() if bits & bit}


def tier(hook: str, fee: int) -> int:
    """0, 1, 2 or 3 -- how far this pool's hook is from the swap.

    `fee` is the `PoolKey`'s, not the pool's current one: a dynamic-fee pool is
    dynamic whatever it last charged.
    """
    if fee == DYNAMIC_FEE:
        return 3
    bits = int(hook, 16) & ALL_HOOK_MASK
    if bits & RETURNS_DELTA:
        return 3
    if int(hook, 16) == 0:
        return 0
    if bits & ON_SWAP:
        return 2
    return 1


def routable(hook: str, fee: int) -> bool:
    """Whether this router will model the pool at all."""
    return tier(hook, fee) in ROUTABLE


@dataclass(frozen=True, slots=True)
class PoolKey:
    """What names a v4 pool.  There is no per-pool address: one singleton holds
    them all, and this tuple is the identity."""

    currency0: str
    currency1: str
    fee: int
    tick_spacing: int
    hooks: str

    @classmethod
    def from_row(cls, row) -> PoolKey:
        """From a census row, `[c0, c1, fee, spacing, hooks, tvl]`."""
        return cls(str(row[0]).lower(), str(row[1]).lower(), int(row[2]),
                   int(row[3]), str(row[4]).lower())

    @property
    def tier(self) -> int:
        return tier(self.hooks, self.fee)

    @property
    def routable(self) -> bool:
        return routable(self.hooks, self.fee)

    def as_tuple(self) -> tuple:
        """The ABI ordering `PoolKey` is encoded in."""
        return (self.currency0, self.currency1, self.fee,
                self.tick_spacing, self.hooks)
