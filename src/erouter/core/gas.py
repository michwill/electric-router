"""Execution gas, as a cost and as a pruning bound (spec §11.1).

§11.1 keeps gas *out of the convex core*, and that is not squeamishness: a
fixed cost per arc is a step function, so the moment gas enters the objective
the program stops being convex and becomes mixed-integer.  The Laplacian
structure -- the entire reason this router is fast -- depends on it staying
out.

But gas can bound the problem from outside without entering it, and that is
worth more than it sounds.  A leg that carries less *value* than the leg costs
to execute cannot pay for itself under any circumstances: even capturing 100%
of the flow through it as profit would not cover the gas.  That makes

    psi_min = gas_leg * gas_price / eth_per_value_unit

a **sound** floor rather than a heuristic one -- it can never prune an arc that
belonged in the answer, because no arc below it can contribute more than it
costs.  It is also loose, deliberately: the true threshold is higher (an extra
leg wins only by the few bp of impact it saves, not by its whole flow), but a
loose sound bound is worth more than a tight unsound one.

The per-kind numbers are curve_solver's, which are calibrated against a
deployed router rather than guessed from quote gas.  Quote gas is the wrong
measure: `get_dy` is a view call, while `exchange` pays for transfers and
storage writes it never touches.
"""

from __future__ import annotations

from .types import ArcKind

# Paid once per transaction, not per leg.
TX_BASE = 71_000
# A split plan pays a little extra to distribute the input across branches.
SPLIT_OVERHEAD = 20_000

_BY_KIND: dict[ArcKind, int] = {
    ArcKind.SWAP_STABLE: 102_000,
    ArcKind.SWAP_CRYPTO: 102_000,
    ArcKind.DEPOSIT_FIXED: 71_000,
    ArcKind.DEPOSIT_DYN: 71_000,
    ArcKind.DEPOSIT_FIXED_NOFLAG: 71_000,
    ArcKind.WITHDRAW_STABLE: 107_000,
    ArcKind.WITHDRAW_CRYPTO: 107_000,
    ArcKind.ERC4626_DEPOSIT: 102_000,
    ArcKind.ERC4626_REDEEM: 102_000,
    # A native wrap is a deposit/withdraw on WETH and nothing else.
    ArcKind.WRAP_NATIVE: 40_000,
    ArcKind.UNWRAP_NATIVE: 40_000,
    ArcKind.WSTETH_WRAP: 60_000,
    ArcKind.WSTETH_UNWRAP: 60_000,
    ArcKind.STAKE_NATIVE: 60_000,
    # Measured: `cDAI.redeem` of 100k cDAI cost 172,906 -- a lending redemption
    # touches an interest-accrual write a swap never does, so it is dearer than
    # any of them.  `facts` replaces this with the real figure per token.
    ArcKind.LEND_MINT: 170_000,
    ArcKind.LEND_REDEEM: 173_000,
}

# What an unrecognised leg is assumed to cost -- the swap figure, so a new arc
# kind is never accidentally free.
DEFAULT_LEG = 102_000


def leg_gas(kind: ArcKind) -> int:
    return _BY_KIND.get(kind, DEFAULT_LEG)


def route_gas(kinds, *, legs: int | None = None) -> int:
    """Total gas for one execution plan, base and split overhead included."""
    kinds = list(kinds)
    total = TX_BASE + sum(leg_gas(k) for k in kinds)
    count = len(kinds) if legs is None else legs
    if count > 1:
        total += SPLIT_OVERHEAD
    return total


class GasTable:
    """Per-leg gas that was *executed* rather than assumed.

    The flat per-kind figures above are wrong in a biased direction and wrong
    by different amounts per pool.  Measured against the same block (see
    `dev/gas_probe.py`): 3pool DAI>USDC 118,778, 3pool USDC>USDT 121,702,
    crvUSD/USDC 124,839, TricryptoUSDC 155,891 -- all priced at a flat 102,000,
    so the error runs from +16% to +53%, and a crypto pool costs a third more
    than a stable one while the table charges them the same.

    Two things this deliberately does not claim:

    - **A measurement is per direction, not per pool.**  USDT's transfer costs
      more than DAI's, and the same pool differs by ~3,000 gas between its own
      pairs, so the key carries `(i, j)`.  A per-pool figure is accepted as a
      fallback under `(-1, -1)` for arcs no route exercised.
    - **Each leg was measured cold, alone.**  A later leg in a real route
      inherits warm accounts from its predecessors and costs less, so a sum
      over legs is an upper bound.  Erring high is the safe direction: gas
      enters only as a candidate-ranking term and as the §11.1 pruning floor,
      and over-charging a leg can only make the router prefer a shorter route
      or decline to prune, never invent a route that does not pay.

    Lookup walks from the specific to the general, so a pool that appeared
    after the last calibration is still priced from something we measured
    rather than from the flat guess:

    1. this direction of this pool, measured;
    2. any direction of this pool, under `(-1, -1)` -- a new pair on a known
       pool;
    3. the measured median for this arc kind -- a wholly new pool;
    4. the static per-kind figure, which by then means we have measured
       nothing of this kind at all.

    Tier 3 is what makes a new pool cheap to price correctly.  The caller is
    expected to narrow it further where it can: `dev/gas_cache.py` fills tier 2
    for every pool in the universe from the median of its own *class*, since a
    tricrypto pool resembles other tricrypto pools far more than it resembles
    the all-kinds median.
    """

    __slots__ = ("kinds", "legs")

    def __init__(self, legs: dict[tuple[str, int, int, int], int] | None = None,
                 kinds: dict[int, int] | None = None):
        self.legs = dict(legs or {})
        self.kinds = {int(k): int(v) for k, v in (kinds or {}).items()}

    def gas(self, kind: ArcKind, target: str = "", i: int = 0, j: int = 0) -> int:
        key = (target.lower(), int(kind), int(i), int(j))
        got = self.legs.get(key)
        if got is None:  # a per-pool figure, if this direction was missed
            got = self.legs.get((target.lower(), int(kind), -1, -1))
        if got is None:  # a pool we have never executed
            got = self.kinds.get(int(kind))
        return int(got) if got is not None else leg_gas(kind)

    def __len__(self) -> int:
        return len(self.legs)

    def __bool__(self) -> bool:
        return bool(self.legs)


#: The empty table: every leg falls back to the static per-kind figure.
STATIC = GasTable()


def plan_gas(legs, table: GasTable | None = None) -> int:
    """Total gas for an executable plan, preferring measured per-leg figures.

    Takes `Leg`s rather than kinds, because the measurement is keyed by which
    pool and which direction -- that is the whole point of measuring.
    """
    table = table or STATIC
    legs = list(legs)
    total = TX_BASE + sum(table.gas(x.kind, x.target, x.i, x.j) for x in legs)
    if len(legs) > 1:
        total += SPLIT_OVERHEAD
    return total


def value_per_gas(gas_price_wei: int, value_units_per_eth: float) -> float:
    """Cost of one unit of gas, in the solver's value units."""
    if gas_price_wei <= 0 or value_units_per_eth <= 0:
        return 0.0
    return gas_price_wei / 1e18 * value_units_per_eth


def min_useful_flow(
    gas_price_wei: int,
    value_units_per_eth: float,
    *,
    kind: ArcKind = ArcKind.SWAP_STABLE,
) -> float:
    """The sound floor described above, in value units.

    Returns 0 when gas is disabled, which restores the previous behaviour
    exactly rather than silently applying some default price.
    """
    return leg_gas(kind) * value_per_gas(gas_price_wei, value_units_per_eth)
