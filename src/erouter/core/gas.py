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
