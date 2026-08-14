"""Which arcs quote and then revert, learned once by trying them.

`get_dy` runs a pool's invariant over its own accounting.  It does not ask
whether the coins exist, whether the protocol beneath will take a deposit, or
whether anyone paused anything -- so a quote can be a faithful reading of a
fiction.  Three shapes of that, all live on mainnet today:

* **Frozen reserves.**  Aave V2's pool quotes 9,996 USDC for 10k DAI and
  reverts, because `exchange_underlying` has to deposit into a frozen reserve.
* **Paused mint.**  Compound V2 answers "mint is paused" to the same move,
  while `USDT->DAI` through the same family executes -- that direction only
  *redeems* on the way out.
* **Retired tokens.**  sUSD/sUSDe quoted 6.5x and reverted with "sUSD retired".
  `check_reserves_are_real` catches that one from the balances; the other two
  leave no trace in state at all.

None of it is visible without executing, and none of it changes between blocks,
which is exactly the shape of a fact worth storing.  So this runs when the
facts file is built and the router reads the answer -- no probing on the route
path, where a slow node would be paying for it.

Deprecated protocols stop taking deposits long before they stop honouring
withdrawals, so capability is recorded per direction.  That is what lets a
lending pool keep its redeem arcs instead of being blacklisted whole.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from ..core.keccak import keccak256
from .gas_probe import (
    CALLER,
    Funder,
    _pad,
    revert_reason,  # re-exported: callers read it from here
)

__all__ = ["WRAPPER_CALLS", "Capability", "revert_reason", "try_wrapper"]


#: How each wrapper is entered and left.  `None` means the protocol has no such
#: call, which is not the same as the call failing.
WRAPPER_CALLS = {
    "ctoken": {
        "mint": lambda token, underlying, amount, owner: (
            token, keccak256(b"mint(uint256)")[:4] + _pad(amount), underlying),
        "redeem": lambda token, underlying, amount, owner: (
            token, keccak256(b"redeem(uint256)")[:4] + _pad(amount), token),
    },
    "erc4626": {
        "mint": lambda token, underlying, amount, owner: (
            token, keccak256(b"deposit(uint256,address)")[:4] + _pad(amount) + _pad(owner),
            underlying),
        "redeem": lambda token, underlying, amount, owner: (
            token,
            keccak256(b"redeem(uint256,address,address)")[:4]
            + _pad(amount) + _pad(owner) + _pad(owner),
            token),
    },
}


@dataclass(slots=True)
class Capability:
    address: str
    family: str
    mint: bool = False
    redeem: bool = False
    notes: dict = field(default_factory=dict)


def try_wrapper(evm, funder: Funder, *, token: str, underlying: str, family: str,
                amount: int) -> Capability:
    """Can this wrapper still be entered, and can it still be left?

    Both are attempted independently, because on a deprecated protocol they
    genuinely differ: Compound V2 answers "mint is paused" and redeems fine.
    """
    out = Capability(address=token.lower(), family=family)
    calls = WRAPPER_CALLS.get(family)
    if calls is None:
        return out
    for direction, build in calls.items():
        target, data, spend = build(token, underlying, amount, CALLER)
        snapshot = evm.snapshot()
        try:
            if not funder.fund(spend, target, amount * 2):
                out.notes[direction] = "not funded"
                continue
            try:
                evm.message_call(caller=CALLER, to=target, calldata=data)
                setattr(out, direction, True)
            except Exception as exc:
                out.notes[direction] = revert_reason(exc)
        finally:
            with contextlib.suppress(Exception):
                evm.revert(snapshot)
    return out
