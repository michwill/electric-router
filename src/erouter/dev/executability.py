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

__all__ = ["WRAPPER_CALLS", "Capability", "discover_wrappers",
           "revert_reason", "try_wrapper"]


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
    """What a wrapper will still do.  `None` means untested, not refused.

    The distinction is the whole point.  cUSDC's redeem could not be funded --
    conjuring a cToken means writing a balance the protocol computes rather
    than stores -- and reporting that as "cannot redeem" would deny a working
    arc for the rest of the file's life.  A direction is only ever recorded
    once it has actually been attempted.
    """

    address: str
    family: str
    mint: bool | None = None
    redeem: bool | None = None
    notes: dict = field(default_factory=dict)


def try_wrapper(evm, funder: Funder, *, token: str, underlying: str, family: str,
                amount: int, holders: dict | None = None) -> Capability:
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
        caller = CALLER
        try:
            if not funder.fund(spend, target, amount * 2):
                # A share balance the protocol computes rather than stores
                # cannot be conjured by writing a slot, so borrow it from
                # someone holding the token -- a pool's own reserves will do.
                lent = (funder.lend_from(spend, (holders or {}).get(spend.lower(), ""),
                                         target, amount)
                        if (holders or {}).get(spend.lower()) else None)
                if lent is None:
                    out.notes[direction] = "not funded"
                    continue      # leaves the direction `None`: untested
                caller = lent
                evm.set_balance(caller, 10 ** 20)
            try:
                evm.message_call(caller=caller, to=target, calldata=data)
                setattr(out, direction, True)
            except Exception as exc:
                setattr(out, direction, False)
                out.notes[direction] = revert_reason(exc)
        finally:
            with contextlib.suppress(Exception):
                evm.revert(snapshot)
    return out


def discover_wrappers(pools, client) -> list[tuple[str, str, str]]:
    """Every coin that wraps another token, found by asking rather than listing.

    Mintability and redeemability are properties of a *token*, not of a pool or
    a swap, and the asymmetry is everywhere once looked for: Compound's mint is
    paused while redeem works, Aave's reserves are frozen, sUSDe mints on
    demand and redeems on a seven-day cooldown, pufETH redeems through a queue,
    sfrxUSD reports `maxDeposit == 0`.  E8 found thirty-one linear ERC4626
    tokens and concluded that linearity says nothing about whether you can get
    back out -- which is why merging them is gated on a hand-written allowlist.

    Executing both directions answers what that allowlist is guessing at, so
    the discovery has to be universal: every coin of every pool at every index,
    not a list someone maintains.

    `asset()` identifies an ERC4626 vault and `underlying()` a cToken.  Both are
    view calls and go out in one batch.
    """
    from ..core.codec import encode_call
    from ..core.transport import Call, Status

    tokens = sorted({coin.address.lower() for pool in pools for coin in pool.coins})
    if not tokens:
        return []
    calls: list[Call] = []
    for token in tokens:
        calls.append(Call(token, encode_call("asset()")))
        calls.append(Call(token, encode_call("underlying()")))
    answers = client.raw(calls)

    found: list[tuple[str, str, str]] = []
    for k, token in enumerate(tokens):
        for answer, family in ((answers[2 * k], "erc4626"),
                               (answers[2 * k + 1], "ctoken")):
            if answer.status is not Status.VALUE or len(answer.data) < 32:
                continue
            underlying = "0x" + answer.data[-20:].hex()
            if int(underlying, 16) == 0 or underlying == token:
                continue
            found.append((token, underlying, family))
            break
    return found
