"""Send a route through `ElectricRouter` -- on a fork today, on a chain later.

`executor.py` runs the *quoter's* leg encoding with every bound switched off,
because its question is "would this route run at all".  This one runs the
router's encoding with the bounds on, because its question is the opposite:
does a call a user would actually sign deliver what the quote promised.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from ..core.routecall import RouteCall

CONTRACT = pathlib.Path(__file__).resolve().parents[3] / "contracts" / "ElectricRouter.vy"

NATIVE = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
GAS_HEADROOM_WEI = 10**19

#: `Error(string)`.
REVERT_SELECTOR = bytes.fromhex("08c379a0")


def revert_reason(exc: Exception) -> str:
    """The router's own message, dug out of boa's call trace.

    Worth the digging: the trace leads with every argument, and a route with
    six pools pushes the reason past any sane truncation -- so a caller reading
    `error` would see addresses where the diagnosis should be.
    """
    from ..core.codec import decode_result

    frame = exc.args[0] if exc.args else None
    output = getattr(getattr(frame, "computation", None), "output", b"")
    if output[:4] == REVERT_SELECTOR:
        try:
            return str(decode_result(["string"], output[4:])[0])
        except Exception:
            pass
    return ""


@dataclass(slots=True)
class Sent:
    """What the router really paid, and what it cost to find out."""

    amount_out: int = 0
    quoted_out: int | None = None
    gas: int = 0
    calldata_bytes: int = 0
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error and self.amount_out > 0

    @property
    def drift_bp(self) -> float:
        if not self.quoted_out:
            return 0.0
        return (self.amount_out - self.quoted_out) / self.quoted_out * 1e4


def deploy(name: str = "ElectricRouter"):
    import boa

    return boa.loads(CONTRACT.read_text(), name=name)


def send(
    call: RouteCall,
    *,
    router=None,
    quoted_out: int | None = None,
    wrapped: str = "",
    holders: list[str] | None = None,
    expect_block: int | None = None,
) -> Sent:
    """Run `call` against the active fork.  Never raises; reports instead.

    A revert is an answer here -- it is what a user would have seen -- so it
    lands in `error` rather than propagating out of a batch.
    """
    import boa

    from .executor import ERC20_ABI, _fund

    result = Sent(quoted_out=quoted_out,
                  calldata_bytes=len(call.calldata(sender=call.receiver)))
    if expect_block is not None:
        at = int(boa.env.evm.patch.block_number)
        if at != expect_block:
            result.error = (
                f"fork is at block {at:,}, the quote was pinned to "
                f"{expect_block:,}; the comparison would be meaningless")
            return result

    contract = router or deploy()
    native_in = call.token_in.lower() == NATIVE

    try:
        with boa.env.anchor():
            who = boa.env.generate_address()
            boa.env.set_balance(
                who, GAS_HEADROOM_WEI + (call.amount_in if native_in else 0))
            if not native_in:
                token = boa.loads_abi(ERC20_ABI).at(call.token_in)
                _fund(boa, token, who, call.amount_in, result, wrapped, holders)
                with boa.env.prank(who):
                    token.approve(contract.address, call.amount_in)
            with boa.env.prank(who):
                result.amount_out = contract.execute(
                    call.amount_in,
                    list(call.pools),
                    list(call.params),
                    call.set_approvals,
                    list(call.tokens),
                    who,
                    call.min_out,
                    value=call.amount_in if native_in else 0,
                )
            result.gas = int(contract._computation.get_gas_used())
    except Exception as exc:
        result.error = (revert_reason(exc)
                        or f"{type(exc).__name__}: {exc}".strip()[:400])
    return result
