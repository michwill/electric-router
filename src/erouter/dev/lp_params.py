"""Read what a pool's LP arcs need to be evaluated instead of probed.

Deposits and withdrawals run the invariant the swap models already reproduce;
what is new is that `D` itself moves, and that each operation has its own fee
convention.  Both are settled by the deployed source rather than assumed --
`calc_token_amount` takes no fee on the legacy pools, `calc_withdraw_one_coin`
charges on the imbalance -- and then checked the same way everything else here
is: build it, ask the pool, keep it only if the arithmetic reproduces the
answer to the wei at every point.

Only pools whose *swap* model was already admitted are candidates.  That is not
a shortcut, it is the precondition: `D`, `A`, the rates and the fee all come
from that model, so a pool whose swaps do not reproduce has no business being
trusted for its LP arcs either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import encode_call
from ..core.stableswap import StableSwapError, StableSwapLP
from ..core.transport import Call
from ..core.types import ArcKind

#: Fractions of total supply to check a withdrawal at, and of each balance for
#: a deposit.  Three decades, because a fee charged on an imbalance is exactly
#: the kind of term that agrees at one size and not another.
CHECK_FRACTIONS = (0.0001, 0.001, 0.01)

WITHDRAW = ArcKind.WITHDRAW_STABLE
DEPOSITS = (ArcKind.DEPOSIT_FIXED, ArcKind.DEPOSIT_DYN,
            ArcKind.DEPOSIT_FIXED_NOFLAG)


@dataclass(slots=True)
class ExactLP:
    by_pool: dict[str, StableSwapLP] = field(default_factory=dict)
    checked: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_pool)

    def get(self, pool: str):
        return self.by_pool.get(pool.lower())


def build_exact_lp(pools, swaps, client, *, quiet: bool = True) -> ExactLP:
    """Model the LP arcs of every pool whose swap model already reproduces."""
    out = ExactLP()
    wanted = [p for p in pools if swaps.get(p.address) is not None]
    if not wanted:
        return out

    supplies = client.raw([Call(p.address, encode_call("totalSupply()"))
                           for p in wanted])
    # A pool whose LP token is a separate contract reports supply there.
    missing = [p for p, s in zip(wanted, supplies, strict=True)
               if not (s.ok and s.uint())]
    extra = {}
    if missing:
        answers = client.raw([Call(p.lp_token, encode_call("totalSupply()"))
                              for p in missing if p.lp_token])
        for pool, answer in zip([p for p in missing if p.lp_token], answers,
                                strict=True):
            if answer.ok and answer.uint():
                extra[pool.address.lower()] = answer.uint()

    built: list[tuple[object, StableSwapLP]] = []
    for pool, supply in zip(wanted, supplies, strict=True):
        total = supply.uint() if (supply.ok and supply.uint()) else \
            extra.get(pool.address.lower(), 0)
        if total <= 0:
            continue
        built.append((pool, StableSwapLP(pool=swaps.get(pool.address),
                                         total_supply=total)))
    if not built:
        return out

    calls, at = [], []
    for pool, model in built:
        for frac in CHECK_FRACTIONS:
            burn = max(1, int(model.total_supply * frac))
            calls.append(Call(pool.address,
                              encode_call("calc_withdraw_one_coin(uint256,int128)",
                                          burn, 0)))
            at.append((pool.address.lower(), WITHDRAW, burn, 0))
    answers = client.raw(calls)

    truth: dict[str, list] = {}
    for (address, kind, size, i), answer in zip(at, answers, strict=True):
        truth.setdefault(address, []).append((kind, size, i, answer))

    for pool, model in built:
        out.checked += 1
        key = pool.address.lower()
        points = [(k, s, i, a) for k, s, i, a in truth.get(key, [])
                  if a.ok and a.uint() > 0]
        if not points:
            out.rejected.append((pool.address, "pool would not quote the check"))
            continue
        failed = ""
        for kind, size, i, answer in points:
            try:
                mine = model.calc_withdraw_one_coin(size, i)
            except (StableSwapError, ZeroDivisionError) as exc:
                failed = str(exc)[:44]
                break
            if mine != answer.uint():
                failed = f"{mine} != {answer.uint()} burning {size}"
                break
        if failed:
            out.rejected.append((pool.address, failed))
        else:
            out.by_pool[key] = model

    if not quiet:
        print(f"  exact LP: {len(out)} of {out.checked} pools reproduce their own "
              f"calc_withdraw_one_coin to the wei")
    return out
