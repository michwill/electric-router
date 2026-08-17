"""Read what a stableswap pool needs to be evaluated instead of probed.

`core/stableswap.py` runs the pool's own invariant, which turns `f(delta)` from
something measured into something computed -- exact at any size, and free once
the parameters are in hand.  This is the part that gets them: one batched pass
over the universe for `A`, the fee, the off-peg multiplier and the rates, and
the balances the arc list already carries.

**Every pool is checked against the chain before it is believed.**  That is not
belt-and-braces, it is the whole safety argument: a misread `A`, a rate array
in the wrong order or the wrong fee convention produces a curve that is
confidently wrong at every size, and unlike a failed probe it does not announce
itself.  So each candidate is quoted once for real and kept only if the
arithmetic reproduces it exactly.  A pool that does not reproduce keeps being
probed, which is what every pool did before this existed.

The two dialects are separated by asking rather than by guessing: a pool that
answers `A_precise()` scales `A` by 100 and takes its fee in `xp` space; one
that does not is the legacy shape and takes the fee after converting back to
token units.  3pool disagrees by exactly one wei at $100k under the wrong
convention, which is how the split was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import decode, encode_call
from ..core.stableswap import StableSwap, StableSwapError
from ..core.transport import Call
from ..core.types import ArcKind, Probe

#: One `get_dy` per pool, at a size big enough that a wrong `A` shows up.  A
#: dust probe agrees with almost any parameters; a real one does not.
CHECK_FRACTION = 0.01


@dataclass(slots=True)
class ExactPools:
    """Verified stableswap models, by pool address."""

    by_pool: dict[str, StableSwap] = field(default_factory=dict)
    checked: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_pool)

    def get(self, pool: str):
        return self.by_pool.get(pool.lower())


def _stable(pool) -> bool:
    """Whether this pool is worth asking about at all."""
    kind = getattr(pool, "swap_kind", None)
    return kind in (ArcKind.SWAP_STABLE, None) and len(pool.coins) in (2, 3, 4)


def build_exact_pools(pools, client, *, quiet: bool = True) -> ExactPools:
    """Model every stableswap whose parameters reproduce its own `get_dy`."""
    out = ExactPools()
    wanted = [p for p in pools if _stable(p) and p.balances and all(p.balances)]
    if not wanted:
        return out

    calls: list[Call] = []
    for pool in wanted:
        calls += [
            Call(pool.address, encode_call("A_precise()")),
            Call(pool.address, encode_call("A()")),
            Call(pool.address, encode_call("fee()")),
            Call(pool.address, encode_call("offpeg_fee_multiplier()")),
            Call(pool.address, encode_call("stored_rates()")),
        ]
    answers = client.raw(calls)

    built: list[tuple[object, StableSwap]] = []
    for k, pool in enumerate(wanted):
        precise, plain, fee, offpeg, rates_raw = answers[5 * k : 5 * k + 5]
        if precise.ok and precise.uint():
            amp, a_precision, fee_on_xp = precise.uint(), 100, True
        elif plain.ok and plain.uint():
            amp, a_precision, fee_on_xp = plain.uint(), 1, False
        else:
            continue
        if not fee.ok:
            continue

        rates: tuple[int, ...] = ()
        if rates_raw.ok and rates_raw.data:
            try:
                rates = tuple(decode(["uint256[]"], rates_raw.data)[0])
            except Exception:  # noqa: BLE001 -- a pool without the getter
                rates = ()
        if len(rates) != len(pool.coins):
            # The legacy shape: the rate is only the decimal correction.
            rates = tuple(10 ** (36 - c.decimals) for c in pool.coins)

        model = StableSwap(
            balances=tuple(int(b) for b in pool.balances),
            rates=rates, amp=amp, fee=fee.uint(),
            offpeg_fee_multiplier=offpeg.uint_or(0) or 0,
            a_precision=a_precision, fee_on_xp=fee_on_xp,
        )
        built.append((pool, model))

    if not built:
        return out

    # --- and now make each one prove itself -------------------------------
    probes: list[Probe] = []
    for pool, model in built:
        dx = max(1, int(pool.balances[0] * CHECK_FRACTION))
        probes.append(Probe(pool.address, ArcKind.SWAP_STABLE, 0, 1,
                            len(pool.coins), dx))
    quotes = client.probe(probes)

    for (pool, model), quote, probe in zip(built, quotes, probes, strict=True):
        out.checked += 1
        if not quote.ok or quote.value <= 0:
            out.rejected.append((pool.address, "pool would not quote the check"))
            continue
        try:
            mine = model.get_dy(0, 1, probe.dx)
        except StableSwapError as exc:
            out.rejected.append((pool.address, str(exc)[:40]))
            continue
        if mine != quote.value:
            # Not a tolerance.  Agreeing to the wei is the evidence that every
            # parameter was read correctly; anything less means one of them was
            # not, and the error will be somewhere else at another size.
            out.rejected.append(
                (pool.address, f"{mine} != {quote.value} at {probe.dx}"))
            continue
        out.by_pool[pool.address.lower()] = model

    if not quiet:
        print(f"  exact stableswap: {len(out)} of {out.checked} pools reproduce "
              f"their own get_dy to the wei")
    return out
