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

    # A metapool is the same invariant with the base pool's LP valued at its
    # `virtual_price` rather than at 1 -- so it is a *rates* problem, not a
    # solver problem, and `StableSwap` already takes rates as an argument.
    # `stored_rates()` reports this on the ng pools that have it; the older
    # ones do not, and their LP coin then reads as a plain 18-decimal token,
    # which is wrong by exactly the accrued yield.
    issuer = {p.lp_token.lower(): p.address for p in pools if p.lp_token}
    lp_coins = {
        pool.address.lower(): [
            issuer[coin.address.lower()] for coin in pool.coins
            if coin.address.lower() in issuer
        ]
        for pool in wanted
    }
    bases = sorted({a for v in lp_coins.values() for a in v})
    virtual: dict[str, int] = {}
    if bases:
        answers = client.raw([Call(a, encode_call("get_virtual_price()")) for a in bases])
        for address, answer in zip(bases, answers, strict=True):
            if answer.ok and answer.uint():
                virtual[address.lower()] = answer.uint()

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

        reported: tuple[int, ...] = ()
        if rates_raw.ok and rates_raw.data:
            try:
                reported = tuple(decode(["uint256[]"], rates_raw.data)[0])
            except Exception:  # noqa: BLE001 -- a pool without the getter
                reported = ()

        # The legacy shape: the rate is only the decimal correction -- except
        # for a base-pool LP, which is worth its virtual price.
        plain = tuple(10 ** (36 - c.decimals) for c in pool.coins)
        meta = tuple(
            virtual.get(issuer.get(c.address.lower(), "").lower(), 0)
            or 10 ** (36 - c.decimals)
            for c in pool.coins
        )
        candidates = []
        if len(reported) == len(pool.coins):
            candidates.append(reported)
        candidates.append(plain)
        if meta != plain:
            candidates.append(meta)

        # Which convention this pool follows is asked, not assumed: the
        # variants are cheap to evaluate and only one of them can reproduce
        # the chain to the wei.
        for rates in candidates:
            for on_xp in (fee_on_xp, not fee_on_xp):
                built.append((pool, StableSwap(
                    balances=tuple(int(b) for b in pool.balances),
                    rates=rates, amp=amp, fee=fee.uint(),
                    offpeg_fee_multiplier=offpeg.uint_or(0) or 0,
                    a_precision=a_precision, fee_on_xp=on_xp,
                )))

    if not built:
        return out

    # --- and now make each one prove itself -------------------------------
    # One probe per pool, however many variants it produced: the quote is the
    # same question, and the variants are decided in Python against it.
    order: list[object] = []
    seen_pools: set[str] = set()
    for pool, _ in built:
        if pool.address.lower() not in seen_pools:
            seen_pools.add(pool.address.lower())
            order.append(pool)
    probes = [
        Probe(pool.address, ArcKind.SWAP_STABLE, 0, 1, len(pool.coins),
              max(1, int(pool.balances[0] * CHECK_FRACTION)))
        for pool in order
    ]
    quotes = client.probe(probes)
    truth = {
        pool.address.lower(): (probe.dx, quote)
        for pool, probe, quote in zip(order, probes, quotes, strict=True)
    }

    for pool in order:
        out.checked += 1
        key = pool.address.lower()
        dx, quote = truth[key]
        if not quote.ok or quote.value <= 0:
            out.rejected.append((pool.address, "pool would not quote the check"))
            continue
        best: str = ""
        for candidate_pool, model in built:
            if candidate_pool.address.lower() != key:
                continue
            try:
                mine = model.get_dy(0, 1, dx)
            except StableSwapError as exc:
                best = best or str(exc)[:40]
                continue
            if mine == quote.value:
                # Not a tolerance.  Agreeing to the wei is the evidence that
                # every parameter was read correctly; anything less means one
                # was not, and the error will surface at another size.
                out.by_pool[key] = model
                break
            best = best or f"{mine} != {quote.value} at {dx}"
        else:
            out.rejected.append((pool.address, best or "no variant matched"))

    if not quiet:
        print(f"  exact stableswap: {len(out)} of {out.checked} pools reproduce "
              f"their own get_dy to the wei")
    return out
