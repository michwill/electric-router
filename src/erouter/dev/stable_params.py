"""Read what a stableswap pool needs to be evaluated instead of probed.

`core/stableswap.py` runs the pool's own invariant, which turns `f(delta)` from
something measured into something computed -- exact at any size, and free once
the parameters are in hand.  This is the part that gets them: one batched pass
over the universe for `A`, the fee, the off-peg multiplier and the rates.

**Every pool is checked against the chain before it is believed.**  That is the
whole safety argument: a misread `A`, a rate array in the wrong order or the
wrong fee convention produces a curve that is confidently wrong at every size
and, unlike a failed probe, does not announce itself.  So each candidate is
quoted once for real and kept only if the arithmetic reproduces it exactly.  A
pool that does not reproduce keeps being probed.

The two dialects are separated by asking rather than by guessing: a pool that
answers `A_precise()` scales `A` by 100 and takes its fee in `xp` space; one
that does not is the legacy shape and takes the fee after converting back to
token units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import encode_call
from ..core.stableswap import StableSwap, StableSwapError
from ..core.transport import Call
from ..core.types import ArcKind, Probe
from .exact_cache import trust as _trust_verdict
from .lending_params import build_exact_lending, build_exact_rate_pools

#: Sizes to check at, as fractions of the input balance, in both directions.
#
# One point is not enough, and that is measured rather than cautious: a
# cryptoswap model built from the wrong invariant reproduced a pool at 1% of
# balance and disagreed with it at five of the six other points tried.  A model
# that matches where it was checked and diverges elsewhere is worse than no
# model, because nothing downstream will ever ask again.  A dust probe also
# agrees with almost any `A`, so the ladder spans three decades.
CHECK_FRACTIONS = (0.001, 0.01, 0.1)


@dataclass(slots=True)
class ExactPools:
    """Verified stableswap models, by pool address."""

    by_pool: dict[str, StableSwap] = field(default_factory=dict)
    checked: int = 0
    #: Admitted on a remembered verdict, without re-gating.
    trusted: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_pool)

    def get(self, pool: str):
        return self.by_pool.get(pool.lower())


def _stable(pool) -> bool:
    """Whether this pool is worth asking about at all."""
    kind = getattr(pool, "swap_kind", None)
    return kind in (ArcKind.SWAP_STABLE, None) and len(pool.coins) in (2, 3, 4)


def _decode_rates(blob: bytes | None, n_coins: int) -> tuple[int, ...]:
    """`stored_rates()`, whichever array shape the pool returns.

    The ng pools return a `DynArray`, which arrives as an offset, a length and
    then the items.  The older factory pools return `uint256[N_COINS]`, which
    for two coins is 64 bytes with no header at all -- and those are precisely
    the pools where the rate is not the decimal correction, because a plain pool
    only grew a `stored_rates` when someone attached a rate oracle to it.
    Demanding a header dropped them in silence and valued ETHx at one ETH: a
    9.5% error the pool announces and we were not reading.

    Length alone cannot separate the two shapes -- 128 bytes is a two-coin
    dynamic array and a four-coin fixed one.  A dynamic array announces itself,
    first word the offset 32 and second its own length, and no rate is ever 32,
    so read the header where it is there and the bare words where it is not.
    """
    if not blob:
        return ()
    words = [int.from_bytes(blob[k : k + 32], "big")
             for k in range(0, len(blob) - 31, 32)]
    if len(words) >= 2 + n_coins and words[0] == 32 and words[1] == n_coins:
        return tuple(words[2 : 2 + n_coins])
    if len(words) == n_coins:
        return tuple(words)
    return ()


def build_exact_pools(pools, client, *, quiet: bool = True,
                      cache=None, resample=(), only=None) -> ExactPools:
    """Model every stableswap whose parameters reproduce its own `get_dy`."""
    out = ExactPools()
    wanted = [p for p in pools if _stable(p) and p.balances and all(p.balances)
              and (only is None or p.address.lower() in only)]
    if not wanted:
        return out

    # A metapool is the same invariant with the base pool's LP valued at its
    # `virtual_price` rather than at 1 -- so it is a *rates* problem, not a
    # solver problem.  `stored_rates()` reports this on the ng pools that have
    # it; on the older ones the LP coin reads as a plain 18-decimal token,
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
            # Only `exchange` needs this, never `get_dy`, so a pool that does
            # not report it still models and quotes; `StableSwap.exchange`
            # then refuses rather than assuming a flattering zero.
            Call(pool.address, encode_call("admin_fee()")),
        ]
    answers = client.raw(calls)

    # `stored_rates()` cannot come through the quoter: its `Res` struct holds one
    # `uint256`, so `raw_batch` hands back 32 bytes where a rates *array* is 128,
    # the first word being the ABI offset.  The truncation is silent and reads as
    # a valid answer, so every rate-bearing ng pool fell back to decimals-only
    # rates and modelled sUSDe as worth exactly one DOLA -- all 77 of the ng
    # rejections.  The transport returns whole returndata, so ask it directly.
    transport = getattr(client, "transport", None)
    rate_data: dict[str, bytes] = {}
    if transport is not None and hasattr(transport, "call_many"):
        wire_calls = [Call(p.address, encode_call("stored_rates()")) for p in wanted]
        for pool, answer in zip(wanted, transport.call_many(wire_calls), strict=True):
            if answer.ok and len(answer.data) >= 32 * len(pool.coins):
                rate_data[pool.address.lower()] = answer.data

    built: list[tuple[object, StableSwap, dict]] = []
    for k, pool in enumerate(wanted):
        precise, plain, fee, offpeg, admin = answers[5 * k : 5 * k + 5]
        if precise.ok and precise.uint():
            amp, a_precision, fee_on_xp = precise.uint(), 100, True
        elif plain.ok and plain.uint():
            amp, a_precision, fee_on_xp = plain.uint(), 1, False
        else:
            continue
        if not fee.ok:
            continue

        reported = _decode_rates(rate_data.get(pool.address.lower()),
                                 len(pool.coins))

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
            candidates.append(("reported", reported))
        candidates.append(("plain", plain))
        if meta != plain:
            candidates.append(("meta", meta))

        # Which convention this pool follows is asked, not assumed: the variants
        # are cheap to evaluate and only one of them can reproduce the chain to
        # the wei.  All are built whether or not a verdict predicts the winner,
        # so a stale verdict falls back to the gate instead of silently
        # dropping the pool.
        for label, rates in candidates:
            for on_xp in (fee_on_xp, not fee_on_xp):
                built.append((pool, StableSwap(
                    balances=tuple(int(b) for b in pool.balances),
                    rates=rates, amp=amp, fee=fee.uint(),
                    offpeg_fee_multiplier=offpeg.uint_or(0) or 0,
                    a_precision=a_precision, fee_on_xp=on_xp,
                    admin_fee=admin.uint_or(-1) if admin.ok else -1,
                ), {"family": "stable", "rates": label, "fee_on_xp": on_xp}))

    if not built:
        return out

    # --- and now make each one prove itself -------------------------------
    # One probe per pool, however many variants it produced: the quote is the
    # same question, and the variants are decided in Python against it.
    order: list[object] = []
    seen_pools: set[str] = set()
    for pool, _, _ in built:
        if pool.address.lower() not in seen_pools:
            seen_pools.add(pool.address.lower())
            order.append(pool)

    # A pool the cache has already seen prove itself does not prove itself
    # again: the gate establishes that a pool's *code* implements this
    # invariant, and Curve pools are not upgradeable.  `A`, the fee terms, the
    # ramp and the balances were all read fresh above, so what is trusted is
    # only "this variant, not one of the other five".
    order = [p for p in order
             if not _trust_verdict(out, cache, resample, built, p.address.lower())
             and not (cache is not None and cache.skip(p.address, p.balances))]

    probes, where = [], []
    for pool in order:
        for i, j in ((0, 1), (1, 0)):
            for frac in CHECK_FRACTIONS:
                probes.append(Probe(pool.address, ArcKind.SWAP_STABLE, i, j,
                                    len(pool.coins),
                                    max(1, int(pool.balances[i] * frac))))
                where.append(pool.address.lower())
    quotes = client.probe(probes)
    truth: dict[str, list] = {}
    for address, probe, quote in zip(where, probes, quotes, strict=True):
        truth.setdefault(address, []).append((probe, quote))

    for pool in order:
        out.checked += 1
        key = pool.address.lower()
        points = [(pr, q) for pr, q in truth.get(key, []) if q.ok and q.value > 0]
        if not points:
            # Recorded, not just reported: a pool that answers nothing is
            # empty or holds dust, and re-deriving that costs a probe per size
            # per direction on every run for as long as it stays that way.
            out.rejected.append((pool.address, "pool would not quote the check"))
            if cache is not None:
                cache.refuse(key, "would not quote", balances=pool.balances)
            continue
        best: str = ""
        for candidate_pool, model, variant in built:
            if candidate_pool.address.lower() != key:
                continue
            failed = ""
            for probe, quote in points:
                try:
                    mine = model.get_dy(probe.i, probe.j, probe.dx)
                except StableSwapError as exc:
                    failed = str(exc)[:40]
                    break
                # Not a tolerance.  Agreeing to the wei at *every* point is
                # the evidence that every parameter was read correctly.
                if mine != quote.value:
                    failed = f"{mine} != {quote.value} at {probe.dx}"
                    break
            if not failed:
                out.by_pool[key] = model
                if cache is not None:
                    cache.record(key, variant)
                break
            best = best or failed
        else:
            why = best or "no variant matched"
            out.rejected.append((pool.address, why))
            if cache is not None:
                cache.refuse(key, why, balances=pool.balances)

    # The lending pools reach the same holder by a different road: their coin
    # list runs past their balances, so the filter at the top of this function
    # never sees them, and their rates come from the wrappers rather than from
    # `stored_rates()`.  `lending_params` settles both and picks each pool's
    # variant by reproducing its own `get_dy` -- this gate, one level down.
    block = (getattr(getattr(client, "transport", None), "block", None)
             or getattr(client, "block", None))
    if block and only is None:
        try:
            lent, refused = build_exact_lending(pools, client, block=int(block))
        except Exception as exc:  # noqa: BLE001 - a reader must not fail a quote
            lent, refused = {}, [("lending", str(exc))]
        out.by_pool.update(lent)
        out.checked += len(lent) + len(refused)
        out.rejected.extend(refused)
        if lent and not quiet:
            print(f"  exact lending: {len(lent)} wrapped-token pools")

        # And the ones the ordinary reading could not reproduce: a coin whose
        # value is not one (rETH, ankrETH) or a moving target like RAI's
        # redemption price.  Only the rejects are retried, so a pool that models
        # correctly with a plain rate cannot be talked out of it.
        stubborn = {a.lower() for a, _ in out.rejected}
        if stubborn:
            try:
                rated = build_exact_rate_pools(
                    pools, client, addresses=stubborn,
                    # `virtual` is keyed by the *issuing pool*; the retry wants
                    # it keyed by the LP token it is the price of.
                    virtual={token: virtual[owner.lower()]
                             for token, owner in issuer.items()
                             if owner.lower() in virtual})
            except Exception:  # noqa: BLE001 - a reader must not fail a quote
                rated = {}
            if rated:
                out.by_pool.update(rated)
                out.rejected = [(a, w) for a, w in out.rejected
                                if a.lower() not in rated]
                if not quiet:
                    print(f"  exact rate pools: {len(rated)} with wrapper rates")

    if not quiet:
        print(f"  exact stableswap: {len(out)} of {out.checked} pools reproduce "
              f"their own get_dy to the wei")
    return out
