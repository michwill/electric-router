"""Read what a twocrypto-ng pool needs to be evaluated instead of probed.

The factory deploys two pools that are indistinguishable by type, name or
coins: cryptoswap proper, and the "FX Swap" -- a *stableswap* invariant wearing
cryptoswap's machinery.  Which one a pool is, is decided by the math contract
it holds as an immutable.

**This does not read that address.**  It builds the FX Swap model for every
twocrypto pool and lets the wei-exact check decide: a cryptoswap pool cannot
reproduce its own `get_dy` from a stableswap invariant, so it fails and keeps
being probed.  That is the same rule the stableswap reader uses, and it has the
property an address list does not -- it works on a chain nobody has surveyed,
and it cannot go stale when a new math implementation is deployed.

Two exclusions are made up front rather than left to the check, because the
check cannot see them:

* **a `POLICY` contract.**  `_fee` asks it first, and `Policy.get_fee(xp)`
  takes the balances -- so the fee may vary with trade size.  One probe would
  agree at the size it was taken and disagree elsewhere, which is worse than
  not modelling the pool at all.
* **an A/gamma ramp in progress.**  `D` then comes from `newton_D` rather than
  storage, and `newton_D` is not implemented here yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import decode, encode_call
from ..core.transport import Call
from ..core.twocrypto import Twocrypto, TwocryptoError
from ..core.types import ArcKind, Probe

#: One `get_dy` per pool, at a size big enough to separate the two invariants.
CHECK_FRACTION = 0.01
UINT64 = 2**64 - 1


@dataclass(slots=True)
class ExactTwocrypto:
    by_pool: dict[str, Twocrypto] = field(default_factory=dict)
    checked: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_pool)

    def get(self, pool: str):
        return self.by_pool.get(pool.lower())


def _candidates(pools):
    for pool in pools:
        if len(pool.coins) != 2 or not pool.balances or not all(pool.balances):
            continue
        if getattr(pool, "swap_kind", None) is not ArcKind.SWAP_CRYPTO:
            continue
        yield pool


def build_exact_twocrypto(pools, client, *, quiet: bool = True) -> ExactTwocrypto:
    """Model every twocrypto pool that reproduces its own `get_dy`."""
    out = ExactTwocrypto()
    wanted = list(_candidates(pools))
    if not wanted:
        return out

    calls: list[Call] = []
    for pool in wanted:
        calls += [
            Call(pool.address, encode_call("price_scale()")),
            Call(pool.address, encode_call("D()")),
            Call(pool.address, encode_call("A()")),
            Call(pool.address, encode_call("gamma()")),
            Call(pool.address, encode_call("packed_fee_params()")),
            Call(pool.address, encode_call("POLICY()")),
            Call(pool.address, encode_call("future_A_gamma_time()")),
            Call(pool.address, encode_call("last_timestamp()")),
        ]
    answers = client.raw(calls)

    # `precisions()` returns two words, which the quoter's one-word `Res`
    # cannot carry -- the same truncation that made every rate-bearing
    # stableswap look wrong.  Ask the transport, which returns whole
    # returndata.
    transport = getattr(client, "transport", None)
    precisions: dict[str, tuple[int, int]] = {}
    if transport is not None and hasattr(transport, "call_many"):
        wire = [Call(p.address, encode_call("precisions()")) for p in wanted]
        for pool, answer in zip(wanted, transport.call_many(wire), strict=True):
            if answer.ok and len(answer.data) >= 64:
                try:
                    got = decode(["uint256[2]"], answer.data)[0]
                except Exception:  # noqa: BLE001
                    continue
                precisions[pool.address.lower()] = (int(got[0]), int(got[1]))

    built: list[tuple[object, Twocrypto]] = []
    for k, pool in enumerate(wanted):
        (scale, d, amp, gamma, packed, policy,
         ramp_until, last) = answers[8 * k : 8 * k + 8]
        key = pool.address.lower()
        if key not in precisions:
            continue
        if not (scale.ok and d.ok and amp.ok and packed.ok):
            continue
        if policy.ok and policy.uint():
            out.rejected.append((pool.address, "has a POLICY contract"))
            continue
        if ramp_until.ok and last.ok and ramp_until.uint() > last.uint():
            out.rejected.append((pool.address, "A/gamma ramp in progress"))
            continue
        blob = packed.uint()
        built.append((pool, Twocrypto(
            balances=(int(pool.balances[0]), int(pool.balances[1])),
            precisions=precisions[key],
            price_scale=scale.uint(),
            d=d.uint(),
            amp=amp.uint(),
            gamma=gamma.uint_or(0) or 0,
            mid_fee=(blob >> 128) & UINT64,
            out_fee=(blob >> 64) & UINT64,
            fee_gamma=blob & UINT64,
            stable=True,
        )))

    if not built:
        return out

    probes = [
        Probe(pool.address, ArcKind.SWAP_CRYPTO, 0, 1, 2,
              max(1, int(pool.balances[0] * CHECK_FRACTION)))
        for pool, _ in built
    ]
    quotes = client.probe(probes)

    for (pool, model), probe, quote in zip(built, probes, quotes, strict=True):
        out.checked += 1
        if not quote.ok or quote.value <= 0:
            out.rejected.append((pool.address, "pool would not quote the check"))
            continue
        try:
            mine = model.get_dy(0, 1, probe.dx)
        except (TwocryptoError, ZeroDivisionError) as exc:
            out.rejected.append((pool.address, str(exc)[:40]))
            continue
        if mine != quote.value:
            # Almost always "this is a cryptoswap pool, not an FX Swap", which
            # is the intended way to tell them apart.
            out.rejected.append(
                (pool.address, f"{mine} != {quote.value} at {probe.dx}"))
            continue
        out.by_pool[pool.address.lower()] = model

    if not quiet:
        print(f"  exact twocrypto: {len(out)} of {out.checked} pools reproduce "
              f"their own get_dy to the wei")
    return out
