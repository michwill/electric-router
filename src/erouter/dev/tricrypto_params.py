"""Read what a tricrypto pool needs to be evaluated instead of probed.

Same shape as the two-coin readers and the same rule: build the model, quote
the pool for real, and keep it only if the arithmetic reproduces the answer to
the wei at every point.  Nothing is decided by pool type or by an address list.

`D` is read from storage, so a pool mid-A/gamma-ramp is refused rather than
approximated -- `newton_D` is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import decode, encode_call
from ..core.transport import Call
from ..core.tricrypto import Tricrypto, TricryptoError
from ..core.types import ArcKind, Probe

CHECK_FRACTIONS = (0.001, 0.01, 0.1)
#: Every ordered pair touching all three coins -- a two-coin ladder would leave
#: one coin's precision and price scale unexercised.
CHECK_PAIRS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
UINT64 = 2**64 - 1


@dataclass(slots=True)
class ExactTricrypto:
    by_pool: dict[str, Tricrypto] = field(default_factory=dict)
    checked: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_pool)

    def get(self, pool: str):
        return self.by_pool.get(pool.lower())


def build_exact_tricrypto(pools, client, *, quiet: bool = True) -> ExactTricrypto:
    """Model every three-coin crypto pool that reproduces its own `get_dy`."""
    out = ExactTricrypto()
    wanted = [
        p for p in pools
        if len(p.coins) == 3 and p.balances and all(p.balances)
        and getattr(p, "swap_kind", None) is ArcKind.SWAP_CRYPTO
    ]
    if not wanted:
        return out

    calls: list[Call] = []
    for pool in wanted:
        calls += [
            Call(pool.address, encode_call("D()")),
            Call(pool.address, encode_call("A()")),
            Call(pool.address, encode_call("gamma()")),
            Call(pool.address, encode_call("packed_fee_params()")),
            Call(pool.address, encode_call("future_A_gamma_time()")),
            Call(pool.address, encode_call("last_timestamp()")),
            Call(pool.address, encode_call("price_scale(uint256)", 0)),
            Call(pool.address, encode_call("price_scale(uint256)", 1)),
        ]
    answers = client.raw(calls)

    # `precisions()` returns three words, which the quoter's one-word `Res`
    # cannot carry.  The transport returns whole returndata.
    transport = getattr(client, "transport", None)
    precisions: dict[str, tuple[int, int, int]] = {}
    if transport is not None and hasattr(transport, "call_many"):
        wire = [Call(p.address, encode_call("precisions()")) for p in wanted]
        for pool, answer in zip(wanted, transport.call_many(wire), strict=True):
            if answer.ok and len(answer.data) >= 96:
                try:
                    got = decode(["uint256[3]"], answer.data)[0]
                except Exception:  # noqa: BLE001
                    continue
                precisions[pool.address.lower()] = tuple(int(v) for v in got)
    for pool in wanted:
        precisions.setdefault(
            pool.address.lower(),
            tuple(10 ** (18 - coin.decimals) for coin in pool.coins[:3]),
        )

    built: list[tuple[object, Tricrypto]] = []
    for k, pool in enumerate(wanted):
        d, amp, gamma, packed, ramp_until, last, ps0, ps1 = answers[8 * k : 8 * k + 8]
        if not (d.ok and amp.ok and packed.ok and ps0.ok and ps1.ok):
            continue
        if ramp_until.ok and last.ok and ramp_until.uint() > last.uint():
            out.rejected.append((pool.address, "A/gamma ramp in progress"))
            continue
        blob = packed.uint()
        built.append((pool, Tricrypto(
            balances=tuple(int(b) for b in pool.balances[:3]),
            precisions=precisions[pool.address.lower()],
            price_scale=(ps0.uint(), ps1.uint()),
            d=d.uint(),
            amp=amp.uint(),
            gamma=gamma.uint_or(0) or 0,
            mid_fee=(blob >> 128) & UINT64,
            out_fee=(blob >> 64) & UINT64,
            fee_gamma=blob & UINT64,
        )))

    if not built:
        return out

    probes, at = [], []
    for pool, _ in built:
        for i, j in CHECK_PAIRS:
            for frac in CHECK_FRACTIONS:
                probes.append(Probe(pool.address, ArcKind.SWAP_CRYPTO, i, j, 3,
                                    max(1, int(pool.balances[i] * frac))))
                at.append(pool.address.lower())
    quotes = client.probe(probes)
    truth: dict[str, list] = {}
    for address, probe, quote in zip(at, probes, quotes, strict=True):
        truth.setdefault(address, []).append((probe, quote))

    for pool, model in built:
        out.checked += 1
        key = pool.address.lower()
        points = [(pr, q) for pr, q in truth.get(key, []) if q.ok and q.value > 0]
        if not points:
            out.rejected.append((pool.address, "pool would not quote the check"))
            continue
        failed = ""
        for probe, quote in points:
            try:
                mine = model.get_dy(probe.i, probe.j, probe.dx)
            except (TricryptoError, ZeroDivisionError) as exc:
                failed = str(exc)[:40]
                break
            if mine != quote.value:
                failed = f"{mine} != {quote.value} at {probe.dx}"
                break
        if failed:
            out.rejected.append((pool.address, failed))
        else:
            out.by_pool[key] = model

    if not quiet:
        print(f"  exact tricrypto: {len(out)} of {out.checked} pools reproduce "
              f"their own get_dy to the wei")
    return out
