"""Read what a rate wrapper needs to be evaluated instead of executed.

A wstETH is a vault without the ERC4626 spelling: no curve, no ladder, one
ratio per direction describing the whole range. `wrappers.py` already reads
that ratio -- it needs one to merge wstETH into stETH as a node -- but the
number stops there, so every conversion leg a route carries was being sent to
the EVM to be told what a multiplication already knew. Measured: 24 of 39
routes in a quote, at 481 us each against 197 for one that is walked.

Same rule as the pools and the vaults: build the model, ask the chain for
real, and keep it only if the arithmetic reproduces the answer **to the wei**,
at every size checked and in each direction separately.

Which is why the rate is not simply believed. `getStETHByWstETH` is
`shares * totalPooledEther / totalShares`, and the 1e18-scaled rate the merge
uses is that expression already rounded once -- linear to 1.3e-19, which is
about a wei, and a wei is what the gate is for. So both conventions are
offered and the chain decides: the derived rate, and the pool's own two
totals. A wrapper that reproduces four sizes spanning eight decades is running
the arithmetic we think it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.codec import encode_call
from ..core.transport import Call
from ..core.types import ArcKind
from ..core.vault import Vault

#: Sizes to check at, spread over eight decades, so a rounding convention that
#: only shows at one end is caught. The same spread `vault_params` uses.
CHECK_SIZES = (10**15, 10**18, 5 * 10**20, 10**23)

WRAP = ArcKind.WSTETH_WRAP
UNWRAP = ArcKind.WSTETH_UNWRAP

#: What the chain is asked, per direction.
ASK = {
    UNWRAP: "getStETHByWstETH(uint256)",
    WRAP: "getWstETHByStETH(uint256)",
}


@dataclass(slots=True)
class ExactWrappers:
    """Verified wrapper models, by (address, direction)."""

    by_key: dict[tuple[str, ArcKind], Vault] = field(default_factory=dict)
    checked: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_key)

    def get(self, token: str, kind):
        return self.by_key.get((token.lower(), kind))


def build_exact_wrappers(pairs, client, *, quiet: bool = True) -> ExactWrappers:
    """Model every wrapper direction that reproduces its own conversion call.

    `pairs` is the chain table's `wsteth_pairs`: `(wrapper, canonical)`, where
    the canonical token is the one holding the totals.
    """
    out = ExactWrappers()
    wanted = [(t.lower(), c.lower()) for t, c in pairs]
    if not wanted:
        return out

    # The totals live on the canonical token, not the wrapper: stETH knows how
    # much ether it pools and how many shares stand against it, and wstETH is
    # a thin shell over `getPooledEthByShares`.
    meta = client.raw([c for _t, canonical in wanted for c in (
        Call(canonical, encode_call("getTotalPooledEther()")),
        Call(canonical, encode_call("getTotalShares()")))])

    probes: list[Call] = []
    for token, _canonical in wanted:
        for kind in (UNWRAP, WRAP):
            probes += [Call(token, encode_call(ASK[kind], x)) for x in CHECK_SIZES]
    quotes = client.raw(probes)
    per = 2 * len(CHECK_SIZES)

    for k, (token, _canonical) in enumerate(wanted):
        pooled, shares = meta[2 * k: 2 * k + 2]
        block = quotes[per * k: per * (k + 1)]
        seen = {UNWRAP: block[:len(CHECK_SIZES)],
                WRAP: block[len(CHECK_SIZES):]}

        totals = None
        if pooled.ok and shares.ok and pooled.uint() > 0 and shares.uint() > 0:
            totals = (pooled.uint(), shares.uint())

        for kind in (UNWRAP, WRAP):
            out.checked += 1
            answers = seen[kind]
            if any(not a.ok for a in answers):
                out.rejected.append((token, f"{kind.name}: no answer"))
                continue
            want = [a.uint() for a in answers]
            if all(v == 0 for v in want):
                out.rejected.append((token, f"{kind.name}: answers zero"))
                continue

            # The chain's own ratio first, then the rate it derives from it.
            # If the two agree the first is still the right one to keep: it is
            # the expression the contract evaluates, so it cannot drift when
            # the totals move between blocks.
            unit = 10**18
            candidates = []
            if totals is not None:
                num, den = totals if kind is UNWRAP else totals[::-1]
                candidates.append(("totals", num, den))
            derived = want[CHECK_SIZES.index(unit)] if unit in CHECK_SIZES else 0
            if derived:
                candidates.append(("rate", derived, unit))

            for name, num, den in candidates:
                model = Vault(num=num, den=den)
                if all(model.convert(x) == w for x, w in zip(CHECK_SIZES, want,
                                                             strict=True)):
                    out.by_key[(token, kind)] = model
                    if not quiet:
                        print(f"  {token} {kind.name}: {name}")
                    break
            else:
                gap = "; ".join(
                    f"{x}: want {w}, {name} gives {Vault(num=num, den=den).convert(x)}"
                    for name, num, den in candidates[:1]
                    for x, w in list(zip(CHECK_SIZES, want, strict=True))[:1])
                out.rejected.append((token, f"{kind.name}: no convention fits ({gap})"))
    return out
