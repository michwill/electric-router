"""Twocrypto-ng, evaluated exactly (§11.3).

The same factory deploys two different pools, and they are not distinguishable
by pool type, name or coins -- only by the math contract each one holds as an
immutable:

* **Twocrypto** proper, the cryptoswap invariant with `gamma`;
* **FX Swap**, which is a *stableswap* invariant wearing all of cryptoswap's
  machinery -- price scale, dynamic fee, EMA oracle, rebalancing.

Everything except `get_y` is shared, which is what makes this module worth
having before either backend exists.  `TwocryptoView.get_dy` is:

    xp[i] += dx
    xp = [xp[0] * p[0], xp[1] * price_scale * p[1] / PRECISION]
    y  = MATH.get_y(A, gamma, xp, D, j)[0]
    dy = xp[j] - y - 1  ;  xp[j] = y
    if j > 0: dy = dy * PRECISION / price_scale
    dy /= p[j]
    dy -= fee(xp) * dy / FEE_PRECISION

Note the fee is charged on the *post-trade* `xp` -- after `xp[j] = y` -- which
is the sort of thing that is invisible in a formula and decides the last basis
point.

Two things this deliberately does not model.  `D` is read from storage, and a
pool mid-A/gamma-ramp recomputes it with `newton_D` instead: `is_ramping` says
when, and such a pool has to be probed until `newton_D` exists here.  And a
pool carrying a `POLICY` contract has its fee -- and its price scale -- driven
by arbitrary external code, which no parameter set can stand in for.  Both are
refused rather than approximated; the caller keeps probing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stableswap import StableSwapError, solve_y

PRECISION = 10**18
FEE_PRECISION = 10**10
N_COINS = 2
#: `A_MULTIPLIER` in twocrypto-ng, where stableswap-ng calls it `A_PRECISION`
#: and sets it to 100.
A_MULTIPLIER = 10_000
MIN_FEE = FEE_PRECISION // 10 // 10_000  # 0.1 bp
MAX_FEE = FEE_PRECISION


class TwocryptoError(ArithmeticError):
    """The pool cannot be evaluated from its parameters."""


@dataclass(frozen=True, slots=True)
class Twocrypto:
    """One twocrypto-ng pool, enough to reproduce `get_dy` exactly.

    `stable` selects the FX Swap backend -- the caller decides it by reading
    the pool's `MATH()` and recognising the implementation, never by guessing
    from the pool's name or its coins.
    """

    balances: tuple[int, int]
    precisions: tuple[int, int]
    price_scale: int
    d: int
    amp: int  # `A`, already carrying A_MULTIPLIER * N**N
    gamma: int
    mid_fee: int
    out_fee: int
    fee_gamma: int
    #: True for an FX Swap (stableswap invariant), False for cryptoswap.
    stable: bool = True

    # ------------------------------------------------------------------ fee

    def fee(self, xp: list[int]) -> int:
        """`Twocrypto._fee`, on the balances *after* the trade.

        `B` is a balance indicator: 1e18 at perfect balance, falling toward 0
        as the pool skews, so the fee slides from `mid_fee` to `out_fee`.
        `fee_gamma` regulates how fast.
        """
        total = xp[0] + xp[1]
        if total <= 0:
            raise TwocryptoError("empty pool")
        b = PRECISION * N_COINS**N_COINS * xp[0] // total * xp[1] // total
        b = self.fee_gamma * b // (self.fee_gamma * b // PRECISION + PRECISION - b)
        fee = (self.mid_fee * b + self.out_fee * (PRECISION - b)) // PRECISION
        return min(MAX_FEE, max(MIN_FEE, fee))

    # ---------------------------------------------------------------- quote

    def get_dy(self, i: int, j: int, dx: int) -> int:
        """Exactly what the pool's `get_dy(i, j, dx)` returns on chain."""
        if i == j or not (0 <= i < N_COINS) or not (0 <= j < N_COINS):
            raise TwocryptoError("coin index out of range")
        if dx <= 0:
            return 0
        if not all(self.balances) or self.d <= 0:
            raise TwocryptoError("empty pool")

        xp = [self.balances[0], self.balances[1]]
        xp[i] += dx
        xp = [
            xp[0] * self.precisions[0],
            xp[1] * self.price_scale * self.precisions[1] // PRECISION,
        ]

        y = self._y(xp, j)
        if y >= xp[j]:
            raise TwocryptoError("unsafe value for y")
        dy = xp[j] - y - 1
        xp[j] = y
        if j > 0:
            dy = dy * PRECISION // self.price_scale
        dy //= self.precisions[j]

        # The fee reads the post-trade balances, which is why `xp[j]` is
        # assigned above rather than after this line.
        return dy - self.fee(xp) * dy // FEE_PRECISION

    def _y(self, xp: list[int], j: int) -> int:
        if not self.stable:
            raise TwocryptoError("cryptoswap get_y is not implemented")
        # `StableswapMath.get_y`: the stableswap iteration at A_MULTIPLIER.
        # `i` and `j` are the other way round from `solve_y`'s signature --
        # there, `i` is the coin whose balance is known.
        other = 1 - j
        try:
            return solve_y(self.amp, A_MULTIPLIER, xp, self.d,
                           other, j, xp[other])
        except StableSwapError as exc:
            raise TwocryptoError(str(exc)) from exc
