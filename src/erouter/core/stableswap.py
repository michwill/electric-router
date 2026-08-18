"""Stableswap, evaluated exactly from its own parameters (§11.3).

The quadratic in `calibrate.py` is what makes routing a convex program, and it
is the right model for choosing *which* pools and *what split*: both quantities
are first-order-flat at the optimum, so an `O(theta^2)` error in the curve does
not move them.  It is the wrong model for asking what a pool actually pays at a
size approaching its own reserves, and past that point no choice of sample
point rescues it -- a secant fitted at the trade size still describes the chord
of a curve that is not a parabola.

Measured on mainnet crvUSD -> sDOLA at $2M, block 25,770,648: the candidate
family's own cheapest path returned 5,800 sDOLA against an expected 1,419,000 --
0.3% of the input, verified on chain, with `max theta` at 0.816.  The refine
pass had already re-fitted 258 arcs at trade-sized steps and the §12.1 size
check another 263.  Sampling closer did not help, because the object being
sampled is not a parabola at 80% of a reserve.

So: read `A`, the fee, the balances and the rates, and evaluate the pool's own
invariant.  `f(delta)` is then exact at any size, which is what candidate
scoring needs.  The relaxation keeps the quadratic.

This is the third way of knowing a curve here, and they answer different
questions.  `calibrate.py` fits two derivatives, which is what the convex
program needs.  `curves.py` interpolates `x/f(x)` through probes, which is
exact enough for a realised route's split and works for *any* pool, including
ones whose parameters we cannot read.  This is neither: it is the pool's own
arithmetic, so it costs no probes at all once the parameters are in hand and it
is exact where the other two are approximations.  It only covers stableswap;
anything else keeps sampling.

Two dialects, and the difference is not cosmetic:

* **legacy** (3pool and everything of that era) stores `A` unscaled, and takes
  the fee *after* converting the output back to token units.
* **ng** (`CurveStableSwapNG`) stores `A * A_PRECISION`, takes the fee in `xp`
  units before converting, and scales that fee with how far off peg the two
  balances are -- the dynamic fee.  A pool at peg charges `fee`; the same pool
  badly imbalanced charges several times that, which is exactly the regime a
  large trade puts it in, so ignoring it understates the cost of the trades
  that need it most.

Integer arithmetic throughout, floor division everywhere the contracts use it,
because the point of this module is to agree with them to the wei.
"""

from __future__ import annotations

from dataclasses import dataclass

PRECISION = 10**18
FEE_DENOMINATOR = 10**10
A_PRECISION = 100
#: Newton's method in the contracts is capped at 255 and asserted to converge.
MAX_ITER = 255


class StableSwapError(ArithmeticError):
    """The invariant did not converge, or the pool cannot serve the trade."""


def solve_y(amp: int, a_precision: int, xp: list[int], d: int,
            i: int, j: int, x: int) -> int:
    """The `j` balance restoring the invariant when `i` holds `x`.

    Module level because two different pool families run exactly this
    iteration.  `CurveStableSwapNG` is the obvious one; the other is
    twocrypto-ng's `StableswapMath`, which the same factory deploys as an "FX
    Swap" -- cryptoswap machinery (price scale, dynamic fee, EMA oracle) around
    a stableswap invariant.  Its only difference is `A_MULTIPLIER = 10000`
    where stableswap-ng uses `A_PRECISION = 100`, which is what `a_precision`
    already parameterises, so the two share this rather than a copy of it.
    """
    n = len(xp)
    ann = amp * n
    c = d
    s = 0
    for k in range(n):
        if k == i:
            below = x
        elif k != j:
            below = xp[k]
        else:
            continue
        if below <= 0:
            raise StableSwapError("empty balance")
        s += below
        c = c * d // (below * n)
    c = c * d * a_precision // (ann * n)
    b = s + d * a_precision // ann
    y = d
    for _ in range(MAX_ITER):
        prev = y
        y = (y * y + c) // (2 * y + b - d)
        if abs(y - prev) <= 1:
            return y
    raise StableSwapError("y did not converge")


@dataclass(frozen=True, slots=True)
class StableSwap:
    """One pool's state, enough to evaluate `get_dy` exactly.

    `rates` are the 1e18-scaled multipliers that take a raw balance into the
    common-precision `xp` space: `10**(36 - decimals)` for a legacy pool's
    `RATES`, `stored_rates()` for ng (which folds in an LST's exchange rate as
    well as its decimals).  Passing them explicitly means this module never has
    to know which of the two it is looking at.

    `amp` is whatever `A_precise()`/`A()` returned *together with* the matching
    `a_precision`, so a caller reading either spelling can describe the pool
    without converting and losing a digit.
    """

    balances: tuple[int, ...]
    rates: tuple[int, ...]
    amp: int
    fee: int
    #: 0 or FEE_DENOMINATOR both mean "no dynamic fee"; ng pools carry a real one.
    offpeg_fee_multiplier: int = 0
    a_precision: int = A_PRECISION
    #: ng takes the fee in `xp` space; legacy takes it in token space.
    fee_on_xp: bool = True

    @property
    def n(self) -> int:
        return len(self.balances)

    def xp(self) -> list[int]:
        return [b * r // PRECISION for b, r in zip(self.balances, self.rates, strict=True)]

    # ------------------------------------------------------------ invariant

    def d(self, xp: list[int] | None = None) -> int:
        """`D`, by the contracts' own Newton iteration.

        Written as the contracts write it, including the order of operations:
        `D_P = D_P * D // (x * n)` accumulates a different rounding than the
        algebraically equal `D**(n+1) // (n**n * prod(x))`, and the difference
        shows up in the last wei of a quote.
        """
        xp = self.xp() if xp is None else xp
        s = sum(xp)
        if s == 0:
            return 0
        n = self.n
        ann = self.amp * n
        d = s
        for _ in range(MAX_ITER):
            d_p = d
            for x in xp:
                if x == 0:
                    raise StableSwapError("empty balance")
                d_p = d_p * d // (x * n)
            prev = d
            d = ((ann * s // self.a_precision + d_p * n) * d
                 // ((ann - self.a_precision) * d // self.a_precision + (n + 1) * d_p))
            if abs(d - prev) <= 1:
                return d
        raise StableSwapError("D did not converge")

    def y(self, i: int, j: int, x: int, xp: list[int] | None = None,
          d: int | None = None) -> int:
        """The `j` balance that restores the invariant when `i` holds `x`."""
        if i == j:
            raise StableSwapError("i and j must differ")
        xp = self.xp() if xp is None else xp
        return solve_y(self.amp, self.a_precision, xp,
                       self.d(xp) if d is None else d, i, j, x)

    # ----------------------------------------------------------------- fees

    def dynamic_fee(self, xpi: int, xpj: int) -> int:
        """The fee this pool charges at that imbalance (ng only).

        `4 xi xj / (xi + xj)^2` is 1 at peg and falls toward 0 as the two sides
        diverge, so the fee rises from `fee` toward `offpeg_fee_multiplier`
        times it.  A pool the trade has pushed off peg charges more for the
        privilege, which is the term that matters exactly when the trade is
        large enough to do the pushing.
        """
        multiplier = self.offpeg_fee_multiplier
        if multiplier <= FEE_DENOMINATOR:
            return self.fee
        xps2 = (xpi + xpj) ** 2
        return ((multiplier * self.fee)
                // ((multiplier - FEE_DENOMINATOR) * 4 * xpi * xpj // xps2
                    + FEE_DENOMINATOR))

    # ----------------------------------------------------------------- quote

    def get_dy(self, i: int, j: int, dx: int) -> int:
        """Exactly what `get_dy(i, j, dx)` returns on chain."""
        if dx <= 0:
            return 0
        xp = self.xp()
        d = self.d(xp)
        x = xp[i] + dx * self.rates[i] // PRECISION
        y = self.y(i, j, x, xp, d)
        raw = xp[j] - y - 1
        if raw <= 0:
            return 0
        if self.fee_on_xp:
            fee = self.dynamic_fee((xp[i] + x) // 2, (xp[j] + y) // 2)
            return (raw - raw * fee // FEE_DENOMINATOR) * PRECISION // self.rates[j]
        out = raw * PRECISION // self.rates[j]
        return out - out * self.fee // FEE_DENOMINATOR


# --------------------------------------------------------------- LP arcs

def solve_y_d(amp: int, a_precision: int, xp: list[int], d: int, i: int,
              n: int) -> int:
    """`get_y_D`: balance `i` when `D` is reduced to `d`, the others held.

    A different question from `solve_y`, which asks what `i` becomes when
    another balance changes at constant `D`.  Here `D` itself moves -- which is
    what a single-sided deposit or withdrawal does -- and the same quadratic is
    iterated with `c` and `b` built from the *target* `D`.

    Ported from the deployed 3pool
    (`0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7`), generalised over
    `A_PRECISION`: the pools that predate it have `a_precision == 1`, which
    makes every scaling term below vanish exactly as in the original.
    """
    ann = amp * n
    c = d
    s = 0
    for k in range(n):
        if k == i:
            continue
        x = xp[k]
        if x <= 0:
            raise StableSwapError("empty balance")
        s += x
        c = c * d // (x * n)
    c = c * d * a_precision // (ann * n)
    b = s + d * a_precision // ann
    y = d
    for _ in range(255):
        prev = y
        y = (y * y + c) // (2 * y + b - d)
        if (y - prev if y > prev else prev - y) <= 1:
            return y
    raise StableSwapError("y_D did not converge")


@dataclass(frozen=True, slots=True)
class StableSwapLP:
    """A pool's LP arcs: what a deposit mints and a withdrawal returns.

    The invariant is the same one `StableSwap` already reproduces; what is new
    is that `D` moves.  Two conventions the deployed source settles, and both
    are easy to assume wrongly:

    * `calc_token_amount` takes **no fee at all** on the legacy pools -- its own
      docstring calls it "needed to prevent front-running, not for precise
      calculations".  It is `(D1 - D0) * totalSupply / D0` and nothing else.
    * `calc_withdraw_one_coin` charges `fee * N / (4 * (N - 1))` on each
      coin's *imbalance* against the ideal, not on the output, and then
      withdraws one wei less "to account for rounding errors".
    """

    pool: StableSwap
    total_supply: int

    @property
    def n(self) -> int:
        return self.pool.n

    def _xp(self) -> list[int]:
        return self.pool.xp()

    def calc_token_amount(self, amounts: list[int], deposit: bool) -> int:
        """LP minted for a deposit, or burned for a withdrawal.  Fee-free."""
        if self.total_supply <= 0:
            raise StableSwapError("no supply")
        p = self.pool
        d0 = p.d()
        moved = list(p.balances)
        for k, amount in enumerate(amounts):
            if deposit:
                moved[k] += amount
            else:
                if amount > moved[k]:
                    raise StableSwapError("withdrawing more than the pool holds")
                moved[k] -= amount
        after = StableSwap(balances=tuple(moved), rates=p.rates, amp=p.amp,
                           fee=p.fee, offpeg_fee_multiplier=p.offpeg_fee_multiplier,
                           a_precision=p.a_precision, fee_on_xp=p.fee_on_xp)
        d1 = after.d()
        diff = d1 - d0 if deposit else d0 - d1
        return diff * self.total_supply // d0

    def calc_withdraw_one_coin(self, token_amount: int, i: int) -> int:
        """Coin `i` returned for burning `token_amount` of LP."""
        if self.total_supply <= 0:
            raise StableSwapError("no supply")
        if not (0 <= i < self.n):
            raise StableSwapError("coin index out of range")
        p = self.pool
        n = self.n
        fee = p.fee * n // (4 * (n - 1))
        xp = self._xp()
        d0 = p.d()
        d1 = d0 - token_amount * d0 // self.total_supply
        new_y = solve_y_d(p.amp, p.a_precision, xp, d1, i, n)

        reduced = list(xp)
        for j in range(n):
            if j == i:
                expected = xp[j] * d1 // d0 - new_y
            else:
                expected = xp[j] - xp[j] * d1 // d0
            reduced[j] -= fee * expected // FEE_DENOMINATOR
        dy = reduced[i] - solve_y_d(p.amp, p.a_precision, reduced, d1, i, n)
        # One wei less, as the pool does, and back out of `xp` space.
        return (dy - 1) * PRECISION // p.rates[i]
