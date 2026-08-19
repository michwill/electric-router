"""One pool touched once, with several ports.

A pool with `N` coins can serve more than two ports in a single interaction:

    XDAI -> (3pool) -> 3Crv + USDC.e
    DAI  -> (3pool) -> USDC + USDT

Modelled as separate arcs those are two entries into one pool, which is two
`psi^2/2G` terms with no cross-term and two calibrations against a state only
the first of them will see.  Modelled as one **element** the pool appears
once, so there is no stale second calibration and nothing to cross-couple --
decision 3 is satisfied by construction rather than enforced.

This module is the representation and the arithmetic.  It does not touch the
solve; see `docs/multi-port-elements.md` for why that is a separate step.

The port bound is the structure's, not a cap:

    #coin-ports in + #coin-ports out <= N

because each port occupies a distinct coin.  One coin cannot be both an
input and an output -- that is a wash -- and two input ports on one coin are
just one larger port, so ports map injectively onto coins.  The **LP token
is not one of the `N`** and so consumes no slot: counting it would reject
`add_liquidity` of both coins of a 2-coin pool, which is a real operation.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A port on the LP token rather than on one of the pool's coins.
LP = -1


class MultiPortError(Exception):
    pass


#: Shares are integers, in basis points, and the last port on a side takes
#: the remainder.  Not a style choice: `int(amount * 0.25)` on a wei-scale
#: integer silently loses the low digits -- float64 carries ~15 of them and a
#: 400,000-token deposit needs 24 -- so a float split disagrees with the
#: integer one it is meant to stand for.  This is also the arithmetic
#: `Leg.bps` already executes, so an element splits the way the router does.
BPS = 10_000


@dataclass(frozen=True, slots=True)
class Port:
    """One side of an element: which token, and its share of that side."""

    coin: int          # a coin index, or `LP`
    bps: int           # share of this side's total, in basis points


@dataclass(frozen=True, slots=True)
class MultiPort:
    """`inputs -> outputs` through one pool, priced on advancing state."""

    pool: str
    n_coins: int
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise MultiPortError("an element needs a port on each side")
        coins_in = [p.coin for p in self.inputs if p.coin != LP]
        coins_out = [p.coin for p in self.outputs if p.coin != LP]
        seen = coins_in + coins_out
        for coin in seen:
            if not (0 <= coin < self.n_coins):
                raise MultiPortError(f"coin {coin} out of range")
        if len(set(seen)) != len(seen):
            # The injection that makes the bound below the right one: a coin
            # on both sides is a wash, and twice on one side is one port.
            raise MultiPortError("a coin may hold at most one port")
        if len(seen) > self.n_coins:
            raise MultiPortError(
                f"{len(seen)} coin-ports on a {self.n_coins}-coin pool")
        if sum(p.coin == LP for p in self.inputs) > 1 or \
                sum(p.coin == LP for p in self.outputs) > 1:
            raise MultiPortError("one LP port per side")
        for side, name in ((self.inputs, "inputs"), (self.outputs, "outputs")):
            if any(not (0 < p.bps <= BPS) for p in side):
                raise MultiPortError(f"{name}: shares must be in (0, {BPS}]")
            if sum(p.bps for p in side) != BPS:
                raise MultiPortError(f"{name}: shares must sum to {BPS}")

    @property
    def ports(self) -> int:
        return len(self.inputs) + len(self.outputs)


def _split(ports, total: int):
    """`(port, amount)` per port, in integers, last one taking the rest."""
    left = total
    for k, port in enumerate(ports):
        share = left if k == len(ports) - 1 else total * port.bps // BPS
        left -= share
        yield port, share


def evaluate(element: MultiPort, pool, lp, amount_in: int) -> tuple[list[int], object, object]:
    """`(outputs, pool after, lp after)` for one unit of `element`.

    `pool` is a `StableSwap` and `lp` its `StableSwapLP`, or `None` when the
    element has no LP port.  Every leg is priced against the pool as the
    previous leg left it, which is what makes this one element rather than
    several arcs: the coupling *is* the advancing state.

    Two shapes are served today, which are the ones that arise:

    * **one in, many out** -- split the input by the output weights, then one
      `exchange` per coin port and one `add_liquidity` for an LP port;
    * **many in, one out** -- an `add_liquidity` whose amounts vector is the
      input weights, which is a single call and needs no ordering.

    The general `j`-in-`k`-out case needs a pairing rule between the sides and
    is deliberately refused rather than guessed at.
    """
    if amount_in <= 0:
        raise MultiPortError("nothing to route")
    if len(element.inputs) > 1 and len(element.outputs) > 1:
        raise MultiPortError("many-in many-out needs a pairing rule")

    if len(element.inputs) > 1:
        # k in, 1 out.  Only the LP token can absorb several coins at once.
        out = element.outputs[0]
        if out.coin != LP or lp is None:
            raise MultiPortError("several inputs pay only an LP port")
        amounts = [0] * element.n_coins
        for port, share in _split(element.inputs, amount_in):
            if port.coin == LP:
                raise MultiPortError("an LP input cannot join a deposit")
            amounts[port.coin] = share
        minted, after = lp.add_liquidity(amounts)
        return [minted], after.pool, after

    # 1 in, k out.
    source = element.inputs[0]
    outs: list[int] = []
    for port, share in _split(element.outputs, amount_in):
        if share <= 0:
            raise MultiPortError("a port was allocated nothing")
        if port.coin == LP:
            if lp is None or source.coin == LP:
                raise MultiPortError("no LP model for an LP port")
            amounts = [0] * element.n_coins
            amounts[source.coin] = share
            minted, lp = lp.add_liquidity(amounts)
            pool = lp.pool
            outs.append(minted)
        elif source.coin == LP:
            if lp is None:
                raise MultiPortError("no LP model for an LP input")
            outs.append(lp.calc_withdraw_one_coin(share, port.coin))
        else:
            dy, pool = pool.exchange(source.coin, port.coin, share)
            if lp is not None:
                from dataclasses import replace
                lp = replace(lp, pool=pool)
            outs.append(dy)
    return outs, pool, lp
