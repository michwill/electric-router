"""How often a pool's own minimum-out would trip before the route lands.

Each leg goes out with a minimum-out at a fraction of that pool's fee, which
is where sandwiching stops paying.  The same number is the revert threshold:
move further than that between quoting and inclusion and the leg fails.  What
this module measures is `p` per arc -- the chance of that -- so `core/risk.py`
can price a route as `output * product(1 - p_i) - gas`.

The horizon is two minutes: a user confirming in a wallet plus propagation, not
the multi-hour spans `drift` samples.  So the rate series is resampled at
five-block steps and the windows scored are two steps wide.

**Counting breaches directly does not work at this sample size.**  Two dozen
windows cannot separate "never" from "rarely": zero events would claim a pool
provably never moves, and the honest smoothing that avoids that claim --
Jeffreys' `(k + 1/2) / (n + 1)` -- puts an unbreached pool at 2%, which over a
twelve-leg route reads as a 22% loss.  Both answers are artefacts of counting
rare events with a short ruler, and both dwarf the routing gains being judged.

So the process is modelled instead, and the model is the one the data shows: a
rate does not diffuse, it **jumps when someone trades**.  Most minutes a pool
sees nothing and its rate is unchanged to the last integer; the risk is that a
window catches a trade large enough to move it past the bound.  That factors:

    p = P(the window contains a move) * P(a move is bigger than the bound)

The first is counted per arc -- it is common enough to count.  The second is a
shape, estimated from every *nonzero* move in the universe standardised by its
own arc's typical move, which is thousands of observations rather than two
dozen, and continued past the edge of the sample as a power law (Hill).  Crypto
moves are fat-tailed and a normal tail would understate a wide bound by orders
of magnitude.

Standardising only the nonzero moves is what makes the pooling legitimate, and
an earlier version that skipped it was badly wrong: with the median move of a
quiet pool at zero, its scale fell to the floor, its occasional trade became a
four-hundred-sigma event, and the tail fitted to *that* priced 3pool -- which
never breached in the sample -- at 4.5% a leg.

Per arc rather than per pool.  TriCRV's CRV/ETH rate and its crvUSD/ETH rate do
not move alike, the minimum-out is per leg, and taking a pool's worst arc would
charge every route the worst of them.
"""

from __future__ import annotations

import math

from ..core.codec import encode_call
from ..core.transport import Call

#: Five-block steps, twenty-five of them: roughly a minute apart over half an
#: hour.  Ordered so consecutive samples are adjacent in time.
STEP_BLOCKS = 5
SAMPLES = 25
FINE_BLOCKS = tuple(k * STEP_BLOCKS for k in range(SAMPLES))
#: Window width in steps: two minutes, the far end of a confirm-and-broadcast.
HORIZON = 2
#: The minimum-out, as a fraction of the pool's own fee.
BOUND_OF_FEE = 0.2
#: Below this a rate is not moving as far as our own quotes can resolve.
SCALE_FLOOR_BP = 0.005
#: Never claim an arc is safer than this: the tail is extrapolated past the
#: data, and the extrapolation is not to be trusted to four figures.
RISK_FLOOR = 1e-5
RISK_CEILING = 0.95


def read_fees(client, pools) -> dict[str, float]:
    """Each pool's fee as a fraction, from the pool itself.

    Curve reports fees in units of 1e10.  A pool that does not answer is left
    out rather than defaulted -- without its own fee there is no bound to
    measure against, and inventing one would be a claim.
    """
    targets = [p.address for p in pools]
    calls = [Call(to=address, data=encode_call("fee()")) for address in targets]
    fees: dict[str, float] = {}
    for address, answer in zip(targets, client.raw(calls), strict=True):
        value = answer.uint_or(None) if answer.ok else None
        if value and 0 < value < 10**10:
            fees[address.lower()] = value / 1e10
    return fees


def moves_bp(rates: list[float], horizon: int = HORIZON) -> list[float]:
    """Absolute log moves over `horizon` samples, in basis points."""
    clean = [r for r in rates if r > 0]
    return [abs(math.log(clean[k] / clean[k + horizon])) * 1e4
            for k in range(len(clean) - horizon)]


def jump_scale_bp(moves: list[float]) -> float:
    """The size of a typical *move*, ignoring the windows with none.

    The median over all windows would be zero for most pools, which says only
    that most minutes are quiet.  What the tail shape needs is the size of a
    move given that one happened.
    """
    jumps = sorted(m for m in moves if m > 0)
    if not jumps:
        return SCALE_FLOOR_BP
    return max(jumps[len(jumps) // 2], SCALE_FLOOR_BP)


def tail_model(standardised: list[float]):
    """P(Z > z) for a move's size in units of its arc's typical move.

    Empirical inside the sample, a power law past its edge.  The exponent comes
    from Hill's estimator on the top decile -- the mean log excess over the
    threshold is 1/alpha -- clamped, since below 1.5 the tail is heavier than
    any market (usually too few samples) and above 8 it is thinner than one,
    which would invent safety.
    """
    ordered = sorted((z for z in standardised if z > 0), reverse=True)
    n = len(ordered)
    if n < 40:  # not enough to say anything about a shape
        return lambda z: 1.0 if z <= 0 else min(1.0, 1.0 / (1.0 + z) ** 3)
    k = max(8, n // 10)
    threshold = ordered[k - 1]
    excess = [math.log(z / threshold) for z in ordered[:k] if z > threshold]
    mean_excess = sum(excess) / len(excess) if excess else 0.0
    alpha = min(max(1.0 / mean_excess if mean_excess > 0 else 4.0, 1.5), 8.0)
    p_threshold = k / n

    def tail(z: float) -> float:
        if z <= 0:
            return 1.0
        if z <= threshold:
            return sum(1 for x in ordered if x > z) / n
        return p_threshold * (z / threshold) ** -alpha

    return tail


def breach_risk(series, fees: dict[str, float], arcs=None, *,
                horizon: int = HORIZON) -> dict[str, dict]:
    """Per-arc risk, from rate series and pool fees.

    `series` is what `drift.sample_rates` returns -- one `PriceSeries` per arc,
    carrying the pool it was read from.  `arcs` is the list handed to
    `sample_rates`, whose entries are `(key, pool, kind, i, j, n_coins, dx)`;
    it supplies the coin indices, so the answer is keyed by direction the way
    the minimum-out is.  Without it the answer falls back to pool-level keys.

    Returns `{"address:i>j": {"p", "bound_bp", "scale_bp", "active", "n"}}`, so
    the committed file says why and not merely how much.
    """
    index = {entry[0]: (entry[1], entry[3], entry[4]) for entry in (arcs or [])}

    measured: list[tuple[str, str, float, list[float]]] = []
    pooled: list[float] = []
    for key, entry in series.items():
        pool = entry.pool.lower()
        if pool not in fees:
            continue
        moves = moves_bp(list(entry.prices), horizon)
        if len(moves) < 4:
            continue
        target, i, j = index.get(key, (pool, -1, -1))
        scale = jump_scale_bp(moves)
        measured.append((f"{target.lower()}:{int(i)}>{int(j)}", pool, scale, moves))
        pooled.extend(m / scale for m in moves if m > 0)

    tail = tail_model(pooled)
    out: dict[str, dict] = {}
    for key, pool, scale, moves in measured:
        bound_bp = fees[pool] * BOUND_OF_FEE * 1e4
        jumps = sum(1 for m in moves if m > 0)
        # How often a window contains a move at all.  Half an event stands in
        # for none, so an arc that saw no trade in half an hour is priced low
        # rather than at zero -- it is quiet, not frozen.
        active = max(jumps, 0.5) / len(moves)
        modelled = active * tail(bound_bp / scale)
        # A move that was actually seen past the bound outranks any model of
        # how likely one was.
        seen = sum(1 for m in moves if m > bound_bp) / len(moves)
        risk = min(max(max(seen, modelled), RISK_FLOOR), RISK_CEILING)
        out[key] = {"p": round(risk, 6), "bound_bp": round(bound_bp, 4),
                    "scale_bp": round(scale, 4), "active": round(active, 4),
                    "n": len(moves)}
    return out
