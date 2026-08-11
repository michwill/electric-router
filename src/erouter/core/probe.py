"""Probe planning: turn arcs into a batch of quote requests (spec §2.3).

A *geometric grid* rather than the bare 4-node ladder, because it costs the
same one round trip and does strictly more:

* `a` can be taken from the smallest node that actually answered.  Measured on
  Ethereum, 6% of arcs fail at `1e-6 * reserve` -- 41 revert, 6 return empty,
  and 7 return **zero**, which is the dangerous one because a zero `a` looks
  valid and then poisons `log a` in the reference-price fit.
* second divided differences fall out on *every* consecutive triple, a stronger
  non-concavity detector than one ladder.
* the stableswap peg boundary shows up as a jump in local curvature, so §2.5's
  arc split becomes a measurement rather than a hope that `d_bar` happened to
  land inside the flat region.

Token decimals in the live universe are `{18: 680, 6: 158, 8: 49, 2: 5, 9: 2}`
-- two-decimal tokens exist, so deltas are floored and de-duplicated in integer
space rather than assuming 1e-6 of a reserve is a meaningful amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import ArcKind, Probe, ProbeLadder

GRID: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)


@dataclass(frozen=True, slots=True)
class ArcRef:
    """Everything needed to probe one direction of one pool."""

    pool: str
    kind: ArcKind
    i: int
    j: int
    n_coins: int
    reserve_in: int
    decimals_in: int = 18
    decimals_out: int = 18

    @property
    def id(self) -> str:
        return f"{self.pool.lower()}:{int(self.kind)}:{self.i}>{self.j}"


def plan_deltas(
    reserve_in: int, decimals_in: int = 18, grid: tuple[float, ...] = GRID
) -> list[int]:
    """Integer probe sizes, floored and strictly increasing.

    A node that would round onto its predecessor is dropped rather than sent:
    two identical deltas produce a zero denominator in the divided differences.
    """
    floor = max(1, 10 ** max(0, decimals_in - 6))
    out: list[int] = []
    for fraction in grid:
        delta = max(int(reserve_in * fraction), floor)
        if out and delta <= out[-1]:
            continue
        out.append(delta)
    return out


@dataclass(slots=True)
class ProbePlan:
    probes: list[Probe] = field(default_factory=list)
    arcs: list[ArcRef] = field(default_factory=list)
    deltas: list[list[int]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.probes)


def plan_grid(arcs: list[ArcRef], grid: tuple[float, ...] = GRID) -> ProbePlan:
    """One batch covering every arc's whole ladder."""
    plan = ProbePlan()
    for arc in arcs:
        deltas = plan_deltas(arc.reserve_in, arc.decimals_in, grid)
        if len(deltas) < 3:  # too little room to fit a curvature
            continue
        start = len(plan.probes)
        for delta in deltas:
            plan.probes.append(Probe(arc.pool, arc.kind, arc.i, arc.j, arc.n_coins, delta))
        plan.arcs.append(arc)
        plan.deltas.append(deltas)
        plan.spans.append((start, len(plan.probes)))
    return plan


@dataclass(slots=True)
class Ladder:
    """One arc's successful probes, ready for `calibrate`."""

    arc: ArcRef
    deltas: list[int] = field(default_factory=list)
    quotes: list[int] = field(default_factory=list)
    attempted: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.deltas) >= 3

    @property
    def coarse_tangent(self) -> bool:
        """True when the smallest answering probe was not small.

        `a` is then a chord over a wide interval rather than a tangent, which
        under-states the marginal rate -- the conservative direction, but worth
        recording so §12.1 can see it.
        """
        if not self.deltas or self.arc.reserve_in <= 0:
            return False
        return self.deltas[0] > 1e-3 * self.arc.reserve_in

    def as_float(self) -> tuple[list[float], list[float]]:
        """Human units, which is the frame `a` and `B` are calibrated in."""
        scale_in = 10.0**self.arc.decimals_in
        scale_out = 10.0**self.arc.decimals_out
        return (
            [d / scale_in for d in self.deltas],
            [q / scale_out for q in self.quotes],
        )

    def provenance(self, block: int) -> ProbeLadder:
        return ProbeLadder(
            deltas=tuple(self.deltas),
            quotes=tuple(self.quotes),
            reserve_in=self.arc.reserve_in,
            decimals_in=self.arc.decimals_in,
            decimals_out=self.arc.decimals_out,
            block=block,
        )


def collect(plan: ProbePlan, results) -> list[Ladder]:
    """Group quote results back into per-arc ladders, dropping failures.

    A failed probe is dropped, never recorded as a zero: `a = 0` would look
    like a valid quote and NaN the reference-price fit.
    """
    ladders: list[Ladder] = []
    for arc, deltas, (lo, hi) in zip(plan.arcs, plan.deltas, plan.spans, strict=True):
        ladder = Ladder(arc=arc, attempted=hi - lo)
        for delta, result in zip(deltas, results[lo:hi], strict=True):
            status = getattr(result, "status", None)
            value = int(getattr(result, "value", 0) or 0)
            if status is not None and status.name != "VALUE":
                ladder.failures[status.name] = ladder.failures.get(status.name, 0) + 1
                continue
            if value <= 0:
                ladder.failures["ZERO"] = ladder.failures.get("ZERO", 0) + 1
                continue
            ladder.deltas.append(delta)
            ladder.quotes.append(value)
        ladders.append(ladder)
    return ladders
