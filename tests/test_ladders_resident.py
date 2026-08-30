"""The resident ladders must do what the Python ones do.

`plan_sized`, `collect`, `merge` and the fit have a Python form that is the
reference and a Rust form that holds the ladders instead of rebuilding them.
The second is only allowed to exist while it answers exactly what the first
does, and "exactly" is the right word here: the planning and the merge are
integer bookkeeping, so there is no tolerance to hide in.  Only the fit reads
floats, and both sides hand the same floats to the same compiled `calibrate`.

Synthetic ladders rather than a warmed session: this is the marshalling and
the bookkeeping being checked, and those do not care whose pool it was.
"""

from __future__ import annotations

import math
import random

import pytest

from erouter.core.accel import ladders_from
from erouter.core.calibrate import DRIFT_TOL
from erouter.core.pipeline import _Fitted, _quantum
from erouter.core.probe import Ladder, collect, merge, plan_sized
from erouter.core.types import ArcKind

erouter_solve = pytest.importorskip("erouter_solve")
pytestmark = pytest.mark.skipif(
    not hasattr(erouter_solve, "Ladders"), reason="rebuild ./rust")


class Ref:
    """The parts of an `ArcRef` a ladder reads."""

    def __init__(self, n: int, decimals_in: int, decimals_out: int, reserve: int):
        self.id = f"arc{n}"
        self.pool = f"0x{n:040x}"
        self.kind = ArcKind.SWAP_STABLE
        self.i, self.j, self.n_coins = 0, 1, 2
        self.decimals_in = decimals_in
        self.decimals_out = decimals_out
        self.reserve_in = reserve
        self.reserve_out = reserve


class Answer:
    """One quote result, in the shape `collect` reads."""

    def __init__(self, value, status=None):
        self.value = value
        self.status = status


class Status:
    def __init__(self, name):
        self.name = name


VALUE, REVERTED = Status("VALUE"), Status("REVERTED")


def _ladders(seed: int, n: int = 40):
    """A coarse pass's worth, with the decimals a real universe mixes."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        di = rng.choice((6, 8, 18))
        do = rng.choice((6, 18))
        reserve = rng.randrange(10**4, 10**7) * 10**di
        ref = Ref(k, di, do, reserve)
        lad = Ladder(arc=ref)
        points = sorted({rng.randrange(1, 10**5) * 10 ** (di - 2)
                         for _ in range(rng.randrange(3, 7))})
        for d in points:
            lad.deltas.append(d)
            lad.quotes.append(int(d * rng.uniform(0.9, 1.1) * 10 ** (do - di)) + 1)
        lad.attempted = len(points) + rng.randrange(0, 3)
        out.append(lad)
    return out


def _sizes(ladders, seed: int):
    """What refine asks for: three fractions of the trade, per arc."""
    rng = random.Random(seed + 1)
    out = {}
    for lad in ladders:
        if rng.random() < 0.2:
            continue
        whole = rng.randrange(1, 10**6) * 10 ** (lad.arc.decimals_in - 3)
        out[lad.arc.id] = [int(whole * f) for f in (0.05, 0.10, 0.20)]
    return out


def _answers(probes, seed: int):
    """One answer per probe, with a share of refusals and zeros."""
    rng = random.Random(seed + 2)
    got = []
    for p in probes:
        roll = rng.random()
        if roll < 0.08:
            got.append(Answer(0, REVERTED))
        elif roll < 0.12:
            got.append(Answer(0, VALUE))
        else:
            got.append(Answer(int(p.dx * rng.uniform(0.8, 1.2)) + 1, VALUE))
    return got


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_resident_plan_is_the_python_plan(seed):
    """Same probes, same order -- which matters because the answers are
    zipped back against it."""
    ladders = _ladders(seed)
    sizes = _sizes(ladders, seed)
    want = plan_sized(ladders, sizes)

    resident = ladders_from(ladders)
    slots, flat, spans = [], [], [0]
    by_id = {lad.arc.id: k for k, lad in enumerate(ladders)}
    for arc_id, values in sizes.items():
        slots.append(by_id[arc_id])
        flat.extend(int(v) for v in values)
        spans.append(len(flat))
    at, deltas = resident.plan_sized(slots, flat, spans)

    assert len(deltas) == len(want.probes)
    assert deltas == [p.dx for p in want.probes]
    assert [ladders[k].arc.id for k in at] == [
        arc.id for arc, span in zip(want.arcs, want.spans, strict=True)
        for _ in range(span[1] - span[0])]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_resident_merge_is_the_python_merge(seed):
    """Every ladder's points after absorbing the same answers."""
    ladders = _ladders(seed)
    sizes = _sizes(ladders, seed)
    want = plan_sized(ladders, sizes)
    answers = _answers(want.probes, seed)

    resident = ladders_from(ladders)
    by_id = {lad.arc.id: k for k, lad in enumerate(ladders)}
    slots, flat, spans = [], [], [0]
    for arc_id, values in sizes.items():
        slots.append(by_id[arc_id])
        flat.extend(int(v) for v in values)
        spans.append(len(flat))
    at, deltas = resident.plan_sized(slots, flat, spans)

    names, status, values = [], [], []
    for got in answers:
        if got.status is not None and got.status.name != "VALUE":
            if got.status.name not in names:
                names.append(got.status.name)
            status.append(names.index(got.status.name) + 1)
            values.append(0)
        else:
            status.append(0)
            values.append(max(0, int(got.value)))
    resident.absorb(at, deltas, values, status, names)

    merge(ladders, collect(want, answers))
    for k, lad in enumerate(ladders):
        deltas_r, quotes_r = resident.points(k)
        assert deltas_r == lad.deltas, f"slot {k} deltas"
        assert quotes_r == lad.quotes, f"slot {k} quotes"
        assert resident.attempted(k) == lad.attempted, f"slot {k} attempted"
        assert resident.failures(k) == lad.failures, f"slot {k} failures"


def _same(a: _Fitted, b: _Fitted) -> bool:
    """Field for field, with NaN equal to itself.

    `eta` is NaN wherever the fit had no drift to report, which is a value and
    not a failure -- and IEEE says NaN is not itself, so a plain `==` would
    call two identical fits different.
    """
    for x, y in zip(a, b, strict=True):
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
        if x != y:
            return False
    return True


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_resident_fit_is_the_python_fit(seed):
    """The fits, field for field.

    Equality rather than a tolerance: both sides hand the same floats to the
    same compiled `calibrate`, so the only way to differ is to have scaled the
    ladder differently -- which is exactly what this is here to catch.
    """
    from erouter.core import accel

    ladders = _ladders(seed)
    resident = ladders_from(ladders)
    slots = list(range(len(ladders)))
    got = resident.recalibrate(slots, DRIFT_TOL)

    rows = [(*lad.as_float(), _quantum(lad.arc.decimals_out)) for lad in ladders]
    want = accel.calibrate_many(rows, DRIFT_TOL)

    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        if len(ladders[k].deltas) < 3:
            assert a is None, f"slot {k}: too short to fit"
            continue
        assert (a is None) == (b is None), f"slot {k}: one side refused"
        if a is not None:
            assert _same(_Fitted(*a), _Fitted(*b)), f"slot {k}: {a} != {b}"


def test_a_fork_leaves_the_warm_ladders_alone():
    """Every quote refines from the same coarse start, which is the whole
    reason the reference copies them."""
    ladders = _ladders(9)
    resident = ladders_from(ladders)
    before = resident.points(0)

    fork = resident.fork()
    fork.absorb([0], [before[0][0] * 7 + 1], [12345], [0], [])
    assert resident.points(0) == before, "the warm copy moved"
    assert fork.points(0) != before, "the fork did not"
