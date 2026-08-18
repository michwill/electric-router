"""The Rust solver, when it is there.

`erouter.core` has to stay importable with nothing but numpy -- that is what
`tests/test_purity.py` guards, and it is the reason the Flet frontend can load
this at all.  So the accelerator is *never* a requirement: this module tries the
import, and every caller has a pure-Python path that answers when it is absent.

The boundary is one call per solve rather than per pivot.  A quote runs ~45
solves and ~2,250 pivots; crossing per pivot would pay the FFI cost fifty times
more often and, worse, would leave the pivot loop in Python, which is the part
that actually costs.

Arrays cross as plain lists rather than numpy buffers.  That keeps the
extension clear of the numpy C API, which is the fiddliest part of shipping a
native module into Pyodide -- the same wheel then works in CPython, in the
browser, and in a Web Worker.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised by whichever half is installed
    import erouter_solve as _rust
except ImportError:  # pragma: no cover
    _rust = None

def available() -> bool:
    """Whether the compiled solver can be used at all."""
    return _rust is not None

def version() -> str:
    return getattr(_rust, "__version__", "unknown") if _rust else "absent"

def problem_for(g):
    """The Rust-side graph for `g`, built once and kept on it.

    A quote solves 45-106 times over the same arcs; only the warm start, the
    forbidden mask and the pins differ.  Handing `tau`, `sig`, `G`, `eps` and
    `cap` across the boundary every time meant tens of thousands of Python
    floats per quote to convey data that had not changed -- which cost about
    what the Rust arithmetic saved.

    Cached on the `ArcArrays` itself, keyed by `id` of the arrays it was built
    from, so a graph that is rebuilt (a new block, a re-fit) gets a new problem
    rather than a stale one.  `ArcArrays` is frozen in practice but not by
    construction, hence the key rather than a plain attribute.  It carries an
    `accel` field for this: `slots=True` leaves nowhere else to put it, and the
    version that wrote through `object.__setattr__` never cached anything.
    """
    if _rust is None:
        return None
    key = (id(g.tau), id(g.sig), id(g.G), id(g.eps), id(g.cap))
    got = g.accel
    if got is not None and got[0] == key:
        return got[1]
    cap = [float(v) if math.isfinite(v) else math.inf
           for v in np.asarray(g.cap, float)]
    problem = _rust.Problem(
        [int(v) for v in np.asarray(g.tau)],
        [int(v) for v in np.asarray(g.sig)],
        [float(v) for v in np.asarray(g.G, float)],
        [float(v) for v in np.asarray(g.eps, float)],
        cap,
        int(g.n_nodes),
    )
    try:
        g.accel = (key, problem)
    except (AttributeError, TypeError):  # a graph that will not hold it
        pass
    return problem


def _seed_mask(a0, m: int) -> list[bool]:
    """`A0` as a length-`m` boolean mask, however it was spelled.

    The Python solver writes `A[np.asarray(A0)] = True`, which accepts either
    a mask or a list of indices -- and `Solution.active` is a list of indices,
    which is exactly what the pipeline warm-starts from.  Handing that to
    `np.asarray(a0, bool)` instead turns `[3, 17, 42]` into three `True`s, so
    the accelerated path silently started from a different basis: measured
    against the Python solver on 54 real problems, only 8 agreed on the pivot
    count, and the graphs that seemed to diverge were never given the same
    starting point to diverge from.
    """
    arr = np.asarray(a0)
    if arr.dtype == np.bool_ and arr.shape == (m,):
        return [bool(v) for v in arr]
    mask = np.zeros(m, bool)
    mask[arr] = True
    return [bool(v) for v in mask]


def solve_arrays(
    g,
    src: int,
    dst: int,
    psi_total: float,
    *,
    a0=None,
    forbidden=None,
    pinned=None,
    tol: float,
    maxit: int,
    min_flow: float,
    gas_cost: float,
    partial_ok: bool,
    rank1: bool = True,
) -> dict | None:
    """Run the Rust solve, or return None if it is not installed.

    `inf` survives the crossing as a float, which is what a capless arc needs;
    NaN would not, and does not occur -- `calibrate` is the only source of
    `cap` and it is either finite or `inf` by construction (§2.3).
    """
    problem = problem_for(g)
    if problem is None:
        return None
    return problem.solve(
        src=int(src),
        dst=int(dst),
        psi_total=float(psi_total),
        a0=None if a0 is None else _seed_mask(a0, int(g.m)),
        forbidden=None if forbidden is None else
        [bool(v) for v in np.asarray(forbidden, bool)],
        pinned=None if not pinned else [(int(k), float(v)) for k, v in pinned.items()],
        tol=float(tol),
        maxit=int(maxit),
        min_flow=float(min_flow),
        gas_cost=float(gas_cost),
        partial_ok=bool(partial_ok),
        rank1=bool(rank1),
    )
