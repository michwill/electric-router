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
) -> dict | None:
    """Run the Rust solve, or return None if it is not installed.

    `inf` survives the crossing as a float, which is what a capless arc needs;
    NaN would not, and does not occur -- `calibrate` is the only source of
    `cap` and it is either finite or `inf` by construction (§2.3).
    """
    if _rust is None:
        return None
    cap = [float(v) if math.isfinite(v) else math.inf for v in np.asarray(g.cap, float)]
    return _rust.solve(
        tau=[int(v) for v in np.asarray(g.tau)],
        sig=[int(v) for v in np.asarray(g.sig)],
        g=[float(v) for v in np.asarray(g.G, float)],
        eps=[float(v) for v in np.asarray(g.eps, float)],
        cap=cap,
        n_nodes=int(g.n_nodes),
        src=int(src),
        dst=int(dst),
        psi_total=float(psi_total),
        a0=None if a0 is None else [bool(v) for v in np.asarray(a0, bool)],
        forbidden=None if forbidden is None else
        [bool(v) for v in np.asarray(forbidden, bool)],
        pinned=None if not pinned else [(int(k), float(v)) for k, v in pinned.items()],
        tol=float(tol),
        maxit=int(maxit),
        min_flow=float(min_flow),
        gas_cost=float(gas_cost),
        partial_ok=bool(partial_ok),
        
    )
