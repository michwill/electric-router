"""Dense linear solves for the graph Laplacian.

Measured at the sizes this router actually runs at (single-threaded, us/solve):

    n=299, 900 arcs   dense LU 2125   dense Cholesky 2516   scipy splu 6272
    n=10              dense LU   53   dense Cholesky   64   scipy splu   78

Three things follow, none of which match the textbook:

* Dense LU beats dense Cholesky by 15-25%: LAPACK's blocked `getrf` wins at this
  size and the n^3/3 flop advantage does not show up until much larger n.
* Sparse is 3x *slower* at n~300 -- per-call symbolic analysis dominates.
* The per-pivot system is tiny anyway (the active set is 3-10 arcs), where
  everything is call overhead.

So: dense, refactorise every pivot -- ~2.5 ms against a ~7 s cold route, 0.03%.
Sparse Cholesky with rank-1 updates is a real optimisation at n~1e5, not here,
and this interface keeps it a drop-in.

Cholesky is kept as a debug mode: it raises on an indefinite matrix where LU
silently returns a plausible answer.  That is a weak net -- a single negative
conductance at 10% of max(G) leaves the Laplacian positive definite and both
succeed -- which is why `G > 0` is enforced at the source instead.
"""

from __future__ import annotations

import numpy as np


class SingularSystem(RuntimeError):
    """The Laplacian is not invertible: usually a disconnected node (§9.4)."""


#: How many times to correct a solve against its own residual.
#
# Backward stability promises a residual of `||L|| ||u|| eps` and no more, and
# that is not enough here: `psi = G (u_tau - u_sig - eps)` is a small
# difference of larger numbers times a conductance running to 1e8, so what
# backward stability allows arrives in `psi` at about `TOL` -- the very
# threshold the drop rule reads.  On a degenerate arc the *rounding* then
# decides whether it leaves the basis, and two solvers rounding differently
# take different pivots from there.
#
# Measured on the graph that exposed it: `cond(L)` 1.6e4, a factored solve
# leaving 9.3e-10 where LAPACK left zero, which put a degenerate arc at
# -1.5e-9 against -4.1e-11.  Refinement closes that, and it *saves* time --
# an accurate `u` wastes fewer pivots, so the solve converges in fewer of them.
#
# Each step squares the residual's relative size, so the loop stops as soon as
# one fails to shrink it: below the working precision there is nothing left to
# win.
REFINE_STEPS = 3


def refine(matrix: np.ndarray, rhs: np.ndarray, x: np.ndarray,
           solve) -> np.ndarray:
    """Correct `x` against its own residual, in place of nothing."""
    best = np.inf
    for _ in range(REFINE_STEPS):
        residual = rhs - matrix @ x
        size = float(np.max(np.abs(residual))) if residual.size else 0.0
        if size == 0.0 or not size < best:
            return x
        best = size
        correction = solve(matrix, residual)
        if not np.all(np.isfinite(correction)):
            return x
        x = x + correction
    return x


class DenseLU:
    """Default.  Raises only on exact singularity."""

    name = "lu"

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            x = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise SingularSystem(str(exc)) from exc
        return refine(matrix, rhs, x, np.linalg.solve)


class DenseCholesky:
    """Debug mode.  Also raises when the matrix loses positive definiteness."""

    name = "cholesky"

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise SingularSystem(f"not positive definite: {exc}") from exc
        def factored(_matrix, b):
            return np.linalg.solve(factor.T, np.linalg.solve(factor, b))

        return refine(matrix, rhs, factored(matrix, rhs), factored)


DEFAULT_SOLVER = DenseLU()


def get_solver(name: str | None = None):
    if name in (None, "lu"):
        return DenseLU()
    if name == "cholesky":
        return DenseCholesky()
    raise ValueError(f"unknown linear solver {name!r}")
