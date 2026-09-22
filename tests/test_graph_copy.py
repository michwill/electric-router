"""Copying a graph must not try to copy the accelerator's resident problem.

`problem_for` parks a native object on `ArcArrays.accel` and keys its cache on
the `id()` of the very arrays it was built from, so a copy could never have
used the original's entry.  The object also does not pickle, which is what
`copy.deepcopy` falls back to -- and `route` deep-copies the graph to refine a
second finalist, so with `EROUTER_ACCEL=1` that raised instead of refining.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from erouter.core.graph import ArcArrays


class _Resident:
    """Stands in for the native problem: deepcopy falls back to pickling it."""

    def __reduce__(self):
        raise TypeError("cannot pickle 'builtins.Problem' object")


def graph_of(size: int = 3, n_nodes: int = 3) -> ArcArrays:
    return ArcArrays(
        tau=np.arange(size, dtype=np.int64),
        sig=np.arange(1, size + 1, dtype=np.int64) % n_nodes,
        a=np.ones(size),
        B=np.ones(size),
        G=np.ones(size),
        eps=np.zeros(size),
        cap=np.full(size, np.inf),
        flagged=np.zeros(size, bool),
        clamped=np.zeros(size, bool),
        n_nodes=n_nodes,
        sources=[[k] for k in range(size)],
    )


def test_a_resident_problem_would_refuse_to_be_copied():
    """The premise: without the guard this is the failure, not a slow copy."""
    with pytest.raises(TypeError):
        copy.deepcopy({"accel": _Resident()})


def test_copying_a_graph_drops_the_resident_problem():
    g = graph_of()
    g.accel = ("some key", _Resident())
    clone = copy.deepcopy(g)
    assert clone.accel is None


def test_the_copy_is_still_a_copy():
    g = graph_of()
    g.accel = ("some key", _Resident())
    clone = copy.deepcopy(g)
    assert np.array_equal(clone.tau, g.tau)
    assert clone.tau is not g.tau          # deep, not shared
    assert clone.sources == g.sources
    assert clone.sources is not g.sources
    clone.G[0] = 99.0
    assert g.G[0] == 1.0                   # and writing to it is safe


def test_a_graph_with_no_resident_copies_as_before():
    g = graph_of()
    clone = copy.deepcopy(g)
    assert clone.accel is None
    assert np.array_equal(clone.eps, g.eps)
