"""`graph.build` in Rust must be `graph.build` in Python, arc for arc.

The reference is `core/graph.py`; this is the mirror check for `rust/src/
graph.rs`.  Everything here is `float`-exact rather than approximate, and that
is the whole point of the module being ported first: `build` does divisions,
comparisons and one median, in an order both sides can follow literally.  A
tolerance would hide the interesting failures -- a floor applied before a
ceiling instead of after, a median taken over the wrong subset, a duplicate
group keyed on a rounding the other side does not do.

The generated universes are deliberately nasty: clamped arcs (`B == 0`, so
`G == inf`), flagged arcs (which the ceiling's reference excludes), exact
duplicates that must merge, and spreads wide enough to move the adaptive dust
floor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import graph
from erouter.core.accel import available

pytestmark = pytest.mark.skipif(not available(), reason="erouter_solve not installed")


def native():
    import erouter_solve

    return erouter_solve.Graph


# --------------------------------------------------------------- universes


def universe(seed: int) -> dict:
    """One randomly shaped graph, with the awkward cases forced in."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    m = int(rng.integers(n, 4 * n))
    tau = rng.integers(0, n, m)
    sig = (tau + rng.integers(1, n, m)) % n  # never a self-loop
    a = np.exp(rng.normal(0.0, 0.4, m))
    # A wide but not pathological spread, which is what the adaptive floor is
    # for: 1e-9 to 1e-2 in B against a ~1 numerator is eight orders in G.
    B = np.exp(rng.uniform(math.log(1e-9), math.log(1e-2), m))
    nu = np.exp(rng.normal(0.0, 0.3, n))
    cap = np.where(rng.random(m) < 0.2, rng.uniform(1.0, 100.0, m), np.inf)
    flagged = rng.random(m) < 0.15

    # A clamped arc has B == 0 and therefore G == inf, and needs a finite cap
    # or `build` refuses it outright.
    clamped_at = rng.random(m) < 0.1
    B[clamped_at] = 0.0
    cap[clamped_at] = rng.uniform(1.0, 50.0, int(clamped_at.sum()))

    # Force some exact duplicates: same endpoints, same a, same B.
    if m > 4:
        for k in range(1, min(4, m)):
            a[k] = a[0]
            B[k] = B[0]
            tau[k] = tau[0]
            sig[k] = sig[0]
            cap[k] = cap[0]
    return {
        "tau": tau.astype(np.int64),
        "sig": sig.astype(np.int64),
        "a": a,
        "B": B,
        "nu": nu,
        "cap": cap,
        "flagged": flagged,
        "n_nodes": n,
        "Psi": float(np.exp(rng.normal(0.0, 2.0))),
    }


SEEDS = list(range(40))


def build_both(u: dict, **kw):
    """The reference and the port over the same universe."""
    reference = graph.build(
        u["tau"], u["sig"], u["a"], u["B"], u["nu"], u["Psi"],
        cap=u["cap"].copy(), flagged=u["flagged"], n_nodes=u["n_nodes"], **kw
    )
    ported = native().build(
        list(u["tau"]), list(u["sig"]), list(u["a"]), list(u["B"]), list(u["nu"]),
        u["Psi"], cap=list(u["cap"]), flagged=[bool(v) for v in u["flagged"]],
        n_nodes=u["n_nodes"],
        **{k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in kw.items()},
    )
    return reference, ported


def same(got, want, what: str) -> None:
    """Bit-exact, with NaN equal to NaN -- which `==` is not."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{what}: {got.shape} != {want.shape}"
    agree = (got == want) | (np.isnan(got) & np.isnan(want))
    assert agree.all(), f"{what}: {got[~agree]} != {want[~agree]}"


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("seed", SEEDS)
def test_build_agrees_arc_for_arc(seed):
    u = universe(seed)
    reference, ported = build_both(u)

    assert len(ported) == reference.m
    same(ported.tau, reference.tau, "tau")
    same(ported.sig, reference.sig, "sig")
    same(ported.a, reference.a, "a")
    same(ported.b, reference.B, "B")
    same(ported.g, reference.G, "G")
    same(ported.eps, reference.eps, "eps")
    same(ported.cap, reference.cap, "cap")
    assert ported.flagged == [bool(v) for v in reference.flagged]
    assert ported.clamped == [bool(v) for v in reference.clamped]
    assert ported.n_nodes == reference.n_nodes
    assert ported.ill_conditioned == reference.ill_conditioned
    assert ported.condition() == reference.condition()


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_arcs_are_dropped_for_the_same_reasons(seed):
    u = universe(seed)
    reference, ported = build_both(u)

    index, reason = ported.dropped()
    assert dict(zip(index, reason, strict=True)) == reference.dropped

    flat, spans = ported.sources()
    groups = [list(flat[spans[k]:spans[k + 1]]) for k in range(len(spans) - 1)]
    assert groups == reference.sources


@pytest.mark.parametrize("seed", SEEDS)
def test_scale_normalises_to_the_same_median(seed):
    u = universe(seed)
    reference, ported = build_both(u)

    scaled, psi_reference = graph.scale(reference, u["Psi"])
    psi_ported = ported.scale(u["Psi"])

    assert psi_ported == psi_reference
    assert ported.g_scale == scaled.g_scale
    same(ported.g, scaled.G, "G after scale")
    same(ported.cap, scaled.cap, "cap after scale")


@pytest.mark.parametrize("seed", SEEDS)
def test_arc_params_and_the_ceiling_agree(seed):
    u = universe(seed)
    G, eps = graph.arc_params(u["tau"], u["sig"], u["a"], u["B"], u["nu"])
    got_G, got_eps = native().arc_params(
        list(u["tau"]), list(u["sig"]), list(u["a"]), list(u["B"]), list(u["nu"])
    )
    same(got_G, G, "G")
    same(got_eps, eps, "eps")

    flagged = [bool(v) for v in u["flagged"]]
    same(
        native().ceiling_conductance(list(G), flagged),
        graph.ceiling_conductance(G, u["flagged"]),
        "ceiling",
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_topology_agrees(seed):
    u = universe(seed)
    n = u["n_nodes"]
    for root in range(n):
        assert native().component_of(root, list(u["tau"]), list(u["sig"]), n) == [
            bool(v) for v in graph.component_of(root, u["tau"], u["sig"], n)
        ]

    keep = sorted(range(n))[: max(1, n - 2)]
    G, _ = graph.arc_params(u["tau"], u["sig"], u["a"], u["B"], u["nu"])
    G = np.where(np.isfinite(G), G, 1e6)  # `laplacian` is not asked to handle inf
    want = graph.laplacian(u["tau"], u["sig"], G, n, np.array(keep))
    got = native().laplacian(list(u["tau"]), list(u["sig"]), list(G), n, keep)
    same(np.array(got).reshape(want.shape), want, "laplacian")


@pytest.mark.parametrize("seed", SEEDS[:12])
def test_the_dust_floor_backs_off_the_same_way(seed):
    """`require` is the branch that trades conditioning for connectivity."""
    u = universe(seed)
    src, dst = 0, u["n_nodes"] - 1
    reference, ported = build_both(u, require=(src, dst))
    same(ported.g, reference.G, "G under require")
    assert ported.ill_conditioned == reference.ill_conditioned
    index, _ = ported.dropped()
    assert list(index) == list(reference.dropped)


# ------------------------------------------------------- refusals, in kind


def test_a_clamped_arc_without_a_cap_is_refused_on_both_sides():
    kw = {"tau": [0], "sig": [1], "a": [1.0], "B": [0.0], "nu": [1.0, 1.0]}
    with pytest.raises(ValueError, match="unbounded"):
        graph.build(**{k: np.asarray(v) for k, v in kw.items()}, Psi=1.0, n_nodes=2)
    with pytest.raises(ValueError, match="unbounded"):
        native().build(kw["tau"], kw["sig"], kw["a"], kw["B"], kw["nu"], 1.0, n_nodes=2)


def test_negative_curvature_is_refused_on_both_sides():
    args = ([0], [1], [1.0], [-1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="negative curvature"):
        graph.arc_params(*[np.asarray(v) for v in args])
    with pytest.raises(ValueError, match="negative curvature"):
        native().arc_params(*args)


def test_the_two_refusals_read_the_same():
    """The message is the contract: `_assemble` matches on part of it."""
    args = ([0], [1], [1.0], [-2.5e-7], [1.0, 1.0])
    try:
        graph.arc_params(*[np.asarray(v) for v in args])
    except ValueError as e:
        want = str(e)
    try:
        native().arc_params(*args)
    except ValueError as e:
        got = str(e)
    assert got == want


def test_a_pathological_spread_is_refused_on_both_sides():
    # 1e16 apart in G, well past PATHOLOGICAL_CONDITION.
    tau, sig = [0, 0], [1, 1]
    a, B, nu = [1.0, 1.0], [1e-16, 1.0], [1.0, 1.0]
    with pytest.raises(ValueError, match="clamped in the wrong space"):
        graph.build(*[np.asarray(v) for v in (tau, sig, a, B, nu)], 1e-30, n_nodes=2)
    with pytest.raises(ValueError, match="clamped in the wrong space"):
        native().build(tau, sig, a, B, nu, 1e-30, n_nodes=2)
