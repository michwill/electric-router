"""Sampled leg curves (§7) -- no chain.

The interpolant replaces the quadratic model inside the split search, so what
matters is not that it is smooth but that it is *monotone* and that composing
several of them tracks the truth to well under a basis point.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from erouter.core.curves import CurveError, fit, linear, sizes


def cpmm(x: float, *, reserve_in: float = 1e6, reserve_out: float = 1e6,
         fee: float = 3e-4) -> float:
    """Constant product with a fee -- concave, and analytically exact."""
    dx = x * (1.0 - fee)
    return reserve_out * dx / (reserve_in + dx)


def test_it_reproduces_a_cpmm_to_far_below_a_basis_point():
    nodes = np.geomspace(1e2, 1e5, 24)
    curve = fit(nodes, [cpmm(v) for v in nodes])
    probe = np.geomspace(2e2, 9e4, 500)
    error = np.array([abs(curve.at(v) / cpmm(v) - 1) for v in probe])
    assert error.max() * 10_000 < 0.01, error.max() * 10_000


def test_the_interpolant_is_exact_at_its_nodes():
    nodes = [1.0, 10.0, 100.0, 1000.0]
    values = [cpmm(v) for v in nodes]
    curve = fit(nodes, values)
    for x, y in zip(nodes, values, strict=True):
        assert curve.at(x) == pytest.approx(y, rel=1e-12)
    assert curve.at(0.0) == 0.0


def test_a_saturation_wall_does_not_curl_upward():
    """The failure a cubic introduces, and the reason for a linear `u`.

    A LLAMMA market that has run out of reachable liquidity returns the same
    output over a thousandfold range of inputs.  An overshooting interpolant
    invents a bump there, and an optimiser maximising output walks into it.
    """
    nodes = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
    wall = [1.0, 9.9, 11.47, 11.472806, 11.472806, 11.472806]
    curve = fit(nodes, wall)
    dense = np.geomspace(1.0, 1e5, 2000)
    values = np.array([curve.at(v) for v in dense])
    assert np.all(np.diff(values) >= -1e-12), "interpolant went backwards"
    assert values.max() <= max(wall) + 1e-9, "interpolant overshot the wall"


def test_monotone_probes_give_a_monotone_curve_at_any_node_density():
    """The structural guarantee: it holds for a ladder as coarse as you like."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        count = int(rng.integers(2, 9))
        nodes = np.sort(rng.uniform(1.0, 1e6, count))
        if np.any(np.diff(nodes) <= 0):
            continue
        values = np.cumsum(rng.uniform(0.1, 1.0, count)) * rng.uniform(0.5, 2.0)
        curve = fit(nodes, values)
        dense = np.geomspace(nodes[0], nodes[-1] * 4, 500)
        got = np.array([curve.at(v) for v in dense])
        assert np.all(np.diff(got) >= -1e-9)
        assert got.max() <= curve.at(dense[-1]) + 1e-9


def test_it_is_monotone_through_a_peg_edge():
    """A stableswap leaving its flat region: a kink, not a smooth curve."""
    nodes = np.geomspace(1e3, 1e7, 20)
    values = [v * (0.9995 if v < 1e6 else 0.9995 - 3e-7 * (v - 1e6) / 1e5) for v in nodes]
    curve = fit(nodes, values)
    dense = np.geomspace(1e3, 1e7, 3000)
    got = np.array([curve.at(v) for v in dense])
    assert np.all(np.diff(got) >= -1e-9)


def test_above_the_last_node_it_keeps_rising_and_keeps_saturating():
    curve = fit([1.0, 2.0, 3.0], [1.0, 1.9, 2.7])
    assert curve.at(10.0) > 2.7, "a flat clamp would trap the optimiser here"
    # Still concave out there: a cubic extrapolation of a concave `f` turns
    # over and reports *less* output for more input, which reads as a wall.
    first = curve.at(20.0) - curve.at(10.0)
    second = curve.at(30.0) - curve.at(20.0)
    assert 0.0 < second < first


def test_a_linear_leg_needs_no_probes():
    curve = linear(1.25)
    assert curve.at(4.0) == pytest.approx(5.0)
    assert curve.at(0.0) == 0.0


def test_it_refuses_a_ladder_too_short_to_interpolate():
    with pytest.raises(CurveError):
        fit([1.0], [1.0])


def test_probe_sizes_are_increasing_and_bounded():
    got = sizes(1_000_000.0)
    assert got == sorted(set(got))
    assert got[-1] <= 1_000_000
    assert len(got) <= 24
    assert sizes(1.0) == [], "nothing useful to sample below two wei"


def test_probe_sizes_do_not_collapse_on_a_low_decimal_token():
    """A 2-decimal token's ladder must not be 24 copies of the same integer."""
    got = sizes(500.0)
    assert len(got) == len(set(got))
    assert all(b > a for a, b in pairwise(got))
