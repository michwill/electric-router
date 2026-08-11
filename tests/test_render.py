"""The diagram model and its terminal rendering."""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core.realize import realize
from erouter.core.render_text import render
from erouter.core.rendermodel import build_diagram, format_units
from test_realize import (
    CRVUSD,
    ETH,
    POOL_A,
    POOL_B,
    POOL_C,
    USDC,
    WETH,
    arc,
    merged_nodes,
)


@pytest.fixture
def hybrid():
    """Split at the source, one branch two hops, ending on native ETH."""
    nodes = merged_nodes()
    arcs = [
        arc(POOL_A, USDC, WETH, nodes, a=1 / 1890.0, B=1e-10),
        arc(POOL_B, USDC, CRVUSD, nodes, a=0.9999, B=1e-10),
        arc(POOL_C, CRVUSD, WETH, nodes, a=1 / 1888.0, B=1e-10),
    ]
    arcs[2].eps = -0.00021  # a favourably dislocated pool
    for a in arcs:
        a.G = 1e6
    nu = np.zeros(nodes.n_nodes)
    nu[nodes.node(USDC)] = 1.0
    nu[nodes.node(CRVUSD)] = 1.0
    nu[nodes.node(WETH)] = 1890.0
    route = realize(
        arcs, np.array([6000.0, 4000.0, 4000.0]), nu, nodes,
        src_token=USDC, dst_token=ETH, amount_in=10_000 * 10**6,
        potentials=np.array([0.001, 0.0, 0.0005, 0.0, 0.0][: nodes.n_nodes]),
    )
    return route, nodes


def test_format_units_is_exact():
    """Amounts must never round-trip through a float and lose wei."""
    assert format_units(1234567890123456789, 18) == "1.234568"
    assert format_units(10**6, 6) == "1.000000"
    assert format_units(0, 18) == "0.000000"
    assert format_units(123456789 * 10**18, 18) == "123,456,789.000000"


def test_diagram_has_a_bus_per_slot_and_an_element_per_leg(hybrid):
    route, nodes = hybrid
    diagram = build_diagram(route, nodes)
    assert len(diagram.buses) == len(route.slots)
    assert len(diagram.elements) == len(route.legs)
    assert diagram.bus(0).is_source
    assert diagram.bus(route.dst_slot).is_dest


def test_bus_order_is_topological(hybrid):
    """A bus must be drawn after everything that feeds it."""
    route, nodes = hybrid
    diagram = build_diagram(route, nodes)
    position = {slot: k for k, slot in enumerate(diagram.order)}
    for element in diagram.elements:
        assert position[element.src_slot] < position[element.dst_slot]
    assert diagram.order[0] == 0
    assert diagram.order[-1] == route.dst_slot


def test_merged_nodes_are_labelled(hybrid):
    route, nodes = hybrid
    diagram = build_diagram(route, nodes)
    weth_bus = next(b for b in diagram.buses if b.symbol == "WETH")
    assert "ETH" in weth_bus.merged_with


def test_render_shows_the_circuit_vocabulary(hybrid):
    route, nodes = hybrid
    diagram = build_diagram(
        route, nodes, title="10,000 USDC -> ETH", certificate=True,
        pool_names={POOL_A.lower(): "TricryptoUSDC", POOL_C.lower(): "TriCRV"},
        ledger={"fee_bp": 1.1, "impact_bp": 3.0, "total_bp": 4.1},
    )
    text = render(diagram, unicode=True, color=False, width=96)

    assert "▷|" in text  # diode
    assert "/\\/\\/\\" in text  # resistor
    assert "↯|" in text  # the dislocated pool, as a battery
    assert "BATTERY" in text
    assert "MERGE" in text  # the zero-resistance node merge
    assert "u = ground" in text  # the grounded destination potential
    assert "certificate" in text
    assert "TricryptoUSDC" in text
    assert "loss ledger" in text
    assert max(len(line) for line in text.splitlines()) <= 130


def test_ascii_mode_has_no_box_drawing(hybrid):
    route, nodes = hybrid
    text = render(build_diagram(route, nodes), unicode=False, color=False, width=90)
    assert text.isascii()
    assert ">|" in text
    assert "-vvv-" in text


def test_colour_is_opt_in(hybrid):
    route, nodes = hybrid
    diagram = build_diagram(route, nodes)
    assert "\x1b[" not in render(diagram, color=False)
    assert "\x1b[" in render(diagram, color=True)


def test_a_false_certificate_states_its_reason(hybrid):
    route, nodes = hybrid
    diagram = build_diagram(
        route, nodes, certificate=False, certificate_reason="CHORD_ACTIVE"
    )
    text = render(diagram, color=False)
    assert "CHORD_ACTIVE" in text  # never swallowed (§15)
