"""A cap that is only how far the ladder reached must not bind like a wall."""

from __future__ import annotations

import numpy as np
import pytest

from erouter.core import accel, graph, solve
from erouter.core.nodes import NodeMap
from erouter.core.pipeline import route
from erouter.core.pools import Coin, PoolSpec
from erouter.core.quoter import Quote
from erouter.core.transport import Status
from erouter.core.types import Dialect

UNIT = 10**18
RESERVE = 1_000_000.0
SRC, MID, DST = ("0x" + f"{k:02x}" * 20 for k in (1, 2, 3))


def deep(d: float) -> float:
    """A plain pool, far deeper than the trade."""
    return 0.9996 * d * RESERVE * 50 / (RESERVE * 50 + d)


def rebalancing(d: float) -> float:
    """A dynamic fee that falls faster than the price moves, up to a tenth of the pool."""
    fee = 0.004 - 0.002 * min(d / (0.1 * RESERVE), 1.0)
    return d * (1.0 - fee) - d * d / (2 * 100 * RESERVE)


CURVES = {(1, 0, 1): deep, (1, 1, 0): deep, (2, 0, 1): rebalancing, (2, 1, 0): deep}


class Transport:
    block = 1
    chain_id = 1


class Quoter:
    local = True
    transport = Transport()

    def _f(self, pool: str, i: int, j: int, wei: int) -> int:
        curve = CURVES.get((int(pool[2:4], 16) - 0xA0, i, j))
        return int(curve(wei / UNIT) * UNIT) if curve and wei > 0 else 0

    def probe(self, probes):
        out = []
        for p in probes:
            value = self._f(p.pool, p.i, p.j, p.dx)
            out.append(Quote(Status.VALUE if value > 0 else Status.REVERTED, value))
        return out

    def quote_routes(self, routes, amounts, dst_slots):
        return [self._walk(legs, a, d) for legs, a, d in zip(routes, amounts, dst_slots, strict=True)]

    def _walk(self, legs, amount_in: int, dst_slot: int) -> int:
        slots, base = {0: amount_in}, {}
        for leg in legs:
            have = slots.get(leg.src_slot, 0)
            base.setdefault(leg.src_slot, have)
            take = have if not leg.bps else min(have, base[leg.src_slot] * leg.bps // 10_000)
            slots[leg.src_slot] = have - take
            slots[leg.dst_slot] = slots.get(leg.dst_slot, 0) + self._f(leg.target, leg.i, leg.j, take)
        return slots.get(dst_slot, 0)


def universe():
    pools = []
    for k, (a, b) in enumerate(((SRC, MID), (MID, DST)), start=1):
        spec = PoolSpec(address="0x" + f"{0xA0 + k:02x}" * 20, name=f"pool{k}", pool_type="main",
                        coins=(Coin(a, f"T{a[2:4]}", 18, 0), Coin(b, f"T{b[2:4]}", 18, 1)),
                        tvl_usd=2 * RESERVE)
        spec.dialect = Dialect.STABLE
        spec.balances = (int(RESERVE * UNIT), int(RESERVE * UNIT))
        pools.append(spec)
    nodes = NodeMap()
    for spec in pools:
        for coin in spec.coins:
            nodes.add_token(coin.address, coin.symbol, coin.decimals)
    return pools, nodes


def test_increasing_returns_past_the_coarse_grid_still_route():
    """5% of a pool whose rate improves to 10%: the coarse grid stops at 1%."""
    pools, nodes = universe()
    amount = int(0.05 * RESERVE * UNIT)
    result = route(pools, nodes, Quoter(), src_token=SRC, dst_token=DST,
                   amount_in=amount, verify_on_chain=True, measure_impact=False)
    hop = Quoter()
    direct = hop._f(pools[1].address, 0, 1, hop._f(pools[0].address, 0, 1, amount))
    assert result.verified_out >= direct * (1 - 1e-9)


def test_the_compiled_solve_names_the_cut(monkeypatch):
    """A short destination cut reads the same from Rust as from the reference."""
    if not accel.available():
        pytest.skip("no compiled solver")
    monkeypatch.setattr(solve, "_ACCEL_ON", True)
    g = graph.build(np.array([0, 1]), np.array([1, 2]), np.array([1.0, 1.0]),
                    np.array([1.0, 1.0]), np.ones(3), 1.0,
                    cap=np.array([np.inf, 0.2]), n_nodes=3)
    sol = solve.active_set_solve(g, 0, 2, 1.0)
    assert not sol.feasible
    assert "into the destination can carry 20.000%" in sol.reason
