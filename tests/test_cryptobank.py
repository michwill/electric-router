"""Cryptoswap arcs as banks: fitted from the pool's own invariant, sized to the
trade, bounded by the pool, and left alone where one parabola is enough."""

from __future__ import annotations

import math

import numpy as np
import pytest

from erouter.core import cryptobank
from erouter.core.bank import capacity, collapse, output
from erouter.core.nodes import Conversion, ConversionKind, NodeMap
from erouter.core.quoter import Quote
from erouter.core.transport import Status
from erouter.core.types import ArcKind, PoolArc

CRV = "0xd533a949740bb3306d119cc777fa900ba034cd52"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
POOL = "0x" + "4e" * 20
FEE = 0.003


class ConstantProduct:
    """A pool the client can compute: constant product with a fee, refusing a
    trade past 30x its reserve the way a cryptoswap invariant stops converging."""

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
        self.sizes: list[float] = []

    def computes(self, pool: str) -> bool:
        return pool == POOL

    def chain(self, dx: float) -> float:
        return self.y * dx / (self.x + dx) * (1 - FEE)

    def probe(self, probes):
        out = []
        for p in probes:
            dx = p.dx / 1e18
            self.sizes.append(dx)
            if dx > 30 * self.x:
                out.append(Quote(Status.REVERTED, 0))
            else:
                out.append(Quote(Status.VALUE, int(self.chain(dx) * 1e18)))
        return out


def setup(x: float, y: float, kind=ArcKind.SWAP_CRYPTO):
    nodes = NodeMap()
    nodes.add_token(CRV, "CRV", 18)
    nodes.add_token(WETH, "WETH", 18)
    spot = y / x * (1 - FEE)
    arc = PoolArc(
        id=f"{POOL}:2>1", pool=POOL, kind=kind, i=2, j=1, n_coins=3,
        token_in=CRV, token_out=WETH,
        tau=nodes.node(CRV), sigma=nodes.node(WETH),
        a=spot, B=1e-12, cap=math.inf, G=1e12, eps=-math.log(spot),
        reserve_in=int(x * 1e18), decimals_in=18, decimals_out=18,
        tvl_usd=1e7, note="TriCRV",
    )
    nu = np.ones(nodes.n_nodes)
    return nodes, arc, nu, ConstantProduct(x, y)


def test_a_bank_follows_the_pool_to_the_size_of_the_trade():
    """TriCRV-sized: 10M CRV against 2M in the pool.  One parabola peaks at a
    fraction of that; the bank has to keep paying."""
    nodes, arc, nu, client = setup(2e6, 256.0)
    arcs, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=2e6)
    bank = banks[(POOL, 2, 1)]
    assert len(bank) == cryptobank.SEGMENTS
    assert [a.id for a in arcs] == [f"{arc.id}#{k}" for k in range(len(bank))]
    assert all(a.parallel for a in arcs)
    for dx in (1e5, 5e5, 1e6, 2e6):
        assert abs(output(bank, dx) / client.chain(dx) - 1) < 0.005, dx


def test_a_pool_the_trade_barely_moves_keeps_its_parabola():
    """Parallel near-identical arcs are the degeneracy that made the solve churn."""
    nodes, arc, nu, client = setup(1e9, 1.28e5)
    arcs, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=1e5)
    assert banks == {} and arcs == [arc]


def test_a_small_pool_is_banked_to_its_own_depth_not_the_trade():
    """Sized to the trade, the first probe sat at 26x the reserve, the invariant
    refused it, and the pool kept an uncapped parabola that took 8M CRV."""
    nodes, arc, nu, client = setup(6e4, 7.0)
    arcs, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=1e7)
    bank = banks[(POOL, 2, 1)]
    assert capacity(bank) <= cryptobank.DEPTH * 6e4 * (1 + 1e-9)
    assert max(client.sizes) <= cryptobank.DEPTH * 6e4 * (1 + 1e-9)
    assert all(math.isfinite(a.cap) for a in arcs)


def test_only_cryptoswap_arcs_are_banked():
    nodes, arc, nu, client = setup(2e6, 256.0, kind=ArcKind.SWAP_STABLE)
    arcs, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=2e6)
    assert banks == {} and arcs == [arc]


STETH = "0xae7ab96520de3a18e5e111b5eaab095312d7fe84"
WST = "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"


def test_a_merged_token_is_banked_in_canonical_units():
    """wstETH is 1.2445 stETH, and a Curve arc leaves `rate_in` at 1.0 on it.
    The pieces were rescaled by that 1.0: every wstETH piece 1.2445x rich, and
    the graph held 219 WETH of free arbitrage in loops."""
    nodes = NodeMap()
    nodes.add_token(STETH, "stETH", 18)
    nodes.add_token(WST, "wstETH", 18)
    nodes.add_token(WETH, "WETH", 18)
    nodes.merge(Conversion(ConversionKind("ERC4626"), WST, STETH, 12, 10, target=WST))
    rate = nodes.rate(WST)
    assert rate == 1.2
    x, y = 1e4, 1.2e4                      # wstETH, WETH: 1.2 WETH a token
    client = ConstantProduct(x, y)
    spot_token = y / x * (1 - FEE)
    arc = PoolArc(
        id=f"{POOL}:0>1", pool=POOL, kind=ArcKind.SWAP_CRYPTO, i=0, j=1, n_coins=2,
        token_in=WST, token_out=WETH, tau=nodes.node(WST), sigma=nodes.node(WETH),
        a=spot_token / rate, B=1e-12, cap=math.inf, G=1e12, eps=0.0,
        reserve_in=int(x * 1e18), decimals_in=18, decimals_out=18, tvl_usd=1e7, note="LSD")
    nu = np.ones(nodes.n_nodes)
    pieces, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=1e4)
    assert abs(pieces[0].a / arc.a - 1) < 0.01, "canonical, like the arc it replaces"

    dx_token = 2e3
    psi = np.array([dx_token * rate if k == 1 else 0.0 for k in range(len(pieces))])
    psi[0] = 0.0
    folded, flows = collapse(pieces, psi, nu, nodes, banks, kind=ArcKind.SWAP_CRYPTO)
    leg = folded[0]
    out_canonical = leg.a * flows[0]
    assert abs(out_canonical / client.chain(dx_token) - 1) < 0.005


class Amplified(ConstantProduct):
    """Flat and then a wall: `p L tanh(dx / L)`, an amplified pool's shape."""

    def __init__(self, p: float, L: float):
        super().__init__(1e12, 1e12)
        self.p, self.L = p, L

    def chain(self, dx: float) -> float:
        return self.p * self.L * math.tanh(dx / self.L)


def test_a_piece_never_prices_above_the_pools_own_rate_at_zero():
    """Three points on a curve that is flat and then meets a wall give a slope
    at zero above the pool's: rETH/ETH's first piece was 10.5% rich."""
    nodes, arc, nu, _ = setup(2e6, 256.0)
    # The first piece spans 1.2L, where the three-point slope at zero is
    # 9.5% above the pool's.
    client = Amplified(p=1.0, L=2e5 / 1.2)
    _, banks = cryptobank.bank_arcs([arc], nu, nodes, client, Psi=1.28e6)
    bank = banks[(POOL, 2, 1)]
    width = bank[0].cap
    assert width == pytest.approx(1.2 * client.L)
    assert bank[0].a <= client.p * (1 + 1e-6)
    paid = bank[0].a * width - 0.5 * bank[0].B * width * width
    assert abs(paid / client.chain(width) - 1) < 1e-6
