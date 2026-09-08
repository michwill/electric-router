"""Holding a chain's v2 pairs across a session, and the filters that decide.

The failures worth testing here are the quiet ones: a venue that loads and
contributes nothing looks identical to a venue that is off, which is how
`--univ3` shipped with its arcs never reaching the graph.
"""

from __future__ import annotations

import json

import pytest

from erouter.venues import univ2
from erouter.venues.univ2_session import Univ2, census_path

WETH = "0x" + "ee" * 20
USDC = "0x" + "cc" * 20
LOOSE = "0x" + "dd" * 20
PAIR = "0x" + "11" * 20
OTHER = "0x" + "22" * 20


class Nodes:
    """A node map that prices WETH and USDC and has never heard of LOOSE."""

    def __init__(self, merged=False):
        self._merged = merged

    def has(self, token):
        return token in (WETH, USDC)

    def node(self, token):
        if self._merged:
            return 0
        return {WETH: 0, USDC: 1}.get(token, -1)

    def decimals(self, token):
        return 18 if token == WETH else 6

    def rate(self, _token):
        return 1.0


class Transport:
    """Answers `getReserves()` for the pairs it was given, and nothing else."""

    def __init__(self, reserves):
        self.reserves = reserves
        self.calls = 0

    def fetch_multi(self, requests, concurrent=False):
        out = []
        for _method, params in requests:
            self.calls += 1
            pool = params[0]["to"].lower()
            got = self.reserves.get(pool)
            if got is None:
                out.append(None)
                continue
            out.append("0x" + got[0].to_bytes(32, "big").hex()
                       + got[1].to_bytes(32, "big").hex()
                       + (0).to_bytes(32, "big").hex())
        return out


def census(tvl=1_000_000.0, token1=USDC):
    return {PAIR: [WETH, token1, 30, tvl]}


def test_a_chain_with_no_census_is_not_an_error(tmp_path):
    assert Univ2.load(tmp_path, "ethereum") is None


def test_an_empty_census_is_refused_rather_than_loaded_empty(tmp_path):
    """A venue that loads and holds nothing is the failure that hides.

    Every flag and boot line says it is on while its arcs never reach the
    graph, which is exactly how `--univ3` shipped.
    """
    path = census_path(tmp_path, "ethereum")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert Univ2.load(tmp_path, "ethereum") is None


def test_a_census_is_read_from_where_the_builder_writes_it(tmp_path):
    path = census_path(tmp_path, "ethereum")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(census()))
    got = Univ2.load(tmp_path, "ethereum")
    assert got is not None and len(got.census) == 1


def test_a_pair_under_the_floor_is_not_read():
    venue = Univ2(census(tvl=5_000.0), floor_usd=10_000.0)
    assert venue.wanted(Nodes()) == {}
    assert venue.considered == (0, 0, 0)


def test_a_pair_the_frame_cannot_price_is_not_read():
    """Above the floor and still no arc: the node map has never seen LOOSE."""
    venue = Univ2(census(token1=LOOSE))
    assert venue.wanted(Nodes()) == {}
    assert venue.considered == (1, 0, 0), "above the floor, not priceable"


def test_a_pair_whose_coins_share_a_node_is_not_read():
    venue = Univ2(census())
    assert venue.wanted(Nodes(merged=True)) == {}
    assert venue.considered == (1, 1, 0), "priceable, then merged away"


def test_refresh_reads_once_per_pair_and_builds_both_directions():
    """One `eth_call` a pair is the whole read -- no ticks, no bitmap."""
    venue = Univ2(census())
    transport = Transport({PAIR: (1_000 * 10**18, 2_500_000 * 10**6)})
    assert venue.refresh(transport, Nodes(), block=100) == 1
    assert transport.calls == 1
    assert len(venue.arcs) == 2
    assert {a.venue for a in venue.arcs} == {"uniswap v2"}
    assert venue.block == 100


def test_a_pair_that_does_not_answer_leaves_no_arc():
    venue = Univ2({**census(), OTHER: [WETH, USDC, 30, 1e6]})
    transport = Transport({PAIR: (1_000 * 10**18, 2_500_000 * 10**6)})
    assert venue.refresh(transport, Nodes(), block=1) == 1
    assert {a.pool for a in venue.arcs} == {PAIR}


def test_an_empty_pair_is_read_and_then_dropped():
    """`live` is false, so it never becomes an arc priced off a rate nobody
    can trade at."""
    venue = Univ2(census())
    assert venue.refresh(Transport({PAIR: (0, 2_500_000 * 10**6)}),
                         Nodes(), block=1) == 0
    assert venue.arcs == []


def test_a_leg_prices_from_the_state_already_held():
    """No second read: a constant product's whole state is two integers."""
    venue = Univ2(census())
    venue.refresh(Transport({PAIR: (1_000 * 10**18, 2_500_000 * 10**6)}),
                  Nodes(), block=1)
    got = venue.output(PAIR, True, 10**18)
    assert got == univ2.output(
        univ2.PairState(1_000 * 10**18, 2_500_000 * 10**6, 18, 6, 30),
        True, 10**18)
    assert venue.output("0x" + "99" * 20, True, 10**18) is None


def test_refreshing_twice_replaces_rather_than_accumulates():
    """A session warms per block, and stale arcs are worse than none."""
    venue = Univ2(census())
    transport = Transport({PAIR: (1_000 * 10**18, 2_500_000 * 10**6)})
    venue.refresh(transport, Nodes(), block=1)
    venue.refresh(transport, Nodes(), block=2)
    assert len(venue.arcs) == 2
    assert venue.block == 2


def test_the_fee_from_the_census_reaches_the_arc():
    """A 25 bp fork priced at 30 bp is a one-sided error on every leg."""
    venue = Univ2({PAIR: [WETH, USDC, 25, 1e6]})
    venue.refresh(Transport({PAIR: (1_000 * 10**18, 2_500_000 * 10**6)}),
                  Nodes(), block=1)
    forward = next(a for a in venue.arcs if a.i == 0)
    assert forward.a == pytest.approx(2_500 * 0.9975, rel=1e-9)


def test_the_pair_count_is_capped_and_keeps_the_deepest():
    """Arc count is a cliff, not a cost, so the venue bounds its own share.

    Most of the v2 factory is pairs seeded once and abandoned, so a census floor
    set a little too low does not cost a little -- it multiplies the graph.  A
    graph taken from 450 arcs to 7,486 elsewhere in this router stopped
    converging at all.  What gets dropped is what holds least.
    """
    census = {f"0x{n:040x}": [WETH, USDC, 30, float(n)] for n in range(1, 11)}
    venue = Univ2(census, floor_usd=0.0, max_pairs=3)
    kept = venue.wanted(Nodes())
    assert len(kept) == 3
    assert set(kept) == {f"0x{n:040x}" for n in (10, 9, 8)}
    assert venue.considered == (10, 10, 3), "counted before the cap, kept after"


def test_a_floor_that_already_bites_makes_the_cap_inert():
    """The floor decides; the cap catches a floor that was wrong."""
    census = {f"0x{n:040x}": [WETH, USDC, 30, float(n) * 1_000] for n in range(1, 11)}
    venue = Univ2(census, floor_usd=8_000.0, max_pairs=500)
    assert len(venue.wanted(Nodes())) == 3      # 8k, 9k, 10k
