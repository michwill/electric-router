"""The seams a venue needs, and that a chain without one pays nothing for them.

`pipeline.route` gained `collapse`, `audit` and `max_spread` alongside the
`extra_arcs` it already had.  All four default to what a Curve-only universe
wants, which is nothing -- and that is the property worth a test, because it is
what every chain but ethereum relies on.
"""

from __future__ import annotations

import json

from erouter.venues.univ3_session import DEFAULT_FLOOR_USD, SPREAD, Univ3, census_path

KNOWN = {"0x" + "a0" * 20: 0, "0x" + "c0" * 20: 1}


class Nodes:
    """Knows two tokens and nothing else."""

    KNOWN = KNOWN

    def has(self, token):
        return token.lower() in self.KNOWN

    def node(self, token):
        return self.KNOWN[token.lower()]

    def decimals(self, token):
        return 6 if self.KNOWN[token.lower()] == 0 else 18


def venue(rows):
    return Univ3(census=rows)


def test_a_chain_with_no_census_is_not_an_error(tmp_path):
    """Every chain but the one with a file routes exactly as it did."""
    assert Univ3.load(tmp_path, "arbitrum") is None


def test_a_census_is_read_from_where_the_cli_looks(tmp_path):
    path = census_path(tmp_path, "ethereum")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"0x" + "11" * 20: ["0xaa", "0xbb", 500, 1e6]}))
    got = Univ3.load(tmp_path, "ethereum")
    assert got is not None and len(got.census) == 1


def test_a_pool_under_the_floor_is_not_read():
    a, b = "0x" + "a0" * 20, "0x" + "c0" * 20
    rows = {"0x" + "11" * 20: [a, b, 500, DEFAULT_FLOOR_USD - 1],
            "0x" + "22" * 20: [a, b, 500, DEFAULT_FLOOR_USD + 1]}
    got = venue(rows)
    assert list(got.wanted(Nodes())) == ["0x" + "22" * 20]
    assert got.considered == (1, 1, 1)      # above, priceable, asked for


def test_a_pool_the_frame_cannot_price_is_not_read():
    """A token no node knows has no arc to give, and inventing one here would
    put a token in the graph nothing else in the router has seen."""
    a, unknown = "0x" + "a0" * 20, "0x" + "ff" * 20
    rows = {"0x" + "11" * 20: [a, unknown, 500, 1e6]}
    got = venue(rows)
    assert got.wanted(Nodes()) == {}
    assert got.considered == (1, 0, 0)


def test_a_pool_whose_coins_share_a_node_is_not_read():
    """Both sides of one merged node is not a trade."""

    class Merged(Nodes):
        def node(self, token):
            return 0

    a, b = "0x" + "a0" * 20, "0x" + "c0" * 20
    got = venue({"0x" + "11" * 20: [a, b, 500, 1e6]})
    assert got.wanted(Merged()) == {}
    assert got.considered == (1, 1, 0)


def test_an_unread_venue_leaves_the_spread_bound_alone():
    """§9.7's default is right until there are arcs it would misread."""
    got = venue({})
    assert got.max_spread < SPREAD
    got.arcs = [object()]
    assert got.max_spread == SPREAD


def test_the_two_bank_maps_are_not_the_same_map():
    """`collapse` sums raw arcs and the walk prices a `Bank`; holding one and
    unwrapping it at each call site is how they got confused."""
    got = venue({})
    assert got.banks is not got.priced


def test_route_takes_the_seams_and_defaults_them_to_nothing():
    """The signature is the contract every other chain relies on."""
    import inspect

    from erouter.core.pipeline import route

    params = inspect.signature(route).parameters
    assert params["collapse"].default is None
    assert params["audit"].default is None
    assert params["extra_arcs"].default is None
    from erouter.core.graph import PATHOLOGICAL_CONDITION
    assert params["max_spread"].default == PATHOLOGICAL_CONDITION


def test_realize_without_a_collapse_is_the_realize_it_was():
    import inspect

    from erouter.core.realize import realize
    assert inspect.signature(realize).parameters["collapse"].default is None


def test_the_session_actually_holds_the_venue_it_was_given():
    """It took `univ3=` and ignored it for one commit, which is worse than not
    taking it: the flag worked, the arcs never arrived, and the quote looked
    fine.  Cheap to assert, and it would have caught that."""
    import inspect

    from erouter.chain.session import RouterSession

    assert "univ3" in inspect.signature(RouterSession.__init__).parameters
    source = inspect.getsource(RouterSession)
    assert "self.univ3 = univ3" in source, "taken and dropped on the floor"
    assert "self.univ3.refresh(" in source, "held and never read"
    assert "late_arcs" in source, "read and not handed to the router"


def test_route_keeps_the_frame_and_the_graph_apart():
    """`extra_arcs` joins before the reference prices and `late_arcs` after.

    Measured on `crvUSD -> sDOLA` at $2M: through `extra_arcs` a v3 universe
    cost 9.50 bp and used no v3 leg, because 146 pools' worth of tick-arcs
    outvoted every Curve pool in the §4 fit.
    """
    import inspect

    from erouter.core.pipeline import route

    params = inspect.signature(route).parameters
    assert params["late_arcs"].default is None
    assert params["extra_arcs"].default is None


def _arc(id_, tau, sigma, venue_=""):
    from erouter.core.types import ArcKind, PoolArc

    return PoolArc(
        id=id_, pool="0x" + "cc" * 20, kind=ArcKind.SWAP_STABLE, i=0, j=1,
        n_coins=2, token_in="0xin", token_out="0xout", tau=tau, sigma=sigma,
        a=1.0, B=1.0, venue=venue_,
    )


def test_a_late_arc_reaching_an_unpriced_node_is_refused():
    """Being in the node map is not being priced.

    `Univ3.wanted` admits a pool when `nodes.has()` both coins, but §4 fits
    `nu` from the arcs it is given -- so a node no arc reaches keeps the
    default 1.0 and a late arc into it reads as enormous arbitrage.  Measured:
    RSR and XYO are in the map with zero Curve arcs, and a `WETH -> RSR` tick
    arc came out at `eps = -5.4e7`.
    """
    from erouter.core.pipeline import admissible_late_arcs

    frame = [_arc("curve:0", 0, 1), _arc("curve:1", 1, 2)]
    good = _arc("v3:0", 0, 2, "uniswap v3")          # both ends priced
    orphan = _arc("v3:1", 0, 9, "uniswap v3")        # node 9 is in no arc
    fresh, unpriced = admissible_late_arcs(frame, [good, orphan])
    assert [a.id for a in fresh] == ["v3:0"]
    assert unpriced == 1


def test_a_late_arc_already_in_the_frame_is_not_added_twice():
    """The seam is re-entered on refit, and a duplicated arc is a second pool."""
    from erouter.core.pipeline import admissible_late_arcs

    frame = [_arc("curve:0", 0, 1), _arc("v3:0", 0, 1, "uniswap v3")]
    fresh, unpriced = admissible_late_arcs(frame, [_arc("v3:0", 0, 1, "uniswap v3")])
    assert fresh == [] and unpriced == 0
