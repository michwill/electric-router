"""Two addresses that are one market but not one balance.

`discover_aliases` merges tokens that agree to the wei -- same supply, same
balance at every holder tried.  Gnosis EURe does not: v1 and v2 hold
different amounts and it refuses them, which is the right call on the
evidence it has.  They are still interchangeable, because a swap denominated
in one settles against the other with no contract in between.  That is not
readable, so it is declared.

Held apart, `--to EURe` picks the deeper side and the market behind the
other is unreachable: three pools become two plus an orphan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from erouter.core.transport import Answer, Status
from erouter.dev import chains as chain_table
from erouter.dev.wrappers import build_node_map

V1 = "0x" + "e1" * 20
V2 = "0x" + "e2" * 20
OTHER = "0x" + "0c" * 20

GNOSIS_EURE = ("0xcB444e90D8198415266c6a2724b7900fb12FC56E",
               "0x420CA0f9B9b604cE0fd9C18EF134C705e5Fa3430")


@dataclass
class Coin:
    address: str
    decimals: int = 18
    symbol: str = "X"


@dataclass
class Pool:
    address: str
    coins: list
    n_coins: int = 2
    balances: tuple = ()
    tvl_usd: float = 100_000.0
    base_pool: str = ""
    lp_token: str = ""
    name: str = "pool"


class Client:
    """Refuses everything.  Nothing here should need a chain read."""

    def raw(self, calls):
        return [Answer(Status.REVERTED, b"") for _ in calls]


def pools():
    return [
        Pool("0x" + "a1" * 20, [Coin(V1, symbol="EURe"), Coin(OTHER, symbol="USDC")],
             tvl_usd=700_000.0),
        Pool("0x" + "a2" * 20, [Coin(V2, symbol="EURe"), Coin(OTHER, symbol="USDC")],
             tvl_usd=170_000.0),
    ]


def chain_with_duals():
    return replace(chain_table.get("gnosis"), duals=((V1, V2),))


def test_a_declared_dual_becomes_one_node():
    nodes, report = build_node_map(pools(), chain_with_duals(), Client())
    assert nodes.has(V1) and nodes.has(V2)
    assert nodes.node(V1) == nodes.node(V2), (
        "declared duals stayed two nodes, so one side's market is unreachable")
    assert (V2.lower(), V1.lower()) in [(a.lower(), b.lower())
                                        for a, b in report.aliases]


def test_the_deeper_side_is_canonical():
    """Legs stay denominated in the token most pools already hold."""
    nodes, _ = build_node_map(pools(), chain_with_duals(), Client())
    assert nodes.canonical_of[nodes.node(V1)].lower() == V1.lower()


def test_without_the_declaration_they_stay_apart():
    """The point of declaring: no read would have merged these."""
    nodes, report = build_node_map(pools(), replace(chain_table.get("gnosis"),
                                                    duals=()), Client())
    assert nodes.node(V1) != nodes.node(V2)
    assert not report.aliases


def test_gnosis_declares_the_eure_pair():
    """The addresses themselves, so a table edit cannot quietly drop them."""
    declared = {(a.lower(), b.lower())
                for a, b in chain_table.get("gnosis").duals}
    assert (GNOSIS_EURE[0].lower(), GNOSIS_EURE[1].lower()) in declared
