"""A vault facts says we cannot mint must not become a mint arc.

srRoyUSDC quotes perfectly well and reverts on execution: it runs a
`WhitelistUserDepositHook` that refuses a depositor not on its list, and no view
says so -- `maxDeposit` answers 2**256-1 whoever asks.  `facts` had already
measured `mint: False` for it, and the arc was built anyway, because this
builder gated on views alone while its sibling `build_lending_arcs` gated on the
measured verdict.  Three such vaults were live when this was written: srRoyUSDC,
USD3 and loAZND.
"""

from __future__ import annotations

import contextlib
from typing import ClassVar

from erouter.chain.wrappers import build_stake_arcs


class Facts:
    def __init__(self, wrappers: dict) -> None:
        self.wrappers = wrappers


class Nodes:
    """Enough of a `NodeMap` for the vault loop to reach its first call.

    `node_of` is empty so discovery finds nothing and the hand list is the
    whole vault set -- the gate is what is under test, not `mintable_vaults`.
    """

    node_of: ClassVar[dict] = {}

    def has(self, address: str) -> bool:
        return True

    def decimals(self, address: str) -> int:
        return 18


class Chain:
    chain_id = 1
    stake_arcs: tuple = ()

    def __init__(self, vaults) -> None:
        self.oneway_vaults = vaults


class Client:
    """Records what was asked about, and answers nothing."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def raw(self, calls):
        self.asked.extend(call.to.lower() for call in calls)
        raise AssertionError("no arc should get this far in these tests")


VAULT = "0x" + "a1" * 20
OTHER = "0x" + "a2" * 20


def _asked(vaults, facts) -> set[str]:
    client = Client()
    with contextlib.suppress(AssertionError):
        build_stake_arcs(Nodes(), Chain(vaults), client, facts)
    return set(client.asked)


def test_a_refused_mint_is_never_asked_about():
    facts = Facts({VAULT: {"mint": False, "redeem": True}})
    assert VAULT.lower() not in _asked([VAULT], facts)


def test_an_untested_mint_is_still_allowed():
    # `None` is "the probe could not fund it", not "the vault said no".
    # Dropping those would delete good arcs for want of a holder.
    facts = Facts({VAULT: {"mint": None, "redeem": True}})
    assert VAULT.lower() in _asked([VAULT], facts)


def test_a_working_mint_is_untouched():
    facts = Facts({VAULT: {"mint": True, "redeem": True}})
    assert VAULT.lower() in _asked([VAULT], facts)


def test_only_the_refused_vault_is_dropped():
    facts = Facts({VAULT: {"mint": False}, OTHER: {"mint": True}})
    asked = _asked([VAULT, OTHER], facts)
    assert VAULT.lower() not in asked
    assert OTHER.lower() in asked


def test_no_facts_at_all_changes_nothing():
    # Every caller passes facts today, but the parameter defaults to None and a
    # missing cache must not silently empty the arc set.
    assert VAULT.lower() in _asked([VAULT], None)
