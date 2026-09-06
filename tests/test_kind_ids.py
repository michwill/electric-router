"""The contracts' leg kinds and `ArcKind` must agree, number for number.

Three hand-maintained lists of the same thing: `core.types.ArcKind` decides
what a leg *is*, `RouteQuoter.vy` decides what call to price it with, and
`ElectricRouter.vy` decides what call to execute it with.  The only thing
joining them is an integer.  Drift does not fail loudly -- a leg is simply
priced or executed as a different operation, and the answer comes back
plausible.

Read from the Vyper sources as text rather than by compiling them, so this
costs nothing and runs in the offline suite where it belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from erouter.core.types import OFF_CHAIN_KINDS, ArcKind

#: Every kind the contracts must know about.  `OFF_CHAIN_KINDS`
#: is priced by a model and never sent, so there is nothing for
#: them to agree with -- see `core.types`.
ON_WIRE = [k for k in ArcKind if k not in OFF_CHAIN_KINDS]

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
NAMES = ("RouteQuoter", "ElectricRouter")

#: Both contracts label the block, so the block is what gets read.  Keying on
#: the type would be keying on a coincidence: the widths differ between them --
#: the quoter is deployed at an address that is a function of its initcode, so
#: its `uint8` stays put while the router's kinds are plain `uint256`.
SECTION = "# --- leg kinds"
DECLARATION = re.compile(
    r"^([A-Z0-9_]+): (?:public\()?constant\(uint(?:8|256)\)\)? = (\d+)",
    re.MULTILINE)


def declared(name: str = "RouteQuoter") -> dict[str, int]:
    text = (CONTRACTS / f"{name}.vy").read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.find("\n# ---", start + 1)
    block = text[start : end if end != -1 else len(text)]
    return {key: int(value) for key, value in DECLARATION.findall(block)}


def test_the_kind_section_is_where_the_kinds_are():
    """A guard on the guard: reading the wrong block would pass every test
    below by finding nothing to disagree with."""
    for name in NAMES:
        found = declared(name)
        assert len(found) == len(ON_WIRE), f"{name}: found {sorted(found)}"
        assert not any(k.startswith("MAX_") for k in found), f"{name}: {sorted(found)}"


@pytest.mark.parametrize("contract", NAMES)
def test_every_contract_kind_exists_in_core_with_the_same_number(contract):
    for name, value in declared(contract).items():
        assert name in ArcKind.__members__, f"{name} is in {contract}, not in core"
        assert int(ArcKind[name]) == value, (
            f"{name}: {contract} says {value}, core says {int(ArcKind[name])} -- "
            "a leg of this kind would be handled as a different operation"
        )


@pytest.mark.parametrize("contract", NAMES)
def test_every_core_kind_is_declared(contract):
    """`WRAP_NATIVE` and friends need no call, but they still need a number."""
    for kind in ON_WIRE:
        assert kind.name in declared(contract), (
            f"{kind.name} is modelled and {contract} has never heard of it")


def test_the_encoder_knows_how_to_place_every_kind():
    """`_DERIVE` says which token each side of a leg is; a gap is a refusal."""
    from erouter.core.routecall import _DERIVE

    for kind in ON_WIRE:
        assert kind in _DERIVE, f"{kind.name} cannot be encoded for the router"


def test_the_router_executes_every_kind_it_declares():
    """A declared kind with no branch in `_run` would revert at the last moment."""
    text = (CONTRACTS / "ElectricRouter.vy").read_text(encoding="utf-8")
    body = text[text.index("def _run("):text.index("def execute(")]
    for kind in ON_WIRE:
        assert f"kind == {kind.name}" in body, f"{kind.name} is declared, not executed"


def test_fourteen_stays_reserved():
    """It was `SWAP_UNDERLYING` on an abandoned branch, and `data/facts` still
    records that survey's findings under it.  Reusing the number would make
    those entries read as a live kind."""
    assert 14 not in declared().values()
    assert 14 not in {int(k) for k in ArcKind}


@pytest.mark.parametrize("name", ["LEND_MINT", "LEND_REDEEM"])
def test_the_lending_kinds_are_not_swaps_and_not_merges(name):
    """A merge is symmetric; these are not.  Compound V2 answers "mint is
    paused" and redeems fine, so the two directions must be separable."""
    kind = ArcKind[name]
    assert kind.is_lending
    assert not kind.is_swap


def test_the_ported_solver_knows_every_kind():
    """`ArcKind` is a fourth hand-maintained list: `rust/src/types.rs`.

    It is not on the wire, so `OFF_CHAIN_KINDS` does not excuse it -- the
    accelerated ballot is handed every arc the reference has, including the
    ones no contract will ever see.  A gap here is not a wrong answer but a
    refusal: `ValueError: no such kind: 17`, which is how `SWAP_UNIV3` was
    found missing after the venue work had already shipped.
    """
    erouter_solve = pytest.importorskip("erouter_solve")
    arcs = erouter_solve.Arcs()
    for kind in ArcKind:
        arcs.add(f"a{int(kind)}", "0x" + "11" * 20, int(kind), 0, 1, 2,
                 "0x" + "22" * 20, "0x" + "33" * 20, 0, 1,
                 1.0, 1.0, float("inf"), 1.0, 0.0, 0, 18, 0.0, 1.0)
    assert len(arcs) == len(list(ArcKind))
