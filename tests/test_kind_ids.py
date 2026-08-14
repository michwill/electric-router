"""The contract's leg kinds and `ArcKind` must agree, number for number.

They are two hand-maintained lists of the same thing: `core.types.ArcKind`
decides what a leg *is*, `RouteQuoter.vy` decides what call to make for it, and
the only thing joining them is an integer.  Drift does not fail loudly -- a leg
is simply priced as a different operation, and the quote comes back plausible.

Read from the Vyper source as text rather than by compiling it, so this costs
nothing and runs in the offline suite where it belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from erouter.core.types import ArcKind

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "RouteQuoter.vy"
DECLARATION = re.compile(
    r"^([A-Z0-9_]+): public\(constant\(uint8\)\) = (\d+)", re.MULTILINE)


def declared() -> dict[str, int]:
    text = CONTRACT.read_text(encoding="utf-8")
    found = dict(DECLARATION.findall(text))
    return {name: int(value) for name, value in found.items()
            if not name.startswith("STATUS_")}


def test_every_contract_kind_exists_in_core_with_the_same_number():
    for name, value in declared().items():
        assert name in ArcKind.__members__, f"{name} is priced but not modelled"
        assert int(ArcKind[name]) == value, (
            f"{name}: contract says {value}, core says {int(ArcKind[name])} -- "
            "a leg of this kind would be priced as a different operation"
        )


def test_every_core_kind_the_quoter_must_price_is_declared():
    """`WRAP_NATIVE` and friends need no call, but they still need a number."""
    for kind in ArcKind:
        assert kind.name in declared(), f"{kind.name} is modelled but not priced"


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
