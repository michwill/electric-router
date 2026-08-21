"""Every ABI signature the contracts spell out by hand.

`raw_call` is unavoidable in both of them -- the quoter has to tell a pool that
does not implement a function from one that reverts, and the router has to try
`coins(uint256)` before `coins(int128)` -- and Vyper has no way to take a
selector off a declared interface, so the signature is a string literal and the
compiler never sees it as anything else.

A typo therefore compiles.  It produces a selector no pool implements, which a
Curve pool answers by falling into `__default__` and returning nothing, which
reads downstream as "this leg produced nothing".  The chain is the only thing
that would notice, and only for the pools a forked run happens to touch.

So the set is pinned here.  Changing a signature means changing this file,
which is the point: it is a two-line diff that makes someone look at the ABI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from erouter.core.codec import parse_type, selector, signature_types
from erouter.core.types import ArcKind

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
LITERAL = re.compile(r'method_id\(\s*"([^"]+)"')

#: What `ElectricRouter` calls.  Grouped as the contract groups them.
ROUTER = {
    # ERC20, and the wrappers that wear its shape
    "totalSupply()",
    # discovery: which spelling, which token
    "coins(uint256)", "coins(int128)",
    "lp_token()", "token()",
    "asset()", "underlying()", "UNDERLYING_ASSET_ADDRESS()", "stETH()",
    # swaps
    "exchange(int128,int128,uint256,uint256)",
    "exchange(uint256,uint256,uint256,uint256)",
    "exchange(uint256,uint256,uint256,uint256,bool)",
    # deposits: N is part of the signature
    "add_liquidity(uint256[],uint256)",
    *(f"add_liquidity(uint256[{n}],uint256)" for n in range(2, 9)),
    # withdrawals
    "remove_liquidity_one_coin(uint256,int128,uint256)",
    "remove_liquidity_one_coin(uint256,uint256,uint256)",
    # wrappers
    "deposit()", "deposit(uint256)", "deposit(uint256,address)",
    "withdraw(uint256)", "redeem(uint256)", "redeem(uint256,address,address)",
    "mint(uint256)", "wrap(uint256)", "unwrap(uint256)", "submit(address)",
}

#: What `RouteQuoter` calls.  Every one is a view; none of them move anything.
QUOTER = {
    "totalSupply()", "virtual_price()",
    "admin_fee()", "ADMIN_FEE()", "xcp_profit()", "xcp_profit_a()",
    "get_dy(int128,int128,uint256)", "get_dy(uint256,uint256,uint256)",
    "calc_token_amount(uint256[],bool)",
    *(f"calc_token_amount(uint256[{n}])" for n in range(2, 9)),
    *(f"calc_token_amount(uint256[{n}],bool)" for n in range(2, 9)),
    "calc_withdraw_one_coin(uint256,int128)",
    "calc_withdraw_one_coin(uint256,uint256)",
    "previewDeposit(uint256)", "previewRedeem(uint256)",
    "getStETHByWstETH(uint256)", "getWstETHByStETH(uint256)",
    "exchangeRateStored()",
}

EXPECTED = {"ElectricRouter": ROUTER, "RouteQuoter": QUOTER}


def spelled(name: str) -> set[str]:
    return set(LITERAL.findall((CONTRACTS / f"{name}.vy").read_text()))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_contract_spells_exactly_what_is_expected(name):
    got, want = spelled(name), EXPECTED[name]
    assert got - want == set(), f"{name} gained: {sorted(got - want)}"
    assert want - got == set(), f"{name} lost: {sorted(want - got)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_signature_is_one_the_abi_can_parse(name):
    """A malformed type is the typo a selector cannot show you."""
    for sig in sorted(spelled(name)):
        assert re.fullmatch(r"[A-Za-z_]\w*\([^()]*\)", sig), f"{sig} is not a signature"
        for arg in signature_types(sig):
            parse_type(arg)                    # raises on anything it cannot read


def test_no_two_signatures_collide():
    """Two spellings sharing a selector would dispatch to one of them."""
    for name, sigs in EXPECTED.items():
        seen: dict[bytes, str] = {}
        for sig in sorted(sigs):
            clash = seen.setdefault(selector(sig), sig)
            assert clash == sig, f"{name}: {sig} and {clash} share a selector"


#: The pairing that has to hold across the two contracts: a leg priced with one
#: index type must be executed with the same one.  Getting this wrong quotes a
#: pool through one ABI and trades it through another, and both calls succeed.
DIALECT = {
    ArcKind.SWAP_STABLE: ("get_dy(int128,int128,uint256)",
                          "exchange(int128,int128,uint256,uint256)"),
    ArcKind.SWAP_CRYPTO: ("get_dy(uint256,uint256,uint256)",
                          "exchange(uint256,uint256,uint256,uint256)"),
    ArcKind.WITHDRAW_STABLE: ("calc_withdraw_one_coin(uint256,int128)",
                              "remove_liquidity_one_coin(uint256,int128,uint256)"),
    ArcKind.WITHDRAW_CRYPTO: ("calc_withdraw_one_coin(uint256,uint256)",
                              "remove_liquidity_one_coin(uint256,uint256,uint256)"),
}


@pytest.mark.parametrize("kind", sorted(DIALECT, key=int))
def test_a_kind_is_priced_and_executed_through_the_same_index_type(kind):
    quoted, executed = DIALECT[kind]
    assert quoted in spelled("RouteQuoter"), f"{kind.name}: quoter lost {quoted}"
    assert executed in spelled("ElectricRouter"), f"{kind.name}: router lost {executed}"
    indexes = ("int128" in quoted, "int128" in executed)
    assert indexes[0] == indexes[1], (
        f"{kind.name} is quoted by {quoted} and executed by {executed} -- one "
        f"takes int128 and the other uint256, so the two disagree about the pool")
