"""Which v4 pools this router will touch, decided from the hook's address.

A hook's permissions are the low fourteen bits of its own address -- the
`PoolManager` validates them at creation -- so this is a mask test, not a call.
Getting the mask wrong is the kind of error that would admit a pool whose hook
rewrites the swap, so the bits are pinned against
`Uniswap/v4-core/src/libraries/Hooks.sol` one by one.
"""

from __future__ import annotations

import pytest

from erouter.venues.univ4 import (
    DYNAMIC_FEE,
    FLAGS,
    PoolKey,
    permissions,
    routable,
    tier,
)

NONE = "0x" + "00" * 20


def hook(bits: int, prefix: str = "ab") -> str:
    """An address carrying exactly `bits`, with a realistic high half."""
    return "0x" + (prefix * 10)[:36] + f"{bits:04x}"


def test_the_flag_values_match_the_v4_core_library():
    """Quoted verbatim from `Hooks.sol`; a shifted bit admits the wrong pools."""
    assert FLAGS == {
        "beforeInitialize": 1 << 13, "afterInitialize": 1 << 12,
        "beforeAddLiquidity": 1 << 11, "afterAddLiquidity": 1 << 10,
        "beforeRemoveLiquidity": 1 << 9, "afterRemoveLiquidity": 1 << 8,
        "beforeSwap": 1 << 7, "afterSwap": 1 << 6,
        "beforeDonate": 1 << 5, "afterDonate": 1 << 4,
        "beforeSwapReturnsDelta": 1 << 3, "afterSwapReturnsDelta": 1 << 2,
        "afterAddLiquidityReturnsDelta": 1 << 1,
        "afterRemoveLiquidityReturnsDelta": 1 << 0,
    }
    assert DYNAMIC_FEE == 0x800000, "0b1000... , above MAX_LP_FEE of 1,000,000"


def test_only_the_low_fourteen_bits_are_permissions():
    """The rest of the address is an address.  Reading it as flags would
    classify almost every hook as doing almost everything."""
    assert permissions("0x" + "ff" * 20) == set(FLAGS)
    assert permissions("0x" + "ff" * 18 + "0000") == set()


def test_no_hook_is_tier_zero():
    assert tier(NONE, 3000) == 0
    assert routable(NONE, 3000)


@pytest.mark.parametrize("name", [
    "beforeInitialize", "afterInitialize", "beforeAddLiquidity",
    "afterAddLiquidity", "beforeRemoveLiquidity", "afterRemoveLiquidity",
    "beforeDonate", "afterDonate",
    "afterAddLiquidityReturnsDelta", "afterRemoveLiquidityReturnsDelta",
])
def test_a_callback_off_the_swap_path_is_tier_one(name):
    """Ten of the fourteen cannot touch a swap however elaborate they are.

    This is the observation the whole venue rests on, and it is worth one
    assertion per bit: a liquidity or donate or initialize callback is simply
    not called when somebody swaps.  Measured, tier 1 is also the *best* tier --
    729 pools usable at $10k against tier 0's 373, at a median 68 bp against
    9,992 -- because a hook means somebody funded the pool on purpose.
    """
    assert tier(hook(FLAGS[name]), 3000) == 1
    assert routable(hook(FLAGS[name]), 3000)


@pytest.mark.parametrize("name", ["beforeSwap", "afterSwap"])
def test_a_hook_on_the_swap_path_is_tier_two_and_refused(name):
    """It cannot alter the amounts without a returns-delta bit, so the
    arithmetic would be ours -- but there is nothing there: two usable pools of
    1,893, and 84% revert on a plain $10,000 quote."""
    assert tier(hook(FLAGS[name]), 3000) == 2
    assert not routable(hook(FLAGS[name]), 3000)


@pytest.mark.parametrize("name",
                         ["beforeSwapReturnsDelta", "afterSwapReturnsDelta"])
def test_a_hook_that_returns_a_delta_is_tier_three(name):
    """It replaces part of the swap, so no reading of the pool predicts it.

    1,413 of these quote *better than spot*, the worst by 2.1e49 bp -- the same
    shape that produced `eps = -5.4e7` on an unpriced node and emptied the
    relaxation into it.
    """
    assert tier(hook(FLAGS[name]), 3000) == 3
    assert not routable(hook(FLAGS[name]), 3000)


def test_a_dynamic_fee_is_tier_three_whatever_the_hook_is():
    """The hook sets the fee per swap, so the `a` of an arc is not knowable.

    Even with no hook at all: the flag is in the `PoolKey`'s fee field, and
    `OVERRIDE_FEE_FLAG` only works on a dynamic pool -- which is exactly what
    makes a *static* fee safe.
    """
    assert tier(NONE, DYNAMIC_FEE) == 3
    assert tier(hook(FLAGS["beforeAddLiquidity"]), DYNAMIC_FEE) == 3
    assert not routable(NONE, DYNAMIC_FEE)


def test_returns_delta_outranks_a_harmless_bit_beside_it():
    """A hook is classified by the worst thing it may do, not the first."""
    bits = (FLAGS["beforeAddLiquidity"] | FLAGS["afterInitialize"]
            | FLAGS["afterSwapReturnsDelta"])
    assert tier(hook(bits), 3000) == 3


def test_a_pool_key_carries_its_own_verdict():
    row = ["0xAAA", "0xBbB", 3000, 60, hook(FLAGS["beforeDonate"]), 1e6]
    key = PoolKey.from_row(row)
    assert (key.currency0, key.currency1) == ("0xaaa", "0xbbb")
    assert key.tier == 1 and key.routable
    assert key.as_tuple()[2:4] == (3000, 60), "the ABI ordering, for encoding"


def test_two_tick_bank_venues_chain_their_collapse():
    """v3 and v4 both fold banks into one leg, and both must run.

    `collapse` takes `(arcs, psi, nu, nodes)` and returns `(arcs, psi)`, so a
    chain that splats the first's output into the second loses `nu` and
    `nodes` -- and it would only ever show with both venues on at once.
    """
    import ast
    import inspect

    from erouter.chain import session as chain_session

    src = inspect.getsource(chain_session.RouterSession._seams) \
        if hasattr(chain_session.RouterSession, "_seams") else \
        inspect.getsource(chain_session)
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
    both = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "both"), None)
    assert both is not None, "no collapse chain; this test needs rewriting"
    body = ast.unparse(both)
    assert "_a(arcs, psi, nu, nodes)" in body
    assert "_b(arcs, psi, nu, nodes)" in body, (
        "the second collapse is not given nu and nodes")
