"""Does a Uniswap v3 leg actually execute?

Every other venue in this router is paid with an allowance: approve the pool,
call it, and it pulls.  A v3 pool does the opposite -- it sends the output
first, then calls `uniswapV3SwapCallback` and checks its own balance rose --
so the executor has to move tokens on the word of whoever called it.  That is
the whole of what is new, and it is the part a view-only quote can never
exercise: `QuoterV2` reverts inside its own callback on purpose and reads the
answer out of the revert data, so it proves the arithmetic and nothing about
settlement.

So this runs the swap for real against `contracts/RouteExecutor.vy` and checks
the output the chain hands back, and that the callback refuses a caller that is
not the pool it is currently inside a swap on.
"""

from __future__ import annotations

import pytest

from erouter.core.realize import RealizedLeg, RealizedRoute
from erouter.core.types import ArcKind, Leg
from erouter.dev.executor import deploy, execute, fork

pytestmark = pytest.mark.forked

# USDC/WETH 0.05%, the deepest v3 pool on mainnet, and its two coins in the
# pool's own index order.
POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

CASES = [
    ("USDC -> WETH", 0, 1, USDC, WETH, 100_000 * 10**6),
    ("WETH -> USDC", 1, 0, WETH, USDC, 40 * 10**18),
]


def one_leg_route(i: int, j: int, token_in: str, token_out: str,
                  amount: int) -> RealizedRoute:
    leg = Leg(target=POOL, kind=ArcKind.SWAP_UNIV3, i=i, j=j, n=2,
              src_slot=0, dst_slot=1, bps=0)
    return RealizedRoute(
        legs=[RealizedLeg(leg=leg, kind=ArcKind.SWAP_UNIV3, target=POOL,
                          token_in=token_in, token_out=token_out,
                          amount_in=amount, amount_out=0)],
        slots={token_in: 0, token_out: 1},
        dst_slot=1, src_token=token_in, dst_token=token_out,
        amount_in=amount,
    )


@pytest.fixture(scope="module")
def forked(rpc):
    from erouter.chain import chains as chain_table
    from erouter.dev import config

    chain = chain_table.get("ethereum")
    fork(config.rpc_url(chain.rpc_attr), rpc.block)
    return rpc.block


def quoted(rpc, block: int, i: int, token_in: str, token_out: str,
           amount: int) -> int:
    """`QuoterV2` at the same block -- the number the router believes."""
    from erouter.core.codec import decode, encode_call

    data = encode_call(
        "quoteExactInputSingle((address,address,uint256,uint24,uint160))",
        (token_in, token_out, amount, 500, 0))
    raw = rpc.fetch("eth_call", [{"to": QUOTER_V2, "data": "0x" + data.hex()},
                                 hex(block)])
    return decode(["uint256", "uint160", "uint32", "uint256"],
                  bytes.fromhex(raw[2:]))[0]


@pytest.mark.parametrize("name,i,j,token_in,token_out,amount", CASES,
                         ids=[c[0] for c in CASES])
def test_a_v3_leg_executes_and_pays_what_the_quoter_said(
        rpc, forked, name, i, j, token_in, token_out, amount):
    route = one_leg_route(i, j, token_in, token_out, amount)
    got = execute(route, expect_block=forked)
    assert not got.error, f"{name}: {got.error}"
    assert got.executed_out > 0, f"{name}: the callback paid nothing"

    want = quoted(rpc, forked, i, token_in, token_out, amount)
    # Wei-exact is the claim: same block, same pool, same arithmetic.  The
    # quoter simulates the identical swap, so anything at all here is a
    # settlement difference rather than a rounding one.
    assert got.executed_out == want, (
        f"{name}: executed {got.executed_out:,} against QuoterV2's {want:,}")


def test_the_callback_refuses_anyone_who_is_not_the_pool(forked):
    """It moves this contract's tokens on the caller's word, so the caller is
    the only thing standing between an approval and a stranger."""
    import boa

    executor = deploy()
    with boa.env.prank(boa.env.generate_address()), pytest.raises(Exception) as bad:
        executor.uniswapV3SwapCallback(10**6, -10**6, bytes.fromhex(
            "0" * 24 + USDC[2:]))
    assert "unexpected v3 callback" in str(bad.value)
