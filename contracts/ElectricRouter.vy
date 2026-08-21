# pragma version 0.4.3
"""
@title ElectricRouter
@notice Executes a routing DAG: a list of pools, and one packed word per pool.
@dev Three things separate this from `RouteExecutor`, which measures routes on a
     fork and is not a router:

     1. EVERY LEG CARRIES ITS OWN MINIMUM RATE.  A single end-to-end bound lets
        a route lose everything in one leg and make it back in another, which is
        exactly the shape a sandwich takes.  Bounding each leg prices the attack
        per pool instead of per trade.
     2. FRACTIONS ARE OF WHAT IS LEFT, not of the original.  A 50/50 split is
        `50%` then `100%`, so the second leg cannot be starved by whatever the
        first one really returned.  `balanceOf` is the source of truth, so a
        rebasing or fee-on-transfer token is measured rather than assumed.
     3. AN ERC20'S REPLY IS CHECKED, NOT DISCARDED.  `default_return_value`
        accepts the tokens that answer with nothing -- USDT and its
        descendants -- while still failing on the ones that answer `False`
        rather than reverting.  Ignoring the word entirely, which is what a
        bare `raw_call` does, treats those two as the same.
     4. THE CALLER NAMES POOLS, NOT TOKENS.  `i` and `j` index the pool's own
        `coins`, and the router reads them, so a caller cannot mis-state which
        token a rate applies to.  Tokens that are not pool coins -- LP tokens,
        vault assets, lending underlyings -- have no cheap getter on every
        deployment, so those legs name theirs in `tokens` and point at it with
        `in_ref` / `out_ref`.

     PACKING, low bit first, one word per pool:

         0..59     frac      1e18 == 100% of the current balance
         60..187   min_rate  1e18-based, out per in, in raw token units
         188..191  i
         192..195  j
         196..199  n_coins   only `add_liquidity` needs it
         200..204  kind      matches `erouter.core.types.ArcKind`
         205..209  in_ref    0 = read it from the pool, else tokens[ref - 1]
         210..214  out_ref   same
         215..255  must be zero

     `min_rate` is a rate between raw units, so it carries the decimal
     difference: 128 bits covers everything from 1e-18 to 3.4e20 out per in.
     A pair outside that band -- an 18-decimal dust token against an 8-decimal
     expensive one -- can only be bounded by its neighbours and `min_out`.
"""

from ethereum.ercs import IERC20

MAX_LEGS: public(constant(uint256)) = 32
MAX_TOKENS: public(constant(uint256)) = 31
MAX_COINS: public(constant(uint256)) = 8

ONE: constant(uint256) = 10**18

# The sentinel Curve uses for native ETH in `coins()`.
NATIVE: public(constant(address)) = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE

# --- leg kinds (must match erouter.core.types.ArcKind) ---------------------

SWAP_STABLE: constant(uint8) = 0
SWAP_CRYPTO: constant(uint8) = 1
DEPOSIT_FIXED: constant(uint8) = 2
DEPOSIT_DYN: constant(uint8) = 3
DEPOSIT_FIXED_NOFLAG: constant(uint8) = 4
WITHDRAW_STABLE: constant(uint8) = 5
WITHDRAW_CRYPTO: constant(uint8) = 6
ERC4626_DEPOSIT: constant(uint8) = 7
ERC4626_REDEEM: constant(uint8) = 8
WRAP_NATIVE: constant(uint8) = 9
UNWRAP_NATIVE: constant(uint8) = 10
WSTETH_UNWRAP: constant(uint8) = 11
WSTETH_WRAP: constant(uint8) = 12
STAKE_NATIVE: constant(uint8) = 13
LEND_MINT: constant(uint8) = 15
LEND_REDEEM: constant(uint8) = 16


struct Step:
    pool: address
    token_in: address
    token_out: address
    kind: uint8
    i: uint8
    j: uint8
    n: uint8
    frac: uint256
    min_rate: uint256


event Routed:
    sender: indexed(address)
    receiver: indexed(address)
    token_in: indexed(address)
    token_out: address
    amount_in: uint256
    amount_out: uint256


# --------------------------------------------------------------- reads


@internal
@view
def _ask(target: address, data: Bytes[68]) -> (bool, Bytes[32]):
    """One staticcall that never reverts.  Empty returndata is not a value.

    A Curve pool answers an unimplemented function with empty data rather than
    a revert, and decoding `0x` gives zero -- which as an address is a token
    nobody holds and as a balance is a leg that quietly does nothing.
    """
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = raw_call(
        target, data, max_outsize=32, is_static_call=True, revert_on_failure=False
    )
    if not ok or len(out) != 32:
        return False, b""
    return True, out


@internal
@view
def _held(token: address) -> uint256:
    if token == NATIVE:
        return self.balance
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = self._ask(
        token, concat(method_id("balanceOf(address)"), abi_encode(self))
    )
    assert ok, "balanceOf"
    return abi_decode(out, uint256)


@internal
@view
def _coin(pool: address, index: uint8) -> address:
    """`coins(index)`, in whichever spelling the pool was compiled with."""
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = self._ask(
        pool,
        concat(method_id("coins(uint256)"), abi_encode(convert(index, uint256))),
    )
    if not ok:
        ok, out = self._ask(
            pool,
            concat(method_id("coins(int128)"), abi_encode(convert(index, int128))),
        )
    assert ok, "coins"
    return abi_decode(out, address)


@internal
@view
def _getter(target: address, data: Bytes[68]) -> address:
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = self._ask(target, data)
    if not ok:
        return empty(address)
    return abi_decode(out, address)


@internal
@view
def _lp_token(pool: address) -> address:
    """The pool's own share token, which is the pool itself on the ng designs.

    Falling back to the pool is gated on it answering `totalSupply()`: fourteen
    mainnet pools -- 3pool among them -- keep their LP token in a separate
    contract and expose no getter for it at all, and quietly treating those as
    their own share token would approve and count the wrong address.  They must
    name it in `tokens` instead.
    """
    found: address = self._getter(pool, method_id("lp_token()"))
    if found != empty(address):
        return found
    found = self._getter(pool, method_id("token()"))
    if found != empty(address):
        return found
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = self._ask(pool, method_id("totalSupply()"))
    assert ok, "lp token unknown -- name it in tokens"
    return pool


@internal
@view
def _underlying(target: address) -> address:
    """What a wrapper is a claim on: ERC4626, Compound, Aave, wstETH."""
    found: address = self._getter(target, method_id("asset()"))
    if found == empty(address):
        found = self._getter(target, method_id("underlying()"))
    if found == empty(address):
        found = self._getter(target, method_id("UNDERLYING_ASSET_ADDRESS()"))
    if found == empty(address):
        found = self._getter(target, method_id("stETH()"))
    assert found != empty(address), "underlying unknown -- name it in tokens"
    return found


# --------------------------------------------------------------- unpacking


@internal
@view
def _unpack(
    pool: address, word: uint256, tokens: DynArray[address, MAX_TOKENS]
) -> Step:
    assert word >> 215 == 0, "reserved bits set"
    kind: uint8 = convert((word >> 200) & 31, uint8)
    i: uint8 = convert((word >> 188) & 15, uint8)
    j: uint8 = convert((word >> 192) & 15, uint8)
    in_ref: uint256 = (word >> 205) & 31
    out_ref: uint256 = (word >> 210) & 31

    step: Step = Step(
        pool=pool,
        token_in=empty(address),
        token_out=empty(address),
        kind=kind,
        i=i,
        j=j,
        n=convert((word >> 196) & 15, uint8),
        frac=word & (2**60 - 1),
        min_rate=(word >> 60) & (2**128 - 1),
    )
    assert step.frac > 0 and step.frac <= ONE, "frac out of range"

    if in_ref > 0:
        assert in_ref <= len(tokens), "in_ref off the token list"
        step.token_in = tokens[in_ref - 1]
    else:
        step.token_in = self._derive(pool, kind, i, True)

    if out_ref > 0:
        assert out_ref <= len(tokens), "out_ref off the token list"
        step.token_out = tokens[out_ref - 1]
    else:
        step.token_out = self._derive(pool, kind, j, False)

    assert step.token_in != step.token_out, "leg does not move between tokens"
    return step


@internal
@view
def _derive(pool: address, kind: uint8, index: uint8, entering: bool) -> address:
    """Which token a leg spends, or produces, read off the pool itself."""
    if kind == SWAP_STABLE or kind == SWAP_CRYPTO:
        return self._coin(pool, index)
    if kind == DEPOSIT_FIXED or kind == DEPOSIT_DYN or kind == DEPOSIT_FIXED_NOFLAG:
        return self._coin(pool, index) if entering else self._lp_token(pool)
    if kind == WITHDRAW_STABLE or kind == WITHDRAW_CRYPTO:
        return self._lp_token(pool) if entering else self._coin(pool, index)
    if kind == ERC4626_DEPOSIT or kind == WSTETH_WRAP or kind == LEND_MINT:
        return self._underlying(pool) if entering else pool
    if kind == ERC4626_REDEEM or kind == WSTETH_UNWRAP or kind == LEND_REDEEM:
        return pool if entering else self._underlying(pool)
    if kind == WRAP_NATIVE or kind == STAKE_NATIVE:
        return NATIVE if entering else pool
    if kind == UNWRAP_NATIVE:
        return pool if entering else NATIVE
    raise "unknown leg kind"


# --------------------------------------------------------------- ERC20


@internal
def _allow(token: address, spender: address):
    """Infinite allowance, set once and only when it is not already there.

    Read first, so the reset through zero happens only when there is something
    to reset -- USDT refuses a non-zero to non-zero change, and every other
    token is charged for the extra write.
    """
    if token == NATIVE:
        return
    held: uint256 = staticcall IERC20(token).allowance(self, spender)
    if held == max_value(uint256):
        return
    if held != 0:
        assert extcall IERC20(token).approve(spender, 0, default_return_value=True), (
            "approve failed")
    assert extcall IERC20(token).approve(
        spender, max_value(uint256), default_return_value=True), "approve failed"


@internal
def _pull(token: address, owner: address, amount: uint256):
    assert extcall IERC20(token).transferFrom(
        owner, self, amount, default_return_value=True), "transferFrom failed"


@internal
def _pay(token: address, to: address, amount: uint256):
    if amount == 0:
        return
    if token == NATIVE:
        # Full gas rather than a stipend: `execute` holds the reentrancy lock,
        # so the only thing this buys an unusual receiver is the ability to be
        # paid at all.
        raw_call(to, b"", value=amount)
        return
    assert extcall IERC20(token).transfer(
        to, amount, default_return_value=True), "transfer failed"


# --------------------------------------------------------------- calldata


@internal
@pure
def _amounts(i: uint8, n: uint8, dx: uint256) -> Bytes[256]:
    """ABI body for a single-sided `uint256[n]`: zeros except slot `i`."""
    words: uint256[MAX_COINS] = empty(uint256[MAX_COINS])
    words[convert(i, uint256)] = dx
    return slice(abi_encode(words), 0, 32 * convert(n, uint256))


@internal
@pure
def _add_liquidity(n: uint8) -> Bytes[4]:
    """`N` is part of the signature, so there is one selector per width."""
    if n == 2:
        return method_id("add_liquidity(uint256[2],uint256)")
    if n == 3:
        return method_id("add_liquidity(uint256[3],uint256)")
    if n == 4:
        return method_id("add_liquidity(uint256[4],uint256)")
    if n == 5:
        return method_id("add_liquidity(uint256[5],uint256)")
    if n == 6:
        return method_id("add_liquidity(uint256[6],uint256)")
    if n == 7:
        return method_id("add_liquidity(uint256[7],uint256)")
    if n == 8:
        return method_id("add_liquidity(uint256[8],uint256)")
    raise "coin count outside 2..8"


# --------------------------------------------------------------- one leg


@internal
def _run(step: Step, dx: uint256) -> uint256:
    """Execute one leg and return what actually arrived.

    Every pool is asked for `min_dy = 0` and the bound is applied here instead,
    on the balance that really moved.  A legacy pool returns nothing from
    `exchange`, an ng one returns `dy`, and a fee-on-transfer token makes both
    of them a lie -- the delta is the only figure true for all three, and it is
    also what the next leg has to spend.
    """
    kind: uint8 = step.kind
    before: uint256 = self._held(step.token_out)

    if kind == SWAP_STABLE:
        raw_call(
            step.pool,
            concat(
                method_id("exchange(int128,int128,uint256,uint256)"),
                abi_encode(
                    convert(step.i, int128), convert(step.j, int128), dx, empty(uint256)
                ),
            ),
        )

    elif kind == SWAP_CRYPTO:
        ok: bool = False
        out: Bytes[32] = b""
        ok, out = raw_call(
            step.pool,
            concat(
                method_id("exchange(uint256,uint256,uint256,uint256)"),
                abi_encode(
                    convert(step.i, uint256),
                    convert(step.j, uint256),
                    dx,
                    empty(uint256),
                ),
            ),
            max_outsize=32,
            revert_on_failure=False,
        )
        # The older crypto pools take a trailing `use_eth` and have no
        # four-argument entry point.  Some of them also swallow an unknown
        # selector in `__default__`, so success is not the discriminator --
        # whether anything moved is.
        if not ok or self._held(step.token_out) == before:
            raw_call(
                step.pool,
                concat(
                    method_id("exchange(uint256,uint256,uint256,uint256,bool)"),
                    abi_encode(
                        convert(step.i, uint256),
                        convert(step.j, uint256),
                        dx,
                        empty(uint256),
                        False,
                    ),
                ),
            )

    elif kind == DEPOSIT_DYN:
        raw_call(
            step.pool,
            concat(
                method_id("add_liquidity(uint256[],uint256)"),
                abi_encode(convert(64, uint256)),
                abi_encode(empty(uint256)),
                abi_encode(convert(step.n, uint256)),
                self._amounts(step.i, step.n, dx),
            ),
        )

    elif kind == DEPOSIT_FIXED or kind == DEPOSIT_FIXED_NOFLAG:
        raw_call(
            step.pool,
            concat(
                self._add_liquidity(step.n),
                self._amounts(step.i, step.n, dx),
                abi_encode(empty(uint256)),
            ),
        )

    elif kind == WITHDRAW_STABLE:
        raw_call(
            step.pool,
            concat(
                method_id("remove_liquidity_one_coin(uint256,int128,uint256)"),
                abi_encode(dx, convert(step.j, int128), empty(uint256)),
            ),
        )

    elif kind == WITHDRAW_CRYPTO:
        raw_call(
            step.pool,
            concat(
                method_id("remove_liquidity_one_coin(uint256,uint256,uint256)"),
                abi_encode(dx, convert(step.j, uint256), empty(uint256)),
            ),
        )

    elif kind == ERC4626_DEPOSIT:
        raw_call(
            step.pool,
            concat(method_id("deposit(uint256,address)"), abi_encode(dx, self)),
        )

    elif kind == ERC4626_REDEEM:
        raw_call(
            step.pool,
            concat(
                method_id("redeem(uint256,address,address)"), abi_encode(dx, self, self)
            ),
        )

    elif kind == WSTETH_WRAP:
        raw_call(step.pool, concat(method_id("wrap(uint256)"), abi_encode(dx)))

    elif kind == WSTETH_UNWRAP:
        raw_call(step.pool, concat(method_id("unwrap(uint256)"), abi_encode(dx)))

    elif kind == LEND_MINT:
        raw_call(step.pool, concat(method_id("mint(uint256)"), abi_encode(dx)))

    elif kind == LEND_REDEEM:
        raw_call(step.pool, concat(method_id("redeem(uint256)"), abi_encode(dx)))

    elif kind == STAKE_NATIVE:
        raw_call(
            step.pool,
            concat(method_id("submit(address)"), abi_encode(empty(address))),
            value=dx,
        )

    elif kind == WRAP_NATIVE:
        if step.token_in == NATIVE:
            raw_call(step.pool, method_id("deposit()"), value=dx)
        else:
            # A 1:1 token adapter wearing the same kind: gnosis converts USDC.e
            # to USDC through one, and it takes `deposit(uint256)` and no value.
            raw_call(step.pool, concat(method_id("deposit(uint256)"), abi_encode(dx)))

    elif kind == UNWRAP_NATIVE:
        raw_call(step.pool, concat(method_id("withdraw(uint256)"), abi_encode(dx)))

    else:
        raise "unknown leg kind"

    return self._held(step.token_out) - before


# --------------------------------------------------------------- external


@external
@payable
@nonreentrant
def execute(
    amount_in: uint256,
    pools: DynArray[address, MAX_LEGS],
    params: DynArray[uint256, MAX_LEGS],
    set_approvals: bool,
    tokens: DynArray[address, MAX_TOKENS] = [],
    receiver: address = msg.sender,
    min_out: uint256 = 0,
) -> uint256:
    """Route `amount_in` through `pools` and pay the proceeds to `receiver`.

    Returns what the route itself produced of the last leg's output token.
    Anything the router was already holding is handed over too but not counted,
    so the answer measures this trade and not the contract's history.
    """
    assert len(pools) == len(params), "one param word per pool"
    assert len(pools) > 0, "empty route"
    assert amount_in > 0, "nothing to route"
    # Not the router itself: it sweeps at the end, so paying itself would
    # leave the proceeds for whoever routes that token next.
    assert receiver != empty(address) and receiver != self, "bad receiver"

    # Unpacked once, up front: every token a leg does not name is read off its
    # pool, and paying for that twice is the whole cost of a short route.
    steps: DynArray[Step, MAX_LEGS] = []
    for k: uint256 in range(MAX_LEGS):
        if k >= len(pools):
            break
        steps.append(self._unpack(pools[k], params[k], tokens))

    first: Step = steps[0]
    last: Step = steps[len(steps) - 1]

    if first.token_in == NATIVE:
        assert msg.value == amount_in, "native input needs msg.value"
    else:
        assert msg.value == 0, "route does not spend native"
        self._pull(first.token_in, msg.sender, amount_in)

    opening: uint256 = self._held(last.token_out)
    seen: DynArray[address, 2 * MAX_LEGS] = []

    for k: uint256 in range(MAX_LEGS):
        if k >= len(pools):
            break
        step: Step = steps[k]
        seen.append(step.token_in)
        seen.append(step.token_out)

        dx: uint256 = self._held(step.token_in) * step.frac // ONE
        if dx == 0:
            continue
        if set_approvals:
            self._allow(step.token_in, step.pool)

        dy: uint256 = self._run(step, dx)
        assert dy > 0, "leg produced nothing"
        assert dy >= dx * step.min_rate // ONE, "leg below its minimum rate"

    produced: uint256 = self._held(last.token_out) - opening
    assert produced >= min_out, "below min_out"

    # Sweep every token the route touched, not just the last one: a branch that
    # rounded down leaves its input behind, and keeping it would make the
    # router's own balance a silent term in the next caller's answer.  A token
    # listed twice is already empty the second time.
    for k: uint256 in range(2 * MAX_LEGS):
        if k >= len(seen):
            break
        self._pay(seen[k], receiver, self._held(seen[k]))

    log Routed(
        sender=msg.sender,
        receiver=receiver,
        token_in=first.token_in,
        token_out=last.token_out,
        amount_in=amount_in,
        amount_out=produced,
    )
    return produced


@external
@payable
def __default__():
    """WETH pays an unwrap back with a bare call, and so does Curve."""
    pass
