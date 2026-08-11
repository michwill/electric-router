# pragma version 0.4.3
"""
@title RouteQuoter
@notice Walks a routing DAG and quotes it, feeding each leg's output into the next.
@dev Three properties are load-bearing and must survive any edit:

     1. STATELESS, NO CONSTRUCTOR, NO IMMUTABLES.  That is what lets the runtime
        bytecode be injected with an `eth_call` state override, so quoting costs
        one round trip and needs no deployment.

     2. EMPTY RETURNDATA IS NOT A VALUE.  A Curve pool that does not implement a
        function returns empty data rather than reverting, and decoding `0x` as a
        uint gives 0 -- which would silently quote every swap at zero.  So every
        call reports one of VALUE / WRONG_ABI / REVERTED, and the caller uses the
        distinction to pick the ABI dialect.  Six mainnet arcs are mis-typed by
        the Curve API today and are only caught this way.

     3. THE LEG ENCODING IS THE ROUTER'S, NOT THE QUOTER'S.  `bps` is a fraction
        of a node's *current* balance, because the amount that actually arrives
        after hop 1 is never exactly what was modelled.  A later phase turns this
        same struct into real execution by swapping the view calls for their
        state-changing twins; nothing about the format is quote-specific.
"""

# --- limits ---------------------------------------------------------------

MAX_PROBES: public(constant(uint256)) = 600
MAX_LEGS: public(constant(uint256)) = 32
MAX_ALL_LEGS: public(constant(uint256)) = 320
MAX_ROUTES: public(constant(uint256)) = 32
MAX_SLOTS: public(constant(uint256)) = 24
MAX_COINS: public(constant(uint256)) = 8

BPS: constant(uint256) = 10_000
CALL_GAS: constant(uint256) = 2_000_000

# --- result status --------------------------------------------------------

STATUS_VALUE: public(constant(uint8)) = 0
STATUS_WRONG_ABI: public(constant(uint8)) = 1  # succeeded, returned no data
STATUS_REVERTED: public(constant(uint8)) = 2

# --- leg kinds (must match erouter.core.types.ArcKind) ---------------------

SWAP_STABLE: public(constant(uint8)) = 0  # get_dy(int128,int128,uint256)
SWAP_CRYPTO: public(constant(uint8)) = 1  # get_dy(uint256,uint256,uint256)
DEPOSIT_FIXED: public(constant(uint8)) = 2  # calc_token_amount(uint256[N],bool)
DEPOSIT_DYN: public(constant(uint8)) = 3  # calc_token_amount(uint256[],bool)
DEPOSIT_FIXED_NOFLAG: public(constant(uint8)) = 4  # calc_token_amount(uint256[N])
WITHDRAW_STABLE: public(constant(uint8)) = 5  # calc_withdraw_one_coin(uint256,int128)
WITHDRAW_CRYPTO: public(constant(uint8)) = 6  # calc_withdraw_one_coin(uint256,uint256)
ERC4626_DEPOSIT: public(constant(uint8)) = 7  # previewDeposit(uint256)
ERC4626_REDEEM: public(constant(uint8)) = 8  # previewRedeem(uint256)
WRAP_NATIVE: public(constant(uint8)) = 9  # 1:1, no call
UNWRAP_NATIVE: public(constant(uint8)) = 10  # 1:1, no call


struct Res:
    status: uint8
    value: uint256


struct Probe:
    pool: address
    kind: uint8
    i: uint8
    j: uint8
    n: uint8
    dx: uint256


struct Leg:
    target: address
    kind: uint8
    i: uint8
    j: uint8
    n: uint8
    src_slot: uint8
    dst_slot: uint8
    bps: uint16


# --------------------------------------------------------------- internals


@internal
@pure
def _amounts_calldata(i: uint8, n: uint8, dx: uint256) -> Bytes[256]:
    """ABI head for a single-sided `uint256[n]`: zeros except slot `i`."""
    words: uint256[MAX_COINS] = empty(uint256[MAX_COINS])
    words[convert(i, uint256)] = dx
    return slice(abi_encode(words), 0, 32 * convert(n, uint256))


@internal
@view
def _static(target: address, data: Bytes[512]) -> Res:
    """One staticcall.  Never reverts; classifies the outcome instead."""
    ok: bool = False
    out: Bytes[32] = b""
    ok, out = raw_call(
        target,
        data,
        max_outsize=32,
        gas=CALL_GAS,
        is_static_call=True,
        revert_on_failure=False,
    )
    if not ok:
        return Res(status=STATUS_REVERTED, value=0)
    if len(out) < 32:
        # Succeeded but said nothing: the wrong ABI, not a zero quote.
        return Res(status=STATUS_WRONG_ABI, value=0)
    return Res(status=STATUS_VALUE, value=abi_decode(out, uint256))


@internal
@view
def _quote(target: address, kind: uint8, i: uint8, j: uint8, n: uint8, dx: uint256) -> Res:
    if kind == WRAP_NATIVE or kind == UNWRAP_NATIVE:
        # Wrapped natives are 1:1 by construction; no call to make.
        return Res(status=STATUS_VALUE, value=dx)

    if kind == SWAP_STABLE:
        return self._static(
            target,
            concat(
                method_id("get_dy(int128,int128,uint256)"),
                abi_encode(convert(i, int128), convert(j, int128), dx),
            ),
        )

    if kind == SWAP_CRYPTO:
        return self._static(
            target,
            concat(
                method_id("get_dy(uint256,uint256,uint256)"),
                abi_encode(convert(i, uint256), convert(j, uint256), dx),
            ),
        )

    if kind == WITHDRAW_STABLE:
        return self._static(
            target,
            concat(
                method_id("calc_withdraw_one_coin(uint256,int128)"),
                abi_encode(dx, convert(j, int128)),
            ),
        )

    if kind == WITHDRAW_CRYPTO:
        return self._static(
            target,
            concat(
                method_id("calc_withdraw_one_coin(uint256,uint256)"),
                abi_encode(dx, convert(j, uint256)),
            ),
        )

    if kind == ERC4626_DEPOSIT:
        return self._static(
            target, concat(method_id("previewDeposit(uint256)"), abi_encode(dx))
        )

    if kind == ERC4626_REDEEM:
        return self._static(
            target, concat(method_id("previewRedeem(uint256)"), abi_encode(dx))
        )

    if kind == DEPOSIT_DYN:
        # StableSwap-NG takes a DynArray, so the amounts need an offset+length
        # header rather than sitting inline.
        amounts: Bytes[256] = self._amounts_calldata(i, n, dx)
        return self._static(
            target,
            concat(
                method_id("calc_token_amount(uint256[],bool)"),
                abi_encode(convert(64, uint256)),  # offset to the array
                abi_encode(True),  # is_deposit
                abi_encode(convert(n, uint256)),  # length
                amounts,
            ),
        )

    if kind == DEPOSIT_FIXED or kind == DEPOSIT_FIXED_NOFLAG:
        amounts_fixed: Bytes[256] = self._amounts_calldata(i, n, dx)
        selector: Bytes[4] = self._deposit_selector(n, kind == DEPOSIT_FIXED)
        if len(selector) == 0:
            return Res(status=STATUS_REVERTED, value=0)
        if kind == DEPOSIT_FIXED:
            return self._static(target, concat(selector, amounts_fixed, abi_encode(True)))
        return self._static(target, concat(selector, amounts_fixed))

    return Res(status=STATUS_REVERTED, value=0)


@internal
@pure
def _deposit_selector(n: uint8, with_flag: bool) -> Bytes[4]:
    """`calc_token_amount` selectors -- N is part of the signature."""
    if with_flag:
        if n == 2:
            return method_id("calc_token_amount(uint256[2],bool)")
        if n == 3:
            return method_id("calc_token_amount(uint256[3],bool)")
        if n == 4:
            return method_id("calc_token_amount(uint256[4],bool)")
        if n == 5:
            return method_id("calc_token_amount(uint256[5],bool)")
        if n == 6:
            return method_id("calc_token_amount(uint256[6],bool)")
        if n == 7:
            return method_id("calc_token_amount(uint256[7],bool)")
        if n == 8:
            return method_id("calc_token_amount(uint256[8],bool)")
        return b""
    if n == 2:
        return method_id("calc_token_amount(uint256[2])")
    if n == 3:
        return method_id("calc_token_amount(uint256[3])")
    if n == 4:
        return method_id("calc_token_amount(uint256[4])")
    if n == 5:
        return method_id("calc_token_amount(uint256[5])")
    if n == 6:
        return method_id("calc_token_amount(uint256[6])")
    if n == 7:
        return method_id("calc_token_amount(uint256[7])")
    if n == 8:
        return method_id("calc_token_amount(uint256[8])")
    return b""


@internal
@view
def _walk(
    legs: DynArray[Leg, MAX_ALL_LEGS],
    lo: uint256,
    hi: uint256,
    amount_in: uint256,
    dst_slot: uint8,
) -> uint256:
    """Run legs[lo:hi] over a slot accumulator; 0 means the route is dead."""
    bal: uint256[MAX_SLOTS] = empty(uint256[MAX_SLOTS])
    bal[0] = amount_in

    cur: uint256 = max_value(uint256)
    base: uint256 = 0

    for k: uint256 in range(MAX_ALL_LEGS):
        if k + lo >= hi:
            break
        leg: Leg = legs[k + lo]
        src: uint256 = convert(leg.src_slot, uint256)
        dst: uint256 = convert(leg.dst_slot, uint256)
        if src >= MAX_SLOTS or dst >= MAX_SLOTS:
            return 0

        # Snapshot the node balance when a group of legs leaving it opens, so
        # `bps` does not depend on the order the group drains.
        if src != cur:
            cur = src
            base = bal[src]

        dx: uint256 = 0
        if leg.bps == 0:
            dx = bal[src]  # last leg out of this node sweeps the remainder
        else:
            dx = base * convert(leg.bps, uint256) // BPS
            if dx > bal[src]:
                dx = bal[src]
        if dx == 0:
            # A split can round a leg down to nothing.  That is a leg with no
            # work to do, not a dead route -- killing the candidate here would
            # discard an otherwise fine split over its smallest branch.
            continue

        res: Res = self._quote(leg.target, leg.kind, leg.i, leg.j, leg.n, dx)
        if res.status != STATUS_VALUE or res.value == 0:
            return 0  # reverting or unimplemented leg kills the candidate

        bal[src] -= dx
        bal[dst] += res.value

    return bal[convert(dst_slot, uint256)]


# --------------------------------------------------------------- external


@external
@view
def probe_batch(probes: DynArray[Probe, MAX_PROBES]) -> DynArray[Res, MAX_PROBES]:
    """Quote many independent (pool, direction, size) points in one eth_call.

    This is the probe ladder's transport: ~5,600 grid points for the whole
    Ethereum universe fit in a handful of calls, so no Multicall3 is needed.
    """
    out: DynArray[Res, MAX_PROBES] = []
    for p: Probe in probes:
        out.append(self._quote(p.pool, p.kind, p.i, p.j, p.n, p.dx))
    return out


@external
@view
def quote_route(
    legs: DynArray[Leg, MAX_LEGS], amount_in: uint256, dst_slot: uint8
) -> uint256:
    """Output of one route.  0 means it reverts or is unquotable."""
    wide: DynArray[Leg, MAX_ALL_LEGS] = []
    for leg: Leg in legs:
        wide.append(leg)
    return self._walk(wide, 0, len(wide), amount_in, dst_slot)


@external
@view
def quote_routes(
    legs: DynArray[Leg, MAX_ALL_LEGS],
    bounds: DynArray[uint16, MAX_ROUTES],
    amounts_in: DynArray[uint256, MAX_ROUTES],
    dst_slots: DynArray[uint8, MAX_ROUTES],
) -> DynArray[uint256, MAX_ROUTES]:
    """All candidate routes in one call.  `bounds[k]` ends route k's legs.

    Flat rather than nested so the calldata stays a single dynamic array, and so
    ~20 multi-hop candidates cost one round trip instead of twenty.
    """
    assert len(bounds) == len(amounts_in), "bounds/amounts length"
    assert len(bounds) == len(dst_slots), "bounds/slots length"

    out: DynArray[uint256, MAX_ROUTES] = []
    lo: uint256 = 0
    for k: uint256 in range(MAX_ROUTES):
        if k >= len(bounds):
            break
        hi: uint256 = convert(bounds[k], uint256)
        if hi > len(legs) or hi < lo:
            out.append(0)
            continue
        out.append(self._walk(legs, lo, hi, amounts_in[k], dst_slots[k]))
        lo = hi
    return out


@external
@view
def raw_batch(
    targets: DynArray[address, MAX_PROBES], calldatas: DynArray[Bytes[128], MAX_PROBES]
) -> DynArray[Res, MAX_PROBES]:
    """Arbitrary batched reads (balances, decimals, asset, maxDeposit, ...).

    Same three-state classification, so `balances(uint256)` vs `balances(int128)`
    can be resolved by which spelling actually answers.
    """
    assert len(targets) == len(calldatas), "length mismatch"
    out: DynArray[Res, MAX_PROBES] = []
    for k: uint256 in range(MAX_PROBES):
        if k >= len(targets):
            break
        out.append(self._static(targets[k], calldatas[k]))
    return out
