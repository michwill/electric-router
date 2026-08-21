# Executing a route

For whoever builds on `contracts/ElectricRouter.vy` -- a frontend, a bot, a
contract that routes as part of something larger.  [`theory.md`](theory.md) is
what decides the route; this is what settles it.

The reference encoder is `erouter.core.routecall`, pure Python and portable, and
`erouter route --calldata` prints exactly what it produces.  Everything below is
also written out in enough detail to build the call from nothing.

---

## 1. The call

```
execute(uint256 amount_in,
        address[] pools,
        uint256[] params,
        bool     set_approvals,
        address[] tokens   = [],
        address   receiver = msg.sender,
        uint256   min_out  = 0) -> uint256
```

`pools` and `params` are the same length and parallel: one pool, one packed word
of everything that pool's leg needs.  Pools stay a plain array of addresses so a
reader can see what is about to be called without unpacking anything.

Vyper writes one entry point per default argument, so **the shortest form that
still says what you mean is the one to send**.  Four selectors exist; dropping
`min_out`, then `receiver`, then `tokens` saves three words:

| you need | signature |
|---|---|
| nothing trailing | `execute(uint256,address[],uint256[],bool)` |
| named tokens | `…,bool,address[])` |
| a third-party receiver | `…,bool,address[],address)` |
| an end-to-end bound | `…,bool,address[],address,uint256)` |

Returns what the route produced of the last leg's output token, measured as a
balance delta.  Payable, `nonreentrant`, and stateless between calls.

## 2. The packed word

One `uint256` per pool, low bit first:

```
bits    width  field
0..59      60  frac       1e18 == 100%
60..187   128  min_rate   1e18-based, out per in, in raw token units
188..191    4  i
192..195    4  j
196..199    4  n_coins    only add_liquidity needs it
200..204    5  kind       ArcKind
205..209    5  in_ref     0 = read it off the pool, else tokens[ref - 1]
210..214    5  out_ref    same
215..255   41  must be zero
```

`kind` is `erouter.core.types.ArcKind`, and the same integers appear in
`RouteQuoter.vy` and `ElectricRouter.vy`.  `tests/test_kind_ids.py` fails the
build if the three lists ever disagree.

```
0  SWAP_STABLE           exchange(int128,int128,uint256,uint256)
1  SWAP_CRYPTO           exchange(uint256,…) then the use_eth spelling
2  DEPOSIT_FIXED         add_liquidity(uint256[N],uint256)
3  DEPOSIT_DYN           add_liquidity(uint256[],uint256)
4  DEPOSIT_FIXED_NOFLAG  add_liquidity(uint256[N],uint256)
5  WITHDRAW_STABLE       remove_liquidity_one_coin(uint256,int128,uint256)
6  WITHDRAW_CRYPTO       remove_liquidity_one_coin(uint256,uint256,uint256)
7  ERC4626_DEPOSIT       deposit(uint256,address)
8  ERC4626_REDEEM        redeem(uint256,address,address)
9  WRAP_NATIVE           payable deposit(), or deposit(uint256) on an adapter
10 UNWRAP_NATIVE         withdraw(uint256)
11 WSTETH_UNWRAP         unwrap(uint256)
12 WSTETH_WRAP           wrap(uint256)
13 STAKE_NATIVE          payable submit(address)
15 LEND_MINT             mint(uint256)
16 LEND_REDEEM           redeem(uint256)
```

14 is permanently reserved.  It was `SWAP_UNDERLYING` on an abandoned branch and
`data/facts` still records that survey under it.

## 3. Fractions are of what is left

`frac` is a share of the balance standing at the leg's input token **when that
leg runs**, read with `balanceOf`.  A 50/50 split is therefore `50%` then
`100%`, not `50%` twice:

```
leg 1   50% of 1,000 USDC   ->  500 spent, 500 left
leg 2  100% of   500 USDC   ->  500 spent, 0   left
```

Three things follow, and they are the reason for the encoding:

**A branch cannot be starved.**  The second leg spends what the first one really
left, not what it was modelled to leave, so a leg that returned a wei less than
expected costs a wei rather than a revert.

**Rebasing and fee-on-transfer tokens are measured.**  stETH loses a wei or two
on transfer; the router counts what arrived.

**The last leg out of a node must be `1e18`.**  Anything less strands the
remainder.  It is swept to the receiver rather than kept, so nothing is lost --
but it is also not routed, which is a worse price than the one you quoted.

Branches need no marking.  Order the legs so that a node's inflows all precede
its outflows -- a topological order -- and write each leg's share of what is left
at that point.  A branch and its merge fall out of that with no special case.

## 4. Every leg carries its own minimum rate

A single end-to-end bound lets a route be robbed in one pool and made whole in
another, which is the shape of a sandwich.  So each leg carries `min_rate` and
the router checks, on the balance that actually moved:

```
dy >= dx * min_rate / 1e18
```

`min_rate` is a rate between *raw* units, so it carries the decimal difference:
USDC (6) to WETH (18) around 4,000 USD is about `2.5e26`.  128 bits covers rates
from 1e-18 to 3.4e20; a pair outside that band -- an 18-decimal dust token
against an 8-decimal expensive one -- can only be bounded by its neighbours and
`min_out`, and the encoder reports those legs in `unbounded` rather than
shipping a silent zero.

### Where the reference rate has to come from

**Not from the route's own modelled leg amounts.**  The solver models each pool
by its first two derivatives; that is accurate enough to choose pools and split
flow and no better.  Measured on a live 13-leg mainnet route, its per-leg
figures were out by up to 37.9 bp against what the pools would really pay, in
both directions.  A minimum rate derived from those is a promise about a number
nothing checked -- and one leg of that route was mis-bounded by 22.4 bp against
a 13.9 bp tolerance, which reverts.

So `pipeline.price_legs` quotes every leg at its final size, in one batch, and
`RealizedLeg.verified_out` is what the bound is set against.  On a pool with an
exact model that costs arithmetic; otherwise one round trip for the route.

### How much tolerance

`erouter.core.routecall.min_rates` grants each leg

```
tolerance = min(CAP_BP, max(FEE_SHARE * fee, FLOOR_BP))
```

with `FEE_SHARE = 0.2`, `CAP_BP = 5`, `FLOOR_BP = 0.1`.  The floor is not
slippage at all -- it is room for the wei a wrap or a rebasing token rounds
away.  The cap is: a pool charging 140 bp would otherwise buy 28 bp of room,
which is far more than that leg is worth protecting for.  Five bp was a *floor*
here once, for volatile pairs; it granted a 1 bp pool twenty-five times what
the fee rule does, and it granted it to whoever front-ran the trade.

**`fee` is the least the pool can charge, not what the leg pays.**  That is the
whole of it: the attacker front-runs and unwinds in small, balanced trades and
is charged near `mid_fee`, while the leg they wrap around pays the dynamic fee
at its own size.  Measured on TricryptoUSDC, 3 bp against 13 bp -- so bounding
on the larger hands over the difference.  `core.poolfee.floor_fee` reads it off
the model: `min(mid_fee, out_fee)` for the cryptoswaps, and the nominal fee for
stableswap, whose off-peg multiplier only ever raises it.

Everything here is computed off chain and arrives as `min_rate` in the packed
word.  The router reads no oracle and re-derives no price; the solver's own
spot quote for that leg, minus this discount, is the bound.

### Whether a sandwich pays is decided by the leg, not by the bound

The attacker pays the fee twice on their own size and is paid the price
displacement the *victim* causes, so with `s` the front-run, `T` the pool and
`v` the victim's trade:

```
gain = (s/T) * v      cost = 2 * fee * s      profit = s * (v/T - 2 * fee)
```

The sign is set by the leg's own price impact against **twice the pool's fee**,
and `t` is nowhere in it -- `t` bounds `s`, which scales the profit without
changing its sign.  So:

**A leg flatter than twice its fee cannot be sandwiched at all.**  The round
trip loses money whatever the victim tolerates.  This is what splitting a route
buys, and on a stableswap it is bought very cheaply: measured on live mainnet
routes, *every* stableswap leg came in under the line -- 0.11 bp of impact at
2.9% of the pool against a 3 bp doubled fee, 0.10 bp at 5.95%.  The invariant is
flat; there is nothing to displace.

**A leg steeper than that is attackable, and then `t` is the cap.**  Ten of
thirty-one pool legs across four live routes sat above the line, all of them
cryptoswap: TricryptoUSDC at 88 bp of impact against a 26 bp doubled fee,
USD-BTC-ETH at 85 against 24.  Note it is the *low-fee* cryptoswaps that are
exposed -- TricryptoUSDT charges 69 bp and is out of reach at the same 0.8% of
reserve.  No tolerance fixes those; only a smaller leg does.

### What the bound guarantees

Exactly this: **the victim either settles at no worse than `(1 - t)` of its
quote, or it does not settle.**  The front-run can only be as large as the bound
will still let through, so extraction is capped at `t` of the trade -- and
measured, it comes to almost exactly `t`, scaling linearly with it and with
nothing else.  A fifth of the fee is a fifth of the fee in what it lets through,
and the 5 bp floor on a pool charging under 25 bp is looser by exactly
`5 / (0.2 * fee)`: twenty-five times on a 1 bp pool.  That is the compromise, as
a number.

Both halves are in `tests/test_sandwich.py`, against a constant-product pool and
a real `StableSwap`, driven through `min_rates` itself rather than a restatement
of it.

`charged_fee` measures the other one -- what this trade pays, by quoting it
twice, once as the pool is and once with its fee fields zeroed.  It is what the
CLI shows and what `leg_fee` returns; it is *not* what the bound is set from.

### What it was measured to buy

Sandwiching the deployed pools on a fork -- real contracts, real dynamic fees,
the attacker front-running and unwinding around a leg-sized victim, at 1 gwei
and ETH at $4,000:

| bound on | victim lost | attacker, net of gas |
|---|---|---|
| what the trade pays (13 bp) | 2.72 bp | **+$3.59 on a $15k leg** |
| the least it can charge (3 bp) | 0.60 bp | **−$0.15, loses** |

The attacker's *return on their own capital* barely moves -- around +180 bp
either way -- because the tolerance scales the front-run rather than the rate
of return.  What changes is that the front-run it permits shrinks from 204 USDC
to 45, and two transactions of gas then cost more than the attack makes.

That is a break-even, not immunity.  Gross extraction is about 0.6 bp of the
leg and gas is flat, so on TricryptoUSDC the attack loses at a $3k leg and at a
$15k leg, and pays from roughly $20k up -- still for no more than the tolerance.
3pool loses at every size, by 2.9 bp of the front-run: its impact never comes
near twice its fee.

`volatile` is passed in as a set of pool addresses rather than inferred:
`core.pools.volatile_pools` classifies by registry type, because nothing in an
arc distinguishes a pegged pair from an oraclised stableswap holding a volatile
one -- and that second shape is exactly what rugs on broadcast.

### What it adds up to

The bounds compound along the route, so `RouteCall.guaranteed_out` runs the
router's own arithmetic with every leg at its minimum and reports what the call
really promises.  On today's mainnet routes that is 0.3 bp under the quote for a
two-leg stable pair and 8-20 bp for a twelve-leg route through pools charging
over a hundred.  Show it before signing.  `min_out` is where a caller takes some
of it back, and it is checked against what the route produced -- not against the
router's balance, so a donation cannot satisfy it.

## 5. Tokens: read or named

`i` and `j` index the pool's own `coins`, and the router reads them.  That is
what makes them binding rather than advisory: a caller cannot point a minimum
rate at a token the pool does not hold at that index.

Tokens that are *not* pool coins have no getter that works on every deployment,
so those are named in `tokens` and pointed at by `in_ref` / `out_ref`.  Which to
name is a straight trade of calldata against gas, and it is the caller's:

| mode | names | when |
|---|---|---|
| `NONE` | nothing | shortest calldata: an L2, or a Curve lending callback that carries the whole call as `bytes` |
| `NEEDED` | LP tokens, vault assets, lending underlyings | the default |
| `ALL` | every token | cheapest to execute |

Measured, mainnet USDC to WETH over thirteen legs: 1,028 bytes and 2.47M gas
against 1,316 bytes and 2.43M.  Calldata wins on an L2 and loses on L1.

**One case cannot be read.**  Fourteen mainnet pools -- 3pool among them -- keep
their LP token in a separate contract and expose no getter for it at all.  The
router tries `lp_token()`, then `token()`, then falls back to the pool itself
only if the pool answers `totalSupply()`; otherwise it reverts with
`lp token unknown -- name it in tokens`.  Quietly treating those as their own LP
token would approve and count the wrong address.

A 1:1 token adapter wearing `WRAP_NATIVE` -- gnosis converts USDC.e to USDC
through one -- is named whatever mode you ask for, because nothing on it can be
read and the alternative is a call that means something else.

## 6. Approvals, native, and what comes back

**`set_approvals`** checks `allowance(self, pool) == max_uint256` per leg and
sets it if not.  Infinite, so it is paid once per (token, pool) and never again.
The reset-through-zero is there for USDT, which refuses a non-zero to non-zero
change and returns no data from `approve` at all.  Pass `false` once the
allowances are in place and save the reads.

**Native ETH** is `0xEeee…eEEeE`, Curve's sentinel.  A route whose first leg
spends it needs `msg.value == amount_in`; a route that does not spend it refuses
a non-zero `msg.value` rather than keeping it.  Payouts go out with a full-gas
call, so a contract wallet can receive them.

**Everything the route touched is swept** to `receiver` at the end -- not just
the destination.  The return value is the destination's balance *delta*, so
anything the router was already holding is handed over but not counted, and its
own balance is never a term in anyone's answer.  A donation to the router is
therefore a gift to whoever routes that token next, exactly as it is on Curve's
own router.

## 7. What it refuses

| revert | means |
|---|---|
| `one param word per pool` | `pools` and `params` are different lengths |
| `empty route` / `nothing to route` | no legs, or `amount_in == 0` |
| `bad receiver` | zero address, or the router itself |
| `reserved bits set` | bits 215+ of a packed word are non-zero |
| `frac out of range` | `frac` is 0 or above 1e18 |
| `in_ref`/`out_ref off the token list` | a ref points past `tokens` |
| `leg does not move between tokens` | input and output resolved to the same address |
| `coins` | the pool answered neither `coins(uint256)` nor `coins(int128)` |
| `lp token unknown` / `underlying unknown` | name it in `tokens` |
| `unknown leg kind` | a kind the router does not execute |
| `native input needs msg.value` | first leg spends native, `msg.value` disagrees |
| `route does not spend native` | `msg.value` sent to a route that wants none |
| `leg produced nothing` | the call succeeded and no balance moved |
| `leg below its minimum rate` | the leg's own bound |
| `below min_out` | the end-to-end bound |

`leg produced nothing` is worth understanding: some older crypto pools have only
the five-argument `exchange` **and** a `__default__` that swallows unknown
selectors, so the four-argument attempt succeeds and does nothing at all.  The
router does not ask whether the call succeeded, it asks whether anything moved --
and retries the other spelling if not.

## 8. Building the call

From this repo:

```python
from erouter.core.routecall import ALL, NONE, encode_route

call = encode_route(result.route, receiver=who, volatile=volatile_pools(pools),
                    quoted_out=result.verified_out)
tx = {"to": ROUTER, "data": call.calldata(sender=who), "value": 0}
```

`call.token_in` is what to hold and approve, `call.guaranteed_out` what the
bounds promise, `call.unbounded` any leg they could not.  `erouter route
--calldata` prints all of it, and `--json` carries it under `call`.

From another language, `routecall.py` is 442 lines including its reasoning
and the whole job is three: build the packed words, decide which tokens to
name, and ABI-encode the shortest signature.  The
two parts worth copying carefully are `fractions` -- shares of what is left, last
one always `1e18` -- and where the reference rate for `min_rate` comes from.

## 9. Traps

**A pinned block.**  Every number in a route is read at one block.  Quote and
execute against different state and the bounds are measuring a market that moved.

**Do not reuse a route.**  The fractions are fine at any size, but the minimum
rates were derived at one, and a dynamic fee moves with the trade.  Re-encode.

**The router is not deployed yet.**  `ElectricRouter` has no address in this
repo; `core.schema.ROUTER_ADDRESS` is empty on purpose, because a wrong address
there is a burnt transaction.  Changing the contract changes its CREATE2 address.

**32 legs, 31 named tokens.**  Beyond that the encoder refuses rather than
truncating.

**A bound is not a guarantee of price.**  It stops a leg settling below a rate.
It cannot tell a pool that will honour its quote from one that will move on
broadcast -- see the blacklist and the reserve check in [`theory.md`](theory.md),
none of which is a proof either.

**Nor is it a guarantee against being sandwiched**, only against being
sandwiched for more than `t` -- and only on a leg steep enough to be worth
attacking at all.  Whoever sets `volatile` is choosing how much, per pool.
