#!/usr/bin/env python3
"""Deploy RouteQuoter.vy, so quoting stops shipping the contract with it.

Until it is deployed, every quote goes out as an `eth_call` with the runtime
bytecode attached as a state override.  That works and needs no transaction --
which is why development runs that way -- but it puts 7,486 bytes on the wire
14,974 hex characters at a time, on *every* call.  Measured on a cold
USDC->WETH route over a 129 ms link: 14 round trips, 1.54 MB sent, of which
0.21 MB (14%) was the contract.

Deploying also takes boa out of the request path entirely.  `core/quoter.py`
already builds calldata for "a RouteQuoter at this address", so once one
exists the browser build needs nothing but `eth_call` -- no compiler, no
state-override support, no boa.  That is the §Portability seam, and this is
the script that opens it.

Dry-runs on a fork by default; `--broadcast` sends it for real.

    python scripts/deploy_quoter.py --chain ethereum
    python scripts/deploy_quoter.py --chain ethereum --broadcast --verify

The deployer key is a brownie keystore under `~/.brownie/accounts/`, decoded
with a passphrase at the prompt -- the same arrangement yb-core's scripts use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from getpass import getpass
from pathlib import Path
from time import sleep

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CONTRACT = REPO / "contracts" / "RouteQuoter.vy"
# The canonical deterministic-deployment proxy: CREATE2 with `salt` as the
# first 32 bytes of calldata and the initcode as the rest.  Checked on every
# chain in the table and present on all sixteen.
#
# This is what makes production affordable.  A scoped drpc endpoint gates
# `eth_call` by target address -- measured, a non-whitelisted target answers
# HTTP 403 "address 0x..." -- so every chain needs its quoter whitelisted, and
# a key holds at most ten addresses against fifteen chains.  CREATE2 fixes the
# address from (proxy, salt, initcode) alone, so one deployment recipe puts the
# quoter at the *same* address everywhere and one whitelist entry covers the
# lot.
CREATE2_PROXY = "0x4e59b44847b379578588920cA78FbF26c0B4956C"
# The salt carries the version, so changing the contract is a deliberate new
# address rather than a silent collision with a whitelist entry that no longer
# describes what is deployed.
SALT_PHRASE = b"erouter.RouteQuoter.v1"
# A pool and a swap that every mainnet fork can answer, used to prove the
# deployed contract behaves identically to the one we have been overriding.
SANITY_POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"  # 3pool
SANITY_DX = 1_000 * 10**6  # 1,000 USDC
# The kinds a redeployment is usually *for*: whatever the deployed contract
# could not price is exactly what nobody notices is missing, because an
# unknown kind comes back REVERTED and the router quietly routes around it.
SANITY_CDAI = "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643"
SANITY_CDAI_DX = 100_000 * 10**8


def account_address(name: str) -> str:
    """The keystore's address, without decoding anything.

    A brownie keystore holds its address in clear next to the encrypted key,
    so asking "where does the deployer need funding" costs no passphrase --
    which matters, because the answer decides whether it is worth typing one.
    """
    path = os.path.expanduser(os.path.join("~", ".brownie", "accounts", name + ".json"))
    if not os.path.exists(path):
        raise SystemExit(f"no keystore at {path}")
    with open(path) as handle:
        return "0x" + json.load(handle)["address"].lower().removeprefix("0x")


def account_load(name: str):
    """A brownie keystore, decoded at the prompt.  Never a plaintext key."""
    from eth_account import account

    path = os.path.expanduser(os.path.join("~", ".brownie", "accounts", name + ".json"))
    if not os.path.exists(path):
        raise SystemExit(f"no keystore at {path}")
    with open(path) as handle:
        key = account.decode_keyfile_json(json.load(handle), getpass(f"passphrase for {name}: "))
    return account.Account.from_key(key)


def verify_with_retries(contract, etherscan, attempts: int = 6, pause: int = 10) -> None:
    """Etherscan needs the code indexed before it will look at it."""
    from boa.verifiers import verify as boa_verify

    for attempt in range(attempts):
        try:
            sleep(pause)
            boa_verify(contract, etherscan, wait=True)
            print("  verified")
            return
        except ValueError as exc:
            if "Already Verified" in str(exc):
                print("  already verified")
                return
            print(f"  verify attempt {attempt + 1}/{attempts}: {exc}")
    print("  ! not verified -- the deployment is still good, verify by hand")


def _sanity_probe(chain):
    """A pool on *this* chain that the deployment can be checked against.

    Mainnet's 3pool was hardcoded, which is no test at all anywhere else: the
    address has no code, the quote returns zero, and a deployment that works
    would look broken.  The deepest pool in the chain's own universe is the
    honest choice -- it is the one a first route is most likely to touch.
    """
    from erouter.core.types import ArcKind, Probe
    from erouter.dev.universe import load_pools

    if chain.name == "ethereum":
        return ([Probe(SANITY_POOL, ArcKind.SWAP_STABLE, 1, 2, 3, SANITY_DX)],
                "1,000 USDC -> USDT on 3pool")
    load = load_pools(chain, min_tvl=1_000.0)
    for pool in sorted(load.pools, key=lambda p: -p.tvl_usd):
        if pool.swap_kind is None:
            continue
        for i, j in pool.swap_pairs():
            # Size from reserves where they are known, and from decimals where
            # they are not.  On the chains this script exists for, reserves are
            # exactly what cannot be read yet -- they come back as zeros, since
            # reading them needs the quoter being deployed here.  A thousandth
            # of a token is small enough that any pool above $1,000 can serve
            # it and large enough not to round to nothing.
            if i < len(pool.balances) and pool.balances[i] > 0:
                dx = max(int(pool.balances[i] * 1e-4), 1)
                note = "1e-4 of reserve"
            else:
                dx = max(10 ** max(pool.coins[i].decimals - 3, 0), 1)
                note = "0.001 token"
            return ([Probe(pool.address, pool.swap_kind, i, j,
                           pool.n_coins, dx)],
                    f"{pool.name[:28]} {i}>{j} at {note}")
    raise SystemExit(f"no quotable pool on {chain.name} to sanity-check against")


def create2_address(salt: bytes, initcode: bytes, proxy: str = CREATE2_PROXY) -> str:
    """Where CREATE2 through `proxy` puts this initcode -- on any chain."""
    from erouter.core.keccak import keccak256

    body = b"\xff" + bytes.fromhex(proxy[2:]) + salt + keccak256(initcode)
    return "0x" + keccak256(body)[12:].hex()


#: Chains not worth the gas, and why.  Printed rather than silently skipped:
#: a chain missing from a deployment sweep should say so.
UNSUPPORTED = {
    "etherlink": "rejects state overrides and access lists, and its tracer "
                 "output cannot be decoded -- a quoter there would still be "
                 "wire-only",
}


def funding_estimate(url: str, chain_id: int, payload: bytes, sender: str | None):
    """(gas, gas price, native cost) for this deployment, from the chain itself.

    Printed before anything is sent, because the question when deploying across
    fourteen chains is not "did it work" but "how much does each one need in
    the deployer".
    """
    from erouter.dev.rpc import JsonRpcTransport

    rpc = JsonRpcTransport(url, chain_id=chain_id)
    tx = {"to": CREATE2_PROXY, "data": "0x" + payload.hex()}
    if sender:
        tx["from"] = sender
    try:
        gas = int(rpc.fetch("eth_estimateGas", [tx]), 16)
    except Exception:
        # What the EVM charges to store code, plus the transaction itself.
        # Only used when the node will not estimate for an unfunded sender.
        gas = 21_000 + 200 * len(payload)
    price = rpc.gas_price()
    return gas, price, gas * price / 1e18


def funding_report(names: list[str], deployer: str, salt_phrase: str) -> int:
    """Balance against deployment cost, per chain.  Sends nothing, asks nothing.

    The deployment sweep needs the deployer funded on every chain at once, and
    a chain discovered to be short *during* the sweep is the expensive way to
    find out -- the passphrase is already typed and half the chains are done.
    """
    import boa

    from erouter.core.keccak import keccak256
    from erouter.dev import chains as chain_table
    from erouter.dev import config
    from erouter.dev.rpc import JsonRpcTransport

    initcode = boa.load_partial(str(CONTRACT)).compiler_data.bytecode
    salt = keccak256(salt_phrase.encode())
    address = create2_address(salt, initcode)
    print(f"  deployer  {deployer}")
    print(f"  quoter    {address} (once deployed)\n")
    print(f"  {'chain':<11} {'native':<8} {'balance':>14} {'needs':>12} {'short by':>12}")
    print("  " + "-" * 62)

    short: list[str] = []
    for name in names:
        chain = chain_table.get(name)
        try:
            rpc = JsonRpcTransport(config.rpc_url(chain.rpc_attr), chain_id=chain.chain_id)
            balance = int(rpc.fetch("eth_getBalance", [deployer, "latest"]), 16) / 1e18
            if bytes.fromhex(
                (rpc.fetch("eth_getCode", [address, "latest"]) or "0x")[2:]
            ):
                print(f"  {name:<11} {chain.native_symbol:<8} {'-':>14} "
                      f"{'deployed':>12} {'':>12}")
                continue
            _, _, cost = funding_estimate(
                config.rpc_url(chain.rpc_attr), chain.chain_id, salt + initcode, deployer)
        except Exception as exc:
            print(f"  {name:<11} {'':<8} {'':>14} {'':>12}   ! {str(exc)[:28]}")
            short.append(name)
            continue
        # Twice the estimate: gas prices move between reading one and sending
        # a transaction, and a sweep that dies half way through is worse than
        # one that asks for a little more up front.
        want = cost * 2
        gap = max(0.0, want - balance)
        print(f"  {name:<11} {chain.native_symbol:<8} {balance:>14.6f} {want:>12.6f} "
              f"{(f'{gap:.6f}' if gap else '-'):>12}")
        if gap:
            short.append(name)

    if short:
        print(f"\n  {len(short)} chain(s) need funding: {', '.join(short)}")
    else:
        print("\n  every chain is funded; deploy with --chain all --create2 --broadcast")
    return 0 if not short else 1


def deploy_one(name: str, args, account=None) -> int:
    import boa

    from erouter.dev import chains as chain_table
    from erouter.dev import config
    from erouter.dev.boa_host import runtime_bytecode

    chain = chain_table.get(name)
    url = config.rpc_url(chain.rpc_attr)
    expected = runtime_bytecode()
    print(f"{chain.name} (chain {chain.chain_id})")
    print(f"  contract  {CONTRACT.relative_to(REPO)}  ({len(expected):,} bytes runtime)")
    print(f"  mode      {'BROADCAST' if args.broadcast else 'fork rehearsal'}")

    if args.broadcast:
        boa.set_network_env(url)
        boa.env.add_account(account)
        deployer = account
        print(f"  deployer  {deployer.address}")
    else:
        # `allow_dirty` because a sweep forks once per chain and the previous
        # fork still holds the contract it just rehearsed deploying; boa
        # refuses to move on otherwise, and every chain after the first fails
        # with "Cannot fork with dirty state".  Nothing is carried over -- each
        # fork starts from that chain's own head.
        # Pinned to a real block rather than boa's default `safe` tag, which
        # polygon's endpoint does not serve ("-32000: safe block not found")
        # and which would otherwise drop that chain out of the sweep.
        from erouter.dev.rpc import JsonRpcTransport

        head = JsonRpcTransport(url, chain_id=chain.chain_id).pin.block
        boa.fork(url, block_identifier=head - 5, allow_dirty=True)
        boa.env.eoa = boa.env.generate_address()
        deployer = None
        print(f"  deployer  {boa.env.eoa} (fork)")

    if args.create2:
        from erouter.core.keccak import keccak256

        initcode = boa.load_partial(str(CONTRACT)).compiler_data.bytecode
        salt = keccak256(args.salt.encode())
        address = create2_address(salt, initcode)
        print(f"  salt      {args.salt!r} -> 0x{salt.hex()}")
        print(f"  proxy     {CREATE2_PROXY}")
        print(f"\n  deterministic address {address}")
        if not bytes(boa.env.get_code(CREATE2_PROXY)):
            print("  ! the deterministic proxy is not deployed on this chain")
            return 1
        existing = bytes(boa.env.get_code(address))
        if existing:
            # Re-running across fourteen chains must be safe: a chain that is
            # already done should say so and cost nothing, not revert half way
            # through the sweep.
            print(f"  already deployed ({len(existing):,} bytes) -- nothing to send")
        else:
            gas, price, cost = funding_estimate(
                url, chain.chain_id, salt + initcode,
                deployer.address if deployer else None)
            print(f"  cost      {gas:,} gas at {price / 1e9:,.3f} gwei "
                  f"= {cost:.6f} {chain.native_symbol}")
            boa.env.execute_code(CREATE2_PROXY, data=salt + initcode, is_modifying=True)
            print(f"  sent {len(salt) + len(initcode):,} bytes through the proxy")
    else:
        quoter = boa.load(str(CONTRACT))
        address = str(quoter.address)
        print(f"\n  deployed at {address}")

    # Same bytes we have been overriding with?  A mismatch means the deployed
    # contract is not the one every measurement in this repo was taken against.
    on_chain = boa.env.get_code(address)
    if bytes(on_chain) != bytes(expected):
        print(f"  ! runtime differs: {len(on_chain):,} on chain vs {len(expected):,} compiled")
        return 1
    print(f"  runtime matches the compiled bytecode ({len(expected):,} bytes)")

    # And does it answer?  Compare it against the override path on the same
    # block, because "deployed" and "usable" are different claims.
    from erouter.core.quoter import QuoterClient
    from erouter.core.types import ArcKind, Probe
    from erouter.dev.boa_host import override_client
    from erouter.dev.rpc import JsonRpcTransport

    probe, label = _sanity_probe(chain)
    if args.broadcast:
        rpc = JsonRpcTransport(url)
        deployed = QuoterClient(rpc, address).probe(probe)[0]
        print(f"\n  sanity {label}")
        print(f"    deployed   {deployed.value:,} ({deployed.status.name})")
        if not deployed.ok or deployed.value <= 0:
            print("  ! the deployed quoter cannot price a pool on this chain")
            return 1
        # The override is the reference *where one can be had*.  The chains
        # this deployment is for are exactly the ones that reject state
        # overrides -- tac and etherlink answer HTTP 400 -- so on those there
        # is nothing to compare against and a positive quote is the whole test.
        try:
            overridden = override_client(rpc).probe(probe)[0]
        except Exception as exc:
            print(f"    override   unavailable on this chain ({str(exc)[:40]})")
            overridden = None
        if overridden is not None and overridden.ok:
            print(f"    override   {overridden.value:,} ({overridden.status.name})")
            if deployed.value != overridden.value:
                print("  ! the deployed quoter disagrees with the override; not usable")
                return 1
            print("    agree")

        # Every kind, not just the one that has always worked.  A kind the
        # deployment is missing does not fail loudly: it answers REVERTED, and
        # the router treats that as "this leg cannot be traded" and silently
        # takes a worse route.
        #
        # cDAI is a mainnet address, so this is a mainnet test.  Running it
        # everywhere would fail every other chain *after* broadcasting, which
        # reads as a broken deployment when it is a missing token.
        lend = [Probe(SANITY_CDAI, ArcKind.LEND_REDEEM, 0, 0, 0, SANITY_CDAI_DX)]
        if chain.name != "ethereum":
            lend = []
        if lend:
            got = QuoterClient(rpc, address).probe(lend)[0]
            want = override_client(rpc).probe(lend)[0]
            print("\n  sanity 100,000 cDAI -> DAI (LEND_REDEEM)")
            print(f"    deployed   {got.value:,} ({got.status.name})")
            print(f"    override   {want.value:,} ({want.status.name})")
            if got.value != want.value or not got.ok:
                print("  ! the deployed quoter cannot price LEND_REDEEM -- lending")
                print("    arcs would be built and then silently routed around")
                return 1
            print("    agree")

        if args.verify:
            from boa.explorer import Etherscan

            key = getattr(config.networks(), "ETHERSCAN_API_KEY", None)
            if key:
                # CREATE2 leaves no deployer object to verify, so bind one to
                # the address the proxy put the code at.
                deployed_at = (boa.load_partial(str(CONTRACT)).at(address)
                               if args.create2 else quoter)
                verify_with_retries(deployed_at, Etherscan(api_key=key))
            else:
                print("  ! no ETHERSCAN_API_KEY in networks.py; skipping verification")

        print("\n  next, in order:")
        print(f'   1. src/erouter/dev/chains.py: quoter="{address}"')
        print("   2. erouter warmcache   -- the local EVM reads the quoter's code")
        print("      from data/evm-state, and this address is not in it yet; without")
        print("      this every local quote returns nothing")
        print(f"   3. whitelist {address} on the RPC key, and drop the old one")
        print("   4. erouter route --from cDAI --to DAI --amount 100000")
        print("      -- should now work without --fresh-quoter")
    else:
        print("\n  fork rehearsal only -- nothing was broadcast.")
        print("  re-run with --broadcast (and --verify) when you want it on chain.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chain", default="ethereum",
        help="a chain name, or 'all' for every chain that can be served",
    )
    parser.add_argument("--account", default="babe", help="brownie keystore name")
    parser.add_argument(
        "--broadcast", action="store_true",
        help="send the transaction. Without it, this deploys on a fork and "
             "throws the result away, which is the safe way to rehearse",
    )
    parser.add_argument("--verify", action="store_true", help="submit to Etherscan")
    parser.add_argument(
        "--create2", action="store_true",
        help="deploy through the deterministic proxy, so the quoter lands on "
             "the same address on every chain and costs one whitelist entry "
             "instead of fifteen",
    )
    parser.add_argument(
        "--salt", default=SALT_PHRASE.decode(),
        help="salt phrase for --create2; changing it changes the address",
    )
    parser.add_argument(
        "--funding", action="store_true",
        help="report the deployer's balance against what each chain's "
             "deployment costs, and stop. Needs no passphrase",
    )
    args = parser.parse_args()

    from erouter.dev import chains as chain_table

    if args.chain == "all":
        wanted = [n for n in chain_table.CHAINS if n not in UNSUPPORTED]
        for name, why in UNSUPPORTED.items():
            print(f"  skipping {name}: {why}\n")
    else:
        wanted = [args.chain]

    if args.funding:
        return funding_report(wanted, account_address(args.account), args.salt)

    # Once, not once per chain: a fourteen-chain sweep should ask for the
    # passphrase a single time.
    account = account_load(args.account) if args.broadcast else None

    results: dict[str, int] = {}
    for name in wanted:
        try:
            results[name] = deploy_one(name, args, account)
        except Exception as exc:
            print(f"  ! {name} failed: {str(exc)[:100]}")
            results[name] = 1
        print()

    if len(wanted) > 1:
        done = [n for n, code in results.items() if code == 0]
        print(f"  {len(done)}/{len(wanted)} chains carry the quoter")
        for name, code in results.items():
            if code != 0:
                print(f"    ! {name}")
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
