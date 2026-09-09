"""What `plan_call` re-reads, and why the token being spent is on the list.

The route's pools are the obvious answer and were the whole of it.  But the
dry run behind a plan begins with `transferFrom`, and what that stands on --
`allowance[sender][ROUTER_ADDRESS]` -- lives on the token, not on any pool.

`LocalEvm.fill` repairs *misses*, and a slot already loaded is not a miss.  So
the allowance read before an approval stayed loaded after it: the second plan
waited for the approval's block, read that block, and never asked about the
one slot the approval had changed.  It refused with
`ERC20InsufficientAllowance(router, 0, amount)` against a chain where the
allowance was there, and only rebuilding the session cleared it.
"""

from __future__ import annotations

from types import SimpleNamespace

from erouter.chain.session import RouterSession

POOL_A = "0x" + "aa" * 20
POOL_B = "0x" + "bb" * 20
ORACLE = "0x" + "cc" * 20
TOKEN = "0x" + "dd" * 20


def route(*targets, src_token=TOKEN):
    return SimpleNamespace(
        legs=[SimpleNamespace(target=t) for t in targets], src_token=src_token)


def accounts(route_, needs=None):
    """`_route_accounts` off a session with nothing else built."""
    session = RouterSession.__new__(RouterSession)
    session.state = SimpleNamespace(arc_needs=needs or {})
    return session._route_accounts(route_)


def test_the_token_being_spent_is_re_read_with_the_pools() -> None:
    got = accounts(route(POOL_A, POOL_B))

    assert got == {POOL_A, POOL_B, TOKEN.lower()}


def test_the_pools_and_what_their_arcs_read_through_still_are() -> None:
    got = accounts(route(POOL_A), needs={POOL_A: (ORACLE,)})

    assert {POOL_A, ORACLE, TOKEN.lower()} <= got


def test_a_route_that_names_no_source_asks_for_no_empty_account() -> None:
    """Native ETH rides on `msg.value` and has no allowance to read; an empty
    string in the set would match nothing and cost a needless comparison."""
    got = accounts(route(POOL_A, src_token=""))

    assert got == {POOL_A}
    assert "" not in got


def test_the_source_is_matched_however_it_was_spelled() -> None:
    """`known_slots` reports addresses lowercased, and the set is compared
    against it -- a checksummed token would never match its own slots."""
    got = accounts(route(POOL_A, src_token=TOKEN.upper()))

    assert TOKEN.lower() in got
