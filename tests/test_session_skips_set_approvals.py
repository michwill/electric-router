"""Not asking the router to top up allowances that are already there.

`set_approvals` makes the router read an allowance before every leg so that it
can grant the ones that are missing.  They are hardly ever missing: the
approval is the router's own rather than the caller's, it is infinite and it is
permanent, so the first person through a (token, pool) pair pays for it and
everyone after finds it there.  Measured by replaying a real two-leg mainnet
route at its own block: 307,617 gas with the flag on against 305,755 with it
off, both allowances already in place -- 936 gas a leg for an answer the local
EVM was holding.
"""

from __future__ import annotations

import asyncio

from erouter.chain import session as sess
from erouter.core.codec import selector


def run(coroutine):
    return asyncio.run(coroutine)


class Leg:
    """Just the two fields `_needs_approvals` reads off a realized leg."""

    def __init__(self, token_in: str, target: str) -> None:
        self.token_in = token_in
        self.target = target


class Route:
    def __init__(self, legs) -> None:
        self.legs = legs


class Backend:
    """Answers `allowance(router, pool)` from a table, and records the asks."""

    def __init__(self, answers: dict, fail: set = frozenset()) -> None:
        self.answers = answers
        self.fail = set(fail)
        self.asked: list[tuple[str, str]] = []

    def call(self, caller, to, calldata, *rest):
        assert calldata[:4] == selector(sess.ALLOWANCE)
        owner = "0x" + calldata[4 + 12:4 + 32].hex()
        spender = "0x" + calldata[36 + 12:36 + 32].hex()
        assert owner.lower() == sess.ROUTER_ADDRESS.lower()
        self.asked.append((to.lower(), spender.lower()))
        if to.lower() in self.fail:
            return {"success": False, "output": b""}
        value = self.answers[(to.lower(), spender.lower())]
        return {"success": True, "output": value.to_bytes(32, "big")}


class Evm:
    """The miss loop, with nothing ever missing."""

    async def fill(self, rpc, run_it, **kw):
        return run_it()


def session_with(backend: Backend) -> sess.RouterSession:
    made = sess.RouterSession.__new__(sess.RouterSession)
    made.backend = backend
    made.evm = Evm()
    made.rpc = None
    return made


TOKEN = "0x" + "a1" * 20
OTHER = "0x" + "a2" * 20
POOL = "0x" + "b1" * 20
POOL2 = "0x" + "b2" * 20


def test_every_pool_already_approved_turns_the_flag_off():
    backend = Backend({(TOKEN, POOL): sess.MAX_UINT,
                       (OTHER, POOL2): sess.MAX_UINT})
    made = session_with(backend)
    route = Route([Leg(TOKEN, POOL), Leg(OTHER, POOL2)])
    assert run(made._needs_approvals(route, 1)) is False
    assert len(backend.asked) == 2


def test_one_missing_allowance_keeps_the_flag_on():
    backend = Backend({(TOKEN, POOL): sess.MAX_UINT, (OTHER, POOL2): 0})
    made = session_with(backend)
    route = Route([Leg(TOKEN, POOL), Leg(OTHER, POOL2)])
    assert run(made._needs_approvals(route, 1)) is True


def test_a_finite_allowance_is_not_enough():
    # `_allow` returns early only on the maximum: anything less and the router
    # resets through zero and approves again, so the flag has work to do.
    backend = Backend({(TOKEN, POOL): sess.MAX_UINT - 1})
    made = session_with(backend)
    assert run(made._needs_approvals(Route([Leg(TOKEN, POOL)]), 1)) is True


def test_a_pair_asked_twice_is_read_once():
    backend = Backend({(TOKEN, POOL): sess.MAX_UINT})
    made = session_with(backend)
    route = Route([Leg(TOKEN, POOL), Leg(TOKEN, POOL)])
    assert run(made._needs_approvals(route, 1)) is False
    assert len(backend.asked) == 1


def test_native_legs_need_no_allowance():
    # Native ETH moves through `msg.value`, and `_allow` returns on it too.
    backend = Backend({})
    made = session_with(backend)
    route = Route([Leg(sess.NATIVE.upper(), POOL)])
    assert run(made._needs_approvals(route, 1)) is False
    assert backend.asked == []


def test_a_token_that_will_not_answer_keeps_the_flag_on():
    backend = Backend({(TOKEN, POOL): sess.MAX_UINT}, fail={TOKEN})
    made = session_with(backend)
    assert run(made._needs_approvals(Route([Leg(TOKEN, POOL)]), 1)) is True
