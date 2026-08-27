"""`min_out` is measured from what the chain said, not from what the model did.

A caller asking for 50 bp of room expects a bound 50 bp under the number they
were shown.  Taken off `route.modelled_out` it was 50 bp under something else:
the model is a *choice*, accurate enough to pick pools and split flow and no
better, and on a volatile leg it can stand tens of basis points off the
quoter.  Measured on ETH -> DOLA at block 25,846,510 the model sat 50.55 bp
above what the route really paid, so a 50 bp bound landed 0.30 bp *above* the
output and the call reverted on `min_out` at its own dry run -- a promise
nothing could keep, given to someone who had asked for room and been given
none.

`verified_out` is the chain's answer and the figure the caller was shown, so it
is what a promise about that figure has to be measured from.  `guaranteed_out`
is wrong for the opposite reason: it already discounts every leg's tolerance,
and deriving from it would compound the two.
"""

from __future__ import annotations

import asyncio

from erouter.chain import session as sess


def run(coroutine):
    return asyncio.run(coroutine)


class Leg:
    def __init__(self) -> None:
        self.target = "0x" + "a1" * 20
        self.token_in = "0x" + "11" * 20


class Route:
    def __init__(self, modelled_out: int) -> None:
        self.legs = [Leg()]
        self.modelled_out = modelled_out
        self.dst_slot = 1


class Result:
    def __init__(self, modelled_out: int, verified_out: int | None) -> None:
        self.route = Route(modelled_out)
        self.verified_out = verified_out
        self.amount_in = 10**18


class Call:
    """Only what `plan_call` reads back off an encoded call."""

    token_in = "0x" + "11" * 20
    guaranteed_out = 0
    quoted_out = 0
    tolerance_bp = 0.0
    unbounded = ()

    def calldata(self, sender: str = "") -> bytes:
        return b"\x00" * 4


class Chain:
    stables: tuple = ()
    forex: tuple = ()


def session_for(monkeypatch, seen: dict) -> sess.RouterSession:
    """A session stubbed down to the one line under test."""
    made = sess.RouterSession.__new__(sess.RouterSession)
    made.chain = Chain()
    made.pools = []
    made.rpc = None
    made.evm = None
    made.client = None

    async def header(_floor):
        return {"number": hex(100)}

    async def read_slots(_slots, at=None):
        return None

    async def needs_approvals(_route, _block):
        return True

    async def dry_run(_data, _sender, _value, _block):
        return 21_000, ""

    class Evm:
        async def fill(self, _rpc, _run, **_kw):
            # The pricing closure is not what this test is about, and running
            # it would need a whole quoter behind it.
            return None

    made.evm = Evm()
    made._header_at_least = header
    made._set_block_env = lambda _h: None
    made._route_accounts = lambda _r: set()
    made._read_slots = read_slots
    made._needs_approvals = needs_approvals
    made._dry_run = dry_run
    made._code_for = None

    class Backend:
        """`plan_call` asks it which slots the route touches; none do."""

        def known_slots(self):
            return []

    made.backend = Backend()

    def capture(route, **kw):
        seen.update(kw)
        return Call()

    monkeypatch.setattr(sess, "encode_route", capture)
    monkeypatch.setattr(sess, "volatile_pools", lambda *a, **kw: ())
    return made


#: Far enough apart that a bound off the wrong one lands above the output,
#: which is the failure being guarded: 50.55 bp was the measured gap.
MODELLED = 252_083_623_693_320_170_518
VERIFIED = 250_815_774_777_133_480_755


def test_the_bound_comes_off_the_verified_output(monkeypatch):
    seen: dict = {}
    made = session_for(monkeypatch, seen)
    run(made.plan_call(Result(MODELLED, VERIFIED),
                       receiver="0x" + "22" * 20, min_out_bp=50.0))
    assert seen["min_out"] == int(VERIFIED * (1 - 50.0 / 1e4))


def test_a_bound_off_the_model_would_have_been_above_the_output(monkeypatch):
    # The bug, stated as arithmetic: 50 bp under the model is *more* than the
    # route pays, so the call cannot settle however honest the market is.
    assert int(MODELLED * (1 - 50.0 / 1e4)) > VERIFIED
    seen: dict = {}
    made = session_for(monkeypatch, seen)
    run(made.plan_call(Result(MODELLED, VERIFIED),
                       receiver="0x" + "22" * 20, min_out_bp=50.0))
    assert seen["min_out"] < VERIFIED


def test_the_model_still_serves_when_nothing_verified_the_route(monkeypatch):
    # A route the chain never priced has only the model to promise against,
    # and a bound off it beats no bound at all.
    seen: dict = {}
    made = session_for(monkeypatch, seen)
    run(made.plan_call(Result(MODELLED, None),
                       receiver="0x" + "22" * 20, min_out_bp=50.0))
    assert seen["min_out"] == int(MODELLED * (1 - 50.0 / 1e4))


def test_no_bound_is_asked_for_and_none_is_given(monkeypatch):
    seen: dict = {}
    made = session_for(monkeypatch, seen)
    run(made.plan_call(Result(MODELLED, VERIFIED), receiver="0x" + "22" * 20))
    assert seen["min_out"] == 0
