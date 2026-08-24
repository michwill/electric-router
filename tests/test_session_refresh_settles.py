"""A refresh rebuilds the exact models against state the EVM actually holds.

The warm gate runs the rebuild under the miss loop because a slot the EVM does
not hold reads **zero**, and a model freezes whatever it read.  `refresh` ran
the same closure bare, so each refresh could bake another zero in -- a fee or
oracle word read as zero gives an absurd conductance, and the session drifted
from a cold one by up to 0.83% and eventually refused a route on KCL.

Reported from the Flet frontend against 2b275ff, with the miss measured: an
oracle a model reads *through* advanced a round, and the slot for the new round
was never fetched.
"""

from __future__ import annotations

import asyncio

from erouter.chain import session as sess


def run(coroutine):
    return asyncio.run(coroutine)


class Rpc:
    async def call(self, method: str, params: list):
        assert method == "eth_getBlockByNumber"
        return {"number": hex(101), "timestamp": "0x1", "gasLimit": "0x1"}


class Evm:
    """Records what ran under the miss loop, and what ran outside it."""

    def __init__(self, log: list) -> None:
        self.log = log

    def repin(self, block: int) -> None:
        self.log.append(("repin", block))

    async def fill(self, rpc, run_it, *, block, code_for=None):
        # Two rounds, which is what a real fill does when the first asks for
        # something: the rebuild must tolerate being run more than once.
        self.log.append(("fill", block))
        run_it()
        return run_it()


class Client:
    def __init__(self, log: list) -> None:
        self.log = log

    def refresh_at(self, block: int) -> int:
        self.log.append(("commit", block))
        return 1


def session_at_block(log: list) -> sess.RouterSession:
    made = sess.RouterSession.__new__(sess.RouterSession)
    made.rpc = Rpc()
    made.evm = Evm(log)
    made.client = Client(log)
    made.block = 100
    made.chain = type("Chain", (), {"chain_id": 1})()
    made.prepared = object()
    made.backend = type("Backend", (), {"known_slots": staticmethod(lambda: [])})()

    def rebuild(block: int = 0, cache=None):
        log.append(("rebuild", block, cache is not None))
        return ((), (), (), (), (), ())

    made._rebuild_models = rebuild
    made._set_block_env = lambda header: None
    made._read_slots = _noop
    made._read_balances = _noop
    return made


async def _noop(*args, **kwargs):
    return None


def test_the_rebuild_runs_under_the_miss_loop():
    log: list = []
    block = run(session_at_block(log).refresh())
    assert block == 101

    kinds = [entry[0] for entry in log]
    assert "fill" in kinds, "the rebuild ran bare, so a missing slot reads zero"
    # Every rebuild happens inside the loop, and the commit comes after it.
    assert kinds.index("fill") < kinds.index("rebuild") < kinds.index("commit")
    assert kinds.count("commit") == 1

    rebuilds = [entry for entry in log if entry[0] == "rebuild"]
    assert len(rebuilds) >= 2, "the loop must be able to run it again"
    for _, at, throwaway in rebuilds:
        assert at == 101, "the rebuild reads the new block, not the old one"
        assert throwaway, "verdicts earned against incomplete state stick"


def test_a_session_that_never_built_models_still_refreshes():
    """A refresh before the gate ran has nothing to rebuild and must not fail."""
    log: list = []
    made = session_at_block(log)
    made._rebuild_models = None

    assert run(made.refresh()) == 101
    assert [entry[0] for entry in log] == ["repin", "commit"]
