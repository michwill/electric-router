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


class Stats:
    def __init__(self) -> None:
        self.unreadable = 0
        self.errors: list = []


class Evm:
    """Records what ran under the miss loop, and what ran outside it."""

    def __init__(self, log: list, short: int = 0) -> None:
        self.log = log
        self.stats = Stats()
        self.short = short

    def repin(self, block: int) -> None:
        self.log.append(("repin", block))

    async def fill(self, rpc, run_it, *, block, code_for=None):
        # Two rounds, which is what a real fill does when the first asks for
        # something: the rebuild must tolerate being run more than once.
        self.log.append(("fill", block))
        run_it()
        result = run_it()
        self.stats.unreadable += self.short
        return result


class Client:
    def __init__(self, log: list) -> None:
        self.log = log

    def refresh_at(self, block: int) -> int:
        self.log.append(("commit", block))
        return 1


def session_at_block(log: list, short: int = 0) -> sess.RouterSession:
    made = sess.RouterSession.__new__(sess.RouterSession)
    made.rpc = Rpc()
    made.evm = Evm(log, short)
    made.unreadable = 0
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


def test_what_a_refresh_could_not_read_is_counted_not_refused():
    """A dropped batch is common in a browser tab, and the caller decides.

    Failing the quote would be worse than the drift, so the count is offered
    and nothing is refused: the frontend re-warms in the background after a run
    of them.
    """
    log: list = []
    clean = session_at_block(log)
    run(clean.refresh())
    assert clean.unreadable == 0

    short = session_at_block(log, short=3)
    run(short.refresh())
    assert short.unreadable == 3, "a slot still on the old block's value is news"


def test_a_session_that_never_built_models_still_refreshes():
    """A refresh before the gate ran has nothing to rebuild and must not fail."""
    log: list = []
    made = session_at_block(log)
    made._rebuild_models = None

    assert run(made.refresh()) == 101
    assert [entry[0] for entry in log] == ["repin", "commit"]


# -- the gas price moves with the block -------------------------------------


class PricedRpc(Rpc):
    """Answers the block *and* the gas price, counting the latter."""

    def __init__(self, price: int = 3_000_000_000) -> None:
        self.price = price
        self.asked = 0

    async def call(self, method: str, params: list):
        if method == "eth_gasPrice":
            self.asked += 1
            return hex(self.price)
        return await super().call(method, params)


def priced_session(log: list, rpc: PricedRpc, was: int = 1_000_000_000):
    made = session_at_block(log)
    made.rpc = rpc
    made.gas_price_wei = was
    return made


def test_a_refresh_re_reads_the_gas_price():
    """It sets the tie tolerance in `verify.score`, whose bucket 0 is sorted by
    fewest legs -- so the price decides how many legs a route may spend.  Read
    once at the warm and never again, a session open since a quiet hour went on
    choosing routes sized for gas that had stopped applying.
    """
    rpc = PricedRpc(3_000_000_000)
    made = priced_session([], rpc)

    run(made.refresh())

    assert rpc.asked == 1, "the gas price was not re-read"
    assert made.gas_price_wei == 3_000_000_000


def test_a_block_that_did_not_move_is_not_asked_again():
    """Gas moves with the block, so an unchanged block has nothing to say."""
    rpc = PricedRpc()
    made = priced_session([], rpc)
    made.block = 101                      # already at what the header reports

    run(made.refresh())

    assert rpc.asked == 0


def test_an_endpoint_that_will_not_price_gas_keeps_the_old_figure():
    """The cheap half of a sweep that has already done the expensive part: a
    gas read that fails must not cost the caller the state that succeeded.
    """
    class Refusing(PricedRpc):
        async def call(self, method: str, params: list):
            if method == "eth_gasPrice":
                self.asked += 1
                raise RuntimeError("no gas price here")
            return await Rpc.call(self, method, params)

    rpc = Refusing()
    made = priced_session([], rpc)

    block = run(made.refresh())

    assert block == 101, "the refresh was lost to the gas read"
    assert made.gas_price_wei == 1_000_000_000, "the old figure was not kept"
    assert rpc.asked == 1


def test_a_zero_answer_does_not_erase_what_was_known():
    """`_gas_price` answers 0 for a shape it cannot read, and a route priced at
    zero gas branches for free -- which is the wrong way to fail."""
    rpc = PricedRpc(0)
    made = priced_session([], rpc)

    run(made.refresh())

    assert made.gas_price_wei == 1_000_000_000
