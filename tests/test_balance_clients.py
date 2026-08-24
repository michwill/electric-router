"""`read_balances` asks two different kinds of contract, and it matters which.

Four calls per coin: the pool's `balances()` in both spellings, and the coin's
`balanceOf(pool)` and `decimals()`.  The last two go to the **token**.

A caller reading pools out of a warmed local EVM has the pools' storage and not
the tokens'.  An unloaded account answers `balanceOf` with zero, which reads as
a pool holding nothing, and `check_reserves_are_real` drops it as insolvent --
so pointing the whole pass at the local EVM took five pools off gnosis and left
no path from WXDAI to EURe at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from erouter.core.transport import Answer, Status
from erouter.dev.universe import read_balances


@dataclass
class Coin:
    """A dataclass because `read_balances` `replace()`s it to fix decimals."""

    address: str
    decimals: int = 18
    symbol: str = "X"


@dataclass
class Pool:
    address: str = "0x" + "aa" * 20
    coins: list = field(default_factory=lambda: [Coin("0x" + "b1" * 20),
                                                 Coin("0x" + "b2" * 20)])
    n_coins: int = 2
    balances: tuple = ()
    held: tuple = ()
    name: str = "pool"
    pool_type: str = "stableswap"


class Client:
    """Answers every call, and remembers who was asked."""

    def __init__(self, value):
        self.value = value
        self.targets: list[str] = []

    def raw(self, calls):
        self.targets.extend(c.to.lower() for c in calls)
        return [Answer(Status.VALUE, self.value.to_bytes(32, "big")) for _ in calls]


def test_token_reads_go_to_the_token_client():
    pool = Pool()
    pools_client = Client(10**21)
    tokens_client = Client(10**21)

    read_balances([pool], pools_client, None, None, token_client=tokens_client)

    assert set(pools_client.targets) == {pool.address.lower()}, (
        "the pool client must only ever be asked about the pool")
    assert set(tokens_client.targets) == {c.address.lower() for c in pool.coins}, (
        "balanceOf and decimals belong to the coin")


def test_one_client_still_answers_everything():
    """The default has to keep working: most callers have only the wire."""
    pool = Pool()
    only = Client(10**21)
    read_balances([pool], only, None, None)
    assert {pool.address.lower()} < set(only.targets)
    assert {c.address.lower() for c in pool.coins} <= set(only.targets)


def test_the_pool_still_gets_its_balances():
    """Splitting the batch must not shuffle the answers."""
    pool = Pool()
    read_balances([pool], Client(7), None, None, token_client=Client(9))
    assert [int(b) for b in pool.balances] == [7, 7]
    assert [int(h) for h in pool.held] == [9, 9], "held comes from the coin"


def test_a_refresh_reads_balances_a_fill_would_leave_alone():
    """A rebuild at a new block has to actually re-read them.

    Filling only empty balances is right at the warm and a no-op afterwards, so
    `refresh_at`'s rebuild froze the warm block's numbers into every model it
    built.  Measured on mainnet: a session warmed 25 blocks back and refreshed
    quoted up to 0.71% away from one warmed cold at the same block, always
    worse, and the EVM under it was provably correct the whole time.
    """
    pool = Pool(balances=(1, 1), held=(1, 1))

    read_balances([pool], Client(7), None, None)
    assert [int(b) for b in pool.balances] == [1, 1], "a fill leaves them alone"

    read_balances([pool], Client(7), None, None, refresh=True)
    assert [int(b) for b in pool.balances] == [7, 7], "a refresh reads them again"


def test_a_llamma_is_left_alone_either_way():
    """It has no `balances()`; its reserves come from the market feed, and a
    refresh that probed for them would replace good numbers with none."""
    pool = Pool(balances=(5, 5), held=(5, 5), pool_type="llamma")
    asked = Client(7)

    read_balances([pool], asked, None, None, refresh=True)

    assert [int(b) for b in pool.balances] == [5, 5]
    assert asked.targets == [], "nothing should have been asked at all"
