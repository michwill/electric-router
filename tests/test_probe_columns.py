"""`probe_columns` must answer what `probe` answers.

The columnar entry point exists because a `Probe` is six fields behind a
validating `__post_init__` -- 885 ns to build against 42 for the tuple under
it -- and refine builds one per probe, 1,380 of them on a warm mainnet quote,
only to take a `Quote` object back for each.  Neither carries anything the
ladders do not already hold in columns.

That saving is only allowed to exist while the two agree, and they have to
agree on all three paths at once: the batch that Rust serves, the Python
fallback for a model the batch declines, and the delegation to the inner
client for a pool with no model at all.  The vectors below reach all three.
"""

from __future__ import annotations

from erouter.chain.exact_probe import ExactQuoterClient
from erouter.core.quoter import Quote, Status
from erouter.core.stableswap import StableSwap
from erouter.core.types import ArcKind, Probe
from erouter.core.vault import Vault

GNOSIS_3POOL = {
    "balances": (142638 * 10**18, 153563 * 10**6, 246110 * 10**6),
    "rates": (10**18, 10**30, 10**30),
    "amp": 200 * 100,
    "fee": 3 * 10**6,
    "a_precision": 100,
    "fee_on_xp": True,
    "admin_fee": 5 * 10**9,
}

POOL = "0x" + "11" * 20
VAULT = "0x" + "22" * 20
#: No model at all, so every probe on it is a hole and goes to the inner
#: client -- which is the third path.
UNKNOWN = "0x" + "33" * 20


class Models:
    """The `.get` / `.by_pool` / `len` an `ExactQuoterClient` reads."""

    def __init__(self, by_pool):
        self.by_pool = by_pool

    def get(self, pool):
        return self.by_pool.get(pool.lower())

    def __len__(self):
        return len(self.by_pool)


class Vaults:
    def __init__(self, model):
        self.model = model

    def get(self, pool, kind):
        return self.model if pool.lower() == VAULT else None


class Inner:
    """The wire, which answers a hole and counts how often it was asked."""

    def __init__(self):
        self.asked = 0

    def probe(self, probes):
        self.asked += len(probes)
        return [Quote(Status.VALUE, 7 * (k + 1)) if p.dx % 3 else
                Quote(Status.REVERTED, 0)
                for k, p in enumerate(probes)]


def _client():
    return ExactQuoterClient(
        Inner(), Models({POOL: StableSwap(**GNOSIS_3POOL)}),
        vaults=Vaults(Vault(num=11 * 10**17, den=10**18)))


def _probes():
    """Enough to reach the batch, the Python path and the wire."""
    out = []
    for dx in (10**18, 10**20, 10**22, 7 * 10**21):
        out.append(Probe(POOL, ArcKind.SWAP_STABLE, 0, 1, 3, dx))
        out.append(Probe(POOL, ArcKind.SWAP_STABLE, 1, 2, 3, dx // 10**12))
        out.append(Probe(VAULT, ArcKind.ERC4626_DEPOSIT, 0, 0, 2, dx))
        out.append(Probe(UNKNOWN, ArcKind.SWAP_STABLE, 0, 1, 2, dx))
    # A size no pool will serve, so a refusal reaches both spellings.
    out.append(Probe(POOL, ArcKind.SWAP_STABLE, 0, 1, 3, 10**40))
    return out


def _as_columns(quotes):
    values, status, names = [], [], []
    for q in quotes:
        state = getattr(q, "status", None)
        values.append(max(0, int(getattr(q, "value", 0) or 0)))
        if state is not None and state.name != "VALUE":
            if state.name not in names:
                names.append(state.name)
            status.append(names.index(state.name) + 1)
        else:
            status.append(0)
    return values, status, names


def test_probe_columns_agrees():
    """Same values, and the same refusals under the same names."""
    probes = _probes()

    want = _as_columns(_client().probe(probes))
    got = _client().probe_columns(
        [p.pool for p in probes], [p.kind for p in probes],
        [p.i for p in probes], [p.j for p in probes],
        [p.n for p in probes], [p.dx for p in probes])

    assert got[0] == want[0], "values differ"
    # The names are discovered in encounter order, so compare what each index
    # means rather than the lists, which is what the caller actually reads.
    assert [None if s == 0 else got[2][s - 1] for s in got[1]] == \
           [None if s == 0 else want[2][s - 1] for s in want[1]], "statuses differ"


def test_all_three_paths_were_reached():
    """A test that only exercised the batch would pass while the other two
    were broken, so this asserts the vectors are doing their job."""
    client = _client()
    probes = _probes()
    client.probe(probes)
    stats = client.stats
    assert stats.computed > 0, "the batch and the Python path served nothing"
    assert stats.delegated > 0, "nothing reached the inner client"
    assert client.client.asked > 0


def test_a_client_with_no_models_still_answers_in_columns():
    """`enabled` off, or no exact models: the whole batch is delegated and the
    columns come back from the inner client's objects."""
    inner = Inner()
    client = ExactQuoterClient(inner, Models({}))
    probes = _probes()[:4]
    values, status, names = client.probe_columns(
        [p.pool for p in probes], [p.kind for p in probes],
        [p.i for p in probes], [p.j for p in probes],
        [p.n for p in probes], [p.dx for p in probes])
    assert len(values) == len(probes)
    assert inner.asked == len(probes)
    assert all(s == 0 or names[s - 1] for s in status)
