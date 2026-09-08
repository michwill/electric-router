"""A refused log window is a hole in the census, and holes have to be re-asked.

The census is an hour of log windows against a public endpoint, and it has been
lost three times: to a span the node would not serve, to a timeout, and -- the
one these tests are about -- to 24 windows refused in a single batch.  That
last one counted the refusals and moved on, so the windows could not be named
afterwards, the count was cached, and every resumed run inherited a refusal for
a slice it would never re-ask.

Fakes here, because what is being tested is what happens when the node says no.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

v2_census = pytest.importorskip("v2_census")


def _log(pair: int, t0: int = 0xAA, t1: int = 0xBB) -> dict:
    return {"data": "0x" + f"{pair:064x}",
            "topics": ["0x" + "00" * 32, "0x" + f"{t0:064x}", "0x" + f"{t1:064x}"]}


class Node:
    """Refuses any window wider than `cap`, and answers the rest.

    That is the real failure: `eth_getLogs` caps the result set, so the same
    query asked again is refused again and a narrower one is not.
    """

    def __init__(self, logs: dict, cap: int = 10**9):
        self.logs, self.cap = logs, cap
        self.asked: list = []

    def fetch_multi(self, requests, concurrent=False):
        out = []
        for _method, params in requests:
            lo = int(params[0]["fromBlock"], 16)
            hi = int(params[0]["toBlock"], 16)
            self.asked.append((lo, hi))
            if hi - lo + 1 > self.cap:
                out.append(RuntimeError("query returned more than 10000 results"))
                continue
            out.append([_log(p) for b, p in self.logs.items() if lo <= b <= hi])
        return out


def test_a_refused_window_is_named_rather_than_counted():
    node = Node({}, cap=5)
    _pairs, refused = v2_census._log_windows(node, [(0, 99), (0, 4)], "0xf")
    assert refused == [(0, 99)], "the window itself, not a tally"


def test_a_refused_window_is_re_asked_narrower_until_it_answers():
    """Halving is the fix for both causes: a capped result set and a timeout."""
    node = Node({10: 0x1, 70: 0x2}, cap=30)
    found, refused = v2_census.retry_refused(node, [(0, 99)], "0xf")
    assert refused == []
    assert set(found) == {"0x" + f"{0x1:040x}", "0x" + f"{0x2:040x}"}


def test_a_window_that_never_answers_is_still_reported():
    """Not every refusal is a width problem, and silence is not success."""
    node = Node({}, cap=0)
    _found, refused = v2_census.retry_refused(node, [(0, 99)], "0xf", depth=3)
    assert refused, "a window that refuses at every width must survive to main"


def test_the_refused_windows_survive_into_the_cache(tmp_path):
    """The bug: the count was cached and the windows were not, so a resumed
    run refused to write over a slice it could no longer name."""
    node = Node({5: 0x1}, cap=0)       # refuses at every width
    cache = tmp_path / "d.json"
    v2_census.discover(node, 999, 100, "0xf", 0, cache)
    held = json.loads(cache.read_text())
    assert isinstance(held["refused"], list)
    assert len(held["refused"]) >= 10, "every window refused and none recovered"
    assert all(len(w) == 2 for w in held["refused"]), "as blocks, re-askable"


def test_a_batch_refused_wholesale_is_recovered_rather_than_lost(tmp_path):
    """The run this came from: 24 windows refused together, 0 pairs, 182 s.

    Under the old code that was a permanent hole and a cached refusal.  Here
    the same batch narrows, answers, and leaves the census complete.
    """
    node = Node({b: b for b in range(5, 1000, 97)}, cap=50)
    cache = tmp_path / "d.json"
    pairs, refused = v2_census.discover(node, 999, 100, "0xf", 0, cache)
    assert refused == []
    assert len(pairs) == len(range(5, 1000, 97))
    assert json.loads(cache.read_text())["refused"] == []


def test_an_old_cache_counting_refusals_rescans_instead_of_trusting_done():
    """`refused: 24` cannot be re-asked, so `done` cannot be believed either."""
    node = Node({5: 0x1})
    cache = Path(pytest.ensuretemp("c") if False else "/dev/null")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "d.json"
        cache.write_text(json.dumps({
            "factory": "0xf", "span": 100, "done": 900, "refused": 24,
            "pairs": {}}))
        v2_census.discover(node, 999, 100, "0xf", 0, cache)
    # Window 0 is re-asked despite `done: 900`.
    assert (0, 99) in node.asked


def test_an_old_cache_with_no_refusals_still_resumes():
    """Rescanning a clean cache would throw away the hour it exists to save."""
    import tempfile
    node = Node({5: 0x1})
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "d.json"
        cache.write_text(json.dumps({
            "factory": "0xf", "span": 100, "done": 900, "refused": 0,
            "pairs": {"0x" + "11" * 20: ["0xa", "0xb"]}}))
        pairs, _refused = v2_census.discover(node, 999, 100, "0xf", 0, cache)
    assert (0, 99) not in node.asked, "resumed, not rescanned"
    assert "0x" + "11" * 20 in pairs, "the cached pairs are kept"


def test_discovery_is_a_union_so_re_asking_costs_only_time():
    """What makes the rescan above safe rather than merely tolerable."""
    node = Node({5: 0x1, 250: 0x2})
    first, _ = v2_census.discover(node, 999, 100, "0xf", 0, None)
    second, _ = v2_census.discover(node, 999, 100, "0xf", 0, None)
    assert first == second and len(first) == 2


class Reader:
    """Answers `getReserves()`, but refuses any batch over `cap`.

    The endpoint caps batches at 100 and the transport defaulted to 500, so
    every request came back refused and was dropped without a word.
    """

    def __init__(self, reserves: dict, cap: int = 10**9, flaky: int = 0):
        self.reserves, self.cap = reserves, cap
        self.flaky = flaky        # refuse this many reads, once each
        self.refused_once: set = set()

    def fetch_multi(self, requests, concurrent=False):
        if len(requests) > self.cap:
            return [RuntimeError(f"batch limit {self.cap} exceeded")] * len(requests)
        out = []
        for _method, params in requests:
            pool = params[0]["to"]
            if self.flaky and pool not in self.refused_once:
                self.refused_once.add(pool)
                self.flaky -= 1
                out.append(RuntimeError("timeout"))
                continue
            got = self.reserves.get(pool)
            if got is None:
                out.append("0x")          # no code behind it
                continue
            out.append("0x" + got[0].to_bytes(32, "big").hex()
                       + got[1].to_bytes(32, "big").hex()
                       + (0).to_bytes(32, "big").hex())
        return out


def test_a_refused_read_is_not_an_empty_pool():
    """The distinction that decides whether a census is small or short."""
    pools = [f"0x{n:040x}" for n in range(1, 6)]
    reader = Reader({}, cap=0)
    got, unread = v2_census.read_reserves(reader, pools, 1, chunk=10, retries=0)
    assert got == {}
    assert sorted(unread) == sorted(pools), "refusals are named, not dropped"


def test_a_pair_with_no_code_is_counted_but_not_re_asked():
    """It answered.  Asking again gets the same `0x`."""
    pools = [f"0x{n:040x}" for n in range(1, 6)]
    got, unread = v2_census.read_reserves(Reader({}), pools, 1, chunk=10)
    assert got == {} and unread == [], "answered, so nothing to re-ask"


def test_a_flaky_read_is_re_asked_and_recovered():
    pools = [f"0x{n:040x}" for n in range(1, 6)]
    reserves = dict.fromkeys(pools, (10 ** 18, 2000 * 10 ** 6))
    got, unread = v2_census.read_reserves(
        Reader(reserves, flaky=3), pools, 1, chunk=10)
    assert unread == []
    assert len(got) == 5, "every pair recovered on a retry"


def test_the_batch_cap_failure_is_visible_rather_than_silent():
    """514,027 pairs read in 63 s, 27 answered -- and nothing said so.

    The 27 were the final partial chunk, the only request under the cap.
    """
    pools = [f"0x{n:040x}" for n in range(1, 128)]
    reserves = dict.fromkeys(pools, (10 ** 18, 2000 * 10 ** 6))
    reader = Reader(reserves, cap=100)
    got, unread = v2_census.read_reserves(reader, pools, 1, chunk=120, retries=0)
    assert len(unread) == 120, "the oversized chunk is reported, not swallowed"
    assert len(got) == 7, "only the chunk under the cap answered"
