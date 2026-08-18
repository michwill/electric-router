"""A remembered verdict must never be believed harder than a fresh one.

The cache exists so a pool that has already reproduced its own `get_dy` is not
made to prove it again: 2,406 gate probes down to 84.  That is all it saves --
it does *not* let the warm skip those pools, which are computed from the very
storage the sweep fetches (see `test_startup_cost.py`).

A wrong verdict would be believed silently, so the properties worth pinning are
the ones that keep it honest:

* editing the maths discards every verdict, because the verdict is a claim
  about *this* arithmetic agreeing with the chain;
* a verdict is matched against the variants actually built this run, never used
  to construct one -- so a remembered variant that is no longer on offer falls
  back to the gate instead of being resurrected.

The second is not hypothetical.  On the production endpoint `stored_rates()` is
refused, so a rate-bearing pool cannot build its `reported` variant at all.  If
its cached verdict were trusted it would be admitted with decimals-only rates
and quote sUSDe as worth exactly one DOLA -- a factor of 1.2427.  Instead the
verdict finds no match, the pool is gated, it fails, and it stays in the set
that gets warmed and read properly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from erouter.dev.exact_cache import ExactCache, math_fingerprint, trust

STABLE = {"family": "stable", "rates": "reported", "fee_on_xp": True}
PLAIN = {"family": "stable", "rates": "plain", "fee_on_xp": True}
POOL = "0x" + "ab" * 20


@dataclass
class FakeOut:
    by_pool: dict = field(default_factory=dict)
    trusted: int = 0


@dataclass
class FakePool:
    address: str


def built_with(*variants):
    """`(pool, model, variant)` triples, the shape the readers produce."""
    pool = FakePool(POOL)
    return [(pool, f"model-{v['rates']}", v) for v in variants]


def test_a_matching_verdict_admits_without_gating(tmp_path: Path):
    cache = ExactCache.load(1, "t", tmp_path)
    cache.record(POOL, STABLE)
    out = FakeOut()

    assert trust(out, cache, set(), built_with(STABLE, PLAIN), POOL.lower())
    assert out.by_pool[POOL.lower()] == "model-reported"
    assert out.trusted == 1


def test_a_verdict_for_a_variant_no_longer_built_falls_back_to_the_gate(tmp_path: Path):
    """The property that keeps a refused `stored_rates()` from being believed."""
    cache = ExactCache.load(1, "t", tmp_path)
    cache.record(POOL, STABLE)          # remembered: the reported-rates variant
    out = FakeOut()

    # ...but this run could not read the rates, so only `plain` exists.
    assert not trust(out, cache, set(), built_with(PLAIN), POOL.lower())
    assert not out.by_pool, "admitted a pool on a variant it could not build"
    assert out.trusted == 0


def test_resampling_forces_a_re_gate(tmp_path: Path):
    cache = ExactCache.load(1, "t", tmp_path)
    cache.record(POOL, STABLE)
    out = FakeOut()

    assert not trust(out, cache, {POOL.lower()}, built_with(STABLE), POOL.lower())
    assert not out.by_pool


def test_an_unknown_pool_is_never_trusted(tmp_path: Path):
    cache = ExactCache.load(1, "t", tmp_path)
    out = FakeOut()
    assert not trust(out, cache, set(), built_with(STABLE), POOL.lower())
    assert not trust(out, None, set(), built_with(STABLE), POOL.lower())


def test_verdicts_survive_a_round_trip(tmp_path: Path):
    cache = ExactCache.load(1, "ethereum", tmp_path)
    cache.record(POOL, STABLE)
    cache.save()

    again = ExactCache.load(1, "ethereum", tmp_path)
    assert again.get(POOL) == STABLE
    assert len(again) == 1


def test_refusals_are_never_written(tmp_path: Path):
    """The file is committed, so it must change only when a verdict does.

    A refusal reason carries the mismatching wei, which moves with the block,
    so persisting it rewrote the file on every route -- a dirty working tree
    after running a read-only command.  Nothing reads them back: a pool absent
    from `verdicts` is re-gated either way.
    """
    cache = ExactCache.load(1, "ethereum", tmp_path)
    cache.record(POOL, STABLE)
    cache.refuse("0x" + "cd" * 20, "23350204197227025 != 21327129196346614")
    cache.save()
    first = (tmp_path / "ethereum.json").read_text()

    again = ExactCache.load(1, "ethereum", tmp_path)
    assert not again.refused, "a refusal was persisted"
    # A second run refusing the same pool with different numbers, as a moving
    # block does, must leave the bytes alone.
    again.refuse("0x" + "cd" * 20, "99999999999999999 != 11111111111111111")
    again.save()
    assert (tmp_path / "ethereum.json").read_text() == first


def test_editing_the_maths_discards_every_verdict(tmp_path: Path, monkeypatch):
    """A verdict is a claim about *this* arithmetic; change it and it is void."""
    cache = ExactCache.load(1, "ethereum", tmp_path)
    cache.record(POOL, STABLE)
    cache.save()
    assert len(ExactCache.load(1, "ethereum", tmp_path)) == 1

    monkeypatch.setattr("erouter.dev.exact_cache.math_fingerprint",
                        lambda: "0000000000000000")
    assert len(ExactCache.load(1, "ethereum", tmp_path)) == 0


def test_recording_clears_a_previous_refusal_and_vice_versa(tmp_path: Path):
    cache = ExactCache.load(1, "t", tmp_path)
    cache.refuse(POOL, "mismatch")
    cache.record(POOL, STABLE)
    assert cache.get(POOL) == STABLE and POOL.lower() not in cache.refused

    cache.refuse(POOL, "mismatch again")
    assert cache.get(POOL) is None and POOL.lower() in cache.refused


def test_the_fingerprint_is_stable_and_not_empty():
    assert math_fingerprint() == math_fingerprint()
    assert len(math_fingerprint()) == 16
