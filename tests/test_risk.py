"""Per-pool minimum-out risk, and what it does to ranking.

The router prices a route as `output * P(lands) - gas`, where `P(lands)` is the
chance every leg's minimum-out -- a fraction of that pool's fee -- still holds
when the transaction is mined a minute or two later.  These tests fix the two
halves of that: the table (`core/risk.py`) and the estimator that fills it
(`dev/revert_risk.py`).
"""

from __future__ import annotations

import math
import random

import numpy as np

from erouter.core.candidates import Candidate, CandidateSet
from erouter.core.realize import RealizedLeg, RealizedRoute
from erouter.core.risk import DEFAULT_RISK, RiskTable
from erouter.core.types import ArcKind, Leg
from erouter.core.verify import verify
from erouter.dev.drift import PriceSeries
from erouter.dev.revert_risk import breach_risk, jump_scale_bp, tail_model

POOL = ["0x" + f"{k:02x}" * 20 for k in range(1, 9)]


class FakeClient:
    def __init__(self, outs):
        self.outs = list(outs)
        self.calls = 0

    def quote_routes(self, routes, amounts_in, dst_slots):
        self.calls += 1
        return self.outs[: len(routes)]


def _candidate(label, targets, out=None, kind=ArcKind.SWAP_STABLE):
    """One candidate whose legs hop through `targets` in order."""
    route = RealizedRoute(dst_slot=1)
    for k, target in enumerate(targets):
        route.legs.append(
            RealizedLeg(
                leg=Leg(target, kind, src_slot=k, dst_slot=k + 1),
                kind=kind, target=target,
                token_in="0xa", token_out="0xb", amount_in=1, amount_out=1,
            )
        )
    candidate = Candidate(label=label, psi=np.array([1.0]), certificate=False)
    candidate.route = route
    candidate.status = "ready"
    if out is not None:
        candidate.verified_out = out
    return candidate


# --------------------------------------------------------------- the table


def test_survival_is_the_product_over_legs():
    table = RiskTable({(POOL[0], 0, 1): 0.25, (POOL[1], 0, 1): 0.1})
    legs = [Leg(POOL[0], ArcKind.SWAP_STABLE, src_slot=0, dst_slot=1),
            Leg(POOL[1], ArcKind.SWAP_CRYPTO, src_slot=1, dst_slot=2)]
    assert table.survival(legs) == 0.75 * 0.9


def test_an_unmeasured_pool_is_not_a_free_one():
    """The one thing a missing measurement cannot say is "never moves"."""
    table = RiskTable({(POOL[0], 0, 1): 0.0})
    priced = Leg(POOL[0], ArcKind.SWAP_STABLE, src_slot=0, dst_slot=1)
    unknown = Leg(POOL[5], ArcKind.SWAP_STABLE, src_slot=0, dst_slot=1)
    assert table.survival([priced]) == 1.0
    assert table.survival([unknown]) == 1.0 - DEFAULT_RISK


def test_an_unsampled_direction_inherits_its_pool_not_the_default():
    """The `(-1, -1)` tier: a pair the sweep missed on a pool it knows is
    priced from that pool, erring high."""
    table = RiskTable({(POOL[0], 0, 1): 0.2, (POOL[0], -1, -1): 0.2})
    missed = Leg(POOL[0], ArcKind.SWAP_STABLE, i=1, j=2, n=3,
                 src_slot=0, dst_slot=1)
    assert table.survival([missed]) == 0.8


def test_arcs_of_one_pool_are_priced_separately():
    """TriCRV's CRV/ETH pair moves several times as far as its crvUSD one, and
    the minimum-out is written per leg."""
    table = RiskTable({(POOL[0], 0, 1): 0.01, (POOL[0], 1, 2): 0.4})
    quiet = Leg(POOL[0], ArcKind.SWAP_CRYPTO, i=0, j=1, n=3, src_slot=0, dst_slot=1)
    loud = Leg(POOL[0], ArcKind.SWAP_CRYPTO, i=1, j=2, n=3, src_slot=0, dst_slot=1)
    assert table.survival([quiet]) == 0.99
    assert table.survival([loud]) == 0.6


def test_wraps_and_redemptions_carry_no_rate_risk():
    """A wrap, a stake or a vault redemption is priced by accrual, which moves
    upward and slowly.  No minimum-out of this kind can trigger on it, so
    charging one would tax exactly the legs that cannot fail."""
    table = RiskTable({(POOL[0], 0, 1): 0.5}, default=0.5)
    for kind in (ArcKind.WRAP_NATIVE, ArcKind.WSTETH_WRAP, ArcKind.STAKE_NATIVE,
                 ArcKind.LEND_REDEEM, ArcKind.ERC4626_DEPOSIT):
        leg = Leg(POOL[0], kind, src_slot=0, dst_slot=1)
        assert table.survival([leg]) == 1.0


# ------------------------------------------------------------- the ranking


def test_a_route_through_a_dangerous_pool_loses_its_advantage():
    """The measured case.  USDC->WETH gains 71 bp by splitting through TriCRV
    and TricryptoUSDC, which breach a quarter and a sixth of the time: about
    62% survival, so the expected outcome is far below a single safe hop."""
    risk = RiskTable({(POOL[0], 0, 1): 1e-5, (POOL[1], 0, 1): 0.25,
                      (POOL[2], 0, 1): 0.167})
    candidates = CandidateSet([
        _candidate("split", [POOL[1], POOL[2]]),
        _candidate("direct", [POOL[0]]),
    ])
    verify(candidates, FakeClient([1_007_100, 1_000_000]), amount_in=10**6,
           risk_table=risk)
    assert candidates.best.label == "direct"
    assert 0.6 < candidates.candidates[0].survival < 0.64


def test_a_long_route_through_safe_pools_keeps_its_gain():
    """The other half of the argument: risk is not a leg budget.  Twelve legs
    through pools whose fees are wide against their volatility cost almost
    nothing, and a 30 bp gain still wins."""
    safe = {(p, 0, 1): 1e-5 for p in POOL}
    candidates = CandidateSet([
        _candidate("long", [POOL[k % 8] for k in range(12)]),
        _candidate("short", [POOL[0]]),
    ])
    verify(candidates, FakeClient([1_003_000, 1_000_000]), amount_in=10**6,
           risk_table=RiskTable(safe))
    assert candidates.best.label == "long"


def test_gas_is_charged_whether_or_not_the_route_lands():
    """Expected output is `out * survival - gas`, not `(out - gas) * survival`:
    a reverted transaction still pays for its execution."""
    risk = RiskTable({(POOL[0], 0, 1): 0.5})
    candidates = CandidateSet([_candidate("one", [POOL[0]])])
    verify(candidates, FakeClient([10**18]), amount_in=10**18,
           gas_price_wei=10**9, dst_wei_per_eth=10**18, risk_table=risk)
    candidate = candidates.candidates[0]
    assert candidate.survival == 0.5
    # 71,000 base + 102,000 for the swap, at 1 gwei.
    assert candidate.gas == 173_000


def test_no_table_means_no_risk_pricing():
    """Ranking must be unchanged when nothing has been measured."""
    candidates = CandidateSet([
        _candidate("long", [POOL[k % 8] for k in range(6)]),
        _candidate("short", [POOL[0]]),
    ])
    verify(candidates, FakeClient([1_000_100, 1_000_000]), amount_in=10**6)
    assert candidates.best.label == "long"
    assert candidates.candidates[0].survival == 1.0


# ------------------------------------------------------------ the estimator


def _walk(steps: int, sigma_bp: float, seed: int) -> list[float]:
    """A rate that moves every sample -- a busy pool."""
    rng = random.Random(seed)
    rate, out = 1.0, []
    for _ in range(steps):
        out.append(rate)
        rate *= math.exp(rng.gauss(0.0, sigma_bp / 1e4))
    return out


def _quiet_walk(steps: int, jump_bp: float, every: int, seed: int) -> list[float]:
    """A rate that holds still and jumps when someone trades, which is what a
    pool's rate actually does -- not a diffusion."""
    rng = random.Random(seed)
    rate, out = 1.0, []
    for k in range(steps):
        out.append(rate)
        if k % every == 0:
            rate *= math.exp(rng.gauss(0.0, jump_bp / 1e4))
    return out


def _arc(key, pool, i, j):
    return (key, pool, int(ArcKind.SWAP_STABLE), i, j, 2, 1000)


def test_a_bound_far_outside_the_data_is_priced_far_below_one_in_n():
    """The reason the tail is modelled rather than counted.  Two dozen windows
    can only resolve down to 4%, and the honest smoothing of a zero count --
    Jeffreys' 2% -- reads as a 22% loss over a twelve-leg route.  A pool whose
    bound sits far outside anything it has ever done is not that."""
    key = "a|b@" + POOL[0]
    series = {key: PriceSeries(token=key, pool=POOL[0],
                               prices=_quiet_walk(25, 0.05, 3, seed=1))}
    # 146 bp fee, as Yield Basis charges: a 29 bp bound on a 0.05 bp jump.
    out = breach_risk(series, {POOL[0].lower(): 0.0146},
                      [_arc(key, POOL[0], 0, 1)])
    assert out[POOL[0].lower() + ":0>1"]["p"] < 1e-3


def test_a_bound_inside_the_typical_move_is_priced_from_the_count():
    """TriCRV's case: a 0.83 bp bound against a rate that moves further than
    that most minutes.  Here the data speaks for itself and no extrapolation
    should talk it down."""
    key = "a|b@" + POOL[1]
    series = {key: PriceSeries(token=key, pool=POOL[1],
                               prices=_walk(25, 2.0, seed=2))}
    out = breach_risk(series, {POOL[1].lower(): 0.000336},
                      [_arc(key, POOL[1], 0, 1)])
    assert out[POOL[1].lower() + ":0>1"]["p"] > 0.2


def test_each_direction_of_a_pool_is_priced_on_its_own():
    """The minimum-out is written per leg, so the quiet pair of a pool with a
    loud one must not inherit its risk."""
    address = POOL[2].lower()
    quiet, loud = "a|b@" + address, "b|c@" + address
    series = {
        quiet: PriceSeries(token=quiet, pool=address,
                           prices=_quiet_walk(25, 0.01, 4, seed=3)),
        loud: PriceSeries(token=loud, pool=address, prices=_walk(25, 4.0, seed=4)),
    }
    out = breach_risk(series, {address: 0.0001},  # 1 bp fee, 0.2 bp bound
                      [_arc(quiet, address, 0, 1), _arc(loud, address, 1, 2)])
    assert out[address + ":1>2"]["p"] > 0.2
    assert out[address + ":0>1"]["p"] < 0.05


def test_a_pool_that_does_not_report_a_fee_is_left_unmeasured():
    """Without its own fee there is no bound, and an invented one would be a
    claim.  Absent means the default, which is the conservative answer."""
    key = "a|b@" + POOL[3]
    series = {key: PriceSeries(token=key, pool=POOL[3],
                               prices=_walk(25, 1.0, seed=5))}
    assert breach_risk(series, {}, [_arc(key, POOL[3], 0, 1)]) == {}


def test_the_scale_is_the_size_of_a_move_not_of_a_window():
    """Most windows contain no trade at all.  Their zeros say how often a pool
    is quiet, which is counted separately; letting them into the scale would
    put it at the floor and turn every real trade into a 400-sigma event -- the
    bug that priced 3pool, which never breached, at 4.5% a leg."""
    mostly_quiet = [0.0] * 18 + [0.3, 0.4, 0.5]
    assert jump_scale_bp(mostly_quiet) == 0.4


def test_the_tail_is_empirical_inside_the_data():
    sample = [float(k) / 10 for k in range(1, 1001)]  # uniform on (0, 100]
    tail = tail_model(sample)
    assert abs(tail(50.0) - 0.5) < 0.02
    # ... and continues past its edge rather than falling off it.
    assert 0 < tail(200.0) < tail(100.0)


def test_the_worst_arc_is_what_an_unsampled_direction_inherits(tmp_path):
    """`FactsCache.risk_table` fills the `(-1, -1)` tier, so a pair the sweep
    missed is priced from its own pool rather than from the global default."""
    from erouter.dev.facts import FactsCache

    cache = FactsCache(chain_id=1, path=tmp_path / "x.json")
    cache.learn_breach({f"{POOL[0].lower()}:0>1": {"p": 0.02},
                        f"{POOL[0].lower()}:1>2": {"p": 0.31}})
    table = cache.risk_table()
    missed = Leg(POOL[0], ArcKind.SWAP_CRYPTO, i=2, j=0, n=3,
                 src_slot=0, dst_slot=1)
    assert table.of(ArcKind.SWAP_CRYPTO, POOL[0], 0, 1) == 0.02
    assert table.survival([missed]) == 1.0 - 0.31


def test_a_deposit_leg_takes_the_pool_figure_not_a_swap_pair():
    """`(i, j)` are coin indices on a swap and something else on a deposit, so
    reading the specific tier for one would hand it whichever swap happened to
    share the numbers."""
    table = RiskTable({(POOL[0], 0, 1): 0.4, (POOL[0], -1, -1): 0.02})
    deposit = Leg(POOL[0], ArcKind.DEPOSIT_FIXED, i=0, j=1, n=3,
                  src_slot=0, dst_slot=1)
    assert table.survival([deposit]) == 0.98
