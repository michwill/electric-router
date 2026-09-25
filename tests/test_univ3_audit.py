"""The winner is ranked on a number the bank produced, so the bank is checked.

Every other venue is ranked on the chain's own answer: `verify` re-quotes each
candidate and a Curve leg is priced by `get_dy`.  A v3 leg has no deployed
quoter the walk can chain, so `teach` answers it from the bank -- the same
arithmetic that proposed the leg.  `audit` closes that by re-walking the winner
with `QuoterV2` standing in, and these hold it to actually refusing one.
"""

from __future__ import annotations

import pytest

from erouter.core.candidates import Candidate, CandidateSet
from erouter.core.codec import encode
from erouter.core.realize import RealizedLeg, RealizedRoute
from erouter.core.types import ArcKind, Leg
from erouter.venues.univ3_client import AUDIT_TOLERANCE_BP, audit

POOL = "0x" + "88" * 20
TOKEN_IN = "0x" + "a0" * 20
TOKEN_OUT = "0x" + "c0" * 20
AMOUNT = 100 * 10**6


class Transport:
    """Answers `quoteExactInputSingle` with whatever it was told to say."""

    def __init__(self, dy: int) -> None:
        self.dy = dy
        self.calls = 0

    def fetch(self, method, params):
        assert method == "eth_call"
        self.calls += 1
        body = encode(["uint256", "uint160", "uint32", "uint256"],
                      [self.dy, 0, 0, 0])
        return "0x" + body.hex()


class Client:
    """Nothing but the hook `audit` needs; no non-v3 leg reaches it."""

    def _stateful_leg(self, legs):
        def quote_leg(leg, dx):
            raise AssertionError("only the v3 leg should be priced here")
        return quote_leg


def one_v3_candidate(ranked: int) -> CandidateSet:
    leg = Leg(target=POOL, kind=ArcKind.SWAP_UNIV3, i=0, j=1, n=2,
              src_slot=0, dst_slot=1, bps=0)
    route = RealizedRoute(
        legs=[RealizedLeg(leg=leg, kind=ArcKind.SWAP_UNIV3, target=POOL,
                          token_in=TOKEN_IN, token_out=TOKEN_OUT,
                          amount_in=AMOUNT, amount_out=ranked)],
        slots={TOKEN_IN: 0, TOKEN_OUT: 1}, dst_slot=1,
        src_token=TOKEN_IN, dst_token=TOKEN_OUT, amount_in=AMOUNT,
    )
    candidate = Candidate(label="the bank's answer", psi=None, certificate=False)
    candidate.route = route
    candidate.verified_out = ranked
    candidate.status = "ok"
    candidate.rank = 0
    return CandidateSet([candidate])


POOLS = {POOL.lower(): (TOKEN_IN, TOKEN_OUT, 500, 6, 18)}


def run(ranked: int, truth: int):
    pool_set = one_v3_candidate(ranked)
    transport = Transport(truth)
    rows = audit(pool_set, Client(), transport, POOLS, block=1)
    return pool_set, rows


def test_a_bank_that_agrees_with_the_pool_is_kept():
    ranked = 40 * 10**18
    pool_set, rows = run(ranked, ranked)
    assert len(rows) == 1 and rows[0][4] is True
    assert pool_set.best is not None
    assert pool_set.best.verified_out == ranked


def test_a_bank_that_drifts_loses_the_candidate_it_priced():
    """One percent out is two hundred times the tolerance, and far less than a
    real defect: the measured banks land inside 0.01 bp of `QuoterV2`."""
    ranked = 40 * 10**18
    pool_set, rows = run(ranked, int(ranked / 1.01))
    _label, _got, _truth, bp, kept = rows[0]
    assert kept is False
    assert bp == pytest.approx(100.0, rel=0.02)      # 1% is 100 bp
    assert pool_set.best is None, "a refused winner must not still win"
    assert pool_set.candidates[0].status == "univ3 audit"
    assert "quoter" in pool_set.candidates[0].note


def test_a_quoter_that_will_not_answer_refuses_the_candidate():
    """Silence is not agreement -- it is the case the audit exists for."""

    class Mute(Transport):
        def fetch(self, method, params):
            raise RuntimeError("no quoter on this chain")

    pool_set = one_v3_candidate(40 * 10**18)
    rows = audit(pool_set, Client(), Mute(0), POOLS, block=1)
    assert rows[0][4] is False
    assert pool_set.best is None


def test_a_route_with_no_v3_leg_is_not_audited():
    """The Curve legs were already ranked on the chain's own answer."""
    leg = Leg(target=POOL, kind=ArcKind.SWAP_STABLE, i=0, j=1, n=2,
              src_slot=0, dst_slot=1, bps=0)
    route = RealizedRoute(
        legs=[RealizedLeg(leg=leg, kind=ArcKind.SWAP_STABLE, target=POOL,
                          token_in=TOKEN_IN, token_out=TOKEN_OUT,
                          amount_in=AMOUNT, amount_out=7)],
        slots={TOKEN_IN: 0, TOKEN_OUT: 1}, dst_slot=1,
        src_token=TOKEN_IN, dst_token=TOKEN_OUT, amount_in=AMOUNT,
    )
    candidate = Candidate(label="all curve", psi=None, certificate=False)
    candidate.route, candidate.verified_out = route, 7
    candidate.status, candidate.rank = "ok", 0
    pool_set = CandidateSet([candidate])
    transport = Transport(0)
    assert audit(pool_set, Client(), transport, POOLS, block=1) == []
    assert transport.calls == 0, "no v3 leg, no reason to ask"


def test_the_tolerance_is_the_one_the_measurements_support():
    """Every v3 leg of a winning route in the 21-case sweep landed inside
    1.7 bp of `QuoterV2`, and most inside 0.01."""
    assert 1.7 < AUDIT_TOLERANCE_BP < 50


class Batching(Transport):
    """A node that also takes a batch; counts round trips of each kind."""

    def __init__(self, dy: int) -> None:
        super().__init__(dy)
        self.batches = 0

    def fetch_multi(self, payloads):
        self.batches += 1
        return [self.fetch(method, params) for method, params in payloads]


class BankedClient(Client):
    """A client whose bank answers the v3 leg, as `teach` makes it."""

    def __init__(self, bank_out: int) -> None:
        self.bank_out = bank_out

    def _quote_leg(self, leg, dx):
        return self.bank_out


def test_an_agreeing_bank_is_audited_in_one_round_trip():
    """Ten sequential `eth_call`s were 1.2 s of a 2.1 s quote; a bank that
    agrees with its pool needs none of them, only one batch."""
    pool_set = one_v3_candidate(12_345)
    transport = Batching(12_345)
    rows = audit(pool_set, BankedClient(12_345), transport, POOLS, block=1)
    assert rows[0][4] is True and rows[0][2] == 12_345
    assert transport.batches == 1 and transport.calls == 1   # the batch's one call, no walk


def test_a_disagreeing_bank_is_settled_from_the_batch_and_refused():
    """The batch already holds the quoter's answer at the input the leg sees,
    so a second walk from it is the truth walk, with no call per leg."""
    pool_set = one_v3_candidate(12_345)
    transport = Batching(11_000)
    rows = audit(pool_set, BankedClient(12_345), transport, POOLS, block=1)
    assert transport.batches == 1 and transport.calls == 1
    assert rows[0][2] == 11_000
    assert rows[0][4] is False and pool_set.candidates[0].verified_out is None


POOL_2 = "0x" + "99" * 20
TOKEN_MID = "0x" + "b0" * 20
RATES = {POOL.lower(): (2.0, 1.99), POOL_2.lower(): (3.0, 3.0)}   # (bank, quoter)


class Pricing(Batching):
    """QuoterV2 at a rate per pool, read from the call it is handed."""

    def __init__(self) -> None:
        super().__init__(0)

    def fetch(self, method, params):
        from erouter.core.codec import decode

        self.calls += 1
        raw = bytes.fromhex(params[0]["data"][2:])[4:]
        token_in, _, dx = decode(["address", "address", "uint256"], raw[:96])
        pool = POOL if token_in.lower() == TOKEN_IN else POOL_2
        body = encode(["uint256", "uint160", "uint32", "uint256"],
                      [int(dx * RATES[pool.lower()][1]), 0, 0, 0])
        return "0x" + body.hex()


class RatedClient(Client):
    def _quote_leg(self, leg, dx):
        return int(dx * RATES[leg.target.lower()][0])


def test_a_leg_downstream_of_a_disagreement_is_asked_again():
    """The first bank is half a percent rich, so the second leg's input moves
    and it is asked again -- two batches, and the answer is the one a call
    per leg in series would give."""
    legs = [Leg(target=pool, kind=ArcKind.SWAP_UNIV3, i=0, j=1, n=2,
                src_slot=k, dst_slot=k + 1, bps=0) for k, pool in enumerate((POOL, POOL_2))]
    tokens = (TOKEN_IN, TOKEN_MID, TOKEN_OUT)
    route = RealizedRoute(
        legs=[RealizedLeg(leg=leg, kind=ArcKind.SWAP_UNIV3, target=leg.target,
                          token_in=tokens[k], token_out=tokens[k + 1],
                          amount_in=AMOUNT, amount_out=1) for k, leg in enumerate(legs)],
        slots={t: k for k, t in enumerate(tokens)}, dst_slot=2,
        src_token=TOKEN_IN, dst_token=TOKEN_OUT, amount_in=AMOUNT,
    )
    candidate = Candidate(label="two v3 legs", psi=None, certificate=False)
    candidate.route, candidate.verified_out = route, int(int(AMOUNT * 2.0) * 3.0)
    candidate.status, candidate.rank = "ok", 0
    pools = {POOL.lower(): (TOKEN_IN, TOKEN_MID, 500, 6, 18),
             POOL_2.lower(): (TOKEN_MID, TOKEN_OUT, 500, 18, 18)}
    transport = Pricing()
    rows = audit(CandidateSet([candidate]), RatedClient(), transport, pools, block=1)
    assert rows[0][2] == int(int(AMOUNT * 1.99) * 3.0)
    assert transport.batches == 2 and transport.calls == 3
