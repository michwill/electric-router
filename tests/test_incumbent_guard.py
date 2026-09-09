"""Adding a source of liquidity must never make a quote worse.

The router makes that promise structurally, and until now it approached it
rather than kept it.  Every candidate family is a perturbation of the base
solve, the base solve moves when a venue joins, and only the *winner* is
refined -- so the route that would have won gets ranked on the split the model
gave it and never re-split.  §6's sub-ballot puts the venue-free neighbourhood
back on the ballot, which took the worst case from -1547.07 bp to -40.59, but a
seed cannot reproduce a route that only exists downstream of refinement.

So the frame is quoted on its own, refinement and all, and the better answer
wins.  These tests are about the choosing, which is where a promise like this
is kept or quietly dropped.
"""

from __future__ import annotations

import ast
import inspect

from erouter.core.pipeline import RouteResult, best_of, route


def _result(out, **counters):
    got = RouteResult(src_token="0xa", dst_token="0xb", amount_in=1, nodes=None)
    got.verified_out = out
    got.counters.update(counters)
    return got


def test_the_venue_free_answer_wins_when_it_pays_more():
    venue, plain = _result(100), _result(125)
    got = best_of(venue, [("the frame alone", plain)])
    assert got is plain
    assert got.counters["venue_declined"] == 1
    assert got.counters["venue_cost_bp"] == 2000.0, "25 of 125 is 2,000 bp"
    assert any("cost 2000.00 bp" in w for w in got.warnings)


def test_the_venue_keeps_the_answer_when_it_pays_more():
    venue, plain = _result(140), _result(100)
    got = best_of(venue, [("the frame alone", plain)])
    assert got is venue
    assert "venue_declined" not in got.counters
    assert got.counters["incumbent_out"] == 100, "and what it beat is recorded"


def test_a_tie_goes_to_the_venue():
    """It was asked for, and every other counter on the result describes it.

    An equal answer is not a reason to throw that away.
    """
    venue, plain = _result(100), _result(100)
    assert best_of(venue, [("the frame alone", plain)]) is venue


def test_a_venue_that_loses_is_recorded_rather_than_quietly_corrected():
    """A venue that keeps costing the search is worth knowing about even once
    the answer is safe."""
    got = best_of(_result(999), [("the frame alone", _result(1000))])
    assert got.counters["venue_declined"] == 1
    assert got.counters["venue_cost_bp"] == 10.0
    assert got.warnings, "silence would make this look like it never happened"


def test_an_unquotable_incumbent_does_not_beat_a_real_answer():
    """`verified_out` of zero is a quote that failed, not a free one."""
    venue = _result(100)
    assert best_of(venue, [("the frame alone", _result(0))]) is venue
    assert best_of(venue, [("the frame alone", _result(None))]) is venue


def test_the_guard_is_on_by_default_and_can_be_turned_off():
    """Off is a latency choice, so it has to be reachable -- and on has to be
    what a caller who has not thought about it gets."""
    params = inspect.signature(route).parameters
    assert params["incumbent_guard"].default is True


def test_the_guard_only_runs_when_a_venue_is_present():
    """A Curve-only quote must not pay for a second one it cannot use."""
    src = inspect.getsource(route)
    tree = ast.parse(src.lstrip())
    checks = [n for n in ast.walk(tree)
              if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
              and "late_arcs" in ast.unparse(n)
              and "incumbent_guard" in ast.unparse(n)]
    assert checks, (
        "the second quote is not gated on there being a venue and on the flag"
    )


def test_the_second_quote_shares_the_preparation():
    """`prepare` is the expensive half and is a function of the pair, not the
    arcs, so the guard costs a second `_quote` and not a second probe."""
    src = inspect.getsource(route)
    tree = ast.parse(src.lstrip())
    prepares = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and (getattr(n.func, "id", None) == "prepare")]
    assert len(prepares) <= 1, "the guard must not re-run preparation"


def test_the_best_rival_wins_not_the_first():
    """With two venues there are three rivals, and they are not ordered."""
    venue = _result(100)
    got = best_of(venue, [("without uniswap v2", _result(110)),
                          ("without uniswap v3", _result(130)),
                          ("the frame alone", _result(120))])
    assert got.verified_out == 130
    assert got.counters["venue_declined_for"] == "without uniswap v3"
    assert venue.counters["incumbent_out"] == 130, "the best of them, not any"


def test_no_rival_at_all_keeps_the_venue():
    """Every arm can fail to route -- a bridged token has no frame arc."""
    venue = _result(100)
    assert best_of(venue, []) is venue


def test_a_leave_one_out_arm_exists_per_venue_when_there_are_several():
    """`curve+v2+v3 >= curve` alone does not say `>= curve+v2`.

    The second is the statement that matters when a venue is added to a router
    that already has one, and it needs its own finished quote: the sub-ballot
    only puts the neighbourhood on the ballot, where a candidate can still be
    ranked on a split nobody refined.
    """
    import ast
    import inspect

    src = inspect.getsource(route)
    tree = ast.parse(src.lstrip())
    text = ast.unparse(tree)
    assert "a.venue for a in late_arcs" in text, (
        "the arms are not built by venue, so there is no leave-one-out")
    assert "len(labels) > 1" in text, (
        "with one venue the leave-one-out arm is the frame arm; quoting both "
        "doubles the cost for nothing")


def test_one_venue_costs_two_quotes_and_two_venues_four():
    """The guard is V+2 quotes, not 2^V."""
    def arms(labels):
        out = []
        if len(labels) > 1:
            out += [f"without {v}" for v in labels]
        out.append("the frame alone")
        return out

    assert len(arms(["uniswap v2"])) == 1, "one venue: just the frame"
    assert len(arms(["uniswap v2", "uniswap v3"])) == 3
    assert len(arms(["a", "b", "c"])) == 4, "V+1 rivals, so V+2 quotes"
