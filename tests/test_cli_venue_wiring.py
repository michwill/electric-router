"""Every path that quotes has to receive the venue, not just some of them.

`--univ3` reads the census, fetches the ticks, prints how many it got -- and
then the interactive loop quoted without them for as long as the flag existed.
The boot line said `uniswap v3: 146 pool(s), 2,419 tick-arc(s)`, so the failure
looked like Uniswap losing on the merits rather than never being asked.
Measured on `WETH -> sDOLA` at 1,000: **1,690,935 sDOLA interactively against
1,750,896 one-shot**, same block, same flags -- 354 bp.

Reading the source is the test on purpose.  Exercising it needs a chain, and
the thing that broke is not behaviour under load but a splat missing from one
call of two, which is exactly what a reader misses and a parser does not.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from erouter.dev import cli

SOURCE = Path(inspect.getfile(cli)).read_text()
TREE = ast.parse(SOURCE)


def _function(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone; this test needs rewriting")


def _route_calls(fn):
    """Every `route(...)` in `fn`, as (line, {names splatted into it})."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "route":
            continue
        splats = {
            kw.value.id for kw in node.keywords
            if kw.arg is None and isinstance(kw.value, ast.Name)
        }
        out.append((node.lineno, splats))
    return out


def test_every_interactive_quote_is_given_the_venue():
    """One of the two `route(` calls had `**route_opts` and not `**venue_opts`.

    The re-quote below it had both, so the flag worked whenever the confirmation
    path happened to run and silently did nothing otherwise.
    """
    calls = _route_calls(_function("_interactive"))
    assert calls, "no route() call found in _interactive"
    missing = [line for line, splats in calls if "venue_opts" not in splats]
    assert not missing, (
        f"route() at line(s) {missing} quotes without **venue_opts; "
        "a venue that is read and not routed is worse than one that is off"
    )


def test_the_one_shot_path_is_given_the_venue_too():
    calls = _route_calls(_function("cmd_route"))
    assert calls, "no route() call found in cmd_route"
    for line, splats in calls:
        assert "venue_opts" in splats, f"route() at line {line} drops the venue"


@pytest.mark.parametrize("name", ["_interactive", "cmd_route"])
def test_the_venue_is_never_folded_into_the_reference_prices(name):
    """A venue's arcs belong in the graph, never in §4's fit.

    146 pools at 32 tick-arcs each outvote every Curve pool in a weighted least
    squares that has no idea they are one pool.  Through `extra_arcs` that cost
    9.50 bp on `crvUSD -> sDOLA`.

    `prepare` does now take `late_arcs`, and takes them *for connectivity
    alone* -- it never fits them and never returns them, which
    `test_late_arc_bridging.py` pins on the two functions that use them.  What
    must not happen is the venue reaching `extra_arcs`, and the way that would
    happen is a splat: `**venue_opts` carries `late_arcs` and `extra_arcs`
    together, and `prepare` would quietly fit both.
    """
    for node in ast.walk(_function(name)):
        if not isinstance(node, ast.Call):
            continue
        called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if called != "prepare":
            continue
        splats = {kw.arg for kw in node.keywords}
        assert None not in splats, "prepare must not be splatted the venue kwargs"
        for kw in node.keywords:
            if kw.arg == "extra_arcs":
                assert "late_arcs" not in ast.unparse(kw.value), (
                    "the venue's arcs are being handed to the fit")


@pytest.mark.parametrize("fn,teacher", [
    ("_venue_options", "teach"),
    ("_univ2_options", "teach"),
])
def test_a_venue_that_adds_arcs_also_teaches_the_client(fn, teacher):
    """Arcs in the graph and a client that cannot price them is a silent loss.

    `verify` re-quotes every candidate through the deployed quoter, which has
    never heard of `SWAP_UNIV3` or `SWAP_UNIV2`.  An untaught client answers
    zero, `verify` reads zero as a revert, and each route carrying the venue is
    dropped before it can win -- which is indistinguishable in the output from
    the venue simply losing.  `--univ2` shipped this way for one commit.
    """
    node = _function(fn)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    adds_arcs = any(
        isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.Constant)
        and n.slice.value == "late_arcs"
        for n in ast.walk(node))
    assert adds_arcs, f"{fn} no longer adds late_arcs; this test needs rewriting"
    taught = [c for c in calls
              if (getattr(c.func, "attr", None) or getattr(c.func, "id", None))
              == teacher]
    assert taught, f"{fn} adds arcs the client is never taught to price"


def test_the_venue_options_take_the_client_to_teach_it():
    """A helper that cannot reach the client cannot teach it."""
    for fn in ("_venue_options", "_univ2_options"):
        params = [a.arg for a in _function(fn).args.args]
        assert "client" in params, f"{fn} has no client to teach"


def test_the_two_venues_concatenate_their_arcs():
    """Both use `late_arcs`, so the second must not replace the first.

    With `--univ3 --univ2` together, an assignment rather than a concatenation
    silently routes one venue only, and the boot line for the other still says
    how many arcs it read.
    """
    for node in ast.walk(_function("_univ2_options")):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "late_arcs"):
            continue
        assert isinstance(node.value, (ast.List, ast.BinOp)), (
            "late_arcs must be built from what is already there")
        if isinstance(node.value, ast.List):
            assert any(isinstance(e, ast.Starred) for e in node.value.elts), (
                "late_arcs is assigned a fresh list; v3's arcs are dropped")
            return
    raise AssertionError("no late_arcs assignment found in _univ2_options")


def test_the_endpoints_are_introduced_before_routability_is_judged():
    """`nodes.has()` decides whether a token is routable at all.

    A token only Uniswap holds has no node, so the check refuses it -- which is
    the whole reason `X -> USDT -> crvUSD` could not be quoted.  Bridging has
    to run first or it may as well not exist.
    """
    fn = _function("cmd_route")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "_introduce_endpoints":
                lines["introduce"] = node.lineno
        # `not nodes.has(src) or not nodes.has(dst)` -- the refusal.
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            src = ast.unparse(node)
            if "nodes.has(src)" in src and "nodes.has(dst)" in src:
                lines["refuse"] = node.lineno
    assert "introduce" in lines, "cmd_route never tries to bridge an endpoint"
    assert "refuse" in lines, "the routability check is gone; rewrite this test"
    assert lines["introduce"] < lines["refuse"], (
        "bridging runs after the token has already been refused")


def test_bridging_runs_before_the_venues_are_built():
    """`wanted()` filters on the node map, so the nodes must exist first.

    Otherwise the pairs reaching the new token are dropped at read time and the
    venue would have to be read twice.
    """
    fn = _function("cmd_route")
    at = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in ("_introduce_endpoints", "_venue_options", "_univ2_options"):
            at.setdefault(name, node.lineno)
    assert at.get("_introduce_endpoints", 1e9) < at.get("_venue_options", 0)
    assert at.get("_introduce_endpoints", 1e9) < at.get("_univ2_options", 0)
