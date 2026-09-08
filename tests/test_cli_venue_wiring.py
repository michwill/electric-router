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
    """`prepare` takes `extra_arcs` only.

    A venue's arcs belong in the graph and not in §4's fit -- 146 pools at 32
    tick-arcs each outvote every Curve pool in a weighted least squares that has
    no idea they are one pool.  Through `extra_arcs` that cost 9.50 bp on
    `crvUSD -> sDOLA`; `late_arcs` is the seam that exists for it.
    """
    for node in ast.walk(_function(name)):
        if not isinstance(node, ast.Call):
            continue
        called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if called != "prepare":
            continue
        splats = {kw.arg for kw in node.keywords}
        assert "late_arcs" not in splats
        assert None not in splats, "prepare must not be splatted the venue kwargs"
