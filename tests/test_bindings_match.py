"""Whatever PyO3 exposes, wasm must expose too -- and the browser shim must
expose whatever the package asks for.

The port has now grown the same hole three times.  `pools` compiled to wasm32 for
months while nothing under `rust/wasm/src/` referenced it, so the linker
stripped it and a browser could not price a pool; `ladders` was added with a
PyO3 binding only and repeated it within the same session.  Then `Graph`,
`Arcs`, `Ballot` and `Pools` reached `core` and `chain` while
`erouter/wasm/_solve.py` stayed as it was, and every browser quote raised
`AttributeError` -- the Rust and the wheel were both fine.  All three were
found by reading, which is not a method.

So this reads the two binding files and compares what they name.  It is a
structural test, not a behavioural one -- `test_wasm_differential` covers
whether they *answer* the same -- and it exists because the failure mode is
silent: nothing errors, the browser simply cannot do something the extension
can.

A binding that is deliberately one-sided goes in `PY_ONLY` with the reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "rust" / "src"
WASM = ROOT / "rust" / "wasm" / "src"

#: Bindings that exist on one side on purpose.
PY_ONLY: dict[str, str] = {}

#: `snake_case` on the PyO3 side, `camelCase` on the wasm side, and a few that
#: are spelled for their host rather than translated.
ALIASES = {
    "__len__": "length",
    "new": "constructor",
}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


def _methods(text: str) -> set[str]:
    """Every name the file exposes, whether a method or a free function.

    Two shapes count.  A method sits inside an `impl` block and is therefore
    indented.  A free function is surface only when it carries an export
    attribute -- `#[pyfunction]` on one side, `#[wasm_bindgen]` on the other --
    which is what tells `kcl_tolerance` from a helper like `err` or `flatten`.

    Attributes were not checked at first, and the whole of `pipeline`'s
    function surface sat at column zero and went unread: nine exported
    functions that the guard was silently ignoring.
    """
    found = {m.group(1) for m in
             re.finditer(r"^[ \t]+(?:pub )?fn\s+([a-z_][a-z0-9_]*)\s*[(<]",
                         text, re.MULTILINE)}
    found |= {m.group(1) for m in
              re.finditer(r"^#\[(?:pyfunction|wasm_bindgen)[^\]]*\]\n"
                          r"(?:^#\[[^\]]*\]\n)*"
                          r"^(?:pub )?fn\s+([a-z_][a-z0-9_]*)\s*[(<]",
                          text, re.MULTILINE)}
    return found


def _read(paths) -> str:
    return "\n".join(p.read_text() for p in paths if p.exists())


CASES = [
    ("pools", [RUST / "pools" / "py.rs"], [WASM / "pools.rs"]),
    ("ladders", [RUST / "ladders_py.rs"], [WASM / "ladders.rs"]),
    ("graph", [RUST / "graph_py.rs"], [WASM / "graph.rs"]),
    ("nodes", [RUST / "nodes_py.rs"], [WASM / "nodes.rs"]),
    ("multiport", [RUST / "multiport_py.rs"], [WASM / "multiport.rs"]),
    ("realize", [RUST / "realize_py.rs"], [WASM / "realize.rs"]),
    ("candidates", [RUST / "candidates_py.rs"], [WASM / "candidates.rs"]),
    ("pipeline", [RUST / "pipeline_py.rs"], [WASM / "pipeline.rs"]),
    ("curves", [RUST / "curves_py.rs"], [WASM / "curves.rs"]),
    ("prices", [RUST / "prices_py.rs"], [WASM / "prices.rs"]),
    ("slippage", [RUST / "slippage_py.rs"], [WASM / "slippage.rs"]),
    ("refit", [RUST / "refit_py.rs"], [WASM / "refit.rs"]),
    ("codec", [RUST / "codec_py.rs"], [WASM / "codec.rs"]),
    ("routecall", [RUST / "routecall_py.rs"], [WASM / "routecall.rs"]),
]


@pytest.mark.parametrize("name,py_files,wasm_files", CASES,
                         ids=[c[0] for c in CASES])
def test_the_wasm_binding_covers_the_pyo3_one(name, py_files, wasm_files):
    py = _methods(_read(py_files))
    wasm = _methods(_read(wasm_files))
    # Helpers the binding uses internally are not surface; only what the other
    # side would have to call is compared.
    missing = set()
    for fn in py:
        if fn in PY_ONLY or fn.startswith("_"):
            continue
        if fn in wasm or _camel(fn) in wasm or ALIASES.get(fn, "") in wasm:
            continue
        missing.add(fn)
    assert not missing, (
        f"{name}: PyO3 exposes {sorted(missing)} and wasm does not. "
        f"A browser cannot do what the extension can. Add the binding, or put "
        f"the name in PY_ONLY with the reason.")


def test_every_pyo3_binding_file_has_a_wasm_counterpart():
    """A new `*_py.rs` with no wasm twin is the hole, one layer up."""
    py_bindings = {p.name for p in RUST.rglob("*.rs")
                   if p.name == "py.rs" or p.name.endswith("_py.rs")}
    # `src/py.rs` is the module's own function exports, whose twin is
    # `wasm/src/solve.rs` rather than a same-named file.
    known = {"py.rs": "solve.rs", "ladders_py.rs": "ladders.rs",
             "graph_py.rs": "graph.rs", "nodes_py.rs": "nodes.rs",
             "multiport_py.rs": "multiport.rs", "realize_py.rs": "realize.rs",
                 "naive_py.rs": "naive.rs",
             "candidates_py.rs": "candidates.rs", "pipeline_py.rs": "pipeline.rs",
             "curves_py.rs": "curves.rs", "prices_py.rs": "prices.rs",
             "slippage_py.rs": "slippage.rs", "refit_py.rs": "refit.rs",
             "codec_py.rs": "codec.rs", "routecall_py.rs": "routecall.rs"}
    for got in sorted(py_bindings):
        twin = known.get(got)
        assert twin is not None, (
            f"{got} is a PyO3 binding with no recorded wasm counterpart. "
            f"Add one under rust/wasm/src/ and name it in `known`.")
        assert (WASM / twin).exists(), f"{got} names {twin}, which is missing"


# ------------------------------------------------- the package and its shim


def test_the_browser_shim_answers_everything_the_package_asks_for():
    """`erouter.wasm._solve` stands in for the extension, so it has to have
    what the extension is asked for.

    Static, and deliberately: `test_pyodide_shim` runs the real thing but
    needs node and a Pyodide distribution, so it skips on most machines and
    skipped on the one where this hole was opened.  This needs neither.

    It reads attribute access rather than importing, because importing the
    shim outside Pyodide gets an unbound module and importing the package
    would need the extension present.
    """
    import ast

    package = ROOT / "src" / "erouter"
    wanted: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        if "wasm" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        # `import erouter_solve as x` is bound per module, and `accel` calls it
        # `_rust`; both are the extension under another name.
        aliases = {"erouter_solve", "_rust"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases |= {a.asname or a.name for a in node.names
                            if a.name == "erouter_solve"}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in aliases):
                wanted.setdefault(node.attr,
                                  f"{path.relative_to(ROOT)}:{node.lineno}")

    shim = (package / "wasm" / "_solve.py").read_text()
    defined = set(re.findall(r"^(?:class|def)\s+(\w+)", shim, re.M))
    defined |= set(re.findall(r"^(\w+)\s*=", shim, re.M))

    missing = {name: where for name, where in wanted.items()
               if name not in defined}
    assert not missing, (
        "the package asks the extension for names the browser shim does not "
        "bind, so every quote in a browser raises AttributeError while every "
        f"CPython test stays green: {missing}")
