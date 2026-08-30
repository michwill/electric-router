"""Whatever PyO3 exposes, wasm must expose too.

The port has now grown the same hole twice.  `pools` compiled to wasm32 for
months while nothing under `rust/wasm/src/` referenced it, so the linker
stripped it and a browser could not price a pool; `ladders` was added with a
PyO3 binding only and repeated it within the same session.  Both were found by
reading, which is not a method.

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
    """Every method the file exposes, by name.

    Indented, because a method sits inside an `impl` block and a free function
    at column zero is a helper -- `err`, `halves`, `whole` -- which is the
    binding's own business rather than surface either side has to match.
    """
    return {m.group(1) for m in
            re.finditer(r"^[ \t]+(?:pub )?fn\s+([a-z_][a-z0-9_]*)\s*[(<]",
                        text, re.MULTILINE)}


def _read(paths) -> str:
    return "\n".join(p.read_text() for p in paths if p.exists())


CASES = [
    ("pools", [RUST / "pools" / "py.rs"], [WASM / "pools.rs"]),
    ("ladders", [RUST / "ladders_py.rs"], [WASM / "ladders.rs"]),
    ("graph", [RUST / "graph_py.rs"], [WASM / "graph.rs"]),
    ("nodes", [RUST / "nodes_py.rs"], [WASM / "nodes.rs"]),
    ("multiport", [RUST / "multiport_py.rs"], [WASM / "multiport.rs"]),
    ("realize", [RUST / "realize_py.rs"], [WASM / "realize.rs"]),
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
             "multiport_py.rs": "multiport.rs", "realize_py.rs": "realize.rs"}
    for got in sorted(py_bindings):
        twin = known.get(got)
        assert twin is not None, (
            f"{got} is a PyO3 binding with no recorded wasm counterpart. "
            f"Add one under rust/wasm/src/ and name it in `known`.")
        assert (WASM / twin).exists(), f"{got} names {twin}, which is missing"
