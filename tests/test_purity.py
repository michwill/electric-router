"""`erouter.core` must stay importable in a browser.

The eventual target is a Pyodide/Flet frontend whose entire runtime dependency
list is `flet` + `flet-charts` -- no web3.py, no eth-abi, no titanoboa, because
compiled dependencies make a wasm32 build impossible or enormous.

The failure mode this guards is slow, not loud: `eth_abi` creeping into the leg
codec, `scipy` into the solve, `requests` into discovery.  Each is individually
reasonable and collectively a rewrite, and nothing else would notice until port
day.  So it is a test, from the first commit.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CORE = pathlib.Path(__file__).resolve().parents[1] / "src" / "erouter" / "core"

# Anything that is not stdlib or numpy.  scipy is allowed *inside functions*
# (an optional fast path) but never at module scope, so it can never be a hard
# import requirement.
FORBIDDEN = {
    "boa",
    "titanoboa",
    "requests",
    "eth_abi",
    "eth_account",
    "eth_utils",
    "web3",
    "vyper",
    "urllib",
    "scipy",
}
FORBIDDEN_AT_MODULE_SCOPE_ONLY = {"scipy"}

CORE_FILES = sorted(CORE.rglob("*.py"))


def test_core_package_is_populated():
    assert CORE_FILES, f"no modules found under {CORE}"


def _module_scope_imports(tree: ast.Module):
    """Top-level imports only -- imports inside a function are lazy and fine."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.lineno


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_forbidden_module_scope_imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for name, lineno in _module_scope_imports(tree):
        root = name.split(".")[0]
        assert root not in FORBIDDEN, (
            f"{path.name}:{lineno} imports {name!r} at module scope. "
            f"erouter.core must stay stdlib + numpy so it can run under Pyodide."
        )


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_imports_from_dev(path):
    """core must never reach into dev; the dependency arrow points one way."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module and "dev" in node.module.split("."):
            pytest.fail(f"{path.name}:{node.lineno} imports from erouter.dev")
        if node.level >= 2:
            pytest.fail(
                f"{path.name}:{node.lineno} uses a relative import that escapes erouter.core"
            )


def test_core_imports_with_forbidden_packages_blocked():
    """Import every core module in a process where the banned packages cannot load.

    Stronger than the AST check: it catches a transitive dependency that only
    shows up at import time, which is exactly how a port breaks.
    """
    import subprocess
    import sys
    import textwrap

    blocked = sorted(FORBIDDEN - {"urllib"})
    modules = [f"erouter.core.{p.stem}" for p in CORE_FILES if p.name != "__init__.py"]
    script = textwrap.dedent(f"""
        import importlib, sys
        BLOCKED = {blocked!r}

        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(
                        f"{{name}} is banned in erouter.core (Pyodide portability)"
                    )
                return None

        sys.meta_path.insert(0, Blocker())
        for name in {modules!r}:
            importlib.import_module(name)
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
