"""The wasm module must be the native extension, byte for byte.

Same crate, same compiler version (`rust/rust-toolchain.toml` pins it), two
targets.  The solver's only non-trivially-rounded operation is `sqrt`, which
IEEE-754 requires be correctly rounded everywhere, and Rust never contracts a
multiply-add on its own -- so an exact match is the expectation, not a hope.
Anything else is a marshalling bug, which is precisely what this catches: the
wasm boundary passes typed arrays where PyO3 passes lists, and flattens the
ragged inputs that PyO3 hands over as nested sequences.

Skipped when either half is missing.  `scripts/build_wasm.sh` makes the wasm
one; `maturin develop -m rust/Cargo.toml` makes the native one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from erouter.core import graph
from erouter.core.accel import available
from erouter.core.calibrate import DRIFT_TOL
from erouter.core.solve import TOL

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "solver_harness.mjs"
PKG = ROOT / "rust" / "wasm" / "pkg"
NODE = shutil.which("node")

pytestmark = [
    pytest.mark.skipif(not available(), reason="erouter_solve not installed"),
    pytest.mark.skipif(NODE is None, reason="node is not installed"),
    pytest.mark.skipif(
        not (PKG / "erouter_wasm_bg.wasm").exists(),
        reason="run scripts/build_wasm.sh",
    ),
]


def run(job: dict) -> dict:
    """One job through the wasm module, as the browser would run it."""
    done = subprocess.run(
        [NODE, str(HARNESS), str(PKG)],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if done.returncode != 0:
        raise AssertionError(f"harness failed:\n{done.stderr}")
    return json.loads(done.stdout)


def floats(hexed: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexed), dtype=np.float64)


def flags(hexed: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexed), dtype=np.uint8).astype(bool)


def make(tau, sig, a, B, *, cap=None, Psi=1.0, n=None):
    tau = np.asarray(tau, np.int64)
    sig = np.asarray(sig, np.int64)
    n = n or int(max(tau.max(), sig.max()) + 1)
    return graph.build(tau, sig, np.asarray(a, float), np.asarray(B, float),
                       np.ones(n), Psi, cap=cap, n_nodes=n, merge_duplicates=False)


def as_job(g, src, dst, psi_total, **kw) -> dict:
    caps = np.asarray(g.cap, float)
    return {
        "op": "solve",
        "tau": np.asarray(g.tau, np.int64).tolist(),
        "sig": np.asarray(g.sig, np.int64).tolist(),
        "g": np.asarray(g.G, float).tolist(),
        "eps": np.asarray(g.eps, float).tolist(),
        # JSON has no infinity; `null` carries it and the harness restores it.
        "cap": [None if not np.isfinite(v) else float(v) for v in caps],
        "n_nodes": int(g.n_nodes),
        "src": int(src),
        "dst": int(dst),
        "psi_total": float(psi_total),
        "a0": kw.get("a0"),
        "forbidden": kw.get("forbidden"),
        "pinned": kw.get("pinned"),
        "tol": kw.get("tol", TOL),
        "maxit": kw.get("maxit", 600),
        "min_flow": kw.get("min_flow", 0.0),
        "gas_cost": kw.get("gas_cost", 0.0),
        "partial_ok": kw.get("partial_ok", False),
        "rank1": kw.get("rank1", True),
    }


def both(g, src, dst, psi_total, **kw):
    """The native answer and the wasm one, ready to compare."""
    import erouter_solve

    caps = np.asarray(g.cap, float)
    problem = erouter_solve.Problem(
        np.asarray(g.tau, np.int64).tolist(),
        np.asarray(g.sig, np.int64).tolist(),
        np.asarray(g.G, float).tolist(),
        np.asarray(g.eps, float).tolist(),
        [float(v) if np.isfinite(v) else float("inf") for v in caps],
        int(g.n_nodes),
    )
    native = problem.solve(
        src=int(src), dst=int(dst), psi_total=float(psi_total),
        a0=None if kw.get("a0") is None else [bool(v) for v in kw["a0"]],
        forbidden=None if kw.get("forbidden") is None
        else [bool(v) for v in kw["forbidden"]],
        pinned=kw.get("pinned"),
        tol=kw.get("tol", TOL), maxit=kw.get("maxit", 600),
        min_flow=kw.get("min_flow", 0.0), gas_cost=kw.get("gas_cost", 0.0),
        partial_ok=kw.get("partial_ok", False), rank1=kw.get("rank1", True),
    )
    return native, run(as_job(g, src, dst, psi_total, **kw))


def same_solve(native, wasm, where: str = "") -> None:
    for key, js in (("psi", "psi"), ("u", "u"),
                    ("psi_upper", "psiUpper"), ("rho", "rho")):
        mine = np.frombuffer(native[key], dtype=np.float64)
        theirs = floats(wasm[js])
        assert mine.tobytes() == theirs.tobytes(), (
            f"{where}{key} differs: max |d| "
            f"{np.max(np.abs(mine - theirs)) if mine.size else 0:.3e}"
        )
    for key, js in (("active", "active"), ("upper", "upper")):
        assert bytes(native[key]) == bytes.fromhex(wasm[js]), f"{where}{key} differs"
    for key, js in (("pivots", "pivots"), ("chol_failures", "cholFailures"),
                    ("keep_changes", "keepChanges"), ("refits", "refits"),
                    ("feasible", "feasible"), ("reason", "reason")):
        assert native[key] == wasm[js], (
            f"{where}{key}: {native[key]!r} native, {wasm[js]!r} wasm"
        )


CASES = {
    "one arc": {"tau": [0], "sig": [1], "a": [1.0], "B": [1.0], "Psi": 0.25},
    "parallel": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 1.0], "B": [1.0, 2.0], "Psi": 1.0},
    "diode": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 0.9999], "B": [1.0, 1.0], "Psi": 0.5},
    "series": {"tau": [0, 0, 1], "sig": [2, 1, 2], "a": [0.997, 0.9995, 0.9995],
               "B": [1.0, 1.0, 1.0], "Psi": 0.4},
    "capped": {"tau": [0, 0], "sig": [1, 1], "a": [1.0, 0.999], "B": [1.0, 1.0],
               "cap": [0.2, np.inf], "Psi": 1.0},
    "network": {"tau": [0, 0, 1, 2, 0], "sig": [1, 2, 3, 3, 3],
                "a": [0.9995, 0.9990, 0.9995, 0.9998, 0.996],
                "B": [1.0, 2.0, 1.0, 3.0, 0.5], "Psi": 2.0},
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_shaped_problems_agree(name):
    spec = dict(CASES[name])
    Psi = spec.pop("Psi")
    g = make(**spec, Psi=Psi)
    dst = int(max(np.max(g.tau), np.max(g.sig)))
    same_solve(*both(g, 0, dst, Psi), where=f"{name}: ")


@pytest.mark.parametrize("seed", range(12))
def test_random_graphs_agree(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    m = int(rng.integers(n, 3 * n))
    tau = rng.integers(0, n, m)
    sig = rng.integers(0, n, m)
    keep = tau != sig
    tau, sig, m = tau[keep], sig[keep], int(keep.sum())
    if m < 2:
        pytest.skip("degenerate draw")
    a = 1.0 - rng.random(m) * 1e-3
    B = 0.5 + rng.random(m) * 3.0
    g = make(tau, sig, a, B, Psi=1.0, n=n)
    same_solve(*both(g, 0, n - 1, 1.0), where=f"seed {seed}: ")


def crowded(n_lanes=14, seed=3):
    """Many near-identical lanes -- the shape that makes a solve oscillate.

    The one that matters here.  `rust/README.md` records a port that agreed on
    every clean problem and returned 4,681 WETH where the reference returned
    9,052 on a real quote, because the real quote reached `maxit` and cycled.
    """
    rng = np.random.default_rng(seed)
    tau = np.zeros(n_lanes, np.int64)
    sig = np.ones(n_lanes, np.int64)
    a = 1.0 - rng.random(n_lanes) * 1e-9
    B = 1.0 + rng.random(n_lanes) * 1e-9
    return make(tau, sig, a, B, Psi=1.0, n=2)


@pytest.mark.parametrize("maxit", [1, 2, 3, 5, 11])
def test_running_out_of_pivots_agrees(maxit):
    same_solve(*both(crowded(), 0, 1, 1.0, maxit=maxit), where=f"maxit {maxit}: ")


@pytest.mark.parametrize("maxit", [1, 2, 3, 5, 11])
def test_a_partial_answer_agrees(maxit):
    same_solve(*both(crowded(), 0, 1, 1.0, maxit=maxit, partial_ok=True),
               where=f"maxit {maxit}: ")


@pytest.mark.parametrize("seed", range(6))
def test_cycling_agrees(seed):
    same_solve(*both(crowded(n_lanes=20, seed=seed), 0, 1, 1.0,
                     maxit=40, partial_ok=True), where=f"seed {seed}: ")


@pytest.mark.parametrize("screen", [0.0, 1e-3, 1e-2, 0.1])
def test_the_flow_screen_agrees(screen):
    same_solve(*both(crowded(), 0, 1, 1.0, min_flow=screen), where=f"min_flow {screen}: ")


@pytest.mark.parametrize("gas", [0.0, 1e-6, 1e-4, 1e-2])
def test_the_gas_screen_agrees(gas):
    same_solve(*both(crowded(), 0, 1, 1.0, gas_cost=gas), where=f"gas {gas}: ")


def test_a_pin_is_honoured_the_same_way():
    g = make([0, 0], [1, 1], [1.0, 0.999], [1.0, 1.0], Psi=1.0)
    same_solve(*both(g, 0, 1, 1.0, pinned=[(1, 0.3)]))


def test_a_forbidden_arc_is_refused_the_same_way():
    g = make([0, 0], [1, 1], [1.0, 0.999], [1.0, 1.0], Psi=1.0)
    same_solve(*both(g, 0, 1, 1.0, forbidden=[False, True]))


def test_a_warm_start_crosses_the_same_way():
    g = crowded()
    same_solve(*both(g, 0, 1, 1.0, a0=[True] + [False] * 13))


# ------------------------------------------------------------- calibrate


def ladder(seed: int):
    rng = np.random.default_rng(seed)
    deltas = np.geomspace(1e2, 1e6, 7)
    # A concave curve with a little noise: what a probe grid looks like.
    quotes = deltas * (1.0 - 1e-4 - deltas * 2e-10) * (1 + rng.normal(0, 1e-9, 7))
    return deltas.tolist(), quotes.tolist()


@pytest.mark.parametrize("seed", range(6))
def test_calibrate_agrees(seed):
    import erouter_solve

    deltas, quotes = ladder(seed)
    native = erouter_solve.calibrate(deltas, quotes, None, False, 0.05, None, None, 0.0)
    wasm = run({
        "op": "calibrate", "deltas": deltas, "quotes": quotes, "delta_bar": None,
        "structural_flag": False, "drift_tol": 0.05, "cap": None,
        "f_at_cap": None, "quantum": 0.0,
    })
    for k, (key, js) in enumerate([("a", "a"), ("B", "b"), ("cap", "cap")]):
        assert np.float64(native[k]).tobytes() == bytes.fromhex(wasm[js]), (
            f"seed {seed}: {key} differs -- {native[k]!r} vs {floats(wasm[js])[0]!r}"
        )
    assert native[3] == wasm["clamped"]
    assert native[4] == wasm["convexFlag"]
    assert native[5] == wasm["flag"]
    for k, js in ((6, "drift"), (7, "eta"), (9, "calibDelta"), (10, "tangentDelta")):
        assert np.float64(native[k]).tobytes() == bytes.fromhex(wasm[js]), (
            f"seed {seed}: field {k} differs"
        )
    assert native[8] == wasm["splitHint"]
    assert native[11] == wasm["note"]


# ---------------------------------------------------------- cycles, paths


@pytest.mark.parametrize("seed", range(6))
def test_cancel_cycles_agrees(seed):
    import erouter_solve

    rng = np.random.default_rng(seed)
    n = 5
    tau = [0, 1, 2, 3, 4, 2]
    sig = [1, 2, 3, 4, 0, 0]
    psi = (rng.random(len(tau)) + 0.1).tolist()
    flow, removed = erouter_solve.cancel_cycles(
        [int(v) for v in tau], [int(v) for v in sig], psi, 1e-12, n)
    wasm = run({"op": "cancel_cycles", "tau": tau, "sig": sig, "psi": psi,
                "tol": 1e-12, "n_nodes": n})
    assert np.asarray(flow, float).tobytes() == floats(wasm["flow"]).tobytes()
    assert removed == wasm["removed"]


def test_find_cycle_agrees():
    import erouter_solve

    tau, sig = [0, 1, 2], [1, 2, 0]
    native = erouter_solve.find_cycle(tau, sig, 3)
    wasm = run({"op": "find_cycle", "tau": tau, "sig": sig, "n_nodes": 3})
    assert list(native or []) == wasm["arcs"]


def test_shortest_path_agrees():
    import erouter_solve

    g = make([0, 0, 1, 2, 0], [1, 2, 3, 3, 3],
             [0.9995, 0.9990, 0.9995, 0.9998, 0.996],
             [1.0, 2.0, 1.0, 3.0, 0.5], Psi=2.0)
    caps = np.asarray(g.cap, float)
    problem = erouter_solve.Problem(
        np.asarray(g.tau, np.int64).tolist(), np.asarray(g.sig, np.int64).tolist(),
        np.asarray(g.G, float).tolist(), np.asarray(g.eps, float).tolist(),
        [float(v) if np.isfinite(v) else float("inf") for v in caps], int(g.n_nodes),
    )
    native = problem.shortest_path(0, 3, None, None, None, 8)
    wasm = run({
        "op": "shortest_path",
        "tau": np.asarray(g.tau, np.int64).tolist(),
        "sig": np.asarray(g.sig, np.int64).tolist(),
        "g": np.asarray(g.G, float).tolist(),
        "eps": np.asarray(g.eps, float).tolist(),
        "cap": [None if not np.isfinite(v) else float(v) for v in caps],
        "n_nodes": int(g.n_nodes), "src": 0, "dst": 3, "max_hops": 8,
    })
    assert list(native["arcs"]) == wasm["arcs"]
    assert native["found"] == wasm["found"]
    assert np.float64(native["length"]).tobytes() == bytes.fromhex(wasm["length"])
    assert list(native["negative_cycle"]) == wasm["negativeCycle"]


# --------------------------------------------------------------- pool models

def _spec_for_js(kind, spec):
    """One model in the shape the harness hands to `Pools`.

    Every integer as a decimal string, which is the boundary both bindings
    use: a balance does not survive a JSON number, and `JSON.parse` would
    round it silently rather than failing.
    """
    out = {"kind": kind}
    if kind == "one_to_one":
        return out
    if kind == "vault":
        out.update(num=str(spec["num"]), den=str(spec["den"]),
                   cap=str(spec.get("cap", 0)))
    elif kind in ("lp_withdraw", "lp_deposit"):
        from test_pools_differential import STABLE_SUPPLY
        out.update(
            balances=[str(v) for v in spec["balances"]],
            rates=[str(v) for v in spec["rates"]],
            amp=str(spec["amp"]), fee=str(spec["fee"]),
            offpeg_fee_multiplier=str(spec.get("offpeg_fee_multiplier", 0)),
            a_precision=str(spec.get("a_precision", 100)),
            fee_on_xp=spec.get("fee_on_xp", True),
            subtract_one=spec.get("subtract_one", True),
            total_supply=str(STABLE_SUPPLY),
            admin_fee=(str(spec["admin_fee"])
                       if spec.get("admin_fee", -1) >= 0 else None))
    elif kind == "tri_lp":
        from test_pools_differential import TRI_SUPPLY
        out.update(
            balances=[str(v) for v in spec["balances"]],
            precisions=[str(v) for v in spec["precisions"]],
            price_scale=[str(v) for v in spec["price_scale"]],
            d=str(spec["d"]), amp=str(spec["amp"]), gamma=str(spec["gamma"]),
            mid_fee=str(spec["mid_fee"]), out_fee=str(spec["out_fee"]),
            fee_gamma=str(spec["fee_gamma"]),
            legacy=spec.get("legacy", False),
            a_multiplier=str(spec.get("a_multiplier", 10000)),
            total_supply=str(TRI_SUPPLY))
    elif kind == "stableswap":
        out.update(
            balances=[str(v) for v in spec["balances"]],
            rates=[str(v) for v in spec["rates"]],
            amp=str(spec["amp"]), fee=str(spec["fee"]),
            offpeg_fee_multiplier=str(spec.get("offpeg_fee_multiplier", 0)),
            a_precision=str(spec.get("a_precision", 100)),
            fee_on_xp=spec.get("fee_on_xp", True),
            subtract_one=spec.get("subtract_one", True),
            admin_fee=(str(spec["admin_fee"])
                       if spec.get("admin_fee", -1) >= 0 else None))
    elif kind == "twocrypto":
        out.update(
            balances=[str(v) for v in spec["balances"]],
            precisions=[str(v) for v in spec["precisions"]],
            price_scale=str(spec["price_scale"]), d=str(spec["d"]),
            amp=str(spec["amp"]), gamma=str(spec["gamma"]),
            mid_fee=str(spec["mid_fee"]), out_fee=str(spec["out_fee"]),
            fee_gamma=str(spec["fee_gamma"]),
            stable=spec.get("stable", True), v21=spec.get("v21", True),
            legacy_fee=spec.get("legacy_fee", False),
            legacy_pool=spec.get("legacy_pool", False),
            legacy_mul2=spec.get("legacy_mul2", False))
    else:
        out.update(
            balances=[str(v) for v in spec["balances"]],
            precisions=[str(v) for v in spec["precisions"]],
            price_scale=[str(v) for v in spec["price_scale"]],
            d=str(spec["d"]), amp=str(spec["amp"]), gamma=str(spec["gamma"]),
            mid_fee=str(spec["mid_fee"]), out_fee=str(spec["out_fee"]),
            fee_gamma=str(spec["fee_gamma"]),
            legacy=spec.get("legacy", False),
            a_multiplier=str(spec.get("a_multiplier", 10000)))
    return out


def _priced_in_the_browser(fast):
    """Every shared vector through the wasm module, in one batch."""
    from test_pools_differential import CASES

    models, which, ii, jj, dd = [], [], [], [], []
    for kind, _n, spec, i, j, dx in CASES:
        models.append(_spec_for_js(kind, spec))
        which.append(len(models) - 1)
        ii.append(i)
        jj.append(j)
        dd.append(dx)
    got = run({"op": "price", "models": models, "which": which, "i": ii,
               "j": jj, "dx": [str(v) for v in dd], "fast": fast})
    return [int(v) if v is not None else None for v in got["dy"]]


def test_the_browser_prices_a_pool_wei_for_wei():
    """The exact path, through wasm, against Python.

    Same crate and same compiler as the extension, so this is not a second
    implementation being checked -- it is the marshalling being checked: the
    balances cross as strings and the answers come back as lo/hi halves of a
    `u128`, and either of those is somewhere a digit can be lost.
    """
    from test_pools_differential import CASES, _ids, _model, _python_price

    want = [_python_price(_model(k, s), i, j, dx, False)
            for k, _n, s, i, j, dx in CASES]
    wrong = [(n, w, g) for n, w, g in
             zip(_ids(), want, _priced_in_the_browser(False), strict=True)
             if w != g]
    assert not wrong, f"{len(wrong)} of {len(CASES)} disagree: {wrong[:5]}"


def test_the_browser_float_path_matches_the_extension_exactly():
    """The float path, through wasm, against the *native* extension.

    Against the extension rather than against Python, and that is the point:
    the two Rust targets run the same source through the same compiler, so
    here an exact match is the expectation.  Python is allowed to differ from
    both by the quote path's budget -- `test_pools_differential` holds that --
    but wasm drifting from the extension would mean the browser ranks routes
    differently from the CLI, which nothing budgets for.
    """
    from test_pools_differential import _batch, _ids

    _want, native = _batch(True)
    wrong = [(n, a, b) for n, a, b in
             zip(_ids(), native, _priced_in_the_browser(True), strict=True)
             if a != b]
    assert not wrong, f"{len(wrong)} disagree with the extension: {wrong[:5]}"


def test_the_browser_splits_an_element_the_same_way():
    """The split search, which moves a loop rather than batching a call."""
    from test_pools_differential import GNOSIS_3POOL

    dx = GNOSIS_3POOL["balances"][0] // 100
    got = run({"op": "element_split",
               "model": _spec_for_js("stableswap", GNOSIS_3POOL),
               "i": 0, "j1": 1, "j2": 2, "dx": str(dx)})
    import erouter_solve

    from test_pools_differential import _add
    pools = erouter_solve.Pools()
    _add(pools, "stableswap", GNOSIS_3POOL)
    assert tuple(got["split"]) == pools.element_split(0, 0, 1, 2, dx)


# ------------------------------------------------------------------ ladders

def _ladder_job(seed: int, *, answer: bool, fit: bool):
    """The refine stage's inputs, in the shape the harness hands to `Ladders`."""
    from erouter.core.probe import plan_sized
    from test_ladders_resident import _answers, _ladders, _sizes

    ladders = _ladders(seed)
    sizes = _sizes(ladders, seed)
    by_id = {lad.arc.id: k for k, lad in enumerate(ladders)}
    slots, want, spans = [], [], [0]
    for arc_id, values in sizes.items():
        slots.append(by_id[arc_id])
        want.extend(int(v) for v in values)
        spans.append(len(want))

    job = {
        "op": "ladders",
        "ladders": [{"decimals_in": lad.arc.decimals_in,
                     "decimals_out": lad.arc.decimals_out,
                     "reserve_in": str(max(0, lad.arc.reserve_in)),
                     "deltas": [str(d) for d in lad.deltas],
                     "quotes": [str(q) for q in lad.quotes],
                     "attempted": lad.attempted} for lad in ladders],
        "slots": slots, "want": [str(v) for v in want], "spans": spans,
    }
    plan = plan_sized(ladders, sizes)
    if answer:
        got = _answers(plan.probes, seed)
        names, status, values = [], [], []
        for one in got:
            if one.status is not None and one.status.name != "VALUE":
                if one.status.name not in names:
                    names.append(one.status.name)
                status.append(names.index(one.status.name) + 1)
                values.append(0)
            else:
                status.append(0)
                values.append(max(0, int(one.value)))
        job.update(values=[str(v) for v in values], status=status, names=names)
    if fit:
        job.update(fit=list(range(len(ladders))), driftTol=DRIFT_TOL)
    return job, ladders, sizes, plan


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_the_browser_plans_the_same_probes(seed):
    """The plan, and its order -- the answers are zipped back against it."""
    job, _lads, _sz, plan = _ladder_job(seed, answer=False, fit=False)
    got = run(job)
    assert [int(v) for v in got["deltas"]] == [p.dx for p in plan.probes]


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_the_browser_merges_the_same_way(seed):
    """Every ladder's points and counts after absorbing the same answers."""
    from erouter.core.probe import collect, merge

    _j, ladders, _sz, plan = _ladder_job(seed, answer=False, fit=False)
    job, _l2, _sz2, _p = _ladder_job(seed, answer=True, fit=False)
    got = run(job)

    from test_ladders_resident import _answers
    merge(ladders, collect(plan, _answers(plan.probes, seed)))
    for k, lad in enumerate(ladders):
        half = len(got["points"][k]) // 2
        assert [int(v) for v in got["points"][k][:half]] == lad.deltas, f"slot {k}"
        assert [int(v) for v in got["points"][k][half:]] == lad.quotes, f"slot {k}"
        assert got["attempted"][k] == lad.attempted, f"slot {k}"


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_the_browser_fits_what_the_extension_fits(seed):
    """The fits, field for field, against the native extension.

    Against the extension rather than Python for the same reason the float
    models are: two targets, one source, one compiler -- so an exact match is
    the expectation and anything else is marshalling.
    """
    import math

    from erouter.core.accel import ladders_from

    job, ladders, _sz, _plan = _ladder_job(seed, answer=False, fit=True)
    got = run(job)["fits"]
    want = ladders_from(ladders).recalibrate(list(range(len(ladders))), DRIFT_TOL)

    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        assert (a is None) == (b is None), f"slot {k}: one side refused"
        if a is None:
            continue
        # (a, B, cap, drift, eta, calib_delta) against the tuple's
        # (a, B, cap, clamped, convex, flag, drift, eta, calib_delta).
        nums = floats(a["nums"])
        for f, (x, y) in enumerate(zip(nums, (b[0], b[1], b[2], b[6], b[7], b[8]),
                                       strict=True)):
            if math.isnan(x) and math.isnan(y):
                continue
            assert x == y, f"slot {k} float {f}: {x} != {y}"
        assert a["clamped"] == b[3], f"slot {k}: clamped"
        assert a["convexFlag"] == b[4], f"slot {k}: convex_flag"
        assert a["flag"] == b[5], f"slot {k}: flag"


# --------------------------------------------------------------- the graph
#
# `graph.build` is the assembly the solver's arrays come out of, and it is the
# first stage above the QP to have a wasm form.  Comparing it against the
# *Python* reference rather than the extension would test the port twice and
# the marshalling not at all -- so this compares the two Rust builds, which is
# where a typed-array bug would live.


def graph_job(seed: int) -> dict:
    """The same nasty universes `test_graph_differential` generates."""
    from test_graph_differential import universe

    u = universe(seed)
    caps = [None if not np.isfinite(v) else float(v) for v in u["cap"]]
    return {
        "op": "graph",
        "tau": [int(v) for v in u["tau"]],
        "sig": [int(v) for v in u["sig"]],
        "a": [float(v) for v in u["a"]],
        "B": [float(v) for v in u["B"]],
        "nu": [float(v) for v in u["nu"]],
        "Psi": u["Psi"],
        "cap": caps,
        "flagged": [int(v) for v in u["flagged"]],
        "n_nodes": int(u["n_nodes"]),
        "scale": u["Psi"],
    }


@pytest.mark.parametrize("seed", [0, 3, 7, 11, 19])
def test_the_browser_assembles_the_same_graph(seed):
    import erouter_solve

    job = graph_job(seed)
    got = run(job)
    want = erouter_solve.Graph.build(
        job["tau"], job["sig"], job["a"], job["B"], job["nu"], job["Psi"],
        cap=[np.inf if v is None else v for v in job["cap"]],
        flagged=[bool(v) for v in job["flagged"]],
        n_nodes=job["n_nodes"],
    )

    assert got["tau"] == list(want.tau)
    assert got["sig"] == list(want.sig)
    for field, native_value in (("a", want.a), ("B", want.b), ("G", want.g),
                                ("eps", want.eps), ("cap", want.cap)):
        assert floats(got[field]).tolist() == list(native_value), field
    assert [bool(v) for v in got["flagged"]] == want.flagged
    assert [bool(v) for v in got["clamped"]] == want.clamped
    assert got["nNodes"] == want.n_nodes
    assert floats(got["condition"])[0] == want.condition()
    assert floats(got["illConditioned"])[0] == want.ill_conditioned

    flat, spans = want.sources()
    assert got["sources"] == list(flat)
    assert got["sourceSpans"] == list(spans)
    index, reason = want.dropped()
    assert got["dropped"] == list(index)
    assert got["droppedReason"] == list(reason)

    # `scale` mutates, so it is compared after everything read from `want`.
    psi_scaled = want.scale(job["scale"])
    assert floats(got["psiScaled"])[0] == psi_scaled
    assert floats(got["gScale"])[0] == want.g_scale
    assert floats(got["scaledG"]).tolist() == list(want.g)
    assert floats(got["scaledCap"]).tolist() == list(want.cap)


# ------------------------------------------------------------- the nodes


def test_the_browser_merges_the_same_nodes():
    import erouter_solve

    from test_nodes_differential import (
        ALL_TOKENS,
        CRVUSD,
        ETH,
        SCRVUSD,
        STETH,
        VAULT_DEN,
        VAULT_NUM,
        WETH,
        WSTETH,
    )

    tokens = [
        {"address": WETH, "symbol": "WETH", "decimals": 18},
        {"address": ETH, "symbol": "ETH", "decimals": 18},
        {"address": CRVUSD, "symbol": "crvUSD", "decimals": 18},
        {"address": SCRVUSD, "symbol": "scrvUSD", "decimals": 18},
        {"address": STETH, "symbol": "stETH", "decimals": 18},
        {"address": WSTETH, "symbol": "wstETH", "decimals": 18},
        {"address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
         "symbol": "USDT", "decimals": 6},
    ]
    merges = [
        {"kind": "NATIVE_WRAP", "token": ETH, "canonical": WETH,
         "rate_num": "1", "rate_den": "1", "target": WETH},
        {"kind": "ERC4626", "token": SCRVUSD, "canonical": CRVUSD,
         "rate_num": str(VAULT_NUM), "rate_den": str(VAULT_DEN), "target": SCRVUSD},
        {"kind": "WSTETH", "token": WSTETH, "canonical": STETH,
         "rate_num": "1204183982113311744", "rate_den": str(10**18), "target": WSTETH},
    ]
    ask = [*ALL_TOKENS, "0xnotatoken"]
    amount = str(3 * 10**24 + 7)
    rescale_args = [0.9997, 4.2e-9, 1.0432519443827714, 1.0]

    got = run({"op": "nodes", "tokens": tokens, "merges": merges, "ask": ask,
               "amount": amount, "rescale": rescale_args})

    want = erouter_solve.NodeMap()
    for t in tokens:
        want.add_token(t["address"], t["symbol"], t["decimals"])
    for m in merges:
        want.merge(m["kind"], m["token"], m["canonical"],
                   m["rate_num"], m["rate_den"], m["target"])

    assert got["nNodes"] == want.n_nodes()
    assert got["mergedNodes"] == want.merged_nodes()
    assert got["node"] == [want.node(t) if want.has(t) else None for t in ask]
    assert got["canonical"] == [want.canonical(t) if want.has(t) else None for t in ask]
    assert got["symbol"] == [want.symbol(t) for t in ask]
    assert got["decimals"] == [want.decimals(t) for t in ask]
    assert floats(got["rate"]).tolist() == [want.rate(t) for t in ask]
    assert got["toCanonical"] == [want.to_canonical_wei(t, amount) for t in ask]
    assert got["fromCanonical"] == [want.from_canonical_wei(t, amount) for t in ask]
    assert got["nodeSymbol"] == [want.node_symbol(k) for k in range(want.n_nodes())]
    assert got["tokensOf"] == [want.tokens_of(k) for k in range(want.n_nodes())]
    # An empty array is how the wasm side spells `None`: JS has no tuple.
    assert got["conversion"] == [list(want.conversion(t) or ()) for t in ask]
    assert got["conversionKinds"] == [list(want.conversion_kinds(t) or ()) for t in ask]
    assert got["isAlias"] == [want.is_alias(t) for t in ask]
    assert floats(got["rescale"]).tolist() == list(
        erouter_solve.NodeMap.rescale(*rescale_args))


# ------------------------------------------------------------ elements
#
# An element is where the shape rules and the advancing state meet, and both
# have to cross intact: the ports go over as parallel coin and share arrays
# because a pair has no typed form, and getting that wrong would silently
# transpose a split.


def test_the_browser_refuses_the_same_shapes():
    from erouter.core.multiport import BPS, LP, MultiPort, MultiPortError, Port

    cases = [
        (3, [(0, BPS)], [(1, 5_000), (2, 5_000)]),
        (2, [(0, BPS)], [(1, 5_000), (2, 5_000)]),
        (3, [(0, BPS)], [(0, 5_000), (1, 5_000)]),
        (2, [(0, 5_000), (1, 5_000)], [(LP, BPS)]),
        (3, [(0, BPS)], [(1, 4_000), (2, 5_000)]),
    ]
    for n_coins, inputs, outputs in cases:
        want = None
        try:
            made = MultiPort(pool="0xp", n_coins=n_coins,
                             inputs=tuple(Port(*p) for p in inputs),
                             outputs=tuple(Port(*p) for p in outputs))
        except MultiPortError as e:
            want = str(e)
        got = run({
            "op": "element", "pool": "0xp", "n_coins": n_coins,
            "in_coins": [c for c, _ in inputs], "in_bps": [b for _, b in inputs],
            "out_coins": [c for c, _ in outputs], "out_bps": [b for _, b in outputs],
        })
        if want is None:
            assert "error" not in got, got
            # `inputs()` interleaves coin and share, which is how a pair
            # crosses without a typed form for it.
            assert got["inputs"] == [v for p in made.inputs for v in (p.coin, p.bps)]
            assert got["outputs"] == [v for p in made.outputs for v in (p.coin, p.bps)]
            assert got["ports"] == made.ports
        else:
            assert got.get("error", "").endswith(want), (got, want)


@pytest.mark.parametrize("dx", [10**18, 10_000 * 10**18])
def test_the_browser_advances_the_pool_between_ports(dx):
    import erouter_solve

    from test_multiport_differential import ADDRESS, POOL, native

    pools, which, _ = native(POOL)
    want = pools.element_evaluate(which, None, 3, [(0, 10_000)],
                                  [(1, 3_000), (2, 7_000)], str(dx))
    got = run({
        "op": "element", "pool": ADDRESS, "n_coins": 3,
        "in_coins": [0], "in_bps": [10_000],
        "out_coins": [1, 2], "out_bps": [3_000, 7_000],
        "dx": str(dx),
        "models": {
            "balances": [str(b) for b in POOL.balances],
            "rates": [str(r) for r in POOL.rates],
            "amp": str(POOL.amp), "fee": str(POOL.fee),
            "offpeg_fee_multiplier": str(POOL.offpeg_fee_multiplier),
            "a_precision": str(POOL.a_precision), "fee_on_xp": POOL.fee_on_xp,
            "subtract_one": POOL.subtract_one, "admin_fee": str(POOL.admin_fee),
        },
        "weights": [str(POOL.rates[1]), str(POOL.rates[2])],
    })
    assert got["dy"] == list(want)
    first, second, payout = pools.element_best_split(
        which, None, 3, [(0, 10_000)], [(1, 3_000), (2, 7_000)], str(dx),
        [str(POOL.rates[1]), str(POOL.rates[2])])
    assert got["split"] == [first, second]
    assert floats(got["payout"])[0] == payout
    assert erouter_solve is not None


# ----------------------------------------------------------- realisation


def test_the_browser_realises_the_same_route():
    """The leg list is the executable artefact, so it crosses field for field."""
    import erouter_solve

    from test_realize_differential import (
        CRVUSD,
        POOL_A,
        POOL_B,
        SCRVUSD,
        USDC,
        WETH,
        build_nodes,
        make_arc,
        port_arcs,
    )

    reference, ported = build_nodes()
    arcs = [
        make_arc(POOL_A, WETH, USDC, reference, a=3000.0, B=1e-3),
        make_arc(POOL_B, USDC, CRVUSD, reference, a=1.0, B=1e-9),
    ]
    psi = [1.0, 1.0]
    nu = [1.0] * reference.n_nodes
    want = erouter_solve.Route.realize(
        port_arcs(arcs), psi, nu, ported, WETH, SCRVUSD, str(10**18), None)

    got = run({
        "op": "realize",
        "tokens": [
            {"address": WETH, "symbol": "WETH", "decimals": 18},
            {"address": USDC, "symbol": "USDC", "decimals": 6},
            {"address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
             "symbol": "USDT", "decimals": 6},
            {"address": CRVUSD, "symbol": "crvUSD", "decimals": 18},
            {"address": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
             "symbol": "ETH", "decimals": 18},
            {"address": SCRVUSD, "symbol": "scrvUSD", "decimals": 18},
        ],
        "merges": [
            {"kind": "NATIVE_WRAP",
             "token": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
             "canonical": WETH, "rate_num": "1", "rate_den": "1", "target": WETH},
            {"kind": "ERC4626", "token": SCRVUSD, "canonical": CRVUSD,
             "rate_num": str(11 * 10**17), "rate_den": str(10**18),
             "target": SCRVUSD},
        ],
        "arcs": [
            {"id": a.id, "pool": a.pool, "kind": int(a.kind), "i": a.i, "j": a.j,
             "n_coins": a.n_coins, "token_in": a.token_in,
             "token_out": a.token_out, "tau": a.tau, "sigma": a.sigma,
             "a": a.a, "B": a.B,
             "cap": None if not np.isfinite(a.cap) else a.cap,
             "G": a.G, "eps": a.eps, "reserve_in": str(a.reserve_in),
             "decimals_in": a.decimals_in, "tvl_usd": a.tvl_usd,
             "gamma_live": None if np.isnan(a.gamma_live) else a.gamma_live,
             "note": a.note}
            for a in arcs
        ],
        "psi": psi, "nu": nu, "src": WETH, "dst": SCRVUSD,
        "amount_in": str(10**18),
    })

    assert got["wireLegs"] == [t for t, *_ in want.wire_legs()]
    assert got["wireNumbers"] == [
        v for _, *rest in want.wire_legs() for v in rest]
    assert got["targets"] == want.targets()
    assert got["kinds"] == list(want.kinds())
    assert got["tokensIn"] == want.tokens_in()
    assert got["tokensOut"] == want.tokens_out()
    assert got["amountsIn"] == want.amounts_in()
    assert got["amountsOut"] == want.amounts_out()
    assert got["arcIds"] == want.arc_ids()
    assert got["poolNames"] == want.pool_names()
    # `gamma_live` is NaN on a conversion leg, and NaN is not equal to itself.
    numbers, expected = floats(got["numbers"]), np.array(want.numbers())
    agree = (numbers == expected) | (np.isnan(numbers) & np.isnan(expected))
    assert agree.all(), (numbers[~agree], expected[~agree])
    assert got["modelled"] == [int(v) for v in want.modelled()]
    assert got["isConversion"] == [int(v) for v in want.is_conversion()]
    assert got["slots"] == [t for t, _ in want.slots()]
    assert got["slotIndices"] == [k for _, k in want.slots()]
    assert got["nodeOfSlot"] == [v for pair in want.node_of_slot() for v in pair]
    assert got["dstSlot"] == want.dst_slot
    assert got["modelledOut"] == want.modelled_out
    assert got["paths"] == [">".join(p) for p in want.paths()]
    assert got["warnings"] == want.warnings()
    assert got["poolsUsed"] == want.pools_used()
    assert floats(got["maxTheta"])[0] == want.max_theta()
    assert floats(got["routeConductance"])[0] == want.route_conductance()
