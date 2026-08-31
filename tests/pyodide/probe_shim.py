"""Every entry point of `erouter.wasm`, exercised inside a real Pyodide.

Run by `tests/js/pyodide_harness.mjs`, which loads Pyodide in node, copies
`src/erouter` into its filesystem, initialises the wasm module and hands this
file to `runPythonAsync`.  One line per entry point; anything not starting
with `OK` is a failure, and `tests/test_pyodide_shim.py` reads them back.

The point is the *differences*.  This shim is the one part of the tree that
only ever runs where CPython is not, so a CPython test of it proves nothing
about the place it runs -- `JsBuffer` has `to_bytes` and no `destroy`, and
that alone broke three of these eight while everything else stayed green.
"""
import sys
import traceback

# `erouter_wasm_glue` is the initialised wasm module, registered by the
# harness; the package comes from the filesystem it copied it into.
import erouter_wasm_glue as glue
import numpy as np

from erouter.wasm import _evm, _solve

_solve.bind(glue)
_evm.bind(glue)
sys.modules["erouter_solve"] = _solve
sys.modules["erouter_evm"] = _evm

report = []
def step(name, fn):
    try:
        report.append(f"OK   {name}: {fn()}")
    except Exception as exc:
        report.append(f"FAIL {name}: {type(exc).__name__}: {exc}\n"
                      + "".join(traceback.format_exc().splitlines(True)[-6:]))

step("version", lambda: _solve.version())

def solve_once():
    p = _solve.Problem([0, 0], [1, 1], [1.0, 2.0], [1e-4, 2e-4],
                       [float("inf")] * 2, 2)
    try:
        got = p.solve(0, 1, 1.0)
        return {k: (len(v) if isinstance(v, bytes) else v)
                for k, v in got.items() if k in ("psi", "pivots", "feasible", "reason")}
    finally:
        p.close()
step("Problem.solve", solve_once)

step("cancel_cycles", lambda: _solve.cancel_cycles([0, 1, 2], [1, 2, 0],
                                                   [1.0, 1.0, 1.0], 1e-12, 3))
step("find_cycle", lambda: _solve.find_cycle([0, 1, 2], [1, 2, 0], 3))

def calibrate_once():
    deltas = np.geomspace(1e2, 1e6, 7)
    quotes = deltas * (1 - 1e-4 - deltas * 2e-10)
    return _solve.calibrate(deltas.tolist(), quotes.tolist())[:3]
step("calibrate", calibrate_once)

def shortest():
    p = _solve.Problem([0, 0, 1], [1, 2, 2], [1.0, 1.0, 1.0],
                       [1e-4, 3e-4, 1e-4], [float("inf")] * 3, 3)
    try:
        return p.shortest_path(0, 2)
    finally:
        p.close()
step("shortest_path", shortest)

def evm_once():
    evm = _evm.Evm("Osaka", 1)
    evm.set_block(number=1, timestamp=1_770_000_000)
    who = "0x" + "22" * 20
    evm.insert_account(who, code=bytes.fromhex("602a5f5260205ff3"))
    got = evm.call("0x" + "11" * 20, who, b"")
    return int.from_bytes(got["output"], "big"), got["gas_used"]
step("Evm.call", evm_once)

def evm_batch():
    evm = _evm.Evm("Osaka", 1)
    evm.set_block(number=1, timestamp=1_770_000_000)
    who = "0x" + "44" * 20
    evm.insert_account(who, code=bytes.fromhex("6007545f5260205ff3"))
    evm.insert_storage_many([(who, "0x7", "0x1234")])
    out = evm.call_many("0x" + "11" * 20, [(who, b""), (who, b"")])
    return [int.from_bytes(o["output"], "big") for o in out], evm.take_misses()
step("Evm.call_many + misses", evm_batch)

def split():
    # `slope` is one shorter than `x` and `u` -- it is `diff(u) / diff(x)`.
    # Equal-length arrays here passed while the real thing overran its own
    # flattened array on every quote, which is the whole reason this probe
    # uses a shape the router actually produces.
    x = [0.0, 0.5, 1.0]
    u = [0.0, 0.5, 0.99]
    slope = [1.0, 0.98]
    return _solve.split_ascend(
        [(x, u, slope, 1.0, 0.0)],
        [0], [1], [None], [[0]], [0], 2, 1, 1.0, [[1.0]], [(0, 0)],
        1e-6, 10, 2, 0.3, 1e-9)
step("split_ascend", split)

# --- what the browser reaches after the probe path grew ----------------
#
# `Graph`, `Arcs`, `Ballot` and `Pools` were added to the package without
# being added here, and the browser raised `AttributeError` on the first
# quote while every CPython test stayed green.  That is the failure class
# this whole file exists for, so each one is exercised, not merely bound.

def pools_price():
    pools = _solve.Pools()
    try:
        which = pools.add_stableswap(
            ["1000000000000000000000", "1000000000000000000000"],
            ["1000000000000000000", "1000000000000000000"],
            "200", "1000000", "0", "100", False, True)
        got = pools.price([which, which], [0, 1], [1, 0],
                          [10 ** 18, 10 ** 30], False)
        # The second amount is past `u64`, which is the point: `u128` crosses
        # as a lo/hi pair and a shim that forgot the hi word would answer it
        # with the wrong number rather than fail.
        return [None if v is None else v for v in got], len(pools)
    finally:
        pools.close()
step("Pools.price", pools_price)

def pools_vault_and_split():
    pools = _solve.Pools()
    try:
        vault = pools.add_vault("1050000000000000000", "10" + "0" * 17, "0")
        one = pools.add_one_to_one()
        priced = pools.price([vault, one], [0, 0], [1, 1],
                             [10 ** 18, 10 ** 18], False)
        # A `u64` scalar parameter, which Pyodide sends as a JS Number unless
        # it is made a BigInt -- the conversion this shim has to do by hand.
        split = pools.element_split(vault, 0, 1, 1, 10 ** 18)
        return priced, split
    finally:
        pools.close()
step("Pools.element_split", pools_vault_and_split)

def ballot_once():
    """A whole generation: `Graph`, `Arcs` and `Ballot` in the order `accel`
    uses them."""
    tau, sig = [0, 0, 1], [1, 2, 2]
    g, eps = [2.0, 1.0, 3.0], [1e-4, 3e-4, 1e-4]
    cap = [float("inf")] * 3
    graph = _solve.Graph.from_arrays(tau, sig, g, eps, cap, [False] * 3, 3)
    arcs = _solve.Arcs()
    try:
        for k in range(3):
            arcs.add(f"a{k}", "0x" + f"{k:02x}" * 20, 0, 0, 1, 2,
                     "0x" + "11" * 20, "0x" + "22" * 20, tau[k], sig[k],
                     1.0, 0.0, cap[k], g[k], eps[k], 10 ** 21, 18, 1e6, 0.0)
        base = _solve.solve(tau, sig, g, eps, cap, 3, 0, 2, 1.0)
        psi = np.frombuffer(base["psi"], dtype=np.float64)
        ballot = _solve.Ballot.generate(
            graph, arcs, 0, 2, 1.0, psi.tolist(), max_candidates=6)
        try:
            return {
                "n": len(ballot),
                "labels": ballot.labels()[:3],
                "kinds": ballot.kinds()[:3],
                "certificates": ballot.certificates()[:3],
                "n_arcs": ballot.n_arcs()[:3],
                "loss": [round(v, 9) for v in ballot.modelled_loss()[:2]],
                "psi0": [round(v, 9) for v in ballot.psi(0)],
                "solves": ballot.solves, "pivots": ballot.pivots,
                "skipped": ballot.skipped, "skipped_wide": ballot.skipped_wide,
            }
        finally:
            ballot.close()
    finally:
        arcs.close()
        graph.close()
step("Ballot.generate", ballot_once)

def ballot_with_pricer():
    """The element pricer crosses as a JS function and answers with a JS
    array; a Python tuple arrives as a proxy with no `length` and is refused
    for the wrong reason."""
    tau, sig = [0, 0, 1], [1, 2, 2]
    g, eps = [2.0, 1.0, 3.0], [1e-4, 3e-4, 1e-4]
    cap = [float("inf")] * 3
    seen = []

    def pricer(k1, k2, psi1, psi2):
        seen.append((k1, k2))
        return (psi1 * 0.5, psi2 * 0.5)

    # Arcs 0 and 1 share a pool and a `tau`, which is what proposes an
    # element pair.  With three distinct pools the family never runs and the
    # callback is never called -- the probe passed that way and proved only
    # that the argument crossed.
    pools = ["0x" + "aa" * 20, "0x" + "aa" * 20, "0x" + "bb" * 20]
    graph = _solve.Graph.from_arrays(tau, sig, g, eps, cap, [False] * 3, 3)
    arcs = _solve.Arcs()
    try:
        for k in range(3):
            arcs.add(f"a{k}", pools[k], 0, 0, k + 1, 3,
                     "0x" + "11" * 20, "0x" + f"{k + 2:02x}" * 20,
                     tau[k], sig[k],
                     1.0, 0.0, cap[k], g[k], eps[k], 10 ** 21, 18, 1e6, 0.0)
        base = _solve.solve(tau, sig, g, eps, cap, 3, 0, 2, 1.0)
        psi = np.frombuffer(base["psi"], dtype=np.float64)
        # The element family runs *after* the pin sweep and breaks as soon as
        # the budget is full, so a small `max_candidates` never reaches it --
        # which is how this probe first passed while calling nothing.
        ballot = _solve.Ballot.generate(
            graph, arcs, 0, 2, 1.0, psi.tolist(), max_candidates=30,
            element_split=pricer)
        try:
            assert seen, "the element pricer was never called"
            return len(ballot), len(seen), seen[0]
        finally:
            ballot.close()
    finally:
        arcs.close()
        graph.close()
step("Ballot element_split", ballot_with_pricer)

def arcs_row():
    arcs = _solve.Arcs()
    try:
        arcs.add("a0", "0x" + "ab" * 20, 0, 0, 1, 2, "0x" + "11" * 20,
                 "0x" + "22" * 20, 0, 1, 1.5, 0.0, 9.0, 2.0, 1e-4,
                 10 ** 30, 6, 1e6, 0.0, "note")
        row = arcs.row(0)
        return {k: row[k] for k in ("id", "i", "j", "reserve_in",
                                    "decimals_in", "note")}
    finally:
        arcs.close()
step("Arcs.row", arcs_row)

"\n".join(report)
