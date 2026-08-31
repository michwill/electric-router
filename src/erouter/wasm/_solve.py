"""`erouter_solve`, served by the wasm module.

Every signature here is the PyO3 extension's, because `core/accel.py` is not
allowed to know which one it got -- including the details that look
incidental: `solve` returns its vectors as `bytes` for `np.frombuffer`, and
`calibrate` returns a twelve-field tuple in `Calibration`'s declaration order.

The crossings are typed arrays.  `to_js` on a contiguous numpy buffer picks
the matching TypedArray by format, which is what the module's `&[f64]` and
`&[i32]` parameters want; a Python list would work and copy element by
element.  Everything that comes back is freed here rather than left to the
JS GC, which does not know how much wasm memory a handle is holding.
"""

from __future__ import annotations

import contextlib

import numpy as np

_mod = None


def bind(module) -> None:
    global _mod
    _mod = module


def version() -> str:
    return str(_mod.version())


__version__ = "wasm"
__doc_module__ = "The router's active-set QP, in Rust, in a browser."


def _typed(values, dtype):
    """A sequence as the TypedArray the module's slice parameter wants.

    Through a `memoryview`, which is the conversion Pyodide documents: it
    picks the array type from the buffer's format, so `float64` arrives as
    `Float64Array` and `int32` as `Int32Array` with no per-element work.  A
    plain Python list would also be accepted by wasm-bindgen and would copy
    one element at a time.
    """
    from pyodide.ffi import to_js

    return to_js(memoryview(np.ascontiguousarray(values, dtype=dtype)))


def _f64(values):
    return _typed(values, np.float64)


def _i32(values):
    return _typed(values, np.int32)


def _u32(values):
    return _typed(values, np.uint32)


def _u8(values):
    return _typed(values, np.uint8)


def _bytes(buffer) -> bytes:
    """A `Float64Array` or `Uint8Array` as Python bytes.

    Not destroyed afterwards, and that is deliberate rather than an omission:
    a Pyodide `JsBuffer` has `to_bytes` and no `destroy` -- `AttributeError:
    destroy` was what the browser answered for every quote, while the same
    code answered fine in CPython, because there is no JsBuffer there to
    differ.  Each of these is a fresh TypedArray the getter just made, so
    letting the JS collector have it is also the right thing.
    """
    return buffer.to_bytes()


def _nan(value) -> float:
    """`None` as NaN, which is how an `f64` parameter carries an absent value."""
    return float("nan") if value is None else float(value)


def _pack(result) -> dict:
    """A `SolveResult` as the dict PyO3 returns, then freed."""
    try:
        return {
            "psi": _bytes(result.psi),
            "u": _bytes(result.u),
            "active": _bytes(result.active),
            "upper": _bytes(result.upper),
            "psi_upper": _bytes(result.psiUpper),
            "rho": _bytes(result.rho),
            "pivots": int(result.pivots),
            "chol_failures": int(result.cholFailures),
            "keep_changes": int(result.keepChanges),
            "refits": int(result.refits),
            "timings": [int(v) for v in result.timings.to_py()],
            "feasible": bool(result.feasible),
            "reason": str(result.reason),
        }
    finally:
        result.free()


class Problem:
    """The graph, resident on the wasm side across a quote's many solves."""

    def __init__(self, tau, sig, g, eps, cap, n_nodes):
        self._inner = _mod.Problem.new(
            _i32(tau), _i32(sig), _f64(g), _f64(eps), _f64(cap), int(n_nodes)
        )

    @property
    def m(self) -> int:
        return int(self._inner.m)

    def shortest_path(self, src, dst, banned_arcs=None, banned_nodes=None,
                      weights=None, max_hops=8):
        got = self._inner.shortestPath(
            int(src), int(dst),
            None if not banned_arcs else _u32(banned_arcs),
            None if not banned_nodes else _u32(banned_nodes),
            None if weights is None else _f64(weights),
            int(max_hops),
        )
        try:
            return {
                "arcs": [int(v) for v in got.arcs.to_py()],
                "length": float(got.length),
                "found": bool(got.found),
                "negative_cycle": [int(v) for v in got.negativeCycle.to_py()],
            }
        finally:
            got.free()

    def solve(self, src, dst, psi_total, a0=None, forbidden=None, pinned=None,
              tol=None, maxit=600, min_flow=0.0, gas_cost=0.0, partial_ok=False,
              rank1=None):
        # An empty array says "absent": a nullable typed array would be a
        # second shape for the module to check on the per-keystroke path.
        pins = list(pinned or [])
        return _pack(self._inner.solve(
            int(src), int(dst), float(psi_total),
            _u8([] if a0 is None else np.asarray(a0, bool)),
            _u8([] if forbidden is None else np.asarray(forbidden, bool)),
            _u32([p for p, _ in pins]),
            _f64([v for _, v in pins]),
            float(TOL if tol is None else tol),
            int(maxit), float(min_flow), float(gas_cost), bool(partial_ok),
            True if rank1 is None else bool(rank1),
        ))

    def close(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.free()

    def __del__(self):  # pragma: no cover - collection order is the GC's
        with contextlib.suppress(Exception):
            self.close()


def _i64(values):
    """`i64` slices -- `Graph`'s `tau`/`sig`, which are wider than `Problem`'s."""
    return _typed(values, np.int64)


def _u64(values):
    return _typed(values, np.uint64)


#: `u128` crosses wasm as a `[lo, hi]` pair of `u64`, because that is the
#: widest integer a typed array carries.  The PyO3 extension takes one Python
#: int, so the pairing is this shim's job on the way in and out.
_MASK64 = (1 << 64) - 1


def _split128(values):
    flat = np.empty(len(values) * 2, dtype=np.uint64)
    for k, value in enumerate(values):
        n = int(value)
        flat[2 * k] = n & _MASK64
        flat[2 * k + 1] = (n >> 64) & _MASK64
    return flat


def _join128(flat, k):
    return int(flat[2 * k]) | (int(flat[2 * k + 1]) << 64)


def _bigint(value):
    """A JS `BigInt`, which a scalar `u64` parameter needs.

    Pyodide sends a Python int as a JS *Number* whenever it fits in a double,
    and wasm-bindgen's `u64` import then throws `Cannot convert a Number to a
    BigInt`.  Small amounts are exactly the ones that fit, so this is not an
    edge case -- it is every ordinary call.
    """
    from js import BigInt

    return BigInt(str(int(value)))


class Graph:
    """Solver arrays, resident on the wasm side.

    Only `from_arrays` is bound: the browser reaches this module after
    `core.graph` has already built, scaled and clamped, and `build` would
    re-derive `G` from `a`, `B` and `nu`.
    """

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def from_arrays(tau, sig, g, eps, cap, flagged, n_nodes):
        return Graph(_mod.Graph.fromArrays(
            _i64(tau), _i64(sig), _f64(g), _f64(eps), _f64(cap),
            _u8(np.asarray(flagged, bool)), int(n_nodes),
        ))

    @property
    def n_nodes(self) -> int:
        return int(self._inner.nNodes)

    def __len__(self) -> int:
        return int(self._inner.length())

    def close(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.free()

    def __del__(self):  # pragma: no cover - collection order is the GC's
        with contextlib.suppress(Exception):
            self.close()


class Arcs:
    """The arc list, built once a quote and read many times behind one call."""

    def __init__(self):
        self._inner = _mod.Arcs.new()

    def add(self, id, pool, kind, i, j, n_coins, token_in, token_out, tau,
            sigma, a, b, cap, g, eps, reserve_in, decimals_in, tvl_usd,
            gamma_live, note="", calib_delta=0.0, decimals_out=18):
        # `reserve_in` is a `u256` on the chain and a decimal string here; the
        # PyO3 side takes the int, so the conversion belongs to this shim.
        return int(self._inner.add(
            str(id), str(pool), int(kind), int(i), int(j), int(n_coins),
            str(token_in), str(token_out), int(tau), int(sigma),
            float(a), float(b), float(cap), float(g), float(eps),
            str(int(reserve_in)), int(decimals_in), float(tvl_usd),
            float(gamma_live), None if note is None else str(note),
            float(calib_delta), int(decimals_out),
        ))

    def row(self, k):
        got = self._inner.row(int(k)).to_py()
        names = ("id", "pool", "kind", "i", "j", "n_coins", "token_in",
                 "token_out", "tau", "sigma", "a", "b", "reserve_in",
                 "decimals_in", "decimals_out", "tvl_usd", "note")
        return dict(zip(names, got, strict=True))

    def __len__(self) -> int:
        return int(self._inner.length())

    def close(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.free()

    def __del__(self):  # pragma: no cover - collection order is the GC's
        with contextlib.suppress(Exception):
            self.close()


class Ballot:
    """One generation's candidates, and what ranking made of them."""

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def generate(graph, arcs, src, dst, psi_total, base_psi, *,
                 base_certificate=False, max_candidates=20, top_k=None,
                 gas_floor=0.0, max_legs=32, max_slots=8, element_split=None):
        proxy = None
        if element_split is not None:
            from pyodide.ffi import create_proxy, to_js

            def priced(k1, k2, psi1, psi2):
                # The module reads the answer as a JS array of two numbers and
                # treats anything else as "no opinion", so an empty one is how
                # a refusal crosses.  A Python tuple would arrive as a proxy
                # with no `length` and be refused for the wrong reason.
                got = element_split(int(k1), int(k2), float(psi1), float(psi2))
                if not got:
                    return to_js([])
                return to_js([float(got[0]), float(got[1])])

            proxy = create_proxy(priced)
        try:
            inner = _mod.Ballot.generate(
                graph._inner, arcs._inner, int(src), int(dst),
                float(psi_total), _f64(base_psi),
                bool(base_certificate), int(max_candidates),
                None if top_k is None else _u32(top_k),
                float(gas_floor), int(max_legs), int(max_slots), proxy,
            )
        finally:
            # The module keeps no reference once `generate` returns, so the
            # proxy is destroyed here rather than leaked for the page's life.
            if proxy is not None:
                proxy.destroy()
        return Ballot(inner)

    def labels(self):
        return [str(v) for v in self._inner.labels().to_py()]

    def kinds(self):
        return [str(v) for v in self._inner.kinds().to_py()]

    def reasons(self):
        return [str(v) for v in self._inner.reasons().to_py()]

    def certificates(self):
        return [bool(v) for v in self._inner.certificates().to_py()]

    def n_arcs(self):
        return [int(v) for v in self._inner.nArcs().to_py()]

    def modelled_loss(self):
        return [float(v) for v in self._inner.modelledLoss().to_py()]

    def psi(self, at):
        return [float(v) for v in self._inner.psi(int(at)).to_py()]

    @property
    def skipped(self) -> int:
        return int(self._inner.skipped)

    @property
    def skipped_wide(self) -> int:
        return int(self._inner.skippedWide)

    @property
    def solves(self) -> int:
        return int(self._inner.solves)

    @property
    def pivots(self) -> int:
        return int(self._inner.pivots)

    def __len__(self) -> int:
        return int(self._inner.length())

    def close(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.free()

    def __del__(self):  # pragma: no cover - collection order is the GC's
        with contextlib.suppress(Exception):
            self.close()


class Pools:
    """The exact pool models, resident across a warm's probes."""

    def __init__(self):
        self._inner = _mod.Pools.new()

    def add_stableswap(self, balances, rates, amp, fee, offpeg_fee_multiplier,
                       a_precision, fee_on_xp, subtract_one, admin_fee=None):
        return int(self._inner.addStableswap(
            _strs(balances), _strs(rates), str(amp), str(fee),
            str(offpeg_fee_multiplier), str(a_precision), bool(fee_on_xp),
            bool(subtract_one), None if admin_fee is None else str(admin_fee)))

    def add_stable_lp(self, balances, rates, amp, fee, offpeg_fee_multiplier,
                      a_precision, fee_on_xp, subtract_one, total_supply,
                      deposit, admin_fee=None):
        return int(self._inner.addStableLp(
            _strs(balances), _strs(rates), str(amp), str(fee),
            str(offpeg_fee_multiplier), str(a_precision), bool(fee_on_xp),
            bool(subtract_one), str(total_supply), bool(deposit),
            None if admin_fee is None else str(admin_fee)))

    def add_twocrypto(self, balances, precisions, price_scale, d, amp, gamma,
                      mid_fee, out_fee, fee_gamma, stable, v21, legacy_fee,
                      legacy_pool, legacy_mul2):
        return int(self._inner.addTwocrypto(
            _strs(balances), _strs(precisions), str(price_scale), str(d),
            str(amp), str(gamma), str(mid_fee), str(out_fee), str(fee_gamma),
            bool(stable), bool(v21), bool(legacy_fee), bool(legacy_pool),
            bool(legacy_mul2)))

    def add_tricrypto(self, balances, precisions, price_scale, d, amp, gamma,
                      mid_fee, out_fee, fee_gamma, legacy, a_multiplier):
        return int(self._inner.addTricrypto(
            _strs(balances), _strs(precisions), _strs(price_scale), str(d),
            str(amp), str(gamma), str(mid_fee), str(out_fee), str(fee_gamma),
            bool(legacy), str(a_multiplier)))

    def add_tricrypto_lp(self, balances, precisions, price_scale, d, amp,
                         gamma, mid_fee, out_fee, fee_gamma, legacy,
                         a_multiplier, total_supply):
        return int(self._inner.addTricryptoLp(
            _strs(balances), _strs(precisions), _strs(price_scale), str(d),
            str(amp), str(gamma), str(mid_fee), str(out_fee), str(fee_gamma),
            bool(legacy), str(a_multiplier), str(total_supply)))

    def add_vault(self, num, den, cap):
        return int(self._inner.addVault(str(num), str(den), str(cap)))

    def add_one_to_one(self):
        return int(self._inner.addOneToOne())

    def element_split(self, which, i, j1, j2, dx):
        n = int(dx)
        got = self._inner.elementSplit(
            int(which), int(i), int(j1), int(j2),
            _bigint(n & _MASK64), _bigint((n >> 64) & _MASK64))
        if got is None:
            return None
        try:
            return (int(got.a), int(got.b))
        finally:
            got.free()

    def price(self, which, i, j, dx, fast=False):
        """A batch, `None` where the pool would refuse."""
        got = self._inner.price(
            _u32(which), _u8(i), _u8(j), _u64(_split128(dx)), bool(fast))
        try:
            ok = got.ok.to_py()
            values = got.values.to_py()
            return [_join128(values, k) if ok[k] else None
                    for k in range(len(ok))]
        finally:
            got.free()

    def __len__(self) -> int:
        return int(self._inner.length)

    def close(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.free()

    def __del__(self):  # pragma: no cover - collection order is the GC's
        with contextlib.suppress(Exception):
            self.close()


def _strs(values):
    return [str(v) for v in values]


#: `core.solve.TOL`, repeated rather than imported: this module stands in for
#: an extension, and an extension does not import from `erouter.core`.
TOL = 1e-10


def solve(tau, sig, g, eps, cap, n_nodes, src, dst, psi_total, a0=None,
          forbidden=None, pinned=None, tol=None, maxit=600, min_flow=0.0,
          gas_cost=0.0, partial_ok=False, rank1=None):
    """One solve on a graph that is not resident.  `Problem` is the hot path."""
    problem = Problem(tau, sig, g, eps, cap, n_nodes)
    try:
        return problem.solve(
            src, dst, psi_total, a0=a0, forbidden=forbidden, pinned=pinned,
            tol=tol, maxit=maxit, min_flow=min_flow, gas_cost=gas_cost,
            partial_ok=partial_ok, rank1=rank1,
        )
    finally:
        problem.close()


def calibrate(deltas, quotes, delta_bar=None, structural_flag=False,
              drift_tol=0.05, cap=None, f_at_cap=None, quantum=0.0):
    """The twelve `Calibration` fields, in declaration order.

    Raises `ValueError` as PyO3 does, so `accel.calibrate_ladder` can turn it
    into the `CalibrationError` its callers catch.
    """
    from pyodide.ffi import JsException

    try:
        got = _mod.calibrate(
            _f64(deltas), _f64(quotes), _nan(delta_bar), bool(structural_flag),
            float(drift_tol), _nan(cap), _nan(f_at_cap), float(quantum),
        )
    except JsException as exc:
        raise ValueError(str(exc)) from None
    try:
        return (
            float(got.a), float(got.b), float(got.cap), bool(got.clamped),
            bool(got.convexFlag), str(got.flag), float(got.drift),
            float(got.eta), bool(got.splitHint), float(got.calibDelta),
            float(got.tangentDelta), str(got.note),
        )
    finally:
        got.free()


def cancel_cycles(tau, sig, psi, tol=1e-12, n_nodes=None):
    got = _mod.cancelCycles(
        _i32(tau), _i32(sig), _f64(psi), float(tol),
        0 if n_nodes is None else int(n_nodes),
    )
    try:
        flow = np.frombuffer(_bytes(got.flow), dtype=np.float64)
        return flow.tolist(), int(got.removed)
    finally:
        got.free()


def find_cycle(tau, sig, n_nodes=None):
    arcs = _mod.findCycle(_i32(tau), _i32(sig),
                          0 if n_nodes is None else int(n_nodes))
    got = [int(v) for v in arcs.to_py()]
    # An empty array is how the module spells `None`; a cycle is never empty.
    return got or None


def split_ascend(curves, src_of, dst_of, static_share, heads, tails, slots,
                 dst_slot, amount_in, start, free, min_weight, iters, sweeps,
                 window, sweep_tol):
    """The split search.  Everything ragged is flattened with offsets.

    A JSON payload would be one line instead of thirty, and this runs on the
    per-keystroke path: the curves alone are ~10,000 floats, which is a
    200 kB parse per quote for data that a typed array carries as bytes.
    """
    curve_x, curve_u, curve_slope = [], [], []
    curve_off, slope_off = [0], [0]
    rate0, tail_values = [], []
    for x, u, slope, r0, tail in curves:
        curve_x.extend(x)
        curve_u.extend(u)
        curve_slope.extend(slope)
        curve_off.append(len(curve_x))
        # Its own offsets: `slope` is `diff(u) / diff(x)`, so it is one shorter
        # than the points it was fitted between.
        slope_off.append(len(curve_slope))
        rate0.append(r0)
        tail_values.append(tail)

    heads_flat, heads_off = [], [0]
    for row in heads:
        heads_flat.extend(row)
        heads_off.append(len(heads_flat))

    start_flat, start_off = [], [0]
    for row in start:
        start_flat.extend(row)
        start_off.append(len(start_flat))

    got = _mod.splitAscend(
        _f64(curve_x), _f64(curve_u), _f64(curve_slope),
        _u32(curve_off), _u32(slope_off), _f64(rate0), _f64(tail_values),
        _u32(src_of), _u32(dst_of),
        # NaN is "not fixed", the same spelling `calibrate` uses for absent.
        _f64([_nan(v) for v in static_share]),
        _u32(heads_flat), _u32(heads_off), _u32(tails),
        int(slots), int(dst_slot), float(amount_in),
        _f64(start_flat), _u32(start_off),
        _u32([s for s, _ in free]), _u32([i for _, i in free]),
        float(min_weight), int(iters), int(sweeps), float(window),
        float(sweep_tol),
    )
    try:
        weights = np.frombuffer(_bytes(got.weights), dtype=np.float64)
        offsets = np.frombuffer(_bytes(got.offsets), dtype=np.uint32)
        rows = [weights[offsets[k]:offsets[k + 1]].tolist()
                for k in range(len(offsets) - 1)]
        return rows, float(got.best), int(got.evaluations)
    finally:
        got.free()
