// Runs one solve through the wasm module and prints the answer as hex.
//
// The point is byte equality with the native extension: same crate, same
// compiler, two targets.  Only `sqrt` is a non-trivially-rounded operation in
// the solver and IEEE requires it be correctly rounded on both, so anything
// other than an exact match is a marshalling bug -- which is what this is
// here to catch.  Hex rather than JSON numbers because a float printed as
// text and read back is not the question being asked.
//
//     node tests/js/solver_harness.mjs <pkg-dir> < problem.json
//
// stdin is `{op, ...}`; stdout is one JSON object.  See
// tests/test_wasm_differential.py, which is the only caller.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const pkg = resolve(process.argv[2]);
const glue = await import(pkg + "/erouter_wasm.js");
// Node's fetch has no `file:` scheme, so the bytes are handed over directly
// rather than letting the glue fetch them the way a browser does.
await glue.default({ module_or_path: readFileSync(pkg + "/erouter_wasm_bg.wasm") });

const hex = (typed) => Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString("hex");

const job = JSON.parse(readFileSync(0, "utf8"));
const out = {};

if (job.op === "solve") {
  const p = new glue.Problem(
    Int32Array.from(job.tau), Int32Array.from(job.sig),
    Float64Array.from(job.g), Float64Array.from(job.eps),
    Float64Array.from(job.cap.map((v) => (v === null ? Infinity : v))),
    job.n_nodes,
  );
  const r = p.solve(
    job.src, job.dst, job.psi_total,
    Uint8Array.from(job.a0 ?? []), Uint8Array.from(job.forbidden ?? []),
    Uint32Array.from((job.pinned ?? []).map((e) => e[0])),
    Float64Array.from((job.pinned ?? []).map((e) => e[1])),
    job.tol, job.maxit, job.min_flow, job.gas_cost, job.partial_ok, job.rank1,
  );
  Object.assign(out, {
    psi: hex(r.psi), u: hex(r.u), psiUpper: hex(r.psiUpper), rho: hex(r.rho),
    active: hex(r.active), upper: hex(r.upper),
    pivots: r.pivots, cholFailures: r.cholFailures, keepChanges: r.keepChanges,
    refits: r.refits, feasible: r.feasible, reason: r.reason,
  });
  r.free();
  p.free();
} else if (job.op === "calibrate") {
  const nan = (v) => (v === null || v === undefined ? NaN : v);
  const c = glue.calibrate(
    Float64Array.from(job.deltas), Float64Array.from(job.quotes),
    nan(job.delta_bar), job.structural_flag, job.drift_tol,
    nan(job.cap), nan(job.f_at_cap), job.quantum,
  );
  Object.assign(out, {
    a: hex(Float64Array.of(c.a)), b: hex(Float64Array.of(c.b)),
    cap: hex(Float64Array.of(c.cap)), clamped: c.clamped,
    convexFlag: c.convexFlag, flag: c.flag,
    drift: hex(Float64Array.of(c.drift)), eta: hex(Float64Array.of(c.eta)),
    splitHint: c.splitHint,
    calibDelta: hex(Float64Array.of(c.calibDelta)),
    tangentDelta: hex(Float64Array.of(c.tangentDelta)),
    note: c.note,
  });
  c.free();
} else if (job.op === "cancel_cycles") {
  const r = glue.cancelCycles(
    Int32Array.from(job.tau), Int32Array.from(job.sig),
    Float64Array.from(job.psi), job.tol, job.n_nodes ?? 0,
  );
  Object.assign(out, { flow: hex(r.flow), removed: r.removed });
  r.free();
} else if (job.op === "find_cycle") {
  const arcs = glue.findCycle(
    Int32Array.from(job.tau), Int32Array.from(job.sig), job.n_nodes ?? 0,
  );
  out.arcs = Array.from(arcs);
} else if (job.op === "shortest_path") {
  const p = new glue.Problem(
    Int32Array.from(job.tau), Int32Array.from(job.sig),
    Float64Array.from(job.g), Float64Array.from(job.eps),
    Float64Array.from(job.cap.map((v) => (v === null ? Infinity : v))),
    job.n_nodes,
  );
  const r = p.shortestPath(
    job.src, job.dst,
    job.banned_arcs ? Uint32Array.from(job.banned_arcs) : undefined,
    job.banned_nodes ? Uint32Array.from(job.banned_nodes) : undefined,
    job.weights ? Float64Array.from(job.weights) : undefined,
    job.max_hops,
  );
  Object.assign(out, {
    arcs: Array.from(r.arcs), length: hex(Float64Array.of(r.length)),
    found: r.found, negativeCycle: Array.from(r.negativeCycle),
  });
  r.free();
  p.free();
} else if (job.op === "price") {
  // The models are built the way a warm builds them -- every parameter as a
  // decimal string -- and then a batch names them by index, so this exercises
  // the marshalling the browser will actually use.
  const pools = new glue.Pools();
  for (const m of job.models) {
    if (m.kind === "stableswap") {
      pools.addStableswap(m.balances, m.rates, m.amp, m.fee,
        m.offpeg_fee_multiplier, m.a_precision, m.fee_on_xp, m.subtract_one,
        m.admin_fee ?? undefined);
    } else if (m.kind === "lp_withdraw" || m.kind === "lp_deposit") {
      pools.addStableLp(m.balances, m.rates, m.amp, m.fee,
        m.offpeg_fee_multiplier, m.a_precision, m.fee_on_xp, m.subtract_one,
        m.total_supply, m.kind === "lp_deposit", m.admin_fee ?? undefined);
    } else if (m.kind === "tri_lp") {
      pools.addTricryptoLp(m.balances, m.precisions, m.price_scale, m.d, m.amp,
        m.gamma, m.mid_fee, m.out_fee, m.fee_gamma, m.legacy, m.a_multiplier,
        m.total_supply);
    } else if (m.kind === "vault") {
      pools.addVault(m.num, m.den, m.cap);
    } else if (m.kind === "one_to_one") {
      pools.addOneToOne();
    } else if (m.kind === "twocrypto") {
      pools.addTwocrypto(m.balances, m.precisions, m.price_scale, m.d, m.amp,
        m.gamma, m.mid_fee, m.out_fee, m.fee_gamma, m.stable, m.v21,
        m.legacy_fee, m.legacy_pool, m.legacy_mul2);
    } else {
      pools.addTricrypto(m.balances, m.precisions, m.price_scale, m.d, m.amp,
        m.gamma, m.mid_fee, m.out_fee, m.fee_gamma, m.legacy, m.a_multiplier);
    }
  }
  // Each dx as lo/hi halves of a u128: there is no BigUint128Array, and a
  // u64 would cap a probe at eighteen tokens at eighteen decimals.
  const M = (1n << 64n) - 1n;
  const dx = new BigUint64Array(job.dx.length * 2);
  job.dx.forEach((v, k) => {
    const b = BigInt(v);
    dx[2 * k] = b & M;
    dx[2 * k + 1] = b >> 64n;
  });
  const r = pools.price(
    Uint32Array.from(job.which), Uint8Array.from(job.i), Uint8Array.from(job.j),
    dx, job.fast);
  const values = r.values, ok = r.ok;
  // Reassembled here rather than crossing as BigInt per probe: the buffer is
  // lo/hi halves, which is how a u128 fits a BigUint64Array.
  const dy = [];
  for (let k = 0; k < ok.length; k++) {
    dy.push(ok[k] ? ((values[2 * k + 1] << 64n) | values[2 * k]).toString() : null);
  }
  Object.assign(out, { dy });
  r.free();
  pools.free();
} else if (job.op === "element_split") {
  const pools = new glue.Pools();
  const m = job.model;
  pools.addStableswap(m.balances, m.rates, m.amp, m.fee,
    m.offpeg_fee_multiplier, m.a_precision, m.fee_on_xp, m.subtract_one,
    m.admin_fee ?? undefined);
  const d = BigInt(job.dx);
  const r = pools.elementSplit(0, job.i, job.j1, job.j2,
    d & ((1n << 64n) - 1n), d >> 64n);
  Object.assign(out, { split: r ? [r.a, r.b] : null });
  if (r) r.free();
  pools.free();
} else if (job.op === "ladders") {
  // The whole refine stage as the browser would run it: build the coarse
  // ladders, plan what is missing, fold the answers in, fit.  u128 crosses as
  // lo/hi halves, which is why every amount here is a pair.
  const M = (1n << 64n) - 1n;
  const pack = (xs) => {
    const out = new BigUint64Array(xs.length * 2);
    xs.forEach((v, k) => { const b = BigInt(v); out[2 * k] = b & M; out[2 * k + 1] = b >> 64n; });
    return out;
  };
  const unpack = (buf) => {
    const out = [];
    for (let k = 0; k < buf.length; k += 2) out.push(((buf[k + 1] << 64n) | buf[k]).toString());
    return out;
  };
  const outHolder = out;
  const L = new glue.Ladders();
  for (const m of job.ladders) {
    L.add(m.decimals_in, m.decimals_out, BigInt(m.reserve_in) & M,
          BigInt(m.reserve_in) >> 64n, pack(m.deltas), pack(m.quotes), m.attempted);
  }
  const plan = L.planSized(Uint32Array.from(job.slots), pack(job.want),
                           Uint32Array.from(job.spans.map((v) => v * 2)));
  const at = Array.from(plan.slot);
  const deltas = unpack(plan.delta);
  const built = { slots: at, deltas };
  if (job.values) {
    L.absorb(Uint32Array.from(at), pack(deltas), pack(job.values),
             Uint8Array.from(job.status), job.names);
    built.points = job.ladders.map((_, k) => unpack(L.points(k)));
    built.attempted = job.ladders.map((_, k) => L.attempted(k));
  }
  if (job.fit) {
    const got = L.recalibrate(Uint32Array.from(job.fit), job.driftTol);
    // Hex, not JSON numbers: a fit's `cap` is routinely Infinity and `eta`
    // NaN, and JSON.stringify writes both as null.  Same reason the solver's
    // vectors cross as hex here.
    built.fits = got.map((f) => f === undefined ? null : {
      nums: hex(Float64Array.of(f.a, f.b, f.cap, f.drift, f.eta, f.calibDelta)),
      clamped: f.clamped, convexFlag: f.convexFlag, flag: f.flag,
    });
  }
  L.free();
  Object.assign(outHolder, built);
} else {
  throw new Error(`unknown op: ${job.op}`);
}

process.stdout.write(JSON.stringify(out));
