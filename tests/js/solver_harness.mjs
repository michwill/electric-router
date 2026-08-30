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
} else if (job.op === "graph") {
  // The whole assembly as the browser would run it: build, then read the
  // arrays back.  Hex for the floats -- a bit-exact comparison is the point,
  // and JSON cannot spell Infinity, which every uncapped arc carries.
  const g = glue.Graph.build(
    BigInt64Array.from(job.tau.map(BigInt)), BigInt64Array.from(job.sig.map(BigInt)),
    Float64Array.from(job.a), Float64Array.from(job.B), Float64Array.from(job.nu),
    job.Psi,
    job.cap ? Float64Array.from(job.cap.map((v) => (v === null ? Infinity : v))) : undefined,
    job.flagged ? Uint8Array.from(job.flagged) : undefined,
    job.clamped ? Uint8Array.from(job.clamped) : undefined,
    job.n_nodes, undefined, undefined, undefined,
    job.require ? Uint32Array.from(job.require) : undefined,
  );
  const built = {
    tau: Array.from(g.tau, Number), sig: Array.from(g.sig, Number),
    a: hex(g.a), B: hex(g.b), G: hex(g.g), eps: hex(g.eps), cap: hex(g.cap),
    flagged: Array.from(g.flagged), clamped: Array.from(g.clamped),
    nNodes: g.nNodes, illConditioned: hex(Float64Array.of(g.illConditioned)),
    condition: hex(Float64Array.of(g.condition())),
    sources: Array.from(g.sources()), sourceSpans: Array.from(g.sourceSpans()),
    dropped: Array.from(g.dropped()), droppedReason: g.droppedReason(),
  };
  if (job.scale !== undefined) {
    built.psiScaled = hex(Float64Array.of(g.scale(job.scale)));
    built.gScale = hex(Float64Array.of(g.gScale));
    built.scaledG = hex(g.g);
    built.scaledCap = hex(g.cap);
  }
  g.free();
  Object.assign(out, built);
} else if (job.op === "nodes") {
  // Built by the same calls a warm makes, then asked the same questions.
  const map = new glue.NodeMap();
  for (const t of job.tokens) map.addToken(t.address, t.symbol, t.decimals);
  for (const m of job.merges) {
    map.merge(m.kind, m.token, m.canonical, m.rate_num, m.rate_den, m.target);
  }
  Object.assign(out, {
    nNodes: map.nNodes(),
    mergedNodes: Array.from(map.mergedNodes()),
    node: job.ask.map((t) => (map.has(t) ? map.node(t) : null)),
    canonical: job.ask.map((t) => (map.has(t) ? map.canonical(t) : null)),
    symbol: job.ask.map((t) => map.symbol(t)),
    decimals: job.ask.map((t) => map.decimals(t)),
    rate: hex(Float64Array.from(job.ask.map((t) => map.rate(t)))),
    toCanonical: job.ask.map((t) => map.toCanonicalWei(t, job.amount)),
    fromCanonical: job.ask.map((t) => map.fromCanonicalWei(t, job.amount)),
    nodeSymbol: Array.from({ length: map.nNodes() }, (_, k) => map.nodeSymbol(k)),
    tokensOf: Array.from({ length: map.nNodes() }, (_, k) => map.tokensOf(k)),
    conversion: job.ask.map((t) => map.conversion(t)),
    conversionKinds: job.ask.map((t) => Array.from(map.conversionKinds(t))),
    isAlias: job.ask.map((t) => map.isAlias(t)),
    rescale: hex(Float64Array.from(glue.NodeMap.rescale(...job.rescale))),
  });
  map.free();
} else if (job.op === "element") {
  // The shape rules and the arithmetic, as the browser gets them.  Ports
  // cross as parallel coin and share arrays; a pair has no typed form.
  const shape = {};
  try {
    const el = new glue.Element(job.pool, job.n_coins,
      Int32Array.from(job.in_coins), BigInt64Array.from(job.in_bps.map(BigInt)),
      Int32Array.from(job.out_coins), BigInt64Array.from(job.out_bps.map(BigInt)));
    shape.inputs = Array.from(el.inputs(), Number);
    shape.outputs = Array.from(el.outputs(), Number);
    shape.ports = el.ports;
    el.free();
  } catch (e) {
    shape.error = String(e.message ?? e);
  }
  Object.assign(out, shape);
  if (job.models) {
    const pools = new glue.Pools();
    const m = job.models;
    const which = pools.addStableswap(m.balances, m.rates, m.amp, m.fee,
      m.offpeg_fee_multiplier, m.a_precision, m.fee_on_xp, m.subtract_one,
      m.admin_fee ?? undefined);
    let lp;
    if (job.supply) {
      lp = pools.addStableLp(m.balances, m.rates, m.amp, m.fee,
        m.offpeg_fee_multiplier, m.a_precision, m.fee_on_xp, m.subtract_one,
        job.supply, false, m.admin_fee ?? undefined);
    }
    const ports = (coins, bps) => [Int32Array.from(coins),
                                   BigInt64Array.from(bps.map(BigInt))];
    try {
      out.dy = pools.elementEvaluate(which, lp, job.n_coins,
        ...ports(job.in_coins, job.in_bps), ...ports(job.out_coins, job.out_bps),
        job.dx);
    } catch (e) {
      out.evaluateError = String(e.message ?? e);
    }
    if (job.weights) {
      const best = pools.elementBestSplit(which, lp, job.n_coins,
        ...ports(job.in_coins, job.in_bps), ...ports(job.out_coins, job.out_bps),
        job.dx, job.weights);
      out.split = [Number(best.a), Number(best.b)];
      out.payout = hex(Float64Array.of(best.payout));
      best.free();
    }
    pools.free();
  }
} else if (job.op === "realize") {
  // A whole route, built the way a quote builds one.  Amounts cross as
  // decimal strings and the diagnostics as hex, for the usual reason: a
  // `cap_in` is routinely Infinity and JSON writes that as null.
  const map = new glue.NodeMap();
  for (const t of job.tokens) map.addToken(t.address, t.symbol, t.decimals);
  for (const m of job.merges) {
    map.merge(m.kind, m.token, m.canonical, m.rate_num, m.rate_den, m.target);
  }
  const arcs = new glue.Arcs();
  for (const a of job.arcs) {
    arcs.add(a.id, a.pool, a.kind, a.i, a.j, a.n_coins, a.token_in, a.token_out,
             a.tau, a.sigma, a.a, a.B, a.cap === null ? Infinity : a.cap,
             a.G, a.eps, a.reserve_in, a.decimals_in, a.tvl_usd,
             a.gamma_live === null ? NaN : a.gamma_live, a.note);
  }
  const route = glue.Route.realize(arcs, Float64Array.from(job.psi),
    Float64Array.from(job.nu), map, job.src, job.dst, job.amount_in,
    job.potentials ? Float64Array.from(job.potentials) : undefined);
  Object.assign(out, {
    wireLegs: route.wireLegs(),
    wireNumbers: Array.from(route.wireNumbers()),
    targets: route.targets(),
    kinds: Array.from(route.kinds()),
    tokensIn: route.tokensIn(),
    tokensOut: route.tokensOut(),
    amountsIn: route.amountsIn(),
    amountsOut: route.amountsOut(),
    arcIds: route.arcIds(),
    poolNames: route.poolNames(),
    numbers: hex(route.numbers()),
    modelled: Array.from(route.modelled()),
    isConversion: Array.from(route.isConversion()),
    slots: route.slots(),
    slotIndices: Array.from(route.slotIndices()),
    nodeOfSlot: Array.from(route.nodeOfSlot()),
    dstSlot: route.dstSlot,
    modelledOut: route.modelledOut,
    paths: route.paths(),
    warnings: route.warnings(),
    poolsUsed: route.poolsUsed(),
    maxTheta: hex(Float64Array.of(route.maxTheta())),
    routeConductance: hex(Float64Array.of(route.routeConductance())),
  });
  route.free();
  arcs.free();
  map.free();
} else if (job.op === "ballot") {
  // Generation and ranking as the browser runs them.  Paths and conflicts
  // cross flat with spans, and the quotes come in as decimal strings.
  const map = new glue.NodeMap();
  for (const t of job.tokens) map.addToken(t.address, t.symbol, t.decimals);
  const g = glue.Graph.build(
    BigInt64Array.from(job.tau.map(BigInt)), BigInt64Array.from(job.sig.map(BigInt)),
    Float64Array.from(job.a), Float64Array.from(job.B), Float64Array.from(job.nu),
    job.Psi, undefined, undefined, undefined, job.n_nodes,
    undefined, undefined, false, undefined);
  const arcs = new glue.Arcs();
  for (const a of job.arcs) {
    arcs.add(a.id, a.pool, a.kind, a.i, a.j, a.n_coins, a.token_in, a.token_out,
             a.tau, a.sigma, a.a, a.B, a.cap === null ? Infinity : a.cap,
             a.G, a.eps, a.reserve_in, a.decimals_in, a.tvl_usd,
             a.gamma_live === null ? NaN : a.gamma_live, a.note);
  }
  const ballot = glue.Ballot.generate(g, arcs, job.src, job.dst, job.Psi,
    Float64Array.from(job.base_psi));
  const out0 = {
    labels: ballot.labels(),
    kinds: ballot.kinds(),
    nArcs: Array.from(ballot.nArcs()),
    modelledLoss: hex(ballot.modelledLoss()),
    solves: ballot.solves, pivots: ballot.pivots,
    skipped: ballot.skipped, skippedWide: ballot.skippedWide,
    psi: Array.from({ length: ballot.length() }, (_, k) => hex(ballot.psi(k))),
    // The helpers, so a mismatch above says which one moved.
    spread: Array.from(glue.Ballot.spread(Uint32Array.from(job.top_k), job.budget)),
    paths: Array.from(glue.Ballot.kShortestPaths(g, job.src, job.dst, job.k)),
    pathSpans: Array.from(glue.Ballot.kShortestPathSpans(g, job.src, job.dst, job.k)),
    carries: Array.from(glue.Ballot.carries(Float64Array.from(job.base_psi), job.Psi)),
  };
  ballot.realizeCandidates(arcs, Float64Array.from(job.nu), map, job.src_token,
                           job.dst_token, job.amount_in);
  out0.statuses = ballot.statuses();
  out0.ready = Array.from(ballot.ready());
  if (job.quotes) {
    ballot.verify(Uint32Array.from(out0.ready), job.quotes, undefined,
                  job.gas_price_wei, job.dst_wei_per_eth);
    // Re-read after verify: `realizeCandidates` leaves them "ready" and
    // verify moves them to "ok" or "reverted".
    out0.statuses = ballot.statuses();
    out0.ranks = Array.from(ballot.ranks());
    out0.gas = Array.from(ballot.gas());
    out0.verifiedOut = ballot.verifiedOut();
    const winner = ballot.best();
    out0.best = winner === undefined ? null : winner;
  }
  Object.assign(out, out0);
  ballot.free();
  arcs.free();
  g.free();
  map.free();
} else if (job.op === "stages") {
  // The quote's stages in order, as a browser would run them between fetches.
  const map = new glue.NodeMap();
  for (const t of job.tokens) map.addToken(t.address, t.symbol, t.decimals);
  for (const m of job.merges ?? []) {
    map.merge(m.kind, m.token, m.canonical, m.rate_num, m.rate_den, m.target);
  }
  const arcs = new glue.Arcs();
  for (const a of job.arcs) {
    arcs.add(a.id, a.pool, a.kind, a.i, a.j, a.n_coins, a.token_in, a.token_out,
             a.tau, a.sigma, a.a, a.B, a.cap === null ? Infinity : a.cap,
             a.G, a.eps, a.reserve_in, a.decimals_in, a.tvl_usd,
             a.gamma_live === null ? NaN : a.gamma_live, a.note);
  }
  const st = new glue.Stages(arcs);
  st.pruneDeadEndNodes(job.src, job.dst);
  const afterPrune = st.arcIds();
  st.restrictToComponent(job.dst, job.n_nodes);
  const afterRestrict = st.arcIds();
  const paired = st.pairDirections();
  const g = st.assemble(Float64Array.from(job.nu), job.Psi, map, job.src, job.dst);
  Object.assign(out, {
    afterPrune, afterRestrict, paired,
    arcIds: st.arcIds(),
    counters: st.counters(),
    counterValues: Array.from(st.counterValues()),
    warnings: st.warnings(),
    arcNumbers: hex(st.arcNumbers()),
    arcFlags: Array.from(st.arcFlags()),
    gammaLive: hex(st.gammaLive()),
    reverseIds: st.reverseIds(),
    G: hex(g.g), eps: hex(g.eps),
    kclTolerance: hex(Float64Array.of(glue.kclTolerance(job.Psi, 1.0))),
    kclDetail: hex(glue.kclDetail(g, Float64Array.from(job.psi), job.src, job.dst,
                                  job.Psi)),
    achievableKcl: hex(Float64Array.of(
      glue.achievableKcl(g, Uint8Array.from(job.active), job.dst))),
    dstPerEth: hex(Float64Array.of(
      glue.dstPerEth(map, Float64Array.from(job.nu), job.dst_token))),
    gasCost: hex(Float64Array.of(
      glue.gasCost(map, Float64Array.from(job.nu), job.dst_token,
                   job.gas_price_wei, job.g_scale))),
    quantum: hex(Float64Array.of(glue.quantum(job.decimals_out))),
  });
  g.free();
  st.free();
  arcs.free();
  map.free();
} else {
  throw new Error(`unknown op: ${job.op}`);
}

process.stdout.write(JSON.stringify(out));
