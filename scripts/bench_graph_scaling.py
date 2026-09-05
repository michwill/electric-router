#!/usr/bin/env python3
"""Does the graph outgrow a dense Laplacian, and where?

Three arms, because the first two disagree and the third settles it.

*Synthetic, uniform.*  Every arc priced alike, so the optimum spreads over the
whole graph and the active set becomes the graph: 3,399 of 3,999 arcs at 1,000
nodes, and a solve of 106 s.  A worst case, and it reads as one.

*Synthetic, hub-and-spoke* with fees over decades, which is the shape of a real
universe.  Better, but still 18% of arcs active at 3,000 nodes and 155 s, since
`laplacian` is dense and costs `pivots x kept^3 / 3` -- a model that predicted
136 s against the 155 s measured.

*Real*, by lowering the TVL floor until the universe doubles.  This is the arm
to believe, and it says the other two are pessimistic: over 2.43x the arcs the
solve grows as `arcs^1.06` -- essentially linear -- while `candidates` grows as
`arcs^1.68`.  The active set stays small, exactly as `linalg` assumed.

    min-tvl   pools   arcs   solve   pivots   candidates
     10,000     396    849   52 ms      771       141 ms
      1,000     584  1,321   65 ms      689       160 ms
        100     747  1,654  107 ms    1,514       305 ms
         10     931  2,065  133 ms    2,988       625 ms

So the first compute wall is candidate generation, not linear algebra -- and
`prototype_univ3_in_graph.py` shows tick decomposition adds no nodes at all, so
it does not touch the Laplacian's size.

The sparse arm is measured too.  `linalg`'s docstring records `splu 6272 us` at
n=299 against dense LU's 2125; neither reproduces here (260 us and 620 us), but
its conclusion holds for the reason it gives: the router factorises the *active
set*, which is ~10 arcs, and at n~10 everything is call overhead.

    uv run python scripts/bench_graph_scaling.py [--arm synthetic|sparse]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from erouter.core.graph import build, laplacian, scale
from erouter.core.linalg import DEFAULT_SOLVER
from erouter.core.seed import seed_subgraph
from erouter.core.solve import solve

HUBS = 12


def hubby(rng, n_nodes: int):
    """Most tokens reach the world through a hub; hubs are richly joined."""
    tau, sig = [], []
    for k in range(HUBS, n_nodes):
        for hub in rng.choice(HUBS, size=rng.integers(1, 3), replace=False):
            tau += [k, int(hub)]
            sig += [int(hub), k]
    for i in range(HUBS):
        for j in range(HUBS):
            if i != j:
                tau.append(i)
                sig.append(j)
    return np.array(tau, np.int64), np.array(sig, np.int64)


def synthetic(sizes: list[int]) -> None:
    rng = np.random.default_rng(11)
    print(f"{'nodes':>8}{'arcs':>10}{'build ms':>10}{'seed ms':>9}"
          f"{'solve ms':>10}{'active':>8}{'act %':>7}{'kept':>7}{'pivots':>8}")
    for n_nodes in sizes:
        tau, sig = hubby(rng, n_nodes)
        m = len(tau)
        # Fees over decades, as a real universe has them, and depth to match.
        a = 1.0 - 10 ** rng.uniform(-4.3, -1.7, m)
        deep = (tau < HUBS) & (sig < HUBS)
        B = np.where(deep, 10 ** rng.uniform(-10, -8, m),
                     10 ** rng.uniform(-7, -5, m))
        nu = np.ones(n_nodes)
        src, dst, Psi = HUBS + 1, HUBS + 2, 1_000.0

        t0 = time.perf_counter()
        g = build(tau, sig, a, B, nu, Psi, n_nodes=n_nodes,
                  merge_duplicates=False, require=(src, dst))
        build_ms = (time.perf_counter() - t0) * 1e3
        g, psi_scaled = scale(g, Psi)

        t0 = time.perf_counter()
        seed = seed_subgraph(g, src, dst, k=8)
        seed_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        report = solve(g, src, dst, psi_scaled, seed=seed, max_rounds=6)
        solve_ms = (time.perf_counter() - t0) * 1e3

        live = np.flatnonzero(np.asarray(report.solution.A))
        kept = len(set(g.tau[live]) | set(g.sig[live])) if live.size else 0
        print(f"{n_nodes:>8,}{m:>10,}{build_ms:>10.1f}{seed_ms:>9.1f}"
              f"{solve_ms:>10.1f}{live.size:>8,}{live.size / g.m * 100:>6.1f}%"
              f"{kept:>7,}{report.solution.pivots:>8,}", flush=True)


def sparse(sizes: list[int], reps: int = 12) -> None:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    rng = np.random.default_rng(3)

    def best(fn, *args, **kw):
        """Min of repeats: a single sample at a millisecond measures the machine."""
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(*args, **kw)
            out.append((time.perf_counter() - t0) * 1e3)
        return min(out)

    def dense_solve(matrix, rhs):
        return np.linalg.solve(matrix, rhs)

    def splu_solve(csc, rhs, **kw):
        return spla.splu(csc, **kw).solve(rhs)

    print(f"{'n':>8}{'nnz %':>8}{'router dense':>14}{'bare dense':>12}"
          f"{'splu default':>14}{'splu MMD':>11}")
    for n_nodes in sizes:
        tau, sig = hubby(rng, n_nodes)
        G = 10 ** rng.uniform(5, 9, len(tau))
        dense = laplacian(tau, sig, G, n_nodes, np.arange(1, n_nodes))
        rhs = rng.standard_normal(dense.shape[0])
        csc = sp.csc_matrix(dense)
        nnz = int(np.count_nonzero(dense)) / dense.shape[0] ** 2 * 100

        want = np.linalg.solve(dense, rhs)
        for name, got in (("splu", spla.splu(csc).solve(rhs)),
                          ("mmd", spla.splu(csc, permc_spec="MMD_AT_PLUS_A"
                                            ).solve(rhs))):
            err = float(np.max(np.abs(got - want))) / max(1e-30,
                                                          float(np.max(np.abs(want))))
            assert err < 1e-6, f"{name} disagrees at n={n_nodes}: {err:.2e}"

        print(f"{dense.shape[0]:>8,}{nnz:>7.2f}%"
              f"{best(DEFAULT_SOLVER.solve, dense, rhs):>14.2f}"
              f"{best(dense_solve, dense, rhs):>12.2f}"
              f"{best(splu_solve, csc, rhs):>14.2f}"
              f"{best(splu_solve, csc, rhs, permc_spec='MMD_AT_PLUS_A'):>11.2f}",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="synthetic", choices=("synthetic", "sparse"))
    p.add_argument("--sizes", default="300,1000,3000")
    args = p.parse_args(argv)
    sizes = [int(v) for v in args.sizes.split(",")]
    (synthetic if args.arm == "synthetic" else sparse)(sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
