"""The ROUTE() driver (spec §5.1).

    nu      <- reference_prices(pools, dst)          one Laplacian solve
    a, B    <- calibrate(pools, nu, X)               M2, vectorised
    G, eps  <- M3, M4
    S       <- seed_subgraph(src, dst, eps, G)
    repeat: solve on S -> price out all m -> extend S    until nothing violates
    realize -> legs

Takes an already-loaded universe and a `QuoterClient`, so it stays in `core`
and can run in the browser against a deployed quoter.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .calibrate import Calibration, CalibrationError, calibrate
from .candidates import Candidate, CandidateSet, generate
from .graph import MAX_CONDITION, ArcArrays, build, scale
from .nodes import NodeMap, rescale
from .pools import PoolSpec
from .prices import check_pair_drops, reference_prices
from .probe import (
    COARSE_GRID,
    RETRY_GRID,
    ArcRef,
    Ladder,
    collect,
    merge,
    plan_grid,
    plan_refine,
)
from .quoter import QuoterClient
from .realize import RealizedRoute, cancel_cycles, check_one_arc_per_pool, realize
from .refit import RefitReport, refit
from .seed import k_shortest_paths, seed_subgraph
from .solve import SolveReport, active_set_solve, solve
from .types import ArcKind, PoolArc, Probe
from .verify import realize_candidates, verify

# §12.4's flow-conservation gate, in two terms -- see `_kcl_tolerance`.
KCL_RELATIVE = 1e-8
KCL_ABSOLUTE = 1e-9


@dataclass(slots=True)
class RouteResult:
    route: RealizedRoute | None = None
    report: SolveReport | None = None
    arcs: list[PoolArc] = field(default_factory=list)
    graph: ArcArrays | None = None
    nu: np.ndarray | None = None
    nodes: NodeMap | None = None
    src_token: str = ""
    dst_token: str = ""
    amount_in: int = 0
    price_out_per_in: float = 0.0
    fee_bp: float = 0.0
    impact_bp: float = 0.0
    candidates: CandidateSet | None = None
    refit_report: RefitReport | None = None
    winner: Candidate | None = None
    verified_out: int | None = None
    timings: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pool_names: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.route is not None

    @property
    def certificate(self) -> bool:
        return bool(self.report and self.report.certificate)

    @property
    def certificate_reason(self) -> str | None:
        if self.certificate:
            return None
        return (self.report.reason if self.report else None) or "NO_SOLUTION"


class RoutingError(RuntimeError):
    pass


def build_arcs(
    pools: list[PoolSpec], nodes: NodeMap
) -> tuple[list[ArcRef], list[tuple[PoolSpec, int, int]]]:
    """Every quotable swap direction, with its probe target and metadata."""
    refs: list[ArcRef] = []
    meta: list[tuple[PoolSpec, int, int]] = []
    for pool in pools:
        kind = pool.swap_kind
        if kind is None or not pool.balances:
            continue
        for i, j in pool.swap_pairs():
            if i >= len(pool.balances) or pool.balances[i] <= 0:
                continue
            token_in, token_out = pool.coins[i].address, pool.coins[j].address
            if not (nodes.has(token_in) and nodes.has(token_out)):
                continue
            if nodes.node(token_in) == nodes.node(token_out):
                continue  # self-loop after merging; see the note in route()
            refs.append(
                ArcRef(
                    pool=pool.address,
                    kind=kind,
                    i=i,
                    j=j,
                    n_coins=pool.n_coins,
                    reserve_in=pool.balances[i],
                    decimals_in=pool.coins[i].decimals,
                    decimals_out=pool.coins[j].decimals,
                )
            )
            meta.append((pool, i, j))
    return refs, meta


def _to_arc(
    pool: PoolSpec, i: int, j: int, ref: ArcRef, fit: Calibration, nodes: NodeMap
) -> PoolArc:
    token_in, token_out = pool.coins[i].address, pool.coins[j].address
    a, B = rescale(fit.a, fit.B, nodes.rate(token_in), nodes.rate(token_out))
    cap = fit.cap
    if math.isfinite(cap):
        cap = cap * nodes.rate(token_in)
    return PoolArc(
        id=ref.id,
        pool=pool.address,
        kind=ArcKind(pool.swap_kind),
        i=i,
        j=j,
        n_coins=pool.n_coins,
        token_in=token_in,
        token_out=token_out,
        tau=nodes.node(token_in),
        sigma=nodes.node(token_out),
        a=a,
        B=B,
        cap=cap,
        calib_delta=fit.calib_delta,
        convex_flag=fit.convex_flag,
        clamped=fit.clamped,
        flag_reason=fit.flag_reason,
        drift=fit.drift,
        eta=fit.eta,
        reserve_in=ref.reserve_in,
        decimals_in=ref.decimals_in,
        decimals_out=ref.decimals_out,
        tvl_usd=pool.tvl_usd,
        note=pool.name,
    )


def calibrate_arcs(
    refs: list[ArcRef],
    meta: list[tuple[PoolSpec, int, int]],
    ladders: list[Ladder],
    nodes: NodeMap,
) -> tuple[list[PoolArc], list[str]]:
    arcs: list[PoolArc] = []
    dropped: list[str] = []
    lookup = {id(lad.arc): lad for lad in ladders}
    for ref, (pool, i, j) in zip(refs, meta, strict=True):
        ladder = next((lad for lad in ladders if lad.arc is ref), None) or lookup.get(id(ref))
        if ladder is None or not ladder.ok:
            dropped.append(f"{ref.id}: only {0 if ladder is None else len(ladder.deltas)} probes")
            continue
        deltas, quotes = ladder.as_float()
        try:
            fit = calibrate(deltas, quotes)
        except CalibrationError as exc:
            dropped.append(f"{ref.id}: {exc}")
            continue
        arcs.append(_to_arc(pool, i, j, ref, fit, nodes))
    return arcs, dropped


def prepare(
    pools: list[PoolSpec],
    nodes: NodeMap,
    client: QuoterClient,
    *,
    src_token: str,
    dst_token: str,
    extra_arcs: list[PoolArc] | None = None,
    timings: dict[str, float] | None = None,
) -> Prepared:
    """The size-independent half: probe, calibrate, restrict, price.

    Everything here is a function of the block and the pair, not the amount,
    so an interactive session pays for it once.
    """
    scratch = RouteResult(src_token=src_token.lower(), dst_token=dst_token.lower(),
                          nodes=nodes)
    clock = _Clock(timings if timings is not None else scratch.timings)
    src_node, dst_node = nodes.node(src_token), nodes.node(dst_token)

    # --- probe, pass 1: the whole universe, coarsely (§2.6) ---------------
    #
    # Pricing out every arc is what the §5.5 certificate rests on, and that
    # needs only `eps`, hence only `a`.  `B` matters only where flow goes.  So
    # the universe gets two points each and the full ladder is spent later on
    # the arcs that could carry something -- ~5,300 probes down to ~1,900.
    with clock("arcs"):
        refs, meta = build_arcs(pools, nodes)
        plan = plan_grid(refs, grid=COARSE_GRID)
    with clock("probe"):
        ladders = collect(plan, client.probe(plan.probes))
        retry = plan_refine(
            ladders, {lad.arc.id for lad in ladders if not lad.ok}, grid=RETRY_GRID
        )
        if retry.probes:
            merge(ladders, collect(retry, client.probe(retry.probes)))
            scratch.counters["probes_retried"] = len(retry)
    with clock("calibrate"):
        arcs, dropped = calibrate_arcs(
            plan.arcs, _align(plan.arcs, refs, meta), ladders, nodes
        )
    scratch.warnings.extend(dropped[:20])
    scratch.counters["pools"] = len(pools)
    scratch.counters["arcs_planned"] = len(refs)
    scratch.counters["probes"] = len(plan)
    scratch.counters["arcs_calibrated"] = len(arcs)
    # Mint/stake arcs are exactly linear by construction, so they are supplied
    # already calibrated rather than probed -- there is no curve to measure.
    if extra_arcs:
        arcs = arcs + [copy.copy(a) for a in extra_arcs]
        scratch.counters["stake_arcs"] = len(extra_arcs)
    if not arcs:
        raise RoutingError("no arc survived calibration")

    # Restrict to the part of the graph that can actually reach the
    # destination.  A disconnected island has no price relative to the
    # numeraire -- `reference_prices` returns 1.0 there as a placeholder, not
    # as a valuation -- and feeding those arcs in gives conductances twelve
    # orders of magnitude off, which wrecks the Laplacian's conditioning for
    # everything else.  Measured: meme-coin pools with no path to WETH produced
    # G ~ 3e12 against a physical maximum near 1e4.
    with clock("component"):
        arcs = _restrict_to_component(arcs, dst_node, nodes.n_nodes, scratch)
    if not arcs:
        raise RoutingError(f"{nodes.symbol(dst_token)} is not reachable from any pool")
    if not any(a.tau == src_node or a.sigma == src_node for a in arcs):
        raise RoutingError(
            f"no path from {nodes.symbol(src_token)} to {nodes.symbol(dst_token)}"
        )

    # --- reference prices (§4) -------------------------------------------
    with clock("prices"):
        a_vec = np.array([a.a for a in arcs])
        # Both directions are already present as separate arcs, so half weight
        # keeps a pool's influence from being counted twice.
        weights = np.array([max(a.tvl_usd, 1.0) / 2 for a in arcs])
        nu = reference_prices(
            np.array([a.tau for a in arcs], dtype=np.int64),
            np.array([a.sigma for a in arcs], dtype=np.int64),
            a_vec, weights, nodes.n_nodes, dst_node,
        )

    return Prepared(
        arcs=arcs, ladders=ladders, nu=nu, src_node=src_node, dst_node=dst_node,
        pool_names={a.pool.lower(): a.note for a in arcs},
        counters=dict(scratch.counters), warnings=list(scratch.warnings),
    )


@dataclass(slots=True)
class Prepared:
    """Everything about a (universe, src, dst) that does not depend on size.

    The split is not a convenience -- it is a property of the model.  Probe
    sizes are fractions of each pool's *reserves*, so the whole derivative
    measurement is amount-independent; and `G_p = nu_tau a_p / B_p`,
    `eps_p = 1 - a_p nu_sig / nu_tau` contain no `Psi` at all.  `Psi` enters
    only at the dust floor, the caps and the right-hand side of the solve.

    So quoting a second size reuses the probes (the RPC), the calibration and
    the reference-price fit, and re-runs only the solve.  `warm` carries the
    previous active set forward, which matters more than it sounds: within one
    active set the KKT system is *affine* in `Psi` -- `L u = Psi(e_src - e_dst)
    + B_A(G_A eps_A)` -- so a nearby size usually needs the same arcs
    conducting, and the solve converges in a handful of pivots instead of ~80.
    """

    arcs: list[PoolArc]
    ladders: list[Ladder]
    nu: np.ndarray
    src_node: int
    dst_node: int
    pool_names: dict[str, str] = field(default_factory=dict)
    warm: np.ndarray | None = None
    counters: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    quotes: int = 0


def route(
    pools: list[PoolSpec],
    nodes: NodeMap,
    client: QuoterClient,
    *,
    src_token: str,
    dst_token: str,
    amount_in: int,
    max_rounds: int = 8,
    seed_k: int = 10,
    verify_on_chain: bool = True,
    max_candidates: int = 20,
    gas_price_wei: int = 0,
    refit_rounds: int = 2,
    prepared: Prepared | None = None,
    extra_arcs: list[PoolArc] | None = None,
) -> RouteResult:
    result = RouteResult(
        src_token=src_token.lower(),
        dst_token=dst_token.lower(),
        amount_in=amount_in,
        nodes=nodes,
    )
    clock = _Clock(result.timings)

    if not nodes.has(src_token) or not nodes.has(dst_token):
        raise RoutingError("source or destination token is not in the universe")
    src_node, dst_node = nodes.node(src_token), nodes.node(dst_token)
    if src_node == dst_node:
        raise RoutingError(
            f"{nodes.symbol(src_token)} and {nodes.symbol(dst_token)} are the same "
            "node after merging; convert directly rather than routing"
        )

    if prepared is None:
        with clock("prepare"):
            prepared = prepare(
                pools, nodes, client, src_token=src_token, dst_token=dst_token,
                extra_arcs=extra_arcs, timings=result.timings,
            )
    else:
        result.counters["reused_preparation"] = 1

    # A fresh copy per quote: `_assemble` writes `G`/`eps` back onto the arcs
    # and the §8 refit re-anchors `B` at one size's realised flows, so handing
    # the same objects to the next quote would leak this size into it.
    arcs = [copy.copy(a) for a in prepared.arcs]
    nu, ladders = prepared.nu, prepared.ladders
    result.arcs = arcs
    result.nu = nu
    result.pool_names = dict(prepared.pool_names)
    result.counters.update(prepared.counters)
    result.warnings.extend(prepared.warnings)

    return _quote(
        result, clock, pools, nodes, client, arcs, ladders, nu,
        src_token=src_token, dst_token=dst_token, amount_in=amount_in,
        src_node=src_node, dst_node=dst_node, max_rounds=max_rounds,
        seed_k=seed_k, verify_on_chain=verify_on_chain,
        max_candidates=max_candidates, gas_price_wei=gas_price_wei,
        refit_rounds=refit_rounds, prepared=prepared,
    )


def _quote(
    result: RouteResult,
    clock,
    pools: list[PoolSpec],
    nodes: NodeMap,
    client: QuoterClient,
    arcs: list[PoolArc],
    ladders: list[Ladder],
    nu: np.ndarray,
    *,
    src_token: str,
    dst_token: str,
    amount_in: int,
    src_node: int,
    dst_node: int,
    max_rounds: int,
    seed_k: int,
    verify_on_chain: bool,
    max_candidates: int,
    gas_price_wei: int,
    refit_rounds: int,
    prepared: Prepared | None,
) -> RouteResult:
    """The size-dependent half: graph, solve, candidates, verify, refit."""
    amount_human = amount_in / 10 ** nodes.decimals(src_token) * nodes.rate(src_token)
    Psi = float(nu[src_node] * amount_human)
    if Psi <= 0:
        raise RoutingError("input amount prices to zero value")

    # --- graph (§3.1, §9.5-9.7) -----------------------------------------
    with clock("graph"):
        arcs, g = _assemble(arcs, nu, Psi, nodes, src_node, dst_node, result)
    with clock("seed"):
        g, Psi_scaled = scale(g, Psi)
        seed = seed_subgraph(g, src_node, dst_node, k=seed_k)

    # --- probe, pass 2: the full ladder, only where it can matter ---------
    with clock("refine"):
        wanted = {arcs[k].id for k in np.flatnonzero(seed)}
        wanted |= {a.id for a in arcs if a.tau in (src_node, dst_node) or a.sigma in (src_node, dst_node)}
        extra = plan_refine(ladders, wanted)
        result.counters["probes_refined"] = len(extra)
        if extra.probes:
            merge(ladders, collect(extra, client.probe(extra.probes)))
            refined = _recalibrate(arcs, ladders, nodes)
            result.counters["arcs_refined"] = refined
            if refined:
                arcs, g = _assemble(arcs, nu, Psi, nodes, src_node, dst_node, result)
                g, Psi_scaled = scale(g, Psi)
                seed = seed_subgraph(g, src_node, dst_node, k=seed_k)
    result.arcs = arcs
    result.graph = g

    # --- solve (§5.4, §5.5) ----------------------------------------------
    with clock("seed"):
        seed = seed_subgraph(g, src_node, dst_node, k=seed_k)
        # §5.4's warm start, and it is worth a lot here.  Starting with all m
        # arcs active means pivoting ~700 of them back *out* one at a time, and
        # every one of those pivots factorises a matrix the size of the whole
        # connected component.  Starting from one path only ever adds arcs, so
        # the matrix stays the size of the active set.
        warm_start = None
        if prepared is not None and prepared.warm is not None:
            # A previous size's support, carried by arc *id*: the dust floor is
            # a function of Psi, so a different size can drop different arcs and
            # index-based reuse would silently point at the wrong pools.
            #
            # Worth carrying because within one active set the KKT system is
            # affine in Psi -- L u = Psi (e_src - e_dst) + B_A (G_A eps_A) -- so
            # the set of conducting arcs is piecewise constant in size, and a
            # nearby amount usually lands in the same piece with nothing to do.
            wanted = set(prepared.warm.tolist()) if hasattr(prepared.warm, "tolist") else set(prepared.warm)
            reuse = [k for k, arc in enumerate(arcs) if arc.id in wanted]
            if reuse:
                warm_start = np.array(reuse, dtype=np.int64)
                result.counters["warm_start_arcs"] = len(reuse)
        if warm_start is None:
            best_path = k_shortest_paths(g, src_node, dst_node, k=1)
            warm_start = np.array(best_path[0]) if best_path else None
    with clock("solve"):
        report = solve(
            g, src_node, dst_node, Psi_scaled, seed=seed,
            max_rounds=max_rounds, A0=warm_start,
        )
    result.report = report
    if not report.solution.feasible:
        raise RoutingError(report.reason or "no feasible route")

    psi = report.solution.psi * g.g_scale
    psi, cycles = cancel_cycles(g.tau, g.sig, psi)
    if cycles:
        result.counters["cycles_cancelled"] = cycles
        result.warnings.append(
            f"{cycles} circulation(s) removed from the optimal flow: the model "
            "found a negative-eps loop it cannot execute as a one-way trade (§2.6)"
        )
    # Decompose the loss *here*, against the graph that produced this flow.
    # Candidate generation and the refit both mutate `g` -- the refit re-anchors
    # B at realised sizes, which at theta ~ 20% can move G by orders of
    # magnitude -- so pairing this psi with a later G reports nonsense
    # (measured: a "resistor" term of 683,668 bp on a 2.7 bp route).
    fee, impact = report.solution.loss_split(g)
    result.fee_bp = fee * g.g_scale / Psi * 10_000
    result.impact_bp = impact * g.g_scale / Psi * 10_000

    active = np.flatnonzero(psi > 0)
    if active.size == 0:
        raise RoutingError("the optimal flow is empty")

    # Carry the *support* to the next size, not the whole final basis.  `A`
    # ends column generation holding every arc that ever priced in -- 199 of
    # them here against 34 carrying flow -- and handing that back as a start
    # set makes the next solve evict the difference one pivot at a time
    # (measured: 18 pivots per quote against 2).  `candidates.py` warm-starts
    # from the circulation-free support for exactly this reason.
    if prepared is not None:
        prepared.warm = np.array([arcs[k].id for k in active], dtype=object)
        prepared.quotes += 1

    # §12.4: KCL must hold on the flow we are about to execute.  This is the
    # invariant that catches conjured or stranded flow before it reaches a leg.
    residual = _kcl_residual(g, psi, src_node, dst_node, Psi)
    result.counters["kcl_residual"] = residual
    tolerance = _kcl_tolerance(Psi, g.g_scale)
    if residual > tolerance:
        raise RoutingError(
            f"flow conservation is violated by {residual:.3e} of the routed value"
        )
    result.counters["active_arcs"] = int(active.size)
    result.counters["pivots"] = report.solution.pivots
    result.counters["cg_rounds"] = report.cg_rounds
    result.counters["arcs_priced_out"] = g.m

    # --- realize (§5.6) ---------------------------------------------------
    with clock("realize"):
        live = [arcs[k] for k in active]
        result.route = realize(
            live, psi[active], nu, nodes,
            src_token=src_token, dst_token=dst_token, amount_in=amount_in,
            potentials=report.solution.u,
        )
    conflicts = check_one_arc_per_pool(result.route)
    if conflicts:
        result.warnings.append(
            f"{len(conflicts)} pool(s) used more than once; a view-only quote "
            "cannot see its own earlier leg (§7 rule 1)"
        )

    # --- candidates and on-chain verification (§6, §7) --------------------
    if verify_on_chain:
        scaled = report.solution.psi.copy()
        with clock("candidates"):
            pool_set = generate(
                g, arcs, src_node, dst_node, Psi_scaled, report.solution,
                base_certificate=report.certificate, seed=seed,
                max_candidates=max_candidates,
            )
            for candidate in pool_set.candidates:
                candidate.psi = candidate.psi * g.g_scale
            del scaled
            realize_candidates(
                pool_set, arcs, nu, nodes,
                src_token=src_token, dst_token=dst_token, amount_in=amount_in,
                potentials=report.solution.u,
            )
        with clock("verify"):
            dst_wei_per_eth = _dst_per_eth(nodes, nu, dst_token)
            verify(
                pool_set, client, amount_in=amount_in,
                gas_price_wei=gas_price_wei, dst_wei_per_eth=dst_wei_per_eth,
            )
        result.candidates = pool_set
        result.counters["candidates"] = len(pool_set)
        result.counters["candidates_quoted"] = sum(
            1 for c in pool_set.candidates if c.verified_out is not None
        )
        result.counters["candidates_reverted"] = sum(
            1 for c in pool_set.candidates if c.status == "reverted"
        )

        # The safety floor: quote every obvious one-hop swap too, so the
        # winner can never be worse than something found by inspection.
        with clock("direct"):
            direct, direct_arcs = direct_candidates(
                pools, nodes, nu, src_token, dst_token, amount_in
            )
            for candidate, arc in zip(direct, direct_arcs, strict=True):
                trial = CandidateSet([candidate])
                realize_candidates(
                    trial, [arc], nu, nodes,
                    src_token=src_token, dst_token=dst_token, amount_in=amount_in,
                )
                if candidate.status == "ready":
                    pool_set.candidates.append(candidate)
            chains, chain_arcs = two_step_candidates(
                pools, nodes, nu, client, src_token, dst_token, amount_in
            )
            for candidate, pair in zip(chains, chain_arcs, strict=True):
                trial = CandidateSet([candidate])
                realize_candidates(
                    trial, pair, nu, nodes,
                    src_token=src_token, dst_token=dst_token, amount_in=amount_in,
                )
                if candidate.status == "ready":
                    pool_set.candidates.append(candidate)
            result.counters["direct_candidates"] = len(direct)
            result.counters["two_step_candidates"] = len(chains)
            if direct or chains:
                verify(
                    pool_set, client, amount_in=amount_in,
                    gas_price_wei=gas_price_wei, dst_wei_per_eth=dst_wei_per_eth,
                )

        winner = pool_set.best
        if winner is None:
            result.warnings.append(
                "no candidate could be quoted on-chain; falling back to the "
                "modelled route (treat the output as unverified)"
            )
        else:
            result.route = winner.route
            result.verified_out = winner.verified_out
            result.winner = winner
            if not winner.certificate:
                report.certificate = False
                report.reason = report.reason or winner.reason or "RESTRICTED"

            # --- §8 refit: re-anchor the winner at its realised sizes --------
            if refit_rounds > 0:
                with clock("refit"):
                    _refit_winner(
                        result, pool_set, winner, g, arcs, nu, nodes, client,
                        src_node=src_node, dst_node=dst_node, Psi=Psi_scaled,
                        src_token=src_token, dst_token=dst_token,
                        amount_in=amount_in, rounds=refit_rounds,
                        gas_price_wei=gas_price_wei,
                    )

    result.price_out_per_in = float(nu[src_node] / nu[dst_node]) if nu[dst_node] else 0.0
    return result


def direct_candidates(
    pools: list[PoolSpec],
    nodes: NodeMap,
    nu: np.ndarray,
    src_token: str,
    dst_token: str,
    amount_in: int,
) -> tuple[list[Candidate], list[PoolArc]]:
    """One-leg candidates through every pool holding both tokens.

    These are the safety floor, and they exist precisely because they do *not*
    depend on the probe grid, the calibration, the reference-price fit or the
    solver.  If a probe fails and an arc is dropped, the model can lose sight
    of an obvious direct swap; measured once on a degraded run, the router
    returned 13,700 USDC for 100,000 DAI and honestly verified it, because
    every candidate it generated really was that bad.

    A router must never be beaten by a swap anyone could find by inspection.
    """
    src_node, dst_node = nodes.node(src_token), nodes.node(dst_token)
    out: list[Candidate] = []
    made: list[PoolArc] = []
    for pool in pools:
        kind = pool.swap_kind
        if kind is None:
            continue
        index = {c.address.lower(): k for k, c in enumerate(pool.coins)}
        for token_in, i in index.items():
            if nodes.node(token_in) != src_node:
                continue
            for token_out, j in index.items():
                if i == j or nodes.node(token_out) != dst_node:
                    continue
                arc = PoolArc(
                    id=f"direct:{pool.address.lower()}:{i}>{j}",
                    pool=pool.address, kind=kind, i=i, j=j, n_coins=pool.n_coins,
                    token_in=token_in, token_out=token_out,
                    tau=src_node, sigma=dst_node,
                    a=float(nu[src_node] / nu[dst_node]) if nu[dst_node] else 1.0,
                    B=0.0, reserve_in=pool.balances[i] if i < len(pool.balances) else 0,
                    decimals_in=pool.coins[i].decimals,
                    decimals_out=pool.coins[j].decimals,
                    tvl_usd=pool.tvl_usd, note=pool.name,
                )
                made.append(arc)
                out.append(
                    Candidate(
                        label=f"direct {pool.name[:22]}",
                        psi=np.array([1.0]), certificate=False,
                        kind="direct", reason="DIRECT", n_arcs=1,
                    )
                )
    return out, made


def two_step_candidates(
    pools: list[PoolSpec],
    nodes: NodeMap,
    nu: np.ndarray,
    client: QuoterClient,
    src_token: str,
    dst_token: str,
    amount_in: int,
    *,
    limit: int = 6,
) -> tuple[list[Candidate], list[list[PoolArc]]]:
    """Model-free `src -> M -> dst` chains, one pool per hop.

    The two-hop half of the safety floor.  `direct_candidates` covers what a
    person would find in one glance; this covers what they would find in two,
    and it depends on no part of the model either.  It exists because the
    quadratic element law degrades badly once a trade is large relative to the
    pools -- measured on wstETH->WETH at 50 units, where the whole Curve
    universe holds ~900 wstETH, the model's best candidate paid 50.30 against
    54.70 from an obvious two-pool chain.

    Two batched rounds pick *which* chains to offer; the quoter then decides
    the actual amounts, so the ranking here only has to be roughly right.
    """
    src_node, dst_node = nodes.node(src_token), nodes.node(dst_token)

    def slots(pool: PoolSpec, node: int) -> list[int]:
        return [
            k for k, c in enumerate(pool.coins)
            if nodes.has(c.address) and nodes.node(c.address) == node
        ]

    # --- round A: src -> M ------------------------------------------------
    probes: list[Probe] = []
    first: list[tuple[PoolSpec, int, int, int]] = []
    for pool in pools:
        if pool.swap_kind is None:
            continue
        for i in slots(pool, src_node):
            for j, coin in enumerate(pool.coins):
                if j == i or not nodes.has(coin.address):
                    continue
                middle = nodes.node(coin.address)
                if middle in (src_node, dst_node):
                    continue
                probes.append(
                    Probe(pool.address, pool.swap_kind, i, j, pool.n_coins, amount_in)
                )
                first.append((pool, i, j, middle))
    if not probes:
        return [], []

    best_first: dict[int, tuple[int, PoolSpec, int, int]] = {}
    for quote, (pool, i, j, middle) in zip(client.probe(probes), first, strict=True):
        if not quote.ok or quote.value <= 0:
            continue
        canonical = nodes.to_canonical_wei(pool.coins[j].address, quote.value)
        if canonical > best_first.get(middle, (0,))[0]:
            best_first[middle] = (canonical, pool, i, j)
    if not best_first:
        return [], []

    ranked = sorted(best_first.items(), key=lambda kv: -kv[1][0])[: 3 * limit]

    # --- round B: M -> dst ------------------------------------------------
    probes = []
    second: list[tuple[int, PoolSpec, int, int]] = []
    for middle, (canonical, _p1, _i1, _j1) in ranked:
        for pool in pools:
            if pool.swap_kind is None:
                continue
            for i in slots(pool, middle):
                start = nodes.from_canonical_wei(pool.coins[i].address, canonical)
                if start <= 0:
                    continue
                for j in slots(pool, dst_node):
                    if i == j:
                        continue
                    probes.append(
                        Probe(pool.address, pool.swap_kind, i, j, pool.n_coins, start)
                    )
                    second.append((middle, pool, i, j))
    if not probes:
        return [], []

    best_chain: dict[int, tuple[int, PoolSpec, int, int]] = {}
    for quote, (middle, pool, i, j) in zip(client.probe(probes), second, strict=True):
        if not quote.ok or quote.value <= 0:
            continue
        value = nodes.to_canonical_wei(pool.coins[j].address, quote.value)
        if value > best_chain.get(middle, (0,))[0]:
            best_chain[middle] = (value, pool, i, j)

    out: list[Candidate] = []
    made: list[list[PoolArc]] = []
    for middle, (_value, pool2, i2, j2) in sorted(
        best_chain.items(), key=lambda kv: -kv[1][0]
    )[:limit]:
        _canonical, pool1, i1, j1 = best_first[middle]
        arcs = [
            _synthetic_arc(pool1, i1, j1, nodes, nu, src_node, middle),
            _synthetic_arc(pool2, i2, j2, nodes, nu, middle, dst_node),
        ]
        out.append(
            Candidate(
                label=f"2-hop via {nodes.symbol(pool1.coins[j1].address)}",
                psi=np.array([1.0, 1.0]), certificate=False,
                kind="direct", reason="TWO_STEP", n_arcs=2,
            )
        )
        made.append(arcs)
    return out, made


def _synthetic_arc(
    pool: PoolSpec, i: int, j: int, nodes: NodeMap, nu: np.ndarray, tau: int, sigma: int
) -> PoolArc:
    """An arc built for realisation only -- never calibrated, never solved."""
    return PoolArc(
        id=f"naive:{pool.address.lower()}:{i}>{j}",
        pool=pool.address, kind=pool.swap_kind, i=i, j=j, n_coins=pool.n_coins,
        token_in=pool.coins[i].address, token_out=pool.coins[j].address,
        tau=tau, sigma=sigma,
        a=float(nu[tau] / nu[sigma]) if nu[sigma] else 1.0, B=0.0,
        reserve_in=pool.balances[i] if i < len(pool.balances) else 0,
        decimals_in=pool.coins[i].decimals, decimals_out=pool.coins[j].decimals,
        tvl_usd=pool.tvl_usd, note=pool.name,
    )


def _refit_winner(
    result: RouteResult,
    pool_set: CandidateSet,
    winner: Candidate,
    g: ArcArrays,
    arcs: list[PoolArc],
    nu: np.ndarray,
    nodes: NodeMap,
    client: QuoterClient,
    *,
    src_node: int,
    dst_node: int,
    Psi: float,
    src_token: str,
    dst_token: str,
    amount_in: int,
    rounds: int,
    gas_price_wei: int,
) -> None:
    """§8 -- refit the winner, re-solve, and let the chain adjudicate again.

    The refitted route is *offered*, never imposed: it goes back through the
    same quoter as an extra candidate, so it only wins if it actually quotes
    higher.  A refit can only improve the model, but the model is not what is
    being reported.
    """
    from .verify import realize_candidates
    from .verify import verify as verify_candidates

    forbidden = np.ones(g.m, bool)
    forbidden[np.flatnonzero(winner.psi > 0)] = False

    def solve_fn(graph_arrays, A0):
        return active_set_solve(
            graph_arrays, src_node, dst_node, Psi, A0=A0, forbidden=forbidden
        )

    report = refit(
        g, arcs, winner.psi / g.g_scale, nu, client, solve_fn, Psi,
        rate_in=lambda arc: nodes.rate(arc.token_in),
        rate_out=lambda arc: nodes.rate(arc.token_out),
        rounds=rounds,
    )
    result.refit_report = report
    result.counters["refit_rounds"] = len(report.rounds)
    if report.rounds:
        result.counters["refit_quoted"] = report.rounds[-1].quoted
        if report.rounds[-1].reflagged:
            result.warnings.append(
                f"{report.rounds[-1].reflagged} arc(s) showed increasing returns at "
                "their realised size and were clamped (§11.2)"
            )
    if report.psi is None or not report.changed:
        return
    if not report.converged:
        result.warnings.append(
            f"refit did not converge in {len(report.rounds)} rounds "
            f"(max |d psi| / Psi = {report.rounds[-1].max_delta_psi:.2e})"
        )

    refitted = Candidate(
        label="refit at realised size",
        psi=report.psi * g.g_scale,
        certificate=False,
        kind="refit",
        reason="REFIT",
        n_arcs=int(np.count_nonzero(report.psi > 0)),
    )
    trial = CandidateSet([refitted])
    realize_candidates(
        trial, arcs, nu, nodes,
        src_token=src_token, dst_token=dst_token, amount_in=amount_in,
    )
    verify_candidates(trial, client, amount_in=amount_in)
    if not refitted.ok:
        return
    refitted.rank = None  # ranked for real against the whole set below

    pool_set.candidates.append(refitted)
    verify_candidates(pool_set, client, amount_in=amount_in, gas_price_wei=gas_price_wei)
    best = pool_set.best
    if best is not None:
        result.route = best.route
        result.verified_out = best.verified_out
        result.winner = best


def _dst_per_eth(nodes: NodeMap, nu: np.ndarray, dst_token: str) -> float:
    """Output-token wei per 1 ETH, for costing gas.  0 when ETH is unpriced."""
    for candidate in ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",):
        if not nodes.has(candidate):
            continue
        weth_value = float(nu[nodes.node(candidate)])
        dst_value = float(nu[nodes.node(dst_token)])
        if weth_value <= 0 or dst_value <= 0:
            return 0.0
        return weth_value / dst_value * 10 ** nodes.decimals(dst_token)
    return 0.0


def _assemble(
    arcs: list[PoolArc],
    nu: np.ndarray,
    Psi: float,
    nodes: NodeMap,
    src_node: int,
    dst_node: int,
    result: RouteResult,
):
    """Build the solver arrays from the current calibration.

    Called twice -- once on the coarse pass, once after refinement -- because
    `build` drops dust and can merge duplicates, so the arc list has to be
    re-aligned to whatever survived.
    """
    bottomless = _clamp_unphysical_depth(arcs, nu, nodes)
    if bottomless:
        result.counters["arcs_clamped_as_bottomless"] = bottomless

    # `cap` is a bound on *value* flow, so convert from canonical token units.
    caps = np.array(
        [a.cap if not math.isfinite(a.cap) else float(nu[a.tau] * a.cap) for a in arcs]
    )
    g = build(
        np.array([a.tau for a in arcs], dtype=np.int64),
        np.array([a.sigma for a in arcs], dtype=np.int64),
        np.array([a.a for a in arcs]),
        np.array([a.B for a in arcs]),
        nu, Psi,
        cap=caps,
        flagged=np.array([a.convex_flag for a in arcs]),
        clamped=np.array([a.clamped for a in arcs]),
        n_nodes=nodes.n_nodes,
        merge_duplicates=False,
        require=(src_node, dst_node),
    )
    arcs = [arcs[group[0]] for group in g.sources]
    result.counters["arcs_dropped_dust"] = sum(
        1 for reason in g.dropped.values() if reason == "DUST"
    )
    for k, arc in enumerate(arcs):
        arc.G, arc.eps = float(g.G[k]), float(g.eps[k])
    if g.ill_conditioned:
        result.counters["condition"] = int(g.ill_conditioned)
        result.warnings.append(
            f"conductance spread is {g.ill_conditioned:.2e}, past §12.4's "
            f"{MAX_CONDITION:.0e} bound: the dust floor was backed off to keep "
            "the pair connected at all. The route is still checked on-chain, "
            "but treat the modelled split as approximate"
        )
    _warn_pair_drops(arcs, result)
    return arcs, g


def _recalibrate(arcs: list[PoolArc], ladders, nodes: NodeMap) -> int:
    """Re-fit the arcs whose ladders just gained points."""
    by_id = {lad.arc.id: lad for lad in ladders}
    changed = 0
    for arc in arcs:
        ladder = by_id.get(arc.id)
        if ladder is None or len(ladder.deltas) < 3:
            continue
        deltas, quotes = ladder.as_float()
        try:
            fit = calibrate(deltas, quotes)
        except CalibrationError:
            continue
        a, B = rescale(fit.a, fit.B, nodes.rate(arc.token_in), nodes.rate(arc.token_out))
        arc.a, arc.B = a, B
        arc.cap = fit.cap * nodes.rate(arc.token_in) if math.isfinite(fit.cap) else math.inf
        arc.clamped, arc.convex_flag = fit.clamped, fit.convex_flag
        arc.flag_reason, arc.drift, arc.eta = fit.flag_reason, fit.drift, fit.eta
        arc.calib_delta = fit.calib_delta
        changed += 1
    return changed


def _kcl_tolerance(Psi: float, g_scale: float) -> float:
    """How much KCL slop is floating-point noise rather than a bug.

    The solve runs in units scaled by `g_scale`, so its roundoff is absolute
    *there* and gets multiplied by `g_scale` on the way out.  A tolerance
    expressed purely as a fraction of `Psi` therefore tightens without limit as
    the trade shrinks: measured on USDC->USDT at this block, `g_scale = 1.5e6`
    turns a clean 1e-11 solve into a 4.7e-5 absolute residual, which is 4.7e-5
    of a $1 trade and 3.8e-11 of a $100k one.  The router was identical in both
    cases; only the yardstick moved, and small trades were rejected outright.

    So allow both terms.  This stays far tighter than the failure the check
    exists to catch -- flow conjured on arcs outside the active component is
    `O(Psi)`, several orders above either bound at any size.
    """
    return KCL_RELATIVE + KCL_ABSOLUTE * g_scale / max(Psi, 1e-30)


def _kcl_residual(
    g: ArcArrays, psi: np.ndarray, src: int, dst: int, Psi: float
) -> float:
    """`||B^T psi - s_hat||_inf / Psi` -- Kirchhoff's current law (§12.4)."""
    net = np.zeros(g.n_nodes)
    np.add.at(net, g.tau, psi)
    np.subtract.at(net, g.sig, psi)
    want = np.zeros(g.n_nodes)
    want[src] += Psi
    want[dst] -= Psi
    return float(np.max(np.abs(net - want)) / Psi) if Psi > 0 else 0.0


def _restrict_to_component(
    arcs: list[PoolArc], dst_node: int, n_nodes: int, result: RouteResult
) -> list[PoolArc]:
    from .graph import component_of

    tau = np.array([a.tau for a in arcs], dtype=np.int64)
    sig = np.array([a.sigma for a in arcs], dtype=np.int64)
    reachable = component_of(dst_node, tau, sig, n_nodes)
    keep = [a for a in arcs if reachable[a.tau] and reachable[a.sigma]]
    result.counters["arcs_unreachable"] = len(arcs) - len(keep)
    result.counters["nodes_reachable"] = int(reachable.sum())
    return keep


# A pool cannot be meaningfully deeper than this multiple of its own input
# reserve.  For constant product `G = nu*y0/2` exactly; a stableswap at the peg
# is far deeper, but not by twelve orders of magnitude.
DEPTH_LIMIT = 1e4


def _clamp_unphysical_depth(arcs: list[PoolArc], nu: np.ndarray, nodes: NodeMap) -> int:
    """Treat immeasurably small curvature as the zero-curvature limit.

    A `B` that implies a conductance far beyond the pool's own reserves is not
    a very deep pool, it is a curvature below the integer noise floor of the
    quotes -- deep pools probed at small sizes look linear.  Leaving it as a
    huge finite `G` wrecks the Laplacian's condition number for every other
    arc; clamping to `B = 0` with a cap is the admissible limit (§2.3) and
    keeps the arc honest: bottomless only up to the size actually probed.
    """
    clamped = 0
    for arc in arcs:
        if arc.B <= 0 or arc.reserve_in <= 0:
            continue
        reserve_canonical = (
            arc.reserve_in / 10**arc.decimals_in * nodes.rate(arc.token_in)
        )
        limit = float(nu[arc.tau]) * reserve_canonical * DEPTH_LIMIT
        conductance = float(nu[arc.tau]) * arc.a / arc.B
        if limit > 0 and conductance > limit:
            arc.B = 0.0
            arc.clamped = True
            arc.convex_flag = True
            probed = arc.calib_delta if arc.calib_delta > 0 else reserve_canonical
            arc.cap = min(arc.cap, probed)
            clamped += 1
    return clamped


def _align(planned, refs, meta):
    """`plan_grid` may skip arcs, so re-align the metadata to what it kept."""
    by_id = {ref.id: m for ref, m in zip(refs, meta, strict=True)}
    return [by_id[ref.id] for ref in planned]


def _warn_pair_drops(arcs: list[PoolArc], result: RouteResult) -> None:
    """§2.6: `eps_f + eps_r <= 0` means `nu` is inconsistent with that pool.

    It manufactures a two-arc negative cycle that does not exist, and the
    solver will happily allocate flow around it.
    """
    forward: dict[tuple[str, int, int], PoolArc] = {}
    for arc in arcs:
        forward[(arc.pool.lower(), arc.i, arc.j)] = arc
    pairs = []
    for (pool, i, j), arc in forward.items():
        reverse = forward.get((pool, j, i))
        if reverse is not None and i < j:
            pairs.append((arc, reverse))
    if not pairs:
        return
    violations = check_pair_drops(
        np.array([f.eps for f, _ in pairs]), np.array([r.eps for _, r in pairs])
    )
    result.counters["eps_pair_violations"] = int(violations.size)
    if violations.size:
        result.warnings.append(
            f"{violations.size} pool(s) have eps_f + eps_r <= 0: the reference price "
            "is inconsistent with them (spurious negative 2-cycle, §2.6)"
        )


class _Clock:
    def __init__(self, sink: dict[str, float]) -> None:
        self.sink = sink

    def __call__(self, name: str):
        return _Span(self.sink, name)


class _Span:
    def __init__(self, sink: dict[str, float], name: str) -> None:
        self.sink, self.name = sink, name

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.sink[self.name] = (time.monotonic() - self.started) * 1000
        return False
