"""Turn a solved value flow into an ordered list of executable legs (§5.6).

Two things make this more than bookkeeping.

**Do not undo the netting.**  The solution is an *edge* flow, so a pool that two
decomposed paths both traverse already carries the net amount.  Re-expanding
into paths and quoting each separately double-counts that pool's impact, which
is a common source of failed quotes.  So legs are emitted per arc, in
topological order, and paths are reconstructed afterwards for display only.

**Merged nodes need conversion legs.**  Routing treats ETH and WETH as one
node, but a pool holds one or the other.  Each node uses its canonical token as
a hub: arriving non-canonical tokens convert in, departing ones convert out.
On a node whose arcs all use the canonical token -- the common case -- no
conversion is emitted at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .nodes import NodeMap
from .types import ArcKind, Leg, PoolArc

BPS = 10_000


class RealizationError(RuntimeError):
    pass


@dataclass(slots=True)
class RealizedLeg:
    leg: Leg
    kind: ArcKind
    target: str
    token_in: str
    token_out: str
    amount_in: int  # raw wei, modelled
    amount_out: int  # raw wei, modelled
    share_of_node: float = 1.0
    arc_id: str | None = None
    pool_name: str = ""
    eps: float = 0.0
    impact_frac: float = 0.0
    theta: float = 0.0
    psi: float = 0.0

    @property
    def is_conversion(self) -> bool:
        return self.kind in (
            ArcKind.WRAP_NATIVE,
            ArcKind.UNWRAP_NATIVE,
            ArcKind.ERC4626_DEPOSIT,
            ArcKind.ERC4626_REDEEM,
        )


@dataclass(slots=True)
class RealizedRoute:
    legs: list[RealizedLeg] = field(default_factory=list)
    slots: dict[str, int] = field(default_factory=dict)
    dst_slot: int = 0
    src_token: str = ""
    dst_token: str = ""
    amount_in: int = 0
    modelled_out: int = 0
    node_of_slot: dict[int, int] = field(default_factory=dict)
    potentials: dict[int, float] = field(default_factory=dict)
    paths: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def wire_legs(self) -> list[Leg]:
        return [rl.leg for rl in self.legs]

    @property
    def pools_used(self) -> list[str]:
        return sorted({rl.target.lower() for rl in self.legs if not rl.is_conversion})


def topological_nodes(tau: np.ndarray, sig: np.ndarray, n_nodes: int) -> list[int]:
    """Kahn's algorithm over the active arcs.  Raises on a cycle."""
    indegree = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(indegree, sig, 1)
    queue = [int(v) for v in np.flatnonzero(indegree == 0)]
    order: list[int] = []
    remaining = indegree.copy()
    while queue:
        node = queue.pop(0)
        order.append(node)
        for arc in np.flatnonzero(tau == node):
            head = int(sig[arc])
            remaining[head] -= 1
            if remaining[head] == 0:
                queue.append(head)
    if len(order) != n_nodes:
        raise RealizationError(
            "the active arcs contain a cycle; flow cannot be ordered for execution"
        )
    return order


def cancel_cycles(
    tau: np.ndarray, sig: np.ndarray, psi: np.ndarray, tol: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Remove circulation from a flow, leaving the same net source-to-sink delivery.

    A cycle appears when the model believes some loop has negative total `eps`
    -- free arbitrage.  §5.3 treats that as real and worth absorbing, but a
    router cannot *execute* a circulation as part of a one-way trade: it would
    need capital it does not have, and the quoter walks a DAG.  Cancelling the
    circulation leaves the delivered amount unchanged and can only lower the
    modelled output (the loop was claimed to be profitable), which is the
    conservative direction.

    Returns the acyclic flow and the number of cycles removed.
    """
    flow = np.array(psi, dtype=float)
    removed = 0
    for _ in range(len(flow) + 1):
        live = np.flatnonzero(flow > tol)
        if live.size == 0:
            break
        cycle = _find_cycle(tau[live], sig[live])
        if cycle is None:
            break
        arcs = live[cycle]
        flow[arcs] -= flow[arcs].min()
        flow[flow <= tol] = 0.0
        removed += 1
    return flow, removed


def _find_cycle(tau: np.ndarray, sig: np.ndarray) -> list[int] | None:
    """One directed cycle as local arc indices, or None.

    Peel arcs that cannot be on a cycle -- those whose head has no way out, or
    whose tail has no way in -- until the set is stable.  Whatever survives has
    every node with both an in- and an out-arc, so following out-edges from any
    of them must revisit a node, and the revisit closes a cycle.
    """
    alive = np.ones(len(tau), dtype=bool)
    while alive.any():
        heads = np.unique(tau[alive])  # nodes with a way out
        tails = np.unique(sig[alive])  # nodes with a way in
        doomed = alive & (~np.isin(sig, heads) | ~np.isin(tau, tails))
        if not doomed.any():
            break
        alive &= ~doomed
    if not alive.any():
        return None

    outgoing: dict[int, int] = {}
    for k in np.flatnonzero(alive):
        outgoing.setdefault(int(tau[k]), int(k))

    node = int(tau[np.flatnonzero(alive)[0]])
    position: dict[int, int] = {}
    path: list[int] = []
    while node not in position:
        position[node] = len(path)
        arc = outgoing[node]
        path.append(arc)
        node = int(sig[arc])
    return path[position[node] :]


def realize(
    arcs: list[PoolArc],
    psi: np.ndarray,
    nu: np.ndarray,
    nodes: NodeMap,
    *,
    src_token: str,
    dst_token: str,
    amount_in: int,
    potentials: np.ndarray | None = None,
) -> RealizedRoute:
    """Build the executable leg list from a solved flow.

    `arcs` and `psi` are parallel and already restricted to the arcs carrying
    flow.  `psi` is value; `delta = psi / nu[tau]` converts back to canonical
    token units, and the node map converts those to the pool's actual token.
    """
    route = RealizedRoute(
        src_token=src_token.lower(),
        dst_token=dst_token.lower(),
        amount_in=amount_in,
    )
    if not arcs:
        raise RealizationError("no arcs carry flow")

    tau = np.array([arc.tau for arc in arcs], dtype=np.int64)
    sig = np.array([arc.sigma for arc in arcs], dtype=np.int64)
    touched = sorted({*tau.tolist(), *sig.tolist()})
    index = {node: k for k, node in enumerate(touched)}
    order = topological_nodes(
        np.array([index[int(t)] for t in tau]),
        np.array([index[int(s)] for s in sig]),
        len(touched),
    )
    node_order = [touched[k] for k in order]

    # --- slots ----------------------------------------------------------
    def slot(token: str) -> int:
        key = token.lower()
        if key not in route.slots:
            route.slots[key] = len(route.slots)
            route.node_of_slot[route.slots[key]] = nodes.node(key)
        return route.slots[key]

    slot(src_token)  # slot 0 is always the input
    if potentials is not None:
        route.potentials = {int(n): float(potentials[n]) for n in touched}

    # --- amounts --------------------------------------------------------
    deltas: list[int] = []
    outs: list[int] = []
    for arc, flow in zip(arcs, psi, strict=True):
        delta_canonical = float(flow) / float(nu[arc.tau])
        delta_token = delta_canonical / nodes.rate(arc.token_in)
        deltas.append(int(delta_token * 10 ** nodes.decimals(arc.token_in)))
        # (M1) is only valid on [0, a/B], where f_hat' hits zero; beyond that
        # the model turns *decreasing*, so it is a hard box constraint rather
        # than something to watch.  Clipping keeps a solver excursion from
        # showing up as a negative -- and therefore zero -- output.
        domain = arc.a / arc.B if arc.B > 0 else math.inf
        d = min(delta_canonical, domain)
        model = arc.a * d - 0.5 * arc.B * d * d
        out_token = max(model, 0.0) / nodes.rate(arc.token_out)
        outs.append(int(out_token * 10 ** nodes.decimals(arc.token_out)))

    by_source: dict[int, list[int]] = {}
    for k, arc in enumerate(arcs):
        by_source.setdefault(arc.tau, []).append(k)

    # --- emit -----------------------------------------------------------
    destination = nodes.node(dst_token)
    for node in node_order:
        canonical = nodes.canonical_of[node]
        outgoing = by_source.get(node, [])

        # (1) fold every non-canonical token held at this node into the hub
        incoming_tokens = {
            arcs[k].token_out.lower() for k in range(len(arcs)) if arcs[k].sigma == node
        }
        if node == nodes.node(src_token):
            incoming_tokens.add(src_token.lower())
        for token in sorted(incoming_tokens):
            if token == canonical:
                continue
            # The destination has no outgoing arcs, so skipping it here left a
            # route that ends in native ETH depositing into the ETH slot while
            # the caller asked for WETH -- the quoter then reads the WETH slot,
            # finds nothing, and the whole candidate reads as "reverted".  That
            # silently removed both big ETH/stETH pools from stETH->WETH and
            # left a shallow factory pool paying half.
            if not outgoing and node != destination:
                continue
            conversion = nodes.conversion.get(token)
            if conversion is None:
                continue
            route.legs.append(
                _conversion_leg(
                    nodes, conversion, forward=True,
                    src=slot(token), dst=slot(canonical), bps=0,
                )
            )

        if not outgoing:
            continue

        # (2) everything leaving the hub, as one contiguous group
        total = sum(deltas[k] for k in outgoing)
        if total <= 0:
            continue
        spokes: list[tuple[int, int]] = []  # (arc index, spoke slot)
        group: list[tuple[int, int]] = []  # (arc index, destination slot)
        for k in outgoing:
            token_in = arcs[k].token_in.lower()
            if token_in == canonical:
                group.append((k, -1))
            else:
                spoke = slot(token_in)
                group.append((k, spoke))
                spokes.append((k, spoke))

        for position, (k, spoke) in enumerate(group):
            last = position == len(group) - 1
            bps = 0 if last else max(1, min(BPS - 1, round(BPS * deltas[k] / total)))
            if spoke >= 0:
                conversion = nodes.conversion[arcs[k].token_in.lower()]
                route.legs.append(
                    _conversion_leg(
                        nodes, conversion, forward=False,
                        src=slot(canonical), dst=spoke, bps=bps,
                    )
                )
            else:
                route.legs.append(
                    _arc_leg(arcs[k], nodes, slot(canonical), slot(arcs[k].token_out),
                             bps, deltas[k], outs[k], float(psi[k]), deltas[k] / total)
                )

        # (3) arcs that had to draw from a spoke, each its own group
        for k, spoke in spokes:
            route.legs.append(
                _arc_leg(arcs[k], nodes, spoke, slot(arcs[k].token_out),
                         0, deltas[k], outs[k], float(psi[k]), deltas[k] / total)
            )

    # --- destination ----------------------------------------------------
    dst_node = nodes.node(dst_token)
    dst_canonical = nodes.canonical_of[dst_node]
    if dst_token.lower() != dst_canonical:
        conversion = nodes.conversion.get(dst_token.lower())
        if conversion is not None:
            route.legs.append(
                _conversion_leg(
                    nodes, conversion, forward=False,
                    src=slot(dst_canonical), dst=slot(dst_token), bps=0,
                )
            )
    route.dst_slot = slot(dst_token)
    route.modelled_out = _forward_simulate(route, nodes)
    route.paths = _decompose(route)
    return route


def _arc_leg(
    arc: PoolArc,
    nodes: NodeMap,
    src: int,
    dst: int,
    bps: int,
    amount_in: int,
    amount_out: int,
    psi: float,
    share: float,
) -> RealizedLeg:
    impact = psi / (2 * arc.G) if arc.G > 0 else 0.0
    theta = 0.0
    if arc.reserve_in > 0:
        theta = amount_in / arc.reserve_in
    return RealizedLeg(
        leg=Leg(
            target=arc.pool,
            kind=arc.kind,
            i=arc.i,
            j=arc.j,
            n=arc.n_coins,
            src_slot=src,
            dst_slot=dst,
            bps=bps,
        ),
        kind=arc.kind,
        target=arc.pool,
        token_in=arc.token_in,
        token_out=arc.token_out,
        amount_in=amount_in,
        amount_out=amount_out,
        share_of_node=share,
        arc_id=arc.id,
        pool_name=arc.note,
        eps=arc.eps,
        impact_frac=impact,
        theta=theta,
        psi=psi,
    )


def _conversion_leg(
    nodes: NodeMap, conversion, *, forward: bool, src: int, dst: int, bps: int
) -> RealizedLeg:
    """`forward` means token -> canonical."""
    kind = conversion.forward_kind if forward else conversion.reverse_kind
    token_in = conversion.token if forward else conversion.canonical
    token_out = conversion.canonical if forward else conversion.token
    return RealizedLeg(
        leg=Leg(
            target=conversion.target or token_out,
            kind=kind,
            i=0,
            j=1,
            n=2,
            src_slot=src,
            dst_slot=dst,
            bps=bps,
        ),
        kind=kind,
        target=conversion.target or token_out,
        token_in=token_in,
        token_out=token_out,
        amount_in=0,
        amount_out=0,
        pool_name=f"{nodes.symbol(token_in)} -> {nodes.symbol(token_out)}",
    )


def _forward_simulate(route: RealizedRoute, nodes: NodeMap) -> int:
    """Replay the legs the way the quoter will, to fill in modelled amounts.

    Uses the same group-snapshot semantics as the contract, so the numbers the
    diagram shows are the numbers the quoter will be asked to confirm.
    """
    balances: dict[int, int] = {0: route.amount_in}
    current = None
    base = 0
    for realized in route.legs:
        src = realized.leg.src_slot
        if src != current:
            current = src
            base = balances.get(src, 0)
        available = balances.get(src, 0)
        take = available if realized.leg.bps == 0 else min(base * realized.leg.bps // BPS, available)
        if take <= 0:
            continue
        if realized.is_conversion:
            conversion = nodes.conversion.get(realized.token_in.lower()) or nodes.conversion.get(
                realized.token_out.lower()
            )
            if conversion is None or realized.kind in (ArcKind.WRAP_NATIVE, ArcKind.UNWRAP_NATIVE):
                produced = take
            elif realized.kind is ArcKind.ERC4626_REDEEM:
                produced = conversion.to_canonical(take)
            else:
                produced = conversion.from_canonical(take)
            realized.amount_in = take
            realized.amount_out = produced
        else:
            # keep the modelled ratio, rescaled to whatever actually arrives
            if realized.amount_in > 0:
                produced = realized.amount_out * take // realized.amount_in
            else:
                produced = 0
            realized.amount_in = take
            realized.amount_out = produced
        balances[src] = available - take
        balances[realized.leg.dst_slot] = balances.get(realized.leg.dst_slot, 0) + produced
    return balances.get(route.dst_slot, 0)


# Paths are display-only, and there can be exponentially many of them: the
# walk below is a DAG enumeration, so a route 16 legs deep that branches three
# ways at each node has tens of millions.  Measured on crvUSD->sDOLA at 5M once
# the savings vaults were merged, this never returned -- §11.6 warns that path
# enumeration is exponential, and it is just as true in the presentation layer
# as in the solver.
MAX_DISPLAY_PATHS = 64


def _decompose(route: RealizedRoute) -> list[list[str]]:
    """Flow decomposition, for display only.

    Deliberately derived *after* the legs: the legs are the truth, and paths
    are a reading of them.  Doing it the other way round is what double-counts
    shared pools.

    Bounded at `MAX_DISPLAY_PATHS`: nobody reads the 65th path, and the legs --
    which are the executable artefact -- are unaffected by the cut.
    """
    outgoing: dict[int, list[RealizedLeg]] = {}
    for realized in route.legs:
        outgoing.setdefault(realized.leg.src_slot, []).append(realized)

    paths: list[list[str]] = []

    def walk(slot: int, trail: list[str], depth: int = 0) -> None:
        if depth > 16 or len(paths) >= MAX_DISPLAY_PATHS:
            return
        legs = outgoing.get(slot)
        if not legs or slot == route.dst_slot:
            if trail:
                paths.append(trail)
            return
        for realized in legs:
            if len(paths) >= MAX_DISPLAY_PATHS:
                return
            label = realized.arc_id or f"{realized.kind.name}:{realized.target[:10]}"
            walk(realized.leg.dst_slot, [*trail, label], depth + 1)

    walk(0, [])
    if len(paths) >= MAX_DISPLAY_PATHS:
        route.warnings.append(
            f"path list truncated at {MAX_DISPLAY_PATHS}; the legs are complete "
            "and are what executes"
        )
    return paths


def check_one_arc_per_pool(route: RealizedRoute) -> list[str]:
    """Decision 3: a pool may appear at most once in a route.

    Deposits and withdrawals mutate pool state and a view-only chained quoter
    cannot see its own earlier leg, so two arcs of the same pool would be
    quoted against stale state.  Returns the offending pool addresses.
    """
    seen: dict[str, int] = {}
    for realized in route.legs:
        if realized.is_conversion:
            continue
        key = realized.target.lower()
        seen[key] = seen.get(key, 0) + 1
    return sorted(pool for pool, count in seen.items() if count > 1)


def max_theta(route: RealizedRoute) -> float:
    return max((rl.theta for rl in route.legs if not rl.is_conversion), default=0.0)


def total_loss_bp(route: RealizedRoute, price_out_per_in: float) -> float:
    """Realised loss against a frictionless trade at the reference price."""
    if route.amount_in <= 0 or price_out_per_in <= 0:
        return math.nan
    ideal = route.amount_in * price_out_per_in
    if ideal <= 0:
        return math.nan
    return (1.0 - route.modelled_out / ideal) * 10_000
