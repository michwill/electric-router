"""Seed subgraph for column generation (spec §5.3).

The union of:

* the `k` shortest `src -> dst` paths under edge length `eps_p`,
* the top-conductance arcs incident to any node those paths touch (breadth,
  not depth -- the solver finds depth by itself),
* any negative-`eps` cycle reachable from the path set, which is free
  arbitrage the router should absorb.

`eps_p` can be negative -- a favourably dislocated pool is an EMF -- so this
uses SPFA (queue-based Bellman-Ford) rather than Dijkstra, and detects negative
cycles instead of looping on them.

**Seed quality only affects how many column-generation rounds run, never the
answer.**  §5.5's certificate is what guarantees correctness, so this file is
allowed to be approximate and is not allowed to be slow.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .graph import ArcArrays

INF = float("inf")


@dataclass(slots=True)
class Adjacency:
    """Outgoing arc indices per node."""

    starts: np.ndarray
    arcs: np.ndarray

    def out(self, node: int) -> np.ndarray:
        return self.arcs[self.starts[node] : self.starts[node + 1]]


def build_adjacency(tau: np.ndarray, sig: np.ndarray, n: int) -> Adjacency:
    del sig
    order = np.argsort(tau, kind="stable")
    counts = np.bincount(tau, minlength=n)
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return Adjacency(starts, order.astype(np.int64))


@dataclass(slots=True)
class ShortestPath:
    arcs: list[int] = field(default_factory=list)
    length: float = INF
    found: bool = False
    negative_cycle: list[int] = field(default_factory=list)


def spfa(
    g: ArcArrays,
    src: int,
    dst: int,
    adj: Adjacency,
    *,
    banned_arcs: set[int] | None = None,
    banned_nodes: set[int] | None = None,
    weights: np.ndarray | None = None,
) -> ShortestPath:
    """Shortest `src -> dst` path by arc length `eps`, tolerating negative arcs.

    Returns the arc indices of the path.  If a negative cycle is reachable, the
    arcs of that cycle are returned separately rather than the search diverging.
    """
    n = g.n_nodes
    banned_arcs = banned_arcs or set()
    banned_nodes = banned_nodes or set()
    cost = g.eps if weights is None else weights

    dist = np.full(n, INF)
    parent = np.full(n, -1, dtype=np.int64)  # arc index used to reach the node
    relaxations = np.zeros(n, dtype=np.int64)
    in_queue = np.zeros(n, dtype=bool)

    dist[src] = 0.0
    queue: list[int] = [src]
    in_queue[src] = True

    while queue:
        node = queue.pop(0)
        in_queue[node] = False
        for arc in adj.out(node):
            arc = int(arc)
            if arc in banned_arcs:
                continue
            head = int(g.sig[arc])
            if head in banned_nodes:
                continue
            candidate = dist[node] + cost[arc]
            if candidate < dist[head] - 1e-15:
                dist[head] = candidate
                parent[head] = arc
                relaxations[head] += 1
                if relaxations[head] > n:
                    return ShortestPath(negative_cycle=_walk_cycle(g, parent, head))
                if not in_queue[head]:
                    queue.append(head)
                    in_queue[head] = True

    if not np.isfinite(dist[dst]):
        return ShortestPath()

    arcs: list[int] = []
    node = dst
    guard = 0
    while node != src:
        arc = int(parent[node])
        if arc < 0 or guard > n:
            return ShortestPath()
        arcs.append(arc)
        node = int(g.tau[arc])
        guard += 1
    arcs.reverse()
    return ShortestPath(arcs, float(dist[dst]), True)


def _walk_cycle(g: ArcArrays, parent: np.ndarray, node: int) -> list[int]:
    """Recover the arcs of a negative cycle from the parent pointers."""
    seen: dict[int, int] = {}
    arcs: list[int] = []
    guard = 0
    while node not in seen and guard <= g.n_nodes + 1:
        seen[node] = len(arcs)
        arc = int(parent[node])
        if arc < 0:
            return []
        arcs.append(arc)
        node = int(g.tau[arc])
        guard += 1
    return arcs[seen.get(node, 0) :]


def k_shortest_paths(
    g: ArcArrays, src: int, dst: int, k: int = 10, adj: Adjacency | None = None
) -> list[list[int]]:
    """Yen's algorithm over `eps`, returning arc-index paths.

    Only ever returns genuine `src -> dst` paths.  A negative cycle is not one:
    conflating the two lets a caller warm-start the solver from a loop that
    does not contain `src`, which then reports the graph as disconnected.
    """
    adj = adj or build_adjacency(g.tau, g.sig, g.n_nodes)

    weights = None
    first = spfa(g, src, dst, adj)
    if first.negative_cycle:
        # §5.3: "shift by min(0, min_p eps_p) and correct".  Shifting changes a
        # path's cost by a constant per hop rather than preserving the order,
        # so the result is only approximately cheapest -- which is fine, since
        # seed quality affects rounds and not correctness.
        weights = g.eps - min(0.0, float(g.eps.min()))
        first = spfa(g, src, dst, adj, weights=weights)
    if not first.found:
        return []

    accepted: list[list[int]] = [first.arcs]
    candidates: list[tuple[float, list[int]]] = []
    seen = {tuple(first.arcs)}

    while len(accepted) < k:
        previous = accepted[-1]
        for i in range(len(previous)):
            root = previous[:i]
            spur_node = int(g.tau[previous[i]])

            banned_arcs = {
                path[i]
                for path in accepted
                if len(path) > i and path[:i] == root
            }
            banned_nodes = {int(g.tau[arc]) for arc in root}

            spur = spfa(
                g, spur_node, dst, adj, weights=weights,
                banned_arcs=banned_arcs, banned_nodes=banned_nodes,
            )
            if not spur.found:
                continue
            whole = root + spur.arcs
            key = tuple(whole)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(candidates, (float(np.sum(g.eps[whole])), whole))

        if not candidates:
            break
        accepted.append(heapq.heappop(candidates)[1])

    return accepted


def seed_subgraph(
    g: ArcArrays,
    src: int,
    dst: int,
    *,
    k: int = 10,
    breadth: int = 8,
    include_negative: bool = True,
) -> np.ndarray:
    """Boolean mask of arcs to start column generation from.

    `breadth` is how many top-conductance arcs to add per touched node.  Adding
    breadth rather than depth is deliberate: the solver discovers depth through
    pricing-out, but it can only price in an arc that some node makes reachable.
    """
    mask = np.zeros(g.m, bool)
    if g.m == 0:
        return mask

    adj = build_adjacency(g.tau, g.sig, g.n_nodes)
    paths = k_shortest_paths(g, src, dst, k, adj)
    for path in paths:
        mask[path] = True

    touched = {src, dst}
    for path in paths:
        for arc in path:
            touched.add(int(g.tau[arc]))
            touched.add(int(g.sig[arc]))

    # Top-G arcs incident to any touched node, in either direction.
    incident_to = np.isin(g.tau, list(touched)) | np.isin(g.sig, list(touched))
    for node in touched:
        local = np.flatnonzero(incident_to & ((g.tau == node) | (g.sig == node)))
        if local.size > breadth:
            local = local[np.argsort(-g.G[local])[:breadth]]
        mask[local] = True

    if include_negative:
        # Free arbitrage the router should absorb; also cheap insurance against
        # a seed that missed a battery path entirely.
        mask[g.eps < 0] = True

    return mask
