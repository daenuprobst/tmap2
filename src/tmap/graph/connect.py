"""Connect a disconnected k-NN graph into a single component.

When ``n_neighbors`` (k) is too low, the k-NN graph can fragment into several
connected components.  Its minimum spanning tree is then a *forest* (one tree
per component), the layout packs the components as disjoint blobs, and TMAP's
``path`` / ``distance`` / ``distances_from`` operations are undefined across
components.

``connect_knn_components`` detects the components and adds the minimum-weight
"bridge" edges needed to join them into one, so the downstream MST yields a
single spanning tree.  Bridges are found by re-querying the ANN index for more
neighbors -- the shortest cross-component links the kept-k graph missed -- using
a Boruvka/Kruskal-style sweep.  If no index is available (e.g. a user-supplied
``knn_graph``), or some components stay isolated even at a large neighbor count,
the remaining components are joined through representative nodes as a last
resort so a single tree is always guaranteed.

Bridge edges are appended as extra neighbor columns on the returned ``KNNGraph``
(unused slots use index ``-1`` / distance ``inf``, which every downstream reader
already skips), so the augmentation is transparent to the rest of the pipeline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from tmap.index.types import KNNGraph

# A callable that returns a fresh KNNGraph with ``k`` neighbors per node, used to
# look for cross-component links the kept-k graph missed.
RequeryFn = Callable[[int], KNNGraph]


class _DSU:
    """Disjoint-set union with path halving."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.n_sets = n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.n_sets -= 1
        return True


def _valid_edges(
    knn: KNNGraph,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Flatten a KNNGraph to (src, dst, dist) arrays, dropping -1/self/out-of-range."""
    idx = knn.indices
    n, k = idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idx.ravel().astype(np.int64)
    dist = knn.distances.ravel().astype(np.float64)
    mask = (dst >= 0) & (dst < n) & (dst != src) & np.isfinite(dist)
    return src[mask], dst[mask], dist[mask]


def _labels(dsu: _DSU, n: int) -> NDArray[np.int64]:
    """Current root (component id) of every node."""
    return np.array([dsu.find(i) for i in range(n)], dtype=np.int64)


def _augment(knn: KNNGraph, bridges: list[tuple[int, int, float]]) -> KNNGraph:
    """Return a KNNGraph with bridge edges appended as extra neighbor columns."""
    if not bridges:
        return knn
    n, k = knn.indices.shape
    inc = np.zeros(n, dtype=np.int64)
    for a, b, _ in bridges:
        inc[a] += 1
        inc[b] += 1
    pad = int(inc.max())

    idx = np.full((n, k + pad), -1, dtype=np.int32)
    dist = np.full((n, k + pad), np.inf, dtype=np.float32)
    idx[:, :k] = knn.indices
    dist[:, :k] = knn.distances

    nxt = np.full(n, k, dtype=np.int64)
    for a, b, w in bridges:
        idx[a, nxt[a]] = b
        dist[a, nxt[a]] = w
        nxt[a] += 1
        idx[b, nxt[b]] = a
        dist[b, nxt[b]] = w
        nxt[b] += 1
    return KNNGraph.from_arrays(idx, dist)


def connect_knn_components(
    knn: KNNGraph,
    requery: RequeryFn | None = None,
    *,
    max_neighbors: int = 512,
) -> tuple[KNNGraph, int, int]:
    """Join a disconnected k-NN graph into a single connected component.

    Parameters
    ----------
    knn : KNNGraph
        The graph to connect (may be disconnected).
    requery : callable, optional
        ``requery(k) -> KNNGraph`` producing a graph with ``k`` neighbors per
        node, used to discover cross-component bridges.  If ``None``, only the
        representative-chaining fallback is available.
    max_neighbors : int, default 512
        Cap on the neighbor count used while searching for bridges.

    Returns
    -------
    (knn, n_components, n_bridges)
        The (possibly augmented) graph, the number of components found in the
        input, and the number of bridge edges added.
    """
    n = knn.n_nodes
    if n < 2:
        return knn, 1, 0

    # 1. Components of the current graph.
    src, dst, _ = _valid_edges(knn)
    dsu = _DSU(n)
    for a, b in zip(src.tolist(), dst.tolist()):
        dsu.union(a, b)
    if dsu.n_sets == 1:
        return knn, 1, 0

    n_components = dsu.n_sets
    bridges: list[tuple[int, int, float]] = []

    # 2. Boruvka/Kruskal sweep: re-query with a growing neighbor count and add
    #    the shortest cross-component edges until everything is joined.
    if requery is not None:
        k_try = max(knn.k * 4, 16)
        while dsu.n_sets > 1:
            k_eff = min(k_try, n - 1, max_neighbors)
            cand = requery(k_eff)
            csrc, cdst, cdist = _valid_edges(cand)
            comp = _labels(dsu, n)
            cross = comp[csrc] != comp[cdst]
            csrc, cdst, cdist = csrc[cross], cdst[cross], cdist[cross]
            if csrc.size:
                order = np.argsort(cdist, kind="stable")
                for e in order:
                    a, b, w = int(csrc[e]), int(cdst[e]), float(cdist[e])
                    if dsu.union(a, b):
                        bridges.append((a, b, w))
                        if dsu.n_sets == 1:
                            break
            if k_eff >= min(n - 1, max_neighbors):
                break  # neighbor count maxed out; rest goes to the fallback
            k_try *= 2

    # 3. Last resort: chain any still-separate components through representatives.
    if dsu.n_sets > 1:
        reps: dict[int, int] = {}
        for i in range(n):
            r = dsu.find(i)
            if r not in reps:
                reps[r] = i
        rep_nodes = list(reps.values())
        # Weight bridges as the most expensive edges so the MST adds them last.
        fallback_w = max((w for _, _, w in bridges), default=0.0)
        if fallback_w == 0.0:
            _, _, d = _valid_edges(knn)
            fallback_w = float(np.max(d)) if d.size else 1.0
        for a, b in zip(rep_nodes[:-1], rep_nodes[1:]):
            if dsu.union(a, b):
                bridges.append((a, b, fallback_w))

    return _augment(knn, bridges), n_components, len(bridges)
