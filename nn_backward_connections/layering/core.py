"""Core algorithms: layer assignment, edge classification, within-layer ordering.

Task spec v1 (full version in layering/README.md):
  input  = a directed weighted graph plus a linear order pi (the MFAS optimizer's
           output, loaded from the pinned champion artifact);
  output = (1) an assignment of nodes to layers such that every feedforward edge
           goes to a strictly later layer (no intra-layer FF edges - the HARD
           version), and (2) an ordering of nodes inside each layer that keeps
           the number of edge crossings low.

Two layer-assignment methods are implemented side by side (decision T2):

  A. ``layers_longest_path`` - earliest-feasible layer from the FF DAG only.
     Fewest possible layers; on node pairs with no FF path the layer order may
     disagree with pi, which is what lets some pi-feedback edges become
     forward-by-layer ("reclaimed").
  B. ``layers_pi_slices`` - layers are contiguous slices of pi. Fully preserves
     pi; feedback can never be reclassified; usually more layers.

Design notes for future work / agents:
  * Deterministic and pure numpy + python loops. No scipy, no np.dot/matmul:
    the ``allen`` env's BLAS aborts the interpreter (campaign queue item P26).
  * The v1 target is the mouse graph (148 nodes, 583 edges), so O(E^2) crossing
    counting and python-loop layerings are deliberate simplicity, not an
    oversight. Scaling to the fly connectome (136k nodes, 5.7M edges) will need
    different internals - keep the API, swap the implementation.
  * pi is always carried as ``order`` (order[k] = node at slot k) together with
    its inverse ``rank`` (rank[v] = slot of node v). The frozen oracle's
    ``score_from_order`` consumes RANKS, not the order permutation.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

__all__ = [
    "FORWARD", "INTRA", "BACKWARD",
    "rank_from_order", "ff_mask_by_order", "mutual_edge_mask",
    "ff_path_exists_mask",
    "layers_longest_path", "layers_pi_slices", "classify_by_layers",
    "initial_slots_from_order", "y_coords", "count_crossings",
    "barycenter_order", "layout_stats",
]

# Edge classes under a layer assignment (see classify_by_layers).
FORWARD, INTRA, BACKWARD = 1, 0, -1


# ---------------------------------------------------------------------------
# The linear order pi and the FF/FB split it induces
# ---------------------------------------------------------------------------

def rank_from_order(order: np.ndarray) -> np.ndarray:
    """Inverse permutation: rank[v] = slot of node v on the pi line."""
    order = np.asarray(order)
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank


def ff_mask_by_order(src: np.ndarray, tgt: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Feedforward mask under pi: edge (u, v) is FF iff rank[u] < rank[v].

    Self-loops (u == v) are neither feedforward nor feedback; they come out
    False here and must be excluded from feedback counts separately (both v1
    datasets have none).
    """
    return rank[np.asarray(src)] < rank[np.asarray(tgt)]


def mutual_edge_mask(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Mask of edges whose reverse edge also exists (halves of mutual pairs).

    For a mutual pair exactly one direction is FF under any order, so the other
    half is feedback that NO order and NO valid layering can reclaim.
    A self-loop would count as its own reverse; both v1 datasets are
    self-loop-free (validated at load).
    """
    src_l = np.asarray(src).tolist()
    tgt_l = np.asarray(tgt).tolist()
    pairs = set(zip(src_l, tgt_l))
    return np.fromiter(((b, a) in pairs for a, b in zip(src_l, tgt_l)),
                       dtype=bool, count=len(src_l))


def ff_path_exists_mask(src, tgt, ff_mask, query_from, query_to) -> np.ndarray:
    """For each query pair i, whether a directed path made ONLY of FF edges
    runs from ``query_from[i]`` to ``query_to[i]`` (BFS per query, early exit;
    a trivial query with from == to returns True).

    Used to CERTIFY feedback edges as unreclaimable: a pi-feedback edge (u, v)
    can become forward in SOME hard layering iff there is NO FF path v -> u,
    because an FF path forces layer[v] < layer[u] in every valid layering.
    O(n_queries * E) - mouse-scale only.
    """
    adj: dict = {}
    for a, b in zip(np.asarray(src)[np.asarray(ff_mask)].tolist(),
                    np.asarray(tgt)[np.asarray(ff_mask)].tolist()):
        adj.setdefault(a, []).append(b)
    q_from = np.asarray(query_from).tolist()
    q_to = np.asarray(query_to).tolist()
    out = np.zeros(len(q_from), dtype=bool)
    for i, (a, b) in enumerate(zip(q_from, q_to)):
        if a == b:
            out[i] = True
            continue
        seen = {a}
        frontier = deque([a])
        found = False
        while frontier and not found:
            u = frontier.popleft()
            for v in adj.get(u, ()):
                if v == b:
                    found = True
                    break
                if v not in seen:
                    seen.add(v)
                    frontier.append(v)
        out[i] = found
    return out


# ---------------------------------------------------------------------------
# Layer assignment (the HARD version: no intra-layer FF edges)
# ---------------------------------------------------------------------------

def _ff_in_edges(src, tgt, ff_mask, n: int) -> List[List[int]]:
    """Per-node list of FF in-neighbours."""
    in_edges: List[List[int]] = [[] for _ in range(n)]
    for s, t in zip(np.asarray(src)[ff_mask].tolist(),
                    np.asarray(tgt)[ff_mask].tolist()):
        in_edges[t].append(s)
    return in_edges


def layers_longest_path(src, tgt, ff_mask, order) -> np.ndarray:
    """Method A: earliest-feasible ("longest path") layers from the FF DAG.

    layer[v] = length of the longest FF path ending at v; FF sources land in
    layer 0. pi enters only through the FF/FB split and as the topological
    sweep order (pi is a topological order of the FF DAG by construction).
    On pairs with no FF path the layer order may DISAGREE with pi - this is
    what makes feedback reclamation possible (decision T2).
    """
    order = np.asarray(order)
    n = order.shape[0]
    in_edges = _ff_in_edges(src, tgt, ff_mask, n)
    layer = np.zeros(n, dtype=np.int64)
    for v in order.tolist():
        if in_edges[v]:
            layer[v] = max(layer[u] for u in in_edges[v]) + 1
    return layer


def layers_pi_slices(src, tgt, ff_mask, order) -> np.ndarray:
    """Method B: layers as contiguous slices of pi.

    Walk pi left to right and cut a new layer right before the first node that
    receives an FF edge from inside the current slice (greedy, so each slice is
    maximal). Layer index is non-decreasing along pi, hence pi is fully
    preserved and a pi-feedback edge can never become forward-by-layer.
    """
    order = np.asarray(order)
    n = order.shape[0]
    in_edges = _ff_in_edges(src, tgt, ff_mask, n)
    layer = np.zeros(n, dtype=np.int64)
    current: set = set()
    cur = 0
    for v in order.tolist():
        if any(u in current for u in in_edges[v]):
            cur += 1
            current = set()
        layer[v] = cur
        current.add(v)
    return layer


def classify_by_layers(src, tgt, layer: np.ndarray) -> np.ndarray:
    """Per-edge class under a layer assignment (decision from the T2 discussion:
    the drawn/reported classification is BY LAYERS, not by pi).

    Returns int8: FORWARD (+1) if layer[u] < layer[v], INTRA (0) if equal,
    BACKWARD (-1) otherwise.
    """
    d = layer[np.asarray(tgt)] - layer[np.asarray(src)]
    return np.sign(d).astype(np.int8)


# ---------------------------------------------------------------------------
# Within-layer ordering (crossing minimization) and the crossing oracle
# ---------------------------------------------------------------------------

def initial_slots_from_order(layer: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Initial within-layer slots: nodes of each layer in pi order."""
    n = layer.shape[0]
    idx = np.lexsort((rank, layer))          # sort by (layer, pi-rank)
    counts = np.bincount(layer)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    slot = np.empty(n, dtype=np.int64)
    slot[idx] = np.arange(n, dtype=np.int64) - starts[layer[idx]]
    return slot


def y_coords(layer: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Vertical drawing coordinate: slots centered around 0 within each layer
    (so layers of different width are vertically centered, like an ANN diagram).
    """
    widths = np.bincount(layer)
    return slot - (widths[layer] - 1) / 2.0


def count_crossings(src, tgt, edge_mask, x: np.ndarray, y: np.ndarray) -> int:
    """Number of PROPER pairwise crossings of straight edge segments.

    Edges are drawn as straight segments (x[u], y[u]) -> (x[v], y[v]); this
    counts exactly the visual crossings of the drawing, including those made by
    layer-skipping edges (decision 3: edges may skip layers; a long straight
    edge pays for everything it passes through). Unweighted, per decision 4.
    Pairs sharing an endpoint and degenerate/collinear touchings count 0.

    O(m^2) pairwise with elementwise numpy only - fine at mouse scale
    (583^2 ~ 3.4e5 pairs), NOT for the fly connectome.
    """
    edge_mask = np.asarray(edge_mask)
    s = np.asarray(src)[edge_mask]
    t = np.asarray(tgt)[edge_mask]
    m = int(s.shape[0])
    if m < 2:
        return 0
    px, py = x[s].astype(np.float64), y[s].astype(np.float64)
    qx, qy = x[t].astype(np.float64), y[t].astype(np.float64)

    def _orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    o1 = _orient(px[:, None], py[:, None], qx[:, None], qy[:, None], px[None, :], py[None, :])
    o2 = _orient(px[:, None], py[:, None], qx[:, None], qy[:, None], qx[None, :], qy[None, :])
    o3 = _orient(px[None, :], py[None, :], qx[None, :], qy[None, :], px[:, None], py[:, None])
    o4 = _orient(px[None, :], py[None, :], qx[None, :], qy[None, :], qx[:, None], qy[:, None])
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    share = ((s[:, None] == s[None, :]) | (s[:, None] == t[None, :]) |
             (t[:, None] == s[None, :]) | (t[:, None] == t[None, :]))
    crossing &= ~share
    return int(crossing.sum()) // 2          # symmetric matrix; diagonal is `share`


def barycenter_order(src, tgt, edge_mask, layer, slot0,
                     n_sweeps: int = 10) -> Tuple[np.ndarray, int]:
    """Iterative barycenter heuristic for within-layer ordering.

    Alternating sweeps (left-to-right then right-to-left): each node's sort key
    is the mean current y of its masked neighbours on the swept-from side; a
    node with no such neighbours keeps its current y. Skip edges participate
    with their true endpoint y, matching the straight-line drawing. Ties break
    by current slot, so the pass is deterministic.

    Returns ``(best_slot, best_crossings)`` - the best slot vector seen under
    ``count_crossings``, never worse than ``slot0``. Stops early on zero
    crossings or a converged sweep.
    """
    layer = np.asarray(layer)
    n = layer.shape[0]
    n_layers = int(layer.max()) + 1
    s = np.asarray(src)[np.asarray(edge_mask)].tolist()
    t = np.asarray(tgt)[np.asarray(edge_mask)].tolist()
    left: List[List[int]] = [[] for _ in range(n)]    # neighbours in earlier layers
    right: List[List[int]] = [[] for _ in range(n)]   # neighbours in later layers
    for u, v in zip(s, t):
        if layer[u] < layer[v]:
            left[v].append(u)
            right[u].append(v)
        elif layer[u] > layer[v]:
            left[u].append(v)
            right[v].append(u)
        # intra-layer edges do not participate (decision T4: arcs are not in
        # the crossing objective for v1)
    nodes_in = [np.where(layer == lyr)[0] for lyr in range(n_layers)]
    x = layer.astype(np.float64)
    slot = np.asarray(slot0).copy()
    best_slot = slot.copy()
    best_c = count_crossings(src, tgt, edge_mask, x, y_coords(layer, slot))
    for _ in range(n_sweeps):
        prev = slot.copy()
        for layer_range, nbrs in ((range(1, n_layers), left),
                                  (range(n_layers - 2, -1, -1), right)):
            for lyr in layer_range:
                nodes = nodes_in[lyr]
                if nodes.shape[0] < 2:
                    continue
                yy = y_coords(layer, slot)
                key = np.array([float(np.mean([yy[u] for u in nbrs[v]])) if nbrs[v]
                                else float(yy[v]) for v in nodes.tolist()])
                new_order = nodes[np.lexsort((slot[nodes], key))]
                slot[new_order] = np.arange(new_order.shape[0], dtype=np.int64)
        c = count_crossings(src, tgt, edge_mask, x, y_coords(layer, slot))
        if c < best_c:
            best_c, best_slot = c, slot.copy()
        if c == 0 or np.array_equal(slot, prev):
            break
    return best_slot, int(best_c)


# ---------------------------------------------------------------------------
# Structural statistics (the "structure" half of the deliverable)
# ---------------------------------------------------------------------------

def layout_stats(src, tgt, weight, rank, layer, method: str) -> dict:
    """Structural summary of a layer assignment.

    Scalars are notebook-table material; the two list-valued keys (``widths``,
    ``span_hist``) feed the profile plots. ``reclaimed_*`` counts pi-feedback
    edges that became forward-by-layer (possible only for method A).
    """
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    w = np.asarray(weight, dtype=np.float64)
    total_w = float(w.sum())
    ff_pi = ff_mask_by_order(src, tgt, rank)
    cls = classify_by_layers(src, tgt, layer)
    fwd, intra, back = cls == FORWARD, cls == INTRA, cls == BACKWARD
    widths = np.bincount(layer, minlength=int(layer.max()) + 1)
    span = (layer[tgt] - layer[src])[fwd]
    reclaimed = (~ff_pi) & fwd
    return {
        "method": method,
        "n_layers": int(widths.shape[0]),
        "max_width": int(widths.max()),
        "mean_width": float(widths.mean()),
        "edges_fwd": int(fwd.sum()),
        "edges_intra": int(intra.sum()),
        "edges_back": int(back.sum()),
        "w_fwd_pct": 100.0 * float(w[fwd].sum()) / total_w,
        "w_intra_pct": 100.0 * float(w[intra].sum()) / total_w,
        "w_back_pct": 100.0 * float(w[back].sum()) / total_w,
        "w_ff_pi_pct": 100.0 * float(w[ff_pi].sum()) / total_w,
        "reclaimed_edges": int(reclaimed.sum()),
        "reclaimed_w_pct": 100.0 * float(w[reclaimed].sum()) / total_w,
        "adjacent_fwd_frac": float((span == 1).mean()) if span.size else float("nan"),
        "mean_span": float(span.mean()) if span.size else float("nan"),
        "widths": widths.tolist(),
        "span_hist": np.bincount(span).tolist() if span.size else [],
    }
