"""Graph analytics: strongly connected components and degree statistics.

These are read-only diagnostics (dataset characterization, future random-graph
comparison). This module is independent of the Rocket path and must not be imported
by ``baseline/rocket.py``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .io import GraphData

__all__ = [
    "build_csr",
    "strongly_connected_components",
    "largest_scc_mask",
    "degree_stats",
]


def build_csr(g: GraphData) -> sp.csr_matrix:
    """Build a weighted CSR adjacency matrix from a :class:`GraphData`."""
    n = g.n_nodes
    return sp.coo_matrix(
        (np.asarray(g.weight, dtype=np.float64),
         (np.asarray(g.src), np.asarray(g.tgt))),
        shape=(n, n),
    ).tocsr()


def strongly_connected_components(g: GraphData) -> Tuple[int, np.ndarray]:
    """Return ``(n_components, labels)`` for strong connectivity (scipy csgraph)."""
    adj = build_csr(g)
    n_comp, labels = connected_components(adj, directed=True, connection="strong")
    return int(n_comp), labels


def largest_scc_mask(g: GraphData) -> np.ndarray:
    """Boolean node mask selecting the largest strongly connected component."""
    _, labels = strongly_connected_components(g)
    counts = np.bincount(labels)
    biggest = int(counts.argmax())
    return labels == biggest


def degree_stats(g: GraphData) -> Dict[str, float]:
    """Compute degree / weighted-degree summary statistics.

    Mirrors the notebook's graph-statistics cell: in/out degree, weighted in/out,
    number of source (in-deg 0) and sink (out-deg 0) vertices.
    """
    n = g.n_nodes
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)

    in_deg = np.bincount(tgt, minlength=n).astype(np.int64)
    out_deg = np.bincount(src, minlength=n).astype(np.int64)
    in_w = np.zeros(n, dtype=np.float64)
    out_w = np.zeros(n, dtype=np.float64)
    np.add.at(in_w, tgt, w)
    np.add.at(out_w, src, w)

    n_comp, labels = strongly_connected_components(g)
    scc_sizes = np.bincount(labels)

    return dict(
        n_nodes=n,
        n_edges=g.n_edges,
        total_weight=g.total_weight,
        density=g.n_edges / (n * (n - 1)) if n > 1 else 0.0,
        mean_in_degree=float(in_deg.mean()),
        mean_out_degree=float(out_deg.mean()),
        max_in_degree=int(in_deg.max()),
        max_out_degree=int(out_deg.max()),
        n_sources=int((in_deg == 0).sum()),
        n_sinks=int((out_deg == 0).sum()),
        weight_min=float(w.min()),
        weight_median=float(np.median(w)),
        weight_mean=float(w.mean()),
        weight_max=float(w.max()),
        weight_p99=float(np.percentile(w, 99)),
        n_scc=n_comp,
        largest_scc=int(scc_sizes.max()),
        largest_scc_frac=float(scc_sizes.max() / n),
    )
