"""Gap diagnostics — measure the Rocket↔best-solution gap (PRIVILEGED / ISOLATED).

This module is the ONLY place ``data/best_solution`` is read (see the subpackage
docstring for the leakage invariant). It provides the building blocks for Stage A of
the Phase-4 diagnosis:

* :func:`load_best_solution`   — parse + leakage-safe remap of the near-optimal order.
* :func:`surrogate_objective` — Rocket's sigmoid surrogate value ``Σ σ(β·Δ)·ŵ`` (the
  thing Rocket maximizes; equals ``-loss`` in :mod:`mfas.baseline.rocket`).
* :func:`order_to_positions` — rank → evenly-spaced ``[-1, 1]`` (Rocket's init convention).
* :func:`optimize_spacing`   — best achievable surrogate for a FIXED order (monotonic
  1-D spacing), so the static surrogate comparison is scale-fair rather than relying on
  an arbitrary evenly-spaced embedding.
* :func:`drift_probe`        — run Rocket FROM the best ordering and watch the discrete
  score drift (DIAGNOSTIC ONLY — it sees the answer; never a reportable score).
* :func:`make_synthetic_graph` — a seeded small graph with a planted near-optimum, used
  as a generalization check / Stage-B guard.

All heavy diagnostics are pure functions returning numbers / arrays; the driver
(``experiments/diagnostics.py``) is responsible for persisting figures/CSVs to
``experiments/outputs/`` (NOT ``results/``).
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..io import GraphData, data_dir
from ..metrics import pct, score_from_positions

__all__ = [
    "best_solution_path",
    "load_best_solution",
    "order_to_positions",
    "surrogate_objective",
    "optimize_spacing",
    "drift_probe",
    "make_synthetic_graph",
]


# ──────────────────────────────────────────────────────────────────────────────
# best_solution loader (the ONLY reader of data/best_solution)
# ──────────────────────────────────────────────────────────────────────────────
def best_solution_path() -> Path:
    """Locate the downloaded near-optimal submission under ``data/best_solution``."""
    d = data_dir() / "best_solution"
    gz = sorted(d.glob("*.csv.gz"))
    if not gz:
        raise FileNotFoundError(f"No *.csv.gz found in {d}")
    return gz[0]


def load_best_solution(g: GraphData, path: Optional[Path] = None
                       ) -> Tuple[np.ndarray, float]:
    """Parse ``best_solution`` and remap to ``g``'s contiguous internal indexing.

    The submission is a CSV ``Node ID,Order`` over the *challenge* node IDs. We map
    each challenge ID to the internal index ``io`` uses (``searchsorted`` on
    ``g.node_ids``), then place ``Order`` at that index. Coverage is asserted exactly
    (full node set, no duplicate ranks) — a silent mis-remap would garble the score.

    Returns
    -------
    order : int64 array, shape (n_nodes,)
        ``order[i]`` is the rank (0 = front/source side) of internal node ``i``.
    score_pct : float
        The exact feedforward % of this ordering under the frozen oracle (sanity:
        connectome ≈ 84.61%).
    """
    path = path or best_solution_path()
    ids, ranks = [], []
    with gzip.open(path, "rt") as f:
        header = next(f)
        assert header.strip().split(",")[:2] == ["Node ID", "Order"], \
            f"unexpected header: {header!r}"
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            ids.append(int(a))
            ranks.append(int(b))
    ids = np.asarray(ids, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)

    node_ids = np.asarray(g.node_ids)
    # Exact node-set coverage check (challenge IDs == graph's node IDs).
    assert ids.shape[0] == node_ids.shape[0], (
        f"row count {ids.shape[0]} != n_nodes {node_ids.shape[0]}")
    assert np.array_equal(np.unique(ids), node_ids), "node-set mismatch vs g.node_ids"
    assert np.array_equal(np.unique(ranks), np.arange(node_ids.shape[0])), \
        "Order is not a permutation of [0, n)"

    internal = np.searchsorted(node_ids, ids)
    assert np.array_equal(node_ids[internal], ids), "searchsorted remap invalid"

    order = np.empty(node_ids.shape[0], dtype=np.int64)
    order[internal] = ranks
    assert np.array_equal(np.unique(order), np.arange(node_ids.shape[0])), \
        "remapped order is not a clean permutation"

    score = score_from_positions(order, np.asarray(g.src), np.asarray(g.tgt), g.weight)
    return order, pct(score, g.total_weight)


# ──────────────────────────────────────────────────────────────────────────────
# Surrogate objective & position helpers
# ──────────────────────────────────────────────────────────────────────────────
def order_to_positions(order: np.ndarray) -> np.ndarray:
    """Map a rank vector in ``[0, n)`` to evenly-spaced positions in ``[-1, 1]``.

    Mirrors H02's ``_init_positions_from_order`` (rank 0 → −1, rank n−1 → +1). The
    surrogate depends on the *gaps* between positions, so this is only a default scale;
    :func:`optimize_spacing` finds the surrogate-optimal monotonic spacing for an order.
    """
    n = order.shape[0]
    return ((order.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0)


def _normalized_weights(g: GraphData) -> np.ndarray:
    """Rocket's ``ŵ = w / max(w)`` in float64 (the loss uses float32; we keep f64 for
    a precise, scale-stable diagnostic value)."""
    w = np.asarray(g.weight, dtype=np.float64)
    return w / float(w.max())


def surrogate_objective(positions: np.ndarray, g: GraphData, beta: float) -> float:
    """Rocket's surrogate value ``Σ σ(β·(pos[v] − pos[u]))·ŵ`` (higher = better).

    This equals ``-loss`` / ``neg_loss`` in :mod:`mfas.baseline.rocket` (same ŵ). It is
    the quantity Rocket maximizes; comparing it at different position vectors is the
    decisive surrogate test. Computed in float64 for stability.
    """
    pos = np.asarray(positions, dtype=np.float64)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    nw = _normalized_weights(g)
    delta = pos[tgt] - pos[src]
    sig = 1.0 / (1.0 + np.exp(-beta * delta))
    return float((sig * nw).sum())


def optimize_spacing(order: np.ndarray, g: GraphData, beta: float,
                     iters: int = 400, lr: float = 0.05, seed: int = 0,
                     device: str = "cpu", free_scale: bool = True
                     ) -> Tuple[np.ndarray, float]:
    """Best surrogate achievable for a FIXED order by optimizing monotonic spacing.

    The surrogate depends on position *gaps*, so an arbitrary (evenly-spaced) embedding of
    ``order`` under-states what the order can achieve. We parametrise strictly-increasing
    positions as ``p = cumsum(softplus(raw))`` (monotone in rank, hence order-preserving)
    and maximise :func:`surrogate_objective` at ``beta`` with Adam. The shape is
    standardised to unit variance; when ``free_scale`` (default) a learnable global scale
    is also optimised, so the order is NOT artificially under-powered relative to Rocket's
    converged positions (whose scale grows freely). The returned surrogate is the
    scale-fair value to compare against Rocket's converged surrogate.

    Returns ``(positions float64, surrogate_value)``. Order is preserved exactly by
    construction (monotone map of rank).
    """
    import torch

    n = order.shape[0]
    dev = torch.device(device)
    rank_of_node = np.asarray(order, dtype=np.int64)        # rank_of_node[node] = rank

    src = torch.as_tensor(np.asarray(g.src), dtype=torch.long, device=dev)
    tgt = torch.as_tensor(np.asarray(g.tgt), dtype=torch.long, device=dev)
    nw = torch.as_tensor(_normalized_weights(g), dtype=torch.float64, device=dev)
    inv = torch.as_tensor(rank_of_node, dtype=torch.long, device=dev)  # node -> rank

    torch.manual_seed(seed)
    # raw gaps between consecutive ranks (n-1 of them); softplus keeps them > 0.
    raw = torch.zeros(n - 1, dtype=torch.float64, device=dev, requires_grad=True)
    # learnable log-scale (overall sharpness); starts at 0 => unit-variance shape.
    log_scale = torch.zeros(1, dtype=torch.float64, device=dev, requires_grad=free_scale)
    params = [raw] + ([log_scale] if free_scale else [])
    opt = torch.optim.Adam(params, lr=lr)

    best_val = -np.inf
    best_pos = None
    for _ in range(iters):
        opt.zero_grad()
        gaps = torch.nn.functional.softplus(raw)              # > 0
        rank_pos = torch.cat([torch.zeros(1, dtype=torch.float64, device=dev),
                              torch.cumsum(gaps, dim=0)])      # increasing, len n
        rank_pos = (rank_pos - rank_pos.mean()) / (rank_pos.std() + 1e-12)
        rank_pos = rank_pos * torch.exp(log_scale)            # learnable overall scale
        pos = rank_pos[inv]                                   # node-space positions
        delta = pos[tgt] - pos[src]
        obj = (torch.sigmoid(beta * delta) * nw).sum()
        (-obj).backward()
        opt.step()
        v = float(obj.item())
        if v > best_val:
            best_val = v
            best_pos = pos.detach().cpu().numpy().copy()
    return best_pos, best_val


# ──────────────────────────────────────────────────────────────────────────────
# Drift probe — DIAGNOSTIC ONLY (sees the answer; keep outputs out of results/)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DriftResult:
    """Output of :func:`drift_probe` (diagnostic only)."""

    init_pct: float                  # discrete % of the best ordering at init
    final_pct: float                 # discrete % after running Rocket from it
    best_pct: float                  # best discrete % seen during the run
    history: object                  # the RocketResult.history DataFrame
    dropped: float                   # init_pct - final_pct (positive => surrogate pulls away)


def drift_probe(g: GraphData, order: np.ndarray, seed: int, device,
                epochs: Optional[int] = None, init_scale="even",
                time_limit: Optional[float] = None, cfg: Optional[object] = None
                ) -> DriftResult:
    """Run UNCHANGED Rocket initialised AT the best ordering; report discrete drift.

    DIAGNOSTIC ONLY. This intentionally leaks the answer into the init to test the
    *surrogate landscape* around the near-optimum: if Rocket's surrogate gradient pulls
    the discrete score DOWN, the surrogate is misaligned with the discrete objective near
    the optimum (DIRECTION R). If it holds/improves, Rocket simply did not reach this
    basin from a cold start (DIRECTION O/I). Its number is NEVER a reportable Rocket score
    and must not be written to ``results/``.

    ``init_scale`` selects the embedding handed to Rocket: ``"even"`` (evenly-spaced
    ``[-1,1]``) or ``"std"`` (evenly-spaced then standardised to unit variance, closer to
    Rocket's converged scale).
    """
    import torch

    from ..baseline.rocket import RocketConfig, run_rocket

    pos = order_to_positions(order)
    if init_scale == "std":
        pos = (pos - pos.mean()) / (pos.std() + 1e-12)
    elif isinstance(init_scale, (int, float)):
        pos = (pos - pos.mean()) / (pos.std() + 1e-12) * float(init_scale)
    init = torch.as_tensor(pos.astype(np.float32), device=device)

    if cfg is None:
        cfg = RocketConfig(epochs=epochs if epochs is not None
                           else {"connectome": 20_000, "mouse": 5_000}.get(g.name, 20_000))
    res = run_rocket(g, cfg, seed=seed, device=device,
                     init_positions=init, time_limit=time_limit)
    init_score = score_from_positions(pos, np.asarray(g.src), np.asarray(g.tgt), g.weight)
    init_pct = pct(init_score, g.total_weight)
    final_pct = float(res.history["pct"].iloc[-1])   # discrete % at the LAST epoch (post-drift)
    return DriftResult(
        init_pct=init_pct,
        final_pct=final_pct,
        best_pct=res.best_pct,                       # best discrete % seen (monotone tracker)
        history=res.history,
        dropped=init_pct - final_pct,                # >0 => surrogate dynamics pull discrete down
    )


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic fixture — planted near-optimum (generalization check / Stage-B guard)
# ──────────────────────────────────────────────────────────────────────────────
def make_synthetic_graph(n: int = 400, avg_out: int = 8, feedback_frac: float = 0.12,
                         weight_hi: int = 50, seed: int = 0,
                         name: str = "synthetic") -> Tuple[GraphData, np.ndarray, float]:
    """Build a seeded directed weighted graph with a PLANTED near-optimal order.

    Construction (leakage-safe — the planted order is returned for DIAGNOSTIC comparison
    only, never used inside any variant's optimization path):

    * nodes ``0..n-1`` carry a hidden ground-truth order = identity (rank i = node i);
    * forward edges connect ``i → j`` with ``i < j`` (consistent with the order),
      carrying the bulk of the weight;
    * a minority of feedback edges ``i → j`` with ``i > j`` carry smaller weight, so the
      planted order is near-optimal but not perfect (some irreducible feedback), like the
      connectomes.

    Returns ``(GraphData, planted_order, planted_pct)`` where ``planted_order[i] = i`` and
    ``planted_pct`` is its exact feedforward %. Weights are int64 (connectome-like).
    """
    rng = np.random.RandomState(seed)
    src, tgt, w = [], [], []
    n_fwd = n * avg_out
    # Forward edges: pick i<j by sampling two distinct nodes and orienting low->high.
    a = rng.randint(0, n, size=n_fwd)
    b = rng.randint(0, n, size=n_fwd)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    src.append(lo); tgt.append(hi)
    w.append(rng.randint(1, weight_hi + 1, size=lo.shape[0]))
    # Feedback edges: orient high->low, fewer and lighter.
    n_fb = int(feedback_frac * lo.shape[0])
    a = rng.randint(0, n, size=n_fb)
    b = rng.randint(0, n, size=n_fb)
    lo2, hi2 = np.minimum(a, b), np.maximum(a, b)
    keep2 = lo2 != hi2
    lo2, hi2 = lo2[keep2], hi2[keep2]
    src.append(hi2); tgt.append(lo2)  # high -> low = feedback wrt identity order
    w.append(rng.randint(1, max(weight_hi // 3, 2) + 1, size=lo2.shape[0]))

    src = np.concatenate(src).astype(np.int32)
    tgt = np.concatenate(tgt).astype(np.int32)
    weight = np.concatenate(w).astype(np.int64)
    node_ids = np.arange(n, dtype=np.int64)
    g = GraphData(src=src, tgt=tgt, weight=weight, node_ids=node_ids, name=name)

    planted_order = np.arange(n, dtype=np.int64)
    sc = score_from_positions(planted_order, src, tgt, weight)
    return g, planted_order, pct(sc, g.total_weight)
