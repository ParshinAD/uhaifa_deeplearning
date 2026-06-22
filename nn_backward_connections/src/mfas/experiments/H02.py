"""Variant H02 — Warm-start Rocket from a greedy MFAS ordering (GreedyAbs-style init).

Hypothesis (backlog H02): initializing the continuous positions from a degree/greedy
ordering of the graph -- instead of the paper-default N(0,1) -- seeds the optimizer in a
better basin and yields a higher final feedforward weight. The paper's Crane phase uses a
topological (TopoShuffle/Kahn) init for the same reason, and greedy MFAS heuristics already
reach ~68-72% feedforward weight on these graphs, so the continuous optimizer starts near a
good discrete solution (a classic continuation trick).

Init source (leakage-safe)
---------------------------
The ordering is the **Eades-Lin-Smyth / GreedyAbs greedy FAS heuristic** (Eades et al. 1993;
Simpson et al. 2016), computed ONLY from the input graph's structure and edge weights:
repeatedly peel isolated nodes, then all sinks (no remaining out-weight -> placed at the
back) and all sources (no remaining in-weight -> placed at the front), otherwise remove the
remaining node with maximum ``out_w - in_w`` (source-like -> front). This is exactly the
greedy ordering that reaches ~68-72%; it NEVER reads the discrete oracle / target metric.
The resulting rank vector ``[0, n)`` is mapped to evenly-spaced standardized positions in
``[-1, 1]`` (front rank -> small position = early = source side), which is the continuous
initialization handed to the UNCHANGED ``run_rocket``.

A full Kahn topological sort is ill-defined on a cyclic graph (both connectomes have
feedback), so we use this greedy DAG-ordering heuristic instead -- it degenerates to a true
topological order on a DAG and is the standard cheap surrogate. The selection of the
max-``delta`` node uses a lazy max-heap so it runs in ~O((n + m) log n) rather than the naive
O(n^2) argmax-per-step (which is ~1.8e10 ops on the 136k-node connectome and infeasible).

Compute-matched
---------------
This is a standard knob-swap (init only): the loss, optimizer, LR/beta schedules and epoch
budget are the baseline's (connectome 20k, mouse 5k). Only ``init_positions`` changes; the
one-time greedy ordering is pre-optimization graph work and adds no optimizer steps.
``RocketResult.n_epochs_done`` is the optimizer steps from ``run_rocket`` (= baseline budget),
so it is screened, at equal total gradient steps, against ``baseline_passthrough`` /
the frozen baseline. The oracle is used only as the baseline already uses it: best-by-oracle
tracking inside ``run_rocket``.
"""
from __future__ import annotations

import heapq
from typing import Optional

import numpy as np
import torch

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData

ID = "H02"
HYPOTHESIS = (
    "Warm-starting Rocket from a GreedyAbs greedy-FAS ordering (graph-structure only, "
    "leakage-safe), mapped to evenly-spaced positions in [-1,1], gives a better basin "
    "and higher final feedforward weight than the N(0,1) default init."
)

# Per-dataset total epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}


def greedy_fas_order(g: GraphData) -> np.ndarray:
    """Eades-Lin-Smyth / GreedyAbs greedy FAS ordering of the graph nodes.

    Leakage-safe: uses ONLY ``g.src``/``g.tgt``/``g.weight`` (input structure + weights),
    never the discrete oracle score. Returns ``order`` where ``order[u]`` is the rank/position
    of node ``u`` in ``[0, n)`` (smaller = earlier = source side).

    Algorithm (peeling, lazy max-heap for the max-(out_w - in_w) tie-break):
      - sinks (no remaining out-weight) are appended to the BACK of the sequence;
      - sources (no remaining in-weight) are prepended to the FRONT;
      - otherwise the remaining node with maximum ``out_w - in_w`` goes to the FRONT.
    On a DAG this degenerates to a topological order; on a cyclic graph it is the standard
    cheap greedy surrogate that reaches ~68-72% feedforward weight on these connectomes.
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    # Remaining out-/in-weight per node (updated as neighbours are removed).
    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, src, w)
    np.add.at(in_w, tgt, w)

    # CSR-style adjacency for O(deg) incremental updates when a node is removed.
    # Out-neighbours of u (sorted by u): removing u decreases in_w of each such v.
    out_order = np.argsort(src, kind="stable")
    out_start = np.searchsorted(src[out_order], np.arange(n + 1))
    out_nbr = tgt[out_order]
    out_wt = w[out_order]
    # In-neighbours of v (sorted by v): removing v decreases out_w of each such u.
    in_order = np.argsort(tgt, kind="stable")
    in_start = np.searchsorted(tgt[in_order], np.arange(n + 1))
    in_nbr = src[in_order]
    in_wt = w[in_order]

    active = np.ones(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    front = 0
    back = n - 1

    EPS = 1e-12  # weights are ints (connectome) or positive floats (mouse)

    # Source / sink queues (nodes whose remaining in-/out-weight has hit zero).
    sinks = [u for u in range(n) if out_w[u] <= EPS]
    sources = [u for u in range(n) if in_w[u] <= EPS and out_w[u] > EPS]
    # Lazy max-heap on (-(out_w - in_w), u); stale entries are skipped on pop.
    heap = [(-(out_w[u] - in_w[u]), u) for u in range(n)
            if out_w[u] > EPS and in_w[u] > EPS]
    heapq.heapify(heap)
    heap_val = (out_w - in_w).copy()  # value each heap entry must match to be current

    def remove(u: int, to_front: bool) -> None:
        nonlocal front, back
        active[u] = False
        if to_front:
            order[u] = front
            front += 1
        else:
            order[u] = back
            back -= 1
        # Update neighbours' remaining degrees; enqueue any that become source/sink.
        for k in range(out_start[u], out_start[u + 1]):
            v = int(out_nbr[k])
            if active[v]:
                in_w[v] -= out_wt[k]
                if in_w[v] <= EPS and out_w[v] > EPS:
                    sources.append(v)
                else:
                    heap_val[v] = out_w[v] - in_w[v]
                    heapq.heappush(heap, (-(heap_val[v]), v))
        for k in range(in_start[u], in_start[u + 1]):
            v = int(in_nbr[k])
            if active[v]:
                out_w[v] -= in_wt[k]
                if out_w[v] <= EPS:
                    sinks.append(v)
                else:
                    heap_val[v] = out_w[v] - in_w[v]
                    heapq.heappush(heap, (-(heap_val[v]), v))

    placed = 0
    while placed < n:
        # Peel all current sinks (back) and sources (front) first.
        progressed = False
        while sinks:
            u = sinks.pop()
            if active[u] and out_w[u] <= EPS:
                remove(u, to_front=False)
                placed += 1
                progressed = True
        while sources:
            u = sources.pop()
            if active[u] and in_w[u] <= EPS and out_w[u] > EPS:
                remove(u, to_front=True)
                placed += 1
                progressed = True
        if placed >= n:
            break
        if progressed:
            continue
        # No sinks/sources: take the active node with max (out_w - in_w) -> front.
        u = -1
        while heap:
            neg, cand = heapq.heappop(heap)
            if not active[cand]:
                continue
            if -neg != heap_val[cand]:  # stale entry
                continue
            u = cand
            break
        if u == -1:
            # Fallback: heap exhausted but nodes remain (all equal / numerical edge case).
            remaining = np.nonzero(active)[0]
            diff = out_w[remaining] - in_w[remaining]
            u = int(remaining[int(np.argmax(diff))])
        remove(u, to_front=True)
        placed += 1

    return order


def _init_positions_from_order(order: np.ndarray, device) -> torch.Tensor:
    """Map a rank vector ``order`` in ``[0, n)`` to evenly-spaced positions in [-1, 1].

    Rank 0 (front / source side) -> -1, rank n-1 (back / sink side) -> +1. Matches the
    Rocket convention where a feedforward edge ``(u, v)`` wants ``pos[u] < pos[v]``.
    """
    n = order.shape[0]
    pos = ((order.astype(np.float32) / max(n - 1, 1)) * 2.0 - 1.0).astype(np.float32)
    return torch.tensor(pos, device=device)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run UNCHANGED Rocket, warm-started from the GreedyAbs greedy-FAS ordering.

    Only the initialization differs from baseline: a one-time, leakage-safe greedy ordering
    (graph structure + weights only) is mapped to evenly-spaced positions and passed as
    ``init_positions`` to ``run_rocket`` at the baseline epoch budget. No post-processing /
    local search is added (that is H04); the PURE Rocket score is reported.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    return run_rocket(g, cfg, seed=seed, device=device,
                      init_positions=init_positions, time_limit=time_limit)
