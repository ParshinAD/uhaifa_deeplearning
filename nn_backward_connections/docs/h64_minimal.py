"""Minimal, self-contained re-implementation of the H64 connectome champion pipeline.

This file exists for TEACHING: it is the code quoted step by step in
``docs/H64_GUIDE.ru.md``. It is deliberately short and readable, not fast.
The production code lives in ``src/mfas/`` and is what every logged number came from:

    stage 1  greedy-FAS warm start         src/mfas/experiments/H02.py   greedy_fas_order
    stage 2  Rocket + ASYM surrogate       src/mfas/experiments/H64.py   _rocket_asym
    stage 3  under-relaxed exact-gain sift src/mfas/refine/underrelax.py sift_underrelaxed
    stage 4  SCC-block <-> sift alternation src/mfas/refine/scc_recursive.py alternate_scc_sift
    oracle   exact feedforward scorer      src/mfas/metrics.py           score_from_order

Every function here mirrors the production logic exactly (same tie-breaks, same
schedules, same constants), so on a small graph the outputs are bit-identical to the
production functions -- ``docs/h64_minimal_check.py`` asserts that on the mouse graph.
The one intentional difference: stage 3 uses a per-node O(n) difference-array loop instead of the
vectorized O(m log m) event-sort kernel of ``insertion.py``. Same math; O(n^2) per sweep, unusable on 136k nodes.

Run on the mouse graph (148 nodes, a few seconds):

    $PY docs/h64_minimal.py --dataset mouse --epochs 2000

Run on the fly connectome with the champion's constants (~20-40 min on a laptop GPU
for stage 2, then several HOURS for the Python-loop sift; use the production code
for real runs):

    $PY docs/h64_minimal.py --dataset connectome
"""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ---------------------------------------------------------------------------
# Constants of the connectome champion (H64 = H42's stages 1,3,4 + H38's surrogate)
# ---------------------------------------------------------------------------
CONFIG = dict(
    # stage 2 -- Rocket (Bader et al. 2025, Algorithm 1) with the ASYM surrogate
    epochs=20_000,          # connectome; microns 80_000; mouse 0 (mouse champion skips it)
    beta_cycles=5,          # cyclic cosine beta schedule, 5 periods
    lr=0.05,                # Adam learning rate
    lr_decay_start=0.5,     # constant LR for the first half of the epochs ...
    lr_end_factor=0.1,      # ... then exponential decay to 10% of lr at the end
    grad_clip=1.0,          # clip_grad_norm_ on the position vector
    log_interval=100,       # exact score every 100 epochs (best-by-oracle)
    asym_M=0.75,            # margin of the one-sided surrogate
    asym_T=1.5,             # temperature of its tanh tail
    # stage 3 -- under-relaxed exact-gain sift
    sift_max_sweeps=40,
    sift_k_full=6,          # first 6 sweeps: full Jacobi step (alpha = 1)
    sift_alpha=0.7,         # then under-relaxed step
    # stage 4 -- alternation of SCC-block refinement and a short sift
    alt_cycles=77,
    alt_sift_sweeps=2,
    alt_k_full=2,
    alt_alpha=0.7,
    min_block=32,
    split_fracs=(0.5, 0.382, 0.618, 0.25, 0.75),
)


# ---------------------------------------------------------------------------
# 0. The oracle: exact feedforward weight of an ordering
# ---------------------------------------------------------------------------
def score(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> float:
    """Total weight of edges (u -> v) with rank[u] < rank[v].

    ``rank[u]`` is the position of node ``u`` on the line (0 = first). Works for
    integer ranks AND for real-valued positions. Weights are summed in int64 /
    float64 (never float32: on 5.7 M integer edges float32 loses units).
    """
    w = np.asarray(w)
    w = w.astype(np.int64) if np.issubdtype(w.dtype, np.integer) else w.astype(np.float64)
    ff = rank[tgt] > rank[src]          # strict: ties count as feedback
    return float(w[ff].sum())


def pct(s: float, total: float) -> float:
    return 100.0 * s / total


# ---------------------------------------------------------------------------
# 1. Greedy-FAS warm start (Eades-Lin-Smyth / GreedyAbs peeling)
# ---------------------------------------------------------------------------
def greedy_fas_order(n: int, src: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return ``rank[u]`` in [0, n): a greedy ordering built by peeling the graph.

    Loop until every node is placed:
      * a node with no remaining OUT-weight is a sink  -> put it at the BACK;
      * a node with no remaining IN-weight  is a source -> put it at the FRONT;
      * otherwise take the node with the largest (out_w - in_w) -> FRONT.
    Removing a node subtracts its edges from the neighbours' remaining degrees.
    A lazy max-heap gives O((n + m) log n) instead of O(n^2).
    """
    src = src.astype(np.int64); tgt = tgt.astype(np.int64); w = w.astype(np.float64)
    out_w = np.zeros(n); in_w = np.zeros(n)
    np.add.at(out_w, src, w); np.add.at(in_w, tgt, w)

    # CSR adjacency: out-edges of u, in-edges of v
    o = np.argsort(src, kind="stable"); out_start = np.searchsorted(src[o], np.arange(n + 1))
    out_nbr, out_wt = tgt[o], w[o]
    i = np.argsort(tgt, kind="stable"); in_start = np.searchsorted(tgt[i], np.arange(n + 1))
    in_nbr, in_wt = src[i], w[i]

    EPS = 1e-12
    active = np.ones(n, dtype=bool)
    rank = np.empty(n, dtype=np.int64)
    front, back = 0, n - 1
    sinks = [u for u in range(n) if out_w[u] <= EPS]
    sources = [u for u in range(n) if in_w[u] <= EPS and out_w[u] > EPS]
    heap = [(-(out_w[u] - in_w[u]), u) for u in range(n) if out_w[u] > EPS and in_w[u] > EPS]
    heapq.heapify(heap)
    heap_val = (out_w - in_w).copy()     # a heap entry is valid only if it matches this

    def remove(u: int, to_front: bool) -> None:
        nonlocal front, back
        active[u] = False
        if to_front:
            rank[u] = front; front += 1
        else:
            rank[u] = back; back -= 1
        for k in range(out_start[u], out_start[u + 1]):       # u -> v : v loses in-weight
            v = int(out_nbr[k])
            if active[v]:
                in_w[v] -= out_wt[k]
                if in_w[v] <= EPS and out_w[v] > EPS:
                    sources.append(v)
                else:
                    heap_val[v] = out_w[v] - in_w[v]
                    heapq.heappush(heap, (-heap_val[v], v))
        for k in range(in_start[u], in_start[u + 1]):         # x -> u : x loses out-weight
            x = int(in_nbr[k])
            if active[x]:
                out_w[x] -= in_wt[k]
                if out_w[x] <= EPS:
                    sinks.append(x)
                else:
                    heap_val[x] = out_w[x] - in_w[x]
                    heapq.heappush(heap, (-heap_val[x], x))

    placed = 0
    while placed < n:
        progressed = False
        while sinks:
            u = sinks.pop()
            if active[u] and out_w[u] <= EPS:
                remove(u, False); placed += 1; progressed = True
        while sources:
            u = sources.pop()
            if active[u] and in_w[u] <= EPS and out_w[u] > EPS:
                remove(u, True); placed += 1; progressed = True
        if placed >= n:
            break
        if progressed:
            continue
        u = -1
        while heap:                                   # pop until a non-stale, active entry
            neg, cand = heapq.heappop(heap)
            if active[cand] and -neg == heap_val[cand]:
                u = cand; break
        if u == -1:                                   # numerical fallback, never hit in practice
            rem = np.nonzero(active)[0]
            u = int(rem[int(np.argmax(out_w[rem] - in_w[rem]))])
        remove(u, True); placed += 1
    return rank


def positions_from_rank(rank: np.ndarray) -> np.ndarray:
    """Map ranks 0..n-1 to evenly spaced float positions in [-1, 1]."""
    n = rank.shape[0]
    return ((rank.astype(np.float32) / max(n - 1, 1)) * 2.0 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Rocket with the one-sided asymmetric surrogate (H38 inside H64)
# ---------------------------------------------------------------------------
def asym_surrogate(z: torch.Tensor, M: float, T: float) -> torch.Tensor:
    """g(z) = 1 for z >= M ; 1 + tanh((z - M)/T) for z < M.  z = beta * (pos[v] - pos[u]).

    Flat (zero gradient) once an edge is feedforward by a margin M; tanh-shaped
    below it, so the whole gradient budget is spent on violated / marginal edges.
    """
    below = 1.0 + torch.tanh((z - M) / T)
    return torch.where(z >= M, torch.ones_like(z), below)


def beta_schedule(epochs: int, cycles: int) -> np.ndarray:
    """Cyclic cosine sharpness: beta in [0.05, 1.05], ``cycles`` full periods."""
    return (np.cos(np.linspace(0, 2 * cycles * np.pi, epochs)) + 1.1) / 2


def rocket_asym(n, src, tgt, w, init_pos: np.ndarray, cfg: dict, seed: int, device,
                verbose: bool = True):
    """Continuous optimisation of node positions (Algorithm 1 of Bader et al. 2025).

    maximise  F(p) = sum_e  w_hat_e * g(beta * (p[tgt_e] - p[src_e]))     w_hat = w / max(w)
    with Adam, gradient clipping, the cyclic beta schedule and a constant->exponential
    LR schedule. The exact score is evaluated every ``log_interval`` epochs and the
    best position vector ever seen is what is returned (best-by-oracle).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
    w_hat = torch.tensor((w / w.max()).astype(np.float32), device=device)

    positions = torch.nn.Parameter(torch.tensor(init_pos, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([positions], lr=cfg["lr"])
    E = cfg["epochs"]
    milestone = int(E * cfg["lr_decay_start"])
    gamma = cfg["lr_end_factor"] ** (1.0 / max(E - milestone, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=milestone),
         torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)],
        milestones=[milestone])
    betas = beta_schedule(E, cfg["beta_cycles"])

    def exact(p):                                   # oracle, always on CPU
        return score(p.detach().cpu().numpy(), src, tgt, w)

    best_score = exact(positions); best_pos = positions.detach().clone()
    t0 = time.time()
    for i in range(E):
        beta = float(betas[i])
        opt.zero_grad()
        delta = positions[tgt_t] - positions[src_t]                  # > 0  <=> feedforward
        loss = -(asym_surrogate(beta * delta, cfg["asym_M"], cfg["asym_T"]) * w_hat).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg["grad_clip"])
        opt.step(); sched.step()
        if i % cfg["log_interval"] == 0 or i == E - 1:
            s = exact(positions)
            if s > best_score:
                best_score, best_pos = s, positions.detach().clone()
            if verbose and i % (cfg["log_interval"] * 20) == 0:
                print(f"  epoch {i:6d}  beta={beta:.3f}  best={best_score:,.0f}  "
                      f"{time.time() - t0:6.1f}s")
    return best_pos.cpu().numpy(), best_score


# ---------------------------------------------------------------------------
# 3. Exact-gain single-node re-insertion ("sift"), under-relaxed
# ---------------------------------------------------------------------------
def build_adj(n, src, tgt, w):
    """CSR adjacency with self-loops removed (they never change under any move)."""
    keep = src != tgt
    src, tgt, w = src[keep].astype(np.int64), tgt[keep].astype(np.int64), w[keep].astype(np.float64)
    o = np.argsort(src, kind="stable"); out_start = np.searchsorted(src[o], np.arange(n + 1))
    i = np.argsort(tgt, kind="stable"); in_start = np.searchsorted(tgt[i], np.arange(n + 1))
    return dict(n=n, out_start=out_start, out_nbr=tgt[o], out_w=w[o],
                in_start=in_start, in_nbr=src[i], in_w=w[i])


def best_gap_for_node(u: int, rank: np.ndarray, adj: dict):
    """Exact best re-insertion point of ONE node, holding every other node fixed.

    Remove u from the line; the others keep their order and get "reduced ranks"
    q(x) = rank[x] if rank[x] < rank[u] else rank[x] - 1.  Re-insert u at gap g
    (g = 0: before everyone, g = n-1: after everyone). The feedforward weight of
    u's own edges as a function of g is piecewise constant:
        out-edge u->v is feedforward iff g <= q(v)   (u before v)
        in-edge  x->u is feedforward iff g >  q(x)   (u after x)
    Build it with a difference array and take the argmax. The plateau tie-break is
    "smallest g" -- the same rule the production kernel uses (tie_break='first').
    Returns (best_gap, gain) with gain = value(best_gap) - value(current gap) >= 0.
    """
    n = adj["n"]; p = int(rank[u])
    D = np.zeros(n + 1)
    for k in range(adj["out_start"][u], adj["out_start"][u + 1]):
        v = int(adj["out_nbr"][k]); qv = rank[v] if rank[v] < p else rank[v] - 1
        D[0] += adj["out_w"][k]; D[qv + 1] -= adj["out_w"][k]     # counts for g in [0, qv]
    for k in range(adj["in_start"][u], adj["in_start"][u + 1]):
        x = int(adj["in_nbr"][k]); qx = rank[x] if rank[x] < p else rank[x] - 1
        D[qx + 1] += adj["in_w"][k]                                # counts for g in [qx+1, n-1]
    total = np.cumsum(D[:n])                                       # value at every gap
    g = int(np.argmax(total))                                      # first maximiser
    return g, float(total[g] - total[p])


def sift_underrelaxed(n, src, tgt, w, rank0, *, max_sweeps, k_full, alpha, verbose=True):
    """Jacobi sift with under-relaxation, best-by-oracle.

    Each sweep computes every node's best gap from the SAME fixed order (Jacobi),
    then rebuilds the order from sort keys:
        non-mover : key = rank
        mover     : key = rank + a * ((best_gap - 0.5) - rank)
    with a = 1 for the first ``k_full`` sweeps (full step) and a = ``alpha`` after.
    The fractional step damps the period-2 oscillation that the full Jacobi step
    falls into on the large connectomes. The working order always advances; the
    returned order is the best one the exact scorer has ever seen.
    """
    adj = build_adj(n, src, tgt, w)
    work = rank0.astype(np.int64).copy()
    best_rank, best_score = work.copy(), score(work, src, tgt, w)
    for s in range(max_sweeps):
        a = 1.0 if s < k_full else alpha
        gaps = np.empty(n, dtype=np.int64); gain = np.empty(n)
        for u in range(n):
            gaps[u], gain[u] = best_gap_for_node(u, work, adj)
        movers = gain > 1e-9
        key = work.astype(np.float64)
        key[movers] = work[movers] + a * ((gaps[movers] - 0.5) - work[movers])
        cand = np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)
        cs = score(cand, src, tgt, w)
        work = cand
        if cs > best_score:
            best_rank, best_score = cand.copy(), cs
        if verbose:
            print(f"  sweep {s:2d} a={a:.1f} movers={int(movers.sum()):6d} "
                  f"cand={cs:,.0f} best={best_score:,.0f}")
        if not movers.any():
            break
    return best_rank, best_score


# ---------------------------------------------------------------------------
# 4. Recursive SCC-topological block refinement, alternated with a short sift
# ---------------------------------------------------------------------------
def topo_order(n_lab, ls, lt, key):
    """Kahn's topological order of the condensation DAG; ties -> smallest ``key``."""
    if ls.size:
        pair = np.unique(ls.astype(np.int64) * n_lab + lt.astype(np.int64))
        us, vs = pair // n_lab, pair % n_lab
    else:
        us = vs = np.empty(0, dtype=np.int64)
    indeg = np.zeros(n_lab, dtype=np.int64); np.add.at(indeg, vs, 1)
    o = np.argsort(us, kind="stable"); us, vs = us[o], vs[o]
    start = np.searchsorted(us, np.arange(n_lab + 1))
    heap = [(float(key[i]), int(i)) for i in np.flatnonzero(indeg == 0)]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap); out.append(u)
        for j in range(start[u], start[u + 1]):
            v = int(vs[j]); indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (float(key[v]), v))
    assert len(out) == n_lab, "condensation must be a DAG"
    return np.asarray(out, dtype=np.int64)


def scc_refine(n, src, tgt, rank, *, min_block, split_frac):
    """One monotone pass. Returns a new rank vector that never scores worse.

    Contiguous-block lemma: permuting the nodes inside a contiguous range of
    positions cannot flip any edge that leaves the range. So each block is an
    independent sub-problem. For a block: split its induced subgraph into strongly
    connected components, lay the SCCs out in topological order (every inter-SCC
    edge becomes feedforward), keep each SCC's internal order, and recurse into
    each SCC. A block that is ONE SCC is cut at ``split_frac`` and both halves are
    refined again (strict subgraphs usually decompose further).
    """
    keep = src != tgt
    src, tgt = src[keep].astype(np.int64), tgt[keep].astype(np.int64)
    pos = rank.astype(np.int64).copy()
    seq = np.empty(n, dtype=np.int64); seq[pos] = np.arange(n)      # seq[position] = node

    def refine(lo, hi, eidx, is_scc):
        nb = hi - lo
        if nb <= min_block or eidx.size == 0:
            return
        if is_scc:
            return split(lo, hi, eidx)
        ls_pos, lt_pos = pos[src[eidx]] - lo, pos[tgt[eidx]] - lo
        mat = coo_matrix((np.ones(eidx.size, dtype=np.int8), (ls_pos, lt_pos)), shape=(nb, nb))
        n_lab, labels = connected_components(mat, directed=True, connection="strong")
        if n_lab == 1:
            return split(lo, hi, eidx)
        ls, lt = labels[ls_pos], labels[lt_pos]
        cross = ls != lt
        lab_key = np.full(n_lab, nb, dtype=np.int64)
        np.minimum.at(lab_key, labels, np.arange(nb))               # first position of each SCC
        order_lab = topo_order(n_lab, ls[cross], lt[cross], lab_key)
        lab_rank = np.empty(n_lab, dtype=np.int64); lab_rank[order_lab] = np.arange(n_lab)
        new_local = np.argsort(lab_rank[labels], kind="stable")     # SCCs in topo order,
        seq[lo:hi] = seq[lo:hi][new_local]                           # nodes inside keep order
        pos[seq[lo:hi]] = np.arange(lo, hi)
        sizes = np.bincount(labels, minlength=n_lab)
        starts = lo + np.concatenate([[0], np.cumsum(sizes[order_lab])[:-1]])
        intra = eidx[~cross]
        if intra.size == 0:
            return
        r_of_edge = lab_rank[ls[~cross]]
        o = np.argsort(r_of_edge, kind="stable")
        bounds = np.searchsorted(r_of_edge[o], np.arange(n_lab + 1))
        for r, lab in enumerate(order_lab):
            if sizes[lab] > min_block:
                refine(int(starts[r]), int(starts[r] + sizes[lab]),
                       intra[o][bounds[r]:bounds[r + 1]], True)

    def split(lo, hi, eidx):
        nb = hi - lo
        mid = lo + max(1, min(nb - 1, int(round(nb * split_frac))))
        ps, pt = pos[src[eidx]], pos[tgt[eidx]]
        refine(lo, mid, eidx[(ps < mid) & (pt < mid)], False)
        refine(mid, hi, eidx[(ps >= mid) & (pt >= mid)], False)

    refine(0, n, np.arange(src.shape[0]), False)
    return pos


def alternate_scc_sift(n, src, tgt, w, rank0, cfg, verbose=True):
    """Stage 4: (block pass) -> (2-sweep sift) repeated; best-by-oracle over all."""
    rank = rank0.astype(np.int64).copy()
    best_rank, best_score = rank.copy(), score(rank, src, tgt, w)
    fr = cfg["split_fracs"]
    for c in range(cfg["alt_cycles"]):
        rank = scc_refine(n, src, tgt, rank, min_block=cfg["min_block"], split_frac=fr[c % len(fr)])
        s1 = score(rank, src, tgt, w)
        if s1 > best_score:
            best_rank, best_score = rank.copy(), s1
        rank, s2 = sift_underrelaxed(n, src, tgt, w, rank, max_sweeps=cfg["alt_sift_sweeps"],
                                     k_full=cfg["alt_k_full"], alpha=cfg["alt_alpha"], verbose=False)
        if s2 > best_score:
            best_rank, best_score = rank.copy(), s2
        if verbose:
            print(f"  cycle {c:3d} split={fr[c % len(fr)]:.3f} after_scc={s1:,.0f} "
                  f"after_sift={s2:,.0f} best={best_score:,.0f}")
    return best_rank, best_score


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------
def run_h64(n, src, tgt, w, cfg, seed=42, device="cpu"):
    total = float(w.sum())
    t0 = time.time()
    print("stage 1: greedy-FAS warm start")
    r1 = greedy_fas_order(n, src, tgt, w)
    s1 = score(r1, src, tgt, w); print(f"  greedy   {pct(s1, total):.4f}%  ({time.time() - t0:.1f}s)")

    print("stage 2: Rocket with the ASYM surrogate")
    pos2, s2 = rocket_asym(n, src, tgt, w, positions_from_rank(r1), cfg, seed, device)
    print(f"  pure Rocket {pct(s2, total):.4f}%  ({time.time() - t0:.1f}s)")
    r2 = np.argsort(np.argsort(pos2, kind="stable"), kind="stable").astype(np.int64)

    print("stage 3: under-relaxed exact-gain sift")
    r3, s3 = sift_underrelaxed(n, src, tgt, w, r2, max_sweeps=cfg["sift_max_sweeps"],
                               k_full=cfg["sift_k_full"], alpha=cfg["sift_alpha"])
    print(f"  after sift {pct(s3, total):.4f}%  ({time.time() - t0:.1f}s)")

    print("stage 4: SCC-block <-> sift alternation")
    r4, s4 = alternate_scc_sift(n, src, tgt, w, r3, cfg)
    print(f"  final {pct(s4, total):.4f}%  ({time.time() - t0:.1f}s)")

    best = max([(s2, r2), (s3, r3), (s4, r4)], key=lambda t: t[0])   # best-by-oracle
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mouse", choices=["mouse", "connectome", "microns"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mfas.io import load_dataset
    g = load_dataset(args.dataset)
    print(g)
    cfg = dict(CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    src, tgt, w = np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight)
    rank, s = run_h64(g.n_nodes, src, tgt, w, cfg, seed=args.seed, device=args.device)
    print(f"RESULT {args.dataset}: {s:,.4f} / {g.total_weight:,.4f} = {pct(s, g.total_weight):.6f}%")


if __name__ == "__main__":
    main()
