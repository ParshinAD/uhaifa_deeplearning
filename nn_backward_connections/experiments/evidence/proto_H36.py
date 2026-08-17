"""H36 prototype — recursive SCC-topological BLOCK refinement (the Vahidi route).

Mechanism
---------
Key observation: if a set of nodes occupies a CONTIGUOUS range of positions in the
line, then permuting those nodes among themselves changes only the intra-block
edges — every edge with an endpoint outside the block keeps its orientation,
because the outside endpoint is either before all block positions or after all of
them. So each contiguous block is an INDEPENDENT sub-problem.

Within a block B (with its induced subgraph G[B]):

  * decompose G[B] into strongly connected components;
  * lay the SCCs out in a topological order of the condensation DAG (ties broken by
    the current order, so we stay as close to the incoming solution as possible);
  * keep each SCC's nodes in their CURRENT relative order and recurse into it.

Every inter-SCC edge inside B becomes feedforward, and every intra-SCC edge keeps
its current orientation, so the block's contribution is NON-DECREASING. A single
SCC has no internal decomposition, so there we split the block in half by current
position and recurse — each half's induced subgraph decomposes again (it is a
strict subgraph, so strong connectivity generally breaks).

=> the whole procedure is monotone non-decreasing in the exact score, by construction.

This is the RECURSIVE form that dr_tmp/FINDINGS_underrelaxation.md E2 asks for. E2
showed the ONE-SHOT top-level condensation gains +0.00013 pp (inter-SCC weight is
only 1.50% and already 99.99% feedforward). The recursion is a different object:
it re-decomposes strict subgraphs, where strong connectivity is much weaker.

Leakage-safety
--------------
Every decision is made from the input edge arrays and the current order alone. The
frozen oracle (mfas.metrics.score_from_order) is used ONLY to score whole candidate
vectors for reporting/accept-reject. data/best_solution is never read.

Scratch prototype (gate rung 2). Not a promotable artifact.
"""
from __future__ import annotations

import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfas.io import load_dataset          # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Condensation topological order, tie-broken by the CURRENT order
# ──────────────────────────────────────────────────────────────────────────────
def topo_order_labels(n_lab: int, ls: np.ndarray, lt: np.ndarray,
                      lab_key: np.ndarray) -> np.ndarray:
    """Kahn topological order of the condensation DAG.

    Parameters
    ----------
    n_lab : number of SCC labels.
    ls, lt : label of the source / target endpoint of each INTER-label edge.
    lab_key : tie-break key per label (its minimum current position), so that
        among all valid topological orders we pick one close to the current order.

    Returns the label sequence, first to last.
    """
    if ls.size:
        pair = np.unique(ls.astype(np.int64) * n_lab + lt.astype(np.int64))
        us = (pair // n_lab).astype(np.int64)
        vs = (pair % n_lab).astype(np.int64)
    else:
        us = vs = np.empty(0, dtype=np.int64)

    indeg = np.zeros(n_lab, dtype=np.int64)
    np.add.at(indeg, vs, 1)
    # CSR-style adjacency over the deduplicated DAG edges.
    o = np.argsort(us, kind="stable")
    us_s, vs_s = us[o], vs[o]
    start = np.searchsorted(us_s, np.arange(n_lab + 1))

    heap = [(float(lab_key[i]), int(i)) for i in np.flatnonzero(indeg == 0)]
    heapq.heapify(heap)
    out = np.empty(n_lab, dtype=np.int64)
    k = 0
    while heap:
        _, u = heapq.heappop(heap)
        out[k] = u
        k += 1
        for j in range(start[u], start[u + 1]):
            v = int(vs_s[j])
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (float(lab_key[v]), v))
    if k != n_lab:                      # cannot happen: the condensation is a DAG
        raise RuntimeError(f"topological sort incomplete: {k}/{n_lab}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The recursive refiner
# ──────────────────────────────────────────────────────────────────────────────
class SccRecursiveRefiner:
    """Recursive SCC-topological block refinement over a rank vector."""

    def __init__(self, src, tgt, n, min_block=32, split_frac=0.5):
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.n = int(n)
        self.min_block = int(min_block)
        self.split_frac = float(split_frac)
        self.n_scc_calls = 0
        self.n_topo_blocks = 0
        self.n_reordered = 0            # blocks whose layout actually changed

    def run(self, rank: np.ndarray) -> np.ndarray:
        """Return a refined rank vector (never worse, by construction)."""
        rank = np.asarray(rank, dtype=np.int64).copy()
        self.seq = np.empty(self.n, dtype=np.int64)
        self.seq[rank] = np.arange(self.n, dtype=np.int64)
        self.pos = rank
        eidx = np.arange(self.src.shape[0], dtype=np.int64)
        keep = self.src != self.tgt                     # self-loops never matter
        self._refine(0, self.n, eidx[keep])
        return self.pos.copy()

    # -- internals ------------------------------------------------------------
    def _refine(self, lo: int, hi: int, eidx: np.ndarray) -> None:
        nb = hi - lo
        if nb <= self.min_block or eidx.size == 0:
            return
        es, et = self.src[eidx], self.tgt[eidx]
        ls_pos = self.pos[es] - lo                      # local positions in [0, nb)
        lt_pos = self.pos[et] - lo

        self.n_scc_calls += 1
        mat = coo_matrix(
            (np.ones(eidx.size, dtype=np.int8), (ls_pos, lt_pos)), shape=(nb, nb))
        n_lab, labels = connected_components(mat, directed=True, connection="strong")

        if n_lab > 1:
            self._apply_topo(lo, hi, eidx, labels, n_lab, ls_pos, lt_pos)
        else:
            self._split_half(lo, hi, eidx)

    def _apply_topo(self, lo, hi, eidx, labels, n_lab, ls_pos, lt_pos) -> None:
        nb = hi - lo
        lab_of_pos = labels                             # label per LOCAL position
        ls = lab_of_pos[ls_pos]
        lt = lab_of_pos[lt_pos]
        cross = ls != lt

        # tie-break key: the label's smallest current local position.
        lab_key = np.full(n_lab, nb, dtype=np.int64)
        np.minimum.at(lab_key, lab_of_pos, np.arange(nb, dtype=np.int64))

        order_lab = topo_order_labels(n_lab, ls[cross], lt[cross], lab_key)
        self.n_topo_blocks += 1

        # New layout: labels in topological order; inside a label, CURRENT order.
        # A stable sort by the label's rank in order_lab does exactly that.
        lab_rank = np.empty(n_lab, dtype=np.int64)
        lab_rank[order_lab] = np.arange(n_lab, dtype=np.int64)
        new_local = np.argsort(lab_rank[lab_of_pos], kind="stable")

        if not np.array_equal(new_local, np.arange(nb, dtype=np.int64)):
            self.n_reordered += 1
            block_seq = self.seq[lo:hi].copy()
            self.seq[lo:hi] = block_seq[new_local]
            self.pos[self.seq[lo:hi]] = np.arange(lo, hi, dtype=np.int64)

        # Recurse into every label block (now contiguous), with its INTRA edges.
        sizes = np.bincount(lab_of_pos, minlength=n_lab)
        starts = lo + np.concatenate(
            [[0], np.cumsum(sizes[order_lab])[:-1]]).astype(np.int64)
        intra = eidx[~cross]
        if intra.size:
            intra_lab = ls[~cross]
            o = np.argsort(lab_rank[intra_lab], kind="stable")
            intra_sorted = intra[o]
            bounds = np.searchsorted(lab_rank[intra_lab][o],
                                     np.arange(n_lab + 1))
            for r, lab in enumerate(order_lab):
                if sizes[lab] <= self.min_block:
                    continue
                self._refine(int(starts[r]), int(starts[r] + sizes[lab]),
                             intra_sorted[bounds[r]:bounds[r + 1]])

    def _split_half(self, lo, hi, eidx) -> None:
        nb = hi - lo
        mid = lo + max(1, min(nb - 1, int(round(nb * self.split_frac))))
        ps, pt = self.pos[self.src[eidx]], self.tgt[eidx]
        pt = self.pos[pt]
        left = (ps < mid) & (pt < mid)
        right = (ps >= mid) & (pt >= mid)
        self._refine(lo, mid, eidx[left])
        self._refine(mid, hi, eidx[right])


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def refine_rounds(g, rank0, rounds=3, min_block=32, fracs=(0.5, 0.382, 0.618),
                  verbose=True):
    """Apply the refiner for several rounds with varying split fractions."""
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    rank = np.asarray(rank0, dtype=np.int64).copy()
    s0 = score_from_order(rank, src_o, tgt_o, g.weight)
    log = [dict(round=-1, pct=pct(s0, total), wall=0.0)]
    if verbose:
        print(f"  start                 {pct(s0, total):.6f} %")
    best_rank, best = rank.copy(), s0
    for r in range(rounds):
        t0 = time.time()
        ref = SccRecursiveRefiner(src_o, tgt_o, g.n_nodes,
                                  min_block=min_block,
                                  split_frac=fracs[r % len(fracs)])
        rank = ref.run(rank)
        sc = score_from_order(rank, src_o, tgt_o, g.weight)
        wall = time.time() - t0
        log.append(dict(round=r, pct=pct(sc, total), wall=wall,
                        split_frac=fracs[r % len(fracs)],
                        n_scc_calls=ref.n_scc_calls,
                        n_topo_blocks=ref.n_topo_blocks,
                        n_reordered=ref.n_reordered))
        if verbose:
            print(f"  round {r} frac={fracs[r % len(fracs)]:.3f}   "
                  f"{pct(sc, total):.6f} %   (+{pct(sc, total) - pct(best, total):+.6f} pp, "
                  f"{wall:.1f}s, {ref.n_reordered} blocks changed)")
        if sc > best:
            best, best_rank = sc, rank.copy()
    return best_rank, float(best), float(s0), log


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}

    # ── proxy 1: mouse, from the champion pipeline's own starting point ───────
    if which in ("all", "mouse"):
        print("== mouse ==")
        g = load_dataset("mouse")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from mfas.experiments.H02 import greedy_fas_order
        from mfas.refine import sift
        order = greedy_fas_order(g)
        rank0 = np.argsort(np.argsort(order, kind="stable"), kind="stable")
        rank0 = np.empty(g.n_nodes, dtype=np.int64)
        rank0[order] = np.arange(g.n_nodes, dtype=np.int64)
        sift_rank, sift_score, _ = sift(g, rank0, max_sweeps=40)
        print(f"  champion-style sift   {pct(sift_score, g.total_weight):.6f} %")
        _, best, s0, log = refine_rounds(g, sift_rank, rounds=3, min_block=8)
        out["mouse"] = dict(sift_pct=pct(sift_score, g.total_weight),
                            refined_pct=pct(best, g.total_weight),
                            delta_pp=pct(best, g.total_weight)
                            - pct(sift_score, g.total_weight), log=log)

    # ── proxy 2: hard synthetic (carries a verified Rocket<->best gap) ────────
    if which in ("all", "synth"):
        print("== hard synthetic ==")
        from mfas.analysis.gap import make_hard_synthetic_graph
        from mfas.experiments.H02 import greedy_fas_order
        from mfas.refine import sift
        gs = make_hard_synthetic_graph(n=400, seed=42)
        g = gs[0] if isinstance(gs, tuple) else gs
        order = greedy_fas_order(g)
        rank0 = np.empty(g.n_nodes, dtype=np.int64)
        rank0[order] = np.arange(g.n_nodes, dtype=np.int64)
        sift_rank, sift_score, _ = sift(g, rank0, max_sweeps=40)
        print(f"  champion-style sift   {pct(sift_score, g.total_weight):.6f} %")
        _, best, s0, log = refine_rounds(g, sift_rank, rounds=3, min_block=8)
        out["hard_synthetic"] = dict(sift_pct=pct(sift_score, g.total_weight),
                                     refined_pct=pct(best, g.total_weight),
                                     delta_pp=pct(best, g.total_weight)
                                     - pct(sift_score, g.total_weight), log=log)

    # ── proxy 3: THE REAL TARGET, from the stored champion order (CPU only) ───
    if which in ("all", "connectome"):
        print("== connectome, from the stored H35 champion order (no GPU) ==")
        g = load_dataset("connectome")
        p = Path("results/20260809T142912Z-H35-connectome-s42-implement-8f52fb"
                 "_positions.npy")
        posv = np.load(p)
        rank0 = np.argsort(np.argsort(posv, kind="stable"),
                           kind="stable").astype(np.int64)
        _, best, s0, log = refine_rounds(g, rank0, rounds=3, min_block=32)
        out["connectome_from_champion"] = dict(
            champion_pct=pct(s0, g.total_weight),
            refined_pct=pct(best, g.total_weight),
            delta_pp=pct(best, g.total_weight) - pct(s0, g.total_weight),
            source=str(p), log=log)

    print(json.dumps(out, indent=2, default=float))
    Path("dr_tmp/proto_H36_out.json").write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
