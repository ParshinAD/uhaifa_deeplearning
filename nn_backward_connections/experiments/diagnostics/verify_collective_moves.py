"""Independent brute-force cross-check of the S1/S2 collective-move gain algebra.

Written by the critic while red-teaming `experiments/size_collective_moves.py`
(see the 2026-08-09 entry in `experiments/log.md`) and promoted here because the
sizing verdict cites its numbers.

Read-only w.r.t. the probe: it imports `experiments/size_collective_moves.py` and
compares that module's closed-form exact-gain formulas against explicit **full-graph**
rescoring by the FROZEN oracle on small random graphs. Nothing here is imported by
the probe, and nothing here writes to `results/`.

What it proves
--------------
* **S1** — the prefix-sum gain of the paired move `[n_1..n_r, u, v, n_{r+1}..n_t]`
  equals the true oracle delta for **every** split `r`. This is also the empirical
  proof of the contiguous-interval lemma the probe relies on.
* **small-SCC DP** — the bitmask DP returns the true optimum (vs `itertools.permutations`)
  and the sequence it returns actually realizes the value it reports.
* **S2** — one block-SCC pass's claimed gain equals the oracle delta, across block
  sizes and grid offsets (including offsets the probe never uses); the result is
  always a valid permutation, the gain is never negative, and it never exceeds the
  reported intra-block feedback ceiling.

Usage:
    /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python \
        experiments/diagnostics/verify_collective_moves.py
"""
import itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mfas.metrics import score_from_positions
import size_collective_moves as S


def rand_graph(rng, n, m, intw=True, maxw=9):
    src = rng.integers(0, n, m)
    tgt = rng.integers(0, n, m)
    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    w = (rng.integers(1, maxw + 1, src.size) if intw
         else rng.random(src.size) * 3.0)
    return src.astype(np.int64), tgt.astype(np.int64), w


# ---------------------------------------------------------------- S1 gain curve
def check_s1(trials=400, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    n_checked = 0
    for t in range(trials):
        n = int(rng.integers(5, 14))
        m = int(rng.integers(n, 5 * n))
        intw = bool(rng.integers(0, 2))
        src, tgt, w = rand_graph(rng, n, m, intw=intw)
        if src.size == 0:
            continue
        rank = rng.permutation(n).astype(np.int64)
        seq = np.argsort(rank, kind="stable")
        out_adj, in_adj = S.build_adjacency(src, tgt, w, n)
        base = score_from_positions(rank, src, tgt, w)
        bw = np.flatnonzero(rank[src] > rank[tgt])
        for e in bw:
            u, v = int(src[e]), int(tgt[e])
            lo, hi = int(rank[v]), int(rank[u])
            got = S._pair_gain_curve(u, v, lo, hi, out_adj, in_adj, rank)
            t_ = hi - lo - 1
            assert got.size == t_ + 1, (got.size, t_)
            mid = seq[lo + 1:hi]
            for r in range(t_ + 1):
                new_seq = np.concatenate((mid[:r], np.array([u, v]), mid[r:]))
                nr = rank.copy()
                nr[new_seq] = np.arange(lo, hi + 1)
                # explicit FULL-graph rescore by the frozen oracle
                truth = score_from_positions(nr, src, tgt, w) - base
                worst = max(worst, abs(truth - got[r]))
                n_checked += 1
    return worst, n_checked


# ------------------------------------------------- bitmask DP vs brute force
def check_dp(trials=300, seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(trials):
        k = int(rng.integers(2, 8))
        W = (rng.random((k, k)) * 5).round(3)
        W[rng.random((k, k)) < 0.4] = 0.0
        np.fill_diagonal(W, 0.0)
        seq, best = S._exact_small_scc_order(W)
        # brute force over all permutations
        bf = -np.inf
        for p in itertools.permutations(range(k)):
            posn = np.empty(k, dtype=int)
            for slot, node in enumerate(p):
                posn[node] = slot
            v = sum(W[i, j] for i in range(k) for j in range(k)
                    if W[i, j] and posn[j] > posn[i])
            bf = max(bf, v)
        # the returned sequence must realize `best`
        posn = np.empty(k, dtype=int)
        for slot, node in enumerate(seq.tolist()):
            posn[node] = slot
        realized = sum(W[i, j] for i in range(k) for j in range(k)
                       if W[i, j] and posn[j] > posn[i])
        worst = max(worst, abs(bf - best), abs(realized - best))
    return worst


# ------------------------------------------------------------- S2 block pass
def check_s2(trials=200, seed=2):
    rng = np.random.default_rng(seed)
    worst = 0.0
    n_nonzero = 0
    for t in range(trials):
        n = int(rng.integers(20, 90))
        m = int(rng.integers(2 * n, 8 * n))
        intw = bool(rng.integers(0, 2))
        src, tgt, w = rand_graph(rng, n, m, intw=intw)
        if src.size == 0:
            continue
        wf = np.asarray(w).astype(np.float64)
        rank = rng.permutation(n).astype(np.int64)
        base = score_from_positions(rank, src, tgt, w)
        for s in (4, 7, 16, 32):
            for off in (0, s // 2, s - 1):
                nr, st = S.s2_pass(n, rank, src, tgt, wf, s, off, 10)
                truth = score_from_positions(nr, src, tgt, w) - base
                worst = max(worst, abs(truth - st["total_gain_weight"]))
                if st["total_gain_weight"] > 0:
                    n_nonzero += 1
                # new_rank must still be a permutation
                assert np.array_equal(np.sort(nr), np.arange(n)), "not a permutation!"
                # gain must never be negative
                assert st["total_gain_weight"] >= -1e-12
                # ceiling must dominate the realized gain
                assert st["total_gain_weight"] <= st["intra_block_feedback_ceiling_weight"] + 1e-9
    return worst, n_nonzero


if __name__ == "__main__":
    w1, n1 = check_s1()
    print(f"S1 gain-curve  : max |brute-force - formula| = {w1:.3e}  over {n1} (edge,split) pairs")
    w2 = check_dp()
    print(f"small-SCC DP   : max |brute-force - DP|      = {w2:.3e}")
    w3, nz = check_s2()
    print(f"S2 block pass  : max |oracle delta - claim|  = {w3:.3e}  ({nz} non-trivial passes)")
