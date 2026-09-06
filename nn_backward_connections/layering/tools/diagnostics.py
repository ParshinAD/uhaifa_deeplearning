"""Structural diagnostics behind layering/RESEARCH_NOTES.md.

Everything here is *analysis* of the pinned champion order; nothing feeds back
into any optimizer (hard invariant 6). Numpy only — no scipy, no torch.

Run (base Anaconda python on the Windows box, or `allen` on a healthy-BLAS box;
``np.linalg.lstsq`` needs a working LAPACK, which `allen` on this box lacks):

    /c/ProgramData/anaconda3/python.exe layering/tools/diagnostics.py [--dataset mouse]

Reports
-------
1. hub concentration      — which node carries the feedback, and how much;
2. SCC condensation       — feedback lives inside strongly connected components;
3. feedback span in pi    — local recurrence vs long-range top-down;
4. soft layering curve    — min intra-slice weight when pi is cut into L slices
                             (exact DP; the "how layered is this graph" curve);
5. trophic levels         — MacKay–Johnson–Jones (2020) continuous hierarchy
                             compared with the MFAS order.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering import io_utils  # noqa: E402

CONFIG = {
    "soft_layer_counts": (3, 5, 8, 10, 15, 20, 26, 49),
    "span_quantiles": (10, 25, 50, 75, 90),
    "top_k": 8,
    "laplacian_pin": 1e-9,
}


def sccs(src: np.ndarray, tgt: np.ndarray, n: int) -> np.ndarray:
    """Kosaraju strongly-connected components; returns component id per node."""
    adj = collections.defaultdict(list)
    radj = collections.defaultdict(list)
    for a, b in zip(src.tolist(), tgt.tolist()):
        adj[a].append(b)
        radj[b].append(a)
    seen = [False] * n
    out = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, iter(adj[s]))]
        while stack:
            v, it = stack[-1]
            for u in it:
                if not seen[u]:
                    seen[u] = True
                    stack.append((u, iter(adj[u])))
                    break
            else:
                out.append(v)
                stack.pop()
    comp = np.full(n, -1)
    c = 0
    for s in reversed(out):
        if comp[s] != -1:
            continue
        comp[s] = c
        st = [s]
        while st:
            v = st.pop()
            for u in radj[v]:
                if comp[u] == -1:
                    comp[u] = c
                    st.append(u)
        c += 1
    return comp


def soft_layering_curve(rank, src, tgt, w, layer_counts):
    """Exact DP: cut the order into L contiguous slices minimising intra-slice
    edge weight. Returns {L: intra_weight}. O(L * n^2) with a 2-D prefix sum."""
    n = len(rank)
    M = np.zeros((n, n))
    a, b = np.minimum(rank[src], rank[tgt]), np.maximum(rank[src], rank[tgt])
    np.add.at(M, (a, b), w)
    C = M.cumsum(0).cumsum(1)

    def intra(i, j):  # ranks i..j-1
        tot = C[j - 1, j - 1]
        if i > 0:
            tot -= C[i - 1, j - 1] + C[j - 1, i - 1] - C[i - 1, i - 1]
        return tot

    res = {}
    INF = float("inf")
    for L in layer_counts:
        dp = np.full((L + 1, n + 1), INF)
        dp[0, 0] = 0.0
        for l in range(1, L + 1):
            for j in range(1, n + 1):
                best = INF
                for i in range(l - 1, j):
                    v = dp[l - 1, i] + intra(i, j)
                    if v < best:
                        best = v
                dp[l, j] = best
        res[L] = float(dp[L, n])
    return res


def trophic_levels(src, tgt, w, n, pin):
    """Weighted trophic levels h (MacKay, Johnson & Jones 2020, Sci. Adv.):
    solve (diag(k_in + k_out) - A - A^T) h = k_in - k_out; and the trophic
    coherence F0 = 1 - sum w (h_t - h_s - 1)^2 / sum w."""
    A = np.zeros((n, n))
    np.add.at(A, (src, tgt), w)
    kin, kout = A.sum(0), A.sum(1)
    Lap = np.diag(kin + kout) - A - A.T
    Lap[0, 0] += pin
    h = np.linalg.lstsq(Lap, kin - kout, rcond=None)[0]
    h -= h.min()
    F0 = 1.0 - ((h[tgt] - h[src] - 1.0) ** 2 * w).sum() / w.sum()
    return h, float(F0)


def main(dataset: str) -> None:
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(dataset)
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(order)
    W = float(w.sum())
    ffm = rank[src] < rank[tgt]
    fb = ~ffm
    ids = g.node_ids
    print(f"dataset={dataset} n={n} m={len(src)} champion FF={ff_pct:.6f}%")

    # 1. hub
    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
    hub = int(deg.argmax())
    touch = (src == hub) | (tgt == hub)
    print(f"[hub] node id={ids[hub]} degree={deg[hub]} pi-rank={rank[hub]}")
    print(f"      FB edges touching hub: {(fb & touch).sum()} of {fb.sum()} "
          f"= {100 * w[fb & touch].sum() / w[fb].sum():.1f}% of FB weight")
    print(f"      FF edges touching hub: {(ffm & touch).sum()} of {ffm.sum()} "
          f"= {100 * w[ffm & touch].sum() / w[ffm].sum():.1f}% of FF weight")
    fbw_in = np.bincount(tgt[fb], weights=w[fb], minlength=n)
    fbw_out = np.bincount(src[fb], weights=w[fb], minlength=n)
    k = CONFIG["top_k"]
    print("      top FB receivers (id, FB w_in, pi-rank):",
          [(int(ids[i]), round(float(fbw_in[i]), 3), int(rank[i])) for i in np.argsort(-fbw_in)[:k]])
    print("      top FB senders   (id, FB w_out, pi-rank):",
          [(int(ids[i]), round(float(fbw_out[i]), 3), int(rank[i])) for i in np.argsort(-fbw_out)[:k]])

    # 2. SCCs
    comp = sccs(src, tgt, n)
    sizes = np.bincount(comp)
    giant = int(sizes.argmax())
    ing = comp == giant
    print(f"[scc] count={len(sizes)} sizes>1={sorted(sizes[sizes > 1].tolist(), reverse=True)}")
    print(f"      giant SCC: {ing.sum()} nodes, pi-rank range [{rank[ing].min()}, {rank[ing].max()}]; "
          f"FB edges inside it: {(fb & ing[src] & ing[tgt]).sum()} of {fb.sum()}")

    # 3. FB span
    span = rank[src] - rank[tgt]
    q = CONFIG["span_quantiles"]
    print(f"[span] FB edge span in pi, quantiles {q}: {np.percentile(span[fb], q).round(1)} (n={n})")

    # 4. soft layering
    curve = soft_layering_curve(rank, src, tgt, w, CONFIG["soft_layer_counts"])
    print("[soft] min intra-slice weight when pi is cut into L slices:")
    for L, v in curve.items():
        print(f"       L={L:2d}: {100 * v / W:6.2f}% of total weight")

    # 5. trophic levels
    h, F0 = trophic_levels(src, tgt, w, n, CONFIG["laplacian_pin"])
    ff_h = 100 * w[h[src] < h[tgt]].sum() / W
    rh = np.argsort(np.argsort(h))
    rho = float(np.corrcoef(rank, rh)[0, 1])
    print(f"[trophic] FF weight under the trophic order: {ff_h:.3f}% (champion pi: {ff_pct:.3f}%)")
    print(f"          trophic coherence F0={F0:.3f}, max level={h.max():.2f}, "
          f"spearman(pi, trophic)={rho:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mouse")
    main(ap.parse_args().dataset)
