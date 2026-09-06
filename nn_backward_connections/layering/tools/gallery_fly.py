"""Representation gallery at connectome scale (fly, 136,648 nodes / 5.66 M edges).

What survives from tools/gallery.py at this size: everything that works on
(layer, layer) or (rank-bin, rank-bin) aggregates - V2 binned adjacency, V4
bundled block diagram, V5 layer matrix - plus the structural statistics. What
does not: per-node drawing (V3/V7), the O(E^2) crossing oracle, and the exact
O(L n^2) slicing DP, which is replaced here by the same DP on a coarse grid of
``grid_bin`` consecutive ranks (exact on the grid, a lower bound on nothing:
the true optimum over all cut positions is <= the grid optimum).

Three ways of cutting the champion order into L layers are compared on the
one objective a layering has - weight left inside layers:

    equal_count   L slices with the same number of nodes
    equal_weight  L slices with the same total incident edge weight
    dp_grid       min intra-slice weight, cuts restricted to the grid

Run (base Anaconda python; needs scipy for the SCC and the trophic solve):

    /c/ProgramData/anaconda3/python.exe layering/tools/gallery_fly.py

Outputs -> layering/outputs/gallery_fly/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering import io_utils                                  # noqa: E402
from layering.draw import COL_FWD, COL_BACK, COL_INTRA         # noqa: E402

CONFIG = {
    "dataset": "connectome",
    "grid_bin": 100,                 # ranks per DP grid cell (136,648 -> 1,367 cells)
    "layer_counts": (8, 10),         # L values for the method comparison and figures
    "curve_L": (3, 5, 8, 10, 15, 20, 30, 50),
    "adjacency_bins": 400,           # V2 heatmap resolution
    "span_bins": 60,                 # V1' span spectrum resolution
    "min_bundle_frac": 0.002,
    "top_hubs": (1, 10, 100, 1000),
    "cg_tol": 1e-6,
    "cg_maxiter": 30000,
    "dpi": 120,
    "out_dir": HERE.parent / "outputs" / "gallery_fly",
}
COL_HUB = "#e6a100"


# ---------------------------------------------------------------------------
# Layer assignment methods (all return layer[v] in 0..L-1 from cut ranks)
# ---------------------------------------------------------------------------

def layer_from_cuts(rank, cuts):
    """cuts: sorted slice starts, cuts[0] == 0."""
    return (np.searchsorted(np.asarray(cuts), rank, side="right") - 1).astype(np.int64)


def cuts_equal_count(n, L):
    return [int(round(k * n / L)) for k in range(L)]


def cuts_equal_weight(rank, src, tgt, w, n, L):
    inc = np.bincount(rank[src], weights=w, minlength=n) + np.bincount(rank[tgt], weights=w, minlength=n)
    c = np.cumsum(inc)
    targets = [k * c[-1] / L for k in range(L)]
    cuts = [0] + [int(np.searchsorted(c, t)) for t in targets[1:]]
    return sorted(set(cuts))


def grid_block_matrix(rank, src, tgt, w, n, b):
    """B x B matrix of edge weight between grid cells (cell = b consecutive ranks),
    symmetrised into the upper triangle so that intra-slice weight of a
    contiguous cell range is a submatrix sum."""
    B = (n + b - 1) // b
    a = np.minimum(rank[src], rank[tgt]) // b
    c = np.maximum(rank[src], rank[tgt]) // b
    M = np.bincount(a * B + c, weights=w, minlength=B * B).reshape(B, B)
    return M, B


def cuts_dp_grid(rank, src, tgt, w, n, L, b):
    """Exact min-intra-weight slicing with cuts restricted to grid boundaries.
    Vectorised over the previous cut, O(L B^2) numpy."""
    M, B = grid_block_matrix(rank, src, tgt, w, n, b)
    C = np.zeros((B + 1, B + 1))
    C[1:, 1:] = M.cumsum(0).cumsum(1)

    def intra_vec(i_vec, j):           # slices [i, j) over cells, vector in i
        return C[j, j] - C[i_vec, j] - C[j, i_vec] + C[i_vec, i_vec]

    INF = np.inf
    dp = np.full((L + 1, B + 1), INF)
    arg = np.full((L + 1, B + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    idx = np.arange(B + 1)
    for l in range(1, L + 1):
        for j in range(l, B + 1):
            i_vec = idx[l - 1:j]
            cand = dp[l - 1, l - 1:j] + intra_vec(i_vec, j)
            k = int(np.argmin(cand))
            dp[l, j], arg[l, j] = cand[k], i_vec[k]
    cuts, j = [], B
    for l in range(L, 0, -1):
        i = arg[l, j]
        cuts.append(int(i) * b)
        j = i
    return sorted(cuts), float(dp[L, B])


def refine_cuts_exact(rank, src, tgt, w, n, cuts, max_sweeps=10):
    """Coordinate descent on the cut positions with EXACT intra-slice weight.

    Moving cut k inside (cuts[k-1], cuts[k+1]) changes only the two adjacent
    slices; for every candidate position p the intra weight of the pair is
    (edges with both ranks < p) + (edges with both ranks >= p), restricted to
    the window - two cumulative sums over ranks. Exact and O(window edges)
    per cut, so the grid DP's answer can only improve. Returns (cuts, moved)."""
    cuts = list(cuts)
    a = np.minimum(rank[src], rank[tgt]); bb = np.maximum(rank[src], rank[tgt])
    moved_total = 0
    for _ in range(max_sweeps):
        moved = 0
        for k in range(1, len(cuts)):
            lo = cuts[k - 1]; hi = cuts[k + 1] if k + 1 < len(cuts) else n
            m = (a >= lo) & (bb < hi)
            aw, bw, ww = a[m] - lo, bb[m] - lo, w[m]
            width = hi - lo
            both_below = np.cumsum(np.bincount(bw + 1, weights=ww, minlength=width + 2))[:width + 1]
            both_above = np.bincount(aw, weights=ww, minlength=width + 1)
            both_above = ww.sum() - np.cumsum(both_above) + both_above   # edges with a >= p
            f = both_below + both_above                                   # index p = 0..width
            p = int(np.argmin(f[1:width])) + 1                            # keep slices non-empty
            new = lo + p
            if new != cuts[k]:
                cuts[k] = new; moved += 1
        moved_total += moved
        if moved == 0:
            break
    return cuts, moved_total


def intra_weight(layer, src, tgt, w):
    return float(w[layer[src] == layer[tgt]].sum())


def layer_matrix(layer, src, tgt, w, L):
    return np.bincount(layer[src] * L + layer[tgt], weights=w, minlength=L * L).reshape(L, L)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_adjacency(rank, src, tgt, w, n, cuts, hub_rank, out, tag):
    B = CONFIG["adjacency_bins"]
    b = (n + B - 1) // B
    M = np.bincount((rank[src] // b) * B + rank[tgt] // b, weights=w, minlength=B * B).reshape(B, B)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=CONFIG["dpi"])
    ax.set_facecolor("#1a1a2e")
    with np.errstate(divide="ignore"):
        im = ax.imshow(np.where(M > 0, np.log10(M), np.nan), cmap="viridis", interpolation="nearest",
                       extent=(0, n, n, 0))
    for c in cuts[1:]:
        ax.axhline(c, color="white", lw=0.7, alpha=0.9)
        ax.axvline(c, color="white", lw=0.7, alpha=0.9)
    ax.axhline(hub_rank, color=COL_HUB, lw=0.6, alpha=0.8)
    ax.axvline(hub_rank, color=COL_HUB, lw=0.6, alpha=0.8)
    ax.plot([0, n], [0, n], color="grey", lw=0.5)
    ax.set_xlabel("target (pi rank)")
    ax.set_ylabel("source (pi rank)")
    ax.set_title(f"V2 fly - adjacency in the MFAS order, {B}x{B} bins of {b} ranks (log10 summed weight)\n"
                 f"above diagonal = feedforward, below = feedback; white = {tag} cuts; gold = top hub")
    fig.colorbar(im, ax=ax, fraction=0.04, label="log10 weight in bin")
    fig.tight_layout()
    fig.savefig(out / f"V2_fly_adjacency_{tag}.png")
    plt.close(fig)


def fig_layer_matrix(P, widths, out, tag):
    L = P.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=CONFIG["dpi"])
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=P.max())
    for i in range(L):
        for j in range(L):
            if P[i, j] >= 0.05:
                ax.annotate(f"{P[i, j]:.1f}", (j, i), ha="center", va="center", fontsize=7,
                            color="white" if P[i, j] > 0.6 * P.max() else "black")
            if j < i:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COL_BACK, lw=0.6))
    ax.set_xticks(range(L)); ax.set_yticks(range(L))
    ax.set_xticklabels([f"L{k}\n({widths[k] // 1000}k)" for k in range(L)], fontsize=7)
    ax.set_yticklabels([f"L{k} ({widths[k] // 1000}k)" for k in range(L)], fontsize=7)
    ax.set_xlabel("target layer (nodes)"); ax.set_ylabel("source layer")
    ff, fb, intra = np.triu(P, 1).sum(), np.tril(P, -1).sum(), np.trace(P)
    ax.set_title(f"V5 fly - layer-to-layer weight, % of total ({tag})\n"
                 f"upper = FF {ff:.2f}%  |  diagonal = intra {intra:.2f}%  |  lower (red) = FB {fb:.2f}%", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, label="% of total weight")
    fig.tight_layout()
    fig.savefig(out / f"V5_fly_layer_matrix_{tag}.png")
    plt.close(fig)
    np.savetxt(out / f"V5_fly_layer_matrix_{tag}.csv", P, delimiter=",", fmt="%.4f")


def fig_bundled(M, W, widths, out, tag):
    L = M.shape[0]
    fig, ax = plt.subplots(figsize=(18, 10), dpi=CONFIG["dpi"])
    for k in range(L):
        h = 0.25 + 0.5 * widths[k] / widths.max()
        ax.add_patch(Rectangle((k - 0.3, -h / 2), 0.6, h, facecolor="#dfe6f0", edgecolor="black", lw=1, zorder=3))
        ax.annotate(f"L{k}\n{widths[k] / 1000:.1f}k", (k, 0), ha="center", va="center", fontsize=8, zorder=4)
        ax.annotate(f"intra {100 * M[k, k] / W:.1f}%", (k, -h / 2 - 0.12), ha="center", va="top",
                    fontsize=7, color=COL_INTRA)
    maxw = M.max()
    t = np.linspace(0, np.pi, 50)
    for i in range(L):
        for j in range(L):
            if i == j or M[i, j] / W < CONFIG["min_bundle_frac"]:
                continue
            lw = 0.6 + 12.0 * (M[i, j] / maxw) ** 0.5
            fwd = j > i
            span = abs(j - i)
            c, r = (i + j) / 2.0, span / 2.0
            xs = c - r * np.cos(t) if fwd else c + r * np.cos(t)
            base = 0.4 if fwd else -0.4
            ys = base + (1 if fwd else -1) * (0.25 + 0.42 * span) * np.sin(t)
            col = COL_FWD if fwd else COL_BACK
            ax.plot(xs, ys, color=col, lw=lw, alpha=0.55 if fwd else 0.85, zorder=2 if fwd else 5)
            ax.annotate("", xy=(j, base), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=min(lw, 3), mutation_scale=12), zorder=6)
            ax.annotate(f"{100 * M[i, j] / W:.1f}", (c, ys[len(t) // 2]), fontsize=6.5, ha="center",
                        va="bottom" if fwd else "top", color=col, zorder=7)
    ax.set_xlim(-0.8, L - 0.2)
    ax.set_ylim(-0.4 - 0.25 - 0.42 * (L - 1) - 0.3, 0.4 + 0.25 + 0.42 * (L - 1) + 0.3)
    ax.set_yticks([]); ax.set_xticks(range(L)); ax.set_xlabel("layer")
    ax.set_title(f"V4 fly - the connectome as an ANN block diagram ({tag}). Arc width ~ sqrt(bundle weight), "
                 f"height = skip length; label = % of total weight; bundles < {100 * CONFIG['min_bundle_frac']:.1f}% hidden.\n"
                 "Blue above = feedforward; red below = feedback; orange = intra-layer share")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"V4_fly_bundled_{tag}.png")
    plt.close(fig)


def fig_span_spectrum(rank, src, tgt, w, n, ff, out):
    """V1' - replaces the arc diagram: weighted histogram of rank distance, FF vs FB."""
    d = (rank[tgt] - rank[src]) / n
    bins = np.linspace(-1, 1, 2 * CONFIG["span_bins"] + 1)
    hf, _ = np.histogram(d[ff], bins=bins, weights=w[ff])
    hb, _ = np.histogram(d[~ff], bins=bins, weights=w[~ff])
    W = w.sum()
    x = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(11, 5), dpi=CONFIG["dpi"])
    ax.bar(x, 100 * hf / W, width=bins[1] - bins[0], color=COL_FWD, label="feedforward")
    ax.bar(x, 100 * hb / W, width=bins[1] - bins[0], color=COL_BACK, label="feedback")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("edge span in the champion order  (rank[target] - rank[source]) / n")
    ax.set_ylabel("% of total weight")
    ax.set_title("V1' fly - span spectrum of the MFAS order: feedforward (right of 0) vs feedback (left of 0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "V1_fly_span_spectrum.png")
    plt.close(fig)
    return x, hf / W, hb / W


def fig_soft_curve(curve_fly, curve_mouse, out):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=CONFIG["dpi"])
    for name, cur, col in (("fly (grid DP bin=%d + exact refinement)" % CONFIG["grid_bin"], curve_fly, "#1f2a44"),
                           ("mouse (exact DP)", curve_mouse, COL_INTRA)):
        if cur:
            Ls = sorted(cur)
            ax.plot(Ls, [cur[L] for L in Ls], "o-", color=col, label=name)
    ax.set_xlabel("number of layers L (contiguous slices of the MFAS order)")
    ax.set_ylabel("min weight left inside layers, % of total")
    ax.set_title("How layered is each brain? Intra-layer weight vs layer count")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "soft_layering_curve_fly_vs_mouse.png")
    plt.close(fig)


# ---------------------------------------------------------------------------

def trophic_levels_sparse(src, tgt, w, n, tol):
    A = sp.coo_matrix((w, (src, tgt)), shape=(n, n)).tocsr()
    kin = np.asarray(A.sum(0)).ravel(); kout = np.asarray(A.sum(1)).ravel()
    Lap = sp.diags(kin + kout) - A - A.T
    Lap = (Lap + sp.diags(np.full(n, 1e-9))).tocsr()
    Minv = sp.diags(1.0 / (kin + kout + 1e-9))
    h, info = spla.cg(Lap, kin - kout, rtol=tol, maxiter=CONFIG["cg_maxiter"], M=Minv)
    h -= h.min()
    F0 = 1.0 - ((h[tgt] - h[src] - 1.0) ** 2 * w).sum() / w.sum()
    return h, float(F0), int(info)


def main():
    out = Path(CONFIG["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    src, tgt, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.float64)
    n = len(order); W = w.sum()
    ff = rank[src] < rank[tgt]
    report = {"dataset": CONFIG["dataset"], "n": n, "m": len(src), "champion_ff_pct": ff_pct}
    print(f"loaded {n} nodes {len(src)} edges, FF {ff_pct:.4f}%  ({time.time() - t0:.0f}s)")

    # --- structure: hubs, SCC, FB span
    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
    fbw_touch = np.bincount(src[~ff], weights=w[~ff], minlength=n) + np.bincount(tgt[~ff], weights=w[~ff], minlength=n)
    FBW = w[~ff].sum()
    top = np.argsort(-fbw_touch)
    hub = int(deg.argmax())
    report["hub"] = {"node_index": hub, "degree": int(deg[hub]), "pi_rank": int(rank[hub])}
    report["fb_weight_share_top_k_nodes_by_fb_touch"] = {
        int(k): float(100 * fbw_touch[top[:k]].sum() / FBW / 2 if k > 1 else 100 * fbw_touch[top[0]] / FBW)
        for k in CONFIG["top_hubs"]}
    # note: each FB edge is counted at both endpoints in fbw_touch; the /2 for k>1 is an
    # upper-bound-safe approximation of the share of FB weight touching the set
    ncomp, comp = csg.connected_components(sp.coo_matrix((np.ones(len(src)), (src, tgt)), shape=(n, n)),
                                           directed=True, connection="strong")
    sizes = np.bincount(comp); giant = int(sizes.argmax()); ing = comp == giant
    report["scc"] = {"count": int(ncomp), "giant_nodes": int(ing.sum()),
                     "fb_edges_inside_giant": int((~ff & ing[src] & ing[tgt]).sum()), "fb_edges": int((~ff).sum()),
                     "giant_rank_span": [int(rank[ing].min()), int(rank[ing].max())]}
    span = (rank[src] - rank[tgt])[~ff] / n
    report["fb_span_over_n_quantiles"] = {q: float(np.percentile(span, q)) for q in (10, 25, 50, 75, 90)}
    print("structure:", json.dumps({k: report[k] for k in ("hub", "fb_weight_share_top_k_nodes_by_fb_touch", "scc",
                                                            "fb_span_over_n_quantiles")}, indent=1))

    # --- method comparison
    b = CONFIG["grid_bin"]
    rows = []
    layers_best = {}
    for L in CONFIG["layer_counts"]:
        for name in ("equal_count", "equal_weight", "dp_grid"):
            t1 = time.time()
            if name == "equal_count":
                cuts = cuts_equal_count(n, L)
            elif name == "equal_weight":
                cuts = cuts_equal_weight(rank, src, tgt, w, n, L)
            else:
                cuts, _ = cuts_dp_grid(rank, src, tgt, w, n, L, b)
                grid_intra = intra_weight(layer_from_cuts(rank, cuts), src, tgt, w)
                cuts, moved = refine_cuts_exact(rank, src, tgt, w, n, cuts)
                print(f"   dp_grid L={L}: grid intra {100 * grid_intra / W:.3f}% -> refined "
                      f"{100 * intra_weight(layer_from_cuts(rank, cuts), src, tgt, w) / W:.3f}% ({moved} cut moves)")
            layer = layer_from_cuts(rank, cuts)
            widths = np.bincount(layer, minlength=L)
            M = layer_matrix(layer, src, tgt, w, L)
            fwd = layer[src] < layer[tgt]
            rows.append({"L": L, "method": name, "cuts": cuts, "widths": widths.tolist(),
                         "intra_pct": 100 * np.trace(M) / W, "ff_by_layer_pct": 100 * np.triu(M, 1).sum() / W,
                         "fb_by_layer_pct": 100 * np.tril(M, -1).sum() / W,
                         "max_width": int(widths.max()), "mean_fwd_span_layers": float((layer[tgt] - layer[src])[fwd].mean()),
                         "seconds": time.time() - t1})
            layers_best[(L, name)] = (layer, widths, M, cuts)
            print(f"L={L:2d} {name:12s} intra={rows[-1]['intra_pct']:.3f}%  ff_by_layer={rows[-1]['ff_by_layer_pct']:.3f}%  "
                  f"max_width={widths.max()}  ({rows[-1]['seconds']:.1f}s)")
    report["methods"] = rows
    with open(out / "method_comparison.csv", "w", encoding="utf-8") as f:
        f.write("L,method,intra_pct,ff_by_layer_pct,fb_by_layer_pct,max_width,mean_fwd_span_layers,cuts\n")
        for r in rows:
            f.write(f"{r['L']},{r['method']},{r['intra_pct']:.4f},{r['ff_by_layer_pct']:.4f},{r['fb_by_layer_pct']:.4f},"
                    f"{r['max_width']},{r['mean_fwd_span_layers']:.3f},\"{r['cuts']}\"\n")

    # --- soft-layering curve (grid DP) + the mouse curve from diagnostics
    curve = {}
    for L in CONFIG["curve_L"]:
        cuts, _ = cuts_dp_grid(rank, src, tgt, w, n, L, b)
        cuts, _ = refine_cuts_exact(rank, src, tgt, w, n, cuts)
        curve[L] = 100 * intra_weight(layer_from_cuts(rank, cuts), src, tgt, w) / W
        print(f"curve L={L:2d}: {curve[L]:.3f}%")
    report["soft_curve_grid_pct"] = curve
    mouse_curve = {3: 20.21, 5: 9.77, 8: 4.99, 10: 3.35, 15: 1.38, 20: 0.77, 26: 0.44, 49: 0.0}  # outputs/diagnostics_mouse.txt
    fig_soft_curve(curve, mouse_curve, out)

    # --- figures for the best method at each L
    for L in CONFIG["layer_counts"]:
        best = min((r for r in rows if r["L"] == L), key=lambda r: r["intra_pct"])["method"]
        layer, widths, M, cuts = layers_best[(L, best)]
        tag = f"L{L}_{best}"
        fig_layer_matrix(100 * M / W, widths, out, tag)
        fig_bundled(M, W, widths, out, tag)
        fig_adjacency(rank, src, tgt, w, n, cuts, rank[hub], out, tag)
        report[f"best_L{L}"] = best
    fig_span_spectrum(rank, src, tgt, w, n, ff, out)

    # --- trophic control
    t1 = time.time()
    h, F0, info = trophic_levels_sparse(src, tgt, w, n, CONFIG["cg_tol"])
    rh = np.argsort(np.argsort(h))
    rho = float(np.corrcoef(rank, rh)[0, 1])
    ff_h = float(100 * w[h[src] < h[tgt]].sum() / W)
    report["trophic"] = {"spearman_vs_pi": rho, "ff_pct_under_trophic_order": ff_h, "F0": F0,
                         "max_level": float(h.max()), "cg_info": info, "seconds": time.time() - t1}
    print("trophic:", report["trophic"])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=CONFIG["dpi"])
    hb = ax.hexbin(rank, h, gridsize=80, bins="log", cmap="Blues")
    ax.set_xlabel("pi rank (MFAS champion)"); ax.set_ylabel("trophic level")
    ax.set_title(f"V6 fly - trophic level vs MFAS rank: Spearman {rho:.2f}, F0={F0:.2f}, "
                 f"trophic order FF {ff_h:.2f}% vs champion {ff_pct:.2f}%")
    fig.colorbar(hb, ax=ax, label="log10 nodes")
    fig.tight_layout(); fig.savefig(out / "V6_fly_trophic_vs_pi.png"); plt.close(fig)

    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
