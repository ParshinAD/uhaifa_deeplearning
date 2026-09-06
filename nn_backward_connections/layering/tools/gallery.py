"""Representation gallery for the MFAS-ordered mouse graph.

Builds several candidate representations of the SAME (graph, champion order)
so they can be compared by eye and ranked (see layering/GALLERY_REVIEW.md).
Nothing here feeds back into any optimizer; pi is the pinned champion.

Run with a python whose BLAS works with matplotlib (base Anaconda on the
Windows box; `allen` on the MacBook):

    /c/ProgramData/anaconda3/python.exe layering/tools/gallery.py

Outputs -> layering/outputs/gallery/*.png, *.csv, *.json (the JSON feeds the
interactive HTML view).

Variants
--------
V1 arc diagram            nodes on a line in pi order; FF arcs above, FB below
V2 sorted adjacency       weight matrix in pi order; upper = FF, lower = FB
V3 soft layers, node-level  L slices of pi (exact DP), hub as its own layer,
                          barycenter within layer, node ids as labels
V4 soft layers, bundled   the same layering with edges bundled per layer pair
                          (an ANN block diagram; width = summed weight)
V5 layer-to-layer matrix  the L x L weight matrix behind V4
V6 hierarchy comparison   trophic level vs pi rank (diagnostic)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering import core, io_utils                      # noqa: E402
from layering.draw import COL_FWD, COL_BACK, COL_INTRA, COL_NODE, edge_linewidths  # noqa: E402
from layering.tools.diagnostics import sccs, trophic_levels  # noqa: E402

CONFIG = {
    "dataset": "mouse",
    "soft_L": 8,                 # content slices; the hub adds one more
    "barycenter_sweeps": 10,
    "dpi": 120,
    "arc_alpha": 0.5,
    "min_bundle_frac": 0.002,    # bundles below this share of total weight are not drawn (V4)
    "global_seed": 42,
    "out_dir": HERE.parent / "outputs" / "gallery",
}
COL_HUB = "#e6a100"
COL_SCC = "#f2e6d9"


# ---------------------------------------------------------------------------
# Soft layering (exact DP) with a forced singleton layer for the hub
# ---------------------------------------------------------------------------

def soft_slices(rank, src, tgt, w, L, singleton_ranks=()):
    """Cut the order into ``L`` contiguous slices minimising intra-slice weight.

    ``singleton_ranks``: pi-ranks that must form a slice of their own (used for
    the hub). Returns ``layer`` per node (0 = first slice) and the cut list.
    Exact O(L n^2) DP over a 2-D prefix sum of the rank-sorted weight matrix.
    """
    n = len(rank)
    M = np.zeros((n, n))
    a, b = np.minimum(rank[src], rank[tgt]), np.maximum(rank[src], rank[tgt])
    np.add.at(M, (a, b), w)
    C = M.cumsum(0).cumsum(1)
    single = set(int(r) for r in singleton_ranks)

    def intra(i, j):
        tot = C[j - 1, j - 1]
        if i > 0:
            tot -= C[i - 1, j - 1] + C[j - 1, i - 1] - C[i - 1, i - 1]
        return tot

    def allowed(i, j):           # slice covers ranks i..j-1
        for r in single:
            if i <= r < j and not (i == r and j == r + 1):
                return False
        return True

    INF = float("inf")
    dp = np.full((L + 1, n + 1), INF)
    arg = np.full((L + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for l in range(1, L + 1):
        for j in range(1, n + 1):
            best, bi = INF, -1
            for i in range(l - 1, j):
                if dp[l - 1, i] == INF or not allowed(i, j):
                    continue
                v = dp[l - 1, i] + intra(i, j)
                if v < best:
                    best, bi = v, i
            dp[l, j], arg[l, j] = best, bi
    assert dp[L, n] < INF, "no feasible slicing"
    cuts = []
    j = n
    for l in range(L, 0, -1):
        i = arg[l, j]
        cuts.append(i)
        j = i
    cuts = sorted(cuts)                       # slice starts, cuts[0] == 0
    layer_of_rank = np.zeros(n, dtype=int)
    for k, start in enumerate(cuts):
        layer_of_rank[start:] = k
    layer = layer_of_rank[rank]
    return layer, cuts, float(dp[L, n])


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_arc(g, rank, ff, hub, in_scc, out):
    """V1: arc diagram along pi."""
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(rank)
    lw = edge_linewidths(w, (0.3, 3.0))
    fig, ax = plt.subplots(figsize=(22, 8), dpi=CONFIG["dpi"])
    lo, hi = rank[in_scc].min(), rank[in_scc].max()
    ax.add_patch(Rectangle((lo - 0.5, -n / 2), hi - lo + 1, n, color=COL_SCC, zorder=0, lw=0))
    t = np.linspace(0, np.pi, 40)
    for e in range(len(src)):
        x0, x1 = rank[src[e]], rank[tgt[e]]
        c = (x0 + x1) / 2
        r = abs(x1 - x0) / 2
        sign = 1.0 if ff[e] else -1.0
        xs = c + r * np.cos(t)
        ys = sign * r * np.sin(t) * 0.9
        ax.plot(xs, ys, color=COL_FWD if ff[e] else COL_BACK, lw=lw[e],
                alpha=CONFIG["arc_alpha"] if ff[e] else 0.7, zorder=1)
    ax.scatter(rank, np.zeros(n), s=18, c=COL_NODE, zorder=3)
    ax.scatter([rank[hub]], [0], s=160, c=COL_HUB, edgecolors="black", zorder=4)
    ax.annotate(f"hub id {g.node_ids[hub]}", (rank[hub], 0), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9, color="black")
    ax.axhline(0, color="black", lw=0.5, zorder=2)
    ax.set_xlabel("position in the champion order pi")
    ax.set_ylabel("feedforward above / feedback below")
    ax.set_title("V1 - arc diagram along the MFAS order (mouse, H63). "
                 "Shaded: span of the giant SCC (102 nodes)")
    ax.legend(handles=[Line2D([], [], color=COL_FWD, lw=2, label="feedforward (pi)"),
                       Line2D([], [], color=COL_BACK, lw=2, label="feedback (pi)")],
              loc="upper right")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "V1_arc_diagram.png")
    plt.close(fig)


def view_adjacency(g, rank, hub, in_scc, cuts, out):
    """V2: weight matrix reordered by pi, log colour, layer cuts as grid."""
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(rank)
    A = np.full((n, n), np.nan)
    A[rank[src], rank[tgt]] = np.log10(w)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=CONFIG["dpi"])
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(A, cmap="viridis", interpolation="nearest")
    for c in cuts[1:]:
        ax.axhline(c - 0.5, color="white", lw=0.7, alpha=0.9)
        ax.axvline(c - 0.5, color="white", lw=0.7, alpha=0.9)
    lo, hi = rank[in_scc].min(), rank[in_scc].max()
    ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1, fill=False,
                           edgecolor=COL_BACK, lw=1.2, ls="--"))
    ax.axhline(rank[hub], color=COL_HUB, lw=0.8, alpha=0.9)
    ax.axvline(rank[hub], color=COL_HUB, lw=0.8, alpha=0.9)
    ax.plot([0, n - 1], [0, n - 1], color="grey", lw=0.5)
    ax.set_xlabel("target (pi rank)")
    ax.set_ylabel("source (pi rank)")
    ax.set_title("V2 - adjacency in the MFAS order. Above diagonal = feedforward, below = feedback.\n"
                 "White grid: soft-layer cuts (L=%d + hub); gold: hub row/col; dashed red: giant SCC"
                 % CONFIG["soft_L"])
    fig.colorbar(im, ax=ax, fraction=0.04, label="log10 weight")
    fig.tight_layout()
    fig.savefig(out / "V2_sorted_adjacency.png")
    plt.close(fig)


def _within_layer_slots(src, tgt, layer, rank):
    slot0 = core.initial_slots_from_order(layer, rank)
    mask = layer[src] != layer[tgt]
    slot, cross = core.barycenter_order(src, tgt, mask, layer, slot0, CONFIG["barycenter_sweeps"])
    return slot, cross


def view_soft_nodes(g, rank, layer, slot, hub, in_scc, out, L_label, fname="V3_soft_layers_nodes.png",
                    title_prefix="V3 - soft layering", xlabel="soft layer (contiguous slice of pi; the hub is its own layer)"):
    """V3: node-level soft layering with labels (Felleman-Van Essen style, columns)."""
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(rank)
    x = layer.astype(float)
    y = core.y_coords(layer, slot)
    cls = core.classify_by_layers(src, tgt, layer)
    lw = edge_linewidths(w, (0.3, 3.0))
    widths = np.bincount(layer)
    fig, ax = plt.subplots(figsize=(18, 11), dpi=CONFIG["dpi"])
    fwd = cls == core.FORWARD
    segs = np.stack([np.stack([x[src[fwd]], y[src[fwd]]], 1),
                     np.stack([x[tgt[fwd]], y[tgt[fwd]]], 1)], 1)
    ax.add_collection(LineCollection(segs, colors=COL_FWD, linewidths=lw[fwd], alpha=0.45, zorder=1))
    for mask, col, rad in ((cls == core.BACKWARD, COL_BACK, 0.35), (cls == core.INTRA, COL_INTRA, 0.8)):
        for e in np.where(mask)[0]:
            u, v = int(src[e]), int(tgt[e])
            ax.add_patch(FancyArrowPatch((x[u], y[u]), (x[v], y[v]),
                                         connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                         mutation_scale=7, color=col, lw=float(lw[e]),
                                         alpha=0.75, zorder=2, shrinkA=6, shrinkB=6))
    face = np.where(in_scc, COL_NODE, "#8a94a6")
    ax.scatter(x, y, s=170, c=face, edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter([x[hub]], [y[hub]], s=320, c=COL_HUB, edgecolors="black", zorder=4)
    for v in range(n):
        ax.annotate(str(g.node_ids[v]), (x[v], y[v]), fontsize=5.5, ha="center", va="center",
                    color="white" if v != hub else "black", zorder=5)
    for k, wd in enumerate(widths):
        ax.annotate(f"L{k}\n{wd} nodes", (k, widths.max() / 2 + 1.5), ha="center", fontsize=8)
    ax.set_xticks(range(len(widths)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("within-layer slot (barycenter)")
    ax.set_title(f"{title_prefix}, {L_label}: node level with ids. "
                 "Grey nodes = outside the giant SCC (pure DAG periphery); gold = hub")
    ax.legend(handles=[Line2D([], [], color=COL_FWD, lw=2, label="feedforward (by layer)"),
                       Line2D([], [], color=COL_BACK, lw=2, label="feedback (by layer)"),
                       Line2D([], [], color=COL_INTRA, lw=2, label="intra-layer (any direction)")],
              loc="upper right", fontsize=8)
    ax.margins(x=0.03, y=0.08)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)


def layer_matrix(src, tgt, w, layer):
    L = int(layer.max()) + 1
    M = np.zeros((L, L))
    np.add.at(M, (layer[src], layer[tgt]), w)
    return M


def view_bundled(g, layer, hub, out, L_label):
    """V4: ANN-style block diagram; one arrow per (layer_i -> layer_j) bundle."""
    src, tgt, w = g.src, g.tgt, g.weight
    M = layer_matrix(src, tgt, w, layer)
    L = M.shape[0]
    W = w.sum()
    widths = np.bincount(layer)
    fig, ax = plt.subplots(figsize=(18, 10), dpi=CONFIG["dpi"])
    for k in range(L):
        h = 0.25 + 0.02 * widths[k]
        col = COL_HUB if layer[hub] == k and widths[k] == 1 else "#dfe6f0"
        ax.add_patch(Rectangle((k - 0.3, -h / 2), 0.6, h, facecolor=col, edgecolor="black", lw=1, zorder=3))
        label = f"L{k}\n{widths[k]} n" if widths[k] > 1 else f"L{k}\nhub {g.node_ids[hub]}"
        ax.annotate(label, (k, 0), ha="center", va="center", fontsize=8, zorder=4)
        if M[k, k] > 0:
            ax.annotate(f"intra {100 * M[k, k] / W:.1f}%", (k, -h / 2 - 0.12), ha="center",
                        va="top", fontsize=7, color=COL_INTRA)
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
            base = 0.35 if fwd else -0.35
            ys = base + (1.0 if fwd else -1.0) * (0.25 + 0.42 * span) * np.sin(t)
            ax.plot(xs, ys, color=COL_FWD if fwd else COL_BACK, lw=lw, alpha=0.55 if fwd else 0.85,
                    solid_capstyle="butt", zorder=2 if fwd else 5)
            # arrow head at the target end
            ax.annotate("", xy=(j, base), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=COL_FWD if fwd else COL_BACK,
                                        lw=min(lw, 3), mutation_scale=12), zorder=6)
            ax.annotate(f"{100 * M[i, j] / W:.1f}", (c, ys[len(t) // 2]), fontsize=6.5,
                        ha="center", va="bottom" if fwd else "top",
                        color=COL_FWD if fwd else COL_BACK, zorder=7)
    ax.set_xlim(-0.8, L - 0.2)
    ax.set_ylim(-0.35 - 0.25 - 0.42 * (L - 1) - 0.3, 0.35 + 0.25 + 0.42 * (L - 1) + 0.3)
    ax.set_yticks([])
    ax.set_xticks(range(L))
    ax.set_xlabel("soft layer")
    ax.set_title(f"V4 - the graph as an ANN block diagram ({L_label}). Arc width ~ sqrt(bundle weight), arc height = skip length; label = % of total weight; "
                 f"bundles < {100 * CONFIG['min_bundle_frac']:.1f}% of total weight hidden.\n"
                 "Blue above = feedforward bundles; red below = feedback bundles; orange = intra-layer share")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "V4_bundled_blocks.png")
    plt.close(fig)
    return M


def view_layer_matrix(M, W, widths, out, L_label):
    """V5: heatmap of the L x L layer-to-layer weight matrix (percent of total)."""
    L = M.shape[0]
    P = 100 * M / W
    fig, ax = plt.subplots(figsize=(8, 7), dpi=CONFIG["dpi"])
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=P.max())
    for i in range(L):
        for j in range(L):
            if P[i, j] > 0:
                ax.annotate(f"{P[i, j]:.1f}", (j, i), ha="center", va="center", fontsize=7,
                            color="white" if P[i, j] > 0.6 * P.max() else "black")
    for k in range(L):
        for i in range(k):
            pass
    ax.add_patch(Rectangle((-0.5, -0.5), L, L, fill=False, lw=0))
    # mark feedback triangle
    for i in range(L):
        for j in range(i):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COL_BACK, lw=0.6))
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels([f"L{k}\n({widths[k]})" for k in range(L)], fontsize=7)
    ax.set_yticklabels([f"L{k} ({widths[k]})" for k in range(L)], fontsize=7)
    ax.set_xlabel("target layer (node count)")
    ax.set_ylabel("source layer")
    ff = np.triu(P, 1).sum()
    fb = np.tril(P, -1).sum()
    intra = np.trace(P)
    ax.set_title(f"V5 - layer-to-layer weight, % of total ({L_label})\n"
                 f"upper = feedforward {ff:.2f}%   diagonal = intra {intra:.2f}%   "
                 f"lower (red boxes) = feedback {fb:.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.045, label="% of total weight")
    fig.tight_layout()
    fig.savefig(out / "V5_layer_matrix.png")
    plt.close(fig)
    np.savetxt(out / "V5_layer_matrix_pct.csv", P, delimiter=",", fmt="%.4f")


def view_trophic(g, rank, hub, in_scc, out):
    """V6: trophic level vs pi rank."""
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(rank)
    h, F0 = trophic_levels(src, tgt, w, n, 1e-9)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=CONFIG["dpi"])
    ax.scatter(rank, h, s=30, c=np.where(in_scc, COL_NODE, "#8a94a6"), zorder=3)
    ax.scatter([rank[hub]], [h[hub]], s=140, c=COL_HUB, edgecolors="black", zorder=4)
    rho = np.corrcoef(rank, np.argsort(np.argsort(h)))[0, 1]
    ax.set_xlabel("pi rank (MFAS champion)")
    ax.set_ylabel("trophic level (MacKay-Johnson-Jones 2020)")
    ax.set_title(f"V6 - two hierarchies of the same graph: Spearman {rho:.2f}, trophic coherence F0={F0:.2f}\n"
                 "grey = outside the giant SCC; gold = hub")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "V6_trophic_vs_pi.png")
    plt.close(fig)


def core_periphery_layers(g, rank, in_scc, hub, L_core):
    """Periphery sources -> column 0, giant SCC soft-layered (hub singleton) ->
    columns 1..L_core, periphery sinks -> last column. A periphery node is
    'upstream' if it sends into the SCC (or has no in-edges), 'downstream' if
    it receives from the SCC; a node touching the SCC on both sides would be in
    the SCC, so the two are exclusive. Nodes touching only the periphery fall
    back to their pi position relative to the hub."""
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(rank)
    core_nodes = np.where(in_scc)[0]
    sub = np.isin(src, core_nodes) & np.isin(tgt, core_nodes)
    local = -np.ones(n, dtype=int)
    local[core_nodes] = np.arange(len(core_nodes))
    core_rank = np.argsort(np.argsort(rank[core_nodes]))
    lay_core, cuts, intra_w = soft_slices(core_rank, local[src[sub]], local[tgt[sub]], w[sub],
                                          L_core, singleton_ranks=[core_rank[local[hub]]])
    layer = np.zeros(n, dtype=int)
    layer[core_nodes] = lay_core + 1
    to_scc = np.zeros(n, bool); from_scc = np.zeros(n, bool)
    to_scc[src[in_scc[tgt] & ~in_scc[src]]] = True
    from_scc[tgt[in_scc[src] & ~in_scc[tgt]]] = True
    for v in np.where(~in_scc)[0]:
        if to_scc[v] and not from_scc[v]:
            layer[v] = 0
        elif from_scc[v] and not to_scc[v]:
            layer[v] = L_core + 1
        else:
            layer[v] = 0 if rank[v] < rank[hub] else L_core + 1
    return layer, intra_w


# ---------------------------------------------------------------------------

def main():
    np.random.seed(CONFIG["global_seed"])
    out = Path(CONFIG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    src, tgt, w = g.src, g.tgt, g.weight
    n = len(order)
    ff = core.ff_mask_by_order(src, tgt, rank)
    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
    hub = int(deg.argmax())
    comp = sccs(src, tgt, n)
    in_scc = comp == np.bincount(comp).argmax()

    L = CONFIG["soft_L"] + 1
    layer, cuts, intra_w = soft_slices(rank, src, tgt, w, L, singleton_ranks=[rank[hub]])
    slot, cross = _within_layer_slots(src, tgt, layer, rank)
    label = f"L={CONFIG['soft_L']}+hub"
    widths = np.bincount(layer)
    print(f"soft layering {label}: cuts={cuts} widths={widths.tolist()} "
          f"intra={100 * intra_w / w.sum():.2f}% crossings(after barycenter)={cross}")

    view_arc(g, rank, ff, hub, in_scc, out)
    view_adjacency(g, rank, hub, in_scc, cuts, out)
    view_soft_nodes(g, rank, layer, slot, hub, in_scc, out, label)
    M = view_bundled(g, layer, hub, out, label)
    view_layer_matrix(M, w.sum(), widths, out, label)
    view_trophic(g, rank, hub, in_scc, out)

    layer_cp, intra_cp = core_periphery_layers(g, rank, in_scc, hub, L)
    slot_cp, cross_cp = _within_layer_slots(src, tgt, layer_cp, rank)
    print(f"core-periphery: widths={np.bincount(layer_cp).tolist()} "
          f"intra(core)={100 * intra_cp / w.sum():.2f}% crossings={cross_cp}")
    view_soft_nodes(g, rank, layer_cp, slot_cp, hub, in_scc, out, label,
                    fname="V7_core_periphery.png", title_prefix="V7 - core-periphery",
                    xlabel="0 = periphery sources | 1..9 = giant SCC soft-layered (hub alone) | 10 = periphery sinks")

    def layout_block(lay, sl):
        yy = core.y_coords(lay, sl)
        c = core.classify_by_layers(src, tgt, lay)
        return {"layer": lay.tolist(), "slot": sl.tolist(), "y": yy.tolist(),
                "widths": np.bincount(lay).tolist(), "cls": c.tolist()}
    data = {
        "dataset": CONFIG["dataset"], "champion_ff_pct": ff_pct, "L_label": label,
        "nodes": [{"i": int(v), "id": int(g.node_ids[v]), "rank": int(rank[v]),
                   "scc": bool(in_scc[v]), "hub": bool(v == hub), "deg": int(deg[v])} for v in range(n)],
        "edges": [{"s": int(src[e]), "t": int(tgt[e]), "w": float(w[e]), "ff_pi": bool(ff[e])}
                  for e in range(len(src))],
        "layouts": {"soft": layout_block(layer, slot), "core_periphery": layout_block(layer_cp, slot_cp)},
    }
    (out / "layout.json").write_text(json.dumps(data), encoding="utf-8")
    np.savetxt(out / "soft_layer_assignment.csv",
               np.stack([g.node_ids, rank, layer, slot], 1), delimiter=",", fmt="%d",
               header="node_id,pi_rank,layer,slot", comments="")

    print("written to", out)


if __name__ == "__main__":
    main()
