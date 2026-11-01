"""The (group, depth) representation of the fly connectome — roadmap item 4, first cut.

GALLERY_REVIEW.md §5.3 concluded that slices of pi are the wrong axis for the fly:
even the best contiguous cut leaves 25.3 % of the weight inside layers (mouse: 3.4 %),
because 16 % of all weight lives within 1.7 % of n in the order. The representation the
fly asks for is two-dimensional — nodes grouped into communities, groups placed by
their pi position, and the layer machinery (V4 bundles, V5 matrix) computed BETWEEN
groups. This tool builds exactly that, with graph communities standing in for the
FlyWire annotations until those are joined in:

  G1  group x pi span chart   — how localised each community is along the champion order
  G2  bundled group diagram   — V4 over (group, depth): blocks at their median-pi order,
                                FF bundles above / FB below (node-level pi classification)
  G3  group weight matrix     — V5 over groups, % of total weight, intra on the diagonal

Grouping: Leiden (modularity) on the UNDIRECTED weight-collapsed graph, fixed seed —
deterministic, structure-only, never touches the oracle (invariant 6). The groups are
a stand-in with a free parameter (the resolution implied by modularity); every figure
states this. Honesty: an edge's FF/FB class always comes from the node-level champion
order, never from the group layout, so a layout choice cannot relabel feedback.

Run (allen env or base python with igraph+leidenalg):

    python layering/tools/gallery_fly_groups.py

Outputs -> layering/outputs/gallery_fly_groups/
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering import io_utils                                  # noqa: E402
from layering.draw import COL_FWD, COL_BACK, COL_INTRA         # noqa: E402

CONFIG = {
    "dataset": "connectome",
    "seed": 42,                     # Leiden seed (the only RNG in this tool)
    "leiden_iterations": -1,        # run to convergence
    "min_group_size": 500,          # communities below this pool into 'rest' (dust: most
                                    # of the 206 communities have <= 25 nodes)
    "max_groups_drawn": 20,         # cap on drawn communities even above the size floor
    "min_bundle_frac": 0.002,       # hide bundles below this share of total weight (as V4)
    "dpi": 120,
    "out_dir": HERE.parent / "outputs" / "gallery_fly_groups",
    # Context numbers quoted in figure titles, from outputs/gallery_fly/report.json:
    "intra_layer_L10_pct": 25.27,   # best contiguous slicing (dp_grid L=10)
}


def leiden_groups(src, tgt, w, n, seed, n_iterations):
    """Leiden modularity communities of the undirected weight-collapsed graph.

    Returns (membership int64[n], modularity float, n_groups int). Deterministic
    for a fixed seed. Direction is deliberately dropped: community structure is
    about who talks to whom, not which way; direction re-enters via the pi order.
    """
    import igraph as ig
    import leidenalg as la

    a = np.minimum(src, tgt)
    b = np.maximum(src, tgt)
    key = a * np.int64(n) + b
    uniq, inv = np.unique(key, return_inverse=True)
    w_und = np.bincount(inv, weights=w)
    ua = (uniq // n).astype(np.int64)
    ub = (uniq % n).astype(np.int64)

    G = ig.Graph(n=int(n), edges=np.column_stack([ua, ub]).tolist())
    part = la.find_partition(G, la.ModularityVertexPartition,
                             weights=w_und.tolist(), seed=seed,
                             n_iterations=n_iterations)
    membership = np.asarray(part.membership, dtype=np.int64)
    return membership, float(part.modularity), int(membership.max() + 1)


def pool_small_groups(membership, rank, k_draw, min_size):
    """Keep the largest communities (>= min_size, at most k_draw), pool the rest.

    Returns (grp int64[n] relabelled 0..K-1 in DEPTH order (median pi rank),
    labels list[str], K, n_kept)."""
    sizes = np.bincount(membership)
    kept = np.argsort(-sizes)[:k_draw]
    kept = kept[sizes[kept] >= min_size]
    remap = np.full(sizes.size, len(kept), dtype=np.int64)      # default -> 'rest'
    remap[kept] = np.arange(len(kept))
    grp0 = remap[membership]
    K = len(kept) + (1 if (grp0 == len(kept)).any() else 0)

    med = np.array([np.median(rank[grp0 == g]) for g in range(K)])
    depth_order = np.argsort(med, kind="stable")
    pos_of = np.empty(K, dtype=np.int64)
    pos_of[depth_order] = np.arange(K)
    grp = pos_of[grp0]

    labels = []
    for x in range(K):
        g0 = int(depth_order[x])
        labels.append("rest" if g0 == len(kept) else f"G{x}")
    return grp, labels, K, len(kept)


def group_direction_matrices(grp, src, tgt, w, ff, K):
    """W_ff[i, j], W_fb[i, j]: weight from group i to group j, split by the
    NODE-LEVEL champion-order class (ff = rank[src] < rank[tgt])."""
    pair = grp[src] * K + grp[tgt]
    W_ff = np.bincount(pair[ff], weights=w[ff], minlength=K * K).reshape(K, K)
    W_fb = np.bincount(pair[~ff], weights=w[~ff], minlength=K * K).reshape(K, K)
    return W_ff, W_fb


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_group_span(grp, rank, n, K, labels, sizes, out):
    """G1 — each group's spread along the champion order (the 'depth' axis)."""
    fig, ax = plt.subplots(figsize=(11, 0.45 * K + 2), dpi=CONFIG["dpi"])
    for x in range(K):
        r = rank[grp == x]
        p10, p25, med, p75, p90 = np.percentile(r, [10, 25, 50, 75, 90])
        ax.plot([p10, p90], [x, x], color="#9aa7b6", lw=2, zorder=2,
                solid_capstyle="butt")
        ax.add_patch(Rectangle((p25, x - 0.22), p75 - p25, 0.44,
                               facecolor=COL_FWD, alpha=0.55, edgecolor="none", zorder=3))
        ax.plot([med], [x], "o", color="#1f2a44", ms=5, zorder=4)
        ax.annotate(f"{sizes[x] / 1000:.1f}k", (n * 1.005, x), fontsize=8,
                    va="center", color="#52514e", annotation_clip=False)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, n)
    ax.set_xlabel("pi rank in the champion order (H64)")
    ax.set_ylabel("community (ordered by median pi rank)")
    ax.set_title("G1 fly — where each Leiden community sits along the MFAS order\n"
                 "bar = p10-p90 of member ranks, box = p25-p75, dot = median; "
                 "right label = nodes",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "G1_fly_groups_span.png")
    plt.close(fig)


def fig_group_bundled(W_ff, W_fb, Wtot, sizes, labels, out):
    """G2 — the V4 bundled diagram over (group, depth) instead of pi slices."""
    K = W_ff.shape[0]
    intra = np.diag(W_ff) + np.diag(W_fb)
    off_ff = W_ff - np.diag(np.diag(W_ff))
    off_fb = W_fb - np.diag(np.diag(W_fb))
    maxw = max(off_ff.max(), off_fb.max())

    fig, ax = plt.subplots(figsize=(18, 10), dpi=CONFIG["dpi"])
    for k in range(K):
        h = 0.25 + 0.5 * sizes[k] / sizes.max()
        ax.add_patch(Rectangle((k - 0.3, -h / 2), 0.6, h, facecolor="#dfe6f0",
                               edgecolor="black", lw=1, zorder=3))
        ax.annotate(f"{labels[k]}\n{sizes[k] / 1000:.1f}k", (k, 0), ha="center",
                    va="center", fontsize=8, zorder=4)
        ax.annotate(f"intra {100 * intra[k] / Wtot:.1f}%", (k, -h / 2 - 0.12),
                    ha="center", va="top", fontsize=7, color=COL_INTRA)

    t = np.linspace(0, np.pi, 50)
    for M, is_ff in ((off_ff, True), (off_fb, False)):
        for i in range(K):
            for j in range(K):
                if i == j or M[i, j] / Wtot < CONFIG["min_bundle_frac"]:
                    continue
                lw = 0.6 + 12.0 * (M[i, j] / maxw) ** 0.5
                span = abs(j - i)
                c, r = (i + j) / 2.0, span / 2.0
                # FF above, FB below — by NODE-level class, not layout direction.
                xs = c - r * np.cos(t) if i < j else c + r * np.cos(t)
                base = 0.4 if is_ff else -0.4
                sgn = 1 if is_ff else -1
                ys = base + sgn * (0.25 + 0.42 * span) * np.sin(t)
                col = COL_FWD if is_ff else COL_BACK
                ax.plot(xs, ys, color=col, lw=lw, alpha=0.55 if is_ff else 0.85,
                        zorder=2 if is_ff else 5)
                ax.annotate("", xy=(j, base), xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=min(lw, 3), mutation_scale=12), zorder=6)
                ax.annotate(f"{100 * M[i, j] / Wtot:.1f}", (c, ys[len(t) // 2]),
                            fontsize=6.5, ha="center",
                            va="bottom" if is_ff else "top", color=col, zorder=7)

    ax.set_xlim(-0.8, K - 0.2)
    ax.set_ylim(-0.4 - 0.25 - 0.42 * (K - 1) - 0.3, 0.4 + 0.25 + 0.42 * (K - 1) + 0.3)
    ax.set_yticks([])
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("community, placed by median pi rank (the depth axis)")
    ax.set_title("G2 fly — the connectome as a (group, depth) block diagram. "
                 "Groups = Leiden communities; x = median-pi order.\n"
                 "Blue above = feedforward, red below = feedback (both by the NODE-level "
                 f"champion order); arc width ~ sqrt(weight); bundles < "
                 f"{100 * CONFIG['min_bundle_frac']:.1f}% hidden; orange = intra-group share")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "G2_fly_groups_bundled.png")
    plt.close(fig)


def fig_group_matrix(P, sizes, labels, out):
    """G3 — the V5 matrix over groups (% of total weight), intra on the diagonal."""
    K = P.shape[0]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=CONFIG["dpi"])
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=P.max())
    for i in range(K):
        for j in range(K):
            if P[i, j] >= 0.05:
                ax.annotate(f"{P[i, j]:.1f}", (j, i), ha="center", va="center",
                            fontsize=6.5,
                            color="white" if P[i, j] > 0.6 * P.max() else "black")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=COL_INTRA, lw=1.0))
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels([f"{labels[k]} ({sizes[k] // 1000}k)" for k in range(K)],
                       fontsize=7)
    ax.set_xlabel("target community"); ax.set_ylabel("source community")
    intra = np.trace(P)
    ax.set_title("G3 fly — community-to-community weight, % of total "
                 "(communities in median-pi order)\n"
                 f"diagonal (orange boxes) = intra-group {intra:.1f}%  vs  best "
                 f"contiguous L=10 slicing {CONFIG['intra_layer_L10_pct']:.1f}%",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, label="% of total weight")
    fig.tight_layout()
    fig.savefig(out / "G3_fly_groups_matrix.png")
    plt.close(fig)
    np.savetxt(out / "G3_fly_groups_matrix_pct.csv", P, delimiter=",", fmt="%.4f")


# ---------------------------------------------------------------------------

def main():
    out = Path(CONFIG["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    src = g.src.astype(np.int64); tgt = g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    n = len(order); Wtot = w.sum()
    ff = rank[src] < rank[tgt]
    print(f"loaded {n} nodes {len(src)} edges, FF {ff_pct:.4f}%  ({time.time() - t0:.0f}s)")

    cache = out / "leiden_membership.npz"
    t1 = time.time()
    if cache.exists():
        with np.load(cache) as npz:
            assert int(npz["seed"]) == CONFIG["seed"] and npz["membership"].shape[0] == n
            membership = npz["membership"].astype(np.int64)
            modularity = float(npz["modularity"])
        n_comm = int(membership.max() + 1)
        print("loaded cached Leiden membership:", cache.name)
    else:
        membership, modularity, n_comm = leiden_groups(
            src, tgt, w, n, CONFIG["seed"], CONFIG["leiden_iterations"])
        np.savez_compressed(cache, membership=membership, modularity=modularity,
                            seed=CONFIG["seed"])
    t_leiden = time.time() - t1
    full_intra = float(w[membership[src] == membership[tgt]].sum())
    print(f"leiden: {n_comm} communities, modularity {modularity:.4f}, "
          f"intra-community weight {100 * full_intra / Wtot:.2f}%  ({t_leiden:.0f}s)")

    grp, labels, K, n_kept = pool_small_groups(membership, rank,
                                               CONFIG["max_groups_drawn"],
                                               CONFIG["min_group_size"])
    sizes = np.bincount(grp, minlength=K)
    coverage = 100.0 * (1.0 - (sizes[np.array(labels) == "rest"].sum() / n if "rest" in labels else 0.0))
    print(f"drawn: {n_kept} largest communities + "
          f"{'a rest pool' if 'rest' in labels else 'no rest'} = {K} groups, "
          f"covering {coverage:.1f}% of nodes")

    W_ff, W_fb = group_direction_matrices(grp, src, tgt, w, ff, K)
    P = 100.0 * (W_ff + W_fb) / Wtot
    drawn_intra = float(np.trace(W_ff) + np.trace(W_fb))

    # Depth-chain statistics over the DRAWN grouping (inter-group weight only).
    iu = np.triu_indices(K, 1)
    inter = (W_ff + W_fb).sum() - drawn_intra
    rightward = float((W_ff + W_fb)[iu].sum())        # earlier-placed -> later-placed
    ff_inter = float(W_ff.sum() - np.trace(W_ff))
    intra_ff = float(np.trace(W_ff))
    report = {
        "dataset": CONFIG["dataset"], "n": n, "m": int(len(src)),
        "champion_ff_pct": ff_pct, "seed": CONFIG["seed"],
        "leiden": {"n_communities": n_comm, "modularity": modularity,
                   "intra_community_pct_full": 100 * full_intra / Wtot,
                   "seconds": t_leiden},
        "drawn_grouping": {
            "k_largest": n_kept, "groups_drawn": K, "node_coverage_pct": coverage,
            "sizes": sizes.tolist(), "labels": labels,
            "intra_group_pct": 100 * drawn_intra / Wtot,
            "intra_group_ff_node_level_pct_of_intra": 100 * intra_ff / drawn_intra,
            "intra_layer_L10_pct_for_comparison": CONFIG["intra_layer_L10_pct"],
        },
        "inter_group": {
            "share_of_total_pct": 100 * inter / Wtot,
            "rightward_in_depth_order_pct_of_inter": 100 * rightward / inter,
            "ff_node_level_pct_of_inter": 100 * ff_inter / inter,
        },
        "group_pi_quantiles": {
            labels[x]: [int(v) for v in
                        np.percentile(rank[grp == x], [10, 25, 50, 75, 90])]
            for x in range(K)},
    }

    fig_group_span(grp, rank, n, K, labels, sizes, out)
    fig_group_bundled(W_ff, W_fb, Wtot, sizes, labels, out)
    fig_group_matrix(P, sizes, labels, out)

    (out / "report_groups.json").write_text(json.dumps(report, indent=1),
                                            encoding="utf-8")
    print(json.dumps(report["drawn_grouping"] | report["inter_group"], indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
