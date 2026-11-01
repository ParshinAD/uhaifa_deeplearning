"""Proposal version of the fly streams figure (print-oriented restyle of fly_streams.py).

Same data, same layout logic (lanes = biological streams, columns = 16 equal-count
depth bins of the champion order), redrawn for a printed page at ~6.5 in width:

  * lane bands get a VARIABLE height ~ sqrt(lane neuron count), so the big lanes
    (central (other), medulla, medulla->lobula) are wide enough that their internal
    bundles separate visually;
  * feedback bundles are drawn near-opaque (forward stays translucent), so the
    crimson mass stands out instead of blending in;
  * every lane is annotated with the share of its OUTGOING weight that points
    backward (e.g. central (other): 17%), and a global forward/backward proportion
    bar is drawn at the top — the point of the figure is that backward is the
    small minority, and that must be readable at a glance;
  * all fonts sized for a 3.7x print shrink (24 in canvas -> 6.5 in page);
  * no internal title / legend — the document caption carries that.

Reuses assign_lanes / build_layout / lane colors from fly_streams.py; nothing here
touches the optimizer or the scorer. Numbers on the figure are recomputed from the
edge list + champion order at draw time.

Run:  $PY layering/tools/fly_streams_proposal.py
Outputs -> layering/outputs/fly_streams/S1_fly_streams_proposal_bio17_B16.png
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fly_streams import LANE_COLORS, FB_COLOR, assign_lanes, build_layout  # noqa: E402

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "fly_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "fly_ann" / "node_ann_meta.json",
    "out_dir": HERE.parent / "outputs" / "fly_streams",
    "out_name": "S1_fly_streams_proposal_bio17_B16.png",
    "n_bins": 16,
    "min_bundle_frac": 2e-4,       # ink a direction iff its weight >= this x total
    "label_top_bundles": 0,        # no % labels on bundles: at print size they are
                                   # tiny and uninformative (advisor round 3)
    "sink_hatch_frac": 0.5,
    "dpi": 150,
    "fig_w": 24.0,
    # lane band half-height (data units): base + scale*sqrt(n_lane/n_max)
    "lane_h_base": 0.30,
    "lane_h_scale": 0.60,
    "lane_gap": 0.14,
    # width / alpha laws for bundles (frac = weight / max inked weight)
    "lw_base": 0.6,
    "lw_scale": 22.0,
    "lw_exp": 0.75,
    "ff_alpha": lambda frac: 0.15 + 0.40 * np.sqrt(frac),
    "fb_alpha": lambda frac: 0.80 + 0.15 * np.sqrt(frac),
    "fs": {                        # font sizes, tuned for the 3.7x shrink
        "lane": 27, "lane_sub": 17, "lane_fb": 22, "cell": 13,
        "bundle_lbl": 17, "depth": 17, "note": 19, "bar": 20,
    },
}


def lane_geometry(lane_sizes: np.ndarray):
    """Per-lane vertical band: center y and half-height ~ sqrt(neuron count)."""
    n_max = lane_sizes.max()
    H = CONFIG["lane_h_base"] + CONFIG["lane_h_scale"] * np.sqrt(lane_sizes / n_max)
    yc = np.empty(len(H))
    cursor = 0.0
    for i, h in enumerate(H):
        yc[i] = cursor + h
        cursor += 2 * h + CONFIG["lane_gap"]
    return yc, H, cursor - CONFIG["lane_gap"]


def per_lane_fb_share(rank, lane, K, src, tgt, w):
    """Share of each lane's outgoing weight that points backward under the order."""
    ff = rank[src] < rank[tgt]
    out_tot = np.bincount(lane[src], weights=w, minlength=K)
    out_ff = np.bincount(lane[src][ff], weights=w[ff], minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 100.0 * (out_tot - out_ff) / out_tot
    return np.nan_to_num(share), out_tot


def draw(L, Wtot, ff_pct_global, lane_fb, out_png):
    K, n_bins = L["K"], L["n_bins"]
    lane_list = L["lane_list"]
    counts = L["counts"].reshape(K, n_bins)
    intra = (L["intra_ff"] + L["intra_fb"]).reshape(K, n_bins)
    sink0 = L["sink0"].reshape(K, n_bins)
    cmax = counts.max()
    intra_max = max(intra.max(), 1e-9)
    thr = CONFIG["min_bundle_frac"] * Wtot
    fs = CONFIG["fs"]

    lane_sizes = counts.sum(axis=1)
    yc, H, y_tot = lane_geometry(lane_sizes)

    x_lo, x_hi = -3.85, n_bins + 2.05
    y_top_pad, y_bot_pad = 2.4, 1.3
    fig_w = CONFIG["fig_w"]
    fig_h = fig_w * (y_tot + y_top_pad + y_bot_pad) / (x_hi - x_lo)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=CONFIG["dpi"])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_tot + y_bot_pad, -y_top_pad)

    # global forward/backward proportion bar (top): lengths are LINEAR in weight
    fb_pct_global = 100.0 - ff_pct_global
    bar_x0, bar_len, bar_h = 3.6, 10.0, 0.34
    ax.add_patch(mpatches.Rectangle((bar_x0, -2.05), bar_len * ff_pct_global / 100,
                                    bar_h, color="#6e6a62", alpha=0.85))
    ax.add_patch(mpatches.Rectangle((bar_x0, -1.45), bar_len * fb_pct_global / 100,
                                    bar_h, color=FB_COLOR, alpha=0.95))
    ax.annotate(f"forward weight  {ff_pct_global:.1f}%",
                (bar_x0 + bar_len * ff_pct_global / 100 + 0.15, -2.05 + bar_h / 2),
                fontsize=fs["bar"], va="center", color="#3d3b37")
    ax.annotate(f"backward weight  {fb_pct_global:.1f}%",
                (bar_x0 + bar_len * fb_pct_global / 100 + 0.15, -1.45 + bar_h / 2),
                fontsize=fs["bar"], va="center", color=FB_COLOR, fontweight="bold")
    ax.annotate("share of all 41.9M synapses under the best ordering:",
                (bar_x0, -2.35), fontsize=fs["bar"] - 2, color="#52514e")

    # lane bands + labels
    for i, name in enumerate(lane_list):
        ax.axhspan(yc[i] - H[i], yc[i] + H[i], color="#000000", alpha=0.030, zorder=0)
        ax.annotate(name, (-3.75, yc[i] - 0.5 * min(H[i], 0.30)), fontsize=fs["lane"],
                    va="center", ha="left", color=LANE_COLORS[name], fontweight="bold")
        n_lane = int(lane_sizes[i])
        sub = f"{n_lane / 1000:.1f}k neurons" if n_lane >= 1000 else f"{n_lane} neurons"
        ax.annotate(sub, (-3.75, yc[i] + 0.5 * min(H[i], 0.30) + 0.17),
                    fontsize=fs["lane_sub"], va="center", ha="left", color="#52514e")
        # per-lane feedback share, right margin (skip the two ~0-weight lanes)
        if lane_fb[1][i] / Wtot > 1e-3:
            ax.annotate(f"{lane_fb[0][i]:.0f}% back", (n_bins + 0.35, yc[i]),
                        fontsize=fs["lane_fb"], va="center", ha="left",
                        color=FB_COLOR, fontweight="bold")

    # cells (same sqrt-area law as the original, capped by the lane band)
    geo = {}
    for i in range(K):
        for jb in range(n_bins):
            c = counts[i, jb]
            if c == 0:
                continue
            hw = min(0.92 * H[i], max(0.06, 0.80 * np.sqrt(c / cmax)))
            ww = 0.5 * (0.18 + 0.50 * np.sqrt(c / cmax))
            geo[(i, jb)] = (jb + 0.5 - ww, jb + 0.5 + ww, hw)
            col = LANE_COLORS[lane_list[i]]
            is_sink = (sink0[i, jb] / c) > CONFIG["sink_hatch_frac"]
            blw = 3.6 * np.sqrt(intra[i, jb] / intra_max)
            ax.add_patch(FancyBboxPatch(
                (jb + 0.5 - ww, yc[i] - hw), 2 * ww, 2 * hw,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor="#c9c4ba" if is_sink else col,
                hatch="////" if is_sink else None,
                edgecolor="#6e6a62" if is_sink else col,
                linewidth=max(blw, 0.4),
                alpha=0.24 + 0.68 * np.sqrt(c / cmax), zorder=4))
            if c > 0.04 * cmax:
                ax.annotate(f"{c / 1000:.1f}k" if c >= 1000 else str(int(c)),
                            (jb + 0.5, yc[i]), fontsize=fs["cell"], ha="center",
                            va="center",
                            color="#faf9f5" if c > 0.3 * cmax else "#141413",
                            zorder=6)

    # bundles: a direction is inked iff its own weight >= thr
    inked = []
    for cs, ct, wf, wb in L["bundles"]:
        if wf >= thr:
            inked.append((cs, ct, wf, True))
        if wb >= thr:
            inked.append((cs, ct, wb, False))
    wmax = max(x[2] for x in inked)

    vert = sorted([x for x in inked if x[0] % n_bins == x[1] % n_bins],
                  key=lambda x: (x[0] % n_bins, x[0] // n_bins, x[1] // n_bins))
    vslot, vcount = {}, {}
    for x in vert:
        col_ = x[0] % n_bins
        vslot[(x[0], x[1], x[3])] = vcount.get(col_, 0)
        vcount[col_] = vcount.get(col_, 0) + 1

    def bezier(x0, y0, x1, y1, sgn):
        dx = max(abs(x1 - x0) * 0.45, 0.35)
        verts = [(x0, y0), (x0 + sgn * dx, y0), (x1 - sgn * dx, y1), (x1, y1)]
        return MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                               MplPath.CURVE4])

    # % labels go on the largest FORWARD bundles only — labels on the opaque
    # feedback arcs cluttered the middle, and the per-lane "% back" annotation
    # already carries the backward numbers.
    label_set = set(id(x) for x in
                    sorted((x for x in inked if x[3]), key=lambda x: -x[2])
                    [:CONFIG["label_top_bundles"]])
    for x in sorted(inked, key=lambda x: x[2]):
        cs, ct, weight, is_ff = x
        ls, bs = cs // n_bins, cs % n_bins
        lt, bt = ct // n_bins, ct % n_bins
        frac = weight / wmax
        lw = CONFIG["lw_base"] + CONFIG["lw_scale"] * frac ** CONFIG["lw_exp"]
        col = LANE_COLORS[lane_list[ls]] if is_ff else FB_COLOR
        al = float(CONFIG["ff_alpha"](frac) if is_ff else CONFIG["fb_alpha"](frac))
        gs, gt = geo[(ls, bs)], geo[(lt, bt)]
        ys, yt = yc[ls], yc[lt]
        if bs == bt:                                   # vertical, offset per bundle
            m_ = vcount[bs]
            xo = bs + 0.5 + (0 if m_ == 1 else
                             -0.24 + 0.48 * vslot[(cs, ct, is_ff)] / (m_ - 1))
            sgn = np.sign(yt - ys)
            pth = bezier(xo, ys + sgn * gs[2], xo, yt - sgn * gt[2], 0)
            mid = (xo, (ys + yt) / 2)
        elif ls == lt:                                 # same lane: bow inside the band
            span = abs(bt - bs)
            h = H[ls] * min(0.34 + 0.11 * span, 0.95)  # FF above / FB below spine
            ya = ys - h if is_ff else ys + h
            x0, x1 = (gs[1], gt[0]) if is_ff else (gs[0], gt[1])
            verts = [(x0, ys), (x0 + (x1 - x0) / 3, ya),
                     (x1 - (x1 - x0) / 3, ya), (x1, ys)]
            pth = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                  MplPath.CURVE4, MplPath.CURVE4])
            mid = ((x0 + x1) / 2, ya + (0.10 if not is_ff else -0.10))
        elif is_ff:                                    # rightward from cell edges
            pth = bezier(gs[1], ys, gt[0], yt, 1)
            mid = ((gs[1] + gt[0]) / 2, (ys + yt) / 2 - 0.08)
        else:                                          # leftward feedback
            pth = bezier(gs[0], ys, gt[1], yt, -1)
            mid = ((gs[0] + gt[1]) / 2, (ys + yt) / 2 + 0.12)
        ax.add_patch(mpatches.PathPatch(pth, fill=False, lw=lw, edgecolor=col,
                                        alpha=al, zorder=3 if is_ff else 5,
                                        capstyle="round"))
        if id(x) in label_set:
            ax.annotate(f"{100 * weight / Wtot:.1f}%", mid, fontsize=fs["bundle_lbl"],
                        ha="center", va="center", color=col, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="#faf9f5",
                                  ec="none", alpha=0.78))

    # depth axis
    n = int(counts.sum())
    for jb in range(0, n_bins + 1, 2):
        r = int(jb * n / n_bins)
        ax.annotate(f"{r // 1000}k", (jb + (0.5 if jb < n_bins else 0.2),
                    y_tot + 0.45), fontsize=fs["depth"], ha="center",
                    color="#6b7385")
    ax.annotate("depth in the optimized order (16 equal-count slices; "
                "tick = neuron rank at slice start)",
                (n_bins / 2, y_tot + 1.05), ha="center", fontsize=fs["note"],
                color="#52514e")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#faf9f5")
    plt.close(fig)


def main():
    t0 = time.time()
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    from layering import io_utils
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    src = g.src.astype(np.int64)
    tgt = g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    Wtot = w.sum()
    out_w = np.bincount(src, weights=w, minlength=g.n_nodes)

    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    scv, ccv = meta["super_class_values"], meta["cell_class_values"]

    lane, lane_list = assign_lanes(rank, d["super_class"], d["cell_class"], scv, ccv)
    K = len(lane_list)
    L = build_layout(rank, lane, lane_list, src, tgt, w, CONFIG["n_bins"], out_w)
    lane_fb = per_lane_fb_share(rank, lane, K, src, tgt, w)

    out_png = out / CONFIG["out_name"]
    draw(L, Wtot, ff_pct, lane_fb, out_png)
    print(f"done in {time.time() - t0:.0f}s -> {out_png}")


if __name__ == "__main__":
    main()
