"""The MICrONS mm^3 drawn as a layered network: cortical layers x MFAS depth.

The fly's streams figure (fly_streams.py) had to ORDER its lanes by median
champion rank because the fly has no single anatomical axis. Mouse cortex does:
depth below pia. So here the lanes are fixed ANATOMICALLY — pia at the top,
white matter at the bottom, excitatory and inhibitory sub-lanes per layer band —
and the horizontal axis is the champion order. Any diagonal structure in the
picture is therefore a MEASURED relation between the optimizer's hierarchy and
real cortical depth, not a layout choice.

  * lanes   = (layer band, E/I) from microns_ann_prep.py: excitatory L2/3, L4,
              L5, L6 (blue, darker = deeper) and inhibitory L1–L2/3, L4, L5, L6
              (green, darker = deeper), plus white matter and unclassified
              (grey). L1 excitatory cells (a few hundred) are merged into
              L2/3 exc; inhibitory L1 into the upper inhibitory lane.
  * columns = contiguous equal-count bins of the champion order pi, so an
              edge's node-level FF/FB class always agrees with its bin
              direction: FF rightward-or-vertical, FB leftward-or-vertical.
  * cells   = (lane, bin) groups, area ~ neuron count; border thickness =
              intra-cell weight; hatched grey = >50 % of members have ZERO
              out-weight in this graph (unproofread truncated axons and volume-
              boundary cuts — reconstruction, not biology).
  * bundles = summed inter-cell weight split into FF and FB (node-level pi
              class). FF bundles take the SOURCE lane's colour; FB is crimson,
              same width/alpha law, drawn on top.

Outputs -> layering/outputs/microns_streams/
  M2_microns_streams_<variant>.png  static preview (geometry the artifact uses)
  layout_<variant>.json             cells + bundles + lane meta for the artifact
  lane_matrix.json                  10x10 lane weights, the un-thresholded backstop
  report_streams.json               all numbers quoted anywhere

Honesty: E/I and positions are a JOIN of public MICrONS tables (provenance in
microns_ann meta); layer bands are OUR density-calibrated depth bands, not
ground truth. Only 249 of 67,534 neurons have proofread axons, so outgoing
connectivity is systematically undersampled; every directional statement in
the figure inherits that caveat. Crimson is reserved for feedback. Nothing
here touches the optimizer or the frozen scorer.

Run:  $PY layering/tools/microns_streams.py
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
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "microns_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "microns_ann" / "node_ann_meta.json",
    "diagnostics": HERE.parent / "outputs" / "microns_ann" / "diagnostics.json",
    "out_dir": HERE.parent / "outputs" / "microns_streams",
    "n_bins_list": [8, 16, 32],
    "min_bundle_frac": 2e-4,
    "export_bundle_frac": 5e-5,
    "label_top_bundles": 10,
    "sink_hatch_frac": 0.5,
    "dpi": 110,
}

# Validated ramps (dataviz six-checks, light surface #fcfcfb):
#   E = blue ordinal (darker = deeper), I = green ordinal, FB = crimson
#   (reserved), reconstruction greys for WM / unclassified / hatched sinks.
FB_COLOR = "#c23b3b"
LANE_SPEC = [
    # (name, ei, layer codes, color)   — anatomical order, pia at the top
    ("L2/3 · exc", 1, (1, 2), "#86b6ef"),
    ("L1–L2/3 · inh", 2, (1, 2), "#57b283"),
    ("L4 · exc", 1, (3,), "#5598e7"),
    ("L4 · inh", 2, (3,), "#37996a"),
    ("L5 · exc", 1, (4,), "#2a78d6"),
    ("L5 · inh", 2, (4,), "#227a50"),
    ("L6 · exc", 1, (5,), "#184f95"),
    ("L6 · inh", 2, (5,), "#115c39"),
    ("white matter", None, (6,), "#8a8578"),
    ("unclassified", 0, None, "#b5a98c"),
]
LANE_COLORS = {name: col for name, _, _, col in LANE_SPEC}


def assign_lanes(ei: np.ndarray, layer: np.ndarray):
    """Fixed anatomical lane order (pia -> WM, E/I interleaved)."""
    n = ei.shape[0]
    lane = np.full(n, len(LANE_SPEC) - 1, dtype=np.int64)     # default: unclassified
    for li, (name, cls, layers, _c) in enumerate(LANE_SPEC):
        if name == "unclassified":
            continue
        if name == "white matter":
            m = layer == 6
        else:
            m = (ei == cls) & np.isin(layer, layers)
        lane[m] = li
    lane_list = [s[0] for s in LANE_SPEC]
    return lane, lane_list


def build_layout(rank, lane, lane_list, src, tgt, w, n_bins, out_w):
    """Aggregate cells and directed cell bundles for one bin count."""
    n = rank.shape[0]
    K = len(lane_list)
    dbin = np.minimum((rank * n_bins) // n, n_bins - 1).astype(np.int64)
    cell = lane * n_bins + dbin
    counts = np.bincount(cell, minlength=K * n_bins)
    sink0 = np.bincount(cell[out_w == 0], minlength=K * n_bins)

    ff = rank[src] < rank[tgt]
    csrc, ctgt = cell[src], cell[tgt]
    same = csrc == ctgt
    intra_ff = np.bincount(csrc[same & ff], weights=w[same & ff],
                           minlength=K * n_bins)
    intra_fb = np.bincount(csrc[same & ~ff], weights=w[same & ~ff],
                           minlength=K * n_bins)

    pair = csrc * (K * n_bins) + ctgt
    m = ~same
    w_ff = np.bincount(pair[m & ff], weights=w[m & ff], minlength=(K * n_bins) ** 2)
    w_fb = np.bincount(pair[m & ~ff], weights=w[m & ~ff], minlength=(K * n_bins) ** 2)
    nz = np.nonzero(w_ff + w_fb)[0]
    bundles = [(int(p // (K * n_bins)), int(p % (K * n_bins)),
                float(w_ff[p]), float(w_fb[p])) for p in nz]

    med_rank_lane = [float(np.median(rank[lane == i])) if (lane == i).any()
                     else float("nan") for i in range(K)]
    return {
        "lane_list": lane_list, "K": K, "n_bins": n_bins,
        "counts": counts, "intra_ff": intra_ff, "intra_fb": intra_fb,
        "sink0": sink0, "bundles": bundles, "med_rank_lane": med_rank_lane,
    }


def layout_stats(L, Wtot, thr):
    """The honesty numbers for one layout — coverage from exactly the inked set."""
    n_bins = L["n_bins"]
    intra_ff = float(L["intra_ff"].sum())
    intra_fb = float(L["intra_fb"].sum())
    b = L["bundles"]
    ffs = sum(x[2] for x in b)
    fbs = sum(x[3] for x in b)
    inked_ff = sum(x[2] for x in b if x[2] >= thr)
    inked_fb = sum(x[3] for x in b if x[3] >= thr)
    same_bin = sum(x[2] + x[3] for x in b if x[0] % n_bins == x[1] % n_bins)
    same_lane_ff = sum(x[2] for x in b if x[0] // n_bins == x[1] // n_bins)
    return {
        "cells_nonempty": int((L["counts"] > 0).sum()),
        "intra_cell_pct": 100 * (intra_ff + intra_fb) / Wtot,
        "intra_cell_ff_pct": 100 * intra_ff / Wtot,
        "intra_cell_fb_pct": 100 * intra_fb / Wtot,
        "inter_ff_pct_of_total": 100 * ffs / Wtot,
        "inter_fb_pct_of_total": 100 * fbs / Wtot,
        "same_bin_vertical_pct_of_inter": 100 * same_bin / (ffs + fbs),
        "same_lane_pct_of_inter_ff": 100 * same_lane_ff / ffs,
        "bundles_total": len(b),
        "bundles_inked_ff": sum(1 for x in b if x[2] >= thr),
        "bundles_inked_fb": sum(1 for x in b if x[3] >= thr),
        "inked_ff_pct_of_inter_ff": 100 * inked_ff / ffs,
        "inked_fb_pct_of_inter_fb": 100 * inked_fb / fbs,
    }


def draw_preview(L, Wtot, stats, ff_pct, out_png, dpi):
    K, n_bins = L["K"], L["n_bins"]
    lane_list = L["lane_list"]
    counts = L["counts"].reshape(K, n_bins)
    intra = (L["intra_ff"] + L["intra_fb"]).reshape(K, n_bins)
    sink0 = L["sink0"].reshape(K, n_bins)
    cmax = counts.max()
    intra_max = max(intra.max(), 1e-9)
    thr = CONFIG["min_bundle_frac"] * Wtot

    fig, ax = plt.subplots(figsize=(24, 11.5), dpi=dpi)
    ax.set_xlim(-3.1, n_bins + 0.6)
    ax.set_ylim(K - 0.25, -3.05)

    for i, name in enumerate(lane_list):
        ax.axhspan(i - 0.42, i + 0.42, color="#000000", alpha=0.025, zorder=0)
        n_lane = int(counts[i].sum())
        ax.annotate(name, (-3.0, i - 0.09), fontsize=10.5, va="center",
                    ha="left", color=LANE_COLORS[name], fontweight="bold")
        med = L["med_rank_lane"][i]
        ax.annotate(f"{n_lane:,} n · med rank {med / 1000:.1f}k",
                    (-3.0, i + 0.22), fontsize=7.5, va="center", ha="left",
                    color="#52514e")

    geo = {}
    for i in range(K):
        for jb in range(n_bins):
            c = counts[i, jb]
            if c == 0:
                continue
            hw = 0.5 * max(0.05, 0.72 * np.sqrt(c / cmax))
            ww = 0.5 * (0.18 + 0.50 * np.sqrt(c / cmax))
            geo[(i, jb)] = (jb + 0.5 - ww, jb + 0.5 + ww, hw)
            col = LANE_COLORS[lane_list[i]]
            is_sink = (sink0[i, jb] / c) > CONFIG["sink_hatch_frac"]
            blw = 3.6 * np.sqrt(intra[i, jb] / intra_max)
            ax.add_patch(FancyBboxPatch(
                (jb + 0.5 - ww, i - hw), 2 * ww, 2 * hw,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor="#c9c4ba" if is_sink else col,
                hatch="////" if is_sink else None,
                edgecolor="#6e6a62" if is_sink else col,
                linewidth=max(blw, 0.4),
                alpha=0.24 + 0.68 * np.sqrt(c / cmax), zorder=4))
            if c > 0.04 * cmax:
                ax.annotate(f"{c / 1000:.1f}k" if c >= 1000 else str(int(c)),
                            (jb + 0.5, i), fontsize=6.4, ha="center", va="center",
                            color="#faf9f5" if c > 0.3 * cmax else "#141413",
                            zorder=6)

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

    label_set = set(id(x) for x in
                    sorted(inked, key=lambda x: -x[2])[:CONFIG["label_top_bundles"]])
    for x in sorted(inked, key=lambda x: x[2]):
        cs, ct, weight, is_ff = x
        ls, bs = cs // n_bins, cs % n_bins
        lt, bt = ct // n_bins, ct % n_bins
        frac = weight / wmax
        lw = 0.5 + 9.0 * np.sqrt(frac)
        col = LANE_COLORS[lane_list[ls]] if is_ff else FB_COLOR
        al = 0.20 + 0.50 * np.sqrt(frac)
        gs, gt = geo[(ls, bs)], geo[(lt, bt)]
        if bs == bt:
            m_ = vcount[bs]
            xo = bs + 0.5 + (0 if m_ == 1 else
                             -0.24 + 0.48 * vslot[(cs, ct, is_ff)] / (m_ - 1))
            sgn = np.sign(lt - ls)
            pth = bezier(xo, ls + sgn * gs[2], xo, lt - sgn * gt[2], 0)
            mid = (xo, (ls + lt) / 2)
        elif ls == lt:
            span = abs(bt - bs)
            h = min(0.12 + 0.045 * span, 0.40)
            ya = ls - h if is_ff else ls + h
            x0, x1 = (gs[1], gt[0]) if is_ff else (gs[0], gt[1])
            verts = [(x0, ls), (x0 + (x1 - x0) / 3, ya),
                     (x1 - (x1 - x0) / 3, ya), (x1, ls)]
            pth = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                  MplPath.CURVE4, MplPath.CURVE4])
            mid = ((x0 + x1) / 2, ya + (0.06 if not is_ff else -0.06))
        elif is_ff:
            pth = bezier(gs[1], ls, gt[0], lt, 1)
            mid = ((gs[1] + gt[0]) / 2, (ls + lt) / 2 - 0.06)
        else:
            pth = bezier(gs[0], ls, gt[1], lt, -1)
            mid = ((gs[0] + gt[1]) / 2, (ls + lt) / 2 + 0.10)
        ax.add_patch(mpatches.PathPatch(pth, fill=False, lw=lw, edgecolor=col,
                                        alpha=al, zorder=3 if is_ff else 5,
                                        capstyle="round"))
        if id(x) in label_set:
            ax.annotate(f"{100 * weight / Wtot:.1f}%", mid, fontsize=7.2,
                        ha="center", va="center", color=col, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="#faf9f5",
                                  ec="none", alpha=0.75))

    n = int(counts.sum())
    for jb in range(0, n_bins + 1, 2):
        r = int(jb * n / n_bins)
        ax.annotate(f"{r // 1000}k", (jb + (0.5 if jb < n_bins else 0.2),
                    K - 0.05), fontsize=7.5, ha="center", color="#6b7385")
    ax.annotate("depth = champion order pi (H42), equal-count bins; tick = rank at bin start",
                (n_bins / 2, K + 0.45), ha="center", fontsize=9, color="#52514e")

    lx = n_bins - 5.4
    ax.annotate("how to read", (lx, -2.85), fontsize=8.5, fontweight="bold",
                color="#1f2a44")
    for dy, txt, col in (
            (0.30, "lanes are ANATOMICAL: pia at top, white matter at bottom (fixed, not fitted)", "#1f2a44"),
            (0.58, "blue = excitatory, green = inhibitory; darker = deeper layer band", "#2a78d6"),
            (0.86, "bundle colour = SOURCE lane (feedforward); crimson = feedback, drawn on top", FB_COLOR),
            (1.14, "cell border thickness = intra-cell weight; numbers in cells = neurons,", "#52514e"),
            (1.42, "on bundles = % of total weight; same-lane bundles bow over/under their lane", "#52514e"),
            (1.70, "hatched grey = >50% of members have no outputs in this graph", "#6e6a62"),
            (1.98, "   (unproofread truncated axons + volume-boundary cuts — reconstruction, not biology)", "#6e6a62")):
        ax.annotate(txt, (lx, -2.85 + dy), fontsize=7.6, color=col)

    ax.set_title(
        "MICrONS mouse visual cortex as a layered network — cortical layers x MFAS depth\n"
        f"67,534 neurons, 10,436,569 synaptic connections; champion feedforward {ff_pct:.2f} % (H42, oracle-verified). "
        f"Lanes: anatomical E/I (public MICrONS tables) x OUR density-calibrated depth bands below a fitted pia proxy (display annotation, not traced borders), pia at the top. "
        f"Columns: {n_bins} equal-count slices of pi.\n"
        f"Cell area ~ neurons (sqrt scale). Feedforward bundles carry their source lane's colour; feedback is crimson, "
        f"same width/alpha law (bins are pi-contiguous, so FF is always rightward-or-vertical and FB "
        f"leftward-or-vertical — no layout choice can relabel an edge).\n"
        f"Hidden inside cells: {stats['intra_cell_ff_pct']:.1f} % FF + {stats['intra_cell_fb_pct']:.1f} % FB "
        f"of total weight. Inked here: {stats['inked_ff_pct_of_inter_ff']:.0f} % of inter-cell FF and "
        f"{stats['inked_fb_pct_of_inter_fb']:.0f} % of inter-cell FB weight "
        f"(threshold {100 * CONFIG['min_bundle_frac']:.2f} % of total per direction); "
        f"same-column flows {stats['same_bin_vertical_pct_of_inter']:.1f} % of inter-cell. "
        f"Only 249/67,534 axons are proofread — outgoing weight is undersampled everywhere.",
        fontsize=10.5, loc="left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#faf9f5")
    plt.close(fig)


def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from layering import io_utils
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("microns")
    src = g.src.astype(np.int64)
    tgt = g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    Wtot = w.sum()
    out_w = np.bincount(src, weights=w, minlength=g.n_nodes)

    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"

    lane, lane_list = assign_lanes(d["ei"], d["layer"])
    K = len(lane_list)
    med_depth = [float(np.nanmedian(np.where(lane == i, d["depth_um"], np.nan)))
                 for i in range(K)]

    # the anatomical lane order vs the data-driven (median-rank) order — the
    # analogue of the fly's lane_order_check, but here the DISAGREEMENT is the
    # finding (inhibitory lanes sort far earlier than their excitatory partners)
    order_med = sorted(range(K), key=lambda i: float(np.median(rank[lane == i])))
    lane_order_check = {
        "anatomical_order": lane_list,
        "median_rank_order": [lane_list[i] for i in order_med],
        "median_ranks": {lane_list[i]: float(np.median(rank[lane == i]))
                         for i in range(K)},
        "note": "lanes are drawn in fixed anatomical order (pia->WM); the "
                "median-rank order shows where the optimizer disagrees — "
                "inhibitory lanes sort much earlier than their excitatory "
                "band partners",
    }

    variants = {}
    for n_bins in CONFIG["n_bins_list"]:
        L = build_layout(rank, lane, lane_list, src, tgt, w, n_bins, out_w)
        thr = CONFIG["min_bundle_frac"] * Wtot
        stats = layout_stats(L, Wtot, thr)
        variants[f"ana{K}_B{n_bins}"] = stats
        draw_preview(L, Wtot, stats, ff_pct,
                     out / f"M2_microns_streams_ana{K}_B{n_bins}.png",
                     CONFIG["dpi"])

        NC = K * n_bins
        tot = {k: np.zeros(NC) for k in ("out_ff", "out_fb", "in_ff", "in_fb")}
        for s, t, wf, wb in L["bundles"]:
            tot["out_ff"][s] += wf; tot["out_fb"][s] += wb
            tot["in_ff"][t] += wf; tot["in_fb"][t] += wb
        cells = []
        for c in np.nonzero(L["counts"])[0]:
            cells.append({"lane": int(c // n_bins), "bin": int(c % n_bins),
                          "count": int(L["counts"][c]),
                          "intra_ff": float(L["intra_ff"][c]),
                          "intra_fb": float(L["intra_fb"][c]),
                          "sink0": int(L["sink0"][c]),
                          "out_ff": float(tot["out_ff"][c]),
                          "out_fb": float(tot["out_fb"][c]),
                          "in_ff": float(tot["in_ff"][c]),
                          "in_fb": float(tot["in_fb"][c])})
        (out / f"layout_ana{K}_B{n_bins}.json").write_text(json.dumps({
            "variant": f"ana{K}_B{n_bins}", "champion_ff_pct": ff_pct,
            "n": int(rank.shape[0]), "m": int(src.shape[0]), "n_bins": n_bins,
            "lanes": [{"name": lane_list[i], "color": LANE_COLORS[lane_list[i]],
                       "n": int((lane == i).sum()),
                       "med_rank": L["med_rank_lane"][i],
                       "med_depth_um": med_depth[i]} for i in range(K)],
            "cells": cells,
            "ink_threshold_frac": CONFIG["min_bundle_frac"],
            "export_floor_frac": CONFIG["export_bundle_frac"],
            "bundles": [{"s": s, "t": t, "ff": round(wf, 2), "fb": round(wb, 2)}
                        for s, t, wf, wb in L["bundles"]
                        if wf + wb >= CONFIG["export_bundle_frac"] * Wtot],
            "stats": stats}), encoding="utf-8")
        print(f"B={n_bins}: intra {stats['intra_cell_pct']:.1f}%  "
              f"inked FF {stats['inked_ff_pct_of_inter_ff']:.0f}% / "
              f"FB {stats['inked_fb_pct_of_inter_fb']:.0f}%")

    # lane x lane matrix (all bins pooled) — the un-thresholded backstop
    ff_edge = rank[src] < rank[tgt]
    lpair = lane[src] * K + lane[tgt]
    lane_W = np.bincount(lpair, weights=w, minlength=K * K).reshape(K, K)
    lane_Wff = np.bincount(lpair[ff_edge], weights=w[ff_edge],
                           minlength=K * K).reshape(K, K)
    (out / "lane_matrix.json").write_text(json.dumps({
        "lane_list": lane_list,
        "W_pct": (100 * lane_W / Wtot).tolist(),
        "W_ff_pct": (100 * lane_Wff / Wtot).tolist(),
        "note": "percent of total weight, source lane = row; FF by node-level "
                "champion order; diagonal = intra-lane"}), encoding="utf-8")

    diag = json.loads(Path(CONFIG["diagnostics"]).read_text(encoding="utf-8"))
    in_w = np.bincount(tgt, weights=w, minlength=g.n_nodes)
    report = {
        "champion_ff_pct": ff_pct,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in CONFIG.items()},
        "lane_sizes": {lane_list[i]: int((lane == i).sum()) for i in range(K)},
        "lane_median_depth_um": {lane_list[i]: med_depth[i] for i in range(K)},
        "lane_order_check": lane_order_check,
        "variants": variants,
        "diagnostics_ref": {k: diag[k] for k in
                            ("d1_spearman_rank_depth_exc",
                             "d1_spearman_rank_depth_inh",
                             "d1_rank_by_layer", "d2_ei_direction",
                             "d3_span_spectrum", "d4_fb_hubs",
                             "d5_reconstruction")},
        "sink_audit": {"pure_sinks_zero_out_weight": int((out_w == 0).sum()),
                       "pure_sources_zero_in_weight": int((in_w == 0).sum()),
                       "note": "hatched cells: >50% zero-out-weight members — "
                               "unproofread truncated axons and volume-boundary "
                               "cuts. Reconstruction, not biology: only 249 "
                               "axons are proofread in this release."},
        "seconds": time.time() - t0,
    }
    (out / "report_streams.json").write_text(json.dumps(report, indent=1),
                                             encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
