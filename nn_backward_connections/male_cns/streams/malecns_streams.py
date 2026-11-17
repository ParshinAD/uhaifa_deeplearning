"""Male CNS drawn as streams x MFAS depth (counterpart of layering/tools/fly_streams.py).

Lanes are the `lane` column of data/processed/malecns_ann.csv.gz (21 lanes, from
male_cns/pipeline/ann_prep.py), ordered top-to-bottom by median rank. Everything that is
dataset-agnostic is imported from fly_streams.py (build_layout, layout_stats,
flow_hierarchy_test, nt_direction_table, CONFIG thresholds); only the static preview is copied,
because its title and colours are FlyWire-specific.

    $PY male_cns/streams/malecns_streams.py   (after malecns_streams_prep.py)
Output: male_cns/results/streams/{layout_*.json, lane_matrix.json, report_streams.json},
        male_cns/outputs/streams/S1_malecns_streams_*.png
"""
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

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from layering.tools.fly_streams import (CONFIG as FW_CONFIG, FB_COLOR, LANE_COLORS as FW_COLORS,  # noqa: E402
                                        build_layout, flow_hierarchy_test, layout_stats,
                                        nt_direction_table)

CONFIG = {
    **{k: FW_CONFIG[k] for k in ("n_bins_list", "min_bundle_frac", "export_bundle_frac",
                                 "label_top_bundles", "sink_hatch_frac", "perm_test_n",
                                 "perm_seed", "dpi")},
    "graph": ROOT / "data/processed/malecns.npz",
    "ann_npz": ROOT / "male_cns/results/streams/node_ann.npz",
    "ann_meta": ROOT / "male_cns/results/streams/node_ann_meta.json",
    "out_dir": ROOT / "male_cns/results/streams",
    "png_dir": ROOT / "male_cns/outputs/streams",
}

# FlyWire colours for shared lanes; "motor (brain)" takes the FlyWire "motor" colour (same
# neurons: head and neck motor neurons). New lanes were picked for CIELAB distance >= 15 to every
# existing lane colour and to the sink hatch; none is from the red family (reserved for feedback).
LANE_COLORS = {**{k: v for k, v in FW_COLORS.items() if k != "motor"},
               "motor (brain)": FW_COLORS["motor"],
               "VNC sensory": "#56b4e9",
               "VNC intrinsic": "#4d6b5e",
               "motor (VNC)": "#3b3b3b",
               "unannotated": "#8f8f8f"}


def assign_lanes(rank, lane_codes, lane_values):
    """Per-node lane index + lane list ordered by median rank (as fly_streams.assign_lanes)."""
    present = sorted(set(int(c) for c in lane_codes),
                     key=lambda c: float(np.median(rank[lane_codes == c])))
    remap = np.zeros(len(lane_values), dtype=np.int64)
    for new, old in enumerate(present):
        remap[old] = new
    return remap[lane_codes], [lane_values[c] for c in present]


def draw_preview(L, Wtot, stats, out_png, dpi, n, m, ff_pct):
    """Copy of fly_streams.draw_preview with Male CNS colours and caption."""
    K, n_bins = L["K"], L["n_bins"]
    lane_list = L["lane_list"]
    counts = L["counts"].reshape(K, n_bins)
    intra = (L["intra_ff"] + L["intra_fb"]).reshape(K, n_bins)
    sink0 = L["sink0"].reshape(K, n_bins)
    cmax = counts.max()
    intra_max = max(intra.max(), 1e-9)
    thr = CONFIG["min_bundle_frac"] * Wtot

    fig, ax = plt.subplots(figsize=(24, 15.5), dpi=dpi)
    ax.set_xlim(-3.1, n_bins + 0.6)
    ax.set_ylim(K - 0.25, -3.05)

    for i, name in enumerate(lane_list):
        ax.axhspan(i - 0.42, i + 0.42, color="#000000", alpha=0.025, zorder=0)
        n_lane = int(counts[i].sum())
        ax.annotate(name, (-3.0, i - 0.09), fontsize=10.5, va="center",
                    ha="left", color=LANE_COLORS[name], fontweight="bold")
        ax.annotate(f"{n_lane:,} n · med rank {round(L['med_rank_lane'][i] / 1000, 1)}k",
                    (-3.0, i + 0.22), fontsize=7.5, va="center", ha="left", color="#52514e")

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
                ax.annotate(f"{round(c / 1000, 1)}k" if c >= 1000 else str(int(c)),
                            (jb + 0.5, i), fontsize=6.4, ha="center", va="center",
                            color="#faf9f5" if c > 0.3 * cmax else "#141413", zorder=6)

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
        return MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    label_set = set(id(x) for x in sorted(inked, key=lambda x: -x[2])[:CONFIG["label_top_bundles"]])
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
            xo = bs + 0.5 + (0 if m_ == 1 else -0.24 + 0.48 * vslot[(cs, ct, is_ff)] / (m_ - 1))
            sgn = np.sign(lt - ls)
            pth = bezier(xo, ls + sgn * gs[2], xo, lt - sgn * gt[2], 0)
            mid = (xo, (ls + lt) / 2)
        elif ls == lt:
            span = abs(bt - bs)
            h = min(0.12 + 0.045 * span, 0.40)
            ya = ls - h if is_ff else ls + h
            x0, x1 = (gs[1], gt[0]) if is_ff else (gs[0], gt[1])
            verts = [(x0, ls), (x0 + (x1 - x0) / 3, ya), (x1 - (x1 - x0) / 3, ya), (x1, ls)]
            pth = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            mid = ((x0 + x1) / 2, ya + (0.06 if not is_ff else -0.06))
        elif is_ff:
            pth = bezier(gs[1], ls, gt[0], lt, 1)
            mid = ((gs[1] + gt[0]) / 2, (ls + lt) / 2 - 0.06)
        else:
            pth = bezier(gs[0], ls, gt[1], lt, -1)
            mid = ((gs[0] + gt[1]) / 2, (ls + lt) / 2 + 0.10)
        ax.add_patch(mpatches.PathPatch(pth, fill=False, lw=lw, edgecolor=col, alpha=al,
                                        zorder=3 if is_ff else 5, capstyle="round"))
        if id(x) in label_set:
            ax.annotate(f"{round(100 * weight / Wtot, 1)}%", mid, fontsize=7.2, ha="center",
                        va="center", color=col, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="#faf9f5", ec="none", alpha=0.75))

    n_tot = int(counts.sum())
    for jb in range(0, n_bins + 1, 2):
        r = int(jb * n_tot / n_bins)
        ax.annotate(f"{r // 1000}k", (jb + (0.5 if jb < n_bins else 0.2), K - 0.05),
                    fontsize=7.5, ha="center", color="#6b7385")
    ax.annotate("depth = MFAS order pi (H64, one seed), equal-count bins; tick = rank at bin start",
                (n_bins / 2, K + 0.45), ha="center", fontsize=9, color="#52514e")

    lx = n_bins - 4.6
    ax.annotate("how to read", (lx, -2.85), fontsize=8.5, fontweight="bold", color="#1f2a44")
    for dy, txt, col in (
            (0.30, "bundle colour = SOURCE stream (feedforward)", "#5b7fa6"),
            (0.58, "crimson = feedback (node-level pi class), drawn on top", FB_COLOR),
            (0.86, "same-lane bundles bow over/under their lane (arc height ~ bin distance)", "#52514e"),
            (1.14, "cell border thickness = intra-cell weight; numbers in cells = neurons,", "#52514e"),
            (1.42, "on bundles = % of total weight", "#52514e"),
            (1.70, "hatched grey = >50% of members have no outputs in this graph", "#6e6a62")):
        ax.annotate(txt, (lx, -2.85 + dy), fontsize=7.6, color=col)

    ax.set_title(
        "Male CNS (brain + nerve cord) as a layered network: biological streams x MFAS depth\n"
        f"{n:,} neurons, {m:,} edges (weight >= 2, traced only); feedforward {round(ff_pct, 2)} % "
        f"(H64 pipeline, one seed, MPS). Lanes: Male CNS annotations mapped to the FlyWire lanes "
        f"(optic lanes via flywireType), ordered by median pi rank. Columns: {n_bins} equal-count slices of pi.\n"
        "Cell area ~ neurons (sqrt scale, floored). Feedforward bundles carry their source lane's colour; "
        "feedback is crimson (bins are pi-contiguous, so FF is always rightward-or-vertical, FB leftward-or-vertical).\n"
        f"Hidden inside cells: {round(stats['intra_cell_ff_pct'], 1)} % FF + "
        f"{round(stats['intra_cell_fb_pct'], 1)} % FB of total weight. Inked here: "
        f"{round(stats['inked_ff_pct_of_inter_ff'])} % of inter-cell FF and "
        f"{round(stats['inked_fb_pct_of_inter_fb'])} % of inter-cell FB weight "
        f"(threshold {round(100 * CONFIG['min_bundle_frac'], 2)} % of total per direction); "
        f"same-column flows {round(stats['same_bin_vertical_pct_of_inter'], 1)} % of inter-cell.",
        fontsize=10.5, loc="left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#faf9f5")
    plt.close(fig)


def main():
    out, png = CONFIG["out_dir"], CONFIG["png_dir"]
    out.mkdir(parents=True, exist_ok=True)
    png.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(CONFIG["graph"])
    src, tgt = z["src"].astype(np.int64), z["tgt"].astype(np.int64)
    w = z["weight"].astype(np.float64)
    Wtot = w.sum()
    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(CONFIG["ann_meta"].read_text(encoding="utf-8"))
    rank = d["rank"]
    n = rank.shape[0]
    ff_pct = meta["champion_ff_pct"]
    assert abs(100 * w[rank[src] < rank[tgt]].sum() / Wtot - ff_pct) < 1e-9, "node_ann.npz is stale"
    out_w = np.bincount(src, weights=w, minlength=n)

    lane, lane_list = assign_lanes(rank, d["lane"].astype(np.int64), meta["lane_values"])
    K = len(lane_list)

    order_mean = sorted(range(K), key=lambda i: float(np.mean(rank[lane == i])))
    pos_mean = {L: p for p, L in enumerate(order_mean)}
    lane_order_check = {
        "identical": order_mean == list(range(K)),
        "position_changes": {lane_list[i]: {"median_pos": i, "mean_pos": pos_mean[i],
                                            "median_rank": float(np.median(rank[lane == i])),
                                            "mean_rank": float(np.mean(rank[lane == i]))}
                             for i in range(K) if pos_mean[i] != i},
    }

    variants = {}
    thr = CONFIG["min_bundle_frac"] * Wtot
    for n_bins in CONFIG["n_bins_list"]:
        L = build_layout(rank, lane, lane_list, src, tgt, w, n_bins, out_w)
        stats = layout_stats(L, Wtot, thr)
        variants[f"mcns{K}_B{n_bins}"] = stats
        draw_preview(L, Wtot, stats, png / f"S1_malecns_streams_mcns{K}_B{n_bins}.png",
                     CONFIG["dpi"], n, src.shape[0], ff_pct)
        (out / f"layout_mcns{K}_B{n_bins}.json").write_text(json.dumps({
            "variant": f"mcns{K}_B{n_bins}", "champion_ff_pct": ff_pct,
            "n": int(n), "m": int(src.shape[0]), "n_bins": n_bins,
            "lanes": [{"name": lane_list[i], "color": LANE_COLORS[lane_list[i]],
                       "n": int((lane == i).sum()), "med_rank": L["med_rank_lane"][i]}
                      for i in range(K)],
            "ink_threshold_frac": CONFIG["min_bundle_frac"],
            "export_floor_frac": CONFIG["export_bundle_frac"],
            "bundles": [{"s": s, "t": t, "ff": round(wf, 2), "fb": round(wb, 2)}
                        for s, t, wf, wb in L["bundles"]
                        if wf + wb >= CONFIG["export_bundle_frac"] * Wtot],
            "stats": stats}), encoding="utf-8")
        print(f"B={n_bins}: intra {round(stats['intra_cell_pct'], 1)}%  inked FF "
              f"{round(stats['inked_ff_pct_of_inter_ff'])}% / FB {round(stats['inked_fb_pct_of_inter_fb'])}%")

    hier = flow_hierarchy_test(rank, d["flow"], meta["flow_values"],
                               CONFIG["perm_test_n"], CONFIG["perm_seed"])
    print("flow hierarchy test:", hier)
    nt = nt_direction_table(rank, d["top_nt"], meta["top_nt_values"], src, tgt, w)
    nt["_note"] = ("top_nt is the Male CNS consensus prediction (malecns_ann.csv.gz), Kenyon cells "
                   "set to acetylcholine; 'unclear' and missing are NA. Not ground truth")

    ff_edge = rank[src] < rank[tgt]
    lpair = lane[src] * K + lane[tgt]
    lane_W = np.bincount(lpair, weights=w, minlength=K * K).reshape(K, K)
    lane_Wff = np.bincount(lpair[ff_edge], weights=w[ff_edge], minlength=K * K).reshape(K, K)
    (out / "lane_matrix.json").write_text(json.dumps({
        "lane_list": lane_list,
        "W_pct": (100 * lane_W / Wtot).tolist(),
        "W_ff_pct": (100 * lane_Wff / Wtot).tolist(),
        "note": "percent of total weight, source lane = row; FF by node-level order; "
                "diagonal = intra-lane"}), encoding="utf-8")

    in_w = np.bincount(tgt, weights=w, minlength=n)
    sink_lanes = np.bincount(lane[out_w == 0], minlength=K)
    report = {
        "champion_ff_pct": ff_pct,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "lane_colors": {L_: LANE_COLORS[L_] for L_ in lane_list},
        "lane_sizes": {lane_list[i]: int((lane == i).sum()) for i in range(K)},
        "lane_order_check": lane_order_check,
        "variants": variants,
        "flow_hierarchy_test": hier,
        "nt_direction": nt,
        "sink_audit": {"pure_sinks_zero_out_weight": int((out_w == 0).sum()),
                       "pure_sources_zero_in_weight": int((in_w == 0).sum()),
                       "sinks_by_lane": {lane_list[i]: int(sink_lanes[i]) for i in range(K) if sink_lanes[i]},
                       "note": "cells with >50% zero-out-weight members are hatched grey"},
        "seconds": time.time() - t0,
    }
    (out / "report_streams.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done in {round(time.time() - t0)} s -> {out}")


if __name__ == "__main__":
    main()
