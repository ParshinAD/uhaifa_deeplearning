"""The fly connectome drawn as a layered ANN: biological streams x MFAS depth.

GALLERY_REVIEW.md §5.3: slices of pi are the wrong sole axis for the fly — the
representation it asks for is two-dimensional (group, depth). fly_ann_prep.py
joined the public FlyWire annotations (99.99 % of nodes); this tool builds the
(stream, depth) grid on top:

  * lanes   = biologically named streams (super_class refined by cell_class),
              ordered top-to-bottom by median champion rank — the ordering is
              DATA-DRIVEN and recovers the textbook sensory->motor pathway.
              Optic lanes are named by the INPUT neuropil encoded in
              cell_class ('LA>ME' takes input in the lamina); the olfactory
              intermediate stage (AL/LH cell classes) gets its own lane;
              optic dust (bilateral/ocellar/NA) goes to 'optic (other)', not
              to a neuropil-named lane. The 13 unannotated nodes go to
              'central (other)' and are listed in fly_ann meta.
  * columns = contiguous depth bins of the champion order pi (equal node count),
              so an edge's node-level FF/FB class ALWAYS agrees with its bin
              direction: FF rightward-or-vertical, FB leftward-or-vertical.
  * cells   = (lane, bin) groups, area ~ neuron count; border thickness =
              intra-cell weight (the mass hidden from the bundles); hatched
              grey = >50 % of members have ZERO out-weight (untraced sinks the
              optimizer necessarily dumps late — not biology);
  * bundles = summed edge weight between cells split into FF and FB parts
              (node-level pi class). FF bundles take the SOURCE lane's colour
              (follow a stream by its colour); FB is always crimson, same
              alpha/width law as FF, drawn on top. A direction is inked iff
              its own weight >= min_bundle_frac * total; coverage is reported
              per class from exactly the inked set.

Outputs -> layering/outputs/fly_streams/
  S1_fly_streams_<variant>.png   static preview (the geometry the artifact uses)
  layout_<variant>.json          cells + bundles + lane meta for the artifact
  report_streams.json            all numbers quoted anywhere, incl. the
                                 flow-field hierarchy permutation test, the
                                 neurotransmitter x direction table, the sink
                                 audit, and the B / lane-order sensitivity.

Honesty: annotations are a JOIN, not oracle data (provenance in fly_ann meta:
upstream commit, sha256, license). Crimson is reserved for feedback — no lane
colour may come from the red family. Nothing touches the optimizer or scorer.

Run:  $PY layering/tools/fly_streams.py
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "fly_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "fly_ann" / "node_ann_meta.json",
    "out_dir": HERE.parent / "outputs" / "fly_streams",
    "n_bins_list": [8, 16, 32],    # sensitivity; 16 is the headline figure
    "min_bundle_frac": 2e-4,       # ink a DIRECTION iff its weight >= this x total
    "export_bundle_frac": 5e-5,    # JSON export floor (the artifact's threshold
                                   # slider can go below the static figure's ink
                                   # threshold, never below this)
    "label_top_bundles": 10,       # % labels on the largest inked bundles
    "sink_hatch_frac": 0.5,        # hatch a cell if > this frac has zero out-weight
    "perm_test_n": 2000,           # permutations for the flow-hierarchy test
    "perm_seed": 42,
    "dpi": 110,
}

# Crimson is reserved for feedback edges — no lane may use the red family.
FB_COLOR = "#c23b3b"
LANE_COLORS = {
    "photoreceptors":      "#1f8a8d",
    "sensory (other)":     "#74a33e",
    "ascending (VNC)":     "#9c9a30",
    "lamina":              "#efd077",
    "medulla":             "#e08a2e",
    "medulla→lobula": "#8a6d15",
    "lobula (plate)":      "#6b4d1e",
    "optic (other)":       "#b5a98c",
    "visual centrifugal":  "#c96fae",
    "olfaction: AL·LH":    "#3f9b6e",
    "visual projection":   "#7a4fa3",
    "mushroom body":       "#3f6ab5",
    "central complex":     "#2a4d8f",
    "central (other)":     "#5b7fa6",
    "descending":          "#1f2a44",
    "endocrine":           "#6b7385",
    "motor":               "#101418",
}

OLFACTION_CLASSES = ("ALPN", "ALLN", "ALIN", "ALON", "LHLN", "LHCENT")
MB_CLASSES = ("Kenyon_Cell", "DAN", "MBON", "MBIN")


def lane_of(sc: str, cc: str) -> str:
    """Map one neuron's (super_class, cell_class) to its lane name."""
    if sc == "sensory":
        return "photoreceptors" if cc in ("visual", "optic_lobes") else "sensory (other)"
    if sc in ("ascending", "sensory_ascending"):
        return "ascending (VNC)"
    if sc == "optic":
        head = cc.split(">")[0].split(".")[0]
        if head == "LA":
            return "lamina"
        if head == "ME":
            return "medulla→lobula" if (">" in cc and cc.split(">")[1][:2] == "LO") else "medulla"
        if head in ("LO", "LOP"):
            return "lobula (plate)"
        return "optic (other)"           # bilateral / ocellar / optic_lobes / NA
    if sc == "visual_centrifugal":
        return "visual centrifugal"
    if sc == "visual_projection":
        return "visual projection"
    if sc == "central":
        if cc in MB_CLASSES:
            return "mushroom body"
        if cc in ("CX", "TuBu"):
            return "central complex"
        if cc in OLFACTION_CLASSES:
            return "olfaction: AL·LH"
        return "central (other)"
    if sc == "descending":
        return "descending"
    if sc == "endocrine":
        return "endocrine"
    if sc == "motor":
        return "motor"
    return "central (other)"             # 13 unannotated nodes (listed in meta)


# ---------------------------------------------------------------------------

def assign_lanes(rank, sc_codes, cc_codes, scv, ccv):
    """Per-node lane index + the lane list ordered by median champion rank."""
    lane_names_by_node = np.array(
        [lane_of(scv[s], ccv[c]) for s, c in zip(sc_codes, cc_codes)])
    lane_list = sorted(set(lane_names_by_node),
                       key=lambda L: float(np.median(rank[lane_names_by_node == L])))
    idx = {L: i for i, L in enumerate(lane_list)}
    lane = np.array([idx[L] for L in lane_names_by_node], dtype=np.int64)
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

    med_rank_lane = [float(np.median(rank[lane == i])) for i in range(K)]
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


def flow_hierarchy_test(rank, flow_codes, flow_values, n_perm, seed):
    """Permutation test: does pi order the flow field afferent<intrinsic<efferent?

    Statistic: Spearman rho between the flow ordinal (afferent=0, intrinsic=1,
    efferent=2; NA excluded) and the champion rank. Null: flow labels permuted.
    Pure numpy (elementwise only, no matmul).
    """
    ordinal = {"afferent": 0.0, "intrinsic": 1.0, "efferent": 2.0}
    keep = np.array([flow_values[c] in ordinal for c in flow_codes])
    x = np.array([ordinal[flow_values[c]] for c in flow_codes[keep]])
    y = rank[keep].astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)      # y is a permutation anyway
    def spearman(xv):
        rx = np.argsort(np.argsort(xv, kind="stable"), kind="stable").astype(np.float64)
        rx -= rx.mean(); ryc = ry - ry.mean()
        return float((rx * ryc).sum() /
                     np.sqrt((rx * rx).sum() * (ryc * ryc).sum()))
    rho = spearman(x)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    xs = x.copy()
    for i in range(n_perm):
        rng.shuffle(xs)
        null[i] = spearman(xs)
    exceed = int((np.abs(null) >= abs(rho)).sum())
    return {"spearman_rho": rho, "n": int(keep.sum()),
            "perms": n_perm, "exceedances": exceed,
            "p_two_sided_leq": max(exceed, 1) / n_perm,
            "note": "rho of flow ordinal (aff<intr<eff) vs champion rank; "
                    "label permutation null; p is a <= bound at the 1/perms floor"}


def nt_direction_table(rank, nt_codes, nt_values, src, tgt, w):
    """Neurotransmitter composition of FF vs FB edge weight (by source node)."""
    ff = rank[src] < rank[tgt]
    Wff, Wfb = float(w[ff].sum()), float(w[~ff].sum())
    table = {}
    for i, v in enumerate(nt_values):
        m = nt_codes[src] == i
        wf, wb = float(w[m & ff].sum()), float(w[m & ~ff].sum())
        if wf + wb == 0:
            continue
        table[v] = {"pct_of_ff": 100 * wf / Wff, "pct_of_fb": 100 * wb / Wfb,
                    "fb_share_within_nt": 100 * wb / (wf + wb)}
    table["_note"] = ("top_nt is a PREDICTION (Eckstein et al. 2024) joined "
                      "from the annotation table, not ground truth")
    return table


# ---------------------------------------------------------------------------
# Static preview: the same geometry the artifact will draw.
# ---------------------------------------------------------------------------

def draw_preview(L, Wtot, stats, out_png, dpi):
    K, n_bins = L["K"], L["n_bins"]
    lane_list = L["lane_list"]
    counts = L["counts"].reshape(K, n_bins)
    intra = (L["intra_ff"] + L["intra_fb"]).reshape(K, n_bins)
    sink0 = L["sink0"].reshape(K, n_bins)
    cmax = counts.max()
    intra_max = max(intra.max(), 1e-9)
    thr = CONFIG["min_bundle_frac"] * Wtot

    fig, ax = plt.subplots(figsize=(24, 13.8), dpi=dpi)
    ax.set_xlim(-3.1, n_bins + 0.6)
    ax.set_ylim(K - 0.25, -3.05)

    # lane bands + labels
    for i, name in enumerate(lane_list):
        ax.axhspan(i - 0.42, i + 0.42, color="#000000", alpha=0.025, zorder=0)
        n_lane = int(counts[i].sum())
        ax.annotate(name, (-3.0, i - 0.09), fontsize=10.5, va="center",
                    ha="left", color=LANE_COLORS[name], fontweight="bold")
        ax.annotate(f"{n_lane:,} n · med rank {L['med_rank_lane'][i] / 1000:.1f}k",
                    (-3.0, i + 0.22), fontsize=7.5, va="center", ha="left",
                    color="#52514e")

    # cells; geometry is kept for bundle anchoring
    geo = {}
    for i in range(K):
        for jb in range(n_bins):
            c = counts[i, jb]
            if c == 0:
                continue
            hw = 0.5 * max(0.05, 0.72 * np.sqrt(c / cmax))       # half height
            ww = 0.5 * (0.18 + 0.50 * np.sqrt(c / cmax))         # half width
            geo[(i, jb)] = (jb + 0.5 - ww, jb + 0.5 + ww, hw)
            col = LANE_COLORS[lane_list[i]]
            is_sink = (sink0[i, jb] / c) > CONFIG["sink_hatch_frac"]
            blw = 3.6 * np.sqrt(intra[i, jb] / intra_max)        # intra border
            ax.add_patch(FancyBboxPatch(
                (jb + 0.5 - ww, i - hw), 2 * ww, 2 * hw,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor="#c9c4ba" if is_sink else col,
                hatch="////" if is_sink else None,
                edgecolor="#6e6a62" if is_sink else col,
                linewidth=max(blw, 0.4),        # intra encoding holds on hatched cells too
                alpha=0.24 + 0.68 * np.sqrt(c / cmax), zorder=4))
            if c > 0.04 * cmax:
                ax.annotate(f"{c / 1000:.1f}k" if c >= 1000 else str(int(c)),
                            (jb + 0.5, i), fontsize=6.4, ha="center", va="center",
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

    # same-column bundles get distinct x offsets inside their column
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
        if bs == bt:                                   # vertical, offset per bundle
            m_ = vcount[bs]
            xo = bs + 0.5 + (0 if m_ == 1 else
                             -0.24 + 0.48 * vslot[(cs, ct, is_ff)] / (m_ - 1))
            sgn = np.sign(lt - ls)
            pth = bezier(xo, ls + sgn * gs[2], xo, lt - sgn * gt[2], 0)
            mid = (xo, (ls + lt) / 2)
        elif ls == lt:                                 # same lane: bow as an arc,
            span = abs(bt - bs)                        # height ~ bin distance,
            h = min(0.12 + 0.045 * span, 0.40)         # FF above / FB below spine
            ya = ls - h if is_ff else ls + h
            x0, x1 = (gs[1], gt[0]) if is_ff else (gs[0], gt[1])
            verts = [(x0, ls), (x0 + (x1 - x0) / 3, ya),
                     (x1 - (x1 - x0) / 3, ya), (x1, ls)]
            pth = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                  MplPath.CURVE4, MplPath.CURVE4])
            mid = ((x0 + x1) / 2, ya + (0.06 if not is_ff else -0.06))
        elif is_ff:                                    # rightward from cell edges
            pth = bezier(gs[1], ls, gt[0], lt, 1)
            mid = ((gs[1] + gt[0]) / 2, (ls + lt) / 2 - 0.06)
        else:                                          # leftward feedback
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

    # depth axis
    n = int(counts.sum())
    for jb in range(0, n_bins + 1, 2):
        r = int(jb * n / n_bins)
        ax.annotate(f"{r // 1000}k", (jb + (0.5 if jb < n_bins else 0.2),
                    K - 0.05), fontsize=7.5, ha="center", color="#6b7385")
    ax.annotate("depth = champion order pi (H64), equal-count bins; tick = rank at bin start",
                (n_bins / 2, K + 0.45), ha="center", fontsize=9, color="#52514e")

    # legend
    lx = n_bins - 4.6
    ax.annotate("how to read", (lx, -2.85), fontsize=8.5, fontweight="bold",
                color="#1f2a44")
    for dy, txt, col in (
            (0.30, "bundle colour = SOURCE stream (feedforward)", "#5b7fa6"),
            (0.58, "crimson = feedback (node-level pi class), drawn on top", FB_COLOR),
            (0.86, "same-lane bundles bow over/under their lane (arc height ~ bin distance)", "#52514e"),
            (1.14, "cell border thickness = intra-cell weight; numbers in cells = neurons,", "#52514e"),
            (1.42, "on bundles = % of total weight", "#52514e"),
            (1.70, "hatched grey = >50% of members have no outputs in this graph", "#6e6a62"),
            (1.98, "   (truncated fragments in sensory lanes; genuine terminals for motor)", "#6e6a62")):
        ax.annotate(txt, (lx, -2.85 + dy), fontsize=7.6, color=col)

    ax.set_title(
        "The fly connectome as a layered network — biological streams x MFAS depth\n"
        f"136,648 neurons, 5,657,719 edges; champion feedforward 84.26 % (H64, oracle-verified). "
        f"Lanes: FlyWire annotations (Schlegel et al. 2024; optic lanes named by INPUT neuropil), "
        f"ordered by median pi rank. Columns: {n_bins} equal-count slices of pi.\n"
        f"Cell area ~ neurons (sqrt scale, floored for visibility). Feedforward bundles carry their source lane's colour; "
        f"feedback is crimson, same width/alpha law (bins are pi-contiguous, so FF is always "
        f"rightward-or-vertical and FB leftward-or-vertical — no layout choice can relabel an edge).\n"
        f"Hidden inside cells: {stats['intra_cell_ff_pct']:.1f} % FF + "
        f"{stats['intra_cell_fb_pct']:.1f} % FB of total weight. Inked here: "
        f"{stats['inked_ff_pct_of_inter_ff']:.0f} % of inter-cell FF and "
        f"{stats['inked_fb_pct_of_inter_fb']:.0f} % of inter-cell FB weight "
        f"(threshold {100 * CONFIG['min_bundle_frac']:.2f} % of total per direction); "
        f"same-column flows {stats['same_bin_vertical_pct_of_inter']:.1f} % of inter-cell.",
        fontsize=10.5, loc="left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#faf9f5")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

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

    # lane-order sensitivity: median vs mean, with the actual swaps named
    order_mean = sorted(range(K), key=lambda i: float(np.mean(rank[lane == i])))
    pos_mean = {L: p for p, L in enumerate(order_mean)}
    lane_order_check = {
        "identical": order_mean == list(range(K)),
        "position_changes": {lane_list[i]: {"median_pos": i,
                                            "mean_pos": pos_mean[i],
                                            "median_rank": float(np.median(rank[lane == i])),
                                            "mean_rank": float(np.mean(rank[lane == i]))}
                             for i in range(K) if pos_mean[i] != i},
        "defense": "median is used because several lanes are heavily skewed "
                   "(optic (other): 205 nodes with a long late tail; "
                   "central (other): a 22k grab-bag) — the mean follows the "
                   "tail, the median follows the mass",
    }

    variants = {}
    for n_bins in CONFIG["n_bins_list"]:
        L = build_layout(rank, lane, lane_list, src, tgt, w, n_bins, out_w)
        thr = CONFIG["min_bundle_frac"] * Wtot
        stats = layout_stats(L, Wtot, thr)
        variants[f"bio{K}_B{n_bins}"] = stats
        draw_preview(L, Wtot, stats,
                     out / f"S1_fly_streams_bio{K}_B{n_bins}.png", CONFIG["dpi"])

        # per-cell totals over ALL bundles (unthresholded) so the artifact can
        # state how much of a cell's flow is below any ink threshold
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
        (out / f"layout_bio{K}_B{n_bins}.json").write_text(json.dumps({
            "variant": f"bio{K}_B{n_bins}", "champion_ff_pct": ff_pct,
            "n": int(rank.shape[0]), "m": int(src.shape[0]), "n_bins": n_bins,
            "lanes": [{"name": lane_list[i], "color": LANE_COLORS[lane_list[i]],
                       "n": int((lane == i).sum()),
                       "med_rank": L["med_rank_lane"][i]} for i in range(K)],
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

    hier = flow_hierarchy_test(rank, d["flow"], meta["flow_values"],
                               CONFIG["perm_test_n"], CONFIG["perm_seed"])
    print("flow hierarchy test:", hier)
    nt = nt_direction_table(rank, d["top_nt"], meta["top_nt_values"], src, tgt, w)

    # lane x lane matrix (all bins pooled) — the honesty backstop for the ink
    # threshold: every gram of inter-lane weight appears here, thresholded or not
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

    sinks = int((out_w == 0).sum())
    in_w = np.bincount(tgt, weights=w, minlength=g.n_nodes)
    report = {
        "champion_ff_pct": ff_pct,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in CONFIG.items()},
        "lane_sizes": {lane_list[i]: int((lane == i).sum()) for i in range(K)},
        "lane_order_check": lane_order_check,
        "variants": variants,
        "flow_hierarchy_test": hier,
        "nt_direction": nt,
        "sink_audit": {"pure_sinks_zero_out_weight": sinks,
                       "pure_sources_zero_in_weight": int((in_w == 0).sum()),
                       "note": "cells with >50% zero-out-weight members are "
                               "hatched grey. In sensory/optic lanes these are "
                               "truncated fragments the optimizer necessarily "
                               "places late (an artifact, not biology); for "
                               "motor/endocrine they are genuine terminals whose "
                               "outputs leave the brain and hence the graph"},
        "seconds": time.time() - t0,
    }
    (out / "report_streams.json").write_text(json.dumps(report, indent=1),
                                             encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
