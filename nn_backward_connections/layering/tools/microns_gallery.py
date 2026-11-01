"""Candidate visualizations of the MICrONS champion order — the selection gallery.

Five candidates besides the streams view (M2, microns_streams.py), each aimed at
one finding from diagnostics.json; the review round then picks which survive
into the interactive artifact:

  M1  depth x rank density (E and I side by side) — THE direct test of
      "does the optimizer's order follow cortical depth", with the layer bands
      and the per-depth median-rank trend drawn on top.
  M3  lane x lane matrix (weight % and FF share) — the aggregate circuit the
      streams view thresholds away; explains WHICH inter-layer weight makes the
      order monotone with depth and why inhibitory lanes sort early.
  M4  real-space views — somas at their true cortical positions (side view
      x-depth, top view x-z), painted by champion rank: the optimizer's "time
      axis" on real anatomy. Unique to MICrONS; the fly had no soma positions.
  M5  E/I direction budget — the anatomical-label version of the fly's
      Chemistry finding, which REVERSES here: excitatory output carries the
      feedback, inhibitory output is forward-aligned.
  M6  span spectrum, fly vs microns — cumulative weight vs order distance for
      all weight and for feedback: the fly's feedback is local chains, the
      cortex's is long-range.

All numbers land in report_gallery.json (incl. the depth-vs-z leak check on the
pia correction and the E<->I weight asymmetry that explains the early-I effect).
Colors follow the validated ramps of microns_streams.py; sequential = blue,
diverging = blue<->red around 50 % FF share, crimson stays reserved for feedback.

Run:  $PY layering/tools/microns_gallery.py
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from microns_streams import LANE_SPEC, LANE_COLORS, FB_COLOR, assign_lanes  # noqa: E402

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "microns_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "microns_ann" / "node_ann_meta.json",
    "diagnostics": HERE.parent / "outputs" / "microns_ann" / "diagnostics.json",
    "out_dir": HERE.parent / "outputs" / "microns_gallery",
    "rank_bins": 64,
    "depth_bin_um": 20.0,
    "depth_max_um": 900.0,
    "matrix_min_w_pct": 0.05,   # grey out FF-share cells below this weight
    "span_grid": np.geomspace(1e-4, 1.0, 61),
    "dpi": 120,
}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
RANK_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURF = "#fcfcfb"
FLY_COLOR = "#eb6834"
MIC_COLOR = "#2a78d6"


def seq_cmap():
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def rank_cmap():
    return LinearSegmentedColormap.from_list("rank_blue", RANK_RAMP)


def div_cmap():
    # FF share: red (feedback-dominant) <-> neutral <-> blue (feedforward)
    return LinearSegmentedColormap.from_list(
        "ffshare", ["#c23b3b", "#e6a0a0", "#f0efec", "#9ec5f4", "#2a78d6"])


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def m1_depth_rank(d, boundaries, out, dpi):
    """M1: depth x rank density, E and I panels + median-rank trend."""
    rank = d["rank"].astype(np.float64)
    depth = d["depth_um"].astype(np.float64)
    ei = d["ei"]
    n = rank.shape[0]
    rb = CONFIG["rank_bins"]
    db = CONFIG["depth_bin_um"]
    dmax = CONFIG["depth_max_um"]
    dedges = np.arange(0, dmax + db, db)
    redges = np.linspace(0, n, rb + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), dpi=dpi, sharey=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(SURF)
    stats = {}
    exc_trend = ([], [])
    for ax, cls, ttl in ((axes[0], 1, "excitatory (55,855)"),
                         (axes[1], 2, "inhibitory (7,219)")):
        m = (ei == cls) & np.isfinite(depth) & (depth >= 0) & (depth <= dmax)
        H, _, _ = np.histogram2d(depth[m], rank[m], bins=[dedges, redges])
        pc = ax.pcolormesh(redges, dedges, H, cmap=seq_cmap(),
                           norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
        # per-depth-row median rank trend; starts at the L1/L23 band edge —
        # somas shallower than that are pia-proxy edge cases with tiny n
        med_x, med_y = [], []
        d_lo = boundaries.get("L1/L23", 60.0)
        for k in range(len(dedges) - 1):
            if dedges[k] < d_lo:
                continue
            mm = m & (depth >= dedges[k]) & (depth < dedges[k + 1])
            if int(mm.sum()) >= 30:
                med_x.append(float(np.median(rank[mm])))
                med_y.append(0.5 * (dedges[k] + dedges[k + 1]))
        if cls == 1:
            exc_trend = (med_x, med_y)
        else:
            ax.plot(exc_trend[0], exc_trend[1], color=INK, lw=1.2,
                    ls=(0, (4, 3)), label="excitatory median (reference)")
        ax.plot(med_x, med_y, color=INK, lw=2.0, solid_capstyle="round",
                label="median rank at this depth")
        for nm, bd in boundaries.items():
            ax.axhline(bd, color=MUTED, lw=0.7, ls=(0, (4, 3)))
            if ax is axes[0]:
                ax.annotate(nm, (n * 0.995, bd - 4), fontsize=7, ha="right",
                            color=MUTED)
        ax.set_title(ttl, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("champion order rank (H42)", fontsize=9, color=INK2)
        style_ax(ax)
        stats[("exc" if cls == 1 else "inh") + "_median_trend"] = {
            "depth_um": med_y, "median_rank": med_x}
    axes[0].set_ylabel("soma depth below pia (µm)", fontsize=9, color=INK2)
    axes[0].invert_yaxis()
    cb = fig.colorbar(pc, ax=axes, pad=0.012, fraction=0.03)
    cb.set_label("neurons per cell (log; colour scale per panel)", fontsize=8,
                 color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    for ax in axes:
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(
        "M1 — Does the optimizer's order follow cortical depth?  "
        "Soma depth vs champion rank, one 2-D histogram per E/I class\n"
        "The excitatory trend is monotone: the order descends the cortex "
        "(L2/3 → L4 → L5 → L6), NOT the canonical L4-first microcircuit — "
        "thalamic input is outside the graph. Inhibitory medians sit earlier in "
        "every layer band (by 6.7–12.9k ranks; not in every thin depth slice — "
        "the deepest small-n bins cross).\n"
        "Layer lines are our density-calibrated bands, not traced borders; the "
        "far-right ranks include the 1,205 zero-output fragments the order "
        "necessarily dumps last. Trend lines start at the L1/L2-3 band edge.",
        fontsize=10, color=INK, x=0.005, ha="left")
    fig.savefig(out / "M1_depth_vs_rank.png", facecolor=SURF,
                bbox_inches="tight")
    plt.close(fig)
    return stats


def m3_matrix(lane, lane_list, rank, src, tgt, w, out, dpi):
    """M3: lane x lane weight and FF share heatmaps."""
    K = len(lane_list)
    Wtot = w.sum()
    ff = rank[src] < rank[tgt]
    lpair = lane[src] * K + lane[tgt]
    W = np.bincount(lpair, weights=w, minlength=K * K).reshape(K, K)
    Wff = np.bincount(lpair[ff], weights=w[ff], minlength=K * K).reshape(K, K)
    Wpct = 100 * W / Wtot
    share = np.where(W > 0, 100 * Wff / np.maximum(W, 1e-12), np.nan)
    mask = Wpct < CONFIG["matrix_min_w_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.0), dpi=dpi)
    fig.patch.set_facecolor(SURF)
    for ax, M, ttl, cmap, vmin, vmax, fmt in (
            (axes[0], Wpct, "share of total weight (%, log color)",
             seq_cmap(), None, None, "{:.1f}"),
            (axes[1], np.ma.masked_where(mask, share),
             "feedforward share of that weight (%)",
             div_cmap(), 0, 100, "{:.0f}")):
        if ttl.startswith("share"):
            pc = ax.pcolormesh(np.ma.masked_where(M <= 0, M), cmap=cmap,
                               norm=LogNorm(vmin=max(Wpct[Wpct > 0].min(), 1e-3),
                                            vmax=Wpct.max()))
        else:
            ax.pcolormesh(np.ones_like(Wpct), cmap=None, color="#dedbd3",
                          shading="auto")   # masked ground: grey, not white
            pc = ax.pcolormesh(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(K) + 0.5, lane_list, rotation=45, ha="right",
                      fontsize=8)
        ax.set_yticks(np.arange(K) + 0.5, lane_list, fontsize=8)
        ax.invert_yaxis()
        ax.set_ylabel("source lane", fontsize=9, color=INK2)
        ax.set_xlabel("target lane", fontsize=9, color=INK2)
        ax.set_title("M3 — " + ttl, fontsize=10, color=INK, loc="left")
        for i in range(K):
            for j in range(K):
                v = M[i, j] if not np.ma.is_masked(M[i, j]) else None
                if v is None or not np.isfinite(v):
                    continue
                if ttl.startswith("share") and Wpct[i, j] < 0.25:
                    continue
                big = (Wpct[i, j] > 3) if ttl.startswith("share") else \
                      (0 <= v <= 25 or v >= 75)
                ax.annotate(fmt.format(v), (j + 0.5, i + 0.5), ha="center",
                            va="center", fontsize=6.6,
                            color="#faf9f5" if (ttl.startswith("share") and
                                                Wpct[i, j] > 3) or
                                               (not ttl.startswith("share") and
                                                (v <= 20 or v >= 80)) else INK)
        style_ax(ax)
        cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.045)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.suptitle(
        "M3 — The aggregate circuit: lane-to-lane synaptic weight and its direction under the champion order\n"
        "Left: where the weight is (row = source; every pair appears, nothing thresholded). Right: how much of "
        "each pair's weight the order calls feedforward (blue = forward-dominant, red = feedback-dominant;\n"
        f"grey = pair under {CONFIG['matrix_min_w_pct']}% of total weight, where a share is noise). Every red cell "
        "is an E→I pair. Caveat: unproofread excitatory axons are truncated, so I-row weights are relatively "
        "over-sampled (249/67,534 axons proofread).",
        fontsize=10, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / "M3_lane_matrix.png", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return {"W_pct": Wpct.tolist(), "ff_share_pct":
            np.where(np.isfinite(share), share, -1).tolist(),
            "lane_list": lane_list}


def m4_real_space(d, boundaries, out, dpi):
    """M4: somas at true positions, painted by champion rank."""
    rank = d["rank"].astype(np.float64)
    x = d["x_um"].astype(np.float64)
    z = d["z_um"].astype(np.float64)
    depth = d["depth_um"].astype(np.float64)
    n = rank.shape[0]
    m = np.isfinite(depth)
    frac = rank / n

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), dpi=dpi,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.patch.set_facecolor(SURF)
    o = np.argsort(np.random.default_rng(42).random(int(m.sum())))  # draw order shuffle
    xs, ds, zs, fs = x[m][o], depth[m][o], z[m][o], frac[m][o]
    sc = axes[0].scatter(xs, ds, c=fs, cmap=rank_cmap(), s=1.5, alpha=0.5,
                         linewidths=0, rasterized=True)
    for nm, bd in boundaries.items():
        axes[0].axhline(bd, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        axes[0].annotate(nm, (float(np.nanmax(x)) * 0.995, bd - 6), fontsize=7,
                         ha="right", color=MUTED)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("x (µm) — the axis crossing V1 → higher visual areas",
                       fontsize=9, color=INK2)
    axes[0].set_ylabel("soma depth below pia (µm)", fontsize=9, color=INK2)
    axes[0].set_title("side view: the depth gradient in real anatomy",
                      fontsize=10, color=INK, loc="left")

    axes[1].scatter(xs, zs, c=fs, cmap=rank_cmap(), s=1.5, alpha=0.5,
                    linewidths=0, rasterized=True)
    axes[1].set_xlabel("x (µm)", fontsize=9, color=INK2)
    axes[1].set_ylabel("z (µm)", fontsize=9, color=INK2)
    axes[1].set_title("top view: no rank trend along x (ρ=+0.05); a weak z "
                      "trend exists (ρ=−0.24)", fontsize=10,
                      color=INK, loc="left")
    for ax in axes:
        style_ax(ax)
        ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.03)
    cb.set_label("champion rank (fraction of order: light = early, dark = late)",
                 fontsize=8, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.suptitle(
        "M4 — The optimizer's hierarchy painted onto real cortex "
        "(every soma at its true position, colored by champion rank; draw order shuffled)\n"
        "Depth below pia is corrected for the tilted pia surface (2-D grid fit, see methods). "
        "The vertical gradient is the finding of M1 seen in anatomy. Horizontally the order is "
        "flat along x — the axis crossing V1 into higher areas (ρ=+0.05) — but carries a weak z "
        "trend (ρ=−0.24, of which about a quarter is the −0.16 depth-correction residual).",
        fontsize=10, color=INK, x=0.005, ha="left")
    fig.savefig(out / "M4_real_space.png", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def m5_ei_budget(diag, pairs, out, dpi):
    """M5: E/I direction budget bars (anatomical-label Chemistry analogue)."""
    d2 = diag["d2_ei_direction"]
    d2t = diag.get("d2_ei_direction_target_side", {})
    fb_pct = diag["fb_pct"]
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.6), dpi=dpi,
                             gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    fig.patch.set_facecolor(SURF)

    ax = axes[0]
    rows = [("excitatory", d2["exc"]["backward_share_pct"], "#2a78d6"),
            ("inhibitory", d2["inh"]["backward_share_pct"], "#37996a"),
            ("unclassified", d2["na"]["backward_share_pct"], "#b5a98c")]
    ypos = np.arange(len(rows))[::-1]
    for (nm, v, c), yp in zip(rows, ypos):
        ax.barh(yp, v, height=0.52, color=c, edgecolor=SURF, linewidth=2)
        ax.annotate(f"{v:.1f}%", (v + 0.5, yp), va="center", fontsize=9,
                    color=INK)
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=9)
    ax.axvline(fb_pct, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.annotate(f"all classes: {fb_pct:.1f}%", (fb_pct + 0.5, ypos[0] + 0.33),
                fontsize=8, color=INK, va="top")
    ax.set_xlim(0, 32)
    ax.set_xlabel("share of the class's outgoing weight\nthat points backward (%)",
                  fontsize=9, color=INK2)
    ax.set_title("backward share by class (source side)", fontsize=10,
                 color=INK, loc="left")
    style_ax(ax)

    ax = axes[1]
    comp = [("feedforward\nweight", d2["exc"]["pct_of_ff"], d2["inh"]["pct_of_ff"]),
            ("backward\nweight", d2["exc"]["pct_of_fb"], d2["inh"]["pct_of_fb"])]
    for k, (nm, e, i) in enumerate(comp):
        na = 100 - e - i
        left = 0
        for v, c, lbl in ((e, "#2a78d6", "E"), (i, "#37996a", "I"),
                          (na, "#b5a98c", "n/a")):
            ax.barh(1 - k, v, left=left, height=0.5, color=c,
                    edgecolor=SURF, linewidth=2)
            if v > 6:
                ax.annotate(f"{lbl} {v:.0f}%", (left + v / 2, 1 - k),
                            ha="center", va="center", fontsize=8.5,
                            color="#faf9f5" if c != "#b5a98c" else INK)
            left += v
    ax.set_yticks([1, 0], [c[0] for c in comp], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("composition by source class (%)", fontsize=9, color=INK2)
    ax.set_title("who carries each direction", fontsize=10, color=INK,
                 loc="left")
    style_ax(ax)

    # the mechanism: the four class-pair channels
    ax = axes[2]
    order = ["I->E", "E->E", "E->I", "I->I"]
    cols = {"I->E": "#37996a", "E->E": "#2a78d6", "E->I": "#2a78d6",
            "I->I": "#37996a"}
    ypos = np.arange(len(order))[::-1]
    pmax = max(pairs[k]["pct_of_total"] for k in order)
    for k, yp in zip(order, ypos):
        p = pairs[k]
        w = p["pct_of_total"]
        ffw = w * p["ff_share_pct"] / 100
        ax.barh(yp, ffw, height=0.52, color=cols[k], edgecolor=SURF, linewidth=2)
        ax.barh(yp, w - ffw, left=ffw, height=0.52, color="#c23b3b",
                edgecolor=SURF, linewidth=2)
        ax.annotate(f"{w:.1f}% of all · {p['ff_share_pct']:.0f}% FF",
                    (w + 0.6, yp), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ypos, [k.replace('->', '→') for k in order], fontsize=9)
    ax.set_xlim(0, pmax * 1.55)
    ax.set_xlabel("share of ALL weight (crimson part = backward)",
                  fontsize=9, color=INK2)
    ax.set_title("the mechanism: pay the loop on the light side", fontsize=10,
                 color=INK, loc="left")
    style_ax(ax)

    into_i = d2t.get("inh", {}).get("backward_share_pct")
    into_e = d2t.get("exc", {}).get("backward_share_pct")
    tgt_line = (f" Target side: {into_i:.0f}% of what interneurons RECEIVE is "
                f"backward, vs {into_e:.0f}% for excitatory cells — the feedback "
                "E carries lands on I." if into_i is not None else "")
    fig.suptitle(
        "M5 — Feedback is EXCITATORY here: the fly's chemistry finding reverses under anatomical E/I labels\n"
        "Fly (predicted transmitters): forward = excitatory ACh, backward enriched for GABA/modulators. "
        "MICrONS (anatomy): inhibitory output is 5.2 % backward because the optimizer places interneurons "
        "EARLY — I→E is 3.6× heavier than E→I, so the loop pays its cost on the light E→I side (48 % FF)."
        + tgt_line + "\n"
        "Caveat: weight = synapse counts, and unproofread excitatory axons are truncated at the volume "
        "boundary while inhibitory axons are local and better sampled — class weight SHARES are biased "
        "toward inhibition; the within-class direction splits are the robust part.",
        fontsize=10, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out / "M5_ei_budget.png", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def m6_span_spectrum(rank_m, src_m, tgt_m, w_m, out, dpi):
    """M6: cumulative weight vs order distance, fly vs microns, all vs FB."""
    from layering import io_utils
    g_fly, _, rank_f, _ = io_utils.load_graph_and_champion_order("connectome")
    curves = {}
    for name, (rk, s, t, ww) in (
            ("microns", (rank_m, src_m, tgt_m, w_m)),
            ("fly", (rank_f, g_fly.src.astype(np.int64),
                     g_fly.tgt.astype(np.int64),
                     g_fly.weight.astype(np.float64)))):
        n = rk.shape[0]
        span = np.abs(rk[s].astype(np.int64) - rk[t].astype(np.int64)) / n
        ff = rk[s] < rk[t]
        Wtot = ww.sum()
        Wfb = ww[~ff].sum()
        grid = CONFIG["span_grid"]
        cum_all = [100 * float(ww[span <= gv].sum()) / Wtot for gv in grid]
        cum_fb = [100 * float(ww[(span <= gv) & ~ff].sum()) / Wfb for gv in grid]
        curves[name] = {"grid": grid.tolist(), "all": cum_all, "fb": cum_fb}

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=dpi)
    fig.patch.set_facecolor(SURF)
    for name, col in (("fly", FLY_COLOR), ("microns", MIC_COLOR)):
        ax.plot(curves[name]["grid"], curves[name]["all"], color=col, lw=2,
                label=f"{name} — all weight")
        ax.plot(curves[name]["grid"], curves[name]["fb"], color=col, lw=2,
                ls=(0, (5, 3)), label=f"{name} — feedback only")
    ax.axvline(0.017, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.annotate("±1.7% of order", (0.017, 97), fontsize=8, color=MUTED,
                rotation=90, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("order distance |rank(u) − rank(v)| as a fraction of the brain",
                  fontsize=9, color=INK2)
    ax.set_ylabel("cumulative share of weight within that distance (%)",
                  fontsize=9, color=INK2)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8.5, framealpha=0.9)
    style_ax(ax)
    g17 = {nm: (float(np.interp(0.017, curves[nm]["grid"], curves[nm]["all"])),
                float(np.interp(0.017, curves[nm]["grid"], curves[nm]["fb"])))
           for nm in curves}
    ax.set_title(
        "M6 — Two feedback architectures: the fly's weight is order-local, the cortex patch's is not\n"
        f"Cumulative weight vs distance along each champion order (order distance, "
        f"not µm). Within ±1.7% of the order the fly holds {g17['fly'][0]:.1f}% "
        f"of all weight and {g17['fly'][1]:.1f}% of its feedback (short chains); "
        f"MICrONS holds {g17['microns'][0]:.1f}% and {g17['microns'][1]:.1f}%.\n"
        "Caveat: this compares a complete brain against a 1.4-mm patch whose "
        "long-range axons are cut at the volume boundary and mostly unproofread.",
        fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "M6_span_spectrum.png", facecolor=SURF,
                bbox_inches="tight")
    plt.close(fig)
    return curves


def main():
    cfg = CONFIG
    out = cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from layering import io_utils
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("microns")
    src = g.src.astype(np.int64)
    tgt = g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    d = np.load(cfg["ann_npz"])
    meta = json.loads(Path(cfg["ann_meta"]).read_text(encoding="utf-8"))
    diag = json.loads(Path(cfg["diagnostics"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank)
    boundaries = meta["layers"]["boundaries_um"]
    lane, lane_list = assign_lanes(d["ei"], d["layer"])

    report = {"champion_ff_pct": ff_pct}

    # supporting numbers: E<->I weight asymmetry (explains the early-I effect)
    ei = d["ei"]
    ff = rank[src] < rank[tgt]
    pairs = {}
    for (a, b), nm in (((1, 2), "E->I"), ((2, 1), "I->E"), ((1, 1), "E->E"),
                       ((2, 2), "I->I")):
        m = (ei[src] == a) & (ei[tgt] == b)
        tw = float(w[m].sum())
        pairs[nm] = {"weight": tw, "pct_of_total": 100 * tw / w.sum(),
                     "ff_share_pct": 100 * float(w[m & ff].sum()) / tw}
    report["ei_pair_budget"] = pairs
    print("E<->I budget:", {k: (round(v['pct_of_total'], 1),
                                round(v['ff_share_pct'], 1)) for k, v in pairs.items()})

    # depth-vs-z leak check on the pia correction
    from microns_diagnostics import spearman
    m = (ei == 1) & np.isfinite(d["depth_um"])
    report["depth_z_leak_check"] = {
        "spearman_depth_z_exc": spearman(d["depth_um"][m].astype(np.float64),
                                         d["z_um"][m].astype(np.float64)),
        "spearman_depth_x_exc": spearman(d["depth_um"][m].astype(np.float64),
                                         d["x_um"][m].astype(np.float64)),
        "note": "near-zero values mean the 2-D pia correction did not leave a "
                "horizontal trend inside the depth coordinate itself"}
    print("depth leak check:", report["depth_z_leak_check"])

    report["m1"] = m1_depth_rank(d, boundaries, out, cfg["dpi"])
    report["m3"] = m3_matrix(lane, lane_list, rank, src, tgt, w, out, cfg["dpi"])
    m4_real_space(d, boundaries, out, cfg["dpi"])
    m5_ei_budget(diag, pairs, out, cfg["dpi"])
    report["m6"] = m6_span_spectrum(rank, src, tgt, w, out, cfg["dpi"])

    report["seconds"] = time.time() - t0
    (out / "report_gallery.json").write_text(json.dumps(report, indent=1),
                                             encoding="utf-8")
    print(f"done in {report['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
