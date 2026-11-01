"""Structural diagnostics of the MICrONS champion order against anatomy.

Answers, with numbers saved to a re-runnable JSON, the questions that decide
which visualizations of the microns solution are scientifically worth building
(mirrors the role tools/diagnostics.py played for the mouse and the early fly
analysis):

  D1  Does the optimizer's rank follow cortical depth — and monotonically, or
      in the canonical-microcircuit order (L4 before L2/3 before L5/L6)?
      Spearman rank/depth + median rank per layer band (E and I separately)
      + a label-permutation test.
  D2  The fly finding 3 test with ANATOMICAL labels: is feedback enriched in
      inhibition? Backward share within E and I outgoing weight; composition
      of FF vs FB weight by (E/I of source).
  D3  Feedback locality: share of total weight within +-d of the order, span
      spectrum of feedback weight (fly: a quarter of weight within +-1.7 %).
  D4  Hubs: top-1 / top-0.1 % / top-1 % neuron share of feedback weight
      (fly: hubless, 0.8 % max; mouse region map: one 40 % hub).
  D5  Reconstruction honesty: rank distribution of zero-out-weight neurons,
      and the proofread-axon subset (n=249) as a sensitivity check.
  D6  Horizontal structure: rank vs x (the axis crossing V1 -> higher visual
      areas) and rank vs z, among E cells.

Spearman is computed with average ranks over ties (np.argsort based, no scipy).
Permutation p-values use a fixed seed and report a <= bound at the 1/perms
floor when no permutation exceeds the observed statistic.

Run:  $PY layering/tools/microns_diagnostics.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering import io_utils                                  # noqa: E402

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "microns_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "microns_ann" / "node_ann_meta.json",
    "out_json": HERE.parent / "outputs" / "microns_ann" / "diagnostics.json",
    "perm_n": 2000,
    "perm_seed": 42,
    "span_edges_frac": [0.001, 0.0025, 0.005, 0.01, 0.017, 0.025, 0.05, 0.1,
                        0.2, 0.4, 1.0],
}

LAYERS = ["NA", "L1", "L2/3", "L4", "L5", "L6", "WM"]


def avg_rank(a: np.ndarray) -> np.ndarray:
    """Average ranks with ties (1..n), no scipy."""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = avg_rank(a), avg_rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def perm_p(a: np.ndarray, b: np.ndarray, observed: float, n: int, seed: int):
    """Two-sided permutation p for |spearman| by shuffling b. Returns (p_bound,
    exceedances) where p is a <= bound at the 1/n floor if exceedances==0."""
    rng = np.random.default_rng(seed)
    ra = avg_rank(a)
    ra = ra - ra.mean()
    na = float(np.sqrt((ra * ra).sum()))
    rb0 = avg_rank(b)
    exceed = 0
    for _ in range(n):
        rb = rng.permutation(rb0)
        rb = rb - rb.mean()
        rho = float((ra * rb).sum() / (na * np.sqrt((rb * rb).sum())))
        if abs(rho) >= abs(observed):
            exceed += 1
    return (max(exceed, 1) / n, exceed)


def main():
    cfg = CONFIG
    t0 = time.time()
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("microns")
    n = g.n_nodes
    ann = np.load(cfg["ann_npz"])
    meta_ann = json.loads(Path(cfg["ann_meta"]).read_text())
    depth = ann["depth_um"].astype(np.float64)
    layer = ann["layer"]
    ei = ann["ei"]
    x_um = ann["x_um"].astype(np.float64)
    z_um = ann["z_um"].astype(np.float64)
    axon_pr = ann["axon_pr"].astype(bool)
    out_w = ann["out_w"].astype(np.int64)
    assert np.array_equal(ann["rank"], rank)
    w = g.weight.astype(np.int64)
    total_w = int(w.sum())
    ff = rank[g.src] < rank[g.tgt]
    fb_w = int(w[~ff].sum())
    res = {"champion_ff_pct": ff_pct, "n": n, "m": g.n_edges,
           "total_weight": total_w, "fb_weight": fb_w,
           "fb_pct": 100.0 * fb_w / total_w}
    print(f"loaded: FF {ff_pct:.4f}%, FB {res['fb_pct']:.2f}%  "
          f"({time.time() - t0:.0f}s)")

    # ---- D1: rank vs depth ---------------------------------------------------
    for cls, name in ((1, "exc"), (2, "inh")):
        m = (ei == cls) & np.isfinite(depth)
        rho = spearman(rank[m].astype(np.float64), depth[m])
        p, exc_n = perm_p(rank[m].astype(np.float64), depth[m], rho,
                          cfg["perm_n"], cfg["perm_seed"])
        res[f"d1_spearman_rank_depth_{name}"] = {
            "rho": rho, "n": int(m.sum()), "p_two_sided_leq": p,
            "exceedances": exc_n, "perms": cfg["perm_n"]}
        print(f"D1 {name}: Spearman(rank, depth) = {rho:+.3f} "
              f"(n={int(m.sum())}, p<={p:.4g})")
    med = {}
    for cls, name in ((1, "exc"), (2, "inh")):
        med[name] = {}
        for code in range(1, 7):
            m = (ei == cls) & (layer == code)
            if int(m.sum()) >= 50:
                med[name][LAYERS[code]] = {
                    "n": int(m.sum()),
                    "median_rank": float(np.median(rank[m])),
                    "q25": float(np.percentile(rank[m], 25)),
                    "q75": float(np.percentile(rank[m], 75))}
    res["d1_rank_by_layer"] = med
    for name in med:
        print(f"D1 {name} median rank by band: " + "  ".join(
            f"{k}:{v['median_rank']:.0f}" for k, v in med[name].items()))

    # ---- D2: E/I direction budget -------------------------------------------
    src_ei = ei[g.src]
    d2 = {}
    for cls, name in ((1, "exc"), (2, "inh")):
        m = src_ei == cls
        tw = int(w[m].sum())
        bw = int(w[m & ~ff].sum())
        d2[name] = {"out_weight": tw, "backward_weight": bw,
                    "backward_share_pct": 100.0 * bw / tw if tw else None,
                    "pct_of_ff": 100.0 * int(w[m & ff].sum()) / int(w[ff].sum()),
                    "pct_of_fb": 100.0 * bw / fb_w}
    m = src_ei == 0
    d2["na"] = {"out_weight": int(w[m].sum()),
                "backward_share_pct": 100.0 * int(w[m & ~ff].sum()) / max(int(w[m].sum()), 1)}
    res["d2_ei_direction"] = d2
    print(f"D2 backward share: E {d2['exc']['backward_share_pct']:.2f}%  "
          f"I {d2['inh']['backward_share_pct']:.2f}%  (brain-wide {res['fb_pct']:.2f}%)")
    # target-side view as well (what E and I RECEIVE):
    tgt_ei = ei[g.tgt]
    d2t = {}
    for cls, name in ((1, "exc"), (2, "inh")):
        m = tgt_ei == cls
        tw = int(w[m].sum())
        d2t[name] = {"in_weight": tw,
                     "backward_share_pct": 100.0 * int(w[m & ~ff].sum()) / tw if tw else None}
    res["d2_ei_direction_target_side"] = d2t

    # ---- D3: locality / span spectrum ---------------------------------------
    span = np.abs(rank[g.src].astype(np.int64) - rank[g.tgt].astype(np.int64))
    fr = np.array(cfg["span_edges_frac"])
    cuts = (fr * n).astype(np.int64)
    spec_all, spec_fb = [], []
    prev = 0
    for c in cuts:
        m = (span > prev) & (span <= c)
        spec_all.append(int(w[m].sum()))
        spec_fb.append(int(w[m & ~ff].sum()))
        prev = c
    res["d3_span_spectrum"] = {
        "edges_frac": cfg["span_edges_frac"],
        "weight_all": spec_all, "weight_fb": spec_fb,
        "within_1p7pct_share_of_total":
            100.0 * int(w[span <= int(0.017 * n)].sum()) / total_w,
        "fb_within_1p7pct_share_of_fb":
            100.0 * int(w[(span <= int(0.017 * n)) & ~ff].sum()) / fb_w}
    print(f"D3 weight within +-1.7% of order: "
          f"{res['d3_span_spectrum']['within_1p7pct_share_of_total']:.1f}% of total; "
          f"{res['d3_span_spectrum']['fb_within_1p7pct_share_of_fb']:.1f}% of FB")

    # ---- D4: feedback hubs ---------------------------------------------------
    fb_in = np.bincount(g.tgt[~ff], weights=w[~ff], minlength=n)
    fb_out = np.bincount(g.src[~ff], weights=w[~ff], minlength=n)
    fb_node = fb_in + fb_out          # each FB unit counted once per endpoint
    top = np.sort(fb_in)[::-1]
    res["d4_fb_hubs"] = {
        "top1_in_share_pct": 100.0 * float(top[0]) / fb_w,
        "top10_in_share_pct": 100.0 * float(top[:10].sum()) / fb_w,
        "top0p1pct_in_share_pct": 100.0 * float(top[:max(n // 1000, 1)].sum()) / fb_w,
        "top1pct_in_share_pct": 100.0 * float(top[:n // 100].sum()) / fb_w,
        "note": "shares of FB weight arriving at (in) the top receiving neurons",
    }
    print(f"D4 FB hubs: top neuron {res['d4_fb_hubs']['top1_in_share_pct']:.2f}%  "
          f"top 0.1% {res['d4_fb_hubs']['top0p1pct_in_share_pct']:.1f}%  "
          f"top 1% {res['d4_fb_hubs']['top1pct_in_share_pct']:.1f}%")

    # ---- D5: reconstruction honesty -----------------------------------------
    zero_out = out_w == 0
    res["d5_reconstruction"] = {
        "zero_out_nodes": int(zero_out.sum()),
        "zero_out_median_rank_frac": float(np.median(rank[zero_out]) / n),
        "zero_out_in_last_10pct": float(
            (rank[zero_out] > 0.9 * n).mean()),
        "proofread_axon_nodes": int(axon_pr.sum()),
        "proofread_out_w_median": float(np.median(out_w[axon_pr])),
        "all_out_w_median": float(np.median(out_w)),
        "note": "only 249 of 67,534 neurons have a proofread axon; unproofread "
                "axons are truncated, so outgoing connectivity is systematically "
                "undersampled — an honesty caveat for every directional claim",
    }
    print(f"D5 zero-out nodes: {int(zero_out.sum())}, median rank frac "
          f"{res['d5_reconstruction']['zero_out_median_rank_frac']:.2f}; "
          f"proofread axon median out_w {res['d5_reconstruction']['proofread_out_w_median']:.0f} "
          f"vs all {res['d5_reconstruction']['all_out_w_median']:.0f}")

    # ---- D6: horizontal structure -------------------------------------------
    for coord, name in ((x_um, "x"), (z_um, "z")):
        m = (ei == 1) & np.isfinite(coord) & np.isfinite(depth)
        rho = spearman(rank[m].astype(np.float64), coord[m])
        res[f"d6_spearman_rank_{name}_exc"] = {"rho": rho, "n": int(m.sum())}
        print(f"D6 Spearman(rank, {name}) among E: {rho:+.3f}")

    res["layers_meta"] = meta_ann["layers"]
    res["seconds"] = time.time() - t0
    Path(cfg["out_json"]).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"saved -> {cfg['out_json']}  ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
