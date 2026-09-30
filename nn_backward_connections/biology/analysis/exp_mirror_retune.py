"""Mirror-swap RE-OPTIMIZATION: how much of the -2.21 pp swap cost is recoverable?

Extends layering/tools/exp_mirror_swap.py (QA_ORDER_AND_ROBUSTNESS.md, Q3).
That experiment swapped every mirror-paired left/right neuron's rank in the
fly-connectome champion order and measured FF 84.26% -> 82.05% (-2.21 pp), with
the stated caveat that -2.21 pp is only an UPPER bound on the cost of mirroring:
the swapped order was never re-optimized.

This experiment applies the champion's own discrete refinement (H64 stages 3+4,
NO stage-2 Rocket restart — a gradient restart would discard the swap) to three
starts, identically:

  (a) champion  — the champion order itself. Control: the champion should be
      ~ a fixed point of its own refinement; whatever it still gains defines
      the yardstick's zero.
  (b) mirror    — the mirror-swapped order (65,452 within-type L<->R rank pairs).
  (c) c1b       — the same-side rank-matched control swap (C1b of the original
      experiment): identical pairing STRUCTURE, no hemisphere crossing. The
      fair structural control.

Headline: irrecoverable mirror gap = retuned(a) - retuned(b), compared with the
structural-control gap retuned(a) - retuned(c). The mirror-specific residual
retuned(c) - retuned(b) is a lower-bound estimate of true left/right wiring
asymmetry as seen by this objective, separated from mere fine-tuning loss.

Secondary: does refinement UN-swap the pairs (revert each pair's within-pair
rank relation to the champion's) or stay in the mirrored configuration
(symmetric twin optimum)?

Regression gate: before anything runs, the reproduced swapped orders must score
exactly what layering/outputs/qa/exp_mirror_swap.json records (all four schemes).
On any mismatch the script aborts.

Refinement = the champion's own stages, exact production code, champion constants
(src/mfas/experiments/H64.py, connectome row):
  stage 3  sift_underrelaxed   max_sweeps=40, k_full=6, alpha=0.7
  stage 4  alternate_scc_sift  n_cycles=77, sift_sweeps=2, k_full=2, alpha=0.7,
                               min_block=32, split_fracs=(0.5,0.382,0.618,0.25,0.75)
Scoring: the frozen oracle only (src/mfas/metrics.py, CPU); refinement itself is
best-by-oracle, so no retuned score can regress below its start.

Run (full, ~10-15 min per start on the MacBook):
  $PY biology/analysis/exp_mirror_retune.py
Smoke test of the plumbing (sift 2 sweeps, 1 alt cycle, separate output files):
  $PY biology/analysis/exp_mirror_retune.py --smoke

Outputs -> biology/results/mirror_retune.json
           biology/results/retuned_{champion,mirror,c1b}_rank.npy
           biology/results/{mirror,c1b}_pairs.npy
           biology/outputs/mirror_retune_recovery.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # biology/analysis
REPO = HERE.parent.parent                        # repo root
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    # -- provenance / gate ---------------------------------------------------
    "seed": 42,                       # MUST match exp_mirror_swap.CONFIG["seed"]
    "reference_json": REPO / "layering" / "outputs" / "qa" / "exp_mirror_swap.json",
    "gate_tol": 1e-9,                 # pct values are exact int sums / total; bit-stable
    # -- H64 champion refinement constants (connectome) ----------------------
    # stage 3 (src/mfas/experiments/H64.py: _MAX_SWEEPS/_K_FULL/_ALPHA)
    "sift_max_sweeps": 40,
    "sift_k_full": 6,
    "sift_alpha": 0.7,
    # stage 4 (H64: _ALT_CYCLES/_ALT_SIFT_SWEEPS/_ALT_K_FULL/_ALPHA/_MIN_BLOCK;
    # split_fracs = scc_recursive.DEFAULT_SPLIT_FRACS, tie_break = "first" default)
    "alt_cycles": 77,
    "alt_sift_sweeps": 2,
    "alt_k_full": 2,
    "alt_alpha": 0.7,
    "min_block": 32,
    "split_fracs": (0.5, 0.382, 0.618, 0.25, 0.75),
    # -- outputs -------------------------------------------------------------
    "out_json": HERE.parent / "results" / "mirror_retune.json",
    "out_fig": HERE.parent / "outputs" / "mirror_retune_recovery.png",
    "results_dir": HERE.parent / "results",
    "dpi": 120,
}

# Reference-palette categorical slots 1-3 (dataviz skill, light mode) + ink.
PALETTE = {"champion": "#2a78d6", "mirror": "#eb6834", "c1b": "#1baf7a",
           "ink": "#0b0b0b", "ink_soft": "#52514e", "surface": "#fcfcfb"}


def build_pairings(rank, side, L, R, types, n, seed):
    """Rebuild the mirror pairing and all three controls of exp_mirror_swap.

    PROVENANCE: replicated verbatim from layering/tools/exp_mirror_swap.py
    main() (pairing + controls block), including the ORDER of RNG consumption
    (C1 pool -> C1 subsample -> C1b -> C2 all draw from one default_rng(seed)),
    so every pairing is bit-identical to the original experiment's. The
    original builds them inline in main(), so they are not importable.

    Returns dict: {"mirror", "c1", "c1b", "c2"} -> int pair array (k, 2).
    """
    # ---- mirror pairing: within type, rank-sorted left[i] <-> right[i] ------
    pairs = []
    per_type = {}
    for t in np.unique(types):
        m = types == t
        li = np.where(m & (side == L))[0]
        ri = np.where(m & (side == R))[0]
        if len(li) < 1 or len(ri) < 1:          # CONFIG["min_per_side"] = 1 there
            continue
        li = li[np.argsort(rank[li], kind="stable")]
        ri = ri[np.argsort(rank[ri], kind="stable")]
        k = min(len(li), len(ri))
        pairs.append(np.column_stack([li[:k], ri[:k]]))
        per_type[t] = k
    pairs = np.concatenate(pairs, axis=0)
    n_pairs = len(pairs)

    rng = np.random.default_rng(seed)

    # C1: same-side same-type random disjoint pairs, subsampled to n_pairs
    c1_pool = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            k2 = (len(idx) // 2) * 2
            c1_pool.append(idx[:k2].reshape(-1, 2))
    c1_pool = np.concatenate(c1_pool, axis=0)
    take = rng.choice(len(c1_pool), size=min(n_pairs, len(c1_pool)), replace=False)
    c1_pairs = c1_pool[take]

    # C1b: same-side same-type, rank-matched halves (fair structural control)
    c1b = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            half = len(idx) // 2
            a = idx[:half]
            b = idx[half:2 * half]
            a = a[np.argsort(rank[a], kind="stable")]
            b = b[np.argsort(rank[b], kind="stable")]
            c1b.append(np.column_stack([a, b]))
    c1b_pairs = np.concatenate(c1b, axis=0)

    # C2: random disjoint pairs across the whole brain
    perm = rng.permutation(n)
    c2_pairs = perm[: 2 * n_pairs].reshape(-1, 2)

    return {"mirror": pairs, "c1": c1_pairs, "c1b": c1b_pairs, "c2": c2_pairs}


def regression_gate(ref_json, got, tol):
    """Compare reproduced swapped-order scores against the recorded experiment.

    ``got``: {scheme_key_in_json: (n_pairs, ff_pct)}. Aborts on any mismatch —
    a mismatched pairing would invalidate every downstream number.
    """
    ref = json.loads(Path(ref_json).read_text(encoding="utf-8"))
    print("regression gate vs", ref_json)
    for key, (n_pairs, ff) in got.items():
        r = ref[key]
        ok_n = int(r["n_pairs"]) == int(n_pairs)
        ok_s = abs(float(r["ff_pct"]) - ff) <= tol
        print(f"  {key:42s} n_pairs {n_pairs:7,} (ref {r['n_pairs']:7,})  "
              f"FF {ff:.10f}% (ref {r['ff_pct']:.10f}%)  "
              f"{'OK' if ok_n and ok_s else 'MISMATCH'}")
        if not (ok_n and ok_s):
            raise SystemExit(f"REGRESSION GATE FAILED on {key}: reproduced pairing "
                             f"does not match the recorded experiment. Aborting.")
    print("  gate PASSED — pairing is bit-identical to the recorded experiment\n")


def retune(g, start_rank, cfg):
    """Champion stages 3+4 (production code, champion constants) on a start order.

    No stage-2 Rocket restart: a gradient restart would discard the swap under
    study. Both stages are best-by-oracle, so the result never scores below
    ``start_rank``. Returns (rank, score, info dict).
    """
    from src.mfas.metrics import pct
    from src.mfas.refine.underrelax import sift_underrelaxed
    from src.mfas.refine.scc_recursive import alternate_scc_sift

    t0 = time.time()
    r3, s3, sweep_log = sift_underrelaxed(
        g, start_rank, k_full=cfg["sift_k_full"], alpha=cfg["sift_alpha"],
        max_sweeps=cfg["sift_max_sweeps"])
    t_sift = time.time() - t0
    print(f"    stage 3 done: {pct(s3, g.total_weight):.4f}%  "
          f"({len(sweep_log)} sweeps, {t_sift:.0f}s)")

    t1 = time.time()
    r4, s4, alt_log = alternate_scc_sift(
        g, r3, n_cycles=cfg["alt_cycles"], sift_sweeps=cfg["alt_sift_sweeps"],
        k_full=cfg["alt_k_full"], alpha=cfg["alt_alpha"],
        min_block=cfg["min_block"], split_fracs=cfg["split_fracs"])
    t_alt = time.time() - t1
    print(f"    stage 4 done: {pct(s4, g.total_weight):.4f}%  "
          f"({len(alt_log)} cycles, {t_alt:.0f}s)")

    info = {"after_sift_pct": pct(s3, g.total_weight),
            "after_alt_pct": pct(s4, g.total_weight),
            "sift_sweeps_run": len(sweep_log), "alt_cycles_run": len(alt_log),
            "t_sift_s": t_sift, "t_alt_s": t_alt,
            "t_total_s": time.time() - t0}
    return r4, s4, info


def pair_relation_stats(pairs, rank_orig, rank_new):
    """Fraction of pairs whose within-pair rank relation matches the original.

    ``reverted``: rank_new orders the pair the same way as the champion (the
    swap has been undone in relative terms). ``stayed``: the mirrored relation
    persists. Ranks are permutations, so no ties; the two fractions sum to 1.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    orig = rank_orig[a] < rank_orig[b]
    new = rank_new[a] < rank_new[b]
    reverted = new == orig
    return {"n_pairs": int(len(pairs)),
            "frac_reverted_to_champion": float(reverted.mean()),
            "frac_stayed_swapped": float((~reverted).mean())}


def make_figure(res, cfg):
    """Before->after dumbbell chart per start, champion score marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    champ = res["champion_ff_pct"]
    starts = ["champion", "mirror", "c1b"]
    labels = {"champion": "champion\n(control: refinement's zero)",
              "mirror": "mirror swap\n(L↔R rank pairs)",
              "c1b": "C1b control\n(same-side rank-matched)"}

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.axhline(champ, color=PALETTE["ink_soft"], linestyle="--", linewidth=1.4,
               zorder=1, label=f"champion order ({champ:.3f}%)")

    for i, s in enumerate(starts):
        r = res["starts"][s]
        col = PALETTE[s]
        before, after = r["start_ff_pct"], r["retuned_ff_pct"]
        tight = abs(after - before) < 0.15          # dumbbell too short for arrow
        if not tight:
            ax.annotate("", xy=(i, after), xytext=(i, before),
                        arrowprops=dict(arrowstyle="-|>", color=col, linewidth=2.0,
                                        shrinkA=6, shrinkB=6), zorder=2)
        ax.plot([i], [before], marker="o", markersize=9, markerfacecolor="none",
                markeredgecolor=col, markeredgewidth=2.0, linestyle="none", zorder=3)
        ax.plot([i], [after], marker="o", markersize=9, color=col,
                linestyle="none", zorder=3)
        if tight:
            # collapsed dumbbell: one combined label, delta to the left
            ax.annotate(f"{before:.3f}% → {after:.3f}%", (i, after),
                        textcoords="offset points", xytext=(14, 6),
                        fontsize=9, color=PALETTE["ink"])
            ax.annotate(f"+{after - before:.3f} pp", (i, after),
                        textcoords="offset points", xytext=(-14, 6), ha="right",
                        fontsize=9, color=col)
        else:
            ax.annotate(f"{before:.3f}%", (i, before), textcoords="offset points",
                        xytext=(14, -4), fontsize=9, color=PALETTE["ink_soft"])
            ax.annotate(f"{after:.3f}%", (i, after), textcoords="offset points",
                        xytext=(14, -4), fontsize=9, color=PALETTE["ink"])
            ax.annotate(f"+{after - before:.3f} pp", (i, (before + after) / 2),
                        textcoords="offset points", xytext=(-8, 0), ha="right",
                        fontsize=9, color=col)
    lo = min(r["start_ff_pct"] for r in res["starts"].values())
    hi = max(max(r["retuned_ff_pct"] for r in res["starts"].values()), champ)
    ax.set_ylim(lo - 0.25, hi + 0.35)

    ax.plot([], [], marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=PALETTE["ink_soft"], markeredgewidth=2.0,
            linestyle="none", label="before re-tuning")
    ax.plot([], [], marker="o", markersize=9, color=PALETTE["ink_soft"],
            linestyle="none", label="after stages 3+4 re-tuning")

    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([labels[s] for s in starts], fontsize=9.5,
                       color=PALETTE["ink"])
    ax.set_xlim(-0.5, len(starts) - 0.2)
    ax.set_ylabel("feedforward weight (% of total)", color=PALETTE["ink"])
    ax.set_title("Re-optimizing after the mirror swap: what refinement claws back",
                 loc="left", fontsize=12, color=PALETTE["ink"], pad=12)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["ink_soft"])
    ax.tick_params(colors=PALETTE["ink_soft"])
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    fig.tight_layout()
    cfg["out_fig"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_fig"], dpi=cfg["dpi"], facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"figure -> {cfg['out_fig']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing test: sift 2 sweeps, 1 alt cycle, "
                         "'_smoke' suffixed outputs")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.smoke:
        cfg["sift_max_sweeps"] = 2
        cfg["alt_cycles"] = 1
        cfg["out_json"] = cfg["out_json"].with_name("mirror_retune_smoke.json")
        cfg["out_fig"] = cfg["out_fig"].with_name("mirror_retune_recovery_smoke.png")
    suffix = "_smoke" if args.smoke else ""

    t_start = time.time()
    from layering import io_utils
    from layering.tools.exp_mirror_swap import CONFIG as SWAP_CONFIG, \
        load_types, swapped_rank
    from src.mfas.metrics import score_from_order, pct
    from scipy.stats import spearmanr

    # ---- load graph + champion (parity-gated inside io_utils) ---------------
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    src, tgt, w = g.src, g.tgt, g.weight
    Wtot = float(w.sum())
    print(f"champion: FF = {ff_pct:.6f}%  (n = {g.n_nodes:,}, m = {len(w):,})")

    # ---- annotations, exactly as the original experiment loaded them --------
    d = np.load(SWAP_CONFIG["ann_npz"])
    meta = json.loads(Path(SWAP_CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    sidev = meta["side_values"]
    side = d["side"]
    L, R = sidev.index("left"), sidev.index("right")
    types = load_types(g.node_ids)
    n = rank.shape[0]

    # ---- rebuild pairings + regression gate ---------------------------------
    pairings = build_pairings(rank, side, L, R, types, n, cfg["seed"])
    swapped = {k: swapped_rank(rank, v) for k, v in pairings.items()}

    def score_of(r):
        return pct(score_from_order(r, src, tgt, w), Wtot)

    json_key = {"mirror": "mirror_swap",
                "c1": "control_same_side_same_type_random",
                "c1b": "control_same_side_same_type_rank_matched",
                "c2": "control_random_pairs"}
    start_scores = {k: score_of(swapped[k]) for k in pairings}
    regression_gate(cfg["reference_json"],
                    {json_key[k]: (len(pairings[k]), start_scores[k])
                     for k in pairings},
                    cfg["gate_tol"])

    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    np.save(cfg["results_dir"] / "mirror_pairs.npy", pairings["mirror"])
    np.save(cfg["results_dir"] / "c1b_pairs.npy", pairings["c1b"])

    # ---- the three starts ----------------------------------------------------
    starts = {"champion": rank.copy(),
              "mirror": swapped["mirror"],
              "c1b": swapped["c1b"]}

    res = {
        "champion_ff_pct": ff_pct,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cfg.items()},
        "smoke": bool(args.smoke),
        "regression_gate": "PASSED (all four schemes bit-identical to "
                           "layering/outputs/qa/exp_mirror_swap.json)",
        "starts": {},
    }

    retuned_ranks = {}
    for name, r0 in starts.items():
        s0 = score_of(r0)
        print(f"[{name}] start FF = {s0:.4f}%  — running stages 3+4 ...")
        r_new, s_new, info = retune(g, r0, cfg)
        s_new_pct = pct(s_new, Wtot)
        deficit = ff_pct - s0
        recovered = s_new_pct - s0
        entry = {
            "start_ff_pct": s0,
            "retuned_ff_pct": s_new_pct,
            "deficit_vs_champion_pp": deficit,
            "recovered_pp": recovered,
            "recovered_frac_of_deficit": (recovered / deficit
                                          if deficit > 1e-12 else None),
            "residual_vs_champion_pp": ff_pct - s_new_pct,
            **info,
        }
        res["starts"][name] = entry
        retuned_ranks[name] = r_new
        np.save(cfg["results_dir"] / f"retuned_{name}_rank{suffix}.npy",
                r_new.astype(np.int64))
        # partial write after every start, so a crash loses nothing
        cfg["out_json"].write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[{name}] retuned FF = {s_new_pct:.4f}%  "
              f"(recovered {recovered:+.4f} pp of {deficit:.4f} pp deficit)\n")

    # ---- headline gaps -------------------------------------------------------
    ra = res["starts"]["champion"]["retuned_ff_pct"]
    rb = res["starts"]["mirror"]["retuned_ff_pct"]
    rc = res["starts"]["c1b"]["retuned_ff_pct"]
    res["headline"] = {
        "irrecoverable_mirror_gap_pp": ra - rb,     # retuned(a) - retuned(b)
        "structural_control_gap_pp": ra - rc,       # retuned(a) - retuned(c)
        "mirror_specific_residual_pp": rc - rb,     # what mirroring costs beyond
                                                    # the same-structure control
    }

    # ---- un-swap analysis ----------------------------------------------------
    rho_swap_vs_champ = spearmanr(swapped["mirror"], rank).statistic
    rho_ret_vs_champ = spearmanr(retuned_ranks["mirror"], rank).statistic
    rho_ret_vs_start = spearmanr(retuned_ranks["mirror"], swapped["mirror"]).statistic
    res["unswap_analysis"] = {
        "mirror_pairs_after_mirror_retune":
            pair_relation_stats(pairings["mirror"], rank, retuned_ranks["mirror"]),
        "c1b_pairs_after_c1b_retune":
            pair_relation_stats(pairings["c1b"], rank, retuned_ranks["c1b"]),
        "spearman_swapped_start_vs_champion": float(rho_swap_vs_champ),
        "spearman_retuned_mirror_vs_champion": float(rho_ret_vs_champ),
        "spearman_retuned_mirror_vs_swapped_start": float(rho_ret_vs_start),
        "mean_abs_rank_shift_retuned_mirror_vs_champion":
            float(np.abs(retuned_ranks["mirror"] - rank).mean()),
        "mean_abs_rank_shift_retuned_mirror_vs_swapped_start":
            float(np.abs(retuned_ranks["mirror"] - swapped["mirror"]).mean()),
    }

    res["seconds_total"] = time.time() - t_start
    res["rerun_command"] = ("$PY biology/analysis/exp_mirror_retune.py"
                            + (" --smoke" if args.smoke else ""))
    cfg["out_json"].write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"results -> {cfg['out_json']}")

    make_figure(res, cfg)

    h = res["headline"]
    print(f"\nHEADLINE  irrecoverable mirror gap  = {h['irrecoverable_mirror_gap_pp']:+.4f} pp")
    print(f"          structural control gap    = {h['structural_control_gap_pp']:+.4f} pp")
    print(f"          mirror-specific residual  = {h['mirror_specific_residual_pp']:+.4f} pp")
    u = res["unswap_analysis"]
    print(f"          mirror pairs reverted to champion relation: "
          f"{100 * u['mirror_pairs_after_mirror_retune']['frac_reverted_to_champion']:.1f}%")
    print(f"done in {res['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
