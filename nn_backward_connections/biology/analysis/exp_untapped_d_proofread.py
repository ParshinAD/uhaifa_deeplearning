"""Item D — MICrONS proofread-axon control (MICRONS_FINDINGS §8.3, never run).

Only 249 of 67,534 MICrONS neurons have a proofread axon; every truncation caveat
in MICRONS_FINDINGS.md points to the 245-node proofread-axon subgraph
(data/processed/microns_proofread.npz: both endpoints proofread, n=245, m=2,424,
total weight 5,466) as the built-in control. This script finally orders it and asks:

  1. FF% of the directly-optimized proofread subgraph vs the FF% of the full-graph
     champion order (pinned H42) RESTRICTED to those 245 nodes;
  2. Spearman rank correlation between the two orders;
  3. do the two headline biological effects survive inside this bias-free subgraph:
     (a) the depth trend (excitatory rank increases with soma depth, full-graph
         rho = +0.356), and
     (b) the interneurons-first effect (inhibitory cells sorted earlier than
         excitatory ones)?

Optimization = greedy-FAS init (src/mfas/experiments/H02.greedy_fas_order) + the
champion stage-3 sift (src/mfas/refine/underrelax.sift_underrelaxed with the H35/H64
constants k_full=6, alpha=0.7, max_sweeps=40), plus CONFIG["n_restarts"] random
permutation restarts through the same sift; best-by-frozen-oracle kept. All scoring
via the frozen oracle (src/mfas/metrics.py). CPU-only, seconds.

Honesty: n = 245 (E/I n reported per class); rank statistics only (Spearman,
Mann-Whitney U) with permutation p-values; this is a robustness check of effect
DIRECTIONS, not a discovery engine.

Run:  $PY biology/analysis/exp_untapped_d_proofread.py
Output: biology/results/untapped/untapped_D_proofread_control.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = dict(
    proofread_npz=str(REPO / "data/processed/microns_proofread.npz"),
    ann_npz=str(REPO / "layering/outputs/microns_ann/node_ann.npz"),
    sift=dict(k_full=6, alpha=0.7, max_sweeps=40),   # H35/H64 champion constants
    n_restarts=16,                                   # random-permutation inits
    seed=42,                                         # master seed for restarts
    n_perm=20000,                                    # label permutations for p-values
    out_json=str(REPO / "biology/results/untapped/untapped_D_proofread_control.json"),
)


def spearman_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """Spearman rho and a two-sided permutation p-value (shuffle y)."""
    rho = stats.spearmanr(x, y).statistic
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = stats.spearmanr(x, rng.permutation(y)).statistic
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (n_perm + 1)
    return float(rho), float(p)


def main() -> None:
    from src.mfas.io import GraphData
    from src.mfas.metrics import pct, score_from_order
    from src.mfas.experiments.H02 import greedy_fas_order
    from src.mfas.refine.underrelax import sift_underrelaxed
    from layering import io_utils

    npz = np.load(CONFIG["proofread_npz"])
    g = GraphData(src=npz["src"], tgt=npz["tgt"], weight=npz["weight"],
                  node_ids=npz["node_ids"], name="microns_proofread")
    n, total = g.n_nodes, g.total_weight
    print(g)

    # ── direct optimization of the 245-node subgraph ─────────────────────────
    t0 = time.time()
    runs = []
    rank_greedy = greedy_fas_order(g)
    greedy_pct = pct(score_from_order(rank_greedy, g.src, g.tgt, g.weight), total)
    best_rank, best_score, _ = sift_underrelaxed(g, rank_greedy, **CONFIG["sift"])
    runs.append(dict(init="greedy_fas", seed=None,
                     pct=pct(best_score, total)))
    rng = np.random.default_rng(CONFIG["seed"])
    restart_seeds = [int(s) for s in rng.integers(0, 2**31 - 1,
                                                  CONFIG["n_restarts"])]
    for s in restart_seeds:
        r0 = np.random.default_rng(s).permutation(n).astype(np.int64)
        r, sc, _ = sift_underrelaxed(g, r0, **CONFIG["sift"])
        runs.append(dict(init="random_perm", seed=s, pct=pct(sc, total)))
        if sc > best_score:
            best_rank, best_score = r, sc
    best_pct = pct(best_score, total)
    opt_seconds = time.time() - t0
    restart_pcts = [r["pct"] for r in runs]

    # ── full-graph champion (H42) restricted to the same 245 nodes ───────────
    g_full, _, rank_full, ff_full = io_utils.load_graph_and_champion_order("microns")
    idx_full = np.searchsorted(g_full.node_ids, g.node_ids)
    assert np.array_equal(g_full.node_ids[idx_full], g.node_ids), \
        "proofread node_ids not all present in the full microns graph"
    rank_restricted = np.argsort(np.argsort(rank_full[idx_full])).astype(np.int64)
    restr_pct = pct(score_from_order(rank_restricted, g.src, g.tgt, g.weight), total)

    rho_orders = stats.spearmanr(best_rank, rank_restricted)

    # ── biology inside the control: depth trend and I-before-E ───────────────
    ann = np.load(CONFIG["ann_npz"])
    depth = ann["depth_um"][idx_full].astype(np.float64)
    ei = ann["ei"][idx_full]              # 0 NA / 1 E / 2 I (microns_ann_prep.py)
    axon_pr = ann["axon_pr"][idx_full]
    n_E, n_I, n_NA = int((ei == 1).sum()), int((ei == 2).sum()), int((ei == 0).sum())
    assert int(axon_pr.sum()) == n, "all 245 control nodes should be proofread-axon"

    rng_p = np.random.default_rng(CONFIG["seed"] + 1)
    res = {}
    for label, r in (("direct_opt", best_rank), ("champion_restricted", rank_restricted)):
        e_mask, i_mask = ei == 1, ei == 2
        rho_all, p_all = spearman_perm_p(r, depth, CONFIG["n_perm"], rng_p)
        rho_e, p_e = spearman_perm_p(r[e_mask], depth[e_mask], CONFIG["n_perm"], rng_p)
        mwu = stats.mannwhitneyu(r[i_mask], r[e_mask], alternative="less")
        res[label] = dict(
            depth_spearman_all=dict(rho=rho_all, p_perm=p_all, n=n),
            depth_spearman_exc=dict(rho=rho_e, p_perm=p_e, n=n_E),
            i_before_e=dict(
                median_rank_I=float(np.median(r[i_mask])),
                median_rank_E=float(np.median(r[e_mask])),
                n_I=n_I, n_E=n_E,
                mannwhitney_U=float(mwu.statistic),
                p_one_sided_I_earlier=float(mwu.pvalue)),
        )

    # ── composition + class-pair weight budget inside the control ────────────
    # (context for interpreting the shifts: the proofread set is NOT a random
    # sample of cortex — proofreading prioritized interneurons.)
    cls = np.where(ei == 1, "E", np.where(ei == 2, "I", "NA"))
    pair_budget = {}
    ff_edge = best_rank[g.tgt] > best_rank[g.src]
    for a in ("E", "I"):
        for b in ("E", "I"):
            m = (cls[g.src] == a) & (cls[g.tgt] == b)
            wsum = float(g.weight[m].sum())
            pair_budget[f"{a}->{b}"] = dict(
                weight=wsum,
                pct_of_total=100.0 * wsum / total,
                ff_share_pct=(100.0 * float(g.weight[m & ff_edge].sum()) / wsum
                              if wsum else None))
    depth_summary = dict(
        median_um=float(np.median(depth)),
        q25_um=float(np.percentile(depth, 25)),
        q75_um=float(np.percentile(depth, 75)),
        full_graph_median_um=float(np.median(ann["depth_um"])),
        inhibitory_frac=n_I / n,
        full_graph_inhibitory_frac=float((ann["ei"] == 2).sum() / ann["ei"].shape[0]),
    )

    out = dict(
        config=CONFIG,
        graph=dict(n=n, m=g.n_edges, total_weight=total),
        greedy_init_pct=float(greedy_pct),
        direct_opt=dict(best_pct=float(best_pct),
                        runs=runs,
                        restart_pct_min=float(min(restart_pcts)),
                        restart_pct_max=float(max(restart_pcts)),
                        seconds=opt_seconds),
        champion_restricted_pct=float(restr_pct),
        full_graph_champion_pct=float(ff_full),
        order_spearman=dict(rho=float(rho_orders.statistic),
                            p=float(rho_orders.pvalue)),
        class_counts=dict(E=n_E, I=n_I, NA=n_NA),
        composition=depth_summary,
        ei_pair_budget_under_direct_opt=pair_budget,
        effects=res,
    )
    Path(CONFIG["out_json"]).write_text(json.dumps(out, indent=1))

    print(f"\ngreedy init            {greedy_pct:.3f}%")
    print(f"direct opt best        {best_pct:.3f}%  "
          f"(restarts {min(restart_pcts):.3f}..{max(restart_pcts):.3f}, "
          f"{opt_seconds:.1f}s)")
    print(f"champion restricted    {restr_pct:.3f}%   (full-graph champion {ff_full:.4f}%)")
    print(f"order Spearman rho     {rho_orders.statistic:.3f}")
    print(f"classes E {n_E} / I {n_I} / NA {n_NA}  "
          f"(I frac {n_I/n:.2f} vs full graph "
          f"{depth_summary['full_graph_inhibitory_frac']:.3f})")
    print(f"depth median {depth_summary['median_um']:.0f}um "
          f"(full graph {depth_summary['full_graph_median_um']:.0f}um)")
    for k, v in pair_budget.items():
        print(f"  {k}: {v['pct_of_total']:5.1f}% of weight, "
              f"FF {v['ff_share_pct']:.1f}%" if v["ff_share_pct"] is not None
              else f"  {k}: 0")
    for label, r in res.items():
        print(f"\n[{label}]")
        print(f"  depth rho (all) {r['depth_spearman_all']['rho']:+.3f} "
              f"p={r['depth_spearman_all']['p_perm']:.4f}")
        print(f"  depth rho (E)   {r['depth_spearman_exc']['rho']:+.3f} "
              f"p={r['depth_spearman_exc']['p_perm']:.4f}")
        ie = r["i_before_e"]
        print(f"  I median {ie['median_rank_I']:.0f} vs E median "
              f"{ie['median_rank_E']:.0f}  (U p={ie['p_one_sided_I_earlier']:.4f})")
    print(f"\nwrote {CONFIG['out_json']}")


if __name__ == "__main__":
    main()
