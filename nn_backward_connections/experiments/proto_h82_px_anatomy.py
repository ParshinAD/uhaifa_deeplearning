"""H82 rung 0b - WHY the best-of-2^k gain is exactly zero, and H83's pre-gate.

Rung 0 found k up to 10,553 common blocks and a realised gain of EXACTLY 0.000000 pp
on all 78 pairs.  This script measures the reason, which is the transferable product:
the block partition is dominated by SINGLETONS (which hold no internal edge and so are
ties by construction) plus ONE giant indivisible core block, so the 2^k offspring set
collapses to the 2 parents.

It also re-runs the D(p) = |prefix_A(p) delta prefix_B(p)| near-cut profile on the pairs
that actually matter (max-k and champion-involving), which rung 0 reported only for the
arbitrary top-5 of an all-zero gain ordering - that is H83's free pre-gate.

Rung 0b additionally runs the DIAGNOSTIC census against data/best_solution, read only
through mfas.analysis.gap: how much of M15's +0.356498 pp residual is block-decomposable?
That arm is diagnostic ONLY and may never enter a scored variant (labelled in the JSON).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from mfas.io import load_dataset
from mfas.metrics import score_from_order
from mfas.analysis.gap import load_best_solution

sys.path.insert(0, "experiments")
from proto_h82_px_census import (ranks_from_positions, common_blocks, sym_diff_profile,
                             px_pair)
import proto_h82_px_census as R


def block_report(g, rank_a, order_a, rank_b, wi, total_w):
    """Block-size and edge-mass anatomy of one pair's common-block partition."""
    n = rank_a.shape[0]
    ends = common_blocks(rank_a, order_a, rank_b)
    k = ends.shape[0]
    sizes = np.diff(np.concatenate(([-1], ends))).astype(np.int64)

    block_of_rank = np.zeros(n, dtype=np.int64)
    if k > 1:
        block_of_rank[ends[:-1] + 1] = 1
        block_of_rank = np.cumsum(block_of_rank)
    block_of_node = np.empty(n, dtype=np.int64)
    block_of_node[order_a] = block_of_rank

    same = block_of_node[g.src] == block_of_node[g.tgt]
    # edge mass that PX can actually re-decide = mass strictly inside a block of size > 1
    big = sizes > 1
    inside_big = same & big[block_of_node[g.src]]
    giant = int(np.argmax(sizes))
    in_giant = same & (block_of_node[g.src] == giant)

    deg = np.bincount(g.src, minlength=n) + np.bincount(g.tgt, minlength=n)
    singleton_nodes = np.isin(block_of_node, np.nonzero(sizes == 1)[0])

    return dict(
        k=int(k),
        n_blocks_size1=int(np.sum(sizes == 1)),
        n_blocks_size_gt1=int(np.sum(big)),
        n_nodes_in_singletons=int(np.sum(sizes[sizes == 1])),
        giant_block_size=int(sizes.max()),
        giant_block_frac_nodes=float(sizes.max()) / n,
        second_largest_block_size=int(np.sort(sizes)[-2]) if k > 1 else 0,
        weight_inside_giant_pct=100.0 * float(wi[in_giant].sum()) / total_w,
        weight_decidable_by_px_pct=100.0 * float(wi[inside_big].sum()) / total_w,
        weight_cross_block_pct=100.0 * float(wi[~same].sum()) / total_w,
        mean_deg_singleton_nodes=float(deg[singleton_nodes].mean()) if singleton_nodes.any() else 0.0,
        mean_deg_giant_nodes=float(deg[~singleton_nodes].mean()),
        median_deg_singleton_nodes=float(np.median(deg[singleton_nodes])) if singleton_nodes.any() else 0.0,
        median_deg_giant_nodes=float(np.median(deg[~singleton_nodes])),
    )


def main():
    t0 = time.time()
    g = load_dataset("connectome")
    R.INTEGRAL = True
    n, total_w = g.n_nodes, float(np.asarray(g.weight, dtype=np.float64).sum())
    wi = np.asarray(g.weight).astype(np.int64)

    prev = json.load(open("experiments/outputs/proto_H82_rung0.json"))
    want = {e["variant"]: e["file"] for e in prev["pool"]}
    pool = {}
    for v, f in want.items():
        rank, order = ranks_from_positions(np.load("results/" + f))
        pool[v] = dict(rank=rank, order=order, file=f,
                       score=int(score_from_order(rank, g.src, g.tgt, g.weight)))
    print("[%6.1fs] loaded %d distinct parents" % (time.time() - t0, len(pool)), flush=True)

    # the pairs that matter: every pair involving the champion, every pair among the
    # structurally-distinct H79 basins, and the max-k pairs rung 0 found.
    champ = "H64"
    basins = ["H64", "H79A", "H79B", "H79C", "H73"]
    interest = set()
    for v in pool:
        if v != champ:
            interest.add((champ, v))
    for i, a in enumerate(basins):
        for b in basins[i + 1:]:
            interest.add((a, b))
    for r in sorted(prev["top_pairs"], key=lambda r: -r["k"])[:6]:
        interest.add((r["a"], r["b"]))

    rows = []
    for a, b in sorted(interest):
        A, B = pool[a], pool[b]
        px = px_pair(g, A["rank"], A["order"], B["rank"])
        rep = block_report(g, A["rank"], A["order"], B["rank"], wi, total_w)
        D = sym_diff_profile(A["rank"], A["order"], B["rank"])
        rep.update(a=a, b=b,
                   score_a=A["score"], score_b=B["score"],
                   gain_pp=100.0 * px["gain_over_max_parents"] / total_w,
                   n_nontrivial_choices=px["n_nontrivial_choices"],
                   a_wins=px["a_wins"], b_wins=px["b_wins"],
                   best_pct=100.0 * px["best_of_2k"] / total_w,
                   # H83 free pre-gate
                   n_D_eq_0=int(np.sum(D == 0)), n_D_le_2=int(np.sum(D <= 2)),
                   n_D_le_8=int(np.sum(D <= 8)), n_D_le_32=int(np.sum(D <= 32)),
                   n_D_le_128=int(np.sum(D <= 128)),
                   median_D=float(np.median(D)), max_D=int(D.max()))
        rows.append(rep)
        print("[%6.1fs] %-5sx%-5s k=%-6d sz>1=%-3d giant=%-6d (%.1f%% nodes) "
              "PX-decidable mass=%.3f%% nontriv=%d gain=%.8f D<=32:%d"
              % (time.time() - t0, a, b, rep["k"], rep["n_blocks_size_gt1"],
                 rep["giant_block_size"], 100 * rep["giant_block_frac_nodes"],
                 rep["weight_decidable_by_px_pct"], rep["n_nontrivial_choices"],
                 rep["gain_pp"], rep["n_D_le_32"]), flush=True)

    # --- DIAGNOSTIC ONLY: the champion against the downloaded reference solution.
    # Read through mfas.analysis.gap; measurement only, never inside an algorithm.
    ref = load_best_solution(g)
    ref_pos = ref[0] if isinstance(ref, tuple) else ref
    ref_rank, ref_order = ranks_from_positions(np.asarray(ref_pos, dtype=np.float64))
    ref_score = int(score_from_order(ref_rank, g.src, g.tgt, g.weight))
    A = pool[champ]
    px = px_pair(g, A["rank"], A["order"], ref_rank)
    rep = block_report(g, A["rank"], A["order"], ref_rank, wi, total_w)
    D = sym_diff_profile(A["rank"], A["order"], ref_rank)
    diag = dict(
        LABEL="DIAGNOSTIC ONLY - uses data/best_solution; may never enter a scored variant",
        a=champ, b="best_solution",
        ref_pct=100.0 * ref_score / total_w,
        champion_pct=100.0 * A["score"] / total_w,
        residual_pp=100.0 * (ref_score - A["score"]) / total_w,
        gain_over_max_parents_pp=100.0 * px["gain_over_max_parents"] / total_w,
        n_nontrivial_choices=px["n_nontrivial_choices"],
        decomposition_exact=(px["score_a_decomposed"] == A["score"]
                             and px["score_b_decomposed"] == ref_score),
        n_D_le_32=int(np.sum(D <= 32)), median_D=float(np.median(D)),
    )
    diag.update({k: v for k, v in rep.items() if k not in ("a", "b")})
    print("\n[%6.1fs] DIAGNOSTIC champion x reference: k=%d sz>1=%d giant=%d (%.1f%% nodes) "
          "PX-decidable mass=%.3f%% gain=%.8f pp of a %.6f pp residual"
          % (time.time() - t0, rep["k"], rep["n_blocks_size_gt1"], rep["giant_block_size"],
             100 * rep["giant_block_frac_nodes"], rep["weight_decidable_by_px_pct"],
             diag["gain_over_max_parents_pp"], diag["residual_pp"]), flush=True)

    out = dict(item="H82", rung="0b", dataset="connectome", generated="2026-08-28",
               n_nodes=int(n), total_weight=total_w,
               champion="H64", champion_pct=84.25817950936937, screen_bar_pp=0.012,
               pairs=rows, diagnostic_vs_reference=diag,
               max_D_le_32_over_pairs=max(r["n_D_le_32"] for r in rows),
               H83_pre_gate_threshold="fewer than 50 positions with D(p) <= 32 kills H83",
               wall_clock_s=time.time() - t0)
    Path("experiments/outputs").mkdir(parents=True, exist_ok=True)
    with open("experiments/outputs/proto_H82_rung0b.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[%6.1fs] WROTE experiments/outputs/proto_H82_rung0b.json" % (time.time() - t0),
          flush=True)
    print("H83 pre-gate: best pair has %d positions with D(p) <= 32 (needs >= 50)"
          % out["max_D_le_32_over_pairs"], flush=True)


if __name__ == "__main__":
    main()
