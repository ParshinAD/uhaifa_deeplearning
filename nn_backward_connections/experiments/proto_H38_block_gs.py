"""H38 prototype gate — block-sequential (Gauss-Seidel) exact-gain sift vs the champion's Jacobi.

The mechanism
-------------
``mfas.refine.underrelax.sift_underrelaxed`` is a JACOBI iteration: every node's exact-optimal
gap is computed against ONE fixed rank vector, and then all movers are applied simultaneously.
Two movers that each individually improve can therefore collide — each computed its gain
assuming the other stayed put. Under-relaxation (``alpha = 0.7``) is the textbook damping cure
for the resulting 2-cycle, and it is what the champion ships.

Gauss-Seidel is the other textbook cure and usually the stronger one: re-insert each node against
the CURRENT order, so later movers see earlier ones. Pure GS destroys the vectorisation, so the
practical form is **block-sequential**: partition the nodes into ``K`` blocks, do a full Jacobi
step within a block, apply it, then move to the next block against the updated rank.
``K = 1`` is exactly the champion's Jacobi; ``K = n`` is pure Gauss-Seidel. ``K`` is a knob that
interpolates between them.

What this gate decides
----------------------
Does block-sequential reach a strictly BETTER FIXED POINT than the champion's under-relaxed
Jacobi, from the same starting order? Cost is deliberately a second question: this prototype
calls the full ``jacobi_best_gaps`` once per block and uses only that block's entries, which is
``K`` times more work than necessary. If the fixed point is not better, the cost question never
arises and the family closes cheaply. If it IS better, the kernel can be restricted to a block's
incident edges (a block of ``n/K`` nodes touches ``~m/K`` edges), which recovers the O(m) sweep.

Both arms start from the SAME order — the greedy-FAS warm start — so no GPU is involved and the
comparison isolates the rebuild rule. The champion's stage 3 from that order reaches
**83.44721 %** on connectome (measured 2026-08-16 as the ``epochs=0`` arm of
``experiments/outputs/proto_S01_connectome.json``), which the ``K = 1`` arm here must reproduce
exactly. That parity check is the reason ``K = 1`` is in the grid.

Leakage-safety: the gap/gain kernel reads only ranks and edge weights; the frozen oracle scores
whole candidate rank vectors for the log and the best-by-oracle choice, exactly as the production
sift does. ``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H38_block_gs.py [dataset] [max_sweeps]
Out:  experiments/outputs/proto_H38_<dataset>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                            # noqa: E402
from mfas.metrics import pct, score_from_order                 # noqa: E402
from mfas.refine.insertion import jacobi_best_gaps             # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed, underrelaxed_rebuild  # noqa: E402
from mfas.experiments.H02 import greedy_fas_order              # noqa: E402
from mfas.experiments import H42                               # noqa: E402

BLOCK_COUNTS = (1, 2, 4, 8)


def block_sequential_sift(g, init_rank, *, n_blocks, k_full, alpha, max_sweeps):
    """Block-sequential under-relaxed exact-gain sift.

    One SWEEP = ``n_blocks`` sub-steps. Each sub-step recomputes every node's exact best gap
    against the CURRENT rank, but only the nodes in that block are allowed to move. With
    ``n_blocks == 1`` this is bit-identical to :func:`sift_underrelaxed`.

    Blocks are contiguous ranges of the CURRENT POSITION, not of node id: the point of the
    ordering is positional, so a positional partition is what makes "later movers see earlier
    ones" mean something. The partition is recomputed each sub-step from the live rank.

    Best-by-oracle, exactly as the production sift: the working order is only accepted as best
    when it scores higher, so this can never return worse than ``init_rank``.
    """
    src, tgt, w = np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight)
    n, total = g.n_nodes, g.total_weight

    work = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(work, src, tgt, g.weight)
    best_rank = work.copy()
    log = []

    for s in range(max_sweeps):
        a = 1.0 if s < k_full else float(alpha)
        t0 = time.time()
        movers_total = 0
        for b in range(n_blocks):
            best_gap, gain = jacobi_best_gaps(work, src, tgt, w, n)
            # restrict the move set to the nodes currently sitting in block b
            lo = (n * b) // n_blocks
            hi = (n * (b + 1)) // n_blocks
            in_block = (work >= lo) & (work < hi)
            masked_gain = np.where(in_block, gain, 0.0)
            movers_total += int((masked_gain > 1e-9).sum())
            work = underrelaxed_rebuild(work, best_gap, masked_gain, a)
        wall = time.time() - t0

        cand = score_from_order(work, src, tgt, g.weight)
        accepted = cand > best_score
        if accepted:
            best_score, best_rank = cand, work.copy()
        log.append(dict(sweep=s, alpha=a, candidate_pct=pct(cand, total),
                        best_pct=pct(best_score, total), accepted=bool(accepted),
                        n_movers=movers_total, wall=wall))
        if movers_total == 0:
            break
    return best_rank, float(best_score), log


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    max_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else H42._MAX_SWEEPS.get("connectome", 40)

    g = io.load_dataset(dataset)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    k_full, alpha = H42._K_FULL, H42._ALPHA

    t0 = time.time()
    rank0 = np.asarray(greedy_fas_order(g), dtype=np.int64)   # NOTE: returns a RANK vector
    t_greedy = time.time() - t0
    start_pct = pct(score_from_order(rank0, src, tgt, g.weight), total)
    print(f"start (greedy-FAS): {start_pct:.5f}%  {t_greedy:.1f}s", flush=True)

    # Reference arm: the production sift itself, so the parity target is produced here and not
    # quoted from another file.
    t0 = time.time()
    _, ref_score, ref_log = sift_underrelaxed(g, rank0, k_full=k_full, alpha=alpha,
                                              max_sweeps=max_sweeps, time_budget_s=None)
    t_ref = time.time() - t0
    ref_pct = pct(ref_score, total)
    print(f"production sift_underrelaxed: {ref_pct:.5f}%  {t_ref:.1f}s  "
          f"sweeps={len(ref_log)}", flush=True)

    # Two damping settings per block count, because using ONE of them would confound the test.
    #
    #   "damped"    alpha = 0.7 after k_full warm sweeps - the champion's schedule.
    #   "undamped"  alpha = 1.0 throughout (k_full = max_sweeps).
    #
    # Under-relaxation is the textbook cure for the JACOBI collision: two movers that each
    # computed their gain assuming the other stayed put. Gauss-Seidel does not have that
    # collision by construction - later movers see earlier ones - so damping a block-sequential
    # arm treats a disease it does not have and may simply handicap it. Testing only the damped
    # form would kill the hypothesis on a confound rather than on its merits.
    rows = []
    for K in BLOCK_COUNTS:
        for tag_name, kf in (("damped", k_full), ("undamped", max_sweeps)):
            if K == 1 and tag_name == "undamped":
                continue                      # K=1 undamped is plain Jacobi, not the comparator
            t0 = time.time()
            _, sc, lg = block_sequential_sift(g, rank0, n_blocks=K, k_full=kf,
                                              alpha=alpha, max_sweeps=max_sweeps)
            dt = time.time() - t0
            row = dict(n_blocks=K, damping=tag_name, pct=pct(sc, total), wall_s=dt,
                       sweeps=len(lg),
                       delta_vs_production_pp=pct(sc, total) - ref_pct,
                       final_movers=lg[-1]["n_movers"] if lg else 0)
            rows.append(row)
            tag = "  <- must equal the production arm" if K == 1 else ""
            print(f"K={K:>2} {tag_name:<8}: {row['pct']:.5f}%  {dt:.1f}s  "
                  f"sweeps={row['sweeps']}  "
                  f"delta={row['delta_vs_production_pp']:+.5f} pp{tag}", flush=True)

    k1 = [r for r in rows if r["n_blocks"] == 1][0]
    parity = abs(k1["delta_vs_production_pp"]) < 1e-9
    best = max(rows, key=lambda r: r["pct"])
    verdict = ("PASS" if best["delta_vs_production_pp"] > 0.012
               else "FAIL (below the 0.012 pp minimum effect size)")
    print(f"PARITY K=1 vs production: {'OK' if parity else 'BROKEN'}", flush=True)
    print(f"BEST: K={best['n_blocks']} {best['delta_vs_production_pp']:+.5f} pp -> {verdict}",
          flush=True)

    out = dict(item="H38", dataset=dataset, start_pct=start_pct,
               k_full=k_full, alpha=alpha, max_sweeps=max_sweeps,
               production_pct=ref_pct, production_wall_s=t_ref,
               production_sweeps=len(ref_log),
               block_counts=list(BLOCK_COUNTS), arms=rows,
               parity_k1_ok=bool(parity), best_delta_pp=best["delta_vs_production_pp"],
               verdict=verdict)
    op = _ROOT / "experiments" / "outputs" / f"proto_H38_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
