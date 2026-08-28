"""Prototype gate for H73 - the sift's argmax tie-break is direction-ASYMMETRIC.

The hypothesis
--------------
``jacobi_best_gaps`` (src/mfas/refine/insertion.py) builds every node's EXACT insertion
profile: a piecewise-constant function of the insertion gap ``g in [0, n-1]`` with at most
``deg(u)`` breakpoints over 136,648 gaps. Its maximum is therefore attained on an
INTERVAL (a "plateau"), not a point, whenever two consecutive breakpoints are far apart.

The current rule picks the FIRST argmax breakpoint (insertion.py:205-209, "ties -> smallest
b, fine"), and gap 0 wins any tie against the breakpoint maximum (:236, ``use_bp = seg_max >
gap0_val``, strict). Both choices are LEFTWARD. That is direction-asymmetric:

* a node whose optimal plateau lies to its RIGHT lands on ``lo``, the NEAR edge -> minimum
  displacement;
* a node whose optimal plateau lies to its LEFT lands on ``lo``, the FAR edge -> maximum
  displacement.

Under-relaxation makes the choice load-bearing rather than cosmetic: ``underrelaxed_rebuild``
moves each mover to ``rank + alpha*((best_gap - 0.5) - rank)``, so the selected gap is the
INTERPOLATION TARGET, and picking the far end of a plateau transports the node ``alpha *
width`` further than picking the near end - for the same exact gain.

RUNG 1 (this script, ``--rung 1``) measures the premise only. It computes, for a given rank
vector, the FULL maximizing gap set per node - as a union of intervals - and reports plateau
widths, the mover direction split, and the displacement the current rule spends versus the
minimum-|displacement| choice among the SAME optimal gaps. It changes no algorithm and
predicts no score.

FREE PRE-GATE (the item's own kill condition): if fewer than 1% of movers per sweep have an
optimal plateau of width > 1, the premise is false, the item dies for free, and meta-rule
M5-no-ties extends from continuous position ties to discrete insertion-profile plateaus.

Novelty note (M5 / H09)
-----------------------
``killed.json`` M5-no-ties reads "the continuous optimizer leaves 0 exact position ties ...
nothing is recoverable from tie-breaking", and H09's revival condition is "never on these
datasets". Its evidence (findings.md #2, the H09 kill) measures ANTI-TIE JITTER ON NEAR-EQUAL
CONTINUOUS POSITIONS. The object here is a different one: exact ties in the argmax of the
DISCRETE insertion profile, which has plateaus by construction because it has at most deg(u)
breakpoints over n gaps. M5 cannot bind on an object it did not measure - but this rung
measures that object FIRST, at zero GPU cost, so if M5's spirit is right the item dies here
and M5 is strengthened rather than evaded.

Leakage-safety
--------------
Everything below is a function of the input edge list and a rank vector. The frozen oracle
(``mfas.metrics.score_from_order``) is read only to report the score of an order that was
already produced; it never enters a gap choice. ``data/best_solution`` is never touched.

Reproduce
---------
    PYTHONPATH=src python experiments/proto_H73_tiebreak.py --rung 1 --dataset connectome
    PYTHONPATH=src python experiments/proto_H73_tiebreak.py --rung 1 --dataset mouse
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mfas.experiments.H02 import greedy_fas_order
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_order
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps
from mfas.refine.underrelax import underrelaxed_rebuild

SENTINEL = np.int64(1) << np.int64(62)


def profile_plateaus(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                     w: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """Exact insertion profile per node, WITH the full maximizing gap set.

    Replicates ``jacobi_best_gaps``' event construction and segmented prefix sum, then
    additionally materialises each node's optimal PLATEAU as a union of gap intervals and
    derives two competing tie-break choices from the SAME maximum:

    * ``gap_first``   - the current production rule (first argmax breakpoint; gap 0 wins ties);
    * ``gap_mindisp`` - the gap in the maximizing set closest to the node's current rank,
      ties broken toward the smaller gap (deterministic, RNG-free).

    Returns a dict of per-node arrays: ``gain``, ``gap_first``, ``gap_mindisp``,
    ``plateau_width`` (number of gaps attaining the maximum), ``dist_first``, ``dist_mindisp``.

    The two rules have IDENTICAL exact gain by construction - they select different points of
    the same argmax set - so the mover set is rule-independent.
    """
    rank = np.asarray(rank, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    base = np.zeros(n, dtype=np.float64)
    np.add.at(base, src, w)

    out = dict(
        gain=np.zeros(n, dtype=np.float64),
        gap_first=np.zeros(n, dtype=np.int64),
        gap_mindisp=np.zeros(n, dtype=np.int64),
        plateau_width=np.ones(n, dtype=np.int64),
        dist_first=np.zeros(n, dtype=np.int64),
        dist_mindisp=np.zeros(n, dtype=np.int64),
    )
    if src.shape[0] == 0:
        return out

    p = rank

    # -- events: (node u, breakpoint b = q+1, delta) -- identical to jacobi_best_gaps ------
    out_nb, out_p = rank[tgt], p[src]
    out_b = np.where(out_nb < out_p, out_nb, out_nb - 1) + 1
    in_nb, in_p = rank[src], p[tgt]
    in_b = np.where(in_nb < in_p, in_nb, in_nb - 1) + 1

    ev_u = np.concatenate([src, tgt])
    ev_b = np.concatenate([out_b, in_b])
    ev_delta = np.concatenate([-w, w])
    np.clip(ev_b, 0, n - 1, out=ev_b)

    order = np.argsort(ev_u * np.int64(n + 2) + ev_b, kind="stable")
    su, sb, sd = ev_u[order], ev_b[order], ev_delta[order]
    del ev_u, ev_b, ev_delta, order

    new_grp = np.empty(su.shape[0], dtype=bool)
    new_grp[0] = True
    new_grp[1:] = (su[1:] != su[:-1]) | (sb[1:] != sb[:-1])
    grp_idx = np.cumsum(new_grp) - 1
    n_grp = int(grp_idx[-1]) + 1
    grp_u, grp_b = su[new_grp], sb[new_grp]
    grp_delta = np.zeros(n_grp, dtype=np.float64)
    np.add.at(grp_delta, grp_idx, sd)
    del su, sb, sd, grp_idx

    node_start = np.empty(n_grp, dtype=bool)
    node_start[0] = True
    node_start[1:] = grp_u[1:] != grp_u[:-1]
    seg_id = np.cumsum(node_start) - 1
    seg_first_pos = np.flatnonzero(node_start)
    seg_node = grp_u[seg_first_pos]

    csum = np.cumsum(grp_delta)
    pre_per_seg = np.empty(seg_first_pos.shape[0], dtype=np.float64)
    pre_per_seg[0] = 0.0
    pre_per_seg[1:] = csum[seg_first_pos[1:] - 1]
    grp_val = base[grp_u] + (csum - pre_per_seg[seg_id])
    del csum, grp_delta, new_grp

    # -- gap INTERVAL owned by each breakpoint group: [grp_b, next_b - 1] -----------------
    is_last = np.empty(n_grp, dtype=bool)
    is_last[:-1] = node_start[1:]
    is_last[-1] = True
    nxt = np.empty(n_grp, dtype=np.int64)
    nxt[:-1] = grp_b[1:]
    nxt[-1] = n
    lo = grp_b
    hi = np.where(is_last, n - 1, nxt - 1)
    del nxt, is_last

    # -- the maximum includes the gap-0 interval [0, b_first - 1] whose value is base ------
    seg_max_grp = np.maximum.reduceat(grp_val, seg_first_pos)
    base_seg = base[seg_node]
    b_first = grp_b[seg_first_pos]
    base_valid = b_first >= 1                     # the gap-0 interval is non-empty
    seg_overall = np.where(base_valid, np.maximum(seg_max_grp, base_seg), seg_max_grp)

    is_max = grp_val == seg_overall[seg_id]          # attains the OVERALL max (plateau set)
    base_is_max = base_valid & (base_seg == seg_overall)
    # The production argmax runs over BREAKPOINT GROUPS ONLY (the gap-0 interval is handled
    # separately by `use_bp` below), so every segment has >= 1 hit and the sentinel is unused.
    is_max_grp_only = grp_val >= seg_max_grp[seg_id]

    # -- plateau width = total number of gaps attaining the maximum ------------------------
    width = np.add.reduceat(np.where(is_max, hi - lo + 1, 0), seg_first_pos)
    width = width + np.where(base_is_max, b_first, 0)

    # -- current production rule ----------------------------------------------------------
    big = np.int64(n_grp + 1)
    cand = np.where(is_max_grp_only, np.arange(n_grp, dtype=np.int64), big)
    seg_best_b = grp_b[np.minimum.reduceat(cand, seg_first_pos)]
    use_bp = seg_max_grp > base_seg               # strict: gap 0 wins ties (insertion.py:236)
    gap_first = np.where(use_bp, seg_best_b, 0).astype(np.int64)
    del cand, seg_best_b

    # -- minimum-|displacement| choice within the SAME maximizing set ---------------------
    p_grp = p[grp_u]
    closest = np.minimum(np.maximum(p_grp, lo), hi)
    dist = np.abs(closest - p_grp)
    key = np.where(is_max, dist * np.int64(n + 1) + closest, SENTINEL)
    seg_key = np.minimum.reduceat(key, seg_first_pos)
    del key, dist, closest, p_grp

    p_seg = p[seg_node]
    c_base = np.minimum(p_seg, np.maximum(b_first - 1, 0))
    key_base = np.where(base_is_max,
                        np.abs(c_base - p_seg) * np.int64(n + 1) + c_base, SENTINEL)
    seg_key = np.minimum(seg_key, key_base)
    gap_mindisp = (seg_key % np.int64(n + 1)).astype(np.int64)

    # -- current value at gap p (the interval containing p) -------------------------------
    inside = (lo <= p[grp_u]) & (p[grp_u] <= hi)
    seg_cur = np.where(base_valid & (p_seg < b_first), base_seg, 0.0)
    hit = np.flatnonzero(inside)
    seg_cur[seg_id[hit]] = grp_val[hit]

    gain = seg_overall - seg_cur
    gain[gain < 0] = 0.0

    nodes = seg_node
    out["gain"][nodes] = gain
    out["gap_first"][nodes] = gap_first
    out["gap_mindisp"][nodes] = gap_mindisp
    out["plateau_width"][nodes] = width
    out["dist_first"][nodes] = np.abs(gap_first - p_seg)
    out["dist_mindisp"][nodes] = np.abs(gap_mindisp - p_seg)
    return out


def _summarise(rank: np.ndarray, prof: Dict[str, np.ndarray], label: str,
               tol: float = 1e-9) -> Dict:
    """Reduce a per-node profile to the rung-1 statistics, over MOVERS only."""
    mv = prof["gain"] > tol
    n_mov = int(mv.sum())
    if n_mov == 0:
        return dict(label=label, n_movers=0, note="fixed point - no movers")
    p = rank[mv]
    wdt = prof["plateau_width"][mv]
    gf, gm = prof["gap_first"][mv], prof["gap_mindisp"][mv]
    df, dm = prof["dist_first"][mv], prof["dist_mindisp"][mv]
    left_f = gf < p
    saved = df - dm
    hist = {}
    for name, lo, hi in [("w=1", 1, 1), ("w=2-10", 2, 10), ("w=11-100", 11, 100),
                         ("w=101-1000", 101, 1000), ("w=1001-10000", 1001, 10000),
                         ("w>10000", 10001, 1 << 62)]:
        hist[name] = int(((wdt >= lo) & (wdt <= hi)).sum())
    n_gap0 = int((gf == 0).sum())
    return dict(
        label=label,
        n_movers=n_mov,
        frac_plateau_gt1=float((wdt > 1).mean()),
        plateau_width_mean=float(wdt.mean()),
        plateau_width_median=float(np.median(wdt)),
        plateau_width_p90=float(np.percentile(wdt, 90)),
        plateau_width_max=int(wdt.max()),
        plateau_hist=hist,
        frac_leftward_first=float(left_f.mean()),
        dist_first_mean=float(df.mean()),
        dist_mindisp_mean=float(dm.mean()),
        dist_first_median=float(np.median(df)),
        dist_mindisp_median=float(np.median(dm)),
        displacement_saved_mean=float(saved.mean()),
        displacement_saved_median=float(np.median(saved)),
        frac_gap_changes=float((gf != gm).mean()),
        # the extreme over-transport case: gap 0 wins the tie and ships the node to the far left
        n_gap0_wins=n_gap0,
        gap0_wins_mean_dist=float(df[gf == 0].mean()) if n_gap0 else 0.0,
        # direction split AFTER the min-displacement rule
        frac_leftward_mindisp=float((gm < p).mean()),
    )


def rung1(dataset: str, sweeps: int, k_full: int, alpha: float,
          extra_orders: List[Tuple[str, np.ndarray]]) -> Dict:
    """Measure plateau structure on real orders and along a real sift trajectory."""
    g = load_dataset(dataset)
    n = g.n_nodes
    src, tgt, w = build_sift_edges(g)
    src_o = np.asarray(g.src, dtype=np.int64)
    tgt_o = np.asarray(g.tgt, dtype=np.int64)

    rows: List[Dict] = []
    t0 = time.time()

    for name, rk in extra_orders:
        prof = profile_plateaus(rk, src, tgt, w, n)
        row = _summarise(rk, prof, name)
        row["order_pct"] = pct(score_from_order(rk, src_o, tgt_o, g.weight), g.total_weight)
        rows.append(row)
        print("[%s] pct=%.6f movers=%d frac_w>1=%.4f saved=%.1f"
              % (name, row["order_pct"], row["n_movers"],
                 row.get("frac_plateau_gt1", 0.0),
                 row.get("displacement_saved_mean", 0.0)), flush=True)

    # -- a real under-relaxed sift trajectory from the greedy-FAS warm start --------------
    work = np.asarray(greedy_fas_order(g), dtype=np.int64)
    work = np.argsort(np.argsort(work, kind="stable"), kind="stable").astype(np.int64)
    for s in range(sweeps):
        a = 1.0 if s < k_full else float(alpha)
        prof = profile_plateaus(work, src, tgt, w, n)
        # The trajectory is advanced by the PRODUCTION kernel, not by this file's
        # reimplementation, so the orders profiled are exactly the ones the champion's
        # stage 3 would visit from the same start. profile_plateaus is used for statistics
        # only. (Its gain agrees with production to ~1e-17 and its mover set exactly;
        # see experiments/check_H73_profile.py.)
        bg_prod, gain_prod = jacobi_best_gaps(work, src, tgt, w, n)
        row = _summarise(work, prof, "sift_sweep_%d" % s)
        row["alpha"] = a
        row["order_pct"] = pct(score_from_order(work, src_o, tgt_o, g.weight), g.total_weight)
        rows.append(row)
        print("[sweep %d] a=%.2f pct=%.6f movers=%d frac_w>1=%.4f d_first=%.1f d_min=%.1f"
              % (s, a, row["order_pct"], row["n_movers"], row.get("frac_plateau_gt1", 0.0),
                 row.get("dist_first_mean", 0.0), row.get("dist_mindisp_mean", 0.0)),
              flush=True)
        if row["n_movers"] == 0:
            break
        work = underrelaxed_rebuild(work, bg_prod, gain_prod, a)

    movers_rows = [r for r in rows if r["n_movers"] > 0]
    min_frac = min((r["frac_plateau_gt1"] for r in movers_rows), default=0.0)
    verdict = "PREMISE HOLDS" if movers_rows and min_frac >= 0.01 \
        else "PREMISE FALSE (<1% plateaus)"
    return dict(
        item="H73", rung=1, dataset=dataset, n_nodes=n,
        params=dict(sweeps=sweeps, k_full=k_full, alpha=alpha),
        rows=rows, verdict=verdict, min_frac_plateau_gt1=min_frac,
        wall_clock_s=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--k-full", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--order", action="append", default=[],
                    help="name=path.npy of an extra position/rank vector to profile")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    extra: List[Tuple[str, np.ndarray]] = []
    for spec in args.order:
        name, _, path = spec.partition("=")
        pos = np.load(path)
        rk = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
        extra.append((name, rk))

    res = rung1(args.dataset, args.sweeps, args.k_full, args.alpha, extra)
    out = Path(args.out or "experiments/outputs/proto_H73_%s.json" % args.dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("\nVERDICT: %s  (min frac_plateau_gt1 = %.6f)"
          % (res["verdict"], res["min_frac_plateau_gt1"]))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
