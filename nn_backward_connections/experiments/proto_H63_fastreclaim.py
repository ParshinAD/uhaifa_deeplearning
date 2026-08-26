"""Prototype rung for cycle 15 (P13 inside H59): is arc reclamation cheap enough for microns?

The question is NOT whether reclamation gains anything — cycle 14 measured that and the answer
is yes on all three datasets. The question is exactly one thing:

    can :mod:`mfas.refine.reclaim2` produce the SAME accepted arc set as
    :mod:`mfas.refine.reclaim` fast enough that the stage fits inside microns' ~68 s of
    guard-deadline slack?

So this script measures two things and asserts a third:

1. **Equivalence.** The fast scan's reclaimable set and the fast conflict resolver's accepted
   set are compared, as sets of (weight, u, v) triples AND as ordered lists, against the
   reference implementation run on the same input. Anything other than identity is a FAIL —
   the whole point of P13 is "no change to the algorithm and therefore no change to the
   result".
2. **Cost**, per phase, so the H63 module can be configured from measurements rather than
   from hope.
3. **The realised score**, through the frozen oracle, which must reproduce the cycle-14
   figure in ``experiments/outputs/proto_H59_<dataset>.json`` exactly.

Run (CPU only, no GPU):

    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    PYTHONPATH=src $PY experiments/proto_H63_fastreclaim.py mouse --ref
    PYTHONPATH=src $PY experiments/proto_H63_fastreclaim.py connectome --ref
    PYTHONPATH=src $PY experiments/proto_H63_fastreclaim.py microns --ref
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfas import io                                          # noqa: E402
from mfas.metrics import pct, score_from_order               # noqa: E402
from mfas.refine import reclaim as ref                       # noqa: E402
from mfas.refine import reclaim2 as fast                     # noqa: E402


def find_champion(dataset):
    """Champion position vector, straight off a logged confirm run — PER DATASET (P14)."""
    champ = "H52" if dataset == "mouse" else "H42"
    for f in reversed(sorted(glob.glob(f"results/*-{champ}-{dataset}-s42-confirm-*.json"))):
        d = json.load(open(f))
        p = d.get("best_positions_path")
        if p and Path(p).exists():
            return champ, p, d["pct"]
    return champ, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="mouse")
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--conflict-budget", type=int, default=200_000)
    ap.add_argument("--ref", action="store_true",
                    help="also run the reference implementation and require identity")
    args = ap.parse_args()

    ds = args.dataset
    g = io.load_dataset(ds)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    n = int(g.n_nodes)
    total = g.total_weight

    champ, pos_path, champ_pct_record = find_champion(ds)
    if pos_path is None:
        print(f"no {champ} champion positions for {ds}")
        return 2
    positions = np.load(pos_path)
    rank = np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)
    base_score = score_from_order(rank, src, tgt, g.weight)
    base_pct = pct(base_score, total)
    print(f"[{ds}] champion {champ} {pos_path}", flush=True)
    print(f"[{ds}] base pct = {base_pct:.8f}  (record {champ_pct_record})", flush=True)

    keep = src != tgt
    e_src, e_tgt, e_w = src[keep], tgt[keep], g.weight[keep]

    rec = dict(item="H63", stage="P13-cheapening", dataset=ds, champion=champ,
               champion_positions=pos_path, champion_pct=base_pct,
               n_nodes=n, n_edges=int(g.n_edges), total_weight=float(total),
               budget=args.budget, conflict_budget=args.conflict_budget,
               ran_reference=bool(args.ref))

    # ── FAST scan ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    arcs_f, (Fs_f, Ft_f), st_f = fast.reclaimable_arcs_fast(
        rank, e_src, e_tgt, e_w, n, budget=args.budget)
    t_scan_fast = time.time() - t0
    print(f"[{ds}] fast scan {t_scan_fast:8.2f}s  reclaimable={st_f['n_reclaimable']:,} "
          f"onehop={st_f['n_onehop']:,} proved={st_f['n_proved']:,} "
          f"searched={st_f['n_searched']:,} unknown={st_f['n_unknown']}", flush=True)

    # ── FAST conflict resolution ────────────────────────────────────────────────────
    t0 = time.time()
    as_f, at_f, aw_f, cst_f = fast.resolve_conflicts_scc(
        Fs_f, Ft_f, arcs_f, n, conflict_budget=args.conflict_budget)
    t_conf_fast = time.time() - t0
    print(f"[{ds}] fast conflict {t_conf_fast:8.2f}s  accepted={cst_f['n_accepted']:,} "
          f"conflicts={cst_f['n_conflicts']:,} budget_rejects={cst_f['n_budget_rejects']} "
          f"nontrivial_sccs={cst_f['n_scc_nontrivial']:,} max_scc={cst_f['n_scc_max']:,} "
          f"arcs_in_scc={cst_f['n_arcs_in_scc']:,}", flush=True)

    rec["fast"] = dict(t_scan_s=t_scan_fast, t_conflict_s=t_conf_fast,
                       t_stage_s=t_scan_fast + t_conf_fast, scan=st_f, conflict=cst_f)

    # ── The realised score, through the frozen oracle ───────────────────────────────
    t0 = time.time()
    new_rank, log = fast.reclaim_arcs_fast(g, rank, rounds=1, budget=args.budget,
                                           conflict_budget=args.conflict_budget)
    t_stage = time.time() - t0
    new_score = score_from_order(new_rank, src, tgt, g.weight)
    new_pct = pct(new_score, total)
    w_acc = float(sum(aw_f))
    lemma = bool(float(new_score) + 1e-9 >= float(base_score) + w_acc)
    print(f"[{ds}] stage(round=1) {t_stage:8.2f}s  {base_pct:.8f} -> {new_pct:.8f} "
          f"= {new_pct - base_pct:+.6f} pp   lemma_holds={lemma}", flush=True)
    rec["stage"] = dict(t_stage_s=t_stage, pct_before=base_pct, pct_after=new_pct,
                        realised_delta_pp=new_pct - base_pct,
                        w_accepted=w_acc, lemma_holds=lemma, log=log)

    # ── Does it reproduce the cycle-14 figure? ──────────────────────────────────────
    p14 = Path(f"experiments/outputs/proto_H59_{ds}.json")
    if p14.exists():
        d14 = json.load(open(p14))
        rec["cycle14_final_pct"] = d14["final_pct"]
        rec["reproduces_cycle14"] = bool(abs(new_pct - d14["final_pct"]) < 1e-12)
        print(f"[{ds}] cycle-14 final {d14['final_pct']:.8f}  reproduced="
              f"{rec['reproduces_cycle14']}", flush=True)

    # ── Reference cross-check ───────────────────────────────────────────────────────
    if args.ref:
        t0 = time.time()
        arcs_r, (Fs_r, Ft_r), st_r = ref.reclaimable_arcs(
            rank, e_src, e_tgt, e_w, n, budget=args.budget)
        t_scan_ref = time.time() - t0
        t0 = time.time()
        as_r, at_r, aw_r, cst_r = ref.resolve_conflicts(
            Fs_r, Ft_r, arcs_r, n, conflict_budget=args.conflict_budget)
        t_conf_ref = time.time() - t0
        print(f"[{ds}] ref  scan {t_scan_ref:8.2f}s  conflict {t_conf_ref:8.2f}s  "
              f"reclaimable={st_r['n_reclaimable']:,} accepted={cst_r['n_accepted']:,}",
              flush=True)

        same_scan_list = arcs_f == arcs_r
        same_scan_set = set(arcs_f) == set(arcs_r)
        acc_f = list(zip(as_f, at_f, aw_f))
        acc_r = list(zip(as_r, at_r, aw_r))
        same_acc_list = acc_f == acc_r
        same_acc_set = set(acc_f) == set(acc_r)
        speedup_scan = t_scan_ref / max(t_scan_fast, 1e-9)
        speedup_conf = t_conf_ref / max(t_conf_fast, 1e-9)
        rec["reference"] = dict(
            t_scan_s=t_scan_ref, t_conflict_s=t_conf_ref,
            t_stage_s=t_scan_ref + t_conf_ref, scan=st_r, conflict=cst_r,
            same_reclaimable_list=same_scan_list, same_reclaimable_set=same_scan_set,
            same_accepted_list=same_acc_list, same_accepted_set=same_acc_set,
            speedup_scan=speedup_scan, speedup_conflict=speedup_conf,
            speedup_stage=(t_scan_ref + t_conf_ref) / max(t_scan_fast + t_conf_fast, 1e-9))
        print(f"[{ds}] EQUIVALENCE reclaimable(list/set)={same_scan_list}/{same_scan_set} "
              f"accepted(list/set)={same_acc_list}/{same_acc_set}", flush=True)
        print(f"[{ds}] SPEEDUP scan x{speedup_scan:.1f}  conflict x{speedup_conf:.1f}  "
              f"stage x{rec['reference']['speedup_stage']:.1f}", flush=True)

    out = Path(f"experiments/outputs/proto_H63_{ds}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2, default=float)
    print(f"[{ds}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
