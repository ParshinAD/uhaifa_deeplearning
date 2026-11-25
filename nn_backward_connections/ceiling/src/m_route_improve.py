"""Method R step 2 and 3: improved routing and a K1 certificate.

Step 2 (capacity schedule): route on the H64 forward DAG in rounds alpha_1 < ... < 1; in round
alpha a forward edge offers only floor(alpha * w) - load, so early (short) paths cannot drain
an edge that later commodities need. Each round runs all hop phases.
Step 3 (residual packing): after that, route the leftover backward demand on the FULL residual
graph (no rank window), so a cycle may contain several backward edges of the H64 order.

Run from the repo root: python ceiling/src/m_route_improve.py --tag i1 --alphas 0.25,0.5,0.75,1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from cert_io import write_cert  # noqa: E402
from layering.io_utils import load_graph_and_champion_order  # noqa: E402
from m_route_core import Router, git_commit, merge_records, verify  # noqa: E402
from m_route_greedy import commodity_order  # noqa: E402

CONFIG = dict(
    alphas="0.25,0.5,0.75,1",
    phases="1,2,3,4,5,6,8,10",
    residual_phases="",
    order="span",
    work_limit=3_000_000,
    residual_work_limit=3_000_000,
    seed=0,
    cert_dir=HERE.parent / "certs",
    res_dir=HERE.parent / "results",
    state_dir=HERE.parent / "results" / "state",
    cert_prefix="c21_route_improve",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    for k in ("alphas", "phases", "residual_phases", "order"):
        ap.add_argument("--" + k.replace("_", "-"), default=CONFIG[k])
    ap.add_argument("--work-limit", type=int, default=CONFIG["work_limit"])
    ap.add_argument("--residual-work-limit", type=int, default=CONFIG["residual_work_limit"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--cert-prefix", default=CONFIG["cert_prefix"])
    ap.add_argument("--interleave", action="store_true",
                    help="per hop limit H: DAG(H) then full residual graph(H); alphas must be 1")
    args = ap.parse_args()
    alphas = [Fraction(a) for a in args.alphas.split(",")]
    phases = [int(h) for h in args.phases.split(",")]
    rphases = [int(h) for h in args.residual_phases.split(",")] if args.residual_phases else []
    t0 = time.time()

    g, order, rank, ff_pct = load_graph_and_champion_order("connectome")
    src, tgt = g.src.astype(np.int64), g.tgt.astype(np.int64)
    w = g.weight.astype(np.int64)
    n, W = g.n_nodes, int(w.sum())
    fwd = rank[src] < rank[tgt]
    back = np.flatnonzero(~fwd)
    F = int(w[back].sum())
    cap = w.copy()
    R = Router(src, tgt, n, fwd, cap, rank)
    comm = commodity_order(back, rank, src, tgt, w, args.order, args.seed)
    fe = np.flatnonzero(fwd)
    parts, log = [], []
    if args.interleave:
        assert alphas == [1] and not rphases
        R2 = Router(src, tgt, n, np.ones(src.size, bool), cap, None)
        dag_routed = 0
        for H in phases:
            for stage, router, wl in (("dag", R, args.work_limit), ("residual", R2, args.residual_work_limit)):
                t1 = time.time()
                left = comm[cap[comm] > 0]
                rec, st = router.run(left, H, wl)
                parts.append(rec)
                dag_routed += st["routed"] if stage == "dag" else 0
                row = dict(stage=stage, H=H, tried=int(left.size), seconds=round(time.time() - t1, 1),
                           routed_total=int(sum(int(p[1].sum()) for p in parts)), **st)
                log.append(row)
                print(row, flush=True)
        alphas = []
    for a in alphas:
        reserve = w[fe] - (w[fe] * a.numerator) // a.denominator
        cap[fe] -= reserve
        assert int(cap[fe].min()) >= 0
        for H in phases:
            t1 = time.time()
            left = comm[cap[comm] > 0]
            rec, st = R.run(left, H, args.work_limit)
            parts.append(rec)
            row = dict(stage="dag", alpha=str(a), H=H, tried=int(left.size),
                       seconds=round(time.time() - t1, 1),
                       routed_total=int(sum(int(p[1].sum()) for p in parts)), **st)
            log.append(row)
            print(row, flush=True)
        cap[fe] += reserve
    if not args.interleave:
        dag_routed = int(sum(int(p[1].sum()) for p in parts))
        assert dag_routed == F - int(cap[back].sum())
    print("dag routed", dag_routed, "share", round(dag_routed / F, 4), flush=True)
    cap_dag = cap.copy()

    if rphases:
        R2 = Router(src, tgt, n, np.ones(src.size, bool), cap, None)
        for H in rphases:
            t1 = time.time()
            left = comm[cap[comm] > 0]
            rec, st = R2.run(left, H, args.residual_work_limit)
            parts.append(rec)
            row = dict(stage="residual", H=H, tried=int(left.size), seconds=round(time.time() - t1, 1),
                       routed_total=int(sum(int(p[1].sum()) for p in parts)), **st)
            log.append(row)
            print(row, flush=True)

    rc, ra, rp, re = merge_records(parts)
    LB = int(ra.sum())
    route_s = round(time.time() - t0, 1)
    print("LB", LB, "ceiling", round(100 * (W - LB) / W, 4), "time", route_s, flush=True)
    CONFIG["state_dir"].mkdir(parents=True, exist_ok=True)
    extra = {} if args.interleave else dict(cap_dag=cap_dag)
    np.savez_compressed(CONFIG["state_dir"] / ("m_route_" + args.tag + ".npz"), cap=cap,
                        **extra, rec_comm=rc, rec_amt=ra, rec_ptr=rp, rec_edges=re)

    cert = CONFIG["cert_dir"] / (args.cert_prefix + "_" + args.tag + ".npz")
    write_cert(cert, graph=g, D=1, cycles=(rp, re), y=ra, method="m_route_improve",
               notes="H64-guided routing, capacity rounds " + args.alphas + ", hop phases "
                     + args.phases + ", residual full-graph phases " + (args.residual_phases or "none")
                     + ", interleave " + str(args.interleave) + ", order " + args.order + ", tag " + args.tag)
    v = verify(cert)
    print("verify", v["line"], v["seconds"], "s", flush=True)
    res = dict(run="m_route_improve", tag=args.tag, command=" ".join(sys.argv), git_commit=git_commit(),
               seed=args.seed, order=args.order, alphas=args.alphas, phases=phases,
               residual_phases=rphases, interleave=args.interleave, work_limit=args.work_limit,
               residual_work_limit=args.residual_work_limit, guide="H64", feedback_guide=F,
               dag_routed=dag_routed, dag_routed_share=round(dag_routed / F, 6), LB=LB,
               ceiling_pct=round(100 * (W - LB) / W, 6), n_cycles=int(ra.size),
               max_cycle_len=int(np.diff(rp).max()) if ra.size else 0, phase_log=log,
               runtime_route_s=route_s, cert=str(cert.relative_to(HERE.parents[1])),
               verifier_status=v["status"], verifier_line=v["line"], verifier_seconds=v["seconds"])
    res["runtime_total_s"] = round(time.time() - t0, 1)
    out = CONFIG["res_dir"] / ("m_route_improve_" + args.tag + ".json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
