"""Method R step 1: greedy integer order-guided routing (H2) with a K1 certificate.

Commodities = backward edges of the guide order, routed on its forward DAG in hop phases
(all commodities with paths of <= H1 hops, then <= H2, ...). Output: certificate, run JSON,
and the residual state (for the improvement step and the H5 table).

Run from the repo root: python ceiling/src/m_route_greedy.py --tag g1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from cert_io import write_cert  # noqa: E402
from layering.io_utils import load_graph_and_champion_order  # noqa: E402
from m_route_core import Router, git_commit, merge_records, verify  # noqa: E402

CONFIG = dict(
    phases=[1, 2, 3, 4, 5, 6, 8, 10, 14, 20],
    order="span",  # span | weight | random
    guide="h64",  # h64 | jitter:S (rank + uniform[0, S) noise, re-sorted) | blockrev:B (reverse inside rank blocks of B)
    work_limit=3_000_000,
    seed=0,
    cert_dir=HERE.parent / "certs",
    res_dir=HERE.parent / "results",
    state_dir=HERE.parent / "results" / "state",
    cert_prefix="c20_route_greedy",
)


def commodity_order(back, rank, src, tgt, w, how, seed):
    if how == "span":
        key = (rank[src[back]] - rank[tgt[back]]).astype(np.int64)
        return back[np.lexsort((-w[back], key))]
    if how == "weight":
        return back[np.argsort(-w[back], kind="stable")]
    rng = np.random.default_rng(seed)
    return back[rng.permutation(back.size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default=CONFIG["order"])
    ap.add_argument("--phases", default=",".join(str(h) for h in CONFIG["phases"]))
    ap.add_argument("--work-limit", type=int, default=CONFIG["work_limit"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--guide", default=CONFIG["guide"])
    ap.add_argument("--no-cert", action="store_true")
    args = ap.parse_args()
    phases = [int(h) for h in args.phases.split(",")]
    t0 = time.time()

    g, order, rank, ff_pct = load_graph_and_champion_order("connectome")
    src, tgt = g.src.astype(np.int64), g.tgt.astype(np.int64)
    w = g.weight.astype(np.int64)
    n, W = g.n_nodes, int(w.sum())
    if args.guide != "h64":
        how, val = args.guide.split(":")
        if how == "jitter":
            key = rank + np.random.default_rng(args.seed).random(n) * int(val)
        else:
            key = (rank // int(val)) * int(val) + (int(val) - 1 - rank % int(val))
        rank = np.empty(n, np.int64)
        rank[np.argsort(key, kind="stable")] = np.arange(n)
        ff_pct = 100 * int(w[rank[src] < rank[tgt]].sum()) / W
    fwd = rank[src] < rank[tgt]
    back = np.flatnonzero(~fwd)
    F = int(w[back].sum())
    print("loaded", round(time.time() - t0, 1), "s; backward edges", back.size, "feedback", F,
          "guide ff_pct", round(ff_pct, 4), flush=True)

    cap = w.copy()
    R = Router(src, tgt, n, fwd, cap, rank)
    comm = commodity_order(back, rank, src, tgt, w, args.order, args.seed)
    parts, phase_log = [], []
    for H in phases:
        t1 = time.time()
        left = comm[cap[comm] > 0]
        rec, st = R.run(left, H, args.work_limit)
        parts.append((rec[0], rec[1], rec[2], rec[3]))
        routed = F - int(cap[back].sum())
        row = dict(H=H, commodities_tried=int(left.size), seconds=round(time.time() - t1, 1),
                   routed_total=routed, **st)
        phase_log.append(row)
        print("phase", row, flush=True)

    rc, ra, rp, re = merge_records(parts)
    routed = F - int(cap[back].sum())
    LB = int(ra.sum())
    assert LB == routed
    route_s = round(time.time() - t0, 1)
    print("routed", routed, "of", F, "share", round(routed / F, 4), "LB", LB,
          "ceiling", round(100 * (W - LB) / W, 4), "time", route_s, flush=True)

    CONFIG["state_dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CONFIG["state_dir"] / ("m_route_" + args.tag + ".npz"), cap=cap,
                        rec_comm=rc, rec_amt=ra, rec_ptr=rp, rec_edges=re)

    res = dict(run="m_route_greedy", tag=args.tag, command=" ".join(sys.argv),
               git_commit=git_commit(), seed=args.seed, order=args.order, phases=phases,
               work_limit=args.work_limit, guide=args.guide, feedback_guide=F, routed=routed,
               routed_share=round(routed / F, 6), LB=LB, ceiling_pct=round(100 * (W - LB) / W, 6),
               n_cycles=int(ra.size), max_cycle_len=int(np.diff(rp).max()) if ra.size else 0,
               phase_log=phase_log, runtime_route_s=route_s)
    if not args.no_cert:
        cert = CONFIG["cert_dir"] / (CONFIG["cert_prefix"] + "_" + args.tag + ".npz")
        write_cert(cert, graph=g, D=1, cycles=(rp, re), y=ra, method="m_route_greedy",
                   notes="order-guided greedy routing on the forward DAG of guide " + args.guide + ", tag " + args.tag
                         + ", order " + args.order + ", hop phases " + args.phases)
        v = verify(cert)
        print("verify", v["line"], v["seconds"], "s", flush=True)
        res.update(cert=str(cert.relative_to(HERE.parents[1])), verifier_status=v["status"],
                   verifier_line=v["line"], verifier_seconds=v["seconds"])
    res["runtime_total_s"] = round(time.time() - t0, 1)
    out = CONFIG["res_dir"] / ("m_route_greedy_" + args.tag + ".json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
