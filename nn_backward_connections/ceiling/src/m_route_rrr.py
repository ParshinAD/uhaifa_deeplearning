"""Method R step 2b: rip-up and reroute on top of a saved H64-DAG routing state.

One round: take a batch of unrouted backward edges b = (u -> v); for each, the shortest
forward path v ~> u in the uncapacitated DAG names the saturated edges that block it; rip up
every routed cycle through those edges, route the batch first, then the ripped commodities (hop phases each).
Keep the round if total routed demand grew, else revert.
Optional final stage: residual full-graph packing (as in m_route_improve.py). K1 certificate.

Run from the repo root: python ceiling/src/m_route_rrr.py --tag x1 --state i1 --minutes 30
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
    batch=500,
    reroute_phases=[1, 2, 3, 4, 5, 6, 8],
    work_limit=3_000_000,
    probe_hops=64,
    max_rip_per_round=20_000,
    residual_phases="",
    seed=0,
    big=10 ** 12,
    cert_dir=HERE.parent / "certs",
    res_dir=HERE.parent / "results",
    state_dir=HERE.parent / "results" / "state",
    cert_prefix="c22_route_rrr",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    ap.add_argument("--residual-phases", default=CONFIG["residual_phases"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    g, order, rank, ff_pct = load_graph_and_champion_order("connectome")
    src, tgt = g.src.astype(np.int64), g.tgt.astype(np.int64)
    w = g.weight.astype(np.int64)
    n, m, W = g.n_nodes, g.n_edges, int(w.sum())
    fwd = rank[src] < rank[tgt]
    back = np.flatnonzero(~fwd)
    F = int(w[back].sum())
    with np.load(CONFIG["state_dir"] / ("m_route_" + args.state + ".npz")) as z:
        assert "cap_dag" not in z.files or np.array_equal(z["cap_dag"], z["cap"]), "state must be DAG-only"
        cap = z["cap"].copy()
        rc, ra, rp, re = z["rec_comm"], z["rec_amt"], z["rec_ptr"], z["rec_edges"]
    lens = np.diff(rp)
    recs = [(rc, ra, rp, re)]
    alive = np.ones(rc.size, bool)
    rec_id = np.repeat(np.arange(rc.size), lens)
    # load check: state must be consistent with the records
    load = np.bincount(re, weights=np.repeat(ra, lens), minlength=m).astype(np.int64)
    assert np.array_equal(load, w - cap)
    del load
    R = Router(src, tgt, n, fwd, cap, rank)
    big = np.where(fwd, CONFIG["big"], 0).astype(np.int64)
    P = Router(src, tgt, n, fwd, big, rank)
    start = F - int(cap[back].sum())
    print("start routed", start, "records", rc.size, "load", round(time.time() - t0, 1), "s", flush=True)

    ent_sorted = None
    rounds, kept, log = 0, 0, []
    while time.time() - t0 < 60 * args.minutes:
        t1 = time.time()
        if ent_sorted is None:
            all_e = np.concatenate([r[3] for r in recs])
            all_id = np.concatenate([np.repeat(np.arange(r[0].size), np.diff(r[2])) for r in recs])
            offs = np.cumsum([0] + [r[0].size for r in recs])
            all_id = all_id + np.repeat(offs[:-1], [r[3].size for r in recs])
            o = np.argsort(all_e, kind="stable")
            ent_sorted = (all_e[o], all_id[o])
        un = back[cap[back] > 0]
        if un.size == 0:
            break
        batch = rng.choice(un, size=min(args.batch, un.size), replace=False)
        big[back] = 0
        big[batch] = 1
        prec, _ = P.run(batch, CONFIG["probe_hops"], CONFIG["work_limit"])
        big[prec[3]] = CONFIG["big"]
        pe = prec[3]
        pe = pe[fwd[pe]]
        blocked = np.unique(pe[cap[pe] == 0])
        es, ids = ent_sorted
        lo, hi = np.searchsorted(es, blocked), np.searchsorted(es, blocked, side="right")
        cand = np.unique(np.concatenate([ids[a:b] for a, b in zip(lo, hi)])) if blocked.size else np.zeros(0, np.int64)
        cand = cand[alive[cand]]
        if cand.size > CONFIG["max_rip_per_round"]:
            cand = rng.choice(cand, CONFIG["max_rip_per_round"], replace=False)
        flat_comm = np.concatenate([r[0] for r in recs])
        flat_amt = np.concatenate([r[1] for r in recs])
        cap_save, alive_save, nrec_save = cap.copy(), alive.copy(), len(recs)
        before = F - int(cap[back].sum())
        # rip
        base = np.cumsum([0] + [r[0].size for r in recs])
        for rid in cand:
            k = int(np.searchsorted(base, rid, side="right") - 1)
            j = int(rid - base[k])
            r = recs[k]
            cap[r[3][r[2][j]:r[2][j + 1]]] += r[1][j]
        alive[cand] = False
        ripped = np.unique(flat_comm[cand])
        new = []
        for H in CONFIG["reroute_phases"]:
            for group in (batch, ripped):
                left = group[cap[group] > 0]
                if left.size:
                    rec, _ = R.run(left, H, CONFIG["work_limit"])
                    new.append(rec)
        after = F - int(cap[back].sum())
        gain = after - before
        if gain > 0:
            merged = merge_records(new)
            recs.append(merged)
            alive = np.concatenate([alive, np.ones(merged[0].size, bool)])
            ent_sorted = None
            kept += 1
        else:
            cap[:] = cap_save
            alive = alive_save
        rounds += 1
        row = dict(round=rounds, batch=int(batch.size), blocked_edges=int(blocked.size),
                   ripped_records=int(cand.size), ripped_amount=int(flat_amt[cand].sum()) if cand.size else 0,
                   gain=int(gain), kept=gain > 0, routed=F - int(cap[back].sum()),
                   seconds=round(time.time() - t1, 1))
        log.append(row)
        if rounds % 10 == 1 or gain > 0:
            print(row, flush=True)

    dag_routed = F - int(cap[back].sum())
    print("rrr done rounds", rounds, "kept", kept, "dag routed", dag_routed, "gain", dag_routed - start, flush=True)
    rc = np.concatenate([r[0] for r in recs])
    ra = np.concatenate([r[1] for r in recs])
    lens = np.concatenate([np.diff(r[2]) for r in recs])
    re_all = np.concatenate([r[3] for r in recs])
    keep_e = np.repeat(alive, lens)
    rc, ra, lens, re_all = rc[alive], ra[alive], lens[alive], re_all[keep_e]
    rp = np.zeros(lens.size + 1, np.int64)
    np.cumsum(lens, out=rp[1:])
    assert int(ra.sum()) == dag_routed
    cap_dag = cap.copy()
    CONFIG["state_dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CONFIG["state_dir"] / ("m_route_" + args.tag + "_dag.npz"), cap=cap_dag,
                        rec_comm=rc, rec_amt=ra, rec_ptr=rp, rec_edges=re_all)
    parts = [(rc, ra, rp, re_all)]
    rlog = []
    if args.residual_phases:
        R2 = Router(src, tgt, n, np.ones(m, bool), cap, None)
        comm = back[np.argsort(rank[src[back]] - rank[tgt[back]], kind="stable")]
        for H in [int(h) for h in args.residual_phases.split(",")]:
            t1 = time.time()
            left = comm[cap[comm] > 0]
            rec, st = R2.run(left, H, CONFIG["work_limit"])
            parts.append(rec)
            row = dict(stage="residual", H=H, tried=int(left.size), seconds=round(time.time() - t1, 1), **st)
            rlog.append(row)
            print(row, flush=True)
    rc, ra, rp, re_all = merge_records(parts)
    LB = int(ra.sum())
    route_s = round(time.time() - t0, 1)
    print("LB", LB, "ceiling", round(100 * (W - LB) / W, 4), "time", route_s, flush=True)
    np.savez_compressed(CONFIG["state_dir"] / ("m_route_" + args.tag + ".npz"), cap=cap, cap_dag=cap_dag,
                        rec_comm=rc, rec_amt=ra, rec_ptr=rp, rec_edges=re_all)
    cert = CONFIG["cert_dir"] / (CONFIG["cert_prefix"] + "_" + args.tag + ".npz")
    write_cert(cert, graph=g, D=1, cycles=(rp, re_all), y=ra, method="m_route_rrr",
               notes="rip-up and reroute from state " + args.state + ", batch " + str(args.batch)
                     + ", seed " + str(args.seed) + ", residual phases " + (args.residual_phases or "none"))
    v = verify(cert)
    print("verify", v["line"], v["seconds"], "s", flush=True)
    res = dict(run="m_route_rrr", tag=args.tag, state=args.state, command=" ".join(sys.argv),
               git_commit=git_commit(), seed=args.seed, batch=args.batch, minutes=args.minutes,
               feedback_guide=F, start_routed=start, rounds=rounds, kept_rounds=kept,
               dag_routed=dag_routed, dag_routed_share=round(dag_routed / F, 6),
               rrr_gain=dag_routed - start, LB=LB, ceiling_pct=round(100 * (W - LB) / W, 6),
               n_cycles=int(ra.size), max_cycle_len=int(np.diff(rp).max()), residual_log=rlog,
               round_log=log, runtime_route_s=route_s, cert=str(cert.relative_to(HERE.parents[1])),
               verifier_status=v["status"], verifier_line=v["line"], verifier_seconds=v["seconds"])
    res["runtime_total_s"] = round(time.time() - t0, 1)
    out = CONFIG["res_dir"] / ("m_route_rrr_" + args.tag + ".json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
