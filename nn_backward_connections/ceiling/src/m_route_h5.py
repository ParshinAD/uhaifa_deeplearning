"""Method R, H5 data: routed vs unrouted demand per backward edge of H64, by rank span and super_class.

Also tests each unrouted backward edge alone on the ORIGINAL forward DAG (all capacities
huge): if no forward path v ~> u exists even then, no H64-guided cycle can ever explain it.

Run from the repo root: python ceiling/src/m_route_h5.py --state i1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from layering.io_utils import load_graph_and_champion_order  # noqa: E402
from m_route_core import Router, git_commit  # noqa: E402

CONFIG = dict(
    ann=ROOT / "layering/outputs/fly_ann/node_ann.npz",
    ann_meta=ROOT / "layering/outputs/fly_ann/node_ann_meta.json",
    state_dir=HERE.parent / "results" / "state",
    out_npz=HERE.parent / "results" / "m_route_unrouted.npz",
    res_dir=HERE.parent / "results",
    span_bins=[(1, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 100000), (100001, 10 ** 9)],
    iso_hops=64,
    iso_work_limit=200_000_000,
    big=10 ** 12,
)


def table(groups, demand, unrouted, iso_dead):
    rows = []
    for name, msk in groups:
        d, u = int(demand[msk].sum()), int(unrouted[msk].sum())
        rows.append(dict(group=name, edges=int(msk.sum()), demand=d, routed=d - u, unrouted=u,
                         unrouted_share_of_group=round(u / max(d, 1), 4),
                         share_of_all_unrouted=round(u / max(int(unrouted.sum()), 1), 4),
                         unrouted_no_path_even_alone=int(unrouted[msk & iso_dead].sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    t0 = time.time()
    g, order, rank, ff_pct = load_graph_and_champion_order("connectome")
    src, tgt = g.src.astype(np.int64), g.tgt.astype(np.int64)
    w = g.weight.astype(np.int64)
    n = g.n_nodes
    fwd = rank[src] < rank[tgt]
    back = np.flatnonzero(~fwd)
    with np.load(CONFIG["state_dir"] / ("m_route_" + args.state + ".npz")) as z:
        cap_dag = z["cap_dag"] if "cap_dag" in z.files else z["cap"]
    demand = w[back]
    unrouted = cap_dag[back]
    routed = demand - unrouted
    span = rank[src[back]] - rank[tgt[back]]

    big = np.where(fwd, CONFIG["big"], w).astype(np.int64)
    R = Router(src, tgt, n, fwd, big, rank)
    todo = back[unrouted > 0]
    rec, st = R.run(todo, CONFIG["iso_hops"], CONFIG["iso_work_limit"])
    iso_ok = np.zeros(g.n_edges, bool)
    iso_ok[rec[0]] = True
    iso_dead = ~iso_ok[back] & (unrouted > 0)
    print("isolation test", st, "seconds", round(time.time() - t0, 1), flush=True)

    meta = json.loads(CONFIG["ann_meta"].read_text())
    names = meta["super_class_values"]
    sc = np.load(CONFIG["ann"], allow_pickle=False)["super_class"].astype(np.int64)
    ss, st_ = sc[src[back]], sc[tgt[back]]
    np.savez_compressed(CONFIG["out_npz"], back_eid=back, src=src[back], tgt=tgt[back],
                        demand=demand, routed=routed, unrouted=unrouted, span=span,
                        super_class_src=ss, super_class_tgt=st_,
                        no_forward_path_alone=iso_dead.astype(np.int8),
                        super_class_values=np.array(names))

    span_groups = [(str(lo) + "-" + str(hi), (span >= lo) & (span <= hi)) for lo, hi in CONFIG["span_bins"]]
    same = ss == st_
    cls_groups = [(names[c], same & (ss == c)) for c in range(len(names)) if (same & (ss == c)).any()]
    cls_groups.append(("cross-class", ~same))
    by_span = table(span_groups, demand, unrouted, iso_dead)
    by_cls = table(cls_groups, demand, unrouted, iso_dead)
    tot = dict(demand=int(demand.sum()), routed=int(routed.sum()), unrouted=int(unrouted.sum()),
               unrouted_edges=int((unrouted > 0).sum()), fully_unrouted_edges=int((routed == 0).sum()),
               unrouted_no_path_even_alone=int(unrouted[iso_dead].sum()),
               edges_no_path_even_alone=int(iso_dead.sum()))
    for title, rows in (("by span", by_span), ("by super_class (both endpoints; else cross)", by_cls)):
        print("==", title)
        for r in rows:
            print(" ", r)
    print("total", tot)
    res = dict(run="m_route_h5", state=args.state, command=" ".join(sys.argv), git_commit=git_commit(),
               seed=None, total=tot, by_span=by_span, by_super_class=by_cls,
               isolation_stats=st, runtime_s=round(time.time() - t0, 1))
    out = CONFIG["res_dir"] / ("m_route_h5_" + args.state + ".json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, CONFIG["out_npz"], flush=True)


if __name__ == "__main__":
    main()
