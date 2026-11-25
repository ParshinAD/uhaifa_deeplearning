"""Merge block runs of m_dual_block with the same k, offset and base order (each run did some windows).
Edges inside window i take X from the run that did window i, nodes of window i take its rounded ranks.
Usage (repo root): m_dual_merge.py <out name> <base_rank.npz or 'champion'> <run name> [<run name> ...]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_block as blk

if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    from src.mfas.metrics import score_from_order
    out, base, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    if base != "champion":
        rank = np.load(base)["rank"].astype(np.int64)
    d = Path(blk.CONFIG["out_dir"])
    X, new_rank, info = None, rank.astype(np.int64).copy(), []
    for name in runs:
        js = json.loads((ROOT / "ceiling" / "results" / ("m_dual_block_" + name + ".json")).read_text())
        z = np.load(d / ("block_" + name + ".npz"))
        k, off = js["k"], js.get("offset", 0)
        ws, wt = (rank[g.src] + off) // k, (rank[g.tgt] + off) // k
        wn = (rank + off) // k
        if X is None:
            X = z["X"].copy()
        sel = np.isin(ws, js["windows_done"]) & (ws == wt)
        X[sel] = z["X"][sel]
        seln = np.isin(wn, js["windows_done"])
        new_rank[seln] = z["rank"][seln]
        info.append(dict(run=name, k=k, offset=off, windows=js["windows_done"]))
    assert np.array_equal(np.sort(new_rank), np.arange(g.n_nodes))
    np.savez(d / ("block_" + out + ".npz"), X=X, rank=new_rank)
    fb = float(g.total_weight - score_from_order(new_rank, g.src, g.tgt, g.weight))
    print(json.dumps(dict(merged=info, sum_wX_over_S=float((g.weight.astype(np.int64) * X).sum() / blk.CONFIG["S"]),
                          rounded_order_feedback=fb)))
