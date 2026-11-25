"""Exact upper bound on the fractional FAS LP value tau* from integer edge lengths X >= 0.

L = min over directed cycles C of X(C) (exact, int64 Dijkstra from every vertex, see m_dual_core).
X / L is feasible for the cycle LP, so tau* <= sum_e w_e X_e / L (exact Fraction).

Usage (repo root): m_dual_check.py <lengths.npz> <name> [key] [S] [divw]
  key 'X' with int64 values is used as is; otherwise X = ceil(S * l) with l = z[key] (divided by w if divw=1).
Writes ceiling/results/m_dual_<name>.json."""
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core

CONFIG = dict(threads=6, S=10 ** 6)


def to_int_lengths(l, S):
    X = np.ceil(np.maximum(np.asarray(l, np.float64), 0.0) * S)
    assert X.max() < 2 ** 52
    return X.astype(np.int64)


def wsum(w, X):
    """Exact sum w X for int64 or float w (Fraction for floats)."""
    if np.issubdtype(np.asarray(w).dtype, np.integer):
        assert int(np.abs(w).max()) * int(X.max()) * w.size < 2 ** 62
        return Fraction(int((np.asarray(w, np.int64) * X).sum()))
    return sum((Fraction(float(a)) * int(b) for a, b in zip(w, X) if b), Fraction(0))


def check(G, w, X, threads=None):
    t0 = time.time()
    L, s, e = G.min_cycle(X, threads=threads or CONFIG["threads"])
    sw = wsum(w, X)
    out = dict(L=L, witness_source=s, witness_edge=e, sum_wX=str(sw), sum_wX_float=float(sw),
               check_s=round(time.time() - t0, 1))
    if L >= np.iinfo(np.int64).max // 4:
        out.update(acyclic=True, UB="0", UB_float=0.0)
    elif L == 0:
        out.update(UB="inf", UB_float=float("inf"))
    else:
        ub = sw / L
        out.update(UB=str(ub), UB_float=float(ub))
    return out


if __name__ == "__main__":
    from src.mfas.io import load_dataset
    t00 = time.time()
    path, name = sys.argv[1], sys.argv[2]
    key = sys.argv[3] if len(sys.argv) > 3 else "X"
    S = int(sys.argv[4]) if len(sys.argv) > 4 else CONFIG["S"]
    divw = len(sys.argv) > 5 and sys.argv[5] == "1"
    g = load_dataset("connectome")
    w = g.weight.astype(np.int64)
    z = np.load(path)
    a = z[key]
    if np.issubdtype(a.dtype, np.integer):
        X, S = a.astype(np.int64), None
    else:
        X = to_int_lengths(a / w if divw else a, S)
    G = core.Graph(g.src, g.tgt, g.n_nodes)
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), input=path,
               key=key, S=S, divw=divw, n=g.n_nodes, m=g.n_edges, nonzero_X=int((X > 0).sum()))
    res.update(check(G, w, X))
    res["runtime_s"] = round(time.time() - t00, 1)
    print(json.dumps(res, indent=1))
    (ROOT / "ceiling" / "results" / ("m_dual_" + name + ".json")).write_text(json.dumps(res, indent=1))
