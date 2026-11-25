"""Baseline K1 certificate: every reciprocal pair (u,v),(v,u) as a 2-cycle, y = min weight, D = 1.

Run from the repo root: python ceiling/src/cert_2cycle.py  -> ceiling/certs/c00_2cycle.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from cert_io import write_cert  # noqa: E402
from src.mfas.io import load_dataset  # noqa: E402

OUT = HERE.parent / "certs" / "c00_2cycle.npz"


def build(graph):
    n = graph.n_nodes
    s, t = graph.src.astype(np.int64), graph.tgt.astype(np.int64)
    key = s * n + t
    order = np.argsort(key)
    skey = key[order]
    fwd = np.flatnonzero(s < t)
    rkey = t[fwd] * n + s[fwd]
    pos = np.minimum(np.searchsorted(skey, rkey), skey.size - 1)
    hit = skey[pos] == rkey
    e1 = fwd[hit]
    e2 = order[pos[hit]]
    y = np.minimum(graph.weight[e1], graph.weight[e2]).astype(np.int64)
    ptr = np.arange(0, 2 * e1.size + 1, 2, dtype=np.int64)
    edges = np.stack([e1, e2], axis=1).ravel().astype(np.int64)
    return ptr, edges, y


def main():
    g = load_dataset("connectome")
    ptr, edges, y = build(g)
    write_cert(OUT, graph=g, D=1, cycles=(ptr, edges), y=y, method="2cycle",
               notes="all reciprocal pairs as 2-cycles, y = min(w(u,v), w(v,u)), D = 1")
    print(OUT, "cycles", len(y), "sum_y", int(y.sum()))


if __name__ == "__main__":
    main()
