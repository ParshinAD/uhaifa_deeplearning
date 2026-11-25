"""Scout 1+2: graph structure relevant to bounds and the feedback of our champion order."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset
from layering.io_utils import load_graph_and_champion_order

CONFIG = dict(dataset="connectome", ann="layering/outputs/fly_ann/node_ann.npz")


def rev_index(si, ti, n):
    key = si.astype(np.int64) * n + ti
    order = np.argsort(key)
    ks = key[order]
    rk = ti.astype(np.int64) * n + si
    pos = np.clip(np.searchsorted(ks, rk), 0, len(ks) - 1)
    has = ks[pos] == rk
    rev = np.where(has, order[pos], -1)
    return key, rev


def q(x):
    x = np.asarray(x)
    return [int(v) for v in np.percentile(x, [0, 50, 90, 99, 99.9, 100])]


t0 = time.time()
g = load_dataset(CONFIG["dataset"])
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
n, m, W = g.n_nodes, g.n_edges, int(w.sum())
print("n", n, "m", m, "W", W)

print("== duplicates / self loops")
key, rev = rev_index(si, ti, n)
print("self_loops", int((si == ti).sum()), "duplicate_edges", int(m - np.unique(key).size))
print("weight min", int(w.min()), "weight percentiles [0,50,90,99,99.9,100]", q(w))
print("edges with w=1", int((w == 1).sum()), "share", round(float((w == 1).mean()), 4))
outdeg = np.bincount(si, minlength=n)
indeg = np.bincount(ti, minlength=n)
outw = np.bincount(si, weights=w, minlength=n)
inw = np.bincount(ti, weights=w, minlength=n)
print("outdeg pct", q(outdeg), "indeg pct", q(indeg))
print("out-strength pct", q(outw), "in-strength pct", q(inw))
print("sources (indeg 0)", int((indeg == 0).sum()), "sinks (outdeg 0)", int((outdeg == 0).sum()))

print("== reciprocal pairs")
has = rev >= 0
w_rev = np.where(has, w[np.maximum(rev, 0)], 0)
pack2 = np.where(has, np.minimum(w, w_rev), 0)
print("reciprocal pairs", int(has.sum() // 2), "edges in pairs", int(has.sum()))
print("weight on reciprocal edges", int(w[has].sum()), "share of W", round(float(w[has].sum() / W), 4))
lb2 = int(pack2.sum() // 2)
print("2-cycle floor LB", lb2, "ceiling", round(100 * (W - lb2) / W, 4))
print("balanced pairs (w_uv == w_vu)", int(((w == w_rev) & has).sum() // 2))

print("== SCC")
A = sp.csr_matrix((np.ones(m, np.int8), (si, ti)), shape=(n, n))
ncomp, lab = connected_components(A, directed=True, connection="strong")
sizes = np.bincount(lab)
big = int(np.argmax(sizes))
print("scc count", ncomp, "giant size", int(sizes[big]), "size>1 count", int((sizes > 1).sum()),
      "second largest", int(np.sort(sizes)[-2]), "singletons", int((sizes == 1).sum()))
inside = lab[si] == lab[ti]
print("weight inside SCCs", int(w[inside].sum()), "between SCCs (free FF)", int(w[~inside].sum()))
ing = inside & (lab[si] == big)
print("giant SCC: edges", int(ing.sum()), "weight", int(w[ing].sum()))
cap = w - pack2
print("giant SCC residual after 2-cycle packing: weight", int(cap[ing].sum()),
      "edges with cap>0", int((ing & (cap > 0)).sum()))
k2 = cap > 0
A2 = sp.csr_matrix((np.ones(int(k2.sum()), np.int8), (si[k2], ti[k2])), shape=(n, n))
nc2, lab2 = connected_components(A2, directed=True, connection="strong")
s2 = np.bincount(lab2)
in2 = lab2[si] == lab2[ti]
print("SCC of residual graph (cap>0 edges): count", nc2, "giant", int(s2.max()),
      "residual weight inside SCCs", int(cap[k2 & in2].sum()), "residual between (no more cycles)", int(cap[k2 & ~in2].sum()))
max_possible_lb = lb2 + int(cap[k2 & in2].sum())
print("trivial max: LB can never exceed 2-cycle LB + residual-in-SCC weight =", max_possible_lb,
      "(=> ceiling >=", round(100 * (W - max_possible_lb) / W, 4), ")")

print("== champion feedback")
g2, order, rank, ff = load_graph_and_champion_order(CONFIG["dataset"])
assert np.array_equal(g2.src, g.src) and np.array_equal(g2.weight, g.weight)
back = rank[si] >= rank[ti]
fb = int(w[back].sum())
print("champion ff%", round(ff, 6), "FF weight", W - fb, "backward weight", fb, "backward edges", int(back.sum()))
print("(a) backward weight on reciprocal edges", int(w[back & has].sum()),
      "(b) on non-reciprocal edges", int(w[back & ~has].sum()))
print("2-cycle floor part: sum over pairs of min(w) =", lb2,
      "; excess paid on reciprocal pairs above min:", int(w[back & has].sum()) - lb2)
both_back = back & has & back[np.maximum(rev, 0)]
print("reciprocal pairs with both edges backward (impossible in strict order):", int(both_back.sum()))
heavier_back = back & has & (w > w_rev)
print("reciprocal pairs where the heavier side is backward:", int(heavier_back.sum()),
      "extra weight", int((w - w_rev)[heavier_back].sum()))
print("backward weight inside giant SCC", int(w[back & ing].sum()), "outside giant", int(w[back & ~ing].sum()))
span = (rank[si] - rank[ti])[back]
print("rank span of backward edges pct [0,50,90,99,99.9,100]", q(span))
for lo, hi in [(1, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 100000), (100001, n)]:
    msk = (span >= lo) & (span <= hi)
    print("  span", lo, "-", hi, "edges", int(msk.sum()), "weight", int(w[back][msk].sum()))
print("backward weight percentiles", q(w[back]), "backward edges w=1", int((w[back] == 1).sum()))

print("== per super_class lane (both endpoints in same class)")
ann = np.load(ROOT / CONFIG["ann"], allow_pickle=False)
import json
meta = json.loads((ROOT / "layering/outputs/fly_ann/node_ann_meta.json").read_text())
names = meta["super_class_values"]
sc = ann["super_class"].astype(np.int64)
same = sc[si] == sc[ti]
print("class | within-class W | within backward | within 2-cycle floor | floor/backward")
for c in range(len(names)):
    msk = same & (sc[si] == c)
    if not msk.any():
        continue
    fl = int(pack2[msk].sum() // 2)
    bw = int(w[msk & back].sum())
    print(" ", names[c], int(w[msk].sum()), bw, fl, round(fl / max(bw, 1), 3))
cross = ~same
print("  cross-class: W", int(w[cross].sum()), "backward", int(w[cross & back].sum()),
      "2-cycle floor", int(pack2[cross].sum() // 2))
print("elapsed s", round(time.time() - t0, 1))
