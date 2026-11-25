"""Independent checker, does not import verify_cert / cert_io / src.mfas.io."""
import gzip, hashlib, json, sys, zipfile
import numpy as np
import pandas as pd

REPO = "/Users/abed359/IdeaProjects/university/deeplearning_thesis_ceiling/nn_backward_connections"
OUT = "/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/redteam/graph.npz"


def build_graph():
    df = pd.read_csv(REPO + "/data/connectome_graph.csv.gz", dtype=str)
    print("columns", list(df.columns), "rows", len(df))
    s = np.array([int(x) for x in df.iloc[:, 0]], dtype=np.int64)
    t = np.array([int(x) for x in df.iloc[:, 1]], dtype=np.int64)
    w = np.array([int(x) for x in df.iloc[:, 2]], dtype=np.int64)
    ids = np.unique(np.concatenate([s, t]))
    si, ti = np.searchsorted(ids, s), np.searchsorted(ids, t)
    assert (ids[si] == s).all() and (ids[ti] == t).all()
    np.savez(OUT, src=si.astype(np.int64), tgt=ti.astype(np.int64), w=w, n=len(ids))


def check(path, src, tgt, w):
    z = np.load(path, allow_pickle=False)
    print(path, sorted(z.files))
    for k in z.files:
        if k != "meta":
            assert z[k].dtype == np.int64, k
    D = int(z["D"])
    ptr, ce, y = z["cyc_ptr"], z["cyc_edges"], z["y"]
    assert D >= 1 and ptr[0] == 0 and ptr[-1] == ce.size and len(y) == len(ptr) - 1
    L = np.diff(ptr)
    assert (L >= 2).all() and (y >= 1).all()
    assert ce.min() >= 0 and ce.max() < len(w)
    C = len(y)
    cid = np.repeat(np.arange(C), L)
    pos = np.arange(ce.size) - ptr[cid]
    nxt = np.where(pos == L[cid] - 1, ptr[cid], np.arange(ce.size) + 1)
    closure_bad = int(np.count_nonzero(tgt[ce] != src[ce[nxt]]))
    nodes = src[ce]
    o = np.lexsort((nodes, cid))
    rep = int(np.count_nonzero((cid[o][1:] == cid[o][:-1]) & (nodes[o][1:] == nodes[o][:-1])))
    # also targets distinct (redundant given closure, checked anyway)
    nodes2 = tgt[ce]
    o2 = np.lexsort((nodes2, cid))
    rep2 = int(np.count_nonzero((cid[o2][1:] == cid[o2][:-1]) & (nodes2[o2][1:] == nodes2[o2][:-1])))
    sy = sum(int(v) for v in y.tolist())
    assert sy < 2**62
    load = np.zeros(len(w), dtype=np.int64)
    np.add.at(load, ce, np.repeat(y, L))
    capw = [int(a) * D for a in w.tolist()]
    over = sum(1 for a, b in zip(load.tolist(), capw) if a > b)
    LB = sy // D
    print(dict(D=D, C=C, entries=int(ce.size), maxlen=int(L.max()), closure_bad=closure_bad,
               repeat_src=rep, repeat_tgt=rep2, overloaded=over, sum_y=sy, LB=LB,
               tight=int(sum(1 for a, b in zip(load.tolist(), capw) if a == b))))
    return ptr, ce, y, D, LB


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build_graph()
    else:
        g = np.load(OUT)
        src, tgt, w = g["src"], g["tgt"], g["w"]
        print("n", int(g["n"]), "m", len(w), "W", int(w.sum()), "selfloops", int((src == tgt).sum()),
              "minw", int(w.min()))
        key = src * int(g["n"]) + tgt
        print("duplicate (u,v) pairs", len(key) - len(np.unique(key)))
        rank = np.load(REPO + "/results/champions/H64_connectome_s42_cuda.npz")["rank"].astype(np.int64)
        assert sorted(rank.tolist()) == list(range(int(g["n"])))
        back = rank[tgt] <= rank[src]
        FB = int(w[back].sum())
        print("champion feedback", FB, "ff", int(w.sum()) - FB)
        for p in sys.argv[1:]:
            ptr, ce, y, D, LB = check(p, src, tgt, w)
            L = np.diff(ptr)
            cid = np.repeat(np.arange(len(y)), L)
            hit = np.bincount(cid, weights=back[ce].astype(np.int64), minlength=len(y))
            print("cycles with no champion-backward edge:", int((hit == 0).sum()),
                  "FB >= LB:", FB >= LB, "gap", FB - LB)
