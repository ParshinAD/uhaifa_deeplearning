"""Scout 4: timings of graph / LP primitives on the giant SCC with residual capacities (after 2-cycle packing).

Usage: python scout_primitives.py graph | lp <n_cols> random|tri3 | torch\ntri3 uses the 3-cycles saved by scout_tri_full.py (residual capacities after 2-cycle packing)."""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra, breadth_first_order

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(tri_file="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/tri3_residual.npz", seed=0, n_batch=100, tri_rows=2000, lp_time_limit=400.0, lp_nnz_per_col=3, lp_rows_per_col=0.5)


def maxrss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)


def residual_giant():
    g = load_dataset("connectome")
    si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
    n, m = g.n_nodes, g.n_edges
    key = si * n + ti
    order = np.argsort(key)
    ks = key[order]
    rk = ti * n + si
    pos = np.clip(np.searchsorted(ks, rk), 0, m - 1)
    has = ks[pos] == rk
    cap = w - np.where(has, np.minimum(w, w[order[pos]]), 0)
    k = cap > 0
    A = sp.csr_matrix((np.ones(int(k.sum())), (si[k], ti[k])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="strong")
    big = np.argmax(np.bincount(lab))
    nodes = np.flatnonzero(lab == big)
    remap = -np.ones(n, np.int64)
    remap[nodes] = np.arange(nodes.size)
    e = k & (lab[si] == big) & (lab[ti] == big)
    return remap[si[e]], remap[ti[e]], cap[e], nodes.size


def run_graph():
    rng = np.random.default_rng(CONFIG["seed"])
    t = time.time()
    s, d, c, nn = residual_giant()
    print("residual giant SCC: nodes", nn, "edges", s.size, "cap sum", int(c.sum()), "build s", round(time.time() - t, 1))
    length = 1.0 / c
    G = sp.csr_matrix((length, (s, d)), shape=(nn, nn))
    src = rng.choice(nn, CONFIG["n_batch"], replace=False)

    t = time.time()
    dist = dijkstra(G, directed=True, indices=int(src[0]))
    print("scipy dijkstra 1 source s", round(time.time() - t, 3), "reachable", int(np.isfinite(dist).sum()))
    t = time.time()
    dist = dijkstra(G, directed=True, indices=src[:10])
    print("scipy dijkstra 10 sources s", round(time.time() - t, 3))
    t = time.time()
    dist = dijkstra(G, directed=True, indices=src)
    print("scipy dijkstra 100 sources s", round(time.time() - t, 3), "out MB", round(dist.nbytes / 2 ** 20, 1))
    t = time.time()
    dist = dijkstra(G, directed=True, indices=int(src[0]), limit=float(np.median(length)) * 3)
    print("scipy dijkstra 1 source with limit=3*median length s", round(time.time() - t, 3),
          "reached", int(np.isfinite(dist).sum()))
    t = time.time()
    bo = breadth_first_order(G, int(src[0]), directed=True, return_predecessors=False)
    print("scipy BFS 1 source s", round(time.time() - t, 3), "visited", bo.size)
    t = time.time()
    du = dijkstra(G, directed=True, indices=int(src[0]), unweighted=True)
    print("scipy dijkstra unweighted (hop BFS distances) s", round(time.time() - t, 3), "max hops", int(du[np.isfinite(du)].max()))

    import igraph as ig
    t = time.time()
    H = ig.Graph(n=nn, edges=np.column_stack([s, d]).tolist(), directed=True)
    print("igraph build s", round(time.time() - t, 1))
    lw = length.tolist()
    t = time.time()
    dd = H.distances(source=[int(src[0])], weights=lw, mode="out")
    print("igraph distances 1 source s", round(time.time() - t, 3))
    t = time.time()
    dd = H.distances(source=src[:10].tolist(), weights=lw, mode="out")
    print("igraph distances 10 sources s", round(time.time() - t, 3))
    t = time.time()
    bf = H.bfs(int(src[0]), mode="out")
    print("igraph BFS 1 source s", round(time.time() - t, 3))
    del H

    print("== vectorized 3-cycle enumeration rate (A[rows] @ A, masked by A^T)")
    B = sp.csr_matrix((np.ones(s.size, np.float32), (s, d)), shape=(nn, nn))
    BT = B.T.tocsr()
    rows = rng.choice(nn, CONFIG["tri_rows"], replace=False)
    t = time.time()
    P = B[rows] @ B
    tp = time.time() - t
    T = P.multiply(BT[rows])
    tri = float(T.sum())
    el = time.time() - t
    print("rows", rows.size, "nnz(A^2 rows)", P.nnz, "3-cycle paths through these start nodes", int(tri),
          "s", round(el, 2), "(matmul", round(tp, 2), ")")
    est_total = tri / rows.size * nn / 3
    print("extrapolated directed 3-cycles in residual giant (rough, random rows)", int(est_total),
          "est full enumeration s", round(el / rows.size * nn, 0), "triangles per s", int(tri / el))
    print("maxrss GB", maxrss_gb())


def run_lp(ncol, kind):
    from scipy.optimize import linprog
    rng = np.random.default_rng(CONFIG["seed"])
    if kind == "random":
        nrow = int(ncol * CONFIG["lp_rows_per_col"])
        k = CONFIG["lp_nnz_per_col"]
        rows = rng.integers(0, nrow, size=(ncol, k)).ravel()
        b = rng.integers(1, 50, size=nrow).astype(float)
    else:
        z = np.load(CONFIG["tri_file"])
        sel = rng.permutation(z["e1"].size)[:ncol]
        ncol = sel.size
        ee = np.column_stack([z["e1"][sel], z["e2"][sel], z["e3"][sel]])
        used, rows = np.unique(ee.ravel(), return_inverse=True)
        nrow = used.size
        b = z["cap"][used].astype(float)
        k = 3
    cols = np.repeat(np.arange(ncol), k)
    M = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(nrow, ncol))
    print("kind", kind, flush=True)
    t = time.time()
    r = linprog(-np.ones(ncol), A_ub=M, b_ub=b, bounds=(0, None), method="highs",
                options=dict(time_limit=CONFIG["lp_time_limit"]))
    el = time.time() - t
    print("LP cols", ncol, "rows", nrow, "nnz", M.nnz, "status", r.status, r.message.strip()[:60],
          "obj", round(float(-r.fun), 2) if r.fun is not None else None, "s", round(el, 2), "maxrss GB", maxrss_gb(), flush=True)
    for meth in ["highs-ipm"]:
        t = time.time()
        r = linprog(-np.ones(ncol), A_ub=M, b_ub=b, bounds=(0, None), method=meth,
                    options=dict(time_limit=CONFIG["lp_time_limit"]))
        print(meth, "status", r.status, "obj", round(float(-r.fun), 2) if r.fun is not None else None,
              "s", round(time.time() - t, 2), "maxrss GB", maxrss_gb(), flush=True)


def run_torch():
    import torch
    rng = np.random.default_rng(CONFIG["seed"])
    s, d, c, nn = residual_giant()
    dev = torch.device("mps")
    idx = torch.tensor(np.vstack([s, d]), dtype=torch.int64)
    val = torch.ones(s.size, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, (nn, nn)).coalesce()
    X = torch.tensor(rng.random((nn, 64), dtype=np.float32))
    t = time.time()
    for _ in range(10):
        Y = torch.sparse.mm(A, X)
    print("torch CPU sparse(A) @ dense(n x 64) x10 s", round(time.time() - t, 3))
    try:
        Am = A.to(dev)
        Xm = X.to(dev)
        Y = torch.sparse.mm(Am, Xm)
        torch.mps.synchronize()
        t = time.time()
        for _ in range(10):
            Y = torch.sparse.mm(Am, Xm)
        torch.mps.synchronize()
        print("torch MPS sparse(A) @ dense(n x 64) x10 s", round(time.time() - t, 3))
    except Exception as ex:
        print("torch MPS sparse mm failed:", type(ex).__name__, str(ex)[:150])
    try:
        Ac = A.to_sparse_csr().to(dev)
        Y = Ac @ Xm
        torch.mps.synchronize()
        t = time.time()
        for _ in range(10):
            Y = Ac @ Xm
        torch.mps.synchronize()
        print("torch MPS CSR @ dense x10 s", round(time.time() - t, 3))
    except Exception as ex:
        print("torch MPS CSR mm failed:", type(ex).__name__, str(ex)[:150])
    As = sp.csr_matrix((np.ones(s.size, np.float32), (s, d)), shape=(nn, nn))
    Xn = X.numpy()
    t = time.time()
    for _ in range(10):
        Yn = As @ Xn
    print("scipy CSR @ dense(n x 64) x10 s", round(time.time() - t, 3))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "graph":
        run_graph()
    elif mode == "lp":
        run_lp(int(float(sys.argv[2])), sys.argv[3])
    elif mode == "torch":
        run_torch()
