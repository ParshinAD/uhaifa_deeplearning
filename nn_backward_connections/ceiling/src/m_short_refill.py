"""Integer greedy refill of a K1 certificate with cycles of any length: in the residual graph
(D*w - load > 0) run BFS from random sources, close the shortest cycles back to the source,
pack each at its residual bottleneck. Rebuilds the residual graph every rebuild_every sources.
Usage: m_short_refill.py <in_cert.npz> <out_cert_name> <run_json> [time_limit_s]"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from cert_io import read_cert, write_cert
from src.mfas.io import load_dataset

CONFIG = dict(seed=0, time_limit_s=float(sys.argv[4]) if len(sys.argv) > 4 else 1500, rebuild_every=2000,
              k_per_source=10, passes=5)
PY = sys.executable

t0 = time.time()
g = load_dataset("connectome")
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
n, m, W = g.n_nodes, g.n_edges, int(w.sum())
meta, a = read_cert(sys.argv[1])
D = int(a["D"])
ptr, ce, y = a["cyc_ptr"], a["cyc_edges"], a["y"]
lens = np.diff(ptr)
load = np.bincount(ce, weights=np.repeat(y, lens), minlength=m).astype(np.int64)
r = D * w - load
assert r.min() >= 0
key = si * n + ti
ko = np.argsort(key)
ks = key[ko]
rng = np.random.default_rng(CONFIG["seed"])
new_c, new_y = [], []
gain = 0
stop = False
for ps in range(CONFIG["passes"]):
    srcs = rng.permutation(n)
    gain_pass = 0
    for b0 in range(0, n, CONFIG["rebuild_every"]):
        k = r > 0
        A = sp.csr_matrix((np.ones(int(k.sum()), np.int8), (si[k], ti[k])), shape=(n, n))
        AT = A.T.tocsr()
        for s in srcs[b0:b0 + CONFIG["rebuild_every"]]:
            ins = AT.indices[AT.indptr[s]:AT.indptr[s + 1]]
            if ins.size == 0 or A.indptr[s + 1] == A.indptr[s]:
                continue
            order, pred = breadth_first_order(A, s, directed=True, return_predecessors=True)
            rank = np.full(n, n, np.int64)
            rank[order] = np.arange(order.size)
            ins = ins[rank[ins] < n]
            for t in ins[np.argsort(rank[ins])][:CONFIG["k_per_source"]]:
                path = [int(t)]
                while path[-1] != s:
                    path.append(int(pred[path[-1]]))
                nodes = np.array(path[::-1] + [s], np.int64)
                e = ko[np.searchsorted(ks, nodes[:-1] * n + nodes[1:])]
                f = int(r[e].min())
                if f > 0:
                    r[e] -= f
                    new_c.append(e)
                    new_y.append(f)
                    gain_pass += f
        if time.time() - t0 > CONFIG["time_limit_s"]:
            stop = True
            break
        if b0 % 20000 == 0:
            print("pass", ps, "sources", b0, "new cycles", len(new_c), "gain/D", round((gain + gain_pass) / D, 1),
                  "s", round(time.time() - t0, 1), flush=True)
    gain += gain_pass
    print("pass", ps, "done gain/D", round(gain_pass / D, 1), flush=True)
    if stop or gain_pass == 0:
        break
cycles = [ce[ptr[i]:ptr[i + 1]] for i in range(y.size)] + new_c
yy = np.concatenate([y, np.array(new_y, np.int64)])
cert = HERE.parent / "certs" / sys.argv[2]
write_cert(cert, graph=g, D=D, cycles=cycles, y=yy, method="m_short_refill",
           notes="refill of " + sys.argv[1] + " with BFS shortest residual cycles")
lb = int(yy.sum()) // D
v = subprocess.run([PY, str(HERE / "verify_cert.py"), str(cert)], capture_output=True, text=True, cwd=ROOT)
print(v.stdout[-600:])
nl = np.bincount([len(c) for c in new_c]) if new_c else []
out = dict(command=" ".join([Path(PY).name] + sys.argv), git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           seed=CONFIG["seed"], config=CONFIG, D=D, base_LB=int(y.sum()) // D, new_cycles=len(new_c),
           new_cycle_len_hist={str(i): int(c) for i, c in enumerate(nl) if c}, LB=lb,
           ceiling_pct=round(100 * (W - lb) / W, 4), cert=str(cert.relative_to(ROOT)), verifier_returncode=v.returncode,
           verifier_stdout=v.stdout[-1200:], runtime_s=round(time.time() - t0, 1))
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
print("LB", lb, "ceiling", round(100 * (W - lb) / W, 4), "rc", v.returncode)
