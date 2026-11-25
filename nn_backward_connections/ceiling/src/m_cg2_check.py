"""Convergence check for the cycle packing LP: exact pricing (Dijkstra with the true lengths x/w, no floor,
no pop limit) from every node, stop per source at the first cycle with l(C) < 1 - margin.
If no source finds one, the pool duals are feasible for the full cycle LP (up to the margin).
Usage: m_cg2_check.py <duals.npz> <tag> [margin]"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from m_cg2_core import lib
from src.mfas.io import load_dataset

CONFIG = dict(duals=sys.argv[1], margin=float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4, k=1, threads=8)
t0 = time.time()
L = lib()
g = load_dataset("connectome")
n, m = g.n_nodes, g.n_edges
w = g.weight.astype(np.float64)
esrc, etgt = g.src.astype(np.int32), g.tgt.astype(np.int32)
x = np.load(sys.argv[1])["x"]
ln = np.maximum(x / w, 0)
oo = np.lexsort((etgt, esrc))
optr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(esrc, minlength=n), out=optr[1:])
io = np.lexsort((esrc, etgt))
iptr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(etgt, minlength=n), out=iptr[1:])
srcs = np.flatnonzero((np.diff(optr) > 0) & (np.diff(iptr) > 0)).astype(np.int64)
mr = srcs.size
rp, rs = np.zeros(mr, np.int64), np.zeros(mr, np.int32)
rl, rh = np.zeros(mr), np.zeros(mr, np.uint64)
ed = np.zeros(mr * 200, np.int32)
st = np.zeros(6, np.int64)
nrec = L.price(n, optr, np.ascontiguousarray(etgt[oo]), oo.astype(np.int32), iptr, np.ascontiguousarray(esrc[io]),
               io.astype(np.int32), esrc, ln, np.ascontiguousarray(ln[oo]), srcs, srcs.size, 1, 1.0 - CONFIG["margin"], 0,
               mr, ed.size, rp, rs, rl, rh, ed, st, CONFIG["threads"])
rl = rl[:nrec]
out = dict(command=" ".join([Path(sys.executable).name] + sys.argv), config=CONFIG,
           git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           sum_x=float(x.sum()), sources=int(srcs.size), sources_with_violated_cycle=int(nrec),
           min_violated_len=float(rl.min()) if nrec else None, mean_violated_len=float(rl.mean()) if nrec else None,
           violated_len_quantiles=[round(float(v), 4) for v in np.quantile(rl, [0.01, 0.1, 0.5, 0.9])] if nrec else None,
           cycle_len_hist={int(a): int(b) for a, b in zip(*np.unique(rs[:nrec], return_counts=True))},
           overflow=int(st[4]), runtime_s=round(time.time() - t0, 1))
(HERE.parent / "results" / ("m_cg2_" + sys.argv[2] + ".json")).write_text(json.dumps(out, indent=1))
print(json.dumps(out))
