"""Round a fractional packing over explicit cycle columns to a K1 certificate:
y_int = floor(D * y), repair any per-edge overload, then greedily refill the leftover integer
capacity D*w - load with columns of the pool (shortest dual length first if x or l is given, else largest
bottleneck first). Writes the cert, runs the verifier, writes the run JSON.
Usage: m_short_cert.py <cols.npz> <y.npz> <D> <cert_name> <run_json> [refill 0/1]"""
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
from cert_io import write_cert
from src.mfas.io import load_dataset

CONFIG = dict(D=int(sys.argv[3]), refill=int(sys.argv[6]) if len(sys.argv) > 6 else 1)
PY = sys.executable

t0 = time.time()
g = load_dataset("connectome")
w = g.weight.astype(np.int64)
W = int(w.sum())
m = g.n_edges
D = CONFIG["D"]
z = np.load(sys.argv[1])
keys = [k for k in z.files if k.startswith("c")]
cols = [z[k].astype(np.int64) for k in keys]
yz = np.load(sys.argv[2])
yf = yz["y"]
ptr = np.cumsum([0] + [c.shape[0] for c in cols])
assert yf.size == ptr[-1]
yi = [np.floor(yf[ptr[i]:ptr[i + 1]] * D).astype(np.int64) for i in range(len(cols))]


def loads(yi):
    tot = np.zeros(m, np.int64)
    for c, yy in zip(cols, yi):
        tot += np.bincount(c.ravel(), weights=np.repeat(yy, c.shape[1]), minlength=m).astype(np.int64)
    return tot


cap = D * w
tot = loads(yi)
over = tot > cap
print("float sum", round(float(yf.sum()), 2), "floored sum/D", round(sum(int(x.sum()) for x in yi) / D, 2),
      "overloaded edges", int(over.sum()), flush=True)
while over.any():
    for c, yy in zip(cols, yi):
        hit = over[c].any(axis=1) & (yy > 0)
        excess = (tot - cap)[c[hit]].max(axis=1)
        yy[hit] -= np.minimum(yy[hit], excess)
    tot = loads(yi)
    over = tot > cap
lb_round = sum(int(x.sum()) for x in yi)
res = cap - tot
refill_gain = 0
if CONFIG["refill"]:
    if "x" in yz.files or "l" in yz.files:
        l = yz["x"] / w if "x" in yz.files else yz["l"]
        pri = np.concatenate([l[c].sum(axis=1) for c in cols])
    else:
        pri = -np.concatenate([w[c].min(axis=1) for c in cols]).astype(np.float64)
    flat = np.argsort(pri, kind="stable")
    grp = np.searchsorted(ptr, flat, side="right") - 1
    for idx, gi in zip(flat.tolist(), grp.tolist()):
        c = cols[gi][idx - ptr[gi]]
        f = int(res[c].min())
        if f > 0:
            res[c] -= f
            yi[gi][idx - ptr[gi]] += f
            refill_gain += f
print("after repair sum/D", round(lb_round / D, 2), "refill gain/D", round(refill_gain / D, 2), flush=True)
cyc, yy = [], []
for c, y_ in zip(cols, yi):
    k = y_ > 0
    cyc.append(c[k])
    yy.append(y_[k])
lens = np.concatenate([np.full(c.shape[0], c.shape[1]) for c in cyc])
cptr = np.zeros(lens.size + 1, np.int64)
np.cumsum(lens, out=cptr[1:])
flat_e = np.concatenate([c.ravel() for c in cyc])
yy = np.concatenate(yy)
cert = HERE.parent / "certs" / sys.argv[4]
write_cert(cert, graph=g, D=D, cycles=(cptr, flat_e), y=yy, method="m_short",
           notes="pool " + sys.argv[1] + " y " + sys.argv[2] + " floor*D + repair + refill " + str(CONFIG["refill"]))
lb = int(yy.sum()) // D
t_build = round(time.time() - t0, 1)
v = subprocess.run([PY, str(HERE / "verify_cert.py"), str(cert)], capture_output=True, text=True, cwd=ROOT)
print(v.stdout[-2000:], v.stderr[-2000:])
src_json = Path(sys.argv[2]).with_suffix(".json")
info = json.loads(src_json.read_text()) if src_json.exists() else {}
out = dict(command=" ".join([Path(PY).name] + sys.argv), git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           seed=None, D=D, columns_by_length={str(c.shape[1]): int(c.shape[0]) for c in cols},
           cycles_in_cert=int(yy.size), LB=lb, ceiling_pct=round(100 * (W - lb) / W, 4),
           lb_before_refill=lb_round // D, fractional_input=float(yf.sum()),
           dual_estimate=info.get("best_dual", info.get("dual_estimate")), solver_info=info,
           cert=str(cert.relative_to(ROOT)), verifier_returncode=v.returncode, verifier_stdout=v.stdout[-1500:],
           runtime_cert_s=t_build, runtime_total_s=round(time.time() - t0, 1))
Path(sys.argv[5]).write_text(json.dumps(out, indent=1))
print("LB", lb, "ceiling", round(100 * (W - lb) / W, 4), "verifier rc", v.returncode)
