import sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
from rcheck import rcheck
D = sys.argv[1]
src, tgt, w = np.load(D + "/src.npy"), np.load(D + "/tgt.npy"), np.load(D + "/w.npy")
X = np.load(D + "/../close/X_5_e.npz")["X"].astype(np.int64)
n, S = 136648, 10**6
assert X.min() >= 0 and X.size == src.size
t = time.time()
vmin, vbad = rcheck(src, tgt, X, n, S, nth=4)
print("time s", round(time.time() - t, 1))
print("min over all edges of X_e + dist(v,u), capped at S+1:", int(vmin.min()), "argmin head", int(vmin.argmin()))
print("edges with cycle < S:", int(vbad.sum()), "heads with min == S:", int((vmin == S).sum()))
np.savez(D + "/check_X5e_out.npz", vmin=vmin, vbad=vbad)
sw = sum(int(a) * int(b) for a, b in zip(w.tolist(), X.tolist()))
print("sum wX", sw, "UB", sw / S)
