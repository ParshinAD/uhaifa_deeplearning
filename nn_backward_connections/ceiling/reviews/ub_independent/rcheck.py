import ctypes, numpy as np
D = "/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/review_dual"
L = ctypes.CDLL(D + "/librc.so")
P = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"); I = ctypes.c_int64
L.rcheck.restype = None
L.rcheck.argtypes = [I, P, P, P, P, P, P, I, I, P, P]

def rcheck(src, tgt, X, n, S, nth=4):
    src, tgt, X = (np.ascontiguousarray(a, np.int64) for a in (src, tgt, X))
    o = np.argsort(src, kind="stable"); optr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))]).astype(np.int64)
    i = np.argsort(tgt, kind="stable"); iptr = np.concatenate([[0], np.cumsum(np.bincount(tgt, minlength=n))]).astype(np.int64)
    vmin = np.zeros(n, np.int64); vbad = np.zeros(n, np.int64)
    L.rcheck(n, optr, np.ascontiguousarray(tgt[o]), np.ascontiguousarray(X[o]), iptr, np.ascontiguousarray(src[i]),
             np.ascontiguousarray(X[i]), int(S), int(nth), vmin, vbad)
    return vmin, vbad
