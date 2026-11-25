"""Method CLOSE core: repair of pool duals into a feasible fractional FAS, with the cycles that forced each raise.

raise_cycles: for every head v of the FAS edges F, one Dijkstra from v under integer lengths X (cap S, as
m_dual_core.raise_heads); for every b = (u, v) in F: out[b] = max(X[b], S - dist(v, u)). If the raise is
larger than thr, the cycle (shortest path v ~> u, then b) is written out in traversal order. These are the
most violated cycles through b under X: they are offered to the column pool.
Same C pattern (system cc + ctypes, pthreads) as m_dual_core.py."""
import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

C_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
typedef int64_t i64;

typedef struct { i64 *hk, *hv, sz, cap; } heap;
static void hpush(heap* h, i64 k, i64 v) {
    if (h->sz == h->cap) { h->cap *= 2; h->hk = realloc(h->hk, h->cap * 8); h->hv = realloc(h->hv, h->cap * 8); }
    i64 i = h->sz++;
    while (i > 0) { i64 p = (i - 1) >> 1; if (h->hk[p] <= k) break; h->hk[i] = h->hk[p]; h->hv[i] = h->hv[p]; i = p; }
    h->hk[i] = k; h->hv[i] = v;
}
static void hpop(heap* h, i64* k, i64* v) {
    *k = h->hk[0]; *v = h->hv[0];
    i64 lk = h->hk[--h->sz], lv = h->hv[h->sz], i = 0;
    for (;;) {
        i64 c = 2 * i + 1;
        if (c >= h->sz) break;
        if (c + 1 < h->sz && h->hk[c + 1] < h->hk[c]) c++;
        if (h->hk[c] >= lk) break;
        h->hk[i] = h->hk[c]; h->hv[i] = h->hv[c]; i = c;
    }
    h->hk[i] = lk; h->hv[i] = lv;
}

typedef struct {
    i64 n; const i64 *optr, *oadj, *tgt, *src, *X, *heads, *gptr, *gedge; i64 nh, S, thr, maxlen, max_rec, max_edge, mode;
    i64* out; i64 *rec_ptr, *rec_len, *rec_edge, *edges; volatile i64 *next, *nrec, *nedge;
} rjob;

static void* rc_worker(void* arg) {
    rjob* J = (rjob*)arg;
    i64 n = J->n;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8); i64* need = calloc(n, 8); i64* par = malloc(n * 8);
    i64* tmp = malloc(J->maxlen * 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nh) break;
        i64 v = J->heads[t], st = t + 1, left = 0, B = J->S;
        for (i64 k = J->gptr[t]; k < J->gptr[t + 1]; k++) { i64 u = J->src[J->gedge[k]]; if (need[u] != st) { need[u] = st; left++; } }
        h.sz = 0; dist[v] = 0; stamp[v] = st; par[v] = -1; hpush(&h, 0, v);
        while (h.sz && left > 0) {
            i64 d, u; hpop(&h, &d, &u);
            if (d >= B) break;
            if (d > dist[u]) continue;
            if (need[u] == st) { need[u] = -st; left--; }
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], y = J->tgt[e], nd = d + J->X[e];
                if (nd >= B) continue;
                if (stamp[y] != st || nd < dist[y]) { stamp[y] = st; dist[y] = nd; par[y] = e; hpush(&h, nd, y); }
            }
        }
        for (i64 k = J->gptr[t]; k < J->gptr[t + 1]; k++) {
            i64 b = J->gedge[k], u = J->src[b];
            i64 d = (need[u] == -st) ? dist[u] : J->S;
            i64 r = J->S - d;
            if (J->mode == 1) { J->out[b] = r > 0 ? r : 0; continue; }
            J->out[b] = J->X[b] > r ? J->X[b] : r;
            if (need[u] != -st || r - J->X[b] <= J->thr) continue;
            i64 len = 0, z = u, ok = 1;
            while (z != v) {
                if (len >= J->maxlen - 1) { ok = 0; break; }
                i64 e = par[z]; tmp[len++] = e; z = J->src[e];
            }
            if (!ok) continue;
            i64 ri = __atomic_fetch_add(J->nrec, 1, __ATOMIC_RELAXED);
            i64 off = __atomic_fetch_add(J->nedge, len + 1, __ATOMIC_RELAXED);
            if (ri >= J->max_rec || off + len + 1 > J->max_edge) continue;
            for (i64 q = 0; q < len; q++) J->edges[off + q] = tmp[len - 1 - q];
            J->edges[off + len] = b;
            J->rec_ptr[ri] = off; J->rec_len[ri] = len + 1; J->rec_edge[ri] = b;
        }
    }
    free(dist); free(stamp); free(need); free(par); free(tmp); free(h.hk); free(h.hv);
    return 0;
}

void raise_cycles(i64 n, const i64* optr, const i64* oadj, const i64* src, const i64* tgt, const i64* X,
                  const i64* heads, const i64* gptr, const i64* gedge, i64 nh, i64 S, i64 thr, i64 maxlen,
                  i64 max_rec, i64 max_edge, i64 mode, i64 nthreads, i64* out, i64* rec_ptr, i64* rec_len, i64* rec_edge,
                  i64* edges, i64* counts) {
    volatile i64 next = 0, nrec = 0, nedge = 0;
    rjob J = {n, optr, oadj, tgt, src, X, heads, gptr, gedge, nh, S, thr, maxlen, max_rec, max_edge, mode,
              out, rec_ptr, rec_len, rec_edge, edges, &next, &nrec, &nedge};
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, rc_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
    counts[0] = nrec; counts[1] = nedge;
}
"""

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha256(C_SRC.encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "m_close_kernel"
        d.mkdir(exist_ok=True)
        so = d / ("close_" + tag + ".so")
        if not so.exists():
            c = d / ("close_" + tag + ".c")
            c.write_text(C_SRC)
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-pthread", "-o", str(so), str(c)], check=True)
        L = ctypes.CDLL(str(so))
        P = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        I = ctypes.c_int64
        L.raise_cycles.restype = None
        L.raise_cycles.argtypes = [I, P, P, P, P, P, P, P, P, I, I, I, I, I, I, I, I, P, P, P, P, P, P]
        _LIB = L
    return _LIB


def c64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def raise_cycles(G, X, F, S, thr, chunks=1, threads=6, maxlen=200, mode=0):
    """Raise of the FAS edges F on m_dual_core.Graph G, heads processed in `chunks` sequential groups (X is
    updated between groups, so later groups see earlier raises). Returns (X1, cyc_ptr, cyc_edges, cyc_edge_b).
    Valid for any chunking: lengths only grow, and every F edge b ends with X_b + dist(v, u) >= S.
    mode 1 (lowering heuristic, no validity claim): X_b = max(0, S - dist(v, u)), the smallest value that keeps the
    cycles through b alone at >= S; edges of one chunk are set simultaneously, so a later raise + check is needed."""
    X = c64(X).copy()
    F = c64(F[np.argsort(G.tgt[F], kind="stable")])
    heads, start = np.unique(G.tgt[F], return_index=True)
    gp = np.append(start, F.size)
    rng = np.random.default_rng(0)
    perm = rng.permutation(heads.size)
    parts = []
    for ch in np.array_split(perm, chunks):
        ch = np.sort(ch)
        hd = c64(heads[ch])
        lens = gp[ch + 1] - gp[ch]
        gptr = np.zeros(ch.size + 1, np.int64)
        np.cumsum(lens, out=gptr[1:])
        ge = c64(F[np.repeat(gp[ch] - gptr[:-1], lens) + np.arange(gptr[-1])])
        max_rec = ge.size
        max_edge = max_rec * 24
        out = X.copy()
        rp, rl, rb = np.zeros(max_rec, np.int64), np.zeros(max_rec, np.int64), np.zeros(max_rec, np.int64)
        ed = np.zeros(max_edge, np.int64)
        cnt = np.zeros(2, np.int64)
        lib().raise_cycles(G.n, G.optr, G.oadj, G.src, G.tgt, X, hd, gptr, ge, hd.size, int(S), int(thr), int(maxlen),
                           max_rec, max_edge, int(mode), int(threads), out, rp, rl, rb, ed, cnt)
        X = out
        k = int(min(cnt[0], max_rec))
        ok = (rl[:k] > 0) & (rp[:k] + rl[:k] <= max_edge)
        rp, rl, rb = rp[:k][ok], rl[:k][ok], rb[:k][ok]
        ptr = np.zeros(rl.size + 1, np.int64)
        np.cumsum(rl, out=ptr[1:])
        parts.append((ptr, ed[np.repeat(rp - ptr[:-1], rl) + np.arange(ptr[-1])], rb))
    ptr = np.zeros(sum(p[2].size for p in parts) + 1, np.int64)
    np.cumsum(np.concatenate([np.diff(p[0]) for p in parts]), out=ptr[1:])
    return X, ptr, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
