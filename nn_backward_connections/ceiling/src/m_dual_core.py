"""Method D core: integer shortest-cycle kernels (C via ctypes, pthreads) for fractional FAS duals.

Lengths X are int64 per edge id. Graph as CSR over out-edges holding edge ids.
- min_cycle: exact L = min over directed cycles C of X(C). Every cycle is found from its vertex of
  smallest rank (Dijkstra from s restricted to vertices of larger rank), pruned by the best L so far.
- pdist: batch of capped point distances dist_X(a, b), returns min(dist, cap).
- cyc_through: for each source s, the shortest cycle through s (unrestricted Dijkstra), with edges.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

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
    i64 n; const i64 *optr, *oadj, *tgt, *X, *rank, *srcs; i64 nsrc;
    volatile i64 *best; volatile i64 *next; i64 *wit; pthread_mutex_t* mu;
    const i64 *qa, *qb, *qcap; i64* qout; i64 nq;
    i64 *cyc_len, *cyc_par_out, *cyc_ptr; i64 maxlen; i64 nk;
} job;

static void* mc_worker(void* arg) {
    job* J = (job*)arg;
    i64 n = J->n;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nsrc) break;
        i64 s = J->srcs[t], rs = J->rank[s], st = t + 1;
        h.sz = 0; dist[s] = 0; stamp[s] = st; hpush(&h, 0, s);
        while (h.sz) {
            i64 d, u; hpop(&h, &d, &u);
            i64 B = *J->best;
            if (d >= B) break;
            if (d > dist[u]) continue;
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], v = J->tgt[e], nd = d + J->X[e];
                if (nd >= B) continue;
                if (v == s) {
                    pthread_mutex_lock(J->mu);
                    if (nd < *J->best) { *J->best = nd; J->wit[0] = s; J->wit[1] = e; }
                    pthread_mutex_unlock(J->mu);
                    B = *J->best;
                    continue;
                }
                if (J->rank[v] <= rs) continue;
                if (stamp[v] != st || nd < dist[v]) { stamp[v] = st; dist[v] = nd; hpush(&h, nd, v); }
            }
        }
    }
    free(dist); free(stamp); free(h.hk); free(h.hv);
    return 0;
}

i64 min_cycle(i64 n, const i64* optr, const i64* oadj, const i64* tgt, const i64* X, const i64* rank,
              const i64* srcs, i64 nsrc, i64 nthreads, i64 cutoff, i64* wit) {
    volatile i64 best = cutoff, next = 0;
    pthread_mutex_t mu; pthread_mutex_init(&mu, 0);
    job J = {n, optr, oadj, tgt, X, rank, srcs, nsrc, &best, &next, wit, &mu};
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, mc_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
    return best;
}

static void* pd_worker(void* arg) {
    job* J = (job*)arg;
    i64 n = J->n;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nq) break;
        i64 a = J->qa[t], b = J->qb[t], C = J->qcap[t], st = t + 1, res = C;
        if (a == b) { J->qout[t] = 0 < C ? 0 : C; continue; }
        h.sz = 0; dist[a] = 0; stamp[a] = st; hpush(&h, 0, a);
        while (h.sz) {
            i64 d, u; hpop(&h, &d, &u);
            if (d >= C) break;
            if (d > dist[u]) continue;
            if (u == b) { res = d; break; }
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], v = J->tgt[e], nd = d + J->X[e];
                if (nd >= C) continue;
                if (stamp[v] != st || nd < dist[v]) { stamp[v] = st; dist[v] = nd; hpush(&h, nd, v); }
            }
        }
        J->qout[t] = res;
    }
    free(dist); free(stamp); free(h.hk); free(h.hv);
    return 0;
}

void pdist(i64 n, const i64* optr, const i64* oadj, const i64* tgt, const i64* X,
           const i64* qa, const i64* qb, const i64* qcap, i64 nq, i64 nthreads, i64* out) {
    volatile i64 next = 0;
    job J = {0};
    J.n = n; J.optr = optr; J.oadj = oadj; J.tgt = tgt; J.X = X; J.next = &next;
    J.qa = qa; J.qb = qb; J.qcap = qcap; J.qout = out; J.nq = nq;
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, pd_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
}

/* shortest cycle through each source: Dijkstra from s, close with any edge u->s.
   writes length (or cap) and up to maxlen edge ids (in cycle order, closing edge last). */
static void* ct_worker(void* arg) {
    job* J = (job*)arg;
    i64 n = J->n;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8); i64* par = malloc(n * 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    i64* tmp = malloc((J->maxlen + 1) * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nsrc) break;
        i64 s = J->srcs[t], st = t + 1, B = J->qcap[t], be = -1;
        h.sz = 0; dist[s] = 0; stamp[s] = st; par[s] = -1; hpush(&h, 0, s);
        while (h.sz) {
            i64 d, u; hpop(&h, &d, &u);
            if (d >= B) break;
            if (d > dist[u]) continue;
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], v = J->tgt[e], nd = d + J->X[e];
                if (nd >= B) continue;
                if (v == s) { B = nd; be = e; continue; }
                if (stamp[v] != st || nd < dist[v]) { stamp[v] = st; dist[v] = nd; par[v] = e; hpush(&h, nd, v); }
            }
        }
        J->cyc_len[t] = B;
        i64 c = 0;
        if (be >= 0) {
            tmp[c++] = be;
            i64 u = J->wit[be];
            while (u != s && c <= J->maxlen) { i64 e = par[u]; tmp[c++] = e; u = J->wit[e]; }
            if (c > J->maxlen) c = 0;
        }
        i64* o = J->cyc_par_out + t * J->maxlen;
        for (i64 i = 0; i < c; i++) o[i] = tmp[c - 1 - i];
        J->cyc_ptr[t] = c;
    }
    free(dist); free(stamp); free(par); free(h.hk); free(h.hv); free(tmp);
    return 0;
}

void cyc_through(i64 n, const i64* optr, const i64* oadj, const i64* src, const i64* tgt, const i64* X,
                 const i64* srcs, const i64* caps, i64 nsrc, i64 nthreads, i64 maxlen,
                 i64* out_len, i64* out_edges, i64* out_cnt) {
    volatile i64 next = 0;
    job J = {0};
    J.n = n; J.optr = optr; J.oadj = oadj; J.tgt = tgt; J.X = X; J.next = &next;
    J.srcs = srcs; J.nsrc = nsrc; J.qcap = caps; J.wit = (i64*)src; J.maxlen = maxlen;
    J.cyc_len = out_len; J.cyc_par_out = out_edges; J.cyc_ptr = out_cnt;
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, ct_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
}

/* up to nk shortest cycles through each source, one per closing edge u->s, all with length < cap. */
static void* cm_worker(void* arg) {
    job* J = (job*)arg;
    i64 n = J->n, K = J->nk;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8); i64* par = malloc(n * 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    i64* tmp = malloc((J->maxlen + 1) * 8);
    i64* ce = malloc(K * 8); i64* cl = malloc(K * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nsrc) break;
        i64 s = J->srcs[t], st = t + 1, B = J->qcap[t], nc = 0;
        h.sz = 0; dist[s] = 0; stamp[s] = st; par[s] = -1; hpush(&h, 0, s);
        while (h.sz) {
            i64 d, u; hpop(&h, &d, &u);
            i64 lim = nc == K ? cl[K - 1] : B;
            if (d >= lim) break;
            if (d > dist[u]) continue;
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], v = J->tgt[e], nd = d + J->X[e];
                if (nd >= B) continue;
                if (v == s) {
                    if (nc == K && nd >= cl[K - 1]) continue;
                    i64 j = nc < K ? nc++ : K - 1;
                    while (j > 0 && cl[j - 1] > nd) { cl[j] = cl[j - 1]; ce[j] = ce[j - 1]; j--; }
                    cl[j] = nd; ce[j] = e;
                    continue;
                }
                if (stamp[v] != st || nd < dist[v]) { stamp[v] = st; dist[v] = nd; par[v] = e; hpush(&h, nd, v); }
            }
        }
        for (i64 j = 0; j < K; j++) {
            i64 slot = t * K + j, c = 0;
            if (j < nc) {
                tmp[c++] = ce[j];
                i64 u = J->wit[ce[j]];
                while (u != s && c <= J->maxlen) { i64 e = par[u]; tmp[c++] = e; u = J->wit[e]; }
                if (c > J->maxlen) c = 0;
                J->cyc_len[slot] = cl[j];
            } else J->cyc_len[slot] = B;
            i64* o = J->cyc_par_out + slot * J->maxlen;
            for (i64 i = 0; i < c; i++) o[i] = tmp[c - 1 - i];
            J->cyc_ptr[slot] = c;
        }
    }
    free(dist); free(stamp); free(par); free(h.hk); free(h.hv); free(tmp); free(ce); free(cl);
    return 0;
}

void cyc_multi(i64 n, const i64* optr, const i64* oadj, const i64* src, const i64* tgt, const i64* X,
               const i64* srcs, const i64* caps, i64 nsrc, i64 nthreads, i64 maxlen, i64 nk,
               i64* out_len, i64* out_edges, i64* out_cnt) {
    volatile i64 next = 0;
    job J = {0};
    J.n = n; J.optr = optr; J.oadj = oadj; J.tgt = tgt; J.X = X; J.next = &next;
    J.srcs = srcs; J.nsrc = nsrc; J.qcap = caps; J.wit = (i64*)src; J.maxlen = maxlen; J.nk = nk;
    J.cyc_len = out_len; J.cyc_par_out = out_edges; J.cyc_ptr = out_cnt;
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, cm_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
}

/* stale raise: for each head v, Dijkstra from v (cap S); for every FAS edge b = (u, v) in group v:
   out[b] = max(X[b], S - dist(v, u)). All distances use the input X (stale, so the raise is valid). */
typedef struct { i64 n; const i64 *optr, *oadj, *tgt, *src, *X, *heads, *gptr, *gedge; i64 nh, S; i64* out; volatile i64* next; } rjob;
static void* rh_worker(void* arg) {
    rjob* J = (rjob*)arg;
    i64 n = J->n;
    i64* dist = malloc(n * 8); i64* stamp = calloc(n, 8); i64* need = calloc(n, 8);
    heap h; h.cap = 1 << 16; h.sz = 0; h.hk = malloc(h.cap * 8); h.hv = malloc(h.cap * 8);
    for (;;) {
        i64 t = __atomic_fetch_add(J->next, 1, __ATOMIC_RELAXED);
        if (t >= J->nh) break;
        i64 v = J->heads[t], st = t + 1, left = 0, B = J->S;
        for (i64 k = J->gptr[t]; k < J->gptr[t + 1]; k++) { i64 u = J->src[J->gedge[k]]; if (need[u] != st) { need[u] = st; left++; } }
        h.sz = 0; dist[v] = 0; stamp[v] = st; hpush(&h, 0, v);
        while (h.sz && left > 0) {
            i64 d, u; hpop(&h, &d, &u);
            if (d >= B) break;
            if (d > dist[u]) continue;
            if (need[u] == st) { need[u] = -st; left--; }
            for (i64 k = J->optr[u]; k < J->optr[u + 1]; k++) {
                i64 e = J->oadj[k], y = J->tgt[e], nd = d + J->X[e];
                if (nd >= B) continue;
                if (stamp[y] != st || nd < dist[y]) { stamp[y] = st; dist[y] = nd; hpush(&h, nd, y); }
            }
        }
        for (i64 k = J->gptr[t]; k < J->gptr[t + 1]; k++) {
            i64 b = J->gedge[k], u = J->src[b];
            i64 d = (need[u] == -st) ? dist[u] : J->S;
            i64 r = J->S - d;
            J->out[b] = J->X[b] > r ? J->X[b] : r;
        }
    }
    free(dist); free(stamp); free(need); free(h.hk); free(h.hv);
    return 0;
}

void raise_heads(i64 n, const i64* optr, const i64* oadj, const i64* src, const i64* tgt, const i64* X,
                 const i64* heads, const i64* gptr, const i64* gedge, i64 nh, i64 S, i64 nthreads, i64* out) {
    volatile i64 next = 0;
    rjob J = {n, optr, oadj, tgt, src, X, heads, gptr, gedge, nh, S, out, &next};
    pthread_t th[64];
    for (i64 i = 0; i < nthreads; i++) pthread_create(&th[i], 0, rh_worker, &J);
    for (i64 i = 0; i < nthreads; i++) pthread_join(th[i], 0);
}
"""

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha256(C_SRC.encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "m_dual_kernel"
        d.mkdir(exist_ok=True)
        so = d / ("dual_" + tag + ".so")
        if not so.exists():
            c = d / ("dual_" + tag + ".c")
            c.write_text(C_SRC)
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-pthread", "-o", str(so), str(c)], check=True)
        L = ctypes.CDLL(str(so))
        P = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        I = ctypes.c_int64
        L.min_cycle.restype = I
        L.min_cycle.argtypes = [I, P, P, P, P, P, P, I, I, I, P]
        L.pdist.restype = None
        L.pdist.argtypes = [I, P, P, P, P, P, P, P, I, I, P]
        L.cyc_through.restype = None
        L.cyc_through.argtypes = [I, P, P, P, P, P, P, P, I, I, I, P, P, P]
        L.raise_heads.restype = None
        L.raise_heads.argtypes = [I, P, P, P, P, P, P, P, P, I, I, I, P]
        L.cyc_multi.restype = None
        L.cyc_multi.argtypes = [I, P, P, P, P, P, P, P, I, I, I, I, P, P, P]
        _LIB = L
    return _LIB


def c64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


class Graph:
    def __init__(self, src, tgt, n):
        self.n = int(n)
        self.src = c64(src)
        self.tgt = c64(tgt)
        o = np.argsort(self.src, kind="stable")
        self.optr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(self.src, minlength=self.n), out=self.optr[1:])
        self.oadj = c64(o)
        deg = np.bincount(self.src, minlength=self.n) + np.bincount(self.tgt, minlength=self.n)
        self.rank = c64(np.argsort(np.argsort(-deg, kind="stable"), kind="stable"))

    def min_cycle(self, X, threads=6, cutoff=None, sources=None):
        """Exact minimum cycle length under X (int64 >= 0), or cutoff if no cycle is shorter."""
        X = c64(X)
        assert X.shape[0] == self.src.shape[0] and X.min() >= 0
        if cutoff is None:
            cutoff = int(np.iinfo(np.int64).max // 4)
        srcs = c64(np.argsort(self.rank) if sources is None else sources)
        wit = np.full(2, -1, np.int64)
        L = lib().min_cycle(self.n, self.optr, self.oadj, self.tgt, X, self.rank, srcs, srcs.size,
                            int(threads), int(cutoff), wit)
        return int(L), int(wit[0]), int(wit[1])

    def pdist(self, X, a, b, cap, threads=6):
        a, b, cap = c64(a), c64(b), c64(cap)
        out = np.zeros(a.size, np.int64)
        lib().pdist(self.n, self.optr, self.oadj, self.tgt, c64(X), a, b, cap, a.size, int(threads), out)
        return out

    def raise_heads(self, X, F, S, threads=6):
        """Stale raise of the edges F: X_b = max(X_b, S - dist_X(head, tail)), one Dijkstra per head."""
        X = c64(X)
        F = c64(F[np.argsort(self.tgt[F], kind="stable")])
        heads, start = np.unique(self.tgt[F], return_index=True)
        gptr = c64(np.append(start, F.size))
        out = X.copy()
        lib().raise_heads(self.n, self.optr, self.oadj, self.src, self.tgt, X, c64(heads), gptr, F, heads.size,
                          int(S), int(threads), out)
        return out

    def cyc_through(self, X, sources, caps, threads=6, maxlen=200):
        sources, caps = c64(sources), c64(caps)
        k = sources.size
        ln = np.zeros(k, np.int64)
        ed = np.zeros(k * maxlen, np.int64)
        cnt = np.zeros(k, np.int64)
        lib().cyc_through(self.n, self.optr, self.oadj, self.src, self.tgt, c64(X), sources, caps, k,
                          int(threads), int(maxlen), ln, ed, cnt)
        return ln, [ed[i * maxlen:i * maxlen + cnt[i]] for i in range(k)]

    def cyc_multi(self, X, sources, caps, nk=8, threads=6, maxlen=200):
        """Up to nk cycles per source (shortest per closing edge) with length < cap; flat lists."""
        sources, caps = c64(sources), c64(caps)
        k = sources.size * nk
        ln = np.zeros(k, np.int64)
        ed = np.zeros(k * maxlen, np.int64)
        cnt = np.zeros(k, np.int64)
        lib().cyc_multi(self.n, self.optr, self.oadj, self.src, self.tgt, c64(X), sources, caps, sources.size,
                        int(threads), int(maxlen), int(nk), ln, ed, cnt)
        return ln, [ed[i * maxlen:i * maxlen + cnt[i]] for i in range(k)]


def scc_internal(src, tgt, n):
    """Mask of edges inside a nontrivial strongly connected component."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.csr_matrix((np.ones(src.size), (src, tgt)), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="strong")
    return lab[src] == lab[tgt]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"
