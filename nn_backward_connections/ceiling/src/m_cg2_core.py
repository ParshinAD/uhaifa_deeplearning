"""Method CG2 core: C kernels (compiled at run time with the system cc, called via ctypes, as in
m_route_core.py) for column generation on the cycle packing LP
    max sum_C y_C  s.t.  sum_{C ni e} y_C / w_e <= 1,  y >= 0.
Pool = CSR over columns (cptr int64, cedge int32 edge ids). Kernels, all pthread-parallel:
- pdhg: diagonally preconditioned PDHG iterations (same scheme as m_short_pdhg.py).
- bounds: feasible primal y / max(1, congestion) and Lagrangian dual estimate.
- price: Dijkstra from every given source under lengths len_e (limit 1), stopped after k
  cycles; each popped in-neighbour t of s with d(t) + len(t->s) < 1 - margin closes a cycle.
  Cycles are rotated to start at their smallest edge id and hashed (64 bit) for dedup.
- refill / intload: integer greedy refill and integer edge loads for the certificate.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

C_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
typedef int64_t i64; typedef int32_t i32; typedef uint64_t u64;
#define MAXT 64
#define MAXL 200

typedef struct { void (*fn)(void*, i64, i64, int); void* arg; i64 lo, hi; int t; } job_t;
static void* job_run(void* p) { job_t* j = (job_t*)p; j->fn(j->arg, j->lo, j->hi, j->t); return 0; }
static void par_ptr(int nt, i64 n, const i64* ptr, void (*fn)(void*, i64, i64, int), void* arg) {
    pthread_t th[MAXT]; job_t jb[MAXT];
    if (nt > MAXT) nt = MAXT; if (nt < 1) nt = 1;
    i64 tot = ptr[n], b = 0;
    for (int t = 0; t < nt; t++) {
        i64 target = tot * (t + 1) / nt, lo = b;
        i64 a = lo, z = n;
        while (a < z) { i64 mid = (a + z) / 2; if (ptr[mid] < target) a = mid + 1; else z = mid; }
        if (t == nt - 1) a = n;
        jb[t].fn = fn; jb[t].arg = arg; jb[t].lo = lo; jb[t].hi = a; jb[t].t = t; b = a;
    }
    for (int t = 1; t < nt; t++) pthread_create(&th[t], 0, job_run, &jb[t]);
    job_run(&jb[0]);
    for (int t = 1; t < nt; t++) pthread_join(th[t], 0);
}

typedef struct {
    const i64* cptr; const i32* cedge; const i64* eptr; const i32* ecol; const double* invw;
    const double* T; const double* S; const double* ybar;
    double *x, *y, *v, *ysum, *xsum, *cong, *yp; double part[MAXT][4];
} P;

static void colstep(void* a, i64 lo, i64 hi, int t) {
    P* p = (P*)a;
    for (i64 c = lo; c < hi; c++) {
        double s = 0;
        for (i64 k = p->cptr[c]; k < p->cptr[c + 1]; k++) { i32 e = p->cedge[k]; s += p->x[e] * p->invw[e]; }
        double yo = p->y[c], yn = yo + p->T[c] * (1.0 - s);
        if (yn < 0) yn = 0;
        p->v[c] = 2 * yn - yo; p->y[c] = yn; p->ysum[c] += yn;
    }
}
static void edgestep(void* a, i64 lo, i64 hi, int t) {
    P* p = (P*)a;
    for (i64 e = lo; e < hi; e++) {
        if (p->eptr[e] == p->eptr[e + 1]) continue;
        double s = 0;
        for (i64 k = p->eptr[e]; k < p->eptr[e + 1]; k++) s += p->v[p->ecol[k]];
        double xn = p->x[e] + p->S[e] * (s * p->invw[e] - 1.0);
        if (xn < 0) xn = 0;
        p->x[e] = xn; p->xsum[e] += xn;
    }
}
void pdhg(i64 K, i64 m, const i64* cptr, const i32* cedge, const i64* eptr, const i32* ecol,
          const double* invw, const double* T, const double* S, double* x, double* y, double* v,
          double* ysum, double* xsum, i64 iters, i64 nt) {
    P p; memset(&p, 0, sizeof(p));
    p.cptr = cptr; p.cedge = cedge; p.eptr = eptr; p.ecol = ecol; p.invw = invw; p.T = T; p.S = S;
    p.x = x; p.y = y; p.v = v; p.ysum = ysum; p.xsum = xsum;
    for (i64 it = 0; it < iters; it++) {
        par_ptr((int)nt, K, cptr, colstep, &p);
        par_ptr((int)nt, m, eptr, edgestep, &p);
    }
}

static void bnd_edge(void* a, i64 lo, i64 hi, int t) {
    P* p = (P*)a; double sx = 0;
    for (i64 e = lo; e < hi; e++) {
        double s = 0;
        for (i64 k = p->eptr[e]; k < p->eptr[e + 1]; k++) s += p->y[p->ecol[k]];
        s *= p->invw[e];
        p->cong[e] = s > 1.0 ? s : 1.0;
        if (p->eptr[e + 1] > p->eptr[e]) sx += p->x[e];
    }
    p->part[t][0] = sx;
}
static void bnd_col(void* a, i64 lo, i64 hi, int t) {
    P* p = (P*)a; double lb = 0, ub = 0;
    for (i64 c = lo; c < hi; c++) {
        double s = 0, mx = 1.0;
        for (i64 k = p->cptr[c]; k < p->cptr[c + 1]; k++) {
            i32 e = p->cedge[k]; s += p->x[e] * p->invw[e];
            if (p->cong[e] > mx) mx = p->cong[e];
        }
        double yy = p->y[c] / mx; p->yp[c] = yy; lb += yy;
        if (s < 1.0) ub += p->ybar[c] * (1.0 - s);
    }
    p->part[t][1] = lb; p->part[t][2] = ub;
}
void bounds(i64 K, i64 m, const i64* cptr, const i32* cedge, const i64* eptr, const i32* ecol,
            const double* invw, const double* ybar, const double* x, const double* y, double* cong,
            double* yp, double* out, i64 nt) {
    P p; memset(&p, 0, sizeof(p));
    p.cptr = cptr; p.cedge = cedge; p.eptr = eptr; p.ecol = ecol; p.invw = invw; p.ybar = ybar;
    p.x = (double*)x; p.y = (double*)y; p.cong = cong; p.yp = yp;
    par_ptr((int)nt, m, eptr, bnd_edge, &p);
    par_ptr((int)nt, K, cptr, bnd_col, &p);
    out[0] = 0; out[1] = 0; out[2] = 0;
    for (int t = 0; t < MAXT; t++) { out[0] += p.part[t][1]; out[1] += p.part[t][0] + p.part[t][2]; }
}

static void clen(void* a, i64 lo, i64 hi, int t) {
    P* p = (P*)a;
    for (i64 c = lo; c < hi; c++) {
        double s = 0;
        for (i64 k = p->cptr[c]; k < p->cptr[c + 1]; k++) s += p->x[p->cedge[k]];
        p->yp[c] = s;
    }
}
void collen(i64 K, const i64* cptr, const i32* cedge, const double* len, double* out, i64 nt) {
    P p; memset(&p, 0, sizeof(p));
    p.cptr = cptr; p.cedge = cedge; p.x = (double*)len; p.yp = out;
    par_ptr((int)nt, K, cptr, clen, &p);
}

/* ---------------- pricing ---------------- */
typedef struct {
    i64 n; const i64* optr; const i32* oadj; const i32* oeid; const i64* iptr; const i32* iadj; const i32* ieid;
    const i32* esrc; const double* len; const double* leno; const i64* srcs; i64 nsrc; i64 k; double lim; i64 poplim;
    volatile i64 next; volatile i64 nrec; volatile i64 nedge; i64 max_rec, max_edge;
    i64* rec_ptr; i32* rec_size; double* rec_len; u64* rec_hash; i32* edges; volatile i64 overflow;
    i64 st[MAXT][4];
} Q;

typedef struct { double d; i32 v; } he;
typedef struct { i64 st; double d; } nd_t;
static inline u64 mix(u64 h) { h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33; return h; }

static void* price_worker(void* a) {
    Q* q = ((Q**)a)[0]; int t = (int)(i64)((void**)a)[1];
    i64 n = q->n;
    nd_t* nd_ = (nd_t*)calloc(n, sizeof(nd_t));
    i64* cm = (i64*)calloc(n, sizeof(i64));
    i32* cle = (i32*)malloc(n * sizeof(i32));
    i32* pred = (i32*)malloc(n * sizeof(i32));
    he* heap = (he*)malloc(n * sizeof(he));
    i32* hpos = (i32*)malloc(n * sizeof(i32));
    i32 path[MAXL + 2], rot[MAXL + 2];
    i64 stamp = 0, pops = 0, relax = 0, found_tot = 0, toolong = 0;
    while (1) {
        i64 si = __sync_fetch_and_add(&q->next, 1);
        if (si >= q->nsrc || q->overflow) break;
        i64 s = q->srcs[si];
        stamp++;
        double cmin = q->lim; i64 ncand = 0;
        for (i64 k = q->iptr[s]; k < q->iptr[s + 1]; k++) {
            i32 u = q->iadj[k], e = q->ieid[k];
            if (u == s || q->len[e] >= q->lim) continue;
            if (cm[u] != stamp) { cm[u] = stamp; cle[u] = e; ncand++; }
            else if (q->len[e] < q->len[cle[u]]) cle[u] = e;
            if (q->len[e] < cmin) cmin = q->len[e];
        }
        if (ncand == 0) continue;
        double lims = q->lim - cmin;
        i64 hn = 0, found = 0, spops = 0;
        nd_[s].st = stamp; nd_[s].d = 0; pred[s] = -1; heap[0].d = 0; heap[0].v = (i32)s; hpos[s] = 0; hn = 1;
        while (hn > 0) {
            i32 u = heap[0].v; double du = heap[0].d; hpos[u] = -1; hn--;
            if (hn > 0) {
                he x = heap[hn]; i64 i = 0;
                while (1) {
                    i64 l = 2 * i + 1; if (l >= hn) break;
                    i64 r = l + 1, c = (r < hn && heap[r].d < heap[l].d) ? r : l;
                    if (heap[c].d >= x.d) break;
                    heap[i] = heap[c]; hpos[heap[i].v] = (i32)i; i = c;
                }
                heap[i] = x; hpos[x.v] = (i32)i;
            }
            pops++;
            if (du >= lims || ++spops > q->poplim) break;
            if (cm[u] == stamp) ncand--;
            if (cm[u] == stamp && du + q->len[cle[u]] < q->lim) {
                i64 L = 0; i32 z = u; int ok = 1;
                while (z != s) { if (L >= MAXL) { ok = 0; break; } path[L++] = pred[z]; z = q->esrc[pred[z]]; }
                if (ok) {
                    i32 cyc[MAXL + 2]; i64 cl = 0;
                    for (i64 i = L - 1; i >= 0; i--) cyc[cl++] = path[i];
                    cyc[cl++] = cle[u];
                    i64 mi = 0; for (i64 i = 1; i < cl; i++) if (cyc[i] < cyc[mi]) mi = i;
                    u64 h = 0x9e3779b97f4a7c15ULL; double tl = 0;
                    for (i64 i = 0; i < cl; i++) { rot[i] = cyc[(mi + i) % cl]; h = mix(h ^ (u64)(uint32_t)rot[i]) + (u64)i; tl += q->len[rot[i]]; }
                    i64 r = __sync_fetch_and_add(&q->nrec, 1);
                    i64 p0 = __sync_fetch_and_add(&q->nedge, cl);
                    if (r >= q->max_rec || p0 + cl > q->max_edge) { q->overflow = 1; break; }
                    memcpy(q->edges + p0, rot, cl * sizeof(i32));
                    q->rec_ptr[r] = p0; q->rec_size[r] = (i32)cl; q->rec_len[r] = tl; q->rec_hash[r] = h;
                    found++;
                    if (found >= q->k) break;
                } else toolong++;
            }
            if (ncand == 0) break;
            for (i64 k = q->optr[u]; k < q->optr[u + 1]; k++) {
                double nd = du + q->leno[k];
                relax++;
                if (nd >= lims) continue;
                i32 v = q->oadj[k];
                i64 i;
                if (nd_[v].st != stamp) { nd_[v].st = stamp; i = hn++; }
                else { if (hpos[v] < 0 || nd >= nd_[v].d) continue; i = hpos[v]; }
                nd_[v].d = nd; pred[v] = q->oeid[k];
                while (i > 0) { i64 pa = (i - 1) / 2; if (heap[pa].d <= nd) break; heap[i] = heap[pa]; hpos[heap[i].v] = (i32)i; i = pa; }
                heap[i].d = nd; heap[i].v = v; hpos[v] = (i32)i;
            }
        }
        found_tot += found;
    }
    q->st[t][0] = pops; q->st[t][1] = relax; q->st[t][2] = found_tot; q->st[t][3] = toolong;
    free(nd_); free(cm); free(cle); free(pred); free(heap); free(hpos);
    return 0;
}

i64 price(i64 n, const i64* optr, const i32* oadj, const i32* oeid, const i64* iptr, const i32* iadj,
          const i32* ieid, const i32* esrc, const double* len, const double* leno, const i64* srcs, i64 nsrc, i64 k, double lim, i64 poplim,
          i64 max_rec, i64 max_edge, i64* rec_ptr, i32* rec_size, double* rec_len, u64* rec_hash, i32* edges,
          i64* stats, i64 nt) {
    Q* q = (Q*)calloc(1, sizeof(Q));
    q->n = n; q->optr = optr; q->oadj = oadj; q->oeid = oeid; q->iptr = iptr; q->iadj = iadj; q->ieid = ieid;
    q->esrc = esrc; q->len = len; q->leno = leno; q->srcs = srcs; q->nsrc = nsrc; q->k = k; q->lim = lim; q->poplim = poplim > 0 ? poplim : n + 1;
    q->max_rec = max_rec; q->max_edge = max_edge; q->rec_ptr = rec_ptr; q->rec_size = rec_size;
    q->rec_len = rec_len; q->rec_hash = rec_hash; q->edges = edges;
    if (nt > MAXT) nt = MAXT;
    pthread_t th[MAXT]; void* args[MAXT][2];
    for (i64 t = 0; t < nt; t++) { args[t][0] = q; args[t][1] = (void*)t; pthread_create(&th[t], 0, price_worker, args[t]); }
    for (i64 t = 0; t < nt; t++) pthread_join(th[t], 0);
    for (int j = 0; j < 4; j++) { stats[j] = 0; for (int t = 0; t < MAXT; t++) stats[j] += q->st[t][j]; }
    stats[4] = q->overflow; stats[5] = q->next < nsrc ? q->next : nsrc;
    i64 r = q->nrec < q->max_rec ? q->nrec : q->max_rec;
    free(q);
    return r;
}


/* ---------------- short-cycle pricing (cycles of 2..5 edges, meet in the middle) ----------------
   For source s, only nodes with index > s are used, so each cycle is searched from its smallest node.
   Forward best paths s->v with exactly 1 and 2 edges, backward best v->s with 1 and 2 edges; joins
   give cycles of 2, 3, 4 edges, and with H >= 5 one middle edge between the two 2-edge halves.
   Score = l(C) + floor * |C|; the k best cycles with l(C) < lim are kept per source. */
typedef struct { double sc; i32 kind, node, eid; } cand_t;
typedef struct { i64 st; double d; i32 e; } lab_t;

typedef struct {
    i64 n; const i64* optr; const i32* oadj; const i32* oeid; const double* leno;
    const i64* iptr; const i32* iadj; const i32* ieid; const double* leni; const i32* esrc; const i32* etgt;
    const double* len; const i64* srcs; i64 nsrc; i64 k; i64 H; double lim, fl;
    volatile i64 next, nrec, nedge, overflow; i64 max_rec, max_edge;
    i64* rec_ptr; i32* rec_size; double* rec_len; u64* rec_hash; i32* edges;
    i64 st[MAXT][4];
} S5;

static void heap_push(cand_t* hp, i64* hn, i64 k, cand_t c) {
    /* max-heap on sc of size <= k: keep the k smallest */
    if (*hn < k) {
        i64 i = (*hn)++;
        while (i > 0) { i64 pa = (i - 1) / 2; if (hp[pa].sc >= c.sc) break; hp[i] = hp[pa]; i = pa; }
        hp[i] = c;
    } else if (c.sc < hp[0].sc) {
        i64 i = 0, n = *hn;
        while (1) {
            i64 l = 2 * i + 1; if (l >= n) break;
            i64 r = l + 1, m = (r < n && hp[r].sc > hp[l].sc) ? r : l;
            if (hp[m].sc <= c.sc) break;
            hp[i] = hp[m]; i = m;
        }
        hp[i] = c;
    }
}

static void* short_worker(void* a) {
    S5* q = ((S5**)a)[0]; int t = (int)(i64)((void**)a)[1];
    i64 n = q->n;
    lab_t *f1 = calloc(n, sizeof(lab_t)), *f2 = calloc(n, sizeof(lab_t)), *b1 = calloc(n, sizeof(lab_t)), *b2 = calloc(n, sizeof(lab_t));
    lab_t *f3 = calloc(n, sizeof(lab_t)), *b3 = calloc(n, sizeof(lab_t));
    i32 *l1 = malloc(n * sizeof(i32)), *l2 = malloc(n * sizeof(i32)), *lb1 = malloc(n * sizeof(i32));
    i32 *l3 = malloc(n * sizeof(i32)), *lb2 = malloc(n * sizeof(i32));
    cand_t* hp = malloc((q->k + 1) * sizeof(cand_t));
    i64 stamp = 0, work = 0, found = 0, cands = 0;
    double lim = q->lim;
    while (1) {
        i64 si = __sync_fetch_and_add(&q->next, 1);
        if (si >= q->nsrc || q->overflow) break;
        i32 s = (i32)q->srcs[si];
        stamp++;
        i64 n1 = 0, n2 = 0, nb1 = 0, hn = 0, n3 = 0, nb2 = 0;
        for (i64 k = q->optr[s]; k < q->optr[s + 1]; k++) {
            i32 v = q->oadj[k]; double d = q->leno[k];
            if (v <= s || d >= lim) continue;
            if (f1[v].st != stamp) { f1[v].st = stamp; f1[v].d = d; f1[v].e = q->oeid[k]; l1[n1++] = v; }
            else if (d < f1[v].d) { f1[v].d = d; f1[v].e = q->oeid[k]; }
        }
        for (i64 k = q->iptr[s]; k < q->iptr[s + 1]; k++) {
            i32 u = q->iadj[k]; double d = q->leni[k];
            if (u <= s || d >= lim) continue;
            if (b1[u].st != stamp) { b1[u].st = stamp; b1[u].d = d; b1[u].e = q->ieid[k]; lb1[nb1++] = u; }
            else if (d < b1[u].d) { b1[u].d = d; b1[u].e = q->ieid[k]; }
        }
        if (n1 == 0 || nb1 == 0) continue;
        for (i64 i = 0; i < n1; i++) {
            i32 a_ = l1[i]; double da = f1[a_].d;
            for (i64 k = q->optr[a_]; k < q->optr[a_ + 1]; k++) {
                i32 v = q->oadj[k]; double d = da + q->leno[k]; work++;
                if (v <= s || d >= lim) continue;
                if (f2[v].st != stamp) { f2[v].st = stamp; f2[v].d = d; f2[v].e = q->oeid[k]; l2[n2++] = v; }
                else if (d < f2[v].d) { f2[v].d = d; f2[v].e = q->oeid[k]; }
            }
        }
        for (i64 i = 0; i < nb1; i++) {
            i32 u = lb1[i]; double du = b1[u].d;
            for (i64 k = q->iptr[u]; k < q->iptr[u + 1]; k++) {
                i32 z = q->iadj[k]; double d = du + q->leni[k]; work++;
                if (z <= s || z == u || d >= lim) continue;
                if (b2[z].st != stamp) { b2[z].st = stamp; b2[z].d = d; b2[z].e = q->ieid[k]; lb2[nb2++] = z; }
                else if (d < b2[z].d) { b2[z].d = d; b2[z].e = q->ieid[k]; }
            }
        }
        /* joins */
        for (i64 i = 0; i < n1; i++) {
            i32 v = l1[i];
            if (b1[v].st == stamp) { double l = f1[v].d + b1[v].d; if (l < lim) { cand_t c = {l + 2 * q->fl, 2, v, -1}; heap_push(hp, &hn, q->k, c); cands++; } }
            if (b2[v].st == stamp) {
                i32 u = q->etgt[b2[v].e];
                if (u != v) { double l = f1[v].d + b2[v].d; if (l < lim) { cand_t c = {l + 3 * q->fl, 3, v, -1}; heap_push(hp, &hn, q->k, c); cands++; } }
            }
        }
        for (i64 i = 0; i < n2; i++) {
            i32 c_ = l2[i];
            if (b2[c_].st != stamp) continue;
            i32 a_ = q->esrc[f2[c_].e], d_ = q->etgt[b2[c_].e];
            if (a_ == d_ || a_ == c_ || d_ == c_) continue;
            double l = f2[c_].d + b2[c_].d;
            if (l < lim) { cand_t c = {l + 4 * q->fl, 4, c_, -1}; heap_push(hp, &hn, q->k, c); cands++; }
        }
        if (q->H >= 6) {
            for (i64 i = 0; i < n2; i++) {
                i32 b_ = l2[i]; double db = f2[b_].d;
                for (i64 k = q->optr[b_]; k < q->optr[b_ + 1]; k++) {
                    i32 v = q->oadj[k]; double d = db + q->leno[k]; work++;
                    if (v <= s || v == b_ || d >= lim) continue;
                    if (f3[v].st != stamp) { f3[v].st = stamp; f3[v].d = d; f3[v].e = q->oeid[k]; l3[n3++] = v; }
                    else if (d < f3[v].d) { f3[v].d = d; f3[v].e = q->oeid[k]; }
                }
            }
            for (i64 i = 0; i < nb2; i++) {
                i32 u = lb2[i]; double du = b2[u].d;
                for (i64 k = q->iptr[u]; k < q->iptr[u + 1]; k++) {
                    i32 z = q->iadj[k]; double d = du + q->leni[k]; work++;
                    if (z <= s || z == u || d >= lim) continue;
                    if (b3[z].st != stamp) { b3[z].st = stamp; b3[z].d = d; b3[z].e = q->ieid[k]; }
                    else if (d < b3[z].d) { b3[z].d = d; b3[z].e = q->ieid[k]; }
                }
            }
            for (i64 i = 0; i < n3; i++) {
                i32 c_ = l3[i];
                if (b3[c_].st != stamp) continue;
                double l = f3[c_].d + b3[c_].d;
                if (l >= lim) continue;
                i32 b_ = q->esrc[f3[c_].e], a_ = q->esrc[f2[b_].e], d_ = q->etgt[b3[c_].e], e_ = q->etgt[b2[d_].e];
                i32 nd6[5] = {a_, b_, c_, d_, e_}; int okn = 1;
                for (int x1 = 0; x1 < 5 && okn; x1++) for (int x2 = x1 + 1; x2 < 5; x2++) if (nd6[x1] == nd6[x2]) { okn = 0; break; }
                if (!okn) continue;
                cand_t c = {l + 6 * q->fl, 6, c_, -1}; heap_push(hp, &hn, q->k, c); cands++;
            }
        }
        if (q->H >= 5) {
            for (i64 i = 0; i < n2; i++) {
                i32 b_ = l2[i]; double db = f2[b_].d; i32 a_ = q->esrc[f2[b_].e];
                if (db >= lim) continue;
                for (i64 k = q->optr[b_]; k < q->optr[b_ + 1]; k++) {
                    i32 c_ = q->oadj[k]; work++;
                    if (c_ <= s || b2[c_].st != stamp) continue;
                    double l = db + q->leno[k] + b2[c_].d;
                    if (l >= lim) continue;
                    i32 d_ = q->etgt[b2[c_].e];
                    if (c_ == a_ || d_ == a_ || d_ == b_ || c_ == b_) continue;
                    cand_t c = {l + 5 * q->fl, 5, b_, q->oeid[k]}; heap_push(hp, &hn, q->k, c); cands++;
                }
            }
        }
        for (i64 j = 0; j < hn; j++) {
            cand_t c = hp[j]; i32 cyc[8]; i64 cl = 0; i32 v = c.node;
            if (c.kind == 2) { cyc[cl++] = f1[v].e; cyc[cl++] = b1[v].e; }
            else if (c.kind == 3) { i32 u = q->etgt[b2[v].e]; cyc[cl++] = f1[v].e; cyc[cl++] = b2[v].e; cyc[cl++] = b1[u].e; }
            else if (c.kind == 4) { i32 a_ = q->esrc[f2[v].e], d_ = q->etgt[b2[v].e];
                cyc[cl++] = f1[a_].e; cyc[cl++] = f2[v].e; cyc[cl++] = b2[v].e; cyc[cl++] = b1[d_].e; }
            else if (c.kind == 6) { i32 b_ = q->esrc[f3[v].e], a_ = q->esrc[f2[b_].e], d_ = q->etgt[b3[v].e], e_ = q->etgt[b2[d_].e];
                cyc[cl++] = f1[a_].e; cyc[cl++] = f2[b_].e; cyc[cl++] = f3[v].e; cyc[cl++] = b3[v].e; cyc[cl++] = b2[d_].e; cyc[cl++] = b1[e_].e; }
            else { i32 a_ = q->esrc[f2[v].e], c_ = q->etgt[c.eid], d_ = q->etgt[b2[c_].e];
                cyc[cl++] = f1[a_].e; cyc[cl++] = f2[v].e; cyc[cl++] = c.eid; cyc[cl++] = b2[c_].e; cyc[cl++] = b1[d_].e; }
            i32 rot[8]; i64 mi = 0; for (i64 i = 1; i < cl; i++) if (cyc[i] < cyc[mi]) mi = i;
            u64 h = 0x9e3779b97f4a7c15ULL; double tl = 0;
            for (i64 i = 0; i < cl; i++) { rot[i] = cyc[(mi + i) % cl]; h = mix(h ^ (u64)(uint32_t)rot[i]) + (u64)i; tl += q->len[rot[i]]; }
            i64 r = __sync_fetch_and_add(&q->nrec, 1);
            i64 p0 = __sync_fetch_and_add(&q->nedge, cl);
            if (r >= q->max_rec || p0 + cl > q->max_edge) { q->overflow = 1; break; }
            memcpy(q->edges + p0, rot, cl * sizeof(i32));
            q->rec_ptr[r] = p0; q->rec_size[r] = (i32)cl; q->rec_len[r] = tl; q->rec_hash[r] = h;
            found++;
        }
    }
    q->st[t][0] = work; q->st[t][1] = cands; q->st[t][2] = found;
    free(f1); free(f2); free(b1); free(b2); free(f3); free(b3); free(l1); free(l2); free(lb1); free(l3); free(lb2); free(hp);
    return 0;
}

i64 price_short(i64 n, const i64* optr, const i32* oadj, const i32* oeid, const double* leno,
                const i64* iptr, const i32* iadj, const i32* ieid, const double* leni, const i32* esrc, const i32* etgt,
                const double* len, const i64* srcs, i64 nsrc, i64 k, i64 H, double lim, double fl,
                i64 max_rec, i64 max_edge, i64* rec_ptr, i32* rec_size, double* rec_len, u64* rec_hash, i32* edges,
                i64* stats, i64 nt) {
    S5* q = (S5*)calloc(1, sizeof(S5));
    q->n = n; q->optr = optr; q->oadj = oadj; q->oeid = oeid; q->leno = leno; q->iptr = iptr; q->iadj = iadj;
    q->ieid = ieid; q->leni = leni; q->esrc = esrc; q->etgt = etgt; q->len = len; q->srcs = srcs; q->nsrc = nsrc;
    q->k = k; q->H = H; q->lim = lim; q->fl = fl; q->max_rec = max_rec; q->max_edge = max_edge;
    q->rec_ptr = rec_ptr; q->rec_size = rec_size; q->rec_len = rec_len; q->rec_hash = rec_hash; q->edges = edges;
    if (nt > MAXT) nt = MAXT;
    pthread_t th[MAXT]; void* args[MAXT][2];
    for (i64 t = 0; t < nt; t++) { args[t][0] = q; args[t][1] = (void*)t; pthread_create(&th[t], 0, short_worker, args[t]); }
    for (i64 t = 0; t < nt; t++) pthread_join(th[t], 0);
    for (int j = 0; j < 3; j++) { stats[j] = 0; for (int t = 0; t < MAXT; t++) stats[j] += q->st[t][j]; }
    stats[3] = q->overflow;
    i64 r = q->nrec < q->max_rec ? q->nrec : q->max_rec;
    free(q);
    return r;
}

/* hash of given CSR cycles (already rotated) */
void hashcols(i64 K, const i64* cptr, const i32* cedge, u64* out) {
    for (i64 c = 0; c < K; c++) {
        u64 h = 0x9e3779b97f4a7c15ULL;
        for (i64 k = cptr[c], i = 0; k < cptr[c + 1]; k++, i++) h = mix(h ^ (u64)(uint32_t)cedge[k]) + (u64)i;
        out[c] = h;
    }
}

void intload(i64 K, const i64* cptr, const i32* cedge, const i64* y, i64* load) {
    for (i64 c = 0; c < K; c++) if (y[c]) for (i64 k = cptr[c]; k < cptr[c + 1]; k++) load[cedge[k]] += y[c];
}

i64 refill(i64 K, const i64* order, const i64* cptr, const i32* cedge, i64* res, i64* y) {
    i64 gain = 0;
    for (i64 j = 0; j < K; j++) {
        i64 c = order[j], f = -1;
        for (i64 k = cptr[c]; k < cptr[c + 1]; k++) { i64 r = res[cedge[k]]; if (f < 0 || r < f) f = r; if (f == 0) break; }
        if (f > 0) { for (i64 k = cptr[c]; k < cptr[c + 1]; k++) res[cedge[k]] -= f; y[c] += f; gain += f; }
    }
    return gain;
}
"""

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha256(C_SRC.encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "m_cg2_kernel"
        d.mkdir(exist_ok=True)
        so = d / ("cg2_" + tag + ".so")
        if not so.exists():
            c = d / ("cg2_" + tag + ".c")
            c.write_text(C_SRC)
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(so), str(c)], check=True)
        L = ctypes.CDLL(str(so))

        def P(dt):
            return np.ctypeslib.ndpointer(dtype=dt, flags="C_CONTIGUOUS")
        I, F = ctypes.c_int64, ctypes.c_double
        i64, i32, f64, u64 = P(np.int64), P(np.int32), P(np.float64), P(np.uint64)
        L.pdhg.restype = None
        L.pdhg.argtypes = [I, I, i64, i32, i64, i32, f64, f64, f64, f64, f64, f64, f64, f64, I, I]
        L.bounds.restype = None
        L.bounds.argtypes = [I, I, i64, i32, i64, i32, f64, f64, f64, f64, f64, f64, f64, I]
        L.collen.restype = None
        L.collen.argtypes = [I, i64, i32, f64, f64, I]
        L.price.restype = I
        L.price.argtypes = [I, i64, i32, i32, i64, i32, i32, i32, f64, f64, i64, I, I, F, I, I, I,
                            i64, i32, f64, u64, i32, i64, I]
        L.price_short.restype = I
        L.price_short.argtypes = [I, i64, i32, i32, f64, i64, i32, i32, f64, i32, i32, f64, i64, I, I, I, F, F,
                                  I, I, i64, i32, f64, u64, i32, i64, I]
        L.hashcols.restype = None
        L.hashcols.argtypes = [I, i64, i32, u64]
        L.intload.restype = None
        L.intload.argtypes = [I, i64, i32, i64, i64]
        L.refill.restype = I
        L.refill.argtypes = [I, i64, i64, i32, i64, i64]
        _LIB = L
    return _LIB


def rotate_min(cedge, cptr):
    """Rotate each CSR cycle so it starts at its smallest edge id (canonical form for hashing)."""
    out = np.empty_like(cedge)
    lens = np.diff(cptr)
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        blk = cedge[cptr[idx][:, None] + np.arange(L)]
        sh = blk.argmin(axis=1)
        blk = blk[np.arange(idx.size)[:, None], (sh[:, None] + np.arange(L)) % L]
        out[cptr[idx][:, None] + np.arange(L)] = blk
    return out


def hashes(cptr, cedge):
    h = np.zeros(cptr.size - 1, np.uint64)
    lib().hashcols(cptr.size - 1, cptr, cedge, h)
    return h


def transpose(cptr, cedge, m):
    """Edge -> column CSR (eptr int64, ecol int32)."""
    K = cptr.size - 1
    col = np.repeat(np.arange(K, dtype=np.int32), np.diff(cptr))
    o = np.argsort(cedge, kind="stable")
    eptr = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(cedge, minlength=m), out=eptr[1:])
    return eptr, np.ascontiguousarray(col[o])
