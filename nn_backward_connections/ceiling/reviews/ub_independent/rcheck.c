/* Independent check: for every head v, Dijkstra from v (unrestricted), cap = S+1 - min X over in-edges of v;
   for each in-edge e=(u,v): cyc = X_e + dist(v,u); report global min (capped at S+1) and count of cyc < S. */
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
typedef int64_t i64;
typedef struct { i64 n; const i64 *optr,*otgt,*oX,*iptr,*isrc,*iX; i64 S; volatile i64 next; i64 *vmin; i64 *vbad; } J_t;
static void* work(void* a) {
  J_t* J = a; i64 n = J->n;
  i64 *dist = malloc(n*8), *st = calloc(n,8), *done = calloc(n,8);
  i64 cap = 1<<16, sz; i64 *hk = malloc(cap*8), *hv = malloc(cap*8);
  for (;;) {
    i64 v = __atomic_fetch_add(&J->next, 1, __ATOMIC_RELAXED);
    if (v >= n) break;
    i64 lo = J->iptr[v], hi = J->iptr[v+1];
    if (lo == hi) { J->vmin[v] = J->S + 1; continue; }
    i64 mx = J->S + 1;
    for (i64 k = lo; k < hi; k++) if (J->iX[k] < mx) mx = J->iX[k];
    i64 C = J->S + 1 - mx, tag = v + 1;
    sz = 0; dist[v] = 0; st[v] = tag;
    hk[0] = 0; hv[0] = v; sz = 1;
    while (sz) {
      i64 d = hk[0], u = hv[0];
      sz--; { i64 lk = hk[sz], lv = hv[sz], i = 0; for (;;) { i64 c = 2*i+1; if (c >= sz) break; if (c+1 < sz && hk[c+1] < hk[c]) c++; if (hk[c] >= lk) break; hk[i]=hk[c]; hv[i]=hv[c]; i=c; } if (sz) { hk[i]=lk; hv[i]=lv; } }
      if (d >= C) break;
      if (done[u] == tag) continue;
      done[u] = tag;
      for (i64 k = J->optr[u]; k < J->optr[u+1]; k++) {
        i64 y = J->otgt[k], nd = d + J->oX[k];
        if (nd >= C) continue;
        if (st[y] != tag || nd < dist[y]) {
          st[y] = tag; dist[y] = nd;
          if (sz == cap) { cap *= 2; hk = realloc(hk, cap*8); hv = realloc(hv, cap*8); }
          i64 i = sz++; while (i > 0) { i64 p = (i-1)/2; if (hk[p] <= nd) break; hk[i]=hk[p]; hv[i]=hv[p]; i=p; } hk[i]=nd; hv[i]=y;
        }
      }
    }
    i64 best = J->S + 1, bad = 0;
    for (i64 k = lo; k < hi; k++) {
      i64 u = J->isrc[k];
      if (done[u] != tag) continue; /* not settled => dist >= C => cyc >= S+1 */
      i64 c = J->iX[k] + dist[u];
      if (c < best) best = c;
      if (c < J->S) bad++;
    }
    J->vmin[v] = best; J->vbad[v] = bad;
  }
  free(dist); free(st); free(done); free(hk); free(hv); return 0;
}
void rcheck(i64 n, const i64* optr, const i64* otgt, const i64* oX, const i64* iptr, const i64* isrc, const i64* iX,
            i64 S, i64 nth, i64* vmin, i64* vbad) {
  J_t J = {n, optr, otgt, oX, iptr, isrc, iX, S, 0, vmin, vbad};
  pthread_t th[64]; for (i64 i = 0; i < nth; i++) pthread_create(&th[i], 0, work, &J);
  for (i64 i = 0; i < nth; i++) pthread_join(th[i], 0);
}
