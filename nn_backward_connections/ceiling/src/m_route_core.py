"""Method R core: integer path routing kernel (C via ctypes) for order-guided cycle packing.

A commodity is an edge b = (u -> v) with residual capacity cap[b]; the kernel looks for a
path v ~> u in the routing graph (edges with cap > 0), closes it with b into a simple cycle
and pushes the bottleneck. Search = bidirectional level BFS with a hop limit H, so a
found cycle has at most H + 1 edges. With use_rank the search is pruned to the rank window
[rank v, rank u] (valid when the routing graph is the forward DAG of that rank).
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

C_SRC = r"""
#include <stdint.h>
typedef int64_t i64;
#define MAXH 64

i64 route(i64 n, const i64* optr, const i64* oadj, const i64* iptr, const i64* iadj,
          const i64* esrc, const i64* etgt, i64* cap,
          const i64* comm, i64 ncomm, i64 pos, i64 H, const i64* rank, i64 use_rank,
          i64 work_limit, i64* fmark, i64* bmark, i64* fpar, i64* bpar, i64* fq, i64* bq,
          i64* stamp_io, i64* rec_comm, i64* rec_amt, i64* rec_ptr, i64* rec_edges,
          i64 max_rec, i64 max_edges, i64* stats)
{
    i64 path[2 * MAXH + 4];
    if (H > MAXH) H = MAXH;
    while (pos < ncomm) {
        i64 b = comm[pos];
        if (cap[b] <= 0) { pos++; continue; }
        i64 u = esrc[b], v = etgt[b];
        i64 stamp = ++(*stamp_io);
        i64 ru = use_rank ? rank[u] : 0, rv = use_rank ? rank[v] : 0;
        fmark[v] = stamp; fpar[v] = -1; fq[0] = v;
        bmark[u] = stamp; bpar[u] = -1; bq[0] = u;
        i64 fs = 0, ft = 1, bs = 0, bt = 1, lf = 0, lb = 0;
        i64 fsum = optr[v + 1] - optr[v], bsum = iptr[u + 1] - iptr[u];
        i64 work = 0, fx = -1, me = -1, by = -1, fail = 0;
        stats[2]++;
        while (lf + lb < H && me < 0 && !fail) {
            if (fsum <= bsum) {
                i64 nt = ft, nsum = 0;
                for (i64 i = fs; i < ft && me < 0; i++) {
                    i64 x = fq[i];
                    for (i64 k = optr[x]; k < optr[x + 1]; k++) {
                        i64 e = oadj[k];
                        work++;
                        if (cap[e] <= 0) continue;
                        i64 y = etgt[e];
                        if (bmark[y] == stamp) { fx = x; me = e; by = y; break; }
                        if (fmark[y] == stamp) continue;
                        if (use_rank && rank[y] >= ru) continue;
                        fmark[y] = stamp; fpar[y] = e; fq[nt++] = y;
                        nsum += optr[y + 1] - optr[y];
                    }
                    if (work > work_limit) { fail = 2; break; }
                }
                if (me >= 0 || fail) break;
                fs = ft; ft = nt; fsum = nsum; lf++;
                if (fs == ft) fail = 1;
            } else {
                i64 nt = bt, nsum = 0;
                for (i64 i = bs; i < bt && me < 0; i++) {
                    i64 x = bq[i];
                    for (i64 k = iptr[x]; k < iptr[x + 1]; k++) {
                        i64 e = iadj[k];
                        work++;
                        if (cap[e] <= 0) continue;
                        i64 z = esrc[e];
                        if (fmark[z] == stamp) { fx = z; me = e; by = x; break; }
                        if (bmark[z] == stamp) continue;
                        if (use_rank && rank[z] <= rv) continue;
                        bmark[z] = stamp; bpar[z] = e; bq[nt++] = z;
                        nsum += iptr[z + 1] - iptr[z];
                    }
                    if (work > work_limit) { fail = 2; break; }
                }
                if (me >= 0 || fail) break;
                bs = bt; bt = nt; bsum = nsum; lb++;
                if (bs == bt) fail = 1;
            }
        }
        stats[3] += work;
        if (me < 0) {
            if (fail == 2) stats[4]++;
            pos++;
            continue;
        }
        i64 len = 0, x = fx;
        while (fpar[x] >= 0) { path[len++] = fpar[x]; x = esrc[fpar[x]]; }
        for (i64 a = 0, c = len - 1; a < c; a++, c--) { i64 t = path[a]; path[a] = path[c]; path[c] = t; }
        path[len++] = me;
        x = by;
        while (bpar[x] >= 0) { path[len++] = bpar[x]; x = etgt[bpar[x]]; }
        if (stats[0] >= max_rec || stats[1] + len + 1 > max_edges) return pos;
        i64 amt = cap[b];
        for (i64 k = 0; k < len; k++) if (cap[path[k]] < amt) amt = cap[path[k]];
        i64 r = stats[0], q = stats[1];
        rec_comm[r] = b; rec_amt[r] = amt;
        rec_edges[q++] = b; cap[b] -= amt;
        for (i64 k = 0; k < len; k++) { rec_edges[q++] = path[k]; cap[path[k]] -= amt; }
        rec_ptr[r + 1] = q;
        stats[0] = r + 1; stats[1] = q; stats[5] += amt;
    }
    return pos;
}
"""

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha256(C_SRC.encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "m_route_kernel"
        d.mkdir(exist_ok=True)
        so = d / ("route_" + tag + ".so")
        if not so.exists():
            c = d / ("route_" + tag + ".c")
            c.write_text(C_SRC)
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(so), str(c)], check=True)
        L = ctypes.CDLL(str(so))
        P = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        I = ctypes.c_int64
        L.route.restype = I
        L.route.argtypes = [I, P, P, P, P, P, P, P, P, I, I, I, P, I, I,
                            P, P, P, P, P, P, P, P, P, P, P, I, I, P]
        _LIB = L
    return _LIB


class Router:
    """Routing graph = edges where mask is True; cap is shared over all m edges."""

    def __init__(self, src, tgt, n, mask, cap, rank=None):
        self.n = int(n)
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.tgt = np.ascontiguousarray(tgt, dtype=np.int64)
        self.cap = cap
        eid = np.flatnonzero(mask).astype(np.int64)
        o = eid[np.argsort(self.src[eid], kind="stable")]
        self.optr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(self.src[o], minlength=n), out=self.optr[1:])
        self.oadj = np.ascontiguousarray(o)
        i = eid[np.argsort(self.tgt[eid], kind="stable")]
        self.iptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(self.tgt[i], minlength=n), out=self.iptr[1:])
        self.iadj = np.ascontiguousarray(i)
        self.use_rank = rank is not None
        self.rank = np.ascontiguousarray(rank if rank is not None else np.zeros(n), dtype=np.int64)
        self.fmark = np.zeros(n, np.int64)
        self.bmark = np.zeros(n, np.int64)
        self.fpar = np.zeros(n, np.int64)
        self.bpar = np.zeros(n, np.int64)
        self.fq = np.zeros(n + 1, np.int64)
        self.bq = np.zeros(n + 1, np.int64)
        self.stamp = np.zeros(1, np.int64)

    def run(self, comm, H, work_limit, max_rec=2_000_000, max_edges=20_000_000):
        """Route commodities in the given order with hop limit H. Returns (records, stats)."""
        L = lib()
        comm = np.ascontiguousarray(comm, dtype=np.int64)
        rc = np.zeros(max_rec, np.int64)
        ra = np.zeros(max_rec, np.int64)
        rp = np.zeros(max_rec + 1, np.int64)
        re = np.zeros(max_edges, np.int64)
        tot = np.zeros(6, np.int64)
        out = []
        pos = 0
        while True:
            st = np.zeros(6, np.int64)
            pos = L.route(self.n, self.optr, self.oadj, self.iptr, self.iadj, self.src, self.tgt,
                          self.cap, comm, comm.size, pos, int(H), self.rank, int(self.use_rank),
                          int(work_limit), self.fmark, self.bmark, self.fpar, self.bpar, self.fq,
                          self.bq, self.stamp, rc, ra, rp, re, max_rec, max_edges, st)
            k, q = int(st[0]), int(st[1])
            out.append((rc[:k].copy(), ra[:k].copy(), rp[:k + 1].copy(), re[:q].copy()))
            tot += st
            if pos >= comm.size:
                break
        return merge_records(out), dict(records=int(tot[0]), edge_entries=int(tot[1]),
                                        searches=int(tot[2]), work=int(tot[3]),
                                        work_limit_hits=int(tot[4]), routed=int(tot[5]))


def merge_records(parts):
    parts = [p for p in parts if p[0].size]
    if not parts:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(1, np.int64), z
    rc = np.concatenate([p[0] for p in parts])
    ra = np.concatenate([p[1] for p in parts])
    re = np.concatenate([p[3] for p in parts])
    lens = np.concatenate([np.diff(p[2]) for p in parts])
    rp = np.zeros(lens.size + 1, np.int64)
    np.cumsum(lens, out=rp[1:])
    return rc, ra, rp, re


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def verify(cert_path):
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE / "verify_cert.py"), str(cert_path)],
                       capture_output=True, text=True, cwd=ROOT)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:]
    return dict(status="PASS" if p.returncode == 0 else "FAIL", line=last,
                stdout=p.stdout[-3000:], seconds=round(time.time() - t0, 1))
