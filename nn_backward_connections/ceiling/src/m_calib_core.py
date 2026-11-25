"""Calibration bounds for min feedback weight on small graphs (local node ids 0..k-1).

Bounds: 2-cycle packing, greedy 2+3-cycle packing, full cycle LP (lazy cycle generation with exact
Floyd-Warshall separation), ordering LP with all 3-dicycle inequalities. Exact optimum: lazy-cycle MILP,
ordering MILP, subset DP (<= 20 nodes). None of this is a certificate (float LP values).

Self-test: python m_calib_core.py   (random tiny graphs vs brute force over all permutations)"""
import itertools
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.sparse.csgraph import connected_components

CONFIG = dict(sep_tol=1e-9, max_lp_rounds=500, max_milp_rounds=200, milp_time=120.0, dp_max_nodes=20,
              selftest_seed=7, selftest_graphs=60)


def dense(k, s, t, w):
    A = np.zeros((k, k))
    np.add.at(A, (s, t), w)
    return A


def lb_2cycle(k, s, t, w):
    A = dense(k, s, t, w)
    return float(np.minimum(A, A.T).sum() / 2)


def greedy_23(k, s, t, w):
    """Exact 2-cycle packing, then all directed 3-cycles of the residual, largest initial min-cap first."""
    A = dense(k, s, t, w)
    p2 = np.minimum(A, A.T)
    lb2 = float(p2.sum() / 2)
    C = A - p2
    tri = []
    for a in range(k):
        b, c = np.nonzero((C[a, :, None] > 0) & (C > 0) & (C[:, a][None, :] > 0))
        keep = (b > a) & (c > a)
        for bb, cc in zip(b[keep], c[keep]):
            tri.append((min(C[a, bb], C[bb, cc], C[cc, a]), a, int(bb), int(cc)))
    tri.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    add = 0.0
    for _, a, b, c in tri:
        y = min(C[a, b], C[b, c], C[c, a])
        if y > 0:
            C[a, b] -= y
            C[b, c] -= y
            C[c, a] -= y
            add += y
    return lb2 + add, len(tri)


def floyd(L):
    """All-pairs shortest paths with successor matrix; L = lengths, inf = no edge."""
    k = L.shape[0]
    D = L.copy()
    np.fill_diagonal(D, 0.0)
    nxt = np.where(np.isfinite(L), np.arange(k)[None, :], -1)
    for x in range(k):
        nd = D[:, x, None] + D[None, x, :]
        better = nd < D
        D = np.where(better, nd, D)
        nxt = np.where(better, nxt[:, x, None], nxt)
    return D, nxt


def path(nxt, i, j):
    p = [i]
    while i != j:
        i = int(nxt[i, j])
        p.append(i)
    return p


def violated_cycles(k, s, t, x, eid, keep=None, tol=None):
    """Cycles e + shortest path t(e)->s(e) with total x-length < 1 - tol (keep: restrict to these edges)."""
    L = np.full((k, k), np.inf)
    idx = np.arange(s.size) if keep is None else np.flatnonzero(keep)
    L[s[idx], t[idx]] = x[idx]
    D, nxt = floyd(L)
    out = set()
    lim = 1.0 - (CONFIG["sep_tol"] if tol is None else tol)
    for e in idx:
        u, v = int(s[e]), int(t[e])
        if x[e] + D[v, u] < lim:
            p = path(nxt, v, u)
            cyc = {int(e)} | {int(eid[p[i], p[i + 1]]) for i in range(len(p) - 1)}
            out.add(tuple(sorted(cyc)))
    return out


def incidence(cycles, m):
    rows = np.repeat(np.arange(len(cycles)), [len(c) for c in cycles])
    cols = np.concatenate([np.array(c, np.int64) for c in cycles])
    return sp.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(len(cycles), m))


def eid_matrix(k, s, t):
    E = -np.ones((k, k), np.int64)
    E[s, t] = np.arange(s.size)
    return E


def cycle_lp(k, s, t, w):
    """min w.x s.t. x(C) >= 1 for every directed cycle C, x >= 0; returns value, cycles, rounds, dual sum."""
    m = s.size
    eid = eid_matrix(k, s, t)
    cycles = sorted(violated_cycles(k, s, t, np.zeros(m), eid))
    if not cycles:
        return 0.0, [], 0, 0.0
    for r in range(CONFIG["max_lp_rounds"]):
        M = incidence(cycles, m)
        res = linprog(w, A_ub=-M, b_ub=-np.ones(len(cycles)), bounds=(0, None), method="highs")
        assert res.status == 0, res.message
        new = violated_cycles(k, s, t, res.x, eid) - set(cycles)
        if not new:
            return float(res.fun), cycles, r + 1, float(-res.ineqlin.marginals.sum())
        cycles = cycles + sorted(new)
    raise RuntimeError("cycle LP did not converge")


def scc_blocks(k, s, t):
    A = sp.csr_matrix((np.ones(s.size), (s, t)), shape=(k, k))
    _, lab = connected_components(A, directed=True, connection="strong")
    return [np.flatnonzero(lab == c) for c in np.unique(lab) if (lab == c).sum() > 1], lab


def ordering_model(nodes, Wd):
    """Variables x_ij (i<j local) = 1 if i before j. Returns c, const, triangle matrix."""
    q = nodes.size
    vid = -np.ones((q, q), np.int64)
    iu, ju = np.triu_indices(q, 1)
    vid[iu, ju] = np.arange(iu.size)
    B = Wd[np.ix_(nodes, nodes)]
    c = (B.T - B)[iu, ju]
    const = float(B[iu, ju].sum())
    i, j, l = [a.ravel() for a in np.meshgrid(np.arange(q), np.arange(q), np.arange(q), indexing="ij")]
    sel = (i < j) & (j < l)
    i, j, l = i[sel], j[sel], l[sel]
    nt = i.size
    rows = np.repeat(np.arange(nt), 3)
    cols = np.column_stack([vid[i, j], vid[j, l], vid[i, l]]).ravel()
    T = sp.csr_matrix((np.tile([1.0, 1.0, -1.0], nt), (rows, cols)), shape=(nt, iu.size))
    return c, const, T


def ordering_bound(k, s, t, w, integral, time_limit=None):
    """Sum over nontrivial SCCs of the ordering LP (integral=False) or MILP (True) with all 3-dicycle
    inequalities 0 <= x_ij + x_jl - x_il <= 1, added lazily (all violated ones per round).
    Edge a->b with a<b is backward iff x_ab = 0: cost w(1 - x_ab); with a>b backward iff x_ba = 1."""
    Wd = dense(k, s, t, w)
    total, rounds, nrows = 0.0, 0, 0
    t0 = time.time()
    tol = CONFIG["sep_tol"]
    for nodes in scc_blocks(k, s, t)[0]:
        c, const, T = ordering_model(nodes, Wd)
        up, lo = np.zeros(T.shape[0], bool), np.zeros(T.shape[0], bool)
        while True:
            rounds += 1
            A = sp.vstack([T[up], -T[lo]]).tocsr()
            b = np.r_[np.ones(int(up.sum())), np.zeros(int(lo.sum()))]
            if integral:
                left = (time_limit or CONFIG["milp_time"]) - (time.time() - t0)
                if left <= 0:
                    return None, False, dict(rounds=rounds, rows=nrows)
                cons = [LinearConstraint(A, lb=-np.inf, ub=b)] if A.shape[0] else []
                r = milp(c, constraints=cons, integrality=np.ones(c.size), bounds=Bounds(0, 1),
                         options=dict(time_limit=left, mip_rel_gap=0.0))
                if r.status != 0:
                    return None, False, dict(rounds=rounds, rows=nrows)
            else:
                r = linprog(c, A_ub=A if A.shape[0] else None, b_ub=b if A.shape[0] else None, bounds=(0, 1),
                            method="highs")
                assert r.status == 0, r.message
            v = T @ r.x
            nu, nl = (v > 1 + tol) & ~up, (v < -tol) & ~lo
            if not nu.any() and not nl.any():
                break
            up |= nu
            lo |= nl
        nrows += int(up.sum() + lo.sum())
        total += float(r.fun) + const
    return total, True, dict(rounds=rounds, rows=nrows)


def lazy_cycle_milp(k, s, t, w, start_cycles, time_limit=None):
    """Exact min FAS: binary y_e, cycle rows added until G minus y is acyclic. Returns value, y, ok, rounds."""
    m = s.size
    eid = eid_matrix(k, s, t)
    cycles = list(start_cycles) or sorted(violated_cycles(k, s, t, np.zeros(m), eid))
    if not cycles:
        return 0.0, np.zeros(m, bool), True, 0
    t0 = time.time()
    for r in range(CONFIG["max_milp_rounds"]):
        left = (time_limit or CONFIG["milp_time"]) - (time.time() - t0)
        if left <= 0:
            return None, None, False, r
        res = milp(w, constraints=LinearConstraint(incidence(cycles, m), lb=1, ub=np.inf),
                   integrality=np.ones(m), bounds=Bounds(0, 1), options=dict(time_limit=left, mip_rel_gap=0.0))
        if res.status != 0:
            return None, None, False, r
        y = np.round(res.x).astype(bool)
        new = violated_cycles(k, s, t, np.zeros(m), eid, keep=~y, tol=0.5)
        if not new:
            return float(w[y].sum()), y, True, r + 1
        cycles = cycles + sorted(new - set(cycles))
    return None, None, False, CONFIG["max_milp_rounds"]


def order_from_removed(k, s, t, y):
    """Topological order (rank array) of G minus edges y; asserts acyclic."""
    keep = ~y
    indeg = np.bincount(t[keep], minlength=k)
    out = [[] for _ in range(k)]
    for a, b in zip(s[keep], t[keep]):
        out[a].append(b)
    stack = sorted(np.flatnonzero(indeg == 0).tolist(), reverse=True)
    rank = -np.ones(k, np.int64)
    r = 0
    while stack:
        v = stack.pop()
        rank[v] = r
        r += 1
        for b in out[v]:
            indeg[b] -= 1
            if indeg[b] == 0:
                stack.append(b)
    assert r == k, "not acyclic"
    return rank


def feedback_of_rank(rank, s, t, w):
    return float(w[rank[s] >= rank[t]].sum())


def dp_exact(k, s, t, w):
    """Exact min feedback by subset DP per nontrivial SCC: f(S) = min_v f(S-v) + w(v -> S-v)."""
    Wd = dense(k, s, t, w)
    total = 0.0
    for nodes in scc_blocks(k, s, t)[0]:
        q = nodes.size
        assert q <= CONFIG["dp_max_nodes"]
        B = Wd[np.ix_(nodes, nodes)]
        N = 1 << q
        outw = np.zeros((q, N))
        for v in range(q):
            arr = np.zeros(1)
            for u in range(q):
                arr = np.concatenate([arr, arr + B[v, u]])
            outw[v] = arr
        masks = np.arange(N, dtype=np.int64)
        pop = np.zeros(N, np.int64)
        for v in range(q):
            pop += (masks >> v) & 1
        f = np.full(N, np.inf)
        f[0] = 0.0
        for layer in range(1, q + 1):
            ms = masks[pop == layer]
            best = np.full(ms.size, np.inf)
            for v in range(q):
                has = (ms >> v) & 1 == 1
                prev = ms[has] ^ (1 << v)
                best[has] = np.minimum(best[has], f[prev] + outw[v][prev])
            f[ms] = best
        total += float(f[N - 1])
    return total


def brute(k, s, t, w):
    return min(feedback_of_rank(np.array(p), s, t, w) for p in itertools.permutations(range(k)))


def selftest():
    rng = np.random.default_rng(CONFIG["selftest_seed"])
    worst = 0.0
    for gi in range(CONFIG["selftest_graphs"]):
        k = int(rng.integers(3, 8))
        P = rng.random((k, k)) < rng.uniform(0.3, 0.9)
        np.fill_diagonal(P, False)
        s, t = np.nonzero(P)
        if s.size == 0:
            continue
        w = rng.integers(1, 10, s.size).astype(np.float64)
        opt = brute(k, s, t, w)
        b1 = lb_2cycle(k, s, t, w)
        b2, _ = greedy_23(k, s, t, w)
        b3, cyc, _, dsum = cycle_lp(k, s, t, w)
        b4, _, _ = ordering_bound(k, s, t, w, integral=False)
        e1, y, ok1, _ = lazy_cycle_milp(k, s, t, w, cyc)
        e2, ok2, _ = ordering_bound(k, s, t, w, integral=True)
        e3 = dp_exact(k, s, t, w)
        e4 = feedback_of_rank(order_from_removed(k, s, t, y), s, t, w)
        tol = 1e-6
        assert ok1 and ok2
        assert b1 <= b2 + tol and b2 <= b3 + tol and b3 <= b4 + tol and b4 <= opt + tol, (gi, b1, b2, b3, b4, opt)
        assert abs(dsum - b3) < 1e-6 * max(1, b3)
        for e in (e1, e2, e3, e4):
            assert abs(e - opt) < tol, (gi, e, opt)
        worst = max(worst, opt - b3)
    print("selftest passed", CONFIG["selftest_graphs"], "graphs; max opt - cycleLP", round(worst, 6))


if __name__ == "__main__":
    selftest()
