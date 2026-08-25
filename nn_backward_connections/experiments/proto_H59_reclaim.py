"""H59 prototype gate - minimal-FAS arc reclamation, measured on the champion order.

Sealed pre-registration: ``experiments/prereg/H59_arc_reclamation.md``, committed BEFORE this
script was ever run. Read it first. The gate it states is the REALISED oracle score, not the
reclaimable-weight capacity sum, because meta-rule M11 forbids capacity as evidence.

The lemma
---------
The champion order induces the forward edge set ``F`` - every edge ``u -> v`` with
``rank[u] < rank[v]``. ``F`` is a DAG and the champion order is one of its topological orders.
For a BACKWARD edge ``(u, v)`` (so ``rank[v] < rank[u]``), if ``u`` is unreachable from ``v`` in
``F`` then ``F u {(u,v)}`` is acyclic, and EVERY topological order of it scores at least
``base + w_uv``: a topological order makes every arc of its own DAG feedforward, so all of ``F``
is retained and ``(u,v)`` is gained. Reclamation is therefore exactly monotone and the realised
gain is bounded BELOW by the reclaimed weight.

Why the search is cheap
-----------------------
Every ``F``-path moves strictly rightward in rank, so any ``v -> u`` path is confined to the open
rank interval ``(rank[v], rank[u])``. No global search is ever needed for the round-opening scan.

Leakage-safety: only edge weights and ranks are read. The frozen oracle SCORES the result; it
never chooses an arc. ``data/best_solution`` is never opened.

Run:  PYTHONPATH=src python experiments/proto_H59_reclaim.py <dataset> [--rounds N] [--budget N]
Out:  experiments/outputs/proto_H59_<dataset>.json
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                    # noqa: E402
from mfas.metrics import pct, score_from_order         # noqa: E402


# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------
def build_csr(row, col, n):
    """CSR of a directed edge list, each row block sorted ascending by column."""
    order = np.lexsort((col, row))
    row_s, col_s = row[order], col[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(row_s, minlength=n), out=indptr[1:])
    return indptr, np.ascontiguousarray(col_s, dtype=np.int64)


# ---------------------------------------------------------------------------
# Reachability inside the rank interval
# ---------------------------------------------------------------------------
def reachable(out_ptr, out_idx, in_ptr, in_idx, rank, v, u, budget):
    """Is ``u`` reachable from ``v`` in ``F``?  True / False / None (budget exhausted).

    Bidirectional BFS confined to the open rank interval ``(rank[v], rank[u])``: an ``F``-path
    only ever moves rightward, so nothing outside the window can lie on one. The two frontiers
    are expanded cheaper-side-first, which is what makes the common "yes, in two hops" case
    cost almost nothing.
    """
    rv = rank[v]
    ru = rank[u]
    if ru <= rv:                       # not a backward pair; nothing to test
        return True

    out_v = out_idx[out_ptr[v]:out_ptr[v + 1]]
    j = np.searchsorted(out_v, u)
    if j < out_v.shape[0] and out_v[j] == u:
        return True                    # 1-hop: a direct v -> u edge in F
    if ru == rv + 1:
        return False                   # empty interval and no direct edge

    fwd = {v}
    bwd = {u}
    f_frontier = [v]
    b_frontier = [u]
    spent = 0

    while f_frontier and b_frontier:
        if len(f_frontier) <= len(b_frontier):
            nxt = []
            for x in f_frontier:
                row = out_idx[out_ptr[x]:out_ptr[x + 1]]
                spent += row.shape[0]
                for y in row:
                    y = int(y)
                    yr = rank[y]
                    if yr >= ru:
                        if y == u:
                            return True
                        continue       # past the right edge of the window
                    if yr <= rv or y in fwd:
                        continue
                    if y in bwd:
                        return True
                    fwd.add(y)
                    nxt.append(y)
            f_frontier = nxt
        else:
            nxt = []
            for x in b_frontier:
                row = in_idx[in_ptr[x]:in_ptr[x + 1]]
                spent += row.shape[0]
                for y in row:
                    y = int(y)
                    yr = rank[y]
                    if yr <= rv:
                        if y == v:
                            return True
                        continue       # past the left edge of the window
                    if yr >= ru or y in bwd:
                        continue
                    if y in fwd:
                        return True
                    bwd.add(y)
                    nxt.append(y)
            b_frontier = nxt
        if spent > budget:
            return None
    return False


def reachable_with_extra(out_ptr, out_idx, extra, v, u):
    """Is ``u`` reachable from ``v`` in ``F u S``?  Exact, unpruned.

    Once arcs have been accepted into ``S`` they run leftward in the champion rank, so the
    interval confinement above no longer holds and the search must be global. ``extra`` is a
    dict node -> list of accepted successors; ``S`` is expected to be tiny, so keeping it on
    the side is far cheaper than rebuilding a 4.5M-edge CSR per acceptance.
    """
    seen = {v}
    stack = [v]
    while stack:
        x = stack.pop()
        for y in out_idx[out_ptr[x]:out_ptr[x + 1]]:
            y = int(y)
            if y == u:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
        for y in extra.get(x, ()):
            if y == u:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


# ---------------------------------------------------------------------------
# Rank-stable topological sort of F u S
# ---------------------------------------------------------------------------
def topo_rank_stable(out_ptr, out_idx, indeg, n, key):
    """Kahn with a min-heap on ``key`` - the topological order closest to the champion order.

    Returns ``order`` (``order[p]`` = node at position ``p``), or ``None`` if a cycle remains,
    which would mean the acyclicity bookkeeping upstream is wrong.
    """
    deg = indeg.copy()
    heap = [int(x) for x in np.flatnonzero(deg == 0)]
    heap = [(int(key[x]), x) for x in heap]
    heapq.heapify(heap)
    order = np.empty(n, dtype=np.int64)
    filled = 0
    while heap:
        _, x = heapq.heappop(heap)
        order[filled] = x
        filled += 1
        for y in out_idx[out_ptr[x]:out_ptr[x + 1]]:
            y = int(y)
            deg[y] -= 1
            if deg[y] == 0:
                heapq.heappush(heap, (int(key[y]), y))
    if filled != n:
        return None
    return order


def find_champion(dataset):
    """Champion position vector, straight off a logged confirm run. Never recomputed."""
    champ = "H52" if dataset == "mouse" else "H42"
    for f in reversed(sorted(glob.glob(f"results/*-{champ}-{dataset}-s42-confirm-*.json"))):
        d = json.load(open(f))
        p = d.get("best_positions_path")
        if p and Path(p).exists():
            return champ, p, d["pct"]
    return champ, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="connectome")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=400_000,
                    help="per-query edge-expansion budget for the bidirectional BFS")
    ap.add_argument("--time-limit", type=float, default=2400.0)
    args = ap.parse_args()

    t_start = time.time()
    g = io.load_dataset(args.dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight)
    n = g.n_nodes
    total = g.total_weight

    champ, pos_path, champ_pct_record = find_champion(args.dataset)
    if pos_path is None:
        print(f"no {champ} champion position vector found for {args.dataset}")
        return 2

    positions = np.load(pos_path)
    rank = np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)
    base_score = score_from_order(rank, src, tgt, g.weight)
    base_pct = pct(base_score, total)
    print(f"[{args.dataset}] champion {champ}  {pos_path}", flush=True)
    print(f"[{args.dataset}] base pct={base_pct:.8f}  (record {champ_pct_record})", flush=True)

    keep = src != tgt
    e_src, e_tgt, e_w = src[keep], tgt[keep], w[keep]

    rec = dict(item="H59", dataset=args.dataset, champion=champ,
               champion_positions=pos_path, champion_pct=base_pct,
               n_nodes=int(n), n_edges=int(g.n_edges), n_self_loops=int((~keep).sum()),
               total_weight=float(total), budget=args.budget,
               rounds_requested=args.rounds,
               prereg="experiments/prereg/H59_arc_reclamation.md", rounds=[])

    cur_rank = rank.copy()
    cur_score = base_score

    for rnd in range(args.rounds):
        t_r = time.time()
        fwd_mask = cur_rank[e_src] < cur_rank[e_tgt]
        F_src, F_tgt = e_src[fwd_mask], e_tgt[fwd_mask]
        B_src, B_tgt, B_w = e_src[~fwd_mask], e_tgt[~fwd_mask], e_w[~fwd_mask]
        n_back = int(B_w.shape[0])
        w_back = float(B_w.sum())

        out_ptr, out_idx = build_csr(F_src, F_tgt, n)
        in_ptr, in_idx = build_csr(F_tgt, F_src, n)

        order_b = np.argsort(-B_w.astype(np.float64), kind="stable")
        print(f"[{args.dataset}] round {rnd}: {n_back:,} backward edges, weight "
              f"{w_back:,.0f} ({100.0 * w_back / total:.4f} pp)", flush=True)

        reclaimable = []
        n_unknown = 0
        w_unknown = 0.0
        n_tested = 0
        truncated = False
        t_scan0 = time.time()
        for i in order_b:
            u, v, wt = int(B_src[i]), int(B_tgt[i]), float(B_w[i])
            r = reachable(out_ptr, out_idx, in_ptr, in_idx, cur_rank, v, u, args.budget)
            n_tested += 1
            if r is None:
                n_unknown += 1
                w_unknown += wt
            elif r is False:
                reclaimable.append((wt, u, v))
            if n_tested % 50_000 == 0:
                print(f"    tested {n_tested:,}/{n_back:,}  reclaimable={len(reclaimable):,}  "
                      f"unknown={n_unknown:,}  {time.time() - t_scan0:.0f}s", flush=True)
            if time.time() - t_start > args.time_limit:
                print(f"    TIME LIMIT after {n_tested:,}/{n_back:,} tests", flush=True)
                truncated = True
                break
        t_scan = time.time() - t_scan0

        w_reclaimable = float(sum(x[0] for x in reclaimable))
        print(f"[{args.dataset}] round {rnd}: reclaimable={len(reclaimable):,}  weight="
              f"{w_reclaimable:,.0f} ({100.0 * w_reclaimable / total:.6f} pp CAPACITY)  "
              f"unknown={n_unknown:,} (w={w_unknown:,.0f})  scan={t_scan:.0f}s", flush=True)

        # greedy heaviest-first re-add, acyclicity re-checked against F u S
        acc_src, acc_tgt, acc_w = [], [], []
        extra = {}
        n_conflicts = 0
        t_g0 = time.time()
        for wt, u, v in sorted(reclaimable, key=lambda x: -x[0]):
            if extra and reachable_with_extra(out_ptr, out_idx, extra, v, u):
                n_conflicts += 1
                continue
            acc_src.append(u)
            acc_tgt.append(v)
            acc_w.append(wt)
            extra.setdefault(u, []).append(v)
        t_greedy = time.time() - t_g0
        w_S = float(sum(acc_w))
        print(f"[{args.dataset}] round {rnd}: accepted={len(acc_w):,} weight={w_S:,.0f} "
              f"({100.0 * w_S / total:.6f} pp)  conflicts={n_conflicts:,}  "
              f"greedy={t_greedy:.0f}s", flush=True)

        # one rank-stable topological re-sort of F u S, then the FROZEN oracle
        fatal = None
        if acc_w:
            A_src = np.concatenate([F_src, np.asarray(acc_src, dtype=np.int64)])
            A_tgt = np.concatenate([F_tgt, np.asarray(acc_tgt, dtype=np.int64)])
            a_ptr, a_idx = build_csr(A_src, A_tgt, n)
            indeg = np.bincount(A_tgt, minlength=n).astype(np.int64)
            new_order = topo_rank_stable(a_ptr, a_idx, indeg, n, cur_rank)
            if new_order is None:
                fatal = "cycle in F u S - acyclicity bookkeeping is wrong"
                print(f"    FATAL: {fatal}", flush=True)
                new_rank, new_score = cur_rank, cur_score
            else:
                new_rank = np.empty(n, dtype=np.int64)
                new_rank[new_order] = np.arange(n, dtype=np.int64)
                new_score = score_from_order(new_rank, src, tgt, g.weight)
        else:
            new_rank, new_score = cur_rank, cur_score

        lemma_holds = bool(float(new_score) + 1e-9 >= float(cur_score) + w_S)
        d_pp = 100.0 * (float(new_score) - float(cur_score)) / total
        rec["rounds"].append(dict(
            round=rnd, n_backward=n_back, w_backward=w_back,
            n_tested=n_tested, scan_truncated=truncated,
            n_reclaimable=len(reclaimable),
            w_reclaimable_CAPACITY=w_reclaimable,
            pp_reclaimable_CAPACITY=100.0 * w_reclaimable / total,
            n_unknown=n_unknown, w_unknown=w_unknown,
            pp_unknown_upper_bound=100.0 * w_unknown / total,
            n_accepted=len(acc_w), w_accepted=w_S, n_conflicts=n_conflicts,
            pp_accepted=100.0 * w_S / total,
            score_before=float(cur_score), score_after=float(new_score),
            pct_before=pct(cur_score, total), pct_after=pct(new_score, total),
            realised_delta_pp=d_pp, lemma_holds=lemma_holds, fatal=fatal,
            t_scan_s=t_scan, t_greedy_s=t_greedy, t_round_s=time.time() - t_r,
            heaviest=[dict(w=float(x[0]), u=int(x[1]), v=int(x[2]))
                      for x in sorted(reclaimable, key=lambda x: -x[0])[:20]]))
        print(f"[{args.dataset}] round {rnd}: REALISED {pct(cur_score, total):.8f} -> "
              f"{pct(new_score, total):.8f} = {d_pp:+.6f} pp   lemma_holds={lemma_holds}",
              flush=True)

        cur_rank, cur_score = new_rank, new_score
        if fatal or not acc_w or truncated:
            if not acc_w and not fatal:
                print(f"[{args.dataset}] round {rnd}: nothing reclaimed - stopping", flush=True)
            break
        if time.time() - t_start > args.time_limit:
            break

    rec["final_pct"] = pct(cur_score, total)
    rec["realised_delta_pp_total"] = 100.0 * (float(cur_score) - float(base_score)) / total
    rec["wall_clock_s"] = time.time() - t_start
    Path("experiments/outputs").mkdir(parents=True, exist_ok=True)
    out = f"experiments/outputs/proto_H59_{args.dataset}.json"
    json.dump(rec, open(out, "w"), indent=2)
    print(f"[{args.dataset}] TOTAL realised {base_pct:.8f} -> {rec['final_pct']:.8f} = "
          f"{rec['realised_delta_pp_total']:+.6f} pp   -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
