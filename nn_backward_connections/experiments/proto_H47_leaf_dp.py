"""Prototype gate for H47 — exact subset-DP ordering of contiguous MICRO-blocks.

The question
------------
``SccRecursiveRefiner._refine`` (``src/mfas/refine/scc_recursive.py:212``) opens with

    if nb <= self.min_block or eidx.size == 0:
        return

and ``min_block = 32``. The block refiner also preserves each SCC's internal relative
order by construction (:248). So the champion's structural stage **optimises nothing at
all below 32 nodes** — not badly, but literally not at all. Vahidi 2025 Algorithms 3 and 5
both exhaustively permute sub-SCCs of size <= 9.

This script measures the CAPACITY of that whole band: how much feedforward weight is
recoverable by permuting contiguous blocks of k <= 16 nodes EXACTLY, starting from the
champion's own final order.

Why a tiling measures the capacity
----------------------------------
Contiguous-block lemma (already proved by ``segment.py`` and used by H36/H46): permuting
the nodes inside a contiguous position range cannot change the orientation of any edge
with an endpoint outside that range. So disjoint contiguous windows compose ADDITIVELY —
the summed exact per-window gain IS the realised delta, and it is checked against the
frozen oracle rather than asserted.

Exactness, in three reductions, none of them approximations
-----------------------------------------------------------
1. A window with no BACKWARD intra edge is already at its exact optimum (its forward
   weight equals its total intra weight, which is the trivial upper bound), so it is
   skipped with gain 0.
2. Nodes touched by no intra-window edge contribute nothing wherever they sit.
3. The intra-window edge digraph splits into weakly connected components. There are no
   edges between components, so their interleaving is irrelevant and each component's
   optimum is independent. The window optimum is the sum of the component optima.

Each component of size m is then solved EXACTLY by Held-Karp subset DP,

    f[S] = max over j in S of ( f[S \\ {j}] + sum over i in S \\ {j} of W[i][j] )

which places j last among S and collects every edge from S\\{j} into j. This is
O(2^m * m^2) against O(m! * m) for exhaustive permutation: at m = 16 that is ~1e6
operations against 16! = 2e13, which is what makes the exact threshold reachable at all.

Rounds
------
A single tiling is one arbitrary cut of the line, so it under-measures the family. Each
round re-tiles at a shifted offset and re-optimises; every round is itself exact and
monotone non-decreasing, so the cumulative gain after R rounds is a GENEROUS estimate of
the family's reach. The kill condition is stated on the honest single-tiling number, and
the multi-round number is reported so that a kill cannot be dismissed as an artefact of
where the cuts fell.

Reproduce
---------
    PYTHONPATH=src python experiments/proto_H47_leaf_dp.py --dataset connectome
    PYTHONPATH=src python experiments/proto_H47_leaf_dp.py --dataset microns
    PYTHONPATH=src python experiments/proto_H47_leaf_dp.py --dataset mouse
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mfas.io import load_dataset
from mfas.metrics import pct, score_from_positions

# The champion order for each dataset, taken EXPLICITLY rather than by a name glob.
# H46's prototype picked its position vector by glob and silently measured a just-killed
# run instead of the champion; the score assertion below is the guard against repeating it.
CHAMPION = {
    "connectome": {
        "variant": "H42",
        "positions": "experiments/evidence/"
                     "20260810T105922Z-H42-connectome-s31415-confirm-f41d7e_positions.npy",
        "expect_pct": 84.15409511,
    },
    "microns": {
        "variant": "H42",
        "positions": "experiments/evidence/"
                     "20260810T121650Z-H42-microns-s31415-confirm-f91b66_positions.npy",
        "expect_pct": 83.2409,
    },
    "mouse": {
        "variant": "H52",
        "positions": "experiments/evidence/"
                     "20260816T211722Z-H52-mouse-s1234-confirm-b458f1_positions.npy",
        "expect_pct": 93.1028,
    },
}

_POPCOUNT_CACHE: dict[int, list[np.ndarray]] = {}


def _popcount_layers(m: int) -> list[np.ndarray]:
    """Subsets of ``range(m)`` grouped by popcount, as arrays of bitmask ints."""
    if m not in _POPCOUNT_CACHE:
        idx = np.arange(1 << m, dtype=np.int64)
        pc = np.zeros(1 << m, dtype=np.int64)
        for b in range(m):
            pc += (idx >> b) & 1
        _POPCOUNT_CACHE[m] = [idx[pc == c] for c in range(m + 1)]
    return _POPCOUNT_CACHE[m]


def held_karp(W: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact maximum internal feedforward weight over all ``m!`` orderings, and the order.

    Parameters
    ----------
    W : (m, m) array
        ``W[i, j]`` is the total weight of edges from local node ``i`` to local node ``j``.

    Returns
    -------
    opt : float
        ``max over permutations of sum of W[i, j] for i placed before j``.
    order : (m,) int array
        ``order[r]`` is the local node placed at rank ``r`` in an optimal ordering.
    """
    m = W.shape[0]
    if m <= 1:
        return 0.0, np.arange(m, dtype=np.int64)

    # inc[S, j] = sum over i in S of W[i, j], by a sum-over-subsets transform.
    inc = np.zeros((1 << m, m), dtype=np.float64)
    for i in range(m):
        inc[1 << i] = W[i]
    for b in range(m):
        with_b = np.flatnonzero((np.arange(1 << m, dtype=np.int64) >> b) & 1)
        inc[with_b] += inc[with_b ^ (1 << b)]

    layers = _popcount_layers(m)
    f = np.full(1 << m, -np.inf, dtype=np.float64)
    arg = np.full(1 << m, -1, dtype=np.int64)
    f[0] = 0.0
    for c in range(1, m + 1):
        Sc = layers[c]
        best = np.full(Sc.shape[0], -np.inf, dtype=np.float64)
        bestj = np.full(Sc.shape[0], -1, dtype=np.int64)
        for j in range(m):
            hi = np.flatnonzero((Sc >> j) & 1)
            if hi.size == 0:
                continue
            prev = Sc[hi] ^ (1 << j)
            # placing j LAST among S collects every edge from S\{j} into j
            cand = f[prev] + inc[prev, j]
            better = cand > best[hi]
            best[hi[better]] = cand[better]
            bestj[hi[better]] = j
        f[Sc] = best
        arg[Sc] = bestj

    order = np.empty(m, dtype=np.int64)
    S = (1 << m) - 1
    for r in range(m - 1, -1, -1):
        j = int(arg[S])
        order[r] = j
        S ^= (1 << j)
    return float(f[(1 << m) - 1]), order


def optimise_window(la: np.ndarray, lb: np.ndarray, w: np.ndarray, k: int,
                    max_dp: int) -> tuple[float, np.ndarray | None, int]:
    """Exactly reorder one window's intra edges.

    Parameters
    ----------
    la, lb : int arrays
        Local (within-window) indices of each intra edge's source / target, in ``[0, k)``.
    w : array
        Edge weights.
    k : int
        Window size.
    max_dp : int
        Components larger than this are left untouched (never happens for k <= 16, but the
        guard keeps the cost bounded and is reported).

    Returns
    -------
    gain : float
        Exact increase in feedforward weight, >= 0.
    new_local : array or None
        The new local ordering (``new_local[r]`` = old local index now at local rank ``r``),
        or None when nothing moved.
    n_skipped : int
        Number of components skipped for exceeding ``max_dp``.
    """
    keep = la != lb
    la, lb, w = la[keep], lb[keep], w[keep]
    if la.size == 0:
        return 0.0, None, 0
    if not (la > lb).any():
        # no backward intra edge => already exactly optimal (reduction 1)
        return 0.0, None, 0

    active = np.unique(np.concatenate([la, lb]))
    remap = np.full(k, -1, dtype=np.int64)
    remap[active] = np.arange(active.size)
    ra, rb = remap[la], remap[lb]

    # reduction 3: weakly connected components of the intra-edge digraph
    n_act = active.size
    mat = coo_matrix((np.ones(ra.size, dtype=np.int8), (ra, rb)), shape=(n_act, n_act))
    n_comp, labels = connected_components(mat, directed=True, connection="weak")

    gain = 0.0
    n_skipped = 0
    # new_local starts as the identity over the whole window and is edited per component
    new_local = np.arange(k, dtype=np.int64)
    moved = False

    for c in range(n_comp):
        members = np.flatnonzero(labels == c)          # local-active indices
        m = members.size
        if m <= 1:
            continue
        emask = labels[ra] == c
        ea, eb, ew = ra[emask], rb[emask], w[emask]
        # component-local indexing, ordered by CURRENT position (members is ascending in
        # active order, and active is ascending in window-local order, so this is current)
        cmap = np.full(n_act, -1, dtype=np.int64)
        cmap[members] = np.arange(m)
        ca, cb = cmap[ea], cmap[eb]
        cur = float(ew[ca < cb].sum())
        if not (ca > cb).any():
            continue
        if m > max_dp:
            n_skipped += 1
            continue

        W = np.zeros((m, m), dtype=np.float64)
        np.add.at(W, (ca, cb), ew.astype(np.float64))
        opt, perm = held_karp(W)                        # perm[r] = component node at rank r
        g = opt - cur
        if g <= 0:
            continue

        slots = active[members]                         # window-local slots this component owns
        new_local[slots] = slots[perm]
        gain += g
        moved = True

    return gain, (new_local if moved else None), n_skipped


def run_round(seq: np.ndarray, src_r: np.ndarray, tgt_r: np.ndarray,
              weight: np.ndarray, n: int, k: int, offset: int,
              max_dp: int) -> tuple[float, np.ndarray, dict]:
    """One exact pass over the tiling of the line at ``offset`` with window size ``k``.

    Returns the total exact gain, the new ``seq`` (node at each rank), and stats.
    """
    # Only FULL windows are optimised: the leading stub [0, offset) and any trailing stub
    # shorter than k are left alone, so every window handled below really has k slots.
    n_full = (n - offset) // k
    wid_s = (src_r - offset) // k
    wid_t = (tgt_r - offset) // k
    in_range = (src_r >= offset) & (tgt_r >= offset) & (wid_s < n_full) & (wid_t < n_full)
    intra = (wid_s == wid_t) & in_range
    ei = np.flatnonzero(intra)
    if ei.size == 0:
        return 0.0, seq, {"n_windows_touched": 0, "n_intra_edges": 0,
                          "n_backward_intra": 0, "n_components_skipped": 0}

    wid = wid_s[ei]
    a = src_r[ei] - (offset + wid * k)
    b = tgt_r[ei] - (offset + wid * k)
    w = weight[ei]

    o = np.argsort(wid, kind="stable")
    wid, a, b, w = wid[o], a[o], b[o], w[o]
    bounds = np.flatnonzero(np.diff(wid)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [wid.size]])

    new_seq = seq.copy()
    total_gain = 0.0
    n_touched = 0
    n_skipped = 0
    n_back = int((a > b).sum())

    for s, e in zip(starts, ends):
        gain, new_local, skipped = optimise_window(a[s:e], b[s:e], w[s:e], k, max_dp)
        n_skipped += skipped
        if new_local is None:
            continue
        lo = offset + int(wid[s]) * k
        new_seq[lo:lo + k] = seq[lo:lo + k][new_local]
        total_gain += gain
        n_touched += 1

    return total_gain, new_seq, {"n_windows_touched": n_touched,
                                 "n_intra_edges": int(ei.size),
                                 "n_backward_intra": n_back,
                                 "n_components_skipped": n_skipped}


def ceiling_round(src_r: np.ndarray, tgt_r: np.ndarray, weight: np.ndarray,
                  n: int, k: int, offset: int) -> dict:
    """STRICT upper bound on any within-window reordering gain, at any ``k``.

    A window's feedforward weight can never exceed its TOTAL intra weight (that is what a
    within-window DAG would score), so the most any reordering of that window can gain is
    the weight of its currently-BACKWARD intra edges. Summed over a tiling this is a strict
    upper bound on the exact optimum, and it costs one O(n_edges) pass instead of a DP.

    This is what lets the micro-block band be bounded ABOVE k = 16, where Held-Karp cannot
    go (2^32 subsets): the bound needs no DP at all, so it covers k = 32 — the champion's
    actual ``min_block`` — and beyond.
    """
    n_full = (n - offset) // k
    wid_s = (src_r - offset) // k
    wid_t = (tgt_r - offset) // k
    in_range = (src_r >= offset) & (tgt_r >= offset) & (wid_s < n_full) & (wid_t < n_full)
    intra = (wid_s == wid_t) & in_range & (src_r != tgt_r)
    back = intra & (src_r > tgt_r)
    return {
        "k": k, "offset": offset,
        "n_intra_edges": int(intra.sum()),
        "n_backward_intra": int(back.sum()),
        "ceiling_gain_weight": float(weight[back].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="connectome",
                    choices=sorted(CHAMPION))
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 12, 16],
                    help="window sizes to measure (H47's stated gate is {8, 12, 16})")
    ap.add_argument("--rounds", type=int, default=6,
                    help="re-tilings at shifted offsets; round 0 is the stated statistic")
    ap.add_argument("--offset-stride", type=int, default=0,
                    help="0 => k//2 (two distinct cuts); 1 => every cut, the generous arm")
    ap.add_argument("--max-dp", type=int, default=16,
                    help="largest weakly-connected component solved exactly")
    ap.add_argument("--ceiling-ks", type=int, nargs="*", default=None,
                    help="run ONLY the cheap strict-upper-bound pass at these window sizes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    spec = CHAMPION[ds]
    t_start = time.time()

    g = load_dataset(ds)
    pos = np.load(spec["positions"])
    base = score_from_positions(pos, g.src, g.tgt, g.weight)
    base_pct = pct(base, g.total_weight)
    print(f"[{ds}] champion {spec['variant']}: {base:,.0f} = {base_pct:.8f}%", flush=True)
    assert abs(base_pct - spec["expect_pct"]) < 1e-3, (
        f"stored order does not reproduce the champion: {base_pct} vs {spec['expect_pct']}")

    n = g.n_nodes
    rank0 = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
    seq0 = np.argsort(rank0, kind="stable").astype(np.int64)
    assert score_from_positions(rank0.astype(np.float64), g.src, g.tgt,
                                g.weight) == base, "rank conversion changed the score"

    weight = g.weight.astype(np.float64)

    if args.ceiling_ks:
        src_r, tgt_r = rank0[g.src], rank0[g.tgt]
        ceil = {}
        for k in args.ceiling_ks:
            per_off = [ceiling_round(src_r, tgt_r, weight, n, k, off)
                       for off in range(0, k, max(1, k // 4))]
            # the tiling that leaves the MOST on the table is the generous bound
            best = max(per_off, key=lambda d: d["ceiling_gain_weight"])
            ceil[str(k)] = {
                "ceiling_gain_pp": 100.0 * best["ceiling_gain_weight"] / g.total_weight,
                "ceiling_pct_if_fully_realised": pct(base + best["ceiling_gain_weight"],
                                                     g.total_weight),
                "worst_case_offset": best["offset"],
                "n_backward_intra": best["n_backward_intra"],
                "n_intra_edges": best["n_intra_edges"],
                "per_offset": per_off,
            }
            print(f"  CEILING k={k:5d}: <= +{ceil[str(k)]['ceiling_gain_pp']:.6f} pp "
                  f"({best['n_backward_intra']:,} backward intra edges)", flush=True)
        out = {
            "item": "H47", "role": "prototype-ceiling", "dataset": ds,
            "champion_variant": spec["variant"], "champion_pct": base_pct,
            "total_weight": float(g.total_weight), "n_nodes": n,
            "note": ("STRICT upper bound: max within-window gain <= weight of backward intra "
                     "edges, because a window cannot score more than its total intra weight. "
                     "No DP, so it is valid at any k including k=32 (the champion's min_block)."),
            "ceiling_by_k": ceil,
            "wall_s_total": time.time() - t_start,
        }
        path = Path(args.out or f"experiments/outputs/proto_H47_ceiling_{ds}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}  ({out['wall_s_total']:.1f} s)")
        return

    results = {}

    for k in args.ks:
        seq = seq0.copy()
        rank = rank0.copy()
        cum = 0.0
        rounds = []
        for r in range(args.rounds):
            stride = args.offset_stride or (k // 2)
            offset = (r * stride) % k if r else 0
            t0 = time.time()
            gain, seq_new, stats = run_round(seq, rank[g.src], rank[g.tgt],
                                             weight, n, k, offset, args.max_dp)
            seq = seq_new
            rank = np.empty(n, dtype=np.int64)
            rank[seq] = np.arange(n, dtype=np.int64)
            cum += gain
            realised = score_from_positions(rank.astype(np.float64), g.src, g.tgt, g.weight)
            rounds.append({
                "round": r, "offset": offset,
                "gain_weight": gain,
                "gain_pp": 100.0 * gain / g.total_weight,
                "cum_gain_pp": 100.0 * cum / g.total_weight,
                "realised_pct": pct(realised, g.total_weight),
                "predicted_pct": pct(base + cum, g.total_weight),
                "oracle_matches": bool(abs(realised - (base + cum)) < 1e-6),
                "wall_s": time.time() - t0,
                **stats,
            })
            print(f"  k={k:2d} round {r} off={offset:2d}: "
                  f"+{100.0 * gain / g.total_weight:.6f} pp  "
                  f"cum {100.0 * cum / g.total_weight:.6f} pp  "
                  f"oracle={'OK' if rounds[-1]['oracle_matches'] else 'MISMATCH'}  "
                  f"({rounds[-1]['wall_s']:.1f} s)", flush=True)

        results[str(k)] = {
            "single_tiling_gain_pp": rounds[0]["gain_pp"],
            "cumulative_gain_pp": rounds[-1]["cum_gain_pp"],
            "final_pct": rounds[-1]["realised_pct"],
            "all_rounds_exact": all(x["oracle_matches"] for x in rounds),
            "rounds": rounds,
        }

    out = {
        "item": "H47",
        "role": "prototype",
        "dataset": ds,
        "champion_variant": spec["variant"],
        "champion_positions": spec["positions"],
        "champion_pct": base_pct,
        "champion_weight": float(base),
        "total_weight": float(g.total_weight),
        "n_nodes": n,
        "min_block_in_champion": 32,
        "max_dp": args.max_dp,
        "rounds_per_k": args.rounds,
        "offset_stride": args.offset_stride or "k//2",
        "by_k": results,
        "wall_s_total": time.time() - t_start,
    }
    path = Path(args.out or f"experiments/outputs/proto_H47_{ds}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}  ({out['wall_s_total']:.1f} s)")


if __name__ == "__main__":
    main()
