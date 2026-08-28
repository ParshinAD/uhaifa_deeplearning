"""H82 rung 0 - partition-crossover (PX / IPT) common-block census.

For two orders A, B over the same node set, scan A's positions left to right and
cut wherever the running max of rank_B over A's prefix equals the prefix length
minus one. Those cuts define k COMMON BLOCKS holding the same node set in the
same position interval in both parents. Every cross-block edge is then oriented
identically in every offspring, so the exact score decomposes as

    score(offspring) = C + sum_i f_i(c_i),    c_i in {A, B}

and the per-block argmax realises the best of 2^k offspring in one O(n) scan plus
one O(m) edge pass.

Reference: Chicano, Whitley, Ochoa, Tinos, arXiv:2407.06742 (PPSN 2024) Cor. 1;
Mobius et al., cond-mat/9902034 (Phys. Rev. E 59, 1999) sec. II.1-II.2.

Rung 0 measures k, the block-size distribution, the per-block win split, the
realised best-of-2^k gain over max(parents), and - free, for H83's pre-gate -
the symmetric-difference profile D(p) = |prefix_A(p) delta prefix_B(p)|.

NOTE (P26): numpy's BLAS is broken in this env (np.dot aborts the interpreter).
Nothing here uses a matrix product; only elementwise ops, bincount, argsort and
maximum.accumulate.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
INTEGRAL = True
from mfas.io import load_dataset
from mfas.metrics import score_from_positions, score_from_order


def _same(x, y):
    """Exact for integer accumulators; 1e-9 relative for float64 re-association."""
    if INTEGRAL:
        return x == y
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def ranks_from_positions(positions):
    """rank[node] = 0-based position of that node on the line (stable argsort)."""
    order = np.argsort(np.asarray(positions), kind="stable")   # node at rank r
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank, order


def common_blocks(rank_a, order_a, rank_b):
    """Block-END positions (inclusive) of the common-block partition, in A's coords.

    A prefix of length p+1 is a common set iff max(rank_B[A_order[0:p+1]]) == p.
    """
    n = rank_a.shape[0]
    x = rank_b[order_a]                       # B-rank of the node at A-rank r
    runmax = np.maximum.accumulate(x)
    return np.nonzero(runmax == np.arange(n, dtype=np.int64))[0]


def sym_diff_profile(rank_a, order_a, rank_b):
    """D(p) = |prefix_A(p) delta prefix_B(p)| for every p, in O(n).

    |I(p)| = #{r <= p : x_r <= p} where x_r = rank_B[A_order[r]]. Its increment is
    [x_p <= p] + [invx[p] < p]; the two indicators cannot both fire for the same
    node, so nothing is double counted.
    """
    n = rank_a.shape[0]
    x = rank_b[order_a]
    invx = np.empty(n, dtype=np.int64)
    invx[x] = np.arange(n, dtype=np.int64)
    p = np.arange(n, dtype=np.int64)
    inter = np.cumsum((x <= p).astype(np.int64) + (invx < p).astype(np.int64))
    return 2 * ((p + 1) - inter)


def px_pair(g, rank_a, order_a, rank_b, want_offspring=False):
    """Full PX decomposition of one parent pair. Exact, integer-safe."""
    n = rank_a.shape[0]
    src, tgt, w = g.src, g.tgt, g.weight
    ends = common_blocks(rank_a, order_a, rank_b)
    k = ends.shape[0]

    block_of_rank = np.zeros(n, dtype=np.int64)
    if k > 1:
        block_of_rank[ends[:-1] + 1] = 1
        block_of_rank = np.cumsum(block_of_rank)
    block_of_node = np.empty(n, dtype=np.int64)
    block_of_node[order_a] = block_of_rank

    bs, bt = block_of_node[src], block_of_node[tgt]
    same = bs == bt
    ff_a = rank_a[tgt] > rank_a[src]
    ff_b = rank_b[tgt] > rank_b[src]

    # THE theory check: every cross-block edge must be oriented identically.
    cross_consistent = bool(np.all(ff_a[~same] == ff_b[~same]))

    # Match the frozen scorer's accumulator: int64 for integer weights (connectome,
    # microns), float64 for float weights (mouse). Casting mouse to int64 would
    # truncate every weight to 0 - caught by the mouse smoke test.
    w = np.asarray(w)
    integral = np.issubdtype(w.dtype, np.integer)
    wi = w.astype(np.int64) if integral else w.astype(np.float64)
    const_c = wi[(~same) & ff_a].sum()

    sel_a = same & ff_a
    sel_b = same & ff_b
    f_a = np.bincount(bs[sel_a], weights=wi[sel_a], minlength=k)
    f_b = np.bincount(bs[sel_b], weights=wi[sel_b], minlength=k)
    if integral:
        # bincount always returns float64; for connectome/microns every weight and
        # partial sum is far below 2^53, so rint recovers the exact integer.
        f_a = np.rint(f_a).astype(np.int64)
        f_b = np.rint(f_b).astype(np.int64)
        const_c = int(const_c)
        cast = int
    else:
        cast = float

    score_a = cast(const_c + f_a.sum())
    score_b = cast(const_c + f_b.sum())
    best = cast(const_c + np.maximum(f_a, f_b).sum())

    sizes = np.diff(np.concatenate(([-1], ends))).astype(np.int64)
    out = dict(
        k=int(k), const_c=const_c,
        score_a_decomposed=score_a, score_b_decomposed=score_b,
        best_of_2k=best,
        gain_over_max_parents=best - max(score_a, score_b),
        cross_block_orientation_consistent=cross_consistent,
        n_blocks_size1=int(np.sum(sizes == 1)),
        n_blocks_size_gt1=int(np.sum(sizes > 1)),
        max_block_size=int(sizes.max()), median_block_size=float(np.median(sizes)),
        a_wins=int(np.sum(f_a > f_b)), b_wins=int(np.sum(f_b > f_a)),
        ties=int(np.sum(f_a == f_b)),
        n_nontrivial_choices=int(np.sum(f_a != f_b)),
    )
    if want_offspring:
        take_b = f_b > f_a
        child = rank_a.copy()
        if take_b.any():
            sel = take_b[block_of_rank]            # A-ranks whose block takes B
            nodes = order_a[sel]                   # the nodes living in those blocks
            slots = np.nonzero(sel)[0]             # the position slots they own
            # inside each block, re-order those nodes by B's opinion, keep the slots
            key = np.lexsort((rank_b[nodes], block_of_node[nodes]))
            child[nodes[key]] = slots
        out["_child_ranks"] = child
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--out", default="experiments/outputs/proto_H82_rung0.json")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--verify-top", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    g = load_dataset(args.dataset)
    total_w = float(np.asarray(g.weight, dtype=np.float64).sum())
    n = g.n_nodes
    global INTEGRAL
    INTEGRAL = bool(np.issubdtype(np.asarray(g.weight).dtype, np.integer))
    print("[%7.1fs] %s: n=%d m=%d W=%.0f" % (time.time() - t0, args.dataset, n,
                                             g.src.shape[0], total_w), flush=True)

    files = sorted(Path("results").glob("*-%s-*_positions.npy" % args.dataset))
    pool, seen = [], {}
    for f in files:
        mo = re.match(r".*Z-([A-Za-z0-9]+)-", f.name)
        variant = mo.group(1) if mo else "?"
        pos = np.load(f)
        if pos.shape[0] != n:
            continue
        rank, order = ranks_from_positions(pos)
        h = hash(rank.tobytes())
        if h in seen:
            seen[h]["dupes"].append(f.name)
            continue
        sc = score_from_positions(pos, g.src, g.tgt, g.weight)
        sc_rank = score_from_order(rank, g.src, g.tgt, g.weight)
        sc = int(sc) if INTEGRAL else float(sc)
        sc_rank = int(sc_rank) if INTEGRAL else float(sc_rank)
        entry = dict(variant=variant, file=f.name, score=sc, score_from_rank=sc_rank,
                     pct=100.0 * sc / total_w, rank=rank, order=order, dupes=[])
        seen[h] = entry
        pool.append(entry)
    print("[%7.1fs] %d files -> %d DISTINCT permutations" % (time.time() - t0, len(files), len(pool)),
          flush=True)
    for e in sorted(pool, key=lambda e: -e["score"]):
        flag = "" if e["score"] == e["score_from_rank"] else "  !! RANK/POS SCORE MISMATCH"
        print("    %-7s %.8f  (%d files)%s" % (e["variant"], e["pct"], 1 + len(e["dupes"]), flag),
              flush=True)

    pairs = [(i, j) for i in range(len(pool)) for j in range(i + 1, len(pool))]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print("[%7.1fs] %d pairs to census" % (time.time() - t0, len(pairs)), flush=True)

    rows = []
    for c, (i, j) in enumerate(pairs):
        A, B = pool[i], pool[j]
        r = px_pair(g, A["rank"], A["order"], B["rank"])
        r.update(a=A["variant"], b=B["variant"], a_file=A["file"], b_file=B["file"],
                 score_a=A["score"], score_b=B["score"],
                 gain_pp=100.0 * r["gain_over_max_parents"] / total_w,
                 best_pct=100.0 * r["best_of_2k"] / total_w)
        r["decomposition_exact"] = (_same(r["score_a_decomposed"], A["score"])
                                    and _same(r["score_b_decomposed"], B["score"]))
        rows.append(r)
        if (c + 1) % 25 == 0 or r["gain_over_max_parents"] > 0:
            print("[%7.1fs] %d/%d %sx%s k=%d gain=%.8fpp exact=%s crossOK=%s"
                  % (time.time() - t0, c + 1, len(pairs), A["variant"], B["variant"],
                     r["k"], r["gain_pp"], r["decomposition_exact"],
                     r["cross_block_orientation_consistent"]), flush=True)

    rows.sort(key=lambda r: -r["gain_over_max_parents"])
    n_exact = sum(r["decomposition_exact"] for r in rows)
    n_crossok = sum(r["cross_block_orientation_consistent"] for r in rows)
    print("\n[%7.1fs] decomposition exact on %d/%d; cross-block consistent on %d/%d"
          % (time.time() - t0, n_exact, len(rows), n_crossok, len(rows)), flush=True)

    # --- rung 1 cross-check: realise the top offspring and score them for real
    verified = []
    for r in rows[: args.verify_top]:
        A = next(e for e in pool if e["file"] == r["a_file"])
        B = next(e for e in pool if e["file"] == r["b_file"])
        full = px_pair(g, A["rank"], A["order"], B["rank"], want_offspring=True)
        child = full.pop("_child_ranks")
        assert np.array_equal(np.sort(child), np.arange(n)), "offspring is not a permutation"
        realised = score_from_order(child, g.src, g.tgt, g.weight)
        realised = int(realised) if INTEGRAL else float(realised)
        ok = _same(realised, r["best_of_2k"])
        verified.append(dict(a=r["a"], b=r["b"], k=r["k"], predicted=r["best_of_2k"],
                             realised=realised, match=ok,
                             realised_pct=100.0 * realised / total_w,
                             a_file=r["a_file"], b_file=r["b_file"]))
        print("    VERIFY %sx%s predicted=%r realised=%r %s"
              % (r["a"], r["b"], r["best_of_2k"], realised,
                 "OK" if ok else "*** MISMATCH ***"), flush=True)
        if ok and realised > max(A["score"], B["score"]):
            np.save("dr_tmp/H82_child_%s_%s_ranks.npy" % (r["a"], r["b"]), child)

    # --- H83 free pre-gate: near-cut profile of the most promising pairs
    near = []
    for r in rows[:5]:
        A = next(e for e in pool if e["file"] == r["a_file"])
        B = next(e for e in pool if e["file"] == r["b_file"])
        D = sym_diff_profile(A["rank"], A["order"], B["rank"])
        near.append(dict(a=r["a"], b=r["b"],
                         n_D_eq_0=int(np.sum(D == 0)),
                         n_D_le_2=int(np.sum(D <= 2)), n_D_le_8=int(np.sum(D <= 8)),
                         n_D_le_32=int(np.sum(D <= 32)), n_D_le_128=int(np.sum(D <= 128)),
                         min_nonzero_D=int(D[D > 0].min()) if np.any(D > 0) else 0,
                         median_D=float(np.median(D)), max_D=int(D.max())))
        print("    NEARCUT %sx%s D==0:%d D<=32:%d medianD:%.0f"
              % (r["a"], r["b"], near[-1]["n_D_eq_0"], near[-1]["n_D_le_32"],
                 near[-1]["median_D"]), flush=True)

    champ = 84.25817950936937
    best_row = rows[0] if rows else None
    out = dict(
        item="H82", rung=0, dataset=args.dataset, generated="2026-08-28",
        n_nodes=int(n), n_edges=int(g.src.shape[0]), total_weight=total_w,
        champion_pct=champ, screen_bar_pp=0.012,
        n_files=len(files), n_distinct=len(pool), n_pairs=len(rows),
        pool=[dict(variant=e["variant"], file=e["file"], score=e["score"], pct=e["pct"],
                   n_dupe_files=len(e["dupes"]))
              for e in sorted(pool, key=lambda e: -e["score"])],
        decomposition_exact_count=n_exact, cross_block_consistent_count=n_crossok,
        best_pair=(None if best_row is None else
                   {k: v for k, v in best_row.items() if not k.startswith("_")}),
        top_pairs=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows[:40]],
        max_k=max((r["k"] for r in rows), default=0),
        n_pairs_k_gt_1=sum(r["k"] > 1 for r in rows),
        n_pairs_positive_gain=sum(r["gain_over_max_parents"] > 0 for r in rows),
        best_gain_pp=(0.0 if best_row is None else best_row["gain_pp"]),
        best_offspring_pct=(0.0 if best_row is None else best_row["best_pct"]),
        best_offspring_delta_vs_champion_pp=(0.0 if best_row is None
                                             else best_row["best_pct"] - champ),
        verified=verified, near_cut_profile=near,
        wall_clock_s=time.time() - t0,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[%7.1fs] WROTE %s" % (time.time() - t0, args.out), flush=True)
    print("MAX k = %d; pairs with k>1 = %d/%d; best gain = %.8f pp; "
          "best offspring vs champion = %+.8f pp"
          % (out["max_k"], out["n_pairs_k_gt_1"], len(rows), out["best_gain_pp"],
             out["best_offspring_delta_vs_champion_pp"]), flush=True)


if __name__ == "__main__":
    main()
