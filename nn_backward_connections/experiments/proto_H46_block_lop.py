"""H46 prototype gate — coarse multilevel k-block permutation, exact, one O(m) pass per level.

The idea
--------
H41's segment move is a 2-block ADJACENT swap capped at ``L, d <= 1024`` — a 2,048-position
window, 1.5% of the connectome line. It fired 4,619 times and died of redundancy (91.9% of what
it found the existing classes already reach). The natural generalisation is not a longer ladder
but a different SHAPE of move: partition the whole line into ``K`` contiguous blocks and permute
the blocks.

Why it is exact, and why it is cheap
-------------------------------------
By the contiguous-block lemma already proved in ``mfas/refine/segment.py``: if every block keeps
its internal order, then for any permutation of the blocks, an edge with both endpoints in the
same block never flips, and an edge between blocks ``a`` and ``b`` is feedforward iff block
``a`` precedes block ``b``. So the score decomposes as

    intra (constant under any block permutation)  +  sum over ordered block pairs (a before b) of W[a][b]

where ``W[a][b]`` is the total weight of edges from block ``a`` to block ``b``. One
``np.bincount`` over all edges builds the whole ``K x K`` matrix, so the exact gain of EVERY
permutation is read off ``W`` with no rescoring at all. Choosing the best permutation of ``K``
blocks is exactly a dense Linear Ordering Problem on ``K`` nodes — for ``K <= 8`` that is 40,320
permutations, evaluated on an 8x8 matrix, i.e. free.

This is the regime the diagnosis points at and no move class has covered: ``diagnosis.md`` puts
the rank-distance percentiles at p25 8,290 / p50 22,580 / p75 54,432, and a K-block partition of
136,648 positions has blocks of 17,081 (K=8) to 68,324 (K=2) — the same order of magnitude.

What this gate decides
----------------------
For each level ``K`` and each offset of the block boundaries, it computes the exact gain of the
best block permutation on the CHAMPION'S order. If the best gain across all levels is below the
0.012 pp minimum effect size, the whole coarse-block family is closed by a measurement that
costs seconds rather than by a screen. A non-trivial gain would be a genuine new move class.

Boundary offsets matter: a fixed equal partition can cut straight through a group that wants to
move. Each level is therefore tried at several shifted origins, which is cheap (one pass each).

Leakage-safety: ``W`` is built from the input edge weights and the current ranks alone. The
frozen oracle is used only to CHECK a realised gain after a permutation is applied, never to
choose one. ``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H46_block_lop.py [dataset]
Out:  experiments/outputs/proto_H46_<dataset>.json
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                          # noqa: E402
from mfas.metrics import pct, score_from_order               # noqa: E402

LEVELS = (2, 3, 4, 5, 6, 7, 8)
OFFSET_FRACS = (0.0, 0.25, 0.5, 0.75)      # boundary origins, as a fraction of a block


def block_matrix(rank, src, tgt, w, bounds):
    """``K x K`` inter-block weight matrix for a contiguous partition given by ``bounds``.

    ``bounds`` is the sorted array of block start positions (length K, first entry 0). One
    ``searchsorted`` maps each endpoint's POSITION to its block, then a single ``bincount``
    over the flattened pair index builds the matrix. O(m) total.
    """
    K = bounds.shape[0]
    b_src = np.searchsorted(bounds, rank[src], side="right") - 1
    b_tgt = np.searchsorted(bounds, rank[tgt], side="right") - 1
    flat = np.bincount(b_src * K + b_tgt, weights=w, minlength=K * K)
    return flat.reshape(K, K)


def best_permutation_gain(W):
    """Exact gain of the best block permutation over the identity, by brute force.

    Score of a permutation ``p`` (blocks laid out in the order ``p[0], p[1], ...``) is
    ``sum_{i<j} W[p[i]][p[j]]``. Intra-block weight is on the diagonal and is invariant, so it
    cancels in the gain and is never added.
    """
    K = W.shape[0]
    idx = np.arange(K)

    def val(p):
        s = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                s += W[p[i], p[j]]
        return s

    base = val(idx)
    best, best_p = base, tuple(idx)
    for p in itertools.permutations(range(K)):
        v = val(p)
        if v > best:
            best, best_p = v, p
    return best - base, best_p, base


def apply_block_perm(seq, bounds, perm, n):
    """Rebuild the position sequence with the blocks laid out in ``perm`` order."""
    ends = np.append(bounds[1:], n)
    parts = [seq[bounds[b]:ends[b]] for b in perm]
    return np.concatenate(parts)


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n, total = g.n_nodes, g.total_weight

    # Take the CHAMPION for this dataset from the registry, not from a glob.
    # A glob over "H4*" sorted by name picks whatever variant ran last, which on the first run
    # of this script was H49 - a variant that had just been KILLED. Measuring a move class on a
    # non-champion order silently answers a different question.
    sota = json.loads((_ROOT / "autoresearch" / "sota.json").read_text())
    champ = sota["datasets"][dataset]["champion"]
    cands = sorted(glob.glob(f"results/*-{champ}-{dataset}-*.json"))
    pos_path = None
    for f in reversed(cands):
        d = json.load(open(f))
        p = d.get("best_positions_path")
        if p and Path(p).exists():
            pos_path, champ_pct = p, d["pct"]
            break
    if pos_path is None:
        print(f"no position vector on disk for champion {champ} on {dataset}")
        return 2
    print(f"champion for {dataset} per sota.json: {champ}", flush=True)
    positions = np.load(pos_path)
    rank = np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)
    base_score = score_from_order(rank, src, tgt, g.weight)
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    print(f"champion order: {os.path.basename(pos_path)}  "
          f"pct={pct(base_score, total):.6f} (record {champ_pct})", flush=True)

    rows, best_overall = [], None
    t0 = time.time()
    for K in LEVELS:
        blk = n // K
        for frac in OFFSET_FRACS:
            off = int(frac * blk)
            if off == 0:
                bounds = np.array([i * blk for i in range(K)], dtype=np.int64)
            else:
                # shifted origin: a short leading block, then K-1 full ones
                bounds = np.array([0] + [off + i * blk for i in range(K)], dtype=np.int64)
                bounds = bounds[bounds < n]
            W = block_matrix(rank, src, tgt, w, bounds)
            gain, perm, _ = best_permutation_gain(W)
            row = dict(K=int(bounds.shape[0]), offset_frac=frac,
                       gain_pp=100.0 * gain / total, perm=[int(x) for x in perm],
                       is_identity=bool(list(perm) == sorted(perm)))
            rows.append(row)
            if best_overall is None or gain > best_overall[0]:
                best_overall = (gain, bounds, perm, row)
    t_scan = time.time() - t0

    print(f"scanned {len(rows)} (level, offset) partitions in {t_scan:.1f}s", flush=True)
    top = sorted(rows, key=lambda r: -r["gain_pp"])[:6]
    for r in top:
        print(f"  K={r['K']:>2} off={r['offset_frac']:<5} gain={r['gain_pp']:+.6f} pp  "
              f"perm={'identity' if r['is_identity'] else r['perm']}", flush=True)

    # EXACTNESS: apply the single best permutation and check against the frozen oracle
    gain, bounds, perm, _ = best_overall
    new_seq = apply_block_perm(seq, bounds, perm, n)
    new_rank = np.empty(n, dtype=np.int64)
    new_rank[new_seq] = np.arange(n, dtype=np.int64)
    realised = score_from_order(new_rank, src, tgt, g.weight) - base_score
    match = abs(realised - gain) < 1e-6
    print(f"EXACTNESS: predicted {100.0*gain/total:+.6f} pp, realised "
          f"{100.0*realised/total:+.6f} pp -> {'MATCH' if match else 'MISMATCH'}", flush=True)

    best_pp = 100.0 * gain / total
    verdict = "PASS" if best_pp > 0.012 else "FAIL (below the 0.012 pp minimum effect size)"
    print(f"BEST coarse-block gain: {best_pp:+.6f} pp  -> {verdict}", flush=True)

    out = dict(item="H46", dataset=dataset, champion_positions=pos_path,
               champion_pct=pct(base_score, total), levels=list(LEVELS),
               offset_fracs=list(OFFSET_FRACS), t_scan_s=t_scan,
               partitions=rows, best_gain_pp=best_pp,
               best_K=int(bounds.shape[0]), best_perm=[int(x) for x in perm],
               exactness_match=bool(match),
               realised_pp=100.0 * realised / total, verdict=verdict)
    op = _ROOT / "experiments" / "outputs" / f"proto_H46_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
