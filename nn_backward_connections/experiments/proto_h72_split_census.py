"""H72 rung 1 - is the SCC recursion's split point leaving repairable weight on the table?

SccRecursiveRefiner._split (src/mfas/refine/scc_recursive.py:275) cuts a block that is a
single SCC at

    mid = lo + round(nb * split_frac),      split_frac cycled over DEFAULT_SPLIT_FRACS

i.e. at a fixed fraction of the block's POSITION range. It never reads an edge weight.
Every BACKWARD edge straddling that cut is dropped from both halves (the masks require
both endpoints on the same side) and is therefore invisible to the rest of the pass; a
forward edge straddling the cut is irrelevant, because the halves only permute inside
themselves and so cannot break it.

The crossing backward weight of a candidate cut is

    X(mid) = sum over edges (u,v) inside the block with pos(v) < pos(u)
             of w * 1[pos(v) < mid <= pos(u)]

which is computable for ALL mid at once in O(|eidx| + nb) with one difference array plus
one cumsum - the same asymptotic cost as the two boolean masks _split already builds.

This script measures, over one full pass of the REAL recursion (the instrumented class
below subclasses the frozen refiner and delegates to it, so the traversal is byte-identical
to production), for every _split call:

    X(mid_fixed)                     what the current rule pays
    min X over a balanced band       what the argmin rule would pay
    min X over every valid cut       the unconstrained floor (diagnostic only)

and aggregates the total avoidable crossing weight into pp of the dataset's total weight.

PRE-REGISTERED KILL CONDITION (rung 1), written before the run:
  Let CAP_band = sum over _split calls of (X_fixed - X_bandmin), in pp, for the split_frac
  of DEFAULT_SPLIT_FRACS most favourable to the hypothesis.
  KILL if CAP_band < 0.012 pp on connectome, because CAP_band is the weight the mechanism
  directly un-hides in one pass and the connectome promotion bar is 0.012 pp: if even a
  100% conversion of un-hidden weight into score falls short of the bar, the mechanism
  cannot deliver it. (M11 in reverse: capacity is an UPPER bound on achievability, so a
  capacity below the bar is a hard kill, while a capacity above it proves nothing.)
  Honesty caveat, stated up front: CAP_band is a first-order capacity, not a formal bound -
  moving the cut also changes the downstream recursion. That is why it is used only to
  KILL, never to keep.

Also reported (whatever the verdict): the per-level relative reduction
(X_fixed - X_bandmin)/X_fixed at the top recursion levels, the argmin's offset from the
fixed cut in units of nb, and the same numbers for the unconstrained argmin.

NOTE (P26): numpy's BLAS is broken in this env (np.dot aborts the interpreter). Nothing
here uses a matrix product.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from mfas.io import load_dataset
from mfas.metrics import score_from_order
from mfas.refine.scc_recursive import SccRecursiveRefiner, DEFAULT_SPLIT_FRACS


class InstrumentedRefiner(SccRecursiveRefiner):
    """The frozen refiner, plus a record of what every _split call could have cut instead.

    Delegates the actual split to the parent, so the traversal, the recursion order and
    the resulting rank are bit-identical to production.
    """

    def __init__(self, src, tgt, n, weight, *, band_frac, **kw):
        super().__init__(src, tgt, n, **kw)
        self.w = np.asarray(weight)
        self.band_frac = float(band_frac)
        self.records = []
        self._depth = 0

    def _refine(self, lo, hi, eidx, is_scc):
        self._depth += 1
        try:
            super()._refine(lo, hi, eidx, is_scc)
        finally:
            self._depth -= 1

    def _split(self, lo, hi, eidx):
        nb = hi - lo
        ps = self.pos[self.src[eidx]]
        pt = self.pos[self.tgt[eidx]]
        w = self.w[eidx]

        # Backward edges inside the block: pos(tgt) < pos(src).
        back = pt < ps
        # X(mid) for every valid cut mid in [lo+1, hi-1]. A backward edge (u,v) crosses
        # mid iff pos(v) < mid <= pos(u), i.e. mid in [pt+1, ps].
        acc = np.float64 if self.w.dtype.kind == "f" else np.int64
        diff = np.zeros(nb + 2, dtype=acc)
        if back.any():
            a = (pt[back] - lo) + 1          # local first crossing cut
            b = (ps[back] - lo)              # local last crossing cut (inclusive)
            np.add.at(diff, a, w[back].astype(acc))
            np.add.at(diff, b + 1, -w[back].astype(acc))
        xcurve = np.cumsum(diff[:nb + 1])    # xcurve[m] = X(lo + m), m in [0, nb]

        mid = lo + max(1, min(nb - 1, int(round(nb * self.split_frac))))
        m_fixed = mid - lo
        valid_lo, valid_hi = 1, nb - 1       # both halves non-empty
        x_fixed = float(xcurve[m_fixed])

        half = max(1, int(round(nb * self.band_frac)))
        b_lo = max(valid_lo, m_fixed - half)
        b_hi = min(valid_hi, m_fixed + half)
        seg = xcurve[b_lo:b_hi + 1]
        j = int(np.argmin(seg))
        m_band = b_lo + j
        x_band = float(seg[j])

        segg = xcurve[valid_lo:valid_hi + 1]
        jg = int(np.argmin(segg))
        m_glob = valid_lo + jg
        x_glob = float(segg[jg])

        self.records.append(dict(
            depth=self._depth, lo=int(lo), hi=int(hi), nb=int(nb),
            n_edges=int(eidx.size), n_back=int(back.sum()),
            back_w_total=float(w[back].sum()) if bool(back.any()) else 0.0,
            split_frac=float(self.split_frac),
            m_fixed=int(m_fixed), x_fixed=x_fixed,
            m_band=int(m_band), x_band=x_band,
            m_glob=int(m_glob), x_glob=x_glob,
        ))
        super()._split(lo, hi, eidx)


def run_pass(g, rank, split_frac, band_frac, min_block=32):
    """One instrumented pass of the production refiner at a given split_frac."""
    ref = InstrumentedRefiner(np.asarray(g.src), np.asarray(g.tgt), g.n_nodes,
                              g.weight, band_frac=band_frac,
                              min_block=min_block, split_frac=split_frac)
    t0 = time.time()
    out = ref.run(np.asarray(rank, dtype=np.int64))
    return out, ref.records, time.time() - t0


def summarise(records, total_weight):
    """Aggregate one pass's split records into the pre-registered numbers."""
    if not records:
        return dict(n_splits=0, cap_band_pp=0.0, cap_glob_pp=0.0,
                    x_fixed_total_pp=0.0, x_band_total_pp=0.0, x_glob_total_pp=0.0,
                    by_level={})
    xf = np.array([r["x_fixed"] for r in records], dtype=np.float64)
    xb = np.array([r["x_band"] for r in records], dtype=np.float64)
    xg = np.array([r["x_glob"] for r in records], dtype=np.float64)
    nb = np.array([r["nb"] for r in records], dtype=np.float64)
    mf = np.array([r["m_fixed"] for r in records], dtype=np.float64)
    mb = np.array([r["m_band"] for r in records], dtype=np.float64)
    depth = np.array([r["depth"] for r in records], dtype=np.int64)
    pp = 100.0 / float(total_weight)
    rel = np.where(xf > 0, (xf - xb) / np.maximum(xf, 1e-12), 0.0)

    by_level = {}
    for d in sorted(set(depth.tolist()))[:8]:
        m = depth == d
        by_level[int(d)] = dict(
            n=int(m.sum()),
            median_nb=float(np.median(nb[m])),
            x_fixed_pp=float(xf[m].sum() * pp),
            x_band_pp=float(xb[m].sum() * pp),
            x_glob_pp=float(xg[m].sum() * pp),
            median_rel_reduction=float(np.median(rel[m])),
            median_offset_frac=float(np.median(np.abs(mb[m] - mf[m]) / nb[m])),
        )
    return dict(
        n_splits=len(records),
        x_fixed_total_pp=float(xf.sum() * pp),
        x_band_total_pp=float(xb.sum() * pp),
        x_glob_total_pp=float(xg.sum() * pp),
        cap_band_pp=float((xf - xb).sum() * pp),
        cap_glob_pp=float((xf - xg).sum() * pp),
        median_rel_reduction=float(np.median(rel)),
        mean_rel_reduction=float(rel.mean()),
        frac_splits_improvable=float((xf - xb > 0).mean()),
        median_offset_frac=float(np.median(np.abs(mb - mf) / nb)),
        largest_block_nb=int(nb.max()),
        by_level=by_level,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--order", required=True, help="stored positions .npy (champion order)")
    ap.add_argument("--band-frac", type=float, default=0.10,
                    help="half-width of the balanced band, as a fraction of nb")
    ap.add_argument("--min-block", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    g = load_dataset(args.dataset)
    pos = np.load(args.order)
    rank = np.asarray(pos, dtype=np.int64)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    base = score_from_order(rank, src, tgt, g.weight)
    pp = 100.0 / float(g.total_weight)
    print(f"{args.dataset}: base order scores {base * pp:.8f} pct")

    per_frac = {}
    for sf in DEFAULT_SPLIT_FRACS:
        out, recs, wall = run_pass(g, rank, sf, args.band_frac, args.min_block)
        s_out = score_from_order(out, src, tgt, g.weight)
        summ = summarise(recs, g.total_weight)
        summ["wall_s"] = wall
        summ["pass_delta_pp"] = float((s_out - base) * pp)
        per_frac[f"{sf}"] = summ
        print(f"  frac={sf}: {summ['n_splits']} splits, "
              f"X_fixed={summ['x_fixed_total_pp']:.6f} pp, "
              f"X_band={summ['x_band_total_pp']:.6f} pp, "
              f"CAP_band={summ['cap_band_pp']:.6f} pp, "
              f"CAP_glob={summ['cap_glob_pp']:.6f} pp, "
              f"pass_delta={summ['pass_delta_pp']:+.6f} pp, {wall:.1f} s")

    caps = {k: v["cap_band_pp"] for k, v in per_frac.items()}
    best_frac = max(caps, key=caps.get)
    payload = dict(
        item="H72", rung=1, dataset=args.dataset, order=args.order,
        band_frac=args.band_frac, min_block=args.min_block,
        base_pct=float(base * pp), total_weight=float(g.total_weight),
        per_split_frac=per_frac,
        cap_band_pp_best=float(caps[best_frac]), best_split_frac=best_frac,
        cap_band_pp_median=float(np.median(list(caps.values()))),
        kill_bar_pp=0.012,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nCAP_band best over fracs = {caps[best_frac]:.6f} pp (frac {best_frac}) "
          f"vs bar 0.012 pp -> {'PASS' if caps[best_frac] >= 0.012 else 'KILL'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
