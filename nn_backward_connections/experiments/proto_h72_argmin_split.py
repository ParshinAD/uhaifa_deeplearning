"""H72 rung 2 - does a weight-aware cut actually buy anything in the alternation?

Rung 1 (experiments/outputs/proto_H72_rung1_connectome.json) established the PREMISE: the
production splitter's cut is not the cheapest cut available inside a balanced band, and the
weight it hides is large in absolute terms. Capacity is not achievability (M11), so this rung
measures the composed effect: two arms of the champion's own stage-4 alternation, identical in
every constant, differing ONLY in where SccRecursiveRefiner._split cuts a single-SCC block.

    control   mid = lo + round(nb * split_frac)                 (production, weight-blind)
    variant   mid = argmin_m X(m) over |m - m_fixed| <= band*nb  (weight-aware, still balanced)

X(m) is the backward weight crossing cut m, computed for every m at once in O(|eidx| + nb) -
the same asymptotic cost as the two boolean masks _split already builds. The band keeps the
round-to-round boundary diversity the cycled split_fracs exist for, and it excludes the
degenerate global argmin, which rung 1 measured to be a maximally unbalanced cut (CAP_glob is
99.9% of X_fixed, i.e. min X over ALL cuts is ~0 because a cut one position from the end
crosses almost nothing).

M12 APPLIES AND IS THE REASON THIS RUNG CAN ONLY KILL. Both arms start from the champion's own
converged order, where the control is at its fixed point and the variant is not, so the
measured incremental is biased UPWARD - 3.76x on connectome for H60, the closest comparable
mechanism (also a change to the block refiner). The control arm's own remaining headroom is
reported beside every number so the bias is visible, per M12's operative consequence.

PRE-REGISTERED KILL CONDITION (rung 2), written before the run:
  KILL if (variant - control) < 0.012 pp on connectome at N = 20 matched cycles, for every band
  tested. A from-champion incremental below the screen bar cannot survive a screen that the
  same measurement overstates.
  Report whatever the verdict: both arms' absolute deltas, the control's headroom, the
  per-cycle trajectory, the realised wall clock of each arm (M8: a class must also clear the
  marginal rate of the stage it displaces - stage 4 is 3.45e-4 pp/s on connectome), and the
  realised reduction in crossing weight so premise and effect can be told apart.

NOTE (P26): numpy's BLAS is broken in this env (np.dot aborts the interpreter). Nothing here
uses a matrix product.
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
from mfas.refine.scc_recursive import (SccRecursiveRefiner, DEFAULT_SPLIT_FRACS,
                                       scc_recursive_refine)
from mfas.refine import sift_underrelaxed


class ArgminSplitRefiner(SccRecursiveRefiner):
    """The production refiner with a weight-aware cut inside a balanced band."""

    def __init__(self, src, tgt, n, weight, *, band_frac, **kw):
        super().__init__(src, tgt, n, **kw)
        self.w = np.asarray(weight)
        self.band_frac = float(band_frac)
        self.x_fixed_sum = 0.0
        self.x_used_sum = 0.0
        self.n_splits = 0

    def _split(self, lo, hi, eidx):
        nb = hi - lo
        ps = self.pos[self.src[eidx]]
        pt = self.pos[self.tgt[eidx]]
        w = self.w[eidx]
        back = pt < ps

        acc = np.float64 if self.w.dtype.kind == "f" else np.int64
        diff = np.zeros(nb + 2, dtype=acc)
        if bool(back.any()):
            a = (pt[back] - lo) + 1
            b = (ps[back] - lo)
            np.add.at(diff, a, w[back].astype(acc))
            np.add.at(diff, b + 1, -w[back].astype(acc))
        xcurve = np.cumsum(diff[:nb + 1])

        m_fixed = max(1, min(nb - 1, int(round(nb * self.split_frac))))
        half = max(1, int(round(nb * self.band_frac)))
        b_lo = max(1, m_fixed - half)
        b_hi = min(nb - 1, m_fixed + half)
        seg = xcurve[b_lo:b_hi + 1]
        m_used = b_lo + int(np.argmin(seg))

        self.n_splits += 1
        self.x_fixed_sum += float(xcurve[m_fixed])
        self.x_used_sum += float(xcurve[m_used])

        mid = lo + m_used
        left = (ps < mid) & (pt < mid)
        right = (ps >= mid) & (pt >= mid)
        self._refine(lo, mid, eidx[left], is_scc=False)
        self._refine(mid, hi, eidx[right], is_scc=False)


def argmin_refine(g, rank, *, min_block, split_frac, band_frac):
    """One monotone pass of the refiner with the weight-aware cut. Returns (rank, stats)."""
    ref = ArgminSplitRefiner(np.asarray(g.src), np.asarray(g.tgt), g.n_nodes, g.weight,
                             band_frac=band_frac, min_block=min_block,
                             split_frac=split_frac)
    out = ref.run(np.asarray(rank, dtype=np.int64))
    return out, dict(n_splits=ref.n_splits, x_fixed=ref.x_fixed_sum, x_used=ref.x_used_sum)


def alternate(g, init_rank, *, n_cycles, band_frac, sift_sweeps=2, k_full=2, alpha=0.7,
              min_block=32, split_fracs=DEFAULT_SPLIT_FRACS):
    """The champion's stage-4 alternation, with band_frac=None meaning the production cut.

    Mirrors mfas.refine.scc_recursive.alternate_scc_sift constant for constant; the only
    difference is which splitter the block refiner uses.
    """
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    pp = 100.0 / float(total)
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(rank, src_o, tgt_o, g.weight)
    best_rank = rank.copy()
    log = []
    x_fixed_sum = x_used_sum = 0.0
    t0 = time.time()
    for c in range(n_cycles):
        sf = split_fracs[c % len(split_fracs)]
        if band_frac is None:
            rank = scc_recursive_refine(g, rank, min_block=min_block, split_frac=sf)
        else:
            rank, st = argmin_refine(g, rank, min_block=min_block, split_frac=sf,
                                     band_frac=band_frac)
            x_fixed_sum += st["x_fixed"]
            x_used_sum += st["x_used"]
        s_scc = score_from_order(rank, src_o, tgt_o, g.weight)
        if s_scc > best_score:
            best_score, best_rank = s_scc, rank.copy()
        rank, s_sift, _ = sift_underrelaxed(g, rank, k_full=k_full, alpha=alpha,
                                            max_sweeps=sift_sweeps)
        if s_sift > best_score:
            best_score, best_rank = s_sift, rank.copy()
        log.append(dict(cycle=c, split_frac=sf, after_scc_pct=float(s_scc * pp),
                        after_sift_pct=float(s_sift * pp), best_pct=float(best_score * pp),
                        cum_wall_s=time.time() - t0))
        print(f"    cycle {c:3d} frac={sf} scc={s_scc * pp:.8f} sift={s_sift * pp:.8f} "
              f"best={best_score * pp:.8f} ({time.time() - t0:.0f} s)", flush=True)
    return best_rank, float(best_score), log, dict(x_fixed=x_fixed_sum, x_used=x_used_sum,
                                                   wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--order", required=True)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--bands", default="0.10,0.25",
                    help="comma-separated band half-widths; the control arm is always run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    g = load_dataset(args.dataset)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    pp = 100.0 / float(g.total_weight)
    rank0 = np.load(args.order).astype(np.int64)
    base = float(score_from_order(rank0, src, tgt, g.weight) * pp)
    print(f"{args.dataset}: base {base:.8f} pct, {args.cycles} matched cycles", flush=True)

    arms = {}
    print("  ARM control (production weight-blind cut)", flush=True)
    _, s_ctl, log_ctl, st_ctl = alternate(g, rank0, n_cycles=args.cycles, band_frac=None)
    arms["control"] = dict(final_pct=float(s_ctl * pp), delta_pp=float(s_ctl * pp) - base,
                           wall_s=st_ctl["wall_s"], log=log_ctl)
    print(f"  control: {s_ctl * pp:.8f} pct ({float(s_ctl * pp) - base:+.6f} pp) "
          f"in {st_ctl['wall_s']:.0f} s", flush=True)

    for b in [float(x) for x in args.bands.split(",")]:
        print(f"  ARM band={b}", flush=True)
        _, s_v, log_v, st_v = alternate(g, rank0, n_cycles=args.cycles, band_frac=b)
        d = float(s_v * pp) - base
        arms[f"band_{b}"] = dict(
            final_pct=float(s_v * pp), delta_pp=d,
            incremental_pp=d - arms["control"]["delta_pp"],
            wall_s=st_v["wall_s"], log=log_v,
            x_fixed_pp=st_v["x_fixed"] * pp, x_used_pp=st_v["x_used"] * pp,
            crossing_weight_reduction_pp=(st_v["x_fixed"] - st_v["x_used"]) * pp)
        print(f"  band={b}: {s_v * pp:.8f} pct ({d:+.6f} pp), incremental "
              f"{d - arms['control']['delta_pp']:+.8f} pp, {st_v['wall_s']:.0f} s", flush=True)

    best_inc = max(v.get("incremental_pp", -9e9) for v in arms.values())
    payload = dict(item="H72", rung=2, dataset=args.dataset, order=args.order,
                   cycles=args.cycles, base_pct=base,
                   control_headroom_pp=arms["control"]["delta_pp"],
                   arms=arms, best_incremental_pp=best_inc, kill_bar_pp=0.012,
                   verdict="PASS" if best_inc >= 0.012 else "KILL")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nbest incremental {best_inc:+.8f} pp vs bar 0.012 pp -> {payload['verdict']}")
    print(f"control arm's own headroom over {args.cycles} cycles: "
          f"{arms['control']['delta_pp']:+.6f} pp (M12: the bias is visible here)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
