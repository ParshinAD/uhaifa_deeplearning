"""Prototype gate for H66 — the ASYM gradient KICK as a TERMINAL stage.

The hypothesis
--------------
Append a stage to the champion pipeline that (1) embeds the champion's final ranks as
continuous positions at a scale chosen so the ASYMMETRIC surrogate's *live* gradient
support spans a target rank REACH, (2) takes K asymmetric gradient steps at a FROZEN
beta, (3) re-ranks, (4) runs one SHORT under-relaxed exact-gain sift, and (5) keeps the
result only if the frozen oracle improves. Does that gain more than +0.012 pp on
connectome over the champion?

Why a from-champion prototype is admissible evidence here (M12)
---------------------------------------------------------------
``killed.json`` meta-rule **M12** says a composed prototype started from the champion's
own converged order systematically OVERSTATES a pipeline change — measured at 3.76x on
H60. Its 2026-08-26 amendment narrows that: the bias needs the variant to diverge from
the control somewhere UPSTREAM, so it applies to a change to an INNER stage and **not**
to a pure TERMINAL APPEND, where the variant arm's input IS the control arm's output.
H59 measured the boundary: prototype +0.017596 pp vs screen +0.017658 pp, 0.35% relative.

H66 is a pure terminal append: stages 1-4 are byte-identical to the champion and this
stage runs after them. So this prototype is a genuine PREDICTION of the screen, not a
screening number. That is the whole reason the item is cheap to decide.

The control arm is mandatory and is reported beside every number (M12's last sentence):
``champion order -> the SAME short sift, no kick``. It states how much of any measured
gain is simply "the champion's own stage 4 had headroom left".

The reach parameterisation (this is the one real design decision)
-----------------------------------------------------------------
``H38._asym_surrogate`` is ``g(z) = 1`` for ``z >= M`` and ``1 + tanh((z-M)/T)`` below,
with ``(M, T) = (0.75, 1.5)``. Its derivative is ``sech^2((z-M)/T)/T`` below the margin
and exactly 0 above it, so an edge receives essentially no gradient once
``(z - M)/T < -Z_SAT``; at ``Z_SAT = 4`` the derivative is 0.13% of its peak. That is the
same saturation threshold ``H64.py``'s docstring uses when it reports ASYM's live support
as 13.10% of backward weight at a reach of ~1,799 ranks.

For a BACKWARD edge spanning ``r`` ranks, ``Delta = -r*s`` where ``s`` is the position
spacing per rank, so ``z = -beta*r*s`` and the edge is live iff

    beta * r * s  <  M + Z_SAT*T  =  5.25          =>   reach R = 5.25 / (beta * s)

Inverting, a target reach ``R`` fixes the embedding scale ``s = 5.25 / (beta * R)``.
Beta is held CONSTANT at ``BETA_KICK`` for the whole kick — the scale never drifts. That
is not a convenience: ``killed.json`` H04's revival condition is *"the in-loop move is the
EXACT-GAIN sift (not barycenter) AND the gradient phase is prevented from drifting
(frozen scale / alternation)"*, and a frozen (beta, s) pair is what discharges it. It also
makes ``R`` mean what it says throughout the kick, which a cyclic beta schedule would not.

Leakage-safety
--------------
The surrogate is a function of positions and beta only, weighted by ``w/max(w)``. The
frozen oracle is read ONLY to (a) track best-by-oracle inside the kick, exactly as the
baseline's own ``run_rocket`` does, and (b) apply the keep-if-improves guard, which is a
max against the input score. ``data/best_solution`` is never touched, no dataset is
special-cased, and no threshold is fitted to the connectome.

Reproduce
---------
    PYTHONPATH=src python experiments/proto_H66_kick.py --dataset connectome
    PYTHONPATH=src python experiments/proto_H66_kick.py --dataset connectome --rounds 3 \
        --reach 20000 --steps 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from mfas.experiments.H38 import M as ASYM_M, T as ASYM_T, _asym_surrogate
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_order
from mfas.refine.underrelax import sift_underrelaxed
from mfas.utils.seeding import select_device

# ── The champion order this stage is appended to (sota.json, 2026-08-27) ──────────────
CHAMPION = {
    "connectome": {
        "variant": "H64",
        "pct": 84.25817950936937,
        "positions": "results/20260827T024801Z-H64-connectome-s42-confirm-a7490c_positions.npy",
    },
    "microns": {
        "variant": "H42",
        "pct": 83.24085291200831,
        "positions": None,   # filled on demand; microns is not the target of this item
    },
}

# ── Constants ────────────────────────────────────────────────────────────────────────
Z_SAT = 4.0                 # (z-M)/T below which tanh' is <0.2% of peak -> "not live"
LIVE_HALFWIDTH = ASYM_M + Z_SAT * ASYM_T      # = 5.25 for (M,T) = (0.75, 1.5)
BETA_KICK = 1.0             # FROZEN. See the docstring: H04's revival condition.
LR = 0.05                   # RocketConfig.lr, unchanged
GRAD_CLIP = 1.0             # RocketConfig.grad_clip, unchanged

# H42/H64 stage-4 SHORT-sift constants (src/mfas/experiments/H64.py:120-124). The kick's
# repair sift is matched to them constant-for-constant so the comparison isolates the kick.
SIFT_SWEEPS = 2
SIFT_K_FULL = 2
SIFT_ALPHA = 0.7

DEFAULT_REACHES = (2_000, 20_000, 60_000)
DEFAULT_STEPS = (50, 200, 800)


def _load_champion_rank(g, dataset: str) -> Tuple[np.ndarray, float]:
    """Load the champion's stored positions, convert to a rank vector, and ANCHOR it.

    Aborts unless the rank vector re-scores to the champion's recorded pct. Without that
    check the whole prototype could be measuring a delta against the wrong order.
    """
    spec = CHAMPION[dataset]
    pos = np.load(spec["positions"])
    rank = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
    if not np.array_equal(np.sort(rank), np.arange(g.n_nodes)):
        raise SystemExit("champion rank vector is not a permutation")
    s = score_from_order(rank, np.asarray(g.src, dtype=np.int64),
                         np.asarray(g.tgt, dtype=np.int64), g.weight)
    got = pct(s, g.total_weight)
    if abs(got - spec["pct"]) > 1e-9:
        raise SystemExit(
            f"ANCHOR FAILED: champion positions re-score to {got!r}, "
            f"sota.json records {spec['pct']!r}. Refusing to measure against it.")
    return rank, float(s)


def _live_support(g, rank: np.ndarray, reach: int) -> Dict[str, float]:
    """Fraction of BACKWARD weight whose span is inside the live band at this reach.

    Comparable to ``H64.py``'s reported 13.10% (ASYM) / 15.66% (sigmoid) figures.
    """
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    span = rank[tgt].astype(np.int64) - rank[src].astype(np.int64)
    back = span < 0
    w_back = float(w[back].sum())
    live = back & (-span < reach)
    return dict(reach=int(reach),
                backward_weight=w_back,
                live_weight=float(w[live].sum()),
                live_frac_of_backward=float(w[live].sum() / w_back) if w_back else 0.0,
                n_backward=int(back.sum()), n_live=int(live.sum()))


def _kick(g, rank: np.ndarray, *, reach: int, steps: int, device,
          log_every: int, base_score: float,
          lr_ranks: Optional[float] = None) -> Tuple[np.ndarray, float, List[Dict]]:
    """K asymmetric gradient steps at a FROZEN (beta, scale); best-by-oracle over the run.

    Returns ``(best_rank, best_score, log)``. ``best_score >= base_score`` always: the
    input order is the initial incumbent, so the kick alone can never regress.

    ``lr_ranks`` — RUNG 2, and it exists because rung 1 had a confound. Adam's update is
    ``m/(sqrt(v)+eps)``, whose magnitude is ~``lr`` for ANY node with a consistent
    gradient, however small. So at ``lr = 0.05`` (the champion's value) every one of the
    136,648 nodes translates by ``lr/s`` ranks per step — 19 ranks at reach 2,000 — whether
    or not the surrogate has anything to say about it. That is a step-size artefact, not a
    statement about the objective. Passing ``lr_ranks = d`` sets ``lr = d * s`` so the
    per-step displacement is ``d`` RANKS by construction, making the probe local.
    """
    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight, dtype=np.float64)
    nw_t = torch.tensor((w_np / w_np.max()).astype(np.float32), device=device)
    src_o = np.asarray(g.src, dtype=np.int64)
    tgt_o = np.asarray(g.tgt, dtype=np.int64)

    # s = 5.25 / (beta * R): the scale that puts the live band exactly `reach` ranks wide.
    s = LIVE_HALFWIDTH / (BETA_KICK * float(reach))
    p0 = (rank.astype(np.float64) - (n - 1) / 2.0) * s
    positions = torch.nn.Parameter(torch.tensor(p0.astype(np.float32), device=device))
    lr = LR if lr_ranks is None else float(lr_ranks) * s
    optimizer = optim.Adam([positions], lr=lr)

    best_score, best_rank = float(base_score), rank.copy()
    fin_rank, fin_score = rank.copy(), float(base_score)
    log: List[Dict] = []
    t0 = time.time()

    def _rerank_and_score() -> Tuple[np.ndarray, float]:
        p = positions.detach().cpu().numpy()
        r = np.argsort(np.argsort(p, kind="stable"), kind="stable").astype(np.int64)
        return r, float(score_from_order(r, src_o, tgt_o, g.weight))

    for i in range(steps):
        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        sig = _asym_surrogate(BETA_KICK * delta)
        loss = -(sig * nw_t).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], GRAD_CLIP)
        optimizer.step()

        if (i + 1) % log_every == 0 or i == steps - 1:
            r, sc = _rerank_and_score()
            fin_rank, fin_score = r, sc
            if sc > best_score:
                best_score, best_rank = sc, r
            log.append(dict(step=i + 1, lr=lr, pct=pct(sc, g.total_weight),
                            best_pct=pct(best_score, g.total_weight),
                            neg_loss=float(-loss.item()),
                            pos_std=float(positions.detach().std().item()),
                            elapsed_s=time.time() - t0))
    return best_rank, best_score, fin_rank, fin_score, log


def _short_sift(g, rank: np.ndarray, sweeps: int = SIFT_SWEEPS
                ) -> Tuple[np.ndarray, float, float]:
    """The champion's own stage-4 SHORT sift, constant-for-constant.

    ``sweeps`` is a knob because the REPAIR half of perturb-and-repair may need more
    sweeps than the champion's stage-4 short sift spends. The 2-sweep setting is the
    production-matched one; anything larger is extra compute and must be declared as such.
    """
    t0 = time.time()
    r, s, _ = sift_underrelaxed(g, rank, k_full=SIFT_K_FULL, alpha=SIFT_ALPHA,
                                max_sweeps=sweeps)
    return r, float(s), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--reach", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--base-positions", default=None,
                    help="POSITIVE CONTROL: run the same grid from an arbitrary stored "
                         "order instead of the champion's, to test whether the kick's "
                         "value is BASE-DEPENDENT. Deltas are then against that order.")
    ap.add_argument("--base-label", default=None)
    ap.add_argument("--sift-sweeps", type=int, nargs="*", default=None,
                    help="repair-sift sweep counts to grid over (default: the "
                         "production-matched 2)")
    ap.add_argument("--lr-ranks", type=float, nargs="*", default=None,
                    help="RUNG 2: per-step Adam displacement in RANKS (sets lr = d*s). "
                         "Omit to use the champion's lr=0.05 unchanged (rung 1).")
    ap.add_argument("--rounds", type=int, default=1,
                    help="alternate kick<->sift this many times per arm")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device, dev_name = select_device("auto")
    g = load_dataset(args.dataset)
    total = g.total_weight
    if args.base_positions:
        # Positive control: an ARBITRARY stored order. No sota anchor applies; the base
        # is whatever it scores, and every delta below is against that number.
        pos = np.load(args.base_positions)
        rank0 = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
        score0 = float(score_from_order(rank0, np.asarray(g.src, dtype=np.int64),
                                        np.asarray(g.tgt, dtype=np.int64), g.weight))
        base_label = args.base_label or Path(args.base_positions).name
    else:
        rank0, score0 = _load_champion_rank(g, args.dataset)
        base_label = CHAMPION[args.dataset]["variant"]
    pct0 = pct(score0, total)
    print(f"[anchor] {base_label} {args.dataset} = {pct0!r}  device={dev_name}", flush=True)

    # ── CONTROL (M12): champion order -> the same short sift, NO kick ────────────────
    t_ctl = time.time()
    _, ctl_score, ctl_wall = _short_sift(g, rank0)
    ctl_score = max(ctl_score, score0)          # keep-if-improves, same guard as the arms
    ctl_pct = pct(ctl_score, total)
    print(f"[control] champion + short sift, no kick = {ctl_pct!r} "
          f"({ctl_pct - pct0:+.6f} pp headroom, {ctl_wall:.1f} s)", flush=True)

    reaches = args.reach or list(DEFAULT_REACHES)
    steps_grid = args.steps or list(DEFAULT_STEPS)

    support = {str(R): _live_support(g, rank0, R) for R in reaches}
    for R, sup in support.items():
        print(f"[support] reach={R:>6}  live={sup['live_frac_of_backward']*100:.2f}% "
              f"of backward weight ({sup['n_live']} edges)", flush=True)

    arms: List[Dict] = []
    lr_grid = args.lr_ranks if args.lr_ranks else [None]
    sweep_grid = args.sift_sweeps or [SIFT_SWEEPS]
    for R in reaches:
      for D in lr_grid:
       for SW in sweep_grid:
        for K in steps_grid:
            t0 = time.time()
            rank, score = rank0.copy(), score0
            per_round = []
            for rd in range(args.rounds):
                k_rank, k_score, f_rank, f_score, klog = _kick(
                    g, rank, reach=R, steps=K, device=device,
                    log_every=args.log_every, base_score=score, lr_ranks=D)
                # PERTURB-AND-REPAIR: the repair sift starts from the PERTURBED order.
                # Sifting from the best-by-oracle order instead would reproduce the
                # control by construction, since the kick never improves on its own.
                s_rank, s_score, s_wall = _short_sift(g, f_rank, sweeps=SW)
                cand = [(k_score, k_rank), (s_score, s_rank), (score, rank)]
                score, rank = max(cand, key=lambda x: x[0])
                per_round.append(dict(round=rd,
                                      kick_best_pct=pct(k_score, total),
                                      kick_final_pct=pct(f_score, total),
                                      repair_pct=pct(s_score, total),
                                      kept_pct=pct(score, total),
                                      kick_log=klog, sift_wall_s=s_wall))
            # keep-if-improves against the champion itself
            final_score = max(score, score0)
            arm = dict(reach=R, steps=K, lr_ranks=D, sift_sweeps=SW, rounds=args.rounds,
                       final_pct=pct(final_score, total),
                       delta_vs_champion_pp=pct(final_score, total) - pct0,
                       delta_vs_control_pp=pct(final_score, total) - ctl_pct,
                       wall_s=time.time() - t0, rounds_detail=per_round)
            arms.append(arm)
            print(f"[arm] reach={R:>6} steps={K:>4} lr={D} sw={SW} "
                  f"kick_final={per_round[-1]['kick_final_pct']:.6f} "
                  f"repair={per_round[-1]['repair_pct']:.6f} -> "
                  f"{arm['final_pct']!r}  vs champion {arm['delta_vs_champion_pp']:+.6f} pp"
                  f"  vs control {arm['delta_vs_control_pp']:+.6f} pp"
                  f"  ({arm['wall_s']:.1f} s)", flush=True)

    best = max(arms, key=lambda a: a["final_pct"])
    bar = 0.012 if args.dataset == "connectome" else 0.002
    verdict = "PASS" if best["delta_vs_champion_pp"] > bar else "FAIL"

    out = dict(
        item="H66", dataset=args.dataset, device=dev_name,
        champion=base_label, champion_pct=pct0,
        base_positions=args.base_positions,
        control_pct=ctl_pct, control_headroom_pp=ctl_pct - pct0, control_wall_s=ctl_wall,
        surrogate=dict(M=ASYM_M, T=ASYM_T, z_sat=Z_SAT,
                       live_halfwidth=LIVE_HALFWIDTH, beta_kick=BETA_KICK,
                       lr=LR, grad_clip=GRAD_CLIP),
        sift=dict(sweeps=SIFT_SWEEPS, k_full=SIFT_K_FULL, alpha=SIFT_ALPHA),
        live_support=support, arms=arms,
        best_arm=dict(reach=best["reach"], steps=best["steps"], lr_ranks=best["lr_ranks"],
                      sift_sweeps=best["sift_sweeps"],
                      final_pct=best["final_pct"],
                      delta_vs_champion_pp=best["delta_vs_champion_pp"],
                      delta_vs_control_pp=best["delta_vs_control_pp"]),
        bar_pp=bar, verdict=verdict,
        total_wall_s=time.time() - t_ctl,
    )
    dest = Path(args.out or f"experiments/outputs/proto_H66_{args.dataset}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{verdict}] best = reach {best['reach']} / {best['steps']} steps -> "
          f"{best['delta_vs_champion_pp']:+.6f} pp vs champion "
          f"(bar {bar}) ; {best['delta_vs_control_pp']:+.6f} pp vs control")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
