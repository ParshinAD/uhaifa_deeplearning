"""A-SUB and A-ALT prototype gates — the researcher's TODO 2 and TODO 3, short version.

Both ideas are already in the backlog with prior evidence; this script tests the two readings
that the prior evidence does NOT already close, on the primary dataset, starting from orders
that are already on disk (so no Rocket run is needed).

────────────────────────────────────────────────────────────────────────────────────────────
A-SUB — "SGD on part of the neurons" (TODO 2)
────────────────────────────────────────────────────────────────────────────────────────────
Two readings, and they have opposite priors.

*Continuous reading* (update a random subset of position coordinates per gradient step) is a
pure DYNAMICS knob, and is NOT tested here — see the write-up. Three independent reasons:
finding #2 (8 mechanisms on the dynamics axis, all null); H13 (stochastic edge subsampling,
connectome −0.84 pp); and the fact that block-coordinate ascent has the SAME critical points as
full-gradient ascent on a smooth objective — it changes the path, not the fixed set, which is
exactly the class of intervention that has never moved this metric.

*Discrete reading* (move a random SUBSET of the sift's movers each sweep) is novel, and it has
an exact relationship to the shipped H35 refiner that makes the comparison well-posed:

    H35 under-relaxation:  key_i = rank_i + alpha * (target_i - rank_i)          (deterministic)
    stochastic subset:     key_i = rank_i + B_i   * (target_i - rank_i),  B_i ~ Bernoulli(p)

so  E[key_i] is IDENTICAL to the under-relaxed key with alpha = p. The displacement fields
agree in expectation; the stochastic one merely adds variance.

⚠ Precise scope of that identity: it holds at the level of the KEY (the displacement field).
The rank vector is `argsort(argsort(key))`, a nonlinear function of the keys, so equal keys in
expectation do NOT imply equal orders in expectation. The identity is what makes `p` and
`alpha` comparable on the same axis, not a proof that the two are the same algorithm.

Both are known cures for the Jacobi period-2 limit cycle that H35 diagnosed (damping and
randomisation are the two textbook fixes). So the ONLY open question is whether the variance
buys anything over the variance-free damping that is already shipped.

  PRE-REGISTERED A-SUB PREDICTIONS
  S1: stochastic p=0.7 lands within noise of deterministic alpha=0.7 (the expectation identity)
  S2: both break the limit cycle (mover count falls, unlike alpha=1.0 Jacobi)
  S3: stochastic does NOT beat deterministic beyond seed noise  => nothing to build

────────────────────────────────────────────────────────────────────────────────────────────
A-ALT — alternate discrete <-> gradient refinement (TODO 3)
────────────────────────────────────────────────────────────────────────────────────────────
H30/H35 run gradient -> sift ONCE. Iterating it is genuinely untested (`dr_tmp/kick_gate.py`
was written but never run — there is no output artifact). Q01 gives a sharp prediction of why
the naive version fails, and yesterday's H38 result gives a reason it might now work.

The Q01 vise: at Rocket's operating point `beta*std ~ 148` the sigmoid surrogate ranks Rocket's
OWN order above a better one, so a gradient phase started from a sifted (better) order flows
AWAY from it. The surrogate only prefers the better order above `beta*std ~ 470` — but Q04
measured that at that scale only ~7% of edges still have a non-zero float32 gradient, so the
gradient phase there does nothing. Drift below the crossover, paralysis above it.

H38's one-sided surrogate escapes the vise on paper: it has NO crossover at any scale and an
alignment ratio of +1.48..+2.00 (vs the sigmoid's −0.591), i.e. it ranks the better order higher
everywhere. If the vise is what kills alternation, swapping the kick's surrogate should show it.

  PRE-REGISTERED A-ALT PREDICTIONS
  A1: sigmoid kick at std=141 drifts the sifted order DOWN substantially
  A2: sigmoid kick at std=1500 (beta*std >> 470) barely moves it (gradient is dead)
  A3: the H38 surrogate does NOT drift the order down at the operating scale
  A4: after re-sifting, no configuration beats the input sifted order by more than seed noise
      (best-by-oracle makes alternation safe, so the realistic failure mode is "no gain")

Leakage: never reads `data/best_solution`. The frozen oracle only scores whole rank vectors,
exactly as the shipped refiners do. Writes to `experiments/outputs/` only, never `results/`.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_asub_aalt.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from mfas.io import load_dataset
from mfas.metrics import pct, score_from_order
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps
from mfas.refine.underrelax import underrelaxed_rebuild

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "experiments" / "outputs"

CONFIG = {
    "dataset": "connectome",
    # Orders already on disk: a pre-sift Rocket order (H02) and a sifted one (H35).
    "presift_positions": "results/20260621T090127Z-H02-connectome-s42-implement-059689_positions.npy",
    "sifted_positions": "results/20260622T213942Z-H35-connectome-s42-implement-8f52fb_positions.npy",
    "asub": {"k_full": 6, "max_sweeps": 22,
             "alphas": [1.0, 0.7], "ps": [0.7, 0.5], "seeds": [42, 123]},
    "aalt": {"scales": [141.0, 500.0, 1500.0], "kick_steps": 150, "lr": 0.05,
             "beta": 1.05, "grad_clip": 1.0, "resift_sweeps": 22, "k_full": 2,
             "alpha": 0.7},
}


# ──────────────────────────────────────────────────────────────────────────────
# A-SUB: stochastic-subset rebuild (the discrete reading)
# ──────────────────────────────────────────────────────────────────────────────
def stochastic_rebuild(rank, best_gap, gain, p, rng, tol=1e-9):
    """Move each mover FULLY with probability ``p``; the rest keep their rank.

    Compare `mfas.refine.underrelax.underrelaxed_rebuild`, which moves EVERY mover a fraction
    ``alpha`` of the way. E[displacement] here equals that one's at ``alpha = p``.
    """
    rank = np.asarray(rank, dtype=np.int64)
    best_gap = np.asarray(best_gap, dtype=np.int64)
    gain = np.asarray(gain, dtype=np.float64)
    key = rank.astype(np.float64).copy()
    movers = np.nonzero(gain > tol)[0]
    chosen = movers[rng.random(movers.shape[0]) < p]
    key[chosen] = best_gap[chosen].astype(np.float64) - 0.5
    return np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)


def run_sift(g, init_rank, *, mode, param, seed, k_full, max_sweeps, tol=1e-9):
    """Unified driver: ``mode='alpha'`` = H35 under-relaxation, ``mode='p'`` = stochastic."""
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    src, tgt, w = build_sift_edges(g)
    n = g.n_nodes
    rng = np.random.default_rng(seed)

    work = np.asarray(init_rank, dtype=np.int64).copy()
    best = work.copy()
    best_score = score_from_order(best, src_o, tgt_o, g.weight)
    movers_log, t0 = [], time.time()
    for s in range(max_sweeps):
        best_gap, gain = jacobi_best_gaps(work, src, tgt, w, n)
        n_movers = int((gain > tol).sum())
        movers_log.append(n_movers)
        if s < k_full:                                   # warm full-Jacobi sweeps (as H35)
            cand = underrelaxed_rebuild(work, best_gap, gain, 1.0, tol=tol)
        elif mode == "alpha":
            cand = underrelaxed_rebuild(work, best_gap, gain, param, tol=tol)
        else:
            cand = stochastic_rebuild(work, best_gap, gain, param, rng, tol=tol)
        cand_score = score_from_order(cand, src_o, tgt_o, g.weight)
        work = cand
        if cand_score > best_score:
            best_score, best = cand_score, cand.copy()
        if n_movers == 0:
            break
    return {"best_pct": pct(best_score, g.total_weight), "movers": movers_log,
            "final_movers": movers_log[-1], "wall_s": time.time() - t0,
            "best_rank": best}


# ──────────────────────────────────────────────────────────────────────────────
# A-ALT: gradient kick from a sifted order, with two surrogates
# ──────────────────────────────────────────────────────────────────────────────
def _sigmoid_surr(z):
    return torch.sigmoid(z)


def _h38_surr(z, M=0.75, T=1.5):
    """H38's one-sided shape (redefined locally so this diagnostic imports no variant)."""
    return torch.where(z >= M, torch.ones_like(z), 1.0 + torch.tanh((z - M) / T))


def gradient_kick(g, rank, *, surrogate, scale, steps, lr, beta, grad_clip, device):
    """Embed ``rank`` evenly at the requested std, ascend the surrogate, return the new order."""
    n = g.n_nodes
    p = (rank.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0
    p = p * (scale / p.std())
    pos = torch.nn.Parameter(torch.tensor(p, dtype=torch.float32, device=device))
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w = np.asarray(g.weight)
    nw = torch.tensor((w / float(w.max())).astype(np.float32), device=device)
    opt = torch.optim.Adam([pos], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        d = pos[tgt_t] - pos[src_t]
        (-(surrogate(beta * d) * nw).sum()).backward()
        torch.nn.utils.clip_grad_norm_([pos], grad_clip)
        opt.step()
    out = pos.detach().cpu().numpy()
    return np.argsort(np.argsort(out, kind="stable"), kind="stable").astype(np.int64)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def main() -> None:
    g = load_dataset(CONFIG["dataset"])
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    report = {"config": CONFIG, "git_commit": git_commit()}

    def order_of(path):
        p = np.load(_ROOT / path)
        return np.argsort(np.argsort(p, kind="stable"), kind="stable").astype(np.int64)

    presift = order_of(CONFIG["presift_positions"])
    sifted = order_of(CONFIG["sifted_positions"])
    pre_pct = pct(score_from_order(presift, src_o, tgt_o, g.weight), g.total_weight)
    sift_pct = pct(score_from_order(sifted, src_o, tgt_o, g.weight), g.total_weight)
    report["inputs"] = {"presift_pct": pre_pct, "sifted_pct": sift_pct}
    print(f"inputs: pre-sift Rocket order {pre_pct:.4f}% | H35 sifted order {sift_pct:.4f}%")

    # ── A-SUB ────────────────────────────────────────────────────────────────
    c = CONFIG["asub"]
    print(f"\n=== A-SUB: stochastic subset vs deterministic under-relaxation "
          f"(from the {pre_pct:.4f}% order, {c['max_sweeps']} sweeps) ===")
    print(f"{'variant':>26} {'best pct':>10} {'final movers':>13} {'wall':>7}")
    asub = {}
    for a in c["alphas"]:
        r = run_sift(g, presift, mode="alpha", param=a, seed=0,
                     k_full=c["k_full"], max_sweeps=c["max_sweeps"])
        asub[f"alpha_{a}"] = {k: v for k, v in r.items() if k != "best_rank"}
        print(f"{f'deterministic alpha={a}':>26} {r['best_pct']:10.4f} "
              f"{r['final_movers']:13d} {r['wall_s']:6.1f}s")
    for p_ in c["ps"]:
        rows = []
        for sd in c["seeds"]:
            r = run_sift(g, presift, mode="p", param=p_, seed=sd,
                         k_full=c["k_full"], max_sweeps=c["max_sweeps"])
            rows.append(r)
            print(f"{f'stochastic p={p_} s{sd}':>26} {r['best_pct']:10.4f} "
                  f"{r['final_movers']:13d} {r['wall_s']:6.1f}s")
        asub[f"p_{p_}"] = {"per_seed_pct": [x["best_pct"] for x in rows],
                           "mean_pct": float(np.mean([x["best_pct"] for x in rows])),
                           "std_pct": float(np.std([x["best_pct"] for x in rows], ddof=1)),
                           "final_movers": [x["final_movers"] for x in rows],
                           "movers": rows[0]["movers"]}
    report["asub"] = asub

    # ── A-ALT ────────────────────────────────────────────────────────────────
    k = CONFIG["aalt"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== A-ALT: gradient kick from the {sift_pct:.4f}% sifted order "
          f"({k['kick_steps']} Adam steps, device={device.type}) ===")
    print(f"{'surrogate':>12} {'std':>7} {'beta*std':>9} {'after kick':>11} "
          f"{'drift':>9}")
    aalt = {}
    for sname, surr in (("sigmoid", _sigmoid_surr), ("H38_asym", _h38_surr)):
        for scale in k["scales"]:
            rk = gradient_kick(g, sifted, surrogate=surr, scale=scale,
                               steps=k["kick_steps"], lr=k["lr"], beta=k["beta"],
                               grad_clip=k["grad_clip"], device=device)
            kp = pct(score_from_order(rk, src_o, tgt_o, g.weight), g.total_weight)
            aalt[f"{sname}_std{scale}"] = {"after_kick_pct": kp,
                                           "drift_pp": kp - sift_pct,
                                           "beta_std": k["beta"] * scale,
                                           "kicked_rank": rk}
            print(f"{sname:>12} {scale:7.0f} {k['beta'] * scale:9.1f} {kp:11.4f} "
                  f"{kp - sift_pct:+9.4f}")

    # Re-sift only the configurations that did not destroy the order (triage: a config that
    # drifts far below the input cannot beat it after a re-sift that starts from there).
    print(f"\n  re-sift (alpha={k['alpha']}, {k['resift_sweeps']} sweeps) of the "
          f"least-damaged configs; net vs the {sift_pct:.4f}% input:")
    ranked = sorted(aalt, key=lambda kk: -aalt[kk]["after_kick_pct"])[:3]
    for name in ranked:
        r = run_sift(g, aalt[name]["kicked_rank"], mode="alpha", param=k["alpha"], seed=0,
                     k_full=k["k_full"], max_sweeps=k["resift_sweeps"])
        aalt[name]["resift_pct"] = r["best_pct"]
        aalt[name]["net_vs_input_pp"] = r["best_pct"] - sift_pct
        print(f"    {name:>22}: {r['best_pct']:.4f}%  net {r['best_pct'] - sift_pct:+.4f} pp")
    for v in aalt.values():
        v.pop("kicked_rank", None)
    report["aalt"] = aalt

    # ── pre-registered checks ────────────────────────────────────────────────
    det = asub["alpha_0.7"]["best_pct"]
    sto = asub["p_0.7"]["mean_pct"]
    checks = {
        "S1_stochastic_matches_deterministic_within_0.01pp": bool(abs(sto - det) < 0.01),
        "S2_both_break_the_limit_cycle": bool(
            asub["alpha_0.7"]["final_movers"] < asub["alpha_1.0"]["final_movers"]
            and asub["p_0.7"]["final_movers"][0] < asub["alpha_1.0"]["final_movers"]),
        "S3_stochastic_does_not_beat_deterministic": bool(sto <= det + 0.01),
        "A1_sigmoid_drifts_down_at_operating_scale": bool(
            aalt["sigmoid_std141.0"]["drift_pp"] < -0.05),
        "A2_sigmoid_frozen_at_large_scale": bool(
            abs(aalt["sigmoid_std1500.0"]["drift_pp"]) < 0.05),
        "A3_h38_does_not_drift_down_at_operating_scale": bool(
            aalt["H38_asym_std141.0"]["drift_pp"] > -0.05),
        "A4_no_config_beats_the_input_after_resift": bool(all(
            v.get("net_vs_input_pp", -9) <= 0.01 for v in aalt.values())),
    }
    report["prediction_check"] = checks
    print("\nPRE-REGISTERED CHECKS:")
    for kk, vv in checks.items():
        print(f"  {kk:>52}: {'HOLDS' if vv else 'FAILS'}")

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "proto_asub_aalt.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
