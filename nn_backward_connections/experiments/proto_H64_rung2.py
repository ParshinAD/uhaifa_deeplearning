"""H64 rung 2 - the PROXY prototype, pre-registered in experiments/prereg/H64_prototype.md.

Six arms per proxy isolate the two defects in H38's original measurement:
    A0/A1 random init, pure      -> reproduces H38's own setting on this device
    B0/B1 greedy-FAS init, pure  -> defect (b), the BASIN
    C0/C1 greedy-FAS init, full  -> defect (c), COMPOSITION with stages 3+4
0 = sigmoid (control), 1 = ASYM.
"""
import json, sys, time
import numpy as np, torch
import torch.optim as optim
sys.path.insert(0, "src")

from mfas.baseline.rocket import RocketConfig, make_beta_schedule
from mfas.metrics import pct, score_from_positions
from mfas.refine import alternate_scc_sift, sift_underrelaxed
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order
from mfas.experiments.H38 import _asym_surrogate
from mfas.utils.seeding import select_device
from mfas.io import load_mouse
from mfas.analysis.gap import make_hard_synthetic_graph

DEV, _DEVNAME = select_device("auto")
print("device:", DEV, flush=True)

_EPOCHS = {"connectome": 20000, "mouse": 5000, "microns": 80000, "hard_synthetic": 20000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12, "hard_synthetic": 40}
_K_FULL, _ALPHA = 6, 0.7
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32, "hard_synthetic": 32}
_ALT_SIFT_SWEEPS = 2
_ALT_K_FULL, _MIN_BLOCK = 2, 32


def rocket_with_surrogate(g, cfg, seed, device, surrogate, init_positions=None):
    """run_rocket's loop verbatim, with the surrogate and the init as parameters."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight); max_w = float(w_np.max())
    nw_t = torch.tensor((w_np / max_w).astype(np.float32), device=device)
    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight

    def dscore(p):
        return score_from_positions(p.detach().cpu().numpy(), src_np, tgt_np, g.weight)

    if init_positions is not None:
        pos_data = init_positions.clone().detach().to(torch.float32)
    else:
        pos_data = torch.randn(n, device=device)
    positions = torch.nn.Parameter(pos_data)

    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    s1 = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    s2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    sched = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2],
                                            milestones=[milestone])
    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    best_score = dscore(positions); best_positions = positions.detach().clone()
    t0 = time.time()
    for i in range(cfg.epochs):
        beta = float(betas[i])
        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        sig = surrogate(beta * delta)
        loss = -(sig * nw_t).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step(); sched.step()
        if i % cfg.log_interval == 0 or i == cfg.epochs - 1:
            sc = dscore(positions)
            if sc > best_score:
                best_score = sc; best_positions = positions.detach().clone()
    return (best_positions.detach().cpu().numpy(), best_score, pct(best_score, total),
            time.time() - t0)


def full_pipeline(g, seed, device, surrogate):
    """H42's stages 1-4 with a parameterised surrogate, greedy-FAS warm start."""
    cfg = RocketConfig(epochs=_EPOCHS[g.name])
    init = _init_positions_from_order(greedy_fas_order(g), device)
    pos, pure_score, pure_pct, wall = rocket_with_surrogate(
        g, cfg, seed, device, surrogate, init_positions=init)
    total = g.total_weight
    rank0 = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
    sift_rank, sift_score, _ = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA, max_sweeps=_MAX_SWEEPS[g.name])
    alt_rank, alt_score, _ = alternate_scc_sift(
        g, sift_rank, n_cycles=_ALT_CYCLES[g.name], sift_sweeps=_ALT_SIFT_SWEEPS,
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK)
    best = max(pure_score, sift_score, alt_score)
    return {"pure_pct": pure_pct, "sift_pct": pct(sift_score, total),
            "alt_pct": pct(alt_score, total), "final_pct": pct(best, total),
            "wall_rocket_s": wall}


SHAPES = {"sigmoid": torch.sigmoid, "asym": _asym_surrogate}
out = {"variant": "H64", "rung": 2, "device": str(DEV),
       "prereg": "experiments/prereg/H64_prototype.md", "proxies": {}}

mouse = load_mouse()
hs, _ref, _rp = make_hard_synthetic_graph(seed=0)
for g in (mouse, hs):
    name = g.name
    rows = {}
    for warm, tag in ((False, "A"), (True, "B")):
        for sh, shape in SHAPES.items():
            cfg = RocketConfig(epochs=_EPOCHS[name])
            init = _init_positions_from_order(greedy_fas_order(g), DEV) if warm else None
            pos, sc, p, wall = rocket_with_surrogate(g, cfg, 42, DEV, shape,
                                                     init_positions=init)
            rows[tag + "_" + sh] = {"pure_pct": p, "wall_rocket_s": wall}
            print(name, tag + "_" + sh, "pure %.6f (%.1fs)" % (p, wall), flush=True)
    for sh, shape in SHAPES.items():
        r = full_pipeline(g, 42, DEV, shape)
        rows["C_" + sh] = r
        print(name, "C_" + sh,
              "pure %.6f sift %.6f alt %.6f FINAL %.6f"
              % (r["pure_pct"], r["sift_pct"], r["alt_pct"], r["final_pct"]), flush=True)

    d = {
      "A1_minus_A0_random_init_pure_pp": rows["A_asym"]["pure_pct"] - rows["A_sigmoid"]["pure_pct"],
      "B1_minus_B0_warm_start_pure_pp":  rows["B_asym"]["pure_pct"] - rows["B_sigmoid"]["pure_pct"],
      "C1_minus_C0_composed_final_pp":   rows["C_asym"]["final_pct"] - rows["C_sigmoid"]["final_pct"],
      "C1_minus_C0_composed_pure_pp":    rows["C_asym"]["pure_pct"] - rows["C_sigmoid"]["pure_pct"],
    }
    dp = d["C1_minus_C0_composed_pure_pp"]
    d["realised_pure_to_final_transfer"] = (
        d["C1_minus_C0_composed_final_pp"] / dp if abs(dp) > 1e-12 else None)
    out["proxies"][name] = {"arms": rows, "deltas": d, "n_nodes": int(g.n_nodes)}
    print("==", name, "DELTAS ==", json.dumps(d, indent=1), flush=True)

per = {n: bool(v["deltas"]["B1_minus_B0_warm_start_pure_pp"] <= 0
               and v["deltas"]["C1_minus_C0_composed_final_pp"] <= 0)
       for n, v in out["proxies"].items()}
out["per_proxy_negative_on_both_questions"] = per
out["prereg_kill_rule_fires"] = bool(all(per.values()))
print("KILL RULE FIRES:", out["prereg_kill_rule_fires"], per, flush=True)

json.dump(out, open("experiments/outputs/proto_H64_rung2.json", "w"), indent=1)
print("WROTE experiments/outputs/proto_H64_rung2.json", flush=True)
