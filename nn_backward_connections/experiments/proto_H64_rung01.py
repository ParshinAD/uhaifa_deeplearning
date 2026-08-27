"""H64 rungs 0+1 — zero GPU. (0) M1's burden (a)+(b) re-derived on this machine.
(1) the REACH diagnostic on the connectome from the stored champion positions."""
import json, sys, numpy as np, torch
sys.path.insert(0, "src")
from mfas.analysis.surrogate_gate import cheap_gate, alignment_report
from mfas.experiments.H38 import _asym_surrogate, M, T
from mfas.io import load_connectome
from mfas.baseline.rocket import RocketConfig, make_beta_schedule

out = {"variant": "H64", "rung": "0+1", "M": M, "T": T}

# ── RUNG 0a: cheap_gate ──────────────────────────────────────────────────────
def to_native(o):
    if isinstance(o, dict):  return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, list):  return [to_native(v) for v in o]
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

gate_asym = cheap_gate(_asym_surrogate, name="ASYM(M=0.75,T=1.5)")
gate_sig  = cheap_gate(torch.sigmoid,   name="sigmoid(baseline)")
out["cheap_gate_asym"] = to_native(gate_asym)
out["cheap_gate_sigmoid"] = to_native(gate_sig)
print("RUNG 0a cheap_gate ASYM   :", gate_asym["verdict"], gate_asym["reasons"])
print("RUNG 0a cheap_gate sigmoid:", gate_sig["verdict"], gate_sig["reasons"])

# ── RUNG 0b: no crossover in beta*std, on the hard synthetic ─────────────────
from mfas.analysis.gap import make_hard_synthetic_graph, _greedy_fas_reference_order
gs, ref_order, ref_pct = make_hard_synthetic_graph(seed=0)
gf = _greedy_fas_reference_order(gs)
n = gs.n_nodes
rng = np.random.RandomState(0)
orders = {
    "best": ref_order.astype(np.int64),
    "rocket": gf.astype(np.int64),
    "random": rng.permutation(n).astype(np.int64),
}
# imbalance sort: rank by (out_w - in_w) descending
w = np.asarray(gs.weight, dtype=np.float64)
outw = np.bincount(np.asarray(gs.src), weights=w, minlength=n)
inw  = np.bincount(np.asarray(gs.tgt), weights=w, minlength=n)
orders["imbalance_sort"] = np.argsort(np.argsort(-(outw - inw), kind="stable"),
                                      kind="stable").astype(np.int64)

sweep = []
for std in [1e-2, 1e-1, 1.0, 1e1, 1e2, 141.0, 1e3, 1e4, 1e5, 1e6]:
    ra = alignment_report(_asym_surrogate, gs, orders, beta=1.05, operating_std=std)
    rs = alignment_report(torch.sigmoid,   gs, orders, beta=1.05, operating_std=std)
    row = {"std": std,
           "beta_std": 1.05 * std,
           "A_asym": ra.get("alignment_ratio"),
           "A_sigmoid": rs.get("alignment_ratio")}
    sweep.append(to_native(row))
    print(f"RUNG 0b beta*std={row['beta_std']:>12.4g}  A_asym={row['A_asym']}  A_sig={row['A_sigmoid']}")
out["crossover_sweep_hard_synthetic"] = sweep
A = [r["A_asym"] for r in sweep if r["A_asym"] is not None]
out["asym_crossover_present"] = bool(min(A) <= 0 < max(A)) if A else None
out["asym_A_min"] = float(min(A)) if A else None
out["asym_A_max"] = float(max(A)) if A else None

# ── RUNG 1: REACH on the connectome from the stored champion positions ──────
gc = load_connectome()
pos = np.load("results/rocket_best_positions.npy")   # parity anchor vector
rank_anchor = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)

cand = {"rocket_best_positions.npy (parity anchor, 82.9161%)": rank_anchor}
import glob, os
champ = sorted(glob.glob("results/*H42-connectome*positions*.npy")) + \
        sorted(glob.glob("results/*H42*connectome*.npy"))
out["champion_npy_candidates"] = champ[:10]
if champ:
    p = champ[-1]
    v = np.load(p)
    cand[os.path.basename(p)] = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.int64)

cfg = RocketConfig()
betas = make_beta_schedule(cfg.epochs, cfg.cycles)
beta_op = float(np.max(betas))
out["beta_max_schedule"] = beta_op

src, tgt = np.asarray(gc.src), np.asarray(gc.tgt)
wc = np.asarray(gc.weight, dtype=np.float64)
reach_rows = []
for label, rk in cand.items():
    d = rk[tgt] - rk[src]                      # rank distance, >0 = feedforward
    back = d < 0
    bw = wc[back]; bd = (-d[back]).astype(np.float64)   # positive rank distance of backward edges
    tot_bw = bw.sum()
    # per-rank position spacing at the operating point: positions are ranks embedded to std S.
    # Use the MEASURED per-rank beta*Delta from the queue item's own figure and from the anchor.
    row = {"order": label,
           "backward_weight": float(tot_bw),
           "backward_weight_frac_of_total": float(tot_bw / wc.sum()),
           "median_backward_rank_distance": float(np.median(bd)),
           "p75_backward_rank_distance": float(np.percentile(bd, 75)),
           "weighted_median_backward_rank_distance": float(
               np.interp(0.5, np.cumsum(bw[np.argsort(bd)]) / tot_bw, np.sort(bd)))}
    # ASYM live support: tanh((z-M)/T) is numerically dead for (z-M)/T < -4  => z < M-4T = -5.25
    # sigmoid: sigmoid'(z) is dead for |z| > ~18 in float32 but practically for |z| > 8.
    for per_rank in [0.003753]:                # the operating-point beta*Delta per rank (queue item)
        z = -bd * per_rank                     # backward edge: Delta<0 => z<0
        asym_live = z > (M - 4.0 * T)
        sig_live  = z > -8.0
        row[f"asym_live_weight_frac@{per_rank}"] = float(bw[asym_live].sum() / tot_bw)
        row[f"sigmoid_live_weight_frac@{per_rank}"] = float(bw[sig_live].sum() / tot_bw)
        row[f"asym_reach_ranks@{per_rank}"] = float((M + 4.0 * T) / per_rank)
        row[f"sigmoid_reach_ranks@{per_rank}"] = float(8.0 / per_rank)
        for r in [1000, 5000, 20000, 60000]:
            row[f"backward_weight_frac_beyond_{r}_ranks"] = float(bw[bd > r].sum() / tot_bw)
    reach_rows.append(row)
    print("RUNG 1", json.dumps(row, indent=1))
out["reach"] = reach_rows

json.dump(out, open("experiments/outputs/proto_H64_rung01.json", "w"), indent=1)
print("WROTE experiments/outputs/proto_H64_rung01.json")
