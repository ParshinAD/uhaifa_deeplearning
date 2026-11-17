"""Order the Male CNS graph with the fly-connectome champion pipeline (H64) and score it.

The FlyWire champion constants are reused unchanged (20,000 epochs, 40 sift sweeps,
77 alternation cycles): the dicts in H64 are keyed by dataset name, so the "malecns"
keys are set here at runtime; no repo source file is edited. Scores come from the
frozen oracle src/mfas/metrics.py.

    $PY male_cns/run_mfas.py --seed 42
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfas.experiments import H64  # noqa: E402
from mfas.experiments.H02 import greedy_fas_order  # noqa: E402
from mfas.io import GraphData  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.utils.seeding import select_device, seed_everything  # noqa: E402

CONFIG = dict(name="malecns", template="connectome", graph="data/processed/malecns.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

z = np.load(ROOT / CONFIG["graph"])
g = GraphData(src=z["src"], tgt=z["tgt"], weight=z["weight"], node_ids=z["node_ids"], name=CONFIG["name"])
for d in (H64._EPOCHS, H64._MAX_SWEEPS, H64._ALT_CYCLES, H64._ALT_SIFT_SWEEPS, H64._PAIR_MAX_POPS,
          H64._RECLAIM_ROUNDS):
    d[CONFIG["name"]] = d[CONFIG["template"]]
print(g, flush=True)

src64, tgt64 = g.src.astype(np.int64), g.tgt.astype(np.int64)
t0 = time.time()
greedy_rank = greedy_fas_order(g)
greedy_score = score_from_order(greedy_rank, src64, tgt64, g.weight)
print("greedy-FAS", round(float(pct(greedy_score, g.total_weight)), 4), round(time.time() - t0, 1), "s", flush=True)

seed_everything(args.seed)
device, dev_name = select_device("auto")
t0 = time.time()
res = H64.run(g, seed=args.seed, device=device)
wall = time.time() - t0
rank = np.argsort(np.argsort(res.best_positions, kind="stable"), kind="stable").astype(np.int64)
final = score_from_order(rank, src64, tgt64, g.weight)
a = res.history.attrs

out = ROOT / "male_cns" / "results"
out.mkdir(parents=True, exist_ok=True)
np.save(out / f"malecns_H64_s{args.seed}_rank.npy", rank)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                        text=True).stdout.strip()
rec = dict(dataset=CONFIG["name"], algo="H64 (connectome constants)", seed=args.seed, device=dev_name,
           git_commit=commit, n_nodes=g.n_nodes, n_edges=g.n_edges, total_weight=int(g.total_weight),
           greedy_score=int(greedy_score), greedy_pct=pct(greedy_score, g.total_weight),
           pure_rocket_pct=a["pure_best_pct"], sift_pct=a["sift_best_pct"], alt_pct=a["alt_best_pct"],
           final_score=int(final), final_pct=pct(final, g.total_weight),
           feedback_pct=100 - pct(final, g.total_weight), wall_clock_s=wall,
           n_epochs_done=res.n_epochs_done, n_sift_sweeps=a["n_sift_sweeps"], n_alt_cycles=a["n_alt_cycles"],
           rank_file=f"male_cns/results/malecns_H64_s{args.seed}_rank.npy")
(out / f"malecns_H64_s{args.seed}.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
