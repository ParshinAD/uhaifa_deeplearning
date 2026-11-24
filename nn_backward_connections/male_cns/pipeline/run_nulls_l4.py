"""L4 cell-type null (finding A7, null-ladder v2 rung `type_dcsbm`) on the Male CNS graph.

Same generator and protocol as the FlyWire L4 rung (biology/analysis/nullmodels2_run.py):
block_configuration_null with cell-type labels, then greedy-FAS -> Rocket+ASYM at 4,000 epochs
-> 8 sift sweeps, frozen oracle. Real comparators are the logged G01-protocol runs in
male_cns/results/nulls/malecns__real__ps*__e4000.json.

Type labels: Male CNS `cell_type`, falling back to "<superclass> (class)", then "unannotated"
(the FlyWire rule of fly_type_labels with the columns this dataset has).

Reading rule, fixed before the first run: the FlyWire inversion replicates if the real male
graph (80.828 / 80.835 % FF) is below every draw; it does not if the real graph is above every
draw; anything else is reported as mixed. The budget probe (--epochs 20000 --sift-sweeps 40, one
pinned draw and the real graph) must keep the sign.

    $PY male_cns/pipeline/run_nulls_l4.py                       # 12 draws, resumes
    $PY male_cns/pipeline/run_nulls_l4.py --stats-only          # labels and generator stats, no run
    $PY male_cns/pipeline/run_nulls_l4.py --n-nulls 1 --epochs 20000 --sift-sweeps 40
    $PY male_cns/pipeline/run_nulls_l4.py --real --epochs 20000 --sift-sweeps 40
    $PY male_cns/pipeline/run_nulls_l4.py --n-nulls 1 --singleton-fallback   # fallback-label control
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "biology" / "analysis"))
import nullmodels_run  # noqa: E402
from mfas.io import GraphData  # noqa: E402
from nullmodels2_generators import two_cycle_bound  # noqa: E402
from nullmodels_generators import block_configuration_null  # noqa: E402
from nullmodels_run import _finish_run, run_protocol  # noqa: E402

CONFIG = dict(
    epochs=4_000, device="mps", pipeline_seed=42,
    n_nulls=12, gen_seed_base=8_000,
    runs_dir=ROOT / "male_cns" / "results" / "nulls",
)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--n-nulls", type=int, default=CONFIG["n_nulls"])
ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
ap.add_argument("--sift-sweeps", type=int, default=None)
ap.add_argument("--real", action="store_true", help="run the real graph at this budget instead of nulls")
ap.add_argument("--stats-only", action="store_true")
ap.add_argument("--singleton-fallback", action="store_true",
                help="control: every neuron without a cell_type is its own type")
args = ap.parse_args()
if args.sift_sweeps is not None:
    nullmodels_run.CONFIG["sift_max_sweeps"] = args.sift_sweeps

z = np.load(ROOT / "data/processed/malecns.npz")
g = GraphData(src=z["src"], tgt=z["tgt"], weight=z["weight"], node_ids=z["node_ids"], name="malecns")
src, tgt, w = np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight)
total = float(w.sum())

ann = pd.read_csv(ROOT / "data/processed/malecns_ann.csv.gz", usecols=["cell_type", "superclass"])
fallback = (ann.superclass + " (class)").fillna("unannotated")
names = ann.cell_type.fillna(fallback)
if args.singleton_fallback:
    names = ann.cell_type.fillna("untyped " + ann.index.astype(str).to_series(index=ann.index))
kind = "type_dcsbm_singlefb" if args.singleton_fallback else "type_dcsbm"
labels = np.unique(names.to_numpy(), return_inverse=True)[1].astype(np.int64)
is_fallback = ann.cell_type.isna().to_numpy()
bound_real = two_cycle_bound(src, tgt, w, g.n_nodes)
print(g, flush=True)
print("types", labels.max() + 1, "fallback nodes", int(is_fallback.sum()),
      "share of W touching a fallback node", round(float(w[is_fallback[src] | is_fallback[tgt]].sum() / total), 4),
      "real 2-cycle floor %W", round(100 * bound_real / total, 4), flush=True)

if args.real:
    stem = f"malecns__real__ps{CONFIG['pipeline_seed']}__e{args.epochs}"
    out = CONFIG["runs_dir"] / f"{stem}.json"
    if not out.exists():
        res = run_protocol(g, epochs=args.epochs, pipeline_seed=CONFIG["pipeline_seed"], device=CONFIG["device"])
        _finish_run(res, out, CONFIG["runs_dir"] / "orders", stem, dict(dataset="malecns", kind="real", gen_seed=None))
    sys.exit(0)

for i in range(args.n_nulls):
    gen_seed = CONFIG["gen_seed_base"] + i
    stem = f"malecns__{kind}__gs{gen_seed}__ps{CONFIG['pipeline_seed']}__e{args.epochs}"
    out = CONFIG["runs_dir"] / f"{stem}.json"
    if out.exists() and not args.stats_only:
        continue
    t0 = time.time()
    s2, t2, w2, stats = block_configuration_null(src, tgt, w, labels, seed=gen_seed)
    t_gen = time.time() - t0
    bound_null = two_cycle_bound(s2, t2, w2, g.n_nodes)
    moved = t2 != tgt
    extra = dict(dataset="malecns", kind=kind, gen_seed=gen_seed, gen_stats=stats,
                 n_types=int(labels.max() + 1),
                 moved_edge_share=float(moved.mean()), moved_weight_share=float(w[moved].sum() / total),
                 bound_2cycle=bound_null, bound_2cycle_pct=100.0 * bound_null / total,
                 bound_2cycle_real=bound_real, bound_2cycle_real_pct=100.0 * bound_real / total)
    if args.stats_only:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in extra.items()}, "t_gen", round(t_gen, 1))
        break
    gn = GraphData(src=s2.astype(np.int32), tgt=t2.astype(np.int32), weight=w2,
                   node_ids=np.arange(g.n_nodes), name="null_type_dcsbm")
    res = run_protocol(gn, epochs=args.epochs, pipeline_seed=CONFIG["pipeline_seed"], device=CONFIG["device"])
    res["t_generate_s"] = t_gen
    _finish_run(res, out, None, stem, extra)
