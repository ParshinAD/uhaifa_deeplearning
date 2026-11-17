"""Random-graph null ladder (finding A1) on the Male CNS graph.

Reuses the FlyWire G01 protocol and generators unchanged (biology/analysis/nullmodels_run.py:
greedy-FAS -> Rocket+ASYM at 4,000 epochs -> 8 sift sweeps, frozen oracle), so the male
ladder is directly comparable with the logged FlyWire runs in biology/results/nullmodels/runs/.
L3 lanes are the Male CNS lanes of male_cns/pipeline/ann_prep.py. Fewer draws per rung than FlyWire
(compute); FlyWire draws have std <= 0.025 pp within a rung.

    $PY male_cns/pipeline/run_nulls.py            # resumes: existing JSONs are skipped
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "biology" / "analysis"))
from mfas.io import GraphData  # noqa: E402
from nullmodels_run import _draw_null, _finish_run, run_protocol  # noqa: E402

CONFIG = dict(
    epochs=4_000, device="mps", pipeline_seed=42,
    real_seeds=(42, 123),
    n_nulls=dict(L1_gnm=3, w_source=3, w_shuffled=2, L3_dcsbm=3),
    gen_seed_base=dict(w_shuffled=1_000, w_source=2_000, L1_gnm=3_000, L3_dcsbm=5_000),
    runs_dir=ROOT / "male_cns" / "results" / "nulls",
)

z = np.load(ROOT / "data/processed/malecns.npz")
g = GraphData(src=z["src"], tgt=z["tgt"], weight=z["weight"], node_ids=z["node_ids"], name="malecns")
src, tgt, w = np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight)
lane = pd.read_csv(ROOT / "data/processed/malecns_ann.csv.gz", usecols=["lane"]).lane.to_numpy()
labels = np.unique(lane, return_inverse=True)[1].astype(np.int64)
print(g, "lanes", labels.max() + 1, flush=True)

for seed in CONFIG["real_seeds"]:
    stem = f"malecns__real__ps{seed}__e{CONFIG['epochs']}"
    out = CONFIG["runs_dir"] / f"{stem}.json"
    if not out.exists():
        res = run_protocol(g, epochs=CONFIG["epochs"], pipeline_seed=seed, device=CONFIG["device"])
        _finish_run(res, out, CONFIG["runs_dir"] / "orders", stem, dict(dataset="malecns", kind="real", gen_seed=None))

# Brain-only subgraph (drop nerve-cord neurons): the closest like-for-like to FlyWire (brain only).
region = pd.read_csv(ROOT / "data/processed/malecns_ann.csv.gz", usecols=["region"]).region.to_numpy()
keep = region == "brain"
stem = f"malecns_brain__real__ps42__e{CONFIG['epochs']}"
out = CONFIG["runs_dir"] / f"{stem}.json"
if not out.exists():
    new = -np.ones(g.n_nodes, dtype=np.int64)
    new[keep] = np.arange(keep.sum())
    e = keep[src] & keep[tgt]
    gb = GraphData(src=new[src[e]].astype(np.int32), tgt=new[tgt[e]].astype(np.int32), weight=w[e],
                   node_ids=g.node_ids[keep], name="malecns_brain")
    res = run_protocol(gb, epochs=CONFIG["epochs"], pipeline_seed=42, device=CONFIG["device"])
    _finish_run(res, out, CONFIG["runs_dir"] / "orders", stem, dict(dataset="malecns_brain", kind="real", gen_seed=None))

for variant, k in CONFIG["n_nulls"].items():
    for i in range(k):
        gen_seed = CONFIG["gen_seed_base"][variant] + i
        stem = f"malecns__{variant}__gs{gen_seed}__ps{CONFIG['pipeline_seed']}__e{CONFIG['epochs']}"
        out = CONFIG["runs_dir"] / f"{stem}.json"
        if out.exists():
            continue
        t0 = time.time()
        s2, t2, w2, stats = _draw_null(variant, src, tgt, w, g.n_nodes, gen_seed, labels)
        gn = GraphData(src=s2.astype(np.int32), tgt=t2.astype(np.int32), weight=w2,
                       node_ids=np.arange(g.n_nodes), name=f"null_{variant}")
        res = run_protocol(gn, epochs=CONFIG["epochs"], pipeline_seed=CONFIG["pipeline_seed"],
                           device=CONFIG["device"])
        res["t_generate_s"] = time.time() - t0
        _finish_run(res, out, None, stem, dict(dataset="malecns", kind=variant, gen_seed=gen_seed, gen_stats=stats))
