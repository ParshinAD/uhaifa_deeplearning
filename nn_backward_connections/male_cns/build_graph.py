"""Build the Male CNS v1.0 graph and node annotations in the repo's formats.

Runs in the isolated reader env male_cns/.venv (pandas + pyarrow; the `allen` env has no
feather reader and is not modified):

    male_cns/.venv/bin/python -I male_cns/build_graph.py

Inputs (public bucket gs://flyem-male-cns, no token; see male_cns/BUILD.md for URLs and sha256):
    data/raw/male_cns/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather
    data/raw/male_cns/body-annotations-male-cns-v1.0-minconf-0.5.feather
    data/raw/male_cns/body-neurotransmitters-male-cns-v1.0.feather

Outputs:
    data/processed/malecns.npz              src, tgt (int32), weight (int64), node_ids (bodyId)
    data/processed/malecns_nodes.csv.gz     one row per graph node, annotations joined
    male_cns/results/build_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "male_cns"
CONFIG = dict(
    weights="connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    annotations="body-annotations-male-cns-v1.0-minconf-0.5.feather",
    nts="body-neurotransmitters-male-cns-v1.0.feather",
    # FlyWire graph (data/README.md) has min weight 2; drop weight-1 edges for a like-for-like comparison.
    min_weight=2,
    ann_cols=["bodyId", "type", "flywireType", "instance", "superclass", "class", "subclass",
              "somaSide", "rootSide", "status", "statusLabel", "itoleeHl", "trumanHl",
              "dimorphism", "fruDsx", "group"],
)

e = pd.read_feather(RAW / CONFIG["weights"], columns=["body_pre", "body_post", "weight"])
n_raw, self_loops = len(e), int((e.body_pre == e.body_post).sum())
e = e[e.body_pre != e.body_post]
w1_edges, w1_weight = int((e.weight < CONFIG["min_weight"]).sum()), int(e.weight[e.weight < CONFIG["min_weight"]].sum())
e = e[e.weight >= CONFIG["min_weight"]]
node_ids = np.union1d(e.body_pre.to_numpy(), e.body_post.to_numpy())
src = np.searchsorted(node_ids, e.body_pre.to_numpy()).astype(np.int32)
tgt = np.searchsorted(node_ids, e.body_post.to_numpy()).astype(np.int32)
w = e.weight.to_numpy().astype(np.int64)
out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "malecns.npz", src=src, tgt=tgt, weight=w, node_ids=node_ids)

a = pd.read_feather(RAW / CONFIG["annotations"], columns=CONFIG["ann_cols"])
nt = pd.read_feather(RAW / CONFIG["nts"], columns=["body", "predicted_nt", "predicted_nt_confidence",
                                                   "celltype_predicted_nt", "consensus_nt"])
nodes = pd.DataFrame({"idx": np.arange(len(node_ids)), "bodyId": node_ids})
nodes = nodes.merge(a, on="bodyId", how="left").merge(nt.rename(columns={"body": "bodyId"}),
                                                     on="bodyId", how="left")
assert len(nodes) == len(node_ids) and (nodes.idx.to_numpy() == np.arange(len(node_ids))).all()
nodes.to_csv(out / "malecns_nodes.csv.gz", index=False)

stats = dict(
    source_rows=n_raw, self_loops_dropped=self_loops, min_weight_kept=CONFIG["min_weight"],
    edges_dropped_below_min=w1_edges, weight_dropped_below_min=w1_weight, n_nodes=int(len(node_ids)),
    n_edges=int(len(w)), total_weight=int(w.sum()), min_weight=int(w.min()), max_weight=int(w.max()),
    share_edges_weight_lt5=round(float((w < 5).mean()), 4),
    share_weight_in_edges_lt5=round(float(w[w < 5].sum() / w.sum()), 4),
    nodes_with_type=int(nodes["type"].notna().sum()),
    nodes_with_superclass=int(nodes["superclass"].notna().sum()),
    nodes_with_flywireType=int(nodes["flywireType"].notna().sum()),
    nodes_with_consensus_nt=int(nodes["consensus_nt"].notna().sum()),
    superclass_counts={str(k): int(v) for k, v in nodes["superclass"].value_counts(dropna=False).items()},
    somaSide_counts={str(k): int(v) for k, v in nodes["somaSide"].value_counts(dropna=False).items()},
    consensus_nt_counts={str(k): int(v) for k, v in nodes["consensus_nt"].value_counts(dropna=False).items()},
)
(ROOT / "male_cns" / "results").mkdir(parents=True, exist_ok=True)
(ROOT / "male_cns" / "results" / "build_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if not k.endswith("counts")}, indent=1))
