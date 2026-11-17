"""Where the Male CNS order cuts the brain <-> nerve-cord loop, and cross-sex rank agreement
without the neurons that FlyWire truncates at the neck.

    $PY male_cns/analysis/vnc_loop.py
Output: male_cns/results/vnc_loop.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "male_cns" / "results"
CONFIG = dict(early=0.1, late=0.9, neck_lanes=["ascending (VNC)", "descending", "sensory (other)", "motor"])

z = np.load(ROOT / "data/processed/malecns.npz")
src, tgt, w = z["src"].astype(np.int64), z["tgt"].astype(np.int64), z["weight"]
rank = np.load(RES / "malecns_H64_s42_rank.npy")
n = len(rank)
rn = rank / (n - 1)
a = pd.read_csv(ROOT / "data/processed/malecns_ann.csv.gz", low_memory=False)
fb = rank[src] > rank[tgt]
vnc = (a.region == "VNC").to_numpy()
lane = a.lane.to_numpy()
R = {}

# FF share and FB budget per block: brain->brain, brain->VNC, VNC->brain, VNC->VNC
blocks = {}
for nm, m in [("brain->brain", ~vnc[src] & ~vnc[tgt]), ("brain->VNC", ~vnc[src] & vnc[tgt]),
              ("VNC->brain", vnc[src] & ~vnc[tgt]), ("VNC->VNC", vnc[src] & vnc[tgt])]:
    blocks[nm] = dict(share_of_W=float(w[m].sum() / w.sum()), ff_share=float(1 - w[m & fb].sum() / w[m].sum()),
                      share_of_FB=float(w[m & fb].sum() / w[fb].sum()))
R["blocks"] = blocks

# DN and AN positions: early / late fractions, and where their input comes from
for ln in ["descending", "ascending (VNC)", "VNC sensory", "VNC intrinsic", "motor (VNC)"]:
    m = lane == ln
    inw = np.bincount(tgt, weights=w.astype(float), minlength=n)
    in_vnc = np.bincount(tgt[vnc[src]], weights=w[vnc[src]].astype(float), minlength=n)
    early, late = m & (rn < CONFIG["early"]), m & (rn > CONFIG["late"])
    R[ln] = dict(n=int(m.sum()), median=float(np.median(rn[m])), frac_early=float(early.mean() / m.mean()),
                 frac_late=float(late.mean() / m.mean()),
                 input_from_vnc_share_early=float(in_vnc[early].sum() / max(inw[early].sum(), 1)),
                 input_from_vnc_share_late=float(in_vnc[late].sum() / max(inw[late].sum(), 1)))

# DN types that sit early: what feeds them
dn = (lane == "descending") & (rn < CONFIG["early"])
inw_lane = pd.DataFrame({"lane": lane[src], "w": w})[dn[tgt]].groupby("lane").w.sum().sort_values(ascending=False)
R["early_DN_input_by_lane_share"] = (inw_lane / inw_lane.sum()).round(4).head(6).to_dict()
R["early_DN_types_top"] = a.cell_type[dn].value_counts().head(10).to_dict()

# Cross-sex type agreement without neck-truncated lanes (FlyWire lacks the cord)
j = pd.read_csv(RES / "type_rank_cross_sex.csv", index_col=0)
k = j[~j.lane_f.isin(CONFIG["neck_lanes"])]
R["cross_sex_spearman_all"] = float(j.median_rank_f.corr(j.median_rank_m, method="spearman"))
R["cross_sex_spearman_without_neck_lanes"] = float(k.median_rank_f.corr(k.median_rank_m, method="spearman"))
R["cross_sex_types_without_neck_lanes"] = int(len(k))
R["cross_sex_abs_delta_gt_0.5_by_lane"] = j[(j.delta.abs() > 0.5)].lane_f.value_counts().to_dict()
(RES / "vnc_loop.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
