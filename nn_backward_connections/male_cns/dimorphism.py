"""Do sexually dimorphic / fru-dsx types move more between the female and male orders than
other matched types of the same lane? A sex-specific signal would show up here; lab and
segmentation differences would not prefer these types.

    $PY male_cns/dimorphism.py
Output: male_cns/results/dimorphism.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "male_cns" / "results"
CONFIG = dict(seed=42, n_perm=10000)

j = pd.read_csv(RES / "type_rank_cross_sex.csv", index_col=0)
a = pd.read_csv(ROOT / "data/processed/malecns_ann.csv.gz", low_memory=False)
f = pd.read_csv(ROOT / "data/external/flywire_neuron_annotations.tsv.gz", sep="\t", low_memory=False,
                usecols=["cell_type", "dimorphism", "fru_dsx"])
dim_m = a[a.dimorphism.notna() | a.fruDsx.notna()].flywireType.dropna().unique()
dim_f = f[f.dimorphism.isin(["sexually dimorphic", "potentially sexually dimorphic"]) | f.fru_dsx.notna()].cell_type.dropna().unique()
j["dimorphic"] = j.index.isin(set(dim_m) | set(dim_f))
j["absd"] = j.delta.abs()
rng = np.random.default_rng(CONFIG["seed"])

# Lane-stratified permutation: shuffle the dimorphic flag within lanes.
obs = j.loc[j.dimorphic, "absd"].median() - j.loc[~j.dimorphic, "absd"].median()
groups = [np.where(j.lane_f.to_numpy() == l)[0] for l in j.lane_f.unique()]
flag = j.dimorphic.to_numpy()
absd = j.absd.to_numpy()
null = []
for _ in range(CONFIG["n_perm"]):
    p = flag.copy()
    for g in groups:
        p[g] = flag[rng.permutation(g)]
    null.append(np.median(absd[p]) - np.median(absd[~p]))
null = np.array(null)
R = dict(types=int(len(j)), dimorphic_types=int(j.dimorphic.sum()),
         median_abs_delta_dimorphic=float(j.loc[j.dimorphic, "absd"].median()),
         median_abs_delta_other=float(j.loc[~j.dimorphic, "absd"].median()),
         diff=float(obs), p_one_sided=float((null >= obs).mean()),
         by_lane=j.groupby(["lane_f", "dimorphic"]).absd.median().unstack().round(4).dropna().to_dict("index"),
         top_dimorphic_movers=j[j.dimorphic].nlargest(10, "absd")[["lane_f", "median_rank_f", "median_rank_m"]].round(3).reset_index().to_dict("records"))
(RES / "dimorphism.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
