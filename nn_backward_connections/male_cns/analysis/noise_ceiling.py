"""Noise ceiling for the cross-sex type comparison: how well do two good orders of the SAME
brain agree on per-type median position? Compared with the female-vs-male Spearman.

    $PY male_cns/analysis/noise_ceiling.py
Output: male_cns/results/noise_ceiling.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "male_cns" / "results"
sys.path.insert(0, str(ROOT / "male_cns" / "analysis"))
from analyze import load  # noqa: E402

CONFIG = dict(
    flywire_ref="results/champions/H64_connectome_s42_cuda.npz",
    flywire_alt=["biology/results/nullmodels/orders/connectome__real__ps42__e4000.npy",
                 "biology/results/nullmodels/orders/connectome__real__ps123__e4000.npy",
                 "biology/results/nullmodels/orders/connectome__real__ps999__e4000.npy"],
    malecns_alt=["male_cns/results/nulls/orders/malecns__real__ps42__e4000.npy",
                 "male_cns/results/nulls/orders/malecns__real__ps123__e4000.npy"],
)


def type_medians(rank, types):
    """Median normalized rank per type."""
    rn = rank / (len(rank) - 1)
    return pd.Series(rn).groupby(types).median()


R = {}
for ds, alts in [("flywire", CONFIG["flywire_alt"]), ("malecns", CONFIG["malecns_alt"])]:
    _, _, _, rank, a = load(ds, None)
    ref = type_medians(rank, a.flywire_type.to_numpy())
    lanes = a.groupby("flywire_type").lane.agg(lambda s: s.mode().iat[0])
    out = {}
    for p in alts:
        if not (ROOT / p).exists():
            continue
        r2 = np.load(ROOT / p).astype(np.int64)
        alt = type_medians(r2, a.flywire_type.to_numpy())
        both = pd.concat([ref, alt], axis=1, keys=["ref", "alt"]).dropna()
        by_lane = both.groupby(lanes.reindex(both.index).to_numpy()).apply(
            lambda d: d.ref.corr(d.alt, method="spearman") if len(d) > 10 else np.nan)
        out[Path(p).name] = dict(types=int(len(both)), spearman=float(both.ref.corr(both.alt, method="spearman")),
                                 neuron_spearman=float(pd.Series(rank).corr(pd.Series(r2), method="spearman")),
                                 by_lane={k: (None if pd.isna(v) else round(float(v), 3)) for k, v in by_lane.items()})
    R[ds] = out
(RES / "noise_ceiling.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))

# Cross-sex agreement with both orders from the same protocol (female G01 ps42, male brain-only
# re-optimized), so the male brain order is not a restriction of a full-CNS order.
_, _, _, _, af = load("flywire", None)
_, _, _, _, am = load("malecns_brain", None)
rf = np.load(ROOT / CONFIG["flywire_alt"][0]).astype(np.int64)
rm = np.load(ROOT / "male_cns/results/nulls/orders/malecns_brain__real__ps42__e4000.npy").astype(np.int64)
tf = type_medians(rf, af.flywire_type.to_numpy())
tm = type_medians(rm, am.flywire_type.to_numpy())
both = pd.concat([tf, tm], axis=1, keys=["f", "m"]).dropna()
lanes_f = af.groupby("flywire_type").lane.agg(lambda s: s.mode().iat[0]).reindex(both.index)
keep = ~lanes_f.isin(["ascending (VNC)", "descending", "sensory (other)", "motor"])
R["cross_sex_protocol_orders"] = dict(types=int(len(both)), spearman=float(both.f.corr(both.m, method="spearman")),
                                      spearman_without_neck_lanes=float(both[keep].f.corr(both[keep].m, method="spearman")))
(RES / "noise_ceiling.json").write_text(json.dumps(R, indent=1))
print(R["cross_sex_protocol_orders"])
