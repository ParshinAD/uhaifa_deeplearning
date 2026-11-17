"""Per-node arrays for the Male CNS streams page (counterpart of layering/tools/fly_ann_prep.py).

Loads the Male CNS graph and the one-seed H64 order, re-scores the order with the frozen
oracle (must give 93,041,515), encodes the per-node annotation columns as integer codes and
computes the spectral affinity coordinate with the FlyWire function
(fly_ann_prep.spectral_coords, same tau rule, seed and iteration count).

    $PY male_cns/streams/malecns_streams_prep.py
Output: male_cns/results/streams/node_ann.npz, node_ann_meta.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from layering.tools.fly_ann_prep import CONFIG as FW_CONFIG, spectral_coords  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402

CONFIG = dict(
    graph=ROOT / "data/processed/malecns.npz",
    rank=ROOT / "male_cns/results/malecns_H64_s42_rank.npy",
    ann=ROOT / "data/processed/malecns_ann.csv.gz",
    out_dir=ROOT / "male_cns/results/streams",
    expected_ff_weight=93_041_515,
    # short NT codes of ann_prep.py -> the FlyWire long names the page uses
    nt_long={"ACh": "acetylcholine", "GABA": "gaba", "Glu": "glutamate", "DA": "dopamine",
             "5-HT": "serotonin", "OA": "octopamine", "HA": "histamine"},
    cols=["lane", "class", "superclass", "side", "flow", "top_nt", "region"],
)


def encode(values):
    """Codes into a value list that starts with 'NA' (FlyWire meta convention)."""
    col = ["NA" if (not isinstance(v, str) or v == "") else v for v in values]
    vals = ["NA"] + sorted(set(col) - {"NA"})
    idx = {v: i for i, v in enumerate(vals)}
    return np.array([idx[v] for v in col], dtype=np.int16), vals


def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(CONFIG["graph"])
    src, tgt, w = z["src"].astype(np.int64), z["tgt"].astype(np.int64), z["weight"].astype(np.int64)
    n = int(z["node_ids"].shape[0])
    rank = np.load(CONFIG["rank"]).astype(np.int64)
    assert rank.shape[0] == n and np.array_equal(np.sort(rank), np.arange(n)), "rank is not a permutation"
    score = int(score_from_order(rank, src, tgt, w))
    W = int(w.sum())
    ff_pct = pct(score, W)
    assert score == CONFIG["expected_ff_weight"], f"order re-scores to {score}, expected {CONFIG['expected_ff_weight']}"
    print(f"n={n} m={src.shape[0]} W={W} FF={score} ({round(float(ff_pct), 4)} %)")

    a = pd.read_csv(CONFIG["ann"], low_memory=False)
    assert np.array_equal(a.idx.to_numpy(), np.arange(n))
    assert np.array_equal(a.bodyId.to_numpy(), z["node_ids"]), "annotation rows are not aligned to graph nodes"
    a["top_nt"] = a.top_nt.map(CONFIG["nt_long"])

    codes, values = {}, {}
    for c in CONFIG["cols"]:
        codes[c], values[c] = encode(a[c].tolist())
        print(f"  {c}: {len(values[c])} values")

    t1 = time.time()
    ((y1, it1), (y2, it2)), tau = spectral_coords(
        src, tgt, w.astype(np.float64), n, k=2, iters=FW_CONFIG["spec_iters"],
        seed=FW_CONFIG["spec_seed"], tol=FW_CONFIG["spec_tol"], tau_factor=FW_CONFIG["spec_tau_factor"])
    print(f"spectral coords: {it1} + {it2} iterations, tau={round(float(tau), 3)} ({round(time.time() - t1)} s)")

    np.savez_compressed(out / "node_ann.npz", rank=rank, spec_y=y1, spec_y2=y2,
                        **{c: codes[c] for c in CONFIG["cols"]})
    meta = {
        "dataset": "malecns", "n": n, "m": int(src.shape[0]), "W": W,
        "ff_weight": score, "champion_ff_pct": ff_pct,
        "order": "male_cns/results/malecns_H64_s42_rank.npy (H64 pipeline, one seed 42, MPS)",
        "annotation": "data/processed/malecns_ann.csv.gz (male_cns/pipeline/ann_prep.py)",
        "spectral": {"iters": [it1, it2], "seed": FW_CONFIG["spec_seed"], "tau": tau,
                     "tau_factor": FW_CONFIG["spec_tau_factor"],
                     "weights": "log1p of undirected collapsed synapse counts",
                     "operator": "tau-regularized random-walk (D+tau I)^-1 (W + tau/n J), deflated "
                                 "power iteration; within-lane use is ORDINAL (rank)"},
        **{f"{c}_values": values[c] for c in CONFIG["cols"]},
    }
    (out / "node_ann_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved -> {out} ({round(time.time() - t0)} s total)")


if __name__ == "__main__":
    main()
