"""REVIEW addendum (3d): the KC->ACh override counterfactual.

Established by review_label_nulls.py: the cache layering/outputs/fly_ann/
node_ann.npz predates the (uncommitted) nt_overrides edit of fly_ann_prep.py —
its top_nt is bit-identical to the raw Schlegel/Eckstein TSV, so the study ran
WITHOUT the KC->ACh correction it claims was applied. Consequence: 5,172 of the
5,854 "dopamine" neurons are Kenyon cells (ground-truth cholinergic).

This script computes the corrected T1 rows: apply KC->ACh, recompute observed
per-NT backward shares, and run the within-super_class null (K=300, review
seed) for the corrected labels. Also decomposes: KC-only backward share,
raw-DA-without-KC share.

Run:  $PY biology/analysis/review_label_nulls_kc.py
Out:  biology/results/review/review_label_nulls_kc.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biology.analysis.review_label_nulls import emp, shuffle_within, RSEED, K


def main():
    t0 = time.time()
    from layering import io_utils
    g, order, rank, ff = io_utils.load_graph_and_champion_order("connectome")
    ann = np.load(REPO_ROOT / "layering/outputs/fly_ann/node_ann.npz")
    meta = json.loads((REPO_ROOT / "layering/outputs/fly_ann/node_ann_meta.json")
                      .read_text())
    nt = ann["top_nt"].astype(np.int64)
    sc = ann["super_class"].astype(np.int64)
    cc = ann["cell_class"].astype(np.int64)
    nt_names = meta["top_nt_values"]
    ach = nt_names.index("acetylcholine")
    da = nt_names.index("dopamine")
    kc = cc == meta["cell_class_values"].index("Kenyon_Cell")

    w = g.weight.astype(np.int64)
    back = rank[g.src] > rank[g.tgt]
    n = rank.shape[0]
    ow = np.zeros(n, np.int64)
    bw = np.zeros(n, np.int64)
    np.add.at(ow, g.src, w)
    np.add.at(bw, g.src[back], w[back])

    def share(mask):
        return 100.0 * bw[mask].sum() / ow[mask].sum()

    def shares_of(lab):
        num = np.bincount(lab, weights=bw, minlength=len(nt_names))
        den = np.bincount(lab, weights=ow, minlength=len(nt_names))
        return 100.0 * num / den

    # corrected labels: KC -> acetylcholine
    nt_ov = nt.copy()
    nt_ov[kc] = ach

    rep = dict(seed=RSEED, n_perm=K,
               decomposition=dict(
                   kc=dict(n=int(kc.sum()), backward_share=share(kc)),
                   da_without_kc=dict(n=int(((nt == da) & ~kc).sum()),
                                      backward_share=share((nt == da) & ~kc)),
                   da_as_published=dict(n=int((nt == da).sum()),
                                        backward_share=share(nt == da)),
                   ach_as_published=dict(n=int((nt == ach).sum()),
                                         backward_share=share(nt == ach)),
                   ach_corrected=dict(n=int((nt_ov == ach).sum()),
                                      backward_share=share(nt_ov == ach))))

    obs = shares_of(nt_ov)
    for tag, strat in [("super_class", sc), ("super_x_cell", sc * 1000 + cc)]:
        rng = np.random.default_rng([RSEED, 99, hash(tag) % 2**31])
        draws = np.empty((K, len(nt_names)))
        for k in range(K):
            draws[k] = shares_of(shuffle_within(nt_ov, strat, rng))
        rep[f"corrected_T1_{tag}"] = {
            nm: dict(emp(draws[:, i], obs[i]),
                     n_neurons=int((nt_ov == i).sum()))
            for i, nm in enumerate(nt_names)}

    rep["seconds"] = round(time.time() - t0, 1)
    out = REPO_ROOT / "biology/results/review/review_label_nulls_kc.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["decomposition"], indent=1))
    for tag in ("corrected_T1_super_class", "corrected_T1_super_x_cell"):
        print("==", tag)
        for nm, st in rep[tag].items():
            print(f"  {nm:14s} n {st['n_neurons']:6d} obs {st['observed']:.3f} "
                  f"mean {st['null_mean']:.3f} sd {st['null_sd']:.3f} "
                  f"z {st['z']:+.2f} p {st['p_two_sided']:.4g}")
    print(f"wrote {out} ({rep['seconds']}s)")


if __name__ == "__main__":
    main()
