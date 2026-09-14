"""Q3 experiment: swap mirror-paired left/right neurons in the champion order
and measure the score drop.

Rationale (QA_ORDER_AND_ROBUSTNESS.md, Q3): if the wiring were perfectly
mirror-symmetric AND the pairing perfect, swapping every left neuron's rank
with its right twin's rank would leave the feedforward score EXACTLY
unchanged (a bijection of edges). The measured drop is therefore an index of
(wiring asymmetry + pairing imperfection + the champion's fine-tuning to this
side's realization).

Pairing (v1, stated choice): within each cell type, sort the left members and
the right members by champion rank and pair i-th with i-th; min(#L,#R) pairs
per type; neurons without a partner keep their rank. Cell type = the
annotation's cell_type with the class fallback, as everywhere in the artifact.

Controls (same number of swapped pairs where possible, seed 42):
  C1  same-side, same-type, RANDOM pairs — generic same-type swap cost, but
      with larger rank displacements than the mirror pairing produces.
  C1b same-side, same-type, RANK-MATCHED pairs (random split of a (type,side)
      group into two halves, each sorted by rank, i-th <-> i-th) — the fair
      control: same pairing structure as the mirror swap, same side.
  C2  random pairs across the whole brain — the destruction baseline.
Mean |rank displacement| is reported per scheme so the controls' fairness is
visible.

Scoring: the frozen oracle only (invariant 6 untouched; we evaluate
permutations, never edit the scorer).

Run:  $PY layering/tools/exp_mirror_swap.py
Outputs -> layering/outputs/qa/exp_mirror_swap.json (+ printed summary)
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "fly_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "fly_ann" / "node_ann_meta.json",
    "ann_tsv": REPO / "data" / "external" / "flywire_neuron_annotations.tsv.gz",
    "out": HERE.parent / "outputs" / "qa" / "exp_mirror_swap.json",
    "seed": 42,
    "min_per_side": 1,     # a type contributes pairs as long as both sides exist
}


def load_types(node_ids):
    """Per-node type string: cell_type, falling back to '<class> (class)'."""
    ctype = {}
    with gzip.open(CONFIG["ann_tsv"], "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["cell_type"].strip()
            if not t:
                c = row["cell_class"].strip() or row["super_class"].strip()
                t = f"{c} (class)" if c else ""
            ctype[int(row["root_id"])] = t or "unannotated"
    return np.array([ctype.get(int(i), "unannotated") for i in node_ids])


def swapped_rank(rank, pairs):
    r = rank.copy()
    a = pairs[:, 0]
    b = pairs[:, 1]
    r[a], r[b] = rank[b], rank[a]
    return r


def main():
    t0 = time.time()
    from layering import io_utils
    from src.mfas.metrics import score_from_order, pct

    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    src, tgt, w = g.src, g.tgt, g.weight
    Wtot = float(w.sum())
    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    sidev = meta["side_values"]
    side = d["side"]
    L, R = sidev.index("left"), sidev.index("right")
    types = load_types(g.node_ids)
    n = rank.shape[0]

    # ---- mirror pairing: within type, rank-sorted left[i] <-> right[i] ------
    pairs = []
    per_type = {}
    for t in np.unique(types):
        m = types == t
        li = np.where(m & (side == L))[0]
        ri = np.where(m & (side == R))[0]
        if len(li) < CONFIG["min_per_side"] or len(ri) < CONFIG["min_per_side"]:
            continue
        li = li[np.argsort(rank[li], kind="stable")]
        ri = ri[np.argsort(rank[ri], kind="stable")]
        k = min(len(li), len(ri))
        pairs.append(np.column_stack([li[:k], ri[:k]]))
        per_type[t] = k
    pairs = np.concatenate(pairs, axis=0)
    n_pairs = len(pairs)
    covered = 2 * n_pairs
    print(f"mirror pairing: {n_pairs:,} pairs over {len(per_type):,} types "
          f"-> {covered:,}/{n} neurons covered ({100*covered/n:.1f}%)")

    rng = np.random.default_rng(CONFIG["seed"])

    # C1: same-side same-type random disjoint pairs, subsampled to n_pairs
    c1_pool = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            k2 = (len(idx) // 2) * 2
            c1_pool.append(idx[:k2].reshape(-1, 2))
    c1_pool = np.concatenate(c1_pool, axis=0)
    take = rng.choice(len(c1_pool), size=min(n_pairs, len(c1_pool)), replace=False)
    c1_pairs = c1_pool[take]

    # C1b: same-side same-type, rank-matched halves (fair structural control)
    c1b = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            half = len(idx) // 2
            a = idx[:half]
            b = idx[half:2 * half]
            a = a[np.argsort(rank[a], kind="stable")]
            b = b[np.argsort(rank[b], kind="stable")]
            c1b.append(np.column_stack([a, b]))
    c1b_pairs = np.concatenate(c1b, axis=0)

    # C2: random disjoint pairs across the whole brain
    perm = rng.permutation(n)
    c2_pairs = perm[: 2 * n_pairs].reshape(-1, 2)

    def score_of(r):
        return pct(score_from_order(r, src, tgt, w), Wtot)

    results = {"champion_ff_pct": ff_pct}
    for name, prs in (("mirror_swap", pairs),
                      ("control_same_side_same_type_random", c1_pairs),
                      ("control_same_side_same_type_rank_matched", c1b_pairs),
                      ("control_random_pairs", c2_pairs)):
        s = score_of(swapped_rank(rank, prs))
        disp = float(np.abs(rank[prs[:, 0]] - rank[prs[:, 1]]).mean())
        results[name] = {"n_pairs": int(len(prs)), "ff_pct": s,
                         "delta_pp": s - ff_pct,
                         "mean_abs_rank_displacement": disp}
        print(f"{name:42s} pairs={len(prs):7,}  FF={s:.4f}%  "
              f"delta={s-ff_pct:+.4f} pp  mean|drank|={disp:,.0f}")

    # localisation: which lane pairs change FF weight under the mirror swap
    from layering.tools.fly_streams import assign_lanes
    lane, lane_list = assign_lanes(rank, d["super_class"], d["cell_class"],
                                   meta["super_class_values"],
                                   meta["cell_class_values"])
    r2 = swapped_rank(rank, pairs)
    ff1 = rank[src] < rank[tgt]
    ff2 = r2[src] < r2[tgt]
    changed = ff1 != ff2
    Kl = len(lane_list)
    key = lane[src].astype(np.int64) * Kl + lane[tgt].astype(np.int64)
    sign = np.where(ff2, 1.0, -1.0)          # became-forward minus became-backward
    delta_w = np.bincount(key[changed], weights=(w[changed].astype(np.float64)
                                                 * sign[changed]),
                          minlength=Kl * Kl).reshape(Kl, Kl)
    flat = [(lane_list[i], lane_list[j], float(delta_w[i, j]))
            for i in range(Kl) for j in range(Kl) if delta_w[i, j] != 0]
    flat.sort(key=lambda x: -abs(x[2]))
    results["mirror_swap_lane_delta_top10"] = [
        {"src": a, "tgt": b, "net_ff_weight_change": v,
         "pp_of_total": 100 * v / Wtot} for a, b, v in flat[:10]]
    results["pairing"] = {
        "method": "within cell_type, rank-sorted left[i]<->right[i]",
        "n_pairs": int(n_pairs), "n_types": int(len(per_type)),
        "coverage_pct": 100 * covered / n,
        "seed": CONFIG["seed"],
    }
    results["seconds"] = time.time() - t0

    CONFIG["out"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["out"].write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"done in {results['seconds']:.0f}s -> {CONFIG['out']}")


if __name__ == "__main__":
    main()
