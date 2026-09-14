"""Q4 experiment: degrade the champion order to ~80 % and ~70 % feedforward
and re-derive the headline conclusions — do they survive suboptimality?

Degradation model (stated choice): seeded LOCAL window shuffles — pick a
random contiguous rank window of `window` ranks and randomly permute the
ranks inside it; repeat in batches until the oracle score falls to the
target. Local shuffles approximate "the optimizer stopped early" better than
global pair swaps (early stopping leaves coarse structure right and local
detail wrong), but it is a model, not a rerun of the optimizer — stated in
the caveat. A fully random order is included as the floor.

For each order we recompute, with the SAME code paths as the main pipeline:
  - oracle feedforward %
  - the 17 lane medians -> Spearman vs the champion's lane medians, plus the
    number of adjacent inversions in the canonical 9-lane staircase
    photoreceptors -> lamina -> medulla -> medulla→lobula -> lobula (plate)
    -> visual projection -> central (other) -> descending -> motor
  - the flow-field hierarchy rho (afferent<intrinsic<efferent vs rank)
  - the backward share within ACh / GABA / glutamate / serotonin
  - a type-contiguity proxy: median over types (n>=100) of IQR(type ranks)/n

Run:  $PY layering/tools/exp_degraded_order.py
Outputs -> layering/outputs/qa/exp_degraded_order.json (+ printed table)
"""
from __future__ import annotations

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
    "out": HERE.parent / "outputs" / "qa" / "exp_degraded_order.json",
    "seed": 42,
    "targets_ff_pct": [80.0, 70.0],
    "window": 2000,          # ranks per shuffle window
    "batch": 40,             # windows between oracle checks
    "max_batches": 400,
    "staircase": ["photoreceptors", "lamina", "medulla",
                  "medulla→lobula", "lobula (plate)",
                  "visual projection", "central (other)", "descending",
                  "motor"],
    "nts": ["acetylcholine", "gaba", "glutamate", "serotonin"],
    "type_min_n": 100,
}


def spearman(x, y):
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def metrics_for(rank, ctx):
    """All Q4 metrics for one rank vector."""
    src, tgt, w = ctx["src"], ctx["tgt"], ctx["w"]
    ff = rank[src] < rank[tgt]
    out = {}
    # lane medians + staircase
    med = np.array([np.median(rank[ctx["lane"] == i]) for i in range(ctx["K"])])
    out["lane_median_spearman_vs_champion"] = spearman(med, ctx["champ_med"])
    st = [med[ctx["lane_idx"][nm]] for nm in CONFIG["staircase"]]
    out["staircase_adjacent_inversions"] = int(
        sum(st[i] > st[i + 1] for i in range(len(st) - 1)))
    # flow rho
    keep = ctx["flow_ord"] >= 0
    out["flow_rho"] = spearman(ctx["flow_ord"][keep].astype(np.float64),
                               rank[keep].astype(np.float64))
    # NT backward shares
    nt_src = ctx["nt"][src]
    shares = {}
    for nm in CONFIG["nts"]:
        i = ctx["ntv"].index(nm)
        m = nt_src == i
        tot = float(w[m].sum())
        shares[nm] = 100 * float(w[m & ~ff].sum()) / tot
    out["nt_backward_share"] = shares
    # type contiguity proxy
    iqrs = []
    for idxs in ctx["big_types"]:
        q1, q3 = np.percentile(rank[idxs], [25, 75])
        iqrs.append((q3 - q1) / ctx["n"])
    out["type_iqr_over_n_median"] = float(np.median(iqrs))
    return out


def main():
    t0 = time.time()
    from layering import io_utils
    from layering.tools.fly_streams import assign_lanes
    from src.mfas.metrics import score_from_order, pct

    g, order, rank0, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    src, tgt, w = g.src, g.tgt, g.weight
    Wtot = float(w.sum())
    n = rank0.shape[0]
    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank0), "annotation cache is stale"

    lane, lane_list = assign_lanes(rank0, d["super_class"], d["cell_class"],
                                   meta["super_class_values"],
                                   meta["cell_class_values"])
    flow_v = meta["flow_values"]
    ford = {"afferent": 0, "intrinsic": 1, "efferent": 2}
    flow_ord = np.array([ford.get(flow_v[c], -1) for c in d["flow"]])
    # big types from the artifact bundle (same fallback naming as everywhere)
    import base64
    art = json.loads((HERE.parent / "outputs" / "fly_artifact" /
                      "artifact_data.json").read_text(encoding="utf-8"))
    t16 = np.frombuffer(base64.b64decode(art["cloud"]["type_u16"]),
                        dtype=np.uint16)
    counts = np.bincount(t16)
    big = [np.where(t16 == i)[0] for i in np.where(counts >= CONFIG["type_min_n"])[0]
           if art["type_names"][i] != "unannotated"]

    ctx = {"src": src, "tgt": tgt, "w": w, "lane": lane, "K": len(lane_list),
           "lane_idx": {nm: i for i, nm in enumerate(lane_list)},
           "champ_med": np.array([np.median(rank0[lane == i])
                                  for i in range(len(lane_list))]),
           "flow_ord": flow_ord, "nt": d["top_nt"].astype(np.int64),
           "ntv": meta["top_nt_values"], "big_types": big, "n": n}

    rng = np.random.default_rng(CONFIG["seed"])

    def degrade_to(target):
        r = rank0.copy()
        pos = np.argsort(r, kind="stable")          # pos[k] = node at rank k
        batches = 0
        while batches < CONFIG["max_batches"]:
            for _ in range(CONFIG["batch"]):
                a = int(rng.integers(0, n - CONFIG["window"]))
                sl = pos[a:a + CONFIG["window"]].copy()
                rng.shuffle(sl)
                pos[a:a + CONFIG["window"]] = sl
            r = np.empty(n, dtype=rank0.dtype)
            r[pos] = np.arange(n)
            s = pct(score_from_order(r, src, tgt, w), Wtot)
            batches += 1
            if s <= target:
                return r, s, batches
        return r, s, batches

    orders = {"champion": (rank0, ff_pct, 0)}
    for tgt_ff in CONFIG["targets_ff_pct"]:
        r, s, b = degrade_to(tgt_ff)
        orders[f"degraded_{int(tgt_ff)}"] = (r, s, b)
        print(f"degraded to {s:.2f}% in {b} batches")
    rr = np.arange(n); rng.shuffle(rr)
    orders["random"] = (rr, pct(score_from_order(rr, src, tgt, w), Wtot), 0)

    results = {"config": {k: v for k, v in CONFIG.items()
                          if not isinstance(v, Path)}}
    for name, (r, s, b) in orders.items():
        m = metrics_for(r, ctx)
        m["ff_pct"] = s
        m["shuffle_batches"] = b
        results[name] = m
        print(f"\n{name}: FF={s:.2f}%  laneSpearman={m['lane_median_spearman_vs_champion']:+.3f}  "
              f"staircaseInv={m['staircase_adjacent_inversions']}  flowRho={m['flow_rho']:+.3f}")
        print("  NT backward shares:", {k: round(v, 1) for k, v in m["nt_backward_share"].items()},
              " typeIQR/n:", round(m["type_iqr_over_n_median"], 4))

    results["seconds"] = time.time() - t0
    CONFIG["out"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["out"].write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {results['seconds']:.0f}s -> {CONFIG['out']}")


if __name__ == "__main__":
    main()
