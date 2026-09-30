"""Item A — who are the fly graph's hubs and feedback carriers? (untapped mining)

The gallery report (layering/outputs/gallery_fly/report.json) established that the
highest-degree node is graph index 78384 (degree 16,784, pi-rank 59,048 of 136,648)
but never looked up its identity. This script maps graph indices -> FlyWire root_ids
(via GraphData.node_ids) and joins them against the community annotation table
(data/external/flywire_neuron_annotations.tsv.gz) for three node lists:

  1. top-K nodes by total degree (in-edges + out-edges, unweighted),
  2. top-K sources of feedback weight (sum of w over edges with rank[src] > rank[tgt],
     grouped by source),
  3. top-K receivers of feedback weight (same edges, grouped by target).

"Feedback" is defined by the pinned champion order (H64, 84.2582 % FF), loaded through
layering.io_utils.load_graph_and_champion_order, which parity-gates the order against
the frozen oracle (src/mfas/metrics.py) before returning it. Nothing here touches the
optimizer; this is pure annotation lookup on an already-verified order.

Run:  $PY biology/analysis/exp_untapped_a_fly_hubs.py
Output: biology/results/untapped/untapped_A_fly_hubs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = dict(
    dataset="connectome",
    ann_tsv=str(REPO / "data/external/flywire_neuron_annotations.tsv.gz"),
    ann_cols=["root_id", "flow", "super_class", "cell_class", "cell_sub_class",
              "cell_type", "hemibrain_type", "top_nt", "side"],
    top_k=12,                    # rows per table (10 asked; 12 for a safety margin)
    focus_index=78384,           # the known max-degree node from the gallery report
    out_json=str(REPO / "biology/results/untapped/untapped_A_fly_hubs.json"),
    seed=42,                     # no RNG used; logged for the record
)


def annotate(indices: np.ndarray, node_ids: np.ndarray, ann: pd.DataFrame,
             extra: dict[str, np.ndarray]) -> list[dict]:
    """Return one JSON-ready row per graph index, joined with the annotation table.

    ``extra`` maps column name -> full-length per-node array; values are sliced at
    each index. Annotation fields missing for a root_id come back as None.
    """
    rows = []
    for idx in indices:
        rid = int(node_ids[idx])
        row = {"graph_index": int(idx), "root_id": rid}
        for k, arr in extra.items():
            v = arr[idx]
            row[k] = int(v) if np.issubdtype(type(v), np.integer) else float(v)
        if rid in ann.index:
            a = ann.loc[rid]
            for c in CONFIG["ann_cols"][1:]:
                val = a[c]
                row[c] = None if (isinstance(val, float) and np.isnan(val)) else str(val)
        else:
            row["annotation"] = "root_id not in annotation table"
        rows.append(row)
    return rows


def main() -> None:
    from layering import io_utils

    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.int64)

    # Degree = number of incident edges (in + out), matching the gallery report.
    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)

    # Feedback edges under the champion order (strictly backward; ties impossible
    # for a permutation rank).
    fb = rank[src] > rank[tgt]
    fb_w_out = np.zeros(n, dtype=np.int64)
    fb_w_in = np.zeros(n, dtype=np.int64)
    np.add.at(fb_w_out, src[fb], w[fb])
    np.add.at(fb_w_in, tgt[fb], w[fb])
    total_fb = int(w[fb].sum())

    # Total in/out strength for context.
    s_out = np.zeros(n, dtype=np.int64)
    s_in = np.zeros(n, dtype=np.int64)
    np.add.at(s_out, src, w)
    np.add.at(s_in, tgt, w)

    ann = pd.read_csv(CONFIG["ann_tsv"], sep="\t", usecols=CONFIG["ann_cols"],
                      low_memory=False)
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")

    k = CONFIG["top_k"]
    extra = {
        "degree": deg, "pi_rank": rank,
        "pi_rank_frac": rank / (n - 1.0),
        "fb_w_out": fb_w_out, "fb_w_in": fb_w_in,
        "fb_out_share_pct": 100.0 * fb_w_out / max(total_fb, 1),
        "fb_in_share_pct": 100.0 * fb_w_in / max(total_fb, 1),
        "strength_out": s_out, "strength_in": s_in,
    }
    top_deg = np.argsort(-deg)[:k]
    top_fb_src = np.argsort(-fb_w_out)[:k]
    top_fb_tgt = np.argsort(-fb_w_in)[:k]

    assert top_deg[0] == CONFIG["focus_index"], (
        f"max-degree node is {top_deg[0]}, expected {CONFIG['focus_index']} "
        f"from the gallery report")

    out = {
        "config": CONFIG,
        "champion_ff_pct": ff_pct,
        "total_weight": g.total_weight,
        "total_fb_weight": total_fb,
        "fb_pct_of_total": 100.0 * total_fb / g.total_weight,
        "top_degree": annotate(top_deg, g.node_ids, ann, extra),
        "top_fb_senders": annotate(top_fb_src, g.node_ids, ann, extra),
        "top_fb_receivers": annotate(top_fb_tgt, g.node_ids, ann, extra),
    }
    Path(CONFIG["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["out_json"]).write_text(json.dumps(out, indent=1))
    print(f"champion FF {ff_pct:.4f}%  total FB weight {total_fb:,} "
          f"({out['fb_pct_of_total']:.3f}% of total)")
    for name in ("top_degree", "top_fb_senders", "top_fb_receivers"):
        print(f"\n== {name} ==")
        for r in out[name]:
            print(f"  idx {r['graph_index']:>6} deg {r['degree']:>6} "
                  f"rank {r['pi_rank']:>6} ({100*r['pi_rank_frac']:.1f}%) "
                  f"fb_out {r['fb_w_out']:>7} fb_in {r['fb_w_in']:>7}  "
                  f"{r.get('super_class')}/{r.get('cell_class')}/"
                  f"{r.get('cell_type')}/{r.get('hemibrain_type')} "
                  f"nt={r.get('top_nt')} side={r.get('side')}")
    print(f"\nwrote {CONFIG['out_json']}")


if __name__ == "__main__":
    main()
