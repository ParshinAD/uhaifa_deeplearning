"""Item C — the mouse id->region gap: data-driven fingerprint + FB-carrier table.

data/table_mouse.txt (148 nodes, ids 1..149, float weights) has no region names, so
the mouse findings (RESEARCH_NOTES.md: the 40.5 %-of-feedback hub id 92, the giant
SCC, the long-range feedback) are biologically uninterpretable. The provenance
investigation (documented in biology/studies/untapped.md, item C) found NO
authoritative id->region mapping, so this script does what CAN be done honestly:

  1. a complete structural fingerprint of the file — enough to positively identify
     the source table if a candidate matrix ever surfaces (exact float weights are
     quoted for the three heaviest edges: floats survive copy-paste, names do not);
  2. the per-node feedback ledger under the pinned H63 champion order (93.1754 %
     FF, parity-gated): top feedback senders/receivers by ORIGINAL node id, so the
     day a mapping arrives the biological reading is one join away.

No optimizer; the frozen oracle only (via layering.io_utils' parity gate).

Run:  $PY biology/analysis/exp_untapped_c_mouse_provenance.py
Output: biology/results/untapped/untapped_C_mouse_fingerprint.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = dict(
    dataset="mouse",
    top_k=10,
    hub_id=92,                # RESEARCH_NOTES.md hub (original id)
    out_json=str(REPO / "biology/results/untapped/untapped_C_mouse_fingerprint.json"),
    seed=42,                  # no RNG used; logged for the record
)


def node_row(i: int, node_ids, rank, n, deg_in, deg_out, fb_in, fb_out,
             total_fb) -> dict:
    """JSON row for contiguous index ``i`` keyed by the ORIGINAL file id."""
    return dict(
        original_id=int(node_ids[i]), pi_rank=int(rank[i]),
        pi_rank_frac=float(rank[i] / (n - 1)),
        in_degree=int(deg_in[i]), out_degree=int(deg_out[i]),
        fb_w_in=float(fb_in[i]), fb_w_out=float(fb_out[i]),
        fb_touch_share_pct=float(100.0 * (fb_in[i] + fb_out[i]) / total_fb),
    )


def main() -> None:
    from layering import io_utils

    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    n = g.n_nodes
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.float64)
    ids = np.asarray(g.node_ids)

    # ── structural fingerprint ────────────────────────────────────────────────
    present = set(int(x) for x in ids)
    missing = sorted(set(range(min(present), max(present) + 1)) - present)
    pair = {(int(s), int(t)): float(wt) for s, t, wt in zip(src, tgt, w)}
    recip = sum(1 for k in pair if (k[1], k[0]) in pair)
    heavy_idx = np.argsort(-w)[:3]
    heaviest = [dict(src_id=int(ids[src[i]]), tgt_id=int(ids[tgt[i]]),
                     weight=repr(float(w[i]))) for i in heavy_idx]
    deg_out = np.bincount(src, minlength=n)
    deg_in = np.bincount(tgt, minlength=n)

    # ── feedback ledger under the champion order ─────────────────────────────
    fb = rank[src] > rank[tgt]
    fb_out = np.zeros(n); fb_in = np.zeros(n)
    np.add.at(fb_out, src[fb], w[fb])
    np.add.at(fb_in, tgt[fb], w[fb])
    total_fb = float(w[fb].sum())

    k = CONFIG["top_k"]
    args = (ids, rank, n, deg_in, deg_out, fb_in, fb_out, total_fb)
    top_senders = [node_row(i, *args) for i in np.argsort(-fb_out)[:k]]
    top_receivers = [node_row(i, *args) for i in np.argsort(-fb_in)[:k]]
    hub_i = int(np.where(ids == CONFIG["hub_id"])[0][0])
    hub = node_row(hub_i, *args)
    hub["degree_total"] = hub["in_degree"] + hub["out_degree"]

    out = dict(
        config=CONFIG,
        champion_ff_pct=float(ff_pct),
        fingerprint=dict(
            n_nodes=n, n_edges=g.n_edges,
            id_range=[int(ids.min()), int(ids.max())], missing_ids=missing,
            total_weight=repr(float(w.sum())),
            weight_min=repr(float(w.min())), weight_max=repr(float(w.max())),
            n_reciprocal_edges=recip,
            heaviest_edges_exact=heaviest,
            file_format="tab-separated, no header, src tgt weight (floats)",
        ),
        total_fb_weight=total_fb,
        fb_pct_of_total=float(100.0 * total_fb / w.sum()),
        hub=hub,
        top_fb_senders=top_senders,
        top_fb_receivers=top_receivers,
    )
    Path(CONFIG["out_json"]).write_text(json.dumps(out, indent=1))

    print(f"champion FF {ff_pct:.4f}%  FB {out['fb_pct_of_total']:.3f}% of weight")
    print(f"fingerprint: n={n} m={g.n_edges} ids {ids.min()}..{ids.max()} "
          f"missing {missing} recip {recip}")
    for e in heaviest:
        print(f"  heaviest: {e['src_id']} -> {e['tgt_id']}  w={e['weight']}")
    print(f"hub id {CONFIG['hub_id']}: rank {hub['pi_rank']} of {n} "
          f"({100*hub['pi_rank_frac']:.1f}%), in {hub['in_degree']} out "
          f"{hub['out_degree']}, touches {hub['fb_touch_share_pct']:.1f}% of FB")
    print("\ntop FB senders (original ids):")
    for r in top_senders:
        print(f"  id {r['original_id']:>3} rank {r['pi_rank']:>3} "
              f"fb_out {r['fb_w_out']:.4f} ({100*r['fb_w_out']/total_fb:.1f}% of FB)")
    print("top FB receivers (original ids):")
    for r in top_receivers:
        print(f"  id {r['original_id']:>3} rank {r['pi_rank']:>3} "
              f"fb_in {r['fb_w_in']:.4f} ({100*r['fb_w_in']/total_fb:.1f}% of FB)")
    print(f"\nwrote {CONFIG['out_json']}")


if __name__ == "__main__":
    main()
