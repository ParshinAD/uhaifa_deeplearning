"""Item B — mine the fly 17x17 lane matrix for counter-hierarchy and one-way pairs.

The fly streams study computed a full lane x lane weight matrix
(layering/outputs/fly_streams/lane_matrix.json) but only ever quoted one cell
(VPN -> visual centrifugal, 24.7 % FF). This script mines the whole matrix:

  * semantics (verified against the champion): W_pct[i][j] = weight of lane pair
    i->j as percent of TOTAL graph weight; W_ff_pct[i][j] = the FEEDFORWARD part
    of that weight, also as percent of total (sum of W_ff_pct = 84.2582, the
    champion FF). So pair FF share = W_ff_pct / W_pct, and pair backward weight
    = W_pct - W_ff_pct.

  1. "Counter-hierarchy" pairs: weight >= min_weight_pct of total AND FF share
     < counter_ff_max — lane pairs the optimal order largely refuses to orient.
  2. "One-way street" pairs: weight >= min_weight_pct AND FF share >= oneway_ff_min.
  3. A feedback budget: every pair's backward weight grouped into pathway families
     (visual / central / periphery-motor; intra-lane vs between-lane), answering
     "where does the fly's 15.7 % backward weight live, by pathway?".

No optimizer, no oracle: pure re-aggregation of an already-verified matrix
(its FF column sums to the pinned H64 champion FF% to 1e-9, asserted below).

Run:  $PY biology/analysis/exp_untapped_b_lane_mining.py
Outputs: biology/results/untapped/untapped_B_lane_mining.json
         biology/outputs/untapped_B_fb_budget.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

CONFIG = dict(
    lane_matrix=str(REPO / "layering/outputs/fly_streams/lane_matrix.json"),
    champion_ff_pct=84.25817950936937,   # pinned H64 (asserted against the matrix)
    min_weight_pct=0.1,                  # pair must carry >= 0.1 % of total weight
    counter_ff_max=65.0,                 # "counter-hierarchy" threshold (FF share %)
    oneway_ff_min=97.0,                  # "one-way street" threshold (FF share %)
    out_json=str(REPO / "biology/results/untapped/untapped_B_lane_mining.json"),
    out_png=str(REPO / "biology/outputs/untapped_B_fb_budget.png"),
    dpi=120,
    seed=42,                             # no RNG used; logged for the record
)

# Pathway families over the 17 bio-lanes (fly streams bio17 layout).
FAMILY = {
    "photoreceptors": "visual", "lamina": "visual", "medulla": "visual",
    "medulla→lobula": "visual", "lobula (plate)": "visual",
    "optic (other)": "visual", "visual projection": "visual",
    "visual centrifugal": "visual",
    "olfaction: AL·LH": "central", "mushroom body": "central",
    "central complex": "central", "central (other)": "central",
    "endocrine": "central",
    "ascending (VNC)": "periphery-motor", "sensory (other)": "periphery-motor",
    "descending": "periphery-motor", "motor": "periphery-motor",
}


def budget_category(src_lane: str, tgt_lane: str) -> str:
    """Assign a lane pair's backward weight to one budget category."""
    fs, ft = FAMILY[src_lane], FAMILY[tgt_lane]
    if src_lane == tgt_lane:
        return f"within one {fs} lane (recurrence)"
    if fs == ft == "visual":
        return "between visual lanes (retro-pathways)"
    if fs == ft == "central":
        return "between central lanes"
    if fs == ft == "periphery-motor":
        return "between periphery-motor lanes"
    if {fs, ft} == {"visual", "central"}:
        return "visual ↔ central"
    return "involving periphery-motor"


def main() -> None:
    d = json.loads(Path(CONFIG["lane_matrix"]).read_text())
    lanes = d["lane_list"]
    W = np.array(d["W_pct"])            # pct of total weight, src row -> tgt col
    F = np.array(d["W_ff_pct"])         # its FF part, pct of total weight
    assert abs(F.sum() - CONFIG["champion_ff_pct"]) < 1e-9, "matrix != champion FF"
    assert abs(W.sum() - 100.0) < 1e-9
    assert set(lanes) == set(FAMILY), "lane list drifted from FAMILY map"

    FB = W - F                          # backward weight per pair, pct of total
    total_fb_pct = FB.sum()             # = 100 - 84.2582 = 15.742

    pairs = []
    for i, ls in enumerate(lanes):
        for j, lt in enumerate(lanes):
            if W[i, j] <= 0:
                continue
            pairs.append(dict(
                src=ls, tgt=lt, intra=(i == j),
                weight_pct_of_total=float(W[i, j]),
                ff_share_pct=float(100.0 * F[i, j] / W[i, j]),
                fb_pct_of_total=float(FB[i, j]),
                fb_share_of_all_fb_pct=float(100.0 * FB[i, j] / total_fb_pct),
                category=budget_category(ls, lt),
            ))

    heavy = [p for p in pairs if p["weight_pct_of_total"] >= CONFIG["min_weight_pct"]]
    counter = sorted((p for p in heavy if p["ff_share_pct"] < CONFIG["counter_ff_max"]),
                     key=lambda p: p["ff_share_pct"])
    oneway = sorted((p for p in heavy if p["ff_share_pct"] >= CONFIG["oneway_ff_min"]),
                    key=lambda p: -p["ff_share_pct"])
    top_fb = sorted(pairs, key=lambda p: -p["fb_pct_of_total"])[:25]

    # Feedback budget by category (shares of ALL backward weight).
    budget: dict[str, float] = {}
    for p in pairs:
        budget[p["category"]] = budget.get(p["category"], 0.0) + p["fb_share_of_all_fb_pct"]
    budget = dict(sorted(budget.items(), key=lambda kv: -kv[1]))
    assert abs(sum(budget.values()) - 100.0) < 1e-6

    # Named visual retro-pathway aggregate: between-visual-lane pairs where the
    # source lane sits ANATOMICALLY deeper than the target (retina->lamina->
    # medulla->lobula/lobula plate->VPN); visual centrifugal is by definition
    # retrograde (central-brain neurons projecting INTO the optic lobe), counted
    # via its family membership above, listed separately here.
    depth = {"photoreceptors": 0, "lamina": 1, "medulla": 2, "medulla→lobula": 3,
             "lobula (plate)": 4, "optic (other)": 4, "visual projection": 5,
             "visual centrifugal": 2.5}
    retro_named = [p for p in pairs
                   if (not p["intra"]) and FAMILY[p["src"]] == FAMILY[p["tgt"]] == "visual"
                   and depth[p["src"]] > depth[p["tgt"]]]
    retro_named.sort(key=lambda p: -p["fb_pct_of_total"])

    out = dict(
        config=CONFIG,
        total_fb_pct_of_total_weight=float(total_fb_pct),
        counter_hierarchy_pairs=counter,
        one_way_pairs=oneway,
        top_fb_pairs=top_fb,
        fb_budget_by_category_pct_of_all_fb=budget,
        visual_retro_named_pairs=retro_named,
        visual_retro_named_fb_share_of_all_fb_pct=float(
            sum(p["fb_share_of_all_fb_pct"] for p in retro_named)),
        family_map=FAMILY,
    )
    Path(CONFIG["out_json"]).write_text(json.dumps(out, indent=1))

    # ── figure: one horizontal bar chart of the budget ────────────────────────
    SURFACE, INK, INK2, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    cats = list(budget.keys())[::-1]
    vals = [budget[c] for c in cats]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=CONFIG["dpi"])
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(cats, vals, height=0.55, color=BLUE, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", fontsize=9, color=INK2)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("share of ALL backward weight (%)", color=INK2, fontsize=10)
    ax.set_title("Fly connectome: where the 15.7% backward weight lives\n"
                 "(champion H64 order, 17 bio-lanes grouped into pathway families)",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.xaxis.grid(True, color="#e8e7e2", zorder=0)
    fig.tight_layout()
    fig.savefig(CONFIG["out_png"], facecolor=SURFACE)
    print(f"wrote {CONFIG['out_png']}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"total FB = {total_fb_pct:.3f}% of all weight")
    print(f"\ncounter-hierarchy pairs (W>={CONFIG['min_weight_pct']}%, "
          f"FF<{CONFIG['counter_ff_max']}%): {len(counter)}")
    for p in counter:
        print(f"  {p['src']:>18} -> {p['tgt']:<18} FF {p['ff_share_pct']:5.1f}%  "
              f"W {p['weight_pct_of_total']:.3f}%  FB share {p['fb_share_of_all_fb_pct']:.2f}%")
    print(f"\none-way pairs (W>={CONFIG['min_weight_pct']}%, "
          f"FF>={CONFIG['oneway_ff_min']}%): {len(oneway)}")
    for p in oneway:
        print(f"  {p['src']:>18} -> {p['tgt']:<18} FF {p['ff_share_pct']:5.1f}%  "
              f"W {p['weight_pct_of_total']:.3f}%")
    print("\nFB budget by category (% of all FB):")
    for c, v in budget.items():
        print(f"  {v:6.2f}%  {c}")
    print(f"\nnamed visual retro pairs FB share: "
          f"{out['visual_retro_named_fb_share_of_all_fb_pct']:.2f}% of all FB")
    for p in retro_named[:8]:
        print(f"  {p['src']:>18} -> {p['tgt']:<18} FF {p['ff_share_pct']:5.1f}%  "
              f"FB share {p['fb_share_of_all_fb_pct']:.2f}%")
    print(f"\nwrote {CONFIG['out_json']}")


if __name__ == "__main__":
    main()
