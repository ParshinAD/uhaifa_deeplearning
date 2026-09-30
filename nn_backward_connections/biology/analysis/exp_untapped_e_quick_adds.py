"""Item E — three quick adds, each one paragraph with a number.

E1  MICrONS within-class recurrence: E->E and I->I feedforward shares (computed by
    the gallery run, layering/outputs/microns_gallery/report_gallery.json, never
    interpreted), pulled together with the proofread-control shares from item D.

E2  The MICrONS "unclassified" lane (ei==0 & not white-matter band; n=4,202,
    median rank ~18.1k = early). Is it early for the interneuron arithmetic
    (genuinely heavy forward output) or is it reconstruction debris (orphan
    fragments with outputs but no inputs, or multi-soma merge bodies)? Computed
    fresh from the canonical graph + annotation cache: per-group degrees,
    in/out weights, zero-output / zero-input shares, multi-soma shares.

E3  The fly trophic-order gap: numbers copied (with provenance) from the logged
    artifacts layering/outputs/gallery_fly/report.json and
    layering/outputs/diagnostics_mouse.txt — no re-run needed, both are
    re-runnable via layering/tools/{gallery_fly.py,diagnostics.py}.

Run:  $PY biology/analysis/exp_untapped_e_quick_adds.py
Output: biology/results/untapped/untapped_E_quick_adds.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = dict(
    ann_npz=str(REPO / "layering/outputs/microns_ann/node_ann.npz"),
    microns_npz=str(REPO / "data/processed/microns.npz"),
    gallery_json=str(REPO / "layering/outputs/microns_gallery/report_gallery.json"),
    proofread_json=str(REPO / "biology/results/untapped/untapped_D_proofread_control.json"),
    fly_report=str(REPO / "layering/outputs/gallery_fly/report.json"),
    mouse_diag=str(REPO / "layering/outputs/diagnostics_mouse.txt"),
    wm_layer_code=6,          # layer int8: 0 NA/1 L1/2 L2-3/3 L4/4 L5/5 L6/6 WM
    out_json=str(REPO / "biology/results/untapped/untapped_E_quick_adds.json"),
    seed=42,                  # no RNG used; logged for the record
)


def group_stats(mask: np.ndarray, rank, out_w, in_w, out_deg, in_deg,
                n_somas) -> dict:
    """Per-group medians and shares used by the E2 fragments-vs-interneurons test."""
    m = mask
    return dict(
        n=int(m.sum()),
        median_rank=float(np.median(rank[m])),
        median_out_deg=float(np.median(out_deg[m])),
        median_in_deg=float(np.median(in_deg[m])),
        median_out_w=float(np.median(out_w[m])),
        median_in_w=float(np.median(in_w[m])),
        agg_out_over_in_w=float(out_w[m].sum() / max(in_w[m].sum(), 1)),
        zero_output_share=float((out_w[m] == 0).mean()),
        zero_input_share=float((in_w[m] == 0).mean()),
        multi_soma_share=float((n_somas[m] > 1).mean()),
    )


def main() -> None:
    ann = np.load(CONFIG["ann_npz"])
    rank, ei, layer = ann["rank"], ann["ei"], ann["layer"]
    out_w, in_w, n_somas = ann["out_w"], ann["in_w"], ann["n_somas"]
    n = rank.shape[0]

    with np.load(CONFIG["microns_npz"]) as g:
        src, tgt = g["src"], g["tgt"]
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(tgt, minlength=n)

    wm = layer == CONFIG["wm_layer_code"]
    groups = {
        "unclassified_lane": (ei == 0) & ~wm,
        "inhibitory": (ei == 2) & ~wm,
        "excitatory": (ei == 1) & ~wm,
        "white_matter_band": wm,
        "all": np.ones(n, dtype=bool),
    }
    e2 = {k: group_stats(m, rank, out_w, in_w, out_deg, in_deg, n_somas)
          for k, m in groups.items()}
    assert e2["unclassified_lane"]["n"] == 4202, \
        f"unclassified lane n drifted: {e2['unclassified_lane']['n']}"

    # Decompose the unclassified lane: merge bodies / zero-input orphans / rest.
    u = groups["unclassified_lane"]
    sub = {
        "multi_soma": u & (n_somas > 1),
        "zero_input_single_soma": u & (n_somas <= 1) & (in_w == 0),
        "remaining_single_soma": u & (n_somas <= 1) & (in_w > 0),
    }
    e2["unclassified_decomposition"] = {
        k: group_stats(m, rank, out_w, in_w, out_deg, in_deg, n_somas)
        for k, m in sub.items()}

    gal = json.loads(Path(CONFIG["gallery_json"]).read_text())
    e1 = dict(source=CONFIG["gallery_json"],
              full_graph=gal["ei_pair_budget"],
              proofread_control=json.loads(
                  Path(CONFIG["proofread_json"]).read_text()
              )["ei_pair_budget_under_direct_opt"])

    fly = json.loads(Path(CONFIG["fly_report"]).read_text())
    mouse_txt = Path(CONFIG["mouse_diag"]).read_text()
    m_ff = float(re.search(r"trophic order: ([\d.]+)%", mouse_txt).group(1))
    m_champ = float(re.search(r"champion pi: ([\d.]+)%", mouse_txt).group(1))
    m_rho = float(re.search(r"spearman\(pi, trophic\)=([\d.]+)", mouse_txt).group(1))
    e3 = dict(
        fly=dict(source=CONFIG["fly_report"],
                 champion_ff_pct=fly["champion_ff_pct"],
                 trophic_ff_pct=fly["trophic"]["ff_pct_under_trophic_order"],
                 gap_pp=fly["champion_ff_pct"] - fly["trophic"]["ff_pct_under_trophic_order"],
                 spearman=fly["trophic"]["spearman_vs_pi"],
                 F0=fly["trophic"]["F0"], max_level=fly["trophic"]["max_level"]),
        mouse=dict(source=CONFIG["mouse_diag"],
                   champion_ff_pct=m_champ, trophic_ff_pct=m_ff,
                   gap_pp=m_champ - m_ff, spearman=m_rho),
    )

    out = dict(config=CONFIG, E1_ei_recurrence=e1, E2_unclassified_lane=e2,
               E3_trophic_gap=e3)
    Path(CONFIG["out_json"]).write_text(json.dumps(out, indent=1))

    print("E1 full graph:", {k: round(v["ff_share_pct"], 1)
                             for k, v in gal["ei_pair_budget"].items()})
    print("E1 proofread :", {k: (round(v["ff_share_pct"], 1)
                                 if v["ff_share_pct"] else None)
                             for k, v in e1["proofread_control"].items()})
    print("\nE2 groups:")
    hdr = ("group", "n", "med_rank", "outdeg", "indeg", "out_w", "in_w",
           "out/in", "zero_out", "zero_in", "multi_soma")
    print(("{:>18}" + "{:>9}" * 10).format(*hdr))
    flat = {**{k: v for k, v in e2.items() if k != "unclassified_decomposition"},
            **{f"uncl:{k}": v
               for k, v in e2["unclassified_decomposition"].items()}}
    for k, s in flat.items():
        print(f"{k:>28}{s['n']:>9}{s['median_rank']:>9.0f}"
              f"{s['median_out_deg']:>9.0f}{s['median_in_deg']:>9.0f}"
              f"{s['median_out_w']:>9.0f}{s['median_in_w']:>9.0f}"
              f"{s['agg_out_over_in_w']:>9.2f}{s['zero_output_share']:>9.3f}"
              f"{s['zero_input_share']:>9.3f}{s['multi_soma_share']:>9.3f}")
    print("\nE3 trophic gaps: fly %.2fpp (rho %.3f), mouse %.2fpp (rho %.3f)"
          % (e3["fly"]["gap_pp"], e3["fly"]["spearman"],
             e3["mouse"]["gap_pp"], e3["mouse"]["spearman"]))
    print(f"\nwrote {CONFIG['out_json']}")


if __name__ == "__main__":
    main()
