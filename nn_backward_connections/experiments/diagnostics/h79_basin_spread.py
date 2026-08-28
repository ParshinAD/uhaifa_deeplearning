"""H79 rung 2 - the SPREAD of refined scores across structurally different basins.

Reads the harness records the screen produced (``results/*-H79*-implement-*.json``) and
computes the ONE quantity H79 was filed to produce: how far apart the champion stack's
FINAL connectome scores are when it is started from constructions that rung 1 showed to
lie outside the champion's own relabelling ball.

It computes no new science; it aggregates. Every figure it prints traces to a
``results/*.json`` written by ``eval/run_variant.py``.

PRE-REGISTERED KILL CONDITION (proto_H79_prereg.json, sealed at 894492e)
------------------------------------------------------------------------
    spread <= 0.019124 pp  ->  the stack ERASES the basin; that is M2/M6's shape and the
                               axis closes.
where ``spread`` = max(final pct over arms) - min(final pct over arms), arms = the three
ADMISSIBLE distinct constructions plus the identity anchor, at the same seed.

VALIDITY GATE, checked first and reported as PASS/VOID
------------------------------------------------------
The identity anchor (H79, construction ``greedy_fas``) must reproduce the H64 champion,
84.25817950936937, bit-for-bit. If it does not, the arms measure this module's refactor
rather than the basin, and every number below is void - the same check H57's prefix study
used to establish that it was measuring the champion's own pipeline.

Run:
    PYTHONPATH=src python experiments/diagnostics/h79_basin_spread.py \
        --out experiments/outputs/proto_H79_rung2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

CHAMPION_CONNECTOME = 84.25817950936937      # sota.json datasets.connectome.pct_mean_exact
NUISANCE_SIGMA_PP = 0.019124                 # M10 / proto_P09_connectome.json
MIN_EFFECT_PP = 0.012                        # campaign.yaml datasets.connectome.min_promotion_delta_pp

ARMS = {
    "H79": "greedy_fas (identity anchor)",
    "H79A": "reverse_greedy_fas",
    "H79B": "scc_topo",
    "H79C": "imbalance_sort",
}


def _load(variant: str, dataset: str, role: str) -> list:
    out = []
    for p in sorted(glob.glob(f"results/*-{variant}-{dataset}-*-{role}-*.json")):
        if p.endswith("_positions.npy"):
            continue
        with open(p) as fh:
            rec = json.load(fh)
        rec["_path"] = os.path.basename(p)
        out.append(rec)
    return out


def _pct(rec: dict) -> float:
    for k in ("pct", "best_pct", "refined_best_pct"):
        if k in rec and rec[k] is not None:
            return float(rec[k])
    raise KeyError(f"no percentage field in {rec.get('_path')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="implement")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_arm = {}
    for variant, label in ARMS.items():
        runs = _load(variant, "connectome", args.role)
        mouse = _load(variant, "mouse", args.role)
        if not runs:
            continue
        pcts = [_pct(r) for r in runs]
        per_arm[variant] = {
            "construction": label,
            "n_runs": len(runs),
            "seeds": sorted({r.get("seed") for r in runs}),
            "pct": pcts,
            "pct_mean": statistics.fmean(pcts),
            "pct_std": statistics.pstdev(pcts) if len(pcts) > 1 else 0.0,
            "distinct_values": len(set(pcts)),
            "delta_vs_champion_pp": statistics.fmean(pcts) - CHAMPION_CONNECTOME,
            "init_pct": [r.get("variant_attrs", {}).get("construction_init_pct") for r in runs],
            "pure_best_pct": [r.get("variant_attrs", {}).get("pure_best_pct") for r in runs],
            "sift_best_pct": [r.get("variant_attrs", {}).get("sift_best_pct") for r in runs],
            "wall_clock_s": [r.get("wall_clock_s") for r in runs],
            "degraded": [bool((r.get("runtime_guard") or {}).get("degraded")) for r in runs],
            "commit": sorted({r.get("git_commit") for r in runs}),
            "files": [r["_path"] for r in runs],
            "mouse_pct": [_pct(r) for r in mouse],
            "mouse_files": [r["_path"] for r in mouse],
        }

    anchor = per_arm.get("H79")
    anchor_ok = bool(anchor and all(abs(p - CHAMPION_CONNECTOME) < 1e-12 for p in anchor["pct"]))

    arm_ids = [a for a in ("H79A", "H79B", "H79C") if a in per_arm]
    means = {a: per_arm[a]["pct_mean"] for a in arm_ids}
    if anchor:
        means["H79"] = anchor["pct_mean"]
    spread = (max(means.values()) - min(means.values())) if len(means) > 1 else 0.0
    arms_only_spread = ((max(per_arm[a]["pct_mean"] for a in arm_ids)
                         - min(per_arm[a]["pct_mean"] for a in arm_ids))
                        if len(arm_ids) > 1 else 0.0)

    out = {
        "id": "H79-rung2",
        "role": args.role,
        "champion_connectome_pct": CHAMPION_CONNECTOME,
        "nuisance_sigma_pp": NUISANCE_SIGMA_PP,
        "min_effect_pp": MIN_EFFECT_PP,
        "validity": {
            "anchor_reproduces_champion": anchor_ok,
            "anchor_pct": anchor["pct"] if anchor else None,
            "verdict": "PASS" if anchor_ok else "VOID",
        },
        "per_arm": per_arm,
        "spread_incl_anchor_pp": spread,
        "spread_arms_only_pp": arms_only_spread,
        "spread_exceeds_nuisance_sigma": bool(spread > NUISANCE_SIGMA_PP),
        "best_arm": max(means, key=means.get) if means else None,
        "best_arm_delta_vs_champion_pp": (max(means.values()) - CHAMPION_CONNECTOME
                                          if means else None),
        "screen_gate_any_arm_passes": bool(
            means and (max(means.values()) - CHAMPION_CONNECTOME) > MIN_EFFECT_PP),
        "decision_rule": ("kill iff spread <= nuisance_sigma_pp (the stack erases the basin); "
                          "the SPREAD is the deliverable, the max is NOT a claim (M9/M10)"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"validity: {out['validity']['verdict']}  anchor={out['validity']['anchor_pct']}")
    for a, d in per_arm.items():
        print(f"{a:>5s} {d['construction']:>28s}  mean {d['pct_mean']:.8f} %  "
              f"n={d['n_runs']} distinct={d['distinct_values']}  "
              f"delta {d['delta_vs_champion_pp']:+.6f} pp  "
              f"wall {max(d['wall_clock_s'] or [0]):.0f} s")
    print(f"SPREAD (incl anchor) = {spread:.6f} pp   vs nuisance sigma {NUISANCE_SIGMA_PP} pp")
    print(f"SPREAD (arms only)   = {arms_only_spread:.6f} pp")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
