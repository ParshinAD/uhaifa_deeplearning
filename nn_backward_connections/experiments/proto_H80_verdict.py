"""Evaluate H80's rung-1 kill conditions MECHANICALLY from the artifacts.

Reads ``experiments/outputs/proto_H80_S{1r,2r,3r}.json`` and applies K1-K5 exactly as
they are written in the sealed pre-registration and its amendments:

* ``proto_H80_prereg.json``            - the arms, the instances, K1-K4
* ``proto_H80_prereg_amendment.json``  - the true ruin-and-recreate, delta_ACCEPT as the
                                         primary, K2/K3 restated on it
* ``proto_H80_prereg_amendment4.json`` - the sideways-drift control and K5
* ``proto_H80_prereg_amendment5.json`` - the calibration fix; S1/S2/S3 superseded by
                                         S1r/S2r/S3r

Writes ``experiments/outputs/proto_H80_verdict.json``. The cycle's verdict is this file's
output, not a reading of the numbers by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "experiments" / "outputs"

K2_BAR_PP = 0.05          # pre-registered minimum effect size on S1
K5_RECOVERY_FRAC = 0.70   # amendment 4: sideways drift recovering >= this kills H80


def load(stage):
    p = OUT / f"proto_H80_{stage}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    s1, s2, s3 = load("S1r"), load("S2r"), load("S3r")
    verdict = {
        "_doc": "MECHANICAL evaluation of H80's rung-1 kill conditions against the "
                "sealed pre-registration and amendments 4 and 5. Produced by "
                "experiments/proto_H80_verdict.py from the S1r/S2r/S3r artifacts.",
        "item": "H80", "cycle": 23,
        "sources": {k: (f"proto_H80_{k}.json" if v else "MISSING")
                    for k, v in (("S1r", s1), ("S2r", s2), ("S3r", s3))},
        "checks": {}, "stages": {},
    }
    if not (s1 and s2):
        verdict["verdict"] = "INCOMPLETE - S1r and S2r are both required"
        (OUT / "proto_H80_verdict.json").write_text(json.dumps(verdict, indent=2),
                                                    encoding="utf-8")
        print(json.dumps(verdict, indent=2))
        return

    for name, d in (("S1r", s1), ("S2r", s2), ("S3r", s3)):
        if not d:
            continue
        su = d["summary"]
        verdict["stages"][name] = {
            "best_cell": su["best_cell"],
            "delta_ACCEPT_pp": su["best_cell_mean_delta"],
            "delta_A3_pp_preregistered": su["mean_delta_A3"],
            "structure_alone_pp": su["mean_structure_alone"],
            "acceptance_alone_pp": su.get("mean_acceptance_alone"),
            "sideways_alone_pp": su.get("mean_sideways_alone"),
            "best_sideways_mean_delta_pp": su.get("best_sideways_mean_delta"),
            "K5_sideways_recovery_fraction": su.get("K5_sideways_recovery_fraction"),
            "temperature_buys_pp": su.get("K5_temperature_buys_pp"),
            "n_adjacent_T0_pairs_positive": su[
                "n_adjacent_T0_pairs_positive_in_best_family"],
            "uphill_accepted_by_T0_A3": su["total_uphill_accepted_by_T0"],
            "max_excursion_pp_by_T0_A3": su["max_excursion_pp_by_T0"],
            "mean_barrier_depth_pp": su.get("mean_barrier_depth_pp"),
            "mean_delta_by_cell": su["mean_delta_by_cell"],
            "calibration_health": sorted({
                a.get("cal_source") for r in d["rows"] for a in r["arms"].values()
                if a.get("cal_source")}),
        }

    c = verdict["checks"]
    su1, su2 = s1["summary"], s2["summary"]

    # K1 - did the schedule ever cross anything on S1?
    tot_up = sum(su1["total_uphill_accepted_by_T0"].values())
    up_a4 = sum(sum(r["arms"][f"A4_topk_anneal_T{m}"]["n_uphill_accepted"]
                    for r in s1["rows"]) for m in ("0.5", "1.0", "2.0", "4.0", "8.0"))
    c["K1_void_if_no_uphill"] = {
        "uphill_accepted_A3": tot_up, "uphill_accepted_A4": up_a4,
        "fires": bool(tot_up + up_a4 == 0),
        "reading": "VOID (mis-sized schedule)" if tot_up + up_a4 == 0 else "not void",
    }
    # K2 - effect size on S1
    c["K2_effect_size_S1"] = {
        "delta_ACCEPT_pp": su1["best_cell_mean_delta"], "bar_pp": K2_BAR_PP,
        "fires": bool(su1["best_cell_mean_delta"] < K2_BAR_PP),
    }
    # K3 - is the winning temperature a singleton?
    c["K3_singleton_temperature"] = {
        "n_adjacent_positive_pairs": su1[
            "n_adjacent_T0_pairs_positive_in_best_family"],
        "fires": bool(su1["n_adjacent_T0_pairs_positive_in_best_family"] < 1),
    }
    # K4 - does it survive the 10x scale-up?
    c["K4_scale_up_S2"] = {
        "delta_ACCEPT_pp": su2["best_cell_mean_delta"],
        "fires": bool(su2["best_cell_mean_delta"] <= 0),
    }
    # K5 - is it barrier crossing, or plateau drift?
    rec = su1.get("K5_sideways_recovery_fraction")
    c["K5_drift_not_barrier_crossing"] = {
        "sideways_recovery_fraction_S1": rec,
        "threshold": K5_RECOVERY_FRAC,
        "temperature_buys_pp_S1": su1.get("K5_temperature_buys_pp"),
        "fires": bool(rec is not None and rec >= K5_RECOVERY_FRAC),
    }

    fired = [k for k, v in c.items() if v.get("fires")]
    verdict["kill_conditions_fired"] = fired
    verdict["verdict"] = "KILL" if fired else "RUNG 1 PASSES -> rung 2"
    (OUT / "proto_H80_verdict.json").write_text(json.dumps(verdict, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"verdict": verdict["verdict"],
                      "fired": fired, "checks": c}, indent=2))


if __name__ == "__main__":
    main()
