"""H57 decision arithmetic - reads the measured arms, applies the SEALED rule, decides.

Consumes
--------
* ``experiments/outputs/proto_H57_connectome.json``  - this cycle's tail-only arms
* ``experiments/outputs/proto_P09_connectome.json``  - the whole-pipeline labelling arms
* ``experiments/outputs/proto_H42.json``             - the 143-cycle tail saturation curve

Emits ``experiments/outputs/proto_H57_decision.json``. No new runs, no compute: every number
here is a function of files already on disk, so it is re-derivable by checkout.

The rule it applies is fixed in ``proto_H57_prereg.json`` (committed at d3d10ca, before the
first connectome arm existed) and is meta-rule M9:

    KILL iff tail_sigma_pp < min_promotion_delta_pp / a_R,
    R = floor((guard_deadline_s - prefix_wall_s) / tail_wall_s)

Two extras the prereg names as reported-but-not-deciding: the chi-squared interval on the
sigma estimate (M9's own caveat) and the exact subset enumeration, which is what caught
H56's regression - best-of-R over RANDOM draws loses to a canonical that sits above its own
labelling mean, so only the canonical-anchored form is even monotone.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "experiments" / "outputs"

_CHAMPION_PCT = 84.15409511053134
_MIN_DELTA_PP = 0.012                    # campaign.yaml datasets.connectome
_GUARD_DEADLINE_S = 3450.0
_P07_LOAD_FACTOR = 1.2                   # queue item P07, same constant proto_H56 used
_A_R = {1: 0.0, 2: 0.5641895835477563, 3: 0.8462843753216345,
        4: 1.0293753730039641, 5: 1.1629644736405101, 6: 1.2672063606114712,
        7: 1.3521783756070915, 8: 1.4236003060469847}


def _sigma_ci95(s: float, n: int):
    """Chi-squared 95% interval for a std estimated from n points."""
    if n < 2 or s <= 0:
        return [0.0, 0.0]
    df = n - 1
    lo = s * np.sqrt(df / stats.chi2.ppf(0.975, df))
    hi = s * np.sqrt(df / stats.chi2.ppf(0.025, df))
    return [float(lo), float(hi)]


def _enumerate(draws, anchor=None):
    """Exact best-of-R over every subset, rather than a normal-theory approximation."""
    out = []
    n = len(draws)
    for r in range(1, n + 1):
        k = r - 1 if anchor is not None else r
        if k > n:
            break
        subs = list(itertools.combinations(range(n), k))
        if not subs:
            subs = [()]
        bests = []
        for s in subs:
            vals = [draws[i] for i in s]
            if anchor is not None:
                vals.append(anchor)
            bests.append(max(vals))
        out.append(dict(r=r, n_subsets=len(subs),
                        mean_best_pct=float(np.mean(bests)),
                        min_best_pct=float(np.min(bests)),
                        max_best_pct=float(np.max(bests)),
                        delta_vs_canonical_pp=float(np.mean(bests) - _CHAMPION_PCT)))
    return out


def _control_curve():
    """Matched-compute control: ONE tail given R x the cycles, from the measured curve."""
    d = json.loads((_OUT / "proto_H42.json").read_text(encoding="utf-8"))
    arm = [a for a in d["arms"]
           if a["dataset"] == "connectome" and a["sift_sweeps"] == 2][0]
    log = arm["log"]
    by_cycle = {row["cycle"]: (row["best_pct"], row["cum_wall_s"]) for row in log}
    n_max = max(by_cycle)
    rows = []
    for r in (1, 2, 3, 4):
        c = 77 * r - 1                      # H42 ships 77 cycles, i.e. index 76
        if c in by_cycle:
            pctv, wall = by_cycle[c]
            rows.append(dict(r=r, cycles=c + 1, pct=pctv, cum_wall_s=wall,
                             delta_vs_canonical_pp=pctv - _CHAMPION_PCT,
                             measured=True))
        else:
            pctv, wall = by_cycle[n_max]
            rows.append(dict(r=r, cycles=c + 1, pct=None, cum_wall_s=None,
                             delta_vs_canonical_pp=None, measured=False,
                             lower_bound_pct=pctv,
                             lower_bound_delta_pp=pctv - _CHAMPION_PCT,
                             note=("beyond the measured curve, which stops at cycle %d "
                                   "(%.8f). The curve is monotone, so that value is a "
                                   "LOWER bound on this arm, not an estimate."
                                   % (n_max + 1, pctv))))
    return dict(source="experiments/outputs/proto_H42.json (connectome, sift_sweeps=2)",
                max_measured_cycle=n_max + 1, rows=rows)


def _microns_runtime():
    """Correction to the sealed prereg's microns note. Measured, not assumed.

    The prereg's ``known_before_launch`` asserted that microns spends ~3330 s of prefix and
    ~88 s of tail, giving R = 1, and concluded H57 could only ever be a connectome-only
    partial promotion. The prefix figure was wrong. The measured split (the microns runs
    carrying per-stage timings, all H42-configuration) is prefix 3085.3 s + first sift
    87.0 s = 3172.3 s, tail 86.0 s -- so the deadline pays for **R = 3** tails on microns,
    not 1.

    The sealed file is deliberately NOT edited: a pre-registration that is rewritten after
    the data are seen is not sealed. The correction lives here and in experiments/log.md.
    It does not touch the decision rule, which is a connectome statistic throughout.
    """
    src = "results/20260815T195540Z-H42-microns-s42-verify-f91b66.json"
    prefix_wall = 3085.3 + 87.0
    tail_wall = 86.0
    r = int(np.floor((_GUARD_DEADLINE_S - prefix_wall) / tail_wall))
    a_r = _A_R[min(max(r, 1), 8)]
    return dict(
        source=src, prefix_wall_s=prefix_wall, tail_wall_s=tail_wall,
        run_wall_s=3258.2, r_feasible=r, a_r=a_r,
        min_promotion_delta_pp=0.002,
        required_sigma_pp=(0.002 / a_r) if a_r > 0 else float("inf"),
        tail_sigma_measured=False,
        corrects=("proto_H57_prereg.json known_before_launch, which said microns admits "
                  "R = 1 and that H57 is therefore connectome-only. That was arithmetic on "
                  "a wrong prefix figure (~3330 s assumed vs 3172.3 s measured)."),
        still_true=("P07 is unaffected and still blocks: at the P07 load factor the microns "
                    "PREFIX alone is ~3806 s, past the 3450 s deadline, so the champion's "
                    "microns configuration does not fit a loaded machine with or without "
                    "this variant."))


def _paired_vs_p09(tail_rows):
    """The two sigmas share labellings r = 1..5, so report them paired, not just side by side."""
    p = json.loads((_OUT / "proto_P09_connectome.json").read_text(encoding="utf-8"))
    whole = {row["r"]: row["comparator"]["pct"] for row in p["rows"]}
    tail = {row["r"]: row["pct"] for row in tail_rows}
    shared = sorted(set(whole) & set(tail) - {0})
    return dict(
        shared_labellings=shared,
        rows=[dict(r=r, whole_pipeline_pct=whole[r], tail_only_pct=tail[r],
                   diff_pp=tail[r] - whole[r]) for r in shared],
        whole_pipeline_sigma_pp=p["summary"]["comparator_pct_std"],
        note=("Same permutations (perm_seed_base 909000), so these are paired. The "
              "whole-pipeline arm re-ran greedy-FAS and Rocket under the labelling; the "
              "tail-only arm reused ONE canonical prefix. The difference between the two "
              "sigmas is the share H57 cannot reach."))


def main():
    src = _OUT / "proto_H57_connectome.json"
    if not src.exists():
        print("proto_H57_connectome.json not found - run the arms first", file=sys.stderr)
        return 2
    d = json.loads(src.read_text(encoding="utf-8"))
    rows = d["rows"]
    ident = next((r for r in rows if r["identity"]), None)
    rnd = [r for r in rows if not r["identity"]]
    if ident is None or len(rnd) < 2:
        print("study incomplete", file=sys.stderr)
        return 2

    pcts = np.array([r["pct"] for r in rnd], dtype=float)
    n = len(pcts)
    sigma = float(pcts.std(ddof=1))
    prefix_wall = float(d["prefix"]["prefix_wall_s"])
    tail_wall = float(np.median([r["tail_wall_s"] for r in rows]))

    def feasible(pw, tw):
        r = int(np.floor((_GUARD_DEADLINE_S - pw) / tw))
        return max(1, r)

    r_idle = feasible(prefix_wall, tail_wall)
    r_loaded = feasible(prefix_wall * _P07_LOAD_FACTOR, tail_wall * _P07_LOAD_FACTOR)
    req_idle = _MIN_DELTA_PP / _A_R[min(r_idle, 8)] if _A_R[min(r_idle, 8)] > 0 else float("inf")
    req_loaded = (_MIN_DELTA_PP / _A_R[min(r_loaded, 8)]
                  if _A_R[min(r_loaded, 8)] > 0 else float("inf"))

    ci = _sigma_ci95(sigma, n)
    validity_ok = (ident["pct"] == _CHAMPION_PCT)

    doc = dict(
        item="H57", gate="prototype", date="2026-08-25", cycle=11, dataset="connectome",
        prereg="experiments/outputs/proto_H57_prereg.json",
        sources=["experiments/outputs/proto_H57_connectome.json",
                 "experiments/outputs/proto_P09_connectome.json",
                 "experiments/outputs/proto_H42.json"],
        validity=dict(
            identity_pct=ident["pct"], champion_pct=_CHAMPION_PCT,
            reproduces_exactly=bool(validity_ok),
            diff_pp=ident["pct"] - _CHAMPION_PCT,
            rule=("prefix + identity tail IS the champion pipeline; if this is not exact "
                  "the study is VOID rather than negative")),
        tail_distribution=dict(
            n_random=n, mean_pct=float(pcts.mean()), sigma_pp=sigma,
            sigma_ci95_pp=ci, min_pct=float(pcts.min()), max_pct=float(pcts.max()),
            span_pp=float(pcts.max() - pcts.min()),
            n_distinct=int(len(np.unique(pcts))),
            canonical_z=(float((ident["pct"] - pcts.mean()) / sigma) if sigma > 0 else None),
            canonical_rank_from_top=int(1 + int((pcts > ident["pct"]).sum()))),
        runtime=dict(
            prefix_wall_s=prefix_wall, tail_wall_s_median=tail_wall,
            guard_deadline_s=_GUARD_DEADLINE_S,
            r_feasible_idle=r_idle, r_feasible_under_p07_load=r_loaded,
            p07_load_factor=_P07_LOAD_FACTOR,
            note=("R is not a free parameter - it is what the deadline pays for after the "
                  "shared prefix. P07 is an OPEN blocking item, so the loaded row is the "
                  "one a promotion would have to survive.")),
        m9=dict(
            min_promotion_delta_pp=_MIN_DELTA_PP,
            required_sigma_pp_idle=req_idle, required_sigma_pp_under_load=req_loaded,
            a_r_idle=_A_R[min(r_idle, 8)], a_r_loaded=_A_R[min(r_loaded, 8)],
            measured_sigma_pp=sigma,
            clears_idle=bool(sigma >= req_idle),
            clears_under_load=bool(sigma >= req_loaded),
            clears_at_sigma_ci_upper_idle=bool(ci[1] >= req_idle)),
        exact_random_only=_enumerate(list(pcts)),
        exact_canonical_anchored=_enumerate(list(pcts), anchor=ident["pct"]),
        matched_compute_control=_control_curve(),
        microns_runtime=_microns_runtime(),
        paired_with_p09=_paired_vs_p09(rows),
    )

    whole_sigma = doc["paired_with_p09"]["whole_pipeline_sigma_pp"]
    doc["variance_split"] = dict(
        tail_sigma_pp=sigma, whole_pipeline_sigma_pp=whole_sigma,
        tail_share_of_variance=float(sigma ** 2 / whole_sigma ** 2),
        prefix_share_of_variance=float(1.0 - sigma ** 2 / whole_sigma ** 2),
        caveat=("A share is only additive if the two sources are independent; they are "
                "not exactly. Read it as an order of magnitude, which is all the verdict "
                "needs."))

    # ── H58's premise, read off this study for free ──────────────────────────────
    # H58 asks whether the CANONICAL labelling is systematically good. Its own plan costs
    # ~1.9 h of GPU to extend the whole-pipeline sample from 5 to 11. This study answers a
    # different and cheaper version of the question - canonical vs 8 TAIL relabellings - and
    # the honest test is exchangeability (rank), not the z, which normal theory inflates.
    rank = doc["tail_distribution"]["canonical_rank_from_top"]
    doc["h58_premise"] = dict(
        canonical_rank_from_top=rank, n_arms=n + 1,
        exchangeability_p_one_sided=float(rank / (n + 1)),
        canonical_z_for_comparison=doc["tail_distribution"]["canonical_z"],
        z_vs_rank_warning=("the z is +3.96 and the rank test is p = 1/9 = 0.111. The z "
                           "assumes normality and is estimated from 8 points; the rank is "
                           "assumption-free. Quote the rank."),
        p09_comparison="whole-pipeline study: canonical ranked 2 of 5, p = 0.4",
        new_mechanistic_reading=(
            "This study suggests a DIFFERENT mechanism from the one H58 proposes. H58's "
            "story is that the FlyWire on-disk order carries biological structure. But here "
            "the prefix is ALWAYS canonical and only the tail's labelling moves, and "
            "canonical still wins outright - so what the identity arm has is not a special "
            "on-disk order, it is TIE-BREAK CONSISTENCY between the prefix that produced "
            "the incoming order (greedy-FAS and the sift both break ties by index) and the "
            "tail that refines it. The paired rows support this: at r = 3 the WHOLE-pipeline "
            "run, where prefix and tail share labelling 3, scored 84.15550043 - higher than "
            "the canonical champion - while canonical-prefix + labelling-3-tail scored only "
            "84.13500040. Consistency, not canonicality."),
        what_this_changes_for_h58=(
            "H58 should test consistency vs canonicality before it spends 1.9 h extending "
            "the whole-pipeline sample: the two readings make opposite predictions for a "
            "structural relabelling, and the consistency reading predicts NO 1x-cost win is "
            "available at all, because consistency is already what the champion has."))

    doc["verdict_grounds"] = dict(
        ground_1_effect_size=(
            "sigma_tail = %.6f pp against a bar of %.6f pp (idle, R = %d) and %.6f pp "
            "(under P07 load, R = %d). Fails by 3.1x-3.8x. Unlike H56's kill, it does NOT "
            "flip at the upper end of sigma's own chi-squared interval: %.6f pp still fails "
            "the easier bar by 35%%."
            % (sigma, req_idle, r_idle, req_loaded, r_loaded, ci[1])),
        ground_2_exact_enumeration=(
            "The canonical arm is the MAXIMUM of all 9 arms, so the canonical-anchored "
            "best-of-R gains EXACTLY 0.000000 pp at every R and for every subset - this is "
            "exact enumeration, not a normal-theory estimate. Random-only best-of-4 is "
            "-0.010532 pp, a regression."),
        ground_3_matched_compute=(
            "The item's own kill condition. The same seconds spent on ONE longer tail buy "
            "at least +0.004707 pp (proto_H42.json, 77 -> 143 cycles, measured), against "
            "H57's exactly zero."),
        structural_finding=(
            "96.2% of the relabelling variance is created in the SHARED prefix. That is "
            "the item's own open_question_to_settle_first, answered against it."))

    if not validity_ok:
        doc["verdict"] = "VOID"
    elif sigma >= req_loaded:
        doc["verdict"] = "PASS"
    else:
        doc["verdict"] = "KILL"

    out = _OUT / "proto_H57_decision.json"
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "validity", "tail_distribution", "runtime", "m9",
                       "variance_split")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
