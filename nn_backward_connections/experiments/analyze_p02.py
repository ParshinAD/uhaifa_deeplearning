"""P02 analysis — assemble the evidence for the 1-seed-screen protocol amendment.

Three things are established here, each from artifacts on disk:

1. PRIMARY-DATASET REPEAT. A fresh process re-runs the champion at seed 42 on connectome
   (H35) and microns (H30) and the resulting position vector is compared BIT-FOR-BIT against
   the P01 run of the same seed, made hours earlier in a different process. This is the
   device-determinism leg (B): if CUDA injected run-to-run noise, repeats would buy
   information and the amendment would have to be "3 repeats", not "1 seed".

2. CLASSIFIER vs GROUND TRUTH. autoresearch/seed_class.py (static call-graph) is checked
   against the 48-run mouse probe in experiments/outputs/proto_P02.json. The check that
   matters is the FALSE-DETERMINISTIC count: a variant the classifier calls deterministic
   that actually varies across seeds. That is P02's stated kill condition.

3. STATISTICAL EQUIVALENCE. Using audit.py's OWN summarize/welch/protocol_se, the screen
   statistics are recomputed from a 3-seed pool and from a 1-seed pool of the same
   deterministic variant. If every quantity the SCREEN verdict depends on is identical, the
   two extra seeds are provably informationless rather than merely observed to be.

   Scope, and why it is narrow. The screen verdict is a POINT-ESTIMATE rule —
   ``delta > screen_delta_pp`` (campaign.yaml gates.screen, agents/implementer.md) — so it
   depends on the pool mean and nothing else. That is the only claim this function makes.
   It deliberately does NOT claim the confidence interval is unaffected, because
   ``audit.protocol_se(comparator, floor)`` is ``max(std_c, floor) * sqrt(2 / n_c)``: a
   function of the COMPARATOR pool alone, structurally blind to how many seeds the variant
   contributed. Reading its unchanged value as evidence would be circular. The honest
   quantity is reported alongside as ``protocol_se_if_variant_n_entered`` =
   ``max(std_c, floor) * sqrt(1/n_v + 1/n_c)``, which at n_v=1 vs n_v=3 is 41% wider — the
   real cost of a 1-seed pool, if a CI were ever computed on one. The amendment keeps the
   full ``confirm_seeds`` precisely so that never happens: every CI in a promotion path is
   computed on a confirm pool of >= 5.

Writes experiments/outputs/proto_P02_primaries.json. Reads only; no results/ writes.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_ROOT / "autoresearch"))
import audit  # noqa: E402  (reuse the auditor's own arithmetic, never a re-implementation)

DEVICE_TAG = "NVIDIA GeForce RTX 4060 Laptop GPU"


def digest(path: Path) -> str:
    a = np.load(path)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def find(variant: str, dataset: str, seed: int, role: str):
    """Result records for one (variant, dataset, seed, role) on THIS device."""
    out = []
    for f in sorted(glob.glob(str(_ROOT / "results" / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if (d.get("experiment_id") != variant or d.get("dataset") != dataset
                or int(d.get("seed", -1)) != seed or d.get("role") != role):
            continue
        if (d.get("env") or {}).get("gpu") != DEVICE_TAG:
            continue
        d["_file"] = Path(f).name
        out.append(d)
    return out


def repeat_check(variant: str, dataset: str) -> dict:
    """Compare the P01 implement run at seed 42 with this cycle's verify repeat at seed 42."""
    base = find(variant, dataset, 42, "implement")
    rep = find(variant, dataset, 42, "verify")
    res = {"variant": variant, "dataset": dataset,
           "n_implement": len(base), "n_verify_repeat": len(rep)}
    if not base or not rep:
        res["status"] = "MISSING"
        return res
    b, r = base[0], rep[-1]          # newest repeat
    bp = _ROOT / b["best_positions_path"]
    rp = _ROOT / r["best_positions_path"]
    res.update(
        implement_file=b["_file"], verify_file=r["_file"],
        implement_pct=b["pct"], verify_pct=r["pct"],
        implement_score=b["score"], verify_score=r["score"],
        implement_wall_s=round(float(b.get("wall_clock_s", 0)), 1),
        verify_wall_s=round(float(r.get("wall_clock_s", 0)), 1),
        score_identical=float(b["score"]) == float(r["score"]),
    )
    if bp.exists() and rp.exists():
        db, dr = digest(bp), digest(rp)
        pb, pr = np.load(bp), np.load(rp)
        res.update(implement_digest=db, verify_digest=dr,
                   positions_bit_identical=bool(db == dr),
                   n_positions_differing=int(np.sum(pb != pr)) if pb.shape == pr.shape else -1)
    else:
        res["positions_bit_identical"] = None
        res["note"] = "a positions .npy is missing (they are gitignored)"
    res["status"] = "OK"
    return res


def classifier_vs_truth() -> dict:
    """Cross-check the static classifier against the 48-run mouse probe."""
    proto = json.loads((_ROOT / "experiments" / "outputs" / "proto_P02.json").read_text())
    cls = json.loads((_ROOT / "autoresearch" / "seed_class.json").read_text())["variants"]

    rows, false_det, disagree = [], [], []
    for vid, obs in proto["variants"].items():
        if "observed_class" not in obs:
            continue
        static = cls.get(vid, {}).get("verdict", "?")
        static_short = "det" if static == "deterministic" else "rng"
        row = dict(variant=vid, classifier=static_short, observed_mouse=obs["observed_class"],
                   device_deterministic=obs["device_deterministic"],
                   seed_inert_mouse=obs["seed_inert_positions"])
        # The dangerous error: called deterministic, but the seed demonstrably changes it.
        if static_short == "det" and not obs["seed_inert_positions"]:
            false_det.append(vid)
            row["verdict"] = "FALSE-DETERMINISTIC"
        elif static_short != obs["observed_class"]:
            disagree.append(vid)
            row["verdict"] = "conservative (static rng, mouse looked det)"
        else:
            row["verdict"] = "agree"
        rows.append(row)

    return {
        "rows": rows,
        "false_deterministic": false_det,
        "conservative_disagreements": disagree,
        "n_device_nondeterministic": sum(1 for r in rows if not r["device_deterministic"]),
        "pass": len(false_det) == 0,
    }


def stat_equivalence(variant: str, dataset: str) -> dict:
    """Recompute the screen statistics from a 3-seed and a 1-seed pool of the same variant.

    Uses audit.py's own summarize/welch/protocol_se so this is the arithmetic a real verdict
    would use, not a paraphrase of it. See the module docstring for what the ``identical``
    flags do and do not claim: ``screen_verdict_identical`` is the load-bearing one.
    """
    import yaml
    campaign = yaml.safe_load((_ROOT / "autoresearch" / "campaign.yaml").read_text())
    sigma_floor = float(campaign["datasets"][dataset]["baseline_sigma_pp"])

    runs3 = audit.load_runs(variant, dataset, "implement", DEVICE_TAG)
    if len(runs3) < 2:
        return {"status": "MISSING", "n": len(runs3)}
    runs1 = [r for r in runs3 if int(r["seed"]) == 42]

    comp = audit.summarize(runs3)          # champion pool (the comparator), unchanged
    out = {"dataset": dataset, "variant": variant, "sigma_floor_pp": sigma_floor,
           "comparator_n": comp["n"], "pools": {}}
    for label, pool in (("screen_3_seeds", runs3), ("screen_1_seed", runs1)):
        v = audit.summarize(pool)
        delta, se, ci = audit.welch(v, comp)
        p_se = audit.protocol_se(comp, sigma_floor)
        # What the SE would be if the variant's own n entered it. audit.protocol_se uses
        # sqrt(2/n_c) — the equal-n form — so it cannot see n_v; at n_v=1 that understates
        # the true spread. Reported, never used as a gate: no CI is computed on a screen pool.
        p_se_nv = (max(comp["std"], sigma_floor)
                   * math.sqrt(1.0 / max(v["n"], 1) + 1.0 / max(comp["n"], 1)))
        out["pools"][label] = dict(
            n=v["n"], seeds=v["seeds"], mean_pct=v["mean"], std_pp=v["std"],
            delta_pp=delta, welch_se=se, welch_ci_lower=ci,
            protocol_se=p_se, protocol_ci_lower=delta - 1.96 * p_se,
            protocol_se_if_variant_n_entered=p_se_nv,
            protocol_ci_lower_if_variant_n_entered=delta - 1.96 * p_se_nv)
    a, b = out["pools"]["screen_3_seeds"], out["pools"]["screen_1_seed"]
    # THE claim: the screen rule is `delta > screen_delta_pp`, a function of the pool mean.
    out["screen_verdict_identical"] = all(abs(a[k] - b[k]) < 1e-12
                                          for k in ("mean_pct", "delta_pp", "std_pp"))
    # NOT a claim of equivalence — protocol_se is comparator-only by construction, so this
    # flag is true tautologically. Kept visible so nobody re-derives it as evidence.
    out["protocol_se_unchanged_but_comparator_only"] = abs(
        a["protocol_se"] - b["protocol_se"]) < 1e-12
    out["protocol_se_widening_at_n1_pct"] = round(
        100.0 * (b["protocol_se_if_variant_n_entered"]
                 / a["protocol_se_if_variant_n_entered"] - 1.0), 1)
    out["status"] = "OK"
    return out


def savings() -> dict:
    """What the amendment actually buys, from LOGGED wall-clocks on this device.

    The queue item guessed "a cycle drops from ~8.5 h to ~2.5 h". That assumed cutting the
    confirm stage too. The amendment is screen-only, so the honest arithmetic is different —
    and the number that matters is the killed-at-screen cycle, because most cycles end there.
    """
    import yaml
    campaign = yaml.safe_load((_ROOT / "autoresearch" / "campaign.yaml").read_text())
    per_run = {}
    for ds, variant in (("connectome", "H35"), ("microns", "H30"), ("mouse", "H30")):
        runs = audit.load_runs(variant, ds, "implement", DEVICE_TAG)
        walls = [float(r.get("wall_clock_s", 0.0)) for r in runs]
        per_run[ds] = dict(variant=variant, n=len(walls),
                           mean_wall_s=round(float(np.mean(walls)), 1) if walls else None)

    n_confirm = {ds: len(campaign["datasets"][ds]["confirm_seeds"])
                 for ds in ("connectome", "microns", "mouse")}
    w = {ds: (per_run[ds]["mean_wall_s"] or 0.0) for ds in per_run}

    # Read the seed counts from the POLICY, not from an assumption. A dataset only appears with
    # a reduced list once its device determinism is verified (PORTING.md § 6b), so while that is
    # pending this reports the saving actually in force, not the one hoped for.
    policy = (campaign.get("seed_policy") or {})
    det = (policy.get("screen_seeds_by_class") or {}).get("deterministic", {})
    n_det = {ds: len(det.get(ds, campaign["datasets"][ds]["screen_seeds"])) for ds in w}
    n_flat = {ds: len(campaign["datasets"][ds]["screen_seeds"]) for ds in w}

    screen_3 = sum(n_flat[ds] * w[ds] for ds in w)
    screen_1 = sum(n_det[ds] * w[ds] for ds in w)
    confirm = sum(n_confirm[ds] * w[ds] for ds in w)
    pending = [ds for ds in ("connectome", "microns")
               if not (policy.get("device_determinism_verified") or {}).get(ds)]
    return {
        "per_run_mean_wall_s": per_run,
        "screen_seeds_in_force": n_det,
        "screen_seeds_if_flat_3": n_flat,
        "device_determinism_pending": pending,
        "screen_3_seeds_h": round(screen_3 / 3600, 2),
        "screen_1_seed_h": round(screen_1 / 3600, 2),
        "screen_saving_h": round((screen_3 - screen_1) / 3600, 2),
        "screen_saving_pct": round(100 * (screen_3 - screen_1) / screen_3, 1) if screen_3 else 0.0,
        "confirm_h_unchanged": round(confirm / 3600, 2),
        "cycle_killed_at_screen_h": [round(screen_3 / 3600, 2), round(screen_1 / 3600, 2)],
        "cycle_reaching_confirm_h": [round((screen_3 + confirm) / 3600, 2),
                                     round((screen_1 + confirm) / 3600, 2)],
        "note": "mouse stays at 3 seeds (it costs ~6 s/run, so there is nothing to save and "
                "it is the cheap corroborating determinism probe). confirm_seeds unchanged. "
                "Seed counts come from campaign.yaml seed_policy, so a dataset still awaiting "
                "its device-determinism evidence is costed at 3 seeds here.",
    }


def main() -> int:
    report = {
        "_doc": "P02 evidence bundle: primary-dataset repeat runs (device determinism), "
                "classifier-vs-ground-truth cross-check, and the 1-seed/3-seed statistical "
                "equivalence computed with audit.py's own functions. "
                "Regenerate: python experiments/analyze_p02.py",
        "device_tag": DEVICE_TAG,
        "repeat_checks": [repeat_check("H35", "connectome"),
                          repeat_check("H30", "microns")],
        "classifier_vs_truth": classifier_vs_truth(),
        "stat_equivalence": [stat_equivalence("H35", "connectome"),
                             stat_equivalence("H30", "microns")],
        "savings": savings(),
    }

    print("=== 1. primary-dataset repeat (same seed, fresh process) ===")
    for r in report["repeat_checks"]:
        if r["status"] != "OK":
            print(f"  {r['variant']:5s} {r['dataset']:11s} {r['status']}")
            continue
        print(f"  {r['variant']:5s} {r['dataset']:11s} "
              f"implement {r['implement_pct']:.4f} ({r['implement_wall_s']}s) vs "
              f"verify {r['verify_pct']:.4f} ({r['verify_wall_s']}s) | "
              f"score_identical={r['score_identical']} "
              f"positions_bit_identical={r.get('positions_bit_identical')} "
              f"n_diff={r.get('n_positions_differing')}")

    print("\n=== 2. classifier vs 48-run mouse ground truth ===")
    cv = report["classifier_vs_truth"]
    for row in cv["rows"]:
        print(f"  {row['variant']:22s} static={row['classifier']:3s} "
              f"mouse={row['observed_mouse']:3s}  {row['verdict']}")
    print(f"  false_deterministic={cv['false_deterministic']}  PASS={cv['pass']}")

    print("\n=== 3. statistical equivalence, 3-seed vs 1-seed pool ===")
    for s in report["stat_equivalence"]:
        if s["status"] != "OK":
            print(f"  {s.get('dataset')}: {s['status']}")
            continue
        a, b = s["pools"]["screen_3_seeds"], s["pools"]["screen_1_seed"]
        print(f"  {s['dataset']:11s} 3-seed: mean={a['mean_pct']:.4f} std={a['std_pp']:.4f} "
              f"welch_se={a['welch_se']:.6f} proto_se={a['protocol_se']:.6f}")
        print(f"  {'':11s} 1-seed: mean={b['mean_pct']:.4f} std={b['std_pp']:.4f} "
              f"welch_se={b['welch_se']:.6f} proto_se={b['protocol_se']:.6f}")
        # Plain ASCII: this goes to a Windows console whose codepage mangles an em-dash.
        print(f"  {'':11s} SCREEN VERDICT IDENTICAL={s['screen_verdict_identical']} "
              f"(mean/delta/std); if a CI were computed on a 1-seed pool its SE would be "
              f"+{s['protocol_se_widening_at_n1_pct']}%, and confirm keeps all seeds so it is not)")

    sv = report["savings"]
    print("\n=== 4. throughput, from logged wall-clocks ===")
    print(f"  screen 3 seeds {sv['screen_3_seeds_h']} h -> policy {sv['screen_1_seed_h']} h "
          f"(saves {sv['screen_saving_h']} h, {sv['screen_saving_pct']}%)  "
          f"seeds in force: {sv['screen_seeds_in_force']}")
    if sv["device_determinism_pending"]:
        print(f"  PENDING device-determinism evidence, still at 3 seeds: "
              f"{sv['device_determinism_pending']}")
    print(f"  cycle killed at screen: {sv['cycle_killed_at_screen_h']} h")
    print(f"  cycle reaching confirm: {sv['cycle_reaching_confirm_h']} h "
          f"(confirm {sv['confirm_h_unchanged']} h, unchanged)")

    dest = _ROOT / "experiments" / "outputs" / "proto_P02_primaries.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
