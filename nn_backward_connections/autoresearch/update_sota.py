#!/usr/bin/env python
"""Promote a variant to champion in ``autoresearch/sota.json`` — the only sanctioned writer.

Refuses to promote unless the mechanical audit passes and the evidence actually exists, so a
champion can never be installed by prose alone. Since 2026-08-15 the audit it runs is
``audit.py --gate promotion``, whose scientific gates are hard failures rather than notes.

Two things this script must get right, both of which it got wrong until 2026-08-15:

* **Full precision.** It stored ``pct_mean = round(mean, 4)`` and then gated the NEXT promotion
  with ``mean <= entry["pct_mean"]``, i.e. against a rounded number. Measured consequence: the
  H36 and H42 mouse confirm pools are BIT-IDENTICAL (n=20 each, one distinct value,
  92.91701410211007, delta exactly 0.0), yet H42 took the mouse championship because
  92.917014102 > round(92.917014102, 4) = 92.917. A 4-decimal rounding artifact was the
  campaign's only mechanical effect-size floor. ``pct_mean_exact`` now carries the unrounded
  value and is what the gate reads; ``pct_mean`` stays rounded because dashboard.py renders it.
* **A minimum effect size.** "Beats the champion" is not "is bigger than the champion". A
  promotion must clear ``campaign.yaml datasets.<ds>.min_promotion_delta_pp``.

Usage
-----
    python autoresearch/update_sota.py --variant H36 --dataset connectome --role confirm
    python autoresearch/update_sota.py --variant H36 --dataset connectome --role confirm --force
    python autoresearch/update_sota.py --variant H36 --dataset mouse --role confirm \\
        --caveat "mouse non-inferiority bound not cleared; PROTOCOL.md decision-table row"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from audit import load_runs, summarize  # noqa: E402


def champion_mean(entry: dict) -> float:
    """The champion's mean at the best precision the registry holds.

    ``pct_mean_exact`` is written by every promotion from 2026-08-15 on. Rows promoted before
    that carry only the 4-decimal ``pct_mean``, so comparisons against them keep up to 5e-5 pp
    of rounding slack — harmless now that ``min_promotion_delta_pp`` (>= 0.002 pp on every
    dataset) is the binding constraint rather than the rounding. Existing rows are deliberately
    NOT rewritten: those are recorded numbers.
    """
    return float(entry.get("pct_mean_exact", entry["pct_mean"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", required=True, choices=["connectome", "microns", "mouse"])
    ap.add_argument("--role", default="confirm")
    ap.add_argument("--pipeline", default=None, help="one-line description of the full pipeline")
    ap.add_argument("--finding", default=None, help="findings.md section documenting it")
    ap.add_argument("--caveat", action="append", default=None, metavar="TEXT",
                    help="short scope/limitation string persisted in the entry's `caveats` list; "
                         "repeatable. Use it whenever the promotion rests on a PROTOCOL.md "
                         "decision-table row rather than a clean pass (e.g. 'GENERAL WIN with "
                         "mouse caveat'), so a consumer reading sota.json alone cannot mistake "
                         "it for an unqualified three-dataset win.")
    ap.add_argument("--force", action="store_true",
                    help="promote even if the audit reports a FAIL or the delta does not clear "
                         "the dataset's minimum effect size. THE escape hatch: it is logged into "
                         "the entry as `forced` and must be justified in experiments/log.md.")
    ap.add_argument("--device-tag", default=None,
                    help="only count runs measured on this device (env.gpu). Defaults to "
                         "campaign.yaml environment.device_tag; pass '' to pool every device.")
    args = ap.parse_args()

    campaign = {}
    cpath = _HERE / "campaign.yaml"
    if cpath.exists():
        try:
            import yaml
            campaign = yaml.safe_load(cpath.read_text()) or {}
        except Exception as e:
            print(f"WARNING: could not parse campaign.yaml ({e!r}) — thresholds unavailable")

    device_tag = args.device_tag
    if device_tag is None:
        device_tag = ((campaign.get("environment") or {}).get("device_tag")) or None

    runs = load_runs(args.variant, args.dataset, args.role, device_tag)
    if not runs:
        print(f"REFUSED: no results/*.json for {args.variant} on {args.dataset} "
              f"at role={args.role}")
        return 1
    stats = summarize(runs)

    sota_path = _HERE / "sota.json"
    sota = json.loads(sota_path.read_text())
    entry = sota["datasets"].get(args.dataset)
    if entry is None:
        print(f"REFUSED: unknown dataset {args.dataset}")
        return 1

    # D2 (2026-08-15). Gate on the UNROUNDED champion mean, and on a real effect size rather
    # than on "> 0". Both halves matter: H42 beat H36 on mouse by exactly 0.0 pp over
    # bit-identical pools, purely because the stored champion had been rounded to 4 decimals.
    incumbent = champion_mean(entry)
    delta = stats["mean"] - incumbent
    d_cfg = ((campaign.get("datasets") or {}).get(args.dataset) or {})
    min_delta = d_cfg.get("min_promotion_delta_pp", d_cfg.get("screen_delta_pp"))
    if min_delta is None:
        print(f"WARNING: no min_promotion_delta_pp/screen_delta_pp for {args.dataset} in "
              f"campaign.yaml — falling back to 'strictly positive', which is NOT an effect size")
        min_delta = 0.0
    if delta <= float(min_delta) and not args.force:
        print(f"REFUSED: {args.variant} mean {stats['mean']!r} on {args.dataset} is "
              f"{delta:+.6f} pp against champion {entry['champion']} at {incumbent!r}, which "
              f"does not clear the minimum effect size {float(min_delta):+.4f} pp "
              f"(campaign.yaml datasets.{args.dataset}.min_promotion_delta_pp). "
              f"Use --force only with a logged justification.")
        return 1

    audit = subprocess.run(
        [sys.executable, str(_HERE / "audit.py"), "--variant", args.variant,
         "--comparator", "champion", "--role", args.role,
         "--comparator-role", entry.get("role", "confirm"),
         "--datasets", args.dataset,
         # D1: the permissive default cannot fail on any scientific ground, so a promotion must
         # ask for the strict shape explicitly. Everything it adds is in campaign.yaml.
         "--gate", "promotion",
         "--device-tag", device_tag or ""],
        cwd=_ROOT, capture_output=True, text=True)
    print(audit.stdout[-4000:])
    if audit.returncode != 0 and not args.force:
        print("REFUSED: audit did not pass under --gate promotion. Fix the evidence, or use "
              "--force with a justification recorded in experiments/log.md.")
        return 1

    prior = {k: entry[k] for k in ("champion", "pct_mean", "pct_mean_exact", "pct_std",
                                   "n_seeds", "caveats") if k in entry}
    entry.update(
        champion=args.variant,
        pipeline=args.pipeline or entry.get("pipeline", ""),
        # pct_mean stays rounded for display (dashboard.py renders it); pct_mean_exact is what
        # the next promotion's gate reads. Never compare against the rounded field again.
        pct_mean=round(stats["mean"], 4),
        pct_mean_exact=stats["mean"],
        pct_std=round(stats["std"], 4),
        pct_std_exact=stats["std"],
        n_seeds=stats["n"],
        seeds=stats["seeds"],
        role=args.role,
        evidence=f"results/*-{args.variant}-{args.dataset}-*-{args.role}-*.json",
        finding=args.finding or entry.get("finding", ""),
        # D7: a promotion whose scope is qualified must say so IN the registry. Caveats belong
        # to the row being written, so they are replaced, never inherited from the predecessor.
        caveats=list(args.caveat or []),
        wall_clock_s_approx=round(stats["max_wall_s"]),
        runner_up=prior,
    )
    if args.force:
        # The escape hatch is documented; it must also be visible in the artifact it wrote.
        entry["forced"] = (f"promoted with --force on {date.today().isoformat()}; "
                           f"delta {delta:+.6f} pp vs {prior.get('champion')}, "
                           f"audit exit {audit.returncode} under --gate promotion. "
                           f"Justification required in experiments/log.md.")
    else:
        entry.pop("forced", None)
    sota["updated"] = date.today().isoformat()
    sota["updated_by"] = f"update_sota.py --variant {args.variant} --dataset {args.dataset}"
    sota_path.write_text(json.dumps(sota, indent=2) + "\n")

    print(f"\nPROMOTED: {args.dataset} champion is now {args.variant} "
          f"{stats['mean']!r} +/- {stats['std']!r} (n={stats['n']}, seeds={stats['seeds']}); "
          f"previous: {prior.get('champion')} {incumbent!r}; delta {delta:+.6f} pp"
          + (f"; caveats: {entry['caveats']}" if entry["caveats"] else "")
          + ("  [FORCED]" if args.force else "") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
