#!/usr/bin/env python
"""Promote a variant to champion in ``autoresearch/sota.json`` — the only sanctioned writer.

Refuses to promote unless the mechanical audit passes and the evidence actually exists, so a
champion can never be installed by prose alone.

Usage
-----
    python autoresearch/update_sota.py --variant H36 --dataset connectome --role confirm
    python autoresearch/update_sota.py --variant H36 --dataset connectome --role confirm --force
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", required=True, choices=["connectome", "microns", "mouse"])
    ap.add_argument("--role", default="confirm")
    ap.add_argument("--pipeline", default=None, help="one-line description of the full pipeline")
    ap.add_argument("--finding", default=None, help="findings.md section documenting it")
    ap.add_argument("--force", action="store_true",
                    help="promote even if the audit reports a FAIL (must be justified in log.md)")
    args = ap.parse_args()

    runs = load_runs(args.variant, args.dataset, args.role)
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

    if stats["mean"] <= entry["pct_mean"] and not args.force:
        print(f"REFUSED: {args.variant} mean {stats['mean']:.4f} does not beat the current "
              f"champion {entry['champion']} at {entry['pct_mean']:.4f} on {args.dataset}. "
              f"Use --force only with a logged justification.")
        return 1

    audit = subprocess.run(
        [sys.executable, str(_HERE / "audit.py"), "--variant", args.variant,
         "--comparator", "champion", "--role", args.role,
         "--comparator-role", entry.get("role", "confirm"),
         "--datasets", args.dataset],
        cwd=_ROOT, capture_output=True, text=True)
    print(audit.stdout[-4000:])
    if audit.returncode != 0 and not args.force:
        print("REFUSED: audit did not pass. Fix the evidence, or use --force with a "
              "justification recorded in experiments/log.md.")
        return 1

    prior = {k: entry[k] for k in ("champion", "pct_mean", "pct_std", "n_seeds") if k in entry}
    entry.update(
        champion=args.variant,
        pipeline=args.pipeline or entry.get("pipeline", ""),
        pct_mean=round(stats["mean"], 4),
        pct_std=round(stats["std"], 4),
        n_seeds=stats["n"],
        seeds=stats["seeds"],
        role=args.role,
        evidence=f"results/*-{args.variant}-{args.dataset}-*-{args.role}-*.json",
        finding=args.finding or entry.get("finding", ""),
        wall_clock_s_approx=round(stats["max_wall_s"]),
        runner_up=prior,
    )
    sota["updated"] = date.today().isoformat()
    sota["updated_by"] = f"update_sota.py --variant {args.variant} --dataset {args.dataset}"
    sota_path.write_text(json.dumps(sota, indent=2) + "\n")

    print(f"\nPROMOTED: {args.dataset} champion is now {args.variant} "
          f"{stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']}, seeds={stats['seeds']}); "
          f"previous: {prior.get('champion')} {prior.get('pct_mean')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
