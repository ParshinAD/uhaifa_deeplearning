#!/usr/bin/env python
"""Resolve the seed plan for one (variant, role) — the operative half of the P02 amendment.

``seed_class.py`` answers *does this variant draw from its seed*. This script turns that answer
into *which seeds each dataset actually runs*, by reading ``campaign.yaml seed_policy``. It
exists so the amendment is executed by code rather than remembered by an agent: a protocol that
depends on somebody recalling it at 3 a.m. is not a protocol.

Rules it enforces (PROTOCOL.md § Phase-7.4):

* ``--role implement`` (the screen) is the ONLY role the policy touches. A ``deterministic``
  variant screens at 1 seed on the primaries; ``rng`` keeps 3.
* Mouse always keeps 3 seeds. It costs ~6 s/run, so there is nothing to save, and three mouse
  runs that disagree falsify a ``deterministic`` classification before any expensive number is
  quoted. The probe may escalate a variant; it may never de-escalate one.
* ``--role confirm`` returns the dataset's full ``confirm_seeds``, untouched. Every CI in a
  promotion path is computed on a confirm pool, never on a 1-seed screen pool.
* ``seed_policy.enabled: false``, an unknown variant, or a classifier error all fall back to the
  dataset's ``screen_seeds``. Every failure mode costs wall-clock; none of them loses variance.

Usage
-----
    python autoresearch/seed_plan.py --variant H36 --role implement
    python autoresearch/seed_plan.py --variant H36 --role implement --json
    python autoresearch/seed_plan.py --variant H36 --role confirm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import seed_class  # noqa: E402

DATASETS = ("connectome", "microns", "mouse")


def _campaign() -> dict:
    import yaml
    return yaml.safe_load((_HERE / "campaign.yaml").read_text())


def classify(variant: str) -> dict:
    """Static classification for ``variant``. Never raises: any failure answers 'rng'."""
    try:
        result = seed_class.analyze(seed_class.ModuleIndex(), variant, strict=True)
    except Exception as exc:                      # pragma: no cover - defensive
        return {"variant": variant, "verdict": "rng",
                "reason": f"classifier raised {type(exc).__name__}: {exc} -> fail-safe rng"}
    if result.get("verdict") == "error":
        result["verdict"] = "rng"
        result["reason"] = result.get("reason", "") + " -> fail-safe rng"
    return result


def plan(variant: str, role: str, datasets=DATASETS) -> dict:
    """Per-dataset seed lists for this (variant, role), plus the reasoning behind them."""
    campaign = _campaign()
    ds_cfg = campaign["datasets"]
    policy = campaign.get("seed_policy") or {}

    cls = classify(variant)
    verdict = cls.get("verdict", "rng")

    seeds: Dict[str, List[int]] = {}
    reason: Dict[str, str] = {}

    for ds in datasets:
        # The screen may skip a dataset (campaign.yaml datasets.<ds>.in_screen: false).
        # CONFIRM never skips: promotion_gate.primary still requires BOTH primaries, so a
        # confirm-only dataset is deferred, not dropped. Executed here rather than left to the
        # agent's memory, because "remember not to run microns" is not a policy.
        if role != "confirm" and ds_cfg[ds].get("in_screen", True) is False:
            seeds[ds] = []
            reason[ds] = f"in_screen=false: {ds} is confirm-only, not screened"
            continue
        if role == "confirm":
            seeds[ds] = list(ds_cfg[ds]["confirm_seeds"])
            reason[ds] = "confirm: full confirm_seeds, policy does not apply"
            continue
        if role != "implement" or not policy.get("enabled"):
            seeds[ds] = list(ds_cfg[ds]["screen_seeds"])
            reason[ds] = (f"role={role} is not the screen"
                          if role != "implement" else "seed_policy disabled")
            continue
        by_class = (policy.get("screen_seeds_by_class") or {}).get(verdict)
        if not by_class or ds not in by_class:
            seeds[ds] = list(ds_cfg[ds]["screen_seeds"])
            reason[ds] = f"no seed_policy entry for class={verdict} -> fall back to screen_seeds"
            continue
        seeds[ds] = list(by_class[ds])
        reason[ds] = (f"class={verdict}" if len(seeds[ds]) > 1
                      else f"class={verdict}: 1-seed screen (P02)")

    n_runs = sum(len(v) for v in seeds.values())
    baseline_runs = sum(len(ds_cfg[ds]["screen_seeds"]) for ds in datasets
                        if role == "confirm" or ds_cfg[ds].get("in_screen", True) is not False)
    return {
        "variant": variant,
        "role": role,
        "classification": verdict,
        "rng_evidence": cls.get("rng_evidence", [])[:3],
        "unresolved_calls": cls.get("unresolved_calls", [])[:3],
        "seeds": seeds,
        "reason": reason,
        "n_runs": n_runs,
        "n_runs_at_3_seeds": baseline_runs,
        "tripwire": (policy.get("standing_tripwire") if verdict == "deterministic"
                     and role == "implement" else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--role", default="implement",
                    choices=["implement", "confirm", "verify", "prototype"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--json", action="store_true", help="emit the full plan as JSON")
    args = ap.parse_args()

    p = plan(args.variant, args.role, tuple(d for d in args.datasets.split(",") if d))
    if args.json:
        print(json.dumps(p, indent=2))
        return 0
    # Shell-consumable: "<dataset> <seed> <seed> ...", one line per dataset.
    for ds, seeds in p["seeds"].items():
        if not seeds:                # skipped dataset (in_screen: false) — emit no plan line
            continue
        print(f"{ds} {' '.join(str(s) for s in seeds)}")
    print(f"# {p['variant']} role={p['role']} class={p['classification']} "
          f"runs={p['n_runs']} (vs {p['n_runs_at_3_seeds']} at a flat 3 seeds)", file=sys.stderr)
    if p["tripwire"]:
        # Plain ASCII: this goes to a Windows console whose codepage mangles an em-dash.
        print(f"# tripwire: {p['tripwire']} - if the 3 mouse runs disagree, the 1-seed "
              f"primary numbers are VOID", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
