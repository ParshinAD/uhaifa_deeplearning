#!/usr/bin/env python
"""Mechanical honesty auditor for the autonomous campaign (Phase 7).

The critic subagent applies *judgement*. This script applies *arithmetic*. It re-derives every
number a cycle claims, straight from ``results/*.json`` and the frozen scorer, so that no
agent's summary can drift from the evidence.

What it checks
--------------
1. FROZEN INTEGRITY  — ``eval.frozen_guard.verify_frozen_manifest()`` plus a git check that no
   frozen path is modified in the working tree.
2. RUN INVENTORY     — every run found for (variant, dataset, role), listed in full. Cherry
   picking is impossible to hide: if 8 runs exist and 3 were reported, all 8 appear here.
3. RE-SCORE          — for every run whose ``best_positions_path`` still exists, re-score the
   positions with the frozen oracle and compare to the recorded ``score``. Any mismatch is a
   hard FAIL (fabricated or corrupted number).
4. SIGNIFICANCE      — per dataset: mean +/- std, Welch SE and 95% CI lower bound versus the
   comparator (default: the champion from ``autoresearch/sota.json``), plus the conservative
   PROTOCOL SE = std_comparator * sqrt(2/n) for continuity with findings.md.
5. LEAKAGE           — static scan of the variant module (and the refiner modules it imports)
   for the target metric, the reference solution, or hardcoded oracle values.
6. RUNTIME           — max wall-clock per run against ``campaign.yaml`` runtime budget.

Exit code 0 = every hard check passed. Non-zero = at least one FAIL. WARNs never fail the run;
they are surfaced for the critic to adjudicate.

Usage
-----
    python autoresearch/audit.py --variant H36
    python autoresearch/audit.py --variant H36 --role confirm --out autoresearch/audit_H36.json
    python autoresearch/audit.py --variant H36 --comparator H35
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FROZEN_PATHS = [
    "src/mfas/metrics.py",
    "eval/harness.py",
    "eval/aggregate.py",
    "tests/test_metrics.py",
    "eval/frozen.sha256",
    "eval/frozen_guard.py",
]

# Tokens that must never appear in an algorithm module. These are the oracle's own values and
# the reference solution: an algorithm that knows them is not solving the problem, it is
# reciting the answer.
LEAKAGE_TOKENS = [
    "best_solution",
    "35463823", "35462925", "34751902", "35165745",
    "84.6147", "84.61", "82.9161",
]
# Allowed-but-noteworthy: keying a compute budget on the dataset name is legitimate and
# documented (H30/H35 do it), but it is also how dataset special-casing would sneak in.
SOFT_TOKENS = ["g.name", "dataset ==", 'dataset=="', "dataset == '"]


class Report:
    def __init__(self) -> None:
        self.checks: List[dict] = []
        self.data: dict = {}

    def add(self, name: str, status: str, detail: str, **extra) -> None:
        """status in {PASS, FAIL, WARN, INFO}."""
        self.checks.append(dict(check=name, status=status, detail=detail, **extra))

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)

    def render(self) -> str:
        width = max(len(c["check"]) for c in self.checks) if self.checks else 10
        lines = []
        for c in self.checks:
            lines.append(f"  [{c['status']:4}] {c['check']:<{width}}  {c['detail']}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Frozen integrity
# ──────────────────────────────────────────────────────────────────────────────
def check_frozen(rep: Report) -> None:
    try:
        from eval.frozen_guard import verify_frozen_manifest  # noqa: E402
        verify_frozen_manifest()
        rep.add("frozen.manifest", "PASS", "SHA-256 manifest matches all frozen files")
    except SystemExit as e:
        rep.add("frozen.manifest", "FAIL", f"frozen guard aborted: {e}")
        return
    except Exception as e:  # pragma: no cover - defensive
        rep.add("frozen.manifest", "FAIL", f"frozen guard error: {e!r}")
        return

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--"] + FROZEN_PATHS,
            cwd=_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        rep.add("frozen.git", "WARN", f"git check failed: {e}")
        return
    if out:
        rep.add("frozen.git", "FAIL", f"frozen path(s) modified in working tree:\n{out}")
    else:
        rep.add("frozen.git", "PASS", "no frozen path modified in the working tree")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Run inventory
# ──────────────────────────────────────────────────────────────────────────────
def load_runs(variant: str, dataset: Optional[str] = None,
              role: Optional[str] = None) -> List[dict]:
    runs = []
    for f in sorted(glob.glob(str(_ROOT / "results" / "*.json"))):
        p = Path(f)
        if p.parent.name != "results":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("experiment_id") != variant and d.get("algo") != variant:
            continue
        if dataset and d.get("dataset") != dataset:
            continue
        if role and d.get("role") != role:
            continue
        d["_file"] = p.name
        runs.append(d)
    return runs


def summarize(runs: List[dict]) -> dict:
    pcts = np.array([r["pct"] for r in runs], dtype=np.float64)
    return dict(
        n=len(runs),
        mean=float(pcts.mean()) if len(pcts) else float("nan"),
        # ddof=1: sample std, matching PROTOCOL/findings.
        std=float(pcts.std(ddof=1)) if len(pcts) > 1 else 0.0,
        seeds=sorted(int(r["seed"]) for r in runs),
        files=[r["_file"] for r in runs],
        max_wall_s=max((float(r.get("wall_clock_s", 0.0)) for r in runs), default=0.0),
        grad_steps=sorted({int(r.get("total_grad_steps") or r.get("n_epochs_done") or 0)
                           for r in runs}),
        # config_hash only covers (algo, dataset), so it CANNOT distinguish two configurations
        # of the same variant (e.g. H30 at 12 vs 40 sweeps). The commit is what separates them.
        commits=sorted({str(r.get("git_commit", "")).split("+")[0][:8] for r in runs}),
        roles=sorted({str(r.get("role", "")) for r in runs}),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Re-score
# ──────────────────────────────────────────────────────────────────────────────
def check_rescore(rep: Report, runs: List[dict]) -> None:
    from mfas import io  # noqa: E402
    from mfas.metrics import score_from_positions  # noqa: E402

    checked = 0
    missing = 0
    graphs: Dict[str, object] = {}
    for r in runs:
        rel = r.get("best_positions_path")
        if not rel:
            missing += 1
            continue
        path = _ROOT / rel
        if not path.exists():
            missing += 1
            continue
        ds = r["dataset"]
        if ds not in graphs:
            graphs[ds] = io.load_dataset(ds)
        g = graphs[ds]
        pos = np.load(path)
        score = score_from_positions(g, pos)
        recorded = float(r["score"])
        if abs(float(score) - recorded) > 1e-6:
            rep.add("rescore", "FAIL",
                    f"{r['_file']}: recorded score {recorded} != re-scored {score}")
            return
        checked += 1

    if checked:
        rep.add("rescore", "PASS",
                f"{checked} run(s) re-scored with the frozen oracle, all exact"
                + (f"; {missing} run(s) had no positions file on disk" if missing else ""))
    else:
        rep.add("rescore", "WARN",
                f"no positions files available to re-score ({missing} run(s) missing them) — "
                "positions .npy are gitignored, so only runs produced locally can be verified")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Significance
# ──────────────────────────────────────────────────────────────────────────────
def welch(a: dict, b: dict) -> Tuple[float, float, float]:
    """Return (delta, se, ci_lower_95) for mean(a) - mean(b), Welch (unequal variances)."""
    delta = a["mean"] - b["mean"]
    va = (a["std"] ** 2) / max(a["n"], 1)
    vb = (b["std"] ** 2) / max(b["n"], 1)
    se = math.sqrt(va + vb)
    return delta, se, delta - 1.96 * se


def protocol_se(comparator: dict) -> float:
    """PROTOCOL.md conservative SE = std_comparator * sqrt(2/n)."""
    n = max(comparator["n"], 1)
    return comparator["std"] * math.sqrt(2.0 / n)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Leakage
# ──────────────────────────────────────────────────────────────────────────────
def _docstring_lines(source: str) -> set:
    """Line numbers occupied by module/class/function docstrings.

    Only *docstrings* are excluded from the leakage scan, not every string literal: a path
    like ``open("data/best_solution/...")`` must still be caught.
    """
    import ast
    lines: set = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                and isinstance(first.value.value, str):
            start = first.lineno
            end = getattr(first, "end_lineno", start) or start
            lines.update(range(start, end + 1))
    return lines


def check_leakage(rep: Report, variant: str) -> None:
    mod = _ROOT / "src" / "mfas" / "experiments" / f"{variant}.py"
    if not mod.exists():
        rep.add("leakage", "WARN", f"variant module not found: {mod.relative_to(_ROOT)}")
        return

    sources = [mod]
    # Follow first-party imports one level (refiners carry the actual move logic).
    text = mod.read_text()
    for m in re.finditer(r"from\s+(?:\.\.|mfas\.)([\w\.]+)\s+import", text):
        cand = _ROOT / "src" / "mfas" / (m.group(1).replace(".", "/") + ".py")
        if cand.exists():
            sources.append(cand)
        pkg = _ROOT / "src" / "mfas" / m.group(1).replace(".", "/")
        if pkg.is_dir():
            sources.extend(sorted(pkg.glob("*.py")))

    hard: List[str] = []
    soft: List[str] = []
    for src in dict.fromkeys(sources):
        body = src.read_text()
        doc_lines = _docstring_lines(body)
        for line_no, line in enumerate(body.splitlines(), 1):
            # Docstrings are prose — this repo's modules *document* the leakage boundary, so
            # scanning them produces false positives. Executable code is what matters.
            if line_no in doc_lines:
                continue
            stripped = line.strip()
            code = stripped.split("#", 1)[0]
            if not code:
                continue
            for tok in LEAKAGE_TOKENS:
                if tok in code:
                    hard.append(f"{src.relative_to(_ROOT)}:{line_no}: {stripped[:110]}")
            for tok in SOFT_TOKENS:
                if tok in code:
                    soft.append(f"{src.relative_to(_ROOT)}:{line_no}: {stripped[:110]}")

    if hard:
        rep.add("leakage", "FAIL",
                "target metric / reference solution referenced in algorithm code:\n    "
                + "\n    ".join(hard))
    else:
        rep.add("leakage", "PASS",
                f"no oracle value or reference solution in {len(sources)} algorithm file(s)")
    if soft:
        rep.add("leakage.dataset_keying", "WARN",
                "dataset-name keying present (legitimate for compute budgets, but the cycle "
                "must state WHAT it keys):\n    " + "\n    ".join(soft[:12]))


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="variant id under audit, e.g. H36")
    ap.add_argument("--comparator", default="champion",
                    help="'champion' (per-dataset, from sota.json) or an explicit variant id")
    ap.add_argument("--role", default=None,
                    help="restrict the variant's runs to this role (implement|verify|confirm)")
    ap.add_argument("--comparator-role", default=None,
                    help="restrict the comparator's runs to this role; use it when the "
                         "comparator id spans several commits/configurations")
    ap.add_argument("--datasets", default="connectome,microns,mouse")
    ap.add_argument("--out", default=None, help="write the full report JSON here")
    args = ap.parse_args()

    rep = Report()
    check_frozen(rep)
    check_leakage(rep, args.variant)

    campaign = {}
    cpath = _HERE / "campaign.yaml"
    if cpath.exists():
        try:
            import yaml
            campaign = yaml.safe_load(cpath.read_text()) or {}
        except Exception as e:
            rep.add("campaign.yaml", "WARN", f"could not parse: {e!r}")

    sota = json.loads((_HERE / "sota.json").read_text())

    all_variant_runs: List[dict] = []
    per_dataset: Dict[str, dict] = {}

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        v_runs = load_runs(args.variant, ds, args.role)
        if not v_runs:
            rep.add(f"runs.{ds}", "WARN", f"no runs found for {args.variant} on {ds}")
            continue
        all_variant_runs.extend(v_runs)

        comp_id = (sota["datasets"].get(ds, {}).get("champion")
                   if args.comparator == "champion" else args.comparator)
        if not comp_id:
            rep.add(f"runs.{ds}", "WARN", f"no champion registered for {ds}")
            continue
        c_runs = load_runs(comp_id, ds, args.comparator_role)
        if not c_runs:
            rep.add(f"comparator.{ds}", "WARN",
                    f"no runs found for comparator {comp_id} on {ds}")
            continue

        v, c = summarize(v_runs), summarize(c_runs)
        delta, se, ci_lo = welch(v, c)
        p_se = protocol_se(c)
        p_ci_lo = delta - 1.96 * p_se

        per_dataset[ds] = dict(
            variant=args.variant, comparator=comp_id, variant_stats=v, comparator_stats=c,
            delta_pp=delta, welch_se=se, welch_ci_lower=ci_lo,
            protocol_se=p_se, protocol_ci_lower=p_ci_lo,
        )
        rep.add(f"significance.{ds}", "INFO",
                f"{args.variant} {v['mean']:.4f}+/-{v['std']:.4f} (n={v['n']}, seeds={v['seeds']}) "
                f"vs {comp_id} {c['mean']:.4f}+/-{c['std']:.4f} (n={c['n']}) | "
                f"delta={delta:+.4f} pp, Welch CI_lo={ci_lo:+.4f}, PROTOCOL CI_lo={p_ci_lo:+.4f}")
        rep.add(f"inventory.{ds}", "INFO",
                f"{v['n']} run(s) for {args.variant}: {', '.join(v['files'])}")

        # A comparator pooled across commits is a MOVING comparator: the same variant id can
        # denote different configurations at different commits (H30 @12 vs @40 is the known
        # case). Pooling them silently biases every delta.
        for label, stats, ident in (("variant", v, args.variant), ("comparator", c, comp_id)):
            if len(stats["commits"]) > 1:
                by_commit = {}
                pool = v_runs if label == "variant" else c_runs
                for r in pool:
                    key = str(r.get("git_commit", "")).split("+")[0][:8]
                    by_commit.setdefault(key, []).append(r["pct"])
                breakdown = "; ".join(
                    f"{k}: n={len(vals)}, mean={float(np.mean(vals)):.4f}"
                    for k, vals in sorted(by_commit.items()))
                rep.add(f"comparator_homogeneity.{ds}.{label}", "WARN",
                        f"{ident} runs span {len(stats['commits'])} commits — the same id may "
                        f"denote different configurations. Restrict by role/commit before "
                        f"quoting a delta. Breakdown: {breakdown}")

        # Runtime budget
        budget = ((campaign.get("runtime") or {}).get("max_wall_clock_s_per_run"))
        warn_at = ((campaign.get("runtime") or {}).get("warn_wall_clock_s_per_run"))
        if budget and v["max_wall_s"] > float(budget):
            rep.add(f"runtime.{ds}", "FAIL",
                    f"max wall {v['max_wall_s']:.0f}s exceeds budget {budget}s")
        elif warn_at and v["max_wall_s"] > float(warn_at):
            rep.add(f"runtime.{ds}", "WARN",
                    f"max wall {v['max_wall_s']:.0f}s over the soft band {warn_at}s "
                    f"(budget {budget}s)")
        else:
            rep.add(f"runtime.{ds}", "PASS", f"max wall {v['max_wall_s']:.0f}s within budget")

        # Equal-compute basis
        if len(v["grad_steps"]) > 1:
            rep.add(f"compute.{ds}", "WARN",
                    f"variant runs used differing total_grad_steps: {v['grad_steps']}")
        elif v["grad_steps"] and c["grad_steps"] and v["grad_steps"] != c["grad_steps"]:
            rep.add(f"compute.{ds}", "WARN",
                    f"grad-step budget differs from comparator: variant {v['grad_steps']} "
                    f"vs {comp_id} {c['grad_steps']} — equal-compute claim needs justification")
        else:
            rep.add(f"compute.{ds}", "PASS",
                    f"equal gradient budget ({v['grad_steps']}) vs {comp_id}")

    check_rescore(rep, all_variant_runs)

    rep.data = dict(variant=args.variant, comparator=args.comparator,
                    per_dataset=per_dataset, checks=rep.checks)

    print(f"\nAUDIT — {args.variant} (comparator: {args.comparator})")
    print(rep.render())
    verdict = "FAIL" if rep.failed else "PASS"
    print(f"\n  VERDICT: {verdict}"
          f"  ({sum(c['status'] == 'WARN' for c in rep.checks)} warning(s))\n")

    if args.out:
        Path(args.out).write_text(json.dumps(rep.data, indent=2))
        print(f"  report written to {args.out}\n")

    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
