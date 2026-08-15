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
   PROTOCOL SE = std_comparator * sqrt(2/n) for continuity with findings.md, and — on
   SUPPORTING datasets — a pooled-variance non-inferiority bound.
5. LEAKAGE           — static scan of the variant module (and the refiner modules it imports)
   for the target metric, the reference solution, or hardcoded oracle values.
6. RUNTIME           — max wall-clock per run against ``campaign.yaml`` runtime budget.
7. PROVENANCE        — does the variant's own module exist at the commit its runs were stamped
   with? A run from a dirty tree at a commit that lacks the module is not reproducible by
   checkout (CLAUDE.md invariant 5, CAMPAIGN.md rule 3).

Exit code 0 = every hard check passed. Non-zero = at least one FAIL. WARNs never fail the run;
they are surfaced for the critic to adjudicate.

--gate: report (default) vs promotion
-------------------------------------
``--gate report`` is the historical behaviour and stays the default, because this script is used
for exploratory checking as well as for promotions: every significance result is INFO, the
degeneracy notice is WARN, and only the mechanical checks (frozen integrity, re-score, leakage,
comparator device, runtime, runtime guard) can fail the run.

``--gate promotion`` is the audit shape required before a variant may become the recorded
champion. It additionally FAILs on, all driven by ``campaign.yaml promotion_gate``:

  * ``delta_pp <= datasets.<ds>.min_promotion_delta_pp`` on any PRIMARY dataset;
  * ``protocol_ci_lower <= 0`` on any PRIMARY dataset;
  * the SUPPORTING (mouse) pooled non-inferiority bound falling at or below
    ``datasets.<ds>.non_inferiority_pp``;
  * an empty variant pool, an empty comparator pool, or an empty ``per_dataset``;
  * runs whose variant module does not exist at the commit they were stamped with.

Added 2026-08-15. Until then this script could not FAIL on any SCIENTIFIC ground at all, so the
question "is this a real gain?" was decided solely by the agent that produced the gain, and
``screen_delta_pp`` was evaluated in no code anywhere in the campaign.

Usage
-----
    python autoresearch/audit.py --variant H36
    python autoresearch/audit.py --variant H36 --role confirm --out autoresearch/audit_H36.json
    python autoresearch/audit.py --variant H36 --comparator H35
    python autoresearch/audit.py --variant H36 --role confirm --gate promotion
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

# Below this, a sample std is float noise, not dispersion. numpy's ddof=1 std over 20 IDENTICAL
# mouse pct values returns 1.458e-14 (catastrophic cancellation in the two-pass sum), which is
# why the old `std == 0.0` degeneracy tripwire did NOT fire on the H42 mouse pool while it fired
# on connectome and microns. Overridable from campaign.yaml promotion_gate.zero_std_tol.
ZERO_STD_TOL = 1e-12


class Report:
    def __init__(self, gate: str = "report") -> None:
        self.checks: List[dict] = []
        self.data: dict = {}
        # "promotion" upgrades the scientific gates from advisory to hard. Everything else is
        # identical, so an exploratory caller sees exactly the report it always saw.
        self.gate = gate

    @property
    def promotion(self) -> bool:
        return self.gate == "promotion"

    def add(self, name: str, status: str, detail: str, **extra) -> None:
        """status in {PASS, FAIL, WARN, INFO}."""
        self.checks.append(dict(check=name, status=status, detail=detail, **extra))

    def gated(self, name: str, ok: bool, detail: str, ok_detail: Optional[str] = None,
              **extra) -> None:
        """A scientific gate: PASS when it holds; FAIL in promotion mode, WARN otherwise.

        ``detail`` describes the VIOLATION. Report mode must never gain an exit code it did not
        have before 2026-08-15, so a violation there is only ever surfaced, never enforced.
        """
        if ok:
            self.add(name, "PASS", ok_detail or f"ok — {detail}", **extra)
        else:
            self.add(name, "FAIL" if self.promotion else "WARN", detail, **extra)

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
def run_device(d: dict) -> str:
    """Device a run was measured on, as recorded by the harness ('Apple MPS', a CUDA name...)."""
    return str((d.get("env") or {}).get("gpu") or "")


def load_runs(variant: str, dataset: Optional[str] = None,
              role: Optional[str] = None,
              device_tag: Optional[str] = None) -> List[dict]:
    """Result records for one variant, optionally restricted to a dataset/role/device.

    ``device_tag`` exists because sota.json is machine-specific (CAMPAIGN.md § Hardware): a score
    is produced by a particular device with its own kernels and its own non-determinism. Pooling
    MPS-measured and CUDA-measured runs of the same id is a moving comparator in the one dimension
    ``config_hash`` cannot see. It is off by default (single-machine history stays unaffected) and
    set from ``campaign.yaml environment.device_tag`` once a checkout has moved.
    """
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
        if device_tag and run_device(d) != device_tag:
            continue
        d["_file"] = p.name
        runs.append(d)
    return runs


def commit_key(d: dict) -> str:
    """Provenance key for one run: short hash PLUS any '+dirty' marker.

    This used to be ``str(git_commit).split("+")[0][:8]``, which STRIPPED the suffix and
    collapsed a clean checkout and a modified working tree at the same hash into one bucket —
    two different pieces of code recorded as one provenance. Verified 2026-08-15: H42's pool
    mixes 3791350d and 3791350d+dirty and the homogeneity check saw a single commit.
    """
    raw = str(d.get("git_commit", ""))
    base, sep, suffix = raw.partition("+")
    return base[:8] + (sep + suffix if sep else "")


def split_commit_key(key: str) -> Tuple[str, bool]:
    """(short hash, was the tree dirty) for a key produced by :func:`commit_key`."""
    base, sep, suffix = key.partition("+")
    return base, bool(sep and suffix)


def summarize(runs: List[dict], zero_std_tol: float = ZERO_STD_TOL) -> dict:
    pcts = np.array([r["pct"] for r in runs], dtype=np.float64)
    # ddof=1: sample std, matching PROTOCOL/findings. But a pool with a single distinct value
    # has a std of exactly 0 mathematically and 1.458e-14 numerically (see ZERO_STD_TOL), and
    # that 1e-14 both defeated the degeneracy tripwire and polluted the recorded sigma. Report
    # the mathematical answer.
    if len(pcts) > 1:
        std = float(pcts.std(ddof=1))
        if len(np.unique(pcts)) == 1 or std < zero_std_tol:
            std = 0.0
    else:
        std = 0.0
    return dict(
        n=len(runs),
        mean=float(pcts.mean()) if len(pcts) else float("nan"),
        std=std,
        seeds=sorted(int(r["seed"]) for r in runs),
        files=[r["_file"] for r in runs],
        max_wall_s=max((float(r.get("wall_clock_s", 0.0)) for r in runs), default=0.0),
        grad_steps=sorted({int(r.get("total_grad_steps") or r.get("n_epochs_done") or 0)
                           for r in runs}),
        # config_hash only covers (algo, dataset), so it CANNOT distinguish two configurations
        # of the same variant (e.g. H30 at 12 vs 40 sweeps). The commit is what separates them.
        commits=sorted({commit_key(r) for r in runs}),
        roles=sorted({str(r.get("role", "")) for r in runs}),
        # Same argument as commits, one level down: a pool spanning two devices is not one
        # measurement of one thing.
        devices=sorted({run_device(r) for r in runs}),
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
        # Frozen signature is (positions, src, tgt, weights) — see src/mfas/metrics.py.
        score = score_from_positions(pos, g.src, g.tgt, g.weight)
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


def protocol_se(comparator: dict, sigma_floor: float = 0.0) -> float:
    """PROTOCOL.md conservative SE = std_comparator * sqrt(2/n), with a noise floor.

    ``sigma_floor`` (campaign.yaml ``datasets.<ds>.baseline_sigma_pp``) matters on a device
    where the pipeline is deterministic: these variants take ``seed`` but never draw from it
    (``init_positions`` comes from greedy-FAS, so ``make_init_positions`` is never called), so
    seed-to-seed spread only ever measured DEVICE non-determinism. On CUDA that reproduces
    bit-identically, std collapses to 0, and an unfloored SE of 0 would make any positive
    delta -- +0.0001 pp included -- clear a 95% CI lower bound. That is a false-positive
    machine, and CAMPAIGN.md's "never claim a win inside noise" is what forbids it.
    """
    n = max(comparator["n"], 1)
    return max(comparator["std"], sigma_floor) * math.sqrt(2.0 / n)


def pooled_sd(a: dict, b: dict) -> float:
    """Standard pooled (within-group) sample SD of two arms."""
    na, nb = max(a["n"], 1), max(b["n"], 1)
    dof = na + nb - 2
    if dof <= 0:
        return max(a["std"], b["std"])
    return math.sqrt(((na - 1) * a["std"] ** 2 + (nb - 1) * b["std"] ** 2) / dof)


def noninferiority_ci(variant: dict, comparator: dict, resolution_pp: float,
                      z: float = 1.96) -> Tuple[float, float, float]:
    """(delta, se, ci_lower) for the SUPPORTING-dataset non-inferiority bound.

        sd    = max(pooled_sd(variant, comparator), resolution_pp)
        se    = sd * sqrt(1/n_v + 1/n_c)
        ci_lo = delta - z*se        ... compared against datasets.<ds>.non_inferiority_pp

    Replaces the reuse of :func:`protocol_se` here (D4, re-specified 2026-08-15). That formula
    was mathematically unpassable on mouse and had never changed a verdict:

      * it used ``sqrt(2/n_comparator)``, so the variant's 20 mouse seeds contributed NOTHING —
        H36 was judged on the comparator's n=3;
      * it floored sigma at ``baseline_sigma_pp`` = 0.2624, which is the RANDOM-INIT baseline's
        dispersion, imported onto a pipeline whose real dispersion is 0.

      Together those demanded delta > +0.16 pp to clear a -0.26 pp margin, on a dataset
      PROTOCOL.md describes as saturated with an expectation of 0.0000 pp. Measured: -0.4047 for
      H36, whose mouse score IMPROVED (92.90180243 -> 92.91701410), and -0.4199 for P02 at
      delta exactly 0. A false-negative machine, left armed.

    The replacement pools both arms' n (the variant's seeds now count) and floors sigma at the
    DATASET's ``measurement_resolution_pp`` — what the oracle can actually resolve — instead of
    at another algorithm's spread. On a deterministic pipeline se collapses to ~0 and the test
    reduces to the honest question "is the point estimate above the margin?"; on a stochastic
    one it is the ordinary pooled-variance non-inferiority test. Pinned by
    tests/test_audit_gates.py, including the historical H36 numbers.
    """
    delta = variant["mean"] - comparator["mean"]
    sd = max(pooled_sd(variant, comparator), max(resolution_pp, 0.0))
    se = sd * math.sqrt(1.0 / max(variant["n"], 1) + 1.0 / max(comparator["n"], 1))
    return delta, se, delta - z * se


# ──────────────────────────────────────────────────────────────────────────────
# 5b. Provenance — is this number reproducible by checkout?
# ──────────────────────────────────────────────────────────────────────────────
def _git_prefix() -> Optional[str]:
    """Path of ``_ROOT`` relative to the git work tree root, '' if identical, None if no git.

    The campaign checkout is a SUBDIRECTORY of the repository (``nn_backward_connections/``),
    so ``git cat-file -e <commit>:src/...`` silently reports "does not exist" for every commit
    unless the prefix is prepended. Getting this wrong would make the check fire on everything.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=_ROOT,
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    return out.stdout.strip()


def module_exists_at(commit: str, rel_path: str, prefix: str) -> Optional[bool]:
    """Does ``rel_path`` (relative to ``_ROOT``) exist in ``commit``? None = git could not say."""
    try:
        out = subprocess.run(["git", "cat-file", "-e", f"{commit}:{prefix}{rel_path}"],
                             cwd=_ROOT, capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode == 0:
        return True
    # git distinguishes "not in this tree" from "no such commit"; only the former is a verdict.
    # Every path verdict is phrased "fatal: path '<p>' ..." — two wordings are in use,
    # "... does not exist in '<rev>'" and "... exists on disk, but not in '<rev>'" (git 2.4x),
    # so match on the prefix rather than on either sentence. An unresolvable revision (shallow
    # clone, pruned history, another machine's hash) says "Not a valid object name" instead and
    # must NOT be reported as a missing module: this check is fail-SAFE toward "unknown",
    # because a false FAIL here would block a legitimate promotion for an infrastructure reason.
    if "fatal: path " in out.stderr:
        return False
    return None


def check_provenance(rep: Report, variant: str, label: str, runs: List[dict],
                     require_module: bool, warn_dirty: bool) -> None:
    """FAIL when the runs backing a promotion cannot be reproduced by ``git checkout``.

    CLAUDE.md invariant 5 and CAMPAIGN.md rule 3 both require it; nothing enforced it until
    2026-08-15. The cheap test is whether the variant's own module exists in the commit the runs
    are stamped with. Verified at the time this landed: all 36 H36 runs carry
    9d43b977...+dirty, a commit where ``src/mfas/experiments/H36.py`` does not exist, so the
    H36 championship rests on code that was never committed at the recorded hash.
    """
    if not runs:
        return
    rel = f"src/mfas/experiments/{variant}.py"
    prefix = _git_prefix()
    if prefix is None:
        rep.add(f"provenance.{label}", "WARN",
                "git unavailable — cannot verify that the runs are reproducible by checkout")
        return

    missing: List[str] = []
    dirty: List[str] = []
    unknown: List[str] = []
    for key in sorted({commit_key(r) for r in runs}):
        sha, is_dirty = split_commit_key(key)
        n = sum(1 for r in runs if commit_key(r) == key)
        if not sha:
            unknown.append(f"{key or '<no commit recorded>'} (n={n})")
            continue
        present = module_exists_at(sha, rel, prefix)
        if present is None:
            unknown.append(f"{key} (n={n})")
        elif not present:
            missing.append(f"{key}: n={n}")
        elif is_dirty:
            dirty.append(f"{key}: n={n}")

    if missing and require_module:
        rep.gated(f"provenance.{label}", False,
                  f"{rel} does not exist at the commit(s) these runs were stamped with, so the "
                  f"number is NOT reproducible by checkout (CLAUDE.md invariant 5): "
                  + "; ".join(missing))
    elif missing:
        rep.add(f"provenance.{label}", "WARN",
                f"{rel} missing at commit(s): " + "; ".join(missing))
    elif unknown:
        rep.add(f"provenance.{label}", "WARN",
                "could not resolve commit(s) against this repository: " + "; ".join(unknown))
    else:
        rep.add(f"provenance.{label}", "PASS",
                f"{rel} exists at every commit backing these runs")

    if dirty and warn_dirty:
        rep.add(f"provenance.{label}.dirty", "WARN",
                f"produced from a MODIFIED working tree at " + "; ".join(dirty)
                + f" — {rel} is present at those commits, so the run is reproducible in "
                  f"principle but the exact tree is unverifiable")


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
    ap.add_argument("--device-tag", default=None,
                    help="only count runs measured on this device (env.gpu). Defaults to "
                         "campaign.yaml environment.device_tag; pass '' to pool every device.")
    ap.add_argument("--gate", default="report", choices=["report", "promotion"],
                    help="'report' (default) = the historical permissive behaviour: significance "
                         "is INFO and only the mechanical checks can fail. 'promotion' = the "
                         "audit shape required before a champion changes; the effect-size, CI, "
                         "non-inferiority, evidence and provenance gates in campaign.yaml "
                         "promotion_gate become hard FAILs. update_sota.py uses 'promotion'.")
    args = ap.parse_args()

    rep = Report(gate=args.gate)
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

    ds_cfg = campaign.get("datasets") or {}
    gate_cfg = campaign.get("promotion_gate") or {}
    z = float(gate_cfg.get("z") or 1.96)
    zero_tol = float(gate_cfg.get("zero_std_tol") or ZERO_STD_TOL)
    g_primary = gate_cfg.get("primary") or {}
    g_supporting = gate_cfg.get("supporting") or {}
    g_evidence = gate_cfg.get("evidence") or {}
    g_prov = gate_cfg.get("provenance") or {}
    if rep.promotion and not gate_cfg:
        # Refusing to run a "strict" audit whose thresholds could not be read is the only safe
        # behaviour: silently falling back to defaults would re-create the very hole D1 names.
        rep.add("promotion_gate.config", "FAIL",
                "--gate promotion requires campaign.yaml promotion_gate, which is missing or "
                "unreadable — refusing to certify a promotion against unknown thresholds")
    elif rep.promotion and not gate_cfg.get("enabled", True):
        # An explicit, file-recorded decision to disable enforcement. It downgrades this run to
        # report mode rather than pretending to enforce; the line below is the audit trail.
        rep.gate = "report"
        rep.add("promotion_gate.config", "WARN",
                "campaign.yaml promotion_gate.enabled is false — scientific gates run as "
                "advisory WARNs and this audit cannot fail on them")
    elif rep.promotion:
        rep.add("promotion_gate.config", "INFO",
                f"strict promotion gate armed from campaign.yaml (z={z}, "
                f"zero_std_tol={zero_tol:g})")

    sota = json.loads((_HERE / "sota.json").read_text())

    # An explicit --device-tag wins; otherwise take the campaign's own device. '' disables it.
    device_tag = args.device_tag
    if device_tag is None:
        device_tag = (campaign.get("environment") or {}).get("device_tag") or None
    if device_tag:
        rep.add("device", "INFO", f"counting only runs measured on '{device_tag}' "
                                  f"(sota.json is machine-specific; see CAMPAIGN.md § Hardware)")

    all_variant_runs: List[dict] = []
    per_dataset: Dict[str, dict] = {}

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        d_cfg = ds_cfg.get(ds) or {}
        ds_role = str(d_cfg.get("role") or "primary")
        # D5 (2026-08-15): these three branches used to WARN-and-continue unconditionally, so an
        # audit with NO significance analysis at all still exited 0 with VERDICT: PASS. Verified
        # in audit_P02.json, where 2 of 3 datasets produced nothing and the audit passed.
        v_runs = load_runs(args.variant, ds, args.role, device_tag)
        if not v_runs:
            rep.gated(f"runs.{ds}", not g_evidence.get("require_variant_runs", True),
                      f"no runs found for {args.variant} on {ds} — nothing to promote on")
            continue
        all_variant_runs.extend(v_runs)

        comp_id = (sota["datasets"].get(ds, {}).get("champion")
                   if args.comparator == "champion" else args.comparator)
        if not comp_id:
            rep.gated(f"runs.{ds}", not g_evidence.get("require_comparator_runs", True),
                      f"no champion registered for {ds} — the delta has no reference point")
            continue
        c_runs = load_runs(comp_id, ds, args.comparator_role, device_tag)
        if not c_runs:
            rep.gated(f"comparator.{ds}", not g_evidence.get("require_comparator_runs", True),
                      f"no runs found for comparator {comp_id} on {ds} — a champion may not be "
                      f"installed on zero comparator evidence")
            continue

        v, c = summarize(v_runs, zero_tol), summarize(c_runs, zero_tol)
        delta, se, ci_lo = welch(v, c)
        sigma_floor = float(d_cfg.get("baseline_sigma_pp") or 0.0)
        resolution = float(d_cfg.get("measurement_resolution_pp") or 0.0)
        p_se = protocol_se(c, sigma_floor)
        p_ci_lo = delta - z * p_se
        ni_delta, ni_se, ni_ci_lo = noninferiority_ci(v, c, resolution, z)
        # D3: `== 0.0` missed the H42 mouse pool, whose std is 1.458e-14 of pure cancellation
        # noise over 20 identical values. summarize() now snaps that to 0.0 and this compares
        # against the same tolerance, so the tripwire cannot be defeated by float noise again.
        if c["std"] < zero_tol or v["std"] < zero_tol:
            rep.add(f"significance.{ds}.degenerate", "WARN",
                    f"zero variance in a pool (variant std={v['std']:.4g}, comparator "
                    f"std={c['std']:.4g}): the pipeline is deterministic on this device, so the "
                    f"Welch CI is degenerate (SE=0 makes any positive delta 'significant'). "
                    f"Use the PROTOCOL CI, floored at baseline_sigma_pp={sigma_floor:.4f}, and "
                    f"the min_promotion_delta_pp minimum effect size. See experiments/log.md P01.")

        per_dataset[ds] = dict(
            variant=args.variant, comparator=comp_id, variant_stats=v, comparator_stats=c,
            dataset_role=ds_role,
            delta_pp=delta, welch_se=se, welch_ci_lower=ci_lo,
            protocol_se=p_se, protocol_ci_lower=p_ci_lo,
            noninferiority_se=ni_se, noninferiority_ci_lower=ni_ci_lo,
        )
        rep.add(f"significance.{ds}", "INFO",
                f"{args.variant} {v['mean']:.4f}+/-{v['std']:.4f} (n={v['n']}, seeds={v['seeds']}) "
                f"vs {comp_id} {c['mean']:.4f}+/-{c['std']:.4f} (n={c['n']}) | "
                f"delta={delta:+.4f} pp, Welch CI_lo={ci_lo:+.4f}, PROTOCOL CI_lo={p_ci_lo:+.4f}")
        rep.add(f"inventory.{ds}", "INFO",
                f"{v['n']} run(s) for {args.variant}: {', '.join(v['files'])}")

        # ── The scientific gates (D1). Advisory in report mode, hard FAILs under --gate
        # promotion. Every threshold below comes from campaign.yaml; none is written here.
        if ds_role == "primary":
            min_delta = d_cfg.get("min_promotion_delta_pp")
            if min_delta is None:
                # Fall back to the screen-stage effect size, which — until 2026-08-15 — was read
                # by NO code anywhere in the campaign despite being quoted in gates.screen,
                # gates.confirm and CLAUDE.md as the minimum effect size.
                min_delta = d_cfg.get("screen_delta_pp")
            if min_delta is None:
                rep.add(f"effect_size.{ds}", "WARN",
                        f"no min_promotion_delta_pp or screen_delta_pp for {ds} — no minimum "
                        f"effect size could be enforced")
            elif g_primary.get("require_delta_gt_min_effect", True):
                rep.gated(f"effect_size.{ds}", delta > float(min_delta),
                          f"delta {delta:+.4f} pp does not clear the minimum effect size "
                          f"{float(min_delta):+.4f} pp for {ds} — a difference this small is not "
                          f"worth a champion change whatever its CI says",
                          ok_detail=f"delta {delta:+.4f} pp clears the minimum effect size "
                                    f"{float(min_delta):+.4f} pp")

            if g_primary.get("require_protocol_ci_lower_gt", None) is not None:
                floor = float(g_primary["require_protocol_ci_lower_gt"])
                rep.gated(f"protocol_ci.{ds}", p_ci_lo > floor,
                          f"PROTOCOL CI lower bound {p_ci_lo:+.4f} does not clear {floor:+.4f} "
                          f"(sigma floored at baseline_sigma_pp={sigma_floor:.4f}, "
                          f"SE={p_se:.4f}) — the gain is inside the protocol noise floor",
                          ok_detail=f"PROTOCOL CI lower bound {p_ci_lo:+.4f} > {floor:+.4f} "
                                    f"(SE={p_se:.4f}, sigma floor {sigma_floor:.4f})")
        else:
            margin = d_cfg.get("non_inferiority_pp")
            if margin is None:
                rep.add(f"non_inferiority.{ds}", "WARN",
                        f"no non_inferiority_pp for supporting dataset {ds} — not checked")
            elif g_supporting.get("require_non_inferiority", True):
                # See noninferiority_ci(): pooled n, sigma floored at the dataset's measurement
                # resolution. The old bound reused protocol_se() and demanded delta > +0.16 pp
                # on a saturated dataset; it returned -0.4047 for H36, whose mouse score rose.
                rep.gated(f"non_inferiority.{ds}", ni_ci_lo > float(margin),
                          f"non-inferiority CI lower bound {ni_ci_lo:+.4f} does not clear "
                          f"{float(margin):+.4f} pp (delta {ni_delta:+.4f}, pooled SE {ni_se:.3g} "
                          f"over n_v={v['n']}+n_c={c['n']}, sigma floored at "
                          f"measurement_resolution_pp={resolution:g})",
                          ok_detail=f"non-inferior: CI lower bound {ni_ci_lo:+.4f} > "
                                    f"{float(margin):+.4f} pp (delta {ni_delta:+.4f}, pooled SE "
                                    f"{ni_se:.3g} over n_v={v['n']}+n_c={c['n']})")
                rep.add(f"non_inferiority.{ds}.detail", "INFO",
                        f"delta={ni_delta:+.6f} pp, pooled SE={ni_se:.3g}, CI_lo={ni_ci_lo:+.6f} "
                        f"vs margin {float(margin):+.4f} pp")

        # D6: is this number reproducible by `git checkout`?
        check_provenance(rep, args.variant, ds, v_runs,
                         require_module=bool(g_prov.get("require_variant_module_at_commit", True)),
                         warn_dirty=bool(g_prov.get("warn_on_dirty_tree", True)))

        # A comparator pooled across commits is a MOVING comparator: the same variant id can
        # denote different configurations at different commits (H30 @12 vs @40 is the known
        # case). Pooling them silently biases every delta.
        for label, stats, ident in (("variant", v, args.variant), ("comparator", c, comp_id)):
            if len(stats["commits"]) > 1:
                by_commit = {}
                pool = v_runs if label == "variant" else c_runs
                for r in pool:
                    by_commit.setdefault(commit_key(r), []).append(r["pct"])
                breakdown = "; ".join(
                    f"{k}: n={len(vals)}, mean={float(np.mean(vals)):.4f}"
                    for k, vals in sorted(by_commit.items()))
                rep.add(f"comparator_homogeneity.{ds}.{label}", "WARN",
                        f"{ident} runs span {len(stats['commits'])} commits — the same id may "
                        f"denote different configurations. Restrict by role/commit before "
                        f"quoting a delta. Breakdown: {breakdown}")
            # A pool spanning devices is the same error in the dimension config_hash cannot see.
            if len(stats["devices"]) > 1:
                pool = v_runs if label == "variant" else c_runs
                by_dev: Dict[str, list] = {}
                for r in pool:
                    by_dev.setdefault(run_device(r) or "?", []).append(r["pct"])
                rep.add(f"comparator_homogeneity.{ds}.{label}.device", "FAIL",
                        f"{ident} runs span {len(stats['devices'])} devices — scores are "
                        f"machine-specific and must not be pooled. Re-run the re-baseline (queue "
                        f"P01) or pass --device-tag. Breakdown: " + "; ".join(
                            f"{k}: n={len(vals)}, mean={float(np.mean(vals)):.4f}"
                            for k, vals in sorted(by_dev.items())))

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

        # Runtime guard (P05). The cap above is only meaningful if something enforced it,
        # and a run the guard TRUNCATED did less work than it was configured to do — it is
        # valid but not comparable, so pooling it into a champion mean is exactly the kind
        # of silent error this auditor exists to make impossible. Hence FAIL, not WARN.
        # Records written before P05 carry no guard block at all; those only WARN, so this
        # check cannot retroactively invalidate the existing champion evidence.
        degraded = [r["_file"] for r in v_runs
                    if (r.get("runtime_guard") or {}).get("degraded")]
        unguarded = [r["_file"] for r in v_runs
                     if not (r.get("runtime_guard") or {}).get("armed")]
        if degraded:
            rep.add(f"runtime_guard.{ds}", "FAIL",
                    f"{len(degraded)}/{len(v_runs)} runs were TRUNCATED by the wall-clock "
                    f"guard and are not comparable to clean runs: {degraded}")
        elif unguarded:
            rep.add(f"runtime_guard.{ds}", "WARN",
                    f"{len(unguarded)}/{len(v_runs)} runs carry no armed runtime guard "
                    f"(pre-P05 records, or campaign.yaml was unreadable) — their wall clock "
                    f"was unbounded: {unguarded[:3]}{' ...' if len(unguarded) > 3 else ''}")
        else:
            rep.add(f"runtime_guard.{ds}", "PASS",
                    f"all {len(v_runs)} runs ran under an armed deadline, none truncated")

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

    # D5, the outer half: an audit that analysed NOTHING must not read as a pass. audit_P02.json
    # is the live example — 2 of 3 datasets produced no significance analysis and the file still
    # says VERDICT: PASS with an empty per_dataset.
    if g_evidence.get("require_nonempty_per_dataset", True):
        rep.gated("evidence.per_dataset", bool(per_dataset),
                  "no dataset produced a significance analysis — this audit certifies nothing",
                  ok_detail=f"significance analysed on {len(per_dataset)} dataset(s): "
                            f"{', '.join(sorted(per_dataset))}")

    rep.data = dict(variant=args.variant, comparator=args.comparator, gate=rep.gate,
                    per_dataset=per_dataset, checks=rep.checks)

    print(f"\nAUDIT — {args.variant} (comparator: {args.comparator}, gate: {rep.gate})")
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
