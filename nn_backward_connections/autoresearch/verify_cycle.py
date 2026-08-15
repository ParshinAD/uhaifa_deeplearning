#!/usr/bin/env python
"""Mechanically re-check a research cycle's BOOKKEEPING against the artifacts on disk.

Why this exists
---------------
``autoresearch/audit.py`` re-derives the numbers behind a *promotion*: it re-scores position
vectors with the frozen oracle, checks device homogeneity, leakage and runtime. It is the
evidence auditor and it is good at that.

Nothing, however, checked the *cycle* - the layer where an autonomous agent writes prose about
what it did. The first 7-cycle run showed exactly why that matters:

* Cycle #5 skipped the critic rung for TIME reasons ("confirm ran to 16:14 against a 17:08 cycle
  cap") and this survived only because the cycle volunteered it in prose. ``grep`` for
  ``gates_run|gate_status|critic_run`` across ``autoresearch/`` and ``.claude/`` returned zero
  hits: no field, no check, no dashboard column.
* Cycles #1 and #4 produced no commit and no ``state.json.history`` entry at all, yet the driver
  logged both as ``finished OK``. ``state.json.history`` runs 1, 2, 3, **5**, 6.
* ``state.json``'s free-text ``notes`` asserted that ``sweep.sh --role confirm`` ignores
  ``confirm_seeds`` for four days after commit ``2d4ba24`` fixed exactly that. Hand-maintained
  prose drifts away from the code it describes.

CAMPAIGN.md rule 3 is "never fabricate a number: every figure in every document traces to a
``results/*.json`` (or ``experiments/outputs/*.json``) plus a re-runnable command." That rule was
enforced by nothing. This script enforces it, plus the structural invariants above.

It is deliberately CHEAP and READ-ONLY: it parses JSON and text, never loads a dataset, never
touches the GPU, and never writes to the campaign's state. Run it after every cycle, and in CI.

Exit codes
----------
0   all checks passed (warnings may still be printed)
1   at least one FAIL
2   could not run (missing files, bad arguments)

Usage
-----
    python autoresearch/verify_cycle.py                    # check the most recent cycle
    python autoresearch/verify_cycle.py --cycle 5          # a specific cycle
    python autoresearch/verify_cycle.py --all              # every cycle in state.json.history
    python autoresearch/verify_cycle.py --json out.json    # machine-readable report
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# A score in this project is a percentage of total edge weight, quoted to 4 decimals
# (e.g. 84.1541). A delta is quoted in percentage points, usually 4 decimals (+0.0569).
# Those are the figures a verdict rests on, so those are the ones we trace. Deliberately NOT
# every number in the prose: dates, seeds, node counts and line numbers would drown the signal.
_PCT_RE = re.compile(r"(?<![\d.])(\d{2}\.\d{4,})(?![\d])")
_DELTA_RE = re.compile(r"(?<![\w.])([+-]\d+\.\d{4,})\s*pp")

# Gates the ladder requires before each verdict. CAMPAIGN.md: "never skip a rung".
_REQUIRED_GATES = {
    "keep":    ["novelty", "prototype", "screen", "confirm", "critic"],
    "kill":    ["novelty"],
    "iterate": ["novelty"],
    "done":    ["novelty"],
}


class Report:
    def __init__(self) -> None:
        self.checks: List[dict] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(dict(check=name, status=status, detail=detail))
        marker = {"PASS": "     ", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[info]"}[status]
        print(f"{marker} {name}: {detail}")

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_artifact_numbers() -> Set[str]:
    """Every score-shaped number that appears anywhere in the machine-written record.

    Sources are exactly the two CAMPAIGN.md names as traceable: ``results/*.json`` (written only
    by ``eval/run_variant.py``) and ``experiments/outputs/*.json`` (prototype outputs). Numbers are
    normalised to 4 decimal places, because that is the precision the prose quotes at, and a
    figure that matches an artifact to 4 dp is traceable in the sense the rule means.
    """
    seen: Set[str] = set()

    def harvest(obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for v in obj:
                harvest(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            f = float(obj)
            if 0.0 < abs(f) < 1e7:
                seen.add(f"{f:.4f}")
                # Also record the pp-delta shape: prose quotes deltas BETWEEN two artifact
                # numbers, which no single artifact contains. Deltas are handled separately.

    for pattern in ("results/*.json", "experiments/outputs/*.json",
                    "autoresearch/audit_*.json", "autoresearch/sota.json"):
        for p in glob.glob(str(_ROOT / pattern)):
            d = _load_json(Path(p))
            if d is not None:
                harvest(d)
    return seen


def collect_artifact_deltas(numbers: Set[str]) -> Set[str]:
    """Pairwise differences of artifact percentages, to 4 dp.

    A cycle's headline is usually a DELTA (`+0.0569 pp`), which appears in no artifact by itself -
    it is the difference of two that do. Rather than hardcode which pair, derive every pairwise
    difference among plausible percentage values. That is O(k^2) on a few thousand values, which
    is milliseconds, and it makes the check honest rather than approximate.
    """
    pcts = sorted({float(x) for x in numbers if 1.0 <= float(x) <= 100.0})
    out: Set[str] = {"0.0000"}
    for i, a in enumerate(pcts):
        for b in pcts[i + 1:]:
            d = round(b - a, 4)
            if 0.0 < d < 10.0:
                out.add(f"{d:.4f}")
    return out


def extract_cycle_section(log_md: str, cycle: int, item: Optional[str]) -> Optional[str]:
    """Pull one cycle's section out of experiments/log.md.

    The log's own convention is one ``## <date> - <ITEM>: <title> - <VERDICT>`` heading per cycle
    (e.g. ``## 2026-08-11 - P05: arm the run-level wall-clock guard - ITERATE``), so the ITEM ID is
    the reliable key, not the cycle number -- the cycle number appears nowhere in those headings.
    Fall back to a cycle-number match for any section written under a different convention, since
    the headings were authored by different sessions and are not perfectly uniform.
    """
    lines = log_md.splitlines()
    pats = []
    if item:
        # Word-boundary on the item id so P05 does not match P050.
        pats.append(re.compile(rf"^(#{{1,4}})\s.*\b{re.escape(item)}\b"))
    pats.append(re.compile(rf"^(#{{1,4}})\s.*\bcycle\s*#?{cycle}\b", re.IGNORECASE))

    for pat in pats:
        start = level = None
        for i, ln in enumerate(lines):
            m = pat.match(ln)
            if m:
                start, level = i, len(m.group(1))
                break
        if start is None:
            continue
        for j in range(start + 1, len(lines)):
            m2 = re.match(r"^(#{1,4})\s", lines[j])
            if m2 and len(m2.group(1)) <= level:
                return "\n".join(lines[start:j])
        return "\n".join(lines[start:])
    return None


def load_declared_figures(rep: Report) -> Dict[str, str]:
    """Figures that legitimately trace to no artifact IN THIS CHECKOUT, each with a reason.

    The honest case for this file: the campaign's own bootstrap number, connectome 83.9101, was
    measured on the previous machine (Apple MPS) and has no CUDA record here -- P01 re-measured
    the same champion at 83.9135. Quoting 83.9101 is not fabrication, but it IS a comparison
    against a foreign-hardware baseline, and the campaign should have to SAY so rather than have
    the number pass silently. So: declared figures are allowed, but only with a written reason,
    and they are reported as INFO so they stay visible.
    """
    path = _HERE / "known_figures.json"
    if not path.exists():
        return {}
    data = _load_json(path) or {}
    figs = {str(k): str(v) for k, v in (data.get("figures") or {}).items()}
    if figs:
        rep.add("figures.declared", "INFO",
                f"{len(figs)} figure(s) declared as legitimately artifact-less in "
                f"autoresearch/known_figures.json")
    return figs


def check_state_schema(rep: Report, state: dict) -> None:
    required = ["campaign", "phase", "cycle", "current_item", "consecutive_kills", "mode",
                "history"]
    missing = [k for k in required if k not in state]
    if missing:
        rep.add("state.schema", "FAIL", f"state.json is missing required key(s): {missing}")
    else:
        rep.add("state.schema", "PASS", "all required keys present")

    hist = state.get("history") or []
    nums = [h.get("cycle") for h in hist if isinstance(h, dict)]
    gaps = [n for n in range(1, (max(nums) if nums else 0) + 1) if n not in nums]
    if gaps:
        rep.add("state.history.contiguous", "FAIL",
                f"history has no entry for cycle(s) {gaps} - a cycle ran and wrote nothing. "
                f"Driver cycles #1 and #4 (2026-08-09/10) did exactly this and burned ~33 min "
                f"of budget invisibly.")
    else:
        rep.add("state.history.contiguous", "PASS",
                f"history is contiguous over {len(nums)} cycle(s)")

    # The prose fields are the ones that drift. Size is a proxy that catches it early.
    blob = sum(len(str(v)) for k, v in state.items() if k in ("last_cycle_outcome", "notes",
                                                             "current_item_note"))
    hist_blob = sum(len(str(h.get("note", ""))) for h in hist if isinstance(h, dict))
    total = blob + hist_blob
    if total > 60_000:
        rep.add("state.prose_size", "FAIL",
                f"{total:,} chars of free-text in state.json. Every cycle must read and rewrite "
                f"this; it grew 672 -> 13,072 bytes over the first 6 cycles. Move the narrative "
                f"to experiments/log.md and keep state.json machine-shaped.")
    elif total > 20_000:
        rep.add("state.prose_size", "WARN",
                f"{total:,} chars of free-text in state.json and growing - budget for a trim.")
    else:
        rep.add("state.prose_size", "PASS", f"{total:,} chars of free-text")


def check_gates(rep: Report, entry: dict, cycle: int) -> None:
    outcome = str(entry.get("outcome", "")).lower().strip()
    gates = entry.get("gates_run")
    if gates is None:
        rep.add(f"cycle{cycle}.gates_run", "FAIL",
                "no `gates_run` field. A skipped rung is then detectable only if the cycle "
                "volunteers it in prose - which is how cycle #5's skipped critic was nearly "
                "lost. Record the gates actually executed.")
        return
    if not isinstance(gates, (list, dict)):
        rep.add(f"cycle{cycle}.gates_run", "FAIL", f"`gates_run` is {type(gates).__name__}, "
                                                   f"expected a list or an object")
        return
    ran = set(gates if isinstance(gates, list) else
              [k for k, v in gates.items() if v not in (False, None, "skipped")])
    need = _REQUIRED_GATES.get(outcome)
    if need is None:
        rep.add(f"cycle{cycle}.gates_run", "WARN", f"unknown outcome {outcome!r}; gates={sorted(ran)}")
        return
    missing = [g for g in need if g not in ran]
    if missing:
        rep.add(f"cycle{cycle}.gates_run", "FAIL",
                f"verdict {outcome!r} requires {need}; missing {missing}. CAMPAIGN.md: "
                f"'never skip a rung'.")
    else:
        rep.add(f"cycle{cycle}.gates_run", "PASS", f"{outcome!r} ran {sorted(ran)}")


def check_numbers(rep: Report, section: str, cycle: int, artifact_nums: Set[str],
                  artifact_deltas: Set[str], declared: Dict[str, str]) -> None:
    """CAMPAIGN.md rule 3, made mechanical."""
    quoted = {m for m in _PCT_RE.findall(section)}
    untraceable, waived = [], []
    for q in sorted(quoted):
        if f"{float(q):.4f}" in artifact_nums:
            continue
        key = f"{float(q):.4f}"
        if key in declared or q in declared:
            waived.append(q)
        else:
            untraceable.append(q)

    if untraceable:
        rep.add(f"cycle{cycle}.figures", "FAIL",
                f"{len(untraceable)} score-shaped figure(s) in log.md trace to no artifact: "
                f"{untraceable[:8]}{' ...' if len(untraceable) > 8 else ''}. "
                f"CAMPAIGN.md rule 3: every figure traces to results/*.json or "
                f"experiments/outputs/*.json. If a figure is legitimately foreign (a prior "
                f"machine, a published reference), declare it with a reason in "
                f"autoresearch/known_figures.json instead of leaving it bare.")
    else:
        rep.add(f"cycle{cycle}.figures", "PASS",
                f"all {len(quoted)} score figure(s) trace to an artifact"
                + (f" ({len(waived)} by declaration)" if waived else ""))
    for w in waived:
        rep.add(f"cycle{cycle}.figures.declared.{w}", "INFO",
                f"{w} is quoted against a declared non-artifact figure: "
                f"{declared.get(f'{float(w):.4f}') or declared.get(w)}")

    deltas = {abs(float(d)) for d in _DELTA_RE.findall(section)}
    bad = sorted({f"{d:+.4f}" for d in deltas if f"{d:.4f}" not in artifact_deltas})
    if bad:
        rep.add(f"cycle{cycle}.deltas", "WARN",
                f"{len(bad)} pp-delta(s) are not a pairwise difference of any two artifact "
                f"figures: {bad[:8]}. This is a weaker check than the figures one (a delta may "
                f"legitimately be against a rounded or pooled value) - read them, do not assume.")
    else:
        rep.add(f"cycle{cycle}.deltas", "PASS",
                f"all {len(deltas)} pp-delta(s) reconcile with artifact figures")


def check_sota_backed(rep: Report) -> None:
    sota = _load_json(_ROOT / "autoresearch" / "sota.json")
    if not sota:
        rep.add("sota.readable", "FAIL", "autoresearch/sota.json missing or unparseable")
        return
    for ds, entry in (sota.get("datasets") or {}).items():
        champ, role = entry.get("champion"), entry.get("role", "confirm")
        n_declared = entry.get("n_seeds")
        files = glob.glob(str(_ROOT / "results" / f"*-{champ}-{ds}-*-{role}-*.json"))
        pcts = []
        for f in files:
            d = _load_json(Path(f))
            if d is not None and "pct" in d:
                pcts.append(float(d["pct"]))
        if not pcts:
            rep.add(f"sota.{ds}.backed", "FAIL",
                    f"champion {champ} has NO results/*.json at role={role}")
        elif n_declared is not None and len(pcts) != n_declared:
            rep.add(f"sota.{ds}.backed", "WARN",
                    f"champion {champ} declares n_seeds={n_declared} but {len(pcts)} run(s) "
                    f"are on disk at role={role}")
        else:
            rep.add(f"sota.{ds}.backed", "PASS",
                    f"champion {champ}: {len(pcts)} run(s) at role={role}")


def check_dashboard_fresh(rep: Report) -> None:
    dash = _ROOT / "autoresearch" / "DASHBOARD.md"
    state = _ROOT / "autoresearch" / "state.json"
    if not dash.exists():
        rep.add("dashboard.exists", "FAIL", "DASHBOARD.md missing - it is what a human reads first")
        return
    if dash.stat().st_mtime + 1 < state.stat().st_mtime:
        rep.add("dashboard.fresh", "FAIL",
                "DASHBOARD.md is older than state.json - regenerate it "
                "(python autoresearch/dashboard.py). It is the human-facing status and a stale "
                "one is worse than none.")
    else:
        rep.add("dashboard.fresh", "PASS", "DASHBOARD.md is at least as new as state.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycle", type=int, default=None, help="cycle number (default: latest)")
    ap.add_argument("--all", action="store_true", help="check every cycle in history")
    ap.add_argument("--json", type=str, default=None, help="write a machine-readable report here")
    args = ap.parse_args()

    state = _load_json(_ROOT / "autoresearch" / "state.json")
    if state is None:
        print("FATAL: autoresearch/state.json missing or unparseable")
        return 2
    log_path = _ROOT / "experiments" / "log.md"
    log_md = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    # The campaign has been bitten twice by Windows' cp1251 console codec (P01 fixed it for the
    # frozen manifest, cycle 3 for dashboard.py). Do not print anything this codec cannot encode:
    # keep output ASCII, and reconfigure the stream where the interpreter allows it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # py3.7+
    except Exception:
        pass

    rep = Report()
    print("=" * 78)
    print("CYCLE BOOKKEEPING VERIFICATION - re-derives the record from artifacts")
    print("=" * 78)

    check_state_schema(rep, state)
    check_sota_backed(rep)
    check_dashboard_fresh(rep)

    declared = load_declared_figures(rep)
    artifact_nums = collect_artifact_numbers()
    artifact_deltas = collect_artifact_deltas(artifact_nums)
    rep.add("artifacts.indexed", "INFO",
            f"{len(artifact_nums):,} score-shaped figures and {len(artifact_deltas):,} pairwise "
            f"deltas indexed from results/ and experiments/outputs/")

    hist = [h for h in (state.get("history") or []) if isinstance(h, dict)]
    if args.all:
        targets = [h.get("cycle") for h in hist]
    elif args.cycle is not None:
        targets = [args.cycle]
    else:
        targets = [hist[-1].get("cycle")] if hist else []

    for cyc in targets:
        entry = next((h for h in hist if h.get("cycle") == cyc), None)
        if entry is None:
            rep.add(f"cycle{cyc}.entry", "FAIL", f"no history entry for cycle {cyc}")
            continue
        check_gates(rep, entry, cyc)
        section = extract_cycle_section(log_md, cyc, entry.get("item"))
        if section is None:
            rep.add(f"cycle{cyc}.log_section", "WARN",
                    f"no section for cycle {cyc} (item {entry.get('item')!r}) found in "
                    f"experiments/log.md - check by hand")
        else:
            check_numbers(rep, section, cyc, artifact_nums, artifact_deltas, declared)

    verdict = "FAIL" if rep.failed else "PASS"
    n_fail = sum(1 for c in rep.checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in rep.checks if c["status"] == "WARN")
    print("-" * 78)
    print(f"VERDICT: {verdict}   ({n_fail} FAIL, {n_warn} WARN, {len(rep.checks)} checks)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(verdict=verdict, checks=rep.checks), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
