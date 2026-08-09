#!/usr/bin/env python
"""Regenerate ``autoresearch/DASHBOARD.md`` — the one page a human reads to know where things are.

Everything here is derived from artifacts (``sota.json``, ``state.json``, ``queue.json``,
``killed.json``, ``results/*.json``). Never hand-write the dashboard: if it disagrees with the
artifacts, the artifacts are right.

    python autoresearch/dashboard.py
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _load(name: str) -> dict:
    p = _HERE / name
    return json.loads(p.read_text()) if p.exists() else {}


def _bar(frac: float, width: int = 32) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    sota, state, queue, killed = (_load(f) for f in
                                  ("sota.json", "state.json", "queue.json", "killed.json"))

    run_files = glob.glob(str(_ROOT / "results" / "*.json"))
    variants: dict = {}
    for f in run_files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        variants.setdefault(d.get("experiment_id") or d.get("algo"), []).append(d)

    L: list = []
    A = L.append

    A("# Campaign dashboard")
    A("")
    A(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `autoresearch/dashboard.py` — "
      "do not hand-edit.*")
    A("")

    phase = state.get("phase", "?")
    mode = state.get("mode", "?")
    A(f"**Phase:** `{phase}` · **cycle:** {state.get('cycle', 0)} · **mode:** `{mode}` · "
      f"**consecutive kills:** {state.get('consecutive_kills', 0)}")
    if mode == "blocked":
        A("")
        A(f"> ⚠ **BLOCKED — waiting for a human.** {state.get('notes', '')}")
    A("")

    # ── progress toward the Phase-1 target ────────────────────────────────────
    conn = sota.get("datasets", {}).get("connectome", {})
    target = sota.get("targets", {}).get("connectome", {})
    if conn and target:
        start = 83.9101          # the champion when the campaign was bootstrapped
        cur = float(conn.get("pct_mean", start))
        tgt = float(target.get("reference_pct", 84.6147))
        frac = (cur - start) / (tgt - start) if tgt > start else 1.0
        A("## Progress — connectome (Phase 1 target)")
        A("")
        A("```")
        A(f"  bootstrap  {start:.4f}%")
        A(f"  now        {cur:.4f}%   {_bar(frac)}  {100*frac:5.1f}% of the way")
        A(f"  target     {tgt:.4f}%   (reference solution / Vahidi 2025)")
        A(f"  remaining  {tgt - cur:+.4f} pp")
        A("```")
        A("")

    # ── champions ─────────────────────────────────────────────────────────────
    A("## Champions")
    A("")
    A("| dataset | champion | score | n | wall/run | evidence |")
    A("|---|---|---|---|---|---|")
    for ds, e in sota.get("datasets", {}).items():
        A(f"| {ds} | **{e.get('champion')}** | {e.get('pct_mean')} ± {e.get('pct_std')} | "
          f"{e.get('n_seeds')} | ~{e.get('wall_clock_s_approx')}s | {e.get('finding') or '—'} |")
    A("")

    # ── queue ─────────────────────────────────────────────────────────────────
    items = queue.get("items", [])
    proposed = [i for i in items if i.get("status") == "proposed"]
    A(f"## Queue — {len(proposed)} proposed / {len(items)} total")
    A("")
    A("| # | id | title | status |")
    A("|---|---|---|---|")
    for i in sorted(items, key=lambda x: x.get("priority", 99))[:12]:
        A(f"| {i.get('priority', '—')} | `{i['id']}` | {i.get('title', '')[:78]} | "
          f"{i.get('status')} |")
    A("")

    # ── history ───────────────────────────────────────────────────────────────
    hist = state.get("history", [])
    if hist:
        A("## Recent cycles")
        A("")
        A("| cycle | item | verdict | note |")
        A("|---|---|---|---|")
        for h in hist[-12:][::-1]:
            A(f"| {h.get('cycle')} | `{h.get('item')}` | **{h.get('verdict')}** | "
              f"{str(h.get('note', ''))[:80]} |")
        A("")

    # ── evidence volume ───────────────────────────────────────────────────────
    A("## Evidence")
    A("")
    A(f"- `results/*.json`: **{len(run_files)}** run records across "
      f"**{len(variants)}** variants")
    A(f"- killed mechanisms on record: **{len(killed.get('killed', []))}** "
      f"(+{len(killed.get('deferred_not_falsified', []))} deferred, "
      f"{len(killed.get('meta_rules', []))} meta-rules)")
    lit = sorted(glob.glob(str(_HERE / "lit" / "*.md")))
    A(f"- literature notes: **{len(lit)}**"
      + (f" (latest: `{Path(lit[-1]).name}`)" if lit else ""))
    A("")

    A("## Control")
    A("")
    A("```bash")
    A("cd " + str(_ROOT))
    A("nohup bash autoresearch/driver.sh > /dev/null 2>&1 &   # start")
    A("tail -f autoresearch/logs/driver.log                   # watch")
    A("touch autoresearch/STOP                                # stop after the current cycle")
    A("bash autoresearch/driver.sh --abort                    # stop now")
    A("```")
    A("")

    out = _HERE / "DASHBOARD.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out.relative_to(_ROOT)}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
