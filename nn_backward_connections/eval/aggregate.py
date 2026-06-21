"""Aggregate ``results/*.json`` into a mean±std Markdown table in experiments/log.md.

Groups records by ``(algo, dataset, config_hash)`` and reports pct / score
mean ± sample std (ddof=1), n_seeds, mean wall-clock. The aggregated table is
written between delimiter markers so manual lab notes around it are preserved.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
_BEGIN = "<!-- BEGIN AGGREGATED RESULTS (auto-generated) -->"
_END = "<!-- END AGGREGATED RESULTS -->"


def load_records(pattern: str) -> List[dict]:
    records = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            records.append(json.load(f))
    return records


def _fmt_mean_std(vals, fmt="{:.4f}"):
    if len(vals) == 1:
        return f"{fmt.format(vals[0])} (n=1)"
    m = statistics.mean(vals)
    s = statistics.stdev(vals)  # sample std, ddof=1
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def aggregate(records: List[dict]) -> str:
    groups = {}
    for r in records:
        key = (r["algo"], r["dataset"], r["config_hash"])
        groups.setdefault(key, []).append(r)

    lines = [
        _BEGIN,
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"from {len(records)} run(s)._",
        "",
        "| algo | dataset | n_seeds | pct mean±std | score mean±std | "
        "wall_clock_s (mean) | seeds | config_hash | git_commit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (algo, dataset, chash), rs in sorted(groups.items()):
        pcts = [r["pct"] for r in rs]
        scores = [r["score"] for r in rs]
        walls = [r["wall_clock_s"] for r in rs]
        seeds = sorted(r["seed"] for r in rs)
        commits = sorted({(r.get("git_commit") or "n/a")[:11] for r in rs})
        # Large integer scores (connectome) -> thousands, no decimals; small float
        # scores (mouse) -> 4 decimals so std is visible.
        score_fmt = "{:,.0f}" if max(abs(s) for s in scores) >= 1000 else "{:,.4f}"
        lines.append(
            f"| {algo} | {dataset} | {len(rs)} | {_fmt_mean_std(pcts)} | "
            f"{_fmt_mean_std(scores, score_fmt)} | "
            f"{statistics.mean(walls):.1f} | {seeds} | {chash[:6]} | "
            f"{', '.join(commits)} |"
        )
    lines += ["", _END]
    return "\n".join(lines)


def write_into(log_path: Path, block: str) -> None:
    """Insert/replace the aggregated block in ``log_path``, preserving other text."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        text = log_path.read_text()
    else:
        text = "# Experiment Log\n\nAppend-only lab notebook. The block below is auto-generated.\n\n"
    if _BEGIN in text and _END in text:
        pre = text[: text.index(_BEGIN)]
        post = text[text.index(_END) + len(_END):]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    log_path.write_text(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate result JSON into log.md")
    p.add_argument("--glob", default=str(_ROOT / "results" / "*.json"))
    p.add_argument("--out", default=str(_ROOT / "experiments" / "log.md"), type=Path)
    args = p.parse_args(argv)

    records = load_records(args.glob)
    if not records:
        print(f"No records matched {args.glob}")
        return 1
    block = aggregate(records)
    write_into(Path(args.out), block)
    print(block)
    print(f"\nWrote aggregated table into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
