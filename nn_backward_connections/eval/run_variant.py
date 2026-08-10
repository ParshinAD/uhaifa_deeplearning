"""Variant runner — run a research variant, score with the FROZEN oracle, log JSON.

The frozen harness (``eval/harness.py``) only knows ``baseline_rocket``. This NON-frozen
runner executes variants from ``src/mfas/experiments/<exp_id>.py`` and scores their output
with the frozen scorer (``mfas.metrics.score_from_positions``). It REUSES (by import, never
editing) the harness's record/provenance helpers so the JSON schema stays identical and the
frozen ``eval/aggregate.py`` keeps working.

Usage
-----
    python -m eval.run_variant --exp <exp_id> --dataset {connectome|mouse} --seed S \
        --out results/ [--role implement|verify|confirm] [--time-limit S] [--device auto]

Records carry the same schema as the baseline harness plus ``experiment_id``, ``role`` and
``hypothesis``, so variant and baseline numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Make ``src`` importable when run as ``python -m eval.run_variant``.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io  # noqa: E402
from mfas.metrics import pct, score_from_positions  # noqa: E402
from mfas.utils.logging import get_logger  # noqa: E402
from mfas.utils.seeding import seed_everything, select_device  # noqa: E402

# Reuse frozen harness helpers by IMPORT (not edit) to keep one source of truth for
# provenance + the JSON schema.
from eval.harness import build_record, config_hash, _repo_relpath  # noqa: E402

# Integrity gate: refuse to produce/accept any score if a frozen file was tampered with
# (closes the Bash-write bypass the PreToolUse hook cannot catch).
from eval.frozen_guard import verify_frozen_manifest  # noqa: E402

# Runtime gate (P05): arm the wall-clock deadline the variants already honour but that
# nothing was ever passing them. See eval/runtime_guard.py.
from eval import runtime_guard  # noqa: E402


def _jsonable(obj):
    """Coerce numpy/pandas scalars and containers into JSON-serialisable Python.

    Variant provenance (``history.attrs``) is written by many different modules and freely
    mixes numpy scalars, lists of dicts and plain floats. One tolerant coercion here is
    cheaper than a convention every variant has to remember.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    return repr(obj)


def load_variant(exp_id: str):
    """Import the variant module ``src/mfas/experiments/<exp_id>.py``."""
    try:
        return importlib.import_module(f"mfas.experiments.{exp_id}")
    except ModuleNotFoundError as e:
        raise SystemExit(f"Unknown variant {exp_id!r}: {e}")


def run(exp_id: str, dataset: str, seed: int, out: Path, role: str,
        time_limit: Optional[float], device: str) -> dict:
    logger = get_logger("mfas.run_variant")

    # ── Frozen-oracle integrity gate (BEFORE anything is run or scored) ──────
    # Aborts with SystemExit if metrics/harness/aggregate/tests differ from
    # eval/frozen.sha256. A tampered oracle must never mint a score.
    verify_frozen_manifest()
    logger.info("frozen-oracle integrity: OK")

    variant = load_variant(exp_id)
    hypothesis = getattr(variant, "HYPOTHESIS", "")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chash = config_hash(dict(algo=exp_id, dataset=dataset))
    run_id = f"{stamp}-{exp_id}-{dataset}-s{seed}-{role}-{chash[:6]}"

    logger.info(f"[{run_id}] loading dataset={dataset}")
    g = io.load_dataset(dataset)
    logger.info(f"  {g!r}")

    seed_everything(seed)
    torch_device, dev_name = select_device(device)
    logger.info(f"  device={dev_name} | variant={exp_id} role={role}")

    # ── Runtime gate (P05) ───────────────────────────────────────────────────
    # An explicit --time-limit wins; otherwise fill the hole that left every run
    # unbounded. On an idle machine this deadline is never reached, so scores are
    # unchanged and bit-identical — verified before it was switched on.
    guard_plan = runtime_guard.resolve(dataset, time_limit)
    time_limit = guard_plan["time_limit_s"]
    if guard_plan["source"] == "campaign":
        logger.info(f"  runtime guard: deadline={time_limit:.0f}s "
                    f"(cap {guard_plan['cap_s']:.0f}s - reserve {guard_plan['reserve_s']:.0f}s)")
    elif guard_plan["source"] == "unavailable":
        logger.warning("  runtime guard: UNARMED — campaign.yaml runtime budget unreadable; "
                       "this run has no wall-clock bound (recorded as source=unavailable)")
    else:
        logger.info(f"  runtime guard: source={guard_plan['source']} limit={time_limit}")

    t0 = time.time()
    result = variant.run(g, seed=seed, device=torch_device, time_limit=time_limit)
    wall = time.time() - t0

    # Authoritative number: re-score produced positions with the FROZEN oracle.
    score = score_from_positions(result.best_positions, np.asarray(g.src),
                                 np.asarray(g.tgt), g.weight)
    pct_val = pct(score, g.total_weight)
    assert score == result.best_score, "run_variant/variant score mismatch"

    out.mkdir(parents=True, exist_ok=True)
    pos_path = out / f"{run_id}_positions.npy"
    np.save(pos_path, result.best_positions)

    record = build_record(
        exp_id=run_id, algo=exp_id, dataset=dataset, seed=seed,
        score=score, pct_val=pct_val, wall_clock_s=wall, cfg_hash=chash,
        total_weight=g.total_weight, n_epochs_done=result.n_epochs_done,
        positions_path=_repo_relpath(pos_path),
    )
    # Variant-specific provenance (extra keys; harmless to frozen aggregate.py).
    record["experiment_id"] = exp_id
    record["role"] = role
    record["hypothesis"] = hypothesis
    # Compute-matched comparison basis: every run records its total gradient steps
    # (= optimizer.step() calls, summed across any restarts). A variant's "win" is only
    # accepted vs a baseline run at the SAME total_grad_steps. Wall-clock (wall_clock_s)
    # stays logged for reference but is NOT the comparison basis (MPS timing is non-det).
    record["budget_basis"] = "total_grad_steps"
    record["total_grad_steps"] = int(result.n_epochs_done)

    # Per-stage provenance (queue item P06a). Every multi-stage variant since H30 computes
    # its stage scores and timings at runtime and stashes them in ``history.attrs`` — and
    # every one of them was thrown away here, which is why H42's runtime attribution had to
    # be INFERRED from stages 1-3 being byte-identical rather than read off a measurement.
    # Persisting it is additive: the frozen schema is untouched and aggregate.py ignores it.
    # P05 needs it too — "did the deadline truncate a stage?" is answered by the requested
    # vs done counts recorded here, and nowhere else.
    attrs = dict(getattr(result.history, "attrs", {}) or {})
    record["variant_attrs"] = _jsonable(attrs)

    # Runtime guard state (P05). A truncated run is valid but NOT comparable to a clean
    # one; this block is what stops it being pooled into a mean unnoticed.
    record["runtime_guard"] = runtime_guard.summarise(
        guard_plan, wall_clock_s=wall,
        variant_wall_s=getattr(result, "wall_clock_s", None), attrs=attrs)
    if record["runtime_guard"]["degraded"]:
        logger.warning(f"[{run_id}] RUNTIME GUARD BOUND: this run is DEGRADED and is not "
                       f"comparable to a clean run: "
                       f"{record['runtime_guard']['stages_truncated'] or 'deadline reached'}")
    if record["runtime_guard"]["over_cap"]:
        logger.warning(f"[{run_id}] wall {wall:.0f}s EXCEEDS the "
                       f"{guard_plan['cap_s']:.0f}s cap")

    json_path = out / f"{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"[{run_id}] DONE  score={score:,.4f}  pct={pct_val:.4f}%  "
                f"epochs={result.n_epochs_done}  wall={wall:.1f}s")
    logger.info(f"  wrote {_repo_relpath(json_path)}")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MFAS variant runner (frozen oracle)")
    p.add_argument("--exp", required=True, help="variant id (module in mfas.experiments)")
    p.add_argument("--dataset", required=True, choices=list(io.DATASETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="results/", type=Path)
    p.add_argument("--role", default="implement",
                   choices=["implement", "verify", "confirm"])
    p.add_argument("--time-limit", type=float, default=None, dest="time_limit")
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    args = p.parse_args(argv)

    run(exp_id=args.exp, dataset=args.dataset, seed=args.seed, out=args.out,
        role=args.role, time_limit=args.time_limit, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
