"""Evaluation harness — run an algorithm, score it with the exact oracle, log JSON.

Usage
-----
    python -m eval.harness --algo baseline_rocket --dataset connectome --seed 42 \
        --out results/ [--config configs/baseline_rocket.yaml] \
        [--epochs N] [--time-limit S] [--device auto|mps|cuda|cpu]

Writes a JSON record with a FIXED schema (see ``build_record``) plus the best
position vector to ``results/<exp_id>_positions.npy``. The reported ``score``/``pct``
always come from the exact scorer in :mod:`mfas.metrics` applied to the produced
positions — never from any algorithm-internal estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Make ``src`` importable when run as ``python -m eval.harness``.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from mfas import io  # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket  # noqa: E402
from mfas.metrics import pct, score_from_positions  # noqa: E402
from mfas.utils.logging import get_logger  # noqa: E402
from mfas.utils.seeding import seed_everything, select_device  # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "baseline_rocket.yaml"


def _repo_relpath(p: Path) -> str:
    """Path relative to the repo root if possible, else absolute (for logging)."""
    try:
        return str(p.resolve().relative_to(_ROOT))
    except ValueError:
        return str(p.resolve())


# ──────────────────────────────────────────────────────────────────────────────
# Environment / provenance
# ──────────────────────────────────────────────────────────────────────────────
def git_commit() -> Optional[str]:
    """Current commit of the enclosing repo, suffixed ``+dirty`` if uncommitted."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             cwd=str(_ROOT), capture_output=True, text=True, check=True)
        repo = top.stdout.strip()
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def env_info() -> dict:
    """Capture python / framework / device provenance."""
    torch_v = None
    cuda = None
    gpu = None
    try:
        import torch
        torch_v = f"torch-{torch.__version__}"
        cuda = torch.version.cuda
        if torch.backends.mps.is_available():
            gpu = "Apple MPS"
        elif torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "CPU"
    except ImportError:
        pass
    return dict(python=platform.python_version(), jax_or_torch=torch_v,
                cuda=cuda, gpu=gpu)


def config_hash(effective: dict) -> str:
    """sha256 (12 hex) of the effective config, EXCLUDING seed and device.

    Seeds of the same experimental setup therefore share a hash, which lets the
    aggregator group them for mean±std.
    """
    payload = {k: v for k, v in effective.items() if k not in ("seed", "device")}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────────────
# Config merge
# ──────────────────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def effective_rocket_config(cfg_yaml: dict, dataset: str,
                            epochs: Optional[int]) -> RocketConfig:
    """Merge defaults ⊕ dataset_overrides ⊕ CLI epoch override into a RocketConfig."""
    rk = dict(cfg_yaml.get("defaults", {}).get("rocket", {}))
    overrides = cfg_yaml.get("dataset_overrides", {}).get(dataset, {})
    rk.update(overrides.get("rocket", {}))
    if epochs is not None:
        rk["epochs"] = epochs
    # keep only valid RocketConfig fields
    valid = RocketConfig.__dataclass_fields__.keys()
    rk = {k: v for k, v in rk.items() if k in valid}
    return RocketConfig(**rk)


# ──────────────────────────────────────────────────────────────────────────────
# Record
# ──────────────────────────────────────────────────────────────────────────────
def build_record(exp_id, algo, dataset, seed, score, pct_val, wall_clock_s,
                 cfg_hash, total_weight, n_epochs_done, positions_path) -> dict:
    """Assemble the fixed-schema JSON record."""
    return dict(
        exp_id=exp_id,
        algo=algo,
        dataset=dataset,
        seed=seed,
        score=score,
        pct=pct_val,
        wall_clock_s=wall_clock_s,
        git_commit=git_commit(),
        env=env_info(),
        config_hash=cfg_hash,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # extras (beyond the required schema) — useful for audit/aggregation
        total_weight=total_weight,
        n_epochs_done=n_epochs_done,
        best_positions_path=positions_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────
def run(algo: str, dataset: str, seed: int, out: Path, config: Path,
        epochs: Optional[int], time_limit: Optional[float], device: str) -> dict:
    logger = get_logger("mfas.harness")
    if algo != "baseline_rocket":
        raise ValueError(f"Unknown algo {algo!r} (only 'baseline_rocket' available)")

    cfg_yaml = load_config(config)
    rcfg = effective_rocket_config(cfg_yaml, dataset, epochs)
    dev_pref = device if device != "auto" else cfg_yaml.get("device", "auto")

    # exp_id / config_hash
    effective = dict(algo=algo, dataset=dataset, **rcfg.__dict__)
    chash = config_hash(effective)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_id = f"{stamp}-{algo}-{dataset}-s{seed}-{chash[:6]}"

    logger.info(f"[{exp_id}] loading dataset={dataset}")
    g = io.load_dataset(dataset)
    logger.info(f"  {g!r}")

    seed_everything(seed)
    torch_device, dev_name = select_device(dev_pref)
    logger.info(f"  device={dev_name} | rocket={rcfg}")

    t0 = time.time()
    result = run_rocket(g, rcfg, seed=seed, device=torch_device,
                        time_limit=time_limit, logger=None)
    wall = time.time() - t0

    # Re-score the produced positions with the exact oracle (authoritative number).
    score = score_from_positions(result.best_positions, np.asarray(g.src),
                                 np.asarray(g.tgt), g.weight)
    pct_val = pct(score, g.total_weight)
    assert score == result.best_score, "harness/runner score mismatch"

    out.mkdir(parents=True, exist_ok=True)
    pos_path = out / f"{exp_id}_positions.npy"
    np.save(pos_path, result.best_positions)

    record = build_record(
        exp_id=exp_id, algo=algo, dataset=dataset, seed=seed,
        score=score, pct_val=pct_val, wall_clock_s=wall, cfg_hash=chash,
        total_weight=g.total_weight, n_epochs_done=result.n_epochs_done,
        positions_path=_repo_relpath(pos_path),
    )
    json_path = out / f"{exp_id}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"[{exp_id}] DONE  score={score:,.0f}  pct={pct_val:.4f}%  "
                f"epochs={result.n_epochs_done}  wall={wall:.1f}s")
    logger.info(f"  wrote {_repo_relpath(json_path)}")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MFAS evaluation harness")
    p.add_argument("--algo", default="baseline_rocket")
    p.add_argument("--dataset", required=True, choices=list(io.DATASETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="results/", type=Path)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None, dest="time_limit")
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    args = p.parse_args(argv)

    run(algo=args.algo, dataset=args.dataset, seed=args.seed, out=args.out,
        config=args.config, epochs=args.epochs, time_limit=args.time_limit,
        device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
