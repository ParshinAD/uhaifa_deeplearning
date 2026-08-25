"""Pin a run's ordering into the repository so the other machine can evaluate it.

WHY THIS EXISTS. The campaign runs on the Windows/RTX box; results are evaluated on the
macOS laptop. That workflow silently did not work: `.gitignore:30` ignores
`results/*_positions.npy` with a single whitelisted exception, so exactly ONE ordering
artifact is tracked in the whole repository -- and it is not a champion. The champion's
order has never left the machine that produced it, which means on this laptop it cannot be
re-scored, diagnosed, or compared against. Every "advantage vs the champion" number is
undefined here for that reason alone, before any device question.

WHAT IT WRITES. `results/champions/<algo>_<dataset>_s<seed>_<device>.npz`, whitelisted in
`.gitignore`, holding:

  rank            int32[n]   the ordering, as ranks (NOT float positions)
  meta            json       exp_id, algo, dataset, seed, device_class, score, pct,
                             source record, git_commit, sha256 of the rank bytes

int32 ranks rather than float32 positions, because ranks are what every downstream probe
consumes, they compress, and they cannot silently lose precision. A connectome ordering is
546,720 bytes as float32 positions and compresses to a fraction of that as int32 ranks.

VERIFICATION IS NOT OPTIONAL. The pinned rank is re-scored through the frozen oracle and
must reproduce the record's own score exactly, in the record's native integer dtype. A
mismatch means the artifact and the record are not the same run, and pinning is refused.

Usage
-----
    $PY tools/pin_champion.py results/<exp_id>.json
    $PY tools/pin_champion.py --verify-only results/champions/H42_connectome_s42_cuda.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mfas.device_policy import device_class, normalise_artifact_path  # noqa: E402
from mfas.io import load_dataset                                      # noqa: E402
from mfas.metrics import pct as pct_of, score_from_order              # noqa: E402

CHAMPIONS = ROOT / "results" / "champions"


def positions_to_rank(arr: np.ndarray) -> np.ndarray:
    """Accept either an integer rank vector or float positions; return int32 ranks."""
    if arr.dtype.kind in "iu":
        rank = arr.astype(np.int64)
    else:
        rank = np.argsort(np.argsort(arr)).astype(np.int64)
    n = rank.size
    if not np.array_equal(np.sort(rank), np.arange(n)):
        raise ValueError("artifact is not a permutation")
    return rank.astype(np.int32)


def pin(record_path: Path, *, force: bool = False) -> Path:
    rec = json.loads(record_path.read_text())
    dev = device_class(rec)

    src = normalise_artifact_path(rec.get("best_positions_path"))
    if src is None:
        raise SystemExit(f"{record_path.name}: record has no best_positions_path")
    src_abs = ROOT / src
    if not src_abs.exists():
        raise SystemExit(
            f"{record_path.name}: the ordering artifact is not on this machine "
            f"({src}). It was never committed -- see .gitignore:30. Run this tool on the "
            f"machine that produced the run, then commit results/champions/."
        )

    rank = positions_to_rank(np.load(src_abs))

    g = load_dataset(rec["dataset"])
    score = score_from_order(rank.astype(np.int64), g.src, g.tgt, g.weight)
    if float(score) != float(rec["score"]):
        raise SystemExit(
            f"REFUSED: re-scored {score} != recorded {rec['score']}. The artifact and the "
            f"record are not the same run."
        )

    CHAMPIONS.mkdir(parents=True, exist_ok=True)
    out = CHAMPIONS / f"{rec['algo']}_{rec['dataset']}_s{rec['seed']}_{dev}.npz"
    if out.exists() and not force:
        raise SystemExit(f"{out.name} already pinned; pass --force to replace")

    meta = {
        "exp_id": rec["exp_id"],
        "algo": rec["algo"],
        "dataset": rec["dataset"],
        "seed": rec["seed"],
        "device_class": dev,
        "score": float(score),
        "pct": float(pct_of(score, g.total_weight)),
        "source_record": record_path.name,
        "git_commit": rec.get("git_commit"),
        "rank_sha256": hashlib.sha256(rank.tobytes()).hexdigest(),
        "n_nodes": int(rank.size),
    }
    np.savez_compressed(out, rank=rank, meta=json.dumps(meta))
    print(f"pinned {out.relative_to(ROOT)}  pct={meta['pct']:.6f}  device={dev}  "
          f"sha256={meta['rank_sha256'][:16]}  ({out.stat().st_size:,} bytes)")
    return out


def verify(npz_path: Path) -> None:
    z = np.load(npz_path, allow_pickle=False)
    rank, meta = z["rank"], json.loads(str(z["meta"]))
    if hashlib.sha256(rank.tobytes()).hexdigest() != meta["rank_sha256"]:
        raise SystemExit(f"{npz_path.name}: rank bytes do not match the recorded sha256")
    g = load_dataset(meta["dataset"])
    score = score_from_order(rank.astype(np.int64), g.src, g.tgt, g.weight)
    if float(score) != meta["score"]:
        raise SystemExit(f"{npz_path.name}: re-score {score} != pinned {meta['score']}")
    print(f"OK {npz_path.name}  pct={pct_of(score, g.total_weight):.6f}  "
          f"device={meta['device_class']}  from {meta['source_record']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path, help="a results/*.json record, or a pinned .npz")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.verify_only or args.path.suffix == ".npz":
        verify(args.path)
    else:
        pin(args.path, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
