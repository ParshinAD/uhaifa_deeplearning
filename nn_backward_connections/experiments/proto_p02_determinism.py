"""P02 prototype gate — is a screen seed inert, and is this device deterministic?

Question (queue item P02). On this machine (CUDA / RTX 4060) the screen runs 3 seeds x 3
datasets. P01 observed that H35 on connectome produced BIT-IDENTICAL positions at seeds
42/123/999, i.e. 2/3 of every screen's wall-clock bought no information. Before amending the
protocol we must separate two effects that P01 could not tell apart:

  A. SEED INERTNESS — the pipeline never draws from ``seed`` (init_positions comes from
     deterministic greedy-FAS, so make_init_positions / torch.randn are never reached).
  B. DEVICE DETERMINISM — repeating the SAME seed reproduces bit-identically, i.e. the CUDA
     kernels (gather/scatter with atomic accumulation over millions of edges) do not inject
     run-to-run noise.

Both must hold for "1 seed is enough". If only A holds, repeats still buy information about
device noise and the protocol must run 3 REPEATS rather than 3 seeds — a materially different
amendment. The 3-run design below (s=42, s=42 again, s=123) is what separates them:

    repeat(42) == run(42)   ->  device deterministic          (B)
    run(123)   == run(42)   ->  seed inert                    (A)

Cheap proxy: mouse (148 nodes, ~6 s/run) exercised across EVERY variant in
src/mfas/experiments/, which is what makes this a classifier test and not an anecdote — a
single variant that is classified deterministic but differs across seeds falsifies the whole
amendment (P02's stated kill condition).

Prototype rung: no writes to results/ (variants are called directly, not through
eval.run_variant), CPU/GPU minutes only. Output: experiments/outputs/proto_P02.json.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io  # noqa: E402
from mfas.metrics import pct, score_from_positions  # noqa: E402
from mfas.utils.seeding import seed_everything, select_device  # noqa: E402

# Every variant module in src/mfas/experiments/ (the classifier must cover all of them).
VARIANTS = [
    "baseline_passthrough", "baseline_multistart",
    "H01", "H02", "H03", "H04", "H05", "H06", "H09", "H11",
    "H13", "H16", "H19", "H30", "H31", "H35",
]

# Static classification, read off the source (verified by the runs below).
#   "deterministic"  = passes init_positions from deterministic greedy-FAS into run_rocket
#                      and draws from no other RNG stream
#   "rng"            = reaches torch.randn / make_init_positions / a seeded RandomState
STATIC_CLASS = {
    "baseline_passthrough": ("rng", "run_rocket init_mode='random' -> torch.randn(n)"),
    "baseline_multistart": ("rng", "K restarts, each a distinct random init"),
    "H01": ("rng", "multi-start: K sub-seeds -> different random inits"),
    "H02": ("det", "greedy_fas_order -> _init_positions_from_order -> init_positions"),
    "H03": ("rng", "own loop, init_mode='random' -> torch.randn(n)"),
    "H04": ("rng", "own loop, torch.randn(n) init"),
    "H05": ("rng", "AdamW swap on the baseline random-init loop"),
    "H06": ("rng", "loss reweighting on the baseline random-init loop"),
    "H09": ("rng", "anti-tie jitter on the baseline random-init loop"),
    "H11": ("rng", "margin surrogate, init_mode='random' -> torch.randn(n)"),
    "H13": ("rng", "mini-batch: RandomState(seed+104729) draws every step"),
    "H16": ("det", "H02 warm start copied verbatim; monotone beta only"),
    "H19": ("rng", "soft-rank surrogate on its own init path"),
    "H30": ("det", "H02 warm start -> run_rocket -> Jacobi sift (no RNG)"),
    "H31": ("rng", "LNS destroy: RandomState(seed + 7919)"),
    "H35": ("det", "H02 warm start -> run_rocket -> under-relaxed sift (no RNG)"),
}

# (label, seed) — the 3-run design. Two runs at seed 42 separate device noise from seed effect.
RUNS = [("a42", 42), ("b42", 42), ("c123", 123)]

DATASET = "mouse"
TIME_LIMIT = None


def _digest(arr: np.ndarray) -> str:
    """Bit-exact fingerprint of a position vector (dtype + shape + raw bytes)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    g = io.load_dataset(DATASET)
    torch_device, dev_name = select_device("auto")
    print(f"dataset={DATASET} n={g.n_nodes} device={dev_name}")

    out = {
        "_doc": "P02 prototype gate: separates seed inertness (A) from device determinism (B). "
                "Runs every variant on mouse at seeds (42, 42, 123); compares positions "
                "bit-for-bit. See experiments/proto_p02_determinism.py.",
        "dataset": DATASET,
        "n_nodes": int(g.n_nodes),
        "device": dev_name,
        "runs": [r[0] for r in RUNS],
        "static_class": {k: v[0] for k, v in STATIC_CLASS.items()},
        "static_reason": {k: v[1] for k, v in STATIC_CLASS.items()},
        "variants": {},
    }

    for vid in VARIANTS:
        try:
            mod = importlib.import_module(f"mfas.experiments.{vid}")
        except Exception as e:  # pragma: no cover - reported, not raised
            out["variants"][vid] = {"error": f"import failed: {e}"}
            print(f"{vid:22s} IMPORT FAIL {e}")
            continue

        rec = {"digests": {}, "scores": {}, "pcts": {}, "wall_s": {}}
        ok = True
        for label, seed in RUNS:
            try:
                seed_everything(seed)
                t0 = time.time()
                res = mod.run(g, seed=seed, device=torch_device, time_limit=TIME_LIMIT)
                wall = time.time() - t0
                pos = np.asarray(res.best_positions)
                score = score_from_positions(pos, np.asarray(g.src), np.asarray(g.tgt),
                                             g.weight)
                rec["digests"][label] = _digest(pos)
                rec["scores"][label] = float(score)
                rec["pcts"][label] = float(pct(score, g.total_weight))
                rec["wall_s"][label] = round(wall, 2)
            except Exception as e:  # pragma: no cover
                ok = False
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()[-800:]
                break

        if ok:
            d = rec["digests"]
            s = rec["scores"]
            # B: same seed twice -> identical?
            rec["device_deterministic"] = d["a42"] == d["b42"]
            # A: different seed -> identical? (only meaningful if B holds)
            rec["seed_inert_positions"] = d["a42"] == d["c123"]
            rec["seed_inert_scores"] = s["a42"] == s["c123"]
            rec["observed_class"] = ("det" if (rec["device_deterministic"]
                                               and rec["seed_inert_positions"]) else "rng")
            rec["static_class"] = STATIC_CLASS.get(vid, ("?", ""))[0]
            rec["classifier_agrees"] = rec["observed_class"] == rec["static_class"]
            print(f"{vid:22s} static={rec['static_class']:3s} obs={rec['observed_class']:3s} "
                  f"devdet={rec['device_deterministic']!s:5s} "
                  f"seedinert={rec['seed_inert_positions']!s:5s} "
                  f"agree={rec['classifier_agrees']!s:5s} "
                  f"pct={rec['pcts']['a42']:.4f} wall={rec['wall_s']['a42']}s")
        else:
            print(f"{vid:22s} RUN FAIL {rec.get('error')}")

        out["variants"][vid] = rec

    # ── Verdict ───────────────────────────────────────────────────────────────
    done = {k: v for k, v in out["variants"].items() if "observed_class" in v}
    disagree = [k for k, v in done.items() if not v["classifier_agrees"]]
    false_det = [k for k, v in done.items()
                 if v["static_class"] == "det" and v["observed_class"] == "rng"]
    nondet = [k for k, v in done.items() if not v["device_deterministic"]]
    out["summary"] = {
        "n_variants_run": len(done),
        "n_failed": len(out["variants"]) - len(done),
        "classifier_disagreements": disagree,
        "false_deterministic": false_det,   # the DANGEROUS error: claimed det, actually varies
        "device_nondeterministic": nondet,
        "prototype_pass": len(false_det) == 0 and len(nondet) == 0,
    }
    print("\n--- summary ---")
    print(json.dumps(out["summary"], indent=2))

    dest = _ROOT / "experiments" / "outputs" / "proto_P02.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
