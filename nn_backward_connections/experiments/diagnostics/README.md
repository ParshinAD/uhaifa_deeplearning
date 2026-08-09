# experiments/diagnostics/ — Track B stable diagnostic scripts

Promoted, re-runnable diagnostic scripts that answer a `Q0x` in
[`../questions.md`](../questions.md). This is the **keeper** home; ephemeral exploration stays in
`dr_tmp/` (gitignored) and is promoted here once it earns a place in a finding.

## Contract (per `../PROTOCOL.md` § "Track B — Diagnostics")
- **Privileged reads via `mfas.analysis.gap` only.** Never `open()` `data/best_solution` directly.
- **Never write to `results/`.** Diagnostic outputs go to `experiments/outputs/*.json` (+ plots);
  conclusions go to `../diagnosis.md` under a `## Q0x —` anchor. (`results/*.json` is
  runner-only and write-protected by the guard hook.)
- **Reproducible by checkout.** No hardcoded paths to gitignored artifacts
  (`results/*_positions.npy` is excluded from git) — regenerate, or read the committed
  `results/rocket_best_positions.npy`.
- Every printed number must be reproducible from the script + a logged command; fix seeds.

## Promotion checklist (dr_tmp → here)
1. Move the script; fix any hardcoded gitignored paths (see above).
2. Add a module docstring: which `Q0x` it answers, the exact run command, expected artifact.
3. Write/point the answer in `../diagnosis.md`; annotate any `findings.md` claim it corrects.
4. Update the pointer in `../roadmap.md` and flip status in `../questions.md`.

## Index
- **`q01_drift_from_optimum.py`** — answers `Q01` (why starting from best drifts down: it doesn't,
  at the right scale). Consolidates the two `dr_tmp/drift_*.py` probes; reads the committed parity
  anchor, not a gitignored positions file. Run: `PYTHONPATH=src python
  experiments/diagnostics/q01_drift_from_optimum.py` → `experiments/outputs/q01_*.{json,png}`.
  Answer: `../diagnosis.md` § Q01.
- **`q02_seed_distance.py`** — answers `Q02` (Rocket's score is seed-stable, its ordering is not).
  Run: `PYTHONPATH=src python experiments/diagnostics/q02_seed_distance.py` →
  `experiments/outputs/q02_seed_distance.{json,png}`. Answer: `../diagnosis.md` § Q02.
- **`verify_collective_moves.py`** — **not** a `Q0x` answer: an independent brute-force check of the
  exact-gain algebra in `experiments/size_collective_moves.py` (the H36 sizing gate), written by the
  critic and promoted here because the 2026-08-09 log entry cites its numbers. Compares the closed-form
  S1 / small-SCC-DP / S2 gains against explicit full-graph oracle rescoring on random graphs.
  Run: `python experiments/diagnostics/verify_collective_moves.py` (prints max abs errors, ~1e-14).
