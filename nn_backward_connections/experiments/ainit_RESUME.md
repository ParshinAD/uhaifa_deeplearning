# A-INIT — PAUSED 2026-08-16 (resume state)

Paused mid-screen at the researcher's request (laptop on battery). Nothing is lost: every
finished run is a `results/*.json` on disk. **One run is missing** — that is the whole to-do.

## What is done

| stage | state |
|---|---|
| theory (derivation + numerical check) | done — `experiments/outputs/proto_ainit_scale.json`, slope 2.006 / 1.996 vs predicted 2 |
| prototype gate (mouse + hard synthetic) | done — PASS, falsified the "tight init is worse" prior |
| variant module | done — `src/mfas/experiments/A_INIT.py` (one changed constant: `INIT_STD = 1e-4`) |
| screen: mouse | done, 3/3 seeds |
| screen: connectome | done, 3/3 seeds |
| screen: microns | **2/3 seeds — s999 MISSING** |
| log.md entry | **not written yet** |

## Screen so far (role=implement, budget-matched `baseline_passthrough`, conservative
`SE = std_baseline * sqrt(2/n)`)

| dataset | A_INIT mean±std | baseline mean±std | Δ pp | 95% CI lo | gate | screen |
|---|---|---|---|---|---|---|
| connectome (n=3) | 82.9472 ± 0.0044 | 82.8957 ± 0.0185 | **+0.0514** | +0.0218 | 0.04 | PASS |
| microns (n=2) | 83.1237 ± 0.0013 | 83.1168 ± 0.0001 | **+0.0069** | +0.0067 | 0.002 | PASS* |
| mouse (n=3) | 92.4359 ± 0.0000 | 92.0696 ± 0.2624 | +0.3663 | −0.0536 | ≥ −0.26 | PASS (non-inf) |

\* microns is 2 seeds, not the protocol's 3 — **the screen is not complete until s999 lands.**

Context (NOT part of the screen): the confirmed H02 warm start scores 82.9292 ± 0.0011 on
connectome, so A_INIT is *above* H02 there while being *below* it on mouse (92.4359 vs 92.4793).
That cross-over is the interesting part and must be stated as graph-dependent, not as "beats H02".

## To resume — exactly one command

```bash
cd /Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
caffeinate -i $PY -m eval.run_variant --exp A_INIT --dataset microns --seed 999 \
    --out results/ --role implement --device auto        # ~9 min awake, ~550 s
```

Then re-aggregate and write the log entry.

## Wall-clock caveat — do NOT quote the logged timings

The machine slept during the overnight batch, so `wall_clock_s` in three JSONs is inflated and
must not be cited:

| run | logged wall | honest wall |
|---|---|---|
| connectome s42 / s123 | 80.7 s / 81.0 s | usable |
| **connectome s999** | **7068.7 s** | re-run needed (~81 s) if wall-clock is quoted |
| **microns s42** | **61942.3 s** | re-run needed (~540 s) if wall-clock is quoted |
| microns s123 | 540.3 s | usable — this is the reference microns timing |

Scores are unaffected (all runs completed their full epoch budget: 20,000 / 80,000). Only the
timing column is contaminated. Wrap every future long batch in `caffeinate -i`.

## Comparator trap hit and fixed (worth remembering)

The first aggregation averaged microns `baseline_passthrough` runs at **20k and 80k epochs**
into one comparator (83.0704 instead of 83.1168), inflating Δ by ~0.047 pp — an ~7× overstatement.
Always filter comparator runs by `n_epochs_done`, not just by variant id + dataset + seed.
