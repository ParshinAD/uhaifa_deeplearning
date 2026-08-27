# Quarantine manifest

See `README.md` for the rule. Every row must name the log entry that excluded the run.

| file | variant | dataset | seed | role | reason | replaced by | log entry |
|---|---|---|---|---|---|---|---|
| `20260826T201937Z-H63-microns-s7-confirm-c3d229.json` | H63 | microns | 7 | confirm | `runtime_guard.degraded=true` — 3547.5 s against a 3450 s deadline; `stage3_sift` 1/12 sweeps and `stage4_alternation` 1/5 cycles executed. Scored 83.20094526 (a truncated computation, not this variant's). | `20260826T232612Z-H63-microns-s7-confirm-c3d229.json` — same seed, clean, 2954.6 s, 83.25965742667618, bit-identical to seeds 42/123/999/31415 | `experiments/log.md`, cycle 15 (H59/H63), 2026-08-27 |

## What the replacement showed

The re-run of seed 7 finished 592.9 s faster than the run it replaces (2954.6 s vs 3547.5 s) on
identical code and identical seed, and landed on exactly the same score as the other four seeds.
That is the falsifiable prediction in `experiments/outputs/prereg_H63_confirm_rerun.json` coming
out as predicted: the truncation was the **box**, not the variant. Had it come out the other way
— a second truncation at the same seed — the sealed rule required the cycle to report `iterate`
and leave both runs in `results/`.
