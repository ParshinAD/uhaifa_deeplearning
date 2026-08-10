# Campaign dashboard

*Generated 2026-08-10 16:18 by `autoresearch/dashboard.py` — do not hand-edit.*

**Phase:** `quality` · **cycle:** 5 · **mode:** `incremental` · **consecutive kills:** 0

## Progress — connectome (Phase 1 target)

```
  bootstrap  83.9101%
  now        84.1541%   ███████████·····················   34.6% of the way
  target     84.6147%   (reference solution / Vahidi 2025)
  remaining  +0.4606 pp
```

## Champions

| dataset | champion | score | n | wall/run | evidence |
|---|---|---|---|---|---|
| connectome | **H42** | 84.1541 ± 0.0 | 5 | ~1238s | findings.md #7 (H42, 2026-08-10) |
| microns | **H42** | 83.2409 ± 0.0 | 5 | ~3418s | findings.md #7 (H42, 2026-08-10) |
| mouse | **H42** | 92.917 ± 0.0 | 20 | ~6s | findings.md #7 (H42, 2026-08-10) |

## Queue — 11 proposed / 15 total

| # | id | title | status |
|---|---|---|---|
| 0 | `P01` | HARDWARE RE-BASELINE — re-measure the champions on this machine before any cyc | done |
| 0 | `P05` | BLOCKING: stage 4 has no wall-clock guard and microns now runs 3398-3418 s aga | proposed |
| 1 | `P02` | The screen seeds are inert for deterministic variants — is 3x the compute buyi | done |
| 1 | `P03` | Cycle handoff: the previous cycle was still writing when the driver launched t | proposed |
| 1 | `H36` | SCC-decomposed recursive bounded-span insertion (the Vahidi route) | confirmed |
| 1 | `H42` | Size up the stage-4 alternation budget - the curve was still rising when we cu | confirmed |
| 2 | `H37` | Cycle-triggered under-relaxation (fix the microns regression, make H35 a gener | proposed |
| 2 | `P06` | Run records discard per-stage timings, and sweep.sh --role confirm ignores con | proposed |
| 3 | `P04` | The mouse non-inferiority test is mis-specified - it can only be passed by a l | proposed |
| 3 | `H38` | Gauss-Seidel / block-sequential exact-gain sift | proposed |
| 4 | `H39` | Alternating discrete <-> continuous refinement with a frozen scale (A-ALT) | proposed |
| 5 | `H40` | Structure-aware destroy operator for ruin-and-recreate (revives H31 under its  | proposed |

## Recent cycles

| cycle | item | verdict | note |
|---|---|---|---|
| 5 | `H42` | **keep** | NEW CHAMPION x3, second score move of the campaign, from two constants. Resumed  |
| 3 | `H36` | **keep** | NEW CHAMPION on all three datasets - the first score move of the autonomous camp |
| 2 | `P02` | **keep** | Protocol amendment, no algorithm change. Established BOTH legs: (A) a static cal |
| 1 | `P01` | **done** | Port MacBook/MPS -> Windows/RTX 4060 (CUDA). 12 runs, 3 seeds x (H35 connectome, |

## Evidence

- `results/*.json`: **351** run records across **19** variants
- killed mechanisms on record: **15** (+7 deferred, 7 meta-rules)
- literature notes: **0**

## Control

```bash
cd D:\1\bot\UHaifa\deep_learning\mfas_autoresearch\nn_backward_connections
nohup bash autoresearch/driver.sh > /dev/null 2>&1 &   # start
tail -f autoresearch/logs/driver.log                   # watch
touch autoresearch/STOP                                # stop after the current cycle
bash autoresearch/driver.sh --abort                    # stop now
```

