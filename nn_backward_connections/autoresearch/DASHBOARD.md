# Campaign dashboard

*Generated 2026-08-09 20:44 by `autoresearch/dashboard.py` — do not hand-edit.*

**Phase:** `quality` · **cycle:** 1 · **mode:** `incremental` · **consecutive kills:** 0

## Progress — connectome (Phase 1 target)

```
  bootstrap  83.9101%
  now        83.9135%   ································    0.5% of the way
  target     84.6147%   (reference solution / Vahidi 2025)
  remaining  +0.7012 pp
```

## Champions

| dataset | champion | score | n | wall/run | evidence |
|---|---|---|---|---|---|
| connectome | **H35** | 83.9135 ± 0.0 | 3 | ~591s | findings.md #5 (re-baselined on CUDA, P01 2026-08-09) |
| microns | **H30** | 83.2063 ± 0.0 | 3 | ~3240s | findings.md #4 (re-baselined on CUDA, P01 2026-08-09) |
| mouse | **H30** | 92.9018 ± 0.0 | 3 | ~6s | findings.md #4 (re-baselined on CUDA, P01 2026-08-09) |

## Queue — 9 proposed / 10 total

| # | id | title | status |
|---|---|---|---|
| 0 | `P01` | HARDWARE RE-BASELINE â€” re-measure the champions on this machine before any c | done |
| 1 | `P02` | The screen seeds are inert for deterministic variants â€” is 3x the compute bu | proposed |
| 1 | `H36` | SCC-decomposed recursive bounded-span insertion (the Vahidi route) | proposed |
| 2 | `H37` | Cycle-triggered under-relaxation (fix the microns regression, make H35 a gener | proposed |
| 3 | `H38` | Gauss-Seidel / block-sequential exact-gain sift | proposed |
| 4 | `H39` | Alternating discrete <-> continuous refinement with a frozen scale (A-ALT) | proposed |
| 5 | `H40` | Structure-aware destroy operator for ruin-and-recreate (revives H31 under its  | proposed |
| 6 | `H41` | Segment / block moves instead of single-node re-insertion | proposed |
| 7 | `S01` | PHASE-2 SIZING (speed): how much Rocket does the pipeline actually need? | proposed |
| 8 | `L01` | Literature scan: post-2024 FAS / linear-arrangement / connectome-ordering meth | proposed |

## Recent cycles

| cycle | item | verdict | note |
|---|---|---|---|
| 1 | `P01` | **None** | Port MacBook/MPS -> Windows/RTX 4060 (CUDA). 12 runs, 3 seeds x (H35 connectome, |

## Evidence

- `results/*.json`: **277** run records across **17** variants
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

