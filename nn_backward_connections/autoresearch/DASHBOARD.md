# Campaign dashboard

*Generated 2026-08-09 15:36 by `autoresearch/dashboard.py` — do not hand-edit.*

**Phase:** `quality` · **cycle:** 0 · **mode:** `incremental` · **consecutive kills:** 0

## Progress — connectome (Phase 1 target)

```
  bootstrap  83.9101%
  now        83.9101%   ································    0.0% of the way
  target     84.6147%   (reference solution / Vahidi 2025)
  remaining  +0.7046 pp
```

## Champions

| dataset | champion | score | n | wall/run | evidence |
|---|---|---|---|---|---|
| connectome | **H35** | 83.9101 ± 0.006 | 3 | ~220s | findings.md #5 |
| microns | **H30** | 83.2069 ± 0.0012 | 5 | ~700s | findings.md #4 |
| mouse | **H30** | 92.9018 ± 0.0 | 20 | ~3s | findings.md #4 |

## Queue — 9 proposed / 9 total

| # | id | title | status |
|---|---|---|---|
| 0 | `P01` | HARDWARE RE-BASELINE — re-measure the champions on this machine before any cyc | proposed |
| 1 | `H36` | SCC-decomposed recursive bounded-span insertion (the Vahidi route) | proposed |
| 2 | `H37` | Cycle-triggered under-relaxation (fix the microns regression, make H35 a gener | proposed |
| 3 | `H38` | Gauss-Seidel / block-sequential exact-gain sift | proposed |
| 4 | `H39` | Alternating discrete <-> continuous refinement with a frozen scale (A-ALT) | proposed |
| 5 | `H40` | Structure-aware destroy operator for ruin-and-recreate (revives H31 under its  | proposed |
| 6 | `H41` | Segment / block moves instead of single-node re-insertion | proposed |
| 7 | `S01` | PHASE-2 SIZING (speed): how much Rocket does the pipeline actually need? | proposed |
| 8 | `L01` | Literature scan: post-2024 FAS / linear-arrangement / connectome-ordering meth | proposed |

## Evidence

- `results/*.json`: **265** run records across **17** variants
- killed mechanisms on record: **15** (+7 deferred, 7 meta-rules)
- literature notes: **0**

## Control

```bash
cd /Users/abed359/IdeaProjects/university/mfas_autoresearch/nn_backward_connections
nohup bash autoresearch/driver.sh > /dev/null 2>&1 &   # start
tail -f autoresearch/logs/driver.log                   # watch
touch autoresearch/STOP                                # stop after the current cycle
bash autoresearch/driver.sh --abort                    # stop now
```

