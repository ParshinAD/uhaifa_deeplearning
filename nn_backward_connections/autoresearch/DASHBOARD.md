# Campaign dashboard

*Generated 2026-08-27 20:32 by `autoresearch/dashboard.py` — do not hand-edit.*

**Phase:** `quality` · **cycle:** 17 · **mode:** `incremental` · **consecutive kills:** 0

## Progress — connectome (Phase 1 target)

```
  bootstrap  83.9101%
  now        84.2582%   ████████████████················   49.4% of the way
  target     84.6147%   (reference solution / Vahidi 2025)
  remaining  +0.3565 pp
```

## Champions

| dataset | champion | score | n | wall/run | evidence |
|---|---|---|---|---|---|
| connectome | **H64** | 84.2582 ± 0.0 | 5 | ~1185s | findings.md #11 (H64, promoted 2026-08-27 cycle 17) |
| microns | **H42** | 83.2409 ± 0.0 | 5 | ~3418s | findings.md #7 (H42, 2026-08-10) |
| mouse | **H63** | 93.1754 ± 0.0 | 20 | ~1s | findings.md #10 (H63, 2026-08-27) |

## Queue — 20 proposed / 52 total

| # | id | title | status |
|---|---|---|---|
| 0 | `P01` | HARDWARE RE-BASELINE Р Р†Р вЂљРІР‚Сњ re-measure the champions on this machine  | done |
| 0 | `P05` | BLOCKING: stage 4 has no wall-clock guard and microns now runs 3398-3418 s aga | done |
| 0 | `P07` | BLOCKING: the champion's microns configuration does not fit 3600 s on a loaded | superseded-by-P19 |
| 0 | `H43` | Re-allocate the microns budget: cut Rocket epochs 80,000 -> 20,000 and spend t | killed |
| 0 | `H44` | Drop the gradient phase on mouse: _EPOCHS['mouse'] = 0 (greedy -> under-relaxe | confirmed |
| 0 | `H50` | The reference ROUTE run standalone: ratio-greedy init + iterated exact-gain pa | killed |
| 0 | `H48` | Ratio greedy init ((out_w+1)/(in_w+1)) instead of greedy-FAS - a +6.3 pp bette | killed |
| 0 | `P09` | The campaign's two significance criteria disagree for the first time - decide  | awaiting-operator |
| 0 | `P13` | Make arc reclamation cheap enough to run on microns (it is worth +0.018759 pp  | proposed |
| 0 | `H64` | Compose the ASYMMETRIC surrogate with the CHAMPION stack on CUDA вЂ” H38 was n | iterate |
| 0 | `P15` | OPERATOR DECISION: is a microns-only championship available while the connecto | done |
| 0 | `P18` | Buy the connectome relabelling study for H64 - the ONLY gate left on the large | done |

## Recent cycles

| cycle | item | verdict | note |
|---|---|---|---|
| 17 | `P18` | **keep-partial** | NEW CONNECTOME CHAMPION H64 84.25817950936937 (+0.104084 pp over H42), the large |
| 16 | `H64` | **iterate** | ASYMMETRIC surrogate composed with the champion stack. connectome +0.104084 pp ( |
| 15 | `H59` | **keep-partial** | Variant H63. NEW MOUSE CHAMPION 93.10282596057695 -> 93.17538325903584 (+0.07255 |
| 14 | `H59` | **iterate** | Minimal-FAS arc reclamation. Prototype PASSED all three with exact certificates  |
| 13 | `H60` | **kill** | H60 killed at the SCREEN (net-digraph condensation; connectome +0.007311 vs a 0. |
| 12 | `H47` | **kill** | H47 killed at the prototype rung (exact subset-DP at the SCC leaves). New meta-r |
| 11 | `H57` | **kill** | H57 killed at the prototype rung (prefix-shared tail multi-start; anchored gain  |
| 10 | `H56` | **kill** | H56 killed at the prototype rung (relabelling multi-start; M9 effect size). New  |
| 9 | `P09` | **iterate** | P09 iterate: the relabelling-robustness gate for degenerate primary pools is IN  |
| 8 | `H52` | **keep-partial** | H52 keep-partial: mouse champion 93.1028; connectome refused. See log.md. |
| 7 | `H44` | **keep** | H50 killed. See log.md. |
| 6 | `P05` | **iterate** | H46 killed. See log.md. |

## Evidence

- `results/*.json`: **547** run records across **34** variants
- killed mechanisms on record: **21** (+7 deferred, 12 meta-rules)
- literature notes: **8** (latest: `vahidi-2025.md`)

## Control

```bash
cd D:\1\bot\UHaifa\deep_learning\mfas_autoresearch\nn_backward_connections
nohup bash autoresearch/driver.sh > /dev/null 2>&1 &   # start
tail -f autoresearch/logs/driver.log                   # watch
touch autoresearch/STOP                                # stop after the current cycle
bash autoresearch/driver.sh --abort                    # stop now
```

