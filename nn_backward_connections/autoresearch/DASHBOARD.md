# Campaign dashboard

*Generated 2026-08-27 00:59 by `autoresearch/dashboard.py` — do not hand-edit.*

**Phase:** `quality` · **cycle:** 14 · **mode:** `divergent` · **consecutive kills:** 5

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
| mouse | **H52** | 93.1028 ± 0.0 | 20 | ~1s | experiments/log.md 2026-08-17 (H52) |

## Queue — 17 proposed / 43 total

| # | id | title | status |
|---|---|---|---|
| 0 | `P01` | HARDWARE RE-BASELINE РІР‚вЂќ re-measure the champions on this machine before a | done |
| 0 | `P05` | BLOCKING: stage 4 has no wall-clock guard and microns now runs 3398-3418 s aga | done |
| 0 | `P07` | BLOCKING: the champion's microns configuration does not fit 3600 s on a loaded | in_progress |
| 0 | `H43` | Re-allocate the microns budget: cut Rocket epochs 80,000 -> 20,000 and spend t | killed |
| 0 | `H44` | Drop the gradient phase on mouse: _EPOCHS['mouse'] = 0 (greedy -> under-relaxe | confirmed |
| 0 | `H50` | The reference ROUTE run standalone: ratio-greedy init + iterated exact-gain pa | killed |
| 0 | `H48` | Ratio greedy init ((out_w+1)/(in_w+1)) instead of greedy-FAS - a +6.3 pp bette | killed |
| 0 | `P09` | The campaign's two significance criteria disagree for the first time - decide  | awaiting-operator |
| 0 | `P13` | Make arc reclamation cheap enough to run on microns (it is worth +0.018759 pp  | proposed |
| 0 | `H64` | Compose the ASYMMETRIC surrogate with the CHAMPION stack on CUDA — H38 was nev | proposed |
| 1 | `P02` | The screen seeds are inert for deterministic variants РІР‚вЂќ is 3x the comput | done |
| 1 | `H36` | SCC-decomposed recursive bounded-span insertion (the Vahidi route) | confirmed |

## Recent cycles

| cycle | item | verdict | note |
|---|---|---|---|
| 14 | `H59` | **iterate** | Minimal-FAS arc reclamation. Prototype PASSED all three with exact certificates  |
| 13 | `H60` | **kill** | Net-digraph condensation inside the SCC refiner, killed at the SCREEN. The proto |
| 12 | `H47` | **kill** | Exact subset-DP at the SCC recursion's leaves, killed at the prototype rung for  |
| 11 | `H57` | **kill** | Prefix-shared tail multi-start, killed at the prototype rung for ~2.1 h by a pre |
| 10 | `H56` | **kill** | Relabelling multi-start, killed at the prototype rung for 2.55 s of GPU. Runtime |
| 9 | `P09` | **iterate** | Half the change shipped, half held for the operator. The relabelling-robustness  |
| 8 | `H52` | **keep-partial** | New mouse champion 93.082880 -> 93.102826 (+0.01995 pp, 20/20, std 0) from a new |
| 7 | `H44` | **keep** | NEW MOUSE CHAMPION, +0.16588 pp (92.917014 -> 93.082880), n=20 seeds, std 0, fro |
| 6 | `P05` | **iterate** | The run-level wall-clock guard is built, armed by default and verified on 2 of 3 |
| 5 | `H42` | **keep** | NEW CHAMPION x3, second score move of the campaign, from two constants. Resumed  |
| 4 | `None` | **lost** | RECONSTRUCTED 2026-08-25 to make the history contiguous, NOT a real cycle record |
| 3 | `H36` | **keep** | NEW CHAMPION on all three datasets - the first score move of the autonomous camp |

## Evidence

- `results/*.json`: **506** run records across **33** variants
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

