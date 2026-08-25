# PRE-REGISTRATION S2 — the champion's residual 1-opt

**Sealed 2026-08-25, before the measurement is taken.** Committed before the run so the branch
cannot be chosen after the answer is known. This is the discipline that the invariant-miner
experiment failed this morning, turned on ourselves.

Author of the prediction: Claude (assistant), this session. Named, because an unattributed
prediction is not a prediction.

## The question

The champion pipeline's stage 3 is a full-range exact-gain single-node insertion sweep ("sift"),
under-relaxed. **Has it actually converged the move class it owns?**

`R` := the residual 1-opt of the champion order, reported by
`experiments/diagnostics/residual_1opt.py` as `naive_sum_pp`, on **connectome**.

Calibration, already measured and pinned in `tests/test_residual_1opt.py`:

| order | pct | movers | naive_sum_pp |
|---|---|---|---|
| parity anchor | 82.916074 | 17,949 | 0.5461 |
| reference | 84.614678 | 131 | 0.0010 |

## Why this decides what gets built

Every generator design proposed today assumes the bottleneck is the SUPPLY OF IDEAS. If the
incumbent has not converged the move class it already contains, that assumption is false and the
cheapest available gain is finishing the existing ladder, not inventing a new operator.

## The branch table — SEALED, chosen before the number is seen

| outcome | reading | action |
|---|---|---|
| **R ≥ 0.05 pp** | The ladder has NOT converged the move it owns. There is more than a screen-gate's worth of unclaimed single-node gain sitting in the champion. | **HALT all generator work.** Open a ladder-convergence cycle instead: why does stage 3 stop early — sweep cap, under-relaxation α, time budget, or the alternation schedule? |
| **0.01 ≤ R < 0.05 pp** | Partial convergence; a real but sub-gate residue. | Pay the ladder debt for one cycle, re-measure, then re-enter this table. Generator work waits. |
| **R < 0.01 pp** | The champion is 1-opt dry, like the reference. More single-node insertion is not the route. | **Supply is the constraint.** The generator build (S5, S6) is earned and proceeds. |

## The prediction, with a name on it

I predict **R < 0.01 pp**, i.e. the champion is essentially 1-opt dry.

Reasoning, so that a wrong prediction is informative rather than embarrassing: stage 3 runs
`sift_underrelaxed(k_full=6, α=0.7, max_sweeps=40)` and stage 4 alternates 32 short sift cycles
after it, so the order is driven to a sift fixed point repeatedly and last; and the reference order,
which is 1.7 pp better, sits at 0.0010 pp, showing that this regime is reachable.

**Alternative hypothesis with a different predicted value:** if the alternation in stage 4 leaves
the order at a fixed point of the *short* sift (2 sweeps) rather than of the full sift (40 sweeps),
R lands in the 0.01–0.05 band. That is the outcome I would find most surprising in a useful way,
because it would mean the champion's own schedule is the thing leaving weight on the table.

## Uninformative outcomes — declared in advance, to be discarded rather than rationalised

1. **The champion does not reproduce.** If the reproduced connectome pct differs from the recorded
   `84.15409511` by more than `2.4e-06` (the oracle's resolution at this total weight), then R is
   measured on a DIFFERENT order and says nothing about the champion. Report the reproduction
   failure; do not report R.
2. **Guard truncation.** If the run hits a time budget or sweep cap rather than converging, R
   measures the budget, not the ladder. Record the stop reason with the number or discard it.
3. **Cross-device comparison.** The recorded champion was produced on CUDA; this is MPS. R is a
   property of the order that is actually in hand here. It may NOT be compared against a CUDA-
   produced champion's R without both being measured on the same device.
4. A change in the pinned anchor numbers (17,949 movers / 0.5461 pp) means the environment moved;
   stop and diagnose rather than reading R.

## Commands

```
$PY -m eval.run_variant --algo H42 --dataset connectome --seed 42        # reproduce + save order
$PY experiments/diagnostics/residual_1opt.py --order-npy <saved order>   # measure R
```

The result, whatever it is, goes into `experiments/log.md` **before anything else is done**.
