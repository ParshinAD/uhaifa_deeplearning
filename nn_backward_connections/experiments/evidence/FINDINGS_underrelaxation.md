# Exploratory research — can we push MFAS past H30 (83.78%) cheaply?

**Date:** 2026-06-23 · **Scope:** exploratory, all code/results under `dr_tmp/` only
(no file outside `dr_tmp/` was modified; the FROZEN oracle/loaders were reused, not
copied). **Not** a harness-promoted finding — see "Status & honest limits" below.

Starting point: H30 = full-range **Jacobi** exact-gain sift = **83.78%** connectome
(published mean), gap to SOTA (Vahidi 2025, **84.61%**) = **0.83 pp**.

I tested the three levers the Phase-6 backlog/REPORT left genuinely untested.

---

## TL;DR

**The production sift's Jacobi (all-nodes-move-at-once) dynamics gets stuck in a
period-2 limit cycle on large dense connectomes and never converges. Under-relaxing
the move (move each node only α=0.7 of the way to its exact-optimal gap) breaks the
cycle, the iterate converges, and the score rises — at the SAME cost (0 gradient
steps, identical per-sweep work).**

| dataset | production Jacobi (α=1) | under-relaxed (α=0.7) | same-basin Δ |
|---|---|---|---|
| connectome s42 | 83.807% (cycling) | **83.905%** (converged) | **+0.097 pp** |
| connectome s123 | 83.824% (cycling) | **83.918%** (converged) | **+0.094 pp** |
| microns s42 | 83.207% (cycling) | **83.215%** (converged) | **+0.008 pp** |
| mouse (all seeds) | 92.902% (converges in 5) | 92.874% | −0.028 pp |

A non-regressing **two-phase schedule** (α=1 for 6 sweeps → α=0.7, single
best-by-oracle tracker) gets the best of both: **connectome 83.908%**, **mouse
92.902%** (no regression). This recovers ~**11%** of the remaining gap to SOTA on the
fly connectome, with no MIP and no extra gradient steps.

---

## E2 — SCC structure: a clean DEAD END (settles a long-standing backlog item)

`scipy.sparse.csgraph.connected_components` (strong) on the connectome:
- **9,626** SCCs, but a single **giant SCC of 126,840 nodes (92.82%)**; 9,503 singletons.
- **inter-SCC weight = 630,145 = only 1.50%** of total; the other **98.50% is INSIDE the
  giant SCC.**
- In H30's order, inter-SCC edges are **already 99.99% feedforward**.
- Forcing all inter-SCC edges feedforward (SCC-topological reorder, intra order kept)
  gains **+0.00013 pp** → nothing.

**Conclusion:** the entire residual gap to SOTA lives *inside* the 127k-node giant SCC.
One-shot SCC condensation cannot help here (confirms REPORT.md's prediction). Vahidi's
SCC must therefore act *recursively / within* local search, not as a top-level split.
This retires backlog hypothesis **H22 (block/SCC-macro warm-start)** for the fly graph.
Script: `dr_tmp/probe_scc.py` → `dr_tmp/probe_scc.json`.

## E1 — Sweep-count: production cap of 12 is too low, but Jacobi limit-cycles

From the H02 order, extending the sift past the production cap of 12:

| cap | 12 | 16 | 20 | 24 | 30 | 40 |
|---|---|---|---|---|---|---|
| best-by-oracle | 83.762 | 83.790 | 83.798 | 83.802 | **83.807** | 83.807 |

Raising the cap 12→30 is worth **+0.045 pp** (free, 0 gradient steps). But beyond
sweep ~21 the **Jacobi iterate enters a period-2 limit cycle**: `n_movers` freezes at
~5,255 and the candidate alternates 83.807 / 83.796 forever. So more sweeps barely
help — the dynamics, not the budget, are the wall. Script: `dr_tmp/exp_sweeps.py`.

## E3 — Under-relaxation breaks the cycle (the actual win)

Diagnosis from E1: simultaneous (Jacobi) moves make conflicting nodes leapfrog each
other every sweep → oscillation. Classic fix = **under-relaxation**: move each node a
fraction α of the way to its exact-optimal gap.

- α=0.7 collapses the mover count 5,255 → ~215–270 (it actually **converges**) and
  reaches **83.905% (s42) / 83.918% (s123)** — a robust **+0.094…+0.097 pp** over
  Jacobi on two independent seeds.
- α=0.5 and an annealed 1.0→0.3 schedule also beat Jacobi but converge slower (still
  climbing at sweep 40); α≈0.7 is the sweet spot.
- **Same pathology + same fix on microns** (the second large graph): Jacobi cycles
  (~350 movers), α=0.7 converges (~60 movers), +0.008 pp. Magnitude is graph-dependent
  (~11×, mirroring H30's own connectome≫microns ratio).
- **Mouse** (148 nodes) has **no** limit cycle — Jacobi converges in 5 sweeps — so α<1
  finds a slightly worse local optimum there (−0.028 pp). This is why the production
  recipe should be the two-phase schedule, not a flat α=0.7.

Scripts: `dr_tmp/exp_damped.py`, `dr_tmp/exp_underrelax.py` (+ per-dataset `.jsonl`),
`dr_tmp/exp_twophase.py`.

---

## What this means for "can we reach the optimum?"

- The cheap, gradient-free part of the gap is now ~**83.91%** on the connectome
  (H30 83.78% + under-relaxation). The remaining **~0.70 pp** to Vahidi's 84.61% is
  **intra-giant-SCC, long-range** (E2 + the prior H22 window-sizing): it is not
  reachable by SCC condensation, by more Jacobi sweeps, or by under-relaxation alone.
- Closing it needs what Vahidi actually uses: **gain-aware bounded-span insertion
  combined with *recursive* SCC** inside the local search (re-decompose the residual
  feedback subgraph, recurse). That is a real build, not a few-minutes tweak, and is
  the single highest-value next experiment.

## Status & honest limits (per CLAUDE.md governance)

- **This is exploratory, not a CONFIRMED finding.** A promotion would require
  `eval/run_variant.py` on **both** primary datasets, ≥3 seeds, mean±std + two-stage CI,
  one git commit — all of which write outside `dr_tmp/` and are intentionally NOT done
  here.
- The Δ values above are the **clean within-basin** comparison (same H02 order fed to
  α=1 vs α=0.7), which removes MPS basin noise — the right way to isolate the dynamics.
- Coverage: connectome 2 seeds, microns 1 seed, mouse (deterministic, 1 effective seed).
- The recommended promotable variant is a one-line change to the sift loop: replace the
  full Jacobi rebuild with the two-phase under-relaxed rebuild (α=1 ×K, then α≈0.7),
  keeping the existing best-by-oracle tracker and `max_sweeps≈30`.

### Reproduce
```
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
$PY dr_tmp/probe_scc.py
$PY dr_tmp/exp_sweeps.py
$PY dr_tmp/exp_damped.py
$PY dr_tmp/exp_underrelax.py --dataset connectome --seed 123 --alphas 1.0,0.7 --sweeps 40
$PY dr_tmp/exp_underrelax.py --dataset microns   --seed 42  --alphas 1.0,0.7,0.5 --sweeps 30
$PY dr_tmp/exp_underrelax.py --dataset mouse      --seed 42  --alphas 1.0,0.7,0.5 --sweeps 200
$PY dr_tmp/exp_twophase.py  --dataset connectome --seed 42  --kfull 6 --sweeps 40
$PY dr_tmp/exp_twophase.py  --dataset mouse       --seed 42  --kfull 6 --sweeps 60
```
```
