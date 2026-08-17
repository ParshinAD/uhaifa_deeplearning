# Roadmap — the cross-track research plan

The single map of *what we plan to do*, across all three research tracks. This file is a
**priority index + pointer**: each idea links to the queue file that owns its live status and
full detail. **Status is NOT restated here** (it lives in the queue file) so the two never drift.

Governance for all tracks: `experiments/PROTOCOL.md` (§ "Multi-track research map").

## The three tracks

| track | question it answers | queue (owns status) | conclusions |
|---|---|---|---|
| **A — Improvement (H)** | beat the baseline on the exact feedforward metric | [`backlog.md`](backlog.md) | [`findings.md`](findings.md) |
| **B — Diagnostics (Q)** | *explain* Rocket's behaviour (mechanism / measurement) | [`questions.md`](questions.md) | [`diagnosis.md`](diagnosis.md) |
| **C — Random graphs (G)** | do real brains have more/less unavoidable feedback than matched random graphs? | [`randomgraph.md`](randomgraph.md) | `randomgraph.md` (findings §) |

> **Historical note (do not renumber).** Two diagnostic items shipped in the H-series before
> Track B existed — **H21** (hard-synthetic fixture) and **H22** (window/SCC sizing). They are
> cited by commit-hash across `findings.md` #3 / `diagnosis.md` / `log.md`, so they keep their H
> IDs. New diagnostics start at **Q01**.

---

## Priority (highest expected value / cost first)

Priority tiers are the planner's call; "prior evidence" is the load-bearing column — **read it
before promoting an idea to its queue**, so we don't re-run an already-closed result.

### Track B — Diagnostics (start here)

| # | idea (TODO ref) | priority | prior evidence — read before running | queue |
|---|---|---|---|---|
| **Q01** | **Why does starting Rocket from the best solution lose score?** (TODO 4) | **answered** | Answered 2026-08-17: at a common scale the surrogate ranks Rocket's 82.92% order ABOVE the 84.61% order (net −174.88 at β=1.05: +296.02 true advantage minus 470.90 smoothing loss); the better order is preferred only above β·std ≈ 470 and Rocket runs at 148. Shape-independent (4 surrogate shapes agree). See `diagnosis.md` § Q01. | `questions.md` |
| Q02 | Seed-to-seed distance / ordering variability **without** greedy start (TODO 1) | med | Largely done in `experiments/rocket_base.ipynb` seed-stability analysis (scores ~82.9% stable; ordering Spearman≈0.96; sinks stabler than sources, Jaccard@1000≈0.65 vs 0.33). **Extend, don't restart**; pick metric (Kendall-τ / Spearman / Jaccard@k, sources vs sinks) and state the purpose. | `questions.md` |
| **Q03** | **How far is our best order from the reference near-optimal one, and is the ~84.6% level degenerate?** | **NEXT** (after the S1/S2 sizing) | All published gap-structure numbers (flip fraction 8.47%, Kendall-τ 0.610, flip rank-distance p50=22,580) were measured against **H02's 82.93% order**, before H30/H35 closed 0.98 pp — they are stale. Q02 showed the ~82.9% level is degenerate; whether the ~84.6% level is too is unmeasured and bounds every multi-basin idea. Also reconciles `data/best_solution` (84.6147%) vs Vahidi's published 35,462,925 (84.6125%) — **not the same solution**. | `questions.md` |

### Track A — Improvement

| # | idea (TODO ref) | priority | prior evidence — read before running | queue |
|---|---|---|---|---|
| **H36** | **Collective (multi-node) discrete refinement = A-PAIR (Vahidi Alg 2) + A-SCC (Vahidi Alg 3), alternated** | **NOW** | **Sizing gate PASSED (2026-08-09), promoted from A-SCC/A-PAIR.** Vahidi's SCC step is *not* a global split (that one is dead: +0.00013 pp) — it decomposes the induced subgraph of a **rank window**, which shatters even inside the 92.82% giant SCC. With the single-node class first exhausted as a control, the collective phase still gains **connectome +0.0599 pp** (s42, 94.5% collective) and **microns +0.00601 ± 0.00025 pp** (n=3, CI_lo +0.00561); a **lower bound** — top-K truncation under-states it ~2.5×. Costs ~2.6× wall (breaches the ~2× ceiling), 0 gradient steps. Read `log.md` 2026-08-09 (incl. the critic verdict) before building. | `backlog.md` |
| A-ALT | Alternate discrete↔gradient refinement (TODO 3) | med | Genuinely new *combination* (H30/H35 do gradient→sift once, not iterated). BUT the drift probe (`findings.md` #3, and Q01) predicts a re-Rocket phase **erases** the sift gain; `best-by-oracle` protects it. Design so the gradient phase can't drift (freeze scale / alternate with a *different* discrete move). | `backlog.md` |
| A-CLU | Cluster-decompose ordering: order clusters, then within (SBM/spectral communities) | med-low | Divide-and-conquer; a soft cousin of A-SCC. **Cost warning:** H33 (magnetic-Laplacian) eigensolve = 304–507 s (5–8× over budget, no AMG) and was a −4.97 pp *warm-start*. Clustering-for-D&C is a different use, but full-graph spectral is expensive here. Overlaps Track C's null-model machinery. | `backlog.md` |
| A-INIT | Very-close / tight-position initialization (TODO 5) | low | Init/dynamics knob. Doubly discouraged: init→plateau is **flat** (≤0.06 pp, `diagnosis.md` Step 4) and finding #2 says dynamics don't move the metric. Cheap falsifier only; has diagnostic tie to Q01's scale-sweep. | `backlog.md` |
| A-SUB | Partial / stochastic sift ("SGD on part of the neurons", TODO 2) | low | **Ambiguous — disambiguate first.** As *continuous* block-coordinate GD it's a dynamics knob (H13 edge-subsample killed −0.84 pp). As a *discrete stochastic sift* (move a random subset of movers) it's novel and relates to under-relaxation (H35). Only the discrete recast is worth building. | `backlog.md` |
| A-SURR | Alternate surrogate (flat top/bottom, slower-decaying tanh, TODO 7) | **done — KILLED as H37/H37B (2026-08-17)** | Ran properly rather than by analogy, and the analogy was right for the wrong reason. `tanh` **is** the sigmoid (2.2e-16) and "slower-decaying tanh" is the sigmoid at a different β (the killed H03/A-SCALE axis); a hard flat top/bottom is the killed H11 form. The one untested axis — the **tail exponent** — passed the theory gate (Q04) and then failed everywhere: prototype −0.22/−2.83 pp (mouse/hard synthetic), connectome **−0.45 pp (H37) / −0.93 pp (H37B)** over 3 seeds. Produced `diagnosis.md` § Q04: static surrogate alignment **anti-correlates** with achieved score. | `backlog.md` |
| A-SCALE | Scale/temperature annealing — push positions above the surrogate crossover so it becomes discriminative | **~dead** | **≡ H03 (already KILLED).** Q01 β-analysis: F sees only the product **β·std**, so "scale annealing" is identical to "sharper terminal β" (H03, connectome −0.038 pp), and Adam normalises the extra β gradient-prefactor. Both hit vanishing-gradient at large β·std (why the schedule caps β≤1.05; H19 soft-rank **stall** is the same failure). Do not build unless a genuinely new mechanism appears. | `backlog.md` |

### Track C — Random graphs vs real brains (thesis goal #2)

| # | idea (TODO ref) | priority | prior evidence / notes | queue |
|---|---|---|---|---|
| G01 | Unavoidable feedback: connectome vs **structure-matched** null models — ER, configuration model, **stochastic block model**, degree-preserving rewiring | med | Compare against *structured* nulls (SBM), not naïve ER. Metric = `total − best_feedforward`, estimated by the H35 pipeline **with a greedy-FAS lower bound as a bias control** (see estimator contract in `randomgraph.md`). Core question: do real brains carry more or less unavoidable feedback than matched random graphs? | `randomgraph.md` |
| G02 | Hierarchical / divide-and-conquer MFAS via SBM / spectral communities | low | Cross-links A-CLU (same clustering machinery, different goal: here it's an *analysis* of where feedback concentrates, there it's an *optimizer*). | `randomgraph.md` |

---

## How to use this file
- **Adding an idea:** append a row here (track, priority, prior-evidence, queue pointer) **and** a
  full entry in the queue file. Never put a mutable `status` here.
- **Picking up an idea:** open its queue file; follow that track's contract in `PROTOCOL.md`.
- **Promotion from scratch:** when a `dr_tmp/` script/finding becomes a keeper, promote it to its
  track's home (`experiments/diagnostics/`, `src/mfas/randomgraph/`, or the relevant `.md`) and
  update the pointer here.
