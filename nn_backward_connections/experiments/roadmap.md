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
| **Q01** | **Why does starting Rocket from the best solution drift the score DOWN?** (TODO 4) | **NOW** | ~80% already answered, un-consolidated, in `dr_tmp/drift_from_optimal_spacing.py` + `dr_tmp/drift_scale_sweep.py`: from the true low-loss point P* (optimal spacing, std≈141) small-lr GD **holds** 84.61%; the logged "collapse" (`diagnosis.json`) was an artefact of even-spacing init (std≈0.58, ~200× too small). Job = consolidate + **correct the over-strong claim in `findings.md` #3** ("unreachable/unholdable under *every* schedule/scale"). | `questions.md` |
| Q02 | Seed-to-seed distance / ordering variability **without** greedy start (TODO 1) | med | Largely done in `experiments/rocket_base.ipynb` seed-stability analysis (scores ~82.9% stable; ordering Spearman≈0.96; sinks stabler than sources, Jaccard@1000≈0.65 vs 0.33). **Extend, don't restart**; pick metric (Kendall-τ / Spearman / Jaccard@k, sources vs sinks) and state the purpose. | `questions.md` |

### Track A — Improvement

| # | idea (TODO ref) | priority | prior evidence — read before running | queue |
|---|---|---|---|---|
| A-SCC | SCC decomposition + SCC-structured insertion (TODO 6) | med | **NOT the quick win it looks like.** `dr_tmp/FINDINGS_underrelaxation.md` E2: inter-SCC edges = 1.50%, giant SCC = 92.82% → a *one-shot* SCC/topo split recovers ≈nothing (this is why H22 was retired). It only has legs as **recursive-within-giant-SCC** insertion. Vahidi 2025 reaches 84.61% via greedy + **bounded-span insertion + SCC** (no MIP) — target the *recursive* form. | `backlog.md` |
| A-ALT | Alternate discrete↔gradient refinement (TODO 3) | med | Genuinely new *combination* (H30/H35 do gradient→sift once, not iterated). BUT the drift probe (`findings.md` #3, and Q01) predicts a re-Rocket phase **erases** the sift gain; `best-by-oracle` protects it. Design so the gradient phase can't drift (freeze scale / alternate with a *different* discrete move). | `backlog.md` |
| A-CLU | Cluster-decompose ordering: order clusters, then within (SBM/spectral communities) | med-low | Divide-and-conquer; a soft cousin of A-SCC. **Cost warning:** H33 (magnetic-Laplacian) eigensolve = 304–507 s (5–8× over budget, no AMG) and was a −4.97 pp *warm-start*. Clustering-for-D&C is a different use, but full-graph spectral is expensive here. Overlaps Track C's null-model machinery. | `backlog.md` |
| A-INIT | Very-close / tight-position initialization (TODO 5) | low | Init/dynamics knob. Doubly discouraged: init→plateau is **flat** (≤0.06 pp, `diagnosis.md` Step 4) and finding #2 says dynamics don't move the metric. Cheap falsifier only; has diagnostic tie to Q01's scale-sweep. | `backlog.md` |
| A-SUB | Partial / stochastic sift ("SGD on part of the neurons", TODO 2) | low | **Ambiguous — disambiguate first.** As *continuous* block-coordinate GD it's a dynamics knob (H13 edge-subsample killed −0.84 pp). As a *discrete stochastic sift* (move a random subset of movers) it's novel and relates to under-relaxation (H35). Only the discrete recast is worth building. | `backlog.md` |
| A-SURR | Alternate surrogate (flat top/bottom, slower-decaying tanh, TODO 7) | low | Re-tread of a consistently-negative axis: H11 margin/hinge killed (−0.039 pp connectome), H34 perturbed-sort killed (−0.64 pp); finding #2/#3 — the continuous family is **exhausted**. Fast to falsify, do not prioritize. | `backlog.md` |
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
