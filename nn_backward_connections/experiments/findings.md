# Findings — ranked conclusions

Ranked, evidence-backed conclusions. Every claim must cite the `results/*.json` run(s) and the
command that produced it. A gain within seed noise (overlapping mean ± std) is not a finding.

## #1 — H02: warm-start Rocket from a greedy-FAS ordering (small, robust, equal-compute win)

**Claim.** Initializing Rocket's continuous positions from a leakage-safe greedy Feedback-Arc-Set
ordering (Eades–Lin–Smyth / GreedyAbs: peel sinks→back, sources→front, else remove max `out_w−in_w`
→front; computed only from `g.src`/`g.tgt`/`g.weight`), mapped to evenly-spaced positions in [−1,1]
and fed to the **unchanged** `run_rocket`, beats the random-N(0,1) baseline on the exact feedforward
metric — at **equal compute** (same epoch budget / `total_grad_steps`) and on **both** datasets.

**Evidence (CONFIRMED, conservative matched-seed baseline; SE = std_base·√(2/n), 95% CI lower bound):**

| dataset | H02 mean±std (n) | baseline mean±std (n) | Δ | 95% CI lower | verdict |
|---|---|---|---|---|---|
| connectome | 82.9292 ± 0.0015 (5) | 82.8844 ± 0.0253 (5) | **+0.0448 pp** | **+0.0135** | CI>0 ✓ |
| mouse | 92.4793 ± 0.0000 (20) | 92.2729 ± 0.2000 (20) | **+0.2064 pp** | **+0.0824** | CI>0 ✓ |

Variant: `src/mfas/experiments/H02.py`. Pure Rocket score (no post-processing). The greedy order
alone scores 68.91% (connectome) / 90.13% (mouse) before any optimization.

**Reproduce** (env `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
```
python -m eval.run_variant --exp H02 --dataset connectome --seed {42,123,999,7,31415} --out results/ --role confirm
python -m eval.run_variant --exp H02 --dataset mouse --seed {20 seeds: 42,123,999,7,31415,2718,1618,1414,1732,2236,9999,8888,7777,6666,5555,4444,3333,2222,1111,1234} --out results/ --role confirm
# matched baseline: same via --exp baseline_passthrough
```
Result JSONs: `results/*-H02-{connectome,mouse}-*-confirm-{059689,a8bbc0}.json`;
baseline `results/*-baseline_passthrough-*-{f8cb3c,7b7cba}.json`. Verified independently by the
verifier (read-only) and red-teamed by the critic (frozen-integrity, leakage, reproducibility,
significance, both-dataset robustness — all PASS).

**Honest caveats.** The win is **modest**: mouse +0.21 pp is solid; connectome +0.045 pp clears the
CI bar by a thin **+0.0135 pp** — fragile to a few baseline seeds, so widen the connectome confirm
seed count before leaning on it. H02 **missed the lenient 2σ_baseline screen** and was promoted only
via the more-rigorous CONFIRM test (justified: the screen gate assumes variant variance ≈ baseline
noise, but H02's init is nearly deterministic; see the log's orchestrator escalation note).
Generality beyond connectome+mouse is asserted from the mechanism (a graph-derived warm start lands
Rocket in a better basin), not proven — only two real graphs are available.

## #2 — Rocket's plateau is set by the starting basin, not the optimization dynamics (structural)

**Claim.** Across a 9-hypothesis screen (8 distinct mechanisms), the **only** lever that improved the
exact feedforward metric was the one that changed **where optimization starts** (H02's warm-start).
Every intervention on the optimization **dynamics / trajectory** re-converged to — or fell below —
Rocket's plateau, and both **objective/loss-landscape** reshapings slightly **regressed** the
connectome. This is a reproducible structural property of Rocket on these connectomes: *how* it
descends barely matters; *where* it begins does.

**Evidence (each row = one screened variant; n=3 seeds 42/123/999; Δ = variant − baseline on the
exact metric; all numbers from logged `results/*.json`, one git commit per experiment):**

| variant | mechanism / axis | Δ connectome | Δ mouse | result |
|---|---|---|---|---|
| **H02** | **starting basin** (greedy-FAS warm-start) | **+0.0448** | **+0.2064** | **CONFIRMED win** |
| H01 | restarts (dynamics) | +0.0003 | +0.0000 | kill |
| H03 | β schedule (dynamics) | −0.0376 | +0.0114 | kill |
| H04 | in-loop refinement (dynamics) | −0.0000 | +0.0000 | kill |
| H13 | edge-subsample noise (dynamics) | −0.8356 | −0.0325 | kill |
| H05 | optimizer → AdamW (dynamics, **falsifier**) | +0.0018 | +0.0000 | kill |
| H06 | weight-aware loss reweight (objective) | −0.0380 | −0.0794 | kill |
| H11 | margin/hinge surrogate (objective) | −0.0387 | +0.1264 | kill |
| H09 | anti-tie jitter (free-edge) | −0.0010 | +0.0000 | kill (0 ties exist) |

Two corroborating sub-results: (a) **free-edge recovery is empty** — the continuous optimizer leaves
**0 exact position ties**, so the strict-`>` oracle drops nothing recoverable (H09 sized this before
running). (b) **The faithful surrogate is already near-optimal for the dynamics** — reshaping the
objective magnitude (H06) or surrogate shape (H11) *lowered* connectome, and injecting gradient noise
(H13) lowered it sharply (−0.84 pp). The H05 optimizer falsifier (the dynamics knob most able to
reach a different basin) re-converging to the plateau is the strongest single piece of evidence.

**Why it matters.** It explains *why* the paper's Crane phase (discrete MIP refinement from a good
order) is what extends quality past Rocket's plateau, and predicts that future gains lie in **better
initial orderings / basins** (stronger discrete FAS heuristics, multi-basin search) rather than in
optimizer/LR/β/loss tuning. It also says reproductions need not chase Rocket's exact training
trajectory — the plateau is basin-determined, not schedule-determined.

**Honest scope.** Negative results over two real connectomes and one screened arm per hypothesis;
some un-screened arms remain (β_max∈{2,8} for H03, FRAC=0.25 for H13, Lion for H05, etc.) and four
LOW-EV dynamics knobs (H07/H08/H10/H12) were **deferred, not falsified**, when the campaign hit its
`EARLY_EXIT` stop (7 consecutive non-improving cycles, no promising items left). The inference was
pressure-tested (the ideator kept H05 as a falsifier rather than assuming the conclusion), but it is
an inductive conclusion, not a proof. Full per-cycle evidence + commands: `experiments/log.md`
(H01–H13 cycles + the 2026-06-21 campaign-stop note).
