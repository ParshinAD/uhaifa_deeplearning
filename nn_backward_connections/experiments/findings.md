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
| connectome (n=5) | 82.9292 ± 0.0015 (5) | 82.8844 ± 0.0253 (5) | +0.0448 pp | +0.0135 | CI>0 ✓ |
| **connectome (n=15, hardened)** | **82.9298 ± 0.0011 (15)** | **82.8790 ± 0.0231 (15)** | **+0.0508 pp** | **+0.0391** | **CI>0 ✓✓** |
| mouse | 92.4793 ± 0.0000 (20) | 92.2729 ± 0.2000 (20) | **+0.2064 pp** | **+0.0824** | CI>0 ✓ |
| **microns (n=5, Phase-5)** | **83.1286 ± 0.0003 (5)** | **83.1172 ± 0.0006 (3)** | **+0.0114 pp** | **+0.0106** | **CI>0 ✓✓** |

> **Phase-4 hardening (2026-06-21).** Re-confirmed on **15 matched connectome seeds** (5 original +
> 10 new) vs `baseline_passthrough` at the same seeds: **Δ = +0.0508 pp, Welch 95% CI lower bound
> = +0.0391** (paired +0.0390) — far more robust than the original thin +0.0135 at n=5. H02's
> warm-start is near-deterministic (std 0.001). The earlier fragility caveat is resolved.

> **Phase-5 generality (2026-06-22).** Confirmed on **MICrONS minnie65** (67k neurons, mouse visual
> cortex, a genuinely independent second large connectome): **Δ = +0.0114 pp, Welch 95% CI lower bound
> = +0.0106 pp** (implementer n=5; verifier n=3 independent re-run: CI_lower = +0.0110 pp). Signal/noise
> ratio ~18σ (microns noise floor σ=0.0006 pp, 29× tighter than connectome), making this the
> **strongest per-σ confirmation of the three**. All six critic checks PASS (leakage, frozen integrity,
> compute fairness 80k equal epochs, significance, cross-dataset consistency, no double-counting).

Variant: `src/mfas/experiments/H02.py`. Pure Rocket score (no post-processing). The greedy order
alone scores 68.91% (connectome) / 90.13% (mouse) before any optimization.

**Reproduce** (env `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
```
python -m eval.run_variant --exp H02 --dataset connectome --seed {42,123,999,7,31415} --out results/ --role confirm
python -m eval.run_variant --exp H02 --dataset mouse --seed {20 seeds: 42,123,999,7,31415,2718,1618,1414,1732,2236,9999,8888,7777,6666,5555,4444,3333,2222,1111,1234} --out results/ --role confirm
# matched baseline: same via --exp baseline_passthrough
```
Result JSONs: `results/*-H02-{connectome,mouse}-*-confirm-{059689,a8bbc0}.json`;
`results/*-H02-microns-*-{implement,confirm}-59bc98.json`;
baseline `results/*-baseline_passthrough-*-{f8cb3c,7b7cba,c2f06f}.json`. Verified independently by the
verifier (read-only) and red-teamed by the critic (frozen-integrity, leakage, reproducibility,
significance, all-dataset robustness — all PASS across Phase 3 and Phase 5).

**Honest caveats.** The win is **modest on large graphs**: mouse +0.21 pp is solid; connectome
+0.0508 pp (hardened, CI lower +0.0391 at n=15); microns +0.011 pp (CI lower +0.011 at n=5+3
independent verify). H02 **missed the lenient 2σ_baseline screen** on connectome and was promoted
via the CONFIRM test (justified: H02 variance is nearly deterministic, so the screen gate's
assumption of variant variance ≈ baseline noise does not apply; see the log's orchestrator escalation
note). **GENERAL WIN confirmed on three real connectomes from two species (fly + mouse visual cortex)**
— the mechanism (greedy-FAS warm-start lands Rocket in a better basin) is graph-species-general.

## #2 — Rocket's plateau is set by the starting basin, not the optimization dynamics (structural)

**Claim.** Across a 9-hypothesis screen (8 distinct mechanisms), the **only** lever that improved the
exact feedforward metric was the one that changed **where optimization starts** (H02's warm-start).
Every intervention on the optimization **dynamics / trajectory** re-converged to — or fell below —
Rocket's plateau, and both **objective/loss-landscape** reshapings slightly **regressed** the
connectome. This is a reproducible structural property of Rocket on these connectomes: *how* it
descends barely matters; *where* it begins does.

**Evidence (each row = one screened variant; n=3 seeds 42/123/999; Δ = variant − baseline on the
exact metric; all numbers from logged `results/*.json`, one git commit per experiment):**

| variant | mechanism / axis | Δ connectome | Δ mouse | Δ microns | result |
|---|---|---|---|---|---|
| **H02** | **starting basin** (greedy-FAS warm-start) | **+0.0448** | **+0.2064** | **+0.0114** | **CONFIRMED win** |
| H01 | restarts (dynamics) | +0.0003 | +0.0000 | — | kill |
| H03 | β schedule (dynamics) | −0.0376 | +0.0114 | −0.0092† | kill |
| H04 | in-loop refinement (dynamics) | −0.0000 | +0.0000 | — | kill |
| H13 | edge-subsample noise (dynamics) | −0.8356 | −0.0325 | — | kill |
| H05 | optimizer → AdamW (dynamics, **falsifier**) | +0.0018 | +0.0000 | — | kill |
| H06 | weight-aware loss reweight (objective) | −0.0380 | −0.0794 | — | kill |
| H11 | margin/hinge surrogate (objective) | −0.0387 | +0.1264‡ | +0.0007† | kill |
| H09 | anti-tie jitter (free-edge) | −0.0010 | +0.0000 | — | kill (0 ties exist) |

† Phase-5 result (MICrONS, 80k epochs, 3 seeds — see log.md Phase-5 section). "—" = not run on microns.
‡ Mouse mean Δ driven by single seed (999: +0.381 pp vs 42: −0.043, 123: +0.041 pp); CI_lower = −0.128 pp — NOT CONFIRMED.

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

**Honest scope.** Negative results over two real connectomes (Phase 3) and one screened arm per
hypothesis; some un-screened arms remain (β_max∈{2,8} for H03, FRAC=0.25 for H13, Lion for H05,
etc.) and four LOW-EV dynamics knobs (H07/H08/H10/H12) were **deferred, not falsified**, when the
campaign hit its `EARLY_EXIT` stop. The inference was pressure-tested (the ideator kept H05 as a
falsifier rather than assuming the conclusion), but it is an inductive conclusion, not a proof.

**Phase-5 corroboration (MICrONS, 2026-06-22).** H11 (objective reshaping) and H03 (β-schedule
dynamics) were re-run on MICrONS (67k neurons, second large connectome): H11 Δ=+0.0007 pp
(NOT CONFIRMED, CI_lower=−0.0003 pp; the previously-reported mouse +0.126 pp is an unconfirmed
single-seed fluctuation); H03 Δ=−0.0092 pp (regression). The objective/dynamics null extends to a
second large connectome from a different species — the conclusion is now supported on two large
graphs and has been pressure-tested by an independent verifier and critic. Full evidence:
`experiments/log.md` (H01–H13 Phase-3 cycles + Phase-5 H11r/H03r entries).

## #3 — The Rocket↔best gap is an OPTIMIZATION-GAP that continuous methods alone cannot close (Phase 4)

**Claim.** Against a downloaded near-optimal ordering (`data/best_solution`, **84.6147%**, vs Rocket-only
**82.93%** → gap **≈1.69 pp**), the gap is a true **optimization-gap, not a surrogate-misalignment**: the
sigmoid surrogate *correctly ranks the near-optimal order higher than Rocket's converged solution at every
β*. Yet the gap is **not recoverable by the Rocket class of continuous/gradient optimization** — across the
diagnosis and two pre-registered Stage-B directions, no continuous lever closes it; it is a distributed
reordering with no discretization slack, and is therefore largely **irreducible to continuous methods**
(explaining why the paper's discrete Crane phase is required to go further).

**Evidence (all from `experiments/outputs/diagnosis.json` + `results/*.json`; reproduce via
`experiments/diagnostics.py` and the Stage-B repro commands in `log.md`):**

| probe | result | implication |
|---|---|---|
| decisive surrogate (scale-fair) | best's order out-surrogates Rocket at **every** β (+238 … +1153) | surrogate aligned → **optimization-gap**, not misalignment |
| drift probe (init AT best) | Rocket collapses 84.61% → 82.75–83.03% under every schedule/scale | best is **unreachable/unholdable** by Adam-on-σ |
| init→plateau (connectome) | **flat** 82.87–82.93% across inits 36–69% | better-init-alone ceiling ≤0.06 pp (DIRECTION I down) |
| gap structure | 7.5% of weight flips; Kendall-τ 0.61; **0 ties**, near-ties 0.03% | distributed reordering, **no discretization slack** |
| H16 (DIRECTION O: monotone β) | connectome **−0.27 pp** vs baseline (KILL) | β-schedule change can't beat the tuned cyclic baseline |
| H19 (DIRECTION R: soft-rank) | mouse −1.94, hard-synth −4.66 pp (EARLY_EXIT) | rank-space stalls (O(1/n) gaps); scale isn't the lever |

The Stage-B negative results were obtained on a **purpose-built hard synthetic** carrying a verified
+0.80 pp optimization gap (`gap.make_hard_synthetic_graph`) plus mouse — i.e. the levers failed even where
a gap demonstrably exists. The Rocket-only best remains **H02 = 82.93% connectome (hardened, CI lower
+0.0391 @ n=15) / 92.48% mouse**.

**Why it matters.** It quantifies the continuous/discrete boundary for Rocket: of the ~1.69 pp Crane gap,
continuous optimization recovers **≈0** beyond H02's warm-start (~0.05 pp). The surrogate is faithful; the
barrier is the non-convex landscape — a near-optimal ordering is not a reachable or even *stable* attractor
of Adam-on-σ, and rank/scale/schedule reparametrizations don't change that. This is the mechanistic reason
the paper needs a discrete MIP (Crane) to surpass Rocket.

**Corroboration — the gap is also irreducible to *bounded-local* discrete refinement (H22 sizing,
2026-06-22).** Sizing the Rocket↔best orientation flips by rank-distance (`experiments/size_localsearch.py`
→ `experiments/outputs/localsearch_sizing.json`; cross-check reproduces gain +4.60 / lose −2.91 / net
+1.69 pp exactly) shows the recoverable weight is **long-range / global**: rank-distance percentiles
p25=8,290 / p50=22,580 / p90=87,497 (of n=136,648), and the net gap recoverable within *any* tractable
window is ≤0 (W=100 → −0.010 pp, W=1000 → −0.170, W=5000 → −0.405; positive only for W≤10). So a
bounded-window single-node local search (sifting/re-insertion, DIRECTION D) cannot close the gap and was
**killed at the sizing gate** before building (H09 pattern). This sharpens the conclusion: closing the
residual requires *global* discrete optimization (the paper's Crane MIP), not local discrete cleanup.

**Honest scope.** best_solution exists only for connectome (mouse and microns have none) so the decisive
surrogate/gap steps are connectome-only; conclusions about DIRECTION R rest on mouse + one hard synthetic.
The optimization-gap diagnosis (surrogate alignment, drift probe, gap structure, window sizing) **cannot be
replicated on MICrONS** — there is no reference near-optimal solution for MICrONS to compare against. The
conclusion that the gap is irreducible to continuous methods therefore rests on the fly connectome + synthetic
evidence only; whether an analogous gap exists for MICrONS is unknown. H17 (basin-hopping), H18 (STE),
H20 (Gumbel-Sinkhorn) were **deferred-by-evidence** (predicted non-improving by H01's prior kill, the drift
probe, and H19's failure — see log.md), not exhaustively falsified. Full evidence + commands:
`experiments/diagnosis.md`, `experiments/log.md` (Phase-4 + Phase-5 sections), `experiments/outputs/diagnosis.json`.
