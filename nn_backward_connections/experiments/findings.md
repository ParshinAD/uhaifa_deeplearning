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

## #3 — At its achievable scale the continuous relaxation ranks the better order LOWER, so no gradient lever reaches it

**Claim.** Against a downloaded near-optimal ordering (`data/best_solution`, **84.6147%**, vs Rocket-only
**82.93%** → gap **≈1.69 pp**), the barrier is **not** "the surrogate points the right way but the optimizer
is too weak". At the scale optimization actually runs at, the surrogate points the **wrong way**: at a common
scale of std ≈ 141 it ranks **Rocket's own 82.92% order above the 84.61% order**, across the entire cyclic
range β ∈ [0.05, 1.05]. The better order becomes preferred only above **β·std ≈ 470**; Rocket operates at
**β·std ≈ 148** and its schedule caps β at 1.05, so that regime is never entered. Consequently no
continuous/gradient lever in the Rocket class closes the gap — confirmed across the diagnosis and two
pre-registered Stage-B directions — while **rank-space discrete refinement does** (findings #4/#5 recover
0.98 pp of the 1.69 pp with 0 gradient steps).

**The decomposition that explains it** (`experiments/outputs/q01_surrogate_ranking.json`, std = 141,
β = 1.05, even spacing for both orders; `F = ceiling − smoothing_loss`, ceiling = discrete score / max w):

| order | ceiling | F | smoothing loss | F as % of ceiling |
|---|---|---|---|---|
| best (84.6147%) | 14,745.87 | 14,215.43 | **530.44** | 96.403% |
| Rocket (82.9161%) | 14,449.86 | **14,390.31** | 59.54 | 99.588% |

    best's true advantage +296.02 − best's extra smoothing loss 470.90 = net −174.88

The better order wins many of its edges *narrowly*, and the surrogate discounts a narrow win toward 0.5;
Rocket's order, being the surrogate's own optimum, holds wide margins on the heavy edges. Full mechanism,
the three scale regimes and the shape-independence check: `diagnosis.md` § Q01.

**Evidence (all from `experiments/outputs/diagnosis.json` + `results/*.json`; reproduce via
`experiments/diagnostics.py` and the Stage-B repro commands in `log.md`):**

| probe | result | implication |
|---|---|---|
| **surrogate ranking at a COMMON scale** (std=141, the operating point) | Rocket's order out-surrogates best across all β ∈ [0.05, 1.05]; net −174.88 at β=1.05 | **misaligned at the achievable scale** — no gradient step points toward the better order |
| **crossover scale** (61-point grid, log-interpolated) | best takes over only at **β·std ≈ 470**, invariant across β ∈ {0.05, 0.3, 1.05}; Rocket runs at 148 | the aligned regime is a factor **3.2** away and the schedule cannot reach it |
| surrogate ranking granting **each order its own optimal spacing** | best wins (+238 … +1153) | alignment holds only if the better order is *also* granted a much larger scale — not the situation optimization is in |
| starting AT the best order (even spacing, std≈0.58) | loses 84.61% → 82.75–83.03% | at small β·std the gradient is the order-independent imbalance vector (cos ≥ 0.98 across three different starting orders) and Adam overwrites any input order |
| init→plateau (connectome) | **flat** 82.87–82.93% across inits 36–69% | better-init-alone ceiling ≤0.06 pp (DIRECTION I down) |
| gap structure | 7.5% of weight flips; Kendall-τ 0.61; **0 ties**, near-ties 0.03% | distributed reordering, **no discretization slack** |
| H16 (DIRECTION O: monotone β) | connectome **−0.27 pp** vs baseline (KILL) | β-schedule change can't beat the tuned cyclic baseline |
| H19 (DIRECTION R: soft-rank) | mouse −1.94, hard-synth −4.66 pp (EARLY_EXIT) | rank-space stalls (O(1/n) gaps); scale isn't the lever |

The Stage-B negative results were obtained on a **purpose-built hard synthetic** carrying a verified
+0.80 pp optimization gap (`gap.make_hard_synthetic_graph`) plus mouse — i.e. the levers failed even where
a gap demonstrably exists. The Rocket-only best remains **H02 = 82.93% connectome (hardened, CI lower
+0.0391 @ n=15) / 92.48% mouse**.

**Why it matters.** It quantifies the continuous/discrete boundary for Rocket: of the ~1.69 pp gap,
continuous optimization recovers **≈0** beyond H02's warm-start (~0.05 pp). The cause is now mechanical
rather than vague: the surrogate scores an edge by a function of the two nodes' *distance*, so it charges a
"smoothing loss" for every narrowly-won edge, and at the achievable β·std that charge (470.90) exceeds the
better order's true advantage (296.02). Rank/scale/schedule reparametrizations move along this trade-off
without escaping it — and four different per-edge shapes (sigmoid, slow tanh, hard clip, cusp) produce the
*identical* ranking at both small and operating scale (`diagnosis.md` § Q01). What escapes it is leaving the
per-edge-distance family altogether, which is exactly what the rank-space discrete sift does (#4/#5).

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

**Phase-6 corroboration — the last untried continuous lever also fails (H34, 2026-06-22).** A
perturbed/blackbox differentiable **sort** surrogate (Berthet 2020 perturb-and-MAP over a sort; the
literature's structural fix for H19's O(1/n) vanishing gradient) was prototyped on the gap-bearing
synthetic. Unlike H19 it does **not stall** (398/400 nodes move, init ~50% → ~73% — the gradient is
genuinely non-vanishing), yet it still **loses to the sigmoid Rocket** (Δ = −0.64 pp, all 3 seeds). This
isolates the bottleneck as the **continuous-relaxation basin itself, not the gradient estimator** —
sharpening #3: no continuous lever closes the gap *regardless of gradient source*. The continuous family
is now exhausted (`experiments/proto_h34_perturbsort.py`; killed by prototype gate, no large-graph compute).

> **⚠ #3 IS BEING REVISED (2026-08-17).** Its scope claim — that *no* continuous/gradient lever
> in the Rocket class closes any of the gap — was tested on its last untried axis and **broke**.
> A **one-sided (asymmetric) surrogate** gains **+0.37 pp on the connectome** as pure Rocket
> (H38, 3 seeds, screen PASS at 9× the gate). The sub-claims about *symmetric* surrogates,
> rank-space and gradient estimators stand; the universal quantifier does not. See § Q04/Q05 in
> `diagnosis.md` and the Phase-6.5 entries in `log.md`; the finding will be rewritten once H38's
> microns arm and CONFIRM stage land.

**Phase-6.5 (a) — the surrogate TAIL axis is closed, at equal core width (H37/H37B, KILL).**
The natural repair of #3's misalignment — a surrogate that ranks the better order higher at the
achievable scale — was built. Q04 (`experiments/outputs/q04_surrogate_tails.json`) first proved
the naive versions are no-ops (`(tanh(z/2)+1)/2 == sigmoid(z)` to 2.2e-16 absolute, so a
"slower-decaying tanh" is the killed H03/A-SCALE β axis) and then found the symmetric shape that
does repair the ranking: an algebraic `z^-4` tail with a narrow core, alignment ratio **+0.173**
vs the sigmoid's **−0.591**. It **lost**: −2.834 pp (hard synthetic) and **−0.932 pp on the
connectome over 3 seeds** through the frozen runner; the tail-isolating arm at matched core width
lost −0.451 pp. The load-bearing comparison holds core width FIXED at the sigmoid's 2.1973 and
varies only the tail: alignment −0.591 → −0.519 → −0.461 against Δ_synthetic 0 → −0.459 → −1.686.
**No tail beats the exponential one at equal core width.** Honest scope, per the critic: mouse
contributes no significant evidence at n=3 (no arm differs from the sigmoid); across *different*
widths the statistic merely re-labels core width (Spearman(width, Δ) = +0.83 vs
Spearman(A, Δ) = −0.89), i.e. the already-killed β axis; and the earlier claim that this
"explains the H11 kill mechanistically" is **RETRACTED** — it contradicts the anti-correlation it
was paired with, and H11's clamp was in fact the only arm to beat the sigmoid on the synthetic
(+0.107 pp, within noise). Full evidence: `diagnosis.md` § Q04, `log.md` (2026-08-17 entries).

## #4 — A cheap full-range discrete sift recovers ~half the connectome gap with NO MIP (Phase 6)

> **⚙ UPDATED by Phase 6.2 (2026-06-23).** H30's sift sweep cap was raised **12 → 40** on
> connectome/mouse (microns stays 12). Re-run H30@40 connectome = **83.8118 ± 0.0143 (3 seeds)**
> (was 83.7761 @12; the +0.045 pp is the extra best-by-oracle sweeps — the Jacobi iterate was
> still improving at the old cap of 12). The old @12 numbers below remain reproducible at the
> prior commit (invariant #5). The Jacobi sift does **not converge** on the large connectomes
> (it enters a period-2 limit cycle); the under-relaxed variant **H35** (finding #5) breaks the
> cycle for a further **+0.098 pp** on connectome.

**Claim.** Appending a **leakage-safe, full-range, exact-gain node re-insertion ("sift")** post-phase to
H02-warm-started Rocket recovers a large, CONFIRMED chunk of feedforward weight the continuous optimizer
leaves on the table — at **equal gradient budget** (the sift adds **0** optimizer steps; only wall-clock)
and on **all three real connectomes**. The move repeatedly places each node at the **exact** rank that
maximizes the feedforward weight of its incident edges given all others fixed (full line, not a window),
via a vectorized-NumPy Jacobi rebuild; the frozen oracle is used **only** to accept/reject whole candidate
vectors (best-by-oracle), never inside a move choice. This is a **GENERAL WIN with graph-dependent
magnitude**.

**Evidence (CONFIRMED, 3-dataset rule; conservative SE = std_comparator·√(2/n), 95% CI lower bound).
H30's pure-Rocket order == H02 by construction, so pure-Rocket is reported separately and the SIFT
INCREMENT is credited as Δ-vs-H02:**

| dataset | H30 refined mean±std (n) | pure-Rocket = H02 (n) | Δ vs H02 (sift) | 95% CI lo | Δ vs baseline_passthrough | 95% CI lo |
|---|---|---|---|---|---|---|
| **connectome** | **83.7761 ± 0.0095 (5)** | 82.9292 (5) | **+0.8468 pp** | **+0.8350** | +0.8917 pp | +0.8606 |
| **microns** | **83.2069 ± 0.0012 (5)** | 83.1287 (5) | **+0.0782 pp** | **+0.0767** | +0.0896 pp | +0.0881 |
| **mouse** | 92.9018 ± 0.0000 (20) | 92.4793 (20) | **+0.4225 pp** | +0.4225 | +0.6289 pp | +0.5049 |

All primaries (connectome AND microns) have CI lower bound > 0 vs **both** comparators; mouse passes
non-inferiority (strongly positive) → **GENERAL WIN** per the Phase-5 decision table. Independently
re-run by the verifier (5/5/20 seeds, clean reproduction) and red-teamed by the critic (frozen integrity,
**no leakage** — re-scored positions equal JSON scores exactly; **no double-counting** — sift increment
credited; significance, compute fairness, generality — all PASS).

**What it means for the gap (revises #3).** On the connectome the sift lifts Rocket **82.93% → 83.78%
(+0.85 pp)**, recovering **~51% of the 1.69 pp Rocket↔best gap with no Crane/MIP**. So the residual #3
deemed to "need the 20-day Crane MIP" is **partly reachable by cheap global discrete refinement**. The
*bounded-local* half of #3 still stands: H30's bounded W=10 sift ≈ 0 (reproduces the H22 kill) — the
recoverable weight is long-range/global, just not MIP-exclusive. The remaining ~0.83 pp to the
challenge SOTA (Vahidi 2025 **84.61%** = 35,462,925/41,912,141, via cheap greedy + bounded-span
insertion + SCC, **no MIP** — verified against arXiv:2506.13799 HTML) is the target for H31 (ILS/LNS)
and SCC-structured insertion.

**Honest caveats.**
- **Graph-dependent magnitude (~11×):** connectome +0.85 pp ≫ microns +0.078 pp. The fly connectome
  leaves far more on the table for full-range exact-gain insertion than MICrONS does. The verdict is
  GENERAL (sign/CI), not a uniform-magnitude claim.
- **Wall-clock ~2× on connectome** (H30 ~157–185s vs H02 ~94s; +37s sift), ~1.2× microns, negligible
  mouse — but **0 extra gradient steps** (`total_grad_steps` == baseline), so the equal-compute basis is
  honest; the wall overhead is disclosed (`sift_time_s` in `history.attrs`, folded into `wall_clock_s`).
- **Jacobi vs Gauss-Seidel:** the production sift uses simultaneous (Jacobi) updates with best-by-oracle
  decoupling (it can transiently overshoot, but the returned best is monotone ≥ pure Rocket); the
  exact-gain kernel is brute-force-verified (`tests/test_refine_insertion.py`). MPS nondeterminism
  affects only the pure-Rocket seed order feeding the sift (σ≈0.01 pp connectome), far below the gain.
- **Thin microns comparators** (H02/baseline n=2 at the matched confirm seeds) — handled conservatively;
  microns signal is ~75σ so the sign is safe.

Variant: `src/mfas/experiments/H30.py` (chains H02 → `run_rocket` → sift); kernel:
`src/mfas/refine/insertion.py` (pure vectorized NumPy; numba/pyamg absent).

**Reproduce** (env `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
```
PYTHONPATH=src python -m pytest tests/test_refine_insertion.py -q   # kernel exactness
for S in 42 123 999 7 31415; do python -m eval.run_variant --exp H30 --dataset connectome --seed $S --out results/ --role confirm --device auto; done
for S in 42 123 999 7 31415; do python -m eval.run_variant --exp H30 --dataset microns    --seed $S --out results/ --role confirm --device auto; done
# mouse: 20 seeds, role confirm; comparators: --exp H02 / --exp baseline_passthrough at matched seeds
```
Result JSONs: `results/*-H30-{connectome,microns,mouse}-*-{implement,verify,confirm}-{1976d9,954bab,ff2174}.json`.
Full cycle (implementer/verifier/critic verdicts): `experiments/log.md` (Phase-6 H30 entry).

## #5 — Under-relaxation breaks the Jacobi sift limit cycle: +0.098 pp on the fly connectome (Phase 6.2)

**Claim.** H30's production sift is a **Jacobi** iteration (every node jumps FULLY to its
exact feedforward-maximising gap each sweep). On the two large dense connectomes it does **not
converge** — it enters a period-2 **limit cycle** (thousands of nodes leapfrogging each other
forever; on connectome ~5,255 movers stuck, candidate alternating 83.807/83.796), so
best-by-oracle only creeps up via the lucky phase. Replacing the rebuild with an
**under-relaxed** step (move each mover only a fraction `α=0.7` of the way to its gap, after
`k_full=6` full warm sweeps) **breaks the cycle**: the iterate converges (movers collapse to a
few hundred) and reaches a strictly higher fixed point — at the **same gradient budget** (the
sift adds 0 optimizer steps) and the same per-sweep cost. Variant **H35** = H02→Rocket→
under-relaxed two-phase sift (`src/mfas/experiments/H35.py`, refiner
`src/mfas/refine/underrelax.py`). At `α=1` it is bit-identical to H30's sift (unit-tested), so
the gain is purely the `α<1` dynamics.

**Evidence (3 seeds 42/123/999; H30 here is H30@40, the new baseline; Δ is seed-matched mean ±
paired-std, conservative 95% CI lower bound = mean − 1.96·SE):**

| dataset | H35 mean±std | H30@40 mean±std | Δ(H35−H30) | 95% CI lo | Δ vs H02 | Δ vs baseline_passthrough |
|---|---|---|---|---|---|---|
| **connectome** | **83.9101 ± 0.0060** | 83.8118 ± 0.0143 | **+0.0983 pp** | **+0.0759** | +0.9808 | +1.0144 |
| mouse | 92.9018 ± 0.0000 | 92.9018 ± 0.0000 | +0.0000 (non-regressing) | — | +0.4225 | +0.8322 |
| microns | 83.2045 ± 0.0005 | 83.2064 ± 0.0010 (@12) | **−0.0019 pp** | −0.0035 | +0.0757 | +0.0874 |

**Verdict: CONFIRMED win on the connectome (the primary large graph and where the Rocket↔best
gap lives); non-regressing on mouse; a marginal regression vs H30 on microns at the 12-sweep
cap — so this is NOT a clean 3-dataset GENERAL WIN like H30.** All three connectome per-seed
deltas are positive (+0.080/+0.119/+0.096) and the CI lower bound is well above 0. On the
connectome this lifts Rocket's 82.93% plateau to **83.91%**, recovering ~58% of the 1.69 pp
Rocket↔best gap (vs H30's ~51%) and closing ~17% of the residual to the 84.61% challenge SOTA
(Vahidi 2025), still with **no MIP and 0 extra gradient steps**.

**Why microns regresses (honest).** The under-relaxation needs more sweeps than Jacobi to
converge; the exploratory `dr_tmp` sizing showed it overtakes Jacobi on microns only at ~30
sweeps (+0.008 pp there), but microns' cap was kept at 12 for adequate runtime (its per-sweep
sift cost is ~3–9× connectome's; the 80k-epoch Rocket dominates the wall regardless). With
`k_full=6` only 6 under-relaxed sweeps run on microns — not enough to overtake H30's 12 full
Jacobi sweeps, so H35 lands −0.0019 pp (≈2× the tiny microns noise floor) below H30 there. H35
still beats H02/baseline on microns (+0.076/+0.087). A general win would need a higher microns
sweep cap or a cycle-triggered α (switch to under-relaxation only once oscillation is detected)
— deferred. Variant config: `_MAX_SWEEPS={connectome:40, mouse:40, microns:12}`, `_K_FULL=6`,
`_ALPHA=0.7`.

**Reproduce** (env `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
```
PYTHONPATH=src python -m pytest tests/test_refine_underrelax.py -q   # α=1 ⇔ H30 sift; monotone
for S in 42 123 999; do python -m eval.run_variant --exp H35 --dataset connectome --seed $S --out results/ --role implement --device auto; done
for S in 42 123 999; do python -m eval.run_variant --exp H30 --dataset connectome --seed $S --out results/ --role implement --device auto; done  # @40 baseline
for S in 42 123 999; do python -m eval.run_variant --exp H35 --dataset mouse   --seed $S --out results/ --role implement; done
for S in 42 123 999; do python -m eval.run_variant --exp H35 --dataset microns --seed $S --out results/ --role implement --device auto; done
```
Exploratory diagnosis (the limit-cycle + α sizing): `dr_tmp/FINDINGS_underrelaxation.md`,
`dr_tmp/exp_sweeps.py`, `dr_tmp/exp_damped.py`, `dr_tmp/exp_underrelax.py`.

## Phase-6 summary — global discrete refinement (H30–H34, 2026-06-22)

**Result: 1 CONFIRMED win (H30), 4 kills.** The backlog (H30–H34) is exhausted.

| ID | idea | verdict | decisive evidence |
|---|---|---|---|
| **H30** | full-range exact-gain sift (post-Rocket) | **CONFIRMED GENERAL WIN** (finding #4) | connectome +0.85 pp over H02 (CI_lo +0.835), microns +0.078, mouse +0.42; ~51% of the gap, no MIP |
| H31 | ILS/LNS wrapper on the sift | kill (screen) | beats sift +0.25 pp on synthetic but Δ vs H30 = −0.0008 pp on connectome at ~1.9× wall |
| H32 | trophic-Laplacian warm-start → sift | kill (prototype gate) | trophic→sift −2.12 pp (synthetic) / −1.26 (mouse) vs greedy→sift |
| H33 | magnetic-Laplacian warm-start → sift | kill (prototype gate) | quality −4.97 pp vs greedy→sift AND eigensolve 304–507 s (~5–8× over budget, no AMG) |
| H34 | perturbed-sort continuous surrogate | kill (prototype gate, FALSIFIED) | moves (non-vanishing grad) but −0.64 pp vs sigmoid Rocket on synthetic |

**Cross-cutting insight (extends #2).** The **discrete refiner does the work; the basin it starts from
barely matters.** Full-range exact-gain sift recovers the gap from a greedy-FAS order about as well as from
a Rocket order (the `dr_tmp` "sift-on-greedy ≈ sift-on-Rocket" result), and two *global linear-algebra*
warm-starts (trophic H32, directional-spectral magnetic H33) are **worse** seeds for the sift than plain
greedy-FAS — so **greedy+sift is the design**; no smarter init earns its cost. This mirrors finding #2
(*where* you start matters, not *how* you descend) one level up: among discrete starting orders for the
sift, the cheap greedy peel is already at least as good as expensive spectral/trophic embeddings.

**Is finding #3 revised? Partially, yes (see the banner on #3).** The ~1.69 pp connectome Rocket↔best gap
is **not** MIP-exclusive: a cheap, leakage-safe *global* (full-range) discrete refinement recovers ~51% of
it with 0 MIP and 0 extra gradient steps (H30). The *bounded-local* half of #3 still stands (H30's W=10
sift ≈ 0 reproduces the H22 kill — the recoverable weight is long-range/global). The *continuous-only*
sub-claims of #3 are reinforced, not revised: H34 shows even a non-vanishing-gradient continuous surrogate
cannot beat the relaxation basin. Independent prior art (Vahidi 2025, arXiv:2506.13799, verified vs HTML)
reaches the **full 84.61%** on this exact graph with cheap greedy + *bounded-span* insertion + SCC and **no
MIP** — consistent with our message that the residual is reachable by cheap combinatorial refinement. The
remaining ~0.83 pp from H30's 83.78% to that 84.61% SOTA is the target for future SCC-structured insertion
(H31's generic ruin-&-recreate did not reach it within budget).

**Runtime.** H30 adds 0 gradient steps; wall ~1.4–2× connectome (+~37 s sift), ~1.2× microns, negligible
mouse — all ≤ the 2× ceiling. The four kills consumed only cheap CPU prototype gates (+ one connectome
eigensolve timing benchmark for H33); no wasted large-graph variant cycles. All numbers trace to
`results/*.json` (H30/H31) or `experiments/outputs/proto_h3{2,3,4}_*.json` (gates) with re-runnable
commands in `experiments/log.md`. One git commit per experiment on branch `phase6-global-discrete`.
