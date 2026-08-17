<!-- TRACK-B ANSWER LEDGER. This file records answered diagnostic questions (see
experiments/questions.md and PROTOCOL.md § "Track B — Diagnostics"). New answers are appended as
`## Q0x — <question>` sections that questions.md links to. The Phase-4 section below predates the
Q-numbering and is kept as-is (cited by hash elsewhere). -->

# Phase-4 Stage-A Diagnosis — the Rocket↔best gap

**Question.** Of the ~1.7 pp gap between Rocket-only quality and a near-optimal ordering, how
much is recoverable by continuous/gradient methods alone, and *why* does the gap exist —
because Rocket fails to **reach** the surrogate optimum (OPTIMIZATION-GAP) or because the sigmoid
surrogate's optimum **≠** the discrete optimum (SURROGATE-MISALIGNMENT)?

**All numbers below come from `experiments/outputs/diagnosis.json`**, produced by
`python experiments/diagnostics.py --steps 0,1,2,3,4` (env `allen`, MPS, seed 42), which imports
the privileged `mfas.analysis.gap` — the only reader of `data/best_solution`. Drift-probe and
best-seeing outputs are diagnostics; none are written to `results/` and none touch any
optimization path (leakage audit clean; frozen-guard OK; `pytest` 11/11).

## Anchors
| quantity | value |
|---|---|
| best_solution (oracle) | **84.6147%** (35,463,823 / 41,912,141) |
| Rocket-only best (H02 warm-start) | 82.9273% |
| Rocket baseline (random init) | 82.9160% |
| Crane reference (paper, discrete MIP) | 84.60% |
| **gap to close (continuous-only)** | **≈ 1.69 pp** |

Coverage verified: best_solution is a clean permutation over the exact connectome node set
(136,648 nodes, ranks 0..n−1, valid `searchsorted` remap). **No mouse best_solution exists** →
best-dependent steps (1, 2) are connectome-only; steps 3–4 cover both datasets.

---

## VERDICT: OPTIMIZATION-GAP (the surrogate is well-aligned; the optimizer does not reach it)

### Step 1 — decisive surrogate experiment
The surrogate value depends on position **gaps**, not just order, so we compare *scale-fair*
quantities: the **maximum surrogate achievable by best's fixed order** (monotone spacing optimized
with a free global scale, `gap.optimize_spacing`) vs **Rocket's converged surrogate**.

| β | best-order max surrogate | Rocket (H02) converged surrogate | best − Rocket |
|---|---|---|---|
| 0.05 | 14,680 | 13,526 | **+1,153** |
| 0.50 | 14,673 | 14,402 | **+271** |
| 1.05 (convergence) | 14,679 | 14,440 | **+238** |

**Read this table with its scope.** It grants **each order its own optimal spacing** (free scale).
Under that comparison best's order out-surrogates Rocket's at every β. That is NOT the situation
optimization is in: at a **common** scale — Rocket's operating std ≈ 141 — the ranking **reverses**
and the surrogate prefers Rocket's own order across the whole cyclic β range (§ Q01). So the correct
reading is: the surrogate is aligned only if the better order is also granted a much larger scale;
at the achievable scale it is **misaligned**, and that is the bottleneck.

**Drift probe (corroborating, diagnostic-only).** Initialising Rocket *at* the 84.61% ordering and
running its optimization:

| schedule from best | final % | min along trajectory | drop |
|---|---|---|---|
| cyclic β (baseline), lr .05 | 82.94 | 78.01 | −1.67 |
| constant β=1.05, lr .05 | 83.03 | 78.87 | −1.58 |
| constant β=1.05, lr .005 | 82.75 | 76.60 | −1.86 |

Under every schedule tried **at even spacing (std≈0.58)**, gradient ascent on the surrogate flows
the near-optimal solution back down to Rocket's ~82.9% plateau (dipping into the 76–79% range first).
The mechanism is measured in § Q01: at that scale the gradient is the **order-independent** imbalance
vector (cos ≥ 0.98 between the gradients evaluated at three completely different starting orders),
and Adam's per-coordinate step (≈ lr regardless of gradient magnitude) traverses ~8.6× the entire
position spread in 1000 steps — so any input order, good or bad, is overwritten. These runs measure
the small-scale blind regime; they say nothing about the optimum's stability.

### Step 4 — init→plateau curve (rules DIRECTION I down)
Post-Rocket plateau vs init quality, leakage-safe inits:

| init | connectome init% → plateau% | mouse init% → plateau% |
|---|---|---|
| random | 50.16 → 82.874 | 57.15 → 91.910 |
| uniform | 49.84 → 82.895 | 71.91 → 92.420 |
| degree_diff | 35.78 → 82.930 | 12.07 → 92.393 |
| degree_abs | 46.86 → **82.934** | 27.41 → 91.902 |
| greedy (H02) | 68.91 → 82.930 | 90.13 → **92.479** |

**Connectome plateau is essentially init-invariant (82.87–82.93%, range 0.06 pp) despite inits
spanning 36–69%.** Better init does *not* buy a better plateau — Rocket re-converges to its own
std≈141 basin regardless of init quality (⚠ the separate 84.6%-init drift "collapse" was an
even-spacing artefact, § Q01; the point stands via *reachability*: Rocket doesn't navigate to a
better basin from any init). **Better-init-alone (DIRECTION I) has a ~0.06 pp ceiling on
connectome** and is therefore *not* the primary lever; H02's confirmed +0.045 pp is approximately
all that init can buy. (Mouse is mildly init-sensitive — greedy best — but is near-saturated.)

---

## Gap structure (Step 2) — a distributed reordering, not heavy edges / hubs / ties
- **8.47% of edges (7.51% of weight) flip direction** between Rocket's order and best.
- Decomposes into **+4.60% best-feedforward-but-Rocket-feedback** vs **−2.91% the reverse**, netting
  the **+1.69%** discrete gap — i.e. two large opposing flows, not a small correction.
- Disagreement is **uniform across edge-weight buckets** (9.8% at w=2 → 5.7% at w>100) and sits on
  **median-degree, not hub, endpoints** (≤median degree carries 3.32% of disagreement weight vs
  0.035% for >p99). No "few heavy/hub edges" story.
- **Order correlation: Spearman 0.757, Kendall-τ 0.610** — *moderate*: substantially more reordering
  than "local swaps", but not a wholly random/different basin. The good basin is partially aligned
  with Rocket's, reachable in principle but missed by gradient flow.

## Self-diagnostic (Step 3) — not ties, and not pure non-convergence
- **0 exact position ties; near-ties negligible** (feedback weight within |Δpos|<1.0 of a flip is
  **0.03%** vs the 1.69% gap). The gap is **genuine misordering** by the surrogate optimizer, not
  strict-`>` discretization loss (confirms Phase-3 finding #2a).
- Surrogate is **still descending at stop** (neg-loss tail slope +52/log-step) but the **discrete
  plateau slope ≈ 0** — extra surrogate progress no longer converts to discrete gain. So it is not
  simply "stop too early"; the late surrogate landscape is flat in discrete terms.

## Synthetic generalization check
A planted-order synthetic (n=400, 96.10% planted optimum): **Rocket reaches 96.56% > planted**, and
its converged surrogate ≥ the planted order's at high β — i.e. **no optimization gap on the easy,
mostly-acyclic graph.** The connectome gap is a property of its **harder cyclic structure**;
a Stage-B prototype graph must inject far more feedback / strong-connectivity to reproduce it.

---

## Selected direction(s) — rule-based, ranked by diagnostic magnitude

Evidence: (1) the surrogate is aligned only when each order is granted its own optimal spacing; at
the common operating scale it **prefers Rocket's own order** (§ Q01). (2) DIRECTION I has a **flat
init→plateau** (≤0.06 pp) → down-ranked. (3) **Best is not *reachable* by free gradient flow from a
generic start** — at β·std ≈ 148 no local step points toward it. (4) Kendall-τ
0.61 → moderate, distributed reordering (partial-basin difference). Per the pre-registered tree
(1a dominant → DIRECTION O), and refined by (3):

1. **PRIMARY — DIRECTION O (reach/hold a better basin via stronger continuous optimization):**
   - **Monotone β-continuation / graduated optimization** — replace the cyclic schedule that
     repeatedly *re-melts* β→0.05 (and demonstrably destroys good orders) with a single slow
     sharpening (low-β convex → high-β), tracking the solution; pair with the H02 init.
   - **Continuous basin-hopping / parallel tempering** on the surrogate (perturb→re-optimize,
     keep best-by-oracle) to escape the dominant 82.9% basin.
   - (L-BFGS/second-order is *low* priority — Step 3 shows local convergence quality is not the
     bottleneck.)
2. **SECONDARY — DIRECTION R (stop the optimizer from collapsing good orders):** motivated directly
   by the drift-probe finding. **Straight-through estimator** (exact discrete order in the forward
   pass, surrogate gradient in the backward) and **position→soft-rank differentiable ranking**
   (rank-space normalizes the scale blow-up that saturates the surrogate). Both are O(m)/O(n log n)
   and scale; **Sinkhorn/Gumbel-Sinkhorn are O(n²) → prototype on mouse + a hard synthetic only.**
3. **DOWN-RANKED — DIRECTION I (better init alone):** flat init→plateau ceiling; H02 already
   captured the available gain. Stronger inits are only worth pursuing *in combination* with O/R
   (a good basin that the dynamics can now hold).

**Honest ceiling read.** Because a perfect init is not *reached* by gradient flow from a generic
start (the surrogate does not rank it higher at the achievable β·std, § Q01), and the gap is a
distributed reordering
rather than recoverable ties, a large share of the 1.69 pp may be
**irreducible to continuous optimization** and genuinely require discrete refinement (the paper's
Crane phase). Stage B will quantify how much O/R recovers; the prototype-first gate on a *hard*
synthetic + mouse decides whether any variant earns connectome compute.

**Reproduce:** `python experiments/diagnostics.py --steps 0,1,2,3,4` → `experiments/outputs/diagnosis.json`.

---

## Q01 — Why does starting Rocket from the best solution lose score?

**Answer (2026-08-17).** Because Rocket does not optimize an *order* — it optimizes *coordinates*,
and the surrogate it maximizes is a different objective at every scale. Embedding a given order
requires choosing a spacing, and everything depends on the single product **β·std**. At Rocket's
operating scale the surrogate **does not rank the better order higher**, so moving away from the
best order is correct behaviour *for the surrogate* — not a failure of the optimizer, and not
evidence about the optimum's stability.

Primary artifact: `experiments/outputs/q01_surrogate_ranking.json`
(`experiments/diagnostics/q01_surrogate_ranking.py`, connectome, float64, even spacing for every
order so the comparison is like-for-like).

### The three regimes of the surrogate

`F(P) = Σ_e ŵ_e · σ(β·Δ_e)`, `Δ_e = p_tgt − p_src`, `ŵ = w/max(w)`. Reference points that do not
depend on β: blind floor `0.5·Σŵ = 8,713.54`; saturation ceilings (= discrete score / max w)
`best = 14,745.87`, `Rocket = 14,449.86`.

**(1) β·std → 0 — the surrogate becomes a DIFFERENT problem.** With `σ(x) = ½ + x/4 + O(x³)`:

    F(P) = Ŵ/2 + (β/4)·Σ_e ŵ_e Δ_e + O((β·std)³) = Ŵ/2 − (β/4)·⟨c, P⟩ + O((β·std)³)

with `c_k = out_ŵ(k) − in_ŵ(k)`. The edge sum **telescopes** into a node-level quantity: it no
longer knows which node precedes which, only where each node sits, weighted by its own imbalance.
By the rearrangement inequality the maximizer over permutations is the **imbalance sort**, not the
feedforward optimum. Measured at std = 0.001, β = 1.05 (linear model reproduces F to **3.1e-10**
relative):

| order | discrete % | ⟨c,P⟩ @std=1 | F @std=0.001 | surrogate's preference |
|---|---|---|---|---|
| imbalance sort | 69.6345 | −14,810.51 | **8,717.431** | **1st** |
| Rocket | 82.9161 | −9,098.92 | 8,715.931 | 2nd |
| best | **84.6147** | −6,749.25 | 8,715.315 | **3rd** |
| random | 50.1583 | −78.32 | 8,713.563 | 4th |

The surrogate's ranking is **exactly inverted** relative to the truth, and it is ordered by `−⟨c,P⟩`
alone. Per-edge monotonicity is intact the whole time — what fails is that comparing two whole
*orders* is not the same as improving one edge at a time.

**(2) β·std ≈ 148 — Rocket's operating point. The better order still loses, by 175.**
Write `F = ceiling − smoothing_loss`, where the ceiling is the true objective in ŵ units and the
smoothing loss is what the surrogate discounts for edges that are only *narrowly* correct
(σ ≈ 0.5 instead of 1). At std = 141, β = 1.05:

| order | ceiling | F | smoothing loss | F as % of ceiling |
|---|---|---|---|---|
| best | 14,745.87 | 14,215.43 | **530.44** | 96.403% |
| Rocket | 14,449.86 | **14,390.31** | 59.54 | 99.588% |
| imbalance sort | 12,135.26 | 12,129.27 | 6.00 | 99.951% |
| random | 8,741.13 | 8,740.73 | 0.41 | 99.995% |

    best's true advantage        +296.02
    best's EXTRA smoothing loss  −470.90
    net surrogate advantage      −174.88   → the surrogate prefers the WORSE order

The best order's genuine +296 advantage is **more than cancelled** by the 8.9× larger smoothing
loss it pays: it wins many of its edges *narrowly*, and the surrogate discounts narrow wins toward
0.5. Rocket's order, being the surrogate's own optimum, has arranged wide margins on the heavy
edges and sits at 99.6% of its own ceiling. (Spacing-dependent but robust: at the best order's own
optimized spacing at std ≈ 141 it still loses, `q01_drift.json` `beta_analysis.at_std141`,
β ∈ {0.05, 0.30, 1.05} → −1,240.8 / −596.6 / −223.1.)

#### Where the smoothing loss comes from: the surrogate's RESOLUTION

`σ(0) = 0.5` exactly — at zero separation the surrogate answers "I cannot tell" and pays half
credit. That is the unavoidable price of differentiability: the step has to be smeared, and the
middle of a smeared step is the middle. What decides whether this matters is how the smear width
compares with the distance between **rank-adjacent** nodes, and on this graph that comparison is
brutal (measured, `resolution` block of the artifact, std = 141, β = 1.05, n = 136,648):

    distance between two rank-adjacent nodes = 0.003574 position units
    beta * that distance                     = 0.003753
    credit for an edge between them          = sigma(0.003753) = 0.5009

| nodes are this many ranks apart | β·Δ | credit |
|---|---|---|
| 1 | 0.004 | **50.1%** |
| 100 | 0.375 | 59.3% |
| **293** | 1.10 | 75% |
| **585** | 2.20 | 90% |
| **1,224** | 4.60 | 99% |

**The surrogate's resolution is ~1,000 ranks out of 136,648** — it cannot resolve the ordering at
any granularity finer than ~1% of the graph. Everything inside a thousand-node window is a blur in
which every edge, correct or not, scores about half.

That is what makes the loss *asymmetric between the two orders*: the better order collects much
more of its winning weight inside the blur.

| order | credit lost on correct edges | credit gifted to wrong edges | net | median rank distance of won weight | share of won weight < 1,000 ranks |
|---|---|---|---|---|---|
| best | 669.07 | 138.63 | **530.44** | 8,776 | **16.1%** |
| Rocket | 71.64 | 12.10 | **59.54** | 22,693 | **3.1%** |

Rocket's order *is* the surrogate's own optimum, so it has learned that narrow wins do not pay: it
spreads its heavy edges out (median won-edge distance 22,693 ranks) and simply does not contest the
fine-grained battles. The best order contests and wins them — at half rate. The second column is the
mirror image of the same effect: a *wrong* edge at short range is also gifted ≈0.5, which is why the
net loss is the difference (669.07 − 138.63 for best, 71.64 − 12.10 for Rocket).

Sharpening β would narrow the blur, but that is the other jaw of the vice: already at β = 1.05, 99%
of edge-level gradient mass sits on 2.83% of edges (insight 4 below). Wide blur ⇒ gradient but no
resolution; narrow blur ⇒ resolution but no gradient. The discrete sift escapes the vice by working
in rank space, where "one rank ahead" counts exactly as much as "twenty thousand ranks ahead" —
precisely as the true metric counts it.

**(3) β·std ≈ 470 — the crossover, which Rocket never reaches.** Log-interpolated (not a grid
node), on a 61-point grid:

| β | crossover std | **β·std** |
|---|---|---|
| 0.05 | 9,403.3 | **470.2** |
| 0.30 | 1,565.3 | **469.6** |
| 1.05 | 447.9 | **470.3** |

The invariance across three βs is the direct verification that **F sees only the product β·std**.
Rocket operates at β·std ≈ 1.05 × 141 = **148**, a factor **3.2** below the crossover, and the
cyclic schedule caps β at 1.05 — so the regime where the surrogate would prefer the better order is
never entered.

### It is not the SHAPE of the sigmoid

Four per-edge shapes, same orders, same spacings — sigmoid, a slower-decaying tanh (`tanh(x/10)`),
a hard-clipped ramp (H11's exact form, M = 5), and a cusped shape (`|x|^0.5`, deliberately
non-differentiable at 0):

| shape | ranking at std = 0.01 | ranking at std = 141 |
|---|---|---|
| sigmoid | imbalance > Rocket > best | Rocket > best > imbalance |
| slow tanh | imbalance > Rocket > best | Rocket > best > imbalance |
| hard clip | imbalance > Rocket > best | Rocket > best > imbalance |
| cusp `\|x\|^0.5` | imbalance > Rocket > best | Rocket > best > imbalance |

**All four agree at both scales.** The bias is therefore not a defect of the sigmoid but a property
of the whole family `Σ_e ŵ_e · g(Δ_e)`: any objective that scores an edge by a function of the two
nodes' *distance* rewards spreading nodes by imbalance, and the true objective (a function of the
*sign* only) is exactly the member of that family with zero gradient everywhere. Changing `g`
moves along this trade-off; it does not escape it. Escaping it requires leaving the family — which
is what rank-space methods do, and why the discrete sift (H30/H35) recovers what no continuous
lever has.

### What this does NOT establish

- **Nothing about the stability of the optimum.** A "hold" measured at very large scale is a frozen
  optimizer, not stability: at std ≈ 5·10⁴ over 99.99% of edges have |β·Δ| > 37, σ′ underflows, and
  *every* configuration holds — Rocket's own 82.92% order holds just as exactly as the 84.61% one.
  Giving Adam a step size proportional to the scale (lr = 19 at std 53,626) reproduces a −0.389 pp
  drop, comparable to the −0.430 pp drop at std 141.
- **‖∇F‖ near zero is not evidence of a critical point** — the norm is not scale-free (the
  dimensionless `‖∇F‖·std` is *larger* at large scale), and F has **no finite-scale maximum**: it
  increases monotonically with scale toward the discrete ceiling.
- **The drop is governed by how far Adam travels, not by local steepness.** Across the scale sweep
  the drop is monotone in `lr·T/std` over six decades and *anti*-correlated with `‖∇F‖·std`. Adam's
  per-coordinate step is ≈ lr regardless of gradient magnitude, so at std ≈ 0.58 it traverses ~8.6×
  the entire position spread in 1000 steps and overwrites any input order. At small scale that
  gradient is also order-independent: cos between the gradients evaluated at the best order,
  Rocket's order and a random order is ≥ 0.98.

### Insights & how to use

1. **The plateau is a property of the relaxation's scale, not of the optimizer.** At the achievable
   β·std the continuous surrogate does not rank the better order higher, so no local gradient step
   points toward it. This is the mechanistic core of finding #3.
2. **This is why discrete refinement works and continuous levers do not.** The sift operates on
   ranks and is immune to the whole β·std trade-off — it finds the reordering the surrogate cannot
   see at β·std ≈ 148, recovering ~0.98 pp with 0 gradient steps (findings #4/#5).
3. **β and position-scale are the SAME knob**, verified two ways: F is bit-identical at
   (β=1.05, std=141), (β=0.3, std=493), (β=3.0, std=49) — all β·std ≈ 148 — and the crossover lands
   at β·std ≈ 470 for all three βs. So "scale annealing" (roadmap **A-SCALE**) *is* "sharper
   terminal β" (**H03, already KILLED**, connectome −0.038 pp). Both hit the same wall: at large
   β·std, σ saturates ⇒ vanishing gradient (why the schedule caps β ≤ 1.05 and re-melts; the H19
   soft-rank stall is the same failure). Across the whole cyclic range β ∈ [0.05, 1.05] at std ≈ 141
   the best order is **never** preferred, so the conclusion is robust to β.
4. **Gradient coverage at the operating point, with the honest caveat.** At β = 1.05 on Rocket's
   converged positions, 99% of the *edge-level* gradient mass sits on 2.83% of edges (159,866 of
   5,657,719); |β·Δ| < 5 covers 1.70% of weight but 95.6% of that mass. After in/out cancellation at
   the *node* level, however, 99% of ‖∇F‖₁ is spread over 42.8% of nodes — so "the surrogate
   discounts almost all edges" is fair, while "the optimizer sees only a sliver of the graph" is not.

**Reproduce** (env `allen`, from the repo root):
```
PYTHONPATH=src python experiments/diagnostics/q01_surrogate_ranking.py     # ~1 min, the primary artifact
PYTHONPATH=src python experiments/diagnostics/q01_drift_from_optimum.py    # ~3-6 min, scale sweep + beta_analysis
```
Artifacts: `q01_surrogate_ranking.json` (regimes, decomposition, crossover, shape comparison);
`q01_drift.json` + `q01_scale_sweep.png`, `q01_F_vs_scale.png` (scale sweep, β analysis).

---

## Q02 — How far apart are Rocket solutions across seeds (no greedy warm-start)?

**Answer (2026-08-03): the discrete SCORE is stable across seeds but the ORDERING is not — plain
Rocket has a degenerate, near-flat set of near-optimal orderings that all score ~the same
feedforward weight.** The one robust structure inside that degeneracy is a source/sink asymmetry:
the **extreme sinks reproduce far more stably than the extreme sources** — but this is a small-k /
**tail** phenomenon that closes as k→n. This is a direct extension of the prior
`experiments/rocket_base.ipynb` seed-stability analysis (which found sinks Jaccard@1000 ≈ 0.65–0.74
vs sources ≈ 0.33–0.40), now quantified with global rank correlations and front/back Jaccard curves.

**Evidence** (`experiments/diagnostics/q02_seed_distance.py` → `experiments/outputs/q02_seed_distance.json`
+ `q02_seed_distance.png`; plain Rocket, random N(0,1) init, **no greedy warm-start**; MPS + random
init = the only stochastic sources):

| quantity | connectome (PRIMARY) | mouse (secondary) |
|---|---|---|
| n, epochs, seeds | 136,648; 20,000; 42/123/999/7/31415 | 148; 5,000; 5 seeds |
| discrete % (mean ± std) | **82.8845 ± 0.0225** | 92.1745 ± 0.2176 |
| Spearman ρ (mean ± std over 10 pairs) | **0.9577 ± 0.0012** | 0.9074 ± 0.0369 |
| Kendall-τ | **0.9095 ± 0.0032** | — |

Front/back Jaccard@k across seed pairs (FRONT = k lowest-rank nodes = **sources**; BACK = k
highest-rank = **sinks**):

| dataset | k | Jaccard FRONT (sources) | Jaccard BACK (sinks) | back − front |
|---|---|---|---|---|
| connectome | 100 | 0.186 | **0.598** | +0.412 |
| connectome | 1000 | 0.370 | **0.684** | +0.314 |
| connectome | 10000 | 0.783 | 0.794 | **+0.011** |
| mouse | 50 | 0.675 | **0.924** | +0.249 |

(microns DEFERRED — 80k epochs × 5 seeds is too expensive; noted in the JSON.)

**Precise framing (what the numbers do and do NOT say).**
- **Score stable, order not.** Score std is 0.0225 pp (connectome) while global rank correlation is
  Spearman 0.958 / Kendall-τ 0.910 — high but *not* 1.0. Many near-equivalent orderings yield
  near-identical feedforward weight: a **degenerate / flat set of near-optimal solutions**, not one
  attractor recovered up to noise.
- **The source/sink asymmetry is a genuine structural signal, not a Jaccard-on-extremes artefact.**
  A symmetric algorithm would give equal front/back overlap at each k; the back−front gap (+0.31 at
  k=1000, verifier's independent 3-seed recompute +0.313 vs builder +0.314) shows the two ends are
  *not* interchangeable.
- **⚠ Qualifier — it is a small-k / TAIL effect, not "the whole back half is stable."** The gap is
  large only for the *extreme* tails (k=100: 0.60 vs 0.19; k=1000: 0.68 vs 0.37) and **nearly
  vanishes by k=10000** (0.783 vs 0.794, gap +0.011). As k→n both ends converge because any two
  near-optimal orderings necessarily share their bulk. The claim is narrowly: the *extreme sinks*
  reproduce stably vs the *extreme sources* — do **not** read it as a stable back half.

**Interpretation (hypothesis, corroborating the prior notebook).** Sinks are terminal,
high-in-degree targets: their late position is **over-constrained** by many incoming edges, so it is
pinned and reproduces across seeds/basins. Extreme sources have weaker positional constraints (fewer
edges anchor them early), so many arrangements of the front are near-equivalent in feedforward
weight and each basin picks a different one. This is stated as a hypothesis consistent with the
`rocket_base.ipynb` finding, not a proven mechanism.

**Connections.**
- **Retroactively explains the H01 (multi-start / best-of-K) KILL.** If every seed lands at ~82.9%
  with only *tail* reshuffling and no rich score tail (score std 0.0225 pp), then best-of-K harvests
  ≈nothing — there is no dispersed score distribution to skim the max from. Q02 quantifies *why*
  multi-start was dead (see `findings.md` #2 and the H01 backlog kill).
- **Feeds Track C (random graphs vs brains).** The degeneracy of the feedback-minimizing order — and
  specifically *which* ends are pinned vs free — is a structural property of the connectome to
  compare later against SBM / configuration-model nulls (does a random graph with matched degree show
  the same sink-pinned / source-free asymmetry?).

**Reproduce:** `PYTHONPATH=src python experiments/diagnostics/q02_seed_distance.py`
(env `allen`). Artifacts: `experiments/outputs/q02_seed_distance.json`,
`experiments/outputs/q02_seed_distance.png`. Extends `experiments/rocket_base.ipynb`.

---

## Q04 — Is the failure a property of *any* g(Delta), or only of exponentially-tailed g?

**Answer (2026-08-17).** Of the whole family — but the reason is sharper than Q01 stated, and
one of Q01's supporting rows was mis-interpreted. **Alignment is bought by narrowing the
surrogate's core, which is the `beta` axis; the tail exponent is a separate, genuinely new
degree of freedom that decides what the narrow core *costs*. Neither buys score.**

Primary artifact: `experiments/outputs/q04_surrogate_tails.json`
(`experiments/diagnostics/q04_surrogate_tails.py`, connectome, float64, even spacing for
every order — the same like-for-like convention as Q01).

### 0. `tanh` IS the sigmoid — the naive reading is a no-op

`(tanh(z/2)+1)/2 == sigmoid(z)` to **2.2e-16** over 2·10⁵ points, so `tanh(z/a) == sigmoid(2z/a)`.
Measured consequence: the `tanh_half` row is bit-identical to the sigmoid everywhere (same
crossover 473.02, same alignment ratio −0.5908, gradient cosine **1.000000**), and
`tanh(x/10)` reproduces the sigmoid's crossover scaled by exactly 5 (2364.4 = 5 × 473.0).

> **This corrects the reading of Q01's four-shape table.** Its "slower-decaying tanh"
> (`tanh(x/10)`) is not a different shape — it is the sigmoid at `beta/5`, i.e. a move along
> the *already-swept* `beta*std` axis. The tail axis that row appeared to test was never
> varied. (That table also had **no committed artifact**; it is re-measured here and its
> ranking rows reproduce — every shape gives `imbalance > rocket > best` at std 0.01 and
> `rocket > best > imbalance` at std 141, except the two narrow-core shapes below.)

### 1. Every shape aligns at ~200× its own transition width

Define `width(g)` = the `z` at which `g` reaches 0.9, and take the crossover as the `beta*std`
at which `F_g(best)` overtakes `F_g(rocket)`:

| shape | width | crossover `beta*std` | ratio |
|---|---|---|---|
| sigmoid | 2.1973 | 473.02 | 215.3 |
| tanh(z/2) *(= sigmoid)* | 2.1973 | 473.02 | 215.3 |
| tanh(z/10) | 10.9865 | 2364.37 | 215.2 |
| cauchy / arctan | 3.0778 | 654.43 | 212.6 |
| poly `z^-1` | 4.0000 | 744.06 | 186.0 |
| poly `z^-2` | 1.2361 | 234.16 | 189.4 |
| **poly `z^-4`** | **0.4954** | **95.05** | 191.9 |
| **sigmoid @ matched width** | **0.4953** | **105.92** | 213.8 |
| H11 clamp (M=5) | 4.0000 | 931.93 | 233.0 |

The ratio is **179–233 across eleven shapes** — i.e. to first order "change the shape" *is*
"change `beta`", which is Q01's law and the killed A-SCALE/H03 axis. Only ~±13% is shape.

### 2. The new degree of freedom is what a narrow core COSTS

Rocket's operating point is fixed at `beta*std ≈ 148`, so a shape aligns there iff
`width(g) ≲ 0.74`. Two shapes qualify, and they pay very differently. Alignment ratio
`A = (F_g(best) − F_g(rocket)) / (ceiling(best) − ceiling(rocket))` — the fraction of the true
+296.02 advantage the surrogate actually sees — measured at `beta*std = 148`, and the
zero-gradient node fraction measured at Rocket's **real converged positions** (std 141.04):

| shape | width | **A @ 148** | nodes with `grad == 0` | cos vs sigmoid |
|---|---|---|---|---|
| sigmoid | 2.1973 | **−0.5908** | 16.1% | 1.000000 |
| sigmoid @ matched width | 0.4953 | **+0.1315** | **42.5%** | 0.335 |
| **poly `z^-4`** | 0.4954 | **+0.1725** | **0.006%** | 0.293 |
| tanh(z/10) | 10.9865 | −1.9048 | 0.009% | −0.241 |
| poly `z^-1` | 4.0000 | −0.7949 | 0.006% | 0.016 |
| H11 clamp (M=5) | 4.0000 | **−1.0630** | **65.8%** | −0.485 |

So the exponential tail can only buy alignment by freezing **42.5%** of the nodes, while the
algebraic tail buys *more* alignment (+0.173) with **0.006%** frozen — a 7,000× difference at
identical core width. That gap is the one thing no rescaling of `beta` can reproduce. The
gradient cosines (0.29–0.34, vs 1.000000 for tanh) confirm these are dynamically distinct
directions, not reparametrizations.

**It also explains the logged H11 kill mechanistically:** a hard flat top/bottom is the worst
cell in the table — worst alignment (−1.063) *and* the most frozen nodes (65.8%). "Constant
outside a band" removes exactly the long-range coupling the objective needs.

### 3. …and the alignment is worthless: it ANTI-correlates with achieved score

The static argument predicts that the two aligned shapes should optimize best. They optimize
**worst**. Prototype gate, 3 seeds, everything else identical to baseline
(`experiments/outputs/proto_h37_tails.json`), Δ vs the sigmoid in pp:

| arm | A @ 148 | Δ mouse | Δ hard synthetic |
|---|---|---|---|
| sigmoid | −0.591 | 0 (92.3455) | 0 (73.6832) |
| sigmoid @ matched width | **+0.132** | +0.004 | **−2.117** |
| **poly `z^-4`** (native, aligned) | **+0.173** | **−0.220** | **−2.834** |
| poly `z^-4` @ sigmoid's width | — | −0.018 | −0.459 |
| poly `z^-1` | −0.795 | −0.130 | −0.708 |

**The two shapes whose surrogate ranks the better order higher are the two worst optimizers**
— on the fixture purpose-built to carry an optimization gap, by −2.1 and −2.8 pp. Confirmed
directly on the connectome through the frozen runner (`H37` / `H37B`, see `log.md`).

### Insights & how to use

1. **Static surrogate alignment is not a proxy for reachable score — here it is an inverted
   one.** Finding #3 diagnosed the surrogate as misaligned at the operating scale; Q04 shows
   that *fixing* the misalignment (the only shape that does) makes the result strictly worse.
   The binding constraint is the optimizability of the landscape, not its ranking fidelity.
2. **Mechanism: alignment and interaction range are the same knob pulled in opposite
   directions.** Aligning requires a narrow core, a narrow core is a short-range interaction,
   and a short-range interaction cannot perform the long-range reordering the gap consists of
   (`findings.md` #3: flip rank-distance p50 ≈ 22,580). The polynomial tail restores a
   *non-zero* long-range force but a very weak one — `g'` falls 10¹⁰ from `z=0` to `z=100` —
   so it does not compensate. Corroborating signature: the narrow-core arms converge to a
   **smaller** position spread (mouse final std 15.9/16.9 vs the sigmoid's 25.1).
3. **The continuous family is now closed on its last untested axis.** H11 killed the core
   shape, H34 the gradient estimator, H03/A-SCALE the scale, and Q04+H37 the tail exponent —
   the one axis whose prior negative evidence was an artifact of a mis-specified arm.
4. **The sigmoid is not merely adequate, it is near-optimal for its job.** Its width 2.20 sits
   where the interaction is long-ranged enough to reorder yet sharp enough to discriminate;
   both directions away from it lose.

**Reproduce** (env `allen`, from the repo root):
```
PYTHONPATH=src python experiments/diagnostics/q04_surrogate_tails.py   # ~4 min, theory gate
PYTHONPATH=src python experiments/proto_h37_tails.py                   # ~8 min, prototype gate
```
Artifacts: `experiments/outputs/q04_surrogate_tails.json`,
`experiments/outputs/proto_h37_tails.json`.
