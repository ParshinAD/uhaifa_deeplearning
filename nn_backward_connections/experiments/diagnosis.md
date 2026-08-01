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

**best's order out-surrogates Rocket's solution at *every* β.** If the surrogate were misaligned,
the higher-discrete best ordering would score *lower* on the surrogate; it scores **higher**. So the
sigmoid surrogate correctly ranks best above Rocket's solution — **the surrogate is not the
bottleneck; the optimizer is.** This holds across the whole β schedule (no β-dependent flip).

**Drift probe (corroborating, diagnostic-only).** Initialising Rocket *at* the 84.61% ordering and
running its optimization:

| schedule from best | final % | min along trajectory | drop |
|---|---|---|---|
| cyclic β (baseline), lr .05 | 82.94 | 78.01 | −1.67 |
| constant β=1.05, lr .05 | 83.03 | 78.87 | −1.58 |
| constant β=1.05, lr .005 | 82.75 | 76.60 | −1.86 |

Under every schedule tried **at even spacing (std≈0.58)**, gradient ascent on the surrogate flows
the near-optimal solution back down to Rocket's ~82.9% plateau (dipping into the 76–79% range first).
At the time this read as: best is *not a reachable or even holdable attractor* of Adam-on-sigmoid.

> **⚠ CORRECTED by Q01 (2026-08-01).** The "not **holdable**" half was an **artefact of the even
> spacing** (std≈0.58 — ~200× below the operating scale). From the best order's surrogate-optimal
> spacing P\* (std≈53,626) the score **HOLDS exactly** under small- and Rocket-default lr. Best IS a
> stable attractor at its own scale; only **reachability** fails (Adam-on-σ never navigates to P\*
> from a generic start). See § Q01 below. The gap is still a real optimization-landscape problem —
> just a **reachability**, not a **stability**, one.

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

Evidence: (1) **OPTIMIZATION-GAP** confirmed — best out-surrogates Rocket at all β; surrogate
aligned. (2) DIRECTION I has a **flat init→plateau** (≤0.06 pp) → down-ranked. (3) **Best is not
*reachable* by free gradient flow from a generic start** (it IS a holdable attractor at its own
scale — corrected by § Q01; the earlier "not holdable" was an even-spacing artefact). (4) Kendall-τ
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
start (⚠ corrected — it IS retained at its own scale; the "collapse" was an even-spacing artefact,
§ Q01 — so the barrier is reachability, not stability), and the gap is a distributed reordering
rather than recoverable ties, a large share of the 1.69 pp may be
**irreducible to continuous optimization** and genuinely require discrete refinement (the paper's
Crane phase). Stage B will quantify how much O/R recovers; the prototype-first gate on a *hard*
synthetic + mouse decides whether any variant earns connectome compute.

**Reproduce:** `python experiments/diagnostics.py --steps 0,1,2,3,4` → `experiments/outputs/diagnosis.json`.

---

## Q01 — Why does starting Rocket from the best solution drift the score DOWN?

**Answer (2026-08-01): it does NOT drift when you start from the true low-loss point. The logged
"collapse" was a SCALE artefact of the probe, not a property of the optimum.** This corrects the
over-strong half of finding #3 ("best is unreachable/unholdable … under *every* schedule/scale").

**The subtlety.** The surrogate F(pos) = Σ σ(β·Δ)·ŵ depends on position **gaps**, not on the order
alone. So the best *order* embedded at an arbitrary (even) spacing is **not** a low-loss point — the
gaps are wrong even though the ranking is optimal. Gradient descent correctly flees that high-loss
point, and in doing so it **reshuffles the order**, which is what drops the discrete score. The
original drift probe (`diagnosis.json`) started from the best order at **even spacing in [−1,1]
(std≈0.58)** — ~90,000× below the surrogate-optimal scale — so it measured the flight from a
high-loss point, not the stability of the optimum.

**Evidence** (`experiments/diagnostics/q01_drift_from_optimum.py` → `experiments/outputs/q01_drift.json`
+ `q01_hold_from_optimum.png`, `q01_scale_sweep.png`; connectome, β=1.05, seed 42):

| quantity | value | reading |
|---|---|---|
| best_solution discrete | 84.6147% | the target |
| **P\*** = best order at surrogate-optimal spacing | F=14,745.8, **std≈53,626**, disc 84.6147% | the true low-loss point (large scale) |
| Rocket reference | F=14,438.5, std≈141, disc 82.9161% | Rocket's basin is **lower-F** |
| **F(P\*) − F(Rocket)** | **+307.3** | GD *ascends* F ⇒ from P\* it cannot flow to Rocket |
| **‖∇F(P\*)‖** | **4.9e-4** (max 2.1e-4) | P\* is a **critical point** (local max of the surrogate) |
| **HOLD from P\***, Adam lr ∈ {5e-4, 5e-3, **5e-2**} | 84.6147% → **84.6147%** (min 84.6146%) | holds exactly — even at Rocket's default lr |

**The drift is entirely a function of position SCALE** (same optimal *shape*, rescaled; drop after
1000 Adam steps at lr 5e-3):

| std(pos) | 0.58 | 2 | 10 | 50 | **141** | 500 | 5000 | 53000 |
|---|---|---|---|---|---|---|---|---|
| ‖∇F‖ @start | 19.9 | 15.6 | 10.2 | 5.2 | 3.5 | 1.7 | 0.41 | 5e-4 |
| discrete **drop** (pp) | **5.51** | 4.32 | 2.35 | 0.97 | **0.47** | 0.18 | 0.009 | **0.00** |

At the logged even-spacing scale (std≈0.58) the drop is 5.5 pp; at Rocket's operating scale
(std≈141) it is 0.47 pp; at the true optimal scale (std≈53k) it is 0.00 pp. So:

- **The intuition is correct.** From the genuine loss-minimizing configuration P\*, small-lr — and
  even Rocket-default-lr — GD **holds** the best score; there is no lower-F basin to fall into.
- **The logged collapse was methodological**, not physical: it started ~200× (in std) too small,
  where the best order is a *high-loss* point that GD correctly leaves (reshuffling the order).
- **What survives from #3 (unchanged):** Adam-on-σ does not *navigate to* P\* from a cold or
  Rocket-scale start (the init→plateau curve is flat; Rocket converges to its own std≈141, lower-F
  basin). The barrier is **reachability**, not **stability** — the optimum is a stable attractor at
  its own scale; the optimizer just never gets there on its own.

**Why (the deeper cause): the surrogate is SCALE-BLIND and prefers the best order only above a
large scale.** F depends on gaps, so the *same* best order has a different surrogate value at
different scales — and the best order out-surrogates Rocket **only above std≈459** (`q01_F_vs_scale.png`):

| std(pos) | 0.29 (≈[0,1]) | 0.58 | 50 | **141 (Rocket)** | **459 (crossover)** | 5000 | 53626 (P\*) |
|---|---|---|---|---|---|---|---|
| F(best order) | 9,224 | 9,669 | 13,921 | **14,273** | ≈14,439 | 14,689 | 14,746 |
| vs F(Rocket)=14,439 | −5,215 | −4,770 | −518 | **−165** | 0 | +250 | +307 |

(discrete score of the best order = 84.61% at every scale — the order never changes; only F moves.)

- At std→0 every Δ→0 ⇒ σ→0.5 for all edges ⇒ F→0.5·Σŵ = **8,714**: the surrogate goes fully blind.
- **At Rocket's own operating scale (std≈141) the best order has LOWER F than Rocket** (14,273 <
  14,438). So a non-ideal-but-well-spread order (Rocket) genuinely beats the ideal-but-squished order
  *in the surrogate's own currency*. The best order only wins once positions spread past std≈459 — a
  regime Rocket never enters (its positions converge to std≈141).
- This is the mechanistic core of the plateau: **at the achievable scale the continuous relaxation
  does not even rank the better order higher**, so no local gradient step points toward it.

### Insights & how to use
1. **The plateau is a surrogate-SCALE property, not an optimizer failure.** At std≈141 Rocket's own
   order is (locally) surrogate-optimal; the better order isn't preferred there. This *sharpens* #3:
   the gap is not "the optimizer is too weak" but "the relaxation, at its natural scale, points the
   wrong way." Reconciles with the scale-fair alignment (each order at its *own* optimal spacing ⇒
   best wins by +307): alignment holds only when the best order is granted a large scale.
2. **This is exactly why discrete refinement (H30/H35) works and continuous levers don't.** The sift
   operates in rank space and is immune to the surrogate's scale-blindness — it finds the reordering
   the surrogate cannot "see" at std≈141, recovering ~half the gap with 0 gradient steps (#4/#5). Q01
   is the mechanistic "why" behind that. Conversely it predicts the continuous Track-A levers are
   dead ends: alt-surrogate (A-SURR / killed H11/H34) reshapes σ but not the operating scale;
   tight-init (A-INIT) starts *smaller*, deeper in the blind regime.
3. **The one continuous lever the crossover suggests — and why it's still low-EV.** Push the optimizer
   to operate above std≈459 (scale/temperature annealing, or growing β·std) so the surrogate becomes
   discriminative. But (a) Rocket's positions already free-scale and *choose* std≈141 because that is
   where ITS order is optimal — growing scale alone won't relocate it to a *different* order; (b) above
   the crossover σ saturates on correct edges ⇒ vanishing gradient (the H19 soft-rank stall and the H03
   β-schedule kill are this failure mode). Logged as roadmap **A-SCALE** (low-EV, prior evidence
   attached). Net: the discrete sift already achieves what a scale lever gropes toward.

**Reproduce:** `PYTHONPATH=src python experiments/diagnostics/q01_drift_from_optimum.py`
(env `allen`, ~3–6 min). Artifacts: `q01_drift.json`, `q01_hold_from_optimum.png`,
`q01_scale_sweep.png`, `q01_F_vs_scale.png`. Supersedes the exploratory
`dr_tmp/drift_from_optimal_spacing.py` and `dr_tmp/drift_scale_sweep.py`.
