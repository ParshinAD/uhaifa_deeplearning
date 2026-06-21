# Backlog — open hypotheses for improving Rocket

Each hypothesis, when picked up, must be run through the **unchanged** harness on **both**
datasets across **≥3 seeds**, with results logged to `results/*.json` and a ranked conclusion
recorded in `findings.md`. A change counts as a win only if it beats the baseline beyond seed
noise on the exact feedforward metric.

---

## Re-prioritization (2026-06-21, post-H04)

**Evidence base (4 cycles, all logged):** H02 (warm-start from greedy-FAS ordering) is the only
CONFIRMED win — connectome +0.0448 pp, mouse +0.2064 pp at equal compute. H01 (multi-start),
H03 (sharper terminal β), H04 (in-loop barycenter refinement) all KILLED.

**Central inference (the lever that matters):** interventions on the *late optimization
trajectory / dynamics* did NOT move the metric, because Rocket robustly re-converges to its
plateau (~82.9% / ~92.1%) regardless of the schedule it took to get there. H01 (restarts under
the same dynamics), H03 (β tail) and H04 (in-loop nudges) all landed back on the plateau within
noise. The single lever that helped — H02 — changed **where optimization starts** (the basin),
not how it moves. **Pressure-test, do not blindly accept:** this is a 1-positive / 3-negative
inference over only 4 cycles; it could instead be that the three killed knobs were each
*individually too weak* (β tail too short, barycenter too weak, restarts under-trained) rather
than that "dynamics never matters." The re-ranking below treats dynamics-only knobs as low-EV
**but keeps two of them alive as cheap falsifiers**, and explicitly favours the three remaining
non-dynamics levers (objective/loss landscape, free-edge recovery, exploration noise on the
loss itself) plus levers that compound with the H02 basin.

**Three EV buckets:**
- **LOW-EV (likely plateau re-converge)** — pure late-dynamics knobs that, like H01/H03/H04, only
  change the path to the same basin: **H05** (optimizer swap), **H07** (LR schedule), **H08** (β+LR
  joint), **H10** (grad-clip), **H12** (EMA averaging). These should be screened cheaply or
  deprioritized; expect plateau re-convergence.
- **HIGHER-EV (basin / objective / free-edge)** — change the loss landscape or recover discrete
  edges the surrogate leaves on the table, i.e. mechanisms *distinct* from the ones that already
  re-converged: **H06** (weight-aware loss reweighting — reshapes the objective), **H09**
  (tie-break jitter — recovers strict-`>` near-tie edges for ~free), **H11** (margin/hinge
  surrogate — keeps gradient on correct-but-thin edges, a genuine landscape change), **H13**
  (stochastic edge subsampling — injects exploration noise into the *loss*, not the schedule).
- **COMPOUNDS WITH H02 (new SOTA)** — **H14, H15** below, which start from the H02 basin and add a
  non-dynamics lever, so their comparator is **H02 at matched seeds**, not the random-init baseline.

**Re-ranked OPEN order (highest EV/cost first), with one-line justification:**

| rank | id | bucket | one-line EV justification |
|---|---|---|---|
| 1 | **H09** | free-edge | Cheapest non-dynamics lever; the strict-`>` oracle silently drops near-tie edges, so anti-tie jitter recovers metric for free *without* changing the basin — orthogonal to everything killed. Size the opportunity (fraction of `|Δ|<ε` edges) first; if ~0 it self-falsifies fast. |
| 2 | **H14** (new) | H02-basin × free-edge | Stack the cheapest free-edge lever (jitter) **on the confirmed H02 win** to push past the *new* SOTA at near-zero cost; comparator = H02@matched-seeds. |
| 3 | **H06** | objective | Reshapes *what* the loss optimizes (heavy/borderline-edge emphasis) — a landscape change, not a path change; aligns gradient effort with the metric's own weighting. Distinct mechanism from all 3 kills. |
| 4 | **H11** | objective/landscape | Margin/hinge surrogate keeps gradient on correct-but-thin-margin edges that σ_β abandons — directly targets *why* Rocket plateaus (saturated gradients), a real landscape change, still cheap. |
| 5 | **H15** (new) | H02-basin × landscape | H02 warm-start + best of {H06,H11} objective lever (whichever screens better) — compounding two non-dynamics mechanisms; comparator = H02@matched-seeds. |
| 6 | **H13** | exploration-noise | Stochastic edge subsampling injects noise into the *loss itself* (unlike H01's naive restarts under the same full-batch dynamics) — a genuinely different escape mechanism, cheap; honest medium uncertainty. |
| 7 | **H05** | LOW (dynamics) | Optimizer swap (AdamW/Lion) — cheap to falsify but expected to re-converge to plateau like H01/H03/H04; keep as a fast falsifier of the "dynamics never matters" inference. |
| 8 | **H10** | LOW (dynamics) | Grad-clip/per-node clip — plausible hub argument but it is still a path-not-basin change; cheap, share machinery with H05, expect plateau. |
| 9 | **H12** | LOW (dynamics) | EMA/Polyak averaging — near-free extra candidate, but only smooths the *same* trajectory; high chance within noise. |
| 10 | **H07** | LOW (dynamics) | LR schedule swap — pure path change; H01/H03 evidence says the basin, not the schedule, sets the plateau. Deprioritize. |
| 11 | **H08** | LOW (dynamics) | β+LR joint annealing — subsumes H03 (killed) + H07 (low-EV); lowest EV of the dynamics knobs. Run last, only if H07 surprises. |

**Recommended NEXT:** **H09** — it is the cheapest open lever, tests a mechanism (strict-`>`
near-tie loss) that none of the 4 prior cycles touched, and self-falsifies immediately if the
`|Δ|<ε` edge fraction is ~0. If H09 shows any signal, **H14** (H09-on-H02) is the immediate
follow-up to chase the new SOTA. H06 is the strongest objective-axis bet behind it.

---

## How to read this backlog

Ranked by **expected value / cost**. Each variant is an isolated module
`src/mfas/experiments/<id>.py` exposing `run(g, seed, device, time_limit=None) -> RocketResult`.
It may change only the *algorithm* (init, loss/surrogate, β schedule, optimizer, LR, gradient
handling, restarts, refinement) — never the scorer or harness.

**Noise floor (from PROTOCOL.md):** connectome σ≈0.019 pp (2σ screen ≈ **0.04 pp**),
mouse σ≈0.262 pp (2σ screen ≈ **0.52 pp**). Mouse is ~14× noisier. Baselines: connectome
**82.896%**, mouse **92.070%**. "Effect vs noise" magnitudes below are honest reasoned
estimates, **not** measured numbers.

**Anti-leakage rule applied to every entry:** the discrete oracle score may be used exactly as
the baseline already uses it (track the best discrete-scoring positions seen during the run); it
must **never** be folded into the differentiable loss, hardcoded, or used to special-case a
dataset. Refinement-style variants reorder via positions/permutations and let the frozen oracle
score the result — they do not optimize against a known target value.

**Measurement (all entries, unless noted):** exact feedforward % via the frozen oracle, on
**both** datasets, **≥3 seeds** (42/123/999) for SCREEN; promote on `Δmean > 2σ` on both;
CONFIRM with 5 (connectome) / 20–30 (mouse) seeds and a 95% CI lower bound > 0 on both.

---

## H01 — Multi-start restarts with best-of selection
- **Hypothesis:** Running K independent short Rocket optimizations from different random inits and
  keeping the best discrete-scoring result beats one long run at equal total compute.
- **Rationale:** Seed-stability analysis (CLAUDE.md) shows scores are tight (~82.9%) but
  orderings only correlate Spearman ≈ 0.96 and sources are unstable (Jaccard@1000 ≈ 0.33–0.40)
  — i.e. runs land in *different* basins. Best-of-K over a high-variance ordering is the cheapest
  way to harvest the right tail. The baseline already keeps best-by-oracle, so this is leakage-safe.
- **Design axis:** restarts / multi-start.
- **Expected effect:** connectome +0.02–0.08 pp (near/just above noise); mouse +0.3–0.8 pp
  (mouse's larger σ means more tail to harvest). Most promising on mouse.
- **Est. compute cost:** **cheap–medium.** Equal total epoch budget split into K=4–6 runs; no
  per-step overhead. Slightly more oracle scoring calls (one extra per restart).
- **Measurement:** standard. Compare K∈{4,8} at fixed total epochs vs 1× baseline epochs.
- **status: killed**  <!-- 2026-06-21 screen FAIL on both datasets → KILL. Δ vs equal-budget
  comparator baseline_multistart(K=4) = +0.0003 pp (connectome) / +0.0000 pp (mouse), both far
  below 2σ. H01's restart scheme is algorithmically identical to naive restarts (same sub-seeds),
  and splitting budget into 4×(total/K) under-trains each restart (connectome −0.84 pp vs single
  long run). Hypothesis falsified. See experiments/log.md 2026-06-21 H01 cycle. -->


## H02 — Warm-start from a topological / greedy ordering (TopoShuffle-style init)
- **Hypothesis:** Initializing positions from a degree/greedy-based DAG ordering (Kahn-style
  topological sort of a high-weight acyclic subgraph) instead of N(0,1) gives Rocket a better
  basin and a higher final feedforward weight.
- **Rationale:** The paper's *Crane* phase explicitly uses TopoShuffle (Kahn 1962) init because a
  topological order guarantees Γ(TSP) ≥ Γ(P) (paper §3.2.1); greedy MFAS heuristics already reach
  68–72% (CLAUDE.md). Seeding the continuous optimizer near a good discrete solution is a classic
  continuation trick and is purely a starting point (no target leakage). The existing
  `degree_diff`/`degree_abs` inits are weak proxies for this.
- **Design axis:** initialization.
- **Expected effect:** connectome +0.03–0.15 pp; mouse +0.3–1.0 pp. Upside if the random basin is
  suboptimal; risk it just converges to the same plateau.
- **Est. compute cost:** **cheap.** One-time greedy/topo sort (O(m log n)) before optimization;
  uses installed `networkx`/`scipy`. Per-step cost unchanged.
- **Measurement:** standard; ablate greedy vs topo-of-greedy as the init source.
- **status: confirmed**  <!-- 2026-06-21 CONFIRMED WIN (first of the campaign). Screen marginally
  missed the 2σ_baseline gate, but that gate assumes variant variance ≈ baseline noise; H02's init
  is ~deterministic (σ≈0.0005/0.0000), so it was escalated to the CONFIRM test (the real bar) and
  PASSED on both datasets. Init = leakage-safe Eades–Lin–Smyth/GreedyAbs greedy-FAS ordering (graph
  structure+weights only) → evenly-spaced positions in [-1,1] → UNCHANGED run_rocket at baseline
  budget (20k/5k). CONFIRM (matched-seed baseline, conservative): connectome Δ=+0.0448 pp, 95% CI
  lower +0.0135 (n=5); mouse Δ=+0.2064 pp, 95% CI lower +0.0824 (n=20). Critic KEEP (all 6 risks
  PASS). Caveat: connectome margin thin (+0.0135). See experiments/log.md + findings.md #1. -->
  

## H03 — Sharper / extended β schedule (raise terminal sharpness)
- **Hypothesis:** Increasing the maximum sigmoid sharpness in the late phase (β_max from ~1.05 up
  to e.g. 3–8, or appending a final monotone-sharpening ramp) tightens the surrogate→discrete gap
  and yields higher exact feedforward weight.
- **Rationale:** The surrogate σ_β only approximates the discrete indicator; at β≈1 a unit position
  gap maps to σ≈0.73, so many "weakly correct" edges contribute little gradient and near-ties are
  fuzzy. The paper flags β as *the* sensitive hyperparameter (§4.6) and uses a cyclic schedule
  purely to dodge manual tuning — it never claims [0.05,1.05] is optimal. Annealing β upward at the
  end is standard sigmoid-annealing / deterministic-annealing practice. No leakage (β is a loss
  shape parameter).
- **Design axis:** β schedule / surrogate.
- **Expected effect:** connectome ±0.05 pp (could help or hurt — too-large β kills gradients);
  mouse ±0.5 pp. Needs a small sweep; medium confidence.
- **Est. compute cost:** **cheap.** Schedule array change only; per-step cost identical.
- **Measurement:** standard; sweep β_max ∈ {1.05(base), 2, 4, 8} as the screened factor.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL → KILL (correctly-specified gate: H03's own std ≈
  baseline noise floor, so NOT an H02-style escalation case). Primary arm = baseline cyclic
  exploration for first 75% then a final monotone linear β ramp to β_max=4 over the last 25%
  (leakage-safe loss-shape change; standard knob-swap, equal budget 20k/5k). connectome Δ=−0.0376 pp
  (REGRESSION, mean 82.8582 ± 0.0181, n=3); mouse Δ=+0.0114 pp (mean 92.0810 ± 0.2730, n=3) — both far
  below the 2σ gate. NOT the H02 low-variance case: H03's own std ~= baseline noise floor on both
  datasets, so the gate is correctly specified — no CONFIRM escalation. β_max ∈ {2,8} remain un-run
  sweep arms but the primary arm is not promising. See experiments/log.md 2026-06-21 H03 cycle. -->


## H04 — In-the-loop discrete refinement (continuous + periodic local swaps)
- **Hypothesis:** Periodically nudging positions toward a locally-improved ordering (e.g. greedy
  adjacent-pair / sink-source moves on the current order, then re-seed positions from the improved
  ranks) lets Rocket escape the surrogate plateau and raises exact feedforward weight.
- **Rationale:** Crane's whole premise is that local/MIP refinement extends quality *beyond
  Rocket's plateau* (paper §4.4, 82.87%→84.60%). A cheap, in-variant local search on the *current*
  ordering (no Gurobi) imports that idea. Reorder-then-reproject is leakage-safe: moves are chosen
  by local edge-weight comparisons, and the oracle only scores the outcome.
- **Design axis:** post-processing-as-refinement-within-variant / hybrid discrete+continuous.
- **Expected effect:** connectome +0.05–0.3 pp; mouse +0.5–2.0 pp. Highest ceiling here but also
  the most implementation risk (must stay cheap and not collapse positions).
- **Est. compute cost:** **medium–expensive.** Local search every N epochs adds O(m) per refine;
  must cap refine frequency. Risk of large wall-clock on 5.6M-edge connectome.
- **Measurement:** standard; report *pre-refinement Rocket score* separately (CLAUDE.md rule).
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL. Move =
  cheap leakage-safe weighted-barycenter rank reposition (input edge weights + current ranks only;
  oracle used only for best-by-oracle adoption, so post-refinement ≥ pure Rocket by construction),
  ≤8 O(m) passes in the second half. Compute-matched comparator = baseline_passthrough at equal
  total_grad_steps (20k/5k); added non-gradient cost = 8 passes/run (~0.002 s mouse, ~3.0 s ≈3.8%
  connectome). 0 refine candidates accepted across all 6 runs (barycenter too weak vs Rocket's
  plateau) → post == pure on every run. connectome 82.8958 ± 0.0183, Δ = −0.0000 pp; mouse
  92.0696 ± 0.2624, Δ = +0.0000 pp — both ≈ zero, far below the 2σ gate. H04 own std ≈ baseline
  noise floor on both → correctly-specified screen, no CONFIRM escalation. Leakage-safe, no
  collapse/NaN. See experiments/log.md 2026-06-21 H04 cycle. -->.

## H05 — Optimizer swap: AdamW / decoupled weight decay or Lion vs Adam
- **Hypothesis:** Replacing Adam with AdamW (small decoupled decay to keep positions bounded) or a
  sign-based optimizer (Lion) changes the basin reached and improves final feedforward weight.
- **Rationale:** Positions are unconstrained and can drift to large magnitudes, where σ saturates
  and gradients vanish; mild weight decay regularizes scale (a scale-invariance argument the paper
  itself makes for *weight* normalization, §3.1.1). Sign-based updates (Lion) are robust to the
  heavy-tailed per-node gradient magnitudes that arise from the heavy-tailed degree distribution
  (paper Fig. 1). Optimizer choice is leakage-safe.
- **Design axis:** optimizer.
- **Expected effect:** connectome ±0.03–0.08 pp; mouse ±0.3–0.6 pp. Plausibly neutral; cheap to
  falsify.
- **Est. compute cost:** **cheap.** Drop-in optimizer change (AdamW in torch; Lion if available
  else hand-coded sign update). Per-step cost ≈ identical.
- **Measurement:** standard; AdamW (decay∈{1e-4,1e-2}) and Lion as two screened arms.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL (deliberate falsifier
  cycle; clean negative). Primary arm = AdamW with DECOUPLED weight_decay=1e-4 (small position-scale
  prior; AdamW (β1,β2,eps) defaults == Adam's, so weight_decay is the ONLY behavioural difference;
  wd=0 reduces exactly to baseline). run_rocket loop replicated VERBATIM, ONLY the optimizer
  constructor optim.Adam → optim.AdamW(weight_decay=1e-4) changed; N(0,1) init / grad-clip=1.0 /
  ConstantLR→ExponentialLR / cyclic-β / budget 20k/5k IDENTICAL to baseline. Leakage-safe: decoupled
  decay reads only the positions, never the oracle/input weights, not dataset-special-cased.
  connectome Δ=+0.0018 pp (mean 82.8976 ± 0.0205, n=3); mouse Δ=+0.0000 pp (mean 92.0696 ± 0.2624,
  n=3, bit-for-bit the baseline mean±std) — both far below the 2σ gate. H05 own std ≈ baseline noise
  floor on both (connectome 0.0205 vs 0.0189; mouse 0.2624 vs 0.2624) → correctly-specified screen,
  NOT an H02-style low-variance escalation. UN-RUN arms: AdamW weight_decay=1e-2, Lion (Lion not
  installed; no dependency added). Confirms the campaign's basin-not-dynamics inference: the optimizer
  — the dynamics knob most able to reach a different basin — did not move the exact metric. See
  experiments/log.md 2026-06-21 H05 cycle. -->


## H06 — Degree/weight-aware loss reweighting (focus gradient on high-weight edges)
- **Hypothesis:** Reweighting the surrogate so that high-weight and currently-borderline edges
  receive proportionally more gradient (vs flat max-normalized ŵ) increases retained high-weight
  feedforward arcs.
- **Rationale:** Both connectomes have extremely skewed edge weights (paper Fig. 2; connectome max
  2,405 vs mostly tiny). Max-normalization keeps the objective faithful but means the bulk of edges
  contribute near-equal small gradients while the few decisive heavy edges are diluted late in
  training. Emphasizing heavy/uncertain edges aligns gradient effort with the metric's own
  weighting — without ever reading the discrete score (weights are part of the input graph, not the
  target). Must preserve the objective's *direction* (still maximizing feedforward weight).
- **Design axis:** loss / surrogate.
- **Expected effect:** connectome +0.02–0.10 pp; mouse +0.2–0.8 pp. Risk: changing the loss too far
  from Eq. 7 biases the surrogate away from the true metric.
- **Est. compute cost:** **cheap.** Elementwise reweighting of existing terms; per-step cost ≈ same.
- **Measurement:** standard; ablate (a) heavy-edge emphasis, (b) borderline (|Δ| small) emphasis.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL. Primary arm =
  detached, strictly-positive, bounded per-edge emphasis m_e = 1 + ALPHA·hat_w·(4·σ(1−σ)) with
  ALPHA=4 (a clean combination of (a) heavy hat_w and (b) borderline |σ−0.5| emphasis), applied to
  the baseline σ_β·hat_w term; everything else (init/Adam/clip/LR/β/budget 20k/5k) IDENTICAL to
  baseline (run_rocket loop replicated verbatim, only two lines changed). Direction-preserving:
  b_e detached ⇒ per-edge gradient = baseline·m_e with m_e≥1>0, so no edge's pull is inverted or
  silenced, only relative magnitudes reshaped; still a monotone feedforward reward → argmax stays
  "maximize feedforward weight". Leakage-safe: m_e uses only input weights (hat_w) + the model's own
  surrogate state σ_β, never the discrete oracle, never dataset-special-cased. connectome Δ=−0.0380
  pp (mean 82.8578 ± 0.0288, n=3, REGRESSION); mouse Δ=−0.0794 pp (mean 91.9902 ± 0.1978, n=3,
  REGRESSION) — both below the 2σ gate. Both Δ negative + H06 own std ≈ baseline noise floor on both
  → correctly-specified screen, NOT an H02-style low-variance escalation. The emphasis biased the
  surrogate slightly off the faithful Eq.-7 weighting (the listed risk). UN-RUN arms: heavy-only
  power (hat_w)^(γ−1), borderline-only 1+ALPHA·b_e, ALPHA sweep. See experiments/log.md 2026-06-21
  H06 cycle. -->
  <!-- Knock-on: H15 (H02 basin × best objective lever) is gated on an objective lever screening
  positive; H06 did not. If H11 also fails, H15 reduces to H02 and should be dropped. -->

## H07 — Cosine/one-cycle LR with warmup instead of constant→exponential
- **Hypothesis:** Replacing the ConstantLR(50%)→ExponentialLR(→10%) schedule with a warmup +
  cosine-decay (one-cycle) LR improves the final exact feedforward weight at equal epochs.
- **Rationale:** The current schedule holds LR=0.05 flat for the entire first half then decays —
  unusual and likely wasteful: early large steps with a *small* β (smooth loss) plus late decay
  with *cyclic* β can desync exploration and exploitation. Warmup→cosine is the modern default and
  co-phasing the LR trough with β's sharp phases is a principled annealing pairing. LR is
  leakage-safe.
- **Design axis:** LR schedule.
- **Expected effect:** connectome ±0.02–0.06 pp; mouse ±0.2–0.5 pp. Likely small; cheap to test.
- **Est. compute cost:** **cheap.** Scheduler swap only.
- **Measurement:** standard; one-cycle (warmup 5–10%, cosine to 1–10% of base) vs baseline.
- **status: deferred (campaign stop)**  <!-- 2026-06-21 NOT RUN. Campaign hit its stop criterion
  (EARLY_EXIT: 7 consecutive non-improving cycles after the H02 win, no remaining promising items).
  This is a LOW-EV pure-dynamics knob; the basin-not-dynamics evidence — 5 dynamics-axis kills
  H01/H03/H04/H13 plus the H05 optimizer falsifier (AdamW re-converged to the plateau) — predicts it
  re-converges too. Deferred, not falsified: re-open if the central inference is later overturned.
  See findings.md and the 2026-06-21 campaign-stop note in experiments/log.md. -->

## H08 — β-phase-synced LR + longer/asymmetric β cycles
- **Hypothesis:** Co-scheduling LR and β (high LR during smooth/low-β explore phases, low LR during
  sharp/high-β refine phases) and using fewer, longer cycles with a slow upward β drift beats the
  independent 5-cycle β + half-flat LR.
- **Rationale:** The paper adopts cyclic β to "explore broadly then refine promising regions"
  (§4.6) but the LR schedule is *independent* of β's phase, so refine phases can still take large
  steps that undo exploration gains. Explicitly aligning step size with the explore/refine cadence
  is a coherent annealing design and a natural extension of the paper's own stated intent. Both are
  leakage-safe loss/step parameters.
- **Design axis:** β schedule + LR (joint annealing).
- **Expected effect:** connectome ±0.03–0.10 pp; mouse ±0.3–0.7 pp. Medium uncertainty; subsumes
  parts of H03/H07 so run after them.
- **Est. compute cost:** **cheap.** Two coupled schedule arrays; per-step cost identical.
- **Measurement:** standard; sweep n_cycles∈{2,3,5} with phase-locked LR.
- **status: deferred (campaign stop)**  <!-- 2026-06-21 NOT RUN. Campaign hit its stop criterion
  (EARLY_EXIT: 7 consecutive non-improving cycles after the H02 win, no remaining promising items).
  This is a LOW-EV pure-dynamics knob; the basin-not-dynamics evidence — 5 dynamics-axis kills
  H01/H03/H04/H13 plus the H05 optimizer falsifier (AdamW re-converged to the plateau) — predicts it
  re-converges too. Deferred, not falsified: re-open if the central inference is later overturned.
  See findings.md and the 2026-06-21 campaign-stop note in experiments/log.md. -->

## H09 — Tie-breaking jitter on near-equal positions (anti-collapse)
- **Hypothesis:** Adding tiny deterministic per-node position jitter / discouraging exact position
  ties prevents the strict-`>` oracle from silently dropping near-tie edges, recovering a small
  amount of feedforward weight "for free."
- **Rationale:** The oracle counts `pos[tgt] > pos[src]` (strict): any near-tie that the surrogate
  treats as ~0.5 contributes nothing to the discrete score even when the intended order is correct.
  Encouraging position separation (small repulsion term or post-hoc symmetric jitter before
  scoring) targets exactly these lost edges. Jitter is data-independent and target-blind →
  leakage-safe. Cheap, isolated, low-risk.
- **Design axis:** gradient handling / position regularization.
- **Expected effect:** connectome +0.005–0.04 pp (likely near or below noise — falsifiable);
  mouse +0.05–0.4 pp (float weights → more genuine near-ties). Honest: may be a no-op.
- **Est. compute cost:** **cheap.** Optional repulsion term O(n) or scoring-time jitter only.
- **Measurement:** standard; measure fraction of edges at |Δ|<ε to size the opportunity first.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL, self-falsified as the
  opportunity sizing predicted. Sizing on logged baseline plateau positions: 0 exact ties on BOTH
  datasets; connectome recoverable near-tie feedback weight 0.000048 pp (|d|<1e-3) / 0.000322 pp
  (|d|<1e-2); mouse has 0 edges with |d|<1e-2 — entire recoverable set ~3 orders of magnitude below
  the screen thresholds. Arm = target-blind deterministic symmetric scoring-time jitter
  (1e-9·hash(node_index,seed), mean ~0) on UNCHANGED run_rocket output, oracle picks the better of
  {raw, jittered} whole-vector candidate (best-by-oracle; never per-edge orientation → no leakage),
  equal budget 20k/5k. connectome 82.8948 ± 0.0187, Δ=−0.0010 pp; mouse 92.0696 ± 0.2624 (bit-identical
  to baseline_passthrough), Δ=+0.0000 pp — both far below the 2σ gate. H09 own std ≈ baseline noise
  floor on both → correctly-specified screen, NOT an H02-style low-variance escalation. The continuous
  Adam optimizer leaves no ties for the strict-`>` oracle to drop. See experiments/log.md 2026-06-21
  H09 cycle. -->

## H10 — Per-node adaptive gradient / no-global-clip (clip-by-value or none)
- **Hypothesis:** The global grad-norm clip at 1.0 throttles updates for the heavy-tailed
  high-degree hub nodes; replacing it with per-element clip-by-value (or removing clipping under
  AdamW) lets hubs move enough to be ordered correctly and raises feedforward weight.
- **Rationale:** Degree distribution is heavy-tailed (paper Fig. 1): a few hubs accumulate huge
  gradients, so a *global* norm clip mostly suppresses the very nodes whose placement matters most
  for high-weight edges. Per-coordinate clipping (or Adam's own normalization with no extra clip)
  treats nodes independently. Gradient handling is leakage-safe.
- **Design axis:** gradient handling.
- **Expected effect:** connectome ±0.02–0.08 pp; mouse ±0.2–0.6 pp. Could backfire (instability);
  cheap to falsify.
- **Est. compute cost:** **cheap.** Replace `clip_grad_norm_` with clip-by-value / none.
- **Measurement:** standard; arms = {clip-value 0.1, no-clip, baseline norm-clip}.
- **status: deferred (campaign stop)**  <!-- 2026-06-21 NOT RUN. Campaign hit its stop criterion
  (EARLY_EXIT: 7 consecutive non-improving cycles after the H02 win, no remaining promising items).
  This is a LOW-EV pure-dynamics knob; the basin-not-dynamics evidence — 5 dynamics-axis kills
  H01/H03/H04/H13 plus the H05 optimizer falsifier (AdamW re-converged to the plateau) — predicts it
  re-converges too. Deferred, not falsified: re-open if the central inference is later overturned.
  See findings.md and the 2026-06-21 campaign-stop note in experiments/log.md. -->

## H11 — Surrogate swap: tanh / smooth-hinge / temperature-annealed softmax-rank
- **Hypothesis:** Replacing the sigmoid surrogate with a margin-shaped one (smooth hinge / tanh)
  that keeps producing gradient for already-correct-but-small-margin edges yields higher exact
  feedforward weight than σ_β.
- **Rationale:** σ_β saturates: once an edge is comfortably feedforward its gradient → 0, so the
  optimizer stops *widening margins* that protect the discrete order against later cyclic-β
  reshuffling. A hinge/margin loss keeps pushing margins past a threshold (max-margin intuition,
  and the paper notes Borst's Heaviside/Pauli terms were dropped for cost — a margin loss is a
  cheaper alternative in that same design space). Surrogate shape is leakage-safe as long as it
  monotonically rewards feedforward orientation.
- **Design axis:** loss / surrogate.
- **Expected effect:** connectome ±0.03–0.10 pp; mouse ±0.3–0.7 pp. Medium risk of drifting from
  the true objective; medium confidence.
- **Est. compute cost:** **cheap.** Loss function swap only.
- **Measurement:** standard; arms = {smooth-hinge, tanh} vs sigmoid baseline.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL. Primary arm =
  bounded smooth-hinge surrogate r_β(Δ)=clamp(0.5 + (β·Δ)/(2·MARGIN), 0, 1) with MARGIN=2.0,
  replacing σ_β(Δ); LINEAR (constant non-zero gradient) across the correct-but-thin band
  |β·Δ|≤MARGIN then flat/bounded outside (keeps gradient widening margins σ_β abandons; bounded
  plateau + grad-clip guard divergence). Per-edge weight hat_w UNCHANGED → isolates surrogate SHAPE
  from H06 reweighting. Everything else (N(0,1) init/Adam/clip=1.0/LR/β/budget 20k/5k) IDENTICAL
  to baseline (run_rocket loop replicated verbatim, ONE line changed). Monotone non-decreasing in Δ
  with β>0 ⇒ argmax preserved (still maximizes feedforward weight); leakage-safe (positions + input
  weights + β only, never the discrete oracle, no dataset special-case). connectome Δ=−0.0387 pp
  (mean 82.8571 ± 0.0235, n=3, REGRESSION); mouse Δ=+0.1264 pp (mean 92.1960 ± 0.2643, n=3, positive
  but ~4× under the 0.52 pp gate and within own seed noise) — both below the 2σ gate. Own std ≈
  baseline noise floor on both + connectome Δ negative → correctly-specified screen, NOT an H02-style
  low-variance escalation. UN-RUN arm: tanh (rejected — affine reparam of sigmoid, saturates
  identically, doesn't test the mechanism). Knock-on: H15 was gated on an objective lever
  (H06 OR H11) screening positive; BOTH have now failed → H15 reduces to H02 and should be dropped.
  See experiments/log.md 2026-06-21 H11 cycle. -->

## H12 — Polyak / EMA averaging of positions
- **Hypothesis:** Maintaining an exponential moving average of the position vector and scoring the
  EMA (in addition to the raw iterate) yields a smoother, higher-scoring ordering than the noisy
  last iterate, especially under the oscillating cyclic-β loss.
- **Rationale:** Cyclic β makes the loss landscape (and thus iterates) oscillate; the raw iterate
  at any step is noisy. Polyak/EMA averaging is a standard, near-free way to reduce that noise and
  often lands in flatter, better-generalizing regions. The baseline already best-tracks discrete
  scores, so also scoring the EMA iterate is leakage-safe and strictly adds a candidate.
- **Design axis:** momentum / averaging.
- **Expected effect:** connectome +0.01–0.05 pp; mouse +0.1–0.5 pp. Low cost, modest upside,
  honest chance of being within noise.
- **Est. compute cost:** **cheap.** One extra O(n) EMA update per step + a few extra oracle scores.
- **Measurement:** standard; sweep EMA decay ∈ {0.99, 0.999}.
- **status: deferred (campaign stop)**  <!-- 2026-06-21 NOT RUN. Campaign hit its stop criterion
  (EARLY_EXIT: 7 consecutive non-improving cycles after the H02 win, no remaining promising items).
  This is a LOW-EV pure-dynamics knob; the basin-not-dynamics evidence — 5 dynamics-axis kills
  H01/H03/H04/H13 plus the H05 optimizer falsifier (AdamW re-converged to the plateau) — predicts it
  re-converges too. Deferred, not falsified: re-open if the central inference is later overturned.
  See findings.md and the 2026-06-21 campaign-stop note in experiments/log.md. -->

## H13 — Mini-batch / stochastic edge subsampling per step
- **Hypothesis:** Computing the surrogate loss on a random subset of edges each step (SGD-style)
  injects useful gradient noise that escapes the surrogate plateau, matching or beating full-batch
  Rocket at equal or lower compute.
- **Rationale:** Rocket is currently full-batch (all 5.6M edges every step) — deterministic
  descent into the nearest basin, consistent with the observed plateau. Stochastic edge sampling is
  the textbook way to add exploration noise and is also a path to *more steps per second* on the
  large connectome. Sampling is target-blind (uniform/weight-proportional over input edges) →
  leakage-safe.
- **Design axis:** optimizer / gradient handling (stochasticity).
- **Expected effect:** connectome ±0.02–0.08 pp (noise can help escape or just add variance);
  mouse ±0.2–0.6 pp. Medium uncertainty.
- **Est. compute cost:** **cheap–medium.** Cheaper per step but may need more steps; net wall-clock
  roughly comparable. Adds sampling overhead.
- **Measurement:** standard; batch fraction ∈ {0.25, 0.5} vs full-batch, at matched wall-clock.
- **status: killed**  <!-- 2026-06-21 SCREEN FAIL on both datasets → KILL. Primary arm =
  per-step UNIFORM without-replacement edge subset, FRAC=0.5, loss scaled by |E|/m so the subset
  gradient is an UNBIASED estimator of the full-batch gradient (|E|/m keeps gradient magnitude on the
  baseline scale → grad-clip/LR/β keep their meaning; only injected effect is SGD noise). Everything
  else (N(0,1) init/Adam/clip=1.0/LR/β/budget 20k/5k) IDENTICAL to baseline (run_rocket loop replicated
  verbatim; ONLY the per-step edge SET added). Target-blind/leakage-safe: subset drawn uniformly over
  input edge indices via a dedicated RandomState(seed+104729), never reads the oracle, never
  dataset-special-cased; discrete score is the EXACT full-graph oracle (best-by-oracle tracking only).
  Compute-matched on total_grad_steps per PROTOCOL (n_epochs_done=20k/5k = baseline). connectome Δ=
  −0.8356 pp (mean 82.0602 ± 0.0045, n=3, LARGE REGRESSION); mouse Δ=−0.0325 pp (mean 92.0371 ± 0.2110,
  n=3, within noise) — both fail the 2σ gate. Δ negative on BOTH → correctly-specified screen, NOT an
  H02-style low-variance escalation (connectome own std 0.0045 < baseline floor but the −0.84 pp gap is
  ~185× the std, decisively a regression). Full-batch deterministic descent reaches a markedly better
  basin than its noisy estimator at equal step count. NOTE: wall-clock is higher per run on connectome
  (~1332 s) because the per-step replace=False draw over 5.6M edges dominates — but the comparison basis
  is gradient steps, not wall-clock. UN-RUN arms: FRAC=0.25, weight-proportional sampling, an
  extra-steps arm (would break the grad-step match). See experiments/log.md 2026-06-21 H13 cycle. -->

## H14 — H02 warm-start + anti-tie jitter (free-edge recovery on the new SOTA basin)
- **Hypothesis:** Adding anti-tie position separation (H09's jitter / repulsion, or symmetric
  scoring-time jitter that breaks `pos[tgt]==pos[src]` near-ties) **on top of the confirmed H02
  greedy-FAS warm-start** recovers additional strict-`>` feedforward edges and beats H02 alone on
  the exact metric.
- **Rationale:** H02 is the new SOTA (connectome 82.93% / mouse 92.48%) but it is still a *pure
  Rocket* score — the oracle counts `pos[tgt] > pos[src]` strictly, so any near-tie it leaves
  contributes nothing even when the intended order is correct. H02's warm-start is nearly
  deterministic (σ≈0.0005 / 0.0000), so any genuine free-edge recovery shows up cleanly above its
  own tiny variance. Crucially this is an **orthogonal, non-dynamics** lever (it changed neither
  the schedule nor the basin in the way the killed H01/H03/H04 did): it stacks a free-edge
  mechanism onto a basin mechanism, the two combinations most likely to compound. The evenly-spaced
  greedy init may itself create exact rank ties that the random init never had, so jitter could
  matter *more* here than on the random baseline.
- **Design axis:** initialization (H02 basin) + gradient handling / position regularization
  (anti-tie). Implemented as `src/mfas/experiments/H14.py`: greedy-FAS init (reuse H02's
  `greedy_fas_order`), then either (a) a small repulsion term added to the Rocket loop, or (b)
  deterministic symmetric jitter on the positions before each oracle scoring — both target-blind.
- **Expected effect (reasoned estimate, NOT measured):** connectome +0.005–0.04 pp over H02 (likely
  near noise — honest chance of a no-op if there are few near-ties); mouse +0.05–0.4 pp over H02
  (float weights → more genuine near-ties to recover). Upside is bounded by the size of the
  `|Δ|<ε` edge set, so measure that fraction first to size the ceiling.
- **Est. compute cost:** **cheap.** One-time greedy order (already O(m log n) in H02) + an O(n)
  repulsion term or O(n) scoring-time jitter; per-step cost ≈ baseline. Same epoch budget as
  baseline (connectome 20k, mouse 5k).
- **Comparator (PROTOCOL §Compute-matched — IMPORTANT):** standard knob-swap (init + position
  regularization, same gradient budget), but the fair baseline to beat is **H02 at matched seeds**,
  NOT the random-init `baseline_passthrough`. Run H02 and H14 on the *same* seed set and compute
  Δ = mean(H14) − mean(H02) per dataset. Because H02 (and likely H14) are near-deterministic, use
  the H02-style CONFIRM escalation: if the 2σ_baseline screen is mis-specified for a low-variance
  variant, escalate to the 95% CI-lower-bound CONFIRM against H02@matched-seeds. (A secondary,
  informational comparison vs the random baseline shows total stacked gain, but the *promotion*
  test is vs H02.)
- **Measurement:** exact feedforward % via the frozen oracle, **both** datasets, ≥3 seeds for
  SCREEN (vs H02@matched-seeds), CONFIRM at 5 (connectome) / 20 (mouse) seeds with 95% CI lower
  bound > 0 vs H02. Report the pure H02 score and the H14 score side by side. Oracle used only for
  best-by-oracle tracking (as baseline) — jitter is target-blind, never folded into the loss.
- **status: killed (by implication)**  <!-- 2026-06-21 KILLED without a separate cycle. H14 stacks
  H09's anti-tie jitter on the H02 basin, but H09 self-falsified at opportunity sizing: 0 exact ties
  and near-tie recoverable weight ~3 orders of magnitude below threshold, because Adam spreads
  positions apart. H02's positions are produced by the same Adam optimizer and are equally spread,
  so the identical null result applies — there are no ties for jitter to recover on the H02 basin
  either. The ideator gated H14 on H09 showing signal; it showed none. Compute conserved. See the
  H09 cycle in experiments/log.md. -->
- **Hypothesis:** Combining the confirmed H02 greedy-FAS warm-start with the best-screening
  *objective-axis* lever — heavy/borderline-edge loss reweighting (H06) **or** a margin/hinge
  surrogate (H11), whichever wins its own screen — beats H02 alone, because a better basin plus a
  better-shaped objective compound rather than redundantly re-converging.
- **Rationale:** The killed cycles (H01/H03/H04) all changed *dynamics* and re-hit the plateau;
  H02 changed the *basin*; H06/H11 change the *objective/landscape* (a third, distinct mechanism).
  Two mechanisms that move the metric for different reasons are the most promising thing to stack.
  Starting Rocket in the H02 basin and then keeping gradient on the heavy / correct-but-thin-margin
  edges (which σ_β abandons once they are weakly feedforward) directly attacks *why* the H02 run
  still plateaus. This is gated on H06/H11 first screening positive standalone — if neither beats
  the random baseline, H15 reduces to H02 and should be dropped (cheap to decide).
- **Design axis:** initialization (H02 basin) + loss / surrogate (H06 reweighting or H11 margin).
  Implemented as `src/mfas/experiments/H15.py`: reuse H02's `greedy_fas_order` init, then replicate
  the Rocket loop with the chosen objective lever (reweighted ŵ or hinge/tanh surrogate). Objective
  stays a monotone reward for feedforward orientation (still maximizing feedforward weight) —
  leakage-safe, uses only input edge weights, never the discrete oracle in the loss.
- **Expected effect (reasoned estimate, NOT measured):** connectome +0.01–0.08 pp over H02; mouse
  +0.1–0.6 pp over H02. Honest risk: the objective lever's gain may not be additive with the basin
  gain (both could be exploiting the same slack), so the combined Δ vs H02 could be smaller than
  the standalone H06/H11 Δ vs baseline.
- **Est. compute cost:** **cheap.** Greedy order (as H02) + elementwise reweighting / a swapped
  loss function; per-step cost ≈ baseline. Same epoch budget (connectome 20k, mouse 5k).
- **Comparator (PROTOCOL §Compute-matched — IMPORTANT):** standard knob-swap at matched gradient
  budget; the fair baseline to beat is **H02 at matched seeds**, NOT the random-init baseline.
  Compute Δ = mean(H15) − mean(H02) per dataset on the same seed set; SCREEN vs H02, then CONFIRM
  (5 connectome / 20 mouse seeds, 95% CI lower bound > 0 vs H02). If H15's variance collapses like
  H02's (deterministic init), apply the same low-variance CONFIRM escalation. Also report the
  standalone H06/H11 Δ vs baseline so the additivity of basin × objective is auditable.
- **Measurement:** exact feedforward % via the frozen oracle, **both** datasets, ≥3 seeds SCREEN /
  5+20 CONFIRM, all vs H02@matched-seeds. Pure Rocket score (no post-processing). Oracle used only
  for best-by-oracle tracking.
- **status: dropped**  <!-- 2026-06-21 DROPPED without a cycle. H15 was explicitly gated on H06 OR
  H11 screening positive standalone; BOTH failed (H06 Δ=−0.0380/−0.0794, H11 Δ=−0.0387/+0.1264, both
  sub-threshold, connectome regressing). With no objective lever that beats the random baseline, H15
  reduces to H02 (no additional mechanism to stack), so it is dropped per its own gate. Compute
  conserved. See the H06 and H11 cycles in experiments/log.md. -->

### Ranking rationale (EV / cost)
H01–H02 first: cheapest, best-evidenced upside (exploit known ordering variance + a refinement
trick the paper itself validates). H03–H08 are cheap schedule/optimizer knobs the paper admits are
under-tuned. H04 has the highest ceiling but the most cost/risk, so it sits behind the trivially
cheap wins. H09–H13 are lower-EV or higher-variance ideas kept for diversity across axes
(near-tie recovery, gradient handling, surrogate shape, averaging, stochasticity). Run H03/H07
before H08, and H05 before H10 (shared optimizer machinery).

---

# Phase-4 hypotheses (H16+) — driven by the Stage-A diagnosis

**Read `experiments/diagnosis.md` first.** The Phase-3 backlog above (H01–H15) is closed:
H02 is the confirmed win; H01/H03/H04/H05/H06/H09/H11/H13 are KILLED; H07/H08/H10/H12 deferred;
H14/H15 dropped/killed-by-implication. **Do NOT re-propose any of them.** Phase-4 starts from a
sharper diagnosis than Phase-3's "basin not dynamics" inference:

**Established facts (do not relitigate, all from `experiments/outputs/diagnosis.json`):**
1. **OPTIMIZATION-GAP, not surrogate-misalignment.** best's fixed order out-surrogates Rocket's
   converged solution at *every* β (Δsurrogate +238 to +1,153). The sigmoid surrogate correctly
   ranks the better order above Rocket's — the optimizer fails to reach/hold it.
2. **best is not a holdable attractor.** Initialising Rocket *at* the 84.61% order and running it
   collapses to ~82.9% under cyclic AND constant β, dipping through 76–79% first. The **cyclic
   schedule that re-melts β→0.05** demonstrably destroys good orders.
3. **init→plateau is flat on connectome** (82.87–82.93% over inits spanning 36–69%): better-init-alone
   (DIRECTION I) has a ~0.06 pp ceiling — H02 already captured it. Stronger init is only worth
   pursuing *in combination* with an O/R lever that can hold the basin.
4. The gap is a **distributed reordering** (Kendall-τ 0.61, 8.47% of edges flip, uniform across
   weight buckets, on median-degree not hub endpoints). **0 exact ties**; near-tie recoverable weight
   ~0.0003% ≪ the 1.69% gap → genuine misordering, no free-edge slack (kills any tie/jitter idea).
5. **Synthetic (n=400, near-acyclic) shows NO gap** (Rocket 96.56% > planted 96.10%). The gap is a
   property of the connectome's *hard cyclic structure* — a prototype graph must inject far more
   feedback / strong-connectivity to reproduce it (motivates H21).

**Two axes for Phase-4 (per diagnosis §Selected directions):**
- **DIRECTION O (optimization)** — reach/hold a better basin via stronger continuous optimization
  (schedule, multi-start/tempering). Highest priority: the diagnosis directly implicates the cyclic
  re-melting schedule and the single 82.9% attractor.
- **DIRECTION R (relaxation)** — stop the optimizer from collapsing good orders by changing what is
  differentiated (straight-through discrete forward, rank-space normalization). Secondary: targets
  *why* gradient flow drifts off best (scale blow-up saturating σ, surrogate≠discrete in the forward).

**Honest ceiling read (applies to all H16+).** Because even a perfect init collapses under gradient
flow and the gap is a distributed reordering with no tie-slack, **a large share of the 1.69 pp may be
irreducible to continuous optimization** and genuinely require discrete refinement (the paper's Crane
MIP). Each entry below states honestly *why it might still beat 82.93%* and roughly *how much* —
estimates are reasoned, NOT measured. Many are bets to recover a *fraction* of the gap, not all of it.

**Comparators (PROTOCOL §Compute-matched).** Standard knob-swaps (schedule/loss/relaxation at the
same gradient budget) screen vs **`baseline_passthrough`**, and — since they build on H02's init —
the *promotion* comparison is **vs H02 at matched seeds** (report both: Δ vs random baseline = total
stacked gain, Δ vs H02 = the new lever's marginal gain; promote on Δ-vs-H02). Multi-start / tempering
variants screen vs **`baseline_multistart`** (K naive restarts × total/K epochs, best-of-K,
`n_epochs_done = total`) so the comparison isolates algorithmic merit from extra sampling.

**Leakage rule (every entry).** The discrete oracle may be used ONLY as the baseline already uses it
— best-by-oracle tracking of whole-vector candidates *after* the algorithm produces positions. It
must NEVER enter the differentiable loss, the init, the perturbation choice, or a dataset special-case.
**No variant may read `data/best_solution`** (that path is privileged to `mfas.analysis.gap`,
diagnostics only). STE/soft-rank variants differentiate a function of *positions and input weights*
only; the exact-feedforward forward in an STE uses the strict-`>` order of the *current* positions
(not any known target).

**Prototype-first gate (heavy/novel relaxations, H18–H20).** Before any connectome compute, run on
**mouse + the hard synthetic (H21)**. The hard synthetic must first be shown to *reproduce a Rocket↔best
gap* (Rocket < a strong reference order on it) — otherwise it cannot screen a gap-closer. A relaxation
earns connectome compute only if it closes ≥ ~0.1 pp of the gap on the hard synthetic AND does not
regress mouse. This conserves the ~80 s/run/seed connectome budget for variants with prototype signal.

**Re-ranked Phase-4 order (highest EV/cost first):**

| rank | id | axis | one-line EV justification |
|---|---|---|---|
| 1 | **H16** | O — β schedule | Cheapest, most-directly-implicated lever: replace cyclic re-melting with a single monotone β rise from the H02 basin. Pure schedule-array swap, equal budget. Distinct from killed H03 (random init + kept cycles + only raised β_max). |
| 2 | **H21** | infra (fixture) | Not a Rocket variant — builds the HARD synthetic gap-proxy that gates every heavy relaxation (H18–H20). Cheap, prototype-scale, unblocks the secondary direction. Run early. |
| 3 | **H17** | O — multi-start / tempering | Basin-hopping / parallel tempering on the surrogate from the H02 basin (perturb→re-optimize, keep best-by-oracle). Compute-matched vs `baseline_multistart`. Directly attacks the single 82.9% attractor; medium cost. |
| 4 | **H19** | R — soft-rank | Position→soft-rank (Blondel O(n log n)) normalizes the scale blow-up that saturates σ — the mechanism behind the drift collapse. Scales to connectome; prototype-gated. |
| 5 | **H18** | R — straight-through | Exact discrete feedforward forward + surrogate gradient backward, O(m). Removes the forward surrogate≠discrete error so the optimizer optimizes the true metric directly. Prototype-gated; bias risk. |
| 6 | **H20** | R — Sinkhorn (note only) | Gumbel-Sinkhorn permutation relaxation. O(n²) → **mouse + hard-synthetic ONLY, never connectome.** Lowest EV/cost here; included for axis-completeness and as the strongest-relaxation upper-bound probe. |

---

## H16 — Monotone / graduated β-continuation from the H02 warm-start
- **Hypothesis:** Replacing the cyclic β schedule (which repeatedly re-melts β→0.05 and demonstrably
  destroys good orders) with a **single slow monotone β rise** (small/convex → large/sharp), started
  from the H02 greedy-FAS init, lets Rocket *hold* a better basin and yields a higher exact
  feedforward weight than H02 (cyclic) at equal gradient budget. Expected direction: **positive**.
- **Exact mechanism (what changes):** swap `make_beta_schedule` for a monotone increasing schedule
  over all `epochs`, e.g. `β = β_min + (β_max−β_min)·(i/(T−1))**p` with primary arm
  `β_min=0.05, β_max=1.05, p=1` (linear; identical endpoints to baseline, only the *path* differs —
  no re-melting), and an arm `β_max∈{1.5,2}` to test holding margins. Everything else
  (Adam/clip=1.0/LR const→exp/budget 20k/5k) and the **H02 greedy-FAS init** unchanged. Implemented
  as `src/mfas/experiments/H16.py`: reuse `H02.greedy_fas_order` + `_init_positions_from_order`, build
  the monotone β array, replicate the `run_rocket` loop with `betas = monotone(...)`.
- **Axis:** DIRECTION O (β-schedule / graduated optimization — the textbook continuation method:
  solve the smooth low-β problem first, use its solution to seed the sharper one).
- **Compute-matching:** standard knob-swap @ same epochs. Screen vs `baseline_passthrough`; **promote
  vs H02 @ matched seeds** (report both Δs).
- **Leakage-safety:** β is a loss-shape scalar; schedule depends only on the step index. No oracle in
  the loss, no `best_solution`, no dataset special-case.
- **Distinct from killed H03:** H03 ran from *random* init, *kept the cyclic schedule* for 75% of
  training, and only appended a sharper terminal ramp to a higher β_max — it tested "sharper tail",
  not "no re-melting from a good basin". H16 removes cycling *entirely* and pairs it with H02's init,
  which is the configuration the drift probe implicates (cyclic re-melting collapses good orders).
- **Expected effect (reasoned, NOT measured):** connectome +0.0–0.15 pp over H02; mouse +0.0–0.6 pp.
  **Why it might beat 82.93%:** the drift probe shows the cyclic schedule actively *drops* a good
  order through 76–79% on every re-melt; a monotone schedule never re-melts, so a good basin entered
  early (H02 starts at 68.9%) is more likely to be held as β sharpens. **Why it might not:** the
  *constant-β* drift probe ALSO collapsed (83.03%), so the dominant 82.9% attractor is not solely a
  re-melting artefact — monotone β may simply re-converge there too. Honest: this is the single most
  likely O-lever to move the metric, but could still be a small/within-noise gain.
- **Est. compute cost:** **cheap** (schedule array only; per-step cost identical; same budget).
- **Measurement:** exact ff% both datasets, ≥3 seeds SCREEN vs H02@matched-seeds, CONFIRM 5/20 with
  95% CI lower bound >0 vs H02. H02's init is near-deterministic → apply the H02-style low-variance
  CONFIRM escalation if the 2σ screen is mis-specified.
- **KILL/keep prediction:** lean **keep-ish but uncertain** (best single O bet). **Falsified if**
  Δ-vs-H02 ≤ 0 on connectome across p∈{1} and β_max∈{1.05,1.5,2}, i.e. monotone β re-converges to
  the same plateau — which would corroborate that the 82.9% attractor is intrinsic to Adam-on-σ, not
  the cyclic schedule, and would up-weight DIRECTION R.
- **status: killed**  <!-- 2026-06-21 (Phase 4) SCREEN FAIL connectome. H16=82.6269 vs baseline
  82.8958 (Δ −0.2688) and vs H02 82.9300 (Δ −0.3031); mouse +0.134 vs H02. The monotone low-β start
  MELTS the warm-start (drift-probe mechanism) and the tuned cyclic baseline beats it on connectome.
  This is exactly the falsifier above → corroborates the 82.9% attractor is intrinsic to Adam-on-σ,
  up-weighting DIRECTION R. See log.md 2026-06-21 H16. -->
- **Falsifier outcome (realized):** monotone β re-converged BELOW plateau on connectome → 82.9%
  attractor is intrinsic to the optimizer, not the cyclic schedule. Up-weights DIRECTION R (rank-space)
  and DOWN-weights further β/LR/schedule (DIRECTION O dynamics) tweaks — consistent with Phase-3 #2.

## H17 — Continuous basin-hopping / parallel tempering on the surrogate, from the H02 basin
- **Hypothesis:** Iterating *perturb → re-optimize → keep-best-by-oracle* (basin-hopping), or running
  a few replicas at different β "temperatures" with occasional swaps (parallel tempering), starting
  from the H02 basin, escapes the dominant 82.9% attractor and beats single-run H02 at **equal total
  gradient budget**. Expected direction: **positive (small)**.
- **Exact mechanism:** outer loop of `R` rounds, each = (a) perturb current best positions by
  target-blind Gaussian noise `σ_pert` (or a partial re-melt: short low-β phase), (b) re-optimize with
  a short monotone β sub-run of `epochs/R` steps, (c) score with the oracle and keep the whole-vector
  best. Parallel-tempering arm: `M` replicas with fixed βs spanning [0.05,1.05], periodic
  metropolis-free "keep-better-by-surrogate" replica swaps, oracle-score all, keep global best. First
  init = H02 greedy-FAS order. `src/mfas/experiments/H17.py`. `n_epochs_done` = total optimizer steps
  summed across rounds/replicas (= baseline 20k/5k).
- **Axis:** DIRECTION O (multi-start / annealing — simulated-annealing-style perturbation is the
  sibling of graduated optimization per the continuation-method literature).
- **Compute-matching:** **multi-start → compare vs `baseline_multistart`** with `MFAS_MULTISTART_K`
  set so total steps match; ALSO report Δ vs H02@matched-seeds (H02 is one round with no perturbation,
  so H17 must beat it to justify the machinery).
- **Leakage-safety:** perturbations are target-blind Gaussian / β re-melts; the oracle is used ONLY to
  select among whole-vector candidates (exactly baseline best-tracking). No `best_solution`, no
  per-edge oracle signal, no dataset special-case.
- **Distinct from killed H01:** H01 was *naive independent restarts from fresh random inits* under the
  unchanged cyclic dynamics (re-converged within +0.0003 pp). H17 (a) starts every round from the
  *current best* (hopping, not independent), (b) uses the monotone/short sub-runs from H16, and (c)
  perturbs an already-good order rather than restarting from scratch — a genuinely different escape
  mechanism (exploit the τ=0.61 partial-basin overlap by hopping within it).
- **Expected effect (reasoned, NOT measured):** connectome +0.0–0.1 pp over H02; mouse +0.1–0.7 pp
  (mouse's 14× larger σ → more tail to harvest, as in the H01 rationale). **Why it might beat
  82.93%:** the drift probe shows neighbouring orders span 76–84.6% — there *is* structure to hop
  between; best-of-K over re-optimized perturbations harvests the right tail without needing to
  *hold* best. **Why it might not:** the diagnosis says the 82.9% basin is dominant and deep; small
  perturbations likely fall back into it (H01 evidence), and splitting the budget under-trains each
  round (H01 lost −0.84 pp from under-training). Net EV is genuinely uncertain — most plausible on mouse.
- **Est. compute cost:** **medium** (R re-optimizations + R extra oracle scores; same total grad
  steps but more scoring overhead; tempering's M replicas add memory but parallelize on device).
- **Measurement:** exact ff% both datasets, ≥3 seeds SCREEN vs `baseline_multistart` AND vs
  H02@matched-seeds; CONFIRM 5/20 with 95% CI lower bound >0 vs H02. Report pure best-of score; size
  `σ_pert` / R on mouse first (cheap) before connectome.
- **KILL/keep prediction:** lean **kill on connectome, possible keep on mouse**. **Falsified if**
  best-of-K over perturbed re-optimizations does not exceed H02 beyond noise on either dataset
  (would confirm the basin is too dominant for cheap continuous hopping → only discrete refinement
  recovers the gap).
- **Status: proposed**

## H18 — Straight-through estimator: exact discrete feedforward forward, surrogate gradient backward
- **Hypothesis:** Making the forward pass score the **exact discrete** feedforward indicator
  `1[pos[v] > pos[u]]` while routing gradients through the sigmoid surrogate in the backward pass
  (a straight-through estimator) removes the forward surrogate≠discrete mismatch, so the optimizer
  optimizes the *true* metric directly and reaches a higher exact feedforward weight than the
  surrogate-only Rocket. Expected direction: **positive on a graph that has a gap (prototype-gated)**.
- **Exact mechanism:** per edge define `hard = (Δ > 0).float()` and
  `ste = hard + (sig − sig.detach())` where `sig = σ(β·Δ)`; loss `= −(ste · ŵ).sum()`. Forward value
  = exact discrete weight; backward gradient = the baseline sigmoid gradient (identity-through the
  hard step). Keep H02 init + monotone β (H16) so the *backward* surrogate still sharpens.
  `src/mfas/experiments/H18.py`, `run_rocket` loop replicated with this loss. O(m) per step (one extra
  elementwise compare + detach) — scales to connectome.
- **Axis:** DIRECTION R (relaxation / gradient estimator). Standard STE construction
  (`forward exact, backward smooth`), biased-but-useful gradient per the STE literature.
- **Compute-matching:** standard knob-swap @ same epochs. Screen vs `baseline_passthrough`; promote
  vs H02@matched-seeds.
- **Leakage-safety:** the "discrete forward" uses ONLY the current positions' own strict order — it is
  literally what the oracle would compute on the *current* iterate, not a known target. No
  `best_solution`, no folding the oracle's *value* into the loss, no dataset special-case. (Subtle but
  clean: STE differentiates the surrogate; the hard term is detached, so no target value leaks into
  gradients.)
- **Prototype-first plan:** mouse + hard synthetic (H21) BEFORE connectome. The near-acyclic synthetic
  has no gap, so H18 must be shown on H21's hard graph; earn connectome compute only on ≥~0.1 pp gap
  closure there + no mouse regression.
- **Expected effect (reasoned, NOT measured):** prototype-dependent; if it helps, connectome
  +0.0–0.2 pp over H02. **Why it might beat 82.93%:** the diagnosis says the surrogate is *aligned*
  but the optimizer's late surrogate progress no longer converts to discrete gain (flat discrete tail
  slope at step 3); an exact forward keeps the objective *pinned to the metric* so late steps that
  raise the surrogate without raising the discrete score get no reward — directly attacking the
  step-3 "surrogate descends, discrete flat" finding. **Why it might not:** STE gradient = the same
  saturated sigmoid gradient that already plateaus; the forward change reweights *which* edges count
  but the descent direction is unchanged near a tie, so it may re-converge. Honest medium-high risk.
- **Est. compute cost:** **cheap–medium** (O(m) extra compare/detach per step; prototype-gated so most
  cost is mouse+synthetic, not connectome).
- **Measurement:** exact ff% on mouse + H21 hard-synthetic for the gate; if passed, both datasets ≥3
  seeds SCREEN vs H02, CONFIRM 5/20. Report whether the discrete-tail slope (the step-3 pathology)
  improves.
- **KILL/keep prediction:** lean **uncertain, prototype decides**. **Falsified if** on the hard
  synthetic H18 does not exceed surrogate-only Rocket beyond noise (then the STE forward adds no
  signal the aligned surrogate lacked) → do not spend connectome compute.
- **Status: proposed**

## H19 — Position→soft-rank differentiable ranking (rank-space surrogate, O(n log n))
- **Hypothesis:** Computing the surrogate in **rank space** — replace raw positions with a
  differentiable soft-rank (Blondel et al. 2020, O(n log n)) before forming Δ — normalizes the
  unbounded position-scale blow-up that saturates σ (Rocket's converged `pos_std ≈ 142`), keeping
  gradients alive on borderline edges and letting the optimizer hold a better order; expected to
  beat surrogate-on-raw-positions Rocket. Expected direction: **positive (prototype-gated)**.
- **Exact mechanism:** `r = soft_rank(positions, regularization_strength=ε)` (torchsort /
  fast-soft-sort, O(n log n)); use `Δ = r[tgt] − r[src]` (optionally normalized to [-1,1]) in the
  existing `loss = −(σ(β·Δ)·ŵ).sum()`. Soft-rank is bounded in [1,n] and monotone in positions, so
  scale cannot blow up and σ cannot globally saturate. H02 init + monotone β. `src/mfas/experiments/H19.py`.
  If torchsort/fast-soft-sort is not installed, no new heavy dependency on connectome path without
  confirming availability first (check env; the diagnosis env is `allen`).
- **Axis:** DIRECTION R (relaxation — rank-space reparametrization; directly targets the scale-blowup
  saturation mechanism implicated by the drift collapse).
- **Compute-matching:** standard knob-swap @ same epochs (extra O(n log n) sort per step, n=136k is
  cheap vs the 5.6M-edge gather). Screen vs `baseline_passthrough`; promote vs H02@matched-seeds.
- **Leakage-safety:** soft-rank is a function of positions only; no oracle, no `best_solution`, no
  special-case. Bounded, monotone → still maximizes feedforward orientation (argmax-preserving).
- **Prototype-first plan:** mouse + hard synthetic (H21) before connectome (novel relaxation + a new
  op). Earn connectome compute on gap closure there.
- **Expected effect (reasoned, NOT measured):** if it helps, connectome +0.0–0.25 pp over H02 (the
  highest-ceiling R lever, because it attacks the *named* mechanism). **Why it might beat 82.93%:**
  the diagnosis Step-1 shows best out-surrogates Rocket and Rocket's positions blow up to std≈142
  where a unit reorder barely moves σ — rank space makes adjacent swaps always carry O(1/n) gradient,
  so the optimizer can keep refining the order instead of inflating scale. This is the cleanest
  hypothesis for *why* gradient flow drifts off best. **Why it might not:** the regularization ε trades
  rank fidelity for smoothness; too-smooth soft-ranks blur exactly the borderline edges we need
  sharp, and the O(n log n) op adds per-step cost. Could also just re-converge if the basin (not the
  parametrization) is the true bottleneck.
- **Est. compute cost:** **medium** (one O(n log n) soft-rank/step; prototype-gated; needs a sorting
  lib — verify install before committing connectome time).
- **Measurement:** exact ff% on mouse + H21 gate; if passed, both datasets ≥3 seeds SCREEN vs H02,
  CONFIRM 5/20. Sweep ε ∈ {small, medium}; log converged `pos`/`rank` std to confirm the scale fix.
- **KILL/keep prediction:** lean **most promising R lever, but uncertain**. **Falsified if** rank-space
  surrogate does not exceed raw-position Rocket on the hard synthetic beyond noise across ε (then
  scale-saturation is not the operative bottleneck) → no connectome compute.
- **Status: screened (prototype) — FALSIFIED; recommend EARLY_EXIT DIRECTION R**
  <!-- 2026-06-21 (Phase 4 Stage-B, DIRECTION R prototype). torchsort/fast-soft-sort NOT installed →
  implemented an O(n^2) all-pairs soft rank r_i=Σ_j σ(α·(p_j−p_i)), normalized to [0,1], loss on rank
  gaps; H02 warm-start, equal compute (mouse 5k, synthetic 4k), connectome guarded/refused
  (src/mfas/experiments/H19.py). PROTOTYPE RESULTS (frozen scorer):
    • mouse (n=3 seeds 42/123/999): H19 = 90.1263 ± 0.0000 vs baseline 92.0696 → Δ = −1.94 pp (REGRESSION).
    • hard synthetic H21 cfg0 (n=3): H19 = 68.6175 ± 0.9638 vs baseline 73.2768 ± 1.1858
      → Δ = −4.66 ± 0.39 pp (large REGRESSION); reference best-known 74.0733 (gap to baseline +0.80 pp).
  Soft-rank lands EXACTLY at the greedy-FAS warm-start value (68.6175) and never improves it; this is
  ROBUST across α∈{2,4,8,16,32} (all 68.6175 ± 0.9638). Normalized rank gaps are O(1/n) → σ(β·gap)
  gradient is too flat to move positions at the baseline LR; the optimizer stalls and best-by-oracle
  keeps the init. Falsified per the KILL prediction (does not exceed raw-position Rocket on the hard
  synthetic, across α) → NO connectome compute. Reproduce:
    python experiments/protoR_softrank_synth.py   (synthetic + reference)
    python -m eval.run_variant --exp H19 --dataset mouse --seed {42,123,999} --out results/ --role implement
  Result files: results/*-H19-mouse-s{42,123,999}-implement-*.json. -->


## H20 — Gumbel-Sinkhorn permutation relaxation (NOTE ONLY — O(n²), prototype-restricted)
- **Hypothesis:** A full doubly-stochastic permutation relaxation (Sinkhorn / Gumbel-Sinkhorn) over a
  learned score matrix optimizes the ordering more globally than per-node positions and closes more
  of the gap on a graph that *has* one. Expected direction: **possibly positive, but cost-prohibitive
  at scale.**
- **Exact mechanism:** learn node scores → build an n×n cost, Sinkhorn-normalize to a soft permutation
  P, score `Σ_{(u,v)} P-implied-order weight`, anneal Sinkhorn temperature; round to a hard
  permutation for the oracle. `src/mfas/experiments/H20.py`.
- **Axis:** DIRECTION R (strongest relaxation — global permutation rather than 1-D embedding).
- **Compute-matching:** standard knob-swap @ same prototype budget vs `baseline_passthrough` **on
  mouse + hard-synthetic ONLY**.
- **Leakage-safety:** scores/cost from positions + input weights only; oracle only rounds-and-scores
  the final P. No `best_solution`, no special-case.
- **Prototype-first plan (HARD CONSTRAINT):** **mouse + hard synthetic (H21) ONLY. NEVER connectome.**
  Gumbel-Sinkhorn is O(n²) per Sinkhorn iteration → infeasible at n=136k (≈1.9e10 entries). It exists
  in the backlog purely as an *upper-bound probe*: "does the strongest available relaxation recover
  the synthetic gap that O(n log n) methods miss?" — a diagnostic for whether the residual gap is
  reachable by *any* continuous method.
- **Expected effect (reasoned, NOT measured):** synthetic/mouse only; informational. Even a positive
  result does **not** transfer to connectome (no scalable path) — so EV/cost is low; ranked last.
- **Est. compute cost:** **expensive** per step (O(n²)); strictly prototype-scale.
- **Measurement:** exact ff% on mouse + H21 hard-synthetic only, ≥3 seeds; report as a relaxation
  *upper bound*, not a connectome candidate.
- **KILL/keep prediction:** lean **diagnostic-only; will not become a connectome variant.** **Useful
  iff** it closes substantially more synthetic gap than H18/H19 (→ evidence the residual gap is
  continuous-reachable but needs a global relaxation, motivating a scalable approximation); **null
  result** corroborates the diagnosis that the residual is discrete-refinement territory.
- **Status: proposed**

## H21 — Build the HARD synthetic gap-proxy fixture (infrastructure, prototype gate)
- **Hypothesis (operational, not a Rocket variant):** A synthetic generator with **high feedback /
  strong-connectivity** (unlike the existing near-acyclic `make_synthetic_graph`, on which Rocket
  96.56% > planted 96.10%, i.e. NO gap) can reproduce a **Rocket↔reference gap** at prototype scale
  (n≈1–5k), giving a cheap proxy on which to screen the relaxation hypotheses H18–H20 before spending
  connectome compute.
- **Exact mechanism (what to build):** extend `mfas.analysis.gap.make_synthetic_graph` (or add a sibling
  `make_hard_synthetic_graph`) that injects far more feedback and strong-connectivity:
  - raise `feedback_frac` toward parity with forward weight (e.g. 0.4–0.9) so the planted order is
    only modestly above chance;
  - add **dense cyclic cores / nested SCCs** (clusters with many bidirectional or cycle edges) so the
    optimal order is a *distributed reordering* (mimicking the connectome's τ≈0.61, uniform-across-
    weight-buckets structure), not a near-DAG;
  - heavy-tailed weights (match connectome's skew) but keep the disagreement *uniform across weight
    buckets* and on *median-degree* endpoints, per diagnosis Step 2.
  Return `(GraphData, reference_order, reference_pct)` where `reference_order` is a strong (e.g.
  greedy-FAS or planted) order used **only as a diagnostic comparator**, never inside any variant.
- **Acceptance criterion (this is the gate's gate):** the fixture is valid ONLY if **baseline Rocket
  scores meaningfully *below* the reference order on it** (e.g. gap ≥ ~0.5 pp), reproducing the
  connectome phenomenon at small scale. If Rocket ≥ reference (like the current easy synthetic), tune
  feedback/SCC density up until a gap appears, or report that a gap cannot be synthesised cheaply.
- **Axis:** infrastructure (prototype proxy for DIRECTION R screening).
- **Compute-matching / leakage:** n/a (fixture). `reference_order` is diagnostic-only and must NEVER
  be read by a variant's init/loss/perturbation; it is the synthetic analogue of `best_solution` and
  carries the same privilege boundary. No connectome data involved.
- **Expected effect:** unblocks H18/H19 (and the H20 probe) by providing a fast gap-bearing graph;
  also a thesis artefact characterizing *what structure creates the gap* (ties into the parent
  CLAUDE.md "real brains vs random graphs" question — how much unavoidable feedback vs structure).
- **Est. compute cost:** **cheap** (n≈1–5k; seconds/run). One-time generator + a validation script
  that prints baseline-Rocket-vs-reference on it.
- **Measurement:** report `(reference_pct, baseline_rocket_pct, gap)` over ≥3 seeds; declare the
  fixture usable iff a stable gap ≥ ~0.5 pp is reproduced. Persist the generator params.
- **KILL/keep prediction:** lean **keep / build first** (it is the dependency for H18–H20).
  **Falsified if** no parameter setting at prototype scale produces a stable Rocket↔reference gap —
  which would itself be a finding (the gap is intrinsically large-scale / connectome-structure-specific,
  and prototype screening of relaxations is not possible → run H18/H19 directly on mouse, accept higher
  connectome risk).
- **Status: built / USABLE** <!-- 2026-06-21 (Phase 4 Stage-B). Added
  `mfas.analysis.gap.make_hard_synthetic_graph(...)` (privileged analysis module; its
  `reference_order` is diagnostic-only, same privilege boundary as best_solution — NEVER read by any
  variant). Construction: inter-block forward backbone + high-feedback inter-block backward edges
  (feedback_frac=0.65) + dense bidirectional cyclic cores inside each of n_clusters blocks
  (intra_cycle_frac=0.55) + heavy-tailed (Pareto, α=2.0) integer weights. Reference = high-effort
  oracle-optimised 'best-known' (best of greedy-FAS / block-macro + 6×12k Rocket runs, refined by an
  oracle-guided barycenter sift). LOCKED config (cfg0): n=400, avg_out=10, feedback_frac=0.65,
  n_clusters=8, intra_cycle_frac=0.55, weight_alpha=2.0 (defaults). VERIFIED GAP (n=3 seeds 42/123/999,
  4000-epoch baseline Rocket): reference 74.0733 ± 1.21 vs baseline-Rocket 73.2768 ± 1.19 →
  gap = +0.80 ± 0.21 pp (PASSES the ≥0.5 pp gate; STABLE). Note: a cheap greedy-FAS order alone is
  ~5 pp BELOW Rocket here, so the gap is only real against the high-effort reference — the gate
  required a strong best-known, not a weak heuristic. Reproduce:
    python experiments/protoR_tune_hardsynth.py   (sweep; cfg0 is the chosen fixture)
  Unblocks H18–H20 prototype screening. First R consumer (H19 soft-rank) FALSIFIED on it (see H19). -->

## H22 — Bounded-window discrete local search (sifting / re-insertion) as a Rocket post-phase
- **Axis:** DIRECTION D (discrete refinement) — NEW. Outside the continuous-only Stage-B scope that
  was stopped; it is the corollary of finding #3 ("irreducible to *continuous*" ≠ "irreducible") and
  is explicitly sanctioned by parent CLAUDE.md goal #1 ("stronger local search, hybrid
  discrete+continuous, partial Crane on subgraphs").
- **Hypothesis:** After `run_rocket` converges from the H02 greedy-FAS init, a cheap bounded-window
  **sifting / re-insertion** local search on the order recovers a fraction of the ~1.69 pp gap that
  the continuous optimizer cannot — because it acts directly on the discrete cyclic-core reorderings
  the smooth surrogate gradient is blind to (finding #3). Expected direction: positive **iff the
  gap's flips are short-range**.
- **Exact mechanism (proposed):** for each node, move it to the rank within ±W maximizing the net
  feedforward weight of its incident edges, by **exact incremental delta from input edge weights +
  current ranks only** (no oracle in the loop); a few sweeps; `src/mfas/experiments/H22.py`.
- **Distinct from killed H04:** H04 moved nodes by a **weighted barycenter** = the same coarse
  "majority-vote" signal the gradient already follows (it accepted **0** moves). Sifting evaluates the
  **exact discrete delta of a concrete reinsertion**, a genuinely different mechanism.
- **Leakage-safety:** moves chosen from input weights + current ranks; the frozen oracle scores only
  the final order (best-by-oracle). Never reads `data/best_solution`; the target value never enters a
  move choice. Pure Rocket score reported separately from the refinement (CLAUDE.md rule).
- **Compute-matching:** Rocket+refine vs H02 at matched seeds; disclose the added non-gradient cost.
- **Sizing gate (run BEFORE building, like H09):** `experiments/size_localsearch.py` →
  `experiments/outputs/localsearch_sizing.json`. Measures, on H02's converged connectome order:
  (1) feedback weight reachable within rank-window W (leakage-safe ceiling), and (2 — privileged via
  `mfas.analysis.gap`) the rank-distance of the Rocket↔best orientation flips. Build only if a
  meaningful fraction of the net +1.69 pp is reachable at a tractable window.
- **status: killed (by sizing)**  <!-- 2026-06-22 SELF-FALSIFIED at the sizing gate, exactly the H09
  pattern (sized → ~0 reachable opportunity → kill before building; compute conserved). Cross-check
  PASSED (validates the measurement): net gap +1.6874 pp = gain +4.5975 − lose +2.9102, reproducing
  the Stage-A diagnosis; H02 order re-scores 82.9273% (rank-faithful, 0 ties). DECISIVE RESULT — the
  gap is LONG-RANGE / GLOBAL, not local: the recoverable (feedback→feedforward) weight has rank-distance
  percentiles p25=8,290 / p50=22,580 / p90=87,497 in Rocket's order (n=136,648) — the median recoverable
  edge needs a node to travel ~22.6k ranks. Net gap recoverable within any tractable window is ≤0:
  W=100 → net −0.0100 pp (gain only 0.24% of total gain), W=1000 → net −0.1696 pp, W=5000 → net
  −0.4045 pp; net is positive only for W≤10 (+0.0003 pp). Measure-1 ceiling agrees: feedback pool within
  W=100 = 0.027 pp, W=1000 = 0.41 pp. A bounded-window single-node sift therefore CANNOT close the gap
  (and, like H04, has no improving local move — within a window the broken `lose` edges outweigh the
  `gain`). Mouse is uninformative here (n=148, whole graph is "local", no mouse best_solution). This
  STRENGTHENS finding #3: the residual is irreducible not only to continuous methods but to *bounded
  local* discrete refinement — it is a global reordering requiring global discrete optimization (the
  paper's Crane MIP). Reproduce:
    /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python experiments/size_localsearch.py
  Output: experiments/outputs/localsearch_sizing.json. -->

### Phase-4 ranking rationale (EV / cost)
**H16 first** — cheapest, most directly implicated by the diagnosis (cyclic re-melting collapses good
orders), pure schedule swap, and the configuration the drift probe points at; if it fails it sharply
informs whether the 82.9% attractor is schedule-induced or intrinsic. **H21 next** (build the gate)
because H18–H20 cannot be responsibly screened without a gap-bearing prototype. **H17** (basin-hopping
/ tempering) is the second O lever, medium cost, most promising on noisy mouse. **H19 (soft-rank)**
ranks above **H18 (STE)** among R levers because it attacks the *named* scale-saturation mechanism with
a scalable O(n log n) op, whereas STE's descent direction near a tie is the same saturated sigmoid
gradient that already plateaus. **H20 (Sinkhorn)** is last — diagnostic-only, O(n²), never connectome.
Across all: honest prior is that **much of the 1.69 pp may be irreducible to continuous optimization**;
Phase-4 aims to recover a *fraction* and to *quantify* how much O/R can buy before conceding the
remainder to discrete (Crane-style) refinement.

**Literature grounding:** graduated optimization / continuation methods (H16) —
[Graduated optimization (Wikipedia)](https://en.wikipedia.org/wiki/Graduated_optimization),
[Hazan et al., On Graduated Optimization for Stochastic Non-Convex Problems](https://arxiv.org/abs/1503.03712),
[constraint/temperature annealing for ranking (arXiv:2004.09702)](https://arxiv.org/pdf/2004.09702),
with a noted caution that sigmoid mappings can trap variables at large magnitudes
([Parallel Quasi-Quantum Annealing, arXiv:2409.02135](https://arxiv.org/pdf/2409.02135)) — corroborating
the scale-blowup motivation for H19. Straight-through estimator (H18) —
[STE overview](https://www.emergentmind.com/topics/straight-through-estimator-ste),
[Decoupled STE (arXiv:2410.13331)](https://arxiv.org/pdf/2410.13331). Differentiable sorting/ranking
O(n log n) (H19) — [Blondel et al., Fast Differentiable Sorting and Ranking, ICML 2020 (arXiv:2002.08871)](https://arxiv.org/abs/2002.08871),
impls [google-research/fast-soft-sort](https://github.com/google-research/fast-soft-sort) and
[torchsort](https://github.com/teddykoker/torchsort). Sinkhorn/Gumbel-Sinkhorn permutation relaxation
(H20) — same differentiable-sorting line (optimal-transport view of ranking, Cuturi et al. 2019,
referenced in the Blondel paper).

---

# Phase-5 re-test backlog (MICrONS)

**Why this section exists.** Phase 3–4 ran on exactly TWO real graphs — the fly `connectome`
(136k nodes, large, *hard* cyclic structure with a verified 1.69 pp optimization-gap) and the tiny
`mouse` (148 nodes, ~14× noisier). The require-improvement-on-BOTH promotion rule (PROTOCOL §Stage-1)
killed several hypotheses that GAINED on `mouse` purely because they REGRESSED the single large
graph we had. With one large graph, "regresses connectome" is indistinguishable from "regresses
large graphs in general." **MICrONS** (minnie65 mouse visual cortex, ~72,789 neuron nodes) is a
SECOND large real connectome and a leakage-clean discriminator: it has **no known MFAS solution**
(`data/best_solution`-style privilege concerns do not apply), so any variant runs the unchanged
harness with no oracle-value to peek at.

**The three outcomes MICrONS can produce for a mouse-gain / connectome-regression hypothesis:**
- **GENERAL WIN** — gains on MICrONS too (and ideally re-examined on mouse): the connectome
  regression was the idiosyncrasy, not the gain. Strongest possible Phase-5 result.
- **RECOVERED / GRAPH-DEPENDENT** — gains on MICrONS + mouse but still regresses connectome: a REAL
  effect on "ordinary" large connectomes that the fly graph's hard cyclic core specifically
  suppresses. A genuine "lost theory," scoped to graph structure.
- **SMALL-GRAPH ARTIFACT** — null/regresses on MICrONS like it did on connectome: the mouse gain was
  a 148-node small-sample artifact (mouse σ≈0.26 pp is huge). Confirms the original KILL, now with
  TWO large graphs of evidence.
- **STILL NULL** — within noise everywhere: corroborates finding #2 (basin-not-dynamics).

**Evidence rule (unchanged).** Every prior number below is cited to the `experiments/log.md` cycle or
`experiments/findings.md` row it came from; I re-verified each against those files. Estimates of what
MICrONS will show are reasoned, NOT measured. All re-tests use the **unchanged frozen oracle/harness**,
exact feedforward %, **all three datasets** (connectome / mouse / **microns**), ≥3 seeds SCREEN →
CONFIRM per the (3-dataset-updated) PROTOCOL. **MICrONS noise floor is unknown until a baseline run
exists** — establishing `baseline_passthrough` mean±std on MICrONS (≥3 seeds, ideally 5) is the
implicit prerequisite for every entry and should be the very first MICrONS run (call it **P00**).

**Comparators.** Standard knob-swaps screen vs `baseline_passthrough` per dataset; H02-stacking
variants promote vs **H02 @ matched seeds**; multi-start/tempering variants vs `baseline_multistart`.
Same leakage rule as all prior phases: oracle used ONLY for best-by-oracle tracking of whole-vector
candidates, NEVER folded into loss/init/perturbation, never dataset-special-cased.

## Ranked summary table

| rank | id | hypothesis (one line) | prior conn Δ | prior mouse Δ | why fly-suppressed | EV/cost |
|---|---|---|---|---|---|---|
| 0 | **P00** | establish MICrONS baseline mean±std (not a variant; gates all below) | — | — | — | must-do / cheap |
| 1 | **H11r** | margin/smooth-hinge surrogate keeps gradient on correct-but-thin edges | **−0.0387** (log L688) | **+0.1264** (log L691) | fly's wide-margin saturated basin already near-optimal; hinge's wider gradient band only helps graphs where σ_β under-separates | **HIGH** / cheap |
| 2 | **H16r** | monotone β-continuation from H02 warm-start (no cyclic re-melt) | −0.2688 vs base / **−0.3031 vs H02** (log L994) | **+0.5435 vs base** / +0.1337 vs H02 (log L996) | fly's deep 82.9% attractor melts the warm-start; mouse CLEARED its 0.52 screen vs baseline → 2nd large graph decides if monotone-β is a large-graph win or fly-melt | **HIGH** / cheap |
| 3 | **H02r** | greedy-FAS warm-start (CONFIRMED win) — test GENERALITY to a 2nd large connectome | **+0.0508** (find #1, n=15) | **+0.2064** (find #1) | n/a (already a win) — MICrONS tests whether the basin lever generalizes beyond the fly graph | **HIGH** / cheap |
| 4 | **H03r** | sharper/extended terminal β ramp (raise late surrogate sharpness) | **−0.0376** (log L354) | **+0.0114** (log L357) | fly's tuned cyclic schedule is near-optimal for its hard structure; terminal sharpening may help an easier large graph | **MED** / cheap |
| 5 | **H17r** | basin-hopping / parallel-tempering from H02 basin | deferred: "lean kill on connectome" (backlog L666) | deferred: "possible keep on mouse" (backlog L666) | deferral was fly-specific (drift probe: every init flows back to fly's 82.9% basin); MICrONS basin may be shallower/multi-modal | **MED** / medium |
| 6 | **H18r** | straight-through estimator: exact discrete forward, surrogate backward | deferred (not run, log L1042) | deferred (not run) | deferral assumed fly's STE backward ≈ baseline trajectory; on a graph with a different surrogate↔discrete tail the exact forward may pin late steps | **MED-LOW** / cheap-prototype |
| 7 | **H06r** | weight-aware (heavy & borderline) loss reweighting | **−0.0380** (log L603) | **−0.0794** (log L606) | regressed BOTH → weak "lost theory"; but MICrONS weight-skew differs from fly, so re-size on its distribution | **LOW** / cheap |
| 8 | **NEW H22** | block/SCC-macro warm-start (condense SCCs → order DAG of blocks → expand) — a STRONGER discrete-FAS init than H02 | — (new) | — (new) | fly's dense cyclic core makes greedy-FAS (H02) near-best already; a 2nd large graph with looser SCC structure may have headroom a coarser block order captures | **MED** / cheap-medium |
| 9 | **NEW H23** | degree/strength-stratified init scale (source/sink-aware spread) motivated by MICrONS hub structure | — (new) | — (new) | fly source instability (Jaccard@1000 0.33–0.40, CLAUDE.md) caps init gains; a graph with more stable sources may reward a structure-aware spread | **LOW-MED** / cheap |
| 10 | **H07r/H08r/H10r/H12r** | LOW-EV pure-dynamics knobs (LR sched / β+LR joint / grad-clip / EMA) | deferred (never run) | deferred (never run) | finding #2 predicts null; kept for completeness only | **LOWEST** / cheap |

---

## P00 — Establish the MICrONS baseline noise floor (prerequisite, not a Rocket variant)
- **Operational task:** run `baseline_passthrough` (unchanged Rocket) on MICrONS, ≥3 seeds
  (42/123/999), ideally 5, to record `baseline mean ± std` and derive the 2σ SCREEN threshold for
  the new dataset. Also run `H02` on MICrONS at the same seeds (H02 is the SOTA init and the
  comparator for every stacking variant).
- **Why first:** the PROTOCOL screen gate is `Δ > 2σ_baseline` per dataset; MICrONS σ is unknown.
  Every entry below is un-screenable until this exists. Mouse taught us σ matters (0.26 pp swamped
  several "gains"); MICrONS at ~73k nodes should be far tighter (closer to connectome's 0.02 pp than
  mouse's 0.26 pp), which is itself informative about whether mouse-only gains survive a low-noise
  large graph.
- **What it REVEALS:** the discriminating power of MICrONS. If σ_MICrONS ≈ connectome's ~0.02 pp,
  then a true +0.1 pp mouse-style effect would be unambiguous here — exactly the resolution mouse
  lacked.
- **Cost:** cheap (3–5 baseline runs + 3–5 H02 runs; ~MICrONS-scale Rocket wall-clock, between mouse
  and connectome). **Falsifiable check:** baseline reproduces a stable plateau with σ ≪ mouse.
- **status: proposed**

## H11r — Margin / smooth-hinge surrogate (re-test; prime "lost theory")
- **One-line hypothesis:** replacing σ_β with a bounded smooth-hinge surrogate
  `r=clamp(0.5+(β·Δ)/(2·MARGIN),0,1)` (constant gradient across the correct-but-thin band, then flat)
  raises exact feedforward weight on MICrONS as it did on mouse.
- **Prior per-dataset result (verified):** connectome **Δ=−0.0387 pp** (mean 82.8571±0.0235, n=3;
  log.md L688–690), mouse **Δ=+0.1264 pp** (mean 92.1960±0.2643, n=3; log.md L691–693), findings.md
  #2 row "H11 | margin/hinge surrogate (objective) | −0.0387 | +0.1264 | kill." Killed ONLY by the
  require-both rule: a clean connectome regression vs a positive (sub-threshold) mouse gain. No
  synthetic/Phase-4 run (objective-axis levers were not re-tested in Phase 4).
- **Mechanism for REAL-but-fly-suppressed:** finding #3 established the fly graph's gap is a
  *distributed reordering* with a deep 82.9% attractor where the faithful sigmoid is already
  near-optimal — its converged positions blow up (std≈142, diagnosis), so edges are *already* far
  past the σ_β saturation knee and a wider-gradient hinge adds nothing but drift, hence the −0.04 pp
  regression. The hinge's mechanism (keep gradient on correct-but-thin-margin edges) only pays off on
  graphs where many edges sit NEAR the margin at convergence. The fly graph's hard cyclic core may be
  the *exception* that saturates margins; an ordinary large connectome (MICrONS) may keep many edges
  thin, where the hinge's sustained gradient genuinely widens protective margins. The mouse +0.1264
  hints at this but mouse σ (0.26) is too large to trust at n=3.
- **What MICrONS REVEALS:**
  - GENERAL WIN if MICrONS Δ > 2σ_MICrONS positive (then re-confirm; the fly regression was the
    idiosyncrasy).
  - RECOVERED/GRAPH-DEPENDENT if MICrONS + mouse positive but connectome still negative (a real
    objective-axis effect that fly's saturated basin suppresses).
  - SMALL-GRAPH ARTIFACT if MICrONS ≈ 0 or negative like connectome (the mouse +0.13 was 148-node
    noise; the strongest single test, since MICrONS σ should be tiny).
- **Falsifiable KILL/keep prediction:** lean **kill** (finding #2/#3 predict the faithful surrogate is
  near-optimal at the basin on any large graph) — but this is the single most likely "lost theory"
  because it gained on the one small graph and its mechanism is structure-contingent. KILL if
  MICrONS Δ ≤ 0 beyond noise; KEEP-as-recovered if MICrONS Δ > 2σ positive.
- **Cost:** cheap (one-line surrogate swap in a verbatim `run_rocket` loop; `src/mfas/experiments/H11.py`
  already exists — re-run on MICrONS only; equal budget). Re-test the `tanh` arm only if hinge shows signal.
- **status: proposed**

## H16r — Monotone β-continuation from the H02 warm-start (re-test; mouse cleared its screen)
- **One-line hypothesis:** replacing the cyclic β schedule (which re-melts β→0.05) with a single
  monotone β rise [0.05,1.05], started from H02's greedy-FAS init, holds a better basin on MICrONS
  as it did on mouse.
- **Prior per-dataset result (verified):** connectome H16=82.6269 (deterministic) → **Δ=−0.2688 vs
  baseline / −0.3031 vs H02** (log.md L994); mouse H16=92.6131 → **Δ=+0.5435 vs baseline** /
  +0.1337 vs H02 (log.md L996). **Crucially the mouse +0.5435 vs baseline CLEARED the 0.52 pp mouse
  SCREEN threshold** — H16 was killed purely on the connectome regression, the textbook Phase-5
  "lost theory" pattern. Falsifier outcome recorded: "monotone β re-converged BELOW plateau on
  connectome → 82.9% attractor is intrinsic to the optimizer" (backlog L625).
- **Mechanism for REAL-but-fly-suppressed:** the drift probe (finding #3) showed the fly graph's
  cyclic re-melt actively destroys good orders BUT also that even constant-β collapses to ~82.9% —
  i.e. the fly attractor is so deep that a low-β monotone *start* MELTS the warm-start before it can
  sharpen (H16 connectome ran 82.63%, BELOW even random baseline). On a 2nd large graph whose basin
  is shallower or whose warm-start sits in a wider valley, a monotone never-re-melting schedule
  should *preserve* the H02 init instead of melting it — exactly the mechanism that helped mouse
  (+0.54). MICrONS is the decisive test of whether "no re-melt holds the warm-start" is a real
  large-graph effect or a fly-only melt.
- **What MICrONS REVEALS:**
  - GENERAL WIN if MICrONS Δ-vs-H02 > 0 beyond noise (monotone-β-from-warm-start is a real
    large-graph improvement; the fly melt was the idiosyncrasy).
  - GRAPH-DEPENDENT if MICrONS improves vs baseline but melts vs H02 like connectome did.
  - ARTIFACT/STILL-NULL if MICrONS melts the warm-start too (fly attractor depth is generic to large
    hard graphs) → strongly corroborates finding #3.
  - **Sweep the start-β:** also try a monotone rise that STARTS at the H02 basin's effective β (skip
    the low-β melt phase), since the connectome failure was specifically the low-β start.
- **Falsifiable KILL/keep prediction:** lean **uncertain → possible recover**; the mouse screen-clear
  makes this the highest-information re-test after H11r. KILL if MICrONS melts the warm-start
  (Δ-vs-H02 < 0); KEEP-as-general if Δ-vs-H02 > 2σ on MICrONS.
- **Cost:** cheap (`src/mfas/experiments/H16.py` exists; schedule-array swap + H02 init; equal budget;
  add one start-β-skip arm). Promote vs H02@matched-seeds on each dataset.
- **status: proposed**

## H02r — Greedy-FAS warm-start: test GENERALITY of the confirmed win to a 2nd large connectome
- **One-line hypothesis:** the H02 greedy-FAS warm-start (the ONLY confirmed Phase-3 win) also beats
  random-N(0,1) init on MICrONS at equal compute.
- **Prior per-dataset result (verified):** connectome **Δ=+0.0508 pp** (82.9298±0.0011 vs 82.8790±0.0231,
  n=15, Welch CI lower +0.0391; findings.md #1 hardened row), mouse **Δ=+0.2064 pp** (92.4793±0.0000 vs
  92.2729±0.2000, n=20, CI lower +0.0824; findings.md #1). CONFIRMED on both, survived critic red-team.
- **Mechanism / why MICrONS matters:** finding #1's honest caveat is explicit — "generality beyond
  connectome+mouse is asserted from the mechanism (a graph-derived warm start lands Rocket in a
  better basin), not proven — only two real graphs are available." MICrONS is the FIRST chance to
  test that assertion on a third (and second large) graph. The greedy-FAS order alone scores 68.91%
  (connectome) / 90.13% (mouse) before optimization; its standalone quality on MICrONS, and whether
  the post-Rocket gain survives, directly tests generality of the basin-not-dynamics finding (#2).
- **What MICrONS REVEALS:**
  - GENERAL WIN if MICrONS Δ > 2σ positive → H02 promoted from "win on 2 graphs" to "win on 3,
    including 2 large" — materially strengthens the thesis's only positive result.
  - GRAPH-DEPENDENT (unexpected) if MICrONS shows no gain → would force a caveat that the warm-start
    benefit depends on graph structure (e.g. fly's particular cyclic core), and re-open whether the
    connectome win generalizes.
- **Falsifiable KILL/keep prediction:** lean **strong keep / general win** (a warm-start from a good
  discrete order is the most mechanism-robust, dataset-agnostic lever found). Falsified-as-general if
  MICrONS Δ ≤ 0 beyond noise.
- **Cost:** cheap (`src/mfas/experiments/H02.py` exists; one-time greedy-FAS order O(m log n) + unchanged
  `run_rocket`; equal budget). This run is also produced as part of P00 (H02 is the stacking comparator).
- **status: proposed**

## H03r — Sharper / extended terminal β ramp (re-test)
- **One-line hypothesis:** appending a monotone terminal β-sharpening ramp (β_max up to ~4) tightens
  the surrogate→discrete gap on MICrONS.
- **Prior per-dataset result (verified):** connectome **Δ=−0.0376 pp** (82.8582±0.0181, n=3; log.md
  L354), mouse **Δ=+0.0114 pp** (92.0810±0.2730, n=3; log.md L357). Killed; findings.md #2 row
  "H03 | β schedule (dynamics) | −0.0376 | +0.0114 | kill." Mouse gain is tiny (well within mouse
  noise) — weaker "lost theory" than H11r/H16r, hence lower rank.
- **Mechanism for REAL-but-fly-suppressed:** at β≈1 a unit position gap maps to σ≈0.74, so "weakly
  correct" edges contribute little gradient. The fly graph's converged positions blow up (std≈142),
  so most edges are ALREADY effectively saturated and a terminal β bump only perturbs them
  (−0.04 pp). A 2nd large graph whose positions converge to a *smaller* scale would have more edges
  near the σ knee, where late sharpening genuinely converts surrogate margin to discrete weight.
  MICrONS's converged `pos_std` (log it) is the diagnostic.
- **What MICrONS REVEALS:** GENERAL WIN if Δ>2σ positive; ARTIFACT if ≈0/negative like connectome.
  Distinguishes "terminal sharpening helps" from "fly is already saturated." Un-run arms β_max∈{2,8}
  and RAMP_FRAC sweep become worth one screen if the primary arm shows signal on MICrONS.
- **Falsifiable KILL/keep prediction:** lean **kill** (dynamics knob; finding #2). KEEP-as-recovered
  only if MICrONS + mouse both > 2σ positive.
- **Cost:** cheap (`src/mfas/experiments/H03.py` exists; β-array swap; equal budget).
- **status: proposed**

## H17r — Continuous basin-hopping / parallel tempering from the H02 basin (re-test deferred lever)
- **One-line hypothesis:** perturb→re-optimize→keep-best-by-oracle (basin-hopping), or a few
  β-temperature replicas with swaps, from the H02 basin, beats single-run H02 on MICrONS at equal
  total gradient budget.
- **Prior per-dataset result (verified):** **NOT RUN** — deferred-by-evidence. Backlog L666:
  "lean **kill on connectome, possible keep on mouse**." Phase-4 stop note (log.md L1039–1041)
  deferred it because "the drift probe showed re-optimization flows back to the ~82.9% basin from ANY
  init … naive restarts already KILLED in Phase-3 (H01, +0.0003)." Related prior: H01 multi-start
  connectome +0.0003 / mouse +0.0000 (findings.md #2).
- **Mechanism for REAL-but-fly-suppressed:** the deferral reasoning was explicitly the **fly drift
  probe** — every perturbation falls back into fly's single dominant deep 82.9% attractor. That
  argument is fly-structure-specific: a 2nd large graph with a *shallower or multi-modal* basin
  landscape (which MICrONS, lacking the fly's dense reciprocal cyclic core, plausibly has) would let
  hopping harvest a right tail that the fly graph forbids. The "possible keep on mouse" half of the
  deferral was never tested at low noise; MICrONS supplies a low-σ large-graph version of that test.
- **What MICrONS REVEALS:** GENERAL WIN if best-of-hops > H02 beyond noise on MICrONS (the fly basin
  was uniquely dominant); STILL NULL if hops fall back to the MICrONS plateau (confirms the deep-basin
  property generalizes to large hard graphs). **Cheap-prototype-first:** size σ_pert / R on mouse +
  (re-run) the hard synthetic before any MICrONS compute.
- **Falsifiable KILL/keep prediction:** lean **kill** (deep-basin generality) but the deferral's
  fly-specificity makes it worth ONE prototype-gated screen. KILL if MICrONS best-of-hops ≤ H02
  beyond noise.
- **Cost:** medium (R re-optimizations + R oracle scores; equal total grad steps; `H17.py` not yet
  built). Prototype on mouse first; only spend MICrONS compute if mouse/synthetic show a hop gain.
- **status: proposed**

## H18r — Straight-through estimator: exact discrete forward, surrogate backward (re-test deferred lever)
- **One-line hypothesis:** scoring the exact discrete `1[Δ>0]` in the forward pass while back-propping
  the σ_β gradient (STE) pins the objective to the true metric and beats surrogate-only Rocket on
  MICrONS.
- **Prior per-dataset result (verified):** **NOT RUN** — deferred-by-evidence. Phase-4 stop note
  (log.md L1042): "STE's backward IS the sigmoid-surrogate gradient, so its trajectory ≈ baseline
  Rocket (which already best-by-oracle tracks the discrete score) → ~no change."
- **Mechanism for REAL-but-fly-suppressed:** the deferral assumed STE's descent direction ≈ baseline
  because the backward is identical. But the FORWARD change (exact discrete) reweights *which* edges
  carry reward and specifically zeroes reward for late surrogate progress that does NOT raise the
  discrete score — directly attacking the fly graph's "surrogate descends, discrete flat" step-3
  pathology (diagnosis). That pathology was characterized ONLY on the fly graph; a 2nd large graph
  with a *different* surrogate↔discrete tail (which MICrONS will have) is where the exact forward
  could actually bite. The deferral's "≈baseline" claim is a fly-specific extrapolation.
- **What MICrONS REVEALS:** GENERAL WIN if STE > H02 beyond noise on MICrONS; STILL NULL if it tracks
  baseline (confirms the deferral). Prototype-gate on mouse + the (re-run) hard synthetic first; log
  whether the discrete-tail slope improves.
- **Falsifiable KILL/keep prediction:** lean **kill** (backward unchanged) but prototype decides; the
  forward-pinning mechanism is graph-tail-specific. KILL if no gain over surrogate-only Rocket on the
  hard synthetic.
- **Cost:** cheap-medium (O(m) extra compare/detach per step; `H18.py` not yet built; prototype-gated
  so most cost is mouse + synthetic, not MICrONS).
- **status: proposed**

## H06r — Weight-aware (heavy & borderline) loss reweighting (re-test; weak candidate)
- **One-line hypothesis:** emphasizing heavy-AND-borderline edges in the surrogate
  (`m_e=1+α·ŵ·4σ(1−σ)`) raises retained high-weight feedforward arcs on MICrONS.
- **Prior per-dataset result (verified):** connectome **Δ=−0.0380 pp** (82.8578±0.0288, n=3; log.md
  L603), mouse **Δ=−0.0794 pp** (91.9902±0.1978, n=3; log.md L606). **Regressed BOTH** → a *weak*
  "lost theory" (no dataset gained), hence low rank. findings.md #2 row "H06 | … | −0.0380 | −0.0794 |
  kill."
- **Mechanism for REAL-but-fly-suppressed:** the emphasis is keyed to the edge-weight distribution
  (`ŵ=w/max(w)`). Both prior graphs have extreme weight skew (fly max 2,405); max-normalization means
  one heavy edge dominates `ŵ` and the emphasis biased the surrogate off the faithful Eq.-7 weighting
  on BOTH. MICrONS's synapse-count weight distribution differs (different max/skew), so the *same*
  formula maps to a different emphasis profile — the reweighting could land differently. **Re-size the
  weight distribution on MICrONS first** and tune α to its skew before screening (target-blind: uses
  only input weights).
- **What MICrONS REVEALS:** since it regressed both prior graphs, a MICrONS gain would be a surprising
  RECOVERED result tied to weight structure; a regression confirms the objective-axis null (finding #2).
- **Falsifiable KILL/keep prediction:** lean **kill** (regressed both already; weakest re-test). Only
  KEEP-as-recovered if MICrONS Δ > 2σ positive AND the distribution argument is borne out.
- **Cost:** cheap (`src/mfas/experiments/H06.py` exists; two-line reweight; equal budget).
- **status: proposed**

## NEW H22 — Block / SCC-macro warm-start (stronger discrete-FAS init than H02)
- **One-line hypothesis:** a warm-start from a *coarser* discrete order — condense strongly-connected
  components into super-nodes, order the resulting near-DAG of blocks (topologically / by greedy-FAS),
  then place nodes block-by-block (greedy-FAS *within* each block) — gives Rocket a better basin than
  H02's flat greedy-FAS order on a large graph whose SCC structure is looser than the fly's.
- **Prior result:** NONE (genuinely new). Motivated by Phase-4 finding #3 (fly gap = distributed
  reordering, Kendall-τ 0.61) and the H21 hard-synthetic construction (the gap appears only with dense
  cyclic cores / nested SCCs), and by H02 finding #1 (warm-start = the one lever that works).
- **Mechanism for why a 2nd large graph has headroom H02 lacks:** on the fly graph H02's greedy-FAS
  order is already near the best continuous-reachable basin (finding #3: better-init-alone has a
  ≤0.06 pp ceiling; init→plateau is flat). That flatness may be BECAUSE the fly's single dense cyclic
  core leaves a greedy peel little room to differ from a block order — they nearly coincide. A large
  connectome with *many smaller SCCs* (plausible for cortical-column-structured MICrONS) gives a
  block-macro order genuine structure that flat greedy-FAS scrambles, so the coarse-to-fine init could
  out-perform H02 specifically where H02 saturates. This is the "structure-aware lever motivated by
  how MICrONS differs from the fly" the brief asks for.
- **What MICrONS REVEALS:** GENERAL WIN if H22 > H02 beyond noise on MICrONS (block structure is a
  real init lever the fly's monolithic core hid); STILL NULL if H22 ≈ H02 (greedy-FAS already captures
  the basin on large graphs too). Always also run on connectome + mouse for the both-datasets rule;
  expect H22 ≈ H02 on the fly graph (its core is monolithic) and small/noisy on 148-node mouse.
- **Falsifiable KILL/keep prediction:** lean **uncertain → modest keep on MICrONS if its SCC structure
  is fragmented**; KILL if H22 ≤ H02 on MICrONS (the fly's flat-init ceiling generalizes). **Compute
  SCC count / size distribution on MICrONS first** to predict headroom before building the variant.
- **Cost:** cheap-medium (Tarjan SCC O(m+n) + per-block greedy-FAS, both leakage-safe input-only;
  `H22.py` to be built; one-time init, per-step cost = baseline). Compares vs H02@matched-seeds.
- **status: proposed**

## NEW H23 — Degree/strength-stratified init spread (structure-aware position scale)
- **One-line hypothesis:** seeding initial positions with a spread that reflects each node's
  out-strength−in-strength (sources spread toward the front, sinks toward the back, magnitude scaled
  by strength imbalance) — rather than H02's evenly-spaced ranks or N(0,1) — gives Rocket a basin that
  better respects hub placement on a graph with stable hubs.
- **Prior result:** NONE (new). Related evidence: the seed-stability analysis (CLAUDE.md / experiments)
  found sinks far more stable than sources on the fly graph (Jaccard@1000 ≈ 0.65–0.74 sinks vs
  0.33–0.40 sources); finding #3 found the fly gap is on *median-degree, not hub* endpoints.
- **Mechanism for why a 2nd large graph could reward it:** on the fly graph, sources are unstable and
  the gap is NOT at hubs, so any hub/strength-aware init has little to grab — explaining why
  degree-based inits were weak proxies (backlog H02 rationale L118). If MICrONS has *more stable
  sources / a cleaner strength gradient* (a structure-aware hypothesis to verify on its degree
  distribution), a strength-stratified spread could place hubs better than a flat rank-spread,
  capturing init headroom the fly graph denies.
- **What MICrONS REVEALS:** GENERAL WIN if H23 > H02 beyond noise on MICrONS; STILL NULL if it tracks
  H02 (strength imbalance adds nothing over the greedy-FAS rank order). Diagnostic: compare source/sink
  Jaccard stability on MICrONS vs the fly graph to see if the precondition holds.
- **Falsifiable KILL/keep prediction:** lean **kill / low-EV** (degree proxies were already weak on
  the fly graph) — kept as a cheap structure-aware probe. KILL if H23 ≤ H02 on MICrONS.
- **Cost:** cheap (O(m) strength tally + a deterministic init map; `H23.py` to be built; per-step cost
  = baseline). Compares vs H02@matched-seeds.
- **status: proposed**

## H07r / H08r / H10r / H12r — LOW-EV pure-dynamics knobs (completeness only)
- **One-line hypotheses:** H07 cosine/one-cycle LR; H08 β-phase-synced LR + longer cycles; H10
  per-node / clip-by-value (or no) grad-clip; H12 Polyak/EMA position averaging.
- **Prior per-dataset result (verified):** **NONE run** — all four deferred at the Phase-3 campaign
  stop (backlog L260–286, L328–333, L380–385; log.md L917–919 "LOW-EV pure-dynamics knobs … predicted
  non-improving … deferred, not falsified").
- **Mechanism for REAL-but-fly-suppressed:** weak. Finding #2 ("plateau set by the starting basin, not
  the optimization dynamics") is the *generalization* these would test, but it was inferred from the
  fly + mouse graphs. There is no specific reason any of these path-not-basin knobs would behave
  differently on MICrONS — the H05 optimizer falsifier (the strongest dynamics knob) re-converged to
  the plateau. They exist here ONLY so finding #2 can be re-tested on a 3rd graph for completeness.
- **What MICrONS REVEALS:** almost certainly STILL NULL on all four → corroborates finding #2 on a 2nd
  large graph. A surprise gain on MICrONS would be a genuine (and important) overturn of the central
  inference — which is the only reason to keep them on the list at all.
- **Falsifiable KILL/keep prediction:** lean **kill all four** (finding #2). Run only if budget remains
  after H11r/H16r/H02r/H03r and the higher-EV new variants; pick at most ONE (H12 EMA, near-free) as a
  cheap finding-#2 re-confirmation on MICrONS.
- **Cost:** cheap each (schedule/optimizer/averaging swaps; variants not yet built). Lowest EV/cost in
  the section.
- **status: proposed**

### Phase-5 ranking rationale (EV / cost)
**P00 first** (un-screenable without the MICrONS noise floor + the H02 comparator). Then the two prime
"lost theories" whose mouse gains were killed only by a fly regression: **H11r** (objective-axis,
mouse +0.1264, mechanism is structure-contingent so a 2nd large graph is the ideal discriminator) and
**H16r** (the ONLY killed lever whose mouse gain *cleared its full screen*, +0.5435 vs baseline — the
strongest single "killed only by connectome" case). **H02r** next — cheap, and generalizing the one
confirmed win to a 2nd large connectome is high thesis value. **H03r** (weaker mouse signal) follows.
Then the fly-specific deferrals **H17r/H18r** whose deferral reasoning was tied to the fly drift probe
/ fly surrogate tail — prototype-gated so cheap to falsify. The two NEW structure-aware inits **H22**
(SCC-macro) and **H23** (strength-stratified) probe headroom the fly's monolithic cyclic core may hide
— H22 ranked above H23 because finding #3 directly implicates SCC/cyclic structure as the gap's cause.
**H06r** is a weak re-test (regressed both prior graphs). The four LOW-EV dynamics knobs are last,
kept only to re-confirm finding #2 on a 2nd large graph. Every entry honors the require-improvement-on-
ALL-THREE rule for promotion; a MICrONS+mouse gain with a persistent connectome regression is reported
as a GRAPH-DEPENDENT "recovered theory," not a general win.
