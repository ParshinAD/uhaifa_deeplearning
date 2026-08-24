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
2. **best is not held by gradient flow at the scales tried.** Initialising Rocket *at* the 84.61% order and running it
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

**Honest ceiling read (applies to all H16+).** Because a perfect init is not *reachable* by gradient
flow from a generic start (mechanism: at the achievable β·std the surrogate ranks the better order
lower — see `diagnosis.md` § Q01) and
the gap is a distributed reordering with no tie-slack, **a large share of the 1.69 pp may be
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

---

# Phase-6 backlog (H30–H34) — DISCRETE-REFINEMENT-LED, post-`dr_tmp` deep-research

**Why this section exists.** Phases 3–5 closed the *continuous* search space: the fly graph's
~1.69 pp gap is an OPTIMIZATION-gap (the surrogate ranks the better order above Rocket; the optimizer
fails to reach/hold it — diagnosis #1), the 82.9% attractor is *intrinsic to Adam-on-σ* (H16 monotone-β
re-converged below plateau), and the gap is a **long-range / global** reordering (H22 sizing:
recoverable-edge rank-distance p50 = 22,580 of 136,648; net recoverable within any tractable window
≤ 0). Finding #3 concluded the residual is "irreducible to continuous + **bounded-local** discrete
methods." **That conclusion is correct but scoped to BOUNDED moves** (H22 = ±W window; H04 = barycenter
mean). The single move the gap's structure calls for — **full-range, exact-gain node re-insertion
(sift)** — was never tried. The `dr_tmp` deep-research read-only agent built and ran the sizing
falsifier (`dr_tmp/size_global_discrete.py` → `.json`, CPU-only, ~41 s) and found full-range sift
recovers **+0.71 … +0.98 pp** over H02 on the repo's own gap-bearing hard synthetic (H21) and
**+0.42 pp** over H02 on mouse, while bounded W=10 sift recovers ≈ 0 (reproducing the H22 kill) —
isolating *move range* as the mechanism. Two independent web searches surfaced the current SOTA on
this exact FlyWire graph — Vahidi (2025), arXiv:2506.13799, ~84.6% via greedy + gain-aware insertion
local refinement + SCC, **Python/Colab, no MIP/Gurobi/spectral/trophic/GNN** — i.e. the gap the repo
deemed "needs the 20-day Crane MIP" is reachable by *cheap combinatorial insertion local search*. The
one unexploited lever even vs SOTA is **global / full-range** insertion (Sakuraba–Yagiura *TREE*).

**Source of truth for this section:** `dr_tmp/REPORT.md` (the ranked TOP-5) and
`dr_tmp/size_global_discrete.json` (prototype numbers). These five entries (H30–H34) are a 1:1
transcription of that report's ranked ideas — not new inventions. All prototype numbers below are
cited as **prototype/estimate**, never as harness-measured wins.

**Budget rule (every Phase-6 entry).** ≤ ~1.5× target / ≤ 2× ceiling of the current best Rocket
(~80 s/seed connectome; ~550 s/seed microns at 80k). No O(n²) at n=136k, no MIP/Gurobi, no multi-day
compute. `numba` and `pyamg` are NOT installed — refiners are vectorized-NumPy / `scipy.sparse` only;
any eigensolve must be benchmarked at connectome scale before spending full-run compute.

**Comparators (PROTOCOL §Compute-matched).** The discrete-refinement post-phases (H30/H31/H32/H33)
add **0 gradient steps** to the continuous run, so `total_grad_steps` is *equal* to the comparator's
and the equal-budget basis is preserved. Each must therefore state **both**:
- **Δ vs H02 @ matched-seeds** — H02's greedy-FAS-warm-started Rocket is the SOTA the refiner is
  bolted onto; this is the *promotion* comparison (marginal gain of the refinement).
- **Δ vs `baseline_passthrough`** — random-init Rocket at the same seeds; the total stacked gain
  (informational). For H32/H33 the warm-start *replaces* the init, so the same two comparators apply
  (vs `baseline_passthrough` = total, vs H02 = "does the new global warm-start beat greedy-FAS").
H34 is a continuous surrogate swap → standard knob-swap, screen vs `baseline_passthrough`, promote
vs H02@matched-seeds.

**Leakage rule (every Phase-6 entry, explicit).** The frozen oracle may be used ONLY for
best-by-oracle accept/reject of a **whole** position/order vector *after* it is produced — exactly as
`run_rocket` already best-tracks the discrete score. Every move/seed is chosen from **input edge
weights + current node ranks only** (closed-form Δ = `w_uv − w_vu + …` for sift; degrees/adjacency for
the spectral/trophic warm-starts; positions + input weights for the surrogate). No variant reads
`data/best_solution` (privileged to `mfas.analysis.gap`), folds the oracle *value* into a move choice
/ loss / init, or is dataset-special-cased.

**Prototype-first gate.** Per the H18–H20 precedent, novel/heavier variants run on **CPU mouse + the
hard synthetic (H21)** before ANY connectome/microns compute:
- **H30** is **prototype-POSITIVE already** (the `dr_tmp` falsifier ran it on mouse + synthetic and it
  closed the gap on both) → go straight to in-repo reproduction of the prototype, then connectome.
- **H31** builds on H30 (it wraps H30's move) → unblocked once H30's refiner exists.
- **H32, H33, H34** are **PROTOTYPE-GATED**: earn connectome/microns compute only after closing
  ≥ ~0.1 pp of the synthetic gap AND not regressing mouse, on CPU.

**Measurement note (refinement variants are near-deterministic).** H30–H33 apply a deterministic
exact-Δ refiner to a (near-deterministic from H02) order, so their per-seed variance is tiny — the
2σ-baseline SCREEN is mis-specified for them (the H02/H16 low-variance pattern). Judge them by the
**95% CI-lower-bound CONFIRM directly** (connectome 5 seeds, microns 5 seeds @ 80k, mouse 20 seeds),
not the 2σ screen. Report the **pure-Rocket score and the Rocket+refinement score separately**
(CLAUDE.md rule). The continuous H34 keeps the normal 2σ-screen-then-CONFIRM path.

**Ranked Phase-6 order (highest EV/cost first):**

| rank | id | idea (one line) | axis | est. runtime mult. | prototype signal |
|---|---|---|---|---|---|
| 1 | **H30** | full-range exact-gain node re-insertion (TREE sift) as a Rocket post-phase | discrete refinement | 1.40×/1.19×/0.92× (conn/mic/mouse) | **SCREENED PASS** — +0.86/+0.078/+0.42 pp over H02 (conn/mic/mouse), beats real 82.93% plateau |
| 2 | ~~**H31**~~ | ILS / LNS wrapper on the H30 insertion move (perturb → re-sift → keep-best) | discrete metaheuristic | 1.92× (conn) | **KILLED** — gate-positive on synthetic (+0.25 pp) but Δ vs H30 = −0.0008 pp (conn) / +0.0000 (mouse); ship H30 alone |
| 3 | **H32** | trophic-level Laplacian global warm-start → H30 sift | init (global) + discrete | ~1.1–1.5× | untried (PROTOTYPE-GATED) |
| 4 | **H33** | magnetic-Laplacian directional spectral warm-start → H30 sift | init (global, directional) + discrete | ~1.3–2× | untried (PROTOTYPE-GATED) |
| 5 | **H34** | perturbed/blackbox differentiable SORT surrogate (non-vanishing gradient) | surrogate (continuous) | ~1.5–2× | untried; lowest EV (PROTOTYPE-GATED) |

---

## H30 — Full-range exact-gain node re-insertion (TREE sift) as a Rocket post-phase  [REPORT #1, TOP PICK]
- **One-line hypothesis:** after H02-warm-started Rocket converges, refining the ORDER by repeatedly
  moving each node to its **exact feedforward-weight-maximising rank** given all other nodes fixed
  (sweeping the WHOLE line, not a window) and accepting only strict improvements, recovers a fraction
  of the ~1.69 pp connectome gap that the continuous optimizer cannot. Expected direction: **positive**
  (prototype-positive on mouse + synthetic; honest residual risk on connectome).
- **Exact mechanism:** for each node `v`, the change in feedforward weight from re-inserting it at any
  rank is a step function of its insertion position whose breakpoints are the current ranks of `v`'s
  in/out neighbours; the argmax over the **full line** is computed from `Δ = w_uv − w_vu + …` (incident
  input edge weights + current ranks only) and `v` is moved there iff it strictly improves. Repeat in
  sweeps. Prototype is O(n²)/sweep (fine at n ≤ 400); **connectome requires a Fenwick / balanced-tree
  position index → O(m log m + n log n) per sweep** (Sakuraba–Yagiura *TREE*, Jacobi rebuild),
  implemented in **vectorized NumPy** (numba unavailable), so a sweep ≈ a few Rocket epochs and a
  handful of sweeps adds ~10–40 s (estimate). `src/mfas/experiments/H30.py`: reuse `H02.greedy_fas_order`
  → unchanged `run_rocket` → discrete sift sweeps on the resulting order; report pure-Rocket vs
  Rocket+sift separately.
- **Design axis:** discrete refinement (hybrid continuous → discrete post-phase). Sanctioned by parent
  CLAUDE.md goal #1 ("stronger local search, hybrid discrete+continuous").
- **Comparators (PROTOCOL §Compute-matched):** the refinement adds **0 gradient steps**, so
  `total_grad_steps` equals the comparator's. Report **Δ vs H02 @ matched-seeds** (the promotion
  comparison — marginal gain of the sift) AND **Δ vs `baseline_passthrough`** (total stacked gain).
  Disclose the added non-gradient wall-clock (the sift sweeps).
- **Leakage rule:** oracle used ONLY for best-by-oracle accept/reject of the whole order vector after
  it is produced; every move uses input edge weights + current ranks only (closed-form Δ); never reads
  `data/best_solution`; never dataset-special-cased. Pure Rocket score reported separately.
- **Distinct-from-killed:**
  - vs **H04** (barycenter, killed, 0 moves accepted): barycenter places a node at the *mean* of its
    neighbours' ranks; sift places it at the *argmax of the exact step-function*. For cyclic-core nodes
    (bimodal neighbour ranks) the argmax is far from the mean — the `dr_tmp` falsifier accepted many
    improving moves where barycenter accepted none.
  - vs **H22** (bounded ±W-window sift, killed by long-range sizing): same exact-Δ code, but H22's
    candidate range was ±W ranks and the gap is long-range (p50 = 22,580 ranks), so net ≤ 0 for all
    tractable W. H30's candidate range is the **whole line** — the falsifier shows bounded W=10 ≈ 0
    while full-range ≈ +0.7–1.0 pp on the same graphs, isolating *range* as the mechanism.
- **Expected effect (ESTIMATE — cites `dr_tmp/size_global_discrete.json`, NOT harness-measured):** on
  the hard synthetic (n=400, H21) full-range sift recovers **+0.71 … +0.98 pp** over H02 (s42 +0.714,
  s123 +0.979, s999 +0.709) and *exceeds the high-effort reference order*; on mouse (n=148) **+0.42 pp**
  over H02 (all three seeds 92.479 → ~92.90); bounded W=10 recovers ≈ 0 on both. **Connectome estimate:
  recover a FRACTION of the 1.69 pp** — magnitude unknown and honestly uncertain (synthetic n=400 ≠
  connectome structure; the connectome order may already be closer to a sift fixed point than the
  proxies are). Given connectome 2σ = 0.04 pp / microns 2σ = 0.002 pp, even a small true gain is
  detectable.
- **Est. compute cost:** **cheap–medium.** One-time greedy order (as H02) + Fenwick/TREE sift sweeps
  (~10–40 s added on connectome, estimate) → **~1.1–1.5×** with Rocket; within the ≤2× ceiling. Note
  the falsifier's efficiency upside: `sift on the cheap greedy order with NO Rocket` matched
  sift-on-Rocket on every row, so a *greedy + TREE-sift* path may even be cheaper than baseline Rocket
  while scoring higher (label: estimate). Pure-NumPy may be slower than the C/numba ideal — the 2×
  ceiling is the gate.
- **Prototype-first gate:** **prototype-POSITIVE already** (`dr_tmp` ran it on mouse + synthetic). Plan:
  (1) reproduce the `dr_tmp` prototype numbers in-repo, (2) implement the Fenwick/TREE sift, (3) first
  **connectome** run (the only outstanding decisive compute), then microns. No further CPU gate needed.
- **Measurement:** exact ff% via the frozen oracle, **all three datasets**, refinement is
  near-deterministic → judge by the **95% CI-lower-bound CONFIRM** (connectome 5 seeds, microns 5 @ 80k,
  mouse 20), NOT the 2σ screen. Report pure-Rocket and Rocket+sift side by side per dataset.
- **KILL/keep prediction + cheapest first test:** lean **KEEP** (strongest of the five; positive on
  both available proxies). **Cheapest first test = the sizing falsifier already run**
  (`dr_tmp/size_global_discrete.py`). **Falsified if**, with the Fenwick structure on connectome, a few
  full-range sweeps from the H02 order yield Δ ≤ noise vs H02 — which would mean the connectome order is
  already a full-range-sift fixed point and would *re-strengthen* finding #3.
- **status: confirmed** (2026-06-22) — CONFIRMED GENERAL WIN, promoted to findings.md #4 (verifier 5/5/20 seeds, critic keep). SCREEN PASS on both primaries + mouse non-inferior. Within-run
  pure→refined gain (n=3, seeds 42/123/999): **connectome +0.860 pp** (82.931 → 83.791; beats the REAL
  82.93% H02 plateau — settles the decisive unknown, recovers ~51% of the 1.69 pp gap), **microns +0.078 pp**
  (83.129 → 83.207), **mouse +0.422 pp** (92.479 → 92.902). Δ vs baseline_passthrough +0.895 / +0.090 /
  +0.832 pp. Runtime multiplier vs H02: 1.40× / 1.19× / 0.92× (≤2× ceiling). Refinement near-deterministic
  (std ≤0.004) → 2σ screen uninformative; recommend CI-lower-bound CONFIRM (connectome 5 / microns 5 /
  mouse 20). Gain is GRAPH-DEPENDENT in magnitude (connectome ≫ microns). Revises finding #3's
  "bounded-local" scope on the connectome. Tests `tests/test_refine_insertion.py` 5/5 green;
  prototype `experiments/proto_sift_fullrange.py`. Result ids in `experiments/log.md` (Phase-6 H30 entry).

## H31 — ILS / LNS wrapper on the H30 insertion move (perturb → re-sift → keep-best)  [REPORT #2]
- **One-line hypothesis:** wrapping H30's exact-gain insertion in an Iterated Local Search / Large
  Neighbourhood Search — perturb the current best order (insert-kick, or ruin-and-recreate: remove the
  k nodes carrying the most back-edge weight and re-insert each at its exact-optimal rank), run a short
  sift sweep, keep-best-by-oracle — recovers *more* of the gap than single-pass H30 by reaching the
  coordinated multi-node reorder the gap requires. Expected direction: **positive, smaller than H30**.
- **Exact mechanism:** outer loop of R rounds, each = (a) target-blind perturbation — random insert-kick
  OR a ruin step that removes k nodes selected by *current* back-edge weight (input weights + ranks
  only) — (b) re-insert each removed node at its exact-Δ-optimal rank (the H30 move), (c) one short
  TREE sweep, (d) oracle-score the whole order and keep the global best. Cap R so total added wall-clock
  holds ≤ 2×. `src/mfas/experiments/H31.py`: reuse H02 init + `run_rocket` + the H30 refiner, add the
  ILS/LNS outer loop. This is the recipe that improved all best-known large LOP instances
  (Sakuraba et al. 2015, ILS-on-TREE).
- **Design axis:** discrete metaheuristic (multi-start / large-neighbourhood — *not* continuous).
- **Comparators (PROTOCOL §Compute-matched):** the wrapper adds **0 gradient steps** (it perturbs and
  re-sifts a discrete order; the continuous run is one H02 pass), so `total_grad_steps` equals the
  comparator's. Report **Δ vs H02 @ matched-seeds** (promotion) AND **Δ vs `baseline_passthrough`**
  (total). Also report **Δ vs single-pass H30** (does the wrapper earn its extra wall-clock?). Disclose
  the added non-gradient cost (R sweeps + repairs).
- **Leakage rule:** perturbations are target-blind (random kicks / highest-back-edge-weight ruin from
  input weights + current ranks); repair uses the closed-form Δ; oracle ONLY for whole-vector
  best-by-oracle accept. Never reads `data/best_solution`; never dataset-special-cased.
- **Distinct-from-killed:** vs **H01** (random multi-start of the CONTINUOUS Rocket, killed +0.0003 pp):
  H01 restarted the *continuous* optimizer from fresh random inits under the same dynamics that
  re-converge to the plateau — its "same basin" argument is about continuous flow. H31 perturbs and
  re-optimizes a **discrete order with a discrete move** in an entirely different search space; H01's
  basin argument does not apply. (Also distinct from H17 continuous basin-hopping, which re-runs the
  surrogate optimizer; H31's inner solver is the exact-Δ sift.)
- **Expected effect (ESTIMATE — cites `dr_tmp/size_global_discrete.json`):** on the hard synthetic LNS
  added **+0.20 pp over full sift** (s42 73.692 → 73.896, s123 73.787 → 74.050, s999 76.023 → 76.258);
  on mouse the marginal gain over H30 was small (+0.006 … +0.02 pp). **Connectome estimate: a small
  increment on top of H30's fraction** — genuinely uncertain and bounded by how many kicks fit in the
  2× budget. Most plausible where the gap is a *distributed* (multi-node) reorder, i.e. the fly graph.
- **Est. compute cost:** **medium.** Each round = one short TREE sweep + an O(k·deg) repair; tune R to
  the budget. **~1.3–2×** (estimate). Cap rounds to hold the 2× ceiling.
- **Prototype-first gate:** builds on H30 → unblocked once H30's refiner exists. Before any
  connectome/microns compute, run an **ILS round-count sweep on mouse + synthetic at matched wall-clock**
  to confirm LNS beats single-pass H30 beyond noise; only then spend large-graph compute.
- **Measurement:** exact ff% all three datasets; near-deterministic → **95% CI-lower-bound CONFIRM**
  (connectome 5 / microns 5 @ 80k / mouse 20), not the 2σ screen. Report pure-Rocket, Rocket+H30, and
  Rocket+H31 side by side.
- **KILL/keep prediction + cheapest first test:** lean **keep, smaller than H30**. **First test = the
  LNS column in `dr_tmp/size_global_discrete.py`** (already positive); then the round-count sweep on
  mouse/synthetic at matched compute. **Falsified if** ILS/LNS does not beat single-pass H30 beyond
  noise at matched compute (then ship H30 alone).
- **status: killed**  <!-- 2026-06-22 SCREEN FAIL on the connectome (primary) → KILL, ship H30 alone.
  Built `src/mfas/refine/lns.py` (ruin-&-recreate: top-k highest-current-back-edge-weight victims, target-
  blind, re-inserted at exact gaps via the H30 kernel; short re-sift; best-by-oracle) + `H31.py` (H02→Rocket
  →H30 sift→ils_lns within residual wall, ≤2× ceiling). Tests `tests/test_refine_lns.py` 5/5 (+H30 5/5)
  green. CHEAPEST-FIRST GATE PASSED on the gap-bearing hard synthetic: ils_lns − single-pass sift = mean
  **+0.2483 pp** (3 seeds, matched wall) — reproduces `dr_tmp` +0.20 pp → earned connectome compute. BUT
  SCREEN vs H30 @ matched seeds 42/123/999: **connectome Δ(H31−H30) = −0.0008 pp** (H31 83.7899±0.0132 vs
  H30 83.7907±0.0044; per-seed +0.0098/−0.0203/+0.0083 — straddles 0, one seed regresses, H31 std 3× H30's),
  **mouse +0.0000 pp** (0 LNS accepts, already at the deterministic sift fixed point). Instrumented
  connectome s42: LNS gained only +0.0057 pp over sift in 6 rounds/1 accept (each round = a full 5.66M-edge
  sift sweep), realized multiplier 1.92× (vs H30 ~1.4×). The +0.25 pp synthetic signal does NOT transfer to
  the connectome's distributed reorder within a ≤2× budget. microns NOT run (connectome — a primary —
  already fails the BOTH-primaries screen; compute conserved). Exactly the backlog's falsification clause.
  See experiments/log.md 2026-06-22 H31 cycle; result ids `…-H31-connectome-s{42,123,999}-implement-80c75b`,
  `…-H31-mouse-s{42,123,999}-implement-8a4eac`; gate `experiments/outputs/proto_h31_lns.json`. -->


## H32 — Trophic-level Laplacian global warm-start → H30 sift  [REPORT #3, PROTOTYPE-GATED]
- **One-line hypothesis:** seeding the order from **trophic levels** (one sparse symmetric-Laplacian
  solve giving a least-squares-optimal global hierarchy) instead of greedy-FAS, then refining with the
  H30 sift, reaches a better basin than H02+sift on a graph where greedy peeling under-captures the
  macro order. Expected direction: **positive iff trophic+sift > greedy+sift on the prototype**.
- **Exact mechanism:** compute trophic levels `h` by one sparse solve `Λ h = v` with
  `Λ = diag(w_in + w_out) − (W + Wᵀ)`, `v = w_in − w_out` (MacKay–Johnson–Sansom 2020) via
  `scipy.sparse.linalg.cg` (`pyamg` unavailable → CG, optionally diagonal-preconditioned); order nodes
  by `argsort(h)` (sources → sinks); feed that global order to the H30 refiner (and/or as a Rocket
  init). `src/mfas/experiments/H32.py`. One SDD/Laplacian solve is near-linear → seconds at m = 5.6M.
- **Design axis:** initialization (global linear solve) + discrete refinement.
- **Comparators (PROTOCOL §Compute-matched):** the warm-start *replaces* the init and the sift adds
  **0 gradient steps**, so report **Δ vs `baseline_passthrough`** (total stacked gain) AND **Δ vs H02
  @ matched-seeds** ("does a global trophic warm-start + sift beat greedy-FAS + sift / + Rocket"). Most
  decisive sub-comparison: **trophic+sift vs greedy+sift** (isolates the warm-start's contribution).
- **Leakage rule:** `h` is a function of the input graph only (degrees + adjacency); oracle ONLY for
  whole-vector best-by-oracle; never reads `data/best_solution`; never dataset-special-cased. Sift
  leakage is H30's (closed-form Δ).
- **Distinct-from-killed:** vs **all repo inits** (`random` / `uniform` / `degree_diff` / `degree_abs`
  / H02 greedy-FAS): those are per-node degree imbalances or a local greedy peel; trophic level is a
  *global* linear solve coupling all nodes (the minimiser of a feedback-coherence proxy
  `F₀ = Σ w(h_v − h_u − 1)²`, conceptually the FAS objective). No spectral/trophic method has been run
  on this instance (Vahidi 2025 and the challenge winners are purely combinatorial), so it is genuinely
  untried. Crucially, finding #3's "flat init → plateau (≤0.06 pp)" is *init → run Rocket → plateau*
  (Rocket melts any init); here the trophic order seeds the **discrete refiner**, which *holds and
  improves* a good global order rather than melting it — so the flat-ceiling argument does not bind.
- **Expected effect (ESTIMATE — no prototype number yet; the `dr_tmp` falsifier did NOT run trophic):**
  **uncertain → modest.** Honest range: +0.0 … +0.3 pp over H02+sift on a graph whose macro order
  greedy under-captures; could TIE H30 where greedy already captures the macro order. `F₀` is a
  squared-difference proxy, not the exact weighted feedforward count, so it may not transfer.
- **Est. compute cost:** **cheap–medium.** One sparse CG Laplacian solve (seconds at m = 5.6M) + the
  H30 refiner → **~1.1–1.5×** (estimate). No heavy dependency beyond `scipy.sparse` (confirm CG
  convergence at connectome scale before committing full compute; `pyamg` is unavailable).
- **Prototype-first gate (PROTOTYPE-GATED):** **CPU mouse + hard synthetic (H21) BEFORE any
  connectome/microns compute.** Compute the trophic order, score it raw, score it after the H30
  refiner; **keep only if trophic+sift ≥ greedy+sift beyond noise on the synthetic (gap-bearing)
  without regressing mouse.** Earn large-graph compute only on ≥ ~0.1 pp synthetic-gap improvement
  over greedy+sift.
- **Measurement:** exact ff% all three datasets; near-deterministic → **95% CI-lower-bound CONFIRM**
  (connectome 5 / microns 5 @ 80k / mouse 20), not the 2σ screen. Report raw trophic order pct,
  trophic+sift, and greedy+sift side by side.
- **KILL/keep prediction + cheapest first test:** lean **uncertain → modest keep**. **Cheapest first
  test = the prototype sizing gate** (trophic+sift vs greedy+sift on mouse + synthetic). **Falsified
  if** trophic+sift ≤ greedy+sift on the synthetic (then greedy already captures the macro order and
  trophic adds nothing) → no large-graph compute.
- **status: killed (by prototype gate)** <!-- 2026-06-22 trophic+sift < greedy+sift on BOTH proxies
  (synthetic Δ=−2.12 pp, mouse Δ=−1.26 pp; CG converged, residual ~1e-10). Trophic is a WORSE basin
  than greedy and the sift does not rescue it (captures coarse source→sink axis, not the cyclic core).
  No large-graph compute spent. proto_h32_trophic.{py,json}. -->

## H33 — Magnetic-Laplacian directional spectral warm-start → H30 sift  [REPORT #4, PROTOTYPE-GATED]
- **One-line hypothesis:** seeding the order from the leading eigenvector of the **Hermitian magnetic
  Laplacian** `L_q` (charge `q ≈ 0.25` encodes arc direction as complex phase), then refining with the
  H30 sift, reaches a better basin than the trophic warm-start (H32) on a graph whose ranking signal
  is carried by *edge direction itself* rather than net in/out imbalance. Expected direction:
  **positive only if it beats H32**.
- **Exact mechanism:** build `L_q` (direction → complex phase), represent the n×n complex-Hermitian
  operator as a **2n×2n real-symmetric** one, take the leading eigenpair via
  `scipy.sparse.linalg.eigsh` / `lobpcg` (no AMG — `pyamg` unavailable), order nodes by the de-rotated
  eigenvector phases, feed that global order to the H30 refiner. `src/mfas/experiments/H33.py`.
  Convergence of the leading eigenpair is spectral-gap-dependent.
- **Design axis:** initialization (global, *direction-aware*) + discrete refinement.
- **Comparators (PROTOCOL §Compute-matched):** as H32 — eigensolve replaces the init, sift adds **0
  gradient steps**: report **Δ vs `baseline_passthrough`** (total) AND **Δ vs H02 @ matched-seeds**
  (promotion). Decisive sub-comparison: **magnetic+sift vs trophic+sift (H32)** — it must beat H32 to
  justify the more fragile, slower eigensolve.
- **Leakage rule:** eigenvector of an operator built only from the input graph; oracle ONLY for
  whole-vector best-by-oracle; never reads `data/best_solution`; never dataset-special-cased. Sift
  leakage is H30's.
- **Distinct-from-killed:** an entirely new axis vs the repo's inits (none are spectral); distinct from
  **H32** (directional complex phase vs undirected net-imbalance solve) and from undirected Fiedler /
  spectral seriation (which *discards* edge direction — the FAS signal — and is therefore down-ranked
  in the report). Same "seed the discrete refiner, not Rocket" logic as H32 sidesteps the flat
  init → plateau ceiling.
- **Expected effect (ESTIMATE — no prototype number; not run by the `dr_tmp` falsifier):** **uncertain.**
  The most theoretically apt *directional* spectral seed, but if it merely ties H32 it is the worse
  EV/cost (more fragile, slower). Honest range: +0.0 … +0.3 pp over H02+sift, *conditional* on beating
  H32.
- **Est. compute cost:** **medium.** One sparse leading-eigenvector solve, near-linear per matvec but
  iteration-count-sensitive, + the H30 refiner → **~1.3–2×** (estimate) — **must be measured.**
  **Requires a connectome-scale eigensolve TIMING benchmark** (the eigensolve alone, no full run)
  before any full compute; **drop in favour of H32 if it cannot hit ≤ 2× or does not beat H32.**
- **Prototype-first gate (PROTOTYPE-GATED):** **CPU mouse + hard synthetic (H21) BEFORE any
  connectome/microns compute** (same sizing gate as H32: magnetic+sift vs trophic+sift vs greedy+sift),
  **PLUS** the connectome-scale eigensolve timing micro-benchmark. Earn large-graph compute only if
  magnetic+sift > trophic+sift on the synthetic AND the eigensolve hits ≤ 2×.
- **Measurement:** exact ff% all three datasets; near-deterministic → **95% CI-lower-bound CONFIRM**
  (connectome 5 / microns 5 @ 80k / mouse 20), not the 2σ screen. Report raw magnetic order pct,
  magnetic+sift, and the H32/greedy+sift comparators side by side; log eigensolve wall-clock.
- **KILL/keep prediction + cheapest first test:** lean **keep only if it beats H32.** **Cheapest first
  test = the prototype sizing gate + the eigensolve timing micro-benchmark.** **Falsified if**
  magnetic+sift ≤ trophic+sift on the synthetic, OR the eigensolve cannot hit the time budget.
- **status: killed (by prototype gate)** <!-- 2026-06-22 fails BOTH gates (either sufficient):
  (1) QUALITY: magnetic+sift < greedy+sift (synthetic Δ=−4.97 pp, mouse Δ=−1.49 pp) — directional
  spectral order is a weaker warm-start than greedy. (2) TIMING: connectome eigensolve ALONE = 304s
  (k=4) / 507s (k=2), ~5–8× over the ≤60s budget without AMG (pyamg absent). proto_h33_magnetic.{py,json}. -->

## H34 — Perturbed / blackbox differentiable SORT surrogate (non-vanishing gradient)  [REPORT #5, lowest EV, PROTOTYPE-GATED]
- **One-line hypothesis:** replacing Rocket's sigmoid (and the killed soft-rank H19) with a surrogate
  whose gradient comes from **perturb-and-MAP over a SORT** (Berthet 2020) or a **blackbox-solver
  interpolation** (Vlastelica 2020) — gradient magnitude set by the perturbation ε/λ, *independent of
  n* — lets the optimizer keep refining the order instead of stalling, beating raw-position Rocket on a
  graph that has a gap. Expected direction: **positive if any continuous lever works; lean falsified.**
- **Exact mechanism:** the feedforward weight `Σ w · 1[order(u) < order(v)]` is linear in the order
  indicator, so its gradient is obtained from 1–8 extra **sort** calls per step (perturb-and-MAP /
  blackbox interpolation); the inner solver MUST be a sort (O(n log n), n = 136k is cheap) — **never a
  greedy-FAS pass** (20k greedy passes would blow the budget). Pair with a graduated sharpness (ε)
  schedule. H02 init. `src/mfas/experiments/H34.py`. Blondel fast-soft-sort (O(n log n), ε decoupled
  from n) is the fallback drop-in if perturbation overhead is too high.
- **Design axis:** surrogate / relaxation (continuous), with a graduated ε-schedule.
- **Comparators (PROTOCOL §Compute-matched):** standard continuous knob-swap at the same gradient
  budget → screen vs **`baseline_passthrough`**, promote vs **H02 @ matched-seeds**. (Continuous → keep
  the normal 2σ-screen-then-CONFIRM path, unlike the near-deterministic H30–H33.)
- **Leakage rule:** the surrogate is a function of positions + input weights; the inner sort sees only
  the *current* positions; oracle ONLY for whole-vector best-by-oracle. Never reads
  `data/best_solution`; never dataset-special-cased.
- **Distinct-from-killed:**
  - vs **H19 soft-rank** (killed): same rank-space idea but a fundamentally different, **non-vanishing**
    gradient estimator. H19 died from **O(1/n) vanishing gradients** (normalized rank gaps → σ gradient
    too flat → optimizer stalled exactly at the H02 init, robust across α). Perturbed/blackbox
    differentiation is the literature's *structural* fix: the gradient does NOT scale with n.
  - vs **H11 hinge** (killed): the hinge kept a saturating *analytic* gradient on raw positions; H34
    changes *where the gradient comes from* (perturb-and-MAP over a sort), not just its shape.
- **Expected effect (ESTIMATE — no prototype number; not run by the `dr_tmp` falsifier):** **uncertain,
  lean small/null.** The drift probe (diagnosis #2) shows continuous gradient flow collapses even a
  *perfect* order to the 82.9% basin — which may hold regardless of the gradient *source*. This is the
  best *continuous* bet but the diagnosis is a strong prior against any continuous lever; ranked last
  of the five and included for axis-completeness (the only untried continuous mechanism).
- **Est. compute cost:** **medium.** 1–8 sorts/step (O(n log n)) → **~1.5–2×** (estimate). Blondel
  fast-soft-sort is the cheaper fallback.
- **Prototype-first gate (PROTOTYPE-GATED):** **CPU mouse + hard synthetic (H21) BEFORE any
  connectome/microns compute.** On the synthetic (which *has* a gap), does the perturbed-sort surrogate
  exceed raw-position Rocket beyond noise across ε? Earn large-graph compute only on ≥ ~0.1 pp synthetic
  gap closure without mouse regression.
- **Measurement:** exact ff% all three datasets; continuous variant → standard **2σ SCREEN** (3 seeds)
  then **CONFIRM** (connectome 5 / microns 5 @ 80k / mouse 20, 95% CI lower bound > 0). Sweep ε; log
  converged `pos`/`rank` std to confirm the gradient stays non-vanishing.
- **KILL/keep prediction + cheapest first test:** lean **kill (prototype decides).** **Cheapest first
  test = the hard-synthetic ε sweep on CPU.** **Falsified if** it stalls like H19 did across ε (then
  continuous is truly exhausted on this problem — itself a clean, citable thesis result).
- **status: killed (by prototype gate) — FALSIFIED (continuous exhausted)** <!-- 2026-06-22 The
  perturb-and-MAP-over-sort surrogate (Berthet 2020) does NOT stall (unlike H19: 398/400 nodes moved,
  init ~50%→~73%, gradient genuinely non-vanishing) yet still LOSES to sigmoid Rocket on the gap-bearing
  synthetic (Δ=−0.64 pp, all 3 seeds). Confirms finding #3: the bottleneck is the continuous-relaxation
  BASIN, not the gradient estimator. The last untried continuous mechanism is exhausted.
  proto_h34_perturbsort.{py,json}. -->

### Phase-6 ranking rationale (EV / cost)
**H30 first** — the only prototype-POSITIVE idea (full-range sift closed the gap on BOTH available
proxies, mouse + hard-synthetic, in `dr_tmp`), the cheapest large-graph lever (~1.1–1.5×, adds 0
gradient steps), and the move the gap's *long-range* structure (H22 sizing: p50 = 22,580 ranks) calls
for; its only outstanding compute is the first connectome run. **H31** builds directly on H30's move
(ILS/LNS for the coordinated multi-node tail) and was +0.20 pp over H30 on the synthetic — gated by a
matched-compute round-count sweep so it ships only if it beats single-pass H30. **H32 (trophic)** and
**H33 (magnetic)** are the two cheap *global* warm-starts for axis diversity (a genuinely new init axis
vs every killed/confirmed repo init), both PROTOTYPE-GATED on the cheap synthetic sizing gate before
any large-graph compute; H32 ranks above H33 because the trophic solve is cheaper and more robust than
a spectral-gap-dependent eigensolve (H33 must additionally pass a connectome-scale timing benchmark and
beat H32 to survive). **H34** is last — the only untried *continuous* mechanism (non-vanishing
perturbed-sort gradient, the structural fix for H19's O(1/n) stall), but the drift-probe diagnosis is a
strong prior that *any* continuous lever re-converges to the 82.9% basin, so its EV is lowest. Across
all five: every entry reports the **pure-Rocket score separately** from the refinement (CLAUDE.md),
honors the require-improvement-on-ALL-THREE-datasets promotion rule, and uses the oracle only for
best-by-oracle whole-vector acceptance.

---

# Phase 6.2 backlog (H35) — dynamics of the discrete refiner (2026-06-23)

## H35 — under-relaxed two-phase exact-gain sift (break the Jacobi limit cycle)
- **One-line hypothesis:** H30's sift is a *Jacobi* iteration that does NOT converge on the large
  dense connectomes (period-2 limit cycle); **under-relaxing** the move (`key = rank + α·(best_gap −
  rank)`, `α=0.7` after `k_full` full warm sweeps) breaks the cycle, the iterate converges, and the
  refined order beats H30's Jacobi sift at equal gradient budget. Expected direction: **positive on
  the large dense connectomes, non-regressing on small/already-converged graphs (mouse).**
- **Design:** `src/mfas/refine/underrelax.py::sift_underrelaxed` (reuses the verified
  `jacobi_best_gaps` kernel; α=1 is bit-identical to `insertion.sift`); variant
  `src/mfas/experiments/H35.py` chains H02 init + `run_rocket` + the under-relaxed sift. H30
  `_MAX_SWEEPS` raised 12→40 (connectome/mouse) in the same commit; H35 matches caps so the
  comparison isolates α. Comparators: H30@40 (isolate α), H02 (sift increment), baseline_passthrough.
- **status: CONFIRMED (connectome) / NOT a general win** <!-- 2026-06-23 connectome Δ(H35−H30@40) =
  +0.098 pp (CI_lo +0.076, 3 seeds, all per-seed +); mouse Δ=0 (non-regressing); microns Δ=−0.0019 pp
  at the 12-sweep cap (under-relaxation needs ~30 sweeps to overtake Jacobi on microns, dr_tmp). Win
  on the fly connectome (82.93%→83.91%); microns-inferior at the reduced budget. See findings.md #5,
  log.md Phase 6.2. -->

### Open follow-up (deferred)
- **H36 (proposed):** cycle-triggered α — run pure Jacobi until oscillation is detected (candidate
  dips while movers plateau), THEN switch to under-relaxation. Would make H35 a clean 3-dataset
  general win (microns no longer penalized by a fixed short budget). Untried.

---

# Phase 6.3 backlog (H36) — collective (multi-node) discrete moves (2026-08-09)

Opened by the S1/S2 sizing gate (`experiments/log.md` 2026-08-09; probe
`experiments/size_collective_moves.py`, artifacts `experiments/outputs/collective_moves_sizing.json`
+ `experiments/outputs/siftfirst_*.json`). Closes the roadmap's `A-SCC` question and opens `A-PAIR`.

## H36 — Collective discrete refinement: paired backward-edge relocation + block-SCC condensation
- **Hypothesis:** After H35, appending a **collective** (multi-node) discrete phase — alternating
  (a) Vahidi-Alg-2 **paired relocation** of a backward edge's two endpoints over the whole span
  between them, and (b) Vahidi-Alg-3 **block-SCC** condensation refinement on contiguous rank
  windows — raises the exact feedforward metric beyond H35, at **0 extra gradient steps**.
- **Rationale (measured, not assumed):** H35's sift is a *single-node* move. With the single-node
  class first exhausted to a fixed point as a control, the collective phase still gains
  **connectome +0.0599 pp (seed 42; 94.5% of the total is outside the single-node class)** and
  **microns +0.00601 ± 0.00025 pp (n=3, 95% CI lower +0.00561, vs the 0.002 pp gate)**. Mouse gains
  +0.0539 pp, 100% of it from the paired move. Every move's gain is exact by the contiguous-interval
  lemma and oracle-verified (`experiments/diagnostics/verify_collective_moves.py`: max error 1.6e-14).
- **Design axis:** discrete refinement, collective move class (post-Rocket, post-sift).
- **Expected effect:** ≥ the sized numbers — the sizing is a **lower bound**: it examined only the
  top 200k of ~1.19M backward edges (≈50% of connectome backward weight), and a full-K single S1
  pass gains **2.5×** more (+0.04293 vs +0.01736 pp). Round count (6) had not converged on connectome.
- **Est. compute cost:** **expensive in wall-clock, free in gradient steps.** The sized configuration
  costs 359 s of collective rounds + 204 s of control on top of H35's 219 s on connectome = **~2.6×**,
  which **breaches the campaign's ~2× wall ceiling** — declare this up front and consider it a
  budget knob (rounds, top-K) rather than a fixed cost.
- **Comparator:** **H35 at matched seeds** (the incumbent), not `baseline_passthrough`. Report the
  pure-Rocket score, the H35 score and the H36 score separately per CLAUDE.md.
- **Leakage:** safe. Gains use ranks + input edge weights only; the frozen oracle only accepts/rejects
  whole candidate vectors. The probe never reads `data/best_solution`.
- **Measurement:** standard — both large connectomes + mouse, ≥3 seeds SCREEN, CONFIRM at 5/20 with
  95% CI lower bound > 0 vs H35.
- **Spec requirements carried from the critic (do not drop):**
  1. Sift to a 1-opt fixed point **before and between** collective rounds — H35's returned order is
     NOT a fixed point (163 / 608 / 0 movers left), so without this the gain is mis-attributed
     (on microns the uncontrolled headline was ~60% single-node).
  2. Report K-sensitivity; top-K truncation under-states the paired move by ~2.5×.
  3. A monotone best-by-oracle refinement "beating the incumbent" is near-tautological — the real
     test is magnitude vs seed noise on ≥3 seeds, and the wall-clock cost.
- **status: open** — promoted by the sizing gate, not yet run as a variant cycle.

---

# Phase 6.4 backlog (A-INIT) — the initialization SCALE axis (2026-08-15)

Opened by the roadmap's `A-INIT` (TODO 5). The roadmap listed it as **low priority, "doubly
discouraged"**; the prototype gate **falsified that prior** on both proxies. Theory derived first,
then measured: `experiments/proto_ainit_scale.py` → `experiments/outputs/proto_ainit_scale.json`,
`proto_ainit_theory.png`, `proto_ainit_scale.png`.

## A-INIT — very tight position initialization (`src/mfas/experiments/A_INIT.py`, std = 1e-4)
- **One-line hypothesis:** shrinking the paper's `N(0,1)` init to `std = 1e-4` — nothing else
  changed — beats `baseline_passthrough` on the exact metric AND removes its seed variance,
  because at small `beta*std` the gradient collapses to the weight-imbalance vector.
- **Theory (derived, then verified numerically).** With `sigma(x) = 1/2 + x/4 - x^3/48 + O(x^5)`
  and `c_k = out_w_hat(k) - in_w_hat(k)`:

      F(P) = W_hat/2 - (beta/4)<c,P> - (beta^3/48) sum_e w_hat_e D_e^3 + O((beta*D)^5)
      grad_k F = -(beta/4) c_k + O((beta*std)^2)

  Every *pairwise* (who-precedes-whom) term is suppressed as **(beta*std)^2**. Measured
  log-log slope of the residual: **2.006** (mouse) / **1.996** (hard synthetic); predicted 2.
  `cos(grad F, -c) = 1.000000` at `beta*std = 1e-6`. This is Q01's scale-blindness seen from the
  gradient side rather than from `F`.
- **Mechanism (measured, not assumed).** Because the small-scale gradient is a *constant vector*,
  Adam (which normalizes per coordinate) keeps only `sign(c)`: the cloud splits into the two
  imbalance blocks — the `sign(c)` split explains **96.5%** (mouse) / **93.2%** (synthetic) of
  position variance after 20 steps — and then refines within blocks as the cubic term wakes up.
  So Rocket **self-warm-starts from the imbalance signal** (the quantity GreedyAbs ranks on)
  instead of spending steps undoing an uninformative random draw. The init is *forgotten*:
  Spearman(final, init) = +0.03 / -0.02 at std <= 1e-4 vs +0.14 / +0.12 at std = 1, and
  Spearman(final, -c) = +0.67 / +0.75.
- **Prototype evidence (3 seeds, Rocket only, exact oracle):**

  | proxy | std<=1e-4 | baseline std=1 | delta |
  |---|---|---|---|
  | mouse | **92.4359 ± 0.0000** | 92.1451 ± 0.2616 | **+0.2908 pp** |
  | hard synthetic | **73.8395 ± 0.0000** | 73.5348 ± 0.2302 | **+0.3047 pp** |

  The sweep is FLAT over `std ∈ [1e-6, 1e-2]` and degrades monotonically above it
  (std=10 → −0.68 / −1.44 pp; std=100 → −13.6 / −12.0 pp), i.e. the effect is the *scale*, not a
  lucky constant. **Zero seed variance** is the theory's signature, not a fluke of 3 seeds.
- **Interaction with our findings (measured in the same run).**
  1. **It is an ALTERNATIVE to H02, never an addition.** Compressing H02's greedy warm start to
     std=1e-4 lands on *exactly* the tight-random score (mouse 92.4359 for both) — the same
     mechanism that forgets a random init forgets a good one. H02 at its native std (0.577) still
     wins (92.4793), so the confirmed win #1 is not threatened.
  2. **The sift erases the whole axis.** After the H35 under-relaxed sift the spread across ALL
     inits collapses from 0.4375 → **0.0483 pp** (mouse) and 1.3712 → 0.3621 pp (synthetic).
     This is a third, independent confirmation of the Phase-6 cross-cutting insight (the discrete
     refiner does the work) — and it means A-INIT cannot help the **champion** pipeline.
- **What it is worth, honestly.** A free (0 extra gradient steps, 0 wall-clock, no greedy peel)
  improvement to the *baseline*, and a **determinism** property that no other variant has. It is
  NOT a route to the 84.61% target: it lives entirely below the sift.
- **Comparator:** `baseline_passthrough` at matched seeds (this is a baseline-level knob, so the
  champion comparator does not apply). Report vs H02 as context.
- **Leakage:** trivially safe — the init is a scaled Gaussian; it reads neither the graph nor the
  oracle. Compute-matched by construction (same `epochs`, same draw, one scalar factor).
- **status: prototype PASS, screen PAUSED at 8/9 runs** <!-- 2026-08-16 theory verified (slope
  2.006/1.996); proxies +0.29/+0.30 pp with zero seed variance; falsifies the roadmap's "doubly
  discouraged" prior and diagnosis.md Q01 insight #2's prediction that a tight init is worse.
  Screen so far: connectome +0.0514 pp (n=3, gate 0.04) PASS, microns +0.0069 pp (n=2 of 3, gate
  0.002) PASS-pending, mouse +0.3663 pp non-inferior. Paused on battery with microns s999
  outstanding; resume state + the one command in experiments/ainit_RESUME.md. No log.md entry
  until the screen is complete. -->

---

# Phase 6.5 backlog (A-SURR) — the surrogate TAIL-EXPONENT axis (2026-08-17)

Opened by the roadmap's `A-SURR` (TODO 7: "alternate surrogate — flat top/bottom, slower-decaying
tanh"). The roadmap listed it **low / near-dead** by analogy to H11 and H34. Analogy is not
evidence, so it was run properly: theory gate first (Q04), then a prototype gate, then the
primary dataset. All three agree — but the analogy was right for the wrong reason, and the
mechanism that emerged is new (see `diagnosis.md` § Q04).

## H37 / H37B — algebraic-tail surrogate (`src/mfas/experiments/H37{,B}.py`)
- **One-line hypothesis:** replacing the sigmoid's EXPONENTIAL tail with an ALGEBRAIC one,
  `g(z) = 1/2 + 1/2 sign(z)(1 - (1+|z|)^-4)`, raises the exact feedforward metric, because the
  long-range pairs the sigmoid sends to (numerically) zero force keep a correctly-signed pull —
  and finding #3 says the recoverable weight IS long-range (flip rank-distance p50 ≈ 22,580).
- **Two arms, because "shape" hides two axes.** `width(g)` := the `z` at which `g` reaches 0.9.
  - **H37** — `SCALE = 4.4361` so `width` equals the sigmoid's **exactly** (2.1973). Isolates
    the TAIL with core sharpness held fixed. This is the honest test of the new axis.
  - **H37B** — `SCALE = 1` (native width 0.4954). The arm the static theory predicted would
    win: the ONLY shape measured that ranks the near-optimal order ABOVE Rocket's own at
    Rocket's operating scale (alignment ratio **+0.173** vs the sigmoid's **−0.591**) while
    leaving essentially every node mobile (0.006% zero gradients vs 42.5% for a width-matched
    sigmoid).
- **Why the literal reading of TODO 7 is NOT this hypothesis (proved, not argued):**
  `(tanh(z/2)+1)/2 == sigmoid(z)` to **2.2e-16**, so a "tanh surrogate" IS the sigmoid and a
  "slower-decaying tanh" `tanh(z/a)` is the sigmoid at `beta·2/a` — a pure move along the
  `beta*std` axis that H03/A-SCALE already killed (and it makes the surrogate's ranking worse:
  alignment −1.90 at `a=10`). A HARD flat top/bottom is the H11 form, already killed. Only the
  tail EXPONENT was genuinely untested. See `diagnosis.md` § Q04.
- **Theory gate (Q04): PASS** — `experiments/outputs/q04_surrogate_tails.json`. Established the
  ~200×width alignment law across 11 shapes and the 7,000× difference in frozen-node fraction
  between an exponential and an algebraic tail at matched width.
- **Prototype gate: FAIL (all arms).** 3 seeds, everything else identical to baseline
  (`experiments/outputs/proto_h37_tails.json`), Δ vs sigmoid in pp:

  | arm | mouse | hard synthetic |
  |---|---|---|
  | H37B `poly z^-4` native | −0.220 | **−2.834** |
  | sigmoid @ matched width (H03 control) | +0.004 | **−2.117** |
  | H37 `poly z^-4` @ matched width | −0.018 | −0.459 |
  | `poly z^-1` | −0.130 | −0.708 |
  | `poly z^-1` @ matched width | −0.101 | −1.686 |
  | H11 clamp M=5 (anchor) | −0.030 | +0.107 *(within noise, σ≈0.38)* |

  All four pre-registered predictions were recorded before the run; **P1 (`poly_q4` > sigmoid)
  FAILED on both proxies**, P2 held on both, P3 held decisively on the synthetic.
- **Primary-dataset confirmation of the kill (connectome, 3 seeds, frozen runner, role
  `implement`, budget-matched `baseline_passthrough` = 82.8958 ± 0.0187):**

  | variant | per-seed | mean ± std | Δ | 95% CI lo | screen (gate +0.04) |
  |---|---|---|---|---|---|
  | H37 | 82.4152 / 82.4492 / 82.4707 | 82.4450 ± 0.0280 | **−0.4508** | −0.4807 | **FAIL** |
  | H37B | 81.9727 / 81.9355 / 81.9828 | 81.9637 ± 0.0249 | **−0.9321** | −0.9621 | **FAIL** |

  Every one of the 6 per-seed deltas is negative; the miss is 11× (H37) and 23× (H37B) the
  screen threshold **in the wrong direction**.
- **status: KILLED** (theory gate PASS → prototype gate FAIL → primary-dataset FAIL). Not run on
  microns or mouse through the frozen runner: the protocol kills a variant that fails the
  connectome screen, and mouse was already covered at the prototype gate.
- **What it bought (the reason this was worth running).** A genuinely new, artifact-backed
  mechanism, written up as `diagnosis.md` § Q04 and finding #3's Phase-6.5 corroboration:
  **static surrogate alignment ANTI-correlates with achieved score** — the two shapes whose
  surrogate ranks the better order higher are the two worst optimizers (−2.1 / −2.8 pp on the
  gap-bearing fixture). Alignment requires a narrow core, a narrow core is a short-range
  interaction, and short-range interactions cannot perform the long-range reordering the gap
  consists of. This also explains the H11 kill mechanistically (its clamp is the worst cell in
  the table: alignment −1.063 AND 65.8% frozen nodes) and closes the continuous family's last
  untested axis.
- **Leakage:** safe. Both variants read only positions, `beta` and the input `hat_w`; the frozen
  oracle is used exactly as baseline (best-by-oracle tracking). Only the Q04 *diagnostic* reads
  `data/best_solution`, via `mfas.analysis.gap`. Frozen-file integrity verified before and after.
- **Compute honesty:** compute-matched on the protocol's basis (`total_grad_steps` = 20,000 for
  both arms and the comparator). **Wall-clock is NOT matched and must be disclosed:** ~161–190 s
  vs the baseline's ~90 s, because `(1+|z|)^-4` costs more per step than `sigmoid`. Since the
  arms lost, the wall-clock penalty only strengthens the kill.

---

# Phase 6.6 backlog (A-SURR, part 2) — the SYMMETRY axis (2026-08-17)

H37 closed the tail exponent *within odd-symmetric shapes*. The researcher then asked the
question that turned the cycle around: **what if the surrogate is constant on the positive
branch and tanh on the negative one?** Every shape tested up to that point satisfied
`g(-z) = 1 - g(z)`; dropping that assumption is a different move, and it works on the fly
connectome.

## H38 — one-sided (asymmetric) surrogate: flat above a margin, tanh below
- **One-line hypothesis:** with `g(z) = 1` for `z >= M` and `g(z) = 1 + tanh((z-M)/T)` for
  `z < M`, comfortably-feedforward edges receive **zero** gradient, so the entire budget pulls
  FEEDBACK edges toward correctness instead of widening margins that are already won.
- **Theory gate (Q05): PASS, on two structural grounds no symmetric shape has**
  (`experiments/outputs/q05_asymmetric_surrogates.json`):
  1. **It does not telescope.** Q01's small-scale degeneracy (the surrogate collapsing into the
     node-level imbalance objective `W/2 - (beta/4)<c,P>`) requires the edge sum to run over ALL
     edges; here the first-order term runs over the VIOLATED subset, which is order-dependent.
     Measured: at `beta*std -> 0` this shape ranks `best > rocket > imbalance_sort > random`
     (correct) where all 11 symmetric shapes rank the imbalance sort first.
  2. **Alignment ratio +1.48 … +2.00** at Rocket's operating point (sigmoid: −0.591), with **no
     crossover anywhere** in `beta*std` ∈ [1e-2, 1e6] — it never prefers the worse order.
- **A degeneracy was derived BEFORE any run, then confirmed.** At `M = 0`, `g(0) = 1`, so the
  collapsed configuration `P = const` attains `F = sum_e w_hat_e`, the surrogate's **global
  maximum**, strictly above every ordering — and the dynamics flow into it (a violated edge pulls
  its endpoints together). Confirmed numerically: `F(collapse) == W_hat` exactly, and the `M = 0`
  arm collapses to final position std **0.0005** on the hard synthetic, scoring 58.24% vs the
  sigmoid's 73.68%. The margin `M > 0` is what removes it (`g(0) = 0.5379 < 1`).
- **Prototype gate: PASS.** 6×3 grid over `(M, T)` on mouse + hard synthetic
  (`proto_h38_asym.json`, `proto_h38_sweep.json`) shows a **plateau, not a knife edge** (a broad
  positive ridge on mouse). `(M, T) = (0.75, 1.5)` was selected as **the only grid point positive
  on BOTH proxies** (mouse +0.396, synthetic +0.404), not the mouse-optimal (0.25, 0.75) which is
  +0.627 on mouse but −6.30 on the synthetic. **The optimum is graph-dependent — disclosed.**
- **Primary datasets (frozen runner, 3 seeds, role `implement`, budget-matched):**

  | dataset | baseline | H38 | Δ | gate | verdict |
  |---|---|---|---|---|---|
  | **connectome** | 82.8958 ± 0.0187 | **83.2626 ± 0.0108** | **+0.3668** (CI_lo +0.3368) | +0.04 | **PASS, 9×** |
  | **microns** | 83.1172 ± 0.0006 | **82.4482 ± 0.0094** | **−0.6689** (CI_lo −0.6700) | +0.002 | **FAIL (regression)** |
  | mouse | 92.0696 ± 0.2624 | 92.2053 ± 0.2602 | +0.1357 | > −0.26 | non-inferior ✓ |

  **Verdict per the Phase-5 decision table (microns ✗ / connectome ✓ / mouse ✓):
  GRAPH-DEPENDENT — a real fly-connectome effect, NOT a general win.**
  All three connectome seeds positive (+0.354/+0.358/+0.389). For scale: the only previously
  CONFIRMED pure-Rocket win, H02, is +0.0508 pp on connectome — H38 is **7× larger**, and at
  83.26% pure Rocket it exceeds H02's 82.93% by +0.33 pp.
- **Controls — both decisive, both run on the connectome at 3 seeds:**

  | control | Δ vs baseline | what it rules out |
  |---|---|---|
  | **H38C** mirror (flat on the FEEDBACK side) | **−2.8890** | not "any one-sided shape", and not the smaller position scale such a shape induces — the **direction** of the asymmetry is the mechanism |
  | **H38D** plain sigmoid at `beta × 4`, whole run | **−0.5804** | not the `beta`/position-scale axis (A-SCALE ≡ H03). Stronger than H03 itself, which only ramped beta over the last 25% of epochs |

- **Not a best-by-oracle sampling artifact:** the gain is the same on the FINAL-epoch score as on
  the tracked best (mouse +0.392 vs +0.396; synthetic +0.360 vs +0.404).
- **Leakage:** safe — `grep` clean for `best_solution` / `analysis.gap` / any target constant;
  imports are `baseline.rocket`, `io`, `metrics` only; the frozen oracle is used exactly as the
  baseline uses it. Frozen integrity verified before and after; all 6 connectome position vectors
  re-score to their logged `pct` to 1e-9. Device parity checked (variant and comparator both MPS).
- **Compute:** matched on `total_grad_steps` (20,000). **Wall-clock NOT matched and disclosed:**
  ~115 s vs the baseline's ~75 s on connectome (≈1.5×), from the `tanh` + `where` kernel.
- **Known defect (documented, not hidden):** `torch.where` makes autograd return 0 exactly at
  `z == M` instead of the left-derivative `1/T`; a measure-zero kink, immaterial in float
  practice, but the claim is "verified except at the kink", not "verified everywhere".
- **status: CONFIRMED-PENDING — screen PASSED on connectome, FAILED on microns.** Next steps, in
  order: (i) CONFIRM stage on connectome at 5 seeds; (ii) diagnose the microns regression — the
  prime suspect is that `M` and `T` are absolute constants in `z` units while microns is ~4×
  denser (155 vs 41 average degree) and runs 80k epochs, so a size/density-scaled `(M, T)` may be
  required; any re-tuning **must not** be selected on microns and then reported on microns.

---

# Optimizer-design idea carried forward from H38 (2026-08-17)

**A-VIOL — "spend gradient only on violated constraints" as a general optimizer principle.**

H38 is a *specific* surrogate, but the reason it works is not specific to it: **the gradient
budget should go to edges that are currently WRONG, not to widening margins that are already
won.** The sigmoid spends its gradient symmetrically around `Delta = 0` — half of it defending
edges that are already feedforward. Measured at Rocket's converged connectome order, only
**20.6%** of the sigmoid's gradient mass sits on feedback edges (which carry 17.1% of `w_hat`);
the one-sided form puts **100%** there, and the mirror form (0%) loses −2.889 pp. That ordering
0% → 20.6% → 100% against −2.889 → 0 → +0.367 pp is the cleanest single statement of the effect
we have.

This is a *constraint-satisfaction* view of MFAS rather than a smooth-relaxation view, and it
suggests several moves that are NOT surrogate swaps and have never been tried here:

| id | idea | note |
|---|---|---|
| **A-VIOL-1** | **Density-scaled `(M, T)`** — the open follow-up that decides whether H38 generalizes. `M`/`T` are absolute constants in `z` units, but microns is ~4× denser (155 vs 41 avg degree) and runs 80k epochs; a margin defined relative to the local `Delta` distribution (e.g. a quantile of `|Delta|` over incident edges) may fix the −0.669 pp microns regression. **Must not be tuned on microns and then reported on microns.** |
| A-VIOL-2 | **Violation-weighted sampling** rather than a one-sided shape: keep the sigmoid but sample/weight the loss toward currently-violated edges. Distinct from the killed H06 (weight-aware reweighting, which used *input* weights, not the *current* violation state) and from H13 (uniform random subsampling, −0.84 pp). |
| A-VIOL-3 | **Margin as a schedule**, not a constant: anneal `M` the way `beta` is annealed, so the "already won" set is defined loosely early and tightly late. Cheap; composes with the existing cyclic schedule. |
| A-VIOL-4 | **Compose with the discrete refiner.** Everything above is pure Rocket. H38's order has never been fed to the H35 under-relaxed sift. Since findings #4/#5 show the sift does most of the work, the question "does a +0.37 pp better starting order survive the sift, or does the sift erase it?" is the highest-value cheap experiment in this group — and A-INIT already showed the sift *can* erase an init advantage (spread 0.4375 → 0.0483 pp on mouse). |

**Gate for all of these:** `mfas.analysis.surrogate_gate` (cheap checks first), then the proxy
prototype, then connectome AND microns, then the controls in
`surrogate_gate.REQUIRED_CONTROLS`. Do not skip the mirror and beta-rescale controls — they are
what turned H38 from "a shape that wins" into "the direction of the asymmetry is the mechanism".

---

# Phase 6.5 backlog (A-MBAND) — the multi-band surrogate (2026-08-18)

Opened directly from the Q01 mechanism (`diagnosis.md` § Q01): the surrogate's resolution on the
connectome is ~293 ranks out of 136,648, so it is structurally blind to the narrow wins the
near-optimal order is built from. This is the reformulation of the roadmap's `A-SURR` (TODO 7) that
survived the earlier shape analysis — it changes the number of length scales, not the shape.

## A-MBAND — add a second, rank-scale sigmoid band on top of the existing one
- **One-line hypothesis:** `F = Σ ŵ σ(β_c Δ) + λ·Σ ŵ σ(β_f Δ)` with `β_f = ρ·n/(√12·std(P))`
  (half-width = 1/ρ *ranks*, hence scale-free) lets the coarse band keep doing long-range transport
  while the fine band supplies the reward for narrow wins the coarse band discounts to ~0.5.
- **status: KILLED by prototype gate (2026-08-18)** — 12 arms, all negative, monotone in λ.

**Gate 1 — ranking: PASS.** At the operating scale (std = 141, β = 1.05, even spacing) the
single-band surrogate ranks Rocket's 82.92% order **above** the 84.61% order by −174.88. Adding the
fine band **flips it**:

| half-width | λ = 0.3 | λ = 1.0 | λ = 3.0 |
|---|---|---|---|
| 20 ranks | −133.49 | −36.91 | +239.04 |
| 3.3 ranks | −102.84 | **+65.27** | **+545.57** |
| 1.0 rank | −91.72 | **+102.32** | **+656.73** |
| 0.33 rank | −86.82 | **+118.67** | **+705.76** |

This is the first continuous objective in the project that prefers the better order **at Rocket's
own operating scale**, with no rescaling. It was not enough.

**Gate 2 — training: FAIL, decisively.** connectome, 20k epochs, seed 42, vs the λ=0 control
(82.9161, which reproduces `baseline_passthrough`):

| arm | pct | Δ vs control | final std | resulting resolution |
|---|---|---|---|---|
| control (λ=0) | **82.9161** | — | **165.4** | **249 ranks** |
| ρ=1.0, λ=0.3 | 81.7728 | −1.1432 | 94.5 | 437 |
| ρ=0.3, λ=1.0 | 81.3424 | −1.5737 | 94.6 | 437 |
| ρ=1.0, λ=3.0 | 80.8884 | −2.0277 | 80.7 | 512 |
| ρ=3.0, λ=3.0 | 80.4852 | −2.4308 | 84.5 | 488 |
| late ramp (λ=0.3 from 50%) | 82.6084 | −0.3077 | 86.9 | 475 |
| late ramp (λ=1.0 from 50%) | 82.0626 | −0.8535 | 66.0 | 625 |
| fixed β_f = 280 (non-adaptive) | 81.4468 | −1.4693 | 73.1 | 565 |

**Measured failure mechanism — the intervention is self-defeating.** In *every* arm the position
scale collapses (165 → 66–107), so the effective resolution gets **worse** (249 → 437–625 ranks),
which is the opposite of the intended effect. The two controls localise the cause: it is **not** the
adaptive `β_f` self-amplifying (fixing `β_f = 280` collapses the scale just as hard), and it is
**not** an early-training transient (ramping λ in only after 50% still collapses it). The cause is
the fine band's gradient magnitude: `β_f/β_c ≈ 267`, so for any node with a short-range neighbour the
fine band dominates the per-coordinate Adam step, the node's position is set by a local tug-of-war
instead of by global structure, and the coarse band's "spread out" signal never accumulates.

**What this buys us (the reason the kill is worth its compute).** It is the sharpest available
evidence for finding #3: even when the continuous objective is **repaired so that it demonstrably
ranks the better order higher**, optimizing it is *worse* than optimizing the misaligned one.
Correct ranking is necessary but not sufficient — the binding constraint is the gradient dynamics,
not the objective's preference. This is strictly stronger than H34's result (which showed a
non-vanishing gradient estimator still loses).

**Revival condition.** Only if a mechanism is found that adds fine-scale reward **without** letting
its gradient dominate the coarse band — e.g. per-band gradient normalization, or applying the fine
band to a *disjoint* parameter (a residual offset) rather than to the same positions. The λ→0 sliver
(λ ≤ 0.03) is untested but the 12-arm trend is monotone toward the control, so no positive region is
predicted there.

**Reproduce** (env `allen`, repo root; ~18 min sweep + ~6 min rescue on MPS):
```
PYTHONPATH=src python experiments/proto_amband.py --stage resolution   # the pathology audit
PYTHONPATH=src python experiments/proto_amband.py --stage ranking      # gate 1
PYTHONPATH=src python experiments/proto_amband.py --stage sweep --datasets connectome   # gate 2
```
Artifacts: `experiments/outputs/proto_amband.json` (`resolution`, `ranking`, `sweep`, `rescue`),
logs `experiments/amband_sweep.log`, `experiments/amband_rescue.log`.

### Side result kept from Stage A — resolution predicts where the sift pays

| graph | n | converged std | resolution | H30 sift gain (finding #4) |
|---|---|---|---|---|
| connectome | 136,648 | 141.0 | **292.6 ranks** | **+0.847 pp** |
| microns | 67,534 | 908.4 | **22.5 ranks** | **+0.078 pp** |

Resolution ratio 13.0× vs sift-gain ratio 10.9× — consistent with the idea that the discrete sift is
paid exactly for the fine-scale structure the surrogate cannot resolve, and it would explain finding
#4's unexplained "graph-dependent magnitude (~11×)" caveat. **Honest scope: n = 2 graphs, and mouse
does NOT fit** (resolution 2.1 ranks yet +0.4225 pp sift gain — though mouse is tiny and
near-saturated, σ = 0.26 pp). Hypothesis-grade, not a law; a third large connectome would test it.

---

# Phase 6.7 — A-SUB and A-ALT prototype gates (2026-08-18)

Both are the researcher's TODO 2 and TODO 3, run as short gates on the primary dataset from
orders already on disk (no Rocket run needed). Artifacts: `experiments/outputs/proto_asub_aalt.json`,
`experiments/outputs/proto_asub_aalt_control.json`; script `experiments/proto_asub_aalt.py`.

## A-SUB — "SGD on part of the neurons" (TODO 2): the DISCRETE reading PASSES

**The continuous reading was NOT run, and that is a considered decision, not an omission.**
Updating a random subset of position *coordinates* per gradient step is a pure dynamics knob:
block-coordinate ascent has the SAME critical points as full-gradient ascent on a smooth
objective, so it changes the path, not the fixed set — precisely the class that finding #2 found
inert across 8 mechanisms, and the closest prior test (H13, stochastic edge subsampling) cost
−0.84 pp on connectome. It would be a cheap falsifier, never a candidate.

**The discrete reading is different and it wins.** H35's shipped rebuild moves EVERY mover a
fraction `alpha` toward its exact-gain target. The stochastic recast moves each mover FULLY with
probability `p`. The displacement fields agree in expectation at `alpha = p`
(`E[key_i] = rank_i + p*(target_i - rank_i)`), so the two are directly comparable — ⚠ but only at
the KEY level: the rank vector is `argsort(argsort(key))`, a nonlinear map, so this is what makes
`p` and `alpha` the same axis, not a proof they are the same algorithm.

Measured on connectome from a fixed pre-sift Rocket order (82.9314%), 22 sweeps, `k_full=6`:

| rebuild | best pct | movers left | wall |
|---|---|---|---|
| Jacobi `alpha=1.0` (H30) | 83.8086 | 5,162 | 120 s |
| under-relaxed `alpha=0.7` (**H35, shipped**) | 83.8931 | 1,386 | 81 s |
| **stochastic `p=0.7`** (n=5 seeds) | **83.9047 ± 0.0009** | **64–173** | **63 s** |
| stochastic `p=0.5` (n=2) | 83.8999 | 359–448 | 63 s |

**Δ vs the shipped H35 rebuild = +0.0116 pp, all 5 seeds above it, σ = 0.0009 (~13σ), at ~20%
LESS wall-clock.** Two pre-registered predictions were FALSIFIED: S1 ("stochastic matches
deterministic within 0.01 pp") and S3 ("stochastic does not beat deterministic"). The
expectation identity does not carry over to the dynamics.

**Mechanism (supported by the mover counts, the direct signature).** The Jacobi limit cycle H35
diagnosed is caused by *simultaneity* — conflicting nodes leapfrog because they all move at once.
Under-relaxation damps the amplitude but keeps every node moving, so the oscillation survives at
reduced size (1,386 movers left). Randomisation instead breaks the *symmetry* of each conflict:
when two nodes want to swap, only one moves, and the conflict resolves. On the already-sifted
order the stochastic rebuild reaches **0 movers — a true 1-opt fixed point** — where `alpha=0.7`
still leaves 74. For a cycle caused by simultaneity, breaking simultaneity beats damping it.

**status: PROTOTYPE PASS -> promote to a variant cycle (H39).** Needs: all 3 datasets, ≥3 seeds
through the frozen runner, comparator = **H35 at matched seeds** (this replaces H35's rebuild,
so H35 is the incumbent), plus the `p` sensitivity. Note it also fixes the microns
under-convergence that H35's entry lists as an open caveat, since it converges far faster.

## A-ALT — alternate discrete <-> gradient (TODO 3): weak positive, needs seeds

`dr_tmp/kick_gate.py` existed but had **never been run** (no artifact). Q01's prediction is now
confirmed for the first time on a *sifted* order (all previous drift probes used the reference
best order). Kicking the 83.9035% H35 order with 150 Adam steps:

| surrogate | std | beta*std | after kick | drift |
|---|---|---|---|---|
| sigmoid | 141 | 148 | 83.7591 | **−0.1444** |
| sigmoid | 500 | 525 | 83.8679 | −0.0356 |
| sigmoid | 1500 | 1575 | 83.8961 | −0.0074 |
| H38 one-sided | 141 | 148 | 83.8468 | −0.0567 |
| H38 one-sided | 500 | 525 | 83.8958 | −0.0077 |
| H38 one-sided | 1500 | 1575 | 83.9033 | −0.0002 |

**The Q01 vise is real and quantified:** at the operating scale the sigmoid pulls a good order
DOWN by 0.14 pp; at a scale above the crossover it barely moves anything (the gradient is dead).
H38's one-sided surrogate drifts **2.5× less** at the operating scale, exactly as its alignment
ratio predicts — but it still drifts, so pre-registered A3 FAILED.

**Attribution (the control the H36 critic taught us to run first).** A monotone best-by-oracle
re-sift beats its input almost tautologically, so the kick must be compared against *the same
re-sift with no kick*:

| arm | result | net vs the 83.9035% input |
|---|---|---|
| plain re-sift, NO kick (control) | 83.9067 | +0.0032 pp |
| H38 kick @ std 500 -> re-sift | 83.9246 | +0.0211 pp |
| **attributable to the gradient kick** | | **+0.0179 pp** |

So alternation is not merely "more sift" — but note what it structurally is: the kick *lowers*
the score and the re-sift repairs it, i.e. a **perturb-and-repair (ruin & recreate) move**, the
same family as **H31**, which was KILLED (Δ vs H30 = −0.0008 pp at ~1.9× wall). The distinction
is that the perturbation here is gradient-guided rather than random, and that does appear to
matter — but the effect is **single-seed, single-config, and ~1/6 of A-SUB's**.

**status: open, LOWER priority than A-SUB.** Before building: ≥3 seeds × ≥2 kick configs, and
the honest comparator is H31's ruin-and-recreate, not the raw incumbent.
