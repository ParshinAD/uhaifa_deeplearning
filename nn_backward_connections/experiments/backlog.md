# Backlog — open hypotheses for improving Rocket

Each hypothesis, when picked up, must be run through the **unchanged** harness on **both**
datasets across **≥3 seeds**, with results logged to `results/*.json` and a ranked conclusion
recorded in `findings.md`. A change counts as a win only if it beats the baseline beyond seed
noise on the exact feedforward metric.

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

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
- **status: proposed**

---

### Ranking rationale (EV / cost)
H01–H02 first: cheapest, best-evidenced upside (exploit known ordering variance + a refinement
trick the paper itself validates). H03–H08 are cheap schedule/optimizer knobs the paper admits are
under-tuned. H04 has the highest ceiling but the most cost/risk, so it sits behind the trivially
cheap wins. H09–H13 are lower-EV or higher-variance ideas kept for diversity across axes
(near-tie recovery, gradient handling, surrogate shape, averaging, stochasticity). Run H03/H07
before H08, and H05 before H10 (shared optimizer machinery).
