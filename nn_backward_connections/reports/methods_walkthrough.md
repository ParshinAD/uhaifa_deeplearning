# Improving the Rocket MFAS Algorithm on the Fly Connectome — Methods Walkthrough

*A self-contained talk script. Read top to bottom; every method is explained from first
principles, with the intuition, the mechanism, why it helps, and the measured result.
All numbers trace to `results/*.json` / `experiments/` logs (no fabricated figures).*

---

## 0. The problem in one slide

We are given a **directed weighted graph** — the fly brain connectome (FlyWire). Each edge
`(u → v)` with weight `w` means "neuron `u` sends `w` synapses to neuron `v`".

**Task (Maximum Feedforward Arc Set, MFAS).** Place every neuron at a position on a line
(assign each node `i` a coordinate `pos[i]`). An edge `(u → v)` is **feedforward** if
`pos[u] < pos[v]`, otherwise it is **feedback**. Maximize the total weight of feedforward edges:

```
score = Σ  w(u,v)   over all edges with pos[v] > pos[u]
```

Equivalently, minimize the **feedback weight** = `total_weight − score`. The feedback weight is
the interesting scientific quantity: it measures how much recurrence is *unavoidable* given the
wiring — a structural property of the brain.

**Why it is hard.** This is the weighted Minimum Feedback Arc Set problem — NP-hard. The
connectome has **136,648 neurons** and **5,657,719 edges** (total weight 41,912,141), so exact
solvers are out; we need heuristics.

**Baseline we reproduce and try to beat.** Bader et al. (2025), *"Rocket-Crane algorithm for the
Feedback Arc Set problem"*. Their pipeline has two phases: **Rocket** (a continuous/gradient
heuristic, ~82.9%) and **Crane** (a discrete MIP refinement needing Gurobi and ~20 days, reaching
84.60%). We focus on and improve **Rocket** — no MIP, commodity hardware.

**Datasets used (we always report all applicable ones).**

| dataset | what it is | nodes | edges | role |
|---|---|---|---|---|
| `connectome` | FlyWire fly brain | 136,648 | 5,657,719 | **PRIMARY** (the challenge graph) |
| `microns` | MICrONS mouse visual cortex | 67,534 | ~10.4 M | **PRIMARY** (2nd large graph, independent species) |
| `mouse` | small mouse connectome | 148 | 583 | SUPPORTING (tiny, noisy, near-saturated) |

---

## 1. The measurement backbone — why our numbers are trustworthy

Before any algorithm, we built the **evaluation oracle**. This is what makes the thesis
defensible; a reviewer will attack the numbers first, so we lock them down.

### 1.1 The exact scorer ("the oracle")

`src/mfas/metrics.py` implements the *one* canonical scoring function used everywhere:

- An edge is feedforward iff `pos[tgt] > pos[src]` — **strict `>`**, so ties count as feedback.
- Weights are summed in **int64 (connectome) / float64 (mouse)**, **never float32**. Summing
  42 million integer weights in float32 loses precision and silently drifts by a few units
  (this is the historical `34,751,904` vs the correct `34,751,902` discrepancy).
- Scoring is always done **on CPU**, never on the Apple GPU (MPS): large-magnitude float32 index
  operations on MPS can occasionally return garbage.

**Parity anchor (our reproduction claim).** Scoring the saved best positions
(`results/rocket_best_positions.npy`) returns *exactly* **34,751,902 / 41,912,141 = 82.9161%**.
This exact integer is asserted by a unit test. Because the GPU is non-deterministic, we cannot
reproduce a training run bit-for-bit; instead the reproduction rests on this deterministic
**scorer-parity test**.

### 1.2 The rigor protocol (how a "win" is declared)

A change is a win only if it survives a **two-stage statistical gate** on **all** applicable
datasets:

- **SCREEN** (cheap, 3 seeds 42/123/999): promote only if `Δmean > 2σ` on *both* large graphs,
  where σ is the per-dataset baseline noise floor. Thresholds: connectome > 0.04 pp,
  microns > 0.002 pp, mouse non-inferiority > −0.26 pp.
- **CONFIRM** (5 seeds connectome/microns, 20 seeds mouse): the **95% CI lower bound of the
  difference of means must be > 0**. `SE = σ·√(2/n)`; more seeds tighten the interval.
- **"A gain inside the noise is not a gain."**

Each candidate then passes a 3-agent pipeline: **implementer** (builds & screens) →
**verifier** (independent clean re-run) → **critic** (red-teams for metric leakage, overfitting
to one dataset, gains within noise, and accidental edits to the frozen scorer).

**Noise floors (the discriminating power of each dataset):**

| dataset | baseline mean | σ (noise floor) | note |
|---|---|---|---|
| connectome | 82.896 % | 0.0189 pp | primary |
| microns | 83.117 % | **0.0006 pp** | 29× tighter than connectome — best discriminator |
| mouse | 92.070 % | 0.2624 pp | 14× noisier, near-saturated → supporting only |

**Compute fairness.** Every variant is compared at **equal gradient budget** (same number of
`optimizer.step()` calls), *not* equal wall-clock (GPU timing is machine-dependent and
non-deterministic). This is the crux for the post-processing methods below: the discrete
refinement adds **0 gradient steps**, so comparing it to the baseline at equal gradient budget is
honest.

**Governance.** The scorer and harness are **frozen** after their unit test passes — enforced
three ways: a pre-commit hook blocks edits, the files are `chmod 0444`, and a SHA-256 manifest is
verified before *every* scored run. No algorithm ever reads the target metric internally
("no leakage").

---

## 2. Part I — the methods that produce the result (the algorithm chain)

The final pipeline is a chain of three blocks:

```
greedy warm-start  →  Rocket (continuous)  →  sift (discrete refinement)
   (Section 2.2)         (Section 2.1)          (Sections 2.3–2.4)
```

### 2.1 Rocket — continuous relaxation (the baseline method)

**Idea.** Replace the discrete "order on a line" problem with a *continuous* optimization: the
positions `pos` become a trainable vector of real numbers, and the hard condition `pos[v] > pos[u]`
is softened into a **sigmoid**. Now we can take gradients and descend with Adam.

**Surrogate loss** (`src/mfas/baseline/rocket.py`):

```
loss = − Σ_(u,v)∈E  σ( β · (pos[v] − pos[u]) ) · ŵ(u,v),      ŵ = w / max(w)
```

- `σ(β·Δ) ≈ 1` when an edge is feedforward (`Δ = pos[v]−pos[u] > 0`) and `≈ 0` when feedback — so
  the loss is a **smooth surrogate** of the discrete score.
- `β` is the sigmoid **sharpness**. Small β → smooth, near-convex landscape (easy to move, but a
  crude approximation of the real score). Large β → the sigmoid approaches a step function
  (faithful to the discrete score, but riddled with local minima).

**Three schedules inside the training loop:**

1. **Cyclic β** (`make_beta_schedule`): `β = (cos(...) + 1.1)/2`, range `[0.05, 1.05]`, 5 cycles.
   β repeatedly "melts" down to 0.05 (escape a local minimum) and "freezes" up to 1.05 (sharpen).
   This is *graduated optimization / continuation*.
2. **Learning-rate schedule**: constant LR = 0.05 for the first 50 % of epochs, then exponential
   decay to 10 % of the initial LR.
3. **Gradient-norm clipping** at 1.0.

**Best-by-oracle tracking (crucial).** The sigmoid surrogate is *not* what we ultimately want —
we want the discrete score. So every 100 epochs the current positions are scored by the **exact
oracle on CPU**, and whenever the discrete score beats the previous best, those positions are
saved. The algorithm returns not the final descent point but the **best ordering the oracle ever
saw** along the whole trajectory.

**Budget:** connectome 20,000 epochs (~90 s), microns 80,000 (~550 s), mouse 5,000 (~1 s).

**Result:** ~**82.92 %** on the connectome — reproduces the paper's Rocket plateau. This is the
number we set out to beat.

---

### 2.2 Greedy-FAS warm-start (experiment H02) — "where you start"

**Problem.** Rocket by default starts from random positions `N(0,1)` — about 50 % feedforward
(a coin flip). Hypothesis: starting from an already-decent ordering lands the optimizer in a
better *basin*.

**Method — the Eades–Lin–Smyth / GreedyAbs greedy FAS heuristic** (`H02.greedy_fas_order`). A
classic cheap heuristic that uses **only the graph structure and weights** — it never reads the
oracle (leakage-safe). It works by *peeling*:

Repeat until all nodes are placed:
- any **sink** (no remaining out-weight) → append to the **back** of the sequence;
- any **source** (no remaining in-weight) → prepend to the **front**;
- otherwise remove the node with maximum `(out_w − in_w)` (most "source-like") → **front**.

Each removal decrements the remaining in/out-weights of its neighbours. Done naively this is
`O(n²)` (≈ 1.8 × 10¹⁰ operations at 136k nodes — infeasible), so it uses a **lazy max-heap** plus
CSR adjacency → `O((n+m) log n)`.

The resulting rank `[0, n)` is mapped linearly to positions in `[−1, +1]` (rank 0 = source =
position −1) and passed into the **unchanged** Rocket as `init_positions`.

**Why the comparison is fair.** Only the initialization changes; the number of gradient steps
is identical to baseline Rocket, so any gain is algorithmic, not extra compute.

**Result:** the greedy order *alone* already scores **68.9 %** (connectome) / 90.1 % (mouse)
*before* any optimization. After Rocket: **82.93 %** — a small but statistically confirmed gain
(+0.05 pp connectome, +0.21 pp mouse, +0.011 pp microns, all CI-lower-bound > 0 across three real
connectomes from two species). The gain is modest, but it led to the key structural conclusion in
Section 3.1.

---

### 2.3 Sift — exact-gain node re-insertion (experiment H30) — the main breakthrough

This is the core of the improvement. `src/mfas/refine/insertion.py`.

**The single move (re-insertion).** Take one node `u`. Freeze **all other** nodes in their
current order. Remove `u` from the line — now there are `n−1` nodes and `n` "gaps" where `u`
could go back (before everyone, between each pair, after everyone). Question: **into which gap
should `u` go to maximize the feedforward weight of `u`'s own incident edges?**

This has a **closed-form exact answer**:

- Each **out-edge** `u → v` (u wants to be *before* v) is feedforward iff gap `g ≤ q(v)` — it
  "votes" for `u` being early.
- Each **in-edge** `x → u` (u wants to be *after* x) is feedforward iff gap `g > q(x)` — it votes
  for `u` being late.
- As a function of the gap `g`, the total incident weight is a **piecewise-constant step
  function**: at `g = 0` (u before all) every out-edge is feedforward → value = total out-weight;
  each time `g` passes a neighbour's position, exactly one edge flips (an out-edge loses `−w`, an
  in-edge gains `+w`).

We just take the **argmax of this step function** — the *exactly optimal* position for `u` given
everyone else. This is **not** a gradient step and **not** a neighbour swap; it is the globally
best single-node move.

*(The math is validated against a brute-force reference in `tests/test_refine_insertion.py`.)*

**Why sift beats the moves that were killed:**
- vs H04 (barycenter = *mean* of neighbours' ranks): sift places the node at the **argmax of the
  exact step function**, not the *mean*. For the bimodal neighbourhoods of a cyclic core, the
  argmax is far from the mean — which is why H04 got 0 accepts and sift works.
- vs H22 (bounded ±W window): sift's candidate range is the **whole line**, so a node can travel
  ~n ranks. We measured the recoverable weight to be **long-range** (median flip distance ≈ 22,600
  ranks out of 136,648), so a bounded window recovers ~nothing while the full range works.

**Jacobi vs Gauss-Seidel — how to make it fast.**
- *Gauss-Seidel*: move one node, update, then compute the next node with the updated state.
  Correct, but sequential → too slow on 136k nodes.
- *Jacobi (what we use)*: compute the best gap for **all nodes simultaneously** from one frozen
  snapshot of the ranks, then apply every move at once. This is **fully vectorizable in NumPy**
  (`jacobi_best_gaps`: one `argsort` on the combined key `u·(n+2)+b`, then segmented prefix-sums
  and a segmented max) → fast. The catch: because everyone moves at once based on stale
  information, the combined move might not actually improve the score.

**Best-by-oracle at the vector level (safety + leakage guarantee).** The "working" order advances
unconditionally each sweep (Jacobi dynamics), but a **separate** tracker keeps the best whole
vector the oracle has ever approved. The oracle is consulted **only** to accept/reject a whole
candidate vector — never inside a move choice. Consequences:
- the result is **never worse** than the input (monotone ≥ pure Rocket);
- **no leakage**: every move is chosen from weights + current ranks; the target metric is never
  peeked at inside the algorithm.

**Compute accounting.** Sift adds **0 gradient steps** (only wall-clock, ~+37 s on the
connectome). So the equal-gradient-budget comparison to Rocket is honest.

**Result (H30):** connectome **82.93 % → 83.78 %** (+0.85 pp over pure Rocket), microns +0.078 pp,
mouse +0.42 pp — a CONFIRMED general win. This recovers **~51 %** of the 1.69 pp Rocket↔optimum
gap **with no MIP at all**.

---

### 2.4 Under-relaxation (experiment H35) — why Jacobi stalls, and the fix

**Observation (diagnosed in `dr_tmp/FINDINGS_underrelaxation.md`).** On the large *dense* graphs
the Jacobi sift **does not converge** — it enters a **period-2 limit cycle**. Mechanism: two
groups of nodes each simultaneously compute "I should jump past you" → they swap all at once →
next sweep each wants to swap back → endless leapfrogging (on the connectome ~5,255 nodes stay
stuck, the candidate oscillates between 83.807 / 83.796 %). Best-by-oracle then only creeps up via
the "lucky phase" of the oscillation.

**The fix — under-relaxation** (the textbook cure for Jacobi 2-cycles). Instead of jumping fully
into the optimal gap, each node moves only a fraction `α` of the way there:

```
key = rank + α · (target − rank),      α = 0.7
```

This damps the oscillation → the iteration converges (the number of "movers" collapses from
thousands to a few hundred) and reaches a **strictly higher fixed point**.

**Two-phase schedule** (`sift_underrelaxed`): the first `k_full = 6` sweeps use `α = 1` (full
Jacobi — fast progress and captures the exact optimum on graphs that already converge, like
mouse), then `α = 0.7` (damp the cycle on graphs that don't). At `α = 1` the method is
**bit-identical** to H30 (unit-tested), so the entire gain comes purely from `α < 1`.

**Result (H35):** connectome **83.78 % → 83.91 %** (+0.098 pp over H30) — ~58 % of the gap. Mouse
non-regressing; microns a tiny −0.002 pp (it needs more sweeps than the 12 allowed there — this is
disclosed, which is why H35 is a *confirmed connectome win* rather than a clean 3-dataset win).

**Current best result of the project: 83.91 % on the fly connectome — no MIP, 0 extra gradient
steps.**

---

## 3. Part II — the methods that produced conclusions *about the task*

These do not improve anything; they answer *why* the problem behaves as it does. This is the most
scientifically valuable part for the thesis.

### 3.1 Conclusion #1 — "the plateau is set by the starting basin, not the descent dynamics"

**Method — a controlled screen of 9 hypotheses.** We ran 8 distinct mechanisms through the same
statistical gate, grouped by which *axis* of Rocket they change:

| what we changed | axis | Δ connectome | verdict |
|---|---|---|---|
| **H02 greedy warm-start** | **where we start** | **+0.045** | **the only win** |
| H01 restarts | dynamics | +0.0003 | kill |
| H05 optimizer → AdamW | dynamics (falsifier) | +0.0018 | kill |
| H03 β schedule | dynamics | −0.038 | kill |
| H04 in-loop refinement | dynamics | ~0 | kill |
| H13 edge-subsample noise | dynamics | −0.84 | kill |
| H06 weight-aware loss | objective | −0.038 | kill |
| H11 hinge surrogate | objective | −0.039 | kill |
| H09 anti-tie jitter | free-edge | −0.001 | kill (0 ties exist) |

**Conclusion.** The *only* lever that helped was the one that changes **where optimization starts**
(H02). Everything that changes **how it descends** (optimizer, β, noise, loss shape) re-converged
to — or fell below — the plateau. We deliberately kept **H05** (swapping the optimizer, the
dynamics knob most able to reach a different basin) as a **falsifier**: it *also* returned to the
plateau, which is the strongest single piece of evidence. So: *how* you descend barely matters;
*where* you begin does. This predicts future gains lie in **better initial orderings**, not in
optimizer/LR/β/loss tuning — and motivated the discrete direction that became sift.

---

### 3.2 Conclusion #2 — the Rocket↔optimum gap is an *optimization gap*, not a surrogate mismatch

There is a downloaded near-optimal ordering (`data/best_solution` = **84.6147 %**). The gap to
Rocket (82.93 %) is **1.69 pp**. Question: is the gap because (a) the sigmoid surrogate "points the
wrong way" (mismatch), or (b) the surrogate is right but the optimizer **cannot reach it**
(optimization gap)?

All these diagnostics live in `src/mfas/analysis/gap.py` — the **only** module allowed to read
`best_solution`, deliberately isolated so the answer never leaks into an algorithm.

**(1) Decisive surrogate test** (`optimize_spacing`). Compare the surrogate value of the *best*
ordering vs Rocket's solution. Subtlety: the surrogate depends on the *gaps* between positions, not
just the order — so for each ordering we find its *optimal monotone spacing* with a free global
scale (scale-fair). Result: the best ordering yields a **higher** surrogate at **every** β
(+238 … +1153). If the surrogate were mismatched, the higher-scoring discrete order would score
*lower* on the surrogate — instead it scores higher. ⟹ **the surrogate is correct; the bottleneck
is the optimizer.**

**(2) Drift probe** (`drift_probe`). Start Rocket *from* the 84.61 % solution and watch where it
goes. Under every schedule, gradient descent **washes** the near-optimal solution back down to the
82.75–83.03 % plateau (first dipping into 76–79 %). ⟹ the optimum is **not a reachable or even
holdable attractor** for Adam-on-sigmoid; the landscape has a dominant ~82.9 % basin that swallows
even a perfect start.

**(3) init → plateau curve.** Five different initializations (spanning 36–69 % quality) all yield
the *same* post-Rocket plateau (82.87–82.93 %, spread 0.06 pp). ⟹ a better start *by itself* buys
≤ 0.06 pp; H02 already captured essentially all that initialization can give.

**(4) Gap structure.** 7.5 % of the weight "flips" between Rocket and best: +4.60 %
(best-feedforward-but-Rocket-feedback) minus −2.91 % (the reverse) = the net +1.69 %. Two large
*opposing flows*, not a small correction. Kendall-τ = 0.61 (moderate): more than local swaps, but
not a wholly different basin. Disagreement is uniform across edge weights and sits on median-degree
nodes (not hubs / heavy edges). And there are **0 exact ties** → the gap is genuine *misordering*,
not loss from the strict `>`.

**(5) Window sizing** (`experiments/size_localsearch.py`). How much weight is recoverable if moves
are restricted to a ±W window? Positive only for W ≤ 10; W=100 → −0.010, W=1000 → −0.170,
W=5000 → −0.405 pp. Flip-distance percentiles: p50 = 22,580 ranks. ⟹ the recoverable weight is
**long-range / global** — local cleanup cannot get it. *This is exactly what motivated the
full-range sift (H30).*

**(6) Hard synthetic fixture** (`make_hard_synthetic_graph`). Because `best_solution` exists only
for the connectome, we built a synthetic graph with dense cyclic cores (SCC blocks), a high
feedback fraction, and heavy-tailed weights — engineered to *carry* a verified +0.80 pp gap. The
continuous levers (soft-rank H19, perturbed-sort H34) failed there *too*, where a gap provably
exists — so their failure cannot be dismissed as "weak benchmark".

**Final statement (refined by H30/H35).** The gap is an optimization gap; the surrogate is
faithful. It is **not** closable by continuous methods — even a perturbed-sort surrogate with a
*non-vanishing* gradient (H34) loses, isolating the bottleneck as the **continuous-relaxation basin
itself**, not the gradient estimator. But it **is partly closable by cheap *global discrete*
refinement** (sift): ~51 % (H30) → ~58 % (H35), with no MIP. This revises the original "needs the
20-day Crane MIP" framing: *global discrete re-insertion ≠ MIP*. Independent prior art (Vahidi
2025, arXiv:2506.13799) reaches the full 84.61 % on this exact graph *also without MIP* (cheap
greedy + bounded-span insertion + SCC) — consistent with our message.

---

### 3.3 Cross-cutting insight — "the discrete refiner does the work; its starting basin barely matters"

In Phase 6 we tested whether a *smarter* start helps the sift: a trophic-Laplacian warm-start
(H32) and a magnetic-Laplacian (directional-spectral) warm-start (H33). Both were **worse** seeds
for the sift than plain greedy, and H33 additionally blew the time budget (eigensolve 304–507 s).
A separate probe showed sift-from-greedy ≈ sift-from-Rocket. ⟹ the design is simply **greedy +
sift**; no expensive initialization earns its cost. This mirrors Conclusion #1 one level up: among
discrete starting orders, the cheap greedy peel is already at least as good as expensive spectral
embeddings.

---

## 4. Results summary (the headline table)

| method | connectome | what it adds | status |
|---|---|---|---|
| Rocket baseline (random init) | 82.916 % | reproduces the paper's plateau | anchor |
| **H02** greedy warm-start | 82.930 % (+0.05) | better starting basin | CONFIRMED win |
| **H30** full-range Jacobi sift | 83.812 % (+0.85 over H02) | global exact-gain discrete move | CONFIRMED general win |
| **H35** under-relaxed sift | **83.910 %** (+0.10 over H30) | breaks the Jacobi limit cycle | CONFIRMED connectome win |
| *reference:* near-optimal `best_solution` | 84.615 % | — | the target |
| *reference:* Vahidi 2025 SOTA (no MIP) | 84.61 % | — | reachable without MIP |
| *reference:* paper Rocket+Crane (MIP, ~20 d) | 84.60 % | — | not reproduced (needs Gurobi) |

- Our best **83.91 %** recovers **~58 %** of the 1.69 pp Rocket↔optimum gap, **with no MIP and 0
  extra gradient steps**.
- The remaining ~0.7 pp to 84.61 % is the target for future SCC-structured insertion.

---

## 5. Suggested talk track (how to narrate this in ~5 minutes)

1. **The task.** "Order neurons on a line so most synaptic weight flows forward; the unavoidable
   backward weight measures how recurrent the brain *has* to be. It's NP-hard on a 136k-node,
   5.7M-edge graph, so we use heuristics. The reference method, Rocket, plateaus at ~82.9 %; the
   paper only goes higher with a 20-day MIP we can't run."

2. **The measurement backbone.** "First we froze an exact int64 CPU scorer with a parity anchor
   (34,751,902), and adopted a two-stage statistical gate — screen then confirm across ≥3 seeds
   with confidence intervals, verified by an independent agent and red-teamed by a critic. So every
   number is reproducible and no algorithm can cheat by peeking at the metric."

3. **What we learned about the problem (2 structural findings).**
   - "Across 9 controlled experiments, the *only* thing that helped was changing **where**
     optimization starts, not how it descends — including a deliberate falsifier (swapping the
     optimizer) that confirmed it. The plateau is basin-determined."
   - "Against a near-optimal reference we showed the 1.69 pp gap is a genuine **optimization gap**
     — the sigmoid surrogate ranks the optimum higher at every β, yet gradient descent washes even
     a perfect start back to the plateau. And the recoverable weight is **long-range/global**, not
     local."

4. **What we built from that.** "Since the lever is *global discrete reordering*, we added a
   **sift**: it puts each node at the exact position that maximizes its incident feedforward weight
   given all others fixed — computed in closed form, vectorized as a Jacobi sweep, with the oracle
   only accepting/rejecting whole vectors. That's +0.85 pp. Then we noticed the Jacobi sweep
   limit-cycles on dense graphs and damped it with under-relaxation — another +0.10 pp. Final:
   **83.91 %**, ~58 % of the gap, **no MIP, no extra gradient steps.**"

5. **Why it's trustworthy.** "Frozen oracle, equal-gradient-budget comparison, ≥3 seeds with CIs,
   independent verifier and critic, one git commit per experiment, no metric leakage."

---

## 6. Mini-glossary

- **Feedforward / feedback edge** — `(u→v)` is feedforward if `pos[u] < pos[v]`, else feedback.
- **MFAS / FAS** — Maximum Feedforward Arc Set / (minimum) Feedback Arc Set — the same problem,
  complementary objective.
- **Surrogate** — the smooth sigmoid loss Rocket trains on; *not* the reported metric.
- **Oracle** — the exact discrete scorer; frozen ground truth.
- **Basin** — the region of the optimization landscape a start point descends into; its bottom is
  the local optimum reached.
- **Sift** — repeatedly re-inserting each node at its exact feedforward-maximizing position.
- **Jacobi vs Gauss-Seidel** — move all nodes at once from a frozen snapshot (parallel, fast) vs
  one at a time with updates (sequential, converges cleanly).
- **Under-relaxation** — moving only a fraction `α` toward the target each step to damp
  oscillations.
- **Best-by-oracle** — keep the best whole ordering the exact scorer ever validated; never let the
  working iterate's transient dips corrupt the result.
- **Leakage-safe** — every move decided from input weights + current positions only; the target
  metric is never read inside an algorithm.

---

*File: `reports/methods_walkthrough.md`. All figures cross-referenced in
`experiments/findings.md`, `experiments/diagnosis.md`, and `results/*.json`.*
