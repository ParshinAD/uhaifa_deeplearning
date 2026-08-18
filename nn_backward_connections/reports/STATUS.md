# Where the project stands — 18 Aug 2026

*Reading time ~10 minutes. Written to be read top to bottom; each section stands alone if you
only need one.*

**The task.** Order all 136,648 neurons of the FlyWire fly connectome on a line so that as much
edge weight as possible points *forward*. An edge `u→v` is feedforward if `pos(u) < pos(v)`.
The score is the share of total edge weight (41,912,141 over 5,657,719 edges) that is
feedforward. This is weighted Minimum Feedback Arc Set — NP-hard — so everything here is a
heuristic measured against an exact scorer.

**Two conventions used throughout.**
- **pp** = percentage points of total edge weight. The whole project moves in hundredths of a pp.
- **Screen gate** = the smallest gain worth believing on a dataset, set at twice its seed-to-seed
  noise: **0.04 pp** on connectome, **0.002 pp** on MICrONS. Mouse (148 nodes) is far too noisy to
  gate on, so there we only require "not appreciably worse".

> ### If you read only one thing
> We started at **82.90%** (the reproduced published algorithm) and are now at **84.15%**.
> The reference solution is **84.61%**, so **73% of the original gap is closed**. Essentially
> all of that came from the **discrete** side of the algorithm, not the gradient side.

---

## 1. The headline numbers

| dataset | published / baseline | **current best** | reference | gap left |
|---|---|---|---|---|
| **connectome** (fly, 136k nodes) | 82.8958 ± 0.0187 | **84.154095** | 84.6147 | **0.4606 pp** |
| **MICrONS** (mouse cortex, 67k nodes) | 83.1172 ± 0.0006 | **83.240853** | — none | — |
| mouse (148 nodes, supporting only) | 92.0696 ± 0.2624 | **93.082880** | — none | — |

Two results sit *just outside* that table and are the immediate next steps:

- **84.176420** on connectome and **93.102826** on mouse — a sequential pair-relocation pass
  applied to a stored champion order. It is a **post-hoc refinement, not a pipeline run**, so it
  is not a champion until it is built in as a stage and screened end to end (filed as H52).
- **83.2696** on connectome from a *pure gradient* run with a one-sided surrogate (H38) — not on
  the champion path, but the largest gradient-side gain the project has produced. See §7.

> ⚠ **These numbers live on two different branches.** See §8 — this is the biggest housekeeping
> risk in the project right now.

---

## 2. How we got from 82.9% to 84.15% — the waterfall

Every row is the connectome, and every row is *cumulative*. This is the single most useful table
in the document: it shows that four of the five gains are discrete.

| # | step | kind | score | gain | where |
|---|---|---|---|---|---|
| 0 | random order | — | 50.05 | — | — |
| 1 | greedy-FAS peel (Eades–Lin–Smyth) | discrete | 68.91 | — | H02 |
| 2 | **Rocket** — gradient ascent on a sigmoid relaxation | gradient | **82.8958** | — | reproduces the paper's 82.87 plateau |
| 3 | start Rocket from the greedy order instead of noise | gradient-adjacent | 82.9292 | **+0.03** | finding #1 (H02) |
| 4 | full-range exact-gain node sift | **discrete** | 83.8118 | **+0.88** | finding #4 (H30) † |
| 5 | under-relax the sift to break its limit cycle | **discrete** | 83.9101 | **+0.10** | finding #5 (H35) |
| 6 | add a recursive SCC **block** move class | **discrete** | 84.0972 | **+0.19** | finding #6 (H36) |
| 7 | re-allocate the budget: many cheap cycles, not few deep ones | **discrete** | **84.154095** | **+0.06** | finding #7 (H42) |
| — | *reference solution (Vahidi 2025, no MIP)* | — | *84.6147* | | |

† Finding #4's headline is **+0.8468 pp**, measured at the older 12-sweep cap (83.7761); the
waterfall uses the current 40-sweep value so the column sums exactly.

**Read the "kind" column.** The gradient phase contributes step 2, plus **+0.0334 pp** of step 3.
Every other gain — **+1.2249 pp of the +1.2583 pp total, i.e. 97%** — comes from discrete
refinement of the order.

---

## 3. The champion pipeline in plain words

Four stages. Stages 1–3 are frozen; stage 4 is where the recent work happens.

1. **Greedy peel.** Repeatedly strip sinks to the back and sources to the front; otherwise remove
   the node with the largest out-weight minus in-weight and put it at the front. Gives 68.91%.
2. **Rocket (the gradient phase).** Give every neuron a real-valued coordinate and maximise
   `Σ w·σ(β·(pos_v − pos_u))`, a smooth stand-in for "is this edge forward?", with Adam and a
   cyclic sharpness schedule β. Reaches ~82.93%. *On mouse this stage is now deleted entirely* —
   it was actively harmful there (H44).
3. **Node sift.** Repeatedly move each neuron to the exact position that maximises the forward
   weight of its own edges, given everyone else. Moving everyone at once oscillates forever, so
   each node moves only 70% of the way (under-relaxation). Reaches ~83.91%.
4. **Block refinement, alternated.** *Contiguous-block lemma:* if a set of neurons occupies an
   unbroken stretch of the line, you may permute them freely — no edge to the outside can change
   direction. So each stretch is an independent sub-problem: find its strongly connected
   components, lay them out in topological order, recurse. This is **correct by construction**,
   never a gamble. Alternating it with a *short* sift reaches **84.15%**.

**Why stage 4 needed the alternation.** Doing SCC decomposition once on the whole graph is
worthless — it gains **+0.00013 pp**, because 92.8% of neurons sit in one giant strongly
connected component. The value is entirely in **recursing inside** that giant component, on
rank windows, and interleaving with the node move class, which sees different improvements.

---

## 4. What is confirmed to work

Seven findings, each backed by logged runs on ≥3 seeds and an independent re-run.

| # | finding | headline | scope |
|---|---|---|---|
| 1 | Greedy warm start beats random init | +0.05 pp | **all 3 graphs** |
| 2 | The plateau is set by *where optimization starts*, not by its dynamics | 8 mechanisms, all null | structural |
| 3 | The remaining gap is not reachable by continuous methods | see §5 | ⚠ being revised |
| 4 | Full-range exact-gain sift | **+0.85 pp** | **all 3 graphs** |
| 5 | Under-relaxation breaks the sift's limit cycle | +0.10 pp | connectome |
| 6 | Recursive SCC **block** move class | **+0.18 pp** | all 3, mouse caveat |
| 7 | Budget re-allocation inside stage 4 | +0.06 pp | connectome + MICrONS |

**The largest single win is #4** (+0.85 pp) and the largest *new-move-class* win is **#6**.

---

## 5. What we learned about *why* — the thesis content

These are the results that explain the algorithm rather than improve it. They are what makes
the negative results meaningful instead of a list of failures.

**The relaxation optimises a different problem at every scale.** Everything depends on one
product, `β·std` (sharpness × how spread out the coordinates are), not on the two separately.
- As `β·std → 0` the objective *telescopes*: the edge sum collapses into a per-neuron quantity
  that no longer knows which neuron precedes which. Its optimum is "sort by in/out imbalance",
  **not** the feedforward optimum.
- At Rocket's actual operating point (`β·std ≈ 148`) the surrogate **ranks Rocket's own 82.92%
  order above the 84.61% reference order**, by 175 units. It prefers the *worse* order, so no
  gradient step points toward the better one.
- It would prefer the better order above `β·std ≈ 470` — but there the gradient has numerically
  died (only ~7% of edges still carry a non-zero float32 gradient). **Drift below, paralysis
  above.** This is the mechanistic reason the discrete side does the work.

**The gap is long-range.** Where Rocket's order and the reference disagree, the median
disagreement is **22,580 rank positions apart**. That is why bounded-window local search
recovers nothing and why the *full-range* sift was worth +0.85 pp.

**The ~82.9% level is degenerate.** Across seeds the score is stable (±0.02 pp) but the ordering
is not (Spearman 0.958) — many near-equivalent orders. This retroactively explains why
multi-start / best-of-K harvests nothing.

**Symmetry, not shape, was the real constraint on the surrogate.** Every surrogate ever tried
was symmetric (`g(−z) = 1 − g(z)`). Within that family the shape axis is ~a relabelling of β and
is closed. Dropping symmetry — constant above a margin, tanh below, so *only violated edges get
gradient* — escapes the telescoping degeneracy entirely and gains **+0.37 pp** on connectome.
See §7.

---

## 6. What does not work — eleven closed doors

Grouped by *mechanism*, because the individual variant IDs are not the interesting part. Every
row is a measured kill, not an assumption.

| axis | verdict | decisive number |
|---|---|---|
| Optimizer swap / restarts / LR / EMA | null | AdamW +0.002; multi-start +0.0003 |
| β schedule, sharper or monotone | negative | −0.038 / −0.27 pp |
| Position-scale annealing | ≡ the β axis | same knob, proved via `β·std` |
| Loss reweighting by edge weight | negative | −0.038 pp |
| Symmetric surrogate reshaping (hinge, perturbed sort, heavy tails) | negative | −0.04 / −0.64 / −0.45 pp |
| Gradient noise (edge subsampling) | strongly negative | −0.84 pp |
| Tie-breaking / free-edge recovery | vacuous | **0** exact ties exist |
| Rank-space soft-sort | stalls | O(1/n) gaps vanish |
| Spectral / trophic warm starts | negative | −1.3 to −5.0 pp |
| A *better starting order* | negative | a 5.7 pp better start ends 0.014 pp **worse** |
| Bounded-window & coarse-block moves | ~zero | window ≈ 0; coarse blocks **exactly 0** |

Two more axes were measured *shut* rather than negative: the **gradient budget** (connectome
epochs are already at the optimum; more is flat-to-negative) and the **starting basin**.

**The pattern.** Nothing on the continuous side moves the metric except changing where it starts
— with the single, recent exception in §7.

---

## 7. Open right now, ranked

| item | what it is | why it is ranked here |
|---|---|---|
| **H52** | Build the sequential pair-relocation pass into the pipeline as a stage | Already measured **+0.0223 pp** post-hoc on connectome (1.86× that branch's 0.012 pp minimum effect size) and +0.020 on mouse. Closes 4.8% of the remaining gap. Just needs to be a real stage and screened. |
| **A-VIOL** | Generalise the one-sided surrogate: spend gradient only on *violated* edges | H38 gains **+0.3668 pp** on connectome (9× the gate, the largest gradient-side gain in the project) but **regresses −0.6689 pp on MICrONS** → *graph-dependent*, not general. Controls rule out the two obvious confounds: the mirror shape loses 2.89 pp, and a β×4 sigmoid loses 0.58 pp. Prime suspect for the MICrONS failure: the margin is an absolute constant while MICrONS is ~4× denser. |
| **A-SUB → H39** | Replace the sift's damping with *randomisation*: move each node fully with probability p, instead of all nodes a fraction α | **+0.0116 pp** over the shipped rebuild (5 seeds, all above it) at ~20% less wall-clock, and it converges to a true fixed point where damping leaves 74 nodes moving. Attacks exactly the component that finding #7 identified as 88% of stage-4 cost. |
| A-ALT | Alternate the discrete and gradient phases | Measured: the gradient phase drags a good order **down 0.14 pp** at the operating scale, exactly as §5 predicts. After the honest control, the kick is worth **+0.018 pp** — real but single-seed, and structurally it is the perturb-and-repair family that was already killed. |
| Q03 | Re-measure the gap structure against the *current* champion | All published gap-structure numbers were measured against the 82.93% order and are stale by 1.25 pp. |
| **Track C** | Random graphs vs. real brains — *thesis goal #2* | **Not started.** Only generator scaffolding exists. This is half the thesis. |

---

## 8. Honest gaps and risks

**1. The record is split across two diverged branches, and neither has the whole story.**

| branch | ahead by | holds |
|---|---|---|
| `auto/campaign` | 39 commits | findings #6–#7, the champion pipeline (84.15%), H41–H52 |
| `phase6-global-discrete` | 9 commits | H37/H38, the surrogate gate, A-SUB/A-ALT, this file |

`git checkout` of either branch reproduces only part of the record. **This must be resolved
before the thesis** — it is the single highest-value piece of housekeeping.

**2. Variant IDs have collided across the branches.** `H36`, `H37`, `H38` and `H39` each mean
*different things* on the two branches (e.g. `H38` = "Gauss-Seidel sift, killed" on one and
"one-sided surrogate, +0.37 pp" on the other). Any merge must renumber, and every citation by ID
in the write-ups needs checking.

**3. Cross-branch numbers in this file were read from that branch's committed documents**
(`findings.md`, `log.md`) — they are not re-verified here. The numbers from
`phase6-global-discrete` were re-scored today against the frozen oracle.

**4. Two numbers are flagged in their own sources as not reproducible from a clean checkout**:
one connectome seed of the collective-move sizing, and a mover-count control that was written but
never executed. Both are declared in `log.md`; neither is load-bearing for a shipped result.

**5. The reference solution's provenance is unresolved.** Our `data/best_solution` scores
84.6147% while Vahidi 2025 publishes 84.6125% — about 907 weight units apart, so they are **not
the same solution**, and some documents conflate them (open as Q03).

**6. MICrONS is under-explored relative to connectome.** Several diagnostics cannot be run there
at all because it has no reference solution.

---

## 9. Where everything lives

| file | what it holds | read it when |
|---|---|---|
| **this file** | current state, both branches | orientation |
| `experiments/findings.md` | the ranked confirmed wins, with full evidence tables | you need the numbers behind §4 |
| `experiments/diagnosis.md` | the *why* — Q01–Q05 | you need §5 in depth |
| `experiments/log.md` | full chronological journal, one entry per experiment | you need to know exactly what was run |
| `experiments/backlog.md` | every hypothesis, live status | picking the next experiment |
| `experiments/roadmap.md` | cross-track priority index | deciding what matters next |
| `experiments/TODO_STATUS.md` | the original 7 ideas, five-minute read | ⚠ items 2 and 3 are outdated by the 18 Aug results |
| `reports/methods_walkthrough.md` | first-principles talk script — ⚠ predates findings #6–#7 | presenting the method |
| `experiments/PROTOCOL.md` | the rules: gates, seeds, what counts as a win | before claiming any result |

**The rules that make the numbers trustworthy** (`PROTOCOL.md` + `CLAUDE.md`): the scorer and
harness are frozen and checksum-verified before every run; every reported number traces to a
logged `results/*.json` with a re-runnable command; ≥3 seeds with mean ± std; a gain inside noise
is not a gain; one git commit per experiment; and no algorithm may ever read the target metric.

---

*Generated 18 Aug 2026. Numbers from `phase6-global-discrete` were re-scored against the frozen
oracle on the day; numbers from `auto/campaign` are quoted from that branch's committed findings
and log.*
