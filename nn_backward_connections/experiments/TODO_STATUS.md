# The original 7-item TODO — what we tried, and what came of it

A one-page status of the seven research ideas from the original list. Written to be read top
to bottom in about five minutes.

**Every number below comes from a logged run**, and where a result is scoped to one graph the
scope is stated. Two small glosses, used throughout:

- **pp** = percentage points of total edge weight — the quantity we are maximising.
- **screen gate** = the minimum improvement a variant must show on a dataset before it is worth
  testing further. It is set at twice that dataset's seed-to-seed noise: 0.04 pp on the fly
  connectome, 0.002 pp on MICrONS. Mouse is too noisy to gate on, so there we only require
  "not appreciably worse" (non-inferiority).

| | status | meaning |
|---|---|---|
| ✅ | **answered** | finished, holds up |
| 🟡 | **partial** | measured, but the answer is incomplete or scoped to one graph |
| ⬜ | **not tried** | still open — the reason it was deprioritised is given |
| ❌ | **tried and killed** | measured, did not work, and we know why |

---

## Summary

| # | idea | status | outcome in one line |
|---|---|---|---|
| 1 | Seed-to-seed distance without the greedy start | ✅ | Score is stable, the **ordering is not** — many near-equal orderings exist |
| 2 | Stochastic gradient descent on a subset of neurons | 🟡 | Discrete version tried once, on one graph: a small but consistent edge over the current rule. Continuous version still untried |
| 3 | Alternate the discrete and gradient phases | 🟡 | Probed once: the gradient phase does pull a refined order downward. A possible small gain rests on one seed |
| 4 | Why does starting from the best solution get worse? | ✅ | The dramatic version was our own measurement error; a **small real loss remains** |
| 5 | Initialise with very close neuron positions | 🟡 | Works and is explained by theory; screen 8/9 runs — but later stages erase the effect |
| 6 | SCC decomposition from the paper | ✅ | **+0.18 pp**, took the lead on all three datasets when it landed |
| 7 | A different objective function (flat top, slower tanh) | 🟡 | As literally posed it was already answered — but **dropping symmetry gave our largest gradient-side gain (+0.37 pp), on one graph only** |

**The pattern across all seven.** Almost everything that generalises has come from the
**discrete** side of the algorithm. The one standing exception is the greedy warm start, which
is confirmed on all three graphs but worth only +0.05 pp on the connectome. The continuous side
has otherwise produced one large but graph-dependent result (item 7) and a series of clean
failures.

---

## 1. How far apart are results across seeds, without the greedy start? ✅

**What we did.** Ran the plain gradient algorithm from random starts at several seeds and
compared both the scores and the resulting orderings. Numbers below are for the **fly
connectome**; MICrONS was deferred.

**Result.**
- Scores barely move: **82.8845 ± 0.0225 %**.
- The orderings do move: **Spearman 0.9577 ± 0.0012** — similar, but not the same solution.
- Neurons at the **end** of the line are placed more consistently than those at the front
  (overlap of the top 1000: **0.68 at the back vs 0.37 at the front**), but that asymmetry
  disappears by the top 10,000 (**front 0.783, back 0.794**).
- On the much smaller mouse graph the spread is larger (score std 0.218, Spearman 0.907).

**What it means.** The algorithm does not converge to *one* good ordering but to a large family
of near-equivalent ones. This retroactively explains why an earlier idea — run it K times and
keep the best — gained nothing (+0.0003 pp): there is no spread in the score to harvest.

*Evidence: `experiments/diagnostics/q02_seed_distance.py` → `diagnosis.md § Q02`.*

---

## 2. Stochastic gradient descent — move only part of the neurons 🟡

**Status: the discrete version has now been tried once. The continuous version still has not.**
The idea still splits in two, and only one half was tested.

- **As a continuous method** (gradient steps on a random subset of *neurons*) — **not tried.**
  It remains the nearest neighbour of H13, which subsampled a random set of *edges* each step and
  lost **−0.84 pp** on the fly connectome.
- **As a discrete method** (apply the refiner's moves to only a random subset of nodes each
  sweep) — **one short test, described below.**

**What was actually run.** Fly connectome only. One fixed starting order, the same for every arm,
22 sweeps each. The comparison is between two ways of damping the refiner: the rule we currently
ship, where *every* node that wants to move goes 70% of the way, versus moving a *random 70% of
those nodes* the whole way. The two are the same in expectation.

| rule | score | nodes still wanting to move | time |
|---|---|---|---|
| every node moves 70% of the way (current) | 83.8931 | 1,386 | 81 s |
| a random 70% of nodes move all the way | **83.9047 ± 0.0009** (5 seeds) | 64–173 | 63 s |
| every node moves all the way (older rule) | 83.8086 | 5,162 | 120 s |

**What we are reasonably confident about.** On this graph and from this starting order the
randomised rule came out **+0.0116 pp** ahead, and it did so on all five random seeds, none
overlapping the other arm. It also left far fewer nodes still wanting to move, and finished
faster.

**What this does not establish.** One graph, one starting order, and it has not been through the
project's official harness or run on the other two datasets. The arm it beats is our stage-3
refiner, which is not the final stage of the current best pipeline. Nothing is promoted on this.

*Evidence: `experiments/outputs/proto_asub_aalt.json` and `..._control.json`; script
`experiments/proto_asub_aalt.py`. H13 in `experiments/log.md`. Filed as `A-SUB`.*

---

## 3. Alternate the discrete and gradient phases 🟡

**Status: probed once, on a single seed.**

- **Tried and won, unchanged:** alternating **two different discrete moves** — block moves and
  single-node moves. Each re-opens improvements the other has exhausted. This is stage 4 of our
  best pipeline.
- **The discrete↔gradient pairing** was probed for the first time; the scaffold for it had been
  written earlier but never actually run.

**What was actually run.** We took one stored refined order (83.9035% on the fly connectome), ran
150 gradient steps from it, and measured the score afterwards — at three different position
scales and with two different objectives.

| objective | position scale | score after the gradient steps |
|---|---|---|
| the standard one | the scale training actually uses | 83.7591 (−0.1444) |
| the standard one | ~10× larger | 83.8961 (−0.0074) |
| the one-sided one from item 7 | the scale training actually uses | 83.8468 (−0.0567) |

**What we are reasonably confident about.** The gradient phase really does pull a refined order
*downward* at the scale the algorithm runs at. Item 4 predicted this, but had only ever measured
it starting from the reference solution; this is the first time it was measured starting from one
of our own refined orders. At a much larger scale the gradient barely moves anything.

**The one number that might be a gain, and why we do not lean on it.** Re-refining after the
gradient step ended **+0.0211 pp** above the order we started from — but re-refining with *no*
gradient step at all already gives **+0.0032 pp**, so at most **+0.0179 pp** can be credited to
the gradient step. That is one seed and one configuration. It is also the same "disturb, then
repair" shape as an earlier idea that was measured and killed, so it would need to be compared
against that, not against the untouched order.

*Evidence: `experiments/outputs/proto_asub_aalt.json` and `..._control.json`; script
`experiments/proto_asub_aalt.py`. Stage 4 in `findings.md` #6 (branch `auto/campaign`).*

---

## 4. Why does starting from the best known solution make things worse? ✅

**Short answer: the dramatic collapse we originally recorded was an artefact of how we set the
probe up. A small genuine loss remains. And the experiment says nothing at all about whether
the good solution is stable — which is what we first concluded from it.**

The original probe placed the best ordering with the neurons spread over a standard deviation of
about **0.58**, while the algorithm actually operates at about **141** — roughly 240× wider. At
that tiny scale the objective cannot distinguish orderings at all, and Adam simply walks across
the whole position range and overwrites whatever it was given. Hence the alarming −5.5 pp.

Re-running the same test across scales:

| starting spread (std) | 0.58 | 10 | 50 | **141 (where the algorithm runs)** | 5,000 | 53,600 |
|---|---|---|---|---|---|---|
| loss from the best solution | −5.51 pp | −2.35 | −0.97 | **−0.47 pp** | −0.01 | −0.00 |

So at the true operating scale the loss is **−0.47 pp** — an order of magnitude smaller than we
reported, but not zero.

**The correction we had to make to ourselves.** We initially read the perfect hold at std ≈
53,600 as "the best solution is a stable resting point". It is not evidence of that: at that
scale the gradients underflow and the optimiser is effectively frozen, so *every* ordering holds
— including our own 82.92% one. That caveat is now written into the diagnosis explicitly.

**What the corrected analysis does establish.** The barrier is **reachability**. There is a
crossover at a spread of about **459**: below it the objective actually ranks our own ordering
*above* the better one, so no gradient step points toward the better solution from where the
algorithm lives. A related result fell out of the same work: the sharpness parameter and the
position scale are provably the same knob (the objective is bit-identical whenever their product
is held fixed), so tuning one is tuning the other.

*Evidence: `experiments/outputs/q01_drift.json`, `diagnosis.md § Q01` (including its "what this does NOT establish" section).*

---

## 5. Initialise all neurons very close together 🟡

**Status: theory done, prototype passed, screen 8 of 9 runs complete.**

**Why it works (derived first, then measured).** When all positions start very close together,
every "which of these two comes first?" term in the gradient is suppressed quadratically, and
what remains is a single fixed vector: each neuron's outgoing weight minus its incoming weight.
Adam keeps only the *sign* of that vector, so the cloud immediately splits by weight imbalance —
the *sign* of the very quantity the classical greedy heuristic sorts on (the magnitude is
discarded, so it is not the full greedy signal). The algorithm effectively
**warm-starts itself** instead of spending gradient steps undoing a meaningless random draw.
Predicted suppression exponent 2; measured **2.006** and **1.996** — on the mouse graph and a
400-node synthetic, never on a large connectome.

| dataset | tight init | baseline | difference | gate | verdict |
|---|---|---|---|---|---|
| connectome | 82.9472 ± 0.0044 | 82.8957 ± 0.0185 | **+0.0514 pp** | 0.04 | pass |
| microns (2 of 3 seeds) | 83.1237 ± 0.0013 | 83.1168 ± 0.0001 | **+0.0069 pp** | 0.002 | pass, incomplete |
| mouse | 92.4359 ± 0.0000 | 92.0696 ± 0.2624 | +0.3663 pp | non-inferiority | pass, not significant |

**Four caveats worth stating before anyone asks.**
1. The mouse figure looks the largest but is the weakest: its 95% confidence lower bound is
   **−0.05 pp**, i.e. not distinguishable from zero. It passes only the "not worse" bar.
2. It does *not* remove seed variance in general. On mouse the result is fully deterministic
   (std 0.0000), which is the theory's signature — but on the connectome the spread only shrinks
   (0.0185 → 0.0044), and on MICrONS it is slightly *larger* than the baseline's (0.0013 vs
   0.0001).
3. **It improves the baseline, not our best pipeline.** After the discrete refinement stages
   run, the difference between all initialisation schemes collapses. That was measured on mouse
   (spread 0.44 pp → 0.05 pp) and, more weakly, on a synthetic (1.37 → 0.36). There is no
   connectome measurement of this, so the conclusion rests on small graphs.
4. On mouse it is *below* the confirmed greedy warm start (92.4359 vs 92.4793). It is an
   alternative to that warm start, not an improvement on it.

One MICrONS run is still missing, so the screen is not formally complete, and these runs are on
disk but **not yet committed to git**.

*Evidence: `experiments/proto_ainit_scale.py`, `experiments/outputs/proto_ainit_scale.json`, `results/*A_INIT*.json`, `experiments/ainit_RESUME.md`.*

---

## 6. SCC decomposition from the paper ✅

**Result: +0.1837 pp on the fly connectome, and it took the lead on all three datasets when it
landed.**

**The idea.** If a group of neurons occupies an *unbroken stretch* of the line, they can be
rearranged among themselves freely: every edge to a neuron outside that stretch keeps its
direction, because the outside neuron lies either before the whole stretch or after all of it.
So each stretch is an independent sub-problem. Inside it: find the strongly connected
components, lay them out in topological order (every edge *between* components then points
forward), keep each component's internal order, and recurse. This can only improve the result
or leave it unchanged — **correct by construction, not a heuristic gamble.**

**The decisive detail.** Doing this **once** on the whole graph is worthless: it gains
**+0.00013 pp**, because 92.8% of the neurons sit in a single giant *strongly connected*
component. The gain comes from **recursing inside that giant component**.

| dataset | previous best | with SCC blocks | gain |
|---|---|---|---|
| connectome | 83.9135 | **84.0972** | **+0.1837 pp** |
| microns | 83.2063 | 83.2338 | +0.0276 pp |
| mouse | 92.9018 | 92.9170 | +0.0152 pp |

Confirmed on 5 / 5 / 20 seeds with **zero extra gradient steps** — though on this machine every
seed returns a bit-identical result, so those seed counts confirm reproducibility rather than
measuring a spread. In a roughly matched-time control (305 s vs 349 s) the new move class
produced **36× more gain** on the connectome than giving the old refiner those seconds instead
(13× on MICrONS).

**Scope it correctly — three qualifications.**
- It is **not** the project's largest win. The original full-range node sift was worth
  **+0.85 pp**, 4.6× more, and beats this on all three datasets. This is the largest gain from a
  genuinely *new* kind of move since then.
- It **no longer holds the record** on any dataset: later variants have overtaken it on both
  large connectomes and, separately, twice on mouse.
- The mouse result passes as "positive" but **fails** the formal non-inferiority bound, so the
  project logged it as a general win *with a mouse caveat*, not a clean three-dataset win. And
  while it adds no gradient steps, it does cost substantial extra CPU time — the project's own
  reviewer noted that "no extra gradient steps" is not by itself a sufficient fairness argument
  for this kind of variant.

*Evidence: `src/mfas/refine/scc_recursive.py`, `findings.md` #6 (branch `auto/campaign`). One supporting figure — the +0.00013 pp one-shot result — still traces only to an uncommitted scratch script and should be re-derived before it is quoted in the thesis.*

---

## 7. A different objective function — flat top, slower tanh 🟡

This produced the most interesting result on the list, and the most carefully qualified one.

### 7a. The idea as literally posed was already answered — ❌

Run properly rather than dismissed by analogy, and the analogy turned out to be right for an
unexpected reason: **`tanh` is an exact rescaling of the S-curve**, so "a slower-decaying tanh"
is just the S-curve at a different sharpness — an axis we had already killed. A hard flat top
*and* bottom is the shape we killed earlier as H11.

The one genuinely untested axis inside that family was the **tail exponent**: a curve whose tail
decays polynomially rather than exponentially, so distant pairs keep exerting a pull. It passed
a theory gate and then lost on the connectome — **−0.45 pp** at matched core width and
**−0.93 pp** at its own narrower width — and was within noise on the small mouse graph.

### 7b. Dropping the symmetry — constant above a margin, tanh below — 🟡 **graph-dependent**

Every shape tested until this point satisfied $g(-z) = 1 - g(z)$. Abandoning that is a different
move, and it is where the result is. **Flat on the feedforward side, tanh on the feedback side:**
edges that are already comfortably correct get *zero* gradient, so the whole budget goes into
pulling feedback edges into place.

$$g(z) = 1 \quad (z \ge M), \qquad g(z) = 1 + \tanh\!\big((z-M)/T\big) \quad (z < M)$$

with $M = 0.75$, $T = 1.5$. Matched to the baseline's gradient budget on each dataset
(connectome 20,000 steps; MICrONS 80,000; mouse 5,000).

| dataset | baseline | asymmetric | difference | screen gate | outcome |
|---|---|---|---|---|---|
| **connectome** | 82.8958 ± 0.0187 | **83.2626 ± 0.0108** | **+0.3668 pp** | 0.04 | **pass, 9× the gate** |
| **microns** | 83.1172 ± 0.0006 | 82.4482 ± 0.0094 | **−0.6689 pp** | 0.002 | **fail** |
| mouse | 92.0696 ± 0.2624 | 92.2053 ± 0.2602 | +0.1357 pp | non-inferiority | pass |

All three connectome seeds positive; all three MICrONS seeds negative.

**How to present this.** It is the largest gradient-side gain we have measured **on the fly
connectome** — seven times the greedy warm start there (+0.0508 pp at n=15). It is also the
first shape change that has ever *helped*: every previous one was null or negative, so this is a
direct counterexample to our own earlier conclusion that reshaping the objective is null on both
large connectomes. (Larger point estimates exist on the 148-node mouse graph — the greedy warm
start and the tight init of item 5 — but that graph has a 0.26 pp noise floor.)
But under our own rules this is **not a general win**: it must clear
both large connectomes, and it loses on the second by nearly twice what it gains on the first.
The correct label is **graph-dependent**, and the honest sentence is *"we found a real effect and
we do not yet know why it does not transfer."*

**Two controls make it trustworthy** (connectome, 3 seeds each):

| control | result | what it rules out |
|---|---|---|
| mirror shape — flat on the *feedback* side instead | **−2.8890 pp** | that *any* one-sided shape would do; the **direction** of the asymmetry is the mechanism |
| plain S-curve at 4× sharpness for the whole run | **−0.5804 pp** | that this is the sharpness/scale knob in disguise |

Two further points of method: a degenerate case at $M = 0$ — where the objective is maximised by
putting every neuron in the same place — was **derived on paper before running** and then
confirmed exactly. We also checked that the gain is not just an artefact of keeping the
best-scoring snapshot — but **only on the two small proxies, not on the connectome where the
+0.37 pp lives**, and those proxy figures are not yet saved in an artifact. That check still
needs doing properly.

**Limitations, disclosed.** Wall clock is ~1.5× the baseline (gradient steps are matched); the
leading suspect for the MICrONS regression is that $M$ and $T$ are absolute constants while
MICrONS is ~4× denser, so they may need to scale with the graph. Two process points matter as
much as the numbers: this result is at **screen stage** (3 seeds), not the 5-seed confirm stage,
and unlike the experiment before it, **it has not yet been through an independent verifier or
critic pass** — the previous one drew a 13-point critique that rewrote most of its surrounding
claims. A second parameter setting, $(M,T) = (0.25, 0.75)$, was also run on the connectome and
also cleared the gate (+0.1442 pp); it is in `results/` but not written up anywhere. **And the most important untested question: this is a
pure gradient-stage result — the improved ordering has never been passed through our discrete
refinement stages, which is exactly where item 5's advantage was erased.**

### 7c. Multi-band objective — ❌ killed at the prototype gate

A follow-up: one blur width cannot serve both near and distant pairs, so add a second, sharper
band on top of the existing curve. **Twelve configurations** on the connectome (a nine-cell
parameter grid plus three rescue variants), one seed each; **the unmodified control beat every
one of them**, by between 0.31 and 2.43 pp. Written up in `backlog.md`, not yet in the log.

*Evidence: `experiments/log.md` (2026-08-17), `experiments/outputs/{q04_surrogate_tails,q05_asymmetric_surrogates,proto_h38_asym,proto_h38_sweep,proto_amband}.json`, `results/*H3[78]*.json`.*

---

## What the list as a whole has taught us

1. **The discrete side carries almost everything that generalises.** Item 6 and the refinement
   stages produced nearly every confirmed, all-three-datasets gain. The one exception is the
   greedy warm start, which is an initialiser for the gradient stage and is confirmed on all
   three graphs — but it is worth only +0.05 pp on the connectome.
2. **Item 7 is the real crack in that story, and it is now the most active thread.** The
   asymmetric objective moved the fly connectome by +0.37 pp — far more than anything else on
   the gradient side ever has. The principle behind it ("spend gradient only on constraints that
   are violated") has been generalised into a follow-up programme, currently the top-priority
   item, with four concrete sub-experiments including composing it with the discrete refiner.
3. **Two items were resolved by finding our own mistakes**, not new effects — item 4 overturned
   a claim we had written down, and item 1 explained a failure we had recorded but not understood.
4. **The remaining gap is 0.46 pp** between our best current pipeline (84.1541%) and the best
   ordering on the FlyWire leaderboard (84.6147%). A caution on that target: it is a downloaded
   submission file whose method we could not retrieve — it is **not** the figure published in
   Vahidi 2025, which is 84.6125%. Item 7's follow-up is the only thread on this list currently
   expected to make a serious dent in the gap.

> **A naming warning.** The label `H38` is used for *two different experiments*: the asymmetric
> objective above (this branch), and a change to the refiner's update order (the
> automated-campaign branch), which was killed. Say "the asymmetric surrogate" rather than "H38".
