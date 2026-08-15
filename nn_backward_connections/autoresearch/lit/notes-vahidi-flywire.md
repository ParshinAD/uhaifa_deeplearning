# Notes — the prior art that actually beats us on this exact graph

Scan L01, 2026-08-16. Sources retrieved as listed in `README.md`.
**Caveat carried through the whole file:** the local `2506.13799v1.pdf` could not be opened (no
PDF rasterizer, Bash disabled). All Vahidi-2025 content below is from the arXiv **HTML**
rendering. The Vahidi & Koutis 2026 paper is **abstract-only** — the full text was not read.

---

## 1. What I read

| # | source | id / venue | date | link |
|---|---|---|---|---|
| S1 | Soroush Vahidi, *Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization* | arXiv:2506.13799v1 | 2025-06-13 | https://arxiv.org/abs/2506.13799 · https://arxiv.org/html/2506.13799v1 |
| S2 | Soroush Vahidi, Ioannis Koutis, *Interval-Based Refinement Algorithms for Large-Scale Weighted Feedback Arc Set* | SSRN 6221201 | posted 2026-02-11 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6221201 — **403, abstract only** |
| S3 | FlyWire *Minimum Feedback Challenge* leaderboard | codex.flywire.ai | read 2026-08-16 | https://codex.flywire.ai/app/mfas_challenge |
| S4 | Soroush Vahidi, Ioannis Koutis, *Minimum Weighted Feedback Arc Sets for Ranking from Pairwise Comparisons* | arXiv:2412.16181v2 | 2025-01-07 | https://arxiv.org/html/2412.16181v2 |

---

## 2. The leaderboard — and a provenance correction the campaign must make

S3, read 2026-08-16. Total edge weight **41,912,141** (matches our `GraphData`). Challenge closed
2024-10-08; submissions still accepted for leaderboard placement.

| rank | author(s) | forward weight | % (computed here) | date |
|---|---|---|---|---|
| 1 | **Soroush Vahidi, Ioannis Koutis** | **35,463,823** | **84.61465** | **2026-01-09** |
| 2 | David A. Bader, et al. | 35,459,266 | 84.60378 | 2024-11-16 |
| 3 | Dritan Hashorva | 35,452,425 | 84.58746 | 2024-10-28 |
| 4 | Justin Ellis-Joyce, et al. | 35,436,406 | 84.54924 | 2024-10-28 |
| 5 | Justin Ellis-Joyce, et al. (challenge winner) | 35,435,948 | 84.54815 | 2024-10-08 |
| 6 | Xin Zheng, et al. | 35,374,656 | 84.40190 | 2024-10-08 |
| 7 | Dritan Hashorva | 35,364,447 | 84.37754 | 2024-10-08 |
| — | **our champion H42** | **35,270,783** | **84.15417** | 2026-08-10 |
| 8 | David A. Bader, et al. | 35,231,406 | 84.06021 | 2024-10-08 |
| — | benchmark (library) | 29,023,882 | 69.24 | 2024-06-13 |

Three consequences, all checkable arithmetic:

**(a) `data/best_solution` is the current world record, not a published paper's number.**
35,463,823 is leaderboard rank 1, dated 2026-01-09 by **Vahidi & Koutis**. It is *not* the number
in arXiv:2506.13799v1, which reports **35,462,925 = 84.61253%** — 898 units and **0.00214 pp**
lower. `autoresearch/sota.json` currently says the 84.6147 reference was *"independently reached by
Vahidi 2025 (arXiv:2506.13799) with greedy + bounded-span insertion + SCC, no MIP"*. That
attribution is off by one paper: the published 2025 method reaches 84.6125%; 84.6147% belongs to
the 2026 SSRN follow-up (S2). `findings.md` #4 quotes the SOTA as "84.61% = 35,462,925/41,912,141",
which is the *paper's* number and is correct there — the two documents disagree with each other by
898 units. Worth a one-line fix in `sota.json` `targets.connectome.reference_source`; it changes no
measured quantity, but this project's rule is that provenance is part of the number.

**(b) Cross-check that the leaderboard and the paper are consistent.** S1 states its improvement
margin over "BA-CD" as **3,659 units**. 35,462,925 − 35,459,266 = 3,659, and 35,459,266 is exactly
leaderboard rank 2 (Bader et al., the Rocket-Crane paper). The two independent sources agree
exactly. So Rocket+Crane, the 20-day Gurobi pipeline our `CLAUDE.md` says is unreproducible here,
scores **84.6038%** — and the cheap combinatorial method beats it by 0.0109 pp.

**(c) Calibration the campaign does not currently have.** Our champion at 84.1541% would sit
**8th of 9** on this leaderboard. The remaining 0.4606 pp is not a rounding artefact — **seven
public solutions live inside it**, five of them from October 2024. It is a genuinely contested
region, not a last-mile grind.

---

## 3. S1 — mechanism extracted (in our vocabulary)

S1 is a pipeline of five algorithms. Reported trajectory: greedy **0.7524** → refinements
**0.8461**. Structural facts it reports about the graph, which match ours
(`dr_tmp/FINDINGS_underrelaxation.md` E2): largest SCC 126,840 nodes = 92.8%; 9,503 singleton SCCs;
density 3.03e-4.

### 3.1 Alg. 1 — greedy ratio ranking (`AdaptiveOutOverInPlus1Ranking`)
Score each node `s(u) = (out_w(u)+1)/(in_w(u)+1)`, push into a max-heap, repeatedly pop the highest
and append it to the front of the order, updating the weighted degrees of the unranked neighbours
as nodes leave. **Reaches 0.7524 on the connectome.** Our `H02` greedy-FAS (Eades–Lin–Smyth /
GreedyAbs peel, `findings.md` #1) reaches **0.6891** on the same graph. Same asymptotic cost, +6.3 pp.

### 3.2 Alg. 2 — `RefineRankingWithExtendedStrategy` — **the one that matters**
This is the step that carries 0.7524 → 0.8461. It is *edge-driven*, which nothing in our pipeline is.

- Maintain a max-heap of the currently **backward** edges `(u,v)` (i.e. `π(u) > π(v)`), keyed by weight.
- Pop the heaviest. If it is still backward, take the block
  `B = {x : π(v) < π(x) < π(u)}`, so the contiguous position range reads `[v, n_1, …, n_t, u]`.
- Search over split indices `r = 0 … |B|` for the rearrangement
  `[n_1, …, n_r, u, v, n_{r+1}, …, n_t]` maximising the gain. The paper's stated gain is

      Δ = w_uv − w_vu + in_v − out_v − in_u + out_u

  computed with **prefix sums**, and it states that only edges with **both endpoints inside the
  block** matter.
- Accept if `Δ > 0`; otherwise try three fallbacks — swap `u` and `v`; push `v` forward; pull `u`
  backward. Then push the newly created backward edges onto the heap. Keep the ranking if the
  forward weight improved, else restore.

**Why the gain is exact, and what the terms are** (my derivation, checked against the paper's
formula — see the caveat at the top; this is not a quotation of a proof in the paper):

The block is a *contiguous position range*, so by the same contiguous-block lemma that
`mfas/refine/scc_recursive.py` and `mfas/refine/segment.py` already rest on, no edge with an
endpoint outside `[π(v), π(u)]` can change orientation. Inside the block the `n_j` keep their
relative order, so edges among the `n_j` never flip either. Only edges incident to `u` or `v` flip,
and each flips on exactly one side of the split. Therefore

    Δ(r) = (w_uv − w_vu)
         + Σ_{j ≤ r} [ w(n_j → v) − w(v → n_j) ]     # v moves from before n_j to after it
         + Σ_{j > r} [ w(u → n_j) − w(n_j → u) ]     # u moves from after n_j to before it

which is exactly the paper's `Δ = w_uv − w_vu + in_v − out_v − in_u + out_u` with `in_v`/`out_v`
accumulated over the prefix `j ≤ r` and `in_u`/`out_u` over the suffix `j > r`. **Exact, no rescore.**

**Cost — much lower than it looks.** `Δ(r)` is piecewise constant in `r` and changes only at block
positions that are neighbours of `u` or of `v`. So the optimal `r` costs
`O((d(u)+d(v)) log(d(u)+d(v)))` and is **independent of the block length**. On the connectome the
mean total degree is `2·5,657,719/136,648 ≈ 83`.

### 3.3 Alg. 3 — `RefineSCCBlocks`
Cut the largest SCC into consecutive blocks of fixed size `s`; for each block take the induced
subgraph, decompose into sub-SCCs, lay them out in topological order of the condensation; **if a
sub-SCC has ≤ 9 nodes, permute it exhaustively to maximise internal forward weight**, otherwise keep
the incoming relative order. Accept the block reorder only if total forward weight improves.

This is our `H36` stage 4 **plus one thing we do not do**: exact optimisation of small components.
`SccRecursiveRefiner` returns immediately for any block of size `≤ min_block = 32` and always keeps
each SCC's nodes in their current relative order.

### 3.4 Alg. 4 — `FlatPartitionReorder`
Take a rank interval `[s,e]`, partition it into `x^ℓ` groups of roughly equal size; slide a window of
`x` consecutive groups; compute the inter-group weight matrix `W_{i,j}`; evaluate **all `x!`
permutations** with `FW(σ) = Σ_{i<j} W_{g_i, g_j}`; take the argmax; reorder if it differs. The paper
notes it deliberately implemented this *flat* rather than recursively, for memory.

This is a **coarsened Linear Ordering Problem**: contract contiguous rank groups into supernodes and
solve the LOP on the small dense matrix. Our `H41` segment move is precisely the `x = 2`,
adjacent-only, span-≤2048, powers-of-two special case of this. The generalisation is the basis of
hypothesis **H44**.

### 3.5 Alg. 5 — `SCCBasedGlobalRanking`
Global condensation → topological order → exhaustive permutation for SCCs of size ≤ 9, otherwise
preserve input order; accept if forward weight improves. Our `dr_tmp/FINDINGS_underrelaxation.md` E2
already measured the one-shot version of this on our graph as worth **+0.00013 pp**, which is why
`H36` is recursive. Nothing new here except, again, the ≤9 exact step.

---

## 4. S2 — the 2026 follow-up (ABSTRACT ONLY — not read)

Retrieved only as an abstract, via two independent search queries returning consistent text.
Attributed statements below are the abstract's claims, **not verified against a full text.**

> "…a hybrid refinement framework that couples interval-based local improvement with SCC-guided
> structural reorganization, using a **dynamic program to select large sets of non-overlapping
> backward-edge intervals**, then refining them **in parallel** using a **two-level local search**,
> yielding substantial early gains with near-linear scaling on multi-core hardware. The structural
> component focuses on **contiguous rank blocks within the giant SCC**, applying block-level updates
> that expose new opportunities for subsequent local exploitation. A **lightweight controller
> alternates between the two mechanisms**…"

Three things follow.

1. **They alternate two disjoint move classes and say so.** That is exactly the architecture of our
   `alternate_scc_sift` (`findings.md` #6/#7). Independent convergence on the design — mildly
   reassuring, and it means the remaining 0.46 pp is *not* an architecture difference.
2. **Their "interval" is S1's Alg. 2 block, industrialised.** A backward edge `(u,v)` defines the
   interval `[π(v), π(u)]`. Two *disjoint* intervals are independent — by the same contiguous-block
   lemma, and their exact gains **add**. So "select a large non-overlapping set by DP" is the classic
   weighted-interval-scheduling DP (sort by right endpoint, `O(k log k)`), and it is what converts a
   strictly sequential repair loop into a **batched, vectorisable sweep**. Our `segment.py`
   `select_disjoint_moves` does the greedy version of exactly this trick for a different move class;
   the DP is the optimal version.
3. **"near-linear scaling on multi-core hardware"** is a runtime lever we do not use at all — every
   refiner in `src/mfas/refine/` is single-threaded NumPy. Relevant to **P07** (microns runtime),
   not to score.

---

## 5. S4 — checked and discarded

Vahidi & Koutis, arXiv:2412.16181v2 (2025-01-07), *Minimum Weighted Feedback Arc Sets for Ranking
from Pairwise Comparisons*. Cycle-cancelling DFS + ternary search on a "ratio upset loss". Largest
instance: **602 vertices, 5,002 edges**; runtimes 0.01–9.65 s in Python. Different objective
(ratio upset loss, not forward weight) and 227× fewer nodes than our graph. **Nothing transfers.**
Recorded so a future scout does not re-read it.

---

## 6. Does it transfer? — per mechanism

| S1/S2 mechanism | transfers? | why |
|---|---|---|
| Alg. 2 backward-edge interval repair | **yes — highest value** | exact gain, cost `O(d(u)+d(v))` per candidate independent of span, unbounded range. Becomes **H45**. |
| S2 non-overlapping-interval DP + batched apply | **yes, as the enabler for the above** | disjoint contiguous intervals compose exactly; turns a sequential loop into a vectorised sweep. Folded into **H45**. |
| Alg. 4 coarsened `x!` group permutation | **yes, and improvable** | `x!` brute force caps `x ≈ 5`; subset-DP (Held–Karp) solves the coarse LOP exactly in `O(2^x · x²)`, so `x = 16` costs ~1e6 ops instead of `16! = 2e13`. Becomes **H44**. |
| Alg. 3/5 exhaustive permutation of SCCs with `|S| ≤ 9` | **maybe — cheap to size** | we currently do *nothing* below `min_block=32`. Becomes **H46**, with a free CPU sizing gate. |
| Alg. 1 ratio greedy (0.7524 vs our 0.6891) | **gated** | see `hypotheses.md` §Gated-1; it is live only if **S01** shows the Rocket phase is cuttable. |
| S2 multi-core parallelism | **out of scope for score** | relevant to P07 runtime only. |

---

## 7. The conflict check that matters most: does S1 contradict meta-rule M4?

`killed.json` **M4** says bounded-window single-node local search recovers ≤ 0 for `W ≥ 100`, and
demands that any local-move proposal be "global-range or structurally decomposed (SCC), not
window-bounded".

`sota.json`, `CAMPAIGN.md`, `findings.md` #3-banner and #4, and `queue.json` H36 all describe the
prior art as **"bounded-span insertion"**. Read against M4 that phrasing looks like a direct
contradiction — the prior art appears to win with exactly the thing M4 killed.

**It is not a contradiction; the campaign's phrasing of the prior art is simply wrong.** S1's
Algorithm 2 is not span-bounded at all: the block is `[π(v), π(u)]`, whose length is set by the
*violated edge*, and the heap is ordered by weight, so the heaviest violations — which by
`diagnosis.md` Step 2 have median rank distance ~22,580 — are processed first. The only genuinely
span-bounded object in S1 is Alg. 3's fixed block size `s`, which is a *structural* decomposition of
the giant SCC, i.e. M4's own named revival condition.

So the prior art **agrees with M4** rather than contradicting it, and the campaign has been carrying
a description of the prior art that made its own kill index look unreliable. That is the single most
useful thing in this scan: it removes a false tension and it identifies, precisely, the one
long-range move class the campaign never built.
