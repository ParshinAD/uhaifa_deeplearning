# Notes — scalable FAS, multilevel ordering, and the honest negatives

Scan L01, 2026-08-16. This file is mostly negative results. Negatives are the point: they stop the
campaign from hoping on an axis where the literature has nothing.

---

## 1. Scalable FAS heuristics — wrong objective, and I could not read them

| # | source | venue | date | link | status |
|---|---|---|---|---|---|
| F1 | *Finding small feedback arc sets on large graphs* | Computers & Operations Research (Elsevier), S0305054824001965 | 2024 | https://www.sciencedirect.com/science/article/abs/pii/S0305054824001965 | **403 — abstract only, via search snippets** |
| F2 | *Efficient heuristics to compute minimal and stable feedback arc sets* | J. Combinatorial Optimization, 10.1007/s10878-024-01209-8 | 2024 | https://link.springer.com/article/10.1007/s10878-024-01209-8 | title/abstract from search results only |
| F3 | *An Exact Method for the Minimum Feedback Arc Set Problem* | ACM JEA, 10.1145/3446429 | 2021 | https://dl.acm.org/doi/10.1145/3446429 | not fetched |
| F4 | Eades, Lin, Smyth / Simpson et al., *A Fast and Effective Algorithm for the Feedback Arc Set Problem* | J. Heuristics | classic | https://link.springer.com/article/10.1023/A:1011315014322 | already implemented here as `H02`'s greedy |

**What F1's abstract claims** (not verified against a full text): systematic **reduction rules** that
shrink the input without losing optimality, usable as a preprocessor for any FASP algorithm, plus a
**divide-and-conquer** heuristic; on circuit benchmarks it improves existing heuristic solutions by
24–40%.

**Does it transfer? Largely no, and for a structural reason.** The FASP literature almost universally
solves the **cardinality / unweighted** problem on **sparse, near-acyclic** graphs (circuit
benchmarks), where reduction rules bite: a vertex with in-degree 0, a bridge, a chain, a
2-cycle-free region can be committed and removed, and the residual "core" is a small fraction of the
input. Our instance is the opposite: **92.8% of the nodes are in one SCC**, the objective is
**weighted forward mass** rather than arc count, and the reference solution still leaves
**15.39%** of the total weight backward (6,448,318 of 41,912,141). There is no near-acyclic skeleton
to peel. Our own measurement already says this: the one-shot condensation is worth **+0.00013 pp**
(`dr_tmp/FINDINGS_underrelaxation.md` E2), and the SCC-recursive version — which is exactly the
"divide and conquer" idea, done properly — is already the champion (`findings.md` #6).

I make **no claim** about F1's actual algorithms; I could not read them. If a future cycle obtains
the full text, the specific thing to look for is whether any reduction rule survives when the
objective is weighted and the graph is one giant SCC. My expectation is that it does not, but that is
an expectation, not a finding.

F3 (exact MFAS) is an ILP/branch-and-cut method. **Out of scope by inspection:** the linear-ordering
ILP formulation needs `O(n²)` ordering variables (1.87e10 for us) or `O(m+n)` variables with `O(m)`
constraints for the arrangement formulation plus exponentially many cycle cuts. `CLAUDE.md` already
records that the Crane MIP needs Gurobi and ~20 days. Nothing here changes that.

---

## 2. Multilevel / coarsening for graph ordering — right idea, wrong objective, one usable skeleton

| # | source | link |
|---|---|---|
| M1 | Safro, Ron, Brandt, *Graph minimum linear arrangement by multilevel weighted edge contractions* | https://www.researchgate.net/publication/222663254 |
| M2 | Koren, Harel, *A Multi-Scale Algorithm for the Linear Arrangement Problem* | https://link.springer.com/chapter/10.1007/3-540-36379-3_26 |
| M3 | Safro, Temkin, *Multiscale approach for the network compression-friendly ordering* | https://arxiv.org/pdf/1004.5186 |
| M4 | Safro, Ron, Brandt, *Relaxation-based coarsening and multiscale graph organization* | https://arxiv.org/pdf/1004.1220 |
| M5 | Karypis, Kumar, *A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs* | https://epubs.siam.org/doi/10.1137/S1064827595287997 |

These solve **minimum linear arrangement**: minimise `Σ w_uv · |pos(u) − pos(v)|` on an **undirected**
graph. The three-phase skeleton is coarsen → solve small → interpolate/refine while uncoarsening, with
a fast Kernighan–Lin variant as the refinement, scaling to millions of nodes. M4 explicitly names
"more sophisticated multiscale organization for **directed** graphs" as future work — i.e. the
directed case is open in that community.

**What does not transfer.** Their coarsening contracts *heavy edges*, because in MinLA a heavy edge
means "these two nodes want to be near each other". In MFAS a heavy edge `u→v` means "`u` wants to be
*before* `v`" and says nothing about proximity; contracting `u` and `v` destroys precisely the
information the objective is made of. Edge-contraction coarsening is therefore **not** a valid
coarsening for our problem, and no amount of tuning fixes that.

**What does transfer — and this is the bridge.** Coarsen by **contiguous position blocks** instead of
by edges. If the current order is cut into consecutive blocks and each block is kept internally rigid,
then by the contiguous-block lemma the campaign already uses (`refine/scc_recursive.py`,
`refine/segment.py` docstrings), permuting the blocks changes the exact score by exactly the change in
`Σ_{i<j} W[block_i][block_j]`. So:

- the coarse problem is an **exact** surrogate, not an approximation — unusual, and it is a property of
  *this* objective that MinLA does not enjoy;
- the coarse problem is a **dense LOP on `K` supernodes**, which is exactly the object the entire LOP
  metaheuristic literature (`notes-lop-and-ocm.md`) is built for and where `Θ(K²)` is affordable;
- the move it enables is **unbounded in range** — a block can travel the whole line — which is what
  `killed.json` **M4** says the recoverable weight requires.

Vahidi 2025's Algorithm 4 (`FlatPartitionReorder`) is this idea in its brute-force form (`x!` over `x`
consecutive groups, so `x ≲ 5`). Hypothesis **H44** is this idea with the exponential replaced by a
Held–Karp subset DP and the sliding window replaced by a global LOP solve.

---

## 3. Connectome-specific ordering — nothing new, and M7 stands

Query 5. What exists is the **trophic-level / trophic-incoherence** framework (assign each node a
continuous level from the directed flow, measure how coherently edges point up-level) and hierarchical
depth assignments used descriptively in Drosophila connectome papers. The campaign has already tested
this as an algorithm and killed it: **H32** (trophic-level Laplacian warm-start) at −2.12 pp synthetic
/ −1.26 pp mouse, and **H33** (magnetic Laplacian) at −4.97 pp plus a 304–507 s eigensolve. Meta-rule
**M7** covers both.

I found **no** post-2024 work that turns a connectome hierarchy measure into a competitive MFAS
ordering. The connectome-hierarchy literature is descriptive: it uses an ordering to *interpret*
circuits, not to maximise forward weight. **The only algorithmic work on the FlyWire MFAS instance
is the leaderboard entrants** (`notes-vahidi-flywire.md` §2), and of those only Bader et al.
(Rocket-Crane) and Vahidi/Vahidi–Koutis have published methods.

---

## 4. Continuous / differentiable relaxations — nothing that revives M1

I looked, in the course of queries 2, 3 and 8, for anything that would give meta-rule **M1** a
revival condition: a relaxation whose operating scale is not blind in the sense of `diagnosis.md`
Q01 (F depends only on `β·std`, and at `std ≈ 141` the better order is *not* preferred at any
`β ∈ [0.05, 1.05]`).

**I found nothing.** No new differentiable-sorting, soft-rank, or continuous-permutation method that
addresses scale-blindness, and nothing that reports competitive results on a weighted FAS instance of
this size. The published large-scale winners on this exact graph are **entirely combinatorial** — S1's
five algorithms contain no gradient step, and S2's abstract describes only interval repair and block
updates. Rocket-Crane's continuous phase (Bader et al.) is itself superseded by them.

This is a real, load-bearing negative: **M1 is not merely unfalsified, it is corroborated by the fact
that the world record on this instance was set without any continuous component at all.** Any future
cycle proposing a continuous lever now has to explain not only Q01 but also why the two methods above
us on the leaderboard did not need one.

---

## 5. Parallelism — an unexploited axis, but not a score axis

S2's abstract claims "near-linear scaling on multi-core hardware". Every refiner in
`src/mfas/refine/` is single-threaded NumPy. The campaign's binding runtime constraint is **P07**
(microns does not fit 3600 s under load). If a cycle ever needs wall-clock rather than score,
multi-threading the disjoint-move sweeps is the obvious lever, and the disjointness proofs the
campaign already has (`segment.py` module docstring) are exactly what make it safe. I am not queueing
it: Phase 1 is a score phase, and `CAMPAIGN.md` forbids starting Phase 2 unilaterally.
