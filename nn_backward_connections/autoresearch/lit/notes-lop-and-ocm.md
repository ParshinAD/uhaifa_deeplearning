# Notes — the Linear Ordering Problem literature, and PACE 2024 crossing minimization

Scan L01, 2026-08-16. This file exists mainly to record a **scale mismatch** honestly, and to
identify the one bridge across it.

---

## 1. What I read

| # | source | id / venue | date | link |
|---|---|---|---|---|
| L1 | F. Fagiolo, M. Baioletti, V. Santucci, *Linear Ordering Problem: Time for a Change* | arXiv:2605.31051 (accepted PPSN 2026) | v1 2026-05-29, v2 2026-06-05 | https://arxiv.org/abs/2605.31051 |
| L2 | P. Kindermann, F. Klute, S. Terziadis, *The PACE 2024 … Challenge: One-Sided Crossing Minimization* | IPEC 2024, LIPIcs vol. 321, art. 26, DOI 10.4230/LIPIcs.IPEC.2024.26 | 2024 | https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.IPEC.2024.26 |
| L3 | K. Boehmer, L. L. George, F. Hauser, J. Palarus, *Arcee: An OCM-Solver* | arXiv:2411.17596v2 | 2024-11 | https://arxiv.org/html/2411.17596 |
| L4 | J. Rauch, *weberknecht — a One-Sided Crossing Minimization solver* | arXiv:2412.06361 | 2024-12-09 | https://arxiv.org/abs/2412.06361 (**abstract only — PDF text not extractable**) |
| L5 | *PACE Solver Description: OCMu64* | IPEC 2024, DOI 10.4230/LIPIcs.IPEC.2024.35 | 2024 | https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.IPEC.2024.35 (read via search snippets + https://curiouscoding.nl/papers/ocmu64.pdf listing) |

---

## 2. The LOP literature is 450× too small — this is a finding, not a gap

Our problem *is* the Linear Ordering Problem: maximise `Σ_{i<j} W[σ(i)][σ(j)]` where `W` is the
weighted adjacency matrix. So the LOP metaheuristic literature ought to be the natural home. It is
not, and the reason is quantitative.

L1 is the 2026 state-of-the-art survey/benchmark paper. Its **new** benchmark suite, introduced
because the field had been running on 1959–1979 macroeconomic matrices, has instance sizes:

- `isic`: n = **10** (small enough for exhaustive evaluation)
- `rxr`: n = **49**
- `pxp`: n = **89–176** (median 149)
- `os300`: n = **300** — the largest instance in the paper

The two algorithms it evaluates as state of the art:

- **CD-RVNS** — variable neighbourhood search over the **insertion** and **interchange**
  neighbourhoods, alternating with a destruction–construction phase.
- **MA-EDM** — memetic algorithm maintaining a population of local optima under the **insertion**
  neighbourhood, using "a classical acceleration technique that allows the evaluation of all the
  neighbours in **Θ(n²)** time".

Two consequences for us:

1. **`n = 136,648` is 455× the largest instance in the 2026 LOP benchmark**, and the field's standard
   acceleration is `Θ(n²)` per full neighbourhood scan — 1.87e10 operations for one sweep on our
   graph, per iteration. Nothing in this literature is directly runnable at our scale. Any campaign
   member tempted by "the LOP people must have solved this" should stop here.
2. **We already have their best move.** The `Θ(n²)` insertion-neighbourhood acceleration *is* our
   full-range exact-gain sift (`mfas/refine/insertion.py`), which exploits the sparsity of `W` to run
   in `O(m + n)` instead of `Θ(n²)`. On this axis the campaign is ahead of the published LOP state of
   the art, because our `W` is sparse (5.66M nonzeros out of 1.87e10 possible pairs) and theirs is
   dense.

**The one neighbourhood they have and we do not: `interchange`** — swap two items at arbitrary
positions. Exact gain of swapping `u` at position `p` with `v` at position `q > p` is
`w_uv − w_vu` plus, over the nodes strictly between them, the flips of edges incident to `u` and to
`v` — computable in `O(d(u)+d(v))`. But the neighbourhood has `O(n²)` pairs, so it is only usable
with a **candidate list**. The natural candidate list is the set of backward edges — which is
precisely Vahidi's Alg. 2 fallback (1) (`notes-vahidi-flywire.md` §3.2). So interchange is not an
independent idea here; it is subsumed by **H45**.

L1 also reports that **the number of global optima grows exponentially with sparsity**. Our `W` has
density 3.03e-4. That is an argument-shaped observation for **Q02**'s finding (many near-equivalent
orderings, Spearman 0.958 across seeds with 0.0225 pp score spread) — the degeneracy the campaign
measured is what a sparse LOP is *expected* to have. It also supports **M3** (no score tail to
harvest): degeneracy in *orderings* need not produce dispersion in *score*.

---

## 3. PACE 2024 (one-sided crossing minimization) — right scale, wrong cost structure

OCM: given a bipartite graph `(A, B)` with `A` fixed on a line, order `B` to minimise crossings.
The objective is `Σ_{u before v} c_{vu}` for a crossing-count matrix `c` — **structurally a LOP**,
and the heuristic track ran on instances up to ~`10^5` vertices in `B` (L3, Fig. 6) with a **5-minute**
limit (L2). That is our scale and roughly our budget, so this is the closest algorithmic community.

What the solvers actually do (L3, and L5 for the exact/parameterized side):

- **Sifting** (Arcee, L3): take one vertex and place it at the position minimising the objective,
  sequentially over all free vertices. **This is our sift.** Independent arrival at the same primitive.
- **Delta computation** (Arcee, L3): a crossing matrix `M` with `M' = M − Mᵀ` for fast sifting; for
  large graphs a **segment tree** maintaining cumulative counts with range-sum queries, at
  `O(Σ|N(a)|(log|N(a)| + log|B|))`. Our vectorised NumPy prefix-sum kernel achieves the same
  asymptotics without a tree.
- **Force swapping** (Arcee, L3): when improvement plateaus (after a fixed iteration count), swap two
  vertices from the free set and re-sift under constraints. A perturbation-restart. Our **H31**
  (ILS/LNS) already tested the generic form of this and it lost (−0.0008 pp at 1.9× wall).
- **Graph splitting** (Arcee, L3): solve **SCCs of the penalty graph individually and concatenate
  topologically**; plus a neighbourhood-interval partition. Again our stage 4, arrived at
  independently — and again *one-shot*, which we already know is worth +0.00013 pp on our graph.
- **Reduction rules** (Arcee, L3): commit `a` before `b` when `c_ab = 0`, or when neighbourhoods are
  equal, or when bounds determine the pair. These are **pair-fixing** rules: they prune the search
  space of an exact solver. They do not improve a heuristic's incumbent, and we run no exact search,
  so they do not transfer.
- **Branch and bound over prefixes** (OCMu64, L5): fix the order left-to-right, with "strongly fixed
  pairs" and "practically fixed pairs" plus a gluing heuristic. This is an *exact* method. It is
  reported for instances far smaller than the heuristic track and it is the wrong tool at
  `n = 136,648`.

**Net transfer from PACE 2024: essentially nothing new.** Every large-scale technique in it is one we
already have (sifting, SCC split, incremental delta), and everything else is exact-search machinery.
This is a real negative and it is worth recording: it means the campaign's move-class inventory is
*not* behind the graph-drawing community's, and the gap to 84.6147% is not going to be closed by
importing a standard trick.

**The one thing worth stealing is a diagnosis, not a move:** OCM instances are ~10^5 free vertices
with a *dense implied* cost matrix and the field still finds sifting + SCC-split adequate; our
instance is 10^5 nodes with an *extremely sparse* cost matrix, where those same two moves demonstrably
leave 0.46 pp on the table. Sparsity is what makes long-range compound moves necessary — which is the
same conclusion **M4** reached from our own data.

---

## 4. Ejection chains / variable-depth (Lin–Kernighan style) — searched, nothing usable

Query 11. The retrievable material is the classical TSP/QAP line: Lin–Kernighan, Stem-and-Cycle
ejection chains, Kernighan–Lin for graph partitioning, and a variable-depth heuristic for QAP
(arXiv:0912.5473). I found **no** paper applying ejection chains or variable-depth search to the LOP
or to FAS at any scale, let alone at `10^5` nodes.

Assessment, stated as an opinion and not as a retrieved result: an ejection chain for our problem
would be a sequence "move `u` to its best slot, which displaces `v`, move `v`, …" with a
tabu-on-reversal rule and a running cumulative gain. It is buildable and it is genuinely a move class
we lack. But it is **inherently sequential** — each link depends on the previous one's applied
positions — and our entire refiner stack gets its speed from vectorising thousands of independent
moves per sweep (`select_disjoint_moves`, the Jacobi sift). A chain of depth `k` costs `k` sequential
`O(d)` steps and yields one move; a segment sweep costs one `O(m)` pass and yields thousands. Without
a batching story an ejection chain is 3–4 orders of magnitude off the campaign's cost curve. I am
**not** proposing it, and I am recording why so the next scout does not re-derive the same dead end.

The interval-repair move (**H45**) is the affordable relative of an ejection chain: it is a
depth-2 compound move (`u` and `v` jointly) with a closed-form gain and a disjointness rule that
makes thousands of them applicable in one batch.
