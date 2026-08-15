# Vahidi 2025 (arXiv:2506.13799) — the algorithm, and the M4 contradiction

**Date of this note:** 2026-08-16
**Author role:** scout (literature only; no compute was run — an unattended job held the machine)
**Question asked:** `killed.json` meta-rule **M4-local-is-empty** says window-bounded local search
recovers nothing on this graph. `CAMPAIGN.md` and `sota.json` say Vahidi 2025 reaches the target
"with cheap greedy + **bounded-span** insertion + SCC, no MIP". Both cannot be straightforwardly
true. Which one is wrong?

**Short answer:** neither, because the premise is wrong. **Vahidi's method contains no bounded-span
insertion.** The words "bounded" and "span" do not occur in the paper. Its main refinement move
searches the *entire* interval between the two endpoints of a backward edge — on this graph that is
routinely tens of thousands of positions, up to 126,650. The campaign's one-line summary of the
paper is a misdescription that has been propagated through `CAMPAIGN.md`, `sota.json`,
`findings.md` #3/#4 and the Phase-6 summary. Separately, the target number **84.6147% is not this
paper's result** — it belongs to a *later, different* Vahidi–Koutis paper that is not publicly
retrievable.

---

## 1. What I read

| # | Source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| S1 | Soroush Vahidi, *Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization* | arXiv:2506.13799v1, cs.DS. Comment field: "This is a preliminary paper" | submitted 2025-06-13 | https://arxiv.org/abs/2506.13799 , HTML https://arxiv.org/html/2506.13799v1 | **yes**, full HTML incl. all 5 algorithm blocks |
| S2 | `SoroushVahidi/minimum-feedback-challenge` — the paper's own code repository | GitHub, 9 commits (code 2025-06-01, README 2026-02-27) | 2025-06-01 | https://github.com/SoroushVahidi/minimum-feedback-challenge | **yes**, all three notebooks read as raw source |
| S3 | FlyWire Codex "Minimum Feedback Challenge" leaderboard | codex.flywire.ai | read 2026-08-16 | https://codex.flywire.ai/app/mfas_challenge?dataset=fafb | **yes** |
| S4 | Soroush Vahidi, Ioannis Koutis, *Interval-Based Refinement Algorithms for Large-Scale Weighted Feedback Arc Set* | SSRN 6221201 | posted ~Feb 2026 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6221201 | **NO — HTTP 403.** Only the indexed abstract text is available |
| S5 | Vahidi & Koutis, *Minimum Weighted Feedback Arc Sets for Ranking from Pairwise Comparisons* | arXiv:2412.16181 | 2024-12 / v2 2025-01 | https://arxiv.org/abs/2412.16181 | title/abstract only; different problem (tournaments/ranking), not read in depth |

The local `2506.13799v1.pdf` in the repo root could **not** be rendered (no poppler on this box); all
S1 content below comes from the arXiv HTML, which is the same v1.

---

## 2. The retrieved algorithm (S1 + S2)

The paper defines five algorithms and states one cascade sentence. **Verbatim (S1, §3):**

> "Our method has a base part that provides an initial order and uses several heuristics to improve
> the initial order. Algorithm 1 provides the initial order, in which the ratio of sum of the weight
> of the forward edges to the sum of the weight of all edges is 0.7524. Then, we use Algorithm 2,
> Algorithm 3 and Algorithm 4 to improve it to 0.8461."

### Algorithm 1 — adaptive greedy ranking (initial order, 0.7524)

Max-heap over `score(u) = (out_w(u)+1)/(in_w(u)+1)`; repeatedly pop the highest-scoring unranked
node, append it to the order, decrement its neighbours' `in_w` / `out_w` and re-push them (lazy
deletion via an `unranked` set). Leftovers are shuffled in. This is Eades-style greedy peeling with a
*smoothed weighted ratio* key instead of `out_w − in_w`.

> **Number to note.** Their greedy scores **75.24%** on this graph. Our greedy-FAS warm-start
> (`H02`) scores **68.91%** (`experiments/outputs/diagnosis.json`, `init_pct` 68.91342773446004) and
> our reproduced `GreedyAbs` scores 72.04%. Their initial order is **+6.3 pp better than ours**.

### Algorithm 2 — "extended strategy": heap-driven, **full-interval**, compound *pair* relocation

This is the load-bearing move and the one the campaign has mis-summarised. Verbatim pseudocode (S1,
Algorithm 2) with the executed implementation (S2, `advanced_backward_refinement (3).ipynb`,
`apply_new_strategy`) in brackets:

```
2: Insert all backward edges (u,v) with pi(u) > pi(v) into max-heap H
3: while H is not empty:
4:   pop the heaviest edge (u,v);  5-7: skip if it is already forward
8:   Let B = { x in V | pi(v) < pi(x) < pi(u) }          <-- the WHOLE interval
10:  for each index r = 0..|B|:
11:    reorder the block to [n_1..n_r, u, v, n_{r+1}..]  <-- u and v placed ADJACENT at a cut
12:    compute the local Delta by prefix sums
17:  if best_delta > 0: apply
20:  else: try three single-node fallbacks — swap(u,v) / push v forward / pull u backward
29:  push newly created backward edges incident to u or v back into H
```

Verbatim from the surrounding prose (S1, §3.2): *"Let the nodes ranked strictly between v and u be
n1, n2, …, nt, so the local block is: [v, n1, n2, …, nt, u]"* and *"Only the edges with both
endpoints inside this block may change direction"* (their statement of the contiguous-block lemma —
the same lemma our `segment.py` docstring proves).

Exact gain, from the executed code (this is the closed form; the paper's Δ formula is a
special-cased version of it):

```python
delta = (w_uv - w_vu
         + ingoing_v[i+1] - outgoing_v[i+1]      # v moves right from the far-left end to the cut
         - ingoing_u[j]   + outgoing_u[j])       # u moves left  from the far-right end to the cut
```

with prefix sums over v's in-interval neighbours (left of the cut) and suffix sums over u's
in-interval neighbours (right of the cut), and `j = bisect_right(u_scores, nv_score)` tying the two
cuts to one common position. Cut candidates are enumerated **only at v's neighbour positions**
(and skipped when `j == len(u_scores)`) — a self-imposed restriction, not a requirement of the
math. After the move every node other than `u`, `v` shifts by at most one position (the code asserts
this).

**Key structural facts, from the source, not inference:**

- `B` is unrestricted: `between = [node for node, rank in scores.items() if idx_v < rank < idx_u]`.
  There is no window, no cap, no radius. The paper never uses the words "bounded" or "span".
- The heap is ordered by **edge weight**, and on this graph heavy backward edges are long-range: the
  paper's own Table (S1, §4.2) reports their solution's back-edge length **mean 20,536, max
  126,650** (BA-CD: mean 20,450, max 136,410). So the typical `|B|` this move searches is ~20,000
  positions, and the largest is ~126,650.
- It is a **compound two-node move**: `u` and `v` are pulled out of *opposite ends* of the interval
  and re-inserted **adjacent to each other** at one common cut. It is not single-node insertion.
- Cost per popped edge in their implementation is `O(n)` (the `between` list comprehension scans the
  whole score dict) plus `O(|B| log|B|)` for a sort, plus `O(m)` for a full rescore and a 136k-row
  CSV round-trip on every accepted move.

### Algorithm 3 — SCC-condensation inside contiguous rank blocks of the giant SCC

Take the largest SCC, list its nodes in current rank order, cut into consecutive blocks of size `s`.
Per block: induce the subgraph, recompute its SCCs, topologically sort the sub-condensation, permute
sub-SCCs of **size ≤ 9** exhaustively, keep larger ones in rank order, accept the block layout only
if total forward weight improves. **The paper gives no numeric value for `s`.**

> This is essentially our `scc_recursive.py`, one level deep, with an exhaustive tail for tiny SCCs
> and without the recursive `split_frac` re-cutting.

### Algorithm 4 — multilevel/"flat" partition block permutation

Partition the rank interval `[s,e]` into `x^ℓ` contiguous equal groups; slide a window of `x`
consecutive groups; evaluate all `x!` permutations of those groups by the group-to-group weight
matrix `W`; keep the argmax if it differs. Exact by the contiguous-block lemma (permuting the
groups preserves the union of positions).

**Executed parameters (S2, `scc_sort_challenge.ipynb`, only values published anywhere):**

```python
x = 6
max_level = min(int(math.log(end - start + 1, x)), int(math.log(900000, x)))   # = 6 on this graph
parts = partition_flat(scores, level=max_level, x=x, start=0, end=len(scores)-1)
```

so `6^6 = 46,656` groups of ≈2.93 nodes, and each window covers 6 groups ≈ **18 positions**. The
paper's Algorithm 4 takes `ℓ` as an input, and an unused `partition_recursive` sits next to it, so
lower levels (longer-range) were presumably also run — but **no level schedule is published**.

### Algorithm 5 — global SCC-topological ranking

Whole-graph condensation, topological sort, exhaustive permutation of SCCs of size ≤ 9, input rank
order inside big ones. On this graph the giant SCC holds 126,840 / 136,648 nodes, so this only
places the ~9,800 non-giant nodes; **we already measured this exact move as worth +0.00013 pp**
(`dr_tmp/FINDINGS_underrelaxation.md` E2, quoted in `scc_recursive.py`). Consistent with the paper's
own back-edge-length table: their max back-edge length (126,650) is just under the giant-SCC size,
BA-CD's (136,410) is nearly the whole line — the signature of Algorithm 5 having been applied last.

### Reported result and environment (S1 §4.2, verbatim numbers)

| | Vahidi 2025 | BA-CD (Bader et al., Rocket+Crane) |
|---|---|---|
| forward weight | **35,462,925** (≈0.8461) | 35,459,266 (≈0.8460) |
| backward edges | 1,156,812 | 1,157,173 |
| backward weight | 6,449,216 | 6,452,875 |
| back-edge length mean / max | 20,536.34 / 126,650 | 20,449.85 / 136,410 |

> *"Our method improves the total forward edge weight by 3,659 units over BA-CD, corresponding to an
> increase of approximately 0.0087% in the total edge weight."*

Hardware: *"Google Colab Pro+ with the High-RAM CPU runtime … up to 32 GB of RAM and
high-performance Intel Xeon CPUs"*, Python + NumPy/Pandas/NetworkX. **No runtime, no iteration
count, no per-algorithm intermediate score is reported anywhere in the paper.**

---

## 3. The target number 84.6147% does NOT come from this paper

This matters and should be corrected in `sota.json` and `CAMPAIGN.md`.

- `sota.json` records `data/best_solution` = **35,463,823 / 41,912,141 = 84.61468%**, attributed to
  "Vahidi 2025 (arXiv:2506.13799)".
- arXiv:2506.13799 reports **35,462,925 = 84.61254%** (the repo's own checker notebook prints
  `Initial forward edge ratio: 0.846125` on the file it ships, `35462925-Soroush-Ioannis.csv`).
- The FlyWire leaderboard (S3) reads:

  | date | team | forward weight |
  |---|---|---|
  | **2026-01-09** | **Soroush Vahidi, Ioannis Koutis** | **35,463,823** |
  | 2025-06-02 | Soroush Vahidi, Ioannis Koutis | 35,462,925 |
  | 2024-11-16 | Bader, Sriram, Chinthalapudi, Du | 35,459,266 |

So our reference solution is the **2026-01-09** entry — 898 units (0.0021 pp) above the arXiv
paper — and it corresponds to the **later** paper S4, *Interval-Based Refinement Algorithms for
Large-Scale Weighted Feedback Arc Set* (SSRN, ~Feb 2026), which I **could not retrieve** (403).
From the indexed abstract only, S4's method is: *a dynamic program that selects large sets of
non-overlapping backward-edge intervals, refined in parallel by a two-level local search*, plus
*contiguous rank blocks within the giant SCC*, with *a lightweight controller alternating between
the two*. That is Algorithm 2 made batch-parallel by choosing a maximum-weight set of *disjoint*
intervals via DP — i.e. exactly the disjointness trick our `segment.py` already uses, applied to
their pair move. **I could not read the paper and will not reconstruct it further.**

---

## 4. Resolution of the contradiction

### 4.1 The stated contradiction dissolves

M4 is about **bounded rank-window single-node** local search. Vahidi's dominant move is
**unbounded-interval compound pair** relocation, driven by the heaviest backward edges — which on
this graph are precisely the long-range ones. M4's own directive is *"any local-move proposal must
be global-range or structurally decomposed (SCC), not window-bounded"*. Vahidi's Algorithm 2 is
global-range; Algorithm 3 is SCC-structural. **Vahidi satisfies M4's directive rather than
violating it.** There is no conflict to resolve at the level of the rule's directive.

Which of the four candidate explanations the parent offered is right: **none of the four fully; the
correct one is "the paper contains no bounded-span insertion at all".** Sub-answers:

- *Different basin (greedy vs Rocket plateau)?* Real but secondary. Their start is greedy at 75.24%;
  M4's measurement is on H02's Rocket order at 82.93%. Their move class would still be long-range
  from any start.
- *Span measured in another space?* No. `between` is literally a rank interval on the raw line.
- *SCC doing the heavy lifting?* No — we measured the top-level condensation at +0.00013 pp, and the
  paper's Algorithm 3 is block-local. Algorithm 2 is the heavy lifter.
- *Spans larger than the tested W?* **Yes, this is the closest true one** — but "larger" understates
  it: the spans are unbounded, mean ≈20,000, max 126,650, i.e. 25× to 4,000× the largest W (5,000) in
  `localsearch_sizing.json`.
- The one genuinely bounded piece — Algorithm 4 at the published `x=6, ℓ=6` — has a window of **≈18
  positions**. M4's Measure 1 says the entire feedback pool within W=20 is **0.0050 pp**. So M4 is
  *right* about that piece: at its executed setting Algorithm 4 can contribute at most ~0.005 pp.

### 4.2 But M4's *evidence* is over-stated and must be narrowed

Reading `experiments/size_localsearch.py`: M4's headline numbers (W=100 −0.010 pp, W=1000 −0.170,
W=5000 −0.405) come from `measure2_gap_flip_distance.cumulative_within_window.net_pp`, defined as

```
gain = edges feedback in H02's order AND feedforward in the reference
lose = edges feedforward in H02's order AND feedback in the reference
net(W) = sum(w[gain & |rank distance in H02| <= W]) - sum(w[lose & ...])
```

That is **"how much of the *reference's own* rearrangement is short-range"**, not "what a local
search can find". A hill-climb never pays the `lose` term: it accepts only strictly-positive exact
gains and would simply decline the reference's short-range concessions. Using `net(W) ≤ 0` as a
ceiling on local search conflates *copying the reference's decisions inside a window* with
*optimising inside a window*. The genuinely leakage-safe ceiling in the same file is **Measure 1**,
and it says the opposite of "empty":

| W | feedback weight with span ≤ W (`pool_pp`) | vs the 0.46 pp gap H42 → reference |
|---|---|---|
| 100 | 0.027 pp | 6% of the gap — empty, agreed |
| 1,000 | 0.414 pp | 90% of the gap |
| 5,000 | **2.322 pp** | **5× the gap** |

So there is no measurement in the repo showing that a *window-bounded optimiser* at W≈1,000–5,000
recovers nothing. What was measured, and is solid, is: (i) the reference's advantage over H02 is
overwhelmingly long-range (p50 gain distance 22,580); (ii) H30's W=10 sift arm ≈ 0; (iii) at W ≤ 100
the reachable pool is genuinely negligible (0.027 pp).

**Proposed amendment to M4 (narrowing, not repeal):**

> **M4-local-is-empty (narrowed 2026-08-16).** Rank-window neighbourhoods with **W ≤ 100** are
> empty: the entire feedback pool inside W=100 is 0.027 pp (`localsearch_sizing.json` Measure 1) and
> H30's W=10 arm recovers ~0. Beyond that, the evidence is about **provenance of the gap, not about
> local search**: `net_pp(W) ≤ 0` for W ≥ 100 measures how much of the *reference's* rearrangement is
> short-range and does **not** bound a hill-climb, which never pays the `lose` term. The reachable
> feedback pool is 0.414 pp at W=1,000 and 2.322 pp at W=5,000 — larger than the remaining gap.
> Any local-move proposal must still be global-range or structurally decomposed, and a proposal
> whose *only* content is a wider single-node window must first show it beats the full-range sift,
> which is already unbounded. **Prior art check: Vahidi 2025 contains no bounded-span insertion; do
> not cite it as evidence for or against this rule.**

### 4.3 What is genuinely unresolved: did 0.7524 → 0.8461 actually happen from scratch?

`CAMPAIGN.md` treats "the target is reachable by cheap combinatorial means" as established fact
("This is not a fishing expedition"). The evidence for that is **one sentence in a paper whose own
comment field says "This is a preliminary paper"**. Against it:

1. The paper reports **no intermediate score** after Algorithm 2, 3 or 4, **no runtime**, **no
   iteration counts** — the 0.7524 and 0.8461 endpoints only.
2. The repository publishes **no implementation of Algorithm 1** at all. The only executable
   refinement code loads an initial ranking that is **already at 84.61%**
   (`initial_ranking_path = ".../35461047-Soroush-Ioannis-advancedimprove.csv"`; the numeric prefix
   is a forward weight, 35,461,047, i.e. already 1,781 units *above* BA-CD) and writes back to the
   same file.
3. BA-CD's ordering was downloadable from the same leaderboard; the paper calls it *"the strongest
   publicly available solution at the time of writing"*. The paper never states that it was, or was
   not, used as a starting order.
4. *(My inference, flagged as such.)* The published implementation's per-accepted-move cost is
   `O(m)` rescore + `pd.read_csv` + `df.iterrows()` over 136,648 rows + `to_csv` ≈ 10–25 s in
   CPython. Climbing 0.7524 → 0.8461 requires Σgain ≈ 3.93 M weight units; at any plausible mean
   per-move gain that is 10^4–10^5 accepted moves, i.e. days of pure-Python bookkeeping on top of a
   `O(n)`-per-pop scan over ≥1.4 M heap entries. Climbing the last +3,659 units needs of order 10^2
   accepted moves. **The published artefact is sized for the small increment, not the large climb.**

Counter-evidence *for* the paper's claim, from our own repo: `findings.md` (Phase-6 summary) records
the `dr_tmp` result **"sift-on-greedy ≈ sift-on-Rocket"** — our full-range exact-gain single-node
sift climbs from greedy-FAS 68.91% to ≈83.8%, i.e. **+15 pp from a greedy start**. So a
75.24 → 84.61 climb by an unbounded-interval exact-gain refiner is entirely credible *in kind*; the
only doubtful part is the last ~0.8 pp, which is exactly where our own pipeline is stuck.

**Verdict: unresolved.** What would resolve it: (a) retrieving S4 (SSRN 6221201) or a v2 of S1 that
reports per-stage scores; (b) e-mailing the authors for the Algorithm-1 code and a stage-by-stage
trace; (c) *cheapest and fully under our control* — implementing Algorithm 2's pair move ourselves
and running it from our own greedy order, which answers the scientific question regardless of what
they did. **`CAMPAIGN.md` should be edited to say the target is "reported reachable without MIP,
provenance not independently verified", not "known to be reachable".**

---

## 5. What we would have to change (concrete, per module)

Current champion pipeline (`sota.json`, H42, connectome 84.1541%):
greedy-FAS → Rocket → under-relaxed full-range single-node sift → { recursive SCC block refine ↔
2-sweep sift } × cycles (+ H41's segment sweep in the H41 variant). Gap to the reference: **0.4606
pp**.

### 5.1 The move class we do not have: exact-gain **joint pair** relocation (their Algorithm 2)

*What is missing.* `insertion.py`/`underrelax.py` compute, for every node independently, the exact
piecewise-constant profile `total_u(g)` and its argmax, then move every node to its own argmax
(Jacobi). Vahidi's move maximises the **joint** objective
`total_u(c) + total_v(c) + w_uv` over a *common* cut `c`, with `u` placed immediately before `v`.
If neither `u` nor `v` profits from moving alone but both profit from meeting, our sift is at a fixed
point and blind — and the heaviest backward edges are exactly the pairs for which this is likely.
`segment.py` cannot express it either: it relocates *contiguous* runs, and `u`, `v` here are ~20,000
positions apart.

*How to build it, affordably.* The kernel is already 90% written. `jacobi_best_gaps` builds, per
node, an event list `(node, breakpoint b, delta)` whose segmented prefix sum is `total_u(·)`. For a
candidate pair `(u,v)` the joint profile is the *merge* of `u`'s and `v`'s event lists restricted to
`b ∈ (rank[v], rank[u])`, so the joint argmax costs `O(deg(u)+deg(v))` after one sort. Then:

- select the top-K heaviest backward edges (K ≈ 20k–100k) — `O(m)`;
- compute all joint gains in one vectorised segmented pass — `O(Σ deg)`;
- keep strictly-positive gains, sort by gain, **greedily accept moves whose intervals `[rank[v],
  rank[u]]` are pairwise disjoint** (identical to `select_disjoint_moves` in `segment.py`;
  disjoint intervals compose exactly by the same contiguous-block lemma), then apply all of them in
  one `argsort` rebuild — `O(n log n)`;
- repeat for a few sweeps, monotone by construction, best-by-oracle as usual.

*Cost at connectome scale.* One sweep ≈ one `jacobi_best_gaps`-sized pass (our sift already does 40
of these inside the budget) plus an `O(n log n)` rebuild. **Order of ~10–30 s per sweep — the same
register as the existing sift.** This is nothing like the days their CPython version needs; the
whole difference is vectorisation plus batching disjoint moves instead of one-at-a-time rescoring.

*New module:* `src/mfas/refine/pairmove.py` + `src/mfas/experiments/H4x.py`.

### 5.2 The second move class: **coarse multilevel block permutation** (their Algorithm 4 at low ℓ)

`segment.py` swaps **two** adjacent blocks with `L, d ≤ 1024` (max window 2,048 ≈ 1.5% of the line).
Algorithm 4 permutes **x = 6** contiguous groups at once, and at low levels those groups are huge
(ℓ=2 → 36 groups of ~3,800; ℓ=1 → 6 groups of ~22,775). Generalising `segment.py` from `k=2` to
`k≥3` at *coarse* levels is a strictly larger, still-exact, still-monotone move class:

- partition the line into `k` contiguous blocks; one `np.bincount` over all edges gives the `k×k`
  block weight matrix `W` — **`O(m)`, ~1 s on connectome**;
- the best permutation of `k` blocks is itself an MFAS instance on `k` nodes: exhaustive for `k ≤ 9`
  (`k! · k²`), or our own sift for larger `k`;
- apply if the exact gain `Σ_{i<j} W[σ_i, σ_j] − Σ_{i<j} W[i,j] > 0`; exact by the contiguous-block
  lemma already proven in `segment.py`'s docstring, case (c);
- sweep levels `ℓ = 1..7` and offset the partition boundaries between sweeps (the same trick
  `DEFAULT_SPLIT_FRACS` uses in `scc_recursive.py`).

*Cost:* `O(m)` per (level, offset) → a full ladder is seconds. **This is the cheapest untested idea
in this note and should be prototyped first**, because a single `O(m)` bincount at ℓ=1..3 answers
"is there any coarse-scale misplacement left in our order at all?" — and if the answer is no at every
level, that is itself a hard, useful negative that closes the whole block-permutation family.

*Where:* extend `src/mfas/refine/segment.py` (new function, do not change the existing kernel) or a
new `src/mfas/refine/blockperm.py`.

### 5.3 Things in the paper we already have, or that are known-worthless here

| Their piece | Ours | verdict |
|---|---|---|
| Alg 3 (SCC inside contiguous rank blocks) | `scc_recursive.py` — same lemma, *recursive*, with `split_frac` re-cutting | ours is strictly stronger, except we never exhaustively permute SCCs of size ≤ 9. Cheap add-on, likely tiny. |
| Alg 5 (global SCC topological order) | measured: **+0.00013 pp** | dead. Do not spend a cycle. |
| Alg 4 at published `x=6, ℓ=6` (18-position window) | `segment.py` covers this range | dead by M4 Measure 1 (≤0.005 pp). Only the *coarse* levels are new. |
| Alg 2's three greedy fallbacks (swap / push v / pull u) | strictly dominated by our full-range exact-gain sift | dead. |
| Alg 1 (ratio greedy, 75.24% vs our 68.91%) | `H02` greedy-FAS | **+6.3 pp better init.** But M6 says init→plateau is flat through Rocket, and M7 says the sift does not care about the seed ("sift-on-greedy ≈ sift-on-Rocket"). Low expected value **as a warm-start**; possibly worth 20 lines as a free experiment, not a cycle. |

### 5.4 Conflict check against `killed.json`

| idea | entries touched | why it is different / which revival condition |
|---|---|---|
| §5.1 joint pair relocation | **M4**; H22 (bounded-window sifting); H31 (LNS) | Not window-bounded: the search interval is the backward edge's own span (mean ~20,000). Not single-node: it is a compound 2-node move with a joint objective our Jacobi sift provably cannot see. Not LNS: the destroy/rebuild is exact-gain and structure-selected (heaviest backward edge), which is *precisely* H31's stated revival condition — "the destroy operator is STRUCTURE-AWARE … and the rebuild is exact-gain". |
| §5.2 coarse block permutation | **M4**; H22 | Range is `n/k` (22,775 at k=6), i.e. global, not a rank window. M4's directive is satisfied. Note that the *fine* levels are dead by M4 Measure 1 — only ℓ ≤ 3 is worth running, and the proposal must state that. |
| §5.3 Alg-1 ratio greedy | **M6-init-is-flat**, **M7-expensive-warmstarts-lose** | M7 does not apply (this init is *cheaper* than ours, not more expensive). M6 does apply and predicts ≈0 through Rocket. Only admissible as a 20-minute free probe, never as a cycle. |
| anything continuous | M1, M2, M34 | Nothing in this paper is continuous. Vahidi uses **no gradient phase at all** — worth recording: the best-known solution on this graph was produced without any continuous relaxation, which corroborates diagnosis Q01 rather than contradicting it. |

---

## 6. What I could not find (the honest negative)

- **S4 (SSRN 6221201) is not retrievable.** SSRN returns 403 to every route I tried. There is no
  arXiv version (the author's arXiv listing has no such entry as of 2026-08-16). Since S4 — not
  S1 — is the paper behind our 84.6147% reference, **the exact algorithm that produced our target
  number is unknown to this campaign.** All I have is the indexed abstract: DP selection of
  non-overlapping backward-edge intervals + parallel two-level local search + contiguous rank blocks
  in the giant SCC + an alternating controller. I have deliberately not reconstructed more than that.
- **No runtime, no per-stage scores, no parameter values** anywhere in S1 for Algorithms 2–4 beyond
  the `≤9` permutation cap and `x = 6, ℓ = 6` in one notebook cell. Any claim of the form "Vahidi got
  X pp from stage Y" is not supportable from the literature.
- **No independent replication of S1 exists** that I could find. 0 stars / 0 forks on the repo;
  no citing work located.
- **Nothing new on the continuous axis.** I looked; the current best-known solution on this exact
  graph uses no relaxation, no gradient, no learning. Q01's picture stands unchallenged.
- **I did not find any published bounded-window local-search result on this graph**, from Vahidi or
  anyone else. M4 remains untested against the literature, in either direction, above W=100.

---

## 7. Proposed queue items (not yet appended — `queue.json` is being written by the running driver)

I deliberately did not touch `autoresearch/queue.json` while `dr_tmp/night_run.sh` is live. The two
survivors, in priority order:

1. **Coarse multilevel block permutation** (§5.2). Prototype first: it is one `O(m)` bincount per
   level. *Hypothesis:* there exists a level `ℓ ≤ 3` and offset at which permuting the `k = 6`
   contiguous blocks of the champion's order yields a strictly positive exact gain.
   *Kill condition:* max exact gain over `ℓ ∈ {1..4}` × 3 offsets is ≤ 0 on connectome **and**
   microns → the whole block-permutation family above `segment.py`'s 2,048-position ladder is closed.
   *Cost:* minutes. *Expected:* 0.00 to +0.15 pp; most likely 0, and a 0 is worth having.
2. **Exact-gain joint pair relocation** (§5.1). *Hypothesis:* a sweep of joint 2-node relocations on
   the top-K heaviest backward edges, applied over disjoint intervals, gains > `screen_delta_pp` over
   the champion on connectome at ≤ 1.3× wall. *Kill condition:* < `screen_delta_pp` on connectome
   **or** the joint argmax coincides with the two independent argmaxes for > 99% of candidate pairs
   (which would prove the sift already covers the move). *Cost:* ~1 day to implement, ~1–3 h to
   screen. *Expected:* +0.05 to +0.3 pp.

Item 2's second kill condition is the cheap falsifier and should be run as the prototype gate: it is
a pure diagnostic over one `jacobi_best_gaps` call and answers "is this move class actually distinct
from our sift?" before any variant is built.
