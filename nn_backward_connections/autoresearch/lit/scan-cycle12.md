# Literature scan L02 — cycle 12 (FORCED DIVERGENT MODE)

**Date:** 2026-08-25
**Author role:** scout (literature only)
**Compute run by this scan: NONE.** Every internal number below is quoted from a file already in
the repo; every external number carries a retrieved source or is explicitly marked unretrieved.
Same discipline as scan L01 (`autoresearch/lit/README.md`).

**Why this scan exists.** `state.json` has `cycles_since_score_move = 3`; H56 and H57 both died at
the prototype rung and together closed the multi-start family (M9, M10). `CAMPAIGN.md § Escalation`
requires attacking a **different level** — move class, decomposition, or formulation — not another
variant of the current design. This file extends L01 (2026-08-16); it does not repeat it.

**Files written by this scan:** this one only. `queue.json` was NOT touched; queue-ready item
bodies are in § 7 for the cycle to paste.

---

## 1. Executive summary — the three things that are actually new

1. **The target's provenance is worse than the campaign thinks, and the arithmetic is now
   explicit (§ 3).** The top two leaderboard entries are **+0.0087 pp and +0.0109 pp** above the
   Rocket + **Crane MIP** result. The 84.6147% mission target is, on the public record, most
   plausibly *a 20-day Gurobi MIP solution plus ~0.011 pp of interval refinement*. Two other teams
   (Hashorva 84.5875%, Zheng–Tang–Okubo 84.4019%) beat our champion with unpublished, non-MIP-attested
   methods — so *some* of the gap has independent non-MIP precedent, but **the last ~0.03 pp does
   not**. This should be a human checkpoint item, not a silent assumption.
2. **A formulation-level move the campaign has never had: FAS *minimality* by arc re-insertion
   (§ 4.1 → H59).** Our whole stack reasons about *positions*. The FAS literature reasons about the
   *removed arc set*, and asks a question our position-space moves cannot ask: *is there a backward
   edge `(u,v)` such that `v` cannot reach `u` through currently-forward edges?* If so, a topological
   sort of `F ∪ {(u,v)}` scores **at least** current + `w_uv`, and no bounded move class can find it
   because it may require re-sorting the whole line. This is a *certificate* either way, and the
   effect-size bar is startlingly low: 0.012 pp is only **5,029 weight units**, and single backward
   edges here carry up to 2,405.
3. **A decomposition-level defect with a theorem behind it (§ 4.2 → H60).** `scc_recursive.py`
   condenses the **raw digraph** (`np.ones` on every edge, `scc_recursive.py:222`). Every reciprocal
   pair `u⇄v` is therefore a hard 2-cycle and can never be separated — even when `w_uv = 100,
   w_vu = 1`. The rank-aggregation literature calls the right object the **finest-Condorcet
   partition**: condense the *net* digraph `D⁺ = {(u,v) : w_uv > w_vu}`, weight `d_uv = w_uv − w_vu`.
   MFAS on `W` and on `D⁺` are the same problem up to the additive constant `Σ_pairs min(w_uv,w_vu)`,
   and the Extended Condorcet Criterion says **every optimum respects** the `D⁺` condensation. `D⁺`'s
   SCCs are a strict refinement of `W`'s, so a single pass with `D⁺` **dominates** the current pass,
   exactly, and costs *fewer* arcs (which matters because microns has no wall-clock headroom).

---

## 2. Search log

### 2.1 Queries actually issued (WebSearch), 2026-08-25

1. `Vahidi Koutis Interval-Based Refinement Algorithms Large-Scale Weighted Feedback Arc Set`
2. `arXiv 2026 feedback arc set heuristic million edges linear ordering refinement`
3. `"non-overlapping backward-edge intervals" "two-level local search" feedback arc set SSRN abstract`
4. `FlyWire MFAS challenge leaderboard 2026 forward weight best solution`
5. `RASstar feedback arc set graph reduction divide-and-conquer FlyWire 33,037,112`
6. `dynasearch dynamic programming exponential neighbourhood independent moves permutation local search`
7. `Charon Hudry noising method linear ordering problem feedback arc set tournaments`
8. `large neighborhood search destroy repair permutation problems 2025 exact repair dynamic programming ordering`
9. `Kemeny rank aggregation large scale heuristic 2025 local search millions items`
10. `"Approximate Condorcet Partitioning" Escobedo rank aggregation decomposition large-scale`
11. `DAG structure learning topological ordering local search 2025 order-based search permutation neighborhood scalable`
12. `PACE challenge 2025 2026 problem directed feedback arc set heuristic track`
13. `multilevel coarsening directed graph ordering 2026 feedback arc set refinement uncoarsening million nodes`
14. `arXiv cs.DS 2026 "feedback arc set" heuristic scalable weighted new algorithm`
15. `simulated annealing GPU parallel permutation local search 100000 nodes linear ordering problem 2025`
16. `Gert-Jan Both Srinivas Turaga FlyWire minimum feedback arc set challenge solution method`
17. `Dritan Hashorva feedback arc set FlyWire codex challenge algorithm`
18. `Soroush Vahidi github feedback arc set 2026 interval refinement code repository`
19. `"minimal and stable feedback arc sets" efficient heuristics Journal Combinatorial Optimization 2024 minimality`
20. `Simpson Srinivasan Thomas "feedback arc set at web-scale" KwikSort SCC algorithm billion edges minimality`

### 2.2 Pages actually opened and read (WebFetch)

| source | result |
|---|---|
| `codex.flywire.ai/app/mfas_challenge?dataset=fafb` | **read** — full 13-row leaderboard, unchanged since 2026-01-09 |
| `arxiv.org/a/vahidi_s_1` (author listing) | **read** — no 2026 arXiv entry exists |
| `arxiv.org/html/2412.16181v2` (Vahidi & Koutis, MWFAS for ranking) | **read** — algorithms extracted |
| `ar5iv.labs.arxiv.org/html/2204.02902` (Grüttemeier et al., parameterized local search on topological orderings) | **read** — neighbourhood definitions + complexities extracted |
| `arxiv.org/html/2506.15097` (Kemeny space reduction by optimized majority rules) | **read** — XCC / finest-Condorcet / MOT / α-MOT extracted verbatim-ish |
| `github.com/SoroushVahidi/minimum-feedback-challenge` | **read** — same 6 files as L01 saw; **no new code**, no interval-refinement implementation |

### 2.3 Retrieval failures — recorded, not worked around

| source | what happened |
|---|---|
| **SSRN 6221201** (Vahidi & Koutis 2026, the paper behind our exact target) | **403 Forbidden again**, on both `papers.ssrn.com/sol3/papers.cfm?abstract_id=6221201` and `ssrn.com/abstract=6221201`. Third documented failure. The Semantic Scholar API returned **429**. ResearchGate profile **403**. **The full text has still never been read by this campaign.** Everything attributed to it below is from indexed abstract snippets, consistent across three independent queries. |
| local `2506.13799v1.pdf` (repo root) | **still unreadable** — `pdftoppm` (poppler) is not installed on this box. L01's coverage gap #1 is therefore still open. |
| local `s13278-025-01491-2 (2).pdf` (Rocket-Crane) | same reason. **The Crane phase has never actually been read by this campaign**; "Gurobi MIP + ~20 days" traces to `CLAUDE.md` prose, not to a quotation. |
| `davidbader.net/publication/2025-bebtscd/2025-bebtscd.pdf` (free Rocket-Crane PDF) | fetched (2 MB) but **the fetcher could not extract text** from the PDF stream. |
| `link.springer.com/article/10.1007/s13278-025-01491-2` and `.../s10878-024-01209-8` | **303 → IdP login**. Abstract-only. |
| `sciencedirect.com/.../S030505482500276X`, `.../S030505482300028X` | **403**. Abstract-only. |
| `vldb.org/pvldb/vol10/p133-simpson.pdf`, `d-nb.info/1354410769/34` | fetched, **text not extractable** (same PDF-decoding failure). |

> **Systemic note for the operator.** *Every* PDF this session — remote or local — failed to yield
> text. Installing poppler (or any `pdftotext`) on this machine would unblock at least four
> primary sources that two consecutive scans have now had to work around. That is the single
> cheapest research-infrastructure fix available.

---

## 3. Recalibration of the mission target (this is the most load-bearing finding)

The leaderboard, read 2026-08-25 (unchanged since L01 — **no new entry in 7 months**):

| date | team | forward weight | % (computed here from 41,912,141) | vs our champion 84.15409% |
|---|---|---|---|---|
| 2026-01-09 | Vahidi, Koutis | 35,463,823 | **84.61468** | +0.46059 |
| 2024-11-16 | Bader, Sriram, Chinthalapudi, Du (**Rocket + Crane**) | 35,459,266 | **84.60381** | +0.44972 |
| 2024-10-28 | Hashorva | 35,452,425 | 84.58748 | +0.43339 |
| 2024-10-28 | Ellis-Joyce, Both, Turaga | 35,436,406 | 84.54925 | +0.39516 |
| 2024-10-08 | Ellis-Joyce, Both, Turaga | 35,435,948 | 84.54816 | +0.39407 |
| 2024-10-08 | Zheng, Tang, Okubo | 35,374,656 | 84.40190 | +0.24781 |
| 2024-10-08 | Hashorva | 35,364,447 | 84.37753 | +0.22344 |
| — | **our champion H42** | 35,270,783 | **84.15409** | — |
| 2024-10-08 | Bader et al. | 35,231,406 | 84.06012 | −0.09397 |

Plus the published paper number: **Vahidi 2025 (arXiv:2506.13799) = 35,462,925 = 84.61254%**.

**The arithmetic nobody has written down.**

```
Vahidi 2025   − Rocket+Crane = 35,462,925 − 35,459,266 =  3,659 units = 0.00873 pp
Vahidi/Koutis − Rocket+Crane = 35,463,823 − 35,459,266 =  4,557 units = 0.01087 pp
```

Both are within **one screen threshold** (`screen_delta_pp = 0.012`) of the MIP result. The
`arxiv.org/html/2412.16181v2`-adjacent snippet for SSRN 6221201 says the method *"improves on the
strongest prior ordering produced by Rocket–Crane"* — a phrase that is at best ambiguous between
"beats its score" and "refines its ordering", and the 2025 repo's only executable notebook already
loads a ranking at 35,461,047, i.e. **1,781 units above Rocket+Crane** (L01 established this).

**What follows, stated carefully:**

- The claim "84.6147% is reachable by cheap combinatorial means with no MIP" rests on **exactly one
  unverified sentence** — Vahidi 2025's "0.7524 → 0.8461" — and the campaign's own H50 measurement
  contradicts its practicality (their Alg. 2 from a ratio-greedy start reached 75.29% after 40
  sweeps / 2,150 s and needed an estimated ~16 h more; `proto_H50_connectome.json`).
- What *is* independently attested without any MIP: **84.5875%** (Hashorva) and **84.4019%**
  (Zheng–Tang–Okubo), both October 2024, methods unpublished, +0.433 and +0.248 pp over us.
- So the honest reading of the remaining 0.4606 pp is: **~0.25–0.43 pp of it has independent
  non-MIP precedent; the last ~0.03–0.21 pp does not.**

**Recommendation (operator decision, not an agent's):** `CAMPAIGN.md`'s Phase-1 exit criterion is
pinned to a number whose most likely provenance is a 20-day Gurobi MIP. An intermediate,
independently-attested milestone — **84.4019% (+0.248 pp), then 84.5875% (+0.433 pp)** — would be
better calibrated and would stop the campaign treating the last 0.011 pp as if it were cheap.

---

## 4. Sources read — mechanism extracted, and what does NOT transfer

### 4.1 The FAS-*minimality* line (formulation level) — **the best transfer in this scan**

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| A1 | Vahidi & Koutis, *Minimum Weighted Feedback Arc Sets for Ranking from Pairwise Comparisons* | arXiv:2412.16181v2 | 2024-12 / v2 2025-01 | https://arxiv.org/html/2412.16181v2 | **yes** (HTML, algorithms section) |
| A2 | Simpson, Srinivasan, Thomo, *Efficient computation of feedback arc set at web-scale* | PVLDB 10(2), 2016 | 2016 | https://www.vldb.org/pvldb/vol10/p133-simpson.pdf | **no** — PDF text not extractable; snippets only |
| A3 | *Efficient heuristics to compute minimal and stable feedback arc sets* | J. Comb. Optim., 10.1007/s10878-024-01209-8 | 2024 | https://link.springer.com/article/10.1007/s10878-024-01209-8 | **no** — 303/IdP. Abstract + indexed snippets only |
| A4 | *Minimal and stable feedback arc sets and graph centrality measures* | Computers & OR, S030505482500276X | 2025 | https://www.sciencedirect.com/science/article/pii/S030505482500276X | **no** — 403. Title/abstract only |

**Mechanism, in our vocabulary.** A1's Algorithm 1 (read from the HTML) is a local-ratio /
cycle-cancelling heuristic: find cycles by DFS, subtract the minimum cycle weight from every arc on
the cycle, delete zero-weight arcs, and then — the step that matters — **re-add deleted arcs that do
not recreate a cycle**, and only then topologically sort. A3's abstract states the property this
step establishes, and states it precisely: a FAS is **minimal** when *"none of the arcs can be
reintroduced in the graph without disrupting acyclicity"*, and **stable** when for each vertex the
number of eliminated outgoing (resp. incoming) arcs is not larger than the number of remaining
incoming (resp. outgoing) arcs.

Translate to positions. Our order `π` is a topological order of `F` = the set of currently-forward
edges. Take any backward edge `(u,v)`, `rank[u] > rank[v]`. **`F ∪ {(u,v)}` is acyclic iff `v`
cannot reach `u` in `F`.** If it is acyclic, *any* topological order of `F ∪ {(u,v)}` makes every
edge of `F` forward **and** `(u,v)` forward, so it scores **≥ current + w_uv**, exactly and
monotonically. Crucially, a forward path from `v` can only move rightward in rank, so the
reachability test is **confined to the rank interval `(rank[v], rank[u])`** — no global search.

**Why our stack cannot see this.** Every move class we own (single-node insertion, SCC-recursive
block, adjacent-segment, pair relocation) rearranges a *bounded, identified* set of nodes and
computes an exact position-space gain. Reclaiming `(u,v)` may require an arbitrary re-sort of the
whole interval — thousands of nodes, none of which individually profits. And the SCC refiner cannot
find it either, because it condenses the **full** graph `G` (forward *and* backward arcs), in which
`u` and `v` almost always sit in the same SCC; the object that matters here is the condensation of
`F ∪ {e}`, a different graph.

**What does NOT transfer.** A2 and A3 are **cardinality / unweighted** FAS on web and circuit graphs
with a near-acyclic skeleton; their headline contributions (reduction rules, KwikSort variants,
web-scale scaling) are irrelevant to us for the reason L01 already recorded — 92.8% of our nodes are
in one SCC and 15.39% of the total weight stays backward even in the reference. The **"stable"**
half of A3's condition is also already subsumed: it is a per-vertex in/out count test, strictly
weaker than our exact-gain full-range sift. **Only the "minimal" half is new to us**, and it is new
precisely because it is a reachability question, not a gain question.

### 4.2 The Condorcet / majority-tournament decomposition (decomposition level)

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| B1 | *Efficient space reduction techniques by optimized majority rules for the Kemeny aggregation problem and beyond* | arXiv:2506.15097 | 2025-06 | https://arxiv.org/html/2506.15097 | **yes** (HTML, full rules section) |
| B2 | Akbari & Escobedo, *Approximate Condorcet Partitioning: solving large-scale rank aggregation problems* | Computers & OR, S030505482300028X | 2023 | https://www.sciencedirect.com/science/article/abs/pii/S030505482300028X | **no** — 403. Abstract + indexed summary only |
| B3 | Charon & Hudry, *An updated survey on the linear ordering problem for weighted or unweighted tournaments* | Ann. Oper. Res. | 2010 | https://link.springer.com/article/10.1007/s10479-009-0648-7 | **no** — abstract only |

**Mechanism, in our vocabulary.** B1 builds the *cumulative ranking matrix* `δ_xy = (votes for x
before y) − (votes for y before x)` — for us, `d_uv = w_uv − w_vu`. It then states the **Extended
Condorcet Criterion (XCC)**: a partition `C = ⋃ X_i` satisfies XCC when `δ_{x_i x_j} > 0` for all
`x_i ∈ X_i`, `x_j ∈ X_j`, `i < j` — and *every* optimal solution ("median") respects such a
partition, so the problem decomposes into independent subproblems. B2 (abstract-only) defines the
**finest** such partition and gives an algorithm for it; operationally the finest-XCC partition is
the **SCC condensation of the net digraph** `D⁺ = {(u,v) : d_uv > 0}`.

**Why this is a defect in our champion.** `SccRecursiveRefiner._refine` builds
`coo_matrix((np.ones(...), (ls_pos, lt_pos)))` (`src/mfas/refine/scc_recursive.py:222`) — the
**raw, unweighted, both-directions** digraph. Any reciprocal pair is a 2-cycle and is welded into
one SCC forever, however lopsided its weights. Since arcs of `D⁺` are a subset of arcs of `G`,
**`D⁺`'s SCCs strictly refine `G`'s.** And the exactness argument is unchanged: for a pair in
different `D⁺`-SCCs the topological layout takes `max(w_uv, w_vu)`; for a pair inside an SCC nothing
moves; edges leaving a contiguous block never flip. So **one pass of `D⁺`-condensation refinement
dominates one pass of the current one, exactly, from the same starting order** — and it moves the
order into the set that contains every optimum, which is a stronger statement than monotonicity.

**What does NOT transfer.** B1's other rules — **MOT** and **α-MOT** (`δ_xy > min_{α+β=1}
Σ_{z ∈ Z_k(x,y)} max(0, α δ_yz + β δ_zx)`, `O(n²)` per pair) — *fix a pair's relative order in every
optimum*, i.e. they prune an **exact solver's** search space. L01's judgement stands: pair-fixing
does not improve a heuristic's incumbent. Their instances are also tiny (n = 50 synthetic,
n = 102–240 PrefLib) and the `O(n²)`-per-pair cost is nominally hopeless at n = 136,648. There is
one non-obvious residual use, kept in § 7 as a low-priority diagnostic: for us `d_uv = 0` for every
non-adjacent pair, so only the 5.66 M *edges* can be constrained at all, and the interference set of
an edge reduces to its **common neighbourhood** — which makes an approximate MOT sweep a triangle
enumeration rather than an `O(n³)` job. Its value is a *certificate*, not a score.

### 4.3 Exactly-searched compound-move neighbourhoods (move-class level)

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| C1 | Grüttemeier, Komusiewicz, Morawietz, *Efficient Bayesian Network Structure Learning via Parameterized Local Search on Topological Orderings* | AAAI 2021 / arXiv:2204.02902 | 2021 / 2022 | https://ar5iv.labs.arxiv.org/html/2204.02902 | **yes** (ar5iv HTML) |
| C2 | Congram, Potts, van de Velde, *An Iterated Dynasearch Algorithm for the Single-Machine Total Weighted Tardiness Scheduling Problem* | INFORMS J. Computing 14(1):52–67 | 2002 | https://pubsonline.informs.org/doi/abs/10.1287/ijoc.14.1.52.7712 | **no** — paywalled; description from indexed summaries |
| C3 | Ergun & Orlin, *A dynamic programming methodology in very large scale neighborhood search applied to the TSP* | Discrete Optimization, S1572528605000733 | 2006 | https://www.sciencedirect.com/science/article/pii/S1572528605000733 | **no** — abstract only |

**Mechanism (C1, read).** The paper formalises four neighbourhoods on a topological ordering and
searches each **exactly**: *insert distance* and *swap distance* are solvable in `n^{O(r)}·poly`
(enumerate all orderings `r`-close, then DP inside each); *inversion distance* (Kendall-τ ball of
radius `r`) is FPT at `2^{O(√r log r)}·poly` via randomized colour-coding; and they introduce
*inversion-window distance* — partition the order into windows, allow ≤ r inversions **per window** —
with the same FPT bound. The framework requires the objective to decompose per variable; **ours
decomposes per pair, which is stronger.** Their honest experimental note is worth repeating: the
bottleneck was *"the slow implementation of the insert operation"*, not the combinatorics — the same
lesson H51 learned here (naive `O(interval)` rebuild was fine; a Dietz–Sleator structure was
unnecessary).

**Mechanism (C2/C3, not opened).** Dynasearch applies, in one iteration, a **set of mutually
independent (non-overlapping) moves** chosen optimally by a DP — an exponentially large
neighbourhood searched in polynomial time. Independence is exactly our disjointness condition. Two
observations for us: (i) `select_disjoint_moves` (`segment.py:315`) is the **greedy** version of the
dynasearch DP, and H45's gate already measured the optimal interval-scheduling DP at **1.93×** the
greedy packing (`proto_H45_pair_connectome.json`) — that is a free, already-quantified upgrade
wherever batched-disjoint application survives; (ii) H51/H52 then showed that on *this* graph
sequential application beats any disjoint batch by 11.6×, so the dynasearch framing is the *wrong*
one for wide-interval moves here. **Recorded so it is not re-derived: dynasearch is what we already
do, and we already know its limit on this instance.**

**Transfer that is genuinely open.** C1's *insert distance r* is the exact name for the one gap in
our move-class ladder: we have `r = 1` (the sift) and `r = 2` (H45/H52 pair relocation) and nothing
above. For a pairwise-decomposable objective the joint-optimal re-insertion of `k` extracted nodes is
a DP over (subset placed, gap), and because each node's insertion profile is piecewise constant with
`O(deg)` breakpoints, the candidate gap set is `O(Σ_{j∈S} deg(j))` rather than `O(n)` — so the DP is
`O((Σ deg) · 2^k · k)`, ≈ 2·10⁵ operations for `k = 6` at our mean total degree of 83. That is
affordable. See H61 in § 7 — but read its honest self-assessment first; M8 is a serious threat to it.

### 4.4 Acceptance policy (search-control level)

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| D1 | Charon & Hudry, *The noising methods: a generalization of some metaheuristics* | Eur. J. Oper. Res. 135(1):86–101 | 2001 | https://ideas.repec.org/a/eee/ejores/v135y2001i1p86-101.html | **no** — abstract/indexed summary only |
| D2 | Charon & Hudry, *A survey on the linear ordering problem for weighted or unweighted tournaments* | 4OR 5:5–60 | 2007 | https://link.springer.com/article/10.1007/s10288-007-0036-6 | **no** — abstract only |

**Mechanism (from indexed summaries; not opened).** The noising methods perturb the *evaluation* of
a move by decaying noise rather than perturbing the acceptance probability, and generalise both
simulated annealing and threshold accepting. Charon & Hudry developed and benchmarked them on
**exactly our problem** — the linear ordering problem on weighted tournaments — reporting optimal
solutions on all but 6 of 5,790 tournaments *up to 100 vertices*.

**What does NOT transfer: the scale.** 100 vertices. Same wall L01 hit with the LOP benchmark suite
(largest instance n = 300, i.e. 455× smaller than ours). A noising trajectory needs many single
moves; our entire architecture gets its speed from thousands of *simultaneous* exact moves per
`O(m)` pass.

**What might.** A *batched* noising is expressible in our kernel with no architecture change: in a
sift sweep, `jacobi_best_gaps` already produces each node's exact profile; taking the argmax of
`profile + noise` instead of the argmax of `profile`, with the noise amplitude decaying to exactly 0
in the final sweeps (so the last phase is the champion's own monotone converger) and best-by-oracle
tracking as the baseline already does. **I am ranking this LAST and I am explicit about why** — see
H62 in § 7. It consumes RNG, which triples the screen (`seed_class.py` would classify it `rng`),
destroys the bit-determinism that `sota.json` and the P02 one-seed policy rest on, and has to clear
P09's relabelling-robustness min-rule *at every draw*. H31 (ILS/LNS at −0.0008 pp) is the local
precedent and it is not encouraging.

---

## 5. Conflict check against `killed.json` — done idea by idea

| idea | rules / kills touched | why it is genuinely different, or which revival condition it satisfies |
|---|---|---|
| **H59** minimal-FAS arc reclamation | **M4** (local is empty; be global-range or structural); **H31** (LNS); **H22** | Not a rank-window and not even a *position* move: the search object is the **arc set**, and the test is reachability in the forward DAG. Its range is the whole line (a reclaimed arc's topological re-sort may move thousands of nodes). It is not ruin-and-recreate: nothing is destroyed at random, and the acceptance is a proof of acyclicity, not a sampled gain. **M8** applies and is answered by construction: no existing class can produce this move, because none of them asks a reachability question — the redundancy fraction is measurable and predicted ≈ 0. Untouched: M1, M2, M3, M5, M6, M7, M9, M10 (no gradient, no dynamics, no multi-start, no ties, no init, no warm start). |
| **H60** net-digraph (`D⁺`) SCC decomposition | **M4** (structural neighbourhood = its named revival condition); **H36** (the confirmed champion class); **H46** revival_if (*"blocks defined by STRUCTURE … but that is H36"*) | This **is** H36 — with the correct structure graph. H46's revival condition asks for structurally-defined blocks; the point here is that H36's blocks are defined by the *wrong* structure, one in which a 100:1 reciprocal pair is an unbreakable 2-cycle. Not a new class, so M8's "classes compete for the same ground" risk is lower than for H41: it makes an existing, already-confirmed class strictly finer. **M7** does not apply — this is *cheaper* than what we run (`D⁺` has fewer arcs), not an expensive embedding. Untouched: M1–M3, M5, M6, M9, M10. |
| **H61** k-node joint optimal re-insertion (`k = 3..6`) | **M8** (credit ≠ advantage — the primary threat); **M4**; **H41** (segment, killed on 91.9% redundancy); **H45/H51/H52** (the `k = 2` case) | Distinct object: `k = 2` is confirmed to exist and to be non-redundant (H45's gate: 989 of 3,498 improving pairs are invisible to the sift), and a 3-node compound move need not decompose into improving 1- or 2-node moves. **But M8 is the reason this is ranked third, not first**: H41 fired 4,619 times and delivered 8.1% of its own credit. The gate below therefore measures the *incremental* gain over `k ≤ 2` directly, and reports the redundancy fraction, as M8 requires. |
| **H63** α-MOT certificate sweep (diagnostic) | **M4**; L01's own negative (*"reduction rules do not improve a heuristic's incumbent"*) | Filed **as a diagnostic, expected_pp ≈ 0**, precisely because L01's negative is right. Its value is the first *certificate* this campaign would own about its champion, and a localisation of the residual onto a small node set. Do not promote a champion from it. |
| **H62** batched noising / threshold acceptance | **M2** (dynamics null); **M3/M9/M10** (dispersion); **H13** revival_if; **H31** (ILS/LNS screen fail) | **M2 is scoped to the CONTINUOUS optimizer** (optimizer, LR, β schedule, clipping, EMA, restarts, gradient noise) and its evidence is a 9-hypothesis screen of gradient-phase interventions; this is a discrete move-selection rule and never touches the gradient phase. **H13's own revival condition names it**: *"Only as a DISCRETE stochastic move (random subset of movers in the sift) — that is a different axis"*. **M9/M10 do not apply as written** — there is no best-of-R and no extra arms; but their *spirit* does, and I say so plainly in § 7: this is a single trajectory whose whole claim is that a biased random walk finds what a monotone one cannot, and the campaign has one negative data point (H31) against exactly that claim. |

---

## 6. What I could not find — the honest negatives

1. **SSRN 6221201 remains unreadable.** Three attempts, three 403s, plus 429 from the Semantic
   Scholar API and 403 from ResearchGate. **The algorithm that produced our exact mission target is
   still unknown to this campaign**, and after § 3 the more important question is not *how* it was
   produced but *from what* — and the public record cannot answer that either.
2. **No new FAS/LOP work at our scale since L01.** Nothing on arXiv cs.DS in 2026 on scalable
   weighted FAS beyond the two Vahidi papers already known. The leaderboard has had **no new entry
   since 2026-01-09**. Vahidi's arXiv author listing has **no 2026 entry**. The GitHub repo has **no
   new code**.
3. **PACE is irrelevant now.** PACE 2025 was Dominating Set / Hitting Set; PACE 2026 is
   Maximum-Agreement Forest. The last FAS-adjacent PACE was 2022 (Directed Feedback Vertex Set), and
   OCM (PACE 2024) was already mined and found empty by L01.
4. **Multilevel coarsening for *directed* ordering is still open in that community.** Query 13
   returned only partitioning/MinLA material — the same conclusion L01 reached, re-confirmed. Nothing
   new. And L01's structural objection stands: edge-contraction coarsening destroys exactly the
   information our objective is made of.
5. **Nothing revives M1.** No 2025–2026 continuous relaxation addresses scale-blindness. Note the
   internal counterweight the campaign already owns and should keep separate: diagnosis **Q05**'s
   asymmetric surrogate is the one measured exception to "no continuous lever works" (+0.3668 pp on
   connectome pure-Rocket, −0.6689 pp on microns, graph-dependent), and **the literature adds nothing
   to it either way**. I found no external work on one-sided/asymmetric surrogates for FAS.
6. **No published method for the two independent teams that beat us** (Hashorva; Zheng, Tang,
   Okubo). The leaderboard lists no methods and neither team has a paper I could find. Ellis-Joyce /
   Both / Turaga are co-authors of Rocket-Crane, so their entries are that lineage.
7. **Ejection chains / variable-depth for LOP or FAS: still nothing**, at any scale. L01 searched
   this and found nothing; query 6 this time surfaced only the TSP/scheduling dynasearch line. Two
   independent scans finding nothing is a real negative — **stop looking here.**
8. **GPU-parallel permutation local search at 10⁵ nodes: nothing usable.** The GPU-SA literature is
   QAP/TSP at instance sizes far below ours, and it is a wall-clock axis, not a score axis
   (`CAMPAIGN.md` forbids starting Phase 2 unilaterally).

---

## 7. Ranked hypotheses — queue-ready

Bars every item must respect, restated so no gate below is quoted out of context:
`screen_delta_pp` connectome **0.012** / microns **0.002**; `min_promotion_delta_pp` the same;
the connectome **PROTOCOL CI** needs **0.02343 pp at n = 5** (H52's refusal); and since P09 the
connectome promotion additionally needs **every** relabelled delta above 0.012 pp plus a paired
one-sided 95% lower bound above 0.012 pp. **And microns has ≈ 190 s of slack** (3258.2 s idle
against the 3450 s guard deadline, from P05's closing measurement), so an item that *adds* seconds
on microns cannot pass a two-primary screen. That asymmetry is why H59 and H60 rank above H61.

---

### Rank 1 — **H59**: minimal-FAS arc reclamation (reachability, not gain)

**Mechanism (two sentences).** Build the DAG `F` of currently-forward edges; for each heavy
backward edge `(u,v)` test whether `v` can reach `u` in `F`, a search provably confined to the rank
interval `(rank[v], rank[u])`. If it cannot, `F ∪ {(u,v)}` is acyclic and any topological order of it
scores at least current + `w_uv`; greedily accept such arcs heaviest-first, maintaining acyclicity,
and re-sort once.

**Why it is not already dead.** § 5 row 1. It is the only mechanism in the campaign whose accept
test is a *proof of acyclicity* rather than a position-space gain, so no existing class can shadow
it; it satisfies M4's directive (unbounded range, structurally determined) and H31's revival
condition in a stronger form than H31 asked for (the "destroy" is a single arc chosen by weight, the
"repair" is exact by construction).

**Cheap CPU-only prototype gate (minutes, zero GPU).** From
`experiments/evidence/20260810T105922Z-H42-connectome-s31415-confirm-f41d7e_positions.npy`:
build `F`; take the top `K = 20,000` backward edges by weight; for each run a rank-interval-confined
bidirectional BFS with early exit; report `n_reclaimable`, `Σ w_reclaimable` in pp, and the
interval-length distribution of the reclaimable ones. Then verify end-to-end on the top few: apply
them one at a time with a full topological re-sort and check the realised oracle delta equals the
predicted `Σ w`. Repeat on microns.

**Kill condition.** `Σ w_reclaimable` over the top 20,000 backward edges is **< 0.012 pp** on
connectome **and** < 0.002 pp on microns → the champion's feedback arc set is effectively minimal,
the whole arc-reinsertion family closes, and the campaign gains its first optimality certificate for
free. (Secondary kill: reclaimable arcs exist but a full topological re-sort of `F ∪ R` loses more
than it gains once conflicts among `R` are resolved — measure, do not assume.)

**Honest expected_pp, and the argument against it.** **0 to +0.05 pp, midpoint +0.01, with maybe a
50% chance of exactly 0.** Against it: `F` holds ~4.48 M forward edges over 136,648 nodes (mean
out-degree ≈ 33) and forward reachability in a DAG that dense is close to universal over a
20,536-position mean interval, so most backward edges will be provably irreclaimable. Also, 5 of the
9 leaderboard entries above us were produced by pipelines that plausibly ran exactly this step
(A1's Algorithm 1 does), which is weak evidence that a well-refined order is already minimal. **For
it:** the effect-size bar is only 5,029 weight units, single backward edges reach 2,405, and 9,808
of our nodes sit outside the giant SCC where reachability is genuinely sparse. The asymmetry of cost
(minutes) to information (a certificate either way) is the reason this is rank 1, not the expected
score.

```json
{
  "id": "H59",
  "title": "Minimal-FAS arc reclamation - re-add backward edges whose endpoints are not forward-reachable",
  "status": "proposed",
  "priority": 1,
  "axis": "problem formulation / arc-set instead of position-space",
  "hypothesis": "The champion's order is a topological order of its forward-edge DAG F. If some backward edge (u,v) has no F-path from v to u, then F union {(u,v)} is acyclic and ANY topological order of it scores at least current + w_uv, exactly and monotonically. No move class in the campaign can find this, because reclaiming the arc may require re-sorting thousands of nodes none of which profits alone, and because the SCC refiner condenses the FULL graph G (where u and v are almost always in one SCC) rather than F union {e}. The claim: the total weight of such reclaimable backward edges on the champion's connectome order exceeds 0.012 pp.",
  "rationale": "This is the 'minimal feedback arc set' property of the FAS literature - 'none of the arcs can be reintroduced in the graph without disrupting acyclicity' (J. Comb. Optim. 10.1007/s10878-024-01209-8, abstract; full text NOT read, 303/IdP). It is an explicit step in Vahidi & Koutis's own MWFAS algorithm (arXiv:2412.16181v2 Alg. 1, HTML read: cancel cycles, delete zero-weight arcs, then 're-add edges that don't recreate cycles'). The campaign has never run it and has no certificate that its FAS is minimal. The test is affordable because a forward path can only move rightward in rank, so reachability from v to u is confined to the rank interval (rank[v], rank[u]) - no global search. The effect-size bar is unusually low in absolute terms: 0.012 pp is 5,029 weight units and a single backward edge here carries up to 2,405.",
  "kill_condition": "Over the top 20,000 backward edges by weight on the stored champion positions, the total weight of edges (u,v) with no F-path v->u is < 0.012 pp on connectome AND < 0.002 pp on microns. Then the FAS is effectively minimal, the arc-reinsertion family closes, and the certificate is the result. Secondary kill: reclaimable arcs exist but resolving conflicts among them (greedy heaviest-first with an acyclicity check) plus one topological re-sort realises < 0.012 pp against the frozen scorer.",
  "prior_evidence": [
    "autoresearch/lit/scan-cycle12.md section 4.1",
    "arXiv:2412.16181v2 Algorithm 1 (retrieved HTML)",
    "src/mfas/refine/scc_recursive.py:222 (the champion condenses the FULL graph, not F union {e})",
    "experiments/outputs/proto_H45_pair_connectome.json (1,178,821 backward edges on the champion order)"
  ],
  "est_cost": "prototype: minutes of CPU, ZERO GPU (interval-confined bidirectional BFS over the top 20k backward edges). Implementation if it clears: one new module src/mfas/refine/reclaim.py + src/mfas/experiments/H59.py. Deterministic - no RNG, so 1-seed screen on the primaries.",
  "expected_pp": "connectome 0 to +0.05, midpoint +0.01; ~50% chance of exactly 0. Queued for the certificate as much as for the score."
}
```

---

### Rank 2 — **H60**: condense the NET digraph, not the raw one (finest-Condorcet blocks)

**Mechanism (two sentences).** Replace the condensation graph inside `SccRecursiveRefiner._refine`
with `D⁺ = {(u,v) : w_uv > w_vu}`, so a reciprocal pair with `w_uv = 100, w_vu = 1` stops being an
unbreakable 2-cycle; everything else in the champion pipeline is untouched. MFAS on `W` equals MFAS
on `D⁺` plus the order-independent constant `Σ_pairs min(w_uv, w_vu)`, `D⁺`'s SCCs strictly refine
`W`'s, and the Extended Condorcet Criterion says every optimum respects the `D⁺` condensation — so a
single pass with `D⁺` **dominates** a single pass with `W`, exactly.

**Why it is not already dead.** § 5 row 2. This is not a new move class competing with the existing
ones (M8's failure mode for H41); it makes the campaign's *highest-yield* confirmed class (H36,
+0.1837 pp) strictly finer, using the structure graph the rank-aggregation literature says is the
right one. It satisfies H46's revival condition ("blocks defined by STRUCTURE") in the only way H46's
own note left open. M7 does not bite: `D⁺` is *cheaper* than `G` (fewer arcs), which matters under
P07.

**Cheap CPU-only prototype gate (minutes, zero GPU).** Three numbers, in order of cost:
(a) how many ordered pairs are reciprocal, and what weight they carry;
(b) the top-level SCC size distribution of `G` versus `D⁺` (one `scipy.sparse.csgraph`
    `connected_components` call each) — is the giant SCC still ~126,840 nodes?
(c) the decisive one: from the stored champion positions, run **one** pass of `scc_recursive_refine`
    with the `D⁺` arc set and measure the realised delta against the frozen scorer. The raw version
    returns ≈ 0 there by construction (the champion is its fixed point), so this delta **is** the new
    gain. Then (d) the composed test: stage-4 alternation at matched cycle counts, both arc sets,
    reporting the per-class credit split as M8 requires.

**Kill condition.** One `D⁺` pass on the champion's connectome order realises **< 0.012 pp** and the
matched-cycle composed alternation realises < 0.012 pp → the reciprocal-pair welding is not costing
us anything measurable and the decomposition-graph family closes. On microns the same at 0.002 pp.

**Honest expected_pp, and the argument against it.** **0 to +0.10 pp, midpoint +0.02**, which is
*below* the 0.02343 pp connectome PROTOCOL CI threshold in the median case — say so up front. Against
it: H46 measured coarse block permutation at **exactly +0.000000 pp** over 28 partitions, which says
the champion's order is extremely well consolidated at every scale from 17,081 to 68,324 nodes; and
the champion is already a joint fixed point of the sift, so any pair that is *net*-misordered and
individually movable has already been fixed. The gain must come from *joint* rearrangements of many
small `D⁺`-SCCs at once, which is exactly what H46 found nothing of at coarse scales. **For it:** the
dominance argument is a proof, not a hope — the gain cannot be negative in a single pass — and the
whole thing is one substitution in a module the campaign already trusts, with a *lower* arc count.
The realistic risk is not that it regresses; it is that it returns 0.000000 and closes another door.

```json
{
  "id": "H60",
  "title": "Condense the NET digraph (finest-Condorcet blocks), not the raw digraph, inside the SCC refiner",
  "status": "proposed",
  "priority": 1,
  "axis": "decomposition",
  "hypothesis": "SccRecursiveRefiner condenses the RAW digraph (scc_recursive.py:222 builds coo_matrix over np.ones on every edge), so every reciprocal pair u<->v is a hard 2-cycle and can never be separated - even at w_uv=100, w_vu=1. Maximising forward weight on W is identical to maximising it on the net digraph D+ = {(u,v) : w_uv > w_vu} with weight d_uv = w_uv - w_vu, up to the order-independent constant sum_pairs min(w_uv,w_vu). D+'s arcs are a subset of G's, so D+'s SCCs STRICTLY REFINE G's, and one pass of the refiner on D+ therefore dominates one pass on G, exactly and monotonically. The claim: substituting D+ for G as the condensation graph, with nothing else changed, gains > 0.012 pp on the champion's connectome order.",
  "rationale": "The rank-aggregation literature names this object: the Extended Condorcet Criterion says every optimum respects a partition with delta_{x_i x_j} > 0 across blocks, and the FINEST such partition is the SCC condensation of the majority (net) digraph (arXiv:2506.15097, HTML retrieved; Akbari & Escobedo, Computers & OR S030505482300028X, abstract only - NOT read). H36 - the largest score move this campaign ever made, +0.1837 pp - is this mechanism computed on the wrong structure graph. This is not a new competing move class (M8's H41 failure mode), it makes an existing confirmed class finer. It also cuts arcs rather than adding work, which matters because microns has only ~190 s of slack against the 3450 s guard deadline (P05 closing measurement, 3258.2 s idle).",
  "kill_condition": "One pass of scc_recursive_refine with the D+ arc set, from the stored H42 champion positions, realises < 0.012 pp on connectome against the frozen scorer AND the matched-cycle composed stage-4 alternation also realises < 0.012 pp; microns the same at 0.002 pp. Then reciprocal-pair welding costs nothing measurable and the decomposition-graph family closes. Report the SCC size distributions of G and D+ either way - that number is a diagnosis the campaign does not have.",
  "prior_evidence": [
    "autoresearch/lit/scan-cycle12.md section 4.2",
    "arXiv:2506.15097 (Extended Condorcet Criterion, retrieved HTML)",
    "src/mfas/refine/scc_recursive.py:222 (the np.ones condensation) and :248 (SCC-internal order preserved)",
    "autoresearch/killed.json H46 revival_if ('blocks defined by STRUCTURE rather than by equal position ranges')",
    "experiments/log.md 2026-08-10 (H36 confirm, +0.1837 pp on connectome)"
  ],
  "est_cost": "prototype: minutes of CPU, ZERO GPU (one connected_components call per arc set, then one refiner pass from stored positions). Implementation if it clears: a net-arc-set constructor plus a keyword on SccRecursiveRefiner - scc_recursive.py's existing behaviour must remain the default so the champion stays bit-reproducible. Deterministic - 1-seed screen on the primaries.",
  "expected_pp": "connectome 0 to +0.10, midpoint +0.02 - BELOW the 0.02343 pp PROTOCOL CI threshold in the median case, stated up front. H46's exact +0.000000 pp over 28 coarse partitions is the strongest prior against it."
}
```

---

### Rank 3 — **H61**: k-node joint optimal re-insertion, `k = 3..6` (insert-distance-`r` neighbourhood)

**Mechanism (two sentences).** Extract a structurally chosen set `S` of `k` nodes from the line and
re-insert them **jointly optimally** — both their internal order and their interleaving into the
remaining line — by a DP over (subset placed, gap), using the fact that each node's insertion profile
is piecewise constant with `O(deg)` breakpoints so the candidate gap set is `O(Σ_{j∈S} deg(j))`
rather than `O(n)`. The champion has `k = 1` (the sift) and, via H52, `k = 2`; `k ≥ 3` is a move no
sequence of improving `k ≤ 2` moves need reach.

**Why it is not already dead.** § 5 row 3. C1 (AAAI 2021 / arXiv:2204.02902) gives this
neighbourhood a name (*insert distance r*) and shows it is `n^{O(r)}` in general; our pairwise
objective plus breakpoint compression is what makes it cheap here. H45's gate already proved the
`k = 2` case is non-redundant with the sift (989 of 3,498 improving pairs invisible to it), so the
ladder is not obviously exhausted.

**Cheap CPU-only prototype gate (tens of minutes, zero GPU).** From the stored champion positions,
generate candidate triples by structure — for each of the top-K heaviest backward edges `(u,v)`,
take `S = {u, v, z}` over the few heaviest `z` incident to both endpoints inside the interval — run
the `k = 3` DP, and report **two** numbers: total realised gain, and, per M8, the fraction of that
gain that survives after the `k ≤ 2` moves have been applied first (the **redundancy fraction**).

**Kill condition.** The `k = 3` gain **incremental over `k ≤ 2`** is < 0.012 pp on connectome, or the
redundancy fraction exceeds 80% (H41 died at 91.9%), or the realised pp/s is below the stage it
would be inserted into. Any of the three kills it.

**Honest expected_pp, and the argument against it.** **0 to +0.05 pp, midpoint +0.01, and I expect
this to die on redundancy.** Against it: M8 was written *because* a plausible new move class fired
4,619 times and delivered 8.1% of its own credit; a `k = 3` class overlaps the `k = 2` class far more
than segments overlapped insertion. It also *adds* seconds, which microns cannot afford (≈190 s of
slack), so even a connectome win is a one-primary result — the exact position H52 is stuck in. **For
it:** it is the only remaining rung of a ladder whose lower rungs are both confirmed non-empty, and
the gate produces the redundancy number the campaign needs before anyone proposes `k = 4`.

```json
{
  "id": "H61",
  "title": "k-node joint optimal re-insertion (k=3..6) by breakpoint-compressed subset DP - the insert-distance-r neighbourhood",
  "status": "proposed",
  "priority": 3,
  "axis": "discrete refinement / move class",
  "hypothesis": "Our ladder has k=1 (single-node exact-gain sift) and k=2 (H45/H52 pair relocation) and nothing above. Extracting k nodes and re-inserting them JOINTLY optimally - internal order and interleaving together - is exact by a DP over (subset placed, gap): f[g][T u {j}] = f[g][T] + prof_j(g) + sum_{i in T} (w_ij - w_ji). Because each node's insertion profile is piecewise constant and changes only at its own neighbours' positions, the candidate gap set is O(sum_{j in S} deg(j)) rather than O(n), so the DP is O((sum deg) * 2^k * k) ~ 2e5 operations at k=6 and mean total degree 83 - NOT O(n^2). The claim: a sweep of k=3 joint re-insertions over structure-selected triples gains more than 0.012 pp INCREMENTALLY over what k<=2 already takes.",
  "rationale": "The neighbourhood has a name and a complexity in the literature - 'insert distance r', n^{O(r)} poly by enumerate-then-DP (Gruttemeier, Komusiewicz, Morawietz, AAAI 2021 / arXiv:2204.02902, ar5iv HTML retrieved) - and their framework needs a per-variable-decomposable objective where ours decomposes per PAIR, which is stronger. H45's prototype established the k=2 rung is real and non-redundant with the sift: 3,498 strictly improving pairs on the champion order, 989 of them (28.3%) invisible because neither endpoint profits alone, best move spanning 92,958 positions. Their own honest note transfers too: the bottleneck was the insert operation, not the combinatorics - which is exactly what H51 rediscovered here.",
  "kill_condition": "ANY of three: (1) the k=3 gain INCREMENTAL over k<=2 is < 0.012 pp on connectome; (2) the redundancy fraction (k<=2 credit before minus after, over the k=3 class's own credit) exceeds 80% - H41 died at 91.9%; (3) realised pp/s is below the marginal rate of the stage it is inserted into. Report all three whatever the verdict, per M8.",
  "prior_evidence": [
    "autoresearch/lit/scan-cycle12.md section 4.3",
    "arXiv:2204.02902 (insert distance r, retrieved via ar5iv)",
    "experiments/outputs/proto_H45_pair_connectome.json (the k=2 rung: 3,498 improving pairs, 989 invisible to the sift)",
    "autoresearch/killed.json M8-credit-is-not-advantage and H41 (killed on 91.9% redundancy)"
  ],
  "est_cost": "prototype: tens of minutes of CPU, ZERO GPU. Implementation if it clears: a new module; it ADDS seconds, which microns (~190 s of slack) cannot afford - so plan for a connectome-only result and read H52's refusal before building.",
  "expected_pp": "connectome 0 to +0.05, midpoint +0.01. I expect this to die on redundancy; it is queued because the redundancy number is needed before anyone proposes k=4."
}
```

---

### Rank 4 — **H63** (DIAGNOSTIC, not a champion path): α-MOT certificate sweep

**Mechanism (two sentences).** For each edge `(u,v)` with `d_uv > 0`, test the majority-order
condition `d_uv > min_{α+β=1} Σ_{z} max(0, α d_vz + β d_zu)`; where it holds, the relative order of
`u` and `v` is fixed in **every** optimum. Since `d = 0` for every non-adjacent pair, only the 5.66 M
edges can be constrained and the interference set reduces to the **common neighbourhood**, so the
sweep is a triangle enumeration rather than an `O(n³)` job.

**Why it is here at all.** It produces two things the campaign has never had: (a) a *certificate* —
"X% of the constrainable pairs in our champion's order are provably in their optimal relative
orientation"; and (b) if any *fixed* pair is oriented the wrong way in our order, a **provably
improving direction** that no local search found. It is also the natural localiser: the unfixed pairs
are where the residual 0.4606 pp must live, and a small enough unfixed set could then justify an
expensive method on it.

**Kill condition.** Zero violated fixed pairs **and** the unfixed set is not materially smaller than
the whole graph → the rule is uninformative here and the pair-fixing axis closes for good.

**Honest expected_pp.** **≈ 0 directly.** L01's negative is right — reduction rules serve exact
solvers, and we run none. On a graph with mean total degree 83 and heavy-tailed hubs the interference
sums will be large and the rule will mostly fire on light pairs where it says nothing. Filed at low
priority, for the certificate.

---

### Rank 5 (LAST, and I argue against it myself) — **H62**: batched noising / non-monotone acceptance

**Mechanism (two sentences).** Inside a sift sweep, take the argmax of `exact profile + decaying
noise` instead of the argmax of the exact profile, with the amplitude decaying to exactly 0 so the
final phase is the champion's own monotone converger, and best-by-oracle tracking exactly as the
baseline already does. This is Charon & Hudry's noising method — developed for the linear ordering
problem — expressed in our batched kernel instead of as a single-move trajectory.

**Why it is not already dead.** § 5 row 5. M2 is scoped to the *continuous* optimizer's dynamics and
its evidence is a nine-hypothesis screen of gradient-phase knobs; this never touches the gradient
phase. H13's revival condition names a discrete stochastic move as "a different axis". M9/M10 do not
apply as written — no best-of-R, no extra arms.

**Kill condition.** On mouse + hard-synthetic + one connectome SCC block at matched wall-clock, no
noise schedule beats the champion's monotone converger → dead before any connectome run.

**Honest expected_pp and the argument against it, which I think is the stronger side.** **0 to
+0.10, midpoint ≈ 0.** Five reasons to distrust it: (1) the literature's success is at **n ≤ 100**,
455–1,366× below us, and a random walk's mixing cost grows with the space; (2) H31 already tested the
generic "perturb and re-descend" idea here and lost (−0.0008 pp at 1.9× wall); (3) it consumes RNG,
so `seed_class.py` classifies it `rng`, the screen goes back to 3 seeds on the primaries and a cycle
goes from ~2.5 h to ~8.5 h; (4) it destroys the bit-determinism that `sota.json` (std = 0) and the
P02 one-seed policy rest on; (5) P09's relabelling min-rule requires **every** draw's delta to clear
0.012 pp, which a stochastic mechanism is structurally worse at than a deterministic one. **The one
reason to keep it on the list:** the acceptance policy is the only level in `CAMPAIGN.md`'s
escalation list that this campaign has never touched — every mechanism it has ever shipped is a
strictly monotone hill-climb. Do not run this before H59 and H60.

---

## 8. Two things for the cycle that are not hypotheses

1. **A Track-B diagnostic that would redirect the whole campaign, but needs a protocol ruling
   first.** Nobody has ever asked: *is the reference order itself a fixed point of our move classes?*
   Running the champion's stage-3 sift and stage-4 alternation **on the reference ordering** answers
   whether our move classes are the limitation (they would improve it) or the basin is (they would
   not). It costs minutes of CPU. **But** `CAMPAIGN.md` rule 4 says the reference is
   "for *measurement only*, read only through `mfas.analysis.gap`", and feeding it into a refiner is
   arguably past that line. Any resulting positions would be irreversibly contaminated and could
   never seed anything. **I am not queueing it.** I am flagging it as the highest-information
   cheap experiment available *if and only if* the operator rules that it counts as measurement.
2. **H45's free 1.93× is still unclaimed.** `select_disjoint_moves` (`segment.py:315`) packs
   disjoint moves greedily; the optimal weighted-interval-scheduling DP measured **1.93×** the greedy
   gain in `proto_H45_pair_connectome.json`. That is the dynasearch construction (§ 4.3) and it is a
   pure code fix, not a hypothesis. It only pays wherever batched-disjoint application still runs —
   which after H51/H52 is `segment.py` and the SCC path, not pair relocation. Worth one line in
   whatever cycle next touches that machinery.
