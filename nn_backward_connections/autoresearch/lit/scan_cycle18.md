# Literature scan L03 — cycle 18 (SCHEDULED REFRESH, not divergent mode)

**Date:** 2026-08-27
**Author role:** scout (literature only)
**Compute run by this scan: NONE.** No GPU, no CPU experiment. Every internal number is quoted
from a file already in the repo; every external number carries a retrieved source or is explicitly
marked unretrieved.
**Files written:** this one, plus three items appended to `autoresearch/queue.json` (H71, H72, H73).
Nothing else was touched.

**Why this scan exists.** `cycles_since_literature_scan` reached 5. The campaign is *not* in
trouble — cycle 17 promoted H64 as the connectome champion (84.25818%, +0.104084 pp). This is a
standing refresh under `CAMPAIGN.md § Escalation` ("force a literature scan every 8 cycles
regardless of outcomes").

**The constraint that shapes this scan, stated up front.** The previous scan (L02,
`scan-cycle12.md`) ran on **2026-08-25 — two days ago**. A "what is new since the last scan" search
is therefore dry *by construction*, and I did not pretend otherwise. The value here had to come
from **axes the earlier scans recorded as un-searched**, and those are named explicitly in
`lit/README.md § Coverage gaps` (spin-glass/Potts annealing; GPU-parallel permutation local search;
DAG structure-learning orderings) plus the two the launch brief added (LP/SDP rounding for linear
ordering; surrogate shapes for continuous relaxations of ordering). All five were covered. Three of
them are dry and I say so in § 5.

---

## 1. Executive summary

1. **The one genuinely new move class in this scan is the `tuck`** (§ 3.1 → **H71**), from the
   DAG-structure-learning literature (GRaSP, Lam–Andrews–Ramsey, UAI 2022 / arXiv:2206.05421,
   Definition 4.1, **retrieved**). Translated into our vocabulary it is the *subset* generalisation
   of H45/H52's pair relocation: instead of meeting `u` and `v` at a single **cut** of the interval
   between them, it splits the interval into two **arbitrary** groups `⟨γ, u, v, γᶜ⟩`. A prefix `γ`
   *is* a cut, so the move strictly generalises the campaign's `k = 2` rung, and the extra gain term
   is one computable coupling sum. Nothing in the campaign moves interval-interior nodes.
2. **A weight-blind constant sits inside the campaign's highest-yield move class** (§ 3.2 → **H72**).
   `scc_recursive.py:275` cuts a single-SCC block at `round(nb * split_frac)` — a fixed *positional*
   fraction from `DEFAULT_SPLIT_FRACS`. The classical divide-and-conquer approximation for min-FAS
   (Leighton–Rao / Even–Naor–Schieber–Sudan) cuts at a **minimum directed balanced cut** at exactly
   this point, and RASstar (Xiong et al., Comput. & OR 2024) recursively partitions "into two
   sequential subgraphs". Our splitter is the degenerate, weight-blind member of that family, and
   the better cut costs the *same asymptotic work* — one difference array plus a cumsum over the
   block's own edges, which `_split` already touches.
3. **The sift's tie-break is an undocumented, direction-asymmetric constant** (§ 3.3 → **H73**).
   `insertion.py:208` takes the **first** argmax breakpoint ("ties -> smallest b, fine") and
   `insertion.py:236` gives gap 0 the win on ties. Consequence, which the comment does not state:
   a node moving **rightward** stops at the *near* edge of its optimal plateau, while a node moving
   **leftward** overshoots to the *far* edge. That asymmetry is applied to ~10⁵ nodes simultaneously
   per Jacobi sweep. The campaign already has a **measured** leverage figure for tie-break structure
   — P09's relabelling study, which perturbs *nothing but tie-breaks*, moves the connectome
   champion by σ = 0.019124 pp — so this is not a speculative knob.
4. **Calibration, and it is unflattering.** The FlyWire leaderboard was re-read today: **unchanged
   since 2026-01-09**, 12 entries. H64's +0.104084 pp — the largest single connectome move of the
   autonomous campaign — **overtook exactly zero leaderboard entries.** Seven public solutions still
   sit above the champion, the same seven that sat above H42.

---

## 2. Search log

### 2.1 Queries actually issued (WebSearch), 2026-08-27

1. `learning to rank pairwise surrogate loss consistency AUC maximization weighted pairwise ranking scalable`
2. `SDP rounding maximum acyclic subgraph linear ordering approximation practical algorithm`
3. `GPU parallel local search permutation ordering problem CUDA large scale 2025 2026 linear ordering`
4. `topological order search DAG structure learning permutation local search scalable 2025 order-based`
5. `GRaSP greedy relaxations sparsest permutation "tuck" move permutation search causal discovery`
6. `"Scaling Up Order-Based Search" reinsertion moves BOSS grow-shrink trees 2025`
7. `arXiv 2026 feedback arc set weighted heuristic large graphs new algorithm`
8. `arXiv 2501.06935 feedback arc set decompositions weighted oriented graphs`
9. `plateau neutral moves local search tie-breaking sifting variable ordering BDD minimum displacement`
10. `divide and conquer feedback arc set balanced cut minimize crossing backward edges recursive splitting heuristic`
11. `RASstar "Finding small feedback arc sets on large graphs" recursive ordering partition rule GreedyAbs reduction`

### 2.2 Pages actually opened (WebFetch)

| source | result |
|---|---|
| `ar5iv.labs.arxiv.org/html/2206.05421` (GRaSP, Lam–Andrews–Ramsey) | **read** — Definition 4.1 (`tuck`) extracted verbatim-ish, plus the DAG-associahedron statement |
| `arxiv.org/html/2605.05568v1` (SCOPE, relaxed sparsest permutation, 2026-05-07) | **read** — and rejected, see § 5.3 |
| `codex.flywire.ai/app/mfas_challenge?dataset=fafb` | **read** — full 12-row leaderboard, **unchanged since 2026-01-09**, no methods published |
| `arxiv.org/pdf/2206.05421` | fetched, **PDF text not extractable** (same failure as every prior scan) |

### 2.3 Retrieval failures — recorded, not worked around

| source | what happened |
|---|---|
| **SSRN 6221201** (Vahidi & Koutis 2026 — the paper behind our exact 84.6147 target) | **403 Forbidden again.** This is the **fourth** documented failure across three scans. The algorithm that produced the mission target is still unknown to this campaign. |
| local `2506.13799v1.pdf` (repo root) | **still unreadable.** The launch brief asked me to "read it properly"; I could not. `Read` needs `pdftoppm` (poppler), which is still not installed, and Bash is disabled in this session. This is the **third** consecutive scan blocked on the same missing binary. Everything this campaign knows about Vahidi 2025 still comes from the arXiv HTML (`lit/vahidi-2025.md`, `lit/notes-vahidi-flywire.md`). |
| `sciencedirect.com/.../S0305054824001965` (RASstar) | 403, third failure. Abstract + indexed snippets only. |

> **Operator note, repeated because it has now cost three scans.** Installing any `pdftotext`
> (poppler) on this box would unblock the campaign's own local reference PDFs. It is still the
> single cheapest research-infrastructure fix available.

---

## 3. Sources read — mechanism extracted, in our vocabulary

### 3.1 The `tuck` move (move-class level) — **the best transfer in this scan**

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| A1 | Lam, Andrews, Ramsey, *Greedy Relaxations of the Sparsest Permutation Algorithm* | UAI 2022 / arXiv:2206.05421 | 2022-06 | https://ar5iv.labs.arxiv.org/html/2206.05421 | **yes** (ar5iv HTML, Def. 4.1) |
| A2 | Andrews, Ramsey et al., *Fast Scalable and Accurate Discovery of DAGs Using BOSS and Grow-Shrink Trees* | arXiv:2310.17679 | 2023-10 | https://arxiv.org/pdf/2310.17679 | abstract/snippets only |
| A3 | Grüttemeier, Komusiewicz, Morawietz | AAAI 2021 / arXiv:2204.02902 | 2021 | — | already mined by L02 (§ 4.3), **not re-mined** |

**Definition, as retrieved (A1, Def. 4.1).** Given a permutation `π` and vertices `j, k` with
`π[j] < π[k]`, write `π = ⟨δ₁, j, δ₂, k, δ₃⟩`. Partition `δ₂` into `γ` (ancestors of `k` in `G_π`)
and `γᶜ`. Then `tuck(π, j, k) = ⟨δ₁, γ, k, j, γᶜ, δ₃⟩`. The paper states that *"traversing any edge
of the DAG-associahedron can be equivalently done via a tuck"*, i.e. it is a **complete** move for
the polytope of orderings under their score, and it is the operation that let GRaSP relax
faithfulness where earlier permutation methods could not.

**Mechanism in our vocabulary.** Take a violated (backward) edge `u → v`, `rank[v] < rank[u]`.
Write the current order as `⟨δ₁, v, δ₂, u, δ₃⟩` where `δ₂` is the *interval interior*. Choose any
subset `γ ⊆ δ₂` and produce `⟨δ₁, γ, u, v, γᶜ, δ₃⟩`, each group keeping its internal relative order.
By the contiguous-block lemma `segment.py` already proves, no edge with an endpoint outside
`[rank v, rank u]` can flip, so the exact gain is

```
Δ(γ) = (w_uv − w_vu)
     + Σ_{z ∈ γ}  (w_zv − w_vz)
     + Σ_{z ∈ γᶜ} (w_uz − w_zu)
     + Σ_{z ∈ γᶜ, z' ∈ γ, pos(z) < pos(z')} (w_{z'z} − w_{zz'})
```

Derivation, term by term: `u` moves in front of every `z ∈ γᶜ` and stays behind every `z ∈ γ`;
`v` moves behind every `z ∈ γ` and stays in front of every `z ∈ γᶜ`; a pair inside `δ₂` flips iff
the earlier one is in `γᶜ` and the later one is in `γ`.

**Why this is not H45/H52.** If `γ` is a **prefix** of `δ₂` the fourth term is identically zero and
`Δ` collapses to exactly the formula in H45's `scout_L01_addendum` — i.e. **H45/H52's move is the
prefix special case of the tuck**. The general subset move is strictly larger, and it is the only
move in the campaign that reorders *interval-interior* nodes: the sift moves one node, H52 moves
two, H41's segments move rigid contiguous blocks, `scc_recursive` permutes whole SCCs. Nothing
splits an interval by a **non-positional predicate**.

**What does NOT transfer.** GRaSP's *choice* of `γ` (ancestors of `k` in the DAG implied by the
current order) is tied to their BIC-style score and to faithfulness relaxations that have no meaning
here; and their evaluation is per-variable-decomposable while ours is per-pair. The **operation**
transfers; the **selection rule** does not, and must be re-derived from our gain formula (see H71's
three candidate `γ` rules). A2's Grow-Shrink-Tree caching also does not transfer — it caches
conditional-independence scores, whereas our per-node profile is already recomputed in one `O(m)`
vectorised pass.

### 3.2 Where a recursion cuts (decomposition level)

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| B1 | Xiong et al., *Finding small feedback arc sets on large graphs* (**RASstar**) | Comput. & Oper. Res., S0305054824001965 | 2024 | https://www.sciencedirect.com/science/article/abs/pii/S0305054824001965 | **no** — 403 (3rd failure). Abstract + indexed snippets |
| B2 | Even, Naor, Schieber, Sudan, *Approximating minimum feedback sets and multicuts* (spreading metrics / Leighton–Rao divide-and-conquer) | classic | 1998 | https://people.orie.cornell.edu/shmoys/pdf/multicut.pdf | snippets; the method is standard textbook material |

**Mechanism in our vocabulary.** The classical `O(log n log log n)` approximation for min-FAS is a
recursion: cut the vertex set with an approximately **minimum directed balanced cut**, order the two
sides, recurse. The cut is chosen to *minimise the weight that crosses backwards*, because that
weight is exactly what the recursion can never repair afterwards. RASstar's indexed description is
the same shape — "the graph is recursively partitioned into two sequential subgraphs … until all
subgraphs collapse into single vertices".

**Where our code sits on that spectrum.** `SccRecursiveRefiner._split` (`scc_recursive.py:272-281`):

```python
mid = lo + max(1, min(nb - 1, int(round(nb * self.split_frac))))
```

The cut is a **fixed fraction of the block's position range**, drawn from
`DEFAULT_SPLIT_FRACS = (0.5, 0.382, 0.618, 0.25, 0.75)` and cycled across alternation rounds. It
never looks at a single edge weight. Every backward edge straddling that cut is invisible to both
halves for the rest of the pass, and only the single-node sift can repair it.

**Why the better cut is affordable.** For a block `[lo, hi)` with edge index set `eidx`, the
backward weight crossing a candidate cut `mid` is
`X(mid) = Σ_{backward e in block} w_e · 1[pos(v) < mid ≤ pos(u)]`. Every backward edge contributes
`+w` on the half-open range `(pos v, pos u]`, so **one difference array plus one cumsum gives
`X(mid)` for all `mid` at once**, in `O(|eidx| + nb)` — the same asymptotic cost as the two boolean
masks `_split` already builds. Choose `mid = argmin X` restricted to a balanced band around the
current `split_frac`, which preserves the round-to-round boundary diversity the module docstring
says the cycled fractions exist for.

**What does NOT transfer.** The *guarantees*. Leighton–Rao style bounds need an approximate
**sparsest**/balanced cut solved by an LP or a spectral relaxation; at 136,648 nodes that is exactly
what M7 forbids (H33's magnetic-Laplacian eigensolve cost 304–507 s and lost). The prefix-scan
`argmin` above is **not** a directed sparsest cut — it is the best cut *among the n positions of the
current order*, a one-dimensional restriction. That restriction is what makes it free, and it is
also the honest reason to expect a small effect.

### 3.3 Tie-breaking as a first-class search decision (update-rule level)

| # | source | id / venue | date | link | retrieved? |
|---|---|---|---|---|---|
| C1 | Rudell, *Dynamic variable ordering for OBDDs* (sifting) | ICCAD 1993 | 1993 | — | snippets; sifting is the ancestor of our move class |
| C2 | Frank, Cheeseman, Stutz, *When Gravity Fails: Local Search Topology* (plateaus / neutral moves) | JAIR 7:249–281 | 1997 | https://www.researchgate.net/publication/1961077 | snippets only |
| C3 | Verel et al., *NILS: a Neutrality-based Iterated Local Search* | arXiv:1207.4450 | 2012 | https://arxiv.org/pdf/1207.4450 | snippets only |

**What the literature says, briefly.** Sifting (C1) is *exactly* our move class under another name:
move one variable to its locally optimal position with all others fixed. The local-search topology
literature (C2/C3) is about what happens when the objective is **flat**: plateaus dominate the
search landscape, "neutral moves" (zero-delta moves) are the mechanism by which a hill-climber
escapes, and one of the indexed results puts the fraction of steps with a tied best score at
**61.2%** in a modern SLS solver — i.e. tie-breaking is not a detail, it is most of the trajectory.

**Where our code sits.** `jacobi_best_gaps` computes each node's exact insertion profile, which is
piecewise constant with **at most `deg(u)` breakpoints over 136,648 gaps** (mean total degree here
is 83). Optimal *plateaus* are therefore the rule, not the exception. Two lines decide what happens
on one:

```python
seg_argmax_pos = np.minimum.reduceat(cand, seg_first_pos)  # first max position   (insertion.py:208)
                                                           # comment: "ties -> smallest b, fine"
use_bp = seg_max > gap0_val                                #                        (insertion.py:236)
```

So a node is always sent to the **leftmost** gap of its optimal plateau. Note the asymmetry this
creates, which the comment does not mention: a node whose plateau lies to its **right** lands on the
*near* edge (minimum displacement); a node whose plateau lies to its **left** is transported to the
*far* edge (maximum displacement). With `n_movers` in the 10⁴–10⁵ range per Jacobi sweep, that is a
systematic direction-dependent over-transport, interacting directly with the Jacobi collision that
under-relaxation (H35, +0.098 pp) exists to damp.

**What does NOT transfer.** C2/C3's actual proposals — random neutral walks, neutrality-based ILS —
consume RNG, and the campaign's own arithmetic is against them: `seed_class.py` would classify the
variant `rng`, tripling the screen, destroying the bit-determinism `sota.json` (std = 0) and the P02
one-seed policy rest on, and P09's relabel gate requires *every* draw to clear the bar. Only the
**deterministic** half of the idea — that the tie-break rule is a design decision and not a
detail — is taken here.

---

## 4. Conflict check against `killed.json` — idea by idea, strictly

| idea | rules / kills touched | why it is genuinely different, or which revival condition it satisfies |
|---|---|---|
| **H71** subset-bipartition ("tuck") interval repair | **H45** (killed as filed) / **H51-H52** (confirmed, promoted on mouse); **H41** (killed at 91.9% redundancy); **H61** (proposed, unrun); **M4**; **M8**; **M11**; **M12** | H45's kill is explicitly *"it falsifies the placement, not the mechanism"*, and its revival ran as H51→H52 and is **confirmed**. So the `k=2` rung is alive, not dead, and H71 strictly generalises it: a prefix `γ` reproduces H45's formula exactly, so `Δ_tuck ≥ Δ_pair` per candidate by construction. Distinct from **H41**: H41 moved *rigid contiguous* blocks; here the interval is split by a **non-positional predicate**, which no contiguous move can express. Distinct from **H61**: H61's neighbourhood is *insert-distance-r* — `r` nodes move, the rest stay — whereas a tuck moves the entire interval interior (mean 20,536 positions). **M4** is satisfied in its directive form (global range, structurally determined). **M8** is the live threat and is answered by construction in the gate: the incremental gain *over* `k ≤ 2` and the redundancy fraction are both required outputs. **M11** is respected: the gate is a **realised** exact gain verified against the frozen scorer, never a confined-weight ceiling. **M12** applies (from-champion prototype ⇒ screening number only) and is stated in the item. Untouched: M1, M2, M3, M5, M6, M7, M9, M10. |
| **H72** minimum-crossing-weight split point | **H60** (killed, decomposition axis, +0.0073 pp screen fail); **H46** (killed, exactly +0.000000 pp); **M4**; **M7**; **M8**; **M12** | Same *axis* as H60 and that must be faced head-on. H60 changed **which graph** is condensed (`G` → `D⁺`); H72 changes **where a single-SCC block is cut**, a lever H60 never touched and which its kill note does not cover — H60's kill is specific ("past cycle 77 the NET curve is FLATTER than the raw one"), not a closure of the decomposition axis. Distinct from **H46**: H46 *permuted* equal-size blocks of the line and found the identity optimal at every `K`; H72 permutes nothing, it relocates a recursion boundary. **M7** does not bite: this is a prefix scan over the block's own edges, `O(|eidx| + nb)`, not a spectral embedding — it costs the same as the code it replaces. **M8**: not a new move class, so no new competitor for the same ground; redundancy is still reported. **M12** applies at full force (inner-stage change, H60 measured 3.76× overstatement from a champion start) and the item says so. **Honest prior against it: H60.** A *strictly dominating* structural improvement to this same refiner returned +0.0073 pp, 1.64× short of the bar. Untouched: M1, M2, M3, M5, M6, M9, M10, M11. |
| **H73** displacement-minimising tie-break | **M5-no-ties** (the strongest objection); **M2**; **H09**; **H55** (killed); **H58** (proposed); **M9/M10** | **M5 must be argued, not waved past.** M5 reads *"the continuous optimizer leaves 0 exact position ties … nothing is recoverable from tie-breaking"*, and its evidence is `findings.md #2` + the **H09** kill (anti-tie jitter on near-equal *continuous positions*, −0.001 pp). The object here is a different one: a tie in the **argmax of the discrete insertion profile**, a piecewise-constant function with ≤ `deg(u)` breakpoints over 136,648 gaps, so optimal plateaus are *structurally guaranteed* to be wide. M5's measurement cannot bind on an object it did not measure — but the item's own prototype rung is to **measure the plateau widths first**, so if M5's spirit is right this dies for free. **M2** is scoped to the continuous optimizer's dynamics; the campaign's own counter-evidence that a *discrete* update-rule change moves the metric is H35 (under-relaxation, +0.098 pp). Counter-precedent stated honestly: **H55** (block-sequential Gauss–Seidel) was also a discrete update-rule change and lost at every `K`. Distinct from **H58**: H58 changes the input **labelling** and needs a ~1.9 h GPU premise study; H73 changes the **rule**, at zero GPU, and partially answers H58's own open question (if the rule matters, tie-break structure is causal). **M9/M10 do not apply at all**: 1× cost, no best-of-R, no RNG — `seed_class.py` still classifies it deterministic. |

---

## 5. What I could not find — the honest negatives

### 5.1 Nothing new on the instance itself, and the calibration got worse

The FlyWire leaderboard (read today) is **unchanged since 2026-01-09**, 12 entries, no methods
published. Recomputing our position: the H64 champion at 84.25818% ≈ 35,314,975 forward weight.
Entries still above it: 35,463,823 / 35,459,266 / 35,452,425 / 35,436,406 / 35,435,948 / 35,374,656
/ 35,364,447 — **seven**, the same seven that sat above H42 at 35,270,783. So the largest single
connectome move of the autonomous campaign, +0.104084 pp, **overtook nobody**. The nearest rung is
Hashorva 2024-10-08 at 84.37753%, +0.11935 pp away. L02 § 3's recommendation stands unchanged and
unactioned: an intermediate, independently-attested milestone would be better calibrated than the
84.6147 target whose provenance is still unread.

### 5.2 LP / SDP rounding for linear ordering is dry, and dry for a structural reason

This is axis (d) of the brief and it was searched properly. What exists: Maximum Acyclic Subgraph
admits a **2-approximation** (i.e. the trivial "random order or its reverse" bound), and *beating
the random ordering by any constant is UG-hard* (Guruswami, Håstad, Manokaran, Raghavendra,
Charikar, https://people.eecs.berkeley.edu/~venkatg/pubs/papers/mas.pdf). The LP-rounding results
that do better — e.g. `2√2 ≈ 2.828` (arXiv:1405.0456) — are for a **restricted** variant (RMAS), not
ours. Newman's SDP relaxation is studied for integrality gaps, not for practice.

**Verdict: nothing transfers, and it is not a search failure.** Every guarantee in this line is
~34 pp *below* where our champion already sits (84.26% vs a 50% random-ordering baseline), the LP
formulations need `Θ(n²)` ordering variables (1.87 × 10¹⁰ for us) or `Θ(m)` constraints plus
exponentially many cycle cuts, and `CLAUDE.md` already records the one MIP route as 20 days of
Gurobi. **Do not search this axis again.**

### 5.3 DAG structure learning: one real transfer, one clean rejection

The `tuck` (§ 3.1) is the transfer. The rejection is worth recording so nobody re-reads it: **SCOPE**
(*Relaxed Sparsest-Permutation Formulation for Causal Discovery at Scale*, arXiv:2605.05568v1,
**2026-05-07**, retrieved) scales to 10⁴ variables and looks superficially relevant, but it uses a
**single** data-driven ordering (an approximate-minimum-degree permutation of a precision matrix)
with **no permutation local search at all**, and its objective is Cholesky-factor sparsity of a
Gaussian precision matrix. Nothing about it touches a weighted arc objective. Likewise
**HiTOC** (Int. J. Data Sci. Anal., 2026) peels sinks layer by layer — that is our `H02` greedy-FAS,
rediscovered in another field.

### 5.4 Surrogate shapes: the closest literature is learning-to-rank, and it does not help

Our continuous objective `F(P) = Σ_e ŵ_e σ(β·Δ_e)` is *literally* a weighted pairwise **AUC
surrogate** with a sigmoid link and `P` as the score vector — a framing this campaign has never
written down. The literature on that object is large and I read into it: Gao & Zhou
(arXiv:1208.0645, IJCAI-15) prove that exponential and logistic pairwise surrogates are
**AUC-consistent** while the hinge is **not**.

**I am not transferring this, and the reason matters.** Consistency is a statement about minimising
*expected* risk over a hypothesis class as sample size → ∞. We optimise a *single fixed finite
instance* to a local optimum, and Q01's failure is not statistical inconsistency — it is that at
`β·std ≈ 148` the surrogate ranks the better *order* lower, a finite-instance geometry fact. A
consistency theorem says nothing about that. Quoting it would be exactly the kind of borrowed
authority this report is supposed to filter out.

The one usable output: **L02's negative is confirmed from a second direction.** Searching the LTR /
AUC literature rather than the FAS literature, I still found **no** work using a one-sided or
asymmetric pairwise surrogate the way H38/H64 do. That mechanism appears to be genuinely the
campaign's own. Given four open queue items already on that axis (H65, H66, H68, H69), I propose
nothing further there.

### 5.5 GPU-parallel permutation local search: real, but not a score axis

Found and read titles/abstracts for: **cuGenOpt** (arXiv:2603.19163, 2026, "one block evolves one
solution", supports permutation encodings), GPU B&B for permutation flow-shop on up to 384 V100s
(arXiv:2012.09511), GPU constraint-based local search (IEEE), CUDA parallel local search for
set-union knapsack (Knowl.-Based Syst. 2024). Every one of them is a **wall-clock** contribution at
instance sizes far below 136,648, and `CAMPAIGN.md` forbids starting Phase 2 unilaterally. L01's
§ 5 note stands: multi-threading the disjoint-move sweeps is the obvious lever *if and when*
runtime becomes the objective (P19), and the disjointness proofs to make it safe already exist.
**Not queued.**

### 5.6 Spin-glass / Potts annealing for MFAS: nothing

`lit/README.md` coverage gap #3 named this. Queries 3 and 9 surfaced nothing that applies a Potts /
spin-glass annealer to a weighted FAS or linear-ordering instance at any scale. The physics-flavoured
annealing literature for ordering problems is TSP/QAP-shaped. **Gap closed as empty.**

### 5.7 Extremal FAS theory is irrelevant to us

arXiv:2501.06935 (Gutin, Nielsen, Yeo, Zhou, 2025) and arXiv:2607.20996 bound `fas_w(D)` for
**oriented** graphs (no 2-cycles) with `Δ(D) ≤ 4` and girth conditions. Our instance has reciprocal
pairs everywhere and degrees in the thousands. No transfer, in either direction.

### 5.8 Ejection chains / variable-depth for LOP or FAS: **third** consecutive scan finding nothing

L01 searched it, L02 searched it, I searched it (queries 5, 6, 10). Nothing exists at any scale for
this problem family. **This is now a settled negative — stop looking.**

### 5.9 The un-queued leftovers from L02, deliberately not revived

L02 § 7 ranked five items and only three were queued. For the record: the **α-MOT certificate
sweep** (its rank 4) and **batched noising** (its rank 5) remain unqueued. I re-read both and
re-endorse the decision — α-MOT's own author estimated `expected_pp ≈ 0`, and noising's RNG cost
against P09's relabel gate is a structural handicap that nothing in this scan improves.

---

## 6. Ranked hypotheses — queue-ready

Bars, restated so no gate below is quoted out of context: `screen_delta_pp` connectome **0.012** /
microns **0.002**; the connectome **PROTOCOL CI** needs **0.02343 pp at n = 5** (H52's refusal);
since P09 the connectome promotion additionally needs a registered relabelling study; and **microns
is currently runtime-blocked (P19)** — the champion itself truncates 3/5 — so an item that *adds*
seconds on microns cannot pass a two-primary screen today. Under the operator's P15 ruling of
2026-08-27, promotion is per-dataset and the connectome leg is what matters, so all three items
below are filed as **connectome-first**.

**Ranking, and why.** H73 first because it is the cheapest decisive measurement in the queue
(minutes of CPU, no new mechanism, one branch in an existing kernel) and it has a *measured*
leverage prior. H71 second because it is the only genuinely new move class and its prototype needs
no GPU. H72 third because H60's kill is a strong prior against the whole axis.

### Rank 1 — **H73**: displacement-minimising tie-break in the exact-gain sift
### Rank 2 — **H71**: subset-bipartition ("tuck") interval repair
### Rank 3 — **H72**: minimum-crossing-weight split point in the SCC recursion

Full item bodies as appended to `queue.json` are not duplicated here; see
`autoresearch/queue.json` items `H71`, `H72`, `H73`. The mechanism, the derivation, the gate and the
conflict argument for each are in §§ 3 and 4 above.

---

## 7. Two things for the cycle that are not hypotheses

1. **The free 1.93× is *still* unclaimed, two scans later.** `select_disjoint_moves`
   (`segment.py:315`) packs disjoint moves greedily; the optimal weighted-interval-scheduling DP
   measured **1.93×** the greedy gain in `proto_H45_pair_connectome.json`. L02 flagged this as a pure
   code fix rather than a hypothesis and nothing has touched it. If H71 is built it will need exactly
   that packer, so the two should land together.
2. **`DEFAULT_SPLIT_FRACS` has never been sized.** Independently of H72's `argmin` cut, the campaign
   ships five fixed fractions cycled over 77 connectome alternation cycles and has no measurement of
   what a different set, or a different cycle length, is worth. H42 sized the *number* of cycles;
   nobody sized *what varies across them*. That is a sizing question in the same register as H42's
   and it is CPU-only. Not filed as a hypothesis because H72 subsumes the interesting half of it —
   but if H72 dies at its prototype, this is the cheap fallback measurement.
