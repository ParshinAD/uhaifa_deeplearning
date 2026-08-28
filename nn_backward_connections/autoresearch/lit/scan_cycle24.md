# Literature scan L02 — cycle 24 (2026-08-28), TARGETED

Scout run for queue item **L02**. Scoped by a measured barrier depth, not by a general "what is new
in FAS" brief. Dedupe base read in full before any search: `autoresearch/killed.json` (17 meta-rules,
21 kills), `autoresearch/queue.json` (all items, in particular H61 / H71 / H72 / H81),
`autoresearch/lit/scan_cycle18.md`, `autoresearch/lit/divergence-cycle21.md`,
`autoresearch/lit/divergence-cycle22.md`.

---

## 1. The scoped question, and the filter every candidate had to pass

**Question.** Which linear-ordering / FAS / permutation-search methods are designed to **cross a
barrier of known depth** rather than descend into a basin — and which of them do so at a cost per
improvement that does **not** scale with `n` the way an acceptance schedule's round count does?

**The two numbers that scope it.**

- **M15** (cycle 21, H78): the reference solution at 84.61467764201586 % sits behind a **0.786097 pp
  valley** from the H64 champion's basin (84.25817950936937 %). The prize is +0.356498 pp; the two
  orders disagree about the orientation of edges carrying **4.40 pp** of weight and **85 % of it
  cancels**. No partial adoption tested was profitable: rank-blend runs through the valley,
  teleport-top-k loses at k = 1, slot-preserving adoption's best proper subset is **exactly
  +0.000000 pp**. M15 states its own scope limit: *"two geometric families of partial adoption were
  tested, not all of them"*.
- **M17** (cycle 23, H80): barrier crossing is bought in **rounds** and rounds are priced in `n`. The
  measured round rate for destroy-and-recreate + repair sift is 1,106.99/s at n = 148 → 173.60/s at
  n = 400 → 15.91/s at n = 4,000 → **0.1467/s at n = 136,648**. n grows 342× from the n = 400
  synthetic; the round rate falls **1,183×**. On the connectome **zero** uphill moves were ever
  accepted and 98.5–100 % of accepted moves were exact ties. Deepest excursion achieved: 4.32 % of
  the barrier at n = 400, 0.10 % at n = 4,000, **0.00 %** on the connectome.

**The filter I applied to every source.** A mechanism is *alive* here only if it delivers a **large,
structured jump per O(1) expensive operations**, where "expensive operation" means at most a small
constant number of O(m) or O(m log n) passes over the graph — 5,657,719 edges, 136,648 nodes. A
mechanism whose cost per barrier crossing is O(rounds), or whose jump size shrinks as the round
budget shrinks, is **already dead** and I did not queue it, however good the paper.

Two hard constraints I did not relax: **M1** closes the continuous/gradient axis and LP/SDP rounding
for linear ordering was not searched (it is a dead axis by explicit instruction); and nothing I
propose reads `data/best_solution` or the target metric *inside an algorithm* — where the reference
appears below it is a **diagnostic**, read only through `mfas.analysis.gap.load_best_solution`,
exactly as H78 read it.

---

## 2. Search log — what I issued and what actually came back

### 2.1 Queries issued (WebSearch), 2026-08-28

1. `partition crossover tunneling between local optima traveling salesman Whitley`
2. `optimal recombination problem permutation combinatorial optimization Eremeev Kovalenko complexity`
3. `partition crossover gray box optimization separable subfunctions "linear ordering problem" OR "feedback arc set" recombination`
4. `"linear ordering problem" partition crossover hill climber permutation gray-box "total weighted tardiness"`
5. `Chicano Whitley 2025 gray-box permutation problems experiments partition crossover linear ordering components`
6. `memetic algorithm linear ordering problem crossover path relinking large instances 2024 2025`
7. `Lugo Segura Miranda "diversity-aware memetic algorithm" linear ordering problem xLOLIB2 crossover operator PDF`
8. `"linear ordering problem" best known solutions xLOLIB2 instance size n=1000 insertion neighborhood local search time`
9. `"iterative partial transcription" tour merging two tours common subchains post-optimization LKH`
10. `feedback arc set memetic evolutionary crossover recombination large graphs 2025 2026 heuristic million edges`
11. `"solution merging" OR "backbone crossover" combinatorial optimization ordering problems 2025 offspring at least as good as parents`
12. `Kemeny rank aggregation extended Condorcet criterion decomposition partition exact subproblems large scale`
13. `iterated greedy strategic oscillation path relinking linear ordering problem new best known values Duarte Marti`
14. `"path relinking" "linear ordering problem" insertion move guiding solution intermediate solutions best local search Duarte Laguna Marti pdf`

### 2.2 Retrieval status of every source I cite

| source | status |
|---|---|
| arXiv:2407.06742 (Chicano, Whitley, Ochoa, Tinós — gray-box operators, PPSN 2024) | **FULL TEXT**, HTML, retrieved **twice independently** (`arxiv.org/html/2407.06742v1` and `ar5iv.labs.arxiv.org/html/2407.06742`); the two reads agree on every load-bearing claim |
| arXiv:cond-mat/9902034 (Möbius, Freisleben, Schnabel, Schneider — Iterative Partial Transcription, Phys. Rev. E 59, 1999) | **FULL TEXT**, HTML |
| arXiv:2605.31051v1 (Fagiolo, Baioletti, Santucci — *Linear Ordering Problem: Time for a Change*, 2026-05-29) | **FULL TEXT**, HTML |
| arXiv:2405.08285 (Segura et al. — *Future Trends in the Design of Memetic Algorithms: the Case of the LOP*) | **FULL TEXT**, HTML |
| arXiv:2506.15097 (Kemeny space reduction by optimized majority rules, 2025-06) | **FULL TEXT**, HTML |
| arXiv:2201.03893 (Heuristic search for rank aggregation with application to label ranking) | **PARTIAL** — PDF fetched and parsed by the summariser; I treat its crossover description as ABSTRACT-LEVEL, not quotable |
| GECCO 2009 "Tunneling between optima: partition crossover for the TSP" (Whitley, Hains, Howe) | **FAILED** — `cs.colostate.edu/sched/pubs/gecco09tunnel.pdf` fetched but the PDF text is not extractable in this environment (same poppler-missing failure as three previous scans). Claims about it below come from indexed abstracts, and I mark them as such |
| Evolutionary Computation 28(2):255 (2020), GPX2 (Tinós, Whitley, Ochoa) | **ABSTRACT ONLY** (MIT Press / Stirling repository listing) |
| FlyWire MFAS leaderboard `codex.flywire.ai/app/mfas_challenge?dataset=fafb` | **FULL TEXT** — 12 rows, top entry still Vahidi & Koutis 35,463,823 dated **2026-01-09**, i.e. **unchanged since scan_cycle18** |
| SSRN 6221201 (Vahidi & Koutis 2026 — the paper behind our exact target) | **FAILED, HTTP 403.** This is the **fifth** documented failure across four scans |
| Springer `10.1007/s12293-022-00378-5` (diversity-aware memetic LOP) | **FAILED**, HTTP 303 → `idp.springer.com` auth wall. Used the arXiv preprint id (2106.02696) instead, whose *abstract page* carried no method detail |
| Springer `10.1007/s00521-025-11171-z` (Future trends, NCA 2025) | **FAILED**, HTTP 303 auth wall; read the arXiv version 2405.08285 instead |
| ScienceDirect `S030505482300028X` (Approximate Condorcet Partitioning) | **FAILED**, HTTP 403 |
| academia.edu / uv.es copies of "Revised GRASP with path-relinking for the LOP" | **FAILED** — 403 and an unextractable PDF respectively |

---

## 3. Sources read — mechanism in our vocabulary, and cost per barrier crossing

### 3.1 Partition crossover for the **Linear Ordering Problem** — the find of this scan

> **Chicano, Whitley, Ochoa, Tinós.** *Generalizing and Unifying Gray-box Combinatorial Optimization
> Operators.* arXiv:2407.06742, 2024-07-09; PPSN XVIII (LNCS 15148), 2024.
> <https://arxiv.org/abs/2407.06742> — **FULL TEXT retrieved (HTML), twice.**

**Mechanism, in our vocabulary.** Take two orders `A` and `B` over the same 136,648 nodes. Look at
the map from A-rank to B-rank, `q = rank_B ∘ rank_A⁻¹` (the paper writes it `σ₂·σ₁⁻¹` and calls the
step "finding permutations of consecutive elements"). Scan positions left to right and cut wherever
the running maximum of `q` equals the current index: that is exactly a position `p` at which the two
orders hold the **same set** of nodes in the first `p` slots. Those cuts split the line into `k`
**common blocks** — same node set, same position range, different internal order. The paper's
decomposition property is *"re-arranging sets of consecutive elements in the permutation does not
affect the contributions to the fitness function of the elements below or above the set"*, with the
non-interaction condition `[min(e(h₁)),max(e(h₁))] ∩ [min(e(h₂)),max(e(h₂))] = ∅`. Corollary 1 then
says the operator *"necessarily finds the best solution from the set of 2^m solutions that can be
generated by applying or not each of the m components"*.

**Why it is exact for our objective, derived here rather than taken on trust.** Our score is
`Σ_edges w(u,v)·1[pos(u) < pos(v)]`. Let `S_i` be common block `i`, occupying the same position
interval in both parents. For an edge with `u ∈ S_i`, `v ∈ S_j`, `i < j`: `u` precedes `v` in **every**
offspring, whatever internal orders are chosen, because block membership and block order are shared.
Same for `i > j`, always backward. Only `i = j` edges depend on a choice, and they depend on
**block `i`'s choice alone**. So

```
score(offspring) = C + Σ_i f_i(c_i),   c_i ∈ {A, B}
```

with `C` the constant cross-block term. Picking `argmax` per block is the best of `2^k`, in `k`
independent decisions, from **one pass over the edge list**. The offspring is `≥ max(score(A),
score(B))` **by construction** — both parents are members of the `2^k` set.

**Cost per barrier crossing.** `O(n)` for the block scan (a running-max over 136,648 positions,
milliseconds) plus `O(m)` for the per-block intra-weight tally (5,657,719 edges, seconds in numpy).
**One pass. No rounds. No temperature. No hyperparameter.** The jump size is not bounded by the
budget: the offspring can differ from both parents on up to `n` positions at once.

**Does the cost scale with n the way M17's rounds do?** No, and this is the whole point. M17's
arithmetic prices a crossing at `exp(D/(c·δ_med))` rounds against a round rate that falls 1,183×
from n = 400 to n = 136,648. A partition crossover pays **one** O(m) pass regardless of `D`, and it
never goes uphill at all, so the "number of accepted uphill moves needed" — M17's question (b) — is
**zero**.

**What the paper does NOT give us, stated plainly.** It is a **theoretical** contribution: *no
experimental results and no instance sizes*. The number of common blocks `k` between two good LOP
solutions is **unmeasured in the literature at any n**. That is precisely the free measurement H82's
rung 1 makes below, and it is the item's main risk: if `k = 1` for our order pairs, exact PX is a
no-op.

### 3.2 Iterative Partial Transcription — the same operator, 1999, and the honest complexity

> **Möbius, Freisleben, Schnabel, Schneider.** *Combinatorial Optimization by Iterative Partial
> Transcription.* arXiv:cond-mat/9902034 (Phys. Rev. E 59, 1999).
> <https://arxiv.org/html/cond-mat/9902034> — **FULL TEXT retrieved (HTML).**

**Mechanism.** Given two solution vectors differing in `k` components, search for decompositions of
the transformation mapping one to the other into **commuting partial transformations** `M_α`, `M_β`
that *"modify disjunct sets of components"*, then copy whichever partial transformation improves.
For the TSP the instantiation is: find subchains of the two tours that *"include the same cities in
a different order, and have the same initial and final cities"* — the tour analogue of our common
blocks. The authors note IPT is applied as a **post-optimisation on pairs of local minima**, and
that combining it with local search ("IPTLS") is the productive form.

**Cost.** *"requires a computational effort which is roughly proportional to N²"* in their
implementation, with small constants. **This matters for us and I will not paper over it:** the
naive IPT search over all `(start, end)` subchain pairs is quadratic. Our linear-ordering variant is
**not** — because in an ordering there is no cyclic freedom, a common block must occupy the *same
position interval* in both parents, and the finest such decomposition is found by a single running-max
scan in `O(n)`. The quadratic cost in IPT comes from the TSP's rotational/reflectional symmetry,
which our problem does not have. (A GECCO 2021 paper "An efficient implementation of iterative
partial transcription for the TSP" exists — **ACM DL, not retrieved** — reducing that constant.)

**Corroboration value.** Two independent literatures (statistical physics 1999, gray-box EC 2024)
converge on the same operator, and LKH ships a tour-merging procedure built on it. This is not a
one-paper idea.

### 3.3 Partition crossover as an explicit *tunnelling* operator

> **Whitley, Hains, Howe.** *Tunneling between optima: partition crossover for the TSP.* GECCO 2009,
> doi:10.1145/1569901.1570026 — **FAILED to extract full text** (PDF unreadable in this environment).
> **Tinós, Whitley, Ochoa.** *A New Generalized Partition Crossover for the TSP: Tunneling between
> Local Optima.* Evolutionary Computation 28(2):255–288, 2020 — **ABSTRACT ONLY.**

From the retrievable abstracts and repository listings: when two local optima are recombined, the
offspring *"are highly likely to also be local optima"*, letting the operator *"jump or tunnel from
two local optima to two new and distinct local optima **without searching intermediate
solutions**"*; PX returns *"the best of 2^k reachable offspring, where k is the number of recombining
components"*; **GPX2 has O(n) runtime** and is reported to improve results on instances up to
n = 100,000. For pseudo-Boolean/NK landscapes (FOGA 2015, abstract) locally optimal parents produce
offspring locally optimal in the full space *"more than 80 percent of the time"*.

**Why the italicised clause is the load-bearing sentence for this campaign.** Every barrier-crossing
mechanism the campaign has tried — H80's Metropolis schedule, H78's rank-blend path, H78's teleport
path — pays for the **interior** of the valley. PX does not visit the interior at all. That is the
only structural answer to M17 I found in the whole scan.

### 3.4 The state of the art in LOP metaheuristics — and why it does not port

> **Segura et al.** *Future Trends in the Design of Memetic Algorithms: the Case of the LOP.*
> arXiv:2405.08285 — **FULL TEXT (HTML).**
> **Lugo, Segura, Miranda.** *A Diversity-Aware Memetic Algorithm for the LOP.* Memetic Computing
> 14(4):395–409, 2022; arXiv:2106.02696 — **Springer FAILED (auth wall), arXiv abstract only.**

MA-EDM, the strongest published LOP memetic algorithm, uses **Cycle Crossover (CX)** as its
recombination, an explicit-diversity-management replacement rule ("Best Non-Penalized", using
Spearman's footrule distance), and insertion-neighbourhood local search. Instances: **xLOLIB2,
n = 300…1,000**; local search cost is stated as `O(n²)` per local-optimum verification with
incremental evaluation, *"making it the primary bottleneck"*.

**Two things extracted.**

1. **CX is not PX and is strictly weaker for our purposes.** CX partitions positions into the cycles
   of `q = rank_B ∘ rank_A⁻¹` and takes each cycle wholly from one parent. That produces a valid
   permutation but the objective does **not** decompose over cycles: a pair with one element in cycle
   `C₁` and the other in `C₂` has both elements moved, so the cross terms are not constant. PX's
   components are exactly the cycle-unions that are also **position intervals** — which is what buys
   exactness. The published state of the art in LOP is using the operator that does *not* have the
   decomposition property. That is a gap, not a precedent against.
2. **Scale.** The whole LOP metaheuristic literature lives at `n ≤ 1,000` and assumes a dense `n×n`
   matrix. Our instance is `n = 136,648` at density ≈ 3·10⁻⁴. **Nothing in that literature ports as
   an algorithm**; only the operator ports, and only because our version of it is `O(n + m)` rather
   than `O(n²)`.

### 3.5 Independent confirmation of M15's degeneracy claim (2026)

> **Fagiolo, Baioletti, Santucci.** *Linear Ordering Problem: Time for a Change.* arXiv:2605.31051v1,
> **2026-05-29** — **FULL TEXT (HTML).**

Argues LOP research should move to (a) modern EXIOBASE instances and (b) a **Multi-Solution LOP**
formulation, because *"LOP instances often exhibit many distinct global optima that differ
substantially from one another"*, and reports that **"the number of global optima increases
exponentially as sparsity grows"**. Instance sizes n = 10, 49, 89–176, 300. Diversity measured with
Kendall-τ nearest-neighbour and Solow–Polasky.

**Transfer: not a mechanism, a *justification*.** This is external, independent, 2026 confirmation of
what H78 measured on our own instance — the champion and the reference are 0.357 pp apart in score
and structurally far apart (median node displacement 6,051 ranks, 85 % of their disagreement
cancelling). Our graph is extreme on exactly the axis the paper identifies (sparsity), so a large
population of mutually distant near-optima is the *expected* structure, not an anomaly. It is also
the strongest single argument for H82: PX is the one operator whose gain **grows with how far apart
equally-good parents are**, whereas every multi-start design the campaign has killed (M3, M9, M10,
M16) harvests only `max` over the parents and is therefore worth exactly nothing when the parents
score the same. PX profits from the case that kills best-of-R.

### 3.6 Rank aggregation / Kemeny — checked, and it lands on ground we already own

> **arXiv:2506.15097** (2025-06), Kemeny space reduction by optimized majority rules — **FULL TEXT.**
> **Akbari & Escobedo**, *Approximate Condorcet Partitioning*, C&OR 2023 — **FAILED, 403.**

The Kemeny literature's decomposition theorem — *"every median respects such a partition"* when
`δ_xy > 0` for all `x` in block `i`, `y` in block `j`, `i < j` — is, in our vocabulary, exactly the
**net-digraph condensation**: blocks such that all net pairwise weight runs forward. That is **H60**,
built, measured and killed at +0.007311 pp (1.64× short of the bar), with the follow-on cycle count
measured and shown not to close the gap. The αMOT / αMOTe pair-fixing certificates in 2506.15097 are
the same family that scan_cycle18 § 5.9 declined and re-endorsed declining; the paper's own scale is
`n ≤ 240`, and its `M`-finest Condorcet partitioning is `O(n²)`. **Nothing new here. Not queued.**

### 3.7 Multilevel memetic partitioning — the "agreement coarsening" crossover

> **Andre, Schulz, Sanders** and successors: *Memetic Multilevel Hypergraph Partitioning*
> (arXiv:1710.01968), *Multilevel Memetic Hypergraph Partitioning with Greedy Recombination*
> (arXiv:2204.03730) — **titles/abstracts only, not opened in full.**

The recombination there is: contract everything two parent partitions **agree** on, refine the
coarser instance, and the offspring is guaranteed no worse than the better parent. It scales to
instances far larger than the LOP literature's. The MFAS analogue would be to contract maximal runs
of nodes that two orders both place consecutively and in the same relative order, then re-run the
refiner on the coarsened line.

**Declined, and the reason is a measured one.** That is a **coarse block permutation** move, and
**H46** measured the achievable gain of coarse block permutation at 17,081–68,324 nodes as **exactly
+0.000000 pp**. Agreement-derived blocks are a different partition from H46's contiguous tiling, but
the move class is the same one, the kill is recent, and the machinery is expensive. PX gets the same
"offspring ≥ better parent" guarantee for `O(n + m)` and no new refiner, so it dominates on cost.
Revisit only if H82 shows `k` is large and the per-block interiors are where the gain sits.

---

## 4. What this scan says about the already-queued **H71 / H72 / H61**

### H71 (subset-bipartition "tuck") — **unchanged, keep at priority 2**

Nothing retrieved this scan touches it either way. GRaSP's `tuck` remains the only genuinely new
*monotone* move class in the queue. One thing to note honestly: **M15 now bears on it directly.**
H71 is a new monotone move class on the existing basin, and M15's operative directive says such an
item *"must now argue against M15 before it is scheduled"*. H71's own predicted midpoint (+0.008 pp)
is already below the 0.012 pp screen bar and far below the 0.02343 pp PROTOCOL CI floor, and M15
plus M17 together now say the residual is not reachable by monotone local classes at all. I would
**not** re-rank it up; its value is the diagnostic ("is the optimal γ ever a strict non-prefix?"),
which is real and cheap, not the pp.

### H72 (minimum-crossing-weight split in the SCC recursion) — **STRENGTHENED, and it gains a second candidate cut rule**

The recursion's cut point is exactly the object PX's block scan computes. Two additions:

1. **A new, free cut rule.** A position `p` where two independently produced good orders hold the
   **same prefix set** is a cut that *both* solutions agree on. Its symmetric difference
   `|prefix_A(p) Δ prefix_B(p)|` is computable for all `p` in one `O(n)` running scan — the same scan
   H82 needs — and gives a *consensus-guided* split criterion alongside H72's `argmin X(mid)`
   crossing-weight criterion. Where the two criteria agree, the cut is doubly justified; where they
   disagree, the disagreement is itself a measurement worth having.
2. **A literature-side note that cuts against the axis, recorded for balance.** The Kemeny XCC
   decomposition (§ 3.6) is the *provably optimality-preserving* version of "cut the line", and this
   campaign already built and killed it (H60, +0.0073 pp). A merely *good* cut is a weaker object
   than a provably safe one. So H72's honest prior is still H60's, and I would **not** raise it above
   priority 3 — but I would attach the consensus-cut rule to it as a second arm, because it costs
   one extra `O(n)` scan and it is the only cut criterion in the queue that uses information from
   outside a single trajectory.

### H61 (k-node joint optimal re-insertion, k = 3…6) — **WEAKENED, re-prioritise DOWN**

Three reasons from this scan, all of them mine rather than the paper's:

1. **M15 + M17 now bracket it from both sides.** H61 is a monotone move class of bounded arity `k`,
   applied inside the champion's basin. M15 says the residual is a property of the whole permutation
   and no partial adoption of the destination is profitable; M17 says the compound-move family is
   priced in rounds. H61's own kill condition already anticipates dying on redundancy (its
   `expected_pp` note says so explicitly), and both the k = 1 and k = 2 rungs of the same ladder are
   now measured: the sift is a fixed point, and H52's pair relocation realises +0.02233 pp fully
   converged and **still cannot clear the 0.02343 pp PROTOCOL CI floor**. A k = 3 rung is a *smaller*
   increment on top of that.
2. **Its cost per improvement scales the wrong way.** The DP is `O((Σ deg) · 2^k · k)` per candidate
   *tuple*, and the number of candidate tuples is what has to be swept — i.e. it is a rounds-priced
   mechanism in exactly M17's sense, and the round rate on the connectome is the thing that collapsed
   1,183×.
3. **H40's kill folds into it and makes it more expensive, not less.** H40's revival condition says
   "if H61 is built, H40 should be folded into it rather than revived on its own" — i.e. H61 now
   carries a second, already-falsified mechanism's repair burden.

**Recommendation: leave H61 queued but move it below H72, and require it to state, per M17, its round
cost at n = 136,648 before it is scheduled.** Nothing in the 2024–2026 literature strengthens it; the
`insert distance r` framework it cites (arXiv:2204.02902) is unchanged and still has no large-scale
implementation.

---

## 5. New hypotheses — 2, not more

Bars restated so nothing below is quoted out of context: connectome `screen_delta_pp` **0.012 pp**;
the connectome **PROTOCOL CI** needs **0.02343 pp at n = 5**; promotion additionally needs a
registered relabelling study (P09); microns is runtime-blocked (P19), so a connectome-first item is
the only shape that can move today. Stage-4 marginal rate for M8's allocation clause: **3.45e-4 pp/s**
on connectome. Champion wall clock ≈ **1,185 s** of a 3,600 s cap.

---

### H82 — Exact block-partition crossover over the campaign's own archive of stored orders

- **id**: `H82`
- **title**: Exact block-partition crossover (PX / IPT) between two independently produced orders — best of `2^k` in one `O(n + m)` pass
- **status**: `proposed`
- **priority**: 1
- **kind**: science
- **axis**: recombination / solution merging (a move class the campaign has never had)

**hypothesis.** For two orders `A`, `B` over the same node set, scan positions left to right and cut
at every `p` where `max(rank_B[A_order[0:p]]) == p-1`; the cuts define `k` **common blocks** holding
the same node set in the same position interval in both orders. Because block membership and block
order are shared, every cross-block edge is oriented identically in every offspring, so the exact
score decomposes as `C + Σ_i f_i(c_i)` with `c_i ∈ {A, B}`, and taking the per-block argmax realises
the **best of `2^k`** offspring — which is `≥ max(score(A), score(B))` by construction — in one
`O(n)` scan plus one `O(m)` edge pass. **THE CLAIM: over the connectome orders already stored in
`results/*_positions.npy`, the best offspring beats the H64 champion by more than +0.012 pp, and that
advantage survives a terminal repair sift (a pure append, so M12's cycle-14 amendment makes the
prototype number a genuine prediction rather than a screening number).**

**rationale.** The operator is Corollary 1 of Chicano–Whitley–Ochoa–Tinós, arXiv:2407.06742 (PPSN
2024, FULL TEXT retrieved): the procedure *"necessarily finds the best solution from the set of 2^m
solutions"*, and the paper's decomposition property for the LOP is stated as *"re-arranging sets of
consecutive elements in the permutation does not affect the contributions to the fitness function of
the elements below or above the set"*. It is the same operator as Möbius et al.'s Iterative Partial
Transcription (cond-mat/9902034, FULL TEXT), which LKH ships as tour merging. Whitley et al. describe
its purpose as tunnelling from two local optima to a new local optimum *"without searching
intermediate solutions"*. **That clause is the entire reason this item exists**: H78's rank-blend and
teleport probes and H80's Metropolis schedule all pay for the interior of the 0.786 pp valley; PX
never enters it. And arXiv:2605.31051 (2026-05-29, FULL TEXT) reports that for the LOP *"the number
of global optima increases exponentially as sparsity grows"* — our instance has density 3·10⁻⁴, so a
population of mutually distant, nearly equal-scoring orders is the expected structure, and PX is the
only operator whose yield **grows** with that distance while best-of-R's yield goes to zero.

**method (three rungs, CPU only, ZERO GPU).**
- *Rung 0 — free pre-gate, minutes.* For every pair among the stored connectome position vectors —
  H64 (champion), H79A / H79B / H79C (three structurally distinct constructions, Kendall τ
  0.0899 / 0.3706 / 0.4329 from the champion's construction), plus older H42 / H36 / H35 orders —
  compute `k`, the block-size distribution, and the exact `f_i(A)`, `f_i(B)` per block. Report the
  realised best-of-`2^k` gain over `max(parents)`. **If `k = 1` for every pair, the item is dead for
  the cost of one scan.**
- *Rung 0b — DIAGNOSTIC ONLY, never shippable.* Run the same census with `data/best_solution` as one
  parent, read solely through `mfas.analysis.gap.load_best_solution`. This measures how much of
  M15's +0.356498 pp residual is **block-decomposable** — a **fourth** geometric family of partial
  adoption, and the only one that is monotone-safe by construction. It answers M15's own stated scope
  limit. **This arm may never enter a scored variant and must be labelled in the artifact.**
- *Rung 1 — validity.* Cross-check ≥ 25 offspring against the frozen scorer with **zero mismatches**
  before quoting any total (H45's prototype scored 1/18 on its first run by aggregating the wrong
  events — do not repeat that). Repeat the whole census on mouse and microns as scale controls.
- *Rung 2 — composition.* Take the best offspring, apply the champion's stage-4 short repair sift as
  a **terminal append**, and report the composed score, the M8 redundancy fraction against stage 4,
  and pp/s. The shippable variant is `H64 run` + one **cheap** second parent (the second parent need
  not be good — PX is monotone, so a stage-3-only or Rocket-free order is safe) + PX + repair; report
  its wall clock against the 3,600 s cap.

**kill_condition.** ANY of four. (1) Rung 0: the best pair over the whole stored archive has `k = 1`,
or its realised best-of-`2^k` gain over `max(parents)` is < 0.012 pp on connectome. (2) Rung 1: any
mismatch against the frozen scorer. (3) Rung 2: the composed connectome delta over the H64 champion
is < 0.012 pp. (4) M8: realised pp/s below 3.45e-4 pp/s once the second parent's wall clock is
charged. Report `k`, the block-size distribution, the per-block win split (how many blocks each
parent wins) and the redundancy fraction **whatever the verdict** — the campaign has never looked at
the common-block structure of its own solutions and that artifact is worth having on a kill.

**novelty — strict, entry by entry.**
- **M3 / M9 / M10 / M16, H01 / H56 / H57 / H79 (the multi-start family).** All of these price a
  design as `mean + σ·a_R`, i.e. they harvest `max` over the parents. PX produces a point **outside
  the parent set**; its gain is `Σ_i max(f_i(A), f_i(B)) − max(Σ_i f_i(A), Σ_i f_i(B))`, which is
  ≥ 0 always, is **independent of the parents' score spread**, and grows with `k`. M16's headline
  finding — four constructions contract to a 0.005879 pp final spread — is a statement that
  best-of-basins gains nothing; it is *not* a statement that the four final orders are the same
  permutation, and PX's yield depends only on the latter. M9's σ-bar (0.021269 pp) and M10's
  prefix/tail arithmetic simply do not apply: no `R`, no `a_R`, no draws.
- **M15.** This is a fifth probe of partial adoption and M15 explicitly scopes itself to the two
  families tested. Unlike all three of H78's probes, PX **cannot lose** — the parents are members of
  the `2^k` set. So the rung-0b number is a clean addition to M15 in either direction.
- **M17.** Cost per crossing: **one** `O(n + m)` pass, not `O(rounds)`. M17's two mandatory
  pre-scheduling questions are answerable now: (a) round cost at n = 136,648 is **1 round of a few
  seconds**; (b) accepted uphill moves required: **zero**, the operator is monotone.
- **H41 (segment/block moves, killed at 91.9 % redundancy)** is the closest relative and it is a
  different move. H41 *relocates* a rigid contiguous block to another position, which single-node
  insertion largely subsumes. PX **moves nothing**: each block stays in its own position interval and
  its **interior is re-permuted wholesale** according to another solution's global opinion — a
  destination no sequence of improving single-node insertions can reach, because the intermediate
  states are worse. Redundancy must still be reported per M8.
- **H46 (coarse multilevel k-block permutation, exactly +0.000000 pp)** permutes coarse blocks
  *relative to each other*; PX holds block order fixed. Distinct, but H46 is the standing warning
  that coarse-scale structure has been barren, and it is why I put a free `k = 1` pre-gate first.
- **H47 (exact leaf DP, +0.004342 pp)** optimises small windows exactly with no external
  information; PX imports a second solution's opinion at arbitrary block size. Distinct.
- **M11.** PX has the unusual property that the census statistic (`k`, block sizes) is a
  **capacity** number *and* the best-of-`2^k` value is **exactly realised**, not a ceiling — the
  argmax is computed, not bounded. Report both anyway; the capacity/achievability gap reappears at
  the composed stage, where M14's absorber lives, which is what rung 2 exists to measure.
- **M7 (expensive warm-starts lose)** — PX buys no embedding and no eigensolve; the second parent can
  be arbitrarily cheap.

**est_cost.** Rungs 0/0b/1: **minutes to ~1 h of CPU, ZERO GPU** (loads stored `.npy` vectors, one
running-max scan and one edge pass per pair). Rung 2: one ~1,185 s connectome pipeline run for the
cheap second parent plus seconds of PX. Implementation if it clears: one module
`src/mfas/refine/px.py` + `src/mfas/experiments/H82.py`. Deterministic, no RNG → 1-seed screen on the
primaries. It **adds** the second parent's seconds, which microns cannot afford under P19 → plan for
a connectome-only result.

**predicted_effect_pp / expected_pp.** connectome **0 to +0.10, midpoint +0.006** — below the 0.012 pp
bar in the median case, and well below the 0.02343 pp PROTOCOL CI floor, stated up front. The
distribution is strongly right-skewed: the gain is roughly linear in `k`, and `k` is completely
unmeasured — the literature has never reported it for a permutation problem at any scale. Against it:
two orders whose median node displacement is thousands of ranks may share **no** proper cut point at
all, in which case the item is worth exactly zero and dies in minutes. For it: it costs almost
nothing to find out, it cannot regress, and it is the only mechanism this scan found whose cost per
barrier crossing is `O(1)` passes.

**prior_evidence.**
- arXiv:2407.06742 § LOP + Corollary 1 (FULL TEXT, HTML, retrieved twice)
- arXiv:cond-mat/9902034 § II.1–II.2 (FULL TEXT)
- arXiv:2605.31051v1 (2026-05-29, FULL TEXT) — exponential growth of the number of global optima with sparsity
- Whitley/Hains/Howe GECCO 2009 and Tinós/Whitley/Ochoa EvCo 28(2) 2020 — ABSTRACT ONLY, cited only for the word "tunnelling" and the `2^k` claim
- `experiments/outputs/proto_H79_rung1.json` (Kendall τ of the four constructions), `proto_H79_rung2.json`, and the 22 stored H79 connectome position vectors
- `autoresearch/killed.json` M9, M10, M15, M16, M17, H41, H46, H47

---

### H83 — Forced-cut partition crossover: buy extra blocks by paying a bounded, exactly-scored repair

- **id**: `H83`
- **title**: Forced-cut PX — create common blocks where the two orders nearly agree, at a bounded and exactly measured repair cost
- **status**: `proposed`
- **priority**: 4 (**conditional**: schedule only if H82's rung 0 returns `k` small but the near-cut profile is rich; if H82's rung 0 already returns large `k` and a gain, this is redundant and should be closed)
- **kind**: science
- **axis**: recombination / solution merging

**hypothesis.** Exact PX needs `prefix_A(p) == prefix_B(p)` as sets. Define the **near-cut profile**
`D(p) = |prefix_A(p) Δ prefix_B(p)| / 2`, computable for all `p` in one `O(n)` pass. Pick a set of
positions with small `D(p)` and **force** cuts there by relocating, in parent `B` only, the `D(p)`
discordant nodes across the boundary — producing `B'`, which is exactly scoreable and shares those
cut points with `A` by construction. PX between `A` and `B'` is then exact again, gives best of
`2^k'` with `k' ≫ k`, and — because `A` is still a member of the offspring set — **cannot score below
`A`**, whatever the repair cost to `B`. **THE CLAIM: on the connectome, choosing the cut set that
maximises `k'` subject to `Σ D(p) ≤ 2,000` relocated nodes yields a best offspring more than
+0.012 pp above the H64 champion.**

**rationale.** This is the linear-ordering analogue of what GPX2 does for the TSP relative to GPX —
*"GPX2 finds more recombining components than GPX or IPT"* (EvCo 28(2), abstract only) — and of IPT's
own framing, which asks only that the two partial transformations *"modify disjunct sets of
components"* (cond-mat/9902034, FULL TEXT). The exact-cut requirement is brittle in a way the
decomposition property is not: the property needs the two parents to **agree on the set** at the
boundary, and one displaced node destroys a cut that is otherwise perfect. Forcing it back is a
local, exactly-priced repair. The monotonicity guarantee survives the repair intact, which is what
makes this safe to run at all: the worst case is a no-op.

**method (CPU only, ZERO GPU).** Rung 1, free: emit the `D(p)` profile for the best parent pairs from
H82's census — the histogram of `min_p D(p)` over windows tells us immediately whether near-cuts
exist. Pre-gate: **if fewer than 50 positions have `D(p) ≤ 32`, the item dies at zero cost.** Rung 2:
choose cuts by a simple DP over positions maximising block count subject to a total-relocation
budget; build `B'`; run exact PX; verify ≥ 25 offspring against the frozen scorer with zero
mismatches; report realised gain, `k'`, the relocation count actually spent, and the sensitivity of
the gain to the budget (100 / 500 / 2,000 / 10,000 relocated nodes). Rung 3: terminal repair sift and
the composed delta, exactly as H82 rung 2.

**kill_condition.** ANY of three. (1) Pre-gate: fewer than 50 positions with `D(p) ≤ 32`. (2) The
best forced-cut offspring is < 0.012 pp above `max(parents)` on connectome at every relocation
budget tested. (3) The composed delta after the terminal repair sift is < 0.012 pp. Also report, per
M11, the ratio of the realised gain to the `Σ_i max(f_i)` capacity — this is the one place in the PX
family where a ceiling and a realisation genuinely differ, because the forced relocations are a real
cost.

**novelty.** Distinct from H82 in that H82 is exact and takes what is there; H83 **manufactures**
decomposition and pays for it, and can therefore be non-empty exactly when H82 is empty. Distinct from
H31 / H40 / H80: no acceptance rule, no temperature, no rounds, and the "damage" (the relocation of
`D(p)` nodes) is *paid inside a single exactly-scored construction*, not accepted as a worse state to
be climbed out of — M17 does not bind, because the number of accepted uphill moves is still **zero**
and the number of rounds is still **one**. Distinct from H41: again, nothing is relocated to improve
the score directly; relocation is a boundary-alignment step whose only purpose is to restore
separability. Distinct from H45/H52 (pair relocation) and H61 (k-node joint re-insertion): those
optimise a small set of nodes exactly; this re-permutes whole intervals using a second solution.

**est_cost.** Rung 1 minutes, rung 2 tens of minutes of CPU, ZERO GPU. Implementation if it clears:
one function added to the same `px.py` module + `src/mfas/experiments/H83.py`.

**predicted_effect_pp / expected_pp.** connectome **0 to +0.06, midpoint +0.004**. Lower midpoint than
H82 because the repair cost is real and unbounded a priori, and because the whole family shares
H82's central risk (the two orders may simply not agree anywhere). Its value is that it converts
H82's binary outcome (`k = 1` → nothing) into a tunable one.

**prior_evidence.** arXiv:cond-mat/9902034 § II.1; EvCo 28(2):255 (2020) GPX2, ABSTRACT ONLY;
whatever H82's rung 0 emits; `autoresearch/killed.json` M11, M17, H41, H80.

---

## 6. What I could not find, and what I deliberately declined

**Declined, with reasons — not queued.**

1. **Path relinking with intermediate acceptance.** The brief asked for it specifically. I could not
   retrieve a primary source for its LOP instantiation: "Revised GRASP with path-relinking for the
   LOP" (J. Comb. Optim., Duarte–Laguna–Martí) failed at academia.edu (403) and as an unextractable
   PDF at uv.es; the "87 new best values" result attached to iterated-greedy + PR is for the LOP
   **with cumulative costs**, a different objective. More importantly, PR is *dominated on this
   campaign's own filter*: it evaluates `O(n)` intermediate solutions on a path whose interior H78
   has already measured to be a 0.786 pp valley in two parameterisations, whereas PX skips the
   interior entirely. I decline to queue a mechanism that pays for the thing the better mechanism
   avoids. If H82 dies because `k = 1` for all pairs, a *least-loss-ordered* insertion relink between
   two of **our own** orders (never the reference) becomes the natural fallback and should be filed
   then, not now.
2. **LNS with large destroy fractions, tabu with long-term memory, basin hopping, order-MCMC.** All
   priced in rounds. M17 disposes of them at n = 136,648 without needing a new measurement, and H80
   already spent a cycle establishing the round rate. Not queued.
3. **KaHyPar-style agreement coarsening ("multilevel crossover").** Same guarantee as PX at far
   higher engineering cost, and its move class is the one H46 measured at exactly +0.000000 pp.
   Declined; revisit only if H82 finds large `k`.
4. **Kemeny / rank-aggregation decomposition (XCC, αMOT certificates, Approximate Condorcet
   Partitioning).** The exact decomposition theorem is our net-digraph condensation = **H60, killed**;
   the certificate sweeps are the family scan_cycle18 § 5.9 declined and re-endorsed declining; the
   published scale is `n ≤ 240` with `O(n²)` partitioning. Not queued.
5. **Block-insertion algorithms for the LOP** (C&OR S030505481930303X, abstract only) — this is H41,
   killed at 91.9 % redundancy. Not queued.

**Honest negatives — things the literature does not have.**

- **No recombination operator has ever been applied to a feedback arc set instance at connectome
  scale**, as far as this scan can establish. Query 10 returned the same 2024 RASstar / GreedyAbs
  line the campaign already knows, plus memetic work on *hypergraph partitioning* and *signed graph
  clustering* — adjacent problems, not ours. The FAS literature at web scale is still "greedy is
  linear-time and hard to beat".
- **The number of PX components between two good LOP solutions is unmeasured at any n.**
  arXiv:2407.06742 is explicitly theoretical (no experiments, no instance sizes). This is
  simultaneously the reason H82 is worth running and the reason its predicted midpoint is low: nobody
  knows the answer, including the people who defined the operator. If H82's rung 0 runs, it is — as
  far as I can tell — the first such measurement, and it is worth writing down on a kill.
- **Ejection chains / variable-depth (Lin–Kernighan-style compound moves) for LOP or FAS: fourth
  consecutive scan finding nothing.** L01, L02, scan_cycle18 and now this one. scan_cycle18 already
  called it "a settled negative — stop looking". I agree and add nothing.
- **The FlyWire leaderboard is unchanged** since 2026-01-09 (12 rows retrieved in full; top entry
  Vahidi & Koutis 35,463,823 = our reference file). Two entries above 84.6 % that we do *not* have on
  disk exist — Bader et al. 35,459,266 (2024-11-16) and Hashorva 35,452,425 (2024-10-28) — so at
  least three independent solutions reach the band. No methods are published on the page.
- **SSRN 6221201 failed again (403) — the fifth documented failure.** The algorithm that produced our
  exact target remains unknown to this campaign.
- **The poppler / `pdftotext` gap cost this scan two primary sources** (the GECCO 2009 partition
  crossover paper and Martí's LOP path-relinking PDF), and it is the *fourth* consecutive scan
  blocked on the same missing binary. Both of those sources are cited above at abstract level only
  and are marked as such. Installing poppler remains the cheapest research-infrastructure fix
  available to this campaign.

**Standing item this scan re-flags for the third time (not a hypothesis).** `select_disjoint_moves`
(`segment.py:315`) packs disjoint moves greedily; the optimal weighted-interval-scheduling DP was
measured at **1.93×** the greedy gain in `proto_H45_pair_connectome.json`. L02 and scan_cycle18 both
called it a pure code fix and nothing has touched it. It is not a barrier crosser, but under M17's
revival clause (a) — "the round cost is reduced by orders of magnitude" — per-round efficiency is now
a first-class axis, and this is 1.93× of it lying on the floor.
