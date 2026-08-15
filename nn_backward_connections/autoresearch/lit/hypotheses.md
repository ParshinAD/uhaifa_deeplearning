# Ranked shortlist — new falsifiable hypotheses from scan L01 (2026-08-16)

> **Queue ids, and a concurrency note.** While this scan was writing, a concurrent cycle allocated
> `H44` and then queued its own `H45` and `H46` — **the same two mechanisms** derived here as ranks 1
> and 2, reached independently from the same paper. That is the strongest novelty signal the queue
> has had, and it is also live evidence for **P03**. Resolution: the duplicates were **merged, not
> overwritten** — the surviving `queue.json` items `H45` and `H46` carry the other agent's framing
> plus a `scout_L01_addendum` holding everything below that they did not have, and this scan's third
> item was renumbered to `H47`. Mapping used in this file: **rank 1 = H45**, **rank 2 = H46**,
> **rank 3 = H47**.
>
> One substantive correction went into the `H45` addendum and is repeated here because it is a
> correctness issue: the concurrent item proposes maximising `total_u(c) + total_v(c) + w_uv` over a
> common cut `c` using `jacobi_best_gaps`' per-node profiles. **That is a heuristic, not the exact
> gain** — `total_u` was computed with `v` still in place and `total_v` with `u` still in place, so
> the `u`–`v` interaction is double-counted and each profile uses a stale position for the other
> endpoint. The exact form is the `Δ(r)` derived in rank 1 below.

Target: beat the champion **H42 = 84.1541%** on connectome (35,270,783 / 41,912,141), without
regressing microns or mouse, inside 3600 s per run. Reference: **84.6147%**, gap **0.4606 pp**.

Everything below shares one structural property, and it is the reason these three were selected out
of everything read: **the exact gain is available in closed form from a contiguous-block argument**,
so no move ever needs the oracle to be evaluated, and disjoint moves compose additively — which is
what makes them affordable at 136,648 nodes. That is the same lemma `mfas/refine/scc_recursive.py`
and `mfas/refine/segment.py` already rest on, so the machinery, the unit-test pattern and the
leakage-safety argument all carry over unchanged.

Minimum effect size in force on connectome: **0.012 pp** (`campaign.yaml`, `findings.md` #7).

---

## Rank 1 — **H45**: violated-edge interval repair (Vahidi's Algorithm 2, batched)

### Mechanism
The unit of search is a **backward edge**, not a node, an SCC, or a position window. For each edge
`(u,v)` that is currently backward (`rank[u] > rank[v]`), the pair of endpoints defines the interval
of positions `I = [rank[v], rank[u]]`. The move extracts `u` and `v` from `I` and re-inserts them as
the **adjacent pair `u, v`** at a split point `r` inside `I`, leaving every other node's relative
order untouched. Its exact gain is

```
Δ(r) = (w_uv − w_vu)
     + Σ_{j ≤ r} [ w(n_j → v) − w(v → n_j) ]
     + Σ_{j > r} [ w(u → n_j) − w(n_j → u) ]
```

where `n_1 … n_t` are the nodes strictly between `v` and `u`. This is exact — the interval is a
contiguous position range, so no outside edge can flip; the `n_j` keep their relative order, so no
edge among them can flip; only edges incident to `u` or `v` flip, each on exactly one side of `r`.
`Δ(r)` is **piecewise constant in `r`, changing only at neighbours of `u` and `v`**, so the optimal
`r` costs `O((d(u)+d(v)) log(d(u)+d(v)))` — **independent of how long the interval is**. Mean total
degree on connectome is 83.

Batching (this is what makes it a *sweep* rather than a loop): two **disjoint** intervals are
independent and their gains **add**, by the same lemma. So compute the best `(r, Δ)` for a large
candidate set in one vectorised pass, then choose a maximum-total-gain set of pairwise
non-overlapping intervals by the classical weighted-interval-scheduling DP (sort by right endpoint,
binary search for the last compatible interval, `O(k log k)`), and apply the whole set at once. The
realised score delta equals the sum of the selected gains, exactly — assertable in a unit test
against the frozen oracle, exactly as `tests/test_refine_segment.py` does today.

Fallbacks when no `r` improves (Vahidi Alg. 2): swap `u`↔`v`; push `v` forward; pull `u` backward.

### Why it is not already killed
| entry | why this is different |
|---|---|
| **M4** (local is empty; moves must be global-range or structurally decomposed) | **Satisfies the revival condition head-on.** The interval length is set by the violated edge, not by a parameter — it is unbounded, and the heap processes the heaviest violations first, which by `diagnosis.md` Step 2 have median rank distance ~22,580. This is the *only* move class in the campaign whose range is data-determined. See `notes-vahidi-flywire.md` §7: the campaign's own description of the prior art as "bounded-span insertion" is wrong, and that error is what made this axis look dead. |
| **H22** (bounded-window sifting, killed at the sizing gate) | H22 moves **one** node inside a **fixed** rank window. This moves **two specific nodes jointly** across a **data-determined, unbounded** span. Neither `u` nor `v` need profit alone. |
| **H31** (ILS/LNS ruin-and-recreate, screen fail) | H31's revival condition, verbatim: *"The destroy operator is STRUCTURE-AWARE (SCC-, block- or cycle-guided) rather than random, and the rebuild is exact-gain."* A backward edge is the cycle-guidance — every backward edge is part of the residual feedback structure — and the rebuild is the closed-form `Δ(r)` above. Condition met. |
| **H41** (segment moves, prototype-passed) | H41 swaps two **adjacent contiguous blocks** with `L,d ≤ 1024`, i.e. span ≤ 2048 = 1.5% of the line, and it moves whole runs. H45 moves two **non-contiguous individual** nodes across an arbitrary span. Disjoint objects. Credit must still be taken as (H41+H45) − H41 if H41 promotes first (`CAMPAIGN.md`, double-counting rule). |
| **M1, M2, M3, M5, M6, M7** | Untouched — no gradient, no dynamics knob, no multi-start, no ties, no init change, no expensive warm-start. |

### Cheap decisive test (no GPU)
From the **stored** H42 champion connectome positions (`results/*-H42-connectome-*-confirm-*_positions.npy`),
run one full interval-repair sweep over the top `N = 200,000` backward edges by weight, with the
disjoint-interval DP, and report the exact realised delta (verified against the frozen oracle).
CPU only, minutes, no dataset re-training.

**Kill condition:** if one sweep on the champion order yields **< 0.012 pp** (the connectome minimum
effect size), the move class finds nothing at our fixed point and H45 is dead — no GPU spent.
Second gate: if 20 alternating sweeps (interval repair ↔ the champion's 2-sweep sift) do not exceed
what the champion's own move classes buy in the same wall-clock, it is compute and not a move class,
and it is dead by the matched-wall-clock rule `findings.md` #6 established.

### Expected effect size and cost
- **Expected: +0.05 to +0.30 pp on connectome.** Prior evidence is unusually strong: this is the
  documented mechanism carrying 0.7524 → 0.8461 in Vahidi 2025, and — per the abstract of Vahidi &
  Koutis 2026, **which I did not read in full** — the interval-selection DP is a component of the
  method that holds the leaderboard entry equal to our target number.
- **Cost:** prototype ~4–6 h of implementation + minutes of CPU, **0 GPU**. If it passes, a normal
  screen + confirm. New module `src/mfas/refine/interval_repair.py` (nothing existing changes) plus
  `src/mfas/experiments/H45.py` = H42's stages 1–3 then an alternation with the interval sweep added.
- **Runtime at scale:** ~1.16M backward edges exist at 84.15%; a full pass at `O(d(u)+d(v))` each is
  ~1e8 elementary operations. Restricting to the top 100k–200k candidates per sweep and vectorising
  the neighbour scan should give ~10–30 s per sweep on connectome, against ~2,368 s of unused
  headroom under the 3600 s cap (`queue.json` S01). **This is not an `O(n²)` method** — that is the
  whole point of the piecewise-constant argument.

---

## Rank 2 — **H46**: multi-resolution coarse-block LOP (exact-gain, unbounded range)

### Mechanism
Cut the current order into `K` **consecutive blocks** of size `s ≈ n/K`, each kept internally rigid.
One `O(m)` pass builds the `K × K` inter-block weight matrix `W[i][j] = Σ w` over edges from a node in
block `i` to a node in block `j`. Because the blocks are contiguous and rigid, **permuting them
changes the exact feedforward score by exactly the change in `Σ_{i before j} W[i][j]`** — intra-block
edges never flip, inter-block edges flip iff the block order flips. The coarse problem is therefore an
**exact** small dense Linear Ordering Problem, not an approximation.

Solve it with the LOP insertion neighbourhood on `D = W − Wᵀ`: moving block `a` from position `p` to
position `q` gains `−Σ_{b ∈ (p,q]} D[a][b]` (rightward) or `+Σ_{b ∈ [q,p)} D[a][b]` (leftward), which
is a prefix sum along the current coarse order — `O(K)` for **all** target positions at once, `O(K²)`
for a full sweep. Iterate to a coarse local optimum, apply, re-rank.

Multi-resolution and boundary cycling: run at `K ∈ {32, 128, 512, 2048, 4096}` with block-boundary
**offsets** cycled deterministically across rounds (same idea and same no-RNG discipline as
`DEFAULT_SPLIT_FRACS` in `scc_recursive.py`), so a node sitting on a boundary at one round is interior
at the next.

Optional exactness upgrade over the prior art: for a window of `x ≤ 16` coarse blocks the LOP can be
solved **exactly** by Held–Karp subset DP, `f[S] = max_{j∈S} f[S\{j}] + Σ_{i∈S\{j}} W[i][j]`, in
`O(2^x · x²)` — `x = 16` costs ~1e6 operations against `16! = 2e13`. Vahidi's Alg. 4 brute-forces
`x!`, so it is capped near `x = 5`.

### Why it is not already killed
| entry | why this is different |
|---|---|
| **M4** | Revival condition met: **unbounded range**. A block at position 0 can move to position `n−1` in a single move. This is the exact failure mode `segment.py`'s own docstring describes ("*if a run of 300 nodes belongs 800 positions earlier but no single one of them profits from moving alone, the sift is at a fixed point*") — extended from 800 positions to the whole line. |
| **H22** | Not a rank window: this is a global permutation of a coarsening of the entire line. |
| **H41** | H41 **is the `x = 2`, adjacent-only, span-≤2048, powers-of-two special case of this**. H46 is the multi-block, unbounded-range generalisation. A cyclic rotation of five blocks is unreachable by any sequence of *individually improving* adjacent swaps — the classic local-optimum trap H41 is subject to by construction. Credit must be taken as (H41+H46) − H41. |
| **H31** | Deterministic, exact-gain, structure-defined. No random destroy. |
| **H32 / H33 / M7** (expensive global embeddings lose) | Not an embedding and not a warm-start: it **refines the champion's own order** and costs one `O(m)` pass plus a `K²` dense solve. H33's kill was quality **and** a 304–507 s eigensolve; H46 has no eigensolve. |
| **M1, M2, M3, M5, M6** | Untouched. |

### Cheap decisive test (no GPU)
From the stored H42 champion connectome positions: for each `K` in the ladder and each offset, build
`W`, run the coarse LOP to a local optimum, and report the **exact predicted gain** (and verify it
against the frozen oracle on the rebuilt rank vector). This is seconds of CPU per level.

**Kill condition:** if the total exact gain summed over the whole `K` × offset ladder is **< 0.012 pp**,
the champion's order is already coarse-optimal and H46 is dead — no GPU spent.

This gate is worth running **first of the three**, even though H46 ranks second, because it is the
cheapest decisive test in the queue *and* because either outcome is a diagnosis the campaign does not
currently have: it partitions the remaining 0.4606 pp into "coarse mis-ordering of whole regions" vs
"fine mis-ordering inside regions", which tells H45 and H47 where to look.

### Expected effect size and cost
- **Expected: 0 to +0.15 pp on connectome**, honest midpoint +0.05. Lower prior evidence than H45
  (Vahidi's Alg. 4 is not credited with a specific increment in the paper), higher structural
  argument: this is the one hole that provably none of the three existing move classes covers.
- **Cost:** prototype ~2–3 h of implementation + minutes of CPU, **0 GPU**. New module
  `src/mfas/refine/coarse_lop.py` + `src/mfas/experiments/H46.py`.
- **Memory/runtime at scale:** `W` is built sparsely (`scipy.sparse.coo_matrix` on
  `block(src)·K + block(tgt)`, at most `m = 5.66M` nonzeros) and densified only for `K ≤ 4096`
  (16.8M float64 = 134 MB). A coarse sweep is `O(K²) = 1.7e7` at `K = 4096`, ~0.1 s vectorised. Whole
  ladder ≲ 60 s per stage-4-equivalent call.

---

## Rank 3 — **H47**: exact subset-DP ordering at the recursion's leaves

### Mechanism
`SccRecursiveRefiner._refine` returns immediately when `hi − lo <= min_block` (**32**), and inside
every SCC it preserves the current relative order. So the champion's block refiner **never optimises
anything below 32 nodes**. Vahidi's Algorithms 3 and 5 both do: any sub-SCC with `|S| ≤ 9` is permuted
**exhaustively** to maximise internal forward weight.

Replace exhaustive permutation with Held–Karp subset DP —
`f[S] = max_{j∈S} f[S\{j}] + Σ_{i∈S\{j}} w(i→j)`, giving the **exact optimum** internal order of a
`k`-node block in `O(2^k · k²)` instead of `O(k! · k)`; `k = 16` is ~1e6 operations against
`16! = 2e13`. Apply at every leaf of the recursion and at every SCC of size `≤ k_max` met during the
descent, with `min_block` lowered accordingly.

### Why it is not already killed
| entry | why this is different |
|---|---|
| **M4** | This is *not* a rank window: it is the **terminal case of the SCC recursion**, i.e. structurally decomposed — M4's own named revival condition. It also does not add a class; it makes an existing, already-confirmed class exact where it currently does nothing. |
| **H22** | H22 was single-node re-insertion in a window. This is the **exact optimum over all `k!` orderings** of a structurally-derived block. A single-node sift can be at a fixed point while the block is far from its optimum. |
| **M1–M3, M5–M7** | Untouched. |

**Honest self-check:** M4 is nonetheless the strongest prior *against* this one. M4 measured net
recoverable weight within a window as ≤ 0 for `W ≥ 100`, and while `k ≤ 16` is far below that, the
champion's sift is already single-node optimal, which removes most of what a tiny window can hold.
I expect this to die at its gate. It is queued anyway because the gate costs almost nothing and the
number it produces — how much of the residual is intra-window at all — is a diagnosis worth owning.

### Cheap decisive test (no GPU)
From the stored H42 champion connectome positions: for `k ∈ {8, 12, 16}`, tile the line with disjoint
contiguous windows of size `k`, DP-optimise each, and sum the exact gains (then verify the total
against the frozen oracle). Minutes of CPU.

**Kill condition:** total gain **< 0.012 pp** at every `k` → dead.

### Expected effect size and cost
- **Expected: 0 to +0.03 pp.** Low.
- **Cost:** prototype ~1–2 h of implementation + minutes of CPU, **0 GPU**. New module
  `src/mfas/refine/leaf_exact.py` providing a subclass of `SccRecursiveRefiner` that overrides only
  the base case, so `scc_recursive.py` is untouched and the champion is bit-reproducible; variant
  `src/mfas/experiments/H47.py`.

---

## Rank 4 (GATED — do **not** queue yet) — replace greedy-FAS with Vahidi's ratio greedy

Vahidi's Alg. 1 scores `s(u) = (out_w(u)+1)/(in_w(u)+1)`, peels by max-heap, and reaches **0.7524**
on this exact graph. Our `H02` greedy (Eades–Lin–Smyth / GreedyAbs) reaches **0.6891**
(`findings.md` #1). Same asymptotic cost. **+6.3 pp of starting quality for free.**

**Why it is not queued now.** `killed.json` **M6** says the init→plateau curve through Rocket is flat
(82.87–82.93% across inits spanning 36–69%), so a better init buys ≤ 0.06 pp *as long as Rocket sits
between the init and the refiner*. **M7** is not violated — this is a cheap greedy, not an expensive
embedding — but M6 makes the whole thing null under the current pipeline.

**The gate.** Queue item **S01/P07** is measuring right now whether the Rocket phase is cuttable at
all (`proto_P07.json`: microns at 0 epochs = 83.11152% vs 83.2409% at 80,000). If S01 reports that
connectome can drop to a small epoch count, then the init feeds the **sift** directly, and M6 — which
is a statement about *Rocket's* plateau and nothing else — stops covering the case. At that moment
this becomes a live, cheap, one-module hypothesis. **Re-read this section when S01 lands.**

---

## Rank 5 (NOT proposed — recorded so it is not re-derived)

**Ejection chains / variable-depth Lin–Kernighan search.** A genuine move-class gap and the
literature has nothing on it for LOP or FAS at any scale (`notes-lop-and-ocm.md` §4). It is
**inherently sequential**: depth `k` costs `k` dependent steps and yields one move, where our sweeps
cost one `O(m)` pass and yield thousands of independent moves. Without a batching story it is 3–4
orders of magnitude off this campaign's cost curve. H45 is the affordable depth-2 relative of it.

**Reduction rules / preprocessing from the FASP literature.** They exploit near-acyclicity; 92.8% of
our nodes are in one SCC and the reference solution still leaves 15.39% of the weight backward. See
`notes-scalable-fas-and-negatives.md` §1.

**Anything continuous.** Nothing found that addresses Q01's scale-blindness, and the two orderings
above ours on the leaderboard were produced with no gradient step at all. **M1 is corroborated, not
merely unfalsified.**

---

## Summary table

| rank | queue id | move class | range | exact gain? | GPU for the kill test | expected pp | main risk |
|---|---|---|---|---|---|---|---|
| 1 | **H45** | joint pair relocation, edge-driven | **unbounded, data-set** | yes, `O(d(u)+d(v))` | **none** | +0.05 … +0.30 | implementation weight; candidate-set sizing |
| 2 | **H46** | permutation of coarse contiguous blocks | **unbounded** | yes, `O(m + K²)` | **none** | 0 … +0.15 | champion may already be coarse-optimal |
| 3 | **H47** | exact `k!`-optimal ordering of recursion leaves | `k ≤ 16` | yes, `O(2^k k²)` | **none** | 0 … +0.03 | M4 says local is empty |
| 4 | gated | ratio greedy init | — | — | — | 0 (now) | blocked by M6 until S01 lands |

**Suggested execution order** (all three kill tests are CPU-only and need **no GPU**): run **H46**'s
gate first because it is the cheapest and its number tells the other two where to look; then **H45**,
which carries the strongest prior evidence and the largest expected effect; then **H47**, which I
expect to die.
