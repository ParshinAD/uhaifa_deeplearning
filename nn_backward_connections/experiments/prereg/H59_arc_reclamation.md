# H59 — pre-registration, sealed before launch

**Cycle 14, 2026-08-26. Divergent mode** (`cycles_since_score_move = 5`, `consecutive_kills = 4`).
Written and committed BEFORE the prototype is run (queue item P11; the H60 provenance lesson).

## The claim

The champion's order induces a forward edge set `F` (every edge `u->v` with `rank[u] < rank[v]`).
`F` is a DAG and the champion order is one of its topological orders. Let `(u, v)` be a **backward**
edge, i.e. `rank[v] < rank[u]`, so it is currently feedback.

> **Reclamation lemma.** If there is no `F`-path from `v` to `u`, then `F ∪ {(u,v)}` is acyclic, and
> **every** topological order of `F ∪ {(u,v)}` scores at least `base + w_uv`.
>
> *Proof.* A topological order of a DAG makes every one of its arcs feedforward. `F ∪ {(u,v)}` is
> acyclic exactly when adding `(u,v)` closes no cycle, i.e. when `u` is unreachable from `v` in `F`.
> Any topological order of it therefore keeps all of `F` feedforward — the `base` score — and makes
> `(u,v)` feedforward as well, adding `w_uv`. Edges outside `F ∪ {(u,v)}` can only add more. ∎

So the realised gain is **monotone by construction and bounded below by the reclaimed weight**. This
is the classic *minimality* property of a feedback arc set ("no deleted arc can be reintroduced
without creating a cycle"); the campaign has never tested whether its FAS has it.

**Why no existing move class can find this.** Reclaiming one arc may require re-sorting thousands of
nodes, none of which profits alone, so the single-node sift (H35) and the pair move (H45) are blind
to it. The SCC refiner (H36/H42) condenses `G`, where `u` and `v` sit in one giant SCC — and H60
showed that even the exact net-digraph condensation `D+` does not separate them. The move here is
not in position space at all: it edits the **arc set** and then re-derives positions from it.

## THE GATE — re-specified before launch

The queue item's stated `kill_condition` is *"the total weight of reclaimable backward edges exceeds
0.012 pp"*. **That is a capacity statistic and meta-rule M11 forbids it as evidence** — H47 measured
capacity and achievability on the same order and they differ by two orders of magnitude. The queue
item's own SECONDARY condition is the realised number, so it is promoted to PRIMARY here:

| | statistic | role |
|---|---|---|
| **PRIMARY (the gate)** | realised percentage from the **frozen oracle**, after greedy heaviest-first re-add + one rank-stable topological re-sort, minus the champion's percentage | decides keep/kill |
| SECONDARY | `n_reclaimable`, `W_reclaimable` | **context only** — reported, never a gate (M11) |

**Pre-registered kill condition.** H59 dies at the prototype rung iff the realised delta is
`< 0.012 pp` on connectome **or** `< 0.002 pp` on microns (the `screen_delta_pp` of
`campaign.yaml`). It proceeds to a screen only if it clears both.

**M12 applies.** This prototype is measured FROM the champion's converged order, and M12 says such a
number systematically **overstates** what the pipeline would deliver. That bias is in the wrong
direction for a pass, so a *failure* here is safe to act on; a *pass* would still have to survive a
real screen. The control arm's own remaining headroom is stated with any positive number.

## The algorithm being prototyped

1. Load the champion position vector off a logged run (never recomputed). `rank = argsort(argsort())`.
2. `F` = all non-self-loop edges forward under `rank`; `B` = all backward edges, weight-desc.
3. **Reclaimability test** for each `(u,v) ∈ B`: is `u` reachable from `v` in `F`? Every `F`-path
   moves strictly rightward in rank, so the search is confined to the open rank interval
   `(rank[v], rank[u])` — no global search. Resolved by a cascade, cheapest first:
   1-hop (a direct `v->u` edge in `F`), 2-hop (`out_F(v) ∩ in_F(u)` inside the interval), then a
   bounded bidirectional BFS confined to the interval.
   A query that exhausts its expansion budget is recorded as **UNKNOWN** and treated as *reachable*
   (i.e. not reclaimable). That is conservative for the capacity statistic, and it cannot inflate
   the primary gate, which only ever counts arcs actually added. `n_unknown` is reported.
4. **Greedy heaviest-first re-add.** In weight-desc order over the reclaimable set, add `(u,v)` to
   the accepted set `S` iff `F ∪ S ∪ {(u,v)}` is still acyclic (checked against the maintained
   arc set, not against `F` alone — accepted arcs create new reachability).
5. **One rank-stable topological re-sort** of `F ∪ S`: Kahn's algorithm with a min-heap keyed on the
   champion rank, i.e. the topological order of `F ∪ S` that is lexicographically closest to the
   champion order. This is the least-disruptive choice; the lemma holds for any of them.
6. **Score with the frozen oracle** (`mfas.metrics.score_from_order`). Report realised delta.
7. **Exactness check (a lower-bound identity, not an equality):** the lemma guarantees
   `realised_score >= base_score + W_S`. A violation means the implementation is wrong, and the run
   is void. This is checked and reported as `lemma_holds`.
8. Repeat 2-7 for up to `--rounds` rounds (default 3): a re-sort can expose newly reclaimable arcs.
   Stop early when a round reclaims nothing.

Datasets: connectome and microns (the primaries, which carry the gate) and mouse (context).
CPU only, zero GPU. Leakage-safe: only edge weights and ranks are read; `data/best_solution` is
never touched.

## What each outcome means

- **Realised delta below both bars** → **kill**, and the kill carries a *certificate*: the champion's
  feedback arc set is (effectively) minimal, and the whole arc-reinsertion family closes with it.
  That is a positive result about the search space and it is worth the cycle even at zero score.
- **Realised delta above both bars** → the mechanism is real; implement `src/mfas/experiments/H59.py`
  and run the full screen. M12 says expect the screen to come in lower.
- **`n_unknown` large enough that the untested weight could itself clear the bar** → the prototype is
  inconclusive rather than negative, and it must say so instead of claiming a kill.

Run:
```
PYTHONPATH=src python experiments/proto_H59_reclaim.py <dataset> [--rounds N] [--budget N]
```
Out: `experiments/outputs/proto_H59_<dataset>.json`
