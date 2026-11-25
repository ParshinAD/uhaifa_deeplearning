# Engineering scout (phase 0), 2026-11-25

All numbers below come from the runs listed with each section (MacBook, Apple Silicon, env `allen`,
`PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`, run from the repo root
`nn_backward_connections/`). Several scripts ran at the same time on the 14-core machine, so wall
times are indicative, not benchmarks. None of the LB values here is a certificate; they are
probes of which methods are worth certifying.

W = 41,912,141, n = 136,648, m = 5,657,719. Ceiling % = 100 * (W - LB) / W.

## 1. Graph structure

Command: `$PY ceiling/src/scout_structure.py` -> `ceiling/logs/scout_structure.log` (36 s).

- No self loops, no duplicate edges. Min weight 2 (no weight-1 edges). Weight percentiles
  (0/50/90/99/99.9/100): 2 / 4 / 15 / 54 / 147 / 2405.
- Out-degree p50 27, p99 256, max 8064; in-degree p50 25, p99 293, max 8840.
  6,651 sources (in-degree 0), 2,447 sinks.
- Reciprocal pairs: 519,895 (1,039,790 edges), carrying 10,419,331 = 24.9 % of W.
  67,860 pairs are balanced (w_uv = w_vu).
- 2-cycle floor LB = 2,815,303 -> ceiling 93.2828 %.
- SCCs: 9,626 in total, giant SCC 126,840 nodes, only 123 SCCs of size > 1 (second largest 8),
  9,503 singletons. Weight between SCCs (free FF) = 630,145; inside SCCs 41,281,996.
  Giant SCC: 5,606,125 edges, weight 41,278,263. So SCC decomposition gives almost nothing.
- After removing the 2-cycle packing: residual weight in giant SCC 35,648,499 on 5,018,557 edges.
  The residual graph (edges with cap > 0) has a giant SCC of 124,984 nodes, 5,000,481 edges,
  residual capacity 35,392,715; residual weight between residual SCCs (can never be in a cycle
  of the residual) 888,791.

## 2. Feedback of the champion order (H64)

Same script and log.

- FF 35,314,407, backward weight 6,597,734 (matches W - FF), 1,173,245 backward edges.
- (a) backward weight on reciprocal edges 3,398,872 = 2,815,303 (2-cycle floor, paid by any order)
  + 583,569 excess (99,871 pairs where the heavier side is backward).
  (b) backward weight on non-reciprocal edges 3,198,862.
- So the gap between the 2-cycle floor and our order is 3,782,431 syn; a bound method has to
  "explain" this part. Our order puts 6,597,309 of the backward weight in the giant SCC.
- Rank span of backward edges (rank[src] - rank[tgt]) p50 18,156, p90 61,014, p99 113,129.
  By span: 1-10: 262,617 w; 11-100: 205,999; 101-1e3: 481,675; 1e3-1e4: 1,577,670;
  1e4-1e5: 3,751,842; >1e5: 317,931. Most feedback is long-range in our order.
- Per super_class (both endpoints in the class; there is no separate lane field in
  `layering/outputs/fly_ann/node_ann.npz`, so super_class is used): 2-cycle floor / our backward
  weight = optic 1,505,589 / 3,117,142 (0.48), central 876,100 / 2,167,029 (0.40),
  visual_projection 26,544 / 66,096 (0.40), cross-class edges 394,650 / 1,216,658 (0.32).
  Small classes (sensory 0.90, motor 0.70) are nearly explained by 2-cycles.

## 3. Old probe (2-cycle + greedy 3-cycle) and full 3-cycle enumeration

Command: `$PY -u ceiling/src/scout_probe3.py` -> `ceiling/logs/scout_probe3.log` (copy of the
old probe logic, seed 0, loader instead of CSV, 900 s cap).

| time | seed edges done | LB | ceiling % |
|---|---|---|---|
| 60 s | 228,919 | 3,240,239 | 92.2690 |
| 120 s | 454,415 | 3,547,203 | 91.5366 |
| 420 s | 1,920,087 | 4,372,450 | 89.5676 |
| 900 s (cap, not finished) | 4,489,697 of 5,069,964 | 4,562,821 | 89.1134 |

Rate: about 5,000 seed edges/s, 7,400 triangle hits/s (pure Python loop). It does not complete
in 15 min (89 % of seed edges); extrapolated finish about 17 min. The 2026-08-19 run reached
4,537,757 at 420 s; here 420 s gave 4,372,450, so that run was about 2x faster per second
(guess: no concurrent load then; algorithm and seed are the same).

Vectorized enumeration is much faster. Command: `$PY -u ceiling/src/scout_tri_full.py` ->
`ceiling/logs/scout_tri_full.log` (69 s total):
- The residual graph (after exact 2-cycle packing) has exactly 2,765,514 directed 3-cycles,
  enumerated in 27 s by chunked sparse matmul (A[rows] @ A masked by A^T, then middle vertices).
- Sum over 3-cycles of their min residual cap (no sharing) = 8,488,017 (a loose cap on what
  3-cycles could ever add).
- Greedy packing of all 3-cycles, random order: +1,717,011 -> LB 4,532,314 (89.1862 %), 3.5 s.
- Greedy, largest min-cap first: +1,785,371 -> LB 4,600,674 (89.0231 %). Best number in this scout.
- The edge-seeded probe order beat random triangle order (4,562,821 at 89 % done), so packing
  order is worth about 70k syn here.
Conclusion: with 2- and 3-cycles only, greedy saturates near 89.0 %. The remaining gap to our
order is about 2.0M syn (LB 4.60M vs feedback 6.60M), and 3-cycles are exhausted.

## 4. Primitive timings (residual giant SCC: 124,984 nodes, 5,000,481 edges)

Command: `$PY -u ceiling/src/scout_primitives.py graph` -> `ceiling/logs/scout_primitives_graph.log`.

| primitive | time |
|---|---|
| build residual graph from loader | 29 s |
| scipy dijkstra, 1 source (lengths 1/cap) | 0.115 s |
| scipy dijkstra, 10 / 100 sources (batched) | 1.13 s / 11.0 s (linear, no batching gain) |
| scipy dijkstra, 1 source with `limit` | 0.16 s (limit reached almost all nodes) |
| scipy BFS (breadth_first_order) | 0.014 s |
| scipy unweighted dijkstra (hop distances) | 0.071 s, max 8 hops from the sample source |
| igraph build | 3.8 s |
| igraph distances, 1 / 10 sources | 0.167 s / 0.873 s |
| igraph BFS | 0.035 s |
| 3-cycle matmul, 2,000 rows | 0.24 s (584k cycle hits/s) |

So one shortest-cycle oracle call is about 0.1 s per source with scipy; one full round
"shortest cycle through every node" is about 125k * 0.1 s = 3.5 h single core (about 15 min on
14 processes). igraph is not faster for weighted distances, only for build.

Torch (`scout_primitives.py torch` -> `ceiling/logs/scout_primitives_torch.log`): sparse COO and
CSR matmul are not implemented on MPS in this torch (NotImplementedError). CPU: torch sparse
@ dense(n x 64) 0.23 s per product, scipy CSR 0.15 s. MPS is not useful for this.

LP (HiGHS via `scipy.optimize.linprog`), packing LP max sum y, M y <= cap, 3 nonzeros per column;
time limit 400 s per solver. Commands `scout_primitives.py lp <ncols> random|tri3`, logs
`ceiling/logs/scout_primitives_lp1e5.log`, `ceiling/logs/scout_primitives_lp_tri3_big.log`.

| LP | cols | rows | highs (simplex) | highs-ipm | peak RSS |
|---|---|---|---|---|---|
| random rows | 1e5 | 5e4 | time limit 400 s | 33.5 s | 0.6 GB |
| real 3-cycles (random subset) | 1e5 | 262,018 | 0.6 s | 0.6 s | 0.8 GB |
| real 3-cycles (random subset) | 1e6 | 1,511,195 | 37 s | 43 s | 3.3 GB |
| real 3-cycles (all) | 2,765,514 | 2,500,751 | time limit 400 s | time limit 400 s | 6.0 GB |

Random 1e6 was not run (1e5 random already needs IPM). The real cycle LP is much easier than a
random one, but the full 3-cycle LP did not finish in 400 s with either solver. LP values from a
subset of columns are valid but not comparable with the greedy over all cycles (the 1e6 subset LP
gives 1,297,340.5). HiGHS returns floats; a certificate needs rounding down to integers over a
denominator D (K1 format), which is easy for a packing (scale, floor, check feasibility).

## 5. Mouse meso graph (calibration instance)

Commands: `$PY -u ceiling/src/scout_mouse.py lazy` -> `ceiling/logs/scout_mouse_lazy.log`,
`$PY -u ceiling/src/scout_mouse.py order` -> `ceiling/logs/scout_mouse_order.log`.

- n 148, m 583, W 9.159036 (float weights 0.000184 to 0.613959). No self loops or duplicates.
- Reciprocal pairs 83 (28.5 % of edges, 40.0 % of weight). 2-cycle LB 0.515373 -> ceiling 94.3731 %.
- SCCs: one of 102 nodes and one of 2; 465 edges inside SCCs; 1.727607 weight between SCCs.
- Champion H63: 93.175383 %, feedback 0.625069, 124 backward edges.
- Exact MFAS, lazy cycle constraints (binary y_e, shortest cycle through every remaining edge,
  `mip_rel_gap=0`): 3 MILP rounds, 527 cycles, under 0.1 s. OPT feedback 0.622613 ->
  OPT FF 93.202194 %. A topological order of G minus the solution scored by the frozen oracle
  gives 93.202194 %.
- Ordering formulation (x_ij for i<j on the 104 SCC nodes, 5,356 binaries, 182,104 transitivity
  rows): HiGHS optimal in 1.8 s, same 0.622613, dual bound equal.
- So the mouse optimum is known: 93.2022 %, and the mouse champion is 0.0268 pp below it.
  (This also means the mouse champion is not optimal; not acted on here.)

## 6. Assessment

3-cycles are exhausted: all 2.77M residual 3-cycles fit in memory and greedy packing over all of
them gives LB 4.60M (89.02 %), only about 70k better than the old probe. The LP optimum over the
same 3-cycles was not obtained (full LP timed out); how much it adds over greedy is unknown (guess:
modest, since it is bounded by the 8.49M no-sharing sum and greedy already packs 1.79M). Getting below 89 % needs longer cycles, so the feasible path is LP-optimal cycle
packing with a shortest-cycle oracle: Garg-Konemann / multiplicative weights or column generation,
where each pricing round is a Dijkstra (0.1 s per source with scipy) and the master is a packing
LP that HiGHS solves in about 40 s for 1e6 real cycle columns but not in 400 s for 2.8M. That
points to MWU (no LP solver, integer-friendly, parallel over 14 processes) rather than a
monolithic HiGHS master, or HiGHS only on decomposed blocks (the cycle-edge incidence of 3-cycles
likely splits into components; not measured). Exact methods (ordering MILP, lazy cycle MILP) are
instant on the 148-node mouse, so they are usable as calibration and on small dense pieces of the
fly graph (K2 certificates), but not on the 125k-node giant SCC. GPU (MPS) sparse ops are not
available in this torch, so everything is CPU.
