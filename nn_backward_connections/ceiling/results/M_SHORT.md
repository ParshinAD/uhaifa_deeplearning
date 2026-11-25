# Method P: LP-quality packing of short cycles, then longer (R1, R2; R6 not done)

Date 2026-11-25. Git HEAD 657786fb (worktree not committed). All runs on the MacBook, CPU only,
at most 6 cores, shared with other agents (wall times are indicative).
W = 41,912,141. Ceiling % = 100 (W - LB) / W. Our order: feedback 6,597,734 (84.2582 %).

## What was done

1. `src/m_short_enum.py`: all directed 2-cycles and 3-cycles of the FULL graph (not of the residual
   after 2-cycle removal, as in the scout's `tri3_residual.npz`): 519,895 2-cycles and
   7,435,509 3-cycles, 123 s. Joint LP over both, no 2-cycles fixed first.
2. Fractional packing LP max sum y, sum over cycles through e of y <= w(e), over explicit columns.
   - `src/m_short_mwu.py` (Garg-Konemann style multiplicative weights, all near-shortest columns
     routed per round). Too slow: after 400 rounds (about 4 min) the dual estimate was still 5.38 M
     against a primal of 4.8 M; stopped.
   - `src/m_short_pdhg.py` (diagonally preconditioned PDHG with restarts, scipy sparse matvec,
     1 core). Gap 0.25 % after 400 iterations (4 min), 0.007 % after 3000 iterations (15 min).
     Chosen by this measurement. Primal = each y_C divided by the max congestion on its edges
     (always feasible). Dual estimate = Lagrangian bound sum_e x_e + sum_C min_e w_e max(0, 1 - x(C)),
     valid for the column set used (float arithmetic, so an estimate, not a proof).
3. `src/m_short_cert.py`: y_int = floor(D y) with D = 10,000, overload repair (never needed), then
   greedy integer refill of the leftover D w - load with pool columns in order of dual length.
   Writes the K1 cert and runs `src/verify_cert.py`.
4. Column generation (`src/m_short_price.py`): lengths l_e = x_e / w_e + 0.001, Dijkstra (limit 1)
   from a random sample of sources, close with every in-edge, keep up to 20 cycles per source with
   l(C) < 1, length >= 4. Then warm-started PDHG on the bigger pool (`src/m_short_pool.py`).
5. `src/m_short_refill.py`: integer greedy refill of a certificate with any-length cycles: BFS
   from random sources in the residual graph (D w - load > 0), pack the shortest closing cycles.

Observation on the duals: after the 2+3-cycle LP, 78 % of edges have l_e = 0 and 13 % have l_e
close to 1, so almost all violated cycles are cycles in the unsaturated subgraph (mean dual length
of the priced cycles 0.019 in r1 rising to 0.082 in r6).

## Runs (all certificates verified PASS by `src/verify_cert.py`)

| run (results/*.json) | cert | columns in pool (new) | LB verified | ceiling % | LP dual estimate (pool) | time |
|---|---|---|---|---|---|---|
| m_short_greedy23_joint | c10_greedy23_joint | 7,955,404 | 4,556,165 | 89.1292 | - | 55 s |
| m_short_pdhg23_test400 (PDHG 400 it) | c11_pdhg23_test400 | 7,955,404 | 4,887,334 | 88.3391 | 4,897,387 | 4 min + 75 s |
| m_short_pdhg23 (PDHG 3000 it) | c13_pdhg23 | 7,955,404 | 4,889,638 | 88.3336 | 4,889,963 | 15 min + 74 s |
| m_short_pdhg23_refill1500 | c14_pdhg23_refill1500 | c13 + 215,107 BFS cycles | 5,130,285 | 87.7594 | - | 26 min |
| m_short_cg_r1 | c15_cg_r1 | 8,241,049 (+285,645) | 5,165,389 | 87.6757 | 5,166,240 | 22 + 6 + 2 min |
| m_short_cg_r1_refill1500 | c16_cg_r1_refill1500 | c15 + 342,866 | 5,442,629 | 87.0142 | - | 26 min |
| m_short_cg_r2 | c17_cg_r2 | 8,661,806 (+420,757) | 5,436,968 | 87.0277 | 5,439,247 | 14 + 3 + 1 min |
| m_short_cg_r3 | c18_cg_r3 | 9,223,945 (+562,139) | 5,654,864 | 86.5078 | 5,657,391 | 18 + 3 + 1 min |
| m_short_cg_r3_refill1500 | c19_cg_r3_refill1500 | c18 + BFS cycles | 5,739,014 | 86.3070 | - | 26 min |
| m_short_cg_r4 | c1r4_cg_r4 | 9,793,845 (+569,900) | 5,784,581 | 86.1983 | 5,787,722 | 17 + 4 + 1 min |
| m_short_cg_r4_refill1500 | c1r4_cg_r4_refill1500 | c1r4 + BFS cycles | 5,825,201 | 86.1014 | - | 26 min |
| m_short_cg_r5 | c1r5_cg_r5 | 10,367,399 (+573,554) | 5,855,428 | 86.0293 | 5,858,603 | 16 + 4 + 1 min |
| **m_short_cg_r6** | **c1r6_cg_r6** | 10,941,007 (+573,608) | **5,890,045** | **85.9467** | 5,892,487 | 15 + 8 + 1 min |

Times: pricing + PDHG + rounding/verify. Pricing of round k uses the PDHG duals of round k-1
(round 1 used the 400-iteration duals), 20k sources in r1, 30k in r2, 40k from r3 on, a new seed
each round (seed = round - 1). Pricing logs `logs/m_short_price_r*.log`, stats
`results/m_short_price_r*.json`; PDHG logs `logs/m_short_pdhg*.log`. Cycle lengths found by
pricing drift up each round: mostly 4-5 in r1, 7-8 in r6 (longest 25).
LB gain per CG round: +275k, +271k, +218k, +130k, +71k, +35k.

## Verified numbers vs estimates

Verified (verifier PASS, `src/verify_cert.py`): every LB in the table. Best: c1r6_cg_r6,
LB = 5,890,045, ceiling 85.9467 %. Gap to our order (6,597,734): 707,689 syn = 1.69 pp of W.
Gap to the leaderboard top (6,367,908): 477,863 syn.

Estimates (float, not proven): the LP dual column. For the joint 2+3-cycle LP the PDHG run
brackets the optimum in [4,889,601, 4,889,963] (fractional primal / Lagrangian dual), so the
2+3-cycle LP is 4.89 M, about 289k above the best greedy (scout, 2-cycles first, 4,600,674). The 2+3 LP alone gives 88.33 %. The rounding loss is tiny
(D = 10,000; floor loses about 190 syn, refill takes back 200-900 syn).
For the larger pools the dual estimate is only for that pool; it is not a bound on the full cycle
LP, because pricing still finds many violated cycles (r6: 582,883 from 40,000 sources, cap 20 per source).

How far longer cycles can go (guess): the CG gains fall by about half per round
(+130k, +71k, +35k), so with this pricing scheme I expect another 30-60k, i.e. a full cycle
packing LP near 5.92-6.0 M (ceiling about 85.7-85.9 %). Not measured: a full pricing pass over
all 136k sources until no violated cycle is left, which would also give the cycle-LP value tau*
as a rigorous-ish dual. Even then about 0.6-0.7 M syn (1.5 pp) to our order would remain, which
would be the cycle-LP integrality gap plus the heuristic gap of our order.

## What did not work

- MWU / Garg-Konemann in numpy: correct but slow convergence (the dual estimate fell from 6.9 M to
  5.4 M in 400 rounds while PDHG reached a 0.25 % gap in the same time). Its first primal
  (scale all y by the max congestion) was very weak (1.8 M at round 200); per-column scaling
  fixed that (4.6 M at round 200). Logs `logs/m_short_mwu23_e*.log`, no certificate.
- Joint greedy over 2+3-cycles, largest bottleneck first: 4,556,165, below the scout's
  2-cycles-first greedy (4,600,674). Greedy order matters more than the joint column set.
- No rigorous dual bound for the full cycle LP: pricing still finds about 15 violated cycles per
  sampled source, so the dual estimates above hold only for the column sets listed.

## What I would try next

- Converge the column generation: price ALL sources per round (about 1.9 h on 4 cores at the
  current 20 sources/s; parallelize to 6+ cores or a server), stop when no cycle with
  l(C) < 1 is found. That gives tau* and the last 30-60k of LB.
- Cheaper pricing: the duals are almost 0/1, so violated cycles are cycles in the unsaturated
  subgraph; BFS on that subgraph instead of Dijkstra, and k per source above 20 (the cap was hit
  in r4-r6).
- Prune the pool between rounds (drop columns with y = 0 and positive reduced cost) to keep PDHG
  at about 0.2 s per iteration; pool is 10.9 M columns now.
- Integer refill (`m_short_refill.py`) is worth less as CG improves (+277k on c15, +84k on c18,
  +41k on c1r4 per 1500 s), so it is a final polishing step only.
- To close the remaining 0.6-0.7 M, cycle packing is not enough: stronger certificates on pieces
  (K2, R4/R6) where the LP duals are fractional. R6 (heavy-layer split) was not tried.
