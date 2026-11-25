# Method CG2: column generation on the cycle packing LP with C pricing (H7)

Date 2026-11-25. Worktree on branch research/ff-ceiling, base commit deecf6b (ceiling/ files of this
method are not committed; `git_commit` in the JSONs is deecf6b). MacBook, CPU only, 8 threads, shared
with another agent. W = 41,912,141. Ceiling % = 100 (W - LB) / W.
Our order: feedback 6,597,734 (84.2582 %). Leaderboard top: 6,367,908 (84.8065 %).
Start point: c1r6_cg_r6, LB 5,890,045 (85.9467 %), method P (`M_SHORT.md`).

## Best verified result

**c34_fin: LB = 6,138,414 syn, ceiling 85.3541 %, verifier PASS** (`ceiling/src/verify_cert.py`).
Gain over c1r6_cg_r6: +248,369 syn (-0.59 pp of ceiling).
Remaining gap to our order: 459,320 syn (1.10 pp of W). To the leaderboard top: 229,494 syn (0.55 pp).

## Code

- `ceiling/src/m_cg2_core.py`: C kernels compiled at run time with `cc` and called with ctypes
  (same pattern as `m_route_core.py`), pthreads, no new package.
  - `pdhg`: the PDHG iteration of `m_short_pdhg.py` (diagonal preconditioning, restart to the average),
    multithreaded over columns and over edges (pool stored as CSR, edge -> column transpose).
    About 0.1-0.2 s per iteration on 10-15 M columns with 8 threads, vs about 0.33 s single-thread scipy before.
  - `bounds`: feasible primal y / max(1, congestion) and the Lagrangian dual estimate, as in method P.
  - `price`: Dijkstra from each source under l_e = x_e / w_e + floor, binary heap with keys inside,
    stopped after k cycles, or when the distance passes 1 - min closing length, or when all
    in-neighbours of the source are popped. Optional pop budget (not used in the runs).
  - `price_short`: new short-cycle pricer, meet in the middle. For source s only nodes with index > s
    are used (each cycle is searched from its smallest node). Best forward paths with exactly 1, 2, 3
    edges and best backward paths with 1, 2, 3 edges are joined into cycles of 2 to 6 edges (plus
    5-cycles through one middle edge). The k best cycles with l(C) < 1 - 1e-4 are kept per source.
    All 136,648 sources take about 8 s (H = 5) or 45 s (H = 6) with 8 threads.
  - All found cycles are rotated to start at the smallest edge id and hashed (64 bit) for dedup
    against the pool. A hash collision can only drop a column, never make a certificate wrong.
- `ceiling/src/m_cg2_run.py`: driver. Rounds of [PDHG on the pool, price, add new columns, prune
  columns with y = 0 and dual length > 1.05 for 2 rounds in a row], then a last PDHG, then the
  certificate exactly as in `m_short_cert.py` (y_int = floor(D y) with D = 10,000, overload repair,
  greedy integer refill over the pool in order of dual length), `verify_cert.py`, run JSON, state file
  (pool + iterates) in the scratchpad to continue from. Options tried: `--route-seed` (add the cycles of
  a K1 certificate as columns), `--resid-phases` (route cycles in the residual graph of the primal
  with the method-R kernel), `--sat-weight` (search length + floor * weight * primal load / w).
- `ceiling/src/m_cg2_check.py`: convergence check. Exact pricing (true lengths, no floor, no pop limit)
  from every source, first violated cycle per source.

## Runs

All certificates verified PASS. Times are wall times on the shared machine.

| run (results/m_cg2_*.json) | what | rounds | pool at end | pool LP primal (est.) | pool LP dual (est.) | LB verified | ceiling % | wall |
|---|---|---|---|---|---|---|---|---|
| smoke | 2,000 sources, 200 PDHG it, pipeline test | 1 | 10,970,794 | 5,888,695 | 5,898,250 | 5,890,802 (cert deleted, equal to start) | 85.9449 | 114 s |
| **0_r1** | start pool r6 + 548,273 new cycles of c21_route_improve_r1; Dijkstra pricing, all 127,550 sources, k 30, floor 0.001 | 2 | 10,312,196 | 6,041,371 | 6,047,545 | **6,042,797** (c30_r1) | 85.5822 | 48 min |
| 1_s5 | short pricing H = 5, k 50 | 5 (killed, no JSON, log only) | - | 6,117,289 | 6,122,038 | - | - | 8 min |
| **2_s6** | from 0_r1, short pricing H = 6, k 50, PDHG 600 it per round | 9 | 12,528,514 | 6,133,569 | 6,136,841 | **6,134,076** (c32_s6) | 85.3644 | 21 min |
| **3_dj** | from 2_s6, short H 6 + Dijkstra 40,000 sources, k 30, floor 0.01 | 4 | 14,725,318 | 6,135,534 | 6,140,539 | **6,136,666** (c33_dj) | 85.3583 | 27 min |
| **4_fin** | from 3_dj, short H 6 + Dijkstra 40,000 sources, floor 0.001, sat-weight 20, PDHG 1200 it | 2 | 14,610,798 | 6,138,043 | 6,140,143 | **6,138,414** (c34_fin) | 85.3541 | 14 min of compute (*) |

(*) run 4_fin shows t = 11,216 s because the machine slept for about 2.9 h during its second PDHG
(log: 835 s at PDHG iteration 1000, 11,176 s at iteration 1100). The run then hit its time limit and
skipped the planned 3000-iteration final PDHG, so the certificate comes from the 1100-iteration primal.

Pool LP primal / dual are float estimates for the pool only (primal = feasible fractional packing,
dual = Lagrangian bound for the columns in the pool). They are not bounds on the full cycle LP.

### Per-round trajectory (pool LP primal estimate after the PDHG of that round)

| round | run | new columns from the previous pricing | pool LP primal | gain |
|---|---|---|---|---|
| r6 (method P) | - | - | 5,889,594 | - |
| 0 | 0_r1 | +548,273 route cycles | 6,002,952 | +113,358 |
| 1 | 0_r1 | +2,444,532 Dijkstra (mean length 13 edges) | 6,027,757 | +24,805 |
| 2 | 0_r1 | +2,442,032 Dijkstra | 6,041,371 | +13,614 |
| 3 | 2_s6 | (re-solve) | 6,042,357 | +986 |
| 4 | 2_s6 | +3,499,018 short | 6,066,943 | +24,586 |
| 5 | 2_s6 | +3,022,307 short | 6,094,247 | +27,304 |
| 6 | 2_s6 | +2,382,904 short | 6,112,114 | +17,867 |
| 7 | 2_s6 | +1,940,850 short | 6,121,355 | +9,241 |
| 8 | 2_s6 | +1,597,446 short | 6,127,185 | +5,830 |
| 9 | 2_s6 | +1,237,127 short | 6,130,501 | +3,316 |
| 10 | 2_s6 | +976,753 short | 6,132,393 | +1,892 |
| 11 | 2_s6 | +686,272 short | 6,133,569 | +1,176 |
| 12 | 3_dj | (re-solve) | 6,134,578 | +1,009 |
| 13 | 3_dj | +428,594 short, +633,773 Dijkstra | 6,133,847 | -731 (PDHG not converged) |
| 14 | 3_dj | +580,570 short, +561,452 Dijkstra | 6,134,308 | +461 |
| 15 | 3_dj | +486,104 short, +452,402 Dijkstra | 6,135,534 | +1,226 |
| 16 | 4_fin | (re-solve, 1200 it) | 6,137,765 | +2,231 |
| 17 | 4_fin | +228,490 short, +169,835 Dijkstra | 6,138,043 | +278 |

The run-1_s5 trajectory (H = 5, killed): 6,042,357, 6,073,457, 6,099,368, 6,111,201, 6,117,289 for
rounds 3-7, slightly ahead of H = 6 for two rounds and behind from round 7 on (one run each).

## Observations

- Seeding with the method-R cycles was the largest single step: 548,273 new short cycles (length
  up to 11) raised the pool LP by 113k. The 2.4 M cycles from all-source Dijkstra pricing added only
  25k and then 14k: Dijkstra under the floor-0.001 lengths returns long cycles (mean 12-14 edges,
  up to 64) through the many x = 0 edges, and almost none of them get y > 0 (state after 0_r1:
  of 2.6 M columns with 13 to 21 edges only about 1,100 have y > 0; the y mass sits on 2- to 7-cycles).
- Short-cycle pricing (cycles of at most 6 edges, best per meeting node) gave +92k in 8 rounds at
  about 2.5 min per round. Its gains halve every round; by round 11 the short cycles found are only
  barely violated (mean dual length 0.993-0.996).
- Residual-graph routing (method-R kernel on the slack of the fractional primal) found almost
  nothing: 200-350 syn per hop phase and slow at H >= 4 (smoke `m_cg2_smoke_resid.log`). Near the
  pool optimum the residual graph has few cycles; the gains need exchanges, not augmentations.
- Order-guided pricing (cycles = one H64 backward edge + forward path; test on the 2_s6 duals,
  scratch script, 2,000 sources): fewer than 1 violated cycle per source, mean dual length 0.95. Under
  these duals the one-backward-edge cycles are almost all satisfied; the violated cycles pass several
  H64 backward edges.
- Sat-weight pricing (prefer edges with primal slack) found 4 violated cycles per source instead of 15
  and did not change the trend (+278 in round 17).
- The PDHG gap per round (0.03-0.08 %, i.e. 2-5k syn) is now larger than the per-round CG gain, so
  late rounds are partly noise.

## Convergence status

CG did NOT converge. Exact check on `duals_final.npz` (`m_cg2_check.py`, true lengths x/w, no floor,
no pop limit, first violated cycle per source, 473 s, `results/m_cg2_check_final.json`): 119,470 of
127,550 sources still close a cycle with l(C) < 1 - 1e-4. Most of these are violated only at the PDHG
precision level (median dual length 0.9993, 90 % quantile 0.9999), but a real tail remains
(1 % quantile 0.786, 10 % quantile 0.874, smallest found 0.686; cycle lengths 2 to 30, mostly 3-5).
So the pool LP value (about 6.138-6.140 M, estimate) is NOT the full cycle-LP value, and these duals
are not feasible for the full cycle LP. Scaling them by the minimum cycle length (x / alpha with
alpha <= 0.686) would give a useless upper bound above 8.9 M; a real upper bound needs a converged or
repaired dual (left to the upper-bound agent).

## H7 verdict

Prediction: certified LB >= 6.10 M within the phase-2 budget; cycle-LP optimum between 6.25 M and
6.37 M. Kill: certified LB < 5.95 M.

- Certified LB: 6,138,414 >= 6.10 M. Prediction met, kill line not hit. Note: the method that got
  there is column generation with PDHG and short-cycle pricing, seeded with method-R cycles, not
  Garg-Konemann as written in the claim.
- Cycle-LP optimum in [6.25, 6.37] M: not tested. What we have: the full cycle LP is at least
  6,138,414 (certified). The pool LP stalls near 6.140 M (dual estimate for the pool), but pricing still
  finds violated long cycles, so the full cycle LP may be higher; we have no upper bound on it here.
  The final edge lengths are saved for the upper-bound agent (see below).

## Files

- Certificates: `ceiling/certs/c30_r1.npz`, `c32_s6.npz`, `c33_dj.npz`, `c34_fin.npz` (all PASS).
- Run JSONs: `ceiling/results/m_cg2_{smoke,0_r1,2_s6,3_dj,4_fin,check_final}.json`.
- Logs: `ceiling/logs/m_cg2_{smoke,smoke_resid,0_r1,1_s5,2_s6,3_dj,4_fin,check_final}.log`.
- Scratchpad `/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/cg2/`:
  `state_<tag>.npz` (pool CSR cptr/cedge, PDHG iterates y and x, prune ages), `duals_<tag>.npz`
  (best-dual x of each run), and **`duals_final.npz`**: edge-indexed float64 `x` (5,657,719 entries,
  sum 6,139,722.5) from run 4_fin. Edge lengths for the cycle LP are l_e = x_e / w_e; the run's
  Lagrangian dual estimate for its pool is in `dual_estimate_pool` (6,140,143.3).

## Reproduce

From the worktree root, with the `allen` python:

    python ceiling/src/m_cg2_run.py --tag 0_r1 --init --route-seed ceiling/certs/c21_route_improve_r1.npz --rounds 2 --k 30 --pdhg-iters 1000 --pdhg-final-iters 1500 --time-limit 3000
    python ceiling/src/m_cg2_run.py --tag 2_s6 --state <scratch>/cg2/state_0_r1.npz --rounds 30 --k 0 --short-H 6 --short-k 50 --pdhg-iters 600 --pdhg-final-iters 1500 --time-limit 2900   (stopped by the stop_2_s6 file after round 11)
    python ceiling/src/m_cg2_run.py --tag 3_dj --state <scratch>/cg2/state_2_s6.npz --rounds 30 --k 30 --n-sources 40000 --floor-len 0.01 --short-H 6 --short-k 50 --pdhg-iters 600 --pdhg-final-iters 1500 --time-limit 2700 --seed 100   (stopped after round 15)
    python ceiling/src/m_cg2_run.py --tag 4_fin --state <scratch>/cg2/state_3_dj.npz --rounds 2 --k 30 --n-sources 40000 --floor-len 0.001 --sat-weight 20 --short-H 6 --short-k 50 --pdhg-iters 1200 --pdhg-final-iters 3000 --pdhg-tol 0.004 --time-limit 3000 --seed 200
    python ceiling/src/m_cg2_check.py <scratch>/cg2/duals_final.npz check_final

`--init` reads method P's pool_r6.npz and pdhg_r6.npz from `<scratch>/mshort/`. The PDHG is
multithreaded floating point, so a rerun gives close but not bit-identical numbers.

## What I would try next

- Converge the pool LP properly (several thousand PDHG iterations) before each pricing round; the
  late rounds are below the PDHG noise.
- Dual stabilization (price with a smoothed dual, e.g. the average of the last rounds) against the
  degenerate x = 0 edges that make Dijkstra return long cycles.
- More method-R style columns: the route cycles were worth far more per column than priced ones.
  Rerun the router on other guide orders or from the LP residual with integer scaling, and add them.
- Stronger than the cycle LP (K2 pieces) for the remaining 0.46 M to our order.
