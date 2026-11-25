# Method CLOSE: closing the cycle-LP bracket (H9)

Date 2026-11-25. Worktree base commit b01ab22 (the files of this method are not committed). MacBook, CPU only,
6 threads, shared with two other agents. W = 41,912,141. Ceiling % = 100 (W - LB) / W.
Start: LB 6,138,414 (c34_fin, `M_CG2.md`), tau* <= 6,192,600.95 (`M_DUAL.md`), width 54,187.

## Result

    6,141,103 <= tau* <= 1228406814123/200000 = 6,142,034.07      (width 931 syn)

- **LB 6,141,103**, certificate `ceiling/certs/c45_e.npz`, `verify_cert.py` PASS (in the run and re-run
  standalone, `logs/m_close_recheck_final.log`). Ceiling **85.3477 %** FF (was 85.3541 %). Gain +2,689.
- **UB 6,142,034.07** (exact Fraction above), integer lengths X (scratch `close/X_5_e.npz`) with min cycle
  L = S = 10^6, exact check `m_dual_check.check` (int64 Dijkstra from every vertex, sum w X in Python int).
  Checked twice: inside run 5_e and again from the saved file (`logs/m_close_recheck_final.log`).
  Gain -50,567 vs 6,192,600.95.
- So no cycle packing can certify a ceiling below 85.3455 % FF; the current certificate (85.3477 %) is within
  0.0022 pp of the best possible cycle-packing ceiling.
- Width 54,187 -> 931 (58x narrower). tau* is 6.1411-6.1420 M; the leaderboard order (6,367,908) is at least
  225,873 above tau*, our H64 order (6,597,734) at least 455,699.

## Method

Same LP and the same PDHG / certificate code path as CG2 (`m_cg2_core.py` kernels). The change is in pricing and
in tracking the upper bound every round.

1. Chunked repair (`m_close_core.raise_cycles`, `m_close_repair.py`). DUAL's repair raised every H64 backward
   edge b = (u, v) to X_b = max(X_b, S - dist(v, u)) with all distances taken from the input X (one Dijkstra
   per head, all heads in parallel). Here the heads are split into k random groups processed one after the
   other; each group sees the raises of the earlier groups. Still valid: lengths only grow, so after its group
   each b has X_b + dist_final(v, u) >= X_b + dist_then(v, u) >= S, and every cycle has an H64 backward edge.
   On CG2's final duals alone: raise cost 52,872 (k = 1, DUAL) -> 43,574 (k = 8), UB 6,183,302.31
   (`m_close_repair_final_ch8.json`), i.e. -9,299 with no new LP work.
2. Repair cycles as columns. The same Dijkstra keeps parent edges; for every raised b with raise > 10^-3 S the
   cycle (shortest path v ~> u, then b) is written out. It is the most violated cycle through b under the
   current duals. Analysis of DUAL's repair showed where the cost sits: 51k of the 52.9k came from raises of
   0.01-0.5 (about 98k edges), i.e. from a real tail of violated cycles, not from PDHG noise (raises below
   10^-3 cost only 356 syn). These cycles are added to the pool together with short-cycle pricing (H = 6,
   k = 50, as CG2).
3. Round = PDHG 1500 iterations on the pool (warm start) -> repair of the best-dual x + exact check (UB of
   that round) -> add raise cycles (+ short cycles) -> prune (as CG2). Final round: PDHG, repair + check,
   certificate floor(D y) + overload repair + greedy refill (D = 10,000, as CG2), verifier.

Not used: stabilization (option `--alpha` in `m_close_run.py`, never run) and extra-precise PDHG; the plain
loop kept cutting the raise cost by 25-40 % per round, so there was no reason to switch.

## Trajectory (all UBs exact, L >= S, PASS; all LBs verifier PASS)

Pool LP primal / dual est. are float estimates for the pool only (as in CG2). sum w X0 = cost of the rounded
PDHG duals before the raise; UB = (sum w X0 + raise) / L.

| run | round | pool | pool LP primal (est.) | pool dual (est.) | sum w X0 / S | raise cost | UB (exact) | raise cycles added | short added |
|---|---|---|---|---|---|---|---|---|---|
| DUAL | CG2 final | 14,610,798 | 6,138,043 | 6,140,143 | 6,139,729 | 52,872 (k=1) | 6,192,600.95 | - | - |
| repair ch8 | CG2 final | - | - | - | 6,139,729 | 43,574 (k=8) | 6,183,302.31 | - | - |
| 1_a | 18 | 14,610,798 | 6,138,838 | 6,139,903 | 6,139,732 | 44,958 | 6,184,689.32 | 114,171 | 84,028 |
| 1_a | 19 | 14,414,648 | 6,139,063 | 6,140,781 | 6,140,375 | 28,024 | 6,168,399.21 | 92,816 | 191,984 |
| 1_a | 20 (final) | 14,532,819 | 6,139,605 | 6,141,089 | 6,140,734 | 16,239 | 6,156,973.01 | - | - |
| 2_b | 21 (re-solve) | 14,532,819 | 6,140,157 | 6,140,905 | 6,140,749 | 16,471 | 6,157,220.24 | 65,887 | 81,027 |
| 2_b | 22 | 14,526,150 | 6,140,037 | 6,141,260 | 6,140,954 | 11,067 | 6,152,021.29 | 53,703 | 126,858 |
| 2_b | 23 (final) | 14,606,739 | 6,140,289 | 6,141,333 | 6,141,071 | 7,877 | 6,148,948.55 | - | - |
| 3_c | 24 (re-solve) | 14,606,739 | 6,140,653 | 6,141,206 | 6,141,080 | 7,849 | 6,148,929.04 | 39,875 | 45,422 |
| 3_c | 25 | 14,637,754 | 6,140,481 | 6,141,367 | 6,141,148 | 5,663 | 6,146,811.52 | 33,591 | 58,895 |
| 3_c | 26 (final) | 14,686,153 | 6,140,606 | 6,141,391 | 6,141,196 | 4,126 | 6,145,321.68 | 27,321 | - |
| 4_d | 27 | 14,713,474 | 6,140,687 | 6,141,403 | 6,141,229 | 2,906 | 6,144,135.22 | 23,149 | 58,285 |
| 4_d | 28 | 14,762,353 | 6,140,787 | 6,141,399 | 6,141,250 | 2,216 | 6,143,466.15 | 18,495 | 33,465 |
| 4_d | 29 (final) | 14,788,014 | 6,140,919 | 6,141,370 | 6,141,266 | 1,675 | 6,142,940.81 | 15,350 | - |
| 5_e | 30 | 14,803,364 | 6,140,928 | 6,141,384 | 6,141,275 | 1,225 | 6,142,500.31 | 13,015 | 22,799 |
| 5_e | 31 | 14,828,160 | 6,140,973 | 6,141,377 | 6,141,282 | 1,072 | 6,142,354.56 | 10,418 | 12,461 |
| 5_e | 32 (final) | 14,838,738 | 6,141,049 | 6,141,356 | 6,141,287 | 747 | **6,142,034.07** | 8,802 | - |

Rounds 21 and 24 re-solve the pool of the previous run's final round, whose raise cycles were not yet saved
(fixed from run 3_c on: the final round's raise cycles are added to the saved state, outside the certificate).

Bracket per run (certificate / best UB of the run):

| run | certificate | LB | ceiling % | UB | width | wall |
|---|---|---|---|---|---|---|
| start | c34_fin | 6,138,414 | 85.3541 | 6,192,600.95 | 54,187 | - |
| repair ch8 | - | - | - | 6,183,302.31 | 44,888 | 846 s |
| 1_a | c41_a | 6,139,869 | 85.3506 | 6,156,973.01 | 17,104 | 2,931 s |
| 2_b | c42_b | 6,140,483 | 85.3492 | 6,148,948.55 | 8,466 | 2,653 s |
| 3_c | c43_c | 6,140,748 | 85.3485 | 6,145,321.68 | 4,574 | 2,618 s |
| 4_d | c44_d | 6,140,997 | 85.3479 | 6,142,940.81 | 1,944 | 2,723 s |
| lower 4_d | - | - | - | 6,142,455.75 | 1,459 | 1,430 s |
| **5_e** | **c45_e** | **6,141,103** | **85.3477** | **6,142,034.07** | **931** | 2,392 s |

Lowering pass (`m_close_lower.py`, heuristic, the bound rests on the exact check only): on X_4_d set every non-H64-
backward SCC edge e = (u, v) with X_e > 0 to max(0, S - dist(v, u)) (heads in 32 sequential groups), then re-raise
the H64 backward edges (16 groups), exact check. 640,730 edges lowered, -495.9; re-raise +10.8; L = 10^6 PASS:
**UB 49139645981/8000 = 6,142,455.75** (`m_close_lower_4_d.json`, 1,313 s for the pass + start re-check).
Not repeated on X_5_e (time budget); the next CG run alone went below it. A lowering pass on X_5_e would likely
take a few hundred more off the UB (guess, by analogy with 4_d).

## Observations

- The UB side was the loose one. Its gap came from cycles missing in the pool (raise tail 0.01-0.5), and
  pricing exactly those cycles (one per raised H64 backward edge) shrank the raise cost by about 30 % per round:
  45.0k, 28.0k, 16.2k, 11.1k, 7.9k, 5.7k, 4.1k, 2.9k, 2.2k, 1.7k, 1.2k, 1.1k, 0.7k. CG2's all-source Dijkstra pricing had added
  millions of cycles per round with much smaller effect.
- The rounded dual cost sum w X0 rose only 1,555 over the 15 rounds (6,139,732 -> 6,141,287), and the pool LP
  primal 2,211 (6,138,838 -> 6,141,049). So tau* is close to 6.141 M; CG2's pool was already near-optimal for
  the LB, and almost all of the 54k came from the dual side.
- The raise cycles at the end are only mildly violated (dual length quantiles 1 % / 10 % / 50 %: 0.974 / 0.981 /
  0.993 in round 32, vs 0.793 / 0.899 / 0.973 in round 18).
- The LB grows by about 250-600 per run; it is limited by the PDHG precision on the pool (primal-dual gap of the
  pool 450-1,100 syn) and the integer rounding (refill gives back 400-600).

## H9 verdict

Prediction: one more CG campaign from the repaired duals, with pricing focused on the cycles the repair had to
lift, raises LB by >= 25k and/or lowers the UB by >= 10k. Kill: LB gain < 10k and UB gain < 5k after 3 h.

- UB: -50,567 (6,192,600.95 -> 6,142,034.07). Already -24,202 in round 19 (34 min into run 1_a), -35,628 at the
  end of run 1_a, -49,135 at the 3 h mark of the kill clause (6,143,466, round 28, about 2.97 h of cumulated runs).
  Prediction met (>= 10k) by a factor of 5.
- LB: +2,689 only. The LB part of the prediction fails, and it could not have been met: the new UB proves
  tau* <= 6,142,035, so no cycle packing can exceed the old LB by more than 3,620.
- Bracket width 931 syn < 20k: **H9 CONFIRMED** (via the UB side). The cycle-LP value is pinned at
  tau* in [6,141,103, 6,142,035], 85.3455-85.3477 % FF.
- Run time: 15,592 s wall (4.3 h) for all runs together (repair ch8 846, 1_a 2,931, 2_b 2,653, 3_c 2,618,
  4_d 2,723, lowering 1,430, 5_e 2,392 s); no machine sleep (all under `caffeinate -i`).

Consequence for the study (interpretation): the cycle-packing ceiling is now essentially final (85.348 %). The
0.54 pp between it and the leaderboard top (84.8065 %) is either the leaderboard's suboptimality or the
integrality gap of the cycle LP on the full graph; cycle packings cannot tell which. Closing it needs a
relaxation stronger than the cycle LP or better orders (H8 line).

## Files

- Code: `ceiling/src/m_close_core.py` (C kernel `raise_cycles`: chunk-sequential raise + cycle output, mode 1 =
  lowering), `m_close_repair.py` (one repair + check), `m_close_run.py` (driver), `m_close_lower.py`
  (lowering pass). Kernel self-test (scratch `close/t_core.py`): 200 random digraphs, 1 and 3 chunks,
  min cycle after raise >= S, every output cycle a simple closed walk ending in its raised edge: 0 failures.
- Certificates: `ceiling/certs/c41_a.npz`, `c42_b.npz`, `c43_c.npz`, `c44_d.npz`, `c45_e.npz` (all PASS; c45_e is the best).
- Results: `ceiling/results/m_close_{repair_final_ch8,1_a,2_b,3_c,4_d,lower_4_d,5_e}.json`.
  Logs: `ceiling/logs/m_close_{repair_final_ch8,1_a,2_b,3_c,4_d,lower_4_d,5_e,recheck_final}.log`.
- Scratchpad `/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/close/`:
  `X_<tag>.npz` (integer lengths X, S = 10^6, best UB of each run; **`X_5_e.npz` is the UB 6,142,034.07
  witness**; `X_lower_4_d.npz` the lowered 4_d lengths), `state_<tag>.npz` (pool + PDHG iterates to continue from), `duals_<tag>.npz`.

## Reproduce

From the worktree root, `P=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`, `SP` = scratchpad:

    caffeinate -i $P -u ceiling/src/m_close_repair.py $SP/cg2/duals_final.npz x final_ch8 8
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 1_a --state $SP/cg2/state_4_fin.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 2_b --state $SP/close/state_1_a.npz --center $SP/close/X_1_a.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 3_c --state $SP/close/state_2_b.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 4_d --state $SP/close/state_3_c.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 2000 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_lower.py $SP/close/X_4_d.npz 4_d 32 16 1
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 5_e --state $SP/close/state_4_d.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 2000 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2600

`--center` in 2_b only re-checks and carries the previous best UB (alpha = 1, no stabilization). Runs 1_a and
2_b used the driver before the "save final-round raise cycles" change (code otherwise identical). PDHG is
multithreaded floating point and the chunk order of the raise is fixed (seed 0), so reruns give close but not
bit-identical numbers; every quoted bound is re-checkable from its certificate or X file.
