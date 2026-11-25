# Method W: exact optimum of H64 windows and dense pieces (test of H8)

Date 2026-11-25. Worktree base commit b01ab22 (files of this method are not committed). MacBook, CPU, 1 thread
per run (2 for `heavy41`), up to 5 runs in parallel on a machine shared with two other agents, so runtimes are
indicative only. Not a certificate: LP and MILP values are HiGHS floats. Every reported optimum is checked by the
feedback of a topological order of the piece minus the chosen edges (assert in the code).

## Result in short

- On 71 exactly solved pieces (windows of 300 to 3,000 consecutive H64 ranks, 3 dense reciprocal clusters of 500
  nodes) the cycle LP is 0.016 % below the exact optimum on average, worst 0.27 %, above 0.2 % in 2 pieces.
- On the 34 solved windows of the M_DUAL partition (3,000 ranks each) the sums are: H64 617,644, OPT 609,608,
  LP 609,560. H64 is 1.32 % above OPT, OPT is 0.008 % above LP. So the ~1 % window gap that M_DUAL saw between
  the LP and the order is almost all order suboptimality.
- H8 prediction (LP within 0.2 % of OPT on windows) holds on the solved pieces. The kill line (OPT >= 0.5 % above
  LP on average) is not reached. Caveat: the densest central windows are not solved (see Limits).

## Method (`ceiling/src/m_win_run.py`, table by `ceiling/src/m_win_summary.py`)

Instance = induced subgraph of the connectome on a node set H, node order inside H = H64 order.

- `win:<k>:<part>:<rep>`: k consecutive H64 ranks, start drawn uniformly inside a rank range, seed
  `[20261125, 1, k, part id, rep]` (numpy default_rng). Ranges were set from the dominant super_class per 3,000
  ranks: early 9,000-21,000 (sensory and optic mix), optic 21,000-99,000, central 117,000-129,000, late
  129,000-136,648. k = 300, 600, 1000, 2000, 3000; 2 reps per part for k <= 2000, 1 for k = 3000.
- `at:3000:<start>`: the aligned windows of M_DUAL (`m_dual_block.py`, k = 3000), no randomness.
- `recip:<k>:<rep>`: dense reciprocal cluster as in `m_calib_pieces.py`: seed node drawn from the 2,000
  giant-SCC nodes with the largest reciprocal weight, grown by the largest reciprocal weight to the set; seed
  `[20261125, 2, k, 0, rep]`.

Per instance:

- `F_h64`: feedback of H64 restricted to H (all edges of H).
- `F_ls`: H64 restricted to H, then node insertion local search (each node to its best slot, passes until no
  move gains). Checked inline (not saved as a script) against brute force over all single moves on 200 random
  6-node graphs.
- `F_lpround_ls`: topological order of H minus the LP edges with x >= 0.5, then the same local search.
- `lp`: cycle LP over SCC-internal edges, `m_dual_block.block_lp` (lazy rows, separation = up to 10 shortest
  cycles through every node, tolerance 1e-4, LP time cap 600 s, 1500 s for heavy41). Value of an LP over a subset
  of cycle rows, so always <= the true cycle LP; when converged it is within 1e-4 relative of it.
- `opt`: lazy cycle MILP (scipy 1.10 `milp`, HiGHS, `mip_rel_gap=0`) started from the LP rows; separation = up to
  10 shortest-hop cycles through every node of H minus the chosen edges; optimal when none is left. On time
  limit: lower = max(lp, MILP dual bound over the current rows, optimal values of earlier rounds), upper = best
  order found (F_h64, F_ls, F_lpround_ls, local search on the best rounded MILP incumbent).

Cross-check: on two smoke pieces (300 and 1,000 nodes) the exact Floyd-Warshall cycle LP and lazy MILP of
`m_calib_core.py` give the same LP and OPT (1089 / 1089 and 4344 / 4344).

Script changes during the campaign (all runs from the same file, commit b01ab22 plus local edits):
`F_lpround_ls` added after the smoke run; `at:` specs added before the `al3000*` runs; local search on the best
rounded MILP incumbent (only changes the upper bound of unsolved instances) added before `hard_*`, `al3000c`,
`heavy41`, `dense_b`; optional threads / LP cap arguments added before `heavy41`. Instances that timed out were
re-run with more time in `hard_*` runs; the summary keeps the narrowest bracket per instance. Run `al3000b` was
stopped by hand after 4 windows to stay under 60 min; its remaining windows went to `al3000c` and `heavy41`.

## Table (`logs/m_win_summary.log`)

Sum-based ratios over solved instances of each group.

| group | n | solved | sum OPT | (OPT-LP)/OPT % | worst piece % | (H64-OPT)/OPT % | (LS-OPT)/OPT % | (LPround+LS-OPT)/OPT % |
|---|---|---|---|---|---|---|---|---|
| win 300 | 8 | 8 | 15,789 | 0.0 | 0.0 | 1.805 | 1.805 | 0.0 |
| win 600 | 8 | 8 | 32,814 | 0.106 | 0.271 | 3.188 | 3.105 | 1.012 |
| win 1000 | 8 | 8 | 26,445 | 0.028 | 0.061 | 3.275 | 3.275 | 1.066 |
| win 2000 | 8 | 8 | 78,702 | 0.079 | 0.225 | 2.445 | 2.367 | 0.912 |
| win 3000 | 4 | 3 | 40,165 | 0.001 | 0.001 | 2.268 | 2.159 | 0.025 |
| at 3000 (M_DUAL windows) | 37 | 34 | 609,608 | 0.008 | 0.081 | 1.318 | 1.288 | 0.139 |
| recip 500 | 3 | 2 | 111,300 | 0.01 | 0.012 | 23.862 | 1.169 | 0.129 |
| recip 1000 | 1 | 0 | - | - | - | - | - | - |
| recip 2000 | 1 | 0 | - | - | - | - | - | - |

Over all 71 solved pieces: mean per-piece (OPT-LP)/OPT 0.016 %, max 0.271 % (win:600:late:1), next 0.225 %
(win:2000:central:1, needed 793 s). On the 33 solved pieces with >= 5,000 SCC edges the mean is 0.019 %.
Several windows have small SCCs (late and some optic windows: 2 to 110 SCC edges) and add little information.

Unsolved (lower = max(LP, MILP bound), upper = best order found):

| instance | LP converged | LP | lower | upper | H64 | (lower-LP)/lower % | (upper-LP)/upper % | time s |
|---|---|---|---|---|---|---|---|---|
| at:3000:87000 | yes | 7,283.4 | 7,293 | 7,305 | 7,688 | 0.132 | 0.296 | 2,400 |
| at:3000:81000 | yes | 4,649.9 | 4,660 | 4,721 | 4,968 | 0.216 | 1.506 | 1,600 |
| at:3000:123000 (heavy41) | yes | 38,531.4 | 38,546 | 40,487 | 40,581 | 0.038 | 4.83 | 3,300 |
| recip:500:2 | yes | 88,060.9 | 88,089 | 89,648 | 100,092 | 0.032 | 1.77 | 1,500 |
| recip:1000:0 | no (600 s cap) | 110,019.5 | 110,026 | 110,200 | 133,103 | - | 0.164 | 1,126 |
| win:3000:central:0 | no | 52,503.1 | 52,503.1 | 56,356 | 56,430 | - | - | 896 |
| recip:2000:0 | no | 292,888.2 | 292,889 | 323,744 | 350,329 | - | - | 1,129 |

recip:1000:0 is informative even without LP convergence: its OPT is within 0.16 % of a lower bound. The last
two give no usable number. In heavy41 and recip:500:2 the MILP did not finish its first round, so the upper bound
is only a heuristic order.

## H8 verdict

Observation: wherever the exact optimum (or a bracket of width <= 0.3 %) was reached, the cycle LP is within 0.3 %
of it, 0.016 % on average; the integrality gap shows up mainly in dense central and late windows (0.1-0.3 %). On the
same windows H64 is 1.3-3.3 % above the window optimum (group sums). Insertion local search barely moves H64 (it
improves 19 of 73 windows, by at most 0.48 % of the window feedback), so H64 is close to insertion-locally
optimal but clearly not optimal inside windows.

Verdict: H8 SUPPORTED on windows up to 3,000 nodes and on 500-node dense clusters. The kill line is not reached.
Interpretation (cautious): on these pieces the gap between the LP and the order is order suboptimality, not
integrality gap. If the same holds for the long-span cycles of the full graph, the true OPT is close to tau*
(6.14-6.19 M) and most of the 175k between tau* and the leaderboard is leaderboard suboptimality. This
transfer is not tested here.

## Limits

- Pieces are induced subgraphs: their optimum ignores all edges that leave the piece. For contiguous rank windows
  that is harmless for the order (edges leaving the window keep their direction under any reordering inside it),
  but the window LP and OPT only see cycles that stay inside the window. M_DUAL found that windows of 3,000 ranks
  hold only 16 % of H64's feedback (1,053,179 of 6,597,734); the remaining 84 % lies on long-span backward edges
  whose cycles leave every window. Nothing here measures the integrality gap of those long cycles.
- Selection bias: the hardest pieces are exactly the ones not solved. Of the 46 M_DUAL windows, 9 dense central
  windows (starts 90,000-96,000, 105,000-108,000, 117,000-129,000; M_DUAL F_pi 382,298 in total, about 36 % of
  the in-window feedback) were not run, because their LP alone did not converge in 400 s on 6 threads in M_DUAL;
  the two tried (win:3000:central:0, at:3000:123000) did not finish. The gap there could be larger.
- Few dense clusters (3 at 500, 1 at 1,000 bracketed); the 2,000-node cluster gave no result.
- Sums over windows are not a statement about the full graph: window optima do not add up to the full-graph
  optimum, and window LPs do not add up to tau*.
- HiGHS floats; optimal pieces are checked by an integer order feedback, LP values are not exact.

## Bonus: how much would exact window re-solving gain for the optimizer track

For a contiguous rank window, any reordering inside it changes the full-graph feedback by exactly the change of the
window feedback, and disjoint windows add up. On the 37 M_DUAL windows that were run, replacing H64 inside each
window by the best order found lowers the full-graph feedback by 8,760 (exact, achievable now: the window orders
are optimal for 34 windows); the most any reordering inside these 37 windows can gain is 10,774 (H64 minus the sum
of lower bounds). That is 0.13-0.16 % of H64's feedback. The 9 missing central windows hold 382,298 of window
feedback; at the observed 1.3-3 % excess they might add roughly 5-11k (estimate, not measured). So exact
re-solving of 3,000-rank windows is worth about 14-22k (partly estimated), small against H64's 229,826 gap to the leaderboard: the
rest of the order's excess sits in long-span structure that windows of this size do not see. Larger windows
(central 2,000-3,000 already take 10-55 min each) would need a faster exact method.

## Files

- `ceiling/src/m_win_run.py`, `ceiling/src/m_win_summary.py`
- `ceiling/results/m_win_<run>.json` (command, git commit, config, seeds, per-instance numbers, node lists,
  MILP round history, solver messages, runtimes): smoke, small, w2000, w3000a, al3000a, al3000b, al3000c,
  dense_a, dense_b, hard_c2000a, hard_c2000b, hard_al3000, hard_al51000, hard_al87000, heavy41
- `ceiling/logs/m_win_<run>.log` for the same runs (the smoke run printed to the terminal, no log),
  `ceiling/logs/m_win_summary.log`
