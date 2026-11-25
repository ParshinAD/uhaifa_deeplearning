# Method D: upper bounds on the cycle LP value tau* (fractional feedback arc sets)

Date 2026-11-25. Worktree base commit deecf6b (method files committed in b01ab22).
MacBook, CPU only, at most 6 threads, shared with the CG2 agent. W = 41,912,141.
Our order (H64): feedback 6,597,734. Leaderboard top: 6,367,908. Best certified LB at the time of
writing: 6,138,414 (CG2, `M_CG2.md`).

    tau* = min sum_e w_e x_e  s.t.  x >= 0,  x(C) >= 1 for every directed cycle C.

Any cycle packing proves at most tau*. Any feasible x proves tau* <= w.x. This method builds such x.

## Result

**tau* <= 3096300477189/500000 = 6,192,600.95** (exact check: min cycle L = S = 10^6 under integer
lengths X, sum w X = 6,192,600,954,378; `results/m_dual_repair_cg2_final_heads.json`). X = CG2's final
pool duals plus a one-shot raise on H64 backward edges (+52,872), 11 min in total.

With the certified LB 6,138,414 (CG2) this brackets the cycle LP:

    6,138,414 <= tau* <= 6,192,601     (width 54,187 syn = 0.13 pp of W)

- No cycle packing can prove more than 6,192,601, i.e. no cycle-packing ceiling below 85.2248 % FF.
  The current certificate (85.3541 %) is at most 0.13 pp from the end of this road.
- The leaderboard order (6,367,908) is 175,307 syn (0.42 pp of W, 2.8 % of tau*) above tau*; our order is
  405,133 syn above.
- The extrapolation "tau* about 5.92-6.0 M" (M_SHORT.md) is wrong: tau* >= 6.138 M is already certified.

## Exact checker (`ceiling/src/m_dual_check.py`, kernels in `m_dual_core.py`)

Input: integer lengths X_e >= 0 (float lengths l are turned into X = ceil(S l)). The checker computes
L = min over directed cycles C of X(C) exactly, with int64 Dijkstra from every vertex s restricted to
vertices of larger rank (each cycle is found from its smallest-rank vertex; rank = degree order), pruned by
the best L found so far, 6 pthreads. X / L is feasible, so tau* <= sum_e w_e X_e / L, reported as an exact
Fraction (sum in Python int / int64 with an overflow guard). Edges outside nontrivial SCCs are in no cycle;
methods set them to 0.

Self-test `ceiling/src/m_dual_selftest.py` (log `logs/m_dual_selftest.log`), all PASS:
- 300 random digraphs (2-11 nodes, random int lengths with many zeros): `min_cycle`, point distances,
  shortest cycle through each vertex, k shortest closing cycles per vertex and the stale raise kernel
  against Floyd-Warshall, 1 and 6 threads, 0 mismatches.
- 60 tiny graphs: LP over ALL simple cycles (networkx), X = ceil(S x*): checker L equals the brute-force
  minimum and the UB lies in [tau*, tau* (1 + 1e-4)].
- Calibration instances with known cycle LP (`m_calib_*.json`): the mouse (UB 0.622613 = cycLP) and the
  12 largest pieces of `m_calib_pieces_rc.json` (up to 80 nodes, 4,286 edges): UB = cycLP of the JSON to
  within 1e-4 relative (most exactly equal).

Cost on the full graph: 0/1 lengths (champion backward indicator) 105 s wall, 6 threads. Fractional
lengths (block run below): 757 s CPU (about 2-3 min on 6 threads; the wall time in the JSON, 10,500 s,
includes about 2.8 h when the machine slept, the same sleep CG2 reports).

## Runs

| run (results/m_dual_*.json) | what | S | L | sum w X / S | UB on tau* | time |
|---|---|---|---|---|---|---|
| champion_backward | x = indicator of H64 backward edges | 1 | 1 | 6,597,734 | 6,597,734 | 117 s |
| block_k3000all (merge of block_k3000, block_k3000b) | H64 cut into 46 windows of 3,000 ranks; window cycle LP inside, indicator between windows | 1e6 | 1e6 | 6,581,707.22 | 6581707224547/1000000 = 6,581,707.2 | 76 + 33 min + check |
| repair_pdhg_r6 (estimate) | PDHG duals of the r6 pool (method P) | 1e6 | - | 5,892,113 | estimate 8.05 M (not a bound) | 32 s |
| repair_cg2_final (estimate) | CG2 final duals | 1e6 | - | 6,139,729 | estimate 6.19 M (not a bound) | 62 s |
| **repair_cg2_final_heads** | CG2 final duals + one stale raise on all H64 backward edges | 1e6 | 1e6 | 6,192,600.95 | **3096300477189/500000 = 6,192,600.95** | raise 535 s + check 101 s |

### Block bound (`m_dual_block.py`, `m_dual_merge.py`)

Cut the order into rank windows H_1 < H_2 < ... . X = S on edges going backward between windows,
X = S x_i inside window i, where x_i is feasible for the cycle LP of the window graph, X = 0 elsewhere.
Feasible: a cycle inside one window is covered by x_i; a cycle that leaves its window has to come back to
a lower window, through an inter-window backward edge. So tau* <= F_between + sum_i tau*(H_i).
Window LP: lazy cycle generation with HiGHS, separation = up to 10 shortest cycles through every node
(C kernel), rows that are slack are dropped (rows that come back are kept). Each window X is rescaled by
its exact min cycle; unconverged windows (time cap) are repaired by a raise on the window's backward edges,
or fall back to the order indicator if that is cheaper.

Over all 46 windows of 3,000 nodes: the order pays 1,053,179 inside windows, the window UBs sum to
1,037,152 (gain 16,027 = 1.5 %). On the 42 windows whose LP converged: order 776,080, LP 761,902, best
integral window order known (order or LP rounding) 769,590, so the window LP is 1.0 % below the best known
window order. The windows hold only 16 % of the order's feedback (cycles of span < 3,000 ranks), so this
UB stays near the order: 6,581,707. Windows of 1,000 nodes gave gains of 46-80 syn each (smoke).
Not measured: how much of the 1 % is integrality gap and how much is a suboptimal order inside the window
(`m_dual_winmilp.py`, exact window MILP, was written but not run for lack of time).

### Repair of LP duals (`m_dual_repair.py`)

x = CG/PDHG duals are optimal for a restricted column pool, not feasible for all cycles. Repair rule:
X0 = ceil(S x_e / w_e) on SCC edges; for every H64 backward edge b = (u, v) set
X_b = max(X_b, S - dist_X0(v, u)). Every cycle contains a backward edge, and the distances only grow, so
the result has min cycle >= S even though all raises use the stale X0 (they overpay, never underpay).
`heads` mode runs this with one Dijkstra per head node v (C kernel `raise_heads`), then the exact check.

- PDHG r6 duals (pool LP 5.89 M): 83 % of sampled sources lie on a cycle of length 0, the raise would
  add about 2.15 M (estimate from 2,000 sampled backward edges): UB about 8.05 M, useless.
- CG2 final duals (pool LP 6.140 M): no zero cycles in the sample; shortest cycle through a sampled source
  has median 0.89; 69 % of backward edges need a raise; estimated raise cost +51 k, i.e. UB about 6.19 M.
  Run in full (`heads`): 576,529 edges raised, cost +52,872 (the sample estimate +51,407 was close),
  exact check L = 10^6 = S, UB 6,192,600.95. Effort curve of the dual side: order indicator 6,597,734
  (2 min) -> window LPs 6,581,707 (about 2 h) -> CG2 duals + raise 6,192,601 (11 min on top of CG2's
  duals). The quality of the starting duals decides everything; the repair itself is cheap.

## Interpretation

Observation: tau* is pinned to [6,138,414, 6,192,601]. The leaderboard order pays 6,367,908, 175,307
more than any cycle packing can prove (our order: 405,133 more).

What this does and does not prove. leaderboard - tau* = (leaderboard - OPT) + (OPT - tau*). The first term
is the leaderboard's suboptimality, the second the integrality gap of the cycle LP. Their sum is at least
175,307 syn, proven. How it splits is open: an integrality gap is only proven by a lower bound on OPT above
tau*, and cycle packings cannot give one. So the result is not "the cycle LP has a proven gap of 175 k".
It is: "either the leaderboard is up to 175 k above OPT, or the cycle LP has a gap of that size on this
graph, or a mix; cycle packing alone can never certify the leaderboard order to better than 2.8 %".

Evidence on the split (weak, from smaller graphs):
- Calibration (pieces up to 300 nodes, mouse): cycle LP = OPT within 0.28 %, mean 0.002 %.
- 3,000-node windows of H64 (this method): the window LP is 1.0 % below the best window order we know (the
  order itself or the LP rounding). Exact window optima were not computed, so this 1 % is an upper bound
  on the window integrality gap, not a measurement of it.
- Guess, not measured: if the global integrality gap were about 1 % of tau* (60 k), OPT would be near
  6.25 M and the leaderboard about 0.1 M above OPT. A gap of the full 2.8 % would need OPT >= 6.37 M,
  i.e. the leaderboard optimal; nothing in the calibration points that way, but the long-span cycles that
  carry most of the feedback are exactly what the pieces and windows do not contain.

For the ceiling study: the cycle-packing line is nearly finished (at most +54 k left). Closing the rest
needs either a stronger relaxation than the cycle LP on the full graph or better orders, not more cycles.
Next cheap step on the dual side: a lowering pass on the repaired X (lower each raised edge by its slack)
or repair from the next CG2 duals; both can only move the UB inside the 54 k bracket.

## Side task: orders from the window LPs

Rounding each window LP (cut edges with x >= 0.5, Kahn topological sort by H64 rank, keep the window
order if not better) and keeping H64 between windows gives an order with feedback 6,591,244 (frozen
oracle `score_from_order`, `m_dual_merge` output, log `logs/m_dual_merge_k3000all.log`), 6,490 syn better
than H64 (84.2737 % vs 84.2582 % FF). One deterministic pass, no seeds involved; far from the leaderboard
(6,367,908). The rank vector is in the scratchpad (`mdual/block_k3000all.npz`, key `rank`), not pinned.
Not tried: dual-guided rounding of the full-graph x (delete large-x edges, topological sort, local search).

## Files

- Code: `ceiling/src/m_dual_core.py` (C kernels), `m_dual_check.py`, `m_dual_selftest.py`,
  `m_dual_block.py`, `m_dual_merge.py`, `m_dual_repair.py`, `m_dual_winmilp.py` (not run).
- Results: `ceiling/results/m_dual_*.json`. Logs: `ceiling/logs/m_dual_*.log`, including killed runs
  (`m_dual_block_smoke3000_v2_killed.log`: separation cycling, fixed with sticky rows;
  `m_dual_block_k3000b_v1_killed.log`: no repair for unconverged windows yet).
- Lengths X: scratchpad `mdual/` (`champ_back.npz`, `block_k3000all.npz`, `repair_*.npz`), not in git.

Commands (repo root, `P=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`,
`SP` = scratchpad dir):

    $P -u ceiling/src/m_dual_selftest.py > ceiling/logs/m_dual_selftest.log
    $P -u ceiling/src/m_dual_check.py $SP/mdual/champ_back.npz champion_backward X
    $P -u ceiling/src/m_dual_block.py 3000 0 46 k3000 3300        # sep_tol 1e-6, no window cap (older config)
    $P -u ceiling/src/m_dual_block.py 3000 37 46 k3000b 3000      # sep_tol 1e-4, window cap 400 s, repair
    $P ceiling/src/m_dual_merge.py k3000all champion k3000 k3000b
    $P -u ceiling/src/m_dual_check.py $SP/mdual/block_k3000all.npz block_k3000all X
    $P -u ceiling/src/m_dual_repair.py $SP/mshort/pdhg_r6.npz x pdhg_r6 estimate
    $P -u ceiling/src/m_dual_repair.py $SP/cg2/duals_final.npz x cg2_final estimate
    $P -u ceiling/src/m_dual_repair.py $SP/cg2/duals_final.npz x cg2_final_heads heads
