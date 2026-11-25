# Calibration arm (method C): how tight is each relaxation where the optimum is known?

Date 2026-11-25. MacBook (Apple Silicon), env `allen`, run from the repo root. Shared machine (load 4 to 13
during the runs), so runtimes are indicative only. No number here is a certificate: LP values are HiGHS
floats. They measure how tight each relaxation is, to choose which certificate kind to scale.

## Method

For every instance (local graph G with weights w), `ceiling/src/m_calib_core.py` computes:

1. `lb2`: 2-cycle bound, sum over reciprocal pairs of min(w_uv, w_vu).
2. `g23`: greedy 2+3-cycle packing. Exact 2-cycle packing, then all directed 3-cycles of the residual,
   packed in order of largest initial min-capacity (same rule as the scout's best greedy).
3. `cycLP`: full cycle LP, min w.x s.t. x(C) >= 1 for every directed cycle C, x >= 0 (value = max
   fractional cycle packing). Lazy generation: solve the LP over the current cycles, run Floyd-Warshall
   with lengths x, add for every edge (u,v) with x_uv + d(v,u) < 1 - 1e-9 that cycle, repeat until none.
   The separation is exact, so the final value is the LP over all cycles. LP dual sum checked equal.
4. `triLP`: ordering LP on each nontrivial SCC, x_ij for i<j, all 3-dicycle inequalities
   0 <= x_ij + x_jl - x_il <= 1 for every node triple (complete digraph), added lazily until none is
   violated. Skipped above 120 SCC nodes (310 s at 200 nodes); there it is only reported as "implied"
   when cycLP = opt, since cycLP <= triLP <= opt.
5. `opt`: exact minimum feedback. Lazy-cycle MILP (binary y_e, cycle rows until G minus y is acyclic,
   `mip_rel_gap=0`), cross-checked by the feedback of a topological order of G minus y, by the ordering
   MILP with lazy triangles (SCCs <= 120 nodes) and by subset DP (SCCs <= 20 nodes). All methods that
   finished agree on every instance (0 disagreements).

`python ceiling/src/m_calib_core.py` is a self-test: 60 random graphs of 3 to 7 nodes, all bounds and all
three exact methods against brute force over all permutations, and lb2 <= g23 <= cycLP <= triLP <= opt.
It passes.

Reported per instance: each bound as a share of opt, and "gap closed" = (bound - lb2) / (opt - lb2).

### Instances

- Mouse meso, 148 nodes, float weights (`m_calib_mouse.py`).
- Induced subgraphs of the fly connectome (`m_calib_pieces.py`), selection seed per instance =
  20261125 + kind offset (recip 0, celltype 1000, bfs 2000, window 3000) + size index * reps + rep, stored
  in the JSON together with the node list:
  - `recip`: seed node drawn from the 2,000 giant-SCC nodes with the largest reciprocal weight, then grow
    by the largest reciprocal weight to the set (dense reciprocal clusters).
  - `celltype`: a random cell type with 3 to 10 members in the giant SCC (FlyWire annotations,
    `data/external/flywire_neuron_annotations.tsv.gz`), then partners by largest total weight to the set.
  - `bfs`: undirected BFS ball in the giant SCC from a random root, neighbors in random order.
  - `window`: consecutive nodes of the H64 champion order.
  Sizes 15, 18, 20, 30, 45, 60, 80 with 2 reps per kind (56, runs `wb`, `rc`); 16 and 20 with 20 reps for
  recip, celltype, bfs (120, run `small`); 120, 200, 300 with 2 reps (runs `big`, `bigbfs`, `bigct`).
  Because seeds restart per run, pieces of the same kind and seed in different runs are nested
  (for example recip seed 20261125 at 15, 16 and 120 nodes). Total 192 pieces.
- Pieces whose exact MILP hit the 120 s limit were re-solved by `m_calib_retry.py` (300 s, start from the
  cycles with positive packing weight). If still unsolved the optimum is bracketed: lower = max(cycLP,
  MILP dual bound over a subset of cycle rows, which is a relaxation), upper = H64 order on the piece.

## Results

### Mouse (148 nodes, W 9.159036)

| bound | feedback | share of opt | gap closed | FF ceiling % |
|---|---|---|---|---|
| lb2 | 0.515373 | 0.8278 | 0 | 94.373065 |
| g23 | 0.586634 | 0.9422 | 0.6645 | 93.595020 |
| cycLP | 0.622613 | 1.0000 | 1.0000 | 93.202194 |
| triLP | 0.622613 | 1.0000 | 1.0000 | 93.202194 |
| opt | 0.622613 | 1 | 1 | 93.202194 |

Exact optimum 0.622613 by the lazy-cycle MILP (1 round, starting from the 517 cycles of the cycle LP) and
by the ordering MILP with lazy triangles (13 rounds, 39,363 rows). The topological order of G minus the
MILP solution scores 93.202194 % with the frozen oracle `score_from_order`. This agrees with the scout
(93.202194 %). opt - cycLP = -1.1e-16, so the cycle LP is integral on the mouse. H63 champion: 93.175383 %.

### Fly pieces, solved instances with opt > 0

Mean share of opt (minimum in brackets) and mean gap closed over pieces with opt > lb2.

| size | kind | pieces | solved, opt>0 | lb2 share | g23 share | cycLP share | g23 gap closed | cycLP gap closed | cycLP < opt |
|---|---|---|---|---|---|---|---|---|---|
| <=20 | recip | 46 | 46 | 0.934 | 0.991 | 1.0 (1.0) | 0.864 | 1.0 | 0 |
| <=20 | celltype | 46 | 46 | 0.871 | 0.989 | 1.0 (1.0) | 0.897 | 1.0 | 0 |
| <=20 | bfs | 46 | 44 | 0.788 | 0.992 | 1.0 (1.0) | 0.940 | 1.0 | 0 |
| <=20 | window | 6 | 3 | 0.900 | 1.0 | 1.0 (1.0) | 1.0 | 1.0 | 0 |
| 30-80 | recip | 8 | 8 | 0.810 | 0.973 | 1.0 (0.9998) | 0.857 | 0.9995 | 1 |
| 30-80 | celltype | 8 | 8 | 0.771 | 0.948 | 1.0 (1.0) | 0.803 | 1.0 | 0 |
| 30-80 | bfs | 8 | 8 | 0.750 | 0.934 | 1.0 (1.0) | 0.824 | 1.0 | 0 |
| 30-80 | window | 8 | 6 | 0.865 | 0.963 | 1.0 (1.0) | 0.665 | 1.0 | 0 |
| 120-300 | recip | 6 | 4 | 0.817 | 0.959 | 1.0 (1.0) | 0.791 | 1.0 | 0 |
| 120-300 | celltype | 4 | 2 | 0.671 | 0.897 | 0.9986 (0.9972) | 0.671 | 0.9967 | 1 |
| 120-300 | bfs | 6 | 6 | 0.747 | 0.913 | 0.9999 (0.9996) | 0.686 | 0.9998 | 1 |
| all <=20 | | 144 | 139 | 0.866 | 0.991 | 1.0 (1.0) | 0.898 | 1.0 | 0 |
| all 30-80 | | 32 | 30 | 0.795 | 0.954 | 1.0 (0.9998) | 0.800 | 0.9999 | 1 |
| all 120-300 | | 16 | 12 | 0.758 | 0.926 | 0.9997 (0.9972) | 0.719 | 0.9993 | 2 |

"solved, opt>0" excludes acyclic pieces (opt = 0; 5 windows, 2 BFS balls) and the 4 pieces whose MILP hit
the time limit in the main run. triLP was computed on 182 pieces (SCC <= 120 nodes) and never exceeded
cycLP by more than 1e-6.

### Pieces where the cycle LP is below the optimum (all found)

| piece | m | lb2 | g23 | cycLP | triLP | opt | opt - cycLP | relative |
|---|---|---|---|---|---|---|---|---|
| recip 45, seed 20261133 | 1,088 | 7,446 | 7,773 | 7,804.5 | 7,804.5 | 7,806 | 1.5 | 0.019 % |
| celltype 120, seed 20262125 | 2,067 | 5,202 | 7,925 | 8,909 | 8,909 | 8,934 | 25 | 0.28 % |
| bfs 300, seed 20263129 | 3,974 | 2,635 | 3,458 | 3,984.5 | n/a | 3,986 | 1.5 | 0.038 % |
| celltype 200, seed 20262127 (retry) | 5,336 | 56,167 | 65,873 | 69,425.5 | n/a | 69,449 | 23.5 | 0.034 % |
| recip 200, seed 20261128 (retry, unsolved) | 5,948 | 30,384 | 41,995 | 45,712.4 | n/a | in [45,717, 52,538] | >= 4.6 | >= 0.010 % |
| recip 300, seed 20261129 (retry, unsolved) | 9,339 | 24,639 | 28,209 | 29,594 | n/a | in [29,597, 37,351] | >= 3 | >= 0.010 % |
| celltype 300, seed 20262125 (retry, unsolved) | 8,620 | 21,865 | 39,016 | 44,602.8 | n/a | in [44,607, 50,701] | >= 4.2 | >= 0.009 % |

On the two pieces where both exist, triLP = cycLP: the triangle inequalities close 0 % of the
integrality gap. The upper ends of the three unsolved brackets come from H64 and are loose; the lower ends
come from the MILP dual bound and prove a gap of at least 4.6, 3 and 4.2 syn.

### H4 check: pieces with <= 20 nodes

144 pieces of 15 to 20 nodes (exact by DP and two MILPs). Exact minimum minus cycle LP: 0 on all 144
(max 0, sum 0). Not one piece of this size has an integrality gap.

## Verdicts

- **H6: confirmed.** Mouse: the cycle LP closes 100 % of the gap between the 2-cycle bound and the optimum
  (prediction >= 70 %, kill < 50 %). Fly pieces: on the 182 solved pieces with opt > 0 the cycle LP is on
  average 0.002 % below the exact optimum feedback, worst piece 0.28 % (prediction: within 5 %).
- **Cycle LP vs optimum on connectome pieces: the integrality gap is real but tiny.** 7 of the 185
  pieces with opt > 0 (182 solved, 3 bracketed) have cycLP < opt, all with 45 or more nodes, the
  largest at 0.28 % of the piece optimum. The gap appears more often in bigger pieces (0 of 139 at <= 20
  nodes, 1 of 30 at 30 to 80, 6 of 16 at 120 to 300 including the 3 bracketed ones), so on the 126,840-node
  giant SCC it may be larger. These data cannot measure that; they give no sign of an integrality gap near
  the 0.5 pp of W that H3 predicts.
- **Triangle (ordering) LP: no gain.** On 182 pieces it never beat the cycle LP, including the two pieces
  with a gap where it was computed. A new certificate kind with an ordering-LP dual is not justified by
  this calibration.
- **H4 (exact pieces beat cycles): no support.** On <= 20-node pieces, the size the K2 verifier accepts,
  the exact piece minimum never exceeds the piece cycle LP. A K2 piece can only add what its own
  integrality gap is, so the expected K2 gain is about 0. Formal kill of H4 needs the fly run, but this
  predicts it.
- **What matters instead: the cycle packing itself.** The greedy 2+3-cycle packing closes 90 % of the
  lb2-to-opt gap on <= 20-node pieces, 80 % on 30-80 and 72 % on 120-300 nodes. On the mouse it is 66 %.
  The share falls with size because longer cycles matter more. The main line for the fly bound is
  LP-optimal packing of long cycles (methods R1/R2/P), not stronger relaxations or exact pieces.

## Caveats

- Small pieces may not represent the giant SCC. An induced piece keeps only the cycles inside it; the
  integrality gap of the cycle LP can grow with graph size (it is unbounded in general), and our largest
  pieces have 300 nodes against 126,840. The trend (more gap cases at larger sizes) points that way.
- The piece shares cannot be carried over to the fly. For example, g23 closes about 70-90 % of the gap on
  pieces, but on the fly g23 = 4.60 M and the leaderboard feedback is 6.37 M; we do not know the fly cycle
  LP, so no ratio from here should be used to estimate it.
- Selection bias: all 4 pieces that the main run could not solve have a gap (3 of them stay unsolved
  after 300 s more), so the averages over solved pieces understate the gap a little. Shares above are means
  over solved pieces only.
- Windows of the H64 order and BFS balls are sparse (several acyclic); dense recip and celltype pieces
  are the harder tests.
- The triangle LP was not computed above 120 SCC nodes. On the 200 and 300-node pieces with a gap we do
  not know whether it would help; on all smaller ones it did not.
- Float LP values; a near-integer like 29,594.0 is HiGHS output, not exact arithmetic.
- Runs killed at the 30 min budget: `big` (after 9 of 18 pieces; the bfs part was rerun as `bigbfs`, the
  celltype 300 part as `bigct`, killed after 1 of 2 pieces) and the first retry (`m_calib_retry_v1.log`,
  replaced by the bracketing version).

## Files

- Code: `ceiling/src/m_calib_core.py` (bounds, exact solvers, self-test), `m_calib_mouse.py`,
  `m_calib_pieces.py`, `m_calib_retry.py`, `m_calib_summary.py`.
- Results: `ceiling/results/m_calib_mouse.json`, `m_calib_pieces_{wb,rc,small,big,bigbfs,bigct}.json`,
  `m_calib_retry.json`, `m_calib_retry_bigct.json`.
- Logs: `ceiling/logs/m_calib_*.log`; `m_calib_summary.log` has the per-instance table.

Commands (repo root, `PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):

    $PY ceiling/src/m_calib_core.py
    $PY -u ceiling/src/m_calib_mouse.py > ceiling/logs/m_calib_mouse.log
    $PY -u ceiling/src/m_calib_pieces.py window,bfs wb > ceiling/logs/m_calib_pieces_wb.log
    $PY -u ceiling/src/m_calib_pieces.py recip,celltype rc > ceiling/logs/m_calib_pieces_rc.log
    $PY -u ceiling/src/m_calib_pieces.py recip,celltype,bfs small 16,20 20 > ceiling/logs/m_calib_pieces_small.log
    $PY -u ceiling/src/m_calib_pieces.py recip,celltype,bfs big 120,200,300 > ceiling/logs/m_calib_pieces_big.log
    $PY -u ceiling/src/m_calib_pieces.py bfs bigbfs 120,200,300 > ceiling/logs/m_calib_pieces_bigbfs.log
    $PY -u ceiling/src/m_calib_pieces.py celltype bigct 300 > ceiling/logs/m_calib_pieces_bigct.log
    $PY -u ceiling/src/m_calib_retry.py > ceiling/logs/m_calib_retry.log
    $PY -u ceiling/src/m_calib_retry.py bigct > ceiling/logs/m_calib_retry_bigct.log
    $PY ceiling/src/m_calib_summary.py > ceiling/logs/m_calib_summary.log
