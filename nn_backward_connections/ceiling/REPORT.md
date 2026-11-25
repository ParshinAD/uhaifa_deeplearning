# A certified ceiling on feedforward weight for the FlyWire connectome

Study report, 2026-11-25. Branch `research/ff-ceiling`. Paths are relative to `ceiling/` unless stated.

## Abstract

The FlyWire minimum feedback arc set (MFAS) challenge asks for an order of the 136,648 neurons that keeps as much synapse weight as possible on forward edges. The total weight is W = 41,912,141 synapses (syn). The optimum is unknown. Before this study the only proven limit was the 2-cycle bound. Every order pays the lighter side of each reciprocal pair, so no order can exceed 93.2828 % feedforward (FF).

We prove that **no order of this graph can exceed 85.3477 % FF**. The proof is a certificate, `certs/c45_e.npz`. It is a fractional packing of 6,748,248 directed cycles with integer multiplicities over D = 10,000. It shows that every order has feedback weight at least LB = 6,141,103 synapses. An independent verifier checks it from the raw edge list in exact integer arithmetic (PASS).

| order or bound | FF % | feedback (syn) | distance to the ceiling |
|---|---|---|---|
| proven ceiling (this study) | 85.3477 | >= 6,141,103 | 0 |
| leaderboard top (C. Westfall, 2026-11-08, fetched 2026-11-25) | 84.8065 | 6,367,908 | 0.5411 percentage points (pp), 226,805 syn |
| repo `data/best_solution/` (Vahidi and Koutis), comparator only | 84.6147 | 6,448,318 | 0.7330 pp (307,215 syn) |
| our H64 order | 84.2582 | 6,597,734 | 1.0895 pp (456,631 syn) |

H64 is our best order, from the project's earlier heuristic optimizer track (H63 is its mouse counterpart). Optimizer variants carry numbers above 9; the hypotheses H1 to H9 of section 5 are a separate list.

We also bound the method itself. Let tau* be the value of the cycle LP (fractional feedback arc set). It is the best bound any cycle packing can give. We show 6,141,103 <= tau* <= 6,142,034.07. So no cycle-packing certificate can push the ceiling below 85.3455 %, and ours is within 0.0022 pp of that limit.

What lies between the leaderboard (84.8065 %) and the ceiling (85.3477 %) is not known. The proven statement is FF(OPT) <= 85.3477 %. On every smaller graph where we could compute the exact optimum, the cycle LP was within 0.3 % (relative) of it (mean 0.016 % on 71 windows and clusters of 300 to 3,000 nodes). On the 34 solved windows of 3,000 consecutive H64 positions, H64 was 1.3 % above the window optimum (1.3 to 3.3 % across window groups). This suggests, but does not prove, that most of the remaining 0.54 to 1.09 pp is order suboptimality rather than an integrality gap of the LP.

## 1. Problem and why weak duality gives certificates

Graph: the FlyWire connectome as loaded by the project's reference loader `src.mfas.io.load_dataset("connectome")` (unchanged during the study) (repo root). n = 136,648 nodes, m = 5,657,719 edges, integer weights (min 2), no self loops, no duplicate edges. 519,895 reciprocal pairs hold 24.9 % of W. The giant strongly connected component (SCC) has 126,840 nodes and 41,278,263 of the weight, so splitting into SCCs barely reduces the problem (`SCOUT.md`).

For an order pi, the feedback F(pi) is the total weight of edges that go backward, and FF % = 100 (W - F) / W. We want a lower bound LB <= min over pi of F(pi). Then every order has FF % <= 100 (W - LB) / W.

Any order puts at least one edge of every directed cycle backward. Otherwise the positions would strictly increase around the cycle. Take cycles C with weights y_C >= 0 so that no edge carries more than its weight: sum over cycles through e of y_C <= w_e. Then in any order the backward weight is at least sum y_C. Each cycle pays from its own backward edge, and the capacities are never exceeded. This is weak LP duality between

    tau* = min sum_e w_e x_e   s.t.  x(C) >= 1 for every directed cycle C,  x >= 0     (fractional FAS)

and its dual, max sum y_C under the capacity constraints (fractional cycle packing). Any packing gives a valid LB <= tau* <= OPT. Any feasible x gives tau* <= w.x, which limits what packings can ever prove. Both sides can be checked without trusting the solver that produced them.

## 2. Certificate format and verifier

Format (`src/CERT_FORMAT.md`). One `.npz` of int64 arrays, no pickles. A header binds the certificate to the graph by the sha256 of the loader's src, tgt and weight arrays. Kind K1 lists cycles as edge-id sequences with integer multiplicities y over a common denominator D. Kind K2 (pieces of at most 20 nodes whose exact minimum the verifier recomputes) exists, but no quoted bound uses it.

The verifier checks three things:
- every cycle is a closed simple directed walk on real edges;
- for every edge the load is at most D w(e), in exact integers;
- the reported bound is LB = floor(sum y / D).

Independence. `src/verify_cert.py` was written in a separate LLM coding session that wrote none of the bound methods (it also produced the trivial 2-cycle certificate c00). Its tests (`tests/test_verify_cert.py`, 106 tests) include adversarial and overflow cases.

Red team (`reviews/REDTEAM_1.md`, a reviewer that had not seen the bound code). Verdict PASS.
- None of 22 crafted attacks made an invalid certificate pass: 21 were rejected, and one (floor vs ceil rounding) is a valid certificate that correctly gets LB 0. The attacks included wrong dtypes, int64 wrap-around, negative ids, non-monotone pointers and duplicate zip members.
- The reviewer parsed the raw CSV itself and reproduced the edge checksum.
- With its own checker it re-checked two earlier certificates, c1r6 and c21 (section 3), and got exactly the same LB.
- In those two certificates every cycle contains a backward edge of H64.
- Minor remark: duplicate zip members are accepted (the last one wins; not a bypass).

Final certificate c45_e:
- D = 10,000, 6,748,248 cycles, longest cycle 22 edges.
- sum y = 61,411,034,003, LB = 6,141,103, ceiling 3577103800/41912141 % = 85.347675 %.
- verify_cert PASS inside the run and again in a separate re-run.
- sha256 `e243142ccf6e3beafb44f69a7e523caa8550997515504d13b45adbe629817766`.
- 110 MB, not in git. Section 10 shows how to rebuild it.

Upper-bound side. An upper bound on tau* is a vector of integer edge lengths X >= 0 and a scale S with min over cycles of X(C) >= S. Then X / S is feasible and tau* <= sum w X / S. `src/m_dual_check.py` (kernels in `src/m_dual_core.py`) computes the exact minimum X-length L of a directed cycle with int64 Dijkstra from every vertex, and the sum in Python ints. Its self-test `src/m_dual_selftest.py` compares it with brute force on 300 random graphs and on the calibration instances.

## 3. Methods in order

### 3.1 Scout: 2-cycles and greedy 3-cycles (`SCOUT.md`)
- 2-cycle bound: LB 2,815,303, 93.2828 %, `certs/c00_2cycle.npz` (PASS).
- Greedy packing of all 2,765,514 3-cycles left after packing 2-cycles: 4,600,674, 89.0231 % (probe, not certified). The certified joint greedy is c10_greedy23_joint, 4,556,165, 89.1292 %.
- Short cycles explain much of the feedback, but 2- and 3-cycles saturate near 89 %. Hypothesis H1 (section 5) was rejected.

### 3.2 Method R: order-guided routing (`results/M_ROUTE.md`)
Each H64 backward edge (u, v) is a commodity with demand w(u, v). It is routed along forward paths v ~> u with growing hop limits, in integer amounts.
- Pure one-backward-edge routing (certificate c20, run g1): 5,604,533, 86.6279 %.
- Plus routing of the leftover demand on the full residual graph (c21_route_improve_r1): **5,843,849, 86.0569 %** (PASS).
- A good order points to useful cycles of any length. Using weaker orders as the guide gave 117k to 204k less.

### 3.3 Method P: LP packing and column generation (`results/M_SHORT.md`)
The LP over all 2- and 3-cycles is solved with the primal-dual hybrid gradient method (PDHG), then 6 rounds of column generation with Dijkstra pricing. The certificate is y_int = floor(D y) with D = 10,000, plus a greedy integer refill.
- 2+3-cycle LP: [4,889,601, 4,889,963], certified 4,889,638 (c13), 88.3336 %.
- Best: c1r6_cg_r6, **5,890,045, 85.9467 %** (PASS).
- The LP over the same cycles beats greedy by about 289k, and pricing adds longer cycles. Not converged.

### 3.4 Calibration (`results/CALIBRATION.md`)
We measured tightness where the optimum is known (section 6). The cycle LP was equal or very close to OPT, and the triangle LP and K2 pieces added nothing. This set the main line: solve the full cycle LP.

### 3.5 CG2: second column-generation campaign, close to convergence (`results/M_CG2.md`)
C kernels for PDHG and pricing, a meet-in-the-middle short-cycle pricer (cycles up to 6 edges), and seeding with the 548,273 method-R cycles.
- Best: c34_fin, **6,138,414, 85.3541 %** (PASS, also in a separate re-run).
- The method-R seed was the largest single step (+113k on the pool LP), and short-cycle pricing added about 92k. Most packing mass sits on 2- to 7-edge cycles.
- This certified LB above 6.0 M refuted method P's extrapolation of tau* at 5.92 to 6.0 M.
- Not converged: exact pricing still found violated cycles from 119,470 of 127,550 sources.

### 3.6 DUAL: rigorous upper bound on tau* (`results/M_DUAL.md`, `reviews/REVIEW_DUAL.md`)
We repair the approximate CG2 dual lengths x into exact integer lengths. First X0 = ceil(S x / w). Then every H64 backward edge b = (u, v) is raised to max(X_b, S - dist(v, u)). Every cycle contains an H64 backward edge, so min cycle >= S. The exact check gives a minimum cycle length L equal to S = 10^6.
- tau* <= 3096300477189/500000 = 6,192,600.95, so no cycle packing can go below 85.2248 %. The bracket became 6,138,414 to 6,192,601.
- Independent review PASS: own C checker over all 136,648 heads, min cycle exactly S, same sum.
- Pinned as `certs/ub_tau_repair_heads.npz`, X int64 array-bytes sha256 `43b2b56d455106ffd17ecbf5086ca77e95b3141811607a3101ab23bb1efd9f88`, S = 10^6.
- It showed that the cycle-packing road ends between 85.2248 and 85.3541 %.

### 3.7 H8: exact optima on windows (`results/M_WIN.md`)
Lazy-cycle MILP (HiGHS) on windows (blocks of 300 to 3,000 neurons that are consecutive in the H64 order) and on dense reciprocal clusters of 500 to 2,000 nodes. 71 of 78 pieces were solved exactly (none of the clusters above 500 nodes). This does not move the bound. It tests where the gap between the LP and the orders lives (section 6).

### 3.8 H9: closing the bracket (`results/M_CLOSE.md`)
Same LP and certificate path as CG2, with two changes.
- The repair runs over the heads of H64 backward edges in 16 groups, processed one after another. This is cheaper and still valid, because lengths only grow.
- The most violated cycle through each raised backward edge is added to the pool.

Results:
- Best certificate: c45_e, **6,141,103, 85.3477 %** (PASS, also in a separate re-run).
- Best UB: tau* <= 1228406814123/200000 = 6,142,034.07, pinned as `certs/ub_tau_close_5e.npz`.
- The total weight the repair had to add fell about 30 % per round, from 45.0k to 0.7k over 15 rounds. The pool LP moved only 2.2k, so the old 54k gap was almost all on the dual side.
- Pricing exactly the cycles the repair had to lift is what fixed the dual.

### Ladder

| step | certificate | LB (syn) | ceiling FF % | status |
|---|---|---|---|---|
| 2-cycle bound | c00_2cycle | 2,815,303 | 93.2828 | PASS |
| greedy 2+3 cycles (scout) | none | 4,600,674 | 89.0231 | not certified |
| greedy 2+3 cycles, joint | c10_greedy23_joint | 4,556,165 | 89.1292 | PASS |
| LP over 2+3 cycles (P) | c13_pdhg23 | 4,889,638 | 88.3336 | PASS |
| order-guided routing (R) | c21_route_improve_r1 | 5,843,849 | 86.0569 | PASS |
| short-cycle CG (P) | c1r6_cg_r6 | 5,890,045 | 85.9467 | PASS |
| CG2 | c34_fin | 6,138,414 | 85.3541 | PASS |
| H9 close | c45_e | 6,141,103 | 85.3477 | PASS |

Upper bounds on tau* (limits for any cycle packing; run ids as in `results/M_CLOSE.md`):

| step | UB on tau* | lowest possible cycle-packing ceiling | check |
|---|---|---|---|
| H64 backward indicator | 6,597,734 | 84.2582 % | exact |
| window LPs of 3,000 ranks | 6,581,707.22 | 84.2964 % | exact |
| DUAL repair of CG2 duals | 6,192,600.95 | 85.2248 % | exact, independent review PASS |
| H9 run 1_a | 6,156,973.01 | 85.3098 % | exact, independent C checker confirmed |
| H9 run 5_e (final) | 6,142,034.07 | 85.3455 % | exact, independent C checker confirmed |

## 4. The final bracket: what it proves and what it does not

    6,141,103 <= tau* <= 6,142,034.07          (width 931 syn, 0.0022 pp of W)
    FF(any order) <= FF(OPT) <= 85.3477 %       (proven, certificate c45_e)
    no cycle-packing certificate can prove a ceiling below 85.3455 %

Status of the final UB 6,142,034.07:
- Author check PASS with `m_dual_check`, twice: inside run 5_e and again from the saved file (`logs/m_close_recheck_final.log`, L = 10^6 = S).
- Independent check PASS: the REVIEW_DUAL reviewer's own C checker (`reviews/ub_independent/`, unrestricted Dijkstra from all 136,648 heads, 1,273 s) finds 0 edges on a cycle shorter than S, min cycle exactly S, and the same sum w X = 6,142,034,070,615.
- Pinned lengths: `certs/ub_tau_close_5e.npz`, keys X and S. X int64 array-bytes sha256 `22b9799a509b06c088b1a2b6860e503ac66f2092e29d876052809daf418a120b`, S = 10^6.

The earlier UBs 6,192,600.95 and 6,156,973.01 are independently confirmed as well. The ceiling 85.3477 % does not depend on any UB.

What it proves:
- Every order of this graph has FF <= 85.3477 %. The leaderboard top is at most 0.5411 pp below the optimum, our H64 at most 1.0895 pp. This needs no UB.
- From the final UB: the cycle LP is pinned to 6.1411 to 6.1420 M. The leaderboard order pays at least 225,873 more than tau*, H64 at least 455,699 more.

What it does not prove:
- Where OPT lies inside [84.8065 %, 85.3477 %]. We have leaderboard - tau* = (leaderboard - OPT) + (OPT - tau*). The first term is the order's suboptimality, the second is the integrality gap. Their split is open. A cycle packing can never prove an integrality gap.
- That the leaderboard order is suboptimal.

## 5. Hypotheses: predictions and outcomes

Predictions were written in `HYPOTHESES.md` before the runs that test them.

| id | prediction | outcome | verdict |
|---|---|---|---|
| H1 | greedy 2+3-cycle packing reaches LB >= 5.0 M (rejected if < 4.8 M) | 4,600,674 (89.0231 %) | rejected |
| H2 | order-guided greedy routing beats H1; LP of the restricted flow in [5.3, 6.2] M | greedy 5,604,533 pure, 5,843,849 with residual stage; LP not solved | supported (greedy part), LP part untested |
| H3 | integrality gap is the main remaining gap (>= 0.5 pp); tau* / F(H64) in [0.85, 0.97] | tau* / F(H64) = 0.931, inside the range, but calibration and H8 show no gap of that size on pieces | open, weakened |
| H4 | exact small pieces add +0.05 to +0.3 pp | on 144 pieces of <= 20 nodes the exact minimum never exceeds the cycle LP; not run on the full graph | no support, deprioritized |
| H5 | unexplained H64 feedback is central and long-span, not optic | 87 % long span (1e3 to 1e5); unrouted share central brain 0.19 vs optic lobe 0.11, but optic holds 36 % | partly supported |
| H6 | mouse cycle LP closes >= 70 % of the gap; fly pieces within 5 % | mouse 100 % (LP = OPT); 182 pieces mean 0.002 % below OPT, worst 0.28 % | confirmed |
| H7 | certified LB >= 6.10 M; tau* in [6.25, 6.37] M | LB 6,138,414; tau* <= 6,142,034.07 | first part supported, second rejected |
| H8 | window LP within 0.2 % of OPT (rejected if >= 0.5 %) | mean 0.016 %, worst 0.271 % on 71 pieces; H64 1.318 % above OPT on 34 windows | supported on windows; long cycles untested |
| H9 | bracket < 20k: LB +25k and/or UB -10k | UB -50,567, LB +2,689; width 931 | confirmed via the UB side; LB part missed (at most +3,620 was possible) |

The literature pass guessed that the best certifiable ceiling within hours would be 87 to 88.5 % (`lit/LITERATURE.md`, marked as a guess there). The result went lower.

## 6. Calibration

These are HiGHS float values, not certificates.
- Mouse mesoscale connectome (148 nodes). The exact (MILP) optimum is 93.202194 % FF, from two MILP formulations, with the order re-scored by the reference scorer. The cycle LP equals it, closing 100 % of the gap from the 2-cycle bound (94.3731 %). Our best mouse order H63 (93.175383 %) is 0.0268 pp below this MILP optimum.
- 182 exactly solved fly pieces (15 to 300 nodes). The cycle LP is 0.002 % below OPT on average, worst 0.28 %. A gap appears in 7 of 185 pieces, all with 45 or more nodes, and more often in larger ones. The triangle LP never beat the cycle LP. Greedy 2+3 cycles close a falling share of the gap as size grows (0.90, 0.80, 0.72).
- H8 windows and clusters (300 to 3,000 nodes, 71 solved). The cycle LP is 0.016 % below OPT on average, worst 0.271 %. Sums over the 34 solved 3,000-rank windows: H64 617,644, OPT 609,608, LP 609,560.

## 7. Where the gap may live (interpretation)

This section is interpretation, not measurement.
- The ceiling is 0.5411 pp above the leaderboard and 1.0895 pp above H64. On every graph we could check, the cycle LP was within 0.3 % of OPT. If that held on the full graph, OPT would be near tau* (6.14 M), and most of the 226k between the ceiling and the leaderboard would be order suboptimality. That is a guess.
- The checked graphs miss the structure that matters most. Windows of 3,000 ranks hold only 16 % of H64 feedback. The other 84 % sits on long-span backward edges whose cycles leave every window. The gap grew with piece size in calibration, and the 9 densest central windows were not solved. A real integrality gap on the full graph cannot be ruled out.
- For improving orders: exact re-solving of 3,000-rank windows is worth only about 14k to 22k syn. Of this, 8,760 is exact and achievable now on 37 windows, and the rest is an estimate. H64's gap to the leaderboard is 229,826 syn, so most of our order's excess sits in long-span structure.
- Method R's unrouted demand is 87 % on spans of 1e3 to 1e5 ranks, with a higher rate in the central brain (0.19) than in the optic lobe (0.11). This describes one certificate, not where OPT pays.

## 8. Limitations

- The cycle LP is the barrier for this certificate type: no K1 certificate can go below 85.3455 %. Going further needs a stronger relaxation on the full graph, and none was tried at full scale. On small pieces the triangle LP and K2 pieces gave nothing, but that says little about the giant SCC.
- Calibration and window values come from floating solvers. Only the LB certificates (verify_cert) and the UB length vectors (m_dual_check) are integer-checked. Window optima are checked through the integer feedback of an order; LP values are not exact.
- Certificates, including c45_e (110 MB), are not in git. The two UB length vectors are pinned in `certs/`. Rebuilding c45_e needs the scratchpad pool states of the chain P, CG2, CLOSE. If those are lost, the chain must be rerun from method P, whose pool-building steps (`m_short_enum.py`, `m_short_pdhg.py`, `m_short_price.py`, `m_short_pool.py`) are documented only through their logs (`logs/m_short_pdhg_r*.log`, `logs/m_short_price_r*.log`), not as recorded commands. PDHG is multithreaded floating point, so a rerun is close but not bit-identical.
- The K2 path of the verifier is tested only on toy graphs. No quoted bound uses it.
- Wall times are indicative. The machine was shared, and it slept about 2.9 h during the CG2 and DUAL runs.
- The window-LP rounding order (84.2737 %, 6,490 syn better than H64) is one deterministic pass, and its rank vector is not pinned.

## 9. What may interest the community

- To our knowledge this is the first certified upper bound on FF % for this public challenge. Our literature search (`lit/LITERATURE.md`) found no published lower bound, LP value or optimality gap. A challenge-related page states there is "no known optimality bound". A GitHub repo `mdindoost/mfascert` with the same certificate kind is indexed by web search, but it returned HTTP 404 on 2026-11-25 and no bound from it could be checked.
- The result gives the leaderboard a scale: the top entry is within 0.5411 pp of the best possible order.
- The cycle LP is nearly tight on every connectome piece we could solve, and equal to OPT on the mouse meso graph. Whether this holds for the whole graph is an open question.
- Anyone can check the pipeline. The certificate is int64 and bound to the graph by a checksum, the verifier has no solver inside, and the dual side has a separate exact integer checker. Floating-point solvers are used only to find certificates, not to check them.
- Three method findings may transfer to other instances:
  - order-guided routing gives strong column seeds;
  - repairing LP duals through the backward edges of a good order turns approximate duals into exact upper bounds;
  - pricing the cycles the repair lifts closes the dual side fast.

## 10. Reproduction

Run from the project directory `nn_backward_connections/`. Set `P=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`. `SP` is the temporary work folder that held the pool states; its full path is in the result files.

Check the final bounds (cheap):

    $P -m pytest -q ceiling/tests/test_verify_cert.py
    $P ceiling/src/verify_cert.py ceiling/certs/c45_e.npz
    $P -u ceiling/src/m_dual_check.py ceiling/certs/ub_tau_close_5e.npz <tag> X
    $P -u ceiling/src/m_dual_check.py ceiling/certs/ub_tau_repair_heads.npz <tag> X

`m_dual_check.py <file> <tag> <key>` is the form used in `M_DUAL.md`; it writes a results JSON named after the tag. X is int64, so the stored S is not read: the checker computes L and reports sum w X / L. PASS means L = 10^6, the stored S.

2-cycle bound and method R:

    $P ceiling/src/cert_2cycle.py
    $P ceiling/src/m_route_greedy.py --tag g1
    $P ceiling/src/m_route_improve.py --tag r1 --alphas 1 --phases 1,2,3,4,5,6,8,10 --residual-phases 2,3,4,5,6,8

Method P: the certificate and refill commands are in the `command` field of `results/m_short_cg_r*.json`, `m_short_pdhg23*.json` and the `*_refill1500.json` files; they read pools in `$SP/mshort/`. The pool-building steps have no recorded command (see section 8).

CG2 (`results/M_CG2.md`):

    $P ceiling/src/m_cg2_run.py --tag 0_r1 --init --route-seed ceiling/certs/c21_route_improve_r1.npz --rounds 2 --k 30 --pdhg-iters 1000 --pdhg-final-iters 1500 --time-limit 3000
    $P ceiling/src/m_cg2_run.py --tag 2_s6 --state $SP/cg2/state_0_r1.npz --rounds 30 --k 0 --short-H 6 --short-k 50 --pdhg-iters 600 --pdhg-final-iters 1500 --time-limit 2900
    $P ceiling/src/m_cg2_run.py --tag 3_dj --state $SP/cg2/state_2_s6.npz --rounds 30 --k 30 --n-sources 40000 --floor-len 0.01 --short-H 6 --short-k 50 --pdhg-iters 600 --pdhg-final-iters 1500 --time-limit 2700 --seed 100
    $P ceiling/src/m_cg2_run.py --tag 4_fin --state $SP/cg2/state_3_dj.npz --rounds 2 --k 30 --n-sources 40000 --floor-len 0.001 --sat-weight 20 --short-H 6 --short-k 50 --pdhg-iters 1200 --pdhg-final-iters 3000 --pdhg-tol 0.004 --time-limit 3000 --seed 200
    $P ceiling/src/m_cg2_check.py $SP/cg2/duals_final.npz check_final

Run 2_s6 was stopped by a stop file after round 11, and run 3_dj after round 15.

DUAL (`results/M_DUAL.md`):

    $P -u ceiling/src/m_dual_selftest.py > ceiling/logs/m_dual_selftest.log
    $P -u ceiling/src/m_dual_repair.py $SP/cg2/duals_final.npz x cg2_final_heads heads

H9 close (`results/M_CLOSE.md`), which produces c45_e and the final UB:

    caffeinate -i $P -u ceiling/src/m_close_repair.py $SP/cg2/duals_final.npz x final_ch8 8
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 1_a --state $SP/cg2/state_4_fin.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 2_b --state $SP/close/state_1_a.npz --center $SP/close/X_1_a.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 3_c --state $SP/close/state_2_b.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 1500 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 4_d --state $SP/close/state_3_c.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 2000 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2700
    caffeinate -i $P -u ceiling/src/m_close_lower.py $SP/close/X_4_d.npz 4_d 32 16 1
    caffeinate -i $P -u ceiling/src/m_close_run.py --tag 5_e --state $SP/close/state_4_d.npz --rounds 2 --pdhg-iters 1500 --pdhg-final-iters 2000 --chunks 16 --add-thr 1e-3 --short-H 6 --time-limit 2600

Calibration commands are listed at the end of `results/CALIBRATION.md`. Window (H8) commands are in the `command` field of `results/m_win_*.json`.

## 11. File map

| path | content |
|---|---|
| `CHARTER.md` | question, validity rule, method menu, process |
| `PROGRESS.md` | resume point and append-only log |
| `HYPOTHESES.md` | H1 to H9, predictions written before the runs, results log |
| `SCOUT.md` | graph structure, H64 feedback profile, 3-cycles, timings, mouse optimum |
| `lit/LITERATURE.md` | literature review, leaderboard snapshot |
| `src/CERT_FORMAT.md` | certificate format v1 |
| `src/verify_cert.py`, `src/cert_io.py`, `tests/test_verify_cert.py` | independent verifier and tests |
| `src/m_dual_check.py`, `src/m_dual_core.py`, `src/m_dual_selftest.py` | exact UB checker on tau* |
| `src/m_route_*.py`, `m_short_*.py`, `m_calib_*.py`, `m_cg2_*.py`, `m_dual_*.py`, `m_win_*.py`, `m_close_*.py` | methods R, P, calibration, CG2, DUAL, H8, H9 |
| `results/M_*.md`, `results/CALIBRATION.md` | method reports |
| `results/*.json` | per-run command, commit, config, seeds, numbers, verifier output |
| `reviews/REDTEAM_1.md`, `reviews/redteam1/` | red-team review and its checker |
| `reviews/REVIEW_DUAL.md` | independent review of the DUAL UB |
| `certs/c45_e.npz` | final certificate, LB 6,141,103 (110 MB, not in git) |
| `certs/c00_2cycle.npz`, `c21_*`, `c1r6_*`, `c34_fin.npz`, `c4*_*.npz` | earlier certificates (not in git) |
| `certs/ub_tau_close_5e.npz` | final UB lengths, tau* <= 6,142,034.07 |
| `certs/ub_tau_repair_heads.npz` | reviewed UB lengths, tau* <= 6,192,600.95 |
| `logs/` | all run logs, including failed and killed runs |
