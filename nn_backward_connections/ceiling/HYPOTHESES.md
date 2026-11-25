# Hypotheses and predictions (written before the runs that test them)

Notation: W = 41,912,141. F(pi) = backward weight of an order pi. Our H64 order: F = 6,597,734
(15.7418 % of W). LB = certified lower bound on min_pi F(pi). gap = F(H64) - LB.
Start: LB = 4,537,757 (89.1732 %, uncertified probe), gap = 2,059,977 syn = 4.91 pp.

## H1. The 3-cycle greedy is far from finished (draft 2026-11-25, before any run)
Claim: running the greedy 3-cycle packing of the 2026-08 probe to completion (it was cut at
420 s) raises LB substantially.
Prediction: LB(2-cycles + all 3-cycles greedy) >= 5.0 M syn (ceiling <= 88.07 %).
Kill: LB < 4.8 M after completion or after 60 min.

## H2. Order-guided packing: one backward edge per cycle (draft 2026-11-25)
Idea: in the H64 order every backward edge (u -> v, rank v < rank u) closes a cycle with any
FORWARD path v ~> u, and such a path lives in the rank window [rank v, rank u]. Packing these
cycles is a multicommodity flow on the forward DAG: commodity per backward edge, demand
w(u,v), forward edges as capacities. If every backward edge could be fully routed, LB = F(H64)
and H64 would be optimal; the unrouted demand is exactly what this certificate cannot explain.
Shortest paths on a DAG in topological order are linear-time, so multiplicative weights
(Garg-Konemann) or greedy augmenting is cheap. Any packed cycle is valid regardless of pi.
Prediction: greedy order-guided routing (short spans first) beats H1's LB; the LP optimum of
the restricted flow lands at LB in [5.3, 6.2] M syn (ceiling 85.2 to 87.4 %).
Kill: restricted-flow LB below H1's LB.

## H3. The integrality gap of the cycle LP is the main remaining gap (draft)
Claim: after H2, a residual gap of >= 0.5 pp remains that no cycle packing can close, because
the fractional FAS LP optimum is below F(OPT). Test needs a primal fractional FAS bound
(lengths x_e with every cycle of length >= 1) or the GK dual estimate.
Prediction: cycle-LP value / F(H64) in [0.85, 0.97].
Kill: cycle-LP value within 0.1 pp of F(H64) (then H64 is near-proven optimal).

## H4. Exact pieces beat cycles where the graph is dense (draft)
Claim: on small dense reciprocal clusters (e.g. mushroom-body, lamina cartridges, tight cell
types) the exact min feedback of the piece exceeds its best cycle packing, so K2 pieces add
LB beyond H2/H3.
Prediction: positive but small, +0.05 to +0.3 pp of W.
Kill: total K2 gain < 0.02 pp.

## H5. Where the gap lives (descriptive, draft)
Prediction: the unexplained backward weight of H64 (demand H2 cannot route) concentrates in
the central brain and long-span edges, not in the optic lobe.

## Results log (append-only)

- 2026-11-25 H1 REJECTED (scout run `src/scout_tri_full.py`, `logs/scout_*`): all 2,765,514
  residual 3-cycles enumerated (69 s); greedy packing to completion = 4,532,314 (random order)
  / 4,600,674 (largest min-capacity first, 89.0231 %). Below the 4.8 M kill line. Greedy
  2+3-cycle packing saturates near 89.0 %; the LP over the same cycles is still unknown
  (HiGHS hit 400 s on 2.77 M columns).

## H6. Calibration (added 2026-11-25, before the calibration runs)
Mouse meso (148 nodes) has an exact optimum (scout: 93.202194 % FF, two MILP formulations,
oracle-confirmed). Prediction: the full cycle-LP value on the mouse closes >= 70 % of the
gap between the 2-cycle bound (94.3731 %) and the optimum; on small connectome subgraphs
the cycle LP is within 5 % of the exact optimum feedback on average.
Kill: cycle LP closes < 50 % of the mouse gap (then cycle packing alone cannot get close on
the fly and stronger certificates are the main line).
- 2026-11-25 H2 SUPPORTED (greedy part; `results/M_ROUTE.md`): order-guided greedy routing
  c20 g1 LB 5,604,533 > H1 level 4.60 M; with multi-backward-edge cycles on the full graph
  (c21 r1) 5,843,849 = 86.0569 %, verified PASS. LP-optimum part of the prediction untested.
- 2026-11-25 H5 PARTLY SUPPORTED: unrouted demand is 87 % long-span (1e3..1e5) but the
  unrouted share per span bin does not grow past 1e4; central 41 % of unrouted (share 0.19
  of its demand), optic 36 % (0.11) - optic is not free of it. Describes one certificate,
  not where OPT pays.
- 2026-11-25 H6 CONFIRMED (`results/CALIBRATION.md`): mouse cycle LP = exact optimum
  (closes 100 % of the lb2-to-opt gap; prediction >= 70 %); on 182 exactly solved fly pieces
  (15-300 nodes) the cycle LP is 0.002 % below opt on average, worst 0.28 %; gap appears only
  in pieces with 45+ nodes. Triangle LP never beat the cycle LP. Greedy 2+3 closes a falling
  share of the gap with size (0.90 / 0.80 / 0.72; mouse 0.66): long cycles matter.
- 2026-11-25 H4 predicted dead (calibration): on 144 pieces <= 20 nodes the exact minimum
  never exceeds the cycle LP. Not run on the fly; deprioritised.
- 2026-11-25 H3 weakened: no sign of a >= 0.5 pp integrality gap on pieces; cannot be ruled
  out on the giant SCC. The working assumption becomes: the cycle-LP value on the full graph
  is close to OPT, so the ceiling is limited by how well we solve that LP.

## H7. Near-optimal fractional packing of long cycles approaches the leaderboard (added
2026-11-25, before the phase-2 runs)
Claim: a fractional packing (Garg-Konemann / Fleischer style, cycle oracle = shortest cycle
through an edge under current lengths, warm-started from the method-R integer packing)
raises LB well beyond the greedy 5.84 M.
Prediction: certified LB >= 6.10 M (ceiling <= 85.45 %) within the phase-2 budget; the LP
dual (fractional FAS lengths) puts the cycle-LP optimum between 6.25 M and 6.37 M (i.e. at
or slightly below the leaderboard top's 6,367,908 feedback).
Kill: certified LB < 5.95 M after the phase-2 budget.
- 2026-11-25 H7 SUPPORTED (first part; `results/M_CG2.md`): certified LB 6,138,414
  (85.3541 %, c34_fin, verifier PASS, re-run by orchestrator) >= 6.10 M prediction. Second
  part (cycle-LP optimum in [6.25, 6.37] M) open: CG not converged (exact pricing on final
  duals finds violated cycles from 119,470 of 127,550 sources, min length 0.686, mostly at
  PDHG precision). Biggest single step: seeding the pool with method-R cycles (+113k).
  Long cycles (13-21 edges) get almost no weight; packing mass sits on 2-7 edge cycles.
- 2026-11-25 DUAL result (`results/M_DUAL.md`, not yet independently reviewed): exact
  integer check of repaired CG2 duals proves tau* <= 6,192,600.95. With the certified LB:
  6,138,414 <= tau* <= 6,192,601. No cycle packing can certify a ceiling below 85.2248 %.
  P's extrapolation (tau* ~ 5.92-6.0 M) is refuted. H3 restated: leaderboard feedback
  (6,367,908) exceeds tau* by >= 175,307; the split between leaderboard suboptimality and
  integrality gap is unknown.

## H8. Who owns the 175k: the orders or the LP? (added 2026-11-25, before the runs)
Test: exact optimum (MILP with lazy cycle constraints, HiGHS) of contiguous windows of the
H64 order (and of a few dense subgraphs of 500-3000 nodes), vs the window's cycle-LP value
and vs the best known order of the window.
Prediction: on windows the cycle LP stays within 0.2 % of the exact optimum (as on pieces),
so most of the ~1 % window gap between LP and the H64 order is order suboptimality.
Kill / reverse: if exact window optima sit >= 0.5 % above the window LP on average, the
integrality gap is real at that scale and the true OPT is likely far above tau*.

## H9. The bracket can be closed to < 20k (added 2026-11-25)
Prediction: one more CG campaign started from the repaired (feasible) duals, with high
precision PDHG and pricing focused on the cycles the repair had to lift, raises LB by
>= 25k and/or lowers the UB by >= 10k.
Kill: LB gain < 10k and UB gain < 5k after 3 h.
