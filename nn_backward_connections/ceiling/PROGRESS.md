# FF ceiling study: progress (resume point)

Worktree: `/Users/abed359/IdeaProjects/university/deeplearning_thesis_ceiling`, branch
`research/ff-ceiling` (from `layering/v1` at 657786f). Study complete; final write-up in REPORT.md / REPORT.ru.md.

## Current best certified ceiling

| cert | LB (syn) | ceiling FF % | verified by | notes |
|---|---|---|---|---|
| c45_e (H9 close, run 5_e) | 6,141,103 | 85.3477 | verify_cert PASS, re-run by orchestrator | D=10000, 6,748,248 cycles, len <= 22, 110 MB, sha256 e243142c...7766 |
| c00_2cycle | 2,815,303 | 93.2828 | verify_cert PASS (2026-11-25) | certs/c00_2cycle.npz |
| c34_fin (CG2, round 17) | 6,138,414 | 85.3541 | verify_cert PASS, re-run by orchestrator | D=10000, 6,710,328 cycles, len <= 19 |
| c1r6_cg_r6 (method P, CG round 6) | 5,890,045 | 85.9467 | verify_cert PASS, re-run by orchestrator 2026-11-25 | D=10000, 3,734,535 cycles, len 2..15 |
| c21_route_improve_r1 (method R) | 5,843,849 | 86.0569 | verify_cert PASS, re-run by orchestrator 2026-11-25 | 1,760,774 cycles, max len 11 |
| prior: 2-cycle + 420 s greedy 3-cycles (2026-08-19 probe) | 4,537,757 | 89.1732 | not certified (no cert file) | |

Our order 84.2582 % (H64, FB 6,597,734); leaderboard top 84.8065 % (FB 6,367,908, fetched 2026-11-25); repo best_solution 84.6147 %.

## Phase 0 (started 2026-11-25)

- literature agent -> `lit/LITERATURE.md` (running)
- verifier agent -> `src/cert_io.py`, `src/verify_cert.py`, `tests/test_verify_cert.py`,
  `src/cert_2cycle.py`, `certs/c00_2cycle.npz` (running)
- scout agent -> `SCOUT.md`, `src/scout_*.py`, `logs/scout_*` (running)

## Next

Done. Open options are listed in REPORT.md sections 8-9 and REPORT.ru.md (next steps).

## Log (append-only)

- 2026-11-25: charter v1, certificate format v1, phase 0 launched.
- 2026-11-25: verifier done (106 tests pass, incl. adversarial + overflow); c00_2cycle
  verifies PASS, LB 2,815,303 exactly. Mouse (float weights) rejected by format v1.
- 2026-11-25: literature done (lit/LITERATURE.md): no published bound for this instance; leaderboard top re-fetched myself = 35,544,233. Possible competitor repo mdindoost/mfascert (404).
- 2026-11-25: scout done (SCOUT.md). H1 rejected: greedy 2+3-cycles saturate at 4,600,674
  (89.0231 %, uncertified). Mouse exact optimum 93.202194 % (H63 champion 0.0268 pp below);
  to be independently confirmed. Phase 1 launched: methods P (short-cycle LP packing),
  R (order-guided routing, H2), C (calibration, H6).
- 2026-11-25: method R done (results/M_ROUTE.md). Best c21 r1 LB 5,843,849 = 86.0569 % (re-verified by me). Pure H2 routing g1 5,604,533 (86.6279 %). Gap to our order 1.80 pp, to leaderboard top 1.25 pp. Greedy only, LP not solved; 986k of 993k unrouted syn lost to capacity competition, not to missing paths. Router core is C compiled by system cc via ctypes (no new package).
- 2026-11-25: calibration done (results/CALIBRATION.md). Mouse exact optimum independently
  re-derived = 93.202194 % (agrees with scout). Cycle LP ~ OPT on all calibration instances;
  triangle LP and K2 pieces add nothing. Main line now: solve the full cycle LP (H7).
- 2026-11-25: method P done (results/M_SHORT.md). Joint 2+3 LP optimum in [4,889,601, 4,889,963] (88.33 %); column generation with Dijkstra pricing, 6 rounds, gains halving; best c1r6 LB 5,890,045 = 85.9467 % (re-verified by me). Not converged: pricing still finds violated cycles. Warm-start pools in scratchpad/mshort (3.3 GB, volatile).
- Open tension: P extrapolates the full cycle LP at 5.92-6.0 M (integrality gap ~1.5 pp), calibration says the cycle LP ~ OPT on pieces. Needs a rigorous upper bound on the cycle LP (scaled dual lengths) to decide.
- 2026-11-25: checkpoint commit deecf6b (local, not pushed; certs gitignored). Phase 2 launched:
  CG2 (column generation to convergence, C pricing, H7), DUAL (rigorous upper bound on the
  cycle-LP value tau* from repaired dual lengths), RED-TEAM 1 (try to break verifier + proof
  chain, own independent checker).
- 2026-11-25: RED-TEAM 1 PASS (reviews/REDTEAM_1.md): 0/22 crafted invalid certs pass;
  independent from-scratch checker (copied to reviews/redteam1/mycheck.py) reproduces
  LB 5,890,045 (c1r6) and 5,843,849 (c21) exactly; every certified cycle contains a backward
  edge of H64. Minor: duplicate zip members accepted (last wins), meta compared with ==.
- 2026-11-25: CG2 c34_fin LB 6,138,414 = 85.3541 % (re-verified by me). Pool LP primal 6,138,043 vs dual est 6,140,143 (gap 0.034 %). H7 prediction (>= 6.10 M) met. DUAL rough estimate (not rigorous) for tau* ~ 6.19 M. Agents still finishing.
- 2026-11-25: CG2 done (results/M_CG2.md). Note: the Mac slept ~2.9 h mid-run; future long
  runs go under `caffeinate -i`. Pool state for continuation: scratchpad/cg2/state_4_fin.npz.
- 2026-11-25: DUAL done: 6,138,414 <= tau* <= 6,192,601 (cycle-packing ceiling can never go
  below 85.2248 %). Side: window-LP rounding gives an order 6,490 syn better than H64
  (84.2737 %, one pass, not pinned). Phase 3 launched: REVIEW-DUAL (fresh reviewer of the UB
  proof), H8 (exact window MILPs), H9 (close the bracket).
- 2026-11-25: REVIEW-DUAL PASS (reviews/REVIEW_DUAL.md): independent C check over all
  136,648 heads, min cycle = S exactly, same sum; tau* <= 6,192,600.95 is proven. UB length
  vector pinned as certs/ub_tau_repair_heads.npz (X int64 sha256 43b2b56d...9f88, S = 1e6,
  compressed 8.6 MB; will be committed with the final certs). Source duals sha256 6f4e3a4d...f696.
- 2026-11-25: H8 done (results/M_WIN.md): exact optima on 71/78 pieces (H64 windows 300-3000
  ranks, reciprocal clusters of 500). Cycle LP below OPT by 0.016 % on average (worst 0.271 %);
  on 34 solved M_DUAL windows H64 is 1.318 % above OPT, OPT only 0.008 % above LP. Window gap
  is order suboptimality, not integrality. Caveats: induced windows hold 16 % of H64 feedback;
  9 densest central windows not solved. Exact window re-solve is worth only ~14-22k syn (est.).
- 2026-11-25: H9 run 1_a: c41_a LB 6,139,869 = 85.3506 % (verify_cert PASS, re-run by
  orchestrator). Claimed UB tau* <= 6,156,973.01 (X_1_a, S = 1e6), independent check by the
  REVIEW-DUAL C checker running. Run 2_b still in progress.
- 2026-11-25: independent REVIEW-DUAL checker (scratchpad review_dual/rcheck) confirms the run
  1_a UB: min cycle = S, sum wX = 6,156,973,012,593 / 1e6.
- 2026-11-25: H9 done (results/M_CLOSE.md). Final bracket 6,141,103 <= tau* <= 6,142,034.07
  (width 931). Best cert c45_e LB 6,141,103 = 85.3477 % (verify_cert PASS, re-run by
  orchestrator; sha256 e243142c...7766, 107 MB, too big for git, reproduce via M_CLOSE.md).
  No cycle packing can give a ceiling below 85.3455 %. UB lengths pinned as
  certs/ub_tau_close_5e.npz (X sha256 22b9799a...120b, S = 1e6). Independent check running.
- 2026-11-25: independent C checker (reviews/ub_independent/, from REVIEW-DUAL) confirms the
  final UB X_5_e: 0 edges on a cycle < S, min = S, sum wX = 6,142,034,070,615 / 1e6. Bracket
  6,141,103 <= tau* <= 6,142,034.07 is fully verified on both sides.
- 2026-11-25: final reports REPORT.md / REPORT.ru.md written; accuracy review (3 errors, 6
  overstatements) and style review (undefined tags, RU calques) applied. Study complete.
