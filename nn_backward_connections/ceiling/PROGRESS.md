# FF ceiling study: progress (resume point)

Worktree: `/Users/abed359/IdeaProjects/university/deeplearning_thesis_ceiling`, branch
`research/ff-ceiling` (from `layering/v1` at 657786f). Not committed yet.

## Current best certified ceiling

| cert | LB (syn) | ceiling FF % | verified by | notes |
|---|---|---|---|---|
| c00_2cycle | 2,815,303 | 93.2828 | verify_cert PASS (2026-11-25) | certs/c00_2cycle.npz |
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

Phase 1: hypotheses with predictions (`HYPOTHESES.md`), from literature + scout, then one
agent per bound method.

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
