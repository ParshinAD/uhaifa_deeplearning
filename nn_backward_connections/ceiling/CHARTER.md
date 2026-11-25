# FF ceiling study: how high can feedforward weight possibly go on the fly connectome?

Status: charter v1 (2026-11-25), phase 0 started. Resume point: `PROGRESS.md`.

## 1. Question

Every order of the 136,648 FlyWire neurons leaves some weight on backward edges. Our champion
order (H64) keeps 84.2582 % of W = 41,912,141 forward (35,314,407 syn). Comparators only:
the repo's `data/best_solution/` file (Vahidi and Koutis, 84.6147 %) and the live challenge
leaderboard top, 35,544,233 syn = 84.8065 % (C. Westfall, 2026-11-08; fetched from
https://codex.flywire.ai/app/mfas_challenge on 2026-11-25), i.e. feedback 6,367,908.
Nobody knows the true optimum OPT. This study builds the lowest PROVEN upper bound on FF%
(equivalently the highest proven lower bound LB on the minimum feedback weight) and reports
the gap between it and our order and the leaderboard top.

    FF%(our order) <= FF%(OPT) <= ceiling = 100 * (W - LB) / W

Starting point (both are valid certified bounds):
- 2-cycle floor: LB = 2,815,303 -> ceiling 93.2828 % (biology studies, A1).
- greedy 2-cycle + 420 s of greedy 3-cycle packing (2026-08-19,
  `experiments/diagnostics/probe_cyclepacking_upper_bound.py`): LB = 4,537,757 -> 89.1732 %.

Main deliverable: the best certified ceiling, its gap to 84.2582 % and to 84.6147 %, and
where in the brain the remaining gap sits.

## 2. Validity rule (the core of the study)

A bound counts only if it comes with a CERTIFICATE that `src/verify_cert.py` (written by a
separate agent, never by the author of a bound method) checks from scratch against the raw
edge list loaded by the frozen loader `src.mfas.io.load_dataset`. Weak duality makes this
possible: every feedback arc set must hit every directed cycle, and for edge-disjoint
(or capacity-split) subgraphs the minimum feedback weights add up.

Certificate kinds (format in `src/CERT_FORMAT.md`):
- K1 fractional cycle packing: cycles as edge-id lists, integer multiplicities y over a
  common denominator D; checked: each cycle is a closed directed walk on real edges, simple,
  and for every edge sum of y over cycles using it <= D * w(e). LB = sum(y) / D.
- K2 piece packing: pieces = (edge ids, integer capacities c_e) whose capacities, together
  with K1 loads, never exceed w(e); each piece carries a claimed exact minimum feedback
  weight under capacities c, which the verifier recomputes itself (exact DP over node
  subsets, at most 20 nodes per piece). A cycle packing inside a piece is not a separate
  case: move its cycles to K1 and lower the piece capacities by their load.
- Any further certificate kind must be added to the verifier first, with a toy test that
  compares against brute force, before a bound using it is quoted.

Numbers are integers (int64 / Python int). A bound with a float in its proof is not a bound.

## 3. Methods (initial menu; the ideator and literature agents extend it)

1. Better greedy cycle packing (3-cycles to completion, then 4-, 5-, k-cycles, with
   ordering heuristics).
2. LP-optimal fractional cycle packing (dual of the fractional FAS LP), by column
   generation or multiplicative weights (Garg-Konemann) with a shortest-cycle oracle.
   Gives the cycle-LP value itself; the integral champion vs this LP is the integrality gap.
3. Stronger than cycle LP: exact MFAS on small dense pieces (edge-disjoint), ordering-LP
   with 3-dicycle inequalities on pieces, other valid inequalities from the literature.
4. Structure: SCC decomposition (bounds add over SCCs), lane / cell-type blocks.

Calibration arm: graphs whose optimum is known or provable (toy graphs, mouse meso 148 nodes
if exactly solvable, small connectome subgraphs) measure how tight each method is.

## 4. Process

- Phase 0: literature review (`lit/LITERATURE.md`), engineering scout (`SCOUT.md`),
  independent certificate verifier with tests (`src/verify_cert.py`, `tests/`).
- Phase 1: hypotheses with predictions and kill conditions, written in `HYPOTHESES.md`
  BEFORE the run that tests them.
- Phase 2: one subagent per method, own files; every result = certificate + run JSON in
  `results/` (command, git commit, seed, runtime, LB, ceiling) + verifier output.
- Phase 3: reviewer from fresh context per result (correctness of the proof chain, not the
  number); red-team pass that tries to break the verifier.
- Phase 4: `REPORT.md` (English, full method), `REPORT.ru.md` (Russian explainer for the
  owner and supervisor).

## 5. Rules

- Frozen files (`src/mfas/metrics.py`, `eval/*`, `tests/test_metrics.py`) are never edited.
- The best-known solution file is a comparator only; no bound method reads it. Methods may
  use our own H64 order (`results/champions/H64_connectome_s42_cuda.npz`) as a guide, since
  a guide cannot make a certificate invalid.
- Runs: background, logged in `logs/`, each under 60 min unless the log states why; seeds
  pinned. Failed and negative results are logged in `PROGRESS.md` too.
- Code style: CONFIG block, minimal comments, no f-string number formats, English.
