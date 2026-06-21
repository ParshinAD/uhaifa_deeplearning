---
description: Run ONE full research cycle (implementer → verifier → critic → log) for a backlog hypothesis.
argument-hint: "[backlog-id]"
---

Run one complete research-loop cycle for backlog item **$1**, following
`experiments/PROTOCOL.md` exactly. Coordinate the subagents below; they communicate only via
the filesystem (backlog.md, src/mfas/experiments/, results/*.json, log.md).

Do this in order, stopping early if a stage fails:

1. **Read context**: `experiments/PROTOCOL.md` (loop, two-stage significance, file contracts)
   and the item **$1** in `experiments/backlog.md`. If `$1` is not found, stop and say so.

2. **Implementer**: use the `implementer` subagent to implement backlog item **$1** as an
   isolated variant `src/mfas/experiments/<id>.py`, run it via `eval/run_variant.py` on BOTH
   datasets × 3 standard seeds (42/123/999), append the Implementer (screen) block to
   `experiments/log.md`, and flip the item's status in `experiments/backlog.md`.
   - If the SCREEN fails (`Δmean ≤ 2×std` on either dataset), record decision **kill** (or
     **iterate** if the idea is salvageable) and skip to step 5.

3. **Verifier**: use the `verifier` subagent to independently re-run the SCREEN, then the
   CONFIRM stage (connectome 5 seeds, mouse 20–30 seeds), compute per-dataset Δ, SE = std·√(2/n)
   and the 95% CI lower bound, and return CONFIRMED / NOT-CONFIRMED. Record its numbers in the
   current `experiments/log.md` entry.

4. **Critic**: use the `critic` subagent to red-team the finding (frozen-file integrity, metric
   leakage, reproducibility, significance, both-dataset robustness) and append its
   `#### Critic verdict` block with a keep/kill/iterate recommendation.

5. **Decision & re-prioritize**: append `#### Decision: keep | kill | iterate` to the log entry;
   update `$1`'s status in `experiments/backlog.md`; if **keep**, add a ranked entry to
   `experiments/findings.md`; then re-prioritize remaining backlog items if warranted.

Rules: never edit a frozen file (`src/mfas/metrics.py`, `eval/harness.py`, `eval/aggregate.py`,
`tests/test_metrics.py`) — a guardrail hook blocks it. Datasets and seed counts come from
PROTOCOL.md, not hardcoded here. Trust only verified numbers; a gain within noise is not a win.
Idempotent: re-running for the same `$1` appends a new dated cycle rather than corrupting prior
entries.
