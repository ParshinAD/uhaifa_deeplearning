"""Analysis subpackage — PRIVILEGED, ISOLATED diagnostics.

This subpackage is the **only** place in the codebase permitted to read the
downloaded near-optimal ordering ``data/best_solution`` (the "answer"). It exists
to *measure* the gap between Rocket's continuous optimum and a near-optimal
ordering, NOT to feed that answer into any optimization.

LEAKAGE INVARIANT (see CLAUDE.md / experiments/diagnosis.md)
------------------------------------------------------------
Nothing under ``mfas.experiments`` or ``mfas.baseline`` may import this subpackage
or otherwise read ``data/best_solution``. Any variant whose ``run()`` / loss / init
can reach the best solution produces a meaningless, non-generalizing score. The
critic must fail such a variant. Treat ``best_solution`` like the frozen oracle:
read here, for diagnostics only, and keep its outputs out of ``results/``.
"""
