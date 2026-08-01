"""Random-graph null models — Track C (real brains vs random feedback).

This subpackage generates **answer-free** random directed weighted graphs used as null
models for the "unavoidable feedback" study (thesis goal #2). Its output is compared to
the real connectome(s): does a real brain carry more/less unavoidable feedback than a
random graph matched on its size/degree/block structure?

BOUNDARY (do not blur with mfas.analysis.gap)
---------------------------------------------
The privileged ``mfas.analysis.gap`` also builds synthetic graphs
(``make_synthetic_graph``, ``make_hard_synthetic_graph``) but those **plant a known
near-optimal ordering** for diagnostics. Generators here must plant **no** answer — a
random null has no reference solution, so ``data/best_solution`` and the leakage firewall
do not apply. Scoring still goes through the frozen ``mfas.metrics``.

USAGE
-----
Generators return a :class:`mfas.io.GraphData` (answer-free). ``unavoidable_feedback`` is
estimated by a Track-C runner (see ``experiments/randomgraph.md`` § "Estimator contract"),
NOT by ``eval/run_variant.py`` (which only accepts registered datasets and whose H35 budget
is keyed on ``g.name``).

STATUS: skeleton — signatures fixed, bodies not yet implemented.
"""
from .generators import (
    erdos_renyi_like,
    configuration_model_like,
    stochastic_block_model,
    degree_preserving_rewire,
)

__all__ = [
    "erdos_renyi_like",
    "configuration_model_like",
    "stochastic_block_model",
    "degree_preserving_rewire",
]
