"""Experiment variants for the Rocket-improvement campaign.

Each variant is a module exposing a single entry point::

    ID: str            # short stable identifier (matches the file name)
    HYPOTHESIS: str    # one-line description of what is being tried
    def run(g, seed, device, time_limit=None) -> mfas.baseline.rocket.RocketResult

Variants are executed by ``eval/run_variant.py`` (NOT the frozen harness) and scored
with the frozen oracle (``mfas.metrics``). A variant must NEVER import/peek at the
target metric to steer optimization, and must NEVER edit a frozen file. See
``experiments/PROTOCOL.md`` and ``CLAUDE.md``.
"""
