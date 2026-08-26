# results/quarantine/ — runs excluded from every pool, retained in full

A run lands here when the harness itself recorded that it did not execute the computation the
variant specifies. Today there is exactly one such criterion:

    runtime_guard.degraded == true

`eval/runtime_guard.py` sets it when the wall-clock deadline is reached and stages are aborted at
their boundaries; the run record then carries `stages_truncated` with requested-vs-done counts. The
score of such a run is a sample of a smaller algorithm, not of the variant, so pooling it with clean
runs would be a category error. `autoresearch/audit.py:860` already FAILs any pool containing one —
this directory is what to do instead of arguing with that check.

**This is not deletion and it is not data selection on the metric.** The criterion is a boolean the
harness sets from wall clock before any score is read; the file stays under version control and is
quoted in the log entry that excluded it. `audit.load_runs` globs `results/*.json` non-recursively,
so a subdirectory is out of pool by construction rather than by a filter someone has to remember.

**Excluding a run obliges you to replace it**: re-run the same seed at the same role. Seeds are
never re-drawn, added or dropped, and there is no second re-run — a best-of-N over re-runs is
exactly the cherry-picking this directory is meant to make impossible. If the replacement is also
degraded, that is the result: the variant is runtime-blocked on that dataset and the verdict is
`iterate`, with both runs left in `results/`.

Every file here must be listed in `MANIFEST.md` with the log entry that excluded it.
