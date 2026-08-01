# results/randomgraph/ — Track C run records

Provenance JSON for random-graph null-model runs (thesis goal #2). **Separate namespace on
purpose:** `eval/aggregate.py` globs `results/*.json` **non-recursively**, so records in this
subdirectory never pollute the connectome/mouse/microns improvement tables.

- Written **only** by the Track-C runner (see `experiments/randomgraph.md` § "Estimator contract"),
  never hand-authored (hand-editing a results JSON = fabrication; the guard hook blocks it).
- Each record carries: `git_commit`, `config_hash`, `seed`, `generator` + params, `n_nodes`,
  `n_edges`, `estimator_budget`, `h35_pct`, `greedy_pct` (the lower-bound bias control).
- Large per-graph arrays (positions / adjacency) are not committed; the JSON + generator seed
  regenerate them.
